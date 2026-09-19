import copy
import json
import numpy as np

from rb3charts.events import interpret_events, write_event_trace
from rb3charts.model import Note, TempoMap, save_json


def candidate(start, pitch=60, confidence=.5, onset=0.0, source_id=None):
    return dict(start=start, end=start+.15, pitch=pitch, confidence=confidence,
                model_onset_strength=onset, source_id=source_id or str(start), source="guitar")


def tone(y, start, duration=.15, amplitude=.05, sr=8000):
    t = np.arange(round(duration*sr))/sr
    a = round(start*sr)
    y[a:a+len(t)] += amplitude*np.sin(2*np.pi*440*t)


def test_quiet_attack_retained_and_silent_model_hallucination_rejected():
    y = np.zeros(16000)
    tone(y, .5, amplitude=.0003)  # Below the old fixed RMS cutoff.
    rows = [candidate(.5), candidate(1, confidence=.95, onset=.9)]
    original = copy.deepcopy(rows)
    accepted, report = interpret_events(rows, y, 8000)
    assert [e["start"] for e in accepted] == [.5]
    assert report["events"][1]["classification"] == "artifact"
    assert rows == original
    assert accepted[0]["start"] == .5  # Evidence cannot quantize timing.


def test_grid_cannot_rescue_unsupported_transient_but_legato_survives():
    y = np.zeros(24000)
    tone(y, 0, duration=3)
    rows = [candidate(.5), candidate(1, confidence=.8)]
    accepted, report = interpret_events(rows, y, 8000, TempoMap([0,.5,1,1.5,2,2.5,3]))
    assert [e["start"] for e in accepted] == [1]
    assert report["events"][0]["evidence"]["beat_alignment"] == 1
    assert report["events"][0]["classification"] == "uncertain"


def test_repetition_rescues_weak_audible_attack_not_silent_copy():
    y = np.zeros(48000)
    rows = []
    for repeat in range(3):
        for i, pitch in enumerate([60,62,64,62]):
            start = .5 + repeat*1.5 + i*.25
            rows.append(candidate(start, pitch, .3 if i == 2 else .8))
            if repeat != 2:
                tone(y, start)
    accepted, report = interpret_events(rows, y, 8000, TempoMap(list(np.arange(0,6.5,.5))))
    assert len(accepted) == 8
    assert all(e["evidence"]["repeated_phrase"] for e in accepted)
    assert all(e["classification"] == "artifact" for e in report["events"][-4:])


def test_chord_does_not_count_as_repeated_phrase_and_off_grid_attack_survives():
    y = np.zeros(16000)
    tone(y, .537)
    rows = [candidate(.537, p, .8) for p in (60,64,67,72)]
    accepted, _ = interpret_events(rows, y, 8000, TempoMap([0,.5,1,1.5,2]))
    assert len(accepted) == 4
    assert not any(e["evidence"]["repeated_phrase"] for e in accepted)
    assert all(e["start"] == .537 for e in accepted)


def test_empty_stem_and_no_candidates():
    accepted, report = interpret_events([], np.array([]), 8000)
    assert accepted == [] and report["candidate_count"] == 0


def test_silent_repetitions_cannot_rescue_a_weak_isolated_phrase():
    y = np.zeros(32000)
    rows = []
    for repeat in range(2):
        for i, pitch in enumerate([60,62,64,62]):
            start = .5 + repeat*1.5 + i*.25
            rows.append(candidate(start, pitch, .3))
            if repeat == 0:
                tone(y, start)
    accepted, _ = interpret_events(rows, y, 8000, TempoMap(list(np.arange(0,4.5,.5))))
    assert accepted == []


def test_backend_keeps_rejected_candidates_in_audit_and_handles_empty_refinement(tmp_path, monkeypatch):
    from rb3charts import transcribe
    from basic_pitch import inference
    from basic_pitch.note_creation import model_frames_to_time, MIDI_OFFSET
    sr = 22050
    y = np.zeros(sr*2)
    tone(y, .5, sr=sr)
    activation = np.zeros((172,88))
    times = model_frames_to_time(len(activation))
    activation[np.argmin(abs(times-.5)),60-MIDI_OFFSET] = .85
    monkeypatch.setattr(inference, "predict", lambda *a, **k: (
        {"onset":activation}, None, [(.5,.65,60,.2,None)]))
    monkeypatch.setattr(transcribe, "read_mono", lambda path:(y,sr))
    result = transcribe.pitched(tmp_path/"unused.wav", "guitar", .35, lambda x:None,
                               analysis_folder=tmp_path)
    assert result == []
    report = json.loads((tmp_path/"guitar-guitar-events.json").read_text())
    assert report["candidate_count"] == 1 and report["rejected_count"] == 1
    assert report["events"][0]["model_onset_strength"] == .85
    monkeypatch.setattr(inference, "predict", lambda *a, **k:({"onset":activation},None,[]))
    assert transcribe.pitched(tmp_path/"unused.wav", "guitar", .35, lambda x:None,
                              refine_articulations=True) == []


def test_trace_distinguishes_detection_role_and_gameplay_decisions(tmp_path):
    y = np.zeros(40000)
    rows = [candidate(t, source_id=str(i)) for i,t in enumerate([.5,1,1.5,2])]
    for t in (.5,1,1.5):
        tone(y, t)
    accepted, report = interpret_events(rows, y, 8000)
    save_json(tmp_path/"guitar-guitar-events.json", report)
    derived = candidate(3, source_id="kick-bass-1")
    expert = [Note(.51,.6,(0,2),source_id="0+kick-bass-1")]
    write_event_trace(tmp_path,"guitar",[accepted[0],accepted[2],derived],expert)
    trace = json.loads((tmp_path/"guitar-event-trace.json").read_text())
    outcomes = {e["source_id"]:e["outcome"] for e in trace["events"]}
    assert outcomes == {"0":"expert", "1":"role_filtered",
                        "2":"mapping_or_playability_filtered", "3":"detection_rejected",
                        "kick-bass-1":"expert"}
    assert trace["events"][-1]["classification"] == "derived"
    assert trace["events"][0]["expert_notes"][0]["start"] == .51
    assert (tmp_path/"guitar-event-trace.csv").is_file()
