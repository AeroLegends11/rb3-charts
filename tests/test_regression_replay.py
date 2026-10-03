import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from rb3charts.chart import enforce_playability, map_pitched, reductions, stabilize_repeated_phrases
from rb3charts.model import Note, TempoMap
from rb3charts.phrases import author_guitar_phrases
from rb3charts.pipeline import replay_cached
from rb3charts.regression import (
    benchmark_replays,
    compare_chart_attacks,
    compute_regression_metrics,
    stage_entry,
)


def grid():
    return TempoMap(np.arange(0,24,0.5).tolist())


def _write_replay_fixture(source: Path, profile="band", instruments=None):
    instruments = instruments or ["guitar"]
    for part in ("analysis","metadata","source"):
        (source/part).mkdir(parents=True,exist_ok=True)
    sf.write(source/"source"/"decoded.wav",np.zeros((44100*8,2),dtype=np.float32),44100)
    (source/"source"/"source.json").write_text(json.dumps({"path":"demo.wav","sha256":"fixture"}),encoding="utf-8")
    (source/"metadata"/"song.json").write_text(json.dumps({
        "title":"Replay Demo","artist":"RB3 Charts","album":"","year":"","genre":"Other",
        "song_id":"rb3c_demo","source_seconds":8.0,"chart_profile":profile,"chart_profile_requested":profile
    }),encoding="utf-8")
    (source/"metadata"/"options.json").write_text(json.dumps({"instruments":instruments,"chart_profile":profile}),encoding="utf-8")
    (source/"analysis"/"tempo.json").write_text(json.dumps({"beats":np.arange(0,20,0.5).tolist(),"numerator":4}),encoding="utf-8")
    (source/"analysis"/"chart-profile.json").write_text(json.dumps({"resolved":profile,"reason":"fixture"}),encoding="utf-8")
    rows=[{"start":1+i*0.25,"end":1+i*0.25+0.1,"pitch":pitch,"confidence":0.85,"source_id":f"g-{i}"} for i,pitch in enumerate((48,50,52,53,48,50,52,53))]
    (source/"analysis"/"guitar-transcription.json").write_text(json.dumps(rows),encoding="utf-8")
    (source/"analysis"/"guitar-guitar-events.json").write_text(json.dumps({"events":[dict(row,accepted=True,classification="probable",reason="fixture") for row in rows]}),encoding="utf-8")
    if "bass" in instruments:
        bass_rows=[
            {
                "start":1+i*0.5,
                "end":1+i*0.5+0.2,
                "pitch":pitch,
                "confidence":0.8,
                "source_id":f"b-{i}",
            }
            for i,pitch in enumerate((36,38,40,41))
        ]
        (source/"analysis"/"bass-transcription.json").write_text(
            json.dumps(bass_rows),encoding="utf-8"
        )
    if "drums" in instruments:
        drum_rows=[
            {
                "start":1+i*.2,
                "end":1+i*.2+.065,
                "lanes":[0 if i%2==0 else 2],
                "confidence":.9,
                "source_id":f"d-{i}",
            }
            for i in range(8)
        ]
        (source/"analysis"/"charts.json").write_text(
            json.dumps({"drums":{
                difficulty:drum_rows
                for difficulty in ("expert","hard","medium","easy")
            }}),
            encoding="utf-8",
        )


def test_cached_replay_writes_provenance_snapshots_and_metrics(tmp_path, monkeypatch):
    source=tmp_path/"source-run"
    _write_replay_fixture(source, profile="band", instruments=["guitar"])
    monkeypatch.setattr("rb3charts.pipeline.decode",lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("decode should not run")))
    monkeypatch.setattr("rb3charts.pipeline.separate",lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("separate should not run")))
    monkeypatch.setattr("rb3charts.pipeline.pitched",lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("pitched should not run")))
    result=replay_cached(source,output=tmp_path/"output",progress=lambda _:None,compile=False)
    replay_folder=Path(result["folder"])
    assert result["status"]=="draft_ready"
    assert result["analysis_mode"]=="cached_replay"
    assert (replay_folder/"analysis"/"replay-provenance.json").is_file()
    assert (replay_folder/"analysis"/"guitar-stage-snapshots.json").is_file()
    assert (replay_folder/"analysis"/"guitar-voice-continuity.json").is_file()
    assert (replay_folder/"analysis"/"regression-metrics.json").is_file()
    metadata=json.loads((replay_folder/"metadata"/"song.json").read_text(encoding="utf-8"))
    result_json=json.loads((replay_folder/"result.json").read_text(encoding="utf-8"))
    assert metadata["analysis_mode"]=="cached_replay"
    assert result_json["analysis_mode"]=="cached_replay"


def test_cached_replay_writes_snapshots_and_metrics_for_each_instrument(tmp_path):
    source=tmp_path/"source-run"
    _write_replay_fixture(
        source,profile="band",instruments=["guitar","bass"]
    )
    result=replay_cached(
        source,output=tmp_path/"output",progress=lambda _:None,compile=False
    )
    analysis=Path(result["folder"])/"analysis"
    aggregate=json.loads(
        (analysis/"regression-metrics-by-instrument.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(aggregate["instruments"])=={"guitar","bass"}
    for instrument in ("guitar","bass"):
        snapshots=json.loads(
            (analysis/f"{instrument}-stage-snapshots.json").read_text(
                encoding="utf-8"
            )
        )
        assert snapshots["instrument"]==instrument
        assert len(snapshots["stages"])>=2
        assert snapshots["reductions"]["stage"]=="after_reductions"
        assert (analysis/f"{instrument}-regression-metrics.json").is_file()
    assert aggregate["instruments"]["bass"]["stage_count"]==3
    bass_reductions=aggregate["instruments"]["bass"]["reduction_metrics"]
    assert set(bass_reductions)=={"expert","hard","medium","easy"}
    assert bass_reductions["expert"]["attack_retention"]==1
    assert not any(
        row["orphan_source_ids"] for row in bass_reductions.values()
    )
    assert all(
        row["max_timing_error_ms"] in (None,0)
        for row in bass_reductions.values()
    )


def test_cached_replay_rederives_drum_reductions_from_expert(tmp_path):
    source=tmp_path/"source-run"
    _write_replay_fixture(source,profile="band",instruments=["drums"])
    result=replay_cached(
        source,output=tmp_path/"output",progress=lambda _:None,compile=False
    )
    charts=json.loads(
        (Path(result["folder"])/"analysis"/"charts.json").read_text(
            encoding="utf-8"
        )
    )["drums"]
    assert len(charts["expert"])==8
    assert len(charts["hard"])<len(charts["expert"])
    assert {
        note["source_id"] for note in charts["hard"]
    } <= {
        note["source_id"] for note in charts["expert"]
    }


def test_cached_replay_requires_event_reports_for_electronic_roles(tmp_path):
    source=tmp_path/"source-electronic"
    _write_replay_fixture(source, profile="electronic", instruments=["guitar"])
    (source/"analysis"/"guitar-guitar-events.json").unlink()
    with pytest.raises(FileNotFoundError):
        replay_cached(source,output=tmp_path/"output",progress=lambda _:None,compile=False)


def test_benchmark_replays_aggregates_matching_song_metrics(tmp_path):
    def write_run(root,lanes,instrument="guitar"):
        (root/"metadata").mkdir(parents=True,exist_ok=True)
        (root/"analysis").mkdir(exist_ok=True)
        (root/"metadata"/"song.json").write_text(
            json.dumps({"artist":"RB3 Charts","title":"Benchmark Demo"}),encoding="utf-8"
        )
        notes=[
            {"source_id":str(index),"start":index*.2,"end":index*.2+.065,
             "lanes":list(lane) if isinstance(lane,tuple) else [lane],
             "pitch":pitch}
            for index,(lane,pitch) in enumerate(zip(lanes,(48,50,52)))
        ]
        reductions={
            "difficulties":{
                "expert":{"notes":notes},
                "hard":{"notes":notes[:2]},
                "medium":{"notes":notes[:1]},
                "easy":{"notes":notes[:1]},
            }
        }
        (root/"analysis"/f"{instrument}-stage-snapshots.json").write_text(
            json.dumps({
                "instrument":instrument,
                "stages":[{"stage":"final","notes":notes}],
                "reductions":reductions,
            }),
            encoding="utf-8",
        )
        (root/"analysis"/"structure-diagnostics.json").write_text(
            json.dumps({
                "sections":[{},{}],
                "strongest_activity_boundaries":[
                    {"activity_change":.25}
                ],
                "drum_grooves":{
                    "repeated_families":2,
                    "supported_attack_ratio":.5,
                },
            }),
            encoding="utf-8",
        )

    write_run(tmp_path/"baseline"/"run",(0,1,2))
    write_run(tmp_path/"candidate"/"run",(0,0,2))

    report=benchmark_replays(
        tmp_path/"baseline",tmp_path/"candidate",tmp_path/"benchmark.json"
    )

    assert report["songs_matched"]==1
    assert report["instrument"]=="guitar"
    assert report["macro"]["delta"]["attacks"]==0
    assert report["macro"]["delta"]["collapsed_pitch_changes"]==1
    assert report["macro"]["baseline"]["lane_coverage"]==[0,1,2]
    assert report["macro"]["baseline"]["hard_retention"]==.6667
    assert report["macro"]["baseline"]["max_hard_peak_four_second_nps"]==.5
    assert report["structure_macro_song_mean"]["songs_matched"]==1
    assert (
        report["structure_macro_song_mean"]["baseline"]
        ["repeated_drum_groove_families"]==2
    )
    assert report["macro"]["candidate"]["assessment_status_counts"]=={"missing":1}
    assert (tmp_path/"benchmark.json").is_file()

    write_run(tmp_path/"baseline"/"run",(0,1,2),instrument="bass")
    write_run(tmp_path/"candidate"/"run",(0,1,1),instrument="bass")
    bass=benchmark_replays(
        tmp_path/"baseline",
        tmp_path/"candidate",
        tmp_path/"bass-benchmark.json",
        instrument="bass",
    )
    assert bass["instrument"]=="bass"
    assert bass["macro"]["delta"]["collapsed_pitch_changes"]==1

    write_run(
        tmp_path/"baseline"/"run",((0,2),(1,3),2),instrument="drums"
    )
    write_run(
        tmp_path/"candidate"/"run",((0,2),(1,3),2),instrument="drums"
    )
    drums=benchmark_replays(
        tmp_path/"baseline",
        tmp_path/"candidate",
        tmp_path/"drums-benchmark.json",
        instrument="drums",
    )
    assert drums["macro"]["baseline"]["chords"]==0
    assert drums["macro"]["baseline"]["chord_rate"]==0
    assert drums["macro"]["baseline"]["simultaneous_attacks"]==2

    for root in (tmp_path/"baseline"/"run",tmp_path/"candidate"/"run"):
        snapshot=root/"analysis"/"drums-stage-snapshots.json"
        payload=json.loads(snapshot.read_text(encoding="utf-8"))
        payload.pop("reductions")
        snapshot.write_text(json.dumps(payload),encoding="utf-8")
    legacy=benchmark_replays(
        tmp_path/"baseline",
        tmp_path/"candidate",
        tmp_path/"legacy-drums-benchmark.json",
        instrument="drums",
    )
    assert legacy["macro"]["baseline"]["reduction_songs"]==0
    assert legacy["macro"]["baseline"]["hard_retention"] is None

    write_run(tmp_path/"baseline"/"run",(0,1,2),instrument="keys")
    write_run(tmp_path/"candidate"/"run",(0,1,2),instrument="keys")
    candidate_keys=(
        tmp_path/"candidate"/"run"/"analysis"
        /"keys-stage-snapshots.json"
    )
    payload=json.loads(candidate_keys.read_text(encoding="utf-8"))
    payload.pop("reductions")
    candidate_keys.write_text(json.dumps(payload),encoding="utf-8")
    unmatched=benchmark_replays(
        tmp_path/"baseline",
        tmp_path/"candidate",
        tmp_path/"unmatched-keys-benchmark.json",
        instrument="keys",
    )
    assert unmatched["macro"]["baseline"]["reduction_songs"]==0
    assert unmatched["macro"]["baseline"]["hard_retention"] is None


def test_human_chart_comparison_reports_timing_style_and_motifs():
    reference = [
        {"time": index * .25, "lanes": (index % 4,), "duration": .5 if index == 2 else .1}
        for index in range(8)
    ]
    candidate = [
        {"time": row["time"] + .02, "lanes": row["lanes"], "duration": row["duration"]}
        for row in reference
    ]
    candidate[4] = {**candidate[4], "lanes": (0, 2)}

    report = compare_chart_attacks(reference, candidate)

    assert report["attack"]["f1"] == 1
    assert report["attack_windows"]["window_seconds"] == 4
    assert report["attack_windows"]["worst_reference_recall"]["recall"] == 1
    assert report["attack"]["timing_error_ms"]["median"] == pytest.approx(20)
    assert report["chord"]["candidate"] == 1
    assert report["sustain"]["matched"] == 1
    assert report["lane"]["exact_agreement"] == .875
    assert report["density"]["attack_count_ratio"] == 1
    assert report["motifs"]["families"] == 1
    assert report["motifs"]["coverage"] < 1


def test_human_chart_comparison_exposes_local_attack_recall_collapse():
    reference = [
        {"time": time, "lanes": (0,), "duration": .1}
        for time in (0, 1, 4, 5)
    ]
    candidate = [
        {"time": time, "lanes": (0,), "duration": .1}
        for time in (0, 1, 4)
    ]

    windows = compare_chart_attacks(reference, candidate)["attack_windows"]

    assert [row["recall"] for row in windows["windows"]] == [1, .5]
    assert windows["worst_reference_recall"]["start"] == 4


def test_repeated_four_note_runs_metric_coverage():
    tempo=grid()
    notes=[Note(1+i*0.25,1+i*0.25+0.065,(i%4,),source_id=f"s-{i}",pitch=48+i%4) for i in range(8)]
    payload={"instrument":"guitar","stages":[stage_entry("before",notes),stage_entry("after",notes)]}
    metrics=compute_regression_metrics(payload,tempo)
    lane=metrics["transitions"][0]["lane_agreement"]
    assert lane["coverage"]>0
    assert lane["absolute_lane_agreement"]==1
    assert lane["relative_shape_agreement"]==1


def test_transposed_return_tracks_relative_not_absolute_agreement():
    tempo=grid()
    base=[Note(1+i*0.25,1+i*0.25+0.065,(lane,),source_id=f"s-{i}",pitch=p) for i,(p,lane) in enumerate(zip((48,50,52,53,48,50,52,53),(0,1,2,3,0,1,2,3)))]
    shifted=[Note(note.start,note.end,(min(4,note.lanes[0]+1),),source_id=note.source_id,pitch=note.pitch) for note in base]
    payload={"instrument":"guitar","stages":[stage_entry("a",base),stage_entry("b",shifted)]}
    lane=compute_regression_metrics(payload,tempo)["transitions"][0]["lane_agreement"]
    assert lane["relative_shape_agreement"]==1
    assert lane["absolute_lane_agreement"]<1


def test_near_repeats_with_strong_variations_stay_distinct():
    tempo=grid()
    notes=[]
    for occurrence,offset in enumerate((0.0,2.0,4.0,6.0)):
        pitches=(48,50,52,53) if occurrence<3 else (48,54,55,57)
        for step,pitch in enumerate(pitches):
            notes.append(Note(offset+step*0.25,offset+step*0.25+0.065,(step%4,),confidence=0.9,source_id=f"{occurrence}-{step}",pitch=pitch))
    result,report=stabilize_repeated_phrases(notes,tempo,"guitar")
    assert report["families"]>=1
    assert [note.pitch for note in result[12:16]]==[48,54,55,57]


def test_missing_attacks_rests_do_not_get_bridged():
    tempo=grid()
    notes=[]
    for occurrence,offset in enumerate((0.0,2.0,4.0)):
        for step,pitch in enumerate((48,48,50,51)):
            notes.append(Note(offset+step*0.25,offset+step*0.25+0.065,(step%3,),confidence=0.8,source_id=f"base-{occurrence}-{step}",pitch=pitch))
    for step,(time,pitch) in enumerate(zip((6.0,6.25,6.5,7.0),(53,53,55,48))):
        notes.append(Note(time,time+0.065,(3,),confidence=0.4 if step==3 else 0.8,source_id=f"rest-{step}",pitch=pitch))
    _,report=stabilize_repeated_phrases(notes,tempo,"guitar")
    assert report["transposed_occurrences"]==0


def test_chords_are_phrase_boundaries():
    notes=[
        Note(1.0,1.1,(0,),source_id="a",pitch=60),
        Note(1.25,1.35,(1,),source_id="b",pitch=62),
        Note(1.5,1.65,(0,2),source_id="c",pitch=64),
        Note(1.75,1.85,(2,),source_id="d",pitch=65),
        Note(2.0,2.1,(3,),source_id="e",pitch=67),
    ]
    _,report=author_guitar_phrases(notes,grid())
    assert len(report["phrases"])==2
    assert report["phrases"][1]["source_ids"][0]=="d"


def test_unrelated_matching_rhythm_not_forced_to_same_shape():
    tempo=grid()
    notes=[]
    for occurrence,offset in enumerate((0.0,2.0,4.0)):
        pitches=(48,50,52,53) if occurrence<2 else (64,61,67,58)
        for step,pitch in enumerate(pitches):
            notes.append(Note(offset+step*0.25,offset+step*0.25+0.065,(4-step%4,),confidence=0.9,source_id=f"{occurrence}-{step}",pitch=pitch))
    result,_=stabilize_repeated_phrases(notes,tempo,"guitar")
    assert [note.pitch for note in result[8:12]]==[64,61,67,58]


def test_late_confirmed_transposition_informs_earlier_weaker_occurrence():
    tempo=grid()
    notes=[]
    for occurrence,offset in enumerate((0.0,2.0,4.0)):
        for step,pitch in enumerate((48,48,50,51)):
            notes.append(Note(offset+step*0.25,offset+step*0.25+0.065,(step%3,),confidence=0.8,source_id=f"clean-{occurrence}-{step}",pitch=pitch))
    for step,pitch in enumerate((53,53,55)):
        notes.append(Note(5.25+step*0.25,5.25+step*0.25+0.065,(4,),confidence=0.8,source_id=f"partial-{step}",pitch=pitch))
    for step,pitch in enumerate((53,53,55,48)):
        notes.append(Note(6+step*0.25,6+step*0.25+0.065,(4,),confidence=0.4 if step==3 else 0.8,source_id=f"shift-{step}",pitch=pitch))
    result,report=stabilize_repeated_phrases(notes,tempo,"guitar")
    assert [note.lanes[0] for note in result[-7:-4]]==[1,1,2]
    assert report["transposed_occurrences"]==1


def test_variable_tempo_fixture_exercises_authoring_stack():
    tempo=TempoMap([0.0,0.48,0.97,1.43,1.91,2.44,3.02,3.55,4.12,4.66,5.19,5.74,6.32,6.89,7.43,7.98])
    events=[{"start":1.0+i*0.21,"end":1.0+i*0.21+0.1,"pitch":pitch,"confidence":0.8,"source_id":f"v-{i}"} for i,pitch in enumerate((48,50,52,53,48,50,52,53))]
    mapped=map_pitched(events,"guitar")
    playable,_=enforce_playability(mapped,"guitar",tempo,"band",phrase_aware=True)
    authored,_=author_guitar_phrases(playable,tempo)
    reduced=reductions(authored,"guitar",tempo)
    payload={"instrument":"guitar","stages":[stage_entry("map",mapped),stage_entry("playability",playable),stage_entry("phrases",authored),stage_entry("reductions_expert",reduced["expert"])]}
    metrics=compute_regression_metrics(payload,tempo)
    assert metrics["stage_count"]==4
    assert len(metrics["transitions"])==3


def test_phrase_boundary_eight_beat_split():
    tempo=grid()
    notes=[
        Note(1.0,1.08,(0,),source_id="a",pitch=60),
        Note(1.25,1.33,(1,),source_id="b",pitch=62),
        Note(5.0,5.08,(2,),source_id="c",pitch=64),
        Note(5.25,5.33,(3,),source_id="d",pitch=65),
    ]
    _,report=author_guitar_phrases(notes,tempo)
    assert len(report["phrases"])==2
    assert not report["phrases"][1]["joined_previous_phrase"]
