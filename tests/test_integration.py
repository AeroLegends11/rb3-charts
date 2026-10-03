"""Small real-library integration checks, without model downloads."""
from pathlib import Path
import json
import numpy as np
import pytest
import soundfile as sf
from rb3charts.model import Note,TempoMap,Options
from rb3charts.chart import reductions,overdrive,align_drum_activations
from rb3charts.midi import write_midi
from rb3charts.validate import validate
from rb3charts.audio import write_padded_audio
from rb3charts.timing import analyze_tempo
from rb3charts.preview import (
    create_comparison_preview,
    create_expert_hard_reference_preview,
    create_midi_comparison_preview,
    create_preview,
)
from rb3charts.audio import detect_chart_profile

def test_missing_beats_repaired_without_moving_detected_anchors():
    from rb3charts.timing import repair_beat_gaps
    original=[0,.375,.75,1.125,1.5,3.375,3.75,4.125,4.5,4.6]
    result,inserted,discarded=repair_beat_gaps(original,5)
    assert inserted==4
    assert discarded==1
    assert set(original[:-1])<=set(result)
    assert np.allclose(np.diff(result),.375)

def test_real_beat_detector_finds_click_tempo():
    sr=22050
    y=np.zeros(sr*12,dtype=np.float32)
    for t in np.arange(0.5,11.8,0.5):
        a=int(t*sr)
        y[a:a+100]=np.hanning(100)
    tempo=analyze_tempo(y,sr,12)
    assert 115<tempo.bpm<125

def test_half_time_bpm_tag_is_reconciled_with_audio():
    sr=22050
    y=np.zeros(sr*8,dtype=np.float32)
    for t in np.arange(.03,8,.375):
        a=int(t*sr); y[a:a+100]=np.hanning(100)
    tempo=analyze_tempo(y,sr,8,bpm_hint=80)
    assert tempo.bpm==pytest.approx(160,abs=.01)
    assert tempo.beats[0]==pytest.approx(.03,abs=.01)

def test_electronic_tempo_uses_stable_studio_grid():
    sr=22050
    y=np.zeros(sr*10,dtype=np.float32)
    for i,t in enumerate(np.arange(.08,10,60/128)):
        if i not in (5,12):y[int(t*sr):int(t*sr)+100]=np.hanning(100)
    tempo=analyze_tempo(y,sr,10,prefer_constant=True)
    assert tempo.bpm==pytest.approx(128,abs=.1)
    assert np.std(np.diff(tempo.beats))<1e-10

def test_audio_lead_in_matches_midi_lead(tmp_path):
    sr=44100
    y=np.zeros((sr*3,2),dtype=np.float32)
    y[sr]=0.5
    source=tmp_path/"source.wav"
    sf.write(source,y,sr,subtype="FLOAT")
    tempo=TempoMap([0.25,0.75,1.25,1.75,2.25,2.75])
    notes=[Note(1,1.1,(0,),source_id="attack")]
    lead=write_midi(tmp_path/"notes.mid",tempo,{"guitar":reductions(notes,"guitar",tempo)},3,{})
    write_padded_audio(source,tmp_path/"song.wav",lead)
    padded,_=sf.read(tmp_path/"song.wav")
    assert np.argmax(padded[:,0])/sr==pytest.approx(1+lead,abs=1/sr)

def test_drum_fill_ends_on_shared_attack(tmp_path):
    import mido
    tempo=TempoMap(np.arange(0,65,0.5).tolist())
    expert=[Note(i*0.5+0.007,i*0.5+0.072,(0,2),source_id=str(i)) for i in range(120)]
    charts=reductions(expert,"drums",tempo)
    phrases=overdrive(charts,tempo,62)
    corrections=align_drum_activations(charts,tempo,phrases,62)
    assert corrections and all(abs(c["change_ms"])<=12 for c in corrections)
    path=tmp_path/"notes.mid"
    write_midi(path,tempo,{"drums":charts},62,{"drums":phrases})
    absolute=0
    fills=[]
    starts=set()
    for msg in next(t for t in mido.MidiFile(path).tracks if t.name=="PART DRUMS"):
        absolute+=msg.time
        if msg.type=="note_off" and msg.note==120:fills.append(absolute)
        if msg.type=="note_on" and msg.velocity and 96<=msg.note<=100:starts.add(absolute)
    assert fills
    assert set(fills)<=starts
    assert validate({"drums":charts},tempo,62,path)["passed"]

def test_batch_continues_after_one_song_fails(tmp_path,monkeypatch):
    from rb3charts import pipeline
    for name in ("a.wav","b.wav"):(tmp_path/name).write_bytes(b"placeholder")
    def fake(options,progress):
        if options.source.name=="a.wav":raise ValueError("bad audio")
        return {"status":"draft_ready"}
    monkeypatch.setattr(pipeline,"generate",fake)
    results=pipeline.batch(Options(source=tmp_path,output=tmp_path/"output"))
    assert [r["status"] for r in results]==["failed","draft_ready"]
    assert len(list((tmp_path/"output").glob("batch_report_*.json")))==1

def test_reviewed_run_packages_without_regenerating(tmp_path,monkeypatch):
    from rb3charts import pipeline
    run=tmp_path/"reviewed"
    for folder in (run/"metadata",run/"import",run/"logs"):
        folder.mkdir(parents=True)
    (run/"result.json").write_text(json.dumps({"status":"draft_ready","warnings":[],"counts":{}}))
    (run/"metadata"/"song.json").write_text(json.dumps({"song_id":"rb3c_test"}))
    (run/"validation_report.json").write_text(json.dumps({"passed":True}))
    for name in ("notes.mid","song.wav","song.ini"):(run/"import"/name).write_bytes(b"ready")
    package=run/"rb3"/"rb3c_test.con"
    def fake_compile(import_folder,run_folder,song_id,onyx,log):
        package.parent.mkdir()
        package.write_bytes(b"CON "+b"x"*1024)
        return package
    monkeypatch.setattr(pipeline,"compile_rb3",fake_compile)
    monkeypatch.setattr(pipeline,"find_onyx",lambda:Path("onyx.exe"))
    result=pipeline.package_existing(run,lambda message:None)
    assert result["status"]=="compiled_needs_playtest"
    assert result["package"]==str(package)
    assert any("Legacy run" in warning for warning in result["warnings"])
    assert result["validation_recheck"]=="legacy_unavailable"

def test_reviewed_run_rejects_changed_package_input(tmp_path,monkeypatch):
    from rb3charts import pipeline
    run=tmp_path/"reviewed"
    for folder in (run/"metadata",run/"import",run/"analysis",run/"logs"):
        folder.mkdir(parents=True)
    (run/"result.json").write_text(json.dumps({"status":"draft_ready"}))
    (run/"metadata"/"song.json").write_text(
        json.dumps({"song_id":"rb3c_test"})
    )
    (run/"validation_report.json").write_text(json.dumps({"passed":True}))
    for name in ("notes.mid","song.wav","song.ini"):
        (run/"import"/name).write_bytes(b"reviewed")
    pipeline._write_package_input_manifest(run)
    (run/"import"/"notes.mid").write_bytes(b"changed")
    with pytest.raises(
        RuntimeError,match="changed after validation: import.notes.mid"
    ):
        pipeline.package_existing(run,lambda message:None)
    manifest=run/"analysis"/"package-inputs.json"
    payload=json.loads(manifest.read_text(encoding="utf-8"))
    payload["files"].pop()
    manifest.write_text(json.dumps(payload),encoding="utf-8")
    with pytest.raises(RuntimeError,match="manifest is incomplete"):
        pipeline.package_existing(run,lambda message:None)

def test_existing_run_validation_rechecks_midi_and_manifest(
    tmp_path,monkeypatch
):
    from rb3charts import pipeline
    run=tmp_path/"draft"
    for folder in (
        run/"analysis",run/"metadata",run/"import",run/"logs"
    ):
        folder.mkdir(parents=True)
    tempo=TempoMap([0,.5,1,1.5,2,2.5,3])
    charts=reductions(
        [Note(.5,.6,(0,),source_id="original")],"guitar",tempo
    )
    (run/"analysis"/"charts.json").write_text(
        json.dumps({"guitar":{
            difficulty:[note.__dict__ for note in notes]
            for difficulty,notes in charts.items()
        }}),
        encoding="utf-8",
    )
    (run/"analysis"/"tempo.json").write_text(
        json.dumps({"beats":tempo.beats,"numerator":4}),
        encoding="utf-8",
    )
    (run/"metadata"/"song.json").write_text(
        json.dumps({"source_seconds":2,"song_id":"rb3c_demo"}),
        encoding="utf-8",
    )
    write_midi(run/"import"/"notes.mid",tempo,{"guitar":charts},2,{})
    (run/"import"/"song.wav").write_bytes(b"reviewed audio")
    (run/"import"/"song.ini").write_text("[song]\nname=Demo\n")
    pipeline._write_package_input_manifest(run)
    report=pipeline.validate_existing_run(run)
    assert report["passed"]
    assert report["package_input_manifest"]=="verified"
    assert (run/"validation_recheck_report.json").is_file()
    (run/"result.json").write_text(
        json.dumps({"status":"draft_ready"}),encoding="utf-8"
    )
    (run/"validation_report.json").write_text(
        json.dumps({"passed":True}),encoding="utf-8"
    )
    package=run/"rb3"/"rb3c_demo.con"
    def fake_compile(import_folder,run_folder,song_id,onyx,log):
        package.parent.mkdir()
        package.write_bytes(b"CON "+b"x"*1024)
        return package
    monkeypatch.setattr(pipeline,"compile_rb3",fake_compile)
    monkeypatch.setattr(pipeline,"find_onyx",lambda:Path("onyx.exe"))
    packaged=pipeline.package_existing(run,lambda message:None)
    assert packaged["status"]=="compiled_needs_playtest"
    assert packaged["validation_recheck"]=="passed"
    assert packaged["validation_recheck_report"].endswith(
        "validation_recheck_report.json"
    )
    (run/"import"/"song.ini").write_text("[song]\nname=Changed\n")
    changed=pipeline.validate_existing_run(run)
    assert not changed["passed"]
    assert any(
        "changed after validation" in error
        for error in changed["errors"]
    )
    with pytest.raises(
        RuntimeError,match="Current chart revalidation failed"
    ):
        pipeline.package_existing(run,lambda message:None)
    (run/"analysis"/"package-inputs.json").write_text(
        "{invalid",encoding="utf-8"
    )
    malformed=pipeline.validate_existing_run(run)
    assert not malformed["passed"]
    assert "Package-input manifest is malformed" in malformed["errors"]
    (run/"analysis"/"package-inputs.json").write_text(
        json.dumps({"schema_version":1,"files":[None]}),
        encoding="utf-8",
    )
    invalid_item=pipeline.validate_existing_run(run)
    assert not invalid_item["passed"]
    assert "Package-input manifest is malformed" in invalid_item["errors"]

    copy=tmp_path/"draft-copy"
    import shutil
    shutil.copytree(run,copy)
    (copy/"analysis"/"tempo.json").write_text(
        "{invalid",encoding="utf-8"
    )
    batch=pipeline.validate_existing_runs(
        tmp_path,tmp_path/"batch-validation.json"
    )
    assert batch["runs_total"]==2
    assert batch["runs_passed"]==0
    assert batch["runs_failed"]==2
    assert (tmp_path/"batch-validation.json").is_file()

def test_visual_review_contains_synced_controls_and_beat_grid(tmp_path):
    (tmp_path/"audio").mkdir()
    sf.write(tmp_path/"audio"/"song.wav",np.zeros((44100*2,2),dtype=np.float32),44100)
    tempo=TempoMap([0,.5,1,1.5,2])
    notes=[
        Note(.5,.6,(0,),confidence=.4,source_id="one"),
        Note(1.0,1.1,(0,2,4),confidence=.9,source_id="two"),
    ]
    charts={"guitar":{d:notes for d in ("expert","hard","medium","easy")}}
    voice={
        "guitar":{
            "windows":[{
                "start_seconds":.25,
                "end_seconds":1.25,
                "register_separation":14.0,
                "dominant_evidence_share":.6,
                "register_switches":4,
                "candidate_for_tracking":True,
            }]
        }
    }
    patterns={
        "guitar":{
            "top_patterns":[{
                "occurrences":2,
                "source_times":[.5,1.5],
                "lane_variants":2,
                "lane_mismatch_times":[1.25],
            }]
        }
    }
    create_preview(
        tmp_path,
        charts,
        tempo,
        0,
        {"title":"Review","artist":"Tester"},
        voice,
        patterns=patterns,
        structure_diagnostics={
            "strongest_activity_boundaries":[{
                "seconds":1.5,
                "activity_change":.75,
            }]
        },
    )
    page=(tmp_path/"review.html").read_text(encoding="utf-8")
    assert "Play / Pause" in page and "−5 seconds" in page
    assert "Highway speed" in page and "Hyper 5" in page
    assert "highway.onchange=()=>lookAhead=+highway.value" in page
    assert "Playback speed" in page
    assert "const payload=" in page and '"beats": [0, 0.5, 1, 1.5, 2]' in page
    assert '"lead": 0' in page and "local density" in page
    assert '"overlays": {"guitar":' in page and '"register_switches": 4' in page
    assert '"kind": "low confidence"' in page
    assert '"kind": "unusual chord"' in page
    assert '"kind": "motif lane mismatch"' in page
    assert '"kind": "repeated motif candidate"' in page
    assert "family 1; 2 occurrences; instance 1" in page
    assert '"kind": "activity boundary"' in page
    assert "Difficulty attacks" in page
    assert "Review markers jump to low-confidence notes" in page
    assert "Amber regions mark conflicting guitar-register evidence" in page
    assert "voice conflict" in page
    assert 'height="700"' in page and "NOTES HIT THIS LINE" in page
    assert "activeHold=n.end-n.t>=.09&&n.t<time&&n.end>time" in page
    assert "(!activeHold&&rawStart>strike+2)||rawEnd>strike+2" in page
    assert "if(activeHold)continue" in page
    assert (tmp_path/"audio"/"timing-audit-guitar.wav").is_file()


def test_comparison_preview_keeps_versions_synchronized(tmp_path):
    notes=[
        {
            "start":1.0,
            "end":1.1,
            "lanes":[0],
            "confidence":.8,
            "source_id":"shared",
        }
    ]
    extra={
        "start":1.25,
        "end":1.35,
        "lanes":[1],
        "confidence":.7,
        "source_id":"candidate-only",
    }
    for name,hard in (("baseline",notes),("candidate",notes+[extra])):
        folder=tmp_path/name
        (folder/"analysis").mkdir(parents=True)
        (folder/"metadata").mkdir()
        (folder/"source").mkdir()
        (folder/"audio").mkdir()
        (folder/"analysis"/"charts.json").write_text(
            json.dumps({"guitar":{"expert":notes,"hard":hard}}),
            encoding="utf-8",
        )
        (folder/"analysis"/"tempo.json").write_text(
            json.dumps({"beats":[0,.5,1,1.5,2],"numerator":4}),
            encoding="utf-8",
        )
        (folder/"metadata"/"song.json").write_text(
            json.dumps({
                "song_id":"song-id",
                "title":"A/B Song",
                "artist":"Tester",
                "lead_seconds":2.0,
            }),
            encoding="utf-8",
        )
        (folder/"source"/"source.json").write_text(
            json.dumps({"sha256":"abc"}),encoding="utf-8"
        )
        (folder/"audio"/"song.wav").write_bytes(b"audio")

    output=tmp_path/"review"/"ab.html"
    report=create_comparison_preview(
        tmp_path/"baseline",tmp_path/"candidate",output
    )
    page=output.read_text(encoding="utf-8")
    assert report["counts"]["baseline"]["guitar"]["hard"]==1
    assert report["counts"]["candidate"]["guitar"]["hard"]==2
    assert "A · Baseline" in page and "B · Candidate" in page
    assert '"different": true' in page
    assert "A/B keys switch versions" in page
    assert "Highway speed" in page and "Hyper 5" in page
    assert "highway.onchange=()=>lookAhead=+highway.value" in page
    assert "Playback speed" in page
    assert "NOTES HIT THIS LINE" in page
    assert "rawY>strike+2" in page
    assert '"default_difficulties": {"guitar": "hard"}' in page
    assert "diff.value||payload.default_difficulties[inst.value]||'expert'" in page
    assert '"lead": 2.0' in page and "song '+Math.max(0,time-payload.lead)" in page
    assert "../candidate/audio/song.wav" in page


def test_midi_comparison_preview_aligns_reference_to_generated_audio(
    monkeypatch,tmp_path
):
    generated=tmp_path/"generated"
    for part in ("analysis","metadata","source","audio"):
        (generated/part).mkdir(parents=True,exist_ok=True)
    generated_note={
        "start":1.0,
        "end":1.1,
        "lanes":[0],
        "confidence":.8,
        "source_id":"generated",
    }
    (generated/"analysis"/"charts.json").write_text(json.dumps({
        "guitar":{
            "expert":[
                generated_note,
                {**generated_note,"start":1.25,"end":1.35},
            ],
            "hard":[generated_note],
        }
    }),encoding="utf-8")
    (generated/"analysis"/"tempo.json").write_text(json.dumps({
        "beats":[0,.5,1,1.5,2],
        "numerator":4,
    }),encoding="utf-8")
    (generated/"metadata"/"song.json").write_text(json.dumps({
        "song_id":"song-id",
        "title":"Cross Difficulty",
        "artist":"Tester",
        "lead_seconds":2.0,
    }),encoding="utf-8")
    (generated/"source"/"source.json").write_text(
        json.dumps({"sha256":"abc"}),encoding="utf-8"
    )
    (generated/"audio"/"song.wav").write_bytes(b"audio")
    reference=tmp_path/"reference.mid"
    reference.write_bytes(b"midi")
    monkeypatch.setattr(
        "rb3charts.datasets._read_expert_gems",
        lambda path,chart_format,difficulty:[
            {"start":.5,"end":.6,"lane":4}
        ],
    )

    output=tmp_path/"review"/"hard-vs-expert.html"
    report=create_midi_comparison_preview(
        generated,
        reference,
        output,
        generated_difficulty="hard",
        reference_difficulty="expert",
        offset=-1.1,
    )

    page=output.read_text(encoding="utf-8")
    assert report["reference_attacks"]==1
    assert report["alignment_offset_seconds"]==-1.1
    assert "A · Generated Hard" in page
    assert "B · Official Expert" in page
    assert '"t": 3.6' in page
    assert '"different": false' in page
    assert (
        tmp_path/"review"/"hard-vs-expert"/"audio"/"song.wav"
    ).is_file()

    abc_output=tmp_path/"review"/"abc.html"
    abc_report=create_expert_hard_reference_preview(
        generated,reference,abc_output,offset=-1.1
    )
    abc_page=abc_output.read_text(encoding="utf-8")
    assert abc_report["counts"]=={
        "Project Expert":2,
        "Project Hard":1,
        "Official Expert":1,
    }
    assert "A · Project Expert" in abc_page
    assert "B · Project Hard" in abc_page
    assert "C · Official Expert" in abc_page
    assert "Highway speed" in abc_page and "Hyper 5" in abc_page
    assert "highway.onchange=()=>lookAhead=+highway.value" in abc_page
    assert "Playback speed" in abc_page
    assert "selectVersion('generated_hard')" in abc_page
    assert '"t": 3.6' in abc_page


def test_midi_preserves_hold_note_duration(tmp_path):
    import pretty_midi
    tempo=TempoMap([0,.5,1,1.5,2,2.5,3])
    note=Note(.5,1.5,(2,),source_id="hold")
    path=tmp_path/"holds.mid"
    lead=write_midi(path,tempo,{"guitar":{d:[note] for d in ("expert","hard","medium","easy")}},3,{})
    midi=pretty_midi.PrettyMIDI(str(path))
    part=next(instrument for instrument in midi.instruments if instrument.name=="PART GUITAR")
    expert=next(n for n in part.notes if n.pitch==98)
    assert expert.start-lead==pytest.approx(.5,abs=.002)
    assert expert.end-expert.start==pytest.approx(1.0,abs=.002)

def test_auto_profile_detects_guitar_led_band(tmp_path):
    sr=8000
    stems={}
    for name,level in (("guitar",.3),("other",.03),("piano",.01)):
        path=tmp_path/f"{name}.wav"; sf.write(path,np.ones((sr,1),dtype=np.float32)*level,sr); stems[name]=path
    profile,evidence=detect_chart_profile(stems,"Alternative")
    assert profile=="band" and evidence["guitar_to_alternate_ratio"]>1.35

def test_auto_profile_detects_synth_led_song(tmp_path):
    sr=8000
    stems={}
    for name,level in (("guitar",.01),("other",.25),("piano",.02)):
        path=tmp_path/f"{name}.wav"; sf.write(path,np.ones((sr,1),dtype=np.float32)*level,sr); stems[name]=path
    profile,evidence=detect_chart_profile(stems,"Pop")
    assert profile=="electronic" and "exceeds guitar" in evidence["reason"]

def test_auto_profile_respects_electronic_genre_hint(tmp_path):
    profile,evidence=detect_chart_profile({},"Electronic Dance")
    assert profile=="electronic" and "genre" in evidence["reason"]
