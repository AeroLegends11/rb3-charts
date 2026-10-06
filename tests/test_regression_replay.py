import hashlib
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
    audit_playability_evidence,
    benchmark_all_replays,
    benchmark_replays,
    assess_official_quality_release,
    build_comparison_index,
    compare_chart_attacks,
    compare_comparison_indexes,
    compare_reference_midi,
    estimate_attack_offset,
    _authorized_comparison_markdown,
    _paired_source_evidence_markdown,
    _reference_comparison_markdown,
    _snapshot_attacks,
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


def test_cached_replay_fretnet_shadow_is_diagnostic_only(tmp_path):
    source=tmp_path/"source-run"
    _write_replay_fixture(source,profile="band",instruments=["guitar"])
    baseline=replay_cached(
        source,
        output=tmp_path/"baseline",
        progress=lambda _:None,
        compile=False,
    )
    shadow=tmp_path/"shadow.json"
    shadow.write_text(json.dumps({
        "schema":1,
        "kind":"fretnet_shadow_evidence",
        "pitch_runs":[
            {
                "source_id":f"shadow-{index}",
                "onset_s":1+index*.25,
                "offset_s":1+index*.25+.1,
                "midi_pitch":60,
            }
            for index in range(8)
        ],
    }),encoding="utf-8")

    candidate=replay_cached(
        source,
        output=tmp_path/"candidate",
        progress=lambda _:None,
        compile=False,
        fretnet_shadow=shadow,
    )

    baseline_charts=json.loads(
        (Path(baseline["folder"])/"analysis"/"charts.json").read_text(
            encoding="utf-8"
        )
    )
    candidate_folder=Path(candidate["folder"])
    candidate_charts=json.loads(
        (candidate_folder/"analysis"/"charts.json").read_text(encoding="utf-8")
    )
    snapshots=json.loads(
        (candidate_folder/"analysis"/"guitar-stage-snapshots.json").read_text(
            encoding="utf-8"
        )
    )
    report=json.loads(
        (candidate_folder/"analysis"/"guitar-fretnet-shadow.json").read_text(
            encoding="utf-8"
        )
    )

    assert candidate_charts==baseline_charts
    assert any(
        stage["stage"]=="fretnet_shadow_diagnostic"
        for stage in snapshots["stages"]
    )
    assert report["diagnostic_only"] is True


def test_cached_replay_recomputes_band_lead_from_cached_stem(
    tmp_path,monkeypatch
):
    source=tmp_path/"source-run"
    _write_replay_fixture(source,profile="band",instruments=["guitar"])
    stem=tmp_path/"guitar.wav"
    sf.write(stem,np.zeros(44100,dtype=np.float32),44100)
    (source/"analysis"/"stems.json").write_text(
        json.dumps({"guitar":str(stem)}),encoding="utf-8"
    )
    calls=[]

    def focus(path,events,beat_period):
        calls.append((path,len(events),beat_period))
        return [dict(event,pitch=60) for event in events],{
            "candidates":len(events),
            "retained":len(events),
        }

    monkeypatch.setattr("rb3charts.pipeline.band_guitar_lead_events",focus)
    result=replay_cached(
        source,output=tmp_path/"output",progress=lambda _:None,compile=False
    )
    replay_folder=Path(result["folder"])
    transcription=json.loads(
        (replay_folder/"analysis"/"guitar-transcription.json").read_text(
            encoding="utf-8"
        )
    )

    assert calls and calls[0][0]==stem
    assert {event["pitch"] for event in transcription}=={60}


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
        (root/"analysis"/"patterns.json").write_text(
            json.dumps({instrument:{
                "attacks":3,
                "recurring_pattern_families":2,
                "recurring_pattern_occurrences":5,
                "pattern_supported_attacks":2,
                "contour_agreement_ratio":.75,
            }}),
            encoding="utf-8",
        )

    write_run(tmp_path/"baseline"/"run",(0,1,2))
    write_run(tmp_path/"candidate"/"run",(0,0,2))

    report=benchmark_replays(
        tmp_path/"baseline",tmp_path/"candidate",tmp_path/"benchmark.json"
    )

    assert report["songs_matched"]==1
    assert report["schema_version"]==2
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
    assert report["pattern_macro_song_mean"]["songs_matched"]==1
    assert (
        report["pattern_macro_song_mean"]["baseline"]
        ["pattern_supported_attack_ratio"]==.6667
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


def test_benchmark_replays_rejects_duplicate_song_keys(tmp_path):
    for name in ("one","two"):
        root=tmp_path/"baseline"/name
        (root/"metadata").mkdir(parents=True)
        (root/"analysis").mkdir()
        (root/"metadata"/"song.json").write_text(json.dumps({
            "artist":"Same Artist",
            "title":"Same Song",
        }),encoding="utf-8")
        (root/"analysis"/"guitar-stage-snapshots.json").write_text(
            json.dumps({
                "instrument":"guitar",
                "stages":[{"stage":"final","notes":[]}],
            }),
            encoding="utf-8",
        )
    with pytest.raises(ValueError,match="Duplicate replay song"):
        benchmark_replays(
            tmp_path/"baseline",
            tmp_path/"candidate",
            tmp_path/"report.json",
        )


def test_benchmark_replays_rejects_missing_song_identity(tmp_path):
    root=tmp_path/"baseline"/"run"
    (root/"metadata").mkdir(parents=True)
    (root/"analysis").mkdir()
    (root/"metadata"/"song.json").write_text(
        json.dumps({"artist":"","title":"Untitled"}),
        encoding="utf-8",
    )
    (root/"analysis"/"guitar-stage-snapshots.json").write_text(
        json.dumps({
            "instrument":"guitar",
            "stages":[{"stage":"final","notes":[]}],
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError,match="requires artist and title"):
        benchmark_replays(
            tmp_path/"baseline",
            tmp_path/"candidate",
            tmp_path/"report.json",
        )


def test_benchmark_all_replays_reports_available_instruments(tmp_path):
    for side in ("baseline","candidate"):
        root=tmp_path/side/"run"
        (root/"metadata").mkdir(parents=True)
        (root/"analysis").mkdir()
        (root/"metadata"/"song.json").write_text(json.dumps({
            "artist":"RB3 Charts",
            "title":"All Instrument Demo",
        }),encoding="utf-8")
        notes=[{
            "source_id":"one",
            "start":0,
            "end":.1,
            "lanes":[0],
            "pitch":48,
        }]
        (root/"analysis"/"guitar-stage-snapshots.json").write_text(
            json.dumps({
                "instrument":"guitar",
                "stages":[{"stage":"final","notes":notes}],
            }),
            encoding="utf-8",
        )
    report=benchmark_all_replays(
        tmp_path/"baseline",
        tmp_path/"candidate",
        tmp_path/"all.json",
    )
    assert set(report["instruments"])=={"guitar"}
    assert set(report["unavailable"])=={"bass","drums","keys"}
    assert (tmp_path/"all.json").is_file()


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


def test_motif_comparison_reports_source_pitch_contour_ceiling():
    reference = [
        {"time": index * .25, "lanes": (index % 4,), "duration": .1}
        for index in range(8)
    ]
    candidate = [
        {
            "time": row["time"],
            "lanes": row["lanes"],
            "duration": row["duration"],
            "pitch": 60 + index % 4,
        }
        for index, row in enumerate(reference)
    ]

    motifs = compare_chart_attacks(reference, candidate)["motifs"]

    assert motifs["source_pitch_covered_occurrences"] == 2
    assert motifs["source_pitch_contour_agreement"] == 1


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


def test_paired_source_markdown_labels_external_calibration_without_paths():
    evidence={
        "status":"available",
        "scope":"external_rights_cleared_calibration",
        "report":r"C:\private\agreement.json",
        "report_sha256":"a"*64,
        "source_dataset":{"id":"guitarset","version":"1.1.0"},
        "configuration":{"tolerance_ms":50},
        "lineage":{
            "manifest":r"C:\private\manifest.json",
            "manifest_sha256":"b"*64,
            "release":{"published_md5":"c"*32},
        },
        "target_coverage":{
            "status":"unavailable",
            "paired_items":0,
            "total_items":28,
        },
        "splits":{
            "validation":{
                "shared":{"precision":.9,"recall":.82,"f1":.8581},
                "best_source":{"channel":"pickup","f1":.85},
                "delta_vs_best_source":{
                    "precision":.05,"recall":-.03,"f1":.0081,
                },
            },
        },
    }

    markdown=_paired_source_evidence_markdown(evidence)

    assert "External rights-cleared calibration only" in markdown
    assert "not generated-song evidence" in markdown
    assert "unavailable** (0/28 items)" in markdown
    assert "| validation | 0.9000 | 0.8200 | 0.8581 |" in markdown
    assert r"C:\private" not in markdown


def test_paired_source_markdown_omits_metrics_for_unavailable_or_invalid_lineage():
    hidden={
        "validation":{
            "shared":{"precision":.9999,"recall":.9999,"f1":.9999},
        },
    }
    unavailable=_paired_source_evidence_markdown({
        "status":"unavailable",
        "target_coverage":{"paired_items":0,"total_items":28},
        "splits":hidden,
    })
    invalid=_paired_source_evidence_markdown({
        "status":"invalid",
        "reason":"manifest hash does not match",
        "target_coverage":{"paired_items":0,"total_items":28},
        "splits":hidden,
    })

    assert "does not imply" in unavailable
    assert "0.9999" not in unavailable
    assert "Metrics omitted" in invalid
    assert "manifest hash does not match" in invalid
    assert "0.9999" not in invalid


def test_authorized_comparison_markdown_includes_paired_source_state():
    markdown=_authorized_comparison_markdown({
        "song_id":"authorized-song",
        "manifest_sha256":"d"*64,
        "difficulties":{
            "expert":{
                "attack":{
                    "reference":100,
                    "candidate":90,
                    "matched":80,
                    "precision":.8889,
                    "recall":.8,
                    "f1":.8421,
                },
            },
        },
        "paired_source_agreement":{
            "status":"unavailable",
            "target_coverage":{"paired_items":0,"total_items":28},
        },
    })

    assert "Authorized human-chart comparison" in markdown
    assert "authorized human notes were not generation inputs" in markdown
    assert "Lineage status: **unavailable**" in markdown


def test_reference_comparison_markdown_omits_local_paths_and_note_rows():
    report={
        "provenance":{
            "generated_midi":r"C:\private\generated\notes.mid",
            "generated_sha256":"a"*64,
            "reference_midi":r"C:\excluded\official\reference.mid",
            "reference_sha256":"b"*64,
            "difficulty":"expert",
            "alignment":"automatic-onsets-only",
            "alignment_offset_seconds":.125,
        },
        "metrics":{
            "attack":{
                "reference":100,
                "candidate":90,
                "matched":80,
                "precision":.8889,
                "recall":.8,
                "f1":.8421,
            },
            "lane":{
                "exact_agreement":.75,
                "movement_direction_agreement":.7,
            },
            "attack_windows":{
                "windows":[{
                    "start":0,
                    "end":4,
                    "reference":10,
                    "candidate":9,
                    "matched":8,
                    "precision":.8889,
                    "recall":.8,
                    "f1":.8421,
                }],
            },
        },
        "stage_metrics":[],
    }

    markdown=_reference_comparison_markdown(report)

    assert "Generated MIDI: `notes.mid`" in markdown
    assert "Reference MIDI: `reference.mid`" in markdown
    assert "Generated SHA-256" in markdown
    assert "reference note rows are not embedded" in markdown
    assert r"C:\private" not in markdown
    assert r"C:\excluded" not in markdown


def test_reference_comparison_supports_cross_difficulty(monkeypatch,tmp_path):
    generated=tmp_path/"generated.mid"
    reference=tmp_path/"reference.mid"
    generated.write_bytes(b"generated")
    reference.write_bytes(b"reference")
    calls=[]

    def read_gems(path, chart_format, difficulty):
        calls.append((Path(path).name,chart_format,difficulty))
        count=2 if Path(path)==generated else 3
        return [
            {"start":float(index),"end":float(index)+.1,"lane":index}
            for index in range(count)
        ]

    monkeypatch.setattr("rb3charts.datasets._read_expert_gems",read_gems)
    report=compare_reference_midi(
        generated,
        reference,
        tmp_path/"comparison.json",
        difficulty="hard",
        reference_difficulty="expert",
        offset=0,
    )

    assert calls==[
        ("reference.mid","rock_band_midi","expert"),
        ("generated.mid","rock_band_midi","hard"),
    ]
    assert report["metrics"]["attack"]["reference"]==3
    assert report["metrics"]["attack"]["candidate"]==2
    assert report["provenance"]["generated_difficulty"]=="hard"
    assert report["provenance"]["reference_difficulty"]=="expert"
    markdown=(tmp_path/"comparison.md").read_text(encoding="utf-8")
    assert "Generated difficulty: **Hard Guitar**" in markdown
    assert "Reference difficulty: **Expert Guitar**" in markdown


def test_reference_comparison_recovers_run_lead_from_midi_file(monkeypatch,tmp_path):
    generated=tmp_path/"run"
    (generated/"midi").mkdir(parents=True)
    (generated/"metadata").mkdir()
    generated_midi=generated/"midi"/"notes.mid"
    generated_midi.write_bytes(b"generated")
    (generated/"metadata"/"song.json").write_text(
        json.dumps({"lead_seconds":3.5}),
        encoding="utf-8",
    )
    reference=tmp_path/"reference.mid"
    reference.write_bytes(b"reference")

    def read_gems(path, chart_format, difficulty):
        start=4.5 if Path(path)==generated_midi else 1.0
        return [{"start":start,"end":start+.1,"lane":0}]

    monkeypatch.setattr("rb3charts.datasets._read_expert_gems",read_gems)
    report=compare_reference_midi(
        generated_midi,
        reference,
        offset=0,
    )

    assert report["provenance"]["generated_audio_lead_seconds"]==3.5
    assert report["metrics"]["attack"]["matched"]==1


def _write_index_run(root, name, song_id, schema=None, title="Song", artist="Artist"):
    run=root/name
    (run/"metadata").mkdir(parents=True)
    (run/"analysis").mkdir()
    (run/"midi").mkdir()
    generated_midi=run/"midi"/"notes.mid"
    generated_midi.write_bytes(f"generated {song_id}".encode())
    generated_hash=hashlib.sha256(generated_midi.read_bytes()).hexdigest()
    (run/"metadata"/"song.json").write_text(json.dumps({
        "song_id":song_id,
        "title":title,
        "artist":artist,
    }),encoding="utf-8")
    (run/"validation_report.json").write_text(
        json.dumps({"passed":True}),encoding="utf-8"
    )
    if schema is not None:
        metrics=compare_chart_attacks(
            [{"time":0.0,"lanes":(0,),"duration":.5}],
            [{"time":0.02,"lanes":(0,),"duration":.5}],
        )
        if schema=="rb3charts-authorized-chart-comparison":
            report={
                "schema":schema,
                "schema_version":1,
                "song_id":song_id,
                "generated_folder":str(run),
                "generated_identity":{
                    "song_id":song_id,
                    "title":title,
                    "artist":artist,
                },
                "generated_midi_sha256":generated_hash,
                "difficulties":{"expert":metrics},
                "paired_source_agreement":{
                    "status":"available",
                    "scope":"external_rights_cleared_calibration",
                },
            }
            filename="human-chart-comparison.json"
        else:
            report={
                "schema":schema,
                "schema_version":1,
                "provenance":{
                    "generated_midi":str(run/"midi"/"notes.mid"),
                    "generated_sha256":generated_hash,
                    "reference_midi":r"C:\excluded\reference.mid",
                },
                "metrics":metrics,
                "stage_metrics":[],
            }
            filename="reference-comparison.json"
        path=run/"analysis"/filename
        path.write_text(json.dumps(report),encoding="utf-8")
        path.with_suffix(".md").write_text("# Local review\n",encoding="utf-8")
    return run


def _shift_comparison_metrics(run, delta):
    path=next((run/"analysis").glob("*-comparison.json"))
    report=json.loads(path.read_text(encoding="utf-8"))
    metrics=(
        report["difficulties"]["expert"]
        if "difficulties" in report else report["metrics"]
    )
    metrics["attack"]["f1"]=round(metrics["attack"]["f1"]+delta,4)
    weakest=metrics["attack_windows"]["worst_reference_recall"]
    weakest["recall"]=round(weakest["recall"]+delta,4)
    path.write_text(json.dumps(report),encoding="utf-8")


def test_comparison_index_sanitizes_reports_and_keeps_missing_runs(tmp_path):
    _write_index_run(
        tmp_path,
        "run one",
        "song-one",
        "rb3charts-authorized-chart-comparison",
        title="Pipe | Song\nSecond line",
        artist="Safe Artist",
    )
    _write_index_run(tmp_path,"run-two","song-two")

    index=build_comparison_index(tmp_path)
    json_text=(tmp_path/"comparison-index.json").read_text(encoding="utf-8")
    markdown=(tmp_path/"comparison-index.md").read_text(encoding="utf-8")

    assert index["comparison_schema"]=="rb3charts-authorized-chart-comparison"
    entries={row["identity"]["song_id"]:row for row in index["entries"]}
    assert entries["song-one"]["comparison_status"]=="available"
    assert entries["song-one"]["summary"]["paired_source_status"]=="available"
    assert entries["song-two"]["comparison_status"]=="missing"
    assert entries["song-two"]["summary"] is None
    assert "unmatched" not in json_text
    assert str(tmp_path) not in json_text
    assert "C:\\\\excluded" not in json_text
    assert "Pipe \\| Song Second line" in markdown
    assert "run%20one/analysis/human-chart-comparison.md" in markdown
    assert "Reference-note rows" in markdown


def _release_index_entry(index, passed=True):
    high = .9 if passed else .5
    return {
        "identity": {
            "song_id": f"song-{index}",
            "title": f"Song {index}",
            "artist": f"Artist {index % 5}",
        },
        "comparison_status": "available",
        "summary": {
            "attack": {"f1": high},
            "chord": {"reference": 20, "f1": high},
            "sustain": {"reference": 20, "f1": high},
            "lane": {
                "exact_agreement": high,
                "movement_direction_agreement": high,
                "mean_interval_error": 0.2,
            },
            "motif": {
                "reference_occurrences": 20,
                "coverage": high,
                "relative_lane_shape_agreement": high,
            },
            "weakest_window": {"recall": high},
        },
    }


def test_official_quality_release_requires_eighty_percent_and_broad_coverage(
    tmp_path,
):
    index_path = tmp_path / "comparison-index.json"
    index_path.write_text(json.dumps({
        "comparison_schema": "rb3charts-authorized-chart-comparison",
        "entries": [
            _release_index_entry(index, passed=index < 16)
            for index in range(20)
        ],
    }), encoding="utf-8")

    report = assess_official_quality_release(index_path)

    assert report["songs_passed"] == 16
    assert report["pass_rate"] == .8
    assert report["coverage_ready"] is True
    assert report["release_ready"] is True
    assert (tmp_path / "official-quality-release.md").is_file()


def test_official_quality_release_rejects_insufficient_corpus(tmp_path):
    index_path = tmp_path / "comparison-index.json"
    index_path.write_text(json.dumps({
        "comparison_schema": "rb3charts-authorized-chart-comparison",
        "entries": [
            _release_index_entry(index)
            for index in range(19)
        ],
    }), encoding="utf-8")

    report = assess_official_quality_release(index_path)

    assert report["pass_rate"] == 1
    assert report["coverage_ready"] is False
    assert report["release_ready"] is False


def test_comparison_index_rejects_duplicate_song_identities(tmp_path):
    _write_index_run(
        tmp_path,"run-a","duplicate","rb3charts-reference-comparison"
    )
    _write_index_run(tmp_path,"run-b","duplicate")

    with pytest.raises(ValueError,match="Duplicate song identity duplicate"):
        build_comparison_index(tmp_path)

    assert not (tmp_path/"comparison-index.json").exists()


def test_comparison_index_rejects_mixed_and_malformed_schemas(tmp_path):
    _write_index_run(
        tmp_path,"run-a","one","rb3charts-reference-comparison"
    )
    _write_index_run(
        tmp_path,"run-b","two","rb3charts-authorized-chart-comparison"
    )
    with pytest.raises(ValueError,match="mixes authorized and reference"):
        build_comparison_index(tmp_path)

    root=tmp_path/"malformed"
    run=_write_index_run(root,"run","three")
    (run/"analysis"/"reference-comparison.json").write_text(
        json.dumps({"schema":"unknown","schema_version":1}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError,match="Malformed comparison schema"):
        build_comparison_index(root)


def test_comparison_index_rejects_stale_generated_midi_lineage(tmp_path):
    run=_write_index_run(
        tmp_path,"run","song","rb3charts-authorized-chart-comparison"
    )
    (run/"midi"/"notes.mid").write_bytes(b"changed after comparison")

    with pytest.raises(ValueError,match="generated-MIDI lineage differs"):
        build_comparison_index(tmp_path)

    assert not (tmp_path/"comparison-index.json").exists()


def test_authorized_index_separates_dataset_and_package_song_ids(tmp_path):
    run=_write_index_run(
        tmp_path,"run","dataset-song",
        "rb3charts-authorized-chart-comparison",
    )
    metadata_path=run/"metadata"/"song.json"
    metadata=json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["song_id"]="rb3c_package_id"
    metadata_path.write_text(json.dumps(metadata),encoding="utf-8")
    report_path=run/"analysis"/"human-chart-comparison.json"
    report=json.loads(report_path.read_text(encoding="utf-8"))
    report["generated_identity"]["song_id"]="rb3c_package_id"
    report_path.write_text(json.dumps(report),encoding="utf-8")

    index=build_comparison_index(tmp_path)

    assert index["entries"][0]["identity"]["song_id"]=="dataset-song"


def test_comparison_index_rejects_nested_output(tmp_path):
    _write_index_run(
        tmp_path,"run","song","rb3charts-reference-comparison"
    )

    with pytest.raises(ValueError,match="directly inside"):
        build_comparison_index(
            tmp_path,tmp_path/"nested"/"comparison-index.json"
        )


def test_comparison_trend_pairs_shared_runs_and_reports_asymmetry(tmp_path):
    baseline=tmp_path/"baseline"
    candidate=tmp_path/"candidate"
    _write_index_run(
        baseline,"a-run","a","rb3charts-authorized-chart-comparison",
        title="A Song",artist="Artist",
    )
    _write_index_run(
        baseline,"b-run","b","rb3charts-authorized-chart-comparison",
        title="B Song",artist="Artist",
    )
    _write_index_run(
        baseline,"baseline-only","base","rb3charts-authorized-chart-comparison",
        title="Baseline Only",artist="Artist",
    )
    candidate_a=_write_index_run(
        candidate,"a-run","a","rb3charts-authorized-chart-comparison",
        title="A Song",artist="Artist",
    )
    _shift_comparison_metrics(candidate_a,-.1)
    _write_index_run(
        candidate,"b-run","b",title="B Song",artist="Artist"
    )
    _write_index_run(
        candidate,"candidate-only","cand","rb3charts-authorized-chart-comparison",
        title="Candidate Only",artist="Artist",
    )
    baseline_index=baseline/"comparison-index.json"
    candidate_index=candidate/"comparison-index.json"
    build_comparison_index(baseline,baseline_index)
    build_comparison_index(candidate,candidate_index)

    trend=compare_comparison_indexes(baseline_index,candidate_index)
    json_text=(candidate/"comparison-trend.json").read_text(encoding="utf-8")
    markdown=(candidate/"comparison-trend.md").read_text(encoding="utf-8")

    assert trend["coverage"]=={
        "baseline_identities":3,
        "candidate_identities":3,
        "shared_identities":2,
        "paired_available":1,
        "shared_missing":1,
        "baseline_only":1,
        "candidate_only":1,
        "baseline_only_identities":["base"],
        "candidate_only_identities":["cand"],
    }
    assert trend["macro"]["attack_f1"]["paired_values"]==1
    assert trend["macro"]["attack_f1"]["delta"]==-.1
    assert trend["rankings"]["attack_f1"]["regressions"][0]["identity"]["song_id"]=="a"
    song_b=next(row for row in trend["songs"] if row["identity"]["song_id"]=="b")
    assert song_b["metrics"]["attack_f1"]["status"]=="unavailable"
    assert str(tmp_path) not in json_text
    assert "generated_midi" not in json_text
    assert "reference" not in json_text.lower()
    assert "Baseline-only identities: **1**" in markdown


def test_comparison_trend_preserves_null_metric_as_unavailable(tmp_path):
    baseline=tmp_path/"baseline"
    candidate=tmp_path/"candidate"
    for root in (baseline,candidate):
        run=_write_index_run(
            root,"run","song","rb3charts-reference-comparison"
        )
        if root==candidate:
            report_path=run/"analysis"/"reference-comparison.json"
            report=json.loads(report_path.read_text(encoding="utf-8"))
            report["metrics"]["motifs"]["coverage"]=None
            report_path.write_text(json.dumps(report),encoding="utf-8")
        build_comparison_index(root)
    candidate_index=candidate/"comparison-index.json"

    trend=compare_comparison_indexes(
        baseline/"comparison-index.json",candidate_index
    )

    assert trend["macro"]["motif_coverage"]=={
        "paired_values":0,
        "unavailable_values":1,
        "baseline_mean":None,
        "candidate_mean":None,
        "delta":None,
    }
    assert trend["songs"][0]["metrics"]["motif_coverage"]["status"]=="unavailable"


def test_comparison_trend_rejects_incompatible_or_duplicate_indexes(tmp_path):
    baseline=tmp_path/"baseline"
    candidate=tmp_path/"candidate"
    _write_index_run(
        baseline,"run","song","rb3charts-reference-comparison"
    )
    _write_index_run(
        candidate,"run","song","rb3charts-authorized-chart-comparison"
    )
    build_comparison_index(baseline)
    build_comparison_index(candidate)
    with pytest.raises(ValueError,match="incompatible comparison schemas"):
        compare_comparison_indexes(
            baseline/"comparison-index.json",
            candidate/"comparison-index.json",
        )

    duplicate=json.loads(
        (baseline/"comparison-index.json").read_text(encoding="utf-8")
    )
    duplicate["entries"].append(duplicate["entries"][0])
    duplicate_path=baseline/"duplicate.json"
    duplicate_path.write_text(json.dumps(duplicate),encoding="utf-8")
    with pytest.raises(ValueError,match="Duplicate song identity"):
        compare_comparison_indexes(
            duplicate_path,baseline/"comparison-index.json"
        )


def test_comparison_trend_rejects_stale_report_lineage(tmp_path):
    for name in ("baseline","candidate"):
        root=tmp_path/name
        run=_write_index_run(
            root,"run","song","rb3charts-reference-comparison"
        )
        build_comparison_index(root)
        if name=="candidate":
            report=run/"analysis"/"reference-comparison.json"
            report.write_text("{}\n",encoding="utf-8")

    with pytest.raises(ValueError,match="Stale comparison-report lineage"):
        compare_comparison_indexes(
            tmp_path/"baseline"/"comparison-index.json",
            tmp_path/"candidate"/"comparison-index.json",
        )


def test_comparison_trend_rejects_tampered_index_summary(tmp_path):
    for name in ("baseline","candidate"):
        root=tmp_path/name
        _write_index_run(
            root,"run","song","rb3charts-reference-comparison"
        )
        build_comparison_index(root)
    candidate=tmp_path/"candidate"/"comparison-index.json"
    payload=json.loads(candidate.read_text(encoding="utf-8"))
    payload["entries"][0]["summary"]["attack"]["f1"]=.1234
    candidate.write_text(json.dumps(payload),encoding="utf-8")

    with pytest.raises(ValueError,match="summary differs"):
        compare_comparison_indexes(
            tmp_path/"baseline"/"comparison-index.json",candidate
        )


def test_comparison_trend_rejects_link_escape_and_input_overwrite(tmp_path):
    for name in ("baseline","candidate"):
        root=tmp_path/name
        _write_index_run(
            root,"run","song","rb3charts-reference-comparison"
        )
        build_comparison_index(root)
    baseline=tmp_path/"baseline"/"comparison-index.json"
    candidate=tmp_path/"candidate"/"comparison-index.json"
    payload=json.loads(candidate.read_text(encoding="utf-8"))
    payload["entries"][0]["links"]["json"]="../outside.json"
    candidate.write_text(json.dumps(payload),encoding="utf-8")
    with pytest.raises(ValueError,match="escapes index root"):
        compare_comparison_indexes(baseline,candidate)

    build_comparison_index(tmp_path/"candidate")
    with pytest.raises(ValueError,match="must not overwrite"):
        compare_comparison_indexes(baseline,candidate,candidate)


def test_attack_alignment_and_unmatched_events_are_reported():
    reference=[
        {"time":time,"lanes":(0,),"duration":.1}
        for time in (1,2,3,4)
    ]
    candidate=[
        {"time":time-.37,"lanes":(0,),"duration":.1}
        for time in (1,2,3)
    ]
    offset=estimate_attack_offset(reference,candidate)
    aligned=[
        {**row,"time":row["time"]+offset}
        for row in candidate
    ]
    report=compare_chart_attacks(reference,aligned)
    assert offset==pytest.approx(.37)
    assert report["attack"]["matched"]==3
    assert [row["time"] for row in report["unmatched"]["reference"]]==[4]
    assert report["unmatched"]["candidate"]==[]


def test_snapshot_attacks_group_chord_lanes_and_apply_offset():
    attacks=_snapshot_attacks([
        {"start":1,"end":1.2,"lanes":[0]},
        {"start":1,"end":1.3,"lanes":[2]},
    ],offset=.25)
    assert attacks==[{
        "time":1.25,
        "lanes":(0,2),
        "duration":pytest.approx(.3),
        "pitch":None,
    }]


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


def test_playability_evidence_audit_joins_independent_diagnostics(tmp_path):
    run=tmp_path/"run"
    analysis=run/"analysis"
    source=run/"source"
    analysis.mkdir(parents=True)
    source.mkdir()
    audio=tmp_path/"audio.wav"
    samples=np.zeros(22050,dtype=np.float32)
    samples[10900:11050]=1
    sf.write(audio,samples,22050)
    (source/"source.json").write_text(json.dumps({
        "path":str(audio),
        "sha256":"unused",
    }),encoding="utf-8")
    (analysis/"stems.json").write_text(json.dumps({
        "guitar":str(audio),
    }),encoding="utf-8")
    (analysis/"playability.json").write_text(json.dumps({
        "guitar":{"removal_attribution":{
            "per_beat_cap":{"count":1,"events":[{
                "source_id":"candidate-1",
                "start":.5,
                "beat":1.0,
                "confidence":.6,
                "priority":.7,
                "sixteenth_grid_alignment":1.0,
                "duration_ms":65,
                "sustain_supported":False,
                "motif_supported":False,
            }]},
        }},
    }),encoding="utf-8")
    (analysis/"guitar-transcription.json").write_text(json.dumps([{
        "source_id":"candidate-1",
        "start":.5,
        "end":.565,
        "pitch":60,
        "confidence":.6,
        "model_onset_strength":.8,
        "evidence":{
            "attack_strength":.7,
            "relative_level":1.4,
            "evidence_score":.65,
        },
    }]),encoding="utf-8")
    (analysis/"guitar-voice-continuity.json").write_text(json.dumps({
        "windows":[{
            "start_seconds":0,
            "end_seconds":1,
            "candidate_for_tracking":True,
            "dominant_evidence_share":.7,
            "register_separation":12,
        }],
    }),encoding="utf-8")
    structure=tmp_path/"structure.json"
    structure.write_text(json.dumps({"candidates":[{
        "start":.5,
        "attack_probability":.9,
        "periodic_width_attack_rank_percentile":.8,
        "peak_width_rank_percentile":.75,
    }]}),encoding="utf-8")

    report=audit_playability_evidence(
        run,structure_predictions=structure
    )

    assert report["summary"]["removed_attacks"]==1
    assert report["summary"]["structure_supported"]==1
    assert report["independent_stem_mix_available"] is False
    event=report["events"][0]
    assert event["raw_evidence"]["model_onset_strength"]==.8
    assert event["structure_evidence"]["found"] is True
    assert event["voice_continuity"]["candidate_for_tracking"] is True
    assert (analysis/"playability-evidence.json").is_file()
