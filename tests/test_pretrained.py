import json

import numpy as np
import pytest
import soundfile as sf

from rb3charts.pretrained import (
    audio2chart_chart_to_bundle,
    audio2chart_tokens_to_bundle,
    compare_bundles,
    import_event_bundle,
    sha256_file,
    validate_bundle,
    validate_rights_manifest,
)


def _bundle(events, *, name="fixture", duration=5.0, source_hash="a" * 64):
    return {
        "schema_version": 1,
        "backend": {
            "name": name,
            "upstream": "https://example.invalid/model",
            "upstream_revision": "source-revision",
            "model_id": "fixture/model",
            "model_revision": "model-revision",
            "weights_sha256": "b" * 64,
        },
        "source": {
            "sha256": source_hash,
            "sample_rate": 8000,
            "duration_seconds": duration,
        },
        "inference": {
            "device": "cpu",
            "seed": 7,
            "sampling": {"temperature": 0.0},
            "runtime_seconds": 1.25,
        },
        "timebase": {
            "unit": "seconds",
            "origin": "source_audio",
            "grid_ms": 20,
            "offset_applied_seconds": 0.0,
        },
        "instrument": "guitar",
        "difficulty": "expert",
        "tempo": {
            "beats_seconds": [i * 0.5 for i in range(12)],
            "numerator": 4,
            "provenance": "shared synthetic tempo fixture",
        },
        "events": events,
    }


def _event(event_id, onset, lanes=(0,), offset=None, confidence=None):
    return {
        "event_id": event_id,
        "onset_seconds": onset,
        "offset_seconds": offset,
        "duration_status": "predicted" if offset is not None else "unknown",
        "lanes": list(lanes),
        "confidence": confidence,
        "open_note": False,
        "modifiers": [],
    }


def _fake_audio2chart_sequences(duration, grid_ms=20):
    total_samples = round(duration * 24000)
    chunk_samples = 30 * 24000
    starts = list(range(0, total_samples, chunk_samples))
    if starts[-1] + chunk_samples > total_samples:
        starts[-1] = max(0, total_samples - chunk_samples)
    full = 30_000 // grid_ms
    source_times = [1.0, 29.98]
    if duration > 30:
        source_times.extend([30.0, min(duration - 0.02, 30.02), duration - 0.02])
    sequences = []
    for start in starts:
        tokens = [34] * full
        for seconds in source_times:
            local = seconds - start / 24000
            index = round(local * 1000 / grid_ms)
            if 0 <= index < full:
                tokens[index] = 0
        sequences.append(tokens)
    return sequences, total_samples


@pytest.mark.parametrize("duration", [30.0, 30.02, 45.0, 60.0, 61.0])
def test_audio2chart_chunk_normalization_keeps_source_timing_without_overlap(duration):
    sequences, total_samples = _fake_audio2chart_sequences(duration)
    bundle = audio2chart_tokens_to_bundle(
        sequences,
        total_samples=total_samples,
        source_sha256="a" * 64,
        source_sample_rate=44100,
        source_duration_seconds=duration,
        pad_token_id=34,
        grid_ms=20,
        temperature=0.5,
        top_k=32,
    )
    onsets = [event["onset_seconds"] for event in bundle["events"]]
    assert onsets == sorted(set(onsets))
    assert all(onset < duration for onset in onsets)
    assert bundle["backend"]["upstream_revision"] == "b6abaafecd3d39a11793ca64538431263dcd99ad"
    assert all(event["duration_status"] == "unknown" for event in bundle["events"])
    assert all(event["confidence"] is None for event in bundle["events"])
    if duration == 30.02:
        assert onsets == [1.0, 29.98, 30.0]
    if duration == 45.0:
        assert onsets == [1.0, 29.98, 30.0, 30.02, 44.98]
    if duration == 61.0:
        assert onsets[-1] == 60.98


def test_audio2chart_requires_full_chunk_and_rejects_unsupported_tokens():
    with pytest.raises(ValueError, match="at least 30 seconds"):
        audio2chart_tokens_to_bundle(
            [], total_samples=719999, source_sha256="a" * 64,
            source_sample_rate=44100, source_duration_seconds=29.999,
            pad_token_id=34, grid_ms=20,
        )
    sequences, total_samples = _fake_audio2chart_sequences(30.0)
    sequences[0][50] = 31
    with pytest.raises(ValueError, match="open note"):
        audio2chart_tokens_to_bundle(
            sequences, total_samples=total_samples, source_sha256="a" * 64,
            source_sample_rate=44100, source_duration_seconds=30.0,
            pad_token_id=34, grid_ms=20,
        )
    sequences[0][50] = 32
    with pytest.raises(ValueError, match="unknown token"):
        audio2chart_tokens_to_bundle(
            sequences, total_samples=total_samples, source_sha256="a" * 64,
            source_sample_rate=44100, source_duration_seconds=30.0,
            pad_token_id=34, grid_ms=20,
        )


def test_audio2chart_uses_checkpoint_grid_and_keeps_repeated_boundary_attacks():
    duration = 45.0
    total_samples = int(duration * 24000)
    sequences = [[34] * 750, [34] * 750]
    sequences[0][749] = 0
    sequences[1][375] = 0
    bundle = audio2chart_tokens_to_bundle(
        sequences,
        total_samples=total_samples,
        source_sha256="a" * 64,
        source_sample_rate=44100,
        source_duration_seconds=duration,
        pad_token_id=34,
        grid_ms=40,
    )
    assert [event["onset_seconds"] for event in bundle["events"]] == [29.96, 30.0]


def test_audio2chart_chart_normalizer_preserves_chords_and_reported_sustain(tmp_path):
    chart = tmp_path / "notes.chart"
    chart.write_text(
        """[Song]
{
  Name = "Synthetic"
  Offset = 0
  Resolution = 480
}
[SyncTrack]
{
  0 = TS 4
  0 = B 200000
}
[ExpertSingle]
{
  480 = N 0 0
  960 = N 1 240
  960 = N 3 240
}
""",
        encoding="utf-8",
    )
    bundle = audio2chart_chart_to_bundle(
        chart,
        source_sha256="a" * 64,
        source_sample_rate=8000,
        source_duration_seconds=5,
        tempo=_bundle([])["tempo"],
        temperature=0.5,
        top_k=32,
    )
    assert bundle["events"][0]["onset_seconds"] == pytest.approx(0.3)
    assert bundle["events"][0]["duration_status"] == "unknown"
    assert bundle["events"][1]["onset_seconds"] == pytest.approx(0.6)
    assert bundle["events"][1]["offset_seconds"] == pytest.approx(0.75)
    assert bundle["events"][1]["lanes"] == [1, 3]
    assert bundle["timebase"]["chart_resolution"] == 480
    assert bundle["timebase"]["chart_time_signatures"] == [[0, 4, 4]]


def test_audio2chart_chart_normalizer_rejects_open_frets_and_nonzero_offset(tmp_path):
    chart = tmp_path / "notes.chart"
    chart.write_text(
        '[Song]\n{\nOffset = 0\nResolution = 480\n}\n'
        '[SyncTrack]\n{\n0 = B 200000\n}\n'
        '[ExpertSingle]\n{\n480 = N 7 0\n}\n',
        encoding="utf-8",
    )
    kwargs = {
        "source_sha256": "a" * 64,
        "source_sample_rate": 8000,
        "source_duration_seconds": 5,
        "tempo": _bundle([])["tempo"],
    }
    with pytest.raises(ValueError, match="open note"):
        audio2chart_chart_to_bundle(chart, **kwargs)
    chart.write_text(chart.read_text().replace("Offset = 0", "Offset = 0.25"), encoding="utf-8")
    with pytest.raises(ValueError, match="Nonzero chart Offset"):
        audio2chart_chart_to_bundle(chart, **kwargs)


def test_audio2chart_chart_normalizer_rejects_malformed_sync_entries(tmp_path):
    chart = tmp_path / "notes.chart"
    chart.write_text(
        '[Song]\n{\nOffset = 0\nResolution = 480\n}\n'
        '[SyncTrack]\n{\n0 = B 200000\nBPM 140\n}\n'
        '[ExpertSingle]\n{\n480 = N 0 0\n}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported SyncTrack event"):
        audio2chart_chart_to_bundle(
            chart,
            source_sha256="a" * 64,
            source_sample_rate=8000,
            source_duration_seconds=5,
            tempo=_bundle([])["tempo"],
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p["events"][0].update(onset_seconds=float("nan")), "onset_seconds"),
        (lambda p: p["events"][0].update(lanes=[7]), "lanes"),
        (lambda p: p["events"][0].update(open_note=True), "open note"),
        (lambda p: p["events"][0].update(modifiers=["unknown"]), "modifiers"),
        (lambda p: p["events"][0].update(offset_seconds=2), "unknown duration"),
        (lambda p: p["events"].append(p["events"][0].copy()), "unique"),
        (lambda p: p["timebase"].update(offset_applied_seconds=-0.1), "offset_applied_seconds"),
    ],
)
def test_bundle_validation_rejects_malformed_or_unsupported_events(mutate, message):
    payload = _bundle([_event("e1", 1.0)])
    mutate(payload)
    with pytest.raises(ValueError, match=message):
        validate_bundle(payload)


def test_comparison_uses_one_to_one_matching_and_marks_duration_unavailable():
    predicted = _bundle([_event("p1", 0.0), _event("p2", 0.04, lanes=(1,))], name="pred")
    reference = _bundle([_event("r1", 0.02), _event("r2", 0.08, lanes=(1,))], name="truth")
    metrics = compare_bundles(predicted, reference, tolerance_ms=50)
    assert metrics["one_to_one_matches"] == 2
    assert metrics["onset"]["precision"] == 1
    assert metrics["onset"]["recall"] == 1
    assert metrics["exact_lane_set_rate_on_matched_events"] == 1
    assert metrics["duration"]["available"] is False
    assert metrics["playability_or_human_quality"]["available"] is False


def test_comparison_minimizes_error_without_losing_maximum_match_count():
    nearest_prediction = _bundle([_event("p", 0.1)], name="pred")
    alternatives = _bundle(
        [_event("early", 0.06, lanes=(1,)), _event("exact", 0.1)], name="truth"
    )
    nearest = compare_bundles(nearest_prediction, alternatives, tolerance_ms=50)
    assert nearest["onset"]["mean_absolute_error_ms"] == 0
    assert nearest["exact_lane_set_rate_on_matched_events"] == 1

    crowded_predictions = _bundle(
        [_event("p0", 0.05), _event("p1", 0.1, lanes=(1,))], name="pred"
    )
    crowded_truth = _bundle(
        [_event("t0", 0.0, lanes=(2,)), _event("t1", 0.06, lanes=(1,))], name="truth"
    )
    crowded = compare_bundles(crowded_predictions, crowded_truth, tolerance_ms=50)
    assert crowded["one_to_one_matches"] == 2


def test_comparison_without_reference_events_does_not_report_perfect_scores():
    prediction = _bundle([_event("p1", 1.0)], name="pred")
    reference = _bundle([], name="truth")
    metrics = compare_bundles(prediction, reference)
    assert metrics["onset"]["available"] is False
    assert metrics["onset"]["precision"] is None
    assert metrics["onset"]["recall"] is None


def test_comparison_with_no_matching_attacks_reports_zero_coverage():
    prediction = _bundle([_event("p1", 1.0)], name="pred")
    reference = _bundle([_event("r1", 3.0)], name="truth")
    metrics = compare_bundles(prediction, reference, tolerance_ms=50)
    assert metrics["onset"]["available"] is True
    assert metrics["onset"]["precision"] == 0
    assert metrics["onset"]["recall"] == 0
    assert metrics["onset"]["f1"] == 0
    assert metrics["exact_lane_set_rate_on_matched_events"] is None


def test_comparison_requires_an_explicit_shared_tempo_map():
    prediction = _bundle([_event("p1", 1.0)], name="pred")
    reference = _bundle([_event("r1", 1.0)], name="truth")
    prediction["tempo"] = None
    reference["tempo"] = None
    with pytest.raises(ValueError, match="same recorded tempo map"):
        compare_bundles(prediction, reference)


def test_rights_manifest_requires_held_out_attested_hashes(tmp_path):
    prediction = _bundle([_event("p1", 1.0)])
    reference_path = tmp_path / "reference.json"
    reference_path.write_text("{}", encoding="utf-8")
    manifest = {
        "manifest_version": 1,
        "evaluation_split": "held_out",
        "rights_attested": True,
        "audio_rights": "Authorized synthetic fixture",
        "reference_chart_rights": "Authorized synthetic fixture",
        "audio_sha256": prediction["source"]["sha256"],
        "reference_bundle_sha256": sha256_file(reference_path),
    }
    validate_rights_manifest(manifest, prediction, reference_path)
    manifest["evaluation_split"] = "train"
    with pytest.raises(ValueError, match="held-out"):
        validate_rights_manifest(manifest, prediction, reference_path)


def test_event_bundle_import_reaches_existing_midi_validation_and_preview(tmp_path):
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros((8000 * 5, 1), dtype=np.float32), 8000)
    payload = _bundle(
        [
            _event("n0", 1.0, lanes=(0,)),
            _event("n1", 1.5, lanes=(1,)),
            _event("n2", 2.0, lanes=(2,)),
            _event("n3", 2.5, lanes=(3,)),
        ],
        duration=5.0,
        source_hash=sha256_file(source),
    )
    payload["source"]["sample_rate"] = 8000
    bundle_path = tmp_path / "events.json"
    bundle_path.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "review"
    result = import_event_bundle(source, bundle_path, output, title="Synthetic", artist="Fixture")
    assert result["status"] == "draft_ready"
    assert result["validation"]["passed"] is True
    assert (output / "midi" / "song.mid").is_file()
    assert (output / "review.html").is_file()
    assert (output / "audio" / "timing-audit-guitar.wav").is_file()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        import_event_bundle(source, bundle_path, output)


def test_event_bundle_import_does_not_replace_output_created_during_publish(tmp_path, monkeypatch):
    from rb3charts import pretrained

    source = tmp_path / "source.wav"
    sf.write(source, np.zeros((8000 * 5, 1), dtype=np.float32), 8000)
    payload = _bundle(
        [_event(f"n{i}", 1.0 + i * 0.5, lanes=(i,)) for i in range(4)],
        duration=5.0,
        source_hash=sha256_file(source),
    )
    payload["source"]["sample_rate"] = 8000
    bundle_path = tmp_path / "events.json"
    bundle_path.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "review"
    original_publish = pretrained._publish_directory_no_replace

    def race_publish(staging, target):
        target.mkdir()
        original_publish(staging, target)

    monkeypatch.setattr(pretrained, "_publish_directory_no_replace", race_publish)
    with pytest.raises(FileExistsError):
        import_event_bundle(source, bundle_path, output)
    assert output.is_dir()
    assert list(output.iterdir()) == []


def test_event_bundle_import_rejects_a_different_source_wav(tmp_path):
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros((8000 * 5, 1), dtype=np.float32), 8000)
    payload = _bundle(
        [_event("n0", 1.0)],
        duration=5.0,
        source_hash="a" * 64,
    )
    payload["source"]["sample_rate"] = 8000
    bundle_path = tmp_path / "events.json"
    bundle_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        import_event_bundle(source, bundle_path, tmp_path / "review")
    assert not (tmp_path / "review").exists()
