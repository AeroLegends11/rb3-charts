import hashlib
import json

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from rb3charts.datasets import (
    DatasetManifestError,
    validate_chart_pair_manifest,
    validate_dataset_manifest,
)
from rb3charts.training import TrainingDataError, convert_authorized_chart_pairs


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(tmp_path, items):
    for item in items:
        for key in ("audio", "annotations"):
            path = tmp_path / item[key]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{item['song_id']}:{key}".encode())
            item[f"{key}_sha256"] = _hash(path)
    payload = {
        "schema_version": 1,
        "dataset": {
            "id": "fixture",
            "version": "1.0",
            "source_url": "https://example.invalid/fixture",
            "audio_license": "CC BY 4.0",
            "annotation_license": "CC BY 4.0",
            "commercial_use": "allowed",
            "attribution": "Fixture authors",
        },
        "items": items,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_dataset_manifest_verifies_files_hashes_and_artist_disjoint_splits(tmp_path):
    path = _write_manifest(tmp_path, [
        {"song_id": "train-song", "artist_id": "artist-a", "split": "train",
         "audio": "audio/train.wav", "annotations": "labels/train.json"},
        {"song_id": "test-song", "artist_id": "artist-b", "split": "test",
         "audio": "audio/test.wav", "annotations": "labels/test.json"},
    ])
    report = validate_dataset_manifest(path, verify_hashes=True)
    assert report["items"] == 2
    assert report["artists"] == 2
    assert report["splits"] == {"test": 1, "train": 1, "validation": 0}
    assert report["hashes_verified"] == 4


def test_dataset_manifest_rejects_artist_leakage(tmp_path):
    path = _write_manifest(tmp_path, [
        {"song_id": "one", "artist_id": "same-artist", "split": "train",
         "audio": "one.wav", "annotations": "one.json"},
        {"song_id": "two", "artist_id": "same-artist", "split": "validation",
         "audio": "two.wav", "annotations": "two.json"},
    ])
    with pytest.raises(DatasetManifestError, match="must not cross"):
        validate_dataset_manifest(path)


def test_dataset_manifest_rejects_paths_outside_dataset_root(tmp_path):
    outside = tmp_path.parent / "outside.wav"
    outside.write_bytes(b"outside")
    item = {
        "song_id": "one",
        "artist_id": "artist",
        "split": "train",
        "audio": "../outside.wav",
        "annotations": "one.json",
    }
    (tmp_path / "one.json").write_bytes(b"labels")
    payload = {
        "schema_version": 1,
        "dataset": {
            "id": "fixture", "version": "1", "source_url": "https://example.invalid",
            "audio_license": "CC BY 4.0", "annotation_license": "CC BY 4.0",
            "commercial_use": "allowed", "attribution": "Fixture authors",
        },
        "items": [dict(
            item,
            audio_sha256=_hash(outside),
            annotations_sha256=_hash(tmp_path / "one.json"),
        )],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DatasetManifestError, match="escapes"):
        validate_dataset_manifest(path)


def test_dataset_manifest_rejects_hash_mismatch(tmp_path):
    path = _write_manifest(tmp_path, [
        {"song_id": "one", "artist_id": "artist", "split": "train",
         "audio": "one.wav", "annotations": "one.json"},
    ])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["items"][0]["audio_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DatasetManifestError, match="does not match"):
        validate_dataset_manifest(path, verify_hashes=True)


def test_dataset_manifest_rejects_duplicate_content_under_different_paths(tmp_path):
    path = _write_manifest(tmp_path, [
        {"song_id": "one", "artist_id": "artist-a", "split": "train",
         "audio": "one.wav", "annotations": "one.json"},
        {"song_id": "two", "artist_id": "artist-b", "split": "test",
         "audio": "two.wav", "annotations": "two.json"},
    ])
    payload = json.loads(path.read_text(encoding="utf-8"))
    second_audio = tmp_path / "two.wav"
    second_audio.write_bytes((tmp_path / "one.wav").read_bytes())
    payload["items"][1]["audio_sha256"] = _hash(second_audio)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DatasetManifestError, match="Identical audio content"):
        validate_dataset_manifest(path)


def _write_chart_pair_manifest(tmp_path, items):
    for item_index, item in enumerate(items):
        audio_path = tmp_path / item["audio"]
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio = np.zeros(8000 * 12, dtype=np.float32)
        audio[item_index] = .01
        sf.write(audio_path, audio, 8000)
        item["audio_sha256"] = _hash(audio_path)
        chart_path = tmp_path / item["annotations"]
        chart_path.parent.mkdir(parents=True, exist_ok=True)
        midi = pretty_midi.PrettyMIDI(initial_tempo=120)
        guitar = pretty_midi.Instrument(program=29, name="PART GUITAR")
        for index in range(40):
            guitar.notes.append(pretty_midi.Note(
                velocity=96,
                pitch=96 + index % 5,
                start=.2 + item_index * .01 + index * .2,
                end=.3 + item_index * .01 + index * .2,
            ))
        midi.instruments.append(guitar)
        midi.write(str(chart_path))
        item["annotations_sha256"] = _hash(chart_path)
        for kind in ("audio", "chart"):
            evidence = tmp_path / f"permissions/{item['song_id']}-{kind}.txt"
            evidence.parent.mkdir(parents=True, exist_ok=True)
            evidence.write_text(f"training permission for {kind}", encoding="utf-8")
            item[f"{kind}_rights"] = {
                "basis": "explicit_permission",
                "license": "Pilot training permission",
                "attribution": f"{kind.title()} rights holder",
                "training_allowed": True,
                "redistribution_allowed": False,
                "evidence": str(evidence.relative_to(tmp_path)),
                "evidence_sha256": _hash(evidence),
            }
        item["chart_format"] = "rock_band_midi"
        item["synchronization"] = {
            "method": "manual_alignment",
            "audio_offset_seconds": 0.0,
            "verified": True,
        }
    payload = {
        "schema_version": 1,
        "dataset": {
            "id": "authorized-chart-pilot",
            "version": "1.0",
            "purpose": "authorized_human_chart_pairs",
            "source_url": "https://example.invalid/opt-in",
            "audio_license": "Per-item permission",
            "annotation_license": "Per-item permission",
            "commercial_use": "unknown",
            "attribution": "Per-item rights holders",
        },
        "items": items,
    }
    path = tmp_path / "chart-pairs.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_chart_pair_manifest_requires_both_rights_and_author_disjoint_splits(tmp_path):
    path = _write_chart_pair_manifest(tmp_path, [
        {"song_id": "train-song", "artist_id": "artist-a", "chart_author_id": "author-a",
         "split": "train", "audio": "audio/train.wav", "annotations": "charts/train.mid"},
        {"song_id": "test-song", "artist_id": "artist-b", "chart_author_id": "author-b",
         "split": "test", "audio": "audio/test.wav", "annotations": "charts/test.mid"},
    ])

    report = validate_chart_pair_manifest(path, verify_hashes=True)

    assert report["purpose"] == "authorized_human_chart_pairs"
    assert report["chart_authors"] == 2
    assert report["rights_evidence_hashes_verified"] == 4
    assert report["hashes_verified"] == 4
    assert report["validated_items"]["train-song"]["quality"]["expert_attacks"] == 40


def test_chart_pair_manifest_rejects_missing_audio_training_permission(tmp_path):
    path = _write_chart_pair_manifest(tmp_path, [
        {"song_id": "one", "artist_id": "artist", "chart_author_id": "author",
         "split": "train", "audio": "audio.wav", "annotations": "chart.mid"},
    ])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["items"][0]["audio_rights"]["training_allowed"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DatasetManifestError, match="training_allowed must be true"):
        validate_chart_pair_manifest(path)


def test_chart_pair_manifest_rejects_chart_author_leakage(tmp_path):
    path = _write_chart_pair_manifest(tmp_path, [
        {"song_id": "train", "artist_id": "artist-a", "chart_author_id": "same-author",
         "split": "train", "audio": "train.wav", "annotations": "train.mid"},
        {"song_id": "test", "artist_id": "artist-b", "chart_author_id": "same-author",
         "split": "test", "audio": "test.wav", "annotations": "test.mid"},
    ])

    with pytest.raises(DatasetManifestError, match="Chart authors must not cross"):
        validate_chart_pair_manifest(path)


def test_chart_pair_manifest_rejects_chart_without_expert_guitar(tmp_path):
    path = _write_chart_pair_manifest(tmp_path, [
        {"song_id": "one", "artist_id": "artist", "chart_author_id": "author",
         "split": "train", "audio": "audio.wav", "annotations": "chart.mid"},
    ])
    empty = pretty_midi.PrettyMIDI(initial_tempo=120)
    guitar = pretty_midi.Instrument(program=29, name="PART GUITAR")
    guitar.notes.append(pretty_midi.Note(
        velocity=96, pitch=84, start=.2, end=.3
    ))
    empty.instruments.append(guitar)
    chart_path = tmp_path / "chart.mid"
    empty.write(str(chart_path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["items"][0]["annotations_sha256"] = _hash(chart_path)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DatasetManifestError, match="at least 32 Expert guitar gems"):
        validate_chart_pair_manifest(path)


def test_chart_pair_manifest_rejects_too_few_expert_lanes(tmp_path):
    path = _write_chart_pair_manifest(tmp_path, [
        {"song_id": "one", "artist_id": "artist", "chart_author_id": "author",
         "split": "train", "audio": "audio.wav", "annotations": "chart.mid"},
    ])
    chart_path = tmp_path / "chart.mid"
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    guitar = pretty_midi.Instrument(program=29, name="PART GUITAR")
    for index in range(40):
        guitar.notes.append(pretty_midi.Note(
            velocity=96,
            pitch=96 + index % 2,
            start=.2 + index * .2,
            end=.3 + index * .2,
        ))
    midi.instruments.append(guitar)
    midi.write(str(chart_path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["items"][0]["annotations_sha256"] = _hash(chart_path)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DatasetManifestError, match="at least three lanes"):
        validate_chart_pair_manifest(path)


def test_chart_pair_manifest_rejects_chart_outside_audio(tmp_path):
    path = _write_chart_pair_manifest(tmp_path, [
        {"song_id": "one", "artist_id": "artist", "chart_author_id": "author",
         "split": "train", "audio": "audio.wav", "annotations": "chart.mid"},
    ])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["items"][0]["synchronization"]["audio_offset_seconds"] = 10
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DatasetManifestError, match="outside the audio duration"):
        validate_chart_pair_manifest(path)


def test_authorized_chart_pair_conversion_preserves_targets_and_provenance(tmp_path):
    manifest = _write_chart_pair_manifest(tmp_path, [
        {"song_id": "one", "artist_id": "artist", "chart_author_id": "author",
         "split": "train", "audio": "audio.wav", "annotations": "chart.mid"},
    ])
    output = tmp_path / "converted"

    index = convert_authorized_chart_pairs(manifest, output)

    assert index["schema"] == "rb3charts-authoring-index"
    assert index["examples"][0]["split_keys"] == [
        "artist:artist",
        "chart-author:author",
    ]
    example = json.loads(
        (output / index["examples"][0]["artifact"]).read_text(encoding="utf-8")
    )
    assert example["conditioning_scope"].startswith("chart-derived")
    assert example["source"]["audio"] == "audio.wav"
    assert example["source"]["audio_rights"]["training_allowed"] is True
    assert example["source"]["audio_rights"]["redistribution_allowed"] is False
    assert example["conditioning"][0]["attack_id"] == "one:attack:0"
    assert example["expert_targets"][0] == {
        "attack_id": "one:attack:0",
        "lanes": [0],
        "lane_durations_seconds": [0.1],
    }
    assert example["summary"] == {
        "attacks": 40,
        "chord_attacks": 0,
        "held_attacks": 0,
        "lanes_used": [0, 1, 2, 3, 4],
    }


def test_authorized_chart_pair_conversion_always_verifies_hashes(tmp_path):
    manifest = _write_chart_pair_manifest(tmp_path, [
        {"song_id": "one", "artist_id": "artist", "chart_author_id": "author",
         "split": "train", "audio": "audio.wav", "annotations": "chart.mid"},
    ])
    (tmp_path / "audio.wav").write_bytes(b"tampered")

    with pytest.raises(TrainingDataError, match="audio_sha256 does not match"):
        convert_authorized_chart_pairs(manifest, tmp_path / "converted")
