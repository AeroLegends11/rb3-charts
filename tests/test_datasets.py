import hashlib
import json

import pytest

from rb3charts.datasets import DatasetManifestError, validate_dataset_manifest


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
