"""Validation for rights-traceable structure-model training datasets."""

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re


SCHEMA_VERSION = 1
SPLITS = {"train", "validation", "test"}
COMMERCIAL_USE = {"allowed", "noncommercial", "unknown"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class DatasetManifestError(ValueError):
    pass


def _required(mapping, key, context):
    value = mapping.get(key)
    if value is None or value == "":
        raise DatasetManifestError(f"{context} requires {key}")
    return value


def _resolve_item_path(root, value, context):
    relative = Path(str(value))
    if relative.is_absolute():
        raise DatasetManifestError(f"{context} path must be relative: {relative}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise DatasetManifestError(f"{context} path escapes the dataset root: {relative}") from exc
    if not resolved.is_file():
        raise DatasetManifestError(f"{context} file does not exist: {relative}")
    return relative, resolved


def _hash_file(path):
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset_manifest(path, verify_hashes=False):
    """Validate one dataset manifest and return an auditable summary.

    Audio and annotation licenses are recorded separately because an open label
    file does not grant rights to a corresponding commercial recording.
    """
    manifest_path = Path(path).resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetManifestError(f"Cannot read dataset manifest: {exc}") from exc
    if not isinstance(payload, dict):
        raise DatasetManifestError("Dataset manifest must contain a JSON object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise DatasetManifestError(f"schema_version must be {SCHEMA_VERSION}")

    dataset = payload.get("dataset")
    if not isinstance(dataset, dict):
        raise DatasetManifestError("dataset must be an object")
    for key in (
        "id",
        "version",
        "source_url",
        "audio_license",
        "annotation_license",
        "commercial_use",
        "attribution",
    ):
        _required(dataset, key, "dataset")
    if dataset["commercial_use"] not in COMMERCIAL_USE:
        choices = ", ".join(sorted(COMMERCIAL_USE))
        raise DatasetManifestError(f"dataset commercial_use must be one of: {choices}")

    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise DatasetManifestError("items must be a non-empty array")
    root = manifest_path.parent.resolve()
    seen_songs = set()
    seen_paths = set()
    seen_hashes = {kind: {} for kind in ("audio", "annotations")}
    artist_splits = defaultdict(set)
    split_counts = Counter()
    verified = 0
    validated_items = {}
    for index, item in enumerate(items):
        context = f"items[{index}]"
        if not isinstance(item, dict):
            raise DatasetManifestError(f"{context} must be an object")
        song_id = str(_required(item, "song_id", context))
        artist_id = str(_required(item, "artist_id", context))
        split = str(_required(item, "split", context))
        if split not in SPLITS:
            raise DatasetManifestError(f"{context} split must be train, validation, or test")
        if song_id in seen_songs:
            raise DatasetManifestError(f"Duplicate song_id: {song_id}")
        seen_songs.add(song_id)
        artist_splits[artist_id].add(split)
        split_counts[split] += 1
        validated_item = {
            "song_id": song_id,
            "artist_id": artist_id,
            "split": split,
        }

        for kind in ("audio", "annotations"):
            relative, resolved = _resolve_item_path(root, _required(item, kind, context), f"{context}.{kind}")
            normalized = str(resolved).lower()
            if normalized in seen_paths:
                raise DatasetManifestError(f"File is reused by multiple items: {relative}")
            seen_paths.add(normalized)
            expected = str(_required(item, f"{kind}_sha256", context))
            if not _SHA256.fullmatch(expected):
                raise DatasetManifestError(f"{context}.{kind}_sha256 must be 64 lowercase hex characters")
            previous = seen_hashes[kind].get(expected)
            if previous is not None:
                raise DatasetManifestError(
                    f"Identical {kind} content is reused by {previous} and {song_id}: {relative}"
                )
            seen_hashes[kind][expected] = song_id
            validated_item[f"{kind}_sha256"] = expected
            if verify_hashes:
                actual = _hash_file(resolved)
                if actual != expected:
                    raise DatasetManifestError(
                        f"{context}.{kind}_sha256 does not match {relative}: expected {expected}, got {actual}"
                    )
                verified += 1
        validated_items[song_id] = validated_item

    leaks = {
        artist: sorted(splits)
        for artist, splits in artist_splits.items()
        if len(splits) > 1
    }
    if leaks:
        examples = ", ".join(
            f"{artist} ({'/'.join(splits)})"
            for artist, splits in sorted(leaks.items())[:5]
        )
        raise DatasetManifestError(f"Artists must not cross dataset splits: {examples}")

    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset["id"],
        "dataset_version": dataset["version"],
        "commercial_use": dataset["commercial_use"],
        "items": len(items),
        "artists": len(artist_splits),
        "splits": {split: split_counts[split] for split in sorted(SPLITS)},
        "hashes_verified": verified,
        "manifest": str(manifest_path),
        "manifest_sha256": _hash_file(manifest_path),
        "dataset": dict(dataset),
        "validated_items": validated_items,
    }
