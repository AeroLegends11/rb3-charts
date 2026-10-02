"""Validation for rights-traceable structure-model training datasets."""

from collections import Counter, defaultdict
from hashlib import sha256
import json
import numpy as np
from pathlib import Path
import re


SCHEMA_VERSION = 1
SPLITS = {"train", "validation", "test"}
COMMERCIAL_USE = {"allowed", "noncommercial", "unknown"}
CHART_PAIR_PURPOSE = "authorized_human_chart_pairs"
CHART_FORMATS = {"rock_band_midi", "frets_on_fire_midi", "clone_hero_chart"}
RIGHTS_BASES = {"creator_owned", "explicit_permission", "compatible_license"}
SYNC_METHODS = {"manual_alignment", "shared_render", "embedded_audio_offset"}
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


def _validate_training_rights(root, rights, context, verify_hashes):
    if not isinstance(rights, dict):
        raise DatasetManifestError(f"{context} must be an object")
    basis = str(_required(rights, "basis", context))
    if basis not in RIGHTS_BASES:
        choices = ", ".join(sorted(RIGHTS_BASES))
        raise DatasetManifestError(f"{context}.basis must be one of: {choices}")
    if rights.get("training_allowed") is not True:
        raise DatasetManifestError(f"{context}.training_allowed must be true")
    license_name = str(_required(rights, "license", context))
    attribution = str(_required(rights, "attribution", context))
    relative, resolved = _resolve_item_path(
        root, _required(rights, "evidence", context), f"{context}.evidence"
    )
    expected = str(_required(rights, "evidence_sha256", context))
    if not _SHA256.fullmatch(expected):
        raise DatasetManifestError(
            f"{context}.evidence_sha256 must be 64 lowercase hex characters"
        )
    verified = 0
    if verify_hashes:
        actual = _hash_file(resolved)
        if actual != expected:
            raise DatasetManifestError(
                f"{context}.evidence_sha256 does not match {relative}: "
                f"expected {expected}, got {actual}"
            )
        verified = 1
    return {
        "basis": basis,
        "license": license_name,
        "attribution": attribution,
        "training_allowed": True,
        "evidence": str(relative),
        "evidence_sha256": expected,
        "redistribution_allowed": rights.get("redistribution_allowed") is True,
    }, verified


def _find_guitar_instrument(midi, chart_format):
    named = next(
        (instrument for instrument in midi.instruments if instrument.name == "PART GUITAR"),
        None,
    )
    if named is not None or chart_format != "frets_on_fire_midi":
        return named
    candidates = [
        instrument
        for instrument in midi.instruments
        if any(96 <= note.pitch <= 100 for note in instrument.notes)
    ]
    return candidates[0] if len(candidates) == 1 else None


def _chart_tick_converter(tempo_rows, resolution, offset):
    segments = []
    elapsed = float(offset)
    previous_tick = 0
    previous_bpm = 120.0
    for tick, bpm in sorted(tempo_rows):
        elapsed += (tick - previous_tick) * 60.0 / (previous_bpm * resolution)
        segments.append((tick, elapsed, bpm))
        previous_tick = tick
        previous_bpm = bpm
    if not segments or segments[0][0] != 0:
        segments.insert(0, (0, float(offset), 120.0))

    def seconds(tick):
        start_tick, start_seconds, bpm = segments[0]
        for candidate in segments[1:]:
            if candidate[0] > tick:
                break
            start_tick, start_seconds, bpm = candidate
        return start_seconds + (tick - start_tick) * 60.0 / (bpm * resolution)

    return seconds


def _read_clone_hero_expert(path, difficulty="expert"):
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise DatasetManifestError(f"Cannot read Clone Hero chart: {exc}") from exc
    section = None
    resolution = None
    offset = 0.0
    tempo_rows = []
    note_rows = []
    target_section = {
        "easy": "EasySingle",
        "medium": "MediumSingle",
        "hard": "HardSingle",
        "expert": "ExpertSingle",
    }[difficulty]
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        if section == "Song":
            match = re.fullmatch(r"Resolution\s*=\s*(\d+)", line)
            if match:
                resolution = int(match.group(1))
            match = re.fullmatch(r"Offset\s*=\s*(-?\d+(?:\.\d+)?)", line)
            if match:
                offset = float(match.group(1))
        elif section == "SyncTrack":
            match = re.fullmatch(r"(\d+)\s*=\s*B\s+(\d+)", line)
            if match:
                tempo_rows.append((int(match.group(1)), int(match.group(2)) / 1000.0))
        elif section == target_section:
            match = re.fullmatch(r"(\d+)\s*=\s*N\s+([0-4])\s+(\d+)", line)
            if match:
                note_rows.append(tuple(map(int, match.groups())))
    if not resolution or resolution <= 0:
        raise DatasetManifestError("Clone Hero chart requires a positive Resolution")
    if not tempo_rows:
        raise DatasetManifestError("Clone Hero chart requires SyncTrack tempo events")
    to_seconds = _chart_tick_converter(tempo_rows, resolution, offset)
    return [
        {
            "start": to_seconds(tick),
            "end": to_seconds(tick + duration),
            "beat": tick / resolution,
            "lane": lane,
        }
        for tick, lane, duration in note_rows
    ]


def _read_expert_gems(chart_path, chart_format, difficulty="expert"):
    if chart_format == "clone_hero_chart":
        return _read_clone_hero_expert(chart_path, difficulty)
    try:
        import pretty_midi

        midi = pretty_midi.PrettyMIDI(str(chart_path))
    except (EOFError, OSError, RuntimeError, ValueError) as exc:
        raise DatasetManifestError(f"Cannot read five-lane MIDI: {exc}") from exc
    guitar = _find_guitar_instrument(midi, chart_format)
    if guitar is None:
        requirement = (
            "one unambiguous five-lane guitar track"
            if chart_format == "frets_on_fire_midi"
            else "a PART GUITAR track"
        )
        raise DatasetManifestError(f"Chart requires {requirement}")
    base = {"easy": 60, "medium": 72, "hard": 84, "expert": 96}[difficulty]
    return [
        {
            "start": float(note.start),
            "end": float(note.end),
            "beat": float(midi.time_to_tick(note.start) / midi.resolution),
            "lane": int(note.pitch - base),
        }
        for note in guitar.notes
        if base <= note.pitch <= base + 4
    ]


def _inspect_chart_pair(audio_path, chart_path, chart_format, synchronization, context):
    try:
        import soundfile as sf

        audio_duration = float(sf.info(audio_path).duration)
        expert = sorted(
            _read_expert_gems(chart_path, chart_format),
            key=lambda note: (note["start"], note["lane"]),
        )
    except (EOFError, OSError, RuntimeError, ValueError) as exc:
        raise DatasetManifestError(f"{context} cannot inspect audio/chart pair: {exc}") from exc
    if audio_duration < 5:
        raise DatasetManifestError(f"{context} audio must be at least 5 seconds")
    if len(expert) < 32:
        raise DatasetManifestError(
            f"{context} chart requires at least 32 Expert guitar gems"
        )
    lanes = sorted({note["lane"] for note in expert})
    if len(lanes) < 3:
        raise DatasetManifestError(
            f"{context} Expert guitar must use at least three lanes"
        )
    if any(
        not np.isfinite(note["start"])
        or not np.isfinite(note["end"])
        or note["start"] < 0
        or note["end"] < note["start"]
        for note in expert
    ):
        raise DatasetManifestError(f"{context} Expert guitar contains invalid note timing")
    offset = float(synchronization.get("audio_offset_seconds", 0))
    first_audio_time = expert[0]["start"] + offset
    last_audio_time = max(note["end"] for note in expert) + offset
    if first_audio_time < -0.25 or last_audio_time > audio_duration + 0.25:
        raise DatasetManifestError(
            f"{context} synchronized Expert guitar falls outside the audio duration"
        )
    attack_times = sorted({round(float(note["start"]), 6) for note in expert})
    return {
        "audio_duration_seconds": round(audio_duration, 6),
        "expert_gems": len(expert),
        "expert_attacks": len(attack_times),
        "expert_lanes": lanes,
        "first_audio_time": round(first_audio_time, 6),
        "last_audio_time": round(last_audio_time, 6),
    }


def validate_chart_pair_manifest(path, verify_hashes=False):
    """Validate opt-in synchronized audio/chart pairs for supervised training."""
    report = validate_dataset_manifest(path, verify_hashes=verify_hashes)
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = payload["dataset"]
    if dataset.get("purpose") != CHART_PAIR_PURPOSE:
        raise DatasetManifestError(
            f"dataset purpose must be {CHART_PAIR_PURPOSE}"
        )
    root = manifest_path.parent.resolve()
    author_splits = defaultdict(set)
    evidence_verified = 0
    validated = {}
    for index, item in enumerate(payload["items"]):
        context = f"items[{index}]"
        song_id = str(item["song_id"])
        author_id = str(_required(item, "chart_author_id", context))
        author_splits[author_id].add(str(item["split"]))
        chart_format = item.get("chart_format")
        if chart_format not in CHART_FORMATS:
            choices = ", ".join(sorted(CHART_FORMATS))
            raise DatasetManifestError(
                f"{context}.chart_format must be one of: {choices}"
            )
        synchronization = item.get("synchronization")
        if not isinstance(synchronization, dict):
            raise DatasetManifestError(f"{context}.synchronization must be an object")
        method = str(_required(synchronization, "method", f"{context}.synchronization"))
        if method not in SYNC_METHODS:
            choices = ", ".join(sorted(SYNC_METHODS))
            raise DatasetManifestError(
                f"{context}.synchronization.method must be one of: {choices}"
            )
        if synchronization.get("verified") is not True:
            raise DatasetManifestError(
                f"{context}.synchronization.verified must be true"
            )
        try:
            offset = float(synchronization.get("audio_offset_seconds", 0))
        except (TypeError, ValueError) as exc:
            raise DatasetManifestError(
                f"{context}.synchronization.audio_offset_seconds must be numeric"
            ) from exc
        if not (-30 <= offset <= 30):
            raise DatasetManifestError(
                f"{context}.synchronization.audio_offset_seconds must be between -30 and 30"
            )
        audio_rights, audio_verified = _validate_training_rights(
            root, item.get("audio_rights"), f"{context}.audio_rights", verify_hashes
        )
        chart_rights, chart_verified = _validate_training_rights(
            root, item.get("chart_rights"), f"{context}.chart_rights", verify_hashes
        )
        _, audio_path = _resolve_item_path(
            root, item["audio"], f"{context}.audio"
        )
        _, chart_path = _resolve_item_path(
            root, item["annotations"], f"{context}.annotations"
        )
        quality = _inspect_chart_pair(
            audio_path, chart_path, chart_format, synchronization, context
        )
        evidence_verified += audio_verified + chart_verified
        validated[song_id] = {
            **report["validated_items"][song_id],
            "chart_author_id": author_id,
            "chart_format": chart_format,
            "synchronization": {
                "method": method,
                "audio_offset_seconds": offset,
                "verified": True,
            },
            "audio_rights": audio_rights,
            "chart_rights": chart_rights,
            "quality": quality,
        }
    leaks = {
        author: sorted(splits)
        for author, splits in author_splits.items()
        if len(splits) > 1
    }
    if leaks:
        examples = ", ".join(
            f"{author} ({'/'.join(splits)})"
            for author, splits in sorted(leaks.items())[:5]
        )
        raise DatasetManifestError(
            f"Chart authors must not cross dataset splits: {examples}"
        )
    return {
        **report,
        "purpose": CHART_PAIR_PURPOSE,
        "chart_authors": len(author_splits),
        "rights_evidence_hashes_verified": evidence_verified,
        "validated_items": validated,
    }
