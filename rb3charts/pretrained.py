"""Validated interchange and review helpers for externally generated charts.

This module deliberately does not import or install third-party model runtimes.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from itertools import combinations
import hashlib
import json
import math
import os
import sys
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

SCHEMA_VERSION = 1
AUDIO2CHART_UPSTREAM = "https://github.com/3podi/audio2chart"
AUDIO2CHART_REVISION = "b6abaafecd3d39a11793ca64538431263dcd99ad"
AUDIO2CHART_MODEL = "3podi/charter-v1.0-40-M-best-acc"
AUDIO2CHART_SAMPLE_RATE = 24_000
AUDIO2CHART_CHUNK_SECONDS = 30
CLONECHARTER_UPSTREAM = "https://github.com/TheJorseman/CloneCharter"
CLONECHARTER_REVISION = "e33dd4f45cd3d731e50e0ea68163ed890cbe45e5"
CLONECHARTER_MODEL = "thejorseman/CloneCharter"
CLONECHARTER_SUSTAIN_STEPS = list(range(48)) + [54, 60, 66, 72, 78, 84, 90, 96, 110, 120, 144, 165]
SUPPORTED_INSTRUMENTS = {"guitar", "bass", "drums"}
SUPPORTED_DIFFICULTIES = {"easy", "medium", "hard", "expert"}
SUPPORTED_MODIFIERS = {"forced_hopo", "forced_strum", "tap"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish_directory_no_replace(source: Path, target: Path):
    if os.name == "nt":
        os.rename(source, target)
        return
    if sys.platform == "linux":
        import ctypes

        renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
        if renameat2 is None:
            raise OSError("Atomic no-replace directory publication is unavailable on this Linux system")
        renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(target),
            1,
        )
        if result:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(target))
        return
    raise OSError(f"Atomic no-replace directory publication is unsupported on {sys.platform}")


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def validate_bundle(payload: dict) -> dict:
    """Validate versioned source-time events and return the unchanged payload."""
    if not isinstance(payload, dict):
        raise ValueError("Event bundle must be a JSON object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported pretrained event schema version: {payload.get('schema_version')!r}")

    backend = payload.get("backend")
    required_backend = {
        "name",
        "upstream",
        "upstream_revision",
        "model_id",
        "model_revision",
        "weights_sha256",
    }
    if not isinstance(backend, dict) or not required_backend.issubset(backend):
        raise ValueError("backend must identify the name, upstream, upstream revision, model, model revision, and weight hash")
    for field in ("name", "upstream", "model_id"):
        if not isinstance(backend[field], str) or not backend[field].strip():
            raise ValueError(f"backend.{field} must be a non-empty string")
    for field in ("upstream_revision", "model_revision"):
        if backend[field] is not None and not isinstance(backend[field], str):
            raise ValueError(f"backend.{field} must be a string or null")
    weight_hash = backend["weights_sha256"]
    if weight_hash is not None and (
        not isinstance(weight_hash, str)
        or len(weight_hash) != 64
        or any(char not in "0123456789abcdef" for char in weight_hash.lower())
    ):
        raise ValueError("backend.weights_sha256 must be a 64-character SHA-256 or null")

    source = payload.get("source")
    if not isinstance(source, dict):
        raise ValueError("source metadata is required")
    source_hash = source.get("sha256")
    if not isinstance(source_hash, str) or len(source_hash) != 64 or any(
        char not in "0123456789abcdef" for char in source_hash.lower()
    ):
        raise ValueError("source.sha256 must be a 64-character SHA-256")
    sample_rate = source.get("sample_rate")
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
        raise ValueError("source.sample_rate must be a positive integer")
    duration = source.get("duration_seconds")
    if not _finite_number(duration) or duration <= 0:
        raise ValueError("source.duration_seconds must be finite and positive")

    inference = payload.get("inference")
    if (
        not isinstance(inference, dict)
        or not {"device", "seed", "sampling", "runtime_seconds"}.issubset(inference)
        or not isinstance(inference.get("sampling"), dict)
    ):
        raise ValueError("inference.device, seed, sampling parameters, and runtime metadata are required")
    if not isinstance(inference.get("device"), str) or not inference["device"]:
        raise ValueError("inference.device must be a non-empty string")
    seed = inference.get("seed")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise ValueError("inference.seed must be an integer or null")
    runtime = inference.get("runtime_seconds")
    if runtime is not None and (not _finite_number(runtime) or runtime < 0):
        raise ValueError("inference.runtime_seconds must be finite and nonnegative, or null")

    timebase = payload.get("timebase")
    if not isinstance(timebase, dict):
        raise ValueError("timebase metadata is required")
    if timebase.get("unit") != "seconds" or timebase.get("origin") != "source_audio":
        raise ValueError("events must use seconds relative to source-audio time zero")
    offset = timebase.get("offset_applied_seconds")
    if not _finite_number(offset) or offset < 0 or offset >= duration:
        raise ValueError("timebase.offset_applied_seconds must be finite and within source duration")
    if "grid_ms" in timebase and (
        not isinstance(timebase["grid_ms"], int)
        or isinstance(timebase["grid_ms"], bool)
        or timebase["grid_ms"] <= 0
    ):
        raise ValueError("timebase.grid_ms must be a positive integer")

    instrument = payload.get("instrument")
    if instrument not in SUPPORTED_INSTRUMENTS:
        raise ValueError(f"Unsupported pretrained instrument: {instrument!r}")
    if payload.get("difficulty") not in SUPPORTED_DIFFICULTIES:
        raise ValueError(f"Unsupported or missing difficulty: {payload.get('difficulty')!r}")

    tempo = payload.get("tempo")
    if tempo is not None:
        if not isinstance(tempo, dict):
            raise ValueError("tempo must be an object or null")
        beats = tempo.get("beats_seconds")
        if not isinstance(beats, list) or len(beats) < 2 or any(not _finite_number(b) for b in beats):
            raise ValueError("tempo.beats_seconds must contain at least two finite source-time beats")
        if any(right <= left for left, right in zip(beats, beats[1:])):
            raise ValueError("tempo.beats_seconds must be strictly increasing")
        if tempo.get("numerator", 4) not in (3, 4, 5, 6, 7):
            raise ValueError("tempo.numerator must be from 3 through 7")
        if "confidence" in tempo and (
            not _finite_number(tempo["confidence"]) or not 0 <= tempo["confidence"] <= 1
        ):
            raise ValueError("tempo.confidence must be in [0, 1]")
        if "warnings" in tempo and (
            not isinstance(tempo["warnings"], list)
            or any(not isinstance(warning, str) for warning in tempo["warnings"])
        ):
            raise ValueError("tempo.warnings must be a list of strings")
        provenance = tempo.get("provenance")
        if not isinstance(provenance, str) or not provenance.strip():
            raise ValueError("tempo.provenance must identify where the shared tempo map came from")

    events = payload.get("events")
    if not isinstance(events, list):
        raise ValueError("events must be a list")
    prior_onset = -1.0
    ids = set()
    for index, event in enumerate(events):
        tag = f"events[{index}]"
        if not isinstance(event, dict):
            raise ValueError(f"{tag} must be an object")
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or not event_id.strip() or event_id in ids:
            raise ValueError(f"{tag}.event_id must be non-empty and unique")
        ids.add(event_id)
        onset = event.get("onset_seconds")
        if not _finite_number(onset) or onset < 0 or onset >= duration:
            raise ValueError(f"{tag}.onset_seconds must be finite and inside the source audio")
        if onset <= prior_onset:
            raise ValueError("events must be strictly sorted by onset; combine simultaneous lanes into one chord event")
        prior_onset = onset
        offset_seconds = event.get("offset_seconds")
        duration_status = event.get("duration_status")
        if duration_status == "predicted":
            if not _finite_number(offset_seconds) or offset_seconds <= onset or offset_seconds > duration:
                raise ValueError(f"{tag} predicted offset must be after its onset and within source duration")
        elif duration_status == "unknown":
            if offset_seconds is not None:
                raise ValueError(f"{tag} unknown duration must not claim an offset")
        else:
            raise ValueError(f"{tag}.duration_status must be 'predicted' or 'unknown'")
        lanes = event.get("lanes")
        if (
            not isinstance(lanes, list)
            or not lanes
            or len(lanes) > 5
            or any(not isinstance(lane, int) or isinstance(lane, bool) or lane not in range(5) for lane in lanes)
            or len(set(lanes)) != len(lanes)
            or lanes != sorted(lanes)
        ):
            raise ValueError(f"{tag}.lanes must be unique, sorted five-fret lane indices 0..4")
        if event.get("open_note", False):
            raise ValueError(f"{tag} contains an unsupported open note")
        modifiers = event.get("modifiers", [])
        if not isinstance(modifiers, list) or any(
            not isinstance(modifier, str) or modifier not in SUPPORTED_MODIFIERS
            for modifier in modifiers
        ):
            raise ValueError(f"{tag} contains unknown note modifiers")
        confidence = event.get("confidence")
        if confidence is not None and (
            not _finite_number(confidence) or not 0 <= confidence <= 1
        ):
            raise ValueError(f"{tag}.confidence must be in [0, 1] or null")
    return payload


def _audio2chart_lane_map() -> dict[int, tuple[int, ...]]:
    lanes = list(range(5))
    patterns = [
        pattern
        for size in range(1, len(lanes) + 1)
        for pattern in combinations(lanes, size)
    ]
    return {index: pattern for index, pattern in enumerate(patterns)} | {
        len(patterns): (7,)
    }


def _audio2chart_chunk_starts(total_samples: int, sample_rate: int, chunk_seconds: int):
    chunk_samples = sample_rate * chunk_seconds
    if total_samples < chunk_samples:
        raise ValueError(f"audio2chart inference requires at least {chunk_seconds} seconds of audio")
    starts = list(range(0, total_samples, chunk_samples))
    if starts[-1] + chunk_samples > total_samples:
        starts[-1] = max(0, total_samples - chunk_samples)
    return starts, chunk_samples


def audio2chart_tokens_to_bundle(
    sequences: list,
    *,
    total_samples: int,
    source_sha256: str,
    source_sample_rate: int,
    source_duration_seconds: float,
    pad_token_id: int,
    grid_ms: int,
    model_id: str = AUDIO2CHART_MODEL,
    model_revision: str | None = None,
    weights_sha256: str | None = None,
    device: str = "unknown",
    seed: int | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    runtime_seconds: float | None = None,
    tempo: dict | None = None,
):
    """Normalize audio2chart token chunks without inventing holds or confidence."""
    if not isinstance(grid_ms, int) or isinstance(grid_ms, bool) or grid_ms <= 0:
        raise ValueError("grid_ms must be a positive integer from the checkpoint config")
    if 1000 * AUDIO2CHART_CHUNK_SECONDS % grid_ms:
        raise ValueError("audio2chart grid must divide a 30-second chunk exactly")
    if not isinstance(pad_token_id, int) or isinstance(pad_token_id, bool) or pad_token_id < 0:
        raise ValueError("pad_token_id must come from the loaded checkpoint config")
    if not isinstance(total_samples, int) or isinstance(total_samples, bool) or total_samples <= 0:
        raise ValueError("total_samples must be a positive integer at the model sample rate")
    if (
        not isinstance(source_sample_rate, int)
        or isinstance(source_sample_rate, bool)
        or source_sample_rate <= 0
        or not _finite_number(source_duration_seconds)
        or source_duration_seconds <= 0
    ):
        raise ValueError("source sample rate and duration must be positive and finite")
    starts, chunk_samples = _audio2chart_chunk_starts(
        total_samples, AUDIO2CHART_SAMPLE_RATE, AUDIO2CHART_CHUNK_SECONDS
    )
    duration_tolerance = 1 / source_sample_rate + 1 / AUDIO2CHART_SAMPLE_RATE
    if abs(total_samples / AUDIO2CHART_SAMPLE_RATE - source_duration_seconds) > duration_tolerance:
        raise ValueError("resampled model audio duration does not match the source WAV duration")
    if not isinstance(sequences, list):
        raise ValueError("model sequences must be a list of token sequences")
    if pad_token_id in _audio2chart_lane_map():
        raise ValueError("pad_token_id conflicts with a real guitar token or unsupported open-note token")
    if len(sequences) != len(starts):
        raise ValueError(f"expected {len(starts)} generated chunks, received {len(sequences)}")
    full_tokens = 1000 * AUDIO2CHART_CHUNK_SECONDS // grid_ms
    previous_end = 0
    normalized = []
    token_lanes = _audio2chart_lane_map()
    for chunk_index, sequence in enumerate(sequences):
        if not isinstance(sequence, (list, tuple)):
            raise ValueError(f"generated chunk {chunk_index} must be a token list")
        tokens = list(sequence)
        if any(not isinstance(token, int) or isinstance(token, bool) for token in tokens):
            raise ValueError(f"generated chunk {chunk_index} contains a non-integer token")
        if len(tokens) > full_tokens or (chunk_index < len(sequences) - 1 and len(tokens) != full_tokens):
            raise ValueError(f"generated chunk {chunk_index} has an unexpected token-grid length")
        first_grid_index = full_tokens - len(tokens) if chunk_index == len(sequences) - 1 else 0
        chunk_start = starts[chunk_index]
        if chunk_index:
            previous_end = starts[chunk_index - 1] + chunk_samples
        for token_index, token in enumerate(tokens, start=first_grid_index):
            sample_position = chunk_start + round(token_index * grid_ms * AUDIO2CHART_SAMPLE_RATE / 1000)
            if sample_position < previous_end:
                continue
            onset = sample_position / AUDIO2CHART_SAMPLE_RATE
            if onset >= source_duration_seconds:
                continue
            if token == pad_token_id:
                continue
            lanes = token_lanes.get(token)
            if lanes is None:
                raise ValueError(f"audio2chart emitted unknown token id {token} at {onset:.6f}s")
            if lanes == (7,):
                raise ValueError(f"audio2chart emitted an unsupported open note at {onset:.6f}s")
            normalized.append(
                {
                    "event_id": f"audio2chart-guitar-{len(normalized):06d}",
                    "onset_seconds": round(onset, 9),
                    "offset_seconds": None,
                    "duration_status": "unknown",
                    "lanes": list(lanes),
                    "confidence": None,
                    "open_note": False,
                    "modifiers": [],
                }
            )

    sampling = {}
    if temperature is not None:
        if not _finite_number(temperature):
            raise ValueError("temperature must be finite or omitted")
        sampling["temperature"] = float(temperature)
    if top_k is not None:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0:
            raise ValueError("top_k must be a nonnegative integer or omitted")
        sampling["top_k"] = top_k
    payload = {
        "schema_version": SCHEMA_VERSION,
        "backend": {
            "name": "audio2chart",
            "upstream": AUDIO2CHART_UPSTREAM,
            "upstream_revision": AUDIO2CHART_REVISION,
            "model_id": model_id,
            "model_revision": model_revision,
            "weights_sha256": weights_sha256,
        },
        "source": {
            "sha256": source_sha256,
            "sample_rate": source_sample_rate,
            "duration_seconds": source_duration_seconds,
        },
        "inference": {
            "device": device,
            "seed": seed,
            "sampling": sampling,
            "runtime_seconds": runtime_seconds,
        },
        "timebase": {
            "unit": "seconds",
            "origin": "source_audio",
            "grid_ms": grid_ms,
            "offset_applied_seconds": 0.0,
        },
        "instrument": "guitar",
        "difficulty": "expert",
        "tempo": tempo,
        "events": normalized,
    }
    return validate_bundle(payload)


def audio2chart_chart_to_bundle(
    chart_path: str | Path,
    *,
    source_sha256: str,
    source_sample_rate: int,
    source_duration_seconds: float,
    tempo: dict | None,
    model_id: str = AUDIO2CHART_MODEL,
    model_revision: str | None = None,
    weights_sha256: str | None = None,
    device: str = "unknown",
    seed: int | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    runtime_seconds: float | None = None,
):
    """Normalize audio2chart's written ExpertSingle .chart without model imports."""
    sections = {}
    current = None
    inside = False
    for line_number, line in enumerate(Path(chart_path).read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        section = re.fullmatch(r"\[([^\]]+)\]", stripped)
        if section:
            if current is not None and inside:
                raise ValueError(f"Unclosed chart section before line {line_number}")
            current = section.group(1)
            sections.setdefault(current, [])
            inside = False
        elif stripped == "{":
            if current is None or inside:
                raise ValueError(f"Unexpected chart block at line {line_number}")
            inside = True
        elif stripped == "}":
            if current is None or not inside:
                raise ValueError(f"Unexpected chart block end at line {line_number}")
            inside = False
        elif stripped and not stripped.startswith("//"):
            if current is None or not inside:
                raise ValueError(f"Unexpected chart content at line {line_number}")
            sections[current].append((line_number, stripped))
    if inside:
        raise ValueError("Unclosed final chart section")
    if not {"Song", "SyncTrack", "ExpertSingle"}.issubset(sections):
        raise ValueError("audio2chart chart must contain Song, SyncTrack and ExpertSingle sections")

    song = {}
    for line_number, line in sections["Song"]:
        if "=" not in line:
            raise ValueError(f"Malformed Song field at line {line_number}")
        key, value = (part.strip() for part in line.split("=", 1))
        song[key] = value.strip('"')
    try:
        resolution = int(song["Resolution"])
        chart_offset = float(song.get("Offset", "0"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Song Resolution and Offset must be valid numbers") from exc
    if resolution <= 0:
        raise ValueError("Song Resolution must be positive")
    if not math.isfinite(chart_offset) or chart_offset != 0:
        raise ValueError("Nonzero chart Offset cannot be safely normalized to source-audio time")

    bpm_events = []
    time_signatures = []
    for line_number, line in sections["SyncTrack"]:
        match = re.fullmatch(r"(\d+)\s*=\s*B\s+(\d+)", line)
        if match:
            tick, raw_bpm = map(int, match.groups())
            if raw_bpm <= 0:
                raise ValueError(f"Invalid SyncTrack BPM at line {line_number}")
            bpm_events.append((tick, raw_bpm / 1000.0))
        elif match := re.fullmatch(r"(\d+)\s*=\s*TS\s+(\d+)(?:\s+(\d+))?", line):
            tick, numerator, denominator = match.groups()
            numerator = int(numerator)
            denominator = int(denominator) if denominator is not None else 4
            if numerator not in (3, 4, 5, 6, 7) or denominator <= 0:
                raise ValueError(f"Unsupported SyncTrack time signature at line {line_number}")
            time_signatures.append((int(tick), numerator, denominator))
        else:
            raise ValueError(f"Unsupported SyncTrack event at line {line_number}")
    bpm_events.sort()
    if not bpm_events or bpm_events[0][0] != 0:
        raise ValueError("SyncTrack must define a positive BPM at tick zero")
    if any(right[0] == left[0] for left, right in zip(bpm_events, bpm_events[1:])):
        raise ValueError("SyncTrack has conflicting BPM events at one tick")
    segments = []
    seconds = 0.0
    for index, (tick, bpm) in enumerate(bpm_events):
        if index:
            previous_tick, previous_bpm = bpm_events[index - 1]
            seconds += (tick - previous_tick) / resolution * 60 / previous_bpm
        segments.append((tick, bpm, seconds))

    def tick_to_seconds(tick):
        applicable = 0
        for index, segment in enumerate(segments):
            if segment[0] > tick:
                break
            applicable = index
        base_tick, bpm, base_seconds = segments[applicable]
        return base_seconds + (tick - base_tick) / resolution * 60 / bpm

    notes_by_tick = {}
    for line_number, line in sections["ExpertSingle"]:
        match = re.fullmatch(r"(\d+)\s*=\s*N\s+(\d+)\s+(\d+)", line)
        if not match:
            if re.match(r"\d+\s*=", line):
                raise ValueError(f"Unsupported ExpertSingle event at line {line_number}")
            raise ValueError(f"Malformed ExpertSingle note at line {line_number}")
        tick, lane, sustain = map(int, match.groups())
        if lane == 7:
            raise ValueError(f"Unsupported open note at chart tick {tick}")
        if lane not in range(5):
            raise ValueError(f"Unsupported lane or modifier {lane} at chart tick {tick}")
        lanes = notes_by_tick.setdefault(tick, {})
        if lane in lanes:
            raise ValueError(f"Duplicate lane {lane} at chart tick {tick}")
        lanes[lane] = sustain

    events = []
    for tick, lanes_by_id in sorted(notes_by_tick.items()):
        sustains = set(lanes_by_id.values())
        if len(sustains) != 1:
            raise ValueError(f"Chord at chart tick {tick} has inconsistent sustain lengths")
        sustain = next(iter(sustains))
        onset = tick_to_seconds(tick)
        offset_seconds = tick_to_seconds(tick + sustain) if sustain else None
        events.append(
            {
                "event_id": f"audio2chart-guitar-{len(events):06d}",
                "onset_seconds": round(onset, 9),
                "offset_seconds": round(offset_seconds, 9) if offset_seconds is not None else None,
                "duration_status": "predicted" if offset_seconds is not None else "unknown",
                "lanes": sorted(lanes_by_id),
                "confidence": None,
                "open_note": False,
                "modifiers": [],
            }
        )
    sampling = {}
    if temperature is not None:
        if not _finite_number(temperature):
            raise ValueError("temperature must be finite or omitted")
        sampling["temperature"] = float(temperature)
    if top_k is not None:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0:
            raise ValueError("top_k must be a nonnegative integer or omitted")
        sampling["top_k"] = top_k
    payload = {
        "schema_version": SCHEMA_VERSION,
        "backend": {
            "name": "audio2chart",
            "upstream": AUDIO2CHART_UPSTREAM,
            "upstream_revision": AUDIO2CHART_REVISION,
            "model_id": model_id,
            "model_revision": model_revision,
            "weights_sha256": weights_sha256,
        },
        "source": {
            "sha256": source_sha256,
            "sample_rate": source_sample_rate,
            "duration_seconds": source_duration_seconds,
        },
        "inference": {
            "device": device,
            "seed": seed,
            "sampling": sampling,
            "runtime_seconds": runtime_seconds,
        },
        "timebase": {
            "unit": "seconds",
            "origin": "source_audio",
            "offset_applied_seconds": 0.0,
            "chart_resolution": resolution,
            "chart_sync_track": [[tick, bpm] for tick, bpm in bpm_events],
            "chart_time_signatures": [list(item) for item in time_signatures],
        },
        "instrument": "guitar",
        "difficulty": "expert",
        "tempo": tempo,
        "events": events,
    }
    return validate_bundle(payload)


def clonecharter_tokens_to_bundle(
    tokens: list,
    *,
    source_sha256: str,
    source_sample_rate: int,
    source_duration_seconds: float,
    tempo: dict,
    instrument: str,
    difficulty: str,
    resolution: int = 192,
    model_id: str = CLONECHARTER_MODEL,
    model_revision: str | None = None,
    weights_sha256: str | None = None,
    device: str = "unknown",
    seed: int | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    runtime_seconds: float | None = None,
):
    """Normalize CloneCharter token IDs using its pinned vocabulary grammar."""
    if not isinstance(tokens, list) or not tokens:
        raise ValueError("CloneCharter output must be a non-empty integer token list")
    if any(not isinstance(token, int) or isinstance(token, bool) or token not in range(187) for token in tokens):
        raise ValueError("CloneCharter token sequence contains a value outside the 187-token vocabulary")
    if not isinstance(resolution, int) or isinstance(resolution, bool) or resolution <= 0 or resolution % 16:
        raise ValueError("CloneCharter resolution must be a positive multiple of its 16-tick token grid")
    if instrument not in SUPPORTED_INSTRUMENTS:
        raise ValueError(f"Unsupported CloneCharter instrument: {instrument!r}")
    if difficulty not in SUPPORTED_DIFFICULTIES:
        raise ValueError("Only RB3 Easy/Medium/Hard/Expert difficulties are supported")
    if not isinstance(tempo, dict) or not isinstance(tempo.get("beats_seconds"), list):
        raise ValueError("The exact source-time beat map used for inference is required")
    beat_times = tempo["beats_seconds"]
    if len(beat_times) < 2 or any(not _finite_number(value) for value in beat_times):
        raise ValueError("tempo.beats_seconds must contain at least two finite source-time beats")
    if any(right <= left for left, right in zip(beat_times, beat_times[1:])):
        raise ValueError("tempo.beats_seconds must be strictly increasing")
    if not isinstance(tempo.get("provenance"), str) or not tempo["provenance"].strip():
        raise ValueError("tempo.provenance must identify the beat map supplied to the model")

    instrument_tokens = {"guitar": 6, "bass": 7, "drums": 8}
    if len(tokens) < 2 or tokens[0] != 1 or tokens[1] != instrument_tokens[instrument]:
        raise ValueError("CloneCharter BOS/instrument tokens do not match the selected instrument")

    def seconds_at_tick(tick):
        beat_position = tick / resolution
        if beat_position > len(beat_times) - 1:
            raise ValueError("The supplied source tempo map does not cover a generated CloneCharter event")
        beat_index = int(math.floor(beat_position))
        fraction = beat_position - beat_index
        if beat_index < len(beat_times) - 1:
            interval = beat_times[beat_index + 1] - beat_times[beat_index]
            return beat_times[beat_index] + fraction * interval
        return beat_times[-1]

    def flush_event_groups(grouped):
        rows = []
        for tick, event in sorted(grouped.items()):
            onset = seconds_at_tick(tick)
            offset_seconds = seconds_at_tick(tick + event["sustain_ticks"]) if event["sustain_ticks"] else None
            rows.append(
                {
                    "event_id": f"clonecharter-{instrument}-{difficulty}-{len(rows):06d}",
                    "onset_seconds": round(onset, 9),
                    "offset_seconds": round(offset_seconds, 9) if offset_seconds is not None else None,
                    "duration_status": "predicted" if offset_seconds is not None else "unknown",
                    "lanes": sorted(event["lanes"]),
                    "confidence": None,
                    "open_note": False,
                    "modifiers": sorted(event["modifiers"]),
                }
            )
        return rows

    grouped = {}
    cursor_tick = 0
    index = 2
    saw_eos = False
    while index < len(tokens):
        token = tokens[index]
        if token == 2:
            saw_eos = True
            if any(value != 0 for value in tokens[index + 1 :]):
                raise ValueError("Only PAD tokens may follow CloneCharter EOS")
            break
        if token == 0:
            if any(value != 0 for value in tokens[index:]):
                raise ValueError("Only PAD tokens may follow CloneCharter PAD")
            break
        if 9 <= token <= 56:
            cursor_tick += (token - 8) * 16
            index += 1
            continue
        if token in (4, 5):
            index += 1
            continue
        if token in (183, 184, 185, 186):
            raise ValueError("CloneCharter star-power/solo tokens are not represented by the current RB3 importer")
        is_drum = instrument == "drums"
        if 57 <= token <= 87 and not is_drum:
            bitmask = token - 56
            lanes = {lane for lane in range(5) if bitmask & (1 << lane)}
            index += 1
            if index >= len(tokens) or not 123 <= tokens[index] <= 182:
                raise ValueError(f"CloneCharter guitar/bass note at tick {cursor_tick} lacks its sustain token")
            sustain_ticks = CLONECHARTER_SUSTAIN_STEPS[tokens[index] - 123] * 16
            index += 1
            modifiers = set()
            while index < len(tokens) and tokens[index] in (88, 89, 90, 91):
                modifier = {
                    88: "forced_hopo",
                    89: "tap",
                    90: "open_note",
                    91: "forced_strum",
                }[tokens[index]]
                if modifier == "open_note":
                    raise ValueError(f"CloneCharter emitted an unsupported open note at tick {cursor_tick}")
                modifiers.add(modifier)
                index += 1
        elif 92 <= token <= 122 and is_drum:
            bitmask = token - 91
            lanes = {lane for lane in range(5) if bitmask & (1 << lane)}
            sustain_ticks = 0
            modifiers = set()
            index += 1
        else:
            raise ValueError(f"Unsupported CloneCharter token {token} at sequence index {index}")

        event = grouped.get(cursor_tick)
        if event is None:
            event = {
                "lanes": set(),
                "sustain_ticks": sustain_ticks,
                "modifiers": set(modifiers),
            }
            grouped[cursor_tick] = event
        elif event["sustain_ticks"] != sustain_ticks or event["modifiers"] != modifiers:
            raise ValueError(f"Simultaneous CloneCharter events at tick {cursor_tick} disagree on sustain/modifiers")
        if event["lanes"] & lanes:
            raise ValueError(f"Duplicate CloneCharter lane at tick {cursor_tick}")
        event["lanes"].update(lanes)
        event["modifiers"].update(modifiers)

    if not saw_eos:
        raise ValueError("CloneCharter token sequence ended without EOS")

    events = flush_event_groups(grouped)
    sampling = {}
    for key, value in (("temperature", temperature), ("top_k", top_k), ("top_p", top_p)):
        if value is not None:
            if not _finite_number(value):
                raise ValueError(f"{key} must be finite or omitted")
            if key == "temperature" and value <= 0:
                raise ValueError("temperature must be positive")
            if key == "top_k" and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise ValueError("top_k must be a nonnegative integer")
            if key == "top_p" and not 0 < value <= 1:
                raise ValueError("top_p must be in (0, 1]")
            sampling[key] = value
    payload = {
        "schema_version": SCHEMA_VERSION,
        "backend": {
            "name": "clonecharter",
            "upstream": CLONECHARTER_UPSTREAM,
            "upstream_revision": CLONECHARTER_REVISION,
            "model_id": model_id,
            "model_revision": model_revision,
            "weights_sha256": weights_sha256,
        },
        "source": {
            "sha256": source_sha256,
            "sample_rate": source_sample_rate,
            "duration_seconds": source_duration_seconds,
        },
        "inference": {
            "device": device,
            "seed": seed,
            "sampling": sampling,
            "runtime_seconds": runtime_seconds,
        },
        "timebase": {
            "unit": "seconds",
            "origin": "source_audio",
            "offset_applied_seconds": 0.0,
            "tick_resolution": resolution,
            "token_grid_ticks": 16,
        },
        "instrument": instrument,
        "difficulty": difficulty,
        "tempo": tempo,
        "events": events,
    }
    return validate_bundle(payload)


def load_bundle(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_bundle(payload)


def _matching_pairs(predicted: list, reference: list, tolerance_seconds: float):
    truth_times = [event["onset_seconds"] for event in reference]
    tree = [None] * (len(reference) + 1)
    paths = []
    empty = (0, 0.0, None)

    def better(left, right):
        if right is None:
            return True
        if left[0] != right[0]:
            return left[0] > right[0]
        if left[1] != right[1]:
            return left[1] < right[1]
        return False

    def query(count):
        best = empty
        index = count
        while index:
            if tree[index] is not None and better(tree[index], best):
                best = tree[index]
            index -= index & -index
        return best

    for p_index, event in enumerate(predicted):
        onset = event["onset_seconds"]
        first = bisect_left(truth_times, onset - tolerance_seconds)
        stop = bisect_right(truth_times, onset + tolerance_seconds)
        updates = []
        for t_index in range(first, stop):
            previous = query(t_index)
            node = len(paths)
            paths.append((p_index, t_index, previous[2]))
            state = (
                previous[0] + 1,
                previous[1] + abs(onset - truth_times[t_index]),
                node,
            )
            updates.append((t_index + 1, state))
        for index, state in updates:
            while index < len(tree):
                if better(state, tree[index]):
                    tree[index] = state
                index += index & -index

    best = query(len(reference))
    pairs = []
    node = best[2]
    while node is not None:
        p_index, t_index, node = paths[node]
        pairs.append((p_index, t_index))
    return list(reversed(pairs))


def compare_bundles(prediction: dict, reference: dict, tolerance_ms: float = 50.0) -> dict:
    """Compare event onsets one-to-one; return unavailable where evidence is absent."""
    validate_bundle(prediction)
    validate_bundle(reference)
    if not _finite_number(tolerance_ms) or tolerance_ms <= 0:
        raise ValueError("onset tolerance must be finite and positive")
    for key in ("sha256", "sample_rate", "duration_seconds"):
        if prediction["source"][key] != reference["source"][key]:
            raise ValueError(f"prediction and reference do not share source {key}")
    if prediction["instrument"] != reference["instrument"] or prediction["difficulty"] != reference["difficulty"]:
        raise ValueError("prediction and reference instrument/difficulty must match")
    prediction_tempo = prediction["tempo"]
    reference_tempo = reference["tempo"]
    same_tempo = (
        isinstance(prediction_tempo, dict)
        and isinstance(reference_tempo, dict)
        and prediction_tempo.get("beats_seconds") == reference_tempo.get("beats_seconds")
        and prediction_tempo.get("numerator", 4) == reference_tempo.get("numerator", 4)
    )
    if not same_tempo:
        raise ValueError("prediction and reference must use the same recorded tempo map")

    predicted = prediction["events"]
    truth = reference["events"]
    pairs = _matching_pairs(predicted, truth, tolerance_ms / 1000)
    errors_ms = [
        abs(predicted[p]["onset_seconds"] - truth[t]["onset_seconds"]) * 1000
        for p, t in pairs
    ]
    if truth:
        precision = len(pairs) / len(predicted) if predicted else 0.0
        recall = len(pairs) / len(truth)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        onset = {
            "available": True,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "coverage": round(recall, 6),
            "mean_absolute_error_ms": round(sum(errors_ms) / len(errors_ms), 6) if errors_ms else None,
            "max_absolute_error_ms": round(max(errors_ms), 6) if errors_ms else None,
        }
    else:
        onset = {
            "available": False,
            "reason": "reference contains no events",
            "precision": None,
            "recall": None,
            "f1": None,
            "coverage": None,
            "mean_absolute_error_ms": None,
            "max_absolute_error_ms": None,
        }

    lane_accuracy = (
        sum(predicted[p]["lanes"] == truth[t]["lanes"] for p, t in pairs) / len(pairs)
        if pairs
        else None
    )
    known_durations = [
        abs(predicted[p]["offset_seconds"] - truth[t]["offset_seconds"]) * 1000
        for p, t in pairs
        if predicted[p]["duration_status"] == truth[t]["duration_status"] == "predicted"
    ]
    duration_metrics = (
        {
            "available": True,
            "matched_events": len(known_durations),
            "mean_absolute_error_ms": round(sum(known_durations) / len(known_durations), 6),
        }
        if known_durations
        else {
            "available": False,
            "reason": "one or both backends do not predict note offsets",
            "matched_events": 0,
            "mean_absolute_error_ms": None,
        }
    )
    single_lane = [event["lanes"][0] for event in predicted if len(event["lanes"]) == 1]
    jumps = [abs(right - left) for left, right in zip(single_lane, single_lane[1:])]
    return {
        "metrics_version": 1,
        "prediction_backend": prediction["backend"]["name"],
        "reference_backend": reference["backend"]["name"],
        "source_sha256": prediction["source"]["sha256"],
        "instrument": prediction["instrument"],
        "difficulty": prediction["difficulty"],
        "onset_tolerance_ms": float(tolerance_ms),
        "prediction_events": len(predicted),
        "reference_events": len(truth),
        "one_to_one_matches": len(pairs),
        "onset": onset,
        "exact_lane_set_rate_on_matched_events": (
            round(lane_accuracy, 6) if lane_accuracy is not None else None
        ),
        "duration": duration_metrics,
        "predicted_density_events_per_minute": round(
            len(predicted) * 60 / prediction["source"]["duration_seconds"], 6
        ),
        "single_lane_transition": {
            "available": bool(jumps),
            "count": len(jumps),
            "mean_lane_distance": round(sum(jumps) / len(jumps), 6) if jumps else None,
            "wide_jumps_gt_2_lanes": sum(jump > 2 for jump in jumps) if jumps else None,
        },
        "playability_or_human_quality": {
            "available": False,
            "reason": "event similarity and lane-transition counts are not human playability or listening review",
        },
    }


def validate_rights_manifest(manifest: dict, prediction: dict, reference_path: str | Path):
    if not isinstance(manifest, dict) or manifest.get("manifest_version") != 1:
        raise ValueError("rights manifest must be a version 1 JSON object")
    if manifest.get("evaluation_split") != "held_out":
        raise ValueError("evaluation requires an explicitly held-out rights manifest")
    if manifest.get("rights_attested") is not True:
        raise ValueError("rights manifest must explicitly attest audio and reference-chart use")
    if not isinstance(manifest.get("audio_rights"), str) or not manifest["audio_rights"].strip():
        raise ValueError("rights manifest must describe audio rights")
    if not isinstance(manifest.get("reference_chart_rights"), str) or not manifest["reference_chart_rights"].strip():
        raise ValueError("rights manifest must describe reference-chart rights")
    if manifest.get("audio_sha256") != prediction["source"]["sha256"]:
        raise ValueError("rights manifest audio hash does not match the prediction source")
    if manifest.get("reference_bundle_sha256") != sha256_file(reference_path):
        raise ValueError("rights manifest reference hash does not match the reference event bundle")


def import_event_bundle(
    source_audio: str | Path,
    bundle_path: str | Path,
    output_folder: str | Path,
    *,
    title: str | None = None,
    artist: str | None = None,
) -> dict:
    """Create a review-only MIDI/preview from an Expert normalized event bundle."""
    import soundfile as sf

    from .chart import reductions
    from .midi import write_midi
    from .model import Note, TempoMap, save_json
    from .preview import create_preview
    from .validate import validate

    source_audio = Path(source_audio).resolve()
    bundle_path = Path(bundle_path).resolve()
    target = Path(output_folder).resolve()
    bundle = load_bundle(bundle_path)
    if bundle["difficulty"] != "expert":
        raise ValueError("Only canonical Expert event bundles can be expanded into lower difficulties")
    if not source_audio.is_file():
        raise FileNotFoundError(f"Source audio not found: {source_audio}")
    info = sf.info(source_audio)
    duration = float(info.frames / info.samplerate)
    source_hash = sha256_file(source_audio)
    if source_hash != bundle["source"]["sha256"]:
        raise ValueError("Source WAV SHA-256 does not match the event bundle")
    if info.samplerate != bundle["source"]["sample_rate"] or not math.isclose(
        duration, bundle["source"]["duration_seconds"], rel_tol=0, abs_tol=1 / info.samplerate
    ):
        raise ValueError("Source WAV sample rate or duration does not match the event bundle")
    tempo_payload = bundle.get("tempo")
    if tempo_payload is None:
        raise ValueError("A shared source-time tempo map is required to serialize normalized events")
    tempo = TempoMap(
        list(tempo_payload["beats_seconds"]),
        tempo_payload.get("numerator", 4),
        tempo_payload.get("confidence", 0.0),
        tempo_payload.get("warnings", []),
    )
    notes = []
    for event in bundle["events"]:
        if event["modifiers"]:
            raise ValueError(f"Existing RB3 MIDI writer cannot preserve modifiers for event {event['event_id']}")
        if event["duration_status"] == "predicted":
            end = event["offset_seconds"]
        else:
            beat = tempo.beat_at(event["onset_seconds"])
            end = tempo.seconds_at(beat + 1 / 960)
        if end > duration:
            raise ValueError(f"Event {event['event_id']} has no serializable duration before source end")
        notes.append(
            Note(
                event["onset_seconds"],
                end,
                tuple(event["lanes"]),
                0.0 if event["confidence"] is None else event["confidence"],
                event["event_id"],
            )
        )
    charts = {bundle["instrument"]: reductions(notes, bundle["instrument"], tempo)}
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to overwrite review output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        for folder in ("audio", "analysis", "metadata", "midi", "source"):
            (staging / folder).mkdir()
        shutil.copy2(source_audio, staging / "audio" / "song.wav")
        shutil.copy2(source_audio, staging / "source" / "source.wav")
        save_json(
            staging / "source" / "source.json",
            {"path": str(source_audio), "sha256": source_hash},
        )
        save_json(staging / "analysis" / "pretrained-events.json", bundle)
        midi_path = staging / "midi" / "song.mid"
        lead = write_midi(midi_path, tempo, charts, duration, {})
        report = validate(charts, tempo, duration, midi_path)
        if not report["passed"]:
            raise ValueError("Imported event chart failed structural validation: " + "; ".join(report["errors"]))
        metadata = {
            "title": title or source_audio.stem,
            "artist": artist or "Unknown Artist",
            "album": "",
            "year": "",
            "genre": "Other",
            "chart_profile": "pretrained_review",
            "source_seconds": duration,
            "analysis_mode": "external_pretrained_event_import",
            "backend": bundle["backend"],
            "difficulty": bundle["difficulty"],
            "confidence_policy": "unknown confidence is represented as 0 only for Note compatibility; it is not a model score",
            "duration_policy": "unknown durations serialize as one MIDI tick; this is not a predicted sustain",
        }
        save_json(staging / "metadata" / "song.json", metadata)
        save_json(staging / "analysis" / "structural-validation.json", report)
        create_preview(staging, charts, tempo, lead, metadata)
        result = {
            "status": "draft_ready",
            "analysis_mode": "external_pretrained_event_import",
            "folder": str(target),
            "source_sha256": source_hash,
            "event_bundle": str(bundle_path),
            "midi": "midi/song.mid",
            "preview": "review.html",
            "validation": report,
        }
        save_json(staging / "result.json", result)
        _publish_directory_no_replace(staging, target)
        return {**result, "folder": str(target)}
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
