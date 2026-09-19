from __future__ import annotations

from collections import defaultdict
from importlib import metadata as importlib_metadata
from pathlib import Path
import hashlib
import platform
import subprocess

import numpy as np

from .model import Note

MOTIF_RHYTHM_SUBDIVISION = 12  # twelfth-beat bins match existing phrase signature quantization


def note_snapshot(note: Note):
    source_ids = [item for item in str(note.source_id).split("+") if item]
    return {
        "source_id": note.source_id,
        "source_ids": source_ids,
        "start": round(float(note.start), 6),
        "end": round(float(note.end), 6),
        "lanes": list(note.lanes),
        "pitch": (float(note.pitch) if note.pitch is not None else None),
        "duration": round(float(note.end - note.start), 6),
        "attack": round(float(note.start), 6),
    }


def by_source_id(notes: list[Note]):
    groups = defaultdict(list)
    for note in sorted(notes, key=lambda row: (row.start, row.end, row.source_id)):
        groups[str(note.source_id)].append(note_snapshot(note))
    return dict(groups)


def stage_entry(stage: str, notes: list[Note]):
    ordered = sorted(notes, key=lambda row: (row.start, row.end, row.source_id))
    return {"stage": stage, "notes": [note_snapshot(note) for note in ordered], "by_source_id": by_source_id(ordered)}


def reductions_entry(stage: str, charts: dict[str, list[Note]]):
    return {"stage": stage, "difficulties": {difficulty: stage_entry(f"{stage}:{difficulty}", notes) for difficulty, notes in charts.items()}}


def _region_key(tempo, seconds: float, beats_per_region: int = 8):
    beat = float(tempo.beat_at(seconds))
    start = int(np.floor(beat / beats_per_region) * beats_per_region)
    return f"beat_{start}_{start + beats_per_region}"


def _snapshots_to_notes(rows):
    return [
        Note(
            start=float(row["start"]),
            end=float(row["end"]),
            lanes=tuple(int(lane) for lane in row["lanes"]),
            confidence=1.0,
            source_id=str(row["source_id"]),
            pitch=(float(row["pitch"]) if row.get("pitch") is not None else None),
        )
        for row in rows
    ]


def _shape_metrics(notes: list[Note]):
    ordered = sorted(notes, key=lambda row: row.start)
    if not ordered:
        return {
            "attacks": 0,
            "single_notes": 0,
            "chord_rate": 0.0,
            "hold_rate": 0.0,
            "collapsed_pitch_changes": 0,
            "reversed_pitch_changes": 0,
        }
    chords = sum(len(note.lanes) > 1 for note in ordered)
    holds = sum(note.end - note.start >= 0.22 for note in ordered)
    singles = sum(len(note.lanes) == 1 for note in ordered)
    pairs = [(a, b) for a, b in zip(ordered, ordered[1:]) if len(a.lanes) == len(b.lanes) == 1 and b.start - a.start < 0.8]
    collapsed = sum(
        a.pitch is not None and b.pitch is not None and a.pitch != b.pitch and a.lanes == b.lanes for a, b in pairs
    )
    reversed_moves = sum(
        a.pitch is not None and b.pitch is not None and (b.pitch - a.pitch) * (b.lanes[0] - a.lanes[0]) < 0 for a, b in pairs
    )
    return {
        "attacks": len(ordered),
        "single_notes": singles,
        "chord_rate": round(chords / len(ordered), 4),
        "hold_rate": round(holds / len(ordered), 4),
        "collapsed_pitch_changes": int(collapsed),
        "reversed_pitch_changes": int(reversed_moves),
    }


def _motif_windows(notes: list[Note], tempo, window: int = 4):
    ordered = sorted((note for note in notes if len(note.lanes) == 1), key=lambda row: row.start)
    motifs = defaultdict(list)
    considered = 0
    for index in range(len(ordered) - window + 1):
        phrase = ordered[index:index + window]
        if any(note.pitch is None for note in phrase):
            continue
        beat_gaps = [float(tempo.beat_at(b.start) - tempo.beat_at(a.start)) for a, b in zip(phrase, phrase[1:])]
        if any(gap <= 0 or gap > 2 for gap in beat_gaps):
            continue
        contour = tuple(int(np.sign(b.pitch - a.pitch)) for a, b in zip(phrase, phrase[1:]))
        rhythm = tuple(int(round(gap * MOTIF_RHYTHM_SUBDIVISION)) for gap in beat_gaps)
        motifs[(contour, rhythm)].append(phrase)
        considered += 1
    repeated = {key: rows for key, rows in motifs.items() if len(rows) >= 2}
    return repeated, considered


def _lane_agreement(previous_notes: list[Note], next_stage_by_source: dict[str, list[dict]], tempo):
    motifs, candidate_windows = _motif_windows(previous_notes, tempo)
    matched_windows = 0
    absolute_matches = 0
    relative_matches = 0
    false_locks = 0
    for occurrences in motifs.values():
        for phrase in occurrences:
            target_lanes = []
            source_lanes = [note.lanes[0] for note in phrase]
            for note in phrase:
                snapshots = next_stage_by_source.get(str(note.source_id), [])
                if not snapshots:
                    target_lanes = []
                    break
                lanes = snapshots[0].get("lanes", [])
                if len(lanes) != 1:
                    target_lanes = []
                    break
                target_lanes.append(int(lanes[0]))
            if len(target_lanes) != len(source_lanes):
                continue
            matched_windows += 1
            absolute_ok = tuple(target_lanes) == tuple(source_lanes)
            relative_ok = tuple(lane - target_lanes[0] for lane in target_lanes) == tuple(lane - source_lanes[0] for lane in source_lanes)
            absolute_matches += int(absolute_ok)
            relative_matches += int(relative_ok)
            if not (absolute_ok or relative_ok):
                false_locks += 1
    return {
        "candidate_windows": candidate_windows,
        "detected_motif_windows": matched_windows,
        "coverage": round(matched_windows / max(1, candidate_windows), 4),
        "absolute_lane_agreement": round(absolute_matches / max(1, matched_windows), 4),
        "relative_shape_agreement": round(relative_matches / max(1, matched_windows), 4),
        "false_lock_windows": false_locks,
        "false_lock_rate": round(false_locks / max(1, matched_windows), 4),
    }


def transition_metrics(previous_stage: dict, next_stage: dict, tempo):
    previous = previous_stage["notes"]
    following = next_stage["notes"]
    following_by_source = next_stage["by_source_id"]
    previous_ids = {str(note["source_id"]) for note in previous}
    following_ids = {str(note["source_id"]) for note in following}
    retained_ids = sorted(previous_ids & following_ids)
    previous_lookup = defaultdict(list)
    for note in previous:
        previous_lookup[str(note["source_id"])].append(note)
    timing_errors = []
    by_region = defaultdict(lambda: {"previous": 0, "retained": 0, "errors": []})
    for note in previous:
        by_region[_region_key(tempo, float(note["start"]))]["previous"] += 1
    for source_id in retained_ids:
        previous_note = previous_lookup[source_id][0]
        next_note = following_by_source[source_id][0]
        error_ms = abs(float(next_note["start"]) - float(previous_note["start"])) * 1000
        timing_errors.append(error_ms)
        region = _region_key(tempo, float(previous_note["start"]))
        by_region[region]["retained"] += 1
        by_region[region]["errors"].append(error_ms)
    regions = {}
    for key, row in sorted(by_region.items()):
        regions[key] = {
            "previous_attacks": row["previous"],
            "retained_attacks": row["retained"],
            "attack_retention": round(row["retained"] / max(1, row["previous"]), 4),
            "mean_timing_error_ms": (round(float(np.mean(row["errors"])), 4) if row["errors"] else None),
        }
    previous_notes = _snapshots_to_notes(previous)
    following_notes = _snapshots_to_notes(following)
    return {
        "from_stage": previous_stage["stage"],
        "to_stage": next_stage["stage"],
        "previous_attacks": len(previous),
        "retained_attacks": len(retained_ids),
        "attack_retention": round(len(retained_ids) / max(1, len(previous)), 4),
        "mean_timing_error_ms": (round(float(np.mean(timing_errors)), 4) if timing_errors else None),
        "max_timing_error_ms": (round(float(np.max(timing_errors)), 4) if timing_errors else None),
        "contour": {"from": _shape_metrics(previous_notes), "to": _shape_metrics(following_notes)},
        "lane_agreement": _lane_agreement(previous_notes, following_by_source, tempo),
        "regions": regions,
    }


def compute_regression_metrics(snapshot_payload: dict, tempo):
    stages = [stage for stage in snapshot_payload.get("stages", []) if "notes" in stage]
    transitions = [transition_metrics(a, b, tempo) for a, b in zip(stages, stages[1:])]
    stage_rates = {stage["stage"]: _shape_metrics(_snapshots_to_notes(stage["notes"])) for stage in stages}
    return {
        "instrument": snapshot_payload.get("instrument"),
        "stage_count": len(stages),
        "stage_rates": stage_rates,
        "transitions": transitions,
        "reductions": snapshot_payload.get("reductions", {}),
    }


def summarize_metrics(metrics: dict):
    transitions = metrics.get("transitions", [])
    if not transitions:
        return "Regression metrics: no stage transitions were recorded."
    retention = float(np.mean([item["attack_retention"] for item in transitions]))
    errors = [item["mean_timing_error_ms"] for item in transitions if item["mean_timing_error_ms"] is not None]
    timing = float(np.mean(errors)) if errors else 0.0
    return f"Regression metrics: {len(transitions)} transitions, mean attack retention {retention:.3f}, mean matched timing error {timing:.2f} ms."


def hash_file(path: Path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def runtime_versions():
    versions = {"python": platform.python_version()}
    for package in ("numpy", "scipy", "librosa", "soundfile", "onnxruntime", "basic-pitch", "pretty_midi"):
        try:
            versions[package] = importlib_metadata.version(package)
        except Exception:
            pass
    return versions


def git_state(root: Path):
    try:
        revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        revision = None
    try:
        dirty = bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip())
    except Exception:
        dirty = None
    return {"revision": revision, "dirty": dirty}
