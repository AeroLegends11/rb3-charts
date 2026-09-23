from __future__ import annotations

from collections import defaultdict
from importlib import metadata as importlib_metadata
from pathlib import Path
import hashlib
import json
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


def _benchmark_runs(root: Path):
    runs={}
    for metadata_path in sorted(Path(root).rglob("metadata/song.json")):
        metadata=json.loads(metadata_path.read_text(encoding="utf-8"))
        key=(
            str(metadata.get("artist","")).strip().casefold(),
            str(metadata.get("title","")).strip().casefold(),
        )
        folder=metadata_path.parents[1]
        if (folder/"analysis"/"guitar-stage-snapshots.json").is_file():
            runs[key]=(folder,metadata)
    return runs


def _benchmark_stage_metrics(folder: Path):
    payload=json.loads(
        (folder/"analysis"/"guitar-stage-snapshots.json").read_text(encoding="utf-8")
    )
    stages=[stage for stage in payload.get("stages",[]) if "notes" in stage]
    if not stages:
        raise ValueError(f"No guitar note stages in {folder}")
    notes=_snapshots_to_notes(stages[-1]["notes"])
    shape=_shape_metrics(notes)
    shape["chords"]=sum(len(note.lanes)>1 for note in notes)
    shape["holds"]=sum(note.end-note.start>=.22 for note in notes)
    starts=np.asarray(sorted(note.start for note in notes),dtype=float)
    peak=0
    right=0
    for left,start in enumerate(starts):
        right=max(right,left)
        while right<len(starts) and starts[right]<start+4:
            right+=1
        peak=max(peak,right-left)
    shape.update({
        "lane_coverage":sorted({lane for note in notes for lane in note.lanes}),
        "peak_four_second_nps":round(peak/4,3),
    })
    assessment_path=folder/"analysis"/"quality-assessment.json"
    shape["assessment_status"]=(
        json.loads(assessment_path.read_text(encoding="utf-8")).get("status")
        if assessment_path.is_file() else None
    )
    return shape


def benchmark_replays(baseline_root: Path, candidate_root: Path, output: Path):
    baseline=_benchmark_runs(Path(baseline_root))
    candidate=_benchmark_runs(Path(candidate_root))
    shared=sorted(set(baseline)&set(candidate))
    if not shared:
        raise ValueError("No matching replay songs with guitar stage snapshots")
    songs=[]
    total_keys=("attacks","single_notes","chords","holds",
                "collapsed_pitch_changes","reversed_pitch_changes")
    totals={
        "baseline":{key:0 for key in total_keys},
        "candidate":{key:0 for key in total_keys},
    }
    densities={"baseline":[],"candidate":[]}
    lanes={"baseline":set(),"candidate":set()}
    statuses={"baseline":defaultdict(int),"candidate":defaultdict(int)}
    for key in shared:
        baseline_folder,metadata=baseline[key]
        candidate_folder,_=candidate[key]
        before=_benchmark_stage_metrics(baseline_folder)
        after=_benchmark_stage_metrics(candidate_folder)
        for metric in total_keys:
            totals["baseline"][metric]+=before[metric]
            totals["candidate"][metric]+=after[metric]
        for label,metrics in (("baseline",before),("candidate",after)):
            densities[label].append(metrics["peak_four_second_nps"])
            lanes[label].update(metrics["lane_coverage"])
            statuses[label][metrics["assessment_status"] or "missing"]+=1
        songs.append({
            "artist":metadata.get("artist",""),
            "title":metadata.get("title",""),
            "baseline_folder":str(baseline_folder.resolve()),
            "candidate_folder":str(candidate_folder.resolve()),
            "baseline":before,
            "candidate":after,
            "delta":{
                metric:round(after[metric]-before[metric],4)
                for metric in before if isinstance(before[metric],(int,float))
            },
        })
    for label in ("baseline","candidate"):
        total_attacks=max(1,totals[label]["attacks"])
        totals[label].update({
            "chord_rate":round(totals[label]["chords"]/total_attacks,4),
            "hold_rate":round(totals[label]["holds"]/total_attacks,4),
            "mean_peak_four_second_nps":round(float(np.mean(densities[label])),3),
            "max_peak_four_second_nps":round(float(np.max(densities[label])),3),
            "lane_coverage":sorted(lanes[label]),
            "assessment_status_counts":dict(sorted(statuses[label].items())),
        })
    delta={
        key:round(totals["candidate"][key]-totals["baseline"][key],4)
        for key in totals["baseline"]
        if isinstance(totals["baseline"][key],(int,float))
    }
    report={
        "schema_version":1,
        "baseline_root":str(Path(baseline_root).resolve()),
        "candidate_root":str(Path(candidate_root).resolve()),
        "songs_matched":len(shared),
        "baseline_only":[{"artist":key[0],"title":key[1]} for key in sorted(set(baseline)-set(candidate))],
        "candidate_only":[{"artist":key[0],"title":key[1]} for key in sorted(set(candidate)-set(baseline))],
        "macro":{"baseline":totals["baseline"],"candidate":totals["candidate"],"delta":delta},
        "songs":songs,
    }
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return report


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
