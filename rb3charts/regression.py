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
    reductions = snapshot_payload.get("reductions", {})
    difficulties = reductions.get("difficulties", {})
    expert_rows = difficulties.get("expert", {}).get("notes", [])
    expert_by_source = {
        str(row["source_id"]):row
        for row in expert_rows
    }
    reduction_metrics = {}
    for difficulty in ("expert", "hard", "medium", "easy"):
        rows = difficulties.get(difficulty, {}).get("notes", [])
        row_by_source = {
            str(row["source_id"]):row
            for row in rows
        }
        common = sorted(set(expert_by_source) & set(row_by_source))
        timing_errors = [
            abs(
                float(row_by_source[source_id]["start"])
                - float(expert_by_source[source_id]["start"])
            ) * 1000
            for source_id in common
        ]
        lane_changes = sum(
            tuple(row_by_source[source_id]["lanes"])
            != tuple(expert_by_source[source_id]["lanes"])
            for source_id in common
        )
        reduction_metrics[difficulty] = {
            "attacks":len(rows),
            "attack_retention":round(len(rows) / max(1, len(expert_rows)), 4),
            "retained_source_ids":len(common),
            "orphan_source_ids":sorted(set(row_by_source) - set(expert_by_source)),
            "lane_changes_from_expert":lane_changes,
            "mean_timing_error_ms":(
                round(float(np.mean(timing_errors)), 4)
                if timing_errors else None
            ),
            "max_timing_error_ms":(
                round(float(np.max(timing_errors)), 4)
                if timing_errors else None
            ),
        }
    return {
        "instrument": snapshot_payload.get("instrument"),
        "stage_count": len(stages),
        "stage_rates": stage_rates,
        "transitions": transitions,
        "reductions": reductions,
        "reduction_metrics": reduction_metrics,
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


def _chart_attacks(gems, offset=0.0):
    grouped = defaultdict(list)
    for gem in gems:
        grouped[round(float(gem["start"]) + offset, 6)].append(gem)
    return [
        {
            "time": time,
            "lanes": tuple(sorted(int(gem["lane"]) for gem in rows)),
            "duration": max(float(gem["end"]) - float(gem["start"]) for gem in rows),
        }
        for time, rows in sorted(grouped.items())
    ]


def _match_attacks(reference, candidate, tolerance):
    left = right = 0
    matches = []
    while left < len(reference) and right < len(candidate):
        error = candidate[right]["time"] - reference[left]["time"]
        if abs(error) <= tolerance:
            matches.append((reference[left], candidate[right], error))
            left += 1
            right += 1
        elif error < 0:
            right += 1
        else:
            left += 1
    return matches


def _windowed_attack_metrics(reference, candidate, tolerance, window=4.0):
    if not reference and not candidate:
        return {"window_seconds": window, "worst_reference_recall": None, "windows": []}
    first = min(row["time"] for row in reference + candidate)
    last = max(row["time"] for row in reference + candidate)
    start = np.floor(first / window) * window
    windows = []
    while start <= last:
        end = start + window
        left = [row for row in reference if start <= row["time"] < end]
        right = [row for row in candidate if start <= row["time"] < end]
        if left or right:
            matches = len(_match_attacks(left, right, tolerance))
            precision = matches / max(1, len(right))
            recall = matches / max(1, len(left))
            windows.append({
                "start": round(float(start), 3),
                "end": round(float(end), 3),
                "reference": len(left),
                "candidate": len(right),
                "matched": matches,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(
                    2 * precision * recall / max(1e-12, precision + recall), 4
                ),
            })
        start = end
    active = [row for row in windows if row["reference"]]
    worst = min(active, key=lambda row: (row["recall"], row["start"])) if active else None
    return {
        "window_seconds": window,
        "worst_reference_recall": worst,
        "windows": windows,
    }


def _binary_event_metrics(matches, reference, candidate, predicate):
    true_positive = sum(predicate(left) and predicate(right) for left, right, _ in matches)
    reference_positive = sum(predicate(row) for row in reference)
    candidate_positive = sum(predicate(row) for row in candidate)
    precision = true_positive / max(1, candidate_positive)
    recall = true_positive / max(1, reference_positive)
    return {
        "reference": reference_positive,
        "candidate": candidate_positive,
        "matched": true_positive,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / max(1e-12, precision + recall), 4),
    }


def _peak_density(attacks, window=4.0):
    starts = [row["time"] for row in attacks]
    peak = right = 0
    for left, start in enumerate(starts):
        right = max(right, left)
        while right < len(starts) and starts[right] < start + window:
            right += 1
        peak = max(peak, right - left)
    return round(peak / window, 3)


def _rest_boundaries(attacks, minimum_gap=1.0):
    return [
        (left["time"] + right["time"]) / 2
        for left, right in zip(attacks, attacks[1:])
        if right["time"] - left["time"] >= minimum_gap
    ]


def _motif_agreement(reference, matches, window=4):
    matched_by_time = {
        round(left["time"], 6): right
        for left, right, _ in matches
    }
    families = defaultdict(list)
    for index in range(len(reference) - window + 1):
        phrase = reference[index:index + window]
        if any(len(row["lanes"]) != 1 for row in phrase):
            continue
        gaps = tuple(round(right["time"] - left["time"], 2)
                     for left, right in zip(phrase, phrase[1:]))
        if any(gap <= 0 or gap > 2 for gap in gaps):
            continue
        lanes = tuple(row["lanes"][0] for row in phrase)
        shape = tuple(lane - lanes[0] for lane in lanes)
        families[(gaps, shape)].append(phrase)
    repeated = [phrases for phrases in families.values() if len(phrases) >= 2]
    occurrences = [phrase for phrases in repeated for phrase in phrases]
    covered = agreed = 0
    for phrase in occurrences:
        candidate = [matched_by_time.get(round(row["time"], 6)) for row in phrase]
        if any(row is None or len(row["lanes"]) != 1 for row in candidate):
            continue
        covered += 1
        lanes = tuple(row["lanes"][0] for row in candidate)
        candidate_shape = tuple(lane - lanes[0] for lane in lanes)
        reference_lanes = tuple(row["lanes"][0] for row in phrase)
        reference_shape = tuple(lane - reference_lanes[0] for lane in reference_lanes)
        agreed += int(candidate_shape == reference_shape)
    return {
        "families": len(repeated),
        "reference_occurrences": len(occurrences),
        "covered_occurrences": covered,
        "coverage": round(covered / max(1, len(occurrences)), 4),
        "relative_lane_shape_agreement": round(agreed / max(1, covered), 4),
    }


def compare_chart_attacks(reference, candidate, tolerance=0.08, hold_seconds=0.35):
    matches = _match_attacks(reference, candidate, tolerance)
    errors = [abs(error) * 1000 for _, _, error in matches]
    precision = len(matches) / max(1, len(candidate))
    recall = len(matches) / max(1, len(reference))
    single_pairs = [
        (left, right)
        for left, right, _ in matches
        if len(left["lanes"]) == len(right["lanes"]) == 1
    ]
    movement_pairs = list(zip(single_pairs, single_pairs[1:]))
    direction_matches = 0
    direction_changes = 0
    interval_errors = []
    for ((left_a, right_a), (left_b, right_b)) in movement_pairs:
        reference_move = left_b["lanes"][0] - left_a["lanes"][0]
        candidate_move = right_b["lanes"][0] - right_a["lanes"][0]
        if reference_move:
            direction_changes += 1
            direction_matches += int(np.sign(reference_move) == np.sign(candidate_move))
        interval_errors.append(abs(reference_move - candidate_move))
    duration_errors = [
        abs(left["duration"] - right["duration"]) * 1000
        for left, right, _ in matches
    ]
    reference_boundaries = _rest_boundaries(reference)
    candidate_boundaries = _rest_boundaries(candidate)
    boundary_matches = _match_attacks(
        [{"time": time} for time in reference_boundaries],
        [{"time": time} for time in candidate_boundaries],
        0.25,
    )
    boundary_precision = len(boundary_matches) / max(1, len(candidate_boundaries))
    boundary_recall = len(boundary_matches) / max(1, len(reference_boundaries))
    return {
        "attack": {
            "reference": len(reference),
            "candidate": len(candidate),
            "matched": len(matches),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(2 * precision * recall / max(1e-12, precision + recall), 4),
            "timing_error_ms": {
                "median": round(float(np.median(errors)), 3) if errors else None,
                "p95": round(float(np.quantile(errors, .95)), 3) if errors else None,
            },
        },
        "attack_windows": _windowed_attack_metrics(reference, candidate, tolerance),
        "chord": _binary_event_metrics(
            matches, reference, candidate, lambda row: len(row["lanes"]) > 1
        ),
        "sustain": {
            **_binary_event_metrics(
                matches, reference, candidate,
                lambda row: row["duration"] >= hold_seconds,
            ),
            "matched_duration_error_ms": {
                "median": round(float(np.median(duration_errors)), 3)
                if duration_errors else None,
                "p95": round(float(np.quantile(duration_errors, .95)), 3)
                if duration_errors else None,
            },
        },
        "lane": {
            "exact_agreement": round(
                sum(left["lanes"] == right["lanes"] for left, right, _ in matches)
                / max(1, len(matches)), 4
            ),
            "single_note_pairs": len(single_pairs),
            "movement_direction_agreement": round(
                direction_matches / max(1, direction_changes), 4
            ),
            "mean_interval_error": round(float(np.mean(interval_errors)), 4)
            if interval_errors else None,
        },
        "density": {
            "reference_peak_four_second_nps": _peak_density(reference),
            "candidate_peak_four_second_nps": _peak_density(candidate),
            "attack_count_ratio": round(len(candidate) / max(1, len(reference)), 4),
        },
        "phrase_boundaries": {
            "reference": len(reference_boundaries),
            "candidate": len(candidate_boundaries),
            "matched_250ms": len(boundary_matches),
            "precision": round(boundary_precision, 4),
            "recall": round(boundary_recall, 4),
            "f1": round(
                2 * boundary_precision * boundary_recall
                / max(1e-12, boundary_precision + boundary_recall), 4
            ),
        },
        "motifs": _motif_agreement(reference, matches),
    }


def compare_authorized_chart(manifest, song_id, generated_folder, output=None):
    from .datasets import validate_chart_pair_manifest, _read_expert_gems

    manifest = Path(manifest).resolve()
    generated_folder = Path(generated_folder).resolve()
    validation = validate_chart_pair_manifest(manifest, verify_hashes=True)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    item = next((row for row in payload["items"] if str(row["song_id"]) == str(song_id)), None)
    if item is None:
        raise ValueError(f"song_id is not present in chart-pair manifest: {song_id}")
    metadata_path = generated_folder / "metadata" / "song.json"
    generated_midi = generated_folder / "midi" / "notes.mid"
    if not metadata_path.is_file() or not generated_midi.is_file():
        raise FileNotFoundError("Generated folder requires metadata/song.json and midi/notes.mid")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    lead = float(metadata.get("lead_seconds", 0))
    reference_path = manifest.parent / item["annotations"]
    difficulties = {}
    for difficulty in ("expert", "hard", "medium", "easy"):
        reference = _chart_attacks(
            _read_expert_gems(reference_path, item["chart_format"], difficulty)
        )
        candidate = _chart_attacks(
            _read_expert_gems(generated_midi, "rock_band_midi", difficulty),
            offset=-lead,
        )
        difficulties[difficulty] = compare_chart_attacks(reference, candidate)
    expert_attacks = difficulties["expert"]["attack"]["reference"]
    report = {
        "schema_version": 1,
        "scope": (
            "post-generation evaluation only; authorized human notes were not "
            "generation inputs"
        ),
        "song_id": str(song_id),
        "manifest_sha256": validation["manifest_sha256"],
        "generated_folder": str(generated_folder),
        "generated_audio_lead_seconds": lead,
        "difficulties": difficulties,
        "difficulty_retention": {
            difficulty: {
                "reference": round(
                    difficulties[difficulty]["attack"]["reference"]
                    / max(1, expert_attacks), 4
                ),
                "candidate": round(
                    difficulties[difficulty]["attack"]["candidate"]
                    / max(1, difficulties["expert"]["attack"]["candidate"]), 4
                ),
            }
            for difficulty in ("hard", "medium", "easy")
        },
    }
    output = (
        Path(output)
        if output is not None
        else generated_folder / "analysis" / "human-chart-comparison.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
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
