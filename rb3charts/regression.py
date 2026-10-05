from __future__ import annotations

from collections import Counter, defaultdict
from importlib import metadata as importlib_metadata
from pathlib import Path
import hashlib
import json
import os
import platform
import subprocess
from urllib.parse import quote

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


def _benchmark_runs(root: Path, instrument: str):
    runs={}
    for metadata_path in sorted(Path(root).rglob("metadata/song.json")):
        metadata=json.loads(metadata_path.read_text(encoding="utf-8"))
        key=(
            str(metadata.get("artist","")).strip().casefold(),
            str(metadata.get("title","")).strip().casefold(),
        )
        if not all(key):
            raise ValueError(
                f"Replay metadata requires artist and title: {metadata_path}"
            )
        folder=metadata_path.parents[1]
        if (folder/"analysis"/f"{instrument}-stage-snapshots.json").is_file():
            if key in runs:
                raise ValueError(
                    f"Duplicate replay song for {instrument}: "
                    f"{metadata.get('artist','')} - "
                    f"{metadata.get('title','')} appears in "
                    f"{runs[key][0]} and {folder}"
                )
            runs[key]=(folder,metadata)
    return runs


def _benchmark_stage_metrics(folder: Path, instrument: str):
    payload=json.loads(
        (folder/"analysis"/f"{instrument}-stage-snapshots.json").read_text(
            encoding="utf-8"
        )
    )
    stages=[stage for stage in payload.get("stages",[]) if "notes" in stage]
    if not stages:
        raise ValueError(f"No {instrument} note stages in {folder}")
    notes=_snapshots_to_notes(stages[-1]["notes"])
    shape=_shape_metrics(notes)
    simultaneous=sum(len(note.lanes)>1 for note in notes)
    shape["simultaneous_attacks"]=simultaneous
    shape["chords"]=simultaneous if instrument!="drums" else 0
    shape["holds"]=sum(note.end-note.start>=.22 for note in notes)
    shape.update({
        "lane_coverage":sorted({lane for note in notes for lane in note.lanes}),
        "peak_four_second_nps":_peak_four_second_nps(notes),
    })
    reductions=payload.get("reductions",{}).get("difficulties",{})
    shape["reductions_available"]=all(
        difficulty in reductions
        for difficulty in ("expert","hard","medium","easy")
    )
    for difficulty in ("expert","hard","medium","easy"):
        reduction=reductions.get(difficulty,{})
        shape[f"{difficulty}_attacks"]=(
            len(reduction.get("notes",[]))
            if shape["reductions_available"] else None
        )
        shape[f"{difficulty}_peak_four_second_nps"]=(
            _peak_four_second_nps(
                _snapshots_to_notes(reduction.get("notes",[]))
            )
            if shape["reductions_available"] else None
        )
    assessment_path=folder/"analysis"/"quality-assessment.json"
    shape["assessment_status"]=(
        json.loads(assessment_path.read_text(encoding="utf-8")).get("status")
        if assessment_path.is_file() else None
    )
    return shape


def _benchmark_structure_metrics(folder: Path):
    path=folder/"analysis"/"structure-diagnostics.json"
    if not path.is_file():
        return None
    payload=json.loads(path.read_text(encoding="utf-8"))
    boundaries=payload.get("strongest_activity_boundaries",[])
    grooves=payload.get("drum_grooves",{})
    return {
        "activity_blocks":len(payload.get("sections",[])),
        "activity_boundaries":len(boundaries),
        "mean_boundary_change":round(float(np.mean([
            float(item.get("activity_change",0))
            for item in boundaries
        ])),4) if boundaries else 0.0,
        "max_boundary_change":round(max(
            [float(item.get("activity_change",0)) for item in boundaries],
            default=0.0,
        ),4),
        "repeated_drum_groove_families":int(
            grooves.get("repeated_families",0)
        ),
        "drum_groove_supported_attack_ratio":float(
            grooves.get("supported_attack_ratio",0)
        ),
    }


def _benchmark_pattern_metrics(folder: Path, instrument: str):
    path=folder/"analysis"/"patterns.json"
    if not path.is_file():
        return None
    payload=json.loads(path.read_text(encoding="utf-8")).get(instrument)
    if not payload:
        return None
    attacks=max(1,int(payload.get("attacks",0)))
    metrics={
        "recurring_pattern_families":int(
            payload.get("recurring_pattern_families",0)
        ),
        "recurring_pattern_occurrences":int(
            payload.get("recurring_pattern_occurrences",0)
        ),
        "pattern_supported_attack_ratio":round(
            int(payload.get("pattern_supported_attacks",0))/attacks,4
        ),
    }
    for key in (
        "contour_agreement_ratio",
        "flattened_change_ratio",
        "adjacent_motion_rate",
        "wide_jump_rate",
    ):
        if key in payload:
            metrics[key]=float(payload[key])
    return metrics


def _peak_four_second_nps(notes):
    starts=np.asarray(sorted(note.start for note in notes),dtype=float)
    peak=0
    right=0
    for left,start in enumerate(starts):
        right=max(right,left)
        while right<len(starts) and starts[right]<start+4:
            right+=1
        peak=max(peak,right-left)
    return round(peak/4,3)


def benchmark_replays(
    baseline_root: Path,
    candidate_root: Path,
    output: Path | None,
    instrument: str = "guitar",
):
    if instrument not in ("guitar","bass","drums","keys"):
        raise ValueError(f"Unsupported benchmark instrument: {instrument}")
    baseline=_benchmark_runs(Path(baseline_root),instrument)
    candidate=_benchmark_runs(Path(candidate_root),instrument)
    shared=sorted(set(baseline)&set(candidate))
    if not shared:
        raise ValueError(
            f"No matching replay songs with {instrument} stage snapshots"
        )
    songs=[]
    core_total_keys=(
        "attacks","single_notes","simultaneous_attacks","chords","holds",
        "collapsed_pitch_changes","reversed_pitch_changes",
    )
    reduction_total_keys=(
        "expert_attacks","hard_attacks","medium_attacks","easy_attacks",
    )
    total_keys=core_total_keys+reduction_total_keys
    totals={
        "baseline":{key:0 for key in total_keys},
        "candidate":{key:0 for key in total_keys},
    }
    densities={"baseline":[],"candidate":[]}
    lanes={"baseline":set(),"candidate":set()}
    statuses={"baseline":defaultdict(int),"candidate":defaultdict(int)}
    reduction_pairs=0
    structure_pairs=[]
    pattern_pairs=[]
    reduction_densities={
        label:{difficulty:[] for difficulty in (
            "expert","hard","medium","easy"
        )}
        for label in ("baseline","candidate")
    }
    for key in shared:
        baseline_folder,metadata=baseline[key]
        candidate_folder,_=candidate[key]
        before=_benchmark_stage_metrics(baseline_folder,instrument)
        after=_benchmark_stage_metrics(candidate_folder,instrument)
        before_structure=_benchmark_structure_metrics(baseline_folder)
        after_structure=_benchmark_structure_metrics(candidate_folder)
        before_pattern=_benchmark_pattern_metrics(
            baseline_folder,instrument
        )
        after_pattern=_benchmark_pattern_metrics(
            candidate_folder,instrument
        )
        for metric in core_total_keys:
            totals["baseline"][metric]+=before[metric]
            totals["candidate"][metric]+=after[metric]
        if (
            before["reductions_available"]
            and after["reductions_available"]
        ):
            reduction_pairs+=1
            for metric in reduction_total_keys:
                totals["baseline"][metric]+=before[metric]
                totals["candidate"][metric]+=after[metric]
            for difficulty in ("expert","hard","medium","easy"):
                reduction_densities["baseline"][difficulty].append(
                    before[f"{difficulty}_peak_four_second_nps"]
                )
                reduction_densities["candidate"][difficulty].append(
                    after[f"{difficulty}_peak_four_second_nps"]
                )
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
                for metric in before
                if (
                    isinstance(before[metric],(int,float))
                    and not isinstance(before[metric],bool)
                    and after[metric] is not None
                )
            },
            "baseline_structure":before_structure,
            "candidate_structure":after_structure,
            "baseline_pattern":before_pattern,
            "candidate_pattern":after_pattern,
        })
        if before_structure is not None and after_structure is not None:
            structure_pairs.append((before_structure,after_structure))
        if before_pattern is not None and after_pattern is not None:
            pattern_pairs.append((before_pattern,after_pattern))
    for label in ("baseline","candidate"):
        total_attacks=max(1,totals[label]["attacks"])
        totals[label].update({
            "chord_rate":round(totals[label]["chords"]/total_attacks,4),
            "simultaneous_attack_rate":round(
                totals[label]["simultaneous_attacks"]/total_attacks,4
            ),
            "hold_rate":round(totals[label]["holds"]/total_attacks,4),
            "mean_peak_four_second_nps":round(float(np.mean(densities[label])),3),
            "max_peak_four_second_nps":round(float(np.max(densities[label])),3),
            "lane_coverage":sorted(lanes[label]),
            "assessment_status_counts":dict(sorted(statuses[label].items())),
        })
        for difficulty in ("hard","medium","easy"):
            totals[label][f"{difficulty}_retention"]=(
                round(
                    totals[label][f"{difficulty}_attacks"]
                    / max(1,totals[label]["expert_attacks"]),
                    4,
                )
                if reduction_pairs else None
            )
        totals[label]["reduction_songs"]=reduction_pairs
        for difficulty in ("expert","hard","medium","easy"):
            values=reduction_densities[label][difficulty]
            totals[label][
                f"mean_{difficulty}_peak_four_second_nps"
            ]=round(float(np.mean(values)),3) if values else None
            totals[label][
                f"max_{difficulty}_peak_four_second_nps"
            ]=round(float(np.max(values)),3) if values else None
    delta={}
    for key,value in totals["baseline"].items():
        if not isinstance(value,(int,float)):
            continue
        delta[key]=(
            None
            if key in reduction_total_keys and not reduction_pairs
            else round(totals["candidate"][key]-value,4)
        )
    report={
        "schema_version":2,
        "instrument":instrument,
        "baseline_root":str(Path(baseline_root).resolve()),
        "candidate_root":str(Path(candidate_root).resolve()),
        "songs_matched":len(shared),
        "baseline_only":[{"artist":key[0],"title":key[1]} for key in sorted(set(baseline)-set(candidate))],
        "candidate_only":[{"artist":key[0],"title":key[1]} for key in sorted(set(candidate)-set(baseline))],
        "macro":{"baseline":totals["baseline"],"candidate":totals["candidate"],"delta":delta},
        "songs":songs,
    }
    structure_keys=(
        "activity_blocks",
        "activity_boundaries",
        "mean_boundary_change",
        "max_boundary_change",
        "repeated_drum_groove_families",
        "drum_groove_supported_attack_ratio",
    )
    structure_macro={"songs_matched":len(structure_pairs)}
    for label,index in (("baseline",0),("candidate",1)):
        structure_macro[label]={
            key:round(float(np.mean([
                pair[index][key] for pair in structure_pairs
            ])),4)
            for key in structure_keys
        } if structure_pairs else None
    structure_macro["delta"]={
        key:round(
            structure_macro["candidate"][key]
            - structure_macro["baseline"][key],
            4,
        )
        for key in structure_keys
    } if structure_pairs else None
    report["structure_macro_song_mean"]=structure_macro
    pattern_macro={"songs_matched":len(pattern_pairs)}
    pattern_keys=(
        sorted(set.intersection(*[
            set(before)&set(after)
            for before,after in pattern_pairs
        ]))
        if pattern_pairs else []
    )
    for label,index in (("baseline",0),("candidate",1)):
        pattern_macro[label]={
            key:round(float(np.mean([
                pair[index][key] for pair in pattern_pairs
            ])),4)
            for key in pattern_keys
        } if pattern_pairs else None
    pattern_macro["delta"]={
        key:round(
            pattern_macro["candidate"][key]
            - pattern_macro["baseline"][key],
            4,
        )
        for key in pattern_keys
    } if pattern_pairs else None
    report["pattern_macro_song_mean"]=pattern_macro
    if output is not None:
        output=Path(output)
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(
            json.dumps(report,indent=2)+"\n",encoding="utf-8"
        )
    return report


def benchmark_all_replays(
    baseline_root: Path,
    candidate_root: Path,
    output: Path,
):
    reports={}
    unavailable={}
    for instrument in ("guitar","bass","drums","keys"):
        try:
            reports[instrument]=benchmark_replays(
                baseline_root,
                candidate_root,
                None,
                instrument=instrument,
            )
        except ValueError as exc:
            if not str(exc).startswith("No matching replay songs"):
                raise
            unavailable[instrument]=str(exc)
    if not reports:
        raise ValueError(
            "No matching replay songs for any supported instrument"
        )
    combined={
        "schema_version":1,
        "baseline_root":str(Path(baseline_root).resolve()),
        "candidate_root":str(Path(candidate_root).resolve()),
        "instruments":reports,
        "unavailable":unavailable,
    }
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(
        json.dumps(combined,indent=2)+"\n",encoding="utf-8"
    )
    return combined


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


def estimate_attack_offset(reference, candidate, tolerance=0.08, limit=10.0):
    """Estimate a global candidate-to-reference offset from attack onsets."""
    if not reference or not candidate:
        return 0.0
    bins=defaultdict(int)
    for left in reference:
        for right in candidate:
            offset=left["time"]-right["time"]
            if abs(offset)<=limit:
                bins[round(offset,2)]+=1
    best_score=None
    best_offset=0.0
    for coarse,_ in sorted(
        bins.items(),key=lambda item:(-item[1],abs(item[0]),item[0])
    )[:40]:
        shifted=[
            {**row,"time":row["time"]+coarse}
            for row in candidate
        ]
        matches=_match_attacks(reference,shifted,tolerance)
        error=sum(abs(delta) for _,_,delta in matches)
        score=(len(matches),-error,-abs(coarse),-coarse)
        if best_score is None or score>best_score:
            best_score=score
            best_offset=coarse
    return round(float(best_offset),6)


def _rank_percentiles(values):
    values=np.asarray(values,dtype=float)
    if not len(values):
        return np.asarray([],dtype=float)
    order=np.argsort(np.argsort(values,kind="mergesort"),kind="mergesort")
    return order / max(1,len(values)-1)


def _audio_onset_peaks(path):
    import librosa
    from scipy import signal

    y,sr=librosa.load(path,sr=22050,mono=True)
    hop=256
    envelope=librosa.onset.onset_strength(y=y,sr=sr,hop_length=hop)
    peaks,properties=signal.find_peaks(
        envelope,
        distance=3,
        prominence=max(1e-8,float(np.max(envelope))*.01),
    )
    widths=signal.peak_widths(envelope,peaks,rel_height=.5)[0]
    return {
        "times":librosa.frames_to_time(peaks,sr=sr,hop_length=hop),
        "height_rank":_rank_percentiles(envelope[peaks]),
        "prominence_rank":_rank_percentiles(properties["prominences"]),
        "width_rank":_rank_percentiles(widths),
    }


def _nearest_onset_feature(payload,seconds,tolerance=.05):
    times=payload["times"]
    if not len(times):
        return {"found":False}
    index=int(np.argmin(np.abs(times-seconds)))
    error=abs(float(times[index])-seconds)
    if error>tolerance:
        return {"found":False}
    return {
        "found":True,
        "error_ms":round(error*1000,3),
        "height_rank":round(float(payload["height_rank"][index]),4),
        "prominence_rank":round(float(payload["prominence_rank"][index]),4),
        "width_rank":round(float(payload["width_rank"][index]),4),
    }


def audit_playability_evidence(
    folder,
    output=None,
    structure_predictions=None,
    reference=None,
    reference_format="rock_band_midi",
    tolerance=.08,
    offset=None,
):
    """Join post-generation removals to independent source evidence."""
    folder=Path(folder).resolve()
    analysis=folder/"analysis"
    playability=json.loads((analysis/"playability.json").read_text(encoding="utf-8"))
    guitar=playability.get("guitar",{})
    attribution=guitar.get("removal_attribution")
    if not isinstance(attribution,dict):
        raise ValueError("Run lacks playability removal attribution")
    transcription=json.loads(
        (analysis/"guitar-transcription.json").read_text(encoding="utf-8")
    )
    raw_exact={str(row["source_id"]):row for row in transcription}
    raw_components={}
    for row in transcription:
        for source_id in str(row["source_id"]).split("+"):
            raw_components.setdefault(source_id,[]).append(row)
    voice_path=analysis/"guitar-voice-continuity.json"
    voice=(
        json.loads(voice_path.read_text(encoding="utf-8")).get("windows",[])
        if voice_path.is_file() else []
    )
    structure=[]
    if structure_predictions is not None:
        payload=json.loads(Path(structure_predictions).read_text(encoding="utf-8"))
        structure=payload.get("candidates",[])
    structure_times=np.asarray(
        [float(row["start"]) for row in structure],dtype=float
    )
    source_payload=json.loads(
        (folder/"source"/"source.json").read_text(encoding="utf-8")
    )
    stems=json.loads((analysis/"stems.json").read_text(encoding="utf-8"))
    source_path=Path(source_payload["path"])
    stem_path=Path(stems["guitar"])
    source_hash=hash_file(source_path)
    stem_hash=hash_file(stem_path)
    source_onsets=_audio_onset_peaks(source_path)
    stem_onsets=(
        source_onsets if source_hash==stem_hash else _audio_onset_peaks(stem_path)
    )
    rows=[]
    for reason,summary in attribution.items():
        for event in summary.get("events",[]):
            source_id=str(event["source_id"])
            raw=[raw_exact[source_id]] if source_id in raw_exact else [
                row
                for component in source_id.split("+")
                for row in raw_components.get(component,[])
            ]
            evidence=[row.get("evidence",{}) for row in raw]
            seconds=float(event["start"])
            structure_row=None
            if len(structure_times):
                index=int(np.argmin(np.abs(structure_times-seconds)))
                if abs(float(structure_times[index])-seconds)<=.05:
                    structure_row=structure[index]
            containing=[
                window for window in voice
                if float(window["start_seconds"])<=seconds<float(window["end_seconds"])
            ]
            voice_row=max(
                containing,
                key=lambda row:float(row.get("dominant_evidence_share",0)),
                default=None,
            )
            source_feature=_nearest_onset_feature(source_onsets,seconds)
            stem_feature=_nearest_onset_feature(stem_onsets,seconds)
            rows.append({
                **event,
                "removal_reason":reason,
                "raw_evidence":{
                    "candidate_count":len(raw),
                    "model_onset_strength":round(max(
                        [float(row.get("model_onset_strength") or 0) for row in raw]
                        or [0]
                    ),4),
                    "attack_strength":round(max(
                        [float(row.get("attack_strength") or 0) for row in evidence]
                        or [0]
                    ),4),
                    "relative_level":round(max(
                        [float(row.get("relative_level") or 0) for row in evidence]
                        or [0]
                    ),4),
                    "evidence_score":round(max(
                        [float(row.get("evidence_score") or 0) for row in evidence]
                        or [0]
                    ),4),
                },
                "structure_evidence":(
                    {
                        "found":True,
                        "error_ms":round(
                            abs(float(structure_row["start"])-seconds)*1000,3
                        ),
                        "attack_probability":round(
                            float(structure_row["attack_probability"]),4
                        ),
                        "periodic_width_attack_rank_percentile":round(
                            float(structure_row[
                                "periodic_width_attack_rank_percentile"
                            ]),4
                        ),
                        "peak_width_rank_percentile":round(
                            float(structure_row["peak_width_rank_percentile"]),4
                        ),
                    } if structure_row is not None else {"found":False}
                ),
                "source_mix_onset":source_feature,
                "guitar_stem_onset":stem_feature,
                "independent_stem_mix_agreement":bool(
                    source_hash!=stem_hash
                    and source_feature.get("found")
                    and stem_feature.get("found")
                ),
                "voice_continuity":(
                    {
                        "inside_window":True,
                        "candidate_for_tracking":bool(
                            voice_row.get("candidate_for_tracking")
                        ),
                        "dominant_evidence_share":float(
                            voice_row.get("dominant_evidence_share",0)
                        ),
                        "register_separation":float(
                            voice_row.get("register_separation",0)
                        ),
                    } if voice_row is not None else {"inside_window":False}
                ),
            })
    reference_scope=None
    if reference is not None:
        from .datasets import _read_expert_gems

        snapshots=json.loads(
            (analysis/"guitar-stage-snapshots.json").read_text(encoding="utf-8")
        )
        stages={row["stage"]:row for row in snapshots.get("stages",[])}
        mapped=_snapshot_attacks(stages["after_map_pitched"]["notes"])
        playable=_snapshot_attacks(stages["after_enforce_playability"]["notes"])
        reference_attacks=_chart_attacks(
            _read_expert_gems(Path(reference),reference_format,"expert")
        )
        resolved_offset=(
            estimate_attack_offset(reference_attacks,mapped,tolerance)
            if offset is None else float(offset)
        )
        aligned_playable=[
            {**row,"time":round(float(row["time"])+resolved_offset,6)}
            for row in playable
        ]
        baseline_matches=_match_attacks(
            reference_attacks,aligned_playable,tolerance
        )
        covered={id(left) for left,_,_ in baseline_matches}
        uncovered=[
            row for row in reference_attacks if id(row) not in covered
        ]
        for row in rows:
            aligned=float(row["start"])+resolved_offset
            row["evaluation_only"]={
                "covers_unmatched_reference_attack":any(
                    abs(float(target["time"])-aligned)<=tolerance
                    for target in uncovered
                )
            }
        reference_scope={
            "reference":str(Path(reference).resolve()),
            "reference_format":reference_format,
            "alignment_offset_seconds":resolved_offset,
            "tolerance_seconds":tolerance,
            "baseline_matches":len(baseline_matches),
            "unmatched_reference_attacks":len(uncovered),
        }
    features={
        "confidence":lambda row:float(row["confidence"]),
        "model_onset_strength":lambda row:row["raw_evidence"]["model_onset_strength"],
        "attack_strength":lambda row:row["raw_evidence"]["attack_strength"],
        "evidence_score":lambda row:row["raw_evidence"]["evidence_score"],
        "structure_rank":lambda row:row["structure_evidence"].get(
            "periodic_width_attack_rank_percentile",0
        ),
        "stem_prominence_rank":lambda row:row["guitar_stem_onset"].get(
            "prominence_rank",0
        ),
        "stem_width_rank":lambda row:row["guitar_stem_onset"].get("width_rank",0),
        "voice_dominant_share":lambda row:row["voice_continuity"].get(
            "dominant_evidence_share",0
        ),
    }
    separability={}
    if reference_scope is not None:
        positives=sum(
            row["evaluation_only"]["covers_unmatched_reference_attack"]
            for row in rows
        )
        for name,getter in features.items():
            values=[getter(row) for row in rows]
            threshold=float(np.quantile(values,.75)) if values else 0.0
            selected=[
                row for row,value in zip(rows,values)
                if value>=threshold and value>0
            ]
            hits=sum(
                row["evaluation_only"]["covers_unmatched_reference_attack"]
                for row in selected
            )
            separability[name]={
                "top_quartile_threshold":round(threshold,4),
                "selected":len(selected),
                "uncovered_reference_hits":hits,
                "precision":round(hits/max(1,len(selected)),4),
                "positive_recall":round(hits/max(1,positives),4),
            }
    report={
        "schema":"rb3charts-reference-comparison",
        "schema_version":1,
        "scope":(
            "post-generation diagnostics; reference data, when supplied, "
            "labels outcomes only and is not generation input"
        ),
        "run_folder":str(folder),
        "source_mix_sha256":source_hash,
        "guitar_stem_sha256":stem_hash,
        "independent_stem_mix_available":source_hash!=stem_hash,
        "structure_predictions":(
            str(Path(structure_predictions).resolve())
            if structure_predictions is not None else None
        ),
        "reference_evaluation":reference_scope,
        "summary":{
            "removed_attacks":len(rows),
            "by_reason":dict(Counter(
                row["removal_reason"] for row in rows
            )),
            "structure_supported":sum(
                row["structure_evidence"]["found"] for row in rows
            ),
            "stem_mix_agreement":sum(
                row["independent_stem_mix_agreement"] for row in rows
            ),
            "voice_continuity_windows":sum(
                row["voice_continuity"]["inside_window"] for row in rows
            ),
            "uncovered_reference_hits":sum(
                row.get("evaluation_only",{}).get(
                    "covers_unmatched_reference_attack",False
                ) for row in rows
            ),
        },
        "feature_separability":separability,
        "events":rows,
    }
    output=Path(output) if output is not None else analysis/"playability-evidence.json"
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return report


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
    covered = agreed = source_covered = source_contour_agreed = 0
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
        if all(row.get("pitch") is not None for row in candidate):
            source_covered += 1
            source_pitches = tuple(float(row["pitch"]) for row in candidate)
            source_contour = tuple(
                0 if right == left else int(np.sign(right - left))
                for left, right in zip(source_pitches, source_pitches[1:])
            )
            reference_contour = tuple(
                0 if right == left else int(np.sign(right - left))
                for left, right in zip(reference_lanes, reference_lanes[1:])
            )
            source_contour_agreed += int(
                source_contour == reference_contour
            )
    return {
        "families": len(repeated),
        "reference_occurrences": len(occurrences),
        "covered_occurrences": covered,
        "coverage": round(covered / max(1, len(occurrences)), 4),
        "relative_lane_shape_agreement": round(agreed / max(1, covered), 4),
        "source_pitch_covered_occurrences": source_covered,
        "source_pitch_contour_agreement": round(
            source_contour_agreed / max(1, source_covered), 4
        ) if source_covered else None,
    }


def compare_chart_attacks(reference, candidate, tolerance=0.08, hold_seconds=0.35):
    matches = _match_attacks(reference, candidate, tolerance)
    matched_reference={id(left) for left,_,_ in matches}
    matched_candidate={id(right) for _,right,_ in matches}
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
        "unmatched": {
            "reference": [
                row for row in reference
                if id(row) not in matched_reference
            ],
            "candidate": [
                row for row in candidate
                if id(row) not in matched_candidate
            ],
        },
    }


def _resolve_midi(path):
    path=Path(path).resolve()
    if path.is_file():
        return path,None
    for relative in (Path("midi")/"notes.mid",Path("import")/"notes.mid"):
        candidate=path/relative
        if candidate.is_file():
            return candidate,path
    raise FileNotFoundError(f"No notes.mid found in: {path}")


def _snapshot_attacks(notes, offset=0.0):
    grouped=defaultdict(list)
    for note in notes:
        grouped[round(float(note["start"])+offset,6)].append(note)
    return [
        {
            "time":time,
            "lanes":tuple(sorted({
                int(lane) for note in rows for lane in note.get("lanes",[])
            })),
            "duration":max(
                float(note.get("end",note["start"]))-float(note["start"])
                for note in rows
            ),
            "pitch":(
                float(rows[0]["pitch"])
                if len(rows) == 1 and rows[0].get("pitch") is not None
                else None
            ),
        }
        for time,rows in sorted(grouped.items())
    ]


def _reference_stage_metrics(folder, reference, offset, tolerance):
    snapshots=folder/"analysis"/"guitar-stage-snapshots.json"
    if not snapshots.is_file():
        return []
    payload=json.loads(snapshots.read_text(encoding="utf-8"))
    stages=[]
    for stage in payload.get("stages",[]):
        notes=stage.get("notes")
        if not isinstance(notes,list):
            continue
        candidate=_snapshot_attacks(notes,offset)
        metrics=compare_chart_attacks(
            reference,candidate,tolerance=tolerance
        )
        stages.append({
            "stage":str(stage.get("stage","unknown")),
            "attack":metrics["attack"],
            "lane":metrics["lane"],
            "motifs":metrics["motifs"],
            "density":metrics["density"],
            "worst_reference_recall":
                metrics["attack_windows"]["worst_reference_recall"],
        })
    return stages


def _reference_comparison_markdown(report):
    provenance=report["provenance"]
    metrics=report["metrics"]
    attack=metrics["attack"]
    generated_name=Path(provenance["generated_midi"]).name
    reference_name=Path(provenance["reference_midi"]).name
    lines=[
        "# Reference chart comparison",
        "",
        (
            "Evaluation only: reference notes were loaded after generation "
            "and were not generation inputs."
        ),
        "",
        f"- Generated MIDI: `{generated_name}`",
        f"- Generated SHA-256: `{provenance['generated_sha256']}`",
        f"- Reference MIDI: `{reference_name}`",
        f"- Reference SHA-256: `{provenance['reference_sha256']}`",
        (
            f"- Difficulty: **{provenance['difficulty'].title()} Guitar**"
            if provenance.get("reference_difficulty",provenance["difficulty"])
            == provenance["difficulty"]
            else (
                f"- Generated difficulty: "
                f"**{provenance['difficulty'].title()} Guitar**\n"
                f"- Reference difficulty: "
                f"**{provenance['reference_difficulty'].title()} Guitar**"
            )
        ),
        (
            f"- Alignment: **{provenance['alignment']}**, "
            f"{provenance['alignment_offset_seconds']:+.3f} seconds"
        ),
        (
            "- Disclosure: summary metrics only; local paths and reference "
            "note rows are not embedded."
        ),
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Reference attacks | {attack['reference']} |",
        f"| Generated attacks | {attack['candidate']} |",
        f"| Matched attacks | {attack['matched']} |",
        f"| Attack precision | {attack['precision']:.4f} |",
        f"| Attack recall | {attack['recall']:.4f} |",
        f"| Attack F1 | {attack['f1']:.4f} |",
        f"| Exact lane agreement | {metrics['lane']['exact_agreement']:.4f} |",
        (
            "| Movement direction agreement | "
            f"{metrics['lane']['movement_direction_agreement']:.4f} |"
        ),
        "",
    ]
    if report["stage_metrics"]:
        lines.extend([
            "## Stage attribution",
            "",
            "| Stage | Attacks | Matched | Precision | Recall | F1 |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for stage in report["stage_metrics"]:
            row=stage["attack"]
            lines.append(
                f"| {stage['stage']} | {row['candidate']} | "
                f"{row['matched']} | {row['precision']:.4f} | "
                f"{row['recall']:.4f} | {row['f1']:.4f} |"
            )
        lines.append("")
    worst=sorted(
        metrics["attack_windows"]["windows"],
        key=lambda row:(row["f1"],row["start"]),
    )[:12]
    lines.extend([
        "## Weakest four-second windows",
        "",
        "| Seconds | Reference | Generated | Matched | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in worst:
        lines.append(
            f"| {row['start']:.0f}-{row['end']:.0f} | "
            f"{row['reference']} | {row['candidate']} | {row['matched']} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1']:.4f} |"
        )
    return "\n".join(lines)+"\n"


def _paired_source_evidence_markdown(evidence):
    status=str(evidence.get("status","invalid"))
    lines=[
        "## Paired-source evidence",
        "",
        f"- Lineage status: **{status}**",
    ]
    coverage=evidence.get("target_coverage") or {}
    paired=int(coverage.get("paired_items",0))
    total=int(coverage.get("total_items",0))
    if status=="unavailable":
        lines.extend([
            (
                "- Generated-song coverage: **unavailable** "
                f"({paired}/{total} explicitly paired corpus items)"
            ),
            (
                "- No paired-source evaluation was attached. This does not "
                "imply that the generated song has an independent paired capture."
            ),
        ])
        return "\n".join(lines)+"\n"
    if status=="invalid":
        lines.extend([
            "- Metrics omitted because the supplied lineage did not validate.",
            f"- Reason: {evidence.get('reason','unspecified lineage failure')}",
        ])
        return "\n".join(lines)+"\n"
    scope=evidence.get("scope")
    if scope=="external_rights_cleared_calibration":
        lines.append(
            "- **External rights-cleared calibration only; this is not "
            "generated-song evidence and does not establish a paired capture "
            "for this song.**"
        )
    else:
        lines.append("- Scope: **direct paired-source corpus coverage**")
    lines.extend([
        (
            f"- Generated-song corpus coverage: **{coverage.get('status','unavailable')}** "
            f"({paired}/{total} items)"
        ),
        (
            f"- Calibration dataset: **{evidence.get('source_dataset',{}).get('id','unknown')} "
            f"{evidence.get('source_dataset',{}).get('version','')}**"
        ),
        f"- Tolerance: **{evidence.get('configuration',{}).get('tolerance_ms')} ms**",
        f"- Evaluation report SHA-256: `{evidence.get('report_sha256','unavailable')}`",
        (
            "- Paired manifest SHA-256: "
            f"`{evidence.get('lineage',{}).get('manifest_sha256','unavailable')}`"
        ),
    ])
    release=(evidence.get("lineage") or {}).get("release")
    if release:
        lines.append(
            f"- Release archive published MD5: `{release.get('published_md5')}`"
        )
    lines.extend([
        "",
        "| Split | Shared P | Shared R | Shared F1 | Best source | Best F1 | ΔP | ΔR | ΔF1 |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ])
    for split,row in sorted((evidence.get("splits") or {}).items()):
        shared=row["shared"]
        best=row["best_source"]
        delta=row["delta_vs_best_source"]
        lines.append(
            f"| {split} | {shared['precision']:.4f} | {shared['recall']:.4f} | "
            f"{shared['f1']:.4f} | {best['channel']} | {best['f1']:.4f} | "
            f"{delta['precision']:+.4f} | {delta['recall']:+.4f} | "
            f"{delta['f1']:+.4f} |"
        )
    return "\n".join(lines)+"\n"


def _authorized_comparison_markdown(report):
    attack=report["difficulties"]["expert"]["attack"]
    lines=[
        "# Authorized human-chart comparison",
        "",
        (
            "Post-generation evaluation only: authorized human notes were "
            "not generation inputs."
        ),
        "",
        f"- Song ID: `{report['song_id']}`",
        f"- Chart-pair manifest SHA-256: `{report['manifest_sha256']}`",
        "",
        "| Metric | Expert |",
        "|---|---:|",
        f"| Reference attacks | {attack['reference']} |",
        f"| Generated attacks | {attack['candidate']} |",
        f"| Matched attacks | {attack['matched']} |",
        f"| Attack precision | {attack['precision']:.4f} |",
        f"| Attack recall | {attack['recall']:.4f} |",
        f"| Attack F1 | {attack['f1']:.4f} |",
        "",
        _paired_source_evidence_markdown(report["paired_source_agreement"]).rstrip(),
    ]
    return "\n".join(lines)+"\n"


def compare_reference_midi(
    generated,
    reference,
    output=None,
    difficulty="expert",
    tolerance=0.08,
    offset=None,
    reference_difficulty=None,
):
    """Compare a completed guitar chart with evaluation-only reference MIDI."""
    from .datasets import _read_expert_gems

    generated_midi,generated_folder=_resolve_midi(generated)
    reference_midi,_=_resolve_midi(reference)
    lead=0.0
    if generated_folder is not None:
        metadata_path=generated_folder/"metadata"/"song.json"
        if metadata_path.is_file():
            metadata=json.loads(metadata_path.read_text(encoding="utf-8"))
            lead=float(metadata.get("lead_seconds",0))
    reference_difficulty=reference_difficulty or difficulty
    reference_attacks=_chart_attacks(_read_expert_gems(
        reference_midi,"rock_band_midi",reference_difficulty
    ))
    candidate_attacks=_chart_attacks(
        _read_expert_gems(generated_midi,"rock_band_midi",difficulty),
        offset=-lead,
    )
    if offset is None:
        offset=estimate_attack_offset(
            reference_attacks,candidate_attacks,tolerance
        )
        alignment="automatic-onsets-only"
    else:
        offset=float(offset)
        alignment="manual"
    aligned=[
        {**row,"time":round(row["time"]+offset,6)}
        for row in candidate_attacks
    ]
    metrics=compare_chart_attacks(
        reference_attacks,aligned,tolerance=tolerance
    )
    report={
        "schema_version":1,
        "scope":(
            "post-generation evaluation only; reference notes were not "
            "generation inputs"
        ),
        "provenance":{
            "generated_midi":str(generated_midi),
            "generated_sha256":hash_file(generated_midi),
            "reference_midi":str(reference_midi),
            "reference_sha256":hash_file(reference_midi),
            "instrument":"guitar",
            "difficulty":difficulty,
            "generated_difficulty":difficulty,
            "reference_difficulty":reference_difficulty,
            "tolerance_seconds":float(tolerance),
            "generated_audio_lead_seconds":lead,
            "alignment":alignment,
            "alignment_offset_seconds":offset,
            "git":git_state(Path(__file__).resolve().parents[1]),
        },
        "metrics":metrics,
        "stage_metrics":(
            _reference_stage_metrics(
                generated_folder,
                reference_attacks,
                offset,
                tolerance,
            )
            if generated_folder is not None and difficulty=="expert" else []
        ),
    }
    output=Path(output) if output else (
        (generated_folder or generated_midi.parent)/
        "analysis"/"reference-comparison.json"
    )
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    output.with_suffix(".md").write_text(
        _reference_comparison_markdown(report),encoding="utf-8"
    )
    return report


def compare_authorized_chart(
    manifest,
    song_id,
    generated_folder,
    output=None,
    source_agreement_report=None,
):
    from .datasets import validate_chart_pair_manifest, _read_expert_gems

    manifest = Path(manifest).resolve()
    generated_folder = Path(generated_folder).resolve()
    validation = validate_chart_pair_manifest(
        manifest,
        verify_hashes=True,
        source_agreement_report=source_agreement_report,
    )
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
        "schema": "rb3charts-authorized-chart-comparison",
        "schema_version": 1,
        "scope": (
            "post-generation evaluation only; authorized human notes were not "
            "generation inputs"
        ),
        "song_id": str(song_id),
        "manifest_sha256": validation["manifest_sha256"],
        "generated_folder": str(generated_folder),
        "generated_identity": {
            "song_id": str(metadata.get("song_id","")),
            "title": str(metadata.get("title","")),
            "artist": str(metadata.get("artist","")),
        },
        "generated_midi_sha256": hash_file(generated_midi),
        "generated_audio_lead_seconds": lead,
        "difficulties": difficulties,
        "stage_metrics": _reference_stage_metrics(
            generated_folder,
            _chart_attacks(_read_expert_gems(
                reference_path,
                item["chart_format"],
                "expert",
            )),
            -lead,
            .08,
        ),
        "paired_source_agreement": validation["paired_source_agreement"],
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
    output.with_suffix(".md").write_text(
        _authorized_comparison_markdown(report),
        encoding="utf-8",
    )
    return report


def _comparison_index_summary(report):
    schema=report["schema"]
    if schema=="rb3charts-authorized-chart-comparison":
        metrics=report["difficulties"]["expert"]
        paired=(report.get("paired_source_agreement") or {}).get(
            "status","unavailable"
        )
    elif schema=="rb3charts-reference-comparison":
        metrics=report["metrics"]
        paired="unavailable"
    else:
        raise ValueError(f"Unsupported comparison schema: {schema}")
    worst=metrics["attack_windows"].get("worst_reference_recall")
    if worst is not None:
        worst={
            key:worst[key]
            for key in (
                "start","end","reference","candidate","matched",
                "precision","recall","f1",
            )
        }
    return {
        "attack":metrics["attack"],
        "chord":metrics["chord"],
        "sustain":metrics["sustain"],
        "lane":{
            key:metrics["lane"].get(key)
            for key in (
                "exact_agreement",
                "movement_direction_agreement",
                "mean_interval_error",
            )
        },
        "motif":{
            key:metrics["motifs"].get(key)
            for key in (
                "families",
                "reference_occurrences",
                "covered_occurrences",
                "coverage",
                "relative_lane_shape_agreement",
            )
        },
        "weakest_window":worst,
        "paired_source_status":paired,
    }


def _markdown_cell(value):
    text=" ".join(str(value).split())
    for source,replacement in (
        ("\\",r"\\"),
        ("|",r"\|"),
        ("[",r"\["),
        ("]",r"\]"),
        ("`",r"\`"),
        ("<","&lt;"),
        (">","&gt;"),
    ):
        text=text.replace(source,replacement)
    return text


def _comparison_index_markdown(index):

    lines=[
        "# Comparison corpus index",
        "",
        (
            "Local summary links only. Reference-note rows, source paths and "
            "detailed authorized records are not embedded."
        ),
        "",
        f"- Comparison schema: **{index['comparison_schema']}**",
        f"- Generated runs: **{len(index['entries'])}**",
        "",
        (
            "| Song | Run | Comparison | Validation | Attack F1 | Chord F1 | "
            "Sustain F1 | Lane exact | Motif coverage | Weakest recall | "
            "Paired source | Report |"
        ),
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for entry in index["entries"]:
        summary=entry.get("summary")
        if summary is None:
            values=["—"]*6
            paired="unavailable"
        else:
            weakest=summary["weakest_window"]
            def display(value):
                return f"{value:.4f}" if value is not None else "—"
            values=[
                display(summary["attack"]["f1"]),
                display(summary["chord"]["f1"]),
                display(summary["sustain"]["f1"]),
                display(summary["lane"]["exact_agreement"]),
                display(summary["motif"]["coverage"]),
                display(weakest["recall"]) if weakest else "—",
            ]
            paired=summary["paired_source_status"]
        link=entry.get("links",{}).get("markdown")
        if link is None:
            link=entry.get("links",{}).get("json")
        report_link=f"[local]({quote(link,safe='/..')})" if link else "—"
        identity=entry["identity"]
        song=_markdown_cell(f"{identity['artist']} — {identity['title']}")
        run=_markdown_cell(entry["run_basename"]).replace("`","'")
        lines.append(
            f"| {song} | `{run}` | "
            f"{entry['comparison_status']} | {entry['validation_status']} | "
            f"{' | '.join(values)} | {paired} | {report_link} |"
        )
    return "\n".join(lines)+"\n"


def build_comparison_index(root, output=None):
    """Build a sanitized deterministic index of local post-generation reports."""
    root=Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Comparison root does not exist: {root}")
    output=Path(output).resolve() if output else root/"comparison-index.json"
    if output.parent!=root:
        raise ValueError(
            "Comparison index output must be directly inside the corpus root"
        )
    metadata_paths=sorted(root.glob("**/metadata/song.json"))
    if not metadata_paths:
        raise ValueError("No generated runs with metadata/song.json were found")
    entries=[]
    identities={}
    schemas=set()
    for metadata_path in metadata_paths:
        run=metadata_path.parent.parent
        try:
            metadata=json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError,json.JSONDecodeError) as exc:
            raise ValueError(f"Malformed generated metadata in {run.name}: {exc}") from exc
        for key in ("song_id","title","artist"):
            if metadata.get(key) in (None,""):
                raise ValueError(f"Generated metadata in {run.name} requires {key}")
        song_id=str(metadata["song_id"])
        reports=[
            path for path in (
                run/"analysis"/"human-chart-comparison.json",
                run/"analysis"/"reference-comparison.json",
            )
            if path.is_file()
        ]
        if len(reports)>1:
            raise ValueError(f"Mixed comparison schemas in generated run {run.name}")
        validation_path=run/"validation_report.json"
        if validation_path.is_file():
            try:
                validation=json.loads(
                    validation_path.read_text(encoding="utf-8")
                )
            except (OSError,json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Malformed validation report in {run.name}: {exc}"
                ) from exc
            if not isinstance(validation.get("passed"),bool):
                raise ValueError(
                    f"Validation report in {run.name} has no boolean passed status"
                )
            validation_status="passed" if validation["passed"] else "failed"
        else:
            validation_status="unavailable"
        entry={
            "identity":{
                "song_id":song_id,
                "title":str(metadata["title"]),
                "artist":str(metadata["artist"]),
            },
            "run_basename":run.name,
            "comparison_status":"missing",
            "validation_status":validation_status,
            "links":{},
            "summary":None,
        }
        if reports:
            report_path=reports[0]
            try:
                report=json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError,json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Malformed comparison report in {run.name}: {exc}"
                ) from exc
            schema=report.get("schema")
            if schema not in {
                "rb3charts-authorized-chart-comparison",
                "rb3charts-reference-comparison",
            } or report.get("schema_version")!=1:
                raise ValueError(
                    f"Malformed comparison schema in generated run {run.name}"
                )
            schemas.add(schema)
            generated_candidates=[
                run/"midi"/"notes.mid",
                run/"import"/"notes.mid",
            ]
            generated_midi=next(
                (path for path in generated_candidates if path.is_file()),
                None,
            )
            if generated_midi is None:
                raise ValueError(
                    f"Comparison report in {run.name} has no current generated MIDI"
                )
            if schema=="rb3charts-authorized-chart-comparison":
                generated_identity=report.get("generated_identity") or {}
                if (
                    str(generated_identity.get("song_id"))!=song_id
                    or str(generated_identity.get("title"))!=str(metadata["title"])
                    or str(generated_identity.get("artist"))!=str(metadata["artist"])
                ):
                    raise ValueError(
                        f"Comparison generated identity differs in {run.name}"
                    )
                entry["identity"]["song_id"]=str(report.get("song_id",""))
                if not entry["identity"]["song_id"]:
                    raise ValueError(
                        f"Comparison song identity is missing in {run.name}"
                    )
                expected_generated_hash=report.get("generated_midi_sha256")
            else:
                expected_generated_hash=(
                    report.get("provenance") or {}
                ).get("generated_sha256")
            if (
                not isinstance(expected_generated_hash,str)
                or len(expected_generated_hash)!=64
                or hash_file(generated_midi)!=expected_generated_hash
            ):
                raise ValueError(
                    f"Comparison generated-MIDI lineage differs in {run.name}"
                )
            relative_json=Path(os.path.relpath(
                report_path,output.parent
            )).as_posix()
            links={"json":relative_json}
            markdown_path=report_path.with_suffix(".md")
            if markdown_path.is_file():
                links["markdown"]=Path(os.path.relpath(
                    markdown_path,output.parent
                )).as_posix()
            entry.update({
                "comparison_status":"available",
                "links":links,
                "lineage":{
                    "comparison_report_sha256":hash_file(report_path),
                    "generated_midi_sha256":expected_generated_hash,
                },
                "summary":_comparison_index_summary(report),
            })
        identity_song_id=entry["identity"]["song_id"]
        if identity_song_id in identities:
            raise ValueError(
                f"Duplicate song identity {identity_song_id}: "
                f"{identities[identity_song_id]} and {run.name}"
            )
        identities[identity_song_id]=run.name
        entries.append(entry)
    if not schemas:
        raise ValueError("No comparison reports were found")
    if len(schemas)>1:
        raise ValueError("Corpus mixes authorized and reference comparison schemas")
    entries.sort(key=lambda row:(
        row["identity"]["artist"].casefold(),
        row["identity"]["title"].casefold(),
        row["identity"]["song_id"],
        row["run_basename"],
    ))
    index={
        "schema":"rb3charts-comparison-corpus-index",
        "schema_version":1,
        "scope":(
            "sanitized local post-generation summary; no reference notes, "
            "source paths or detailed authorized records"
        ),
        "comparison_schema":next(iter(schemas)),
        "metric_semantics":"rb3charts-chart-comparison-v1",
        "entries":entries,
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(index,indent=2)+"\n",encoding="utf-8")
    output.with_suffix(".md").write_text(
        _comparison_index_markdown(index),encoding="utf-8"
    )
    return index


_TREND_METRICS={
    "attack_f1":("attack","f1"),
    "chord_f1":("chord","f1"),
    "sustain_f1":("sustain","f1"),
    "lane_exact_agreement":("lane","exact_agreement"),
    "motif_coverage":("motif","coverage"),
    "motif_shape_agreement":("motif","relative_lane_shape_agreement"),
    "weakest_window_recall":("weakest_window","recall"),
}


def _load_comparison_index(path):
    path=Path(path).resolve()
    try:
        payload=json.loads(path.read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read comparison index: {exc}") from exc
    if (
        payload.get("schema")!="rb3charts-comparison-corpus-index"
        or payload.get("schema_version")!=1
    ):
        raise ValueError("Incompatible comparison index schema")
    if payload.get("metric_semantics")!="rb3charts-chart-comparison-v1":
        raise ValueError("Incompatible comparison metric semantics")
    entries=payload.get("entries")
    if not isinstance(entries,list):
        raise ValueError("Comparison index entries must be an array")
    by_id={}
    display_identities={}
    for entry in entries:
        identity=entry.get("identity") or {}
        song_id=str(identity.get("song_id",""))
        title=str(identity.get("title",""))
        artist=str(identity.get("artist",""))
        if not song_id or not title or not artist:
            raise ValueError("Comparison index contains malformed song identity")
        if song_id in by_id:
            raise ValueError(f"Duplicate song identity {song_id}")
        display_key=(artist.casefold(),title.casefold())
        previous=display_identities.setdefault(display_key,song_id)
        if previous!=song_id:
            raise ValueError(
                f"Duplicate display identity {artist} — {title}"
            )
        status=entry.get("comparison_status")
        if status not in {"available","missing"}:
            raise ValueError(f"Invalid comparison status for {song_id}")
        if entry.get("validation_status") not in {
            "passed","failed","unavailable",
        }:
            raise ValueError(f"Invalid validation status for {song_id}")
        if status=="available":
            links=entry.get("links") or {}
            relative_text=links.get("json")
            if not isinstance(relative_text,str) or not relative_text:
                raise ValueError(f"Invalid comparison link for {song_id}")
            relative=Path(relative_text)
            if relative.is_absolute():
                raise ValueError(f"Invalid comparison link for {song_id}")
            report_path=(path.parent/relative).resolve()
            if not report_path.is_relative_to(path.parent):
                raise ValueError(
                    f"Comparison link escapes index root for {song_id}"
                )
            lineage=entry.get("lineage") or {}
            report_hash=lineage.get("comparison_report_sha256")
            generated_hash=lineage.get("generated_midi_sha256")
            if (
                not isinstance(report_hash,str)
                or len(report_hash)!=64
                or hash_file(report_path)!=report_hash
            ):
                raise ValueError(
                    f"Stale comparison-report lineage for {song_id}"
                )
            try:
                linked_report=json.loads(
                    report_path.read_text(encoding="utf-8")
                )
            except (OSError,json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Malformed linked comparison report for {song_id}: {exc}"
                ) from exc
            if linked_report.get("schema")!=payload.get("comparison_schema"):
                raise ValueError(
                    f"Linked comparison schema differs for {song_id}"
                )
            if _comparison_index_summary(linked_report)!=entry.get("summary"):
                raise ValueError(
                    f"Comparison index summary differs from report for {song_id}"
                )
            run=report_path.parent.parent
            generated_midi=next((
                candidate for candidate in (
                    run/"midi"/"notes.mid",
                    run/"import"/"notes.mid",
                )
                if candidate.is_file()
            ),None)
            if (
                generated_midi is None
                or not isinstance(generated_hash,str)
                or len(generated_hash)!=64
                or hash_file(generated_midi)!=generated_hash
            ):
                raise ValueError(
                    f"Stale generated-MIDI lineage for {song_id}"
                )
            if not isinstance(entry.get("summary"),dict):
                raise ValueError(f"Available comparison has no summary for {song_id}")
        elif entry.get("summary") is not None:
            raise ValueError(f"Missing comparison has a summary for {song_id}")
        by_id[song_id]=entry
    return path,payload,by_id


def _trend_value(entry,metric):
    if entry.get("comparison_status")!="available":
        return None
    value=entry["summary"]
    for key in _TREND_METRICS[metric]:
        if value is None:
            return None
        if not isinstance(value,dict):
            raise ValueError(
                f"Comparison metric {metric} is malformed for "
                f"{entry['identity']['song_id']}"
            )
        value=value.get(key)
    if value is None:
        return None
    if not isinstance(value,(int,float)) or isinstance(value,bool):
        raise ValueError(
            f"Comparison metric {metric} is malformed for "
            f"{entry['identity']['song_id']}"
        )
    if not np.isfinite(value):
        raise ValueError(
            f"Comparison metric {metric} is not finite for "
            f"{entry['identity']['song_id']}"
        )
    if not 0<=value<=1:
        raise ValueError(
            f"Comparison metric {metric} is outside 0..1 for "
            f"{entry['identity']['song_id']}"
        )
    return float(value)


def _comparison_trend_markdown(report):
    coverage=report["coverage"]
    lines=[
        "# Comparison corpus trend",
        "",
        (
            "Sanitized paired aggregate only. Paths, events, reference notes "
            "and detailed authorized records are not embedded."
        ),
        "",
        f"- Shared song identities: **{coverage['shared_identities']}**",
        f"- Paired available comparisons: **{coverage['paired_available']}**",
        f"- Baseline-only identities: **{coverage['baseline_only']}**",
        f"- Candidate-only identities: **{coverage['candidate_only']}**",
        f"- Shared identities with missing comparison: **{coverage['shared_missing']}**",
        "",
        "| Metric | Paired values | Baseline | Candidate | Delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric,row in report["macro"].items():
        if row["paired_values"]:
            lines.append(
                f"| {metric} | {row['paired_values']} | "
                f"{row['baseline_mean']:.4f} | {row['candidate_mean']:.4f} | "
                f"{row['delta']:+.4f} |"
            )
        else:
            lines.append(f"| {metric} | 0 | — | — | — |")
    lines.extend([
        "",
        "## Largest paired attack-F1 changes",
        "",
        "| Direction | Song | Delta |",
        "|---|---|---:|",
    ])
    for direction in ("regressions","improvements"):
        for row in report["rankings"]["attack_f1"][direction][:5]:
            identity=row["identity"]
            song=_markdown_cell(
                f"{identity['artist']} — {identity['title']}"
            )
            lines.append(
                f"| {direction[:-1]} | {song} | {row['delta']:+.4f} |"
            )
    return "\n".join(lines)+"\n"


def compare_comparison_indexes(baseline, candidate, output=None):
    """Compare compatible sanitized indexes without treating asymmetry as paired."""
    baseline_path,baseline_index,baseline_entries=_load_comparison_index(baseline)
    candidate_path,candidate_index,candidate_entries=_load_comparison_index(candidate)
    if baseline_index["comparison_schema"]!=candidate_index["comparison_schema"]:
        raise ValueError("Comparison indexes use incompatible comparison schemas")
    if baseline_index["metric_semantics"]!=candidate_index["metric_semantics"]:
        raise ValueError("Comparison indexes use incompatible metric semantics")
    baseline_ids=set(baseline_entries)
    candidate_ids=set(candidate_entries)
    shared=sorted(baseline_ids&candidate_ids)
    for song_id in shared:
        if baseline_entries[song_id]["identity"]!=candidate_entries[song_id]["identity"]:
            raise ValueError(f"Song identity changed for {song_id}")
    paired=[
        song_id for song_id in shared
        if (
            baseline_entries[song_id]["comparison_status"]=="available"
            and candidate_entries[song_id]["comparison_status"]=="available"
        )
    ]
    songs=[]
    macro={}
    rankings={}
    for metric in _TREND_METRICS:
        rows=[]
        for song_id in paired:
            left=_trend_value(baseline_entries[song_id],metric)
            right=_trend_value(candidate_entries[song_id],metric)
            status="paired" if left is not None and right is not None else "unavailable"
            row={
                "identity":baseline_entries[song_id]["identity"],
                "status":status,
                "baseline":left,
                "candidate":right,
                "delta":round(right-left,4) if status=="paired" else None,
            }
            rows.append(row)
        valid=[row for row in rows if row["status"]=="paired"]
        if valid:
            baseline_mean=float(np.mean([row["baseline"] for row in valid]))
            candidate_mean=float(np.mean([row["candidate"] for row in valid]))
            macro[metric]={
                "paired_values":len(valid),
                "unavailable_values":len(rows)-len(valid),
                "baseline_mean":round(baseline_mean,4),
                "candidate_mean":round(candidate_mean,4),
                "delta":round(candidate_mean-baseline_mean,4),
            }
        else:
            macro[metric]={
                "paired_values":0,
                "unavailable_values":len(rows),
                "baseline_mean":None,
                "candidate_mean":None,
                "delta":None,
            }
        rankings[metric]={
            "regressions":sorted(
                (row for row in valid if row["delta"]<0),
                key=lambda row:(row["delta"],row["identity"]["song_id"]),
            ),
            "improvements":sorted(
                (row for row in valid if row["delta"]>0),
                key=lambda row:(-row["delta"],row["identity"]["song_id"]),
            ),
        }
    for song_id in shared:
        left=baseline_entries[song_id]
        right=candidate_entries[song_id]
        metrics={}
        for metric in _TREND_METRICS:
            baseline_value=_trend_value(left,metric)
            candidate_value=_trend_value(right,metric)
            available=baseline_value is not None and candidate_value is not None
            metrics[metric]={
                "status":"paired" if available else "unavailable",
                "baseline":baseline_value,
                "candidate":candidate_value,
                "delta":(
                    round(candidate_value-baseline_value,4)
                    if available else None
                ),
            }
        songs.append({
            "identity":left["identity"],
            "baseline_status":left["comparison_status"],
            "candidate_status":right["comparison_status"],
            "baseline_validation":left["validation_status"],
            "candidate_validation":right["validation_status"],
            "metrics":metrics,
        })
    report={
        "schema":"rb3charts-comparison-corpus-trend",
        "schema_version":1,
        "comparison_schema":baseline_index["comparison_schema"],
        "metric_semantics":baseline_index["metric_semantics"],
        "lineage":{
            "baseline_index_sha256":hash_file(baseline_path),
            "candidate_index_sha256":hash_file(candidate_path),
        },
        "coverage":{
            "baseline_identities":len(baseline_ids),
            "candidate_identities":len(candidate_ids),
            "shared_identities":len(shared),
            "paired_available":len(paired),
            "shared_missing":len(shared)-len(paired),
            "baseline_only":len(baseline_ids-candidate_ids),
            "candidate_only":len(candidate_ids-baseline_ids),
            "baseline_only_identities":sorted(baseline_ids-candidate_ids),
            "candidate_only_identities":sorted(candidate_ids-baseline_ids),
        },
        "macro":macro,
        "songs":songs,
        "rankings":rankings,
    }
    output=Path(output).resolve() if output else (
        candidate_path.parent/"comparison-trend.json"
    )
    if output in {baseline_path,candidate_path}:
        raise ValueError("Trend output must not overwrite an input index")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    output.with_suffix(".md").write_text(
        _comparison_trend_markdown(report),encoding="utf-8"
    )
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
