"""Evaluate phrase-level source-pitch ownership without changing charts."""

import argparse
import json
from pathlib import Path

import numpy as np

from event_selection_pilot import _candidate_groups, _hash_file
from rb3charts.datasets import _read_expert_gems
from rb3charts.model import save_json


def _reference_attacks(item, root):
    grouped = {}
    for note in _read_expert_gems(
        root / item["annotations"],
        item["chart_format"],
        "expert",
    ):
        grouped.setdefault(round(float(note["start"]), 6), []).append(
            int(note["lane"])
        )
    return [
        {
            "time": time,
            "lanes": sorted(set(lanes)),
        }
        for time, lanes in sorted(grouped.items())
    ]


def _states(group):
    by_pitch = {}
    for event in group["events"]:
        pitch = int(round(float(event["pitch"])))
        evidence = event.get("evidence") or {}
        score = float(evidence.get("evidence_score", event.get("confidence", 0)))
        current = by_pitch.get(pitch)
        if current is None or score > current["score"]:
            by_pitch[pitch] = {
                "pitch": pitch,
                "score": score,
                "accepted": bool(event.get("accepted")),
                "expert": bool(event.get("expert_notes")),
            }
    return sorted(by_pitch.values(), key=lambda row: row["pitch"])


def _baseline_pitch(group):
    states = _states(group)
    return max(
        states,
        key=lambda row: (
            row["expert"],
            row["accepted"],
            row["score"],
            -row["pitch"],
        ),
    )["pitch"]


def _select_path(groups, evidence_weight, jump_weight, octave_penalty):
    selected = []
    phrase_start = 0
    for index in range(1, len(groups) + 1):
        phrase_end = (
            index == len(groups)
            or groups[index]["time"] - groups[index - 1]["time"] >= 1.0
        )
        if not phrase_end:
            continue
        phrase = groups[phrase_start:index]
        state_rows = [_states(group) for group in phrase]
        scores = [
            np.asarray(
                [evidence_weight * state["score"] for state in states],
                dtype=float,
            )
            for states in state_rows
        ]
        back = []
        for position in range(1, len(phrase)):
            previous = state_rows[position - 1]
            current = state_rows[position]
            next_scores = np.full(len(current), -np.inf)
            pointers = np.zeros(len(current), dtype=int)
            for right_index, right in enumerate(current):
                options = []
                for left_index, left in enumerate(previous):
                    jump = abs(right["pitch"] - left["pitch"])
                    transition = (
                        jump_weight * min(jump, 24)
                        + octave_penalty * (jump >= 12)
                    )
                    options.append(scores[position - 1][left_index] - transition)
                best = int(np.argmax(options))
                next_scores[right_index] = (
                    scores[position][right_index] + options[best]
                )
                pointers[right_index] = best
            scores[position] = next_scores
            back.append(pointers)
        state_index = int(np.argmax(scores[-1]))
        path = [state_index]
        for pointers in reversed(back):
            state_index = int(pointers[state_index])
            path.append(state_index)
        path.reverse()
        selected.extend(
            states[state_index]["pitch"]
            for states, state_index in zip(state_rows, path)
        )
        phrase_start = index
    return selected


def _match_groups(groups, reference, tolerance=0.08):
    available = set(range(len(reference)))
    matches = []
    for group_index, group in enumerate(groups):
        candidates = [
            index
            for index in available
            if abs(reference[index]["time"] - group["time"]) <= tolerance
        ]
        if not candidates:
            continue
        best = min(
            candidates,
            key=lambda index: abs(reference[index]["time"] - group["time"]),
        )
        matches.append((group_index, best))
        available.remove(best)
    return matches


def _metrics(records, selector):
    songs = []
    for record in records:
        pitches = selector(record["groups"])
        matched = _match_groups(record["groups"], record["reference"])
        transitions = agreed = 0
        for (left_group, left_reference), (
            right_group,
            right_reference,
        ) in zip(matched, matched[1:]):
            left = record["reference"][left_reference]
            right = record["reference"][right_reference]
            if (
                len(left["lanes"]) != 1
                or len(right["lanes"]) != 1
                or right["time"] - left["time"] > 1.0
            ):
                continue
            lane_delta = right["lanes"][0] - left["lanes"][0]
            pitch_delta = pitches[right_group] - pitches[left_group]
            if lane_delta == 0:
                continue
            transitions += 1
            agreed += int(np.sign(lane_delta) == np.sign(pitch_delta))
        songs.append({
            "song_id": record["song_id"],
            "transitions": transitions,
            "direction_matches": agreed,
            "direction_agreement": agreed / max(1, transitions),
        })
    return {
        "macro_direction_agreement": float(np.mean([
            row["direction_agreement"] for row in songs
        ])),
        "total_direction_agreement": (
            sum(row["direction_matches"] for row in songs)
            / max(1, sum(row["transitions"] for row in songs))
        ),
        "songs": songs,
    }


def _load_records(index_path, manifest_path):
    index = json.loads(index_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = {
        str(item["song_id"]): item
        for item in manifest["items"]
    }
    records = []
    for entry in index["entries"]:
        song_id = str(entry["identity"]["song_id"])
        item = items.get(song_id)
        if item is None or entry.get("comparison_status") != "available":
            continue
        report = json.loads(
            (index_path.parent / entry["links"]["json"]).read_text(
                encoding="utf-8"
            )
        )
        trace = (
            Path(report["generated_folder"])
            / "analysis"
            / "guitar-event-trace.json"
        )
        if not trace.is_file():
            continue
        records.append({
            "song_id": song_id,
            "split": item["split"],
            "groups": _candidate_groups(trace),
            "reference": _reference_attacks(item, manifest_path.parent),
            "trace_sha256": _hash_file(trace),
        })
    return records


def run(index_path, manifest_path, output, evaluate_test=False):
    records = _load_records(index_path, manifest_path)
    validation = [
        record for record in records if record["split"] == "validation"
    ]
    test = [record for record in records if record["split"] == "test"]
    baseline = _metrics(
        validation,
        lambda groups: [_baseline_pitch(group) for group in groups],
    )
    candidates = []
    for evidence_weight in (0.25, 0.5, 1.0, 2.0):
        for jump_weight in (0.01, 0.025, 0.05, 0.1):
            for octave_penalty in (0.0, 0.25, 0.5, 1.0):
                metrics = _metrics(
                    validation,
                    lambda groups, e=evidence_weight, j=jump_weight, o=octave_penalty:
                    _select_path(groups, e, j, o),
                )
                candidates.append({
                    "configuration": {
                        "evidence_weight": evidence_weight,
                        "jump_weight": jump_weight,
                        "octave_penalty": octave_penalty,
                        "rest_reset_seconds": 1.0,
                    },
                    **metrics,
                })
    selected = max(
        candidates,
        key=lambda row: (
            row["macro_direction_agreement"],
            row["total_direction_agreement"],
            -row["configuration"]["jump_weight"],
        ),
    )
    configuration = selected["configuration"]
    report = {
        "schema": "rb3charts-phrase-voice-ownership-pilot",
        "schema_version": 1,
        "scope": (
            "offline authorized evaluation; lane directions are evaluation "
            "targets and never inference inputs"
        ),
        "validation": {
            "baseline_current_source_proxy": baseline,
            "candidate": selected,
        },
        "test_evaluated": bool(evaluate_test and test),
        "input_hashes": {
            "comparison_index_sha256": _hash_file(index_path),
            "manifest_sha256": _hash_file(manifest_path),
        },
    }
    if evaluate_test and test:
        report["test"] = {
            "baseline_current_source_proxy": _metrics(
                test,
                lambda groups: [
                    _baseline_pitch(group) for group in groups
                ],
            ),
            "candidate": _metrics(
                test,
                lambda groups: _select_path(
                    groups,
                    configuration["evidence_weight"],
                    configuration["jump_weight"],
                    configuration["octave_penalty"],
                ),
            ),
        }
    save_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("comparison_index", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    report = run(
        args.comparison_index.resolve(),
        args.manifest.resolve(),
        args.output.resolve(),
        evaluate_test=args.evaluate_test,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
