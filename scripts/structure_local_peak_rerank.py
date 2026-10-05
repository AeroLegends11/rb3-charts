"""Rank precisely timed local attack peaks with phrase-context evidence."""

import argparse
import json
from pathlib import Path

import numpy as np

from structure_score_ensemble import _summarize
from structure_timing_snap import _hash_file, _load_model
from rb3charts.model import save_json
from rb3charts.training import (
    FEATURE_HOP_LENGTH,
    FEATURE_SAMPLE_RATE,
    _attack_peaks,
    _load_feature_artifact,
    _load_feature_index,
)


def _candidate_scores(local_scores, context_scores, local_weight, radius):
    scores = np.zeros_like(local_scores)
    for peak in _attack_peaks(local_scores):
        left = max(0, int(peak) - radius)
        right = min(len(context_scores), int(peak) + radius + 1)
        context = float(np.max(context_scores[left:right]))
        scores[peak] = (
            local_weight * float(local_scores[peak])
            + (1 - local_weight) * context
        )
    return scores


def _reranked_records(records, local_weight, radius):
    return [
        {
            **record,
            "local_scores": _candidate_scores(
                record["local_scores"],
                record["context_scores"],
                local_weight,
                radius,
            ),
            "context_scores": np.zeros_like(record["context_scores"]),
        }
        for record in records
    ]


def _timing_sort_value(row):
    timing = row["macro_median_timing_error_ms"]
    return float(timing) if timing is not None else float("inf")


def _select_validation_candidate(candidates, context_baseline, local_baseline):
    baseline_timings = [
        timing
        for timing in (
            context_baseline["macro_median_timing_error_ms"],
            local_baseline["macro_median_timing_error_ms"],
        )
        if timing is not None
    ]
    timing_limit = min(baseline_timings) if baseline_timings else None
    guarded = [
        row for row in candidates
        if (
            row["macro_f1"] >= context_baseline["macro_f1"]
            and row["macro_recall"] >= context_baseline["macro_recall"]
            and timing_limit is not None
            and row["macro_median_timing_error_ms"] is not None
            and row["macro_median_timing_error_ms"] <= timing_limit
        )
    ]
    selected = max(
        guarded or candidates,
        key=lambda row: (
            row["macro_f1"],
            row["macro_precision"],
            -_timing_sort_value(row),
        ),
    )
    return selected, bool(guarded)


def _test_evaluation(
    test,
    selected,
    context_threshold,
    local_threshold,
    tolerance_frames,
    guard_passed,
):
    if not guard_passed:
        return {}
    test_reranked = _reranked_records(
        test,
        selected["candidate_local_weight"],
        selected["context_radius_frames"],
    )
    return {
        "test_context_baseline": _summarize(
            test,
            0.0,
            context_threshold,
            tolerance_frames,
        ),
        "test_local_baseline": _summarize(
            test,
            1.0,
            local_threshold,
            tolerance_frames,
        ),
        "test_selected": _summarize(
            test_reranked,
            1.0,
            selected["threshold"],
            tolerance_frames,
        ),
    }


def run(index_path, context_path, local_path, output, device="auto"):
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    context_model, context_configuration = _load_model(
        context_path,
        device,
    )
    local_model, local_configuration = _load_model(local_path, device)
    _, items = _load_feature_index(index_path)
    records = []
    with torch.no_grad():
        for item in items:
            if item["split"] not in ("validation", "test"):
                continue
            arrays, _ = _load_feature_artifact(item["path"])
            features = np.concatenate(
                (
                    arrays["log_mel"],
                    arrays["mel_delta"],
                    arrays["beat_phase"],
                ),
                axis=1,
            )
            tensor = torch.from_numpy(features).unsqueeze(0).to(device)
            records.append({
                "song_id": item["song_id"],
                "split": item["split"],
                "context_scores": torch.sigmoid(
                    context_model(tensor)["attack"][0]
                ).cpu().numpy(),
                "local_scores": torch.sigmoid(
                    local_model(tensor)["attack"][0]
                ).cpu().numpy(),
                "actual": np.flatnonzero(arrays["attack_target"] > 0),
            })
    tolerance_frames = max(
        1,
        round(0.05 * FEATURE_SAMPLE_RATE / FEATURE_HOP_LENGTH),
    )
    validation = [
        record for record in records if record["split"] == "validation"
    ]
    context_baseline = _summarize(
        validation,
        0.0,
        float(context_configuration["attack_threshold"]),
        tolerance_frames,
    )
    local_baseline = _summarize(
        validation,
        1.0,
        float(local_configuration["attack_threshold"]),
        tolerance_frames,
    )
    candidates = []
    for radius in range(5):
        for local_weight in np.linspace(0.0, 1.0, 11):
            reranked = _reranked_records(
                validation,
                float(local_weight),
                radius,
            )
            for threshold in np.linspace(0.05, 0.95, 37):
                row = _summarize(
                    reranked,
                    1.0,
                    threshold,
                    tolerance_frames,
                )
                row["context_radius_frames"] = radius
                row["candidate_local_weight"] = float(local_weight)
                candidates.append(row)
    selected, guard_passed = _select_validation_candidate(
        candidates,
        context_baseline,
        local_baseline,
    )
    test = [record for record in records if record["split"] == "test"]
    report = {
        "schema": "rb3charts-structure-local-peak-rerank",
        "schema_version": 1,
        "scope": "offline authorized-data evaluation",
        "selection_rule": (
            "preserve context validation macro F1 and recall, meet the better "
            "baseline timing, then maximize F1"
        ),
        "guard_passed": guard_passed,
        "context_model_sha256": _hash_file(context_path),
        "local_model_sha256": _hash_file(local_path),
        "feature_index_sha256": _hash_file(index_path),
        "validation_context_baseline": context_baseline,
        "validation_local_baseline": local_baseline,
        "validation_selected": selected,
        "test_evaluated": guard_passed,
    }
    report.update(_test_evaluation(
        test,
        selected,
        float(context_configuration["attack_threshold"]),
        float(local_configuration["attack_threshold"]),
        tolerance_frames,
        guard_passed,
    ))
    save_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path)
    parser.add_argument("context_model", type=Path)
    parser.add_argument("local_model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    report = run(
        args.index.resolve(),
        args.context_model.resolve(),
        args.local_model.resolve(),
        args.output.resolve(),
        args.device,
    )
    keys = [
        "validation_context_baseline",
        "validation_local_baseline",
        "validation_selected",
    ]
    if report["test_evaluated"]:
        keys.extend([
            "test_context_baseline",
            "test_local_baseline",
            "test_selected",
        ])
    print(json.dumps({
        "guard_passed": report["guard_passed"],
        **{
            key: {
                "local_weight": report[key]["local_weight"],
                "candidate_local_weight": report[key].get(
                    "candidate_local_weight"
                ),
                "threshold": report[key]["threshold"],
                "radius": report[key].get("context_radius_frames"),
                "macro_precision": report[key]["macro_precision"],
                "macro_recall": report[key]["macro_recall"],
                "macro_f1": report[key]["macro_f1"],
                "macro_median_timing_error_ms": report[key][
                    "macro_median_timing_error_ms"
                ],
            }
            for key in keys
        },
    }, indent=2))


if __name__ == "__main__":
    main()
