"""Blend local and phrase-context attack scores under timing/recall guards."""

import argparse
import json
from pathlib import Path

import numpy as np

from structure_timing_snap import _hash_file, _load_model
from rb3charts.model import save_json
from rb3charts.training import (
    FEATURE_HOP_LENGTH,
    FEATURE_SAMPLE_RATE,
    _load_feature_artifact,
    _load_feature_index,
    _onset_metrics,
)


def _summarize(records, local_weight, threshold, tolerance_frames):
    songs = {}
    for record in records:
        scores = (
            local_weight * record["local_scores"]
            + (1 - local_weight) * record["context_scores"]
        )
        songs[record["song_id"]] = _onset_metrics(
            [(scores, record["actual"])],
            threshold,
            tolerance_frames,
        )
    timing = [
        row["median_timing_error_ms"]
        for row in songs.values()
        if row["median_timing_error_ms"] is not None
    ]
    return {
        "local_weight": float(local_weight),
        "context_weight": float(1 - local_weight),
        "threshold": float(threshold),
        "macro_f1": float(np.mean([
            row["f1"] for row in songs.values()
        ])),
        "macro_precision": float(np.mean([
            row["precision"] for row in songs.values()
        ])),
        "macro_recall": float(np.mean([
            row["recall"] for row in songs.values()
        ])),
        "macro_median_timing_error_ms": (
            float(np.mean(timing)) if timing else None
        ),
        "songs": songs,
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
    local_baseline = _summarize(
        validation,
        1.0,
        float(local_configuration["attack_threshold"]),
        tolerance_frames,
    )
    context_baseline = _summarize(
        validation,
        0.0,
        float(context_configuration["attack_threshold"]),
        tolerance_frames,
    )
    candidates = [
        _summarize(
            validation,
            local_weight,
            threshold,
            tolerance_frames,
        )
        for local_weight in np.linspace(0.1, 0.9, 9)
        for threshold in np.linspace(0.05, 0.95, 37)
    ]
    guarded = [
        row for row in candidates
        if (
            row["macro_recall"] >= context_baseline["macro_recall"]
            and row["macro_median_timing_error_ms"]
            <= local_baseline["macro_median_timing_error_ms"]
        )
    ]
    selected = max(
        guarded or candidates,
        key=lambda row: (
            row["macro_f1"],
            row["macro_precision"],
            -row["macro_median_timing_error_ms"],
        ),
    )
    test = [record for record in records if record["split"] == "test"]
    report = {
        "schema": "rb3charts-structure-score-ensemble",
        "schema_version": 1,
        "scope": "offline authorized-data evaluation",
        "selection_rule": (
            "preserve context validation macro recall and local validation "
            "macro median timing, then maximize macro F1"
        ),
        "guard_passed": bool(guarded),
        "context_model_sha256": _hash_file(context_path),
        "local_model_sha256": _hash_file(local_path),
        "feature_index_sha256": _hash_file(index_path),
        "validation_local_baseline": local_baseline,
        "validation_context_baseline": context_baseline,
        "validation_selected": selected,
        "test_local_baseline": _summarize(
            test,
            1.0,
            float(local_configuration["attack_threshold"]),
            tolerance_frames,
        ),
        "test_context_baseline": _summarize(
            test,
            0.0,
            float(context_configuration["attack_threshold"]),
            tolerance_frames,
        ),
        "test_selected": _summarize(
            test,
            selected["local_weight"],
            selected["threshold"],
            tolerance_frames,
        ),
    }
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
    print(json.dumps({
        "guard_passed": report["guard_passed"],
        **{
            key: {
                "local_weight": report[key]["local_weight"],
                "threshold": report[key]["threshold"],
                "macro_precision": report[key]["macro_precision"],
                "macro_recall": report[key]["macro_recall"],
                "macro_f1": report[key]["macro_f1"],
                "macro_median_timing_error_ms": report[key][
                    "macro_median_timing_error_ms"
                ],
            }
            for key in (
                "validation_local_baseline",
                "validation_context_baseline",
                "validation_selected",
                "test_local_baseline",
                "test_context_baseline",
                "test_selected",
            )
        },
    }, indent=2))


if __name__ == "__main__":
    main()
