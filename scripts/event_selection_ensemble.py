"""Evaluate a fixed multi-seed event-selection ensemble."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from event_selection_pilot import (
    FEATURE_NAMES,
    _hash_file,
    _load_records,
    _metrics,
    build_selector_model,
)
from rb3charts.model import save_json


def _score(records, checkpoints):
    import torch

    loaded = []
    for checkpoint in checkpoints:
        if (
            checkpoint.get("schema") != "rb3charts-event-selection-model"
            or checkpoint.get("schema_version") != 1
            or tuple(checkpoint.get("feature_names") or ()) != FEATURE_NAMES
        ):
            raise ValueError("Unsupported event-selection checkpoint")
        model = build_selector_model()
        model.load_state_dict(checkpoint["state"], strict=True)
        model.eval()
        loaded.append((
            model,
            np.asarray(checkpoint["mean"], dtype=np.float32),
            np.asarray(checkpoint["scale"], dtype=np.float32),
        ))
    scored = []
    with torch.no_grad():
        for record in records:
            probabilities = []
            corrections = []
            for model, mean, scale in loaded:
                features = torch.from_numpy(
                    ((record["features"] - mean) / scale)[None, :, :]
                )
                logits, timing = model(features)
                probabilities.append(torch.sigmoid(logits[0]).numpy())
                corrections.append(torch.tanh(timing[0]).numpy() * 0.08)
            scored.append({
                **record,
                "scores": np.mean(probabilities, axis=0),
                "corrections": np.mean(corrections, axis=0),
            })
    return scored


def _baseline(records):
    return [
        {
            **record,
            "scores": record["accepted"].astype(np.float32),
            "corrections": np.zeros(len(record["times"]), dtype=np.float32),
        }
        for record in records
    ]


def run(
    model_paths,
    index_path,
    manifest_path,
    output,
    evaluate_test=False,
):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    checkpoints = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in model_paths
    ]
    records = _load_records(index_path, manifest_path)
    validation = [
        record for record in records if record["split"] == "validation"
    ]
    test = [record for record in records if record["split"] == "test"]
    validation_baseline = _metrics(_baseline(validation), 0.5, False)
    validation_scored = _score(validation, checkpoints)
    validation_timing_only = [
        {
            **record,
            "scores": original["accepted"].astype(np.float32),
        }
        for record, original in zip(validation_scored, validation)
    ]
    candidates = [
        _metrics(validation_scored, threshold, True)
        for threshold in np.linspace(0.05, 0.95, 37)
    ]
    guarded = [
        row
        for row in candidates
        if (
            row["macro_recall"] >= validation_baseline["macro_recall"]
            and row["macro_weakest_4s_recall"]
            >= validation_baseline["macro_weakest_4s_recall"]
            and row["macro_median_timing_error_ms"]
            <= validation_baseline["macro_median_timing_error_ms"]
        )
    ]
    selected = max(
        guarded or candidates,
        key=lambda row: (
            row["macro_f1"],
            row["macro_precision"],
            row["threshold"],
        ),
    )
    report = {
        "schema": "rb3charts-event-selection-ensemble",
        "schema_version": 1,
        "scope": "offline authorized-data experiment",
        "model_sha256": [_hash_file(path) for path in model_paths],
        "validation": {
            "guard_passed": bool(guarded),
            "baseline_current_acceptance": validation_baseline,
            "timing_only_current_acceptance": _metrics(
                validation_timing_only,
                0.5,
                True,
            ),
            "candidate": selected,
        },
        "test_evaluated": bool(evaluate_test and test),
        "input_hashes": {
            "comparison_index_sha256": _hash_file(index_path),
            "manifest_sha256": _hash_file(manifest_path),
        },
    }
    if evaluate_test and test:
        test_scored = _score(test, checkpoints)
        test_timing_only = [
            {
                **record,
                "scores": original["accepted"].astype(np.float32),
            }
            for record, original in zip(test_scored, test)
        ]
        report["test"] = {
            "baseline_current_acceptance": _metrics(
                _baseline(test),
                0.5,
                False,
            ),
            "timing_only_current_acceptance": _metrics(
                test_timing_only,
                0.5,
                True,
            ),
            "candidate": _metrics(
                test_scored,
                selected["threshold"],
                True,
            ),
        }
    save_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("comparison_index", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    report = run(
        [path.resolve() for path in args.model],
        args.comparison_index.resolve(),
        args.manifest.resolve(),
        args.output.resolve(),
        evaluate_test=args.evaluate_test,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
