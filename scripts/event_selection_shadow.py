"""Evaluate a frozen event selector without changing generated charts."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from event_selection_pilot import (
    FEATURE_NAMES,
    _candidate_groups,
    _hash_file,
    build_selector_model,
)
from rb3charts.datasets import _read_expert_gems
from rb3charts.model import save_json
from rb3charts.regression import compare_chart_attacks


def _attacks(times):
    return [
        {
            "time": float(time),
            "duration": 0.1,
            "lanes": [0],
        }
        for time in sorted(times)
    ]


def run(model_path, trace_path, reference_path, output, offset=-1.1):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )
    if (
        checkpoint.get("schema") != "rb3charts-event-selection-model"
        or checkpoint.get("schema_version") != 1
        or tuple(checkpoint.get("feature_names") or ()) != FEATURE_NAMES
    ):
        raise ValueError("Unsupported event-selection checkpoint")
    groups = _candidate_groups(trace_path)
    features = np.asarray(
        [group["features"] for group in groups],
        dtype=np.float32,
    )
    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["scale"], dtype=np.float32)
    model = build_selector_model()
    model.load_state_dict(checkpoint["state"], strict=True)
    model.eval()
    with torch.no_grad():
        logits, timing = model(torch.from_numpy(
            ((features - mean) / scale)[None, :, :]
        ))
    scores = torch.sigmoid(logits[0]).numpy()
    corrections = torch.tanh(timing[0]).numpy() * 0.08
    threshold = float(checkpoint["threshold"])
    baseline_times = [
        group["time"] + offset
        for group in groups
        if group["accepted"]
    ]
    candidate_times = [
        group["time"] + offset
        + float(np.clip(corrections[index], -0.08, 0.08))
        for index, group in enumerate(groups)
        if scores[index] >= threshold
    ]
    reference_times = sorted({
        round(float(note["start"]), 6)
        for note in _read_expert_gems(
            reference_path,
            "rock_band_midi",
            "expert",
        )
    })
    report = {
        "schema": "rb3charts-event-selection-shadow",
        "schema_version": 1,
        "scope": (
            "post-generation evaluation only; reference notes were not "
            "training, threshold-selection, or generation inputs"
        ),
        "provenance": {
            "model_sha256": _hash_file(model_path),
            "trace_sha256": _hash_file(trace_path),
            "reference_sha256": _hash_file(reference_path),
            "alignment_offset_seconds": float(offset),
            "threshold": threshold,
        },
        "baseline_current_acceptance": compare_chart_attacks(
            _attacks(reference_times),
            _attacks(baseline_times),
            tolerance=0.05,
        ),
        "candidate": compare_chart_attacks(
            _attacks(reference_times),
            _attacks(candidate_times),
            tolerance=0.05,
        ),
    }
    save_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("trace", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--offset", type=float, required=True)
    args = parser.parse_args()
    report = run(
        args.model.resolve(),
        args.trace.resolve(),
        args.reference.resolve(),
        args.output.resolve(),
        offset=args.offset,
    )
    print(json.dumps({
        key: {
            "attack": report[key]["attack"],
            "weakest_window": report[key]["attack_windows"][
                "worst_reference_recall"
            ],
        }
        for key in ("baseline_current_acceptance", "candidate")
    }, indent=2))


if __name__ == "__main__":
    main()
