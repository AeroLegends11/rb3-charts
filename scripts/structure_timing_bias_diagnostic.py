"""Measure signed timing bias for an already-selected attack model."""

import argparse
import json
from pathlib import Path

import numpy as np

from structure_distill_ensemble import _load_student
from structure_highres_timing import _match_times, _selected_records
from structure_timing_snap import _hash_file, _load_model
from audio_authoring_sequence import _load_records
from rb3charts.model import save_json
from rb3charts.training import _load_feature_index


def _signed_metrics(record, tolerance=0.05):
    pairs = _match_times(
        record["reference"],
        record["predicted"],
        tolerance,
    )
    errors = np.asarray([
        (
            record["reference"][reference_index]
            - record["predicted"][predicted_index]
        ) * 1000
        for predicted_index, reference_index in pairs
    ])
    return {
        "matches": len(errors),
        "signed_mean_ms": (
            float(np.mean(errors)) if len(errors) else None
        ),
        "signed_median_ms": (
            float(np.median(errors)) if len(errors) else None
        ),
        "median_absolute_ms": (
            float(np.median(np.abs(errors))) if len(errors) else None
        ),
        "signed_p05_ms": (
            float(np.percentile(errors, 5)) if len(errors) else None
        ),
        "signed_p95_ms": (
            float(np.percentile(errors, 95)) if len(errors) else None
        ),
        "positive_fraction": (
            float(np.mean(errors > 0)) if len(errors) else None
        ),
    }


def run(
    index_path,
    authoring_index,
    teacher_path,
    student_paths,
    output,
    device="auto",
    student_weight=0.2,
    attack_threshold=0.525,
):
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _, items = _load_feature_index(index_path)
    references = {
        record["song_id"]: record
        for record in _load_records(index_path, authoring_index)
    }
    teacher, teacher_configuration = _load_model(teacher_path, device)
    students = [
        _load_student(path, device)[0]
        for path in student_paths
    ]
    records = _selected_records(
        items,
        references,
        teacher,
        teacher_configuration,
        students,
        device,
        student_weight,
        attack_threshold,
    )
    report = {
        "schema": "rb3charts-structure-timing-bias-diagnostic",
        "schema_version": 1,
        "scope": (
            "post-selection diagnostic only; frozen results do not select "
            "thresholds, offsets, or model parameters"
        ),
        "student_weight": student_weight,
        "attack_threshold": attack_threshold,
        "feature_index_sha256": _hash_file(index_path),
        "authoring_index_sha256": _hash_file(authoring_index),
        "teacher_sha256": _hash_file(teacher_path),
        "student_sha256": [_hash_file(path) for path in student_paths],
        "songs": {
            record["song_id"]: {
                "split": record["split"],
                **_signed_metrics(record),
            }
            for record in records
            if record["split"] in ("validation", "test")
        },
    }
    save_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path)
    parser.add_argument("authoring_index", type=Path)
    parser.add_argument("teacher", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("students", type=Path, nargs="+")
    parser.add_argument("--student-weight", type=float, default=0.2)
    parser.add_argument("--attack-threshold", type=float, default=0.525)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    args = parser.parse_args()
    report = run(
        args.index.resolve(),
        args.authoring_index.resolve(),
        args.teacher.resolve(),
        [path.resolve() for path in args.students],
        args.output.resolve(),
        device=args.device,
        student_weight=args.student_weight,
        attack_threshold=args.attack_threshold,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
