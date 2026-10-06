"""Gate distilled attack evidence by deterministic student agreement."""

import argparse
import json
from pathlib import Path

import numpy as np

from structure_distill_ensemble import _load_student
from structure_multires_spectrogram import (
    _audio_path,
    _macro_metrics,
    _short_primary_features,
)
from structure_timing_snap import _hash_file, _load_model
from rb3charts.model import save_json
from rb3charts.training import (
    FEATURE_HOP_LENGTH,
    FEATURE_SAMPLE_RATE,
    _load_feature_artifact,
    _load_feature_index,
    _structure_input_features,
)


def _variance_gated_scores(
    teacher_scores,
    student_scores,
    student_weight,
    variance_threshold,
):
    students = np.asarray(student_scores)
    mean = np.mean(students, axis=0)
    variance = np.var(students, axis=0)
    weight = student_weight * (variance <= variance_threshold)
    return teacher_scores + weight * (mean - teacher_scores)


def _metrics(records, variance_threshold, threshold, tolerance_frames):
    predictions = []
    for record in records:
        scores = _variance_gated_scores(
            record["teacher"],
            record["students"],
            record["student_weight"],
            variance_threshold,
        )
        predictions.append((
            record["song_id"],
            scores,
            record["actual"],
        ))
    return _macro_metrics(predictions, threshold, tolerance_frames)


def _mean_metrics(records, threshold, tolerance_frames):
    return _macro_metrics([
        (
            record["song_id"],
            (
                (1 - record["student_weight"]) * record["teacher"]
                + record["student_weight"]
                * np.mean(record["students"], axis=0)
            ),
            record["actual"],
        )
        for record in records
    ], threshold, tolerance_frames)


def run(
    index_path,
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
    teacher, _ = _load_model(teacher_path, device)
    students = [
        _load_student(path, device)[0]
        for path in student_paths
    ]
    _, items = _load_feature_index(index_path)
    records = []
    with torch.no_grad():
        for item in items:
            if item["split"] not in ("validation", "test"):
                continue
            arrays, _ = _load_feature_artifact(item["path"])
            teacher_scores = torch.sigmoid(
                teacher(
                    torch.from_numpy(
                        _structure_input_features(arrays)
                    ).unsqueeze(0).to(device)
                )["attack"][0]
            ).cpu().numpy()
            short = torch.from_numpy(
                _short_primary_features(
                    _audio_path(item),
                    arrays,
                    short_delta=True,
                )
            ).unsqueeze(0).to(device)
            student_scores = [
                torch.sigmoid(
                    student(short)["attack"][0]
                ).cpu().numpy()
                for student in students
            ]
            records.append({
                "song_id": item["song_id"],
                "split": item["split"],
                "teacher": teacher_scores,
                "students": student_scores,
                "student_weight": student_weight,
                "actual": np.flatnonzero(
                    arrays["attack_target"] > 0
                ).tolist(),
            })
    validation = [
        record for record in records if record["split"] == "validation"
    ]
    test = [record for record in records if record["split"] == "test"]
    active_variances = np.concatenate([
        np.var(record["students"], axis=0)[
            np.maximum(
                record["teacher"],
                np.mean(record["students"], axis=0),
            ) >= 0.1
        ]
        for record in validation
    ])
    tolerance_frames = max(
        1,
        round(0.05 * FEATURE_SAMPLE_RATE / FEATURE_HOP_LENGTH),
    )
    baseline = _mean_metrics(
        validation,
        attack_threshold,
        tolerance_frames,
    )
    candidates = [
        {
            "variance_quantile": float(quantile),
            "variance_threshold": float(np.quantile(
                active_variances,
                quantile,
            )),
            "attack_threshold": float(threshold),
        }
        for quantile in (0.25, 0.5, 0.75, 0.9)
        for threshold in np.linspace(0.4, 0.6, 17)
    ]
    for candidate in candidates:
        candidate["metrics"] = _metrics(
            validation,
            candidate["variance_threshold"],
            candidate["attack_threshold"],
            tolerance_frames,
        )
    guarded = [
        row for row in candidates
        if (
            row["metrics"]["macro_f1"] >= baseline["macro_f1"]
            and row["metrics"]["macro_recall"] >= baseline["macro_recall"]
            and row["metrics"]["macro_median_timing_error_ms"] is not None
            and row["metrics"]["macro_p95_timing_error_ms"] is not None
            and row["metrics"]["macro_median_timing_error_ms"]
            <= baseline["macro_median_timing_error_ms"]
            and row["metrics"]["macro_p95_timing_error_ms"]
            <= baseline["macro_p95_timing_error_ms"]
        )
    ]
    selected = max(
        guarded or candidates,
        key=lambda row: (
            row["metrics"]["macro_f1"],
            row["metrics"]["macro_recall"],
            -(
                row["metrics"]["macro_median_timing_error_ms"]
                if row["metrics"]["macro_median_timing_error_ms"] is not None
                else float("inf")
            ),
        ),
    )
    report = {
        "schema": "rb3charts-structure-distill-variance",
        "schema_version": 1,
        "scope": "offline authorized-data evaluation",
        "selection_rule": (
            "preserve mean-blend validation macro F1, recall, median timing, "
            "and P95 while gating distilled evidence by student agreement"
        ),
        "student_weight": student_weight,
        "attack_threshold": attack_threshold,
        "feature_index_sha256": _hash_file(index_path),
        "teacher_sha256": _hash_file(teacher_path),
        "student_sha256": [_hash_file(path) for path in student_paths],
        "validation_mean_baseline": baseline,
        "validation_candidates": candidates,
        "validation_selected": selected,
        "guard_passed": bool(guarded),
        "test_evaluated": bool(guarded and test),
    }
    if guarded and test:
        report["test_mean_baseline"] = _mean_metrics(
            test,
            attack_threshold,
            tolerance_frames,
        )
        report["test_selected"] = _metrics(
            test,
            selected["variance_threshold"],
            selected["attack_threshold"],
            tolerance_frames,
        )
    save_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path)
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
