"""Calibrate per-song attack density without reading held-out chart targets."""

import argparse
import json
from pathlib import Path

import numpy as np

from structure_distill_ensemble import (
    _audio_path,
    _load_student,
    _short_primary_features,
)
from structure_timing_snap import _hash_file, _load_model
from rb3charts.model import save_json
from rb3charts.training import (
    FEATURE_HOP_LENGTH,
    FEATURE_SAMPLE_RATE,
    _decode_attack_peaks,
    _load_feature_artifact,
    _load_feature_index,
    _onset_metrics,
    _structure_input_features,
)


THRESHOLDS = np.linspace(0.05, 0.95, 181)
FEATURE_THRESHOLDS = np.linspace(0.1, 0.9, 17)


def _density_features(scores):
    duration = len(scores) * FEATURE_HOP_LENGTH / FEATURE_SAMPLE_RATE
    quantiles = np.quantile(
        scores,
        (0.5, 0.75, 0.9, 0.95, 0.975, 0.99),
    )
    peak_rates = [
        len(_decode_attack_peaks(scores, threshold)) / max(duration, 1e-6)
        for threshold in FEATURE_THRESHOLDS
    ]
    return np.asarray([
        float(np.mean(scores)),
        float(np.std(scores)),
        *quantiles.tolist(),
        *peak_rates,
    ])


def _fit_ridge(records, alpha):
    features = np.asarray([
        _density_features(record["scores"])
        for record in records
    ])
    duration = np.asarray([
        len(record["scores"]) * FEATURE_HOP_LENGTH / FEATURE_SAMPLE_RATE
        for record in records
    ])
    target = np.log(np.asarray([
        len(record["actual"]) / max(seconds, 1e-6)
        for record, seconds in zip(records, duration)
    ]))
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-6] = 1.0
    design = np.column_stack((
        np.ones(len(features)),
        (features - mean) / scale,
    ))
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + alpha * penalty,
        design.T @ target,
    )
    return {
        "mean": mean,
        "scale": scale,
        "coefficients": coefficients,
    }


def _predict_rate(model, scores):
    features = (_density_features(scores) - model["mean"]) / model["scale"]
    design = np.concatenate(([1.0], features))
    return float(np.exp(design @ model["coefficients"]))


def _threshold_for_rate(scores, rate):
    duration = len(scores) * FEATURE_HOP_LENGTH / FEATURE_SAMPLE_RATE
    target_count = rate * duration
    return float(min(
        THRESHOLDS,
        key=lambda threshold: (
            abs(
                len(_decode_attack_peaks(scores, threshold))
                - target_count
            ),
            abs(threshold - 0.525),
        ),
    ))


def _summarize(records, model=None, strength=0.0):
    songs = {}
    for record in records:
        scores = record["scores"]
        baseline_rate = (
            len(_decode_attack_peaks(scores, 0.525))
            * FEATURE_SAMPLE_RATE
            / max(1, len(scores) * FEATURE_HOP_LENGTH)
        )
        predicted_rate = (
            _predict_rate(model, scores) if model is not None
            else baseline_rate
        )
        selected_rate = (
            (1 - strength) * baseline_rate
            + strength * predicted_rate
        )
        threshold = _threshold_for_rate(scores, selected_rate)
        metrics = _onset_metrics(
            [(scores, record["actual"])],
            threshold,
            max(
                1,
                round(0.05 * FEATURE_SAMPLE_RATE / FEATURE_HOP_LENGTH),
            ),
        )
        metrics["baseline_rate"] = float(baseline_rate)
        metrics["predicted_rate"] = float(predicted_rate)
        songs[record["song_id"]] = metrics
    timing = [
        row["median_timing_error_ms"]
        for row in songs.values()
        if row["median_timing_error_ms"] is not None
    ]
    return {
        "strength": float(strength),
        "macro_f1": float(np.mean([row["f1"] for row in songs.values()])),
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


def run(
    index_path,
    teacher_path,
    student_paths,
    output,
    device="auto",
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
            arrays, _ = _load_feature_artifact(item["path"])
            teacher_scores = torch.sigmoid(teacher(
                torch.from_numpy(
                    _structure_input_features(arrays)
                ).unsqueeze(0).to(device)
            )["attack"][0]).cpu().numpy()
            short_features = _short_primary_features(
                _audio_path(item),
                arrays,
                short_delta=True,
            )
            student_scores = np.mean([
                torch.sigmoid(student(
                    torch.from_numpy(
                        short_features
                    ).unsqueeze(0).to(device)
                )["attack"][0]).cpu().numpy()
                for student in students
            ], axis=0)
            records.append({
                "song_id": item["song_id"],
                "split": item["split"],
                "scores": 0.8 * teacher_scores + 0.2 * student_scores,
                "actual": np.flatnonzero(arrays["attack_target"] > 0),
            })
    train = [record for record in records if record["split"] == "train"]
    validation = [
        record for record in records if record["split"] == "validation"
    ]
    test = [record for record in records if record["split"] == "test"]
    baseline = _summarize(validation)
    candidates = []
    fitted = {}
    for alpha in (0.1, 1.0, 10.0, 100.0):
        fitted[alpha] = _fit_ridge(train, alpha)
        for strength in (0.25, 0.5, 0.75, 1.0):
            metrics = _summarize(
                validation,
                fitted[alpha],
                strength,
            )
            candidates.append({
                "alpha": alpha,
                **metrics,
            })
    selected = max(
        candidates,
        key=lambda row: (
            row["macro_f1"],
            row["macro_precision"],
            row["macro_recall"],
        ),
    )
    guard_passed = (
        selected["macro_f1"] > baseline["macro_f1"]
        and selected["macro_recall"] >= baseline["macro_recall"]
        and selected["macro_median_timing_error_ms"]
        <= baseline["macro_median_timing_error_ms"]
    )
    report = {
        "schema": "rb3charts-structure-density-calibration",
        "schema_version": 1,
        "scope": (
            "offline authorized-data evaluation; density predictor fits train "
            "targets, configuration selection uses validation, and test stays "
            "sealed unless the validation guard passes"
        ),
        "feature_index_sha256": _hash_file(index_path),
        "teacher_sha256": _hash_file(teacher_path),
        "student_sha256": [
            _hash_file(path) for path in student_paths
        ],
        "train_items": len(train),
        "validation_items": len(validation),
        "test_items": len(test),
        "validation_baseline": baseline,
        "validation_candidates": candidates,
        "validation_selected": selected,
        "guard_passed": guard_passed,
        "test_evaluated": bool(guard_passed and test),
    }
    if guard_passed and test:
        report["test_baseline"] = _summarize(test)
        report["test_selected"] = _summarize(
            test,
            fitted[selected["alpha"]],
            selected["strength"],
        )
    save_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path)
    parser.add_argument("teacher", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("students", type=Path, nargs="+")
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
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
