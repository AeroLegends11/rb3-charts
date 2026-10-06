"""Blend a precise distilled student with its long-context teacher offline."""

import argparse
import json
from pathlib import Path

import numpy as np

from structure_multires_spectrogram import (
    _audio_path,
    _short_primary_features,
)
from structure_score_ensemble import _summarize
from structure_timing_snap import _hash_file, _load_model
from rb3charts.model import save_json
from rb3charts.structure_model import build_structure_model
from rb3charts.training import (
    FEATURE_HOP_LENGTH,
    FEATURE_SAMPLE_RATE,
    _load_feature_artifact,
    _load_feature_index,
    _structure_input_features,
)


def _load_student(path, device):
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("schema") != "rb3charts-structure-short-distillation"
        or checkpoint.get("schema_version") != 1
    ):
        raise ValueError(f"Unsupported distilled checkpoint: {path}")
    model = build_structure_model(
        "multi-scale",
        input_features=258,
        hidden_features=64,
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    return model, checkpoint["configuration"]


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
    teacher, teacher_configuration = _load_model(teacher_path, device)
    if isinstance(student_paths, (str, Path)):
        student_paths = [student_paths]
    students_with_configuration = [
        _load_student(path, device)
        for path in student_paths
    ]
    students = [model for model, _ in students_with_configuration]
    _, items = _load_feature_index(index_path)
    records = []
    with torch.no_grad():
        for item in items:
            if item["split"] not in ("validation", "test"):
                continue
            arrays, _ = _load_feature_artifact(item["path"])
            teacher_features = _structure_input_features(arrays)
            student_features = _short_primary_features(
                _audio_path(item),
                arrays,
                short_delta=True,
            )
            records.append({
                "song_id": item["song_id"],
                "split": item["split"],
                "context_scores": torch.sigmoid(
                    teacher(
                        torch.from_numpy(
                            teacher_features
                        ).unsqueeze(0).to(device)
                    )["attack"][0]
                ).cpu().numpy(),
                "local_scores": np.mean([
                    torch.sigmoid(
                        student(
                            torch.from_numpy(
                                student_features
                            ).unsqueeze(0).to(device)
                        )["attack"][0]
                    ).cpu().numpy()
                    for student in students
                ], axis=0),
                "actual": np.flatnonzero(arrays["attack_target"] > 0),
            })
    tolerance_frames = max(
        1,
        round(0.05 * FEATURE_SAMPLE_RATE / FEATURE_HOP_LENGTH),
    )
    validation = [
        record for record in records if record["split"] == "validation"
    ]
    teacher_baseline = _summarize(
        validation,
        0.0,
        float(teacher_configuration["attack_threshold"]),
        tolerance_frames,
    )
    student_baseline = max(
        (
            _summarize(
                validation,
                1.0,
                threshold,
                tolerance_frames,
            )
            for threshold in np.linspace(0.05, 0.95, 37)
        ),
        key=lambda row: (
            row["macro_f1"],
            row["macro_precision"],
        ),
    )
    candidates = [
        _summarize(
            validation,
            local_weight,
            threshold,
            tolerance_frames,
        )
        for local_weight in np.linspace(0.05, 0.95, 19)
        for threshold in np.linspace(0.05, 0.95, 37)
    ]
    guarded = [
        row for row in candidates
        if (
            row["macro_f1"] >= teacher_baseline["macro_f1"]
            and row["macro_recall"] >= teacher_baseline["macro_recall"]
            and row["macro_median_timing_error_ms"] is not None
            and row["macro_median_timing_error_ms"]
            < teacher_baseline["macro_median_timing_error_ms"]
            and row["macro_median_timing_error_ms"]
            <= student_baseline["macro_median_timing_error_ms"]
        )
    ]
    selected = max(
        guarded or candidates,
        key=lambda row: (
            row["macro_f1"],
            row["macro_precision"],
            -(
                row["macro_median_timing_error_ms"]
                if row["macro_median_timing_error_ms"] is not None
                else float("inf")
            ),
        ),
    )
    test = [record for record in records if record["split"] == "test"]
    report = {
        "schema": "rb3charts-structure-distill-ensemble",
        "schema_version": 1,
        "scope": "offline authorized-data evaluation",
        "selection_rule": (
            "preserve teacher validation macro F1 and recall, strictly improve "
            "teacher timing, and match student timing"
        ),
        "guard_passed": bool(guarded),
        "test_evaluated": bool(guarded and test),
        "feature_index_sha256": _hash_file(index_path),
        "teacher_sha256": _hash_file(teacher_path),
        "student_sha256": [
            _hash_file(path)
            for path in student_paths
        ],
        "validation_teacher_baseline": teacher_baseline,
        "validation_student_baseline": student_baseline,
        "validation_selected": selected,
    }
    if guarded and test:
        report["test_teacher_baseline"] = _summarize(
            test,
            0.0,
            float(teacher_configuration["attack_threshold"]),
            tolerance_frames,
        )
        report["test_student_baseline"] = _summarize(
            test,
            1.0,
            student_baseline["threshold"],
            tolerance_frames,
        )
        report["test_selected"] = _summarize(
            test,
            selected["local_weight"],
            selected["threshold"],
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
