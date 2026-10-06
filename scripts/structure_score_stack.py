"""Train a compact authorized-data ranker over teacher/student attack peaks."""

import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
from scipy import signal

from structure_candidate_classifier import _candidate_targets
from structure_distill_ensemble import (
    _audio_path,
    _load_student,
    _short_primary_features,
)
from structure_score_ensemble import _summarize
from structure_pitch_event_ensemble import (
    _event_support,
    _load_or_extract_events,
)
from structure_timing_snap import _hash_file, _load_model
from rb3charts.model import save_json
from rb3charts.training import (
    FEATURE_HOP_LENGTH,
    FEATURE_SAMPLE_RATE,
    _attack_peaks,
    _load_feature_artifact,
    _load_feature_index,
    _structure_input_features,
)


def _rank_percentiles(values):
    result = np.zeros(len(values), dtype=np.float32)
    if len(values):
        order = np.argsort(values, kind="stable")
        result[order] = np.linspace(0, 1, len(values))
    return result


def _stack_features(
    teacher,
    students,
    peaks,
    event_support=None,
    base_student_weight=0.2,
):
    students = np.asarray(students)
    mean = students.mean(axis=0)
    variance = students.var(axis=0)
    blend = (
        (1 - base_student_weight) * teacher
        + base_student_weight * mean
    )
    if event_support is None:
        event_support = np.zeros_like(blend)
    prominences = signal.peak_prominences(blend, peaks)[0]
    widths = signal.peak_widths(blend, peaks, rel_height=0.5)[0]
    rows = []
    score_rank = _rank_percentiles(blend[peaks])
    prominence_rank = _rank_percentiles(prominences)
    for position, peak in enumerate(peaks):
        neighborhood = blend[
            max(0, peak - 4):min(len(blend), peak + 5)
        ]
        rows.append([
            teacher[peak],
            *students[:, peak].tolist(),
            mean[peak],
            variance[peak],
            blend[peak],
            prominences[position],
            widths[position],
            score_rank[position],
            prominence_rank[position],
            blend[peak] - float(np.mean(neighborhood)),
            float(np.max(neighborhood) - np.min(neighborhood)),
            event_support[peak],
        ])
    return np.asarray(rows, dtype=np.float32)


def _build_model(input_features):
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(input_features, 16),
        nn.GELU(),
        nn.Linear(16, 1),
    )


def _predict(model, records, mean, scale, device):
    import torch

    output = []
    model.eval()
    with torch.no_grad():
        for record in records:
            probabilities = torch.sigmoid(model(
                torch.from_numpy(
                    (record["features"] - mean) / scale
                ).to(device)
            ).squeeze(1)).cpu().numpy()
            scores = np.zeros(len(record["blend"]), dtype=np.float32)
            scores[record["peaks"]] = probabilities
            output.append({
                **record,
                "context_scores": scores,
                "local_scores": np.zeros_like(scores),
            })
    return output


def run(
    index_path,
    teacher_path,
    student_paths,
    old_event_root,
    event_cache,
    output,
    seed=0,
    epochs=12,
    positive_weight_scale=1.0,
    base_student_weight=0.2,
    device="auto",
):
    import torch
    import torch.nn.functional as functional

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher, _ = _load_model(teacher_path, device)
    students = [
        _load_student(path, device)[0]
        for path in student_paths
    ]
    _, items = _load_feature_index(index_path)
    tolerance_frames = max(
        1,
        round(0.05 * FEATURE_SAMPLE_RATE / FEATURE_HOP_LENGTH),
    )
    records = []
    with torch.no_grad():
        for item in items:
            arrays, _ = _load_feature_artifact(item["path"])
            teacher_scores = torch.sigmoid(teacher(
                torch.from_numpy(
                    _structure_input_features(arrays)
                ).unsqueeze(0).to(device)
            )["attack"][0]).cpu().numpy()
            short = torch.from_numpy(_short_primary_features(
                _audio_path(item),
                arrays,
                short_delta=True,
            )).unsqueeze(0).to(device)
            student_scores = [
                torch.sigmoid(student(short)["attack"][0]).cpu().numpy()
                for student in students
            ]
            blend = (
                (1 - base_student_weight) * teacher_scores
                + base_student_weight * np.mean(
                    student_scores,
                    axis=0,
                )
            )
            peaks = _attack_peaks(blend)
            actual = np.flatnonzero(arrays["attack_target"] > 0)
            events = _load_or_extract_events(
                item,
                old_event_root,
                event_cache,
            )
            event_support = _event_support(len(blend), events)
            records.append({
                "song_id": item["song_id"],
                "split": item["split"],
                "features": _stack_features(
                    teacher_scores,
                    student_scores,
                    peaks,
                    event_support,
                    base_student_weight,
                ),
                "targets": _candidate_targets(
                    peaks,
                    actual,
                    tolerance_frames,
                ),
                "peaks": peaks,
                "blend": blend,
                "actual": actual,
            })
    train = [record for record in records if record["split"] == "train"]
    validation = [
        record for record in records if record["split"] == "validation"
    ]
    test = [record for record in records if record["split"] == "test"]
    train_features = np.concatenate([
        record["features"] for record in train
    ])
    train_targets = np.concatenate([
        record["targets"] for record in train
    ])
    mean = train_features.mean(axis=0)
    scale = np.maximum(train_features.std(axis=0), 1e-5)
    normalized = (train_features - mean) / scale
    model = _build_model(normalized.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )
    positives = max(1, int(train_targets.sum()))
    positive_weight = (
        len(train_targets) - positives
    ) / positives * positive_weight_scale
    losses = []
    generator = np.random.default_rng(seed)
    for _ in range(epochs):
        order = generator.permutation(len(normalized))
        epoch_losses = []
        model.train()
        for start in range(0, len(order), 4096):
            batch = order[start:start + 4096]
            features = torch.from_numpy(normalized[batch]).to(device)
            targets = torch.from_numpy(train_targets[batch]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features).squeeze(1)
            loss = functional.binary_cross_entropy_with_logits(
                logits,
                targets,
                pos_weight=torch.tensor(
                    positive_weight,
                    device=device,
                ),
            )
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
    baseline_records = [
        {
            **record,
            "context_scores": record["blend"],
            "local_scores": np.zeros_like(record["blend"]),
        }
        for record in validation
    ]
    baseline = _summarize(
        baseline_records,
        0.0,
        0.525,
        tolerance_frames,
    )
    validation_predictions = _predict(
        model,
        validation,
        mean,
        scale,
        device,
    )
    validation_fused = [
        {
            **predicted,
            "context_scores": record["blend"],
            "local_scores": predicted["context_scores"],
        }
        for predicted, record in zip(
            validation_predictions,
            validation,
        )
    ]
    candidates = []
    for ranker_weight in (0.05, 0.1, 0.2, 0.35, 0.5, 1.0):
        for threshold in np.linspace(0.05, 0.95, 37):
            candidates.append({
                "ranker_weight": ranker_weight,
                **_summarize(
                    validation_fused,
                    ranker_weight,
                    threshold,
                    tolerance_frames,
                ),
            })
    guarded = [
        row for row in candidates
        if (
            row["macro_f1"] > baseline["macro_f1"]
            and row["macro_recall"] >= baseline["macro_recall"]
            and row["macro_median_timing_error_ms"]
            <= baseline["macro_median_timing_error_ms"]
        )
    ]
    selected = max(
        guarded or candidates,
        key=lambda row: (
            row["macro_f1"],
            row["macro_precision"],
            row["macro_recall"],
        ),
    )
    report = {
        "schema": "rb3charts-structure-score-stack",
        "schema_version": 1,
        "scope": (
            "offline authorized-data peak ranking; model fits train only, "
            "selection uses validation, and test stays sealed unless guarded"
        ),
        "seed": seed,
        "epochs": epochs,
        "positive_weight_scale": positive_weight_scale,
        "base_student_weight": base_student_weight,
        "epoch_losses": losses,
        "train_candidates": len(train_targets),
        "train_positive_candidates": int(train_targets.sum()),
        "event_feature": "basic-pitch-proximity",
        "feature_index_sha256": _hash_file(index_path),
        "teacher_sha256": _hash_file(teacher_path),
        "student_sha256": [
            _hash_file(path) for path in student_paths
        ],
        "validation_baseline": baseline,
        "validation_selected": selected,
        "guard_passed": bool(guarded),
        "test_evaluated": bool(guarded and test),
    }
    if guarded and test:
        test_baseline_records = [
            {
                **record,
                "context_scores": record["blend"],
                "local_scores": np.zeros_like(record["blend"]),
            }
            for record in test
        ]
        report["test_baseline"] = _summarize(
            test_baseline_records,
            0.0,
            0.525,
            tolerance_frames,
        )
        test_predictions = _predict(
            model,
            test,
            mean,
            scale,
            device,
        )
        test_fused = [
            {
                **predicted,
                "context_scores": record["blend"],
                "local_scores": predicted["context_scores"],
            }
            for predicted, record in zip(test_predictions, test)
        ]
        report["test_selected"] = _summarize(
            test_fused,
            selected["ranker_weight"],
            selected["threshold"],
            tolerance_frames,
        )
    save_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path)
    parser.add_argument("teacher", type=Path)
    parser.add_argument("old_event_root", type=Path)
    parser.add_argument("event_cache", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("students", type=Path, nargs="+")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--positive-weight-scale", type=float, default=1.0)
    parser.add_argument("--base-student-weight", type=float, default=0.2)
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
        args.old_event_root.resolve(),
        args.event_cache.resolve(),
        args.output.resolve(),
        seed=args.seed,
        epochs=args.epochs,
        positive_weight_scale=args.positive_weight_scale,
        base_student_weight=args.base_student_weight,
        device=args.device,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
