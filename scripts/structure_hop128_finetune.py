"""Fine-tune long-context attack selection at a 128-sample feature hop."""

import argparse
import json
import os
from pathlib import Path
import random

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from audio_authoring_sequence import _load_records
from predicted_authoring_sequence import _blend_attack_scores
from structure_distill_ensemble import _load_student
from structure_highres_timing import _timing_metrics
from structure_multires_spectrogram import (
    _audio_path,
    _short_primary_features,
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


HIGHRES_HOP_LENGTH = 128


def _highres_features(audio_path, arrays):
    import librosa

    y, _ = librosa.load(
        audio_path,
        sr=FEATURE_SAMPLE_RATE,
        mono=True,
    )
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=FEATURE_SAMPLE_RATE,
        n_fft=2048,
        hop_length=HIGHRES_HOP_LENGTH,
        n_mels=128,
        fmin=30,
        fmax=FEATURE_SAMPLE_RATE / 2,
        power=2,
    )
    log_mel = np.clip(
        (
            librosa.power_to_db(mel, ref=1.0, top_db=80)
            + 80
        )
        / 80,
        0,
        1,
    ).T.astype(np.float32)
    delta = np.maximum(
        np.diff(log_mel, axis=0, prepend=log_mel[:1]),
        0,
    ).astype(np.float32)
    old_positions = np.linspace(
        0,
        1,
        len(arrays["beat_phase"]),
    )
    new_positions = np.linspace(0, 1, len(log_mel))
    beat_phase = np.column_stack([
        np.interp(
            new_positions,
            old_positions,
            arrays["beat_phase"][:, column],
        )
        for column in range(2)
    ]).astype(np.float32)
    return np.concatenate(
        (log_mel, delta, beat_phase),
        axis=1,
    )


def _target_frames(times, frames):
    return np.clip(
        np.rint(
            np.asarray(times)
            * FEATURE_SAMPLE_RATE
            / HIGHRES_HOP_LENGTH
        ).astype(int),
        0,
        frames - 1,
    )


def _score_metrics(records, threshold):
    evaluated = []
    for record in records:
        peaks = _attack_peaks(record["scores"], threshold)
        evaluated.append({
            "song_id": record["song_id"],
            "reference": record["reference"],
            "predicted": (
                peaks * record["hop_length"] / FEATURE_SAMPLE_RATE
            ),
        })
    return _timing_metrics(evaluated)


def _blend_resolution_records(records, highres_weight):
    blended = []
    for record in records:
        high_positions = np.linspace(0, 1, len(record["scores"]))
        coarse_positions = np.linspace(
            0,
            1,
            len(record["coarse_scores"]),
        )
        coarse = np.interp(
            high_positions,
            coarse_positions,
            record["coarse_scores"],
        )
        blended.append({
            **record,
            "scores": (
                highres_weight * record["scores"]
                + (1 - highres_weight) * coarse
            ),
        })
    return blended


def _cache_record(
    item,
    references,
    teacher,
    students,
    device,
    cache_root,
    student_weight,
    attack_threshold,
):
    import torch

    cache_path = cache_root / f"{item['song_id']}.npz"
    arrays, _ = _load_feature_artifact(item["path"])
    audio_path = _audio_path(item)
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cached:
            features = cached["features"]
    else:
        features = _highres_features(audio_path, arrays)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, features=features)
    with torch.no_grad():
        teacher_scores = torch.sigmoid(
            teacher(
                torch.from_numpy(
                    _structure_input_features(arrays)
                ).unsqueeze(0).to(device)
            )["attack"][0]
        ).cpu().numpy()
        short = torch.from_numpy(
            _short_primary_features(
                audio_path,
                arrays,
                short_delta=True,
            )
        ).unsqueeze(0).to(device)
        blended_scores = _blend_attack_scores(
            teacher_scores,
            [
                torch.sigmoid(
                    student(short)["attack"][0]
                ).cpu().numpy()
                for student in students
            ],
            student_weight,
        )
    high_positions = np.linspace(0, 1, len(features))
    coarse_positions = np.linspace(0, 1, len(teacher_scores))
    teacher_targets = np.interp(
        high_positions,
        coarse_positions,
        teacher_scores,
    ).astype(np.float32)
    reference = np.asarray(
        references[item["song_id"]]["times"],
        dtype=float,
    )
    attack_target = np.zeros(len(features), dtype=np.float32)
    attack_target[_target_frames(reference, len(features))] = 1
    return {
        "song_id": item["song_id"],
        "split": item["split"],
        "features": features,
        "attack": attack_target,
        "teacher": teacher_targets,
        "reference": reference,
        "coarse_scores": blended_scores,
        "coarse_threshold": attack_threshold,
    }


def run(
    index_path,
    authoring_index,
    teacher_path,
    student_paths,
    cache_root,
    output,
    seed=0,
    epochs=5,
    device="auto",
    teacher_weight=0.5,
    student_weight=0.2,
    attack_threshold=0.525,
    preserve_context_seconds=True,
    blend_coarse=False,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    _, items = _load_feature_index(index_path)
    references = {
        record["song_id"]: record
        for record in _load_records(index_path, authoring_index)
    }
    model, teacher_configuration = _load_model(teacher_path, device)
    students = [
        _load_student(path, device)[0]
        for path in student_paths
    ]
    records = [
        _cache_record(
            item,
            references,
            model,
            students,
            device,
            Path(cache_root).resolve(),
            student_weight,
            attack_threshold,
        )
        for item in items
    ]
    context_pool_sizes = [33, 129, 257]
    if preserve_context_seconds:
        context_pool_sizes = [65, 257, 513]
        model.context_pools = nn.ModuleList([
            nn.AvgPool1d(
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                count_include_pad=False,
            )
            for kernel_size in context_pool_sizes
        ]).to(device)
    train = [record for record in records if record["split"] == "train"]
    validation = [
        record for record in records if record["split"] == "validation"
    ]
    test = [record for record in records if record["split"] == "test"]
    positives = sum(float(record["attack"].sum()) for record in train)
    frames = sum(len(record["attack"]) for record in train)
    positive_weight = min(
        50.0,
        max(1.0, (frames - positives) / positives),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )
    epoch_losses = []
    for _ in range(epochs):
        model.train()
        total = 0.0
        for record in train:
            features = torch.from_numpy(
                record["features"]
            ).unsqueeze(0).to(device)
            attacks = torch.from_numpy(
                record["attack"]
            ).unsqueeze(0).to(device)
            teacher_targets = torch.from_numpy(
                record["teacher"]
            ).unsqueeze(0).to(device)
            logits = model(features)["attack"]
            hard_loss = functional.binary_cross_entropy_with_logits(
                logits,
                attacks,
                pos_weight=torch.tensor(
                    positive_weight,
                    device=device,
                ),
            )
            distill_loss = (
                functional.binary_cross_entropy_with_logits(
                    logits,
                    teacher_targets,
                )
            )
            loss = hard_loss + teacher_weight * distill_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
        epoch_losses.append(total / len(train))

    model.eval()
    with torch.no_grad():
        for record in validation + test:
            record["scores"] = torch.sigmoid(
                model(
                    torch.from_numpy(
                        record["features"]
                    ).unsqueeze(0).to(device)
                )["attack"][0]
            ).cpu().numpy()
            record["hop_length"] = HIGHRES_HOP_LENGTH
    coarse_validation = [{
        **record,
        "scores": record["coarse_scores"],
        "hop_length": FEATURE_HOP_LENGTH,
    } for record in validation]
    baseline = _score_metrics(
        coarse_validation,
        attack_threshold,
    )
    candidates = []
    weights = (
        np.linspace(0.1, 0.9, 9)
        if blend_coarse
        else [1.0]
    )
    for highres_weight in weights:
        candidate_records = _blend_resolution_records(
            validation,
            highres_weight,
        )
        for threshold in np.linspace(0.1, 0.9, 33):
            candidates.append({
                "highres_weight": float(highres_weight),
                "threshold": float(threshold),
                "metrics": _score_metrics(
                    candidate_records,
                    threshold,
                ),
            })
    guarded = [
        row for row in candidates
        if (
            row["metrics"]["macro_f1"] >= baseline["macro_f1"]
            and row["metrics"]["macro_recall"] >= baseline["macro_recall"]
            and row["metrics"]["macro_median_timing_error_ms"] is not None
            and row["metrics"]["macro_p95_timing_error_ms"] is not None
            and row["metrics"]["macro_median_timing_error_ms"]
            < baseline["macro_median_timing_error_ms"]
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
        "schema": "rb3charts-structure-hop128-finetune",
        "schema_version": 1,
        "scope": "offline authorized-data evaluation",
        "selection_rule": (
            "preserve selected coarse-blend validation macro F1 and recall, "
            "strictly improve exact-time median, and not worsen exact-time P95"
        ),
        "seed": seed,
        "epochs": epochs,
        "device": device,
        "teacher_weight": teacher_weight,
        "student_weight": student_weight,
        "coarse_attack_threshold": attack_threshold,
        "highres_hop_length": HIGHRES_HOP_LENGTH,
        "context_pool_sizes": context_pool_sizes,
        "preserve_context_seconds": preserve_context_seconds,
        "blend_coarse": blend_coarse,
        "positive_weight": positive_weight,
        "epoch_losses": epoch_losses,
        "feature_index_sha256": _hash_file(index_path),
        "authoring_index_sha256": _hash_file(authoring_index),
        "teacher_sha256": _hash_file(teacher_path),
        "student_sha256": [_hash_file(path) for path in student_paths],
        "validation_coarse_baseline": baseline,
        "validation_candidates": candidates,
        "validation_selected": selected,
        "guard_passed": bool(guarded),
        "test_evaluated": bool(guarded and test),
    }
    if guarded and test:
        report["test_coarse_baseline"] = _score_metrics(
            [{
                **record,
                "scores": record["coarse_scores"],
                "hop_length": FEATURE_HOP_LENGTH,
            } for record in test],
            attack_threshold,
        )
        report["test_selected"] = _score_metrics(
            _blend_resolution_records(
                test,
                selected["highres_weight"],
            ),
            selected["threshold"],
        )
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": report["schema"],
        "schema_version": report["schema_version"],
        "model_state": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "configuration": {
            **teacher_configuration,
            "hop_length": HIGHRES_HOP_LENGTH,
            "context_pool_sizes": context_pool_sizes,
            "attack_threshold": selected["threshold"],
        },
        "training_report": report,
    }, output)
    save_json(output.with_suffix(".json"), report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path)
    parser.add_argument("authoring_index", type=Path)
    parser.add_argument("teacher", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("students", type=Path, nargs="+")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--teacher-weight", type=float, default=0.5)
    parser.add_argument("--student-weight", type=float, default=0.2)
    parser.add_argument("--attack-threshold", type=float, default=0.525)
    parser.add_argument(
        "--halve-context-seconds",
        action="store_true",
        help="Keep the original frame-count pools instead of time-matched pools",
    )
    parser.add_argument("--blend-coarse", action="store_true")
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
        args.students,
        args.cache.resolve(),
        args.output.resolve(),
        seed=args.seed,
        epochs=args.epochs,
        device=args.device,
        teacher_weight=args.teacher_weight,
        student_weight=args.student_weight,
        attack_threshold=args.attack_threshold,
        preserve_context_seconds=not args.halve_context_seconds,
        blend_coarse=args.blend_coarse,
    )
    print(json.dumps({
        "validation_coarse_baseline": report[
            "validation_coarse_baseline"
        ],
        "validation_selected": report["validation_selected"],
        "guard_passed": report["guard_passed"],
        "test_evaluated": report["test_evaluated"],
        "test_coarse_baseline": report.get("test_coarse_baseline"),
        "test_selected": report.get("test_selected"),
    }, indent=2))


if __name__ == "__main__":
    main()
