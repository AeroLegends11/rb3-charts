"""Distill authorized long-context chartability into short-window audio."""

import argparse
import json
import os
from pathlib import Path
import random

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from structure_multires_spectrogram import (
    _audio_path,
    _macro_metrics,
    _short_primary_features,
)
from structure_timing_snap import _load_model
from rb3charts.model import save_json
from rb3charts.structure_model import build_structure_model
from rb3charts.training import (
    FEATURE_HOP_LENGTH,
    FEATURE_SAMPLE_RATE,
    _hash_file,
    _load_feature_artifact,
    _load_feature_index,
    _structure_input_features,
)


def run(
    index_path,
    teacher_path,
    output,
    seed=0,
    epochs=20,
    device="auto",
    teacher_weight=0.25,
):
    import torch
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
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")

    teacher, teacher_configuration = _load_model(
        teacher_path,
        device,
    )
    if teacher_configuration.get("architecture") != "multi-scale":
        raise ValueError("Distillation teacher must be multi-scale")
    index_path, items = _load_feature_index(index_path)
    records = []
    with torch.no_grad():
        for item in items:
            arrays, metadata = _load_feature_artifact(item["path"])
            teacher_features = _structure_input_features(arrays)
            teacher_probabilities = torch.sigmoid(
                teacher(
                    torch.from_numpy(
                        teacher_features
                    ).unsqueeze(0).to(device)
                )["attack"][0]
            ).cpu().numpy()
            records.append({
                "song_id": item["song_id"],
                "split": item["split"],
                "features": _short_primary_features(
                    _audio_path(item),
                    arrays,
                    short_delta=True,
                ),
                "teacher": teacher_probabilities.astype(np.float32),
                "attack": arrays["attack_target"].astype(np.float32),
                "polyphony": arrays["polyphony_target"].astype(np.int64),
                "sustain": arrays["sustain_target"].astype(np.int64),
                "targets": set(metadata.get("targets") or (
                    "attack",
                    "polyphony",
                    "sustain",
                )),
            })
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
    model = build_structure_model(
        "multi-scale",
        input_features=258,
        hidden_features=64,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
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
            polyphony = torch.from_numpy(
                record["polyphony"]
            ).unsqueeze(0).to(device)
            sustain = torch.from_numpy(
                record["sustain"]
            ).unsqueeze(0).to(device)
            outputs = model(features)
            hard_loss = functional.binary_cross_entropy_with_logits(
                outputs["attack"],
                attacks,
                pos_weight=torch.tensor(positive_weight, device=device),
            )
            teacher_loss = functional.binary_cross_entropy_with_logits(
                outputs["attack"],
                teacher_targets,
            )
            attack_mask = attacks > 0.5
            polyphony_loss = (
                functional.cross_entropy(
                    outputs["polyphony"][attack_mask],
                    polyphony[attack_mask],
                )
                if "polyphony" in record["targets"]
                else outputs["polyphony"].sum() * 0
            )
            sustain_loss = (
                functional.cross_entropy(
                    outputs["sustain"].reshape(-1, 7),
                    sustain.reshape(-1),
                )
                if "sustain" in record["targets"]
                else outputs["sustain"].sum() * 0
            )
            loss = (
                hard_loss
                + teacher_weight * teacher_loss
                + 0.5 * polyphony_loss
                + 0.25 * sustain_loss
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
        epoch_losses.append(total / len(train))

    def predict(selected):
        model.eval()
        result = []
        with torch.no_grad():
            for record in selected:
                scores = torch.sigmoid(model(
                    torch.from_numpy(
                        record["features"]
                    ).unsqueeze(0).to(device)
                )["attack"][0]).cpu().numpy()
                result.append((
                    record["song_id"],
                    scores,
                    np.flatnonzero(record["attack"] > 0).tolist(),
                ))
        return result

    tolerance_frames = max(
        1,
        round(0.05 * FEATURE_SAMPLE_RATE / FEATURE_HOP_LENGTH),
    )
    validation_predictions = predict(validation)
    candidates = [
        _macro_metrics(
            validation_predictions,
            threshold,
            tolerance_frames,
        )
        for threshold in np.linspace(0.05, 0.95, 37)
    ]
    selected = max(
        candidates,
        key=lambda row: (
            row["macro_f1"],
            row["macro_precision"],
            row["f1"],
            row["threshold"],
        ),
    )
    baseline = {
        "macro_f1": 0.759665373617493,
        "macro_recall": 0.718132099284813,
        "macro_median_timing_error_ms": 14.9271137026239,
        "macro_p95_timing_error_ms": 38.14706835115,
    }
    guard_passed = bool(
        selected["macro_f1"] >= baseline["macro_f1"]
        and selected["macro_recall"] >= baseline["macro_recall"]
        and selected["macro_median_timing_error_ms"]
        < baseline["macro_median_timing_error_ms"]
        and selected["macro_p95_timing_error_ms"]
        <= baseline["macro_p95_timing_error_ms"]
    )
    report = {
        "schema": "rb3charts-structure-short-distillation",
        "schema_version": 1,
        "scope": "offline authorized-data evaluation",
        "seed": seed,
        "epochs": epochs,
        "device": device,
        "teacher_weight": teacher_weight,
        "teacher_sha256": _hash_file(teacher_path),
        "feature_index_sha256": _hash_file(index_path),
        "student_representation": {
            "fft_size": 512,
            "mel_bins": 128,
            "positive_delta_bins": 128,
            "beat_phase_features": 2,
        },
        "positive_weight": positive_weight,
        "epoch_losses": epoch_losses,
        "validation_baseline": baseline,
        "validation": selected,
        "guard_passed": guard_passed,
        "test_evaluated": guard_passed and bool(test),
    }
    if guard_passed and test:
        report["test"] = _macro_metrics(
            predict(test),
            selected["threshold"],
            tolerance_frames,
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
            **report["student_representation"],
            "attack_threshold": selected["threshold"],
        },
        "training_report": report,
    }, output)
    save_json(output.with_suffix(".json"), report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path)
    parser.add_argument("teacher", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--teacher-weight", type=float, default=0.25)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    args = parser.parse_args()
    report = run(
        args.index,
        args.teacher,
        args.output,
        seed=args.seed,
        epochs=args.epochs,
        device=args.device,
        teacher_weight=args.teacher_weight,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
