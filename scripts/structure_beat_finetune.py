"""Fine-tune the primary multi-scale model on beat-grid attack supervision."""

import argparse
import json
import os
from pathlib import Path
import random

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from structure_beat_grid import _load_records, _metrics
from rb3charts.model import save_json
from rb3charts.structure_model import build_structure_model
from rb3charts.training import _hash_file


def _predict(model, records, device):
    import torch

    model.eval()
    predictions = []
    with torch.no_grad():
        for record in records:
            outputs = model(
                torch.from_numpy(
                    record["frame_features"]
                ).unsqueeze(0).to(device)
            )
            frames = torch.from_numpy(record["grid_frames"]).to(device)
            probabilities = torch.sigmoid(
                outputs["attack"][0, frames]
            ).cpu().numpy()
            predictions.append({
                **record,
                "probabilities": probabilities,
            })
    return predictions


def run(
    index_path,
    initialize_model,
    output,
    seed=0,
    epochs=5,
    device="auto",
    learning_rate=1e-3,
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

    initialize_model = Path(initialize_model).resolve()
    checkpoint = torch.load(
        initialize_model,
        map_location="cpu",
        weights_only=False,
    )
    configuration = checkpoint["configuration"]
    if configuration.get("architecture") != "multi-scale":
        raise ValueError("Beat fine-tuning requires a multi-scale checkpoint")
    model = build_structure_model(
        "multi-scale",
        input_features=int(configuration["input_features"]),
        hidden_features=int(configuration["hidden_features"]),
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)

    records = _load_records(index_path)
    train = [record for record in records if record["split"] == "train"]
    validation = [
        record for record in records if record["split"] == "validation"
    ]
    test = [record for record in records if record["split"] == "test"]
    positives = sum(float(record["targets"].sum()) for record in train)
    positions = sum(len(record["targets"]) for record in train)
    positive_weight = min(
        50.0,
        max(1.0, (positions - positives) / positives),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    epoch_losses = []
    for _ in range(epochs):
        model.train()
        total = 0.0
        for record in train:
            features = torch.from_numpy(
                record["frame_features"]
            ).unsqueeze(0).to(device)
            frames = torch.from_numpy(record["grid_frames"]).to(device)
            targets = torch.from_numpy(
                record["targets"]
            ).unsqueeze(0).to(device)
            polyphony = torch.from_numpy(
                record["polyphony"]
            ).unsqueeze(0).to(device)
            sustain = torch.from_numpy(
                record["sustain"]
            ).unsqueeze(0).to(device)
            outputs = model(features)
            attack_logits = outputs["attack"][:, frames]
            attack_loss = functional.binary_cross_entropy_with_logits(
                attack_logits,
                targets,
                pos_weight=torch.tensor(positive_weight, device=device),
            )
            attack_mask = targets > 0.5
            polyphony_loss = functional.cross_entropy(
                outputs["polyphony"][:, frames][attack_mask],
                polyphony[attack_mask],
            )
            sustain_loss = functional.cross_entropy(
                outputs["sustain"][:, frames].reshape(-1, 7),
                sustain.reshape(-1),
            )
            loss = (
                attack_loss
                + 0.1 * polyphony_loss
                + 0.05 * sustain_loss
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
        epoch_losses.append(total / len(train))

    validation_predictions = _predict(model, validation, device)
    candidates = [
        _metrics(validation_predictions, threshold)
        for threshold in np.linspace(0.05, 0.95, 37)
    ]
    selected = max(
        candidates,
        key=lambda row: (
            row["macro_f1"],
            row["macro_precision"],
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
        and selected["macro_median_timing_error_ms"] is not None
        and selected["macro_median_timing_error_ms"]
        < baseline["macro_median_timing_error_ms"]
        and selected["macro_p95_timing_error_ms"] is not None
        and selected["macro_p95_timing_error_ms"]
        <= baseline["macro_p95_timing_error_ms"]
    )
    report = {
        "schema": "rb3charts-structure-beat-finetune",
        "schema_version": 1,
        "scope": "offline authorized-data evaluation",
        "seed": seed,
        "epochs": epochs,
        "device": device,
        "learning_rate": learning_rate,
        "positive_weight": positive_weight,
        "initialization_sha256": _hash_file(initialize_model),
        "feature_index_sha256": _hash_file(index_path),
        "epoch_losses": epoch_losses,
        "validation_baseline": baseline,
        "validation": selected,
        "guard_passed": guard_passed,
        "test_evaluated": guard_passed and bool(test),
    }
    if guard_passed and test:
        report["test"] = _metrics(
            _predict(model, test, device),
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
            **configuration,
            "output_representation": "12-subdivision-beat-grid",
        },
        "training_report": report,
    }, output)
    save_json(output.with_suffix(".json"), report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path)
    parser.add_argument("initialize_model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    args = parser.parse_args()
    report = run(
        args.index,
        args.initialize_model,
        args.output,
        seed=args.seed,
        epochs=args.epochs,
        device=args.device,
        learning_rate=args.learning_rate,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
