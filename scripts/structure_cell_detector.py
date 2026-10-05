"""Train a coarse-cell attack detector with explicit within-cell timing."""

import argparse
import json
import os
from pathlib import Path
import random

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from rb3charts.model import save_json
from rb3charts.training import (
    FEATURE_HOP_LENGTH,
    FEATURE_SAMPLE_RATE,
    _hash_file,
    _load_feature_artifact,
    _load_feature_index,
    _structure_input_features,
)


CELL_FRAMES = 4


def _pack_cells(encoded, cell_frames=CELL_FRAMES):
    import torch.nn.functional as functional

    padding = (-encoded.shape[-1]) % cell_frames
    if padding:
        encoded = functional.pad(encoded, (0, padding))
    batch, channels, frames = encoded.shape
    return encoded.reshape(
        batch,
        channels,
        frames // cell_frames,
        cell_frames,
    ).permute(0, 1, 3, 2).reshape(
        batch,
        channels * cell_frames,
        frames // cell_frames,
    )


def _cell_targets(attacks, cell_frames=CELL_FRAMES):
    attacks = np.asarray(attacks) > 0.5
    cells = (len(attacks) + cell_frames - 1) // cell_frames
    objectness = np.zeros(cells, dtype=np.float32)
    phase = np.zeros(cells, dtype=np.int64)
    for frame in np.flatnonzero(attacks):
        cell, offset = divmod(int(frame), cell_frames)
        if objectness[cell]:
            raise ValueError(
                f"Cell {cell} contains multiple attacks; reduce cell size"
            )
        objectness[cell] = 1
        phase[cell] = offset
    return objectness, phase


def _decode_cells(
    objectness,
    phase,
    threshold,
    cell_frames=CELL_FRAMES,
    minimum_distance=4,
):
    candidates = [
        (
            float(score),
            cell * cell_frames + int(phase[cell]),
        )
        for cell, score in enumerate(objectness)
        if score >= threshold
    ]
    selected = []
    for score, frame in sorted(candidates, reverse=True):
        if all(abs(frame - existing) >= minimum_distance for _, existing in selected):
            selected.append((score, frame))
    return sorted(frame for _, frame in selected)


def _metrics(records, threshold, tolerance_frames):
    songs = []
    timing_errors = []
    for record in records:
        predicted = _decode_cells(
            record["objectness"],
            record["phase"],
            threshold,
        )
        actual = record["actual"]
        unmatched = set(range(len(actual)))
        matched = 0
        song_errors = []
        for frame in predicted:
            candidates = [
                index
                for index in unmatched
                if abs(int(actual[index]) - frame) <= tolerance_frames
            ]
            if not candidates:
                continue
            best = min(
                candidates,
                key=lambda index: abs(int(actual[index]) - frame),
            )
            error = abs(int(actual[best]) - frame)
            song_errors.append(error)
            timing_errors.append(error)
            unmatched.remove(best)
            matched += 1
        precision = matched / max(1, len(predicted))
        recall = matched / max(1, len(actual))
        songs.append({
            "song_id": record["song_id"],
            "precision": precision,
            "recall": recall,
            "f1": (
                2 * precision * recall / max(1e-12, precision + recall)
            ),
            "matched_attacks": matched,
            "predicted_attacks": len(predicted),
            "positive_attacks": len(actual),
            "median_timing_error_ms": (
                float(np.median(song_errors))
                * FEATURE_HOP_LENGTH / FEATURE_SAMPLE_RATE * 1000
                if song_errors else None
            ),
            "p95_timing_error_ms": (
                float(np.percentile(song_errors, 95))
                * FEATURE_HOP_LENGTH / FEATURE_SAMPLE_RATE * 1000
                if song_errors else None
            ),
        })
    median_values = [
        row["median_timing_error_ms"]
        for row in songs
        if row["median_timing_error_ms"] is not None
    ]
    p95_values = [
        row["p95_timing_error_ms"]
        for row in songs
        if row["p95_timing_error_ms"] is not None
    ]
    return {
        "threshold": float(threshold),
        "macro_precision": float(np.mean([row["precision"] for row in songs])),
        "macro_recall": float(np.mean([row["recall"] for row in songs])),
        "macro_f1": float(np.mean([row["f1"] for row in songs])),
        "macro_median_timing_error_ms": (
            float(np.mean(median_values)) if median_values else None
        ),
        "macro_p95_timing_error_ms": (
            float(np.mean(p95_values)) if p95_values else None
        ),
        "by_song": {row["song_id"]: row for row in songs},
    }


def _build_model(input_features, hidden_features=64, cell_frames=CELL_FRAMES):
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    class CellDetector(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_projection = nn.Conv1d(
                input_features,
                hidden_features,
                kernel_size=1,
            )
            self.local = nn.Sequential(
                nn.Conv1d(
                    hidden_features,
                    hidden_features,
                    kernel_size=5,
                    padding=2,
                ),
                nn.ReLU(),
                nn.Conv1d(
                    hidden_features,
                    hidden_features,
                    kernel_size=5,
                    padding=2,
                ),
                nn.ReLU(),
            )
            self.context_pools = nn.ModuleList([
                nn.AvgPool1d(
                    kernel_size=kernel,
                    stride=1,
                    padding=kernel // 2,
                    count_include_pad=False,
                )
                for kernel in (33, 129, 257)
            ])
            self.context_fusion = nn.Sequential(
                nn.Conv1d(
                    hidden_features * 4,
                    hidden_features,
                    kernel_size=1,
                ),
                nn.ReLU(),
            )
            self.cell_encoder = nn.Sequential(
                nn.Conv1d(
                    hidden_features * cell_frames * 2,
                    hidden_features,
                    kernel_size=1,
                ),
                nn.ReLU(),
            )
            self.objectness = nn.Conv1d(
                hidden_features,
                1,
                kernel_size=1,
            )
            self.phase = nn.Conv1d(
                hidden_features,
                cell_frames,
                kernel_size=1,
            )

        def forward(self, features):
            projected = self.input_projection(features.transpose(1, 2))
            local = self.local(projected)
            context = self.context_fusion(torch.cat(
                [local] + [pool(local) for pool in self.context_pools],
                dim=1,
            ))
            encoded = torch.cat((local, context), dim=1)
            cells = _pack_cells(encoded, cell_frames)
            cells = self.cell_encoder(cells)
            return {
                "objectness": self.objectness(cells).squeeze(1),
                "phase": self.phase(cells).transpose(1, 2),
            }

    return CellDetector()


def _predict(model, records, device):
    import torch

    model.eval()
    predictions = []
    with torch.no_grad():
        for record in records:
            outputs = model(
                torch.from_numpy(record["features"]).unsqueeze(0).to(device)
            )
            cells = len(record["objectness"])
            predictions.append({
                **record,
                "objectness": torch.sigmoid(
                    outputs["objectness"][0, :cells]
                ).cpu().numpy(),
                "phase": torch.argmax(
                    outputs["phase"][0, :cells],
                    dim=-1,
                ).cpu().numpy(),
            })
    return predictions


def run(
    index_path,
    output,
    seed=0,
    epochs=20,
    device="auto",
    phase_weight=0.05,
    initialize_model=None,
    freeze_backbone=False,
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

    index_path, items = _load_feature_index(index_path)
    records = []
    for item in items:
        arrays, _ = _load_feature_artifact(item["path"])
        objectness, phase = _cell_targets(arrays["attack_target"])
        records.append({
            "song_id": item["song_id"],
            "split": item["split"],
            "features": _structure_input_features(arrays),
            "objectness": objectness,
            "phase": phase,
            "actual": np.flatnonzero(arrays["attack_target"] > 0),
        })
    train = [record for record in records if record["split"] == "train"]
    validation = [
        record for record in records if record["split"] == "validation"
    ]
    test = [record for record in records if record["split"] == "test"]
    positive_cells = sum(float(record["objectness"].sum()) for record in train)
    total_cells = sum(len(record["objectness"]) for record in train)
    positive_weight = min(
        50.0,
        max(1.0, (total_cells - positive_cells) / positive_cells),
    )
    model = _build_model(train[0]["features"].shape[1]).to(device)
    initialization_sha256 = None
    if initialize_model is not None:
        initialize_model = Path(initialize_model).resolve()
        checkpoint = torch.load(
            initialize_model,
            map_location="cpu",
            weights_only=False,
        )
        source = checkpoint["model_state"]
        mapped = {
            "input_projection.weight": source["input_projection.weight"],
            "input_projection.bias": source["input_projection.bias"],
            "local.0.weight": source["encoder.0.weight"],
            "local.0.bias": source["encoder.0.bias"],
            "local.2.weight": source["encoder.2.weight"],
            "local.2.bias": source["encoder.2.bias"],
            "context_fusion.0.weight": source["context_fusion.weight"],
            "context_fusion.0.bias": source["context_fusion.bias"],
        }
        missing, unexpected = model.load_state_dict(mapped, strict=False)
        expected_missing = {
            key
            for key in model.state_dict()
            if key.startswith(("cell_encoder.", "objectness.", "phase."))
        }
        if set(missing) != expected_missing or unexpected:
            raise ValueError("Initialization checkpoint is incompatible")
        initialization_sha256 = _hash_file(initialize_model)
    if freeze_backbone:
        if initialize_model is None:
            raise ValueError("Frozen backbone requires initialization")
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith((
                "cell_encoder.",
                "objectness.",
                "phase.",
            ))
    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=1e-3,
        weight_decay=1e-4,
    )
    epoch_losses = []
    for _ in range(epochs):
        model.train()
        total = 0.0
        for record in train:
            outputs = model(
                torch.from_numpy(record["features"]).unsqueeze(0).to(device)
            )
            cells = len(record["objectness"])
            objectness = torch.from_numpy(
                record["objectness"]
            ).unsqueeze(0).to(device)
            phase = torch.from_numpy(
                record["phase"]
            ).unsqueeze(0).to(device)
            object_loss = functional.binary_cross_entropy_with_logits(
                outputs["objectness"][:, :cells],
                objectness,
                pos_weight=torch.tensor(positive_weight, device=device),
            )
            positive = objectness > 0.5
            phase_loss = functional.cross_entropy(
                outputs["phase"][:, :cells][positive],
                phase[positive],
            )
            loss = object_loss + phase_weight * phase_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
        epoch_losses.append(total / len(train))

    tolerance_frames = max(
        1,
        round(0.05 * FEATURE_SAMPLE_RATE / FEATURE_HOP_LENGTH),
    )
    validation_predictions = _predict(model, validation, device)
    candidates = [
        _metrics(validation_predictions, threshold, tolerance_frames)
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
        "schema": "rb3charts-structure-cell-detector",
        "schema_version": 1,
        "scope": "offline authorized-data evaluation",
        "representation": {
            "cell_frames": CELL_FRAMES,
            "cell_ms": (
                CELL_FRAMES
                * FEATURE_HOP_LENGTH
                / FEATURE_SAMPLE_RATE
                * 1000
            ),
            "heads": ["cell-objectness", "within-cell-phase"],
        },
        "seed": seed,
        "epochs": epochs,
        "device": device,
        "positive_weight": positive_weight,
        "phase_weight": phase_weight,
        "initialization_sha256": initialization_sha256,
        "freeze_backbone": freeze_backbone,
        "epoch_losses": epoch_losses,
        "feature_index_sha256": _hash_file(index_path),
        "validation_baseline": baseline,
        "validation": selected,
        "guard_passed": guard_passed,
        "test_evaluated": guard_passed and bool(test),
    }
    if guard_passed and test:
        test_predictions = _predict(model, test, device)
        report["test"] = _metrics(
            test_predictions,
            selected["threshold"],
            tolerance_frames,
        )
    checkpoint = {
        "schema": report["schema"],
        "schema_version": report["schema_version"],
        "model_state": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "configuration": report["representation"],
        "training_report": report,
    }
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    save_json(output.with_suffix(".json"), report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--phase-weight", type=float, default=0.05)
    parser.add_argument("--initialize-model", type=Path)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    args = parser.parse_args()
    report = run(
        args.index,
        args.output,
        seed=args.seed,
        epochs=args.epochs,
        device=args.device,
        phase_weight=args.phase_weight,
        initialize_model=args.initialize_model,
        freeze_backbone=args.freeze_backbone,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
