"""Train per-beat attack occupancy and subdivision ranking."""

import argparse
import json
import os
from pathlib import Path
import random

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from structure_beat_grid import (
    MATCH_TOLERANCE_SECONDS,
    SUBDIVISIONS,
    _grid_times,
    _label_path,
    _nearest_frames,
    _sample_features,
)
from rb3charts.model import save_json
from rb3charts.training import (
    _hash_file,
    _load_feature_artifact,
    _load_feature_index,
)


def _beat_targets(beat_times, attack_times, subdivisions=SUBDIVISIONS):
    grid_times = _grid_times(beat_times)
    beats = len(beat_times) - 1
    occupancy = np.zeros((beats, subdivisions), dtype=np.float32)
    errors = []
    for attack in attack_times:
        if attack < beat_times[0] or attack >= beat_times[-1]:
            continue
        insertion = int(np.searchsorted(grid_times, attack))
        candidates = [
            index
            for index in (insertion - 1, insertion)
            if 0 <= index < beats * subdivisions
        ]
        best = min(candidates, key=lambda index: abs(grid_times[index] - attack))
        beat, subdivision = divmod(best, subdivisions)
        occupancy[beat, subdivision] = 1
        errors.append(abs(float(grid_times[best]) - float(attack)))
    return occupancy, np.asarray(errors, dtype=np.float64)


def _decode_beats(
    count_probabilities,
    subdivision_scores,
    beat_times,
    count_bias=0.0,
):
    predictions = []
    for beat, probabilities in enumerate(count_probabilities):
        adjusted = np.asarray(probabilities).copy()
        adjusted[1:] += count_bias
        count = int(np.argmax(adjusted))
        if count <= 0:
            continue
        order = np.argsort(
            subdivision_scores[beat],
            kind="stable",
        )[::-1][:count]
        duration = beat_times[beat + 1] - beat_times[beat]
        predictions.extend(
            beat_times[beat] + duration * int(index) / SUBDIVISIONS
            for index in order
        )
    return sorted(predictions)


def _metrics(records, count_bias=0.0):
    rows = []
    for record in records:
        predicted = _decode_beats(
            record["count_probabilities"],
            record["subdivision_scores"],
            record["beat_times"],
            count_bias,
        )
        actual = record["attack_times"]
        unmatched = set(range(len(actual)))
        errors = []
        matched = 0
        for time in predicted:
            candidates = [
                index
                for index in unmatched
                if abs(float(actual[index]) - time)
                <= MATCH_TOLERANCE_SECONDS
            ]
            if not candidates:
                continue
            best = min(
                candidates,
                key=lambda index: abs(float(actual[index]) - time),
            )
            errors.append(abs(float(actual[best]) - time) * 1000)
            unmatched.remove(best)
            matched += 1
        precision = matched / max(1, len(predicted))
        recall = matched / max(1, len(actual))
        rows.append({
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
                float(np.median(errors)) if errors else None
            ),
            "p95_timing_error_ms": (
                float(np.percentile(errors, 95)) if errors else None
            ),
        })
    medians = [
        row["median_timing_error_ms"]
        for row in rows
        if row["median_timing_error_ms"] is not None
    ]
    p95s = [
        row["p95_timing_error_ms"]
        for row in rows
        if row["p95_timing_error_ms"] is not None
    ]
    return {
        "macro_precision": float(np.mean([row["precision"] for row in rows])),
        "macro_recall": float(np.mean([row["recall"] for row in rows])),
        "macro_f1": float(np.mean([row["f1"] for row in rows])),
        "macro_median_timing_error_ms": float(np.mean(medians)),
        "macro_p95_timing_error_ms": float(np.mean(p95s)),
        "by_song": {row["song_id"]: row for row in rows},
    }


def _build_model(input_features, hidden_features=64):
    import torch
    import torch.nn as nn

    class BeatOccupancyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_projection = nn.Conv1d(
                input_features,
                hidden_features,
                kernel_size=1,
            )
            self.encoder = nn.Sequential(
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
                for kernel in (9, 37, 97)
            ])
            self.context_fusion = nn.Sequential(
                nn.Conv1d(
                    hidden_features * 4,
                    hidden_features,
                    kernel_size=1,
                ),
                nn.ReLU(),
            )
            self.beat_encoder = nn.Sequential(
                nn.Conv1d(
                    hidden_features * SUBDIVISIONS,
                    hidden_features,
                    kernel_size=1,
                ),
                nn.ReLU(),
            )
            self.count = nn.Conv1d(
                hidden_features,
                5,
                kernel_size=1,
            )
            self.subdivision = nn.Conv1d(
                hidden_features,
                SUBDIVISIONS,
                kernel_size=1,
            )

        def forward(self, features):
            encoded = self.encoder(
                self.input_projection(features.transpose(1, 2))
            )
            encoded = self.context_fusion(torch.cat(
                [encoded] + [
                    pool(encoded) for pool in self.context_pools
                ],
                dim=1,
            ))
            batch, channels, positions = encoded.shape
            beats = positions // SUBDIVISIONS
            encoded = encoded[:, :, :beats * SUBDIVISIONS]
            encoded = encoded.reshape(
                batch,
                channels,
                beats,
                SUBDIVISIONS,
            ).permute(0, 1, 3, 2).reshape(
                batch,
                channels * SUBDIVISIONS,
                beats,
            )
            encoded = self.beat_encoder(encoded)
            return {
                "count": self.count(encoded).transpose(1, 2),
                "subdivision": self.subdivision(encoded).transpose(1, 2),
            }

    return BeatOccupancyModel()


def _load_records(index_path):
    _, items = _load_feature_index(index_path)
    records = []
    for item in items:
        arrays, metadata = _load_feature_artifact(item["path"])
        labels = json.loads(
            _label_path(item, metadata).read_text(encoding="utf-8")
        )
        beat_times = np.asarray([
            float(row["time"]) for row in labels["beats"]
        ], dtype=np.float64)
        attack_times = np.asarray([
            float(row["start"]) for row in labels["attacks"]
        ], dtype=np.float64)
        occupancy, quantization_errors = _beat_targets(
            beat_times,
            attack_times,
        )
        grid_times = _grid_times(beat_times)
        grid_times = grid_times[:len(occupancy) * SUBDIVISIONS]
        records.append({
            "song_id": item["song_id"],
            "split": item["split"],
            "features": _sample_features(arrays, grid_times),
            "occupancy": occupancy,
            "counts": np.minimum(
                4,
                occupancy.sum(axis=1).astype(np.int64),
            ),
            "beat_times": beat_times,
            "attack_times": attack_times[
                (attack_times >= beat_times[0])
                & (attack_times < beat_times[-1])
            ],
            "quantization_errors": quantization_errors,
        })
    return records


def _predict(model, records, device):
    import torch

    model.eval()
    predictions = []
    with torch.no_grad():
        for record in records:
            outputs = model(
                torch.from_numpy(record["features"]).unsqueeze(0).to(device)
            )
            predictions.append({
                **record,
                "count_probabilities": torch.softmax(
                    outputs["count"][0],
                    dim=-1,
                ).cpu().numpy(),
                "subdivision_scores": torch.sigmoid(
                    outputs["subdivision"][0]
                ).cpu().numpy(),
            })
    return predictions


def run(index_path, output, seed=0, epochs=20, device="auto"):
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

    index_path = Path(index_path).resolve()
    records = _load_records(index_path)
    train = [record for record in records if record["split"] == "train"]
    validation = [
        record for record in records if record["split"] == "validation"
    ]
    test = [record for record in records if record["split"] == "test"]
    model = _build_model(train[0]["features"].shape[1]).to(device)
    count_values = np.concatenate([
        record["counts"] for record in train
    ])
    count_frequencies = np.bincount(count_values, minlength=5).astype(float)
    count_weights = (
        count_frequencies.sum()
        / np.maximum(1.0, count_frequencies)
    )
    count_weights /= np.mean(count_weights[count_frequencies > 0])
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
            outputs = model(
                torch.from_numpy(record["features"]).unsqueeze(0).to(device)
            )
            counts = torch.from_numpy(
                record["counts"]
            ).unsqueeze(0).to(device)
            occupancy = torch.from_numpy(
                record["occupancy"]
            ).unsqueeze(0).to(device)
            count_loss = functional.cross_entropy(
                outputs["count"].reshape(-1, 5),
                counts.reshape(-1),
                weight=torch.tensor(
                    count_weights,
                    dtype=torch.float32,
                    device=device,
                ),
            )
            subdivision_loss = functional.binary_cross_entropy_with_logits(
                outputs["subdivision"],
                occupancy,
            )
            loss = count_loss + subdivision_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
        epoch_losses.append(total / len(train))

    validation_predictions = _predict(model, validation, device)
    candidates = [
        {
            **_metrics(validation_predictions, count_bias),
            "count_bias": float(count_bias),
        }
        for count_bias in np.linspace(0.0, 1.0, 41)
    ]
    validation_metrics = max(
        candidates,
        key=lambda row: (
            row["macro_f1"],
            row["macro_precision"],
            row["count_bias"],
        ),
    )
    baseline = {
        "macro_f1": 0.759665373617493,
        "macro_recall": 0.718132099284813,
        "macro_median_timing_error_ms": 14.9271137026239,
        "macro_p95_timing_error_ms": 38.14706835115,
    }
    guard_passed = bool(
        validation_metrics["macro_f1"] >= baseline["macro_f1"]
        and validation_metrics["macro_recall"] >= baseline["macro_recall"]
        and validation_metrics["macro_median_timing_error_ms"]
        < baseline["macro_median_timing_error_ms"]
        and validation_metrics["macro_p95_timing_error_ms"]
        <= baseline["macro_p95_timing_error_ms"]
    )
    report = {
        "schema": "rb3charts-structure-beat-occupancy",
        "schema_version": 1,
        "scope": "offline authorized-data evaluation",
        "seed": seed,
        "epochs": epochs,
        "device": device,
        "representation": {
            "subdivisions_per_beat": SUBDIVISIONS,
            "maximum_attacks_per_beat": 4,
            "heads": ["beat-attack-count", "subdivision-ranking"],
        },
        "epoch_losses": epoch_losses,
        "count_class_weights": count_weights.tolist(),
        "feature_index_sha256": _hash_file(index_path),
        "validation_baseline": baseline,
        "validation": validation_metrics,
        "guard_passed": guard_passed,
        "test_evaluated": guard_passed and bool(test),
    }
    if guard_passed and test:
        report["test"] = _metrics(
            _predict(model, test, device),
            validation_metrics["count_bias"],
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
        "configuration": report["representation"],
        "training_report": report,
    }, output)
    save_json(output.with_suffix(".json"), report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
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
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
