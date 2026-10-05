"""Train attack selection directly on a beat-synchronous grid."""

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
)


SUBDIVISIONS = 12
MATCH_TOLERANCE_SECONDS = 0.05


def _grid_times(
    beat_times,
    subdivisions=SUBDIVISIONS,
    start_time=None,
    end_time=None,
):
    beat_times = np.asarray(beat_times, dtype=np.float64)
    if len(beat_times) < 2 or np.any(np.diff(beat_times) <= 0):
        raise ValueError("Beat grid requires at least two increasing beats")
    start_position = 0.0
    end_position = float(len(beat_times) - 1)
    if start_time is not None and start_time < beat_times[0]:
        start_position = np.floor(
            (start_time - beat_times[0])
            / (beat_times[1] - beat_times[0])
            * subdivisions
        ) / subdivisions
    if end_time is not None and end_time > beat_times[-1]:
        end_position += np.ceil(
            (end_time - beat_times[-1])
            / (beat_times[-1] - beat_times[-2])
            * subdivisions
        ) / subdivisions
    positions = np.arange(
        round(start_position * subdivisions),
        round(end_position * subdivisions) + 1,
        dtype=np.float64,
    ) / subdivisions
    times = np.interp(
        positions,
        np.arange(len(beat_times), dtype=np.float64),
        beat_times,
    )
    before = positions < 0
    times[before] = (
        beat_times[0]
        + positions[before] * (beat_times[1] - beat_times[0])
    )
    last = len(beat_times) - 1
    after = positions > last
    times[after] = (
        beat_times[-1]
        + (positions[after] - last)
        * (beat_times[-1] - beat_times[-2])
    )
    return times


def _grid_targets(grid_times, attack_times):
    grid_times = np.asarray(grid_times, dtype=np.float64)
    targets = np.zeros(len(grid_times), dtype=np.float32)
    errors = []
    indices = []
    for attack in attack_times:
        insertion = int(np.searchsorted(grid_times, attack))
        candidates = [
            index
            for index in (insertion - 1, insertion)
            if 0 <= index < len(grid_times)
        ]
        best = min(candidates, key=lambda index: abs(grid_times[index] - attack))
        if targets[best]:
            raise ValueError(
                f"Multiple attacks map to beat-grid position {best}"
            )
        targets[best] = 1
        errors.append(abs(float(grid_times[best]) - float(attack)))
        indices.append(best)
    return (
        targets,
        np.asarray(errors, dtype=np.float64),
        np.asarray(indices, dtype=np.int64),
    )


def _nearest_frames(frame_count, grid_times):
    frame_times = (
        np.arange(frame_count, dtype=np.float64)
        * FEATURE_HOP_LENGTH
        / FEATURE_SAMPLE_RATE
    )
    frames = np.searchsorted(frame_times, grid_times)
    frames = np.clip(frames, 0, len(frame_times) - 1)
    previous = np.maximum(0, frames - 1)
    use_previous = (
        np.abs(frame_times[previous] - grid_times)
        < np.abs(frame_times[frames] - grid_times)
    )
    frames[use_previous] = previous[use_previous]
    return frames


def _sample_features(arrays, grid_times):
    frames = _nearest_frames(len(arrays["log_mel"]), grid_times)
    positions = np.arange(len(grid_times), dtype=np.float32)
    phase = 2 * np.pi * (positions % SUBDIVISIONS) / SUBDIVISIONS
    return np.concatenate((
        arrays["log_mel"][frames],
        arrays["mel_delta"][frames],
        np.sin(phase)[:, None],
        np.cos(phase)[:, None],
    ), axis=1).astype(np.float32)


def _select_predictions(
    probabilities,
    grid_times,
    threshold,
    minimum_distance=MATCH_TOLERANCE_SECONDS,
):
    candidates = [
        (float(probability), float(grid_times[index]))
        for index, probability in enumerate(probabilities)
        if probability >= threshold
    ]
    selected = []
    for probability, time in sorted(candidates, reverse=True):
        if all(
            abs(time - existing_time) >= minimum_distance
            for _, existing_time in selected
        ):
            selected.append((probability, time))
    return sorted(time for _, time in selected)


def _metrics(records, threshold):
    rows = []
    for record in records:
        predicted = _select_predictions(
            record["probabilities"],
            record["grid_times"],
            threshold,
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
        "threshold": float(threshold),
        "macro_precision": float(np.mean([row["precision"] for row in rows])),
        "macro_recall": float(np.mean([row["recall"] for row in rows])),
        "macro_f1": float(np.mean([row["f1"] for row in rows])),
        "macro_median_timing_error_ms": (
            float(np.mean(medians)) if medians else None
        ),
        "macro_p95_timing_error_ms": (
            float(np.mean(p95s)) if p95s else None
        ),
        "by_song": {row["song_id"]: row for row in rows},
    }


def _build_model(input_features, hidden_features=64):
    import torch
    import torch.nn as nn

    class BeatGridModel(nn.Module):
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
            self.attack = nn.Conv1d(
                hidden_features,
                1,
                kernel_size=1,
            )
            self.polyphony = nn.Conv1d(
                hidden_features,
                7,
                kernel_size=1,
            )
            self.sustain = nn.Conv1d(
                hidden_features,
                7,
                kernel_size=1,
            )

        def forward(self, features):
            encoded = self.local(
                self.input_projection(features.transpose(1, 2))
            )
            encoded = self.context_fusion(torch.cat(
                [encoded] + [
                    pool(encoded) for pool in self.context_pools
                ],
                dim=1,
            ))
            return {
                "attack": self.attack(encoded).squeeze(1),
                "polyphony": self.polyphony(encoded).transpose(1, 2),
                "sustain": self.sustain(encoded).transpose(1, 2),
            }

    return BeatGridModel()


def _label_path(item, metadata):
    feature_path = Path(item["path"])
    path = feature_path.parent.parent / "labels" / metadata["labels_name"]
    if not path.is_file():
        raise ValueError(f"Beat-grid label artifact is missing: {path}")
    if _hash_file(path) != metadata["labels_sha256"]:
        raise ValueError(f"Beat-grid label hash mismatch: {path}")
    return path


def _load_records(index_path):
    _, items = _load_feature_index(index_path)
    records = []
    for item in items:
        arrays, metadata = _load_feature_artifact(item["path"])
        labels = json.loads(
            _label_path(item, metadata).read_text(encoding="utf-8")
        )
        beat_times = [float(row["time"]) for row in labels["beats"]]
        attack_times = np.asarray(
            [float(row["start"]) for row in labels["attacks"]],
            dtype=np.float64,
        )
        grid_times = _grid_times(
            beat_times,
            start_time=float(np.min(attack_times)),
            end_time=float(np.max(attack_times)),
        )
        targets, quantization_errors, target_indices = _grid_targets(
            grid_times,
            attack_times,
        )
        polyphony = np.zeros(len(grid_times), dtype=np.int64)
        polyphony[target_indices] = np.asarray([
            min(6, int(row["polyphony"]))
            for row in labels["attacks"]
        ], dtype=np.int64)
        frames = _nearest_frames(len(arrays["log_mel"]), grid_times)
        records.append({
            "song_id": item["song_id"],
            "split": item["split"],
            "features": _sample_features(arrays, grid_times),
            "frame_features": np.concatenate((
                arrays["log_mel"],
                arrays["mel_delta"],
                arrays["beat_phase"],
            ), axis=1).astype(np.float32),
            "grid_frames": frames,
            "targets": targets,
            "polyphony": polyphony,
            "sustain": arrays["sustain_target"][frames].astype(np.int64),
            "grid_times": grid_times,
            "attack_times": attack_times,
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
                "probabilities": torch.sigmoid(
                    outputs["attack"][0]
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
    positives = sum(float(record["targets"].sum()) for record in train)
    positions = sum(len(record["targets"]) for record in train)
    positive_weight = min(
        50.0,
        max(1.0, (positions - positives) / positives),
    )
    model = _build_model(train[0]["features"].shape[1]).to(device)
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
            targets = torch.from_numpy(
                record["targets"]
            ).unsqueeze(0).to(device)
            outputs = model(features)
            attack_loss = functional.binary_cross_entropy_with_logits(
                outputs["attack"],
                targets,
                pos_weight=torch.tensor(positive_weight, device=device),
            )
            attack_mask = targets > 0.5
            polyphony = torch.from_numpy(
                record["polyphony"]
            ).unsqueeze(0).to(device)
            sustain = torch.from_numpy(
                record["sustain"]
            ).unsqueeze(0).to(device)
            polyphony_loss = functional.cross_entropy(
                outputs["polyphony"][attack_mask],
                polyphony[attack_mask],
            )
            sustain_loss = functional.cross_entropy(
                outputs["sustain"].reshape(-1, 7),
                sustain.reshape(-1),
            )
            loss = (
                attack_loss
                + 0.5 * polyphony_loss
                + 0.25 * sustain_loss
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
    validation_quantization = np.concatenate([
        record["quantization_errors"] for record in validation
    ]) * 1000
    report = {
        "schema": "rb3charts-structure-beat-grid",
        "schema_version": 1,
        "scope": "offline authorized-data evaluation",
        "representation": {
            "subdivisions_per_beat": SUBDIVISIONS,
            "context_grid_positions": [9, 37, 97],
            "source_features": [
                "log-mel",
                "positive-mel-delta",
                "grid-phase",
            ],
        },
        "seed": seed,
        "epochs": epochs,
        "device": device,
        "positive_weight": positive_weight,
        "epoch_losses": epoch_losses,
        "feature_index_sha256": _hash_file(index_path),
        "validation_quantization": {
            "attacks": len(validation_quantization),
            "median_error_ms": float(np.median(validation_quantization)),
            "p95_error_ms": float(np.percentile(
                validation_quantization,
                95,
            )),
            "within_50ms": float(np.mean(
                validation_quantization <= 50
            )),
        },
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
