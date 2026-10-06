"""Evaluate audio-conditioned five-lane sequence authoring offline."""

import argparse
import json
import os
from pathlib import Path
import random

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from symbolic_authoring_sequence import (
    MASKS,
    _baseline_masks,
    _hash_file,
    _load_records as _load_symbolic_records,
    _metrics,
)
from rb3charts.model import save_json
from rb3charts.training import _load_feature_index


def _load_records(feature_index, authoring_index):
    _, items = _load_feature_index(feature_index)
    feature_paths = {
        item["song_id"]: item["path"]
        for item in items
    }
    records = _load_symbolic_records(authoring_index)
    for record in records:
        record["symbolic_features"] = record["features"]
        path = feature_paths[record["song_id"]]
        with np.load(path, allow_pickle=False) as artifact:
            frame_times = artifact["frame_times"]
            log_mel = artifact["log_mel"]
            mel_delta = artifact["mel_delta"]
        audio_features = []
        for time in record["times"]:
            frame = int(np.argmin(np.abs(frame_times - time)))
            audio_features.append(np.concatenate((
                log_mel[frame],
                mel_delta[frame],
            )))
        record["features"] = np.concatenate(
            (
                np.asarray(audio_features, dtype=np.float32),
                record["features"],
            ),
            axis=1,
        )
    return records


def _baseline(record):
    return _baseline_masks({
        **record,
        "features": record["symbolic_features"],
    })


def _staged_predictions(records, predicted):
    staged = []
    for record, candidate in zip(records, predicted):
        baseline = _baseline(record)
        staged.append([
            candidate_mask
            if max(1, min(5, round(float(symbolic[3]) * 5))) == 1
            else baseline_mask
            for candidate_mask, baseline_mask, symbolic in zip(
                candidate,
                baseline,
                record["symbolic_features"],
            )
        ])
    return staged


def _build_model(input_features, movement_head=False):
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Linear(input_features, 64)
            self.gru = nn.GRU(
                input_size=64,
                hidden_size=64,
                batch_first=True,
                bidirectional=True,
            )
            self.output = nn.Linear(128, len(MASKS))
            self.direction = (
                nn.Linear(128, 3)
                if movement_head else None
            )
            self.activation = nn.ReLU()

        def encode(self, features):
            encoded, _ = self.gru(
                self.activation(self.input(features))
            )
            return encoded

        def forward(self, features):
            encoded = self.encode(features)
            return self.output(encoded)

        def movement_logits(self, encoded):
            if self.direction is None:
                raise ValueError("Model does not have a movement head")
            return self.direction(encoded[:, 1:] - encoded[:, :-1])

    return Model()


def _predict(model, records, mean, scale, device):
    return _predict_with_confidence(
        model,
        records,
        mean,
        scale,
        device,
    )[0]


def _predict_with_confidence(model, records, mean, scale, device):
    import torch

    classes_by_polyphony = {
        polyphony: [
            index for index, mask in enumerate(MASKS)
            if len([lane for lane in range(5) if mask & (1 << lane)])
            == polyphony
        ]
        for polyphony in range(1, 6)
    }
    predicted = []
    confidences = []
    model.eval()
    with torch.no_grad():
        for record in records:
            features = torch.from_numpy(
                ((record["features"] - mean) / scale)[None, :, :]
            ).to(device)
            logits = model(features)[0].cpu().numpy()
            masks = []
            margins = []
            for row, symbolic in zip(logits, record["symbolic_features"]):
                polyphony = max(1, min(5, round(float(symbolic[3]) * 5)))
                allowed = classes_by_polyphony[polyphony]
                ranked = sorted(
                    allowed,
                    key=lambda index: float(row[index]),
                    reverse=True,
                )
                selected = ranked[0]
                masks.append(MASKS[selected])
                margins.append(
                    float(row[ranked[0]] - row[ranked[1]])
                    if len(ranked) > 1
                    else float("inf")
                )
            predicted.append(masks)
            confidences.append(margins)
    return predicted, confidences


def _predict_logit_ensemble(models, records, mean, scale, device):
    import torch

    classes_by_polyphony = {
        polyphony: [
            index for index, mask in enumerate(MASKS)
            if len([lane for lane in range(5) if mask & (1 << lane)])
            == polyphony
        ]
        for polyphony in range(1, 6)
    }
    predicted = []
    confidences = []
    for model in models:
        model.eval()
    with torch.no_grad():
        for record in records:
            features = torch.from_numpy(
                ((record["features"] - mean) / scale)[None, :, :]
            ).to(device)
            logits = np.mean([
                model(features)[0].cpu().numpy()
                for model in models
            ], axis=0)
            masks = []
            margins = []
            for row, symbolic in zip(logits, record["symbolic_features"]):
                polyphony = max(1, min(5, round(float(symbolic[3]) * 5)))
                ranked = sorted(
                    classes_by_polyphony[polyphony],
                    key=lambda index: float(row[index]),
                    reverse=True,
                )
                masks.append(MASKS[ranked[0]])
                margins.append(
                    float(row[ranked[0]] - row[ranked[1]])
                    if len(ranked) > 1
                    else float("inf")
                )
            predicted.append(masks)
            confidences.append(margins)
    return predicted, confidences


def run(
    feature_index,
    authoring_index,
    output,
    seed=0,
    epochs=20,
    evaluate_test=False,
):
    import torch
    import torch.nn.functional as functional

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    records = _load_records(feature_index, authoring_index)
    train = [row for row in records if row["split"] == "train"]
    validation = [row for row in records if row["split"] == "validation"]
    test = [row for row in records if row["split"] == "test"]
    stacked = np.concatenate([row["features"] for row in train])
    mean = stacked.mean(axis=0)
    scale = np.maximum(stacked.std(axis=0), 1e-5)
    model = _build_model(stacked.shape[1]).to(device)
    class_polyphony = torch.tensor(
        [len([lane for lane in range(5) if mask & (1 << lane)]) for mask in MASKS],
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-4,
        weight_decay=1e-4,
    )
    for _ in range(epochs):
        model.train()
        for record in train:
            features = torch.from_numpy(
                ((record["features"] - mean) / scale)[None, :, :]
            ).to(device)
            targets = torch.from_numpy(record["targets"][None, :]).to(device)
            logits = model(features)
            target_polyphony = class_polyphony[targets]
            allowed = (
                target_polyphony.unsqueeze(-1)
                == class_polyphony.reshape(1, 1, -1)
            )
            loss = functional.cross_entropy(
                logits.masked_fill(~allowed, -1e9).reshape(-1, len(MASKS)),
                targets.reshape(-1),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    validation_predictions = _predict(
        model,
        validation,
        mean,
        scale,
        device,
    )
    report = {
        "schema": "rb3charts-audio-authoring-sequence",
        "schema_version": 1,
        "scope": (
            "offline authorized-data experiment; audio features and "
            "lane-independent timing are inputs, chart lanes are targets only"
        ),
        "seed": seed,
        "epochs": epochs,
        "training_items": len(train),
        "validation_items": len(validation),
        "test_items": len(test),
        "test_evaluated": bool(evaluate_test and test),
        "feature_index_sha256": _hash_file(feature_index),
        "authoring_index_sha256": _hash_file(authoring_index),
        "validation": {
            "baseline": _metrics(
                validation,
                [_baseline(record) for record in validation],
            ),
            "candidate": _metrics(
                validation,
                validation_predictions,
            ),
            "staged_candidate": _metrics(
                validation,
                _staged_predictions(validation, validation_predictions),
            ),
        },
    }
    if evaluate_test and test:
        test_predictions = _predict(model, test, mean, scale, device)
        report["test"] = {
            "baseline": _metrics(
                test,
                [_baseline(record) for record in test],
            ),
            "candidate": _metrics(
                test,
                test_predictions,
            ),
            "staged_candidate": _metrics(
                test,
                _staged_predictions(test, test_predictions),
            ),
        }
    save_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("feature_index", type=Path)
    parser.add_argument("authoring_index", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    report = run(
        args.feature_index.resolve(),
        args.authoring_index.resolve(),
        args.output.resolve(),
        args.seed,
        args.epochs,
        args.evaluate_test,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
