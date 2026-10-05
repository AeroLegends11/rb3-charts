"""Train an offline sequence authoring pilot on authorized chart pairs."""

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from rb3charts.model import save_json


MASKS = tuple(range(1, 32))
MASK_TO_CLASS = {mask: index for index, mask in enumerate(MASKS)}


def _hash_file(path):
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mask(lanes):
    value = 0
    for lane in lanes:
        value |= 1 << int(lane)
    return value


def _lanes(mask):
    return [lane for lane in range(5) if mask & (1 << lane)]


def _load_records(index_path):
    index = json.loads(index_path.read_text(encoding="utf-8"))
    records = []
    for entry in index["examples"]:
        artifact_path = index_path.parent / entry["artifact"]
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        targets = {
            row["attack_id"]: row
            for row in artifact["expert_targets"]
        }
        features = []
        masks = []
        times = []
        previous_time = None
        for attack in artifact["conditioning"]:
            time = float(attack["time_seconds"])
            gap = 2.0 if previous_time is None else min(2.0, time - previous_time)
            beat = float(attack["beat"])
            features.append([
                math.log1p(gap * 10),
                math.sin(2 * math.pi * beat),
                math.cos(2 * math.pi * beat),
                min(5, int(attack["polyphony"])) / 5,
                min(2.0, float(attack["duration_seconds"])),
                float(gap >= 1.0),
            ])
            masks.append(_mask(targets[attack["attack_id"]]["lanes"]))
            times.append(time)
            previous_time = time
        records.append({
            "song_id": entry["song_id"],
            "split": entry["split"],
            "features": np.asarray(features, dtype=np.float32),
            "targets": np.asarray(
                [MASK_TO_CLASS[mask] for mask in masks],
                dtype=np.int64,
            ),
            "masks": masks,
            "times": times,
        })
    return records


def _baseline_masks(record):
    choices = {
        1: 1 << 2,
        2: (1 << 1) | (1 << 3),
        3: (1 << 0) | (1 << 2) | (1 << 4),
        4: (1 << 0) | (1 << 1) | (1 << 3) | (1 << 4),
        5: 31,
    }
    return [
        choices[max(1, min(5, round(feature[3] * 5)))]
        for feature in record["features"]
    ]


def _metrics(records, predicted):
    rows = []
    for record, masks in zip(records, predicted):
        exact = []
        single_exact = []
        jaccard = []
        chord_exact = []
        direction = []
        for reference, candidate in zip(record["masks"], masks):
            exact.append(reference == candidate)
            reference_lanes = set(_lanes(reference))
            candidate_lanes = set(_lanes(candidate))
            jaccard.append(
                len(reference_lanes & candidate_lanes)
                / len(reference_lanes | candidate_lanes)
            )
            if len(reference_lanes) > 1:
                chord_exact.append(reference == candidate)
            else:
                single_exact.append(reference == candidate)
        for index in range(1, len(record["masks"])):
            if record["times"][index] - record["times"][index - 1] >= 1.0:
                continue
            reference_delta = (
                np.mean(_lanes(record["masks"][index]))
                - np.mean(_lanes(record["masks"][index - 1]))
            )
            candidate_delta = (
                np.mean(_lanes(masks[index]))
                - np.mean(_lanes(masks[index - 1]))
            )
            if reference_delta:
                direction.append(
                    np.sign(reference_delta) == np.sign(candidate_delta)
                )
        rows.append({
            "song_id": record["song_id"],
            "exact_lanes": float(np.mean(exact)),
            "lane_jaccard": float(np.mean(jaccard)),
            "single_exact": (
                float(np.mean(single_exact)) if single_exact else None
            ),
            "chord_exact": (
                float(np.mean(chord_exact)) if chord_exact else None
            ),
            "movement_direction": (
                float(np.mean(direction)) if direction else None
            ),
        })
    chord_values = [
        row["chord_exact"] for row in rows
        if row["chord_exact"] is not None
    ]
    single_values = [
        row["single_exact"] for row in rows
        if row["single_exact"] is not None
    ]
    movement_values = [
        row["movement_direction"] for row in rows
        if row["movement_direction"] is not None
    ]
    return {
        "macro_exact_lanes": float(np.mean([
            row["exact_lanes"] for row in rows
        ])),
        "macro_lane_jaccard": float(np.mean([
            row["lane_jaccard"] for row in rows
        ])),
        "macro_single_exact": (
            float(np.mean(single_values)) if single_values else None
        ),
        "macro_chord_exact": (
            float(np.mean(chord_values)) if chord_values else None
        ),
        "macro_movement_direction": (
            float(np.mean(movement_values)) if movement_values else None
        ),
        "songs": rows,
    }


def _build_model():
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(
                input_size=6,
                hidden_size=32,
                batch_first=True,
                bidirectional=True,
            )
            self.output = nn.Linear(64, len(MASKS))

        def forward(self, features):
            encoded, _ = self.gru(features)
            return self.output(encoded)

    return Model()


def _predict(model, records, device):
    import torch

    predicted = []
    model.eval()
    with torch.no_grad():
        for record in records:
            features = torch.from_numpy(
                record["features"][None, :, :]
            ).to(device)
            classes = torch.argmax(model(features)[0], dim=-1).cpu().tolist()
            predicted.append([MASKS[index] for index in classes])
    return predicted


def run(index_path, output, seed=0, epochs=20, evaluate_test=False):
    import torch
    import torch.nn.functional as functional

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    records = _load_records(index_path)
    train = [row for row in records if row["split"] == "train"]
    validation = [row for row in records if row["split"] == "validation"]
    test = [row for row in records if row["split"] == "test"]
    model = _build_model().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-4,
        weight_decay=1e-4,
    )
    for _ in range(epochs):
        model.train()
        for record in train:
            features = torch.from_numpy(
                record["features"][None, :, :]
            ).to(device)
            targets = torch.from_numpy(record["targets"][None, :]).to(device)
            loss = functional.cross_entropy(
                model(features).reshape(-1, len(MASKS)),
                targets.reshape(-1),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    baseline = _metrics(
        validation,
        [_baseline_masks(record) for record in validation],
    )
    candidate = _metrics(
        validation,
        _predict(model, validation, device),
    )
    report = {
        "schema": "rb3charts-symbolic-authoring-sequence",
        "schema_version": 1,
        "scope": (
            "offline authorized-data experiment; chart lanes are targets only"
        ),
        "seed": seed,
        "epochs": epochs,
        "training_items": len(train),
        "validation_items": len(validation),
        "test_items": len(test),
        "test_evaluated": bool(evaluate_test and test),
        "feature_index_sha256": _hash_file(index_path),
        "validation": {
            "baseline": baseline,
            "candidate": candidate,
        },
    }
    if evaluate_test and test:
        report["test"] = {
            "baseline": _metrics(
                test,
                [_baseline_masks(record) for record in test],
            ),
            "candidate": _metrics(
                test,
                _predict(model, test, device),
            ),
        }
    save_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    report = run(
        args.index.resolve(),
        args.output.resolve(),
        args.seed,
        args.epochs,
        args.evaluate_test,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
