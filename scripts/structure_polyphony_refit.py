"""Refit only the polyphony head on all non-test authorized songs."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from structure_timing_snap import _hash_file, _load_model
from rb3charts.model import save_json
from rb3charts.training import (
    _load_feature_artifact,
    _load_feature_index,
    _structure_input_features,
)


def _polyphony_metrics(model, records, device):
    import torch

    correct = total = 0
    confusion = Counter()
    model.eval()
    with torch.no_grad():
        for record in records:
            features = torch.from_numpy(
                record["features"]
            ).unsqueeze(0).to(device)
            logits = model(features)["polyphony"][0]
            attack = torch.from_numpy(record["attack"]).to(device) > 0.5
            predicted = torch.argmax(logits[attack], dim=1).cpu().numpy()
            expected = record["polyphony"][record["attack"] > 0.5]
            correct += int(np.sum(predicted == expected))
            total += len(expected)
            confusion.update(
                (int(left), int(right))
                for left, right in zip(expected, predicted)
            )
    return {
        "accuracy": correct / max(1, total),
        "correct": correct,
        "total": total,
        "confusion": {
            f"{expected}->{predicted}": count
            for (expected, predicted), count in sorted(confusion.items())
        },
    }


def run(
    index_path,
    teacher_path,
    output,
    seed=0,
    epochs=5,
    device="auto",
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

    model, configuration = _load_model(teacher_path, device)
    _, items = _load_feature_index(index_path)
    records = []
    for item in items:
        arrays, _ = _load_feature_artifact(item["path"])
        records.append({
            "song_id": item["song_id"],
            "split": item["split"],
            "features": _structure_input_features(arrays),
            "attack": arrays["attack_target"].astype(np.float32),
            "polyphony": arrays["polyphony_target"].astype(np.int64),
        })
    train = [record for record in records if record["split"] != "test"]
    test = [record for record in records if record["split"] == "test"]
    baseline = _polyphony_metrics(model, test, device)

    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.polyphony.parameters():
        parameter.requires_grad = True
    counts = Counter(
        int(value)
        for record in train
        for value in record["polyphony"][record["attack"] > 0.5]
    )
    class_weights = np.ones(7, dtype=np.float32)
    represented = [count for key, count in counts.items() if key > 0]
    maximum = max(represented)
    for polyphony, count in counts.items():
        if polyphony > 0:
            class_weights[polyphony] = np.sqrt(maximum / count)
    weights = torch.from_numpy(class_weights).to(device)
    optimizer = torch.optim.AdamW(
        model.polyphony.parameters(),
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
            attack = torch.from_numpy(
                record["attack"]
            ).unsqueeze(0).to(device) > 0.5
            expected = torch.from_numpy(
                record["polyphony"]
            ).unsqueeze(0).to(device)
            logits = model(features)["polyphony"]
            loss = functional.cross_entropy(
                logits[attack],
                expected[attack],
                weight=weights,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
        epoch_losses.append(total / len(train))
    candidate = _polyphony_metrics(model, test, device)
    report = {
        "schema": "rb3charts-structure-polyphony-refit",
        "schema_version": 1,
        "scope": (
            "fixed final-fit experiment using all non-test authorized songs; "
            "attack model frozen and Persephone retained as author-disjoint test"
        ),
        "seed": seed,
        "epochs": epochs,
        "device": device,
        "training_items": len(train),
        "test_items": len(test),
        "training_polyphony_counts": {
            str(key): value for key, value in sorted(counts.items())
        },
        "class_weights": class_weights.tolist(),
        "epoch_losses": epoch_losses,
        "test_baseline": baseline,
        "test_candidate": candidate,
        "guard_passed": candidate["accuracy"] > baseline["accuracy"],
        "feature_index_sha256": _hash_file(index_path),
        "teacher_sha256": _hash_file(teacher_path),
    }
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": "rb3charts-structure-model",
        "schema_version": 2,
        "model_state": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "configuration": configuration,
        "polyphony_refit_report": report,
    }, output)
    save_json(output.with_suffix(".json"), report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path)
    parser.add_argument("teacher", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    args = parser.parse_args()
    report = run(
        args.index.resolve(),
        args.teacher.resolve(),
        args.output.resolve(),
        seed=args.seed,
        epochs=args.epochs,
        device=args.device,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
