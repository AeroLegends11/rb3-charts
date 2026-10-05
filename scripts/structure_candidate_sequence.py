"""Train an offline sequence ranker over fixed local attack peaks."""

import argparse
import json
import os
from pathlib import Path
import random

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from structure_candidate_classifier import (
    _candidate_features,
    _candidate_targets,
)
from structure_local_peak_rerank import _select_validation_candidate
from structure_score_ensemble import _summarize
from structure_timing_snap import _hash_file, _load_model
from rb3charts.model import save_json
from rb3charts.training import (
    FEATURE_HOP_LENGTH,
    FEATURE_SAMPLE_RATE,
    _attack_peaks,
    _load_feature_artifact,
    _load_feature_index,
)


def _build_model(input_features):
    import torch.nn as nn

    class CandidateSequence(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Linear(input_features, 64)
            self.encoder = nn.GRU(
                input_size=64,
                hidden_size=64,
                batch_first=True,
                bidirectional=True,
            )
            self.output = nn.Linear(128, 1)
            self.activation = nn.ReLU()

        def forward(self, features):
            encoded, _ = self.encoder(self.activation(self.input(features)))
            return self.output(encoded).squeeze(-1)

    return CandidateSequence()


def _prediction_records(model, records, mean, scale, device):
    import torch

    model.eval()
    predicted = []
    with torch.no_grad():
        for record in records:
            features = torch.from_numpy(
                ((record["features"] - mean) / scale)[None, :, :]
            ).to(device)
            probabilities = torch.sigmoid(model(features)[0]).cpu().numpy()
            scores = np.zeros(len(record["local_scores"]), dtype=np.float32)
            scores[record["peaks"]] = probabilities
            predicted.append({
                **record,
                "local_scores": scores,
                "context_scores": np.zeros_like(scores),
            })
    return predicted


def run(
    index_path,
    context_path,
    local_path,
    output,
    seed=0,
    epochs=20,
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
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    context_model, context_configuration = _load_model(context_path, device)
    local_model, local_configuration = _load_model(local_path, device)
    _, items = _load_feature_index(index_path)
    tolerance_frames = max(
        1,
        round(0.05 * FEATURE_SAMPLE_RATE / FEATURE_HOP_LENGTH),
    )
    records = []
    with torch.no_grad():
        for item in items:
            arrays, _ = _load_feature_artifact(item["path"])
            raw = np.concatenate((
                arrays["log_mel"],
                arrays["mel_delta"],
                arrays["beat_phase"],
            ), axis=1)
            tensor = torch.from_numpy(raw).unsqueeze(0).to(device)
            context_scores = torch.sigmoid(
                context_model(tensor)["attack"][0]
            ).cpu().numpy()
            local_scores = torch.sigmoid(
                local_model(tensor)["attack"][0]
            ).cpu().numpy()
            peaks = _attack_peaks(local_scores)
            actual = np.flatnonzero(arrays["attack_target"] > 0)
            candidate_features = _candidate_features(
                arrays,
                local_scores,
                context_scores,
                peaks,
            )
            gaps = np.diff(peaks, prepend=peaks[0]).astype(np.float32)
            gap_features = np.column_stack((
                np.log1p(gaps),
                gaps >= round(
                    FEATURE_SAMPLE_RATE / FEATURE_HOP_LENGTH
                ),
            )).astype(np.float32)
            records.append({
                "song_id": item["song_id"],
                "split": item["split"],
                "features": np.concatenate((
                    candidate_features,
                    gap_features,
                ), axis=1),
                "targets": _candidate_targets(
                    peaks,
                    actual,
                    tolerance_frames,
                ),
                "peaks": peaks,
                "actual": actual,
                "local_scores": local_scores,
                "context_scores": context_scores,
            })
    train = [record for record in records if record["split"] == "train"]
    validation = [
        record for record in records if record["split"] == "validation"
    ]
    test = [record for record in records if record["split"] == "test"]
    stacked = np.concatenate([record["features"] for record in train])
    mean = stacked.mean(axis=0)
    scale = np.maximum(stacked.std(axis=0), 1e-5)
    targets = np.concatenate([record["targets"] for record in train])
    positive = max(1, int(targets.sum()))
    pos_weight = torch.tensor(
        [(len(targets) - positive) / positive],
        device=device,
    )
    model = _build_model(stacked.shape[1]).to(device)
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
            target = torch.from_numpy(record["targets"][None, :]).to(device)
            loss = functional.binary_cross_entropy_with_logits(
                model(features),
                target,
                pos_weight=pos_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    validation_predictions = _prediction_records(
        model,
        validation,
        mean,
        scale,
        device,
    )
    context_baseline = _summarize(
        validation,
        0.0,
        float(context_configuration["attack_threshold"]),
        tolerance_frames,
    )
    local_baseline = _summarize(
        validation,
        1.0,
        float(local_configuration["attack_threshold"]),
        tolerance_frames,
    )
    candidates = []
    for threshold in np.linspace(0.05, 0.95, 37):
        row = _summarize(
            validation_predictions,
            1.0,
            threshold,
            tolerance_frames,
        )
        row["candidate_local_weight"] = 1.0
        row["context_radius_frames"] = 0
        candidates.append(row)
    selected, guard_passed = _select_validation_candidate(
        candidates,
        context_baseline,
        local_baseline,
    )
    report = {
        "schema": "rb3charts-structure-candidate-sequence",
        "schema_version": 1,
        "scope": "offline authorized-data evaluation",
        "seed": seed,
        "epochs": epochs,
        "guard_passed": guard_passed,
        "test_evaluated": guard_passed,
        "feature_index_sha256": _hash_file(index_path),
        "context_model_sha256": _hash_file(context_path),
        "local_model_sha256": _hash_file(local_path),
        "train_candidates": int(len(targets)),
        "train_positive_candidates": int(targets.sum()),
        "validation_context_baseline": context_baseline,
        "validation_local_baseline": local_baseline,
        "validation_selected": selected,
    }
    if guard_passed:
        test_predictions = _prediction_records(
            model,
            test,
            mean,
            scale,
            device,
        )
        report["test_context_baseline"] = _summarize(
            test,
            0.0,
            float(context_configuration["attack_threshold"]),
            tolerance_frames,
        )
        report["test_local_baseline"] = _summarize(
            test,
            1.0,
            float(local_configuration["attack_threshold"]),
            tolerance_frames,
        )
        report["test_selected"] = _summarize(
            test_predictions,
            1.0,
            selected["threshold"],
            tolerance_frames,
        )
    save_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path)
    parser.add_argument("context_model", type=Path)
    parser.add_argument("local_model", type=Path)
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
        args.index.resolve(),
        args.context_model.resolve(),
        args.local_model.resolve(),
        args.output.resolve(),
        seed=args.seed,
        epochs=args.epochs,
        device=args.device,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
