"""Train lane ownership directly on predicted attack sequences offline."""

import argparse
import json
import os
from pathlib import Path
import random

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from audio_authoring_sequence import (
    MASKS,
    _build_model,
    _load_records,
    _predict,
    _staged_predictions,
)
from predicted_authoring_sequence import (
    _predicted_record,
    _timing_lane_metrics,
)
from structure_timing_snap import _hash_file, _load_model
from symbolic_authoring_sequence import (
    MASK_TO_CLASS,
    _baseline_masks,
    _lanes,
)
from rb3charts.model import save_json
from rb3charts.training import (
    FEATURE_HOP_LENGTH,
    FEATURE_SAMPLE_RATE,
    _load_feature_artifact,
    _load_feature_index,
)


def _matched_single_targets(reference, predicted, tolerance_ms=50):
    frame_ms = FEATURE_HOP_LENGTH / FEATURE_SAMPLE_RATE * 1000
    tolerance_frames = max(1, round(tolerance_ms / frame_ms))
    reference_frames = np.rint(
        np.asarray(reference["times"])
        * FEATURE_SAMPLE_RATE
        / FEATURE_HOP_LENGTH
    ).astype(int)
    predicted_frames = np.rint(
        np.asarray(predicted["times"])
        * FEATURE_SAMPLE_RATE
        / FEATURE_HOP_LENGTH
    ).astype(int)
    targets = np.full(len(predicted_frames), -1, dtype=np.int64)
    unmatched = set(range(len(reference_frames)))
    pairs = sorted(
        (
            (abs(int(frame) - int(reference_frames[index])), position, index)
            for position, frame in enumerate(predicted_frames)
            for index in unmatched
            if abs(int(frame) - int(reference_frames[index]))
            <= tolerance_frames
        ),
    )
    used_predictions = set()
    for _, position, reference_index in pairs:
        if position in used_predictions or reference_index not in unmatched:
            continue
        used_predictions.add(position)
        unmatched.remove(reference_index)
        predicted_polyphony = max(
            1,
            min(
                5,
                round(float(predicted["symbolic_features"][position, 3]) * 5),
            ),
        )
        reference_mask = reference["masks"][reference_index]
        if predicted_polyphony == 1 and len(_lanes(reference_mask)) == 1:
            targets[position] = MASK_TO_CLASS[reference_mask]
    return targets


def run(
    feature_index,
    authoring_index,
    structure_checkpoint,
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
    references_list = _load_records(feature_index, authoring_index)
    references = {
        record["song_id"]: record
        for record in references_list
    }
    structure_model, structure_configuration = _load_model(
        structure_checkpoint,
        device,
    )
    _, items = _load_feature_index(feature_index)
    predicted = []
    structure_model.eval()
    with torch.no_grad():
        for item in items:
            arrays, _ = _load_feature_artifact(item["path"])
            features = np.concatenate((
                arrays["log_mel"],
                arrays["mel_delta"],
                arrays["beat_phase"],
            ), axis=1)
            outputs = structure_model(
                torch.from_numpy(features).unsqueeze(0).to(device)
            )
            record = _predicted_record(
                item,
                arrays,
                outputs,
                float(structure_configuration["attack_threshold"]),
            )
            record["targets"] = _matched_single_targets(
                references[item["song_id"]],
                record,
            )
            predicted.append(record)
    train = [record for record in predicted if record["split"] == "train"]
    validation = [
        record for record in predicted if record["split"] == "validation"
    ]
    test = [record for record in predicted if record["split"] == "test"]
    stacked = np.concatenate([record["features"] for record in train])
    mean = stacked.mean(axis=0)
    scale = np.maximum(stacked.std(axis=0), 1e-5)
    lane_model = _build_model(stacked.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        lane_model.parameters(),
        lr=5e-4,
        weight_decay=1e-4,
    )
    training_targets = sum(
        int(np.sum(record["targets"] >= 0))
        for record in train
    )
    for _ in range(epochs):
        lane_model.train()
        for record in train:
            valid = record["targets"] >= 0
            if not np.any(valid):
                continue
            features = torch.from_numpy(
                ((record["features"] - mean) / scale)[None, :, :]
            ).to(device)
            targets = torch.from_numpy(record["targets"][valid]).to(device)
            logits = lane_model(features)[0, valid]
            loss = functional.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    validation_learned = _predict(
        lane_model,
        validation,
        mean,
        scale,
        device,
    )
    validation_baseline = _timing_lane_metrics(
        references,
        validation,
        [
            _baseline_masks({
                **record,
                "features": record["symbolic_features"],
            })
            for record in validation
        ],
    )
    validation_staged = _timing_lane_metrics(
        references,
        validation,
        _staged_predictions(validation, validation_learned),
    )
    guard_passed = (
        validation_staged["joint_exact_f1"]
        > validation_baseline["joint_exact_f1"]
        and validation_staged["matched_exact_lanes"]
        > validation_baseline["matched_exact_lanes"]
        and validation_staged["matched_lane_jaccard"]
        >= validation_baseline["matched_lane_jaccard"]
    )
    report = {
        "schema": "rb3charts-predicted-authoring-self-train",
        "schema_version": 1,
        "scope": (
            "offline authorized-data evaluation trained and inferred on "
            "predicted attack/polyphony sequences; chart lanes are targets only"
        ),
        "seed": seed,
        "epochs": epochs,
        "training_single_targets": training_targets,
        "guard_passed": guard_passed,
        "test_evaluated": bool(guard_passed and test),
        "feature_index_sha256": _hash_file(feature_index),
        "authoring_index_sha256": _hash_file(authoring_index),
        "structure_checkpoint_sha256": _hash_file(structure_checkpoint),
        "validation": {
            "fixed_polyphony_baseline": validation_baseline,
            "staged_audio_candidate": validation_staged,
        },
    }
    if guard_passed and test:
        test_learned = _predict(
            lane_model,
            test,
            mean,
            scale,
            device,
        )
        report["test"] = {
            "fixed_polyphony_baseline": _timing_lane_metrics(
                references,
                test,
                [
                    _baseline_masks({
                        **record,
                        "features": record["symbolic_features"],
                    })
                    for record in test
                ],
            ),
            "staged_audio_candidate": _timing_lane_metrics(
                references,
                test,
                _staged_predictions(test, test_learned),
            ),
        }
    save_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("feature_index", type=Path)
    parser.add_argument("authoring_index", type=Path)
    parser.add_argument("structure_checkpoint", type=Path)
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
        args.feature_index.resolve(),
        args.authoring_index.resolve(),
        args.structure_checkpoint.resolve(),
        args.output.resolve(),
        seed=args.seed,
        epochs=args.epochs,
        device=args.device,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
