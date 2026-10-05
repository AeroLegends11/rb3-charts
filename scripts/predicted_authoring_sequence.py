"""Evaluate lane authoring with predicted attacks and polyphony offline."""

import argparse
import json
import math
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
from structure_timing_snap import _hash_file, _load_model
from symbolic_authoring_sequence import _baseline_masks, _lanes
from rb3charts.model import save_json
from rb3charts.training import (
    FEATURE_HOP_LENGTH,
    FEATURE_SAMPLE_RATE,
    _attack_peaks,
    _load_feature_artifact,
    _load_feature_index,
)


def _remove_duration_input(records):
    for record in records:
        record["symbolic_features"][:, 4] = 0.0
        record["features"][:, 256 + 4] = 0.0


def _context_audio_features(arrays, frames):
    rows = []
    for frame in frames:
        summaries = []
        for radius in (4, 16):
            left = max(0, int(frame) - radius)
            right = min(len(arrays["log_mel"]), int(frame) + radius + 1)
            summaries.extend((
                np.mean(arrays["log_mel"][left:right], axis=0),
                np.max(arrays["mel_delta"][left:right], axis=0),
            ))
        rows.append(np.concatenate((
            arrays["log_mel"][frame],
            arrays["mel_delta"][frame],
            *summaries,
        )))
    return np.asarray(rows, dtype=np.float32)


def _augment_reference_audio(records, items):
    paths = {
        item["song_id"]: item["path"]
        for item in items
    }
    for record in records:
        arrays, _ = _load_feature_artifact(paths[record["song_id"]])
        frames = np.clip(
            np.rint(
                np.asarray(record["times"])
                * FEATURE_SAMPLE_RATE
                / FEATURE_HOP_LENGTH
            ).astype(int),
            0,
            len(arrays["log_mel"]) - 1,
        )
        record["features"] = np.concatenate((
            _context_audio_features(arrays, frames),
            record["symbolic_features"],
        ), axis=1)


def _pitch_contour_features(arrays, frames, times):
    mel = arrays["log_mel"][frames]
    shifted = np.clip(mel - np.max(mel, axis=1, keepdims=True), -20, 0)
    weights = np.exp(shifted)
    bins = np.linspace(0.0, 1.0, mel.shape[1], dtype=np.float32)
    centroids = (weights * bins).sum(axis=1) / np.maximum(
        weights.sum(axis=1),
        1e-6,
    )
    peaks = np.argmax(mel, axis=1) / max(1, mel.shape[1] - 1)
    deltas = np.zeros(len(frames), dtype=np.float32)
    phrase_ranks = np.zeros(len(frames), dtype=np.float32)
    starts = [0]
    for index in range(1, len(frames)):
        if float(times[index]) - float(times[index - 1]) >= 1.0:
            starts.append(index)
        else:
            deltas[index] = centroids[index] - centroids[index - 1]
    starts.append(len(frames))
    for left, right in zip(starts, starts[1:]):
        order = np.argsort(centroids[left:right], kind="stable")
        if right - left == 1:
            phrase_ranks[left] = 0.5
        else:
            phrase_ranks[left + order] = np.linspace(
                0.0,
                1.0,
                right - left,
            )
    change = np.mean(np.abs(arrays["mel_delta"][frames]), axis=1)
    return np.column_stack((
        centroids,
        peaks,
        deltas,
        phrase_ranks,
        change,
    )).astype(np.float32)


def _augment_reference_pitch(records, items, relative_only=False):
    paths = {
        item["song_id"]: item["path"]
        for item in items
    }
    for record in records:
        arrays, _ = _load_feature_artifact(paths[record["song_id"]])
        frames = np.clip(
            np.rint(
                np.asarray(record["times"])
                * FEATURE_SAMPLE_RATE
                / FEATURE_HOP_LENGTH
            ).astype(int),
            0,
            len(arrays["log_mel"]) - 1,
        )
        contour = _pitch_contour_features(
            arrays,
            frames,
            record["times"],
        )
        record["features"] = np.concatenate((
            record["features"],
            contour[:, 2:] if relative_only else contour,
        ), axis=1)


def _predicted_record(
    item,
    arrays,
    outputs,
    threshold,
    context_features=False,
    pitch_contour_features=False,
    relative_pitch_only=False,
):
    import torch

    scores = torch.sigmoid(outputs["attack"][0]).cpu().numpy()
    peaks = _attack_peaks(scores, threshold)
    polyphony = torch.argmax(
        outputs["polyphony"][0],
        dim=-1,
    ).cpu().numpy()
    times = peaks * FEATURE_HOP_LENGTH / FEATURE_SAMPLE_RATE
    symbolic = []
    previous = None
    for peak, seconds in zip(peaks, times):
        gap = 2.0 if previous is None else min(2.0, seconds - previous)
        symbolic.append([
            math.log1p(gap * 10),
            float(arrays["beat_phase"][peak, 0]),
            float(arrays["beat_phase"][peak, 1]),
            max(1, min(5, int(polyphony[peak]))) / 5,
            0.0,
            float(gap >= 1.0),
        ])
        previous = seconds
    symbolic = np.asarray(symbolic, dtype=np.float32)
    audio = (
        _context_audio_features(arrays, peaks)
        if context_features
        else np.concatenate((
            arrays["log_mel"][peaks],
            arrays["mel_delta"][peaks],
        ), axis=1)
    )
    contour = (
        _pitch_contour_features(arrays, peaks, times)
        if pitch_contour_features
        else np.empty((len(peaks), 0), dtype=np.float32)
    )
    if relative_pitch_only:
        contour = contour[:, 2:]
    return {
        "song_id": item["song_id"],
        "split": item["split"],
        "times": times.tolist(),
        "symbolic_features": symbolic,
        "features": np.concatenate((audio, symbolic, contour), axis=1),
    }


def _timing_lane_metrics(references, predictions, masks, tolerance_ms=50):
    frame_ms = FEATURE_HOP_LENGTH / FEATURE_SAMPLE_RATE * 1000
    tolerance_frames = max(1, round(tolerance_ms / frame_ms))
    rows = []
    for predicted, predicted_masks in zip(predictions, masks):
        reference = references[predicted["song_id"]]
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
        unmatched = set(range(len(reference_frames)))
        matched = exact = 0
        jaccard = []
        errors = []
        for frame, mask in zip(predicted_frames, predicted_masks):
            possible = [
                index for index in unmatched
                if abs(int(reference_frames[index]) - int(frame))
                <= tolerance_frames
            ]
            if not possible:
                continue
            best = min(
                possible,
                key=lambda index: abs(int(reference_frames[index]) - int(frame)),
            )
            unmatched.remove(best)
            matched += 1
            errors.append(abs(int(reference_frames[best]) - int(frame)) * frame_ms)
            reference_mask = reference["masks"][best]
            exact += int(reference_mask == mask)
            reference_lanes = set(_lanes(reference_mask))
            predicted_lanes = set(_lanes(mask))
            jaccard.append(
                len(reference_lanes & predicted_lanes)
                / len(reference_lanes | predicted_lanes)
            )
        attack_precision = matched / max(1, len(predicted_frames))
        attack_recall = matched / max(1, len(reference_frames))
        exact_precision = exact / max(1, len(predicted_frames))
        exact_recall = exact / max(1, len(reference_frames))
        rows.append({
            "song_id": predicted["song_id"],
            "predicted_attacks": len(predicted_frames),
            "reference_attacks": len(reference_frames),
            "matched_attacks": matched,
            "attack_precision": attack_precision,
            "attack_recall": attack_recall,
            "attack_f1": (
                2 * attack_precision * attack_recall
                / max(1e-12, attack_precision + attack_recall)
            ),
            "matched_exact_lanes": exact / max(1, matched),
            "matched_lane_jaccard": (
                float(np.mean(jaccard)) if jaccard else 0.0
            ),
            "joint_exact_precision": exact_precision,
            "joint_exact_recall": exact_recall,
            "joint_exact_f1": (
                2 * exact_precision * exact_recall
                / max(1e-12, exact_precision + exact_recall)
            ),
            "median_timing_error_ms": (
                float(np.median(errors)) if errors else None
            ),
        })
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in (
            "attack_precision",
            "attack_recall",
            "attack_f1",
            "matched_exact_lanes",
            "matched_lane_jaccard",
            "joint_exact_precision",
            "joint_exact_recall",
            "joint_exact_f1",
        )
    } | {
        "songs": rows,
    }


def run(
    feature_index,
    authoring_index,
    structure_checkpoint,
    output,
    seed=2,
    epochs=20,
    device="auto",
    context_features=False,
    pitch_contour_features=False,
    relative_pitch_only=False,
):
    import torch
    import torch.nn.functional as functional

    if relative_pitch_only and not pitch_contour_features:
        raise ValueError("relative pitch features require pitch contour features")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    records = _load_records(feature_index, authoring_index)
    _remove_duration_input(records)
    _, items = _load_feature_index(feature_index)
    if context_features:
        _augment_reference_audio(records, items)
    if pitch_contour_features:
        _augment_reference_pitch(
            records,
            items,
            relative_only=relative_pitch_only,
        )
    train = [record for record in records if record["split"] == "train"]
    validation = [
        record for record in records if record["split"] == "validation"
    ]
    test = [record for record in records if record["split"] == "test"]
    stacked = np.concatenate([record["features"] for record in train])
    mean = stacked.mean(axis=0)
    scale = np.maximum(stacked.std(axis=0), 1e-5)
    lane_model = _build_model(stacked.shape[1]).to(device)
    class_polyphony = torch.tensor(
        [len(_lanes(mask)) for mask in MASKS],
        device=device,
    )
    optimizer = torch.optim.AdamW(
        lane_model.parameters(),
        lr=5e-4,
        weight_decay=1e-4,
    )
    for _ in range(epochs):
        lane_model.train()
        for record in train:
            features = torch.from_numpy(
                ((record["features"] - mean) / scale)[None, :, :]
            ).to(device)
            targets = torch.from_numpy(record["targets"][None, :]).to(device)
            logits = lane_model(features)
            allowed = (
                class_polyphony[targets].unsqueeze(-1)
                == class_polyphony.reshape(1, 1, -1)
            )
            loss = functional.cross_entropy(
                logits.masked_fill(~allowed, -1e9).reshape(-1, len(MASKS)),
                targets.reshape(-1),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    structure_model, structure_configuration = _load_model(
        structure_checkpoint,
        device,
    )
    predicted = []
    structure_model.eval()
    with torch.no_grad():
        for item in items:
            if item["split"] not in ("validation", "test"):
                continue
            arrays, _ = _load_feature_artifact(item["path"])
            features = np.concatenate((
                arrays["log_mel"],
                arrays["mel_delta"],
                arrays["beat_phase"],
            ), axis=1)
            outputs = structure_model(
                torch.from_numpy(features).unsqueeze(0).to(device)
            )
            predicted.append(_predicted_record(
                item,
                arrays,
                outputs,
                float(structure_configuration["attack_threshold"]),
                context_features=context_features,
                pitch_contour_features=pitch_contour_features,
                relative_pitch_only=relative_pitch_only,
            ))
    validation_predicted = [
        record for record in predicted if record["split"] == "validation"
    ]
    test_predicted = [
        record for record in predicted if record["split"] == "test"
    ]
    references = {
        record["song_id"]: record
        for record in validation + test
    }
    validation_learned = _predict(
        lane_model,
        validation_predicted,
        mean,
        scale,
        device,
    )
    validation_baseline_masks = [
        _baseline_masks({
            **record,
            "features": record["symbolic_features"],
        })
        for record in validation_predicted
    ]
    validation_staged = _staged_predictions(
        validation_predicted,
        validation_learned,
    )
    baseline_metrics = _timing_lane_metrics(
        references,
        validation_predicted,
        validation_baseline_masks,
    )
    staged_metrics = _timing_lane_metrics(
        references,
        validation_predicted,
        validation_staged,
    )
    guard_passed = (
        staged_metrics["joint_exact_f1"]
        > baseline_metrics["joint_exact_f1"]
        and staged_metrics["matched_exact_lanes"]
        > baseline_metrics["matched_exact_lanes"]
        and staged_metrics["matched_lane_jaccard"]
        >= baseline_metrics["matched_lane_jaccard"]
    )
    report = {
        "schema": "rb3charts-predicted-authoring-sequence",
        "schema_version": 1,
        "scope": (
            "offline authorized-data evaluation with predicted attack times "
            "and polyphony; chart lanes are targets only"
        ),
        "seed": seed,
        "epochs": epochs,
        "context_features": context_features,
        "pitch_contour_features": pitch_contour_features,
        "relative_pitch_only": relative_pitch_only,
        "guard_passed": guard_passed,
        "test_evaluated": bool(guard_passed and test_predicted),
        "feature_index_sha256": _hash_file(feature_index),
        "authoring_index_sha256": _hash_file(authoring_index),
        "structure_checkpoint_sha256": _hash_file(structure_checkpoint),
        "validation": {
            "fixed_polyphony_baseline": baseline_metrics,
            "staged_audio_candidate": staged_metrics,
        },
    }
    if guard_passed and test_predicted:
        test_learned = _predict(
            lane_model,
            test_predicted,
            mean,
            scale,
            device,
        )
        report["test"] = {
            "fixed_polyphony_baseline": _timing_lane_metrics(
                references,
                test_predicted,
                [
                    _baseline_masks({
                        **record,
                        "features": record["symbolic_features"],
                    })
                    for record in test_predicted
                ],
            ),
            "staged_audio_candidate": _timing_lane_metrics(
                references,
                test_predicted,
                _staged_predictions(test_predicted, test_learned),
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
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--context-features", action="store_true")
    parser.add_argument("--pitch-contour-features", action="store_true")
    parser.add_argument("--relative-pitch-only", action="store_true")
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
        context_features=args.context_features,
        pitch_contour_features=args.pitch_contour_features,
        relative_pitch_only=args.relative_pitch_only,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
