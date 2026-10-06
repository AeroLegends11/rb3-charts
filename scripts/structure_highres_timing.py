"""Refine selected attack times from high-resolution local audio patches."""

import argparse
import json
import os
from pathlib import Path
import random

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from audio_authoring_sequence import _load_records
from predicted_authoring_sequence import _blend_attack_scores
from structure_distill_ensemble import _load_student
from structure_multires_spectrogram import (
    _audio_path,
    _short_primary_features,
)
from structure_timing_snap import _hash_file, _load_model
from rb3charts.model import save_json
from rb3charts.training import (
    FEATURE_HOP_LENGTH,
    FEATURE_SAMPLE_RATE,
    _attack_peaks,
    _load_feature_artifact,
    _load_feature_index,
    _structure_input_features,
)


HIGHRES_HOP_LENGTH = 64
HIGHRES_MELS = 32
PATCH_RADIUS = 17
MAX_OFFSET_SECONDS = (
    PATCH_RADIUS * HIGHRES_HOP_LENGTH / FEATURE_SAMPLE_RATE
)


def _match_times(reference, predicted, tolerance=0.05):
    reference = np.asarray(reference, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    unmatched = set(range(len(reference)))
    pairs = []
    for predicted_index, predicted_time in enumerate(predicted):
        possible = [
            reference_index
            for reference_index in unmatched
            if abs(reference[reference_index] - predicted_time) <= tolerance
        ]
        if not possible:
            continue
        reference_index = min(
            possible,
            key=lambda index: (
                abs(reference[index] - predicted_time),
                index,
            ),
        )
        unmatched.remove(reference_index)
        pairs.append((predicted_index, reference_index))
    return pairs


def _timing_metrics(records, tolerance=0.05):
    songs = {}
    for record in records:
        pairs = _match_times(
            record["reference"],
            record["predicted"],
            tolerance,
        )
        errors = [
            abs(
                record["reference"][reference_index]
                - record["predicted"][predicted_index]
            ) * 1000
            for predicted_index, reference_index in pairs
        ]
        precision = len(pairs) / max(1, len(record["predicted"]))
        recall = len(pairs) / max(1, len(record["reference"]))
        songs[record["song_id"]] = {
            "precision": precision,
            "recall": recall,
            "f1": (
                2 * precision * recall
                / max(1e-12, precision + recall)
            ),
            "matched": len(pairs),
            "predicted": len(record["predicted"]),
            "reference": len(record["reference"]),
            "median_timing_error_ms": (
                float(np.median(errors)) if errors else None
            ),
            "p95_timing_error_ms": (
                float(np.percentile(errors, 95)) if errors else None
            ),
        }
    timing = [
        row["median_timing_error_ms"]
        for row in songs.values()
        if row["median_timing_error_ms"] is not None
    ]
    p95 = [
        row["p95_timing_error_ms"]
        for row in songs.values()
        if row["p95_timing_error_ms"] is not None
    ]
    return {
        "macro_precision": float(np.mean([
            row["precision"] for row in songs.values()
        ])),
        "macro_recall": float(np.mean([
            row["recall"] for row in songs.values()
        ])),
        "macro_f1": float(np.mean([
            row["f1"] for row in songs.values()
        ])),
        "macro_median_timing_error_ms": (
            float(np.mean(timing)) if timing else None
        ),
        "macro_p95_timing_error_ms": (
            float(np.mean(p95)) if p95 else None
        ),
        "songs": songs,
    }


def _highres_map(audio_path):
    import librosa

    y, _ = librosa.load(
        audio_path,
        sr=FEATURE_SAMPLE_RATE,
        mono=True,
    )
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=FEATURE_SAMPLE_RATE,
        n_fft=256,
        hop_length=HIGHRES_HOP_LENGTH,
        n_mels=HIGHRES_MELS,
        fmin=30,
        fmax=FEATURE_SAMPLE_RATE / 2,
        power=2,
    )
    log_mel = np.clip(
        (librosa.power_to_db(mel, ref=1.0, top_db=80) + 80) / 80,
        0,
        1,
    ).astype(np.float32)
    positive_delta = np.maximum(
        np.diff(log_mel, axis=1, prepend=log_mel[:, :1]),
        0,
    ).astype(np.float32)
    return np.concatenate((log_mel, positive_delta), axis=0)


def _timing_patches(feature_map, times):
    width = 2 * PATCH_RADIUS + 1
    padded = np.pad(
        feature_map,
        ((0, 0), (PATCH_RADIUS, PATCH_RADIUS)),
        mode="edge",
    )
    centers = np.rint(
        np.asarray(times)
        * FEATURE_SAMPLE_RATE
        / HIGHRES_HOP_LENGTH
    ).astype(int)
    centers = np.clip(centers, 0, feature_map.shape[1] - 1)
    return np.stack([
        padded[:, center:center + width]
        for center in centers
    ]).astype(np.float32)


def _build_model():
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv1d(HIGHRES_MELS * 2, 64, 3, padding=1),
        nn.ReLU(),
        nn.Conv1d(64, 64, 3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64 * (2 * PATCH_RADIUS + 1), 128),
        nn.ReLU(),
        nn.Linear(128, 2 * PATCH_RADIUS + 1),
    )


def _selected_records(
    items,
    references,
    teacher,
    teacher_configuration,
    students,
    device,
    distilled_weight,
    attack_threshold,
):
    import torch

    selected = []
    with torch.no_grad():
        for item in items:
            arrays, _ = _load_feature_artifact(item["path"])
            teacher_scores = torch.sigmoid(
                teacher(
                    torch.from_numpy(
                        _structure_input_features(arrays)
                    ).unsqueeze(0).to(device)
                )["attack"][0]
            ).cpu().numpy()
            short = torch.from_numpy(
                _short_primary_features(
                    _audio_path(item),
                    arrays,
                    short_delta=True,
                )
            ).unsqueeze(0).to(device)
            scores = _blend_attack_scores(
                teacher_scores,
                [
                    torch.sigmoid(
                        student(short)["attack"][0]
                    ).cpu().numpy()
                    for student in students
                ],
                distilled_weight,
            )
            peaks = _attack_peaks(scores, attack_threshold)
            selected.append({
                "song_id": item["song_id"],
                "split": item["split"],
                "audio_path": _audio_path(item),
                "predicted": (
                    peaks * FEATURE_HOP_LENGTH / FEATURE_SAMPLE_RATE
                ).astype(float),
                "reference": np.asarray(
                    references[item["song_id"]]["times"],
                    dtype=float,
                ),
            })
    return selected


def _training_examples(records):
    patches = []
    targets = []
    for record in records:
        pairs = _match_times(
            record["reference"],
            record["predicted"],
            MAX_OFFSET_SECONDS,
        )
        if not pairs:
            continue
        predicted_indices = [
            predicted_index
            for predicted_index, _ in pairs
        ]
        selected_times = record["predicted"][predicted_indices]
        patches.append(_timing_patches(
            _highres_map(record["audio_path"]),
            selected_times,
        ))
        residuals = np.asarray([
            record["reference"][reference_index]
            - record["predicted"][predicted_index]
            for predicted_index, reference_index in pairs
        ])
        targets.append(np.clip(
            np.rint(
                residuals
                * FEATURE_SAMPLE_RATE
                / HIGHRES_HOP_LENGTH
            ).astype(int),
            -PATCH_RADIUS,
            PATCH_RADIUS,
        ) + PATCH_RADIUS)
    return np.concatenate(patches), np.concatenate(targets)


def _corrections(model, records, device):
    import torch

    result = {}
    model.eval()
    with torch.no_grad():
        for record in records:
            patches = _timing_patches(
                _highres_map(record["audio_path"]),
                record["predicted"],
            )
            logits = model(torch.from_numpy(patches).to(device))
            classes = torch.argmax(logits, dim=1).cpu().numpy()
            result[record["song_id"]] = (
                (classes - PATCH_RADIUS)
                * HIGHRES_HOP_LENGTH
                / FEATURE_SAMPLE_RATE
            )
    return result


def _apply_corrections(records, corrections, scale):
    return [{
        **record,
        "predicted": np.sort(
            record["predicted"]
            + float(scale) * corrections[record["song_id"]]
        ),
    } for record in records]


def run(
    index_path,
    authoring_index,
    teacher_path,
    student_paths,
    output,
    seed=0,
    epochs=20,
    device="auto",
    distilled_weight=0.2,
    attack_threshold=0.525,
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

    _, items = _load_feature_index(index_path)
    references = {
        record["song_id"]: record
        for record in _load_records(index_path, authoring_index)
    }
    teacher, teacher_configuration = _load_model(teacher_path, device)
    students = [
        _load_student(path, device)[0]
        for path in student_paths
    ]
    records = _selected_records(
        items,
        references,
        teacher,
        teacher_configuration,
        students,
        device,
        distilled_weight,
        attack_threshold,
    )
    train = [record for record in records if record["split"] == "train"]
    validation = [
        record for record in records if record["split"] == "validation"
    ]
    test = [record for record in records if record["split"] == "test"]
    patches, targets = _training_examples(train)
    model = _build_model().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )
    tensor_patches = torch.from_numpy(patches)
    tensor_targets = torch.from_numpy(targets).long()
    epoch_losses = []
    for _ in range(epochs):
        model.train()
        total = 0.0
        for start in range(0, len(patches), 256):
            batch = tensor_patches[start:start + 256].to(device)
            expected = tensor_targets[start:start + 256].to(device)
            loss = functional.cross_entropy(model(batch), expected)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(batch)
        epoch_losses.append(total / len(patches))

    baseline = _timing_metrics(validation)
    validation_corrections = _corrections(model, validation, device)
    candidates = [
        {
            "scale": scale,
            "metrics": _timing_metrics(_apply_corrections(
                validation,
                validation_corrections,
                scale,
            )),
        }
        for scale in (0.25, 0.5, 0.75, 1.0)
    ]
    guarded = [
        row for row in candidates
        if (
            row["metrics"]["macro_f1"] >= baseline["macro_f1"]
            and row["metrics"]["macro_recall"] >= baseline["macro_recall"]
            and row["metrics"]["macro_median_timing_error_ms"]
            < baseline["macro_median_timing_error_ms"]
            and row["metrics"]["macro_p95_timing_error_ms"]
            <= baseline["macro_p95_timing_error_ms"]
        )
    ]
    selected = max(
        guarded or candidates,
        key=lambda row: (
            row["metrics"]["macro_f1"],
            -row["metrics"]["macro_median_timing_error_ms"],
            -row["metrics"]["macro_p95_timing_error_ms"],
        ),
    )
    report = {
        "schema": "rb3charts-structure-highres-timing",
        "schema_version": 1,
        "scope": "offline authorized-data evaluation",
        "selection_rule": (
            "preserve validation macro F1 and recall while strictly improving "
            "median timing and not worsening P95"
        ),
        "seed": seed,
        "epochs": epochs,
        "device": device,
        "distilled_weight": distilled_weight,
        "attack_threshold": attack_threshold,
        "highres_hop_length": HIGHRES_HOP_LENGTH,
        "patch_radius": PATCH_RADIUS,
        "training_examples": len(patches),
        "epoch_losses": epoch_losses,
        "feature_index_sha256": _hash_file(index_path),
        "authoring_index_sha256": _hash_file(authoring_index),
        "teacher_sha256": _hash_file(teacher_path),
        "student_sha256": [_hash_file(path) for path in student_paths],
        "validation_baseline": baseline,
        "validation_candidates": candidates,
        "validation_selected": selected,
        "guard_passed": bool(guarded),
        "test_evaluated": bool(guarded and test),
    }
    if guarded and test:
        test_corrections = _corrections(model, test, device)
        report["test_baseline"] = _timing_metrics(test)
        report["test_selected"] = _timing_metrics(_apply_corrections(
            test,
            test_corrections,
            selected["scale"],
        ))
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
            "highres_hop_length": HIGHRES_HOP_LENGTH,
            "mel_bins": HIGHRES_MELS,
            "patch_radius": PATCH_RADIUS,
            "correction_scale": selected["scale"],
        },
        "training_report": report,
    }, output)
    save_json(output.with_suffix(".json"), report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path)
    parser.add_argument("authoring_index", type=Path)
    parser.add_argument("teacher", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "students",
        type=Path,
        nargs="+",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--distilled-weight", type=float, default=0.2)
    parser.add_argument("--attack-threshold", type=float, default=0.525)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    args = parser.parse_args()
    report = run(
        args.index.resolve(),
        args.authoring_index.resolve(),
        args.teacher.resolve(),
        [path.resolve() for path in args.students],
        args.output.resolve(),
        seed=args.seed,
        epochs=args.epochs,
        device=args.device,
        distilled_weight=args.distilled_weight,
        attack_threshold=args.attack_threshold,
    )
    print(json.dumps({
        "validation_baseline": report["validation_baseline"],
        "validation_selected": report["validation_selected"],
        "guard_passed": report["guard_passed"],
        "test_evaluated": report["test_evaluated"],
        "test_baseline": report.get("test_baseline"),
        "test_selected": report.get("test_selected"),
    }, indent=2))


if __name__ == "__main__":
    main()
