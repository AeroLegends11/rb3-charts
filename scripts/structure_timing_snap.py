"""Snap context-selected attacks to local-model onset maxima offline."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from rb3charts.model import save_json
from rb3charts.structure_model import build_structure_model
from rb3charts.training import (
    FEATURE_HOP_LENGTH,
    FEATURE_SAMPLE_RATE,
    _attack_peaks,
    _load_feature_artifact,
    _load_feature_index,
)


def _hash_file(path):
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_model(path, device):
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("schema") != "rb3charts-structure-model"
        or checkpoint.get("schema_version") != 2
    ):
        raise ValueError(f"Unsupported structure checkpoint: {path}")
    configuration = checkpoint["configuration"]
    model = build_structure_model(
        configuration["architecture"],
        input_features=int(configuration["input_features"]),
        hidden_features=int(configuration["hidden_features"]),
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    return model, configuration


def _metrics(predicted, actual, tolerance_frames):
    unmatched = set(range(len(actual)))
    errors = []
    for frame in sorted(predicted):
        candidates = [
            index for index in unmatched
            if abs(int(actual[index]) - int(frame)) <= tolerance_frames
        ]
        if not candidates:
            continue
        best = min(
            candidates,
            key=lambda index: abs(int(actual[index]) - int(frame)),
        )
        errors.append(abs(int(actual[best]) - int(frame)))
        unmatched.remove(best)
    matched = len(errors)
    precision = matched / max(1, len(predicted))
    recall = matched / max(1, len(actual))
    frame_ms = FEATURE_HOP_LENGTH / FEATURE_SAMPLE_RATE * 1000
    return {
        "matched": matched,
        "predicted": len(predicted),
        "reference": len(actual),
        "precision": precision,
        "recall": recall,
        "f1": (
            2 * precision * recall / max(1e-12, precision + recall)
        ),
        "median_timing_error_ms": (
            float(np.median(errors) * frame_ms) if errors else None
        ),
        "p95_timing_error_ms": (
            float(np.percentile(errors, 95) * frame_ms) if errors else None
        ),
    }


def _snap(peaks, local_scores, radius):
    snapped = []
    for peak in peaks:
        left = max(0, int(peak) - radius)
        right = min(len(local_scores), int(peak) + radius + 1)
        snapped.append(
            left + int(np.argmax(local_scores[left:right]))
        )
    return sorted(set(snapped))


def _summarize(records, radius, tolerance_frames):
    songs = {}
    for record in records:
        predicted = _snap(
            record["context_peaks"],
            record["local_scores"],
            radius,
        )
        songs[record["song_id"]] = _metrics(
            predicted,
            record["actual"],
            tolerance_frames,
        )
    timing = [
        row["median_timing_error_ms"]
        for row in songs.values()
        if row["median_timing_error_ms"] is not None
    ]
    return {
        "snap_radius_frames": radius,
        "snap_radius_ms": radius * FEATURE_HOP_LENGTH
        / FEATURE_SAMPLE_RATE * 1000,
        "macro_f1": float(np.mean([
            row["f1"] for row in songs.values()
        ])),
        "macro_precision": float(np.mean([
            row["precision"] for row in songs.values()
        ])),
        "macro_recall": float(np.mean([
            row["recall"] for row in songs.values()
        ])),
        "macro_median_timing_error_ms": (
            float(np.mean(timing)) if timing else None
        ),
        "songs": songs,
    }


def run(index_path, context_path, local_path, output, device="auto"):
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    context_model, context_configuration = _load_model(
        context_path,
        device,
    )
    local_model, _ = _load_model(local_path, device)
    _, items = _load_feature_index(index_path)
    records = []
    with torch.no_grad():
        for item in items:
            if item["split"] not in ("validation", "test"):
                continue
            arrays, _ = _load_feature_artifact(item["path"])
            features = np.concatenate(
                (
                    arrays["log_mel"],
                    arrays["mel_delta"],
                    arrays["beat_phase"],
                ),
                axis=1,
            )
            tensor = torch.from_numpy(features).unsqueeze(0).to(device)
            context_scores = torch.sigmoid(
                context_model(tensor)["attack"][0]
            ).cpu().numpy()
            local_scores = torch.sigmoid(
                local_model(tensor)["attack"][0]
            ).cpu().numpy()
            records.append({
                "song_id": item["song_id"],
                "split": item["split"],
                "context_peaks": _attack_peaks(
                    context_scores,
                    float(context_configuration["attack_threshold"]),
                ),
                "local_scores": local_scores,
                "actual": np.flatnonzero(arrays["attack_target"] > 0),
            })
    tolerance_frames = max(
        1,
        round(0.05 * FEATURE_SAMPLE_RATE / FEATURE_HOP_LENGTH),
    )
    validation = [
        record for record in records if record["split"] == "validation"
    ]
    candidates = [
        _summarize(validation, radius, tolerance_frames)
        for radius in range(5)
    ]
    baseline = candidates[0]
    eligible = [
        row for row in candidates
        if row["macro_f1"] >= baseline["macro_f1"]
    ]
    selected = min(
        eligible,
        key=lambda row: (
            row["macro_median_timing_error_ms"],
            -row["macro_f1"],
            row["snap_radius_frames"],
        ),
    )
    test = [record for record in records if record["split"] == "test"]
    report = {
        "schema": "rb3charts-structure-timing-snap",
        "schema_version": 1,
        "scope": "offline authorized-data evaluation",
        "selection_rule": (
            "preserve validation macro F1, then minimize macro median timing "
            "error; test uses the frozen selected radius"
        ),
        "context_model_sha256": _hash_file(context_path),
        "local_model_sha256": _hash_file(local_path),
        "feature_index_sha256": _hash_file(index_path),
        "validation_candidates": candidates,
        "validation_baseline": baseline,
        "validation_selected": selected,
        "test_baseline": _summarize(test, 0, tolerance_frames),
        "test_selected": _summarize(
            test,
            selected["snap_radius_frames"],
            tolerance_frames,
        ),
    }
    save_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path)
    parser.add_argument("context_model", type=Path)
    parser.add_argument("local_model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    report = run(
        args.index.resolve(),
        args.context_model.resolve(),
        args.local_model.resolve(),
        args.output.resolve(),
        args.device,
    )
    print(json.dumps({
        key: {
            "radius_frames": report[key]["snap_radius_frames"],
            "macro_f1": report[key]["macro_f1"],
            "macro_median_timing_error_ms": report[key][
                "macro_median_timing_error_ms"
            ],
        }
        for key in (
            "validation_baseline",
            "validation_selected",
            "test_baseline",
            "test_selected",
        )
    }, indent=2))


if __name__ == "__main__":
    main()
