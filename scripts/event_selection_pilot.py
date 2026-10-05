"""Train an offline phrase-context attack selector on authorized charts."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import random

import numpy as np

from rb3charts.datasets import _read_expert_gems
from rb3charts.model import save_json


FEATURE_NAMES = (
    "confidence",
    "onset_strength",
    "attack_strength",
    "evidence_score",
    "relative_level",
    "rms_log",
    "duration",
    "beat_alignment",
    "repeated_phrase",
    "polyphony",
    "previous_gap",
    "next_gap",
    "local_density_2s",
    "local_pitch_span_2s",
)


def build_selector_model():
    import torch.nn as nn

    class Selector(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(
                len(FEATURE_NAMES),
                32,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.attack = nn.Linear(64, 1)
            self.timing = nn.Linear(64, 1)

        def forward(self, features):
            encoded, _ = self.gru(features)
            return (
                self.attack(encoded).squeeze(-1),
                self.timing(encoded).squeeze(-1),
            )

    return Selector()


def _hash_file(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reference_times(manifest_item, manifest_root):
    notes = _read_expert_gems(
        manifest_root / manifest_item["annotations"],
        manifest_item["chart_format"],
        "expert",
    )
    return sorted({
        round(float(note["start"]), 6)
        for note in notes
    })


def _candidate_groups(trace_path):
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    events = [
        row
        for row in payload["events"]
        if row.get("classification") != "derived"
        and "+" not in str(row.get("source_id", ""))
    ]
    events.sort(key=lambda row: (float(row["start"]), float(row["pitch"])))
    groups = []
    for event in events:
        if groups and float(event["start"]) - groups[-1]["time"] <= 0.03:
            groups[-1]["events"].append(event)
        else:
            groups.append({
                "time": float(event["start"]),
                "events": [event],
            })
    for index, group in enumerate(groups):
        rows = group["events"]
        evidence = [row.get("evidence") or {} for row in rows]
        starts = [float(row["start"]) for row in rows]
        pitches = [float(row["pitch"]) for row in rows]
        left = index
        while left and group["time"] - groups[left - 1]["time"] <= 1.0:
            left -= 1
        right = index + 1
        while right < len(groups) and groups[right]["time"] - group["time"] <= 1.0:
            right += 1
        local_pitches = [
            float(row["pitch"])
            for local in groups[left:right]
            for row in local["events"]
        ]
        group["accepted"] = any(bool(row.get("accepted")) for row in rows)
        group["features"] = [
            max(float(row.get("confidence", 0)) for row in rows),
            max(float(row.get("model_onset_strength") or 0) for row in rows),
            max(float(row.get("evidence", {}).get("attack_strength", 0)) for row in rows),
            max(float(row.get("evidence", {}).get("evidence_score", 0)) for row in rows),
            max(float(row.get("evidence", {}).get("relative_level", 0)) for row in rows),
            np.log1p(max(float(item.get("rms", 0)) for item in evidence) * 100),
            max(float(row["end"]) - float(row["start"]) for row in rows),
            max(float(item.get("beat_alignment") or 0) for item in evidence),
            float(any(bool(item.get("repeated_phrase")) for item in evidence)),
            float(len(set(round(pitch) for pitch in pitches))),
            min(2.0, group["time"] - groups[index - 1]["time"]) if index else 2.0,
            min(2.0, groups[index + 1]["time"] - group["time"])
            if index + 1 < len(groups) else 2.0,
            float(right - left),
            max(local_pitches) - min(local_pitches) if local_pitches else 0.0,
        ]
        group["time"] = float(np.median(starts))
    return groups


def _match_targets(groups, reference, tolerance=0.08):
    available = set(range(len(reference)))
    labels = np.zeros(len(groups), dtype=np.float32)
    offsets = np.zeros(len(groups), dtype=np.float32)
    order = sorted(
        range(len(groups)),
        key=lambda index: -max(
            float(row.get("evidence", {}).get("evidence_score", 0))
            for row in groups[index]["events"]
        ),
    )
    for index in order:
        candidates = [
            ref_index
            for ref_index in available
            if abs(reference[ref_index] - groups[index]["time"]) <= tolerance
        ]
        if not candidates:
            continue
        best = min(
            candidates,
            key=lambda ref_index: abs(
                reference[ref_index] - groups[index]["time"]
            ),
        )
        labels[index] = 1.0
        offsets[index] = reference[best] - groups[index]["time"]
        available.remove(best)
    return labels, offsets


def _metrics(records, threshold, use_corrections):
    song_rows = []
    for record in records:
        selected = [
            record["times"][index]
            + (
                float(np.clip(record["corrections"][index], -0.08, 0.08))
                if use_corrections else 0.0
            )
            for index, score in enumerate(record["scores"])
            if score >= threshold
        ]
        available = set(range(len(record["reference"])))
        errors = []
        matched_reference = set()
        for predicted in selected:
            candidates = [
                index
                for index in available
                if abs(record["reference"][index] - predicted) <= 0.08
            ]
            if candidates:
                best = min(
                    candidates,
                    key=lambda index: abs(
                        record["reference"][index] - predicted
                    ),
                )
                errors.append(abs(record["reference"][best] - predicted) * 1000)
                matched_reference.add(best)
                available.remove(best)
        matched = len(errors)
        precision = matched / max(1, len(selected))
        recall = matched / max(1, len(record["reference"]))
        window_recalls = []
        if record["reference"]:
            start = np.floor(min(record["reference"]) / 4.0) * 4.0
            end = max(record["reference"])
            while start <= end:
                indices = [
                    index
                    for index, time in enumerate(record["reference"])
                    if start <= time < start + 4.0
                ]
                if indices:
                    window_recalls.append(
                        sum(index in matched_reference for index in indices)
                        / len(indices)
                    )
                start += 4.0
        song_rows.append({
            "song_id": record["song_id"],
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(1e-12, precision + recall),
            "matched": matched,
            "selected": len(selected),
            "reference": len(record["reference"]),
            "median_timing_error_ms": (
                float(np.median(errors)) if errors else None
            ),
            "weakest_4s_recall": min(window_recalls)
            if window_recalls else None,
        })
    timing = [
        row["median_timing_error_ms"]
        for row in song_rows
        if row["median_timing_error_ms"] is not None
    ]
    weakest = [
        row["weakest_4s_recall"]
        for row in song_rows
        if row["weakest_4s_recall"] is not None
    ]
    return {
        "threshold": float(threshold),
        "macro_precision": float(np.mean([row["precision"] for row in song_rows])),
        "macro_recall": float(np.mean([row["recall"] for row in song_rows])),
        "macro_f1": float(np.mean([row["f1"] for row in song_rows])),
        "macro_median_timing_error_ms": float(np.mean(timing))
        if timing else None,
        "macro_weakest_4s_recall": float(np.mean(weakest))
        if weakest else None,
        "songs": song_rows,
    }


def _load_records(index_path, manifest_path):
    index_root = index_path.parent
    index = json.loads(index_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_items = {
        str(item["song_id"]): item
        for item in manifest["items"]
    }
    records = []
    for entry in index["entries"]:
        song_id = str(entry["identity"]["song_id"])
        item = manifest_items.get(song_id)
        if item is None or entry.get("comparison_status") != "available":
            continue
        report_path = index_root / entry["links"]["json"]
        report = json.loads(report_path.read_text(encoding="utf-8"))
        trace = Path(report["generated_folder"]) / "analysis" / "guitar-event-trace.json"
        if not trace.is_file():
            continue
        groups = _candidate_groups(trace)
        reference = _reference_times(item, manifest_path.parent)
        labels, offsets = _match_targets(groups, reference)
        records.append({
            "song_id": song_id,
            "split": item["split"],
            "chart_author_id": item["chart_author_id"],
            "features": np.asarray(
                [group["features"] for group in groups],
                dtype=np.float32,
            ),
            "times": np.asarray(
                [group["time"] for group in groups],
                dtype=np.float32,
            ),
            "accepted": np.asarray(
                [group["accepted"] for group in groups],
                dtype=bool,
            ),
            "labels": labels,
            "offsets": offsets,
            "reference": reference,
            "trace_sha256": _hash_file(trace),
        })
    return records


def run(index_path, manifest_path, output, seed=0, epochs=20, evaluate_test=False):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    records = _load_records(index_path, manifest_path)
    train = [record for record in records if record["split"] == "train"]
    validation = [
        record for record in records if record["split"] == "validation"
    ]
    test = [record for record in records if record["split"] == "test"]
    if not train or not validation:
        raise ValueError("Pilot requires train and validation records")

    stacked = np.concatenate([record["features"] for record in train])
    mean = stacked.mean(axis=0)
    scale = np.maximum(stacked.std(axis=0), 1e-5)

    model = build_selector_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    positives = sum(float(record["labels"].sum()) for record in train)
    total = sum(len(record["labels"]) for record in train)
    positive_weight = min(20.0, max(1.0, (total - positives) / positives))
    losses = []
    for epoch in range(epochs):
        model.train()
        order = list(range(len(train)))
        random.Random(seed + epoch).shuffle(order)
        epoch_loss = 0.0
        for index in order:
            record = train[index]
            features = torch.from_numpy(
                (record["features"] - mean) / scale
            ).unsqueeze(0).to(device)
            labels = torch.from_numpy(record["labels"]).unsqueeze(0).to(device)
            offsets = torch.from_numpy(record["offsets"]).unsqueeze(0).to(device)
            logits, corrections = model(features)
            near_positive = torch.zeros_like(labels)
            positive_indices = torch.nonzero(labels[0] > 0.5).flatten()
            for positive_index in positive_indices:
                left = max(0, int(positive_index) - 2)
                right = min(labels.shape[1], int(positive_index) + 3)
                near_positive[0, left:right] = 1
            weights = torch.where(
                (labels < 0.5) & (near_positive > 0.5),
                2.0,
                1.0,
            )
            attack_loss = functional.binary_cross_entropy_with_logits(
                logits,
                labels,
                pos_weight=torch.tensor(positive_weight, device=device),
                weight=weights,
            )
            positive_mask = labels > 0.5
            timing_loss = (
                functional.smooth_l1_loss(
                    torch.tanh(corrections[positive_mask]) * 0.08,
                    offsets[positive_mask],
                )
                if torch.any(positive_mask)
                else corrections.sum() * 0
            )
            loss = attack_loss + 2.0 * timing_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
        losses.append(epoch_loss / len(train))

    def score(source):
        model.eval()
        scored = []
        with torch.no_grad():
            for record in source:
                features = torch.from_numpy(
                    (record["features"] - mean) / scale
                ).unsqueeze(0).to(device)
                logits, corrections = model(features)
                scored.append({
                    **record,
                    "scores": torch.sigmoid(logits[0]).cpu().numpy(),
                    "corrections": (
                        torch.tanh(corrections[0]).cpu().numpy() * 0.08
                    ),
                })
        return scored

    baseline_records = [
        {
            **record,
            "scores": record["accepted"].astype(np.float32),
            "corrections": np.zeros(len(record["times"]), dtype=np.float32),
        }
        for record in validation
    ]
    validation_baseline = _metrics(
        baseline_records,
        0.5,
        False,
    )
    validation_scored = score(validation)
    candidates = [
        _metrics(validation_scored, threshold, True)
        for threshold in np.linspace(0.05, 0.95, 37)
    ]
    guarded = [
        row
        for row in candidates
        if (
            row["macro_recall"] >= validation_baseline["macro_recall"]
            and row["macro_weakest_4s_recall"]
            >= validation_baseline["macro_weakest_4s_recall"]
            and row["macro_median_timing_error_ms"]
            <= validation_baseline["macro_median_timing_error_ms"]
        )
    ]
    selection_pool = guarded or candidates
    selected = max(
        selection_pool,
        key=lambda row: (
            row["macro_f1"],
            row["macro_precision"],
            row["threshold"],
        ),
    )
    report = {
        "schema": "rb3charts-event-selection-pilot",
        "schema_version": 1,
        "scope": (
            "offline authorized-data experiment; official/commercial chart "
            "notes are never inputs"
        ),
        "seed": seed,
        "epochs": epochs,
        "device": device,
        "feature_names": FEATURE_NAMES,
        "positive_weight": positive_weight,
        "epoch_losses": losses,
        "records": {
            split: [record["song_id"] for record in records if record["split"] == split]
            for split in ("train", "validation", "test")
        },
        "validation": {
            "guard_passed": bool(guarded),
            "selection_guards": (
                "preserve macro recall and macro weakest-four-second recall; "
                "do not increase macro median timing error"
            ),
            "baseline_current_acceptance": validation_baseline,
            "candidate": selected,
        },
        "test_evaluated": bool(evaluate_test and test),
        "input_hashes": {
            "comparison_index_sha256": _hash_file(index_path),
            "manifest_sha256": _hash_file(manifest_path),
        },
    }
    if evaluate_test and test:
        test_baseline = [
            {
                **record,
                "scores": record["accepted"].astype(np.float32),
                "corrections": np.zeros(
                    len(record["times"]),
                    dtype=np.float32,
                ),
            }
            for record in test
        ]
        report["test"] = {
            "baseline_current_acceptance": _metrics(
                test_baseline,
                0.5,
                False,
            ),
            "candidate": _metrics(
                score(test),
                selected["threshold"],
                True,
            ),
        }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, report)
    torch.save({
        "schema": "rb3charts-event-selection-model",
        "schema_version": 1,
        "state": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "feature_names": FEATURE_NAMES,
        "mean": mean,
        "scale": scale,
        "threshold": selected["threshold"],
        "report": report,
    }, output.with_suffix(".pt"))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("comparison_index", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    report = run(
        args.comparison_index.resolve(),
        args.manifest.resolve(),
        args.output.resolve(),
        seed=args.seed,
        epochs=args.epochs,
        evaluate_test=args.evaluate_test,
    )
    print(json.dumps({
        "seed": report["seed"],
        "validation": report["validation"],
        "test": report.get("test"),
    }, indent=2))


if __name__ == "__main__":
    main()
