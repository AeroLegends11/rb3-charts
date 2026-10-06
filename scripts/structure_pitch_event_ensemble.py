"""Blend independent Basic Pitch event evidence into selected attack scores."""

import argparse
import json
from pathlib import Path

import numpy as np

from structure_density_calibration import _audio_path
from structure_distill_ensemble import (
    _load_student,
    _short_primary_features,
)
from structure_score_ensemble import _summarize
from structure_timing_snap import _hash_file, _load_model
from rb3charts.model import save_json
from rb3charts.training import (
    FEATURE_HOP_LENGTH,
    FEATURE_SAMPLE_RATE,
    _load_feature_artifact,
    _load_feature_index,
    _structure_input_features,
)


def _existing_event_path(song_id, old_root):
    expanded = old_root / "pitch-events-expanded" / f"{song_id}.json"
    if expanded.exists():
        return expanded
    original = old_root / "pitch-events" / f"{song_id}.json"
    return original if original.exists() else None


def _load_or_extract_events(item, old_root, cache_dir):
    path = _existing_event_path(item["song_id"], old_root)
    if path is None:
        path = cache_dir / f"{item['song_id']}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    from rb3charts.transcribe import pitched

    events = pitched(
        _audio_path(item),
        "guitar",
        0.35,
        lambda message: None,
        source_label=f"authorized-{item['song_id']}",
        refine_articulations=True,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    save_json(path, events)
    return events


def _event_support(frame_count, events):
    support = np.zeros(frame_count, dtype=np.float32)
    for event in events:
        frame = round(
            float(event["start"])
            * FEATURE_SAMPLE_RATE
            / FEATURE_HOP_LENGTH
        )
        score = float(
            event.get("evidence", {}).get(
                "evidence_score",
                event.get("confidence", 0.0),
            )
        )
        for offset in range(-4, 5):
            index = frame + offset
            if 0 <= index < frame_count:
                value = score * np.exp(-0.5 * (offset / 1.5) ** 2)
                support[index] = max(support[index], value)
    return support


def _fuse_scores(base_scores, support, weight):
    base = np.clip(base_scores, 1e-5, 1 - 1e-5)
    logits = np.log(base / (1 - base)) + weight * support
    return 1 / (1 + np.exp(-logits))


def _fused_records(records, weight):
    return [
        {
            **record,
            "context_scores": _fuse_scores(
                record["context_scores"],
                record["event_support"],
                weight,
            ),
        }
        for record in records
    ]


def run(
    index_path,
    teacher_path,
    student_paths,
    old_event_root,
    event_cache,
    output,
    device="auto",
):
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher, _ = _load_model(teacher_path, device)
    students = [
        _load_student(path, device)[0]
        for path in student_paths
    ]
    _, items = _load_feature_index(index_path)
    records = []
    with torch.no_grad():
        for item in items:
            if item["split"] not in ("validation", "test"):
                continue
            arrays, _ = _load_feature_artifact(item["path"])
            teacher_scores = torch.sigmoid(teacher(
                torch.from_numpy(
                    _structure_input_features(arrays)
                ).unsqueeze(0).to(device)
            )["attack"][0]).cpu().numpy()
            short_features = _short_primary_features(
                _audio_path(item),
                arrays,
                short_delta=True,
            )
            student_scores = np.mean([
                torch.sigmoid(student(
                    torch.from_numpy(
                        short_features
                    ).unsqueeze(0).to(device)
                )["attack"][0]).cpu().numpy()
                for student in students
            ], axis=0)
            events = _load_or_extract_events(
                item,
                old_event_root,
                event_cache,
            )
            records.append({
                "song_id": item["song_id"],
                "split": item["split"],
                "context_scores": (
                    0.8 * teacher_scores + 0.2 * student_scores
                ),
                "local_scores": np.zeros_like(teacher_scores),
                "event_support": _event_support(
                    len(teacher_scores),
                    events,
                ),
                "event_count": len(events),
                "actual": np.flatnonzero(arrays["attack_target"] > 0),
            })
    validation = [
        record for record in records if record["split"] == "validation"
    ]
    test = [record for record in records if record["split"] == "test"]
    tolerance_frames = max(
        1,
        round(0.05 * FEATURE_SAMPLE_RATE / FEATURE_HOP_LENGTH),
    )
    baseline = _summarize(
        validation,
        0.0,
        0.525,
        tolerance_frames,
    )
    candidates = []
    for weight in (0.25, 0.5, 1.0, 2.0, 4.0):
        fused = _fused_records(validation, weight)
        for threshold in np.linspace(0.05, 0.95, 37):
            candidates.append({
                "event_weight": weight,
                **_summarize(
                    fused,
                    0.0,
                    threshold,
                    tolerance_frames,
                ),
            })
    guarded = [
        row for row in candidates
        if (
            row["macro_f1"] > baseline["macro_f1"]
            and row["macro_recall"] >= baseline["macro_recall"]
            and row["macro_median_timing_error_ms"]
            <= baseline["macro_median_timing_error_ms"]
        )
    ]
    selected = max(
        guarded or candidates,
        key=lambda row: (
            row["macro_f1"],
            row["macro_precision"],
            row["macro_recall"],
        ),
    )
    report = {
        "schema": "rb3charts-structure-pitch-event-ensemble",
        "schema_version": 1,
        "scope": (
            "offline authorized-data evaluation; Basic Pitch events are "
            "audio-only evidence and test remains sealed unless validation "
            "passes"
        ),
        "feature_index_sha256": _hash_file(index_path),
        "teacher_sha256": _hash_file(teacher_path),
        "student_sha256": [
            _hash_file(path) for path in student_paths
        ],
        "event_counts": {
            record["song_id"]: record["event_count"]
            for record in records
        },
        "validation_baseline": baseline,
        "validation_selected": selected,
        "guard_passed": bool(guarded),
        "test_evaluated": bool(guarded and test),
    }
    if guarded and test:
        report["test_baseline"] = _summarize(
            test,
            0.0,
            0.525,
            tolerance_frames,
        )
        report["test_selected"] = _summarize(
            _fused_records(test, selected["event_weight"]),
            0.0,
            selected["threshold"],
            tolerance_frames,
        )
    save_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path)
    parser.add_argument("teacher", type=Path)
    parser.add_argument("old_event_root", type=Path)
    parser.add_argument("event_cache", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("students", type=Path, nargs="+")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    args = parser.parse_args()
    report = run(
        args.index.resolve(),
        args.teacher.resolve(),
        [path.resolve() for path in args.students],
        args.old_event_root.resolve(),
        args.event_cache.resolve(),
        args.output.resolve(),
        device=args.device,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
