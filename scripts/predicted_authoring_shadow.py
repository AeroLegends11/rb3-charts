"""Evaluate an authorized-data authoring model on an evaluation-only song."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from audio_authoring_sequence import _load_records, _predict
from predicted_authoring_sequence import (
    _augment_reference_pitch,
    _blend_attack_scores,
    _predicted_record,
    _remove_duration_input,
    _supported_staged_predictions,
    _train_lane_models,
)
from structure_distill_ensemble import _load_student
from structure_multires_spectrogram import _short_primary_features
from structure_timing_snap import _hash_file, _load_model
from symbolic_authoring_sequence import _lanes
from rb3charts.datasets import _read_expert_gems
from rb3charts.model import save_json
from rb3charts.regression import compare_chart_attacks
from rb3charts.training import (
    _attack_peaks,
    _extract_audio_features,
    _load_feature_index,
)


def _group_gems(path):
    grouped = {}
    for gem in _read_expert_gems(path, "rock_band_midi", "expert"):
        key = round(float(gem["start"]), 6)
        row = grouped.setdefault(key, {
            "time": key,
            "duration": 0.0,
            "lanes": [],
        })
        row["duration"] = max(
            row["duration"],
            float(gem["end"]) - float(gem["start"]),
        )
        row["lanes"].append(int(gem["lane"]))
    return [
        {**row, "lanes": sorted(set(row["lanes"]))}
        for row in grouped.values()
    ]


def _predicted_attacks(record, masks, offset):
    return [
        {
            "time": float(time) + float(offset),
            "duration": 0.1,
            "lanes": _lanes(mask),
        }
        for time, mask in zip(record["times"], masks)
        if float(time) + float(offset) >= 0
    ]


def _audio_arrays(audio_path, tempo_path):
    tempo = json.loads(Path(tempo_path).read_text(encoding="utf-8"))
    audio = _extract_audio_features(audio_path, tempo["beats"])
    frames = len(audio["frame_times"])
    return {
        "log_mel": audio["log_mel"],
        "mel_delta": audio["mel_delta"],
        "frame_times": audio["frame_times"],
        "beat_phase": audio["beat_phase"],
        "attack_target": np.zeros(frames, dtype=np.float32),
        "polyphony_target": np.zeros(frames, dtype=np.int8),
        "sustain_target": np.zeros(frames, dtype=np.int8),
    }


def _metrics(reference, attacks):
    return {
        str(tolerance_ms): compare_chart_attacks(
            reference,
            attacks,
            tolerance=tolerance_ms / 1000,
        )
        for tolerance_ms in (25, 50)
    }


def run(
    feature_index,
    authoring_index,
    structure_checkpoint,
    student_paths,
    audio_path,
    tempo_path,
    reference_path,
    output,
    offset,
    seed=2,
    epochs=20,
    distilled_weight=0.2,
    attack_threshold=0.525,
    generated_path=None,
    generated_audio_lead=0.0,
    device="auto",
):
    import torch

    torch.use_deterministic_algorithms(True)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    records = _load_records(feature_index, authoring_index)
    _remove_duration_input(records)
    _, items = _load_feature_index(feature_index)
    _augment_reference_pitch(records, items)
    lane_models, mean, scale = _train_lane_models(
        records,
        seed,
        epochs,
        device,
        lane_seeds=[seed],
    )
    lane_model = lane_models[0]

    arrays = _audio_arrays(audio_path, tempo_path)
    teacher, teacher_configuration = _load_model(
        structure_checkpoint,
        device,
    )
    students = [
        _load_student(path, device)[0]
        for path in student_paths
    ]
    features = np.concatenate((
        arrays["log_mel"],
        arrays["mel_delta"],
        arrays["beat_phase"],
    ), axis=1)
    with torch.no_grad():
        outputs = teacher(
            torch.from_numpy(features).unsqueeze(0).to(device)
        )
        teacher_scores = torch.sigmoid(
            outputs["attack"][0]
        ).cpu().numpy()
        short = torch.from_numpy(
            _short_primary_features(
                audio_path,
                arrays,
                short_delta=True,
            )
        ).unsqueeze(0).to(device)
        blended_scores = _blend_attack_scores(
            teacher_scores,
            [
                torch.sigmoid(student(short)["attack"][0]).cpu().numpy()
                for student in students
            ],
            distilled_weight,
        )

    teacher_threshold = float(teacher_configuration["attack_threshold"])
    teacher_peaks = _attack_peaks(teacher_scores, teacher_threshold)
    blended_peaks = _attack_peaks(blended_scores, attack_threshold)
    item = {"song_id": "evaluation-shadow", "split": "shadow"}
    teacher_record = _predicted_record(
        item,
        arrays,
        outputs,
        teacher_threshold,
        pitch_contour_features=True,
        attack_scores=teacher_scores,
    )
    blended_record = _predicted_record(
        item,
        arrays,
        outputs,
        attack_threshold,
        pitch_contour_features=True,
        attack_scores=blended_scores,
        learned_lane_allowed=[
            bool(np.any(np.abs(teacher_peaks - peak) <= 1))
            for peak in blended_peaks
        ],
    )
    teacher_masks = _supported_staged_predictions(
        [teacher_record],
        _predict(lane_model, [teacher_record], mean, scale, device),
    )[0]
    blended_masks = _supported_staged_predictions(
        [blended_record],
        _predict(lane_model, [blended_record], mean, scale, device),
    )[0]

    reference = _group_gems(reference_path)
    teacher_attacks = _predicted_attacks(
        teacher_record,
        teacher_masks,
        offset,
    )
    blended_attacks = _predicted_attacks(
        blended_record,
        blended_masks,
        offset,
    )
    report = {
        "schema": "rb3charts-predicted-authoring-shadow",
        "schema_version": 1,
        "scope": (
            "post-generation evaluation only; reference notes are not "
            "training, model-selection, threshold-selection, or generation inputs"
        ),
        "configuration": {
            "seed": seed,
            "epochs": epochs,
            "distilled_weight": distilled_weight,
            "attack_threshold": attack_threshold,
            "teacher_attack_threshold": teacher_threshold,
            "alignment_offset_seconds": float(offset),
        },
        "provenance": {
            "feature_index_sha256": _hash_file(feature_index),
            "authoring_index_sha256": _hash_file(authoring_index),
            "structure_checkpoint_sha256": _hash_file(structure_checkpoint),
            "student_sha256": [_hash_file(path) for path in student_paths],
            "audio_sha256": _hash_file(audio_path),
            "tempo_sha256": _hash_file(tempo_path),
            "reference_sha256": _hash_file(reference_path),
        },
        "teacher": _metrics(reference, teacher_attacks),
        "distilled_candidate": _metrics(reference, blended_attacks),
    }
    if generated_path is not None:
        generated = [
            {
                **row,
                "time": (
                    float(row["time"])
                    - float(generated_audio_lead)
                    + float(offset)
                ),
            }
            for row in _group_gems(generated_path)
            if (
                float(row["time"])
                - float(generated_audio_lead)
                + float(offset)
            ) >= 0
        ]
        report["provenance"]["generated_sha256"] = _hash_file(
            generated_path
        )
        report["configuration"]["generated_audio_lead_seconds"] = float(
            generated_audio_lead
        )
        report["runtime_generated_baseline"] = _metrics(
            reference,
            generated,
        )
    save_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("feature_index", type=Path)
    parser.add_argument("authoring_index", type=Path)
    parser.add_argument("structure_checkpoint", type=Path)
    parser.add_argument("audio", type=Path)
    parser.add_argument("tempo", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--distilled-student",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--offset", type=float, required=True)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--distilled-weight", type=float, default=0.2)
    parser.add_argument("--attack-threshold", type=float, default=0.525)
    parser.add_argument("--generated", type=Path)
    parser.add_argument("--generated-audio-lead", type=float, default=0.0)
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
        [path.resolve() for path in args.distilled_student],
        args.audio.resolve(),
        args.tempo.resolve(),
        args.reference.resolve(),
        args.output.resolve(),
        args.offset,
        seed=args.seed,
        epochs=args.epochs,
        distilled_weight=args.distilled_weight,
        attack_threshold=args.attack_threshold,
        generated_path=(
            args.generated.resolve() if args.generated else None
        ),
        generated_audio_lead=args.generated_audio_lead,
        device=args.device,
    )
    print(json.dumps({
        name: {
            tolerance: metrics["attack"]
            for tolerance, metrics in result.items()
        }
        for name, result in report.items()
        if name in (
            "runtime_generated_baseline",
            "teacher",
            "distilled_candidate",
        )
    }, indent=2))


if __name__ == "__main__":
    main()
