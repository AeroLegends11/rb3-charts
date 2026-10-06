"""Evaluate lane authoring with predicted attacks and polyphony offline."""

import argparse
from collections import Counter
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
    _predict_with_confidence,
    _predict_logit_ensemble,
    _predict_viterbi,
    _lane_transition_log_probabilities,
    _staged_predictions,
)
from structure_distill_ensemble import _load_student
from structure_multires_spectrogram import (
    _audio_path,
    _short_primary_features,
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


def _blend_attack_scores(
    teacher_scores,
    student_scores,
    student_weight,
):
    return (
        (1 - student_weight) * np.asarray(teacher_scores)
        + student_weight * np.mean(student_scores, axis=0)
    )


def _supported_staged_predictions(records, predicted):
    staged = []
    for record, candidate in zip(records, predicted):
        baseline = _baseline_masks({
            **record,
            "features": record["symbolic_features"],
        })
        supported = record.get(
            "learned_lane_allowed",
            [True] * len(candidate),
        )
        staged.append([
            candidate_mask
            if (
                allowed
                and max(
                    1,
                    min(5, round(float(symbolic[3]) * 5)),
                ) == 1
            )
            else baseline_mask
            for candidate_mask, baseline_mask, symbolic, allowed in zip(
                candidate,
                baseline,
                record["symbolic_features"],
                supported,
            )
        ])
    return staged


def _majority_predictions(records, prediction_sets):
    majority = []
    baselines = [
        _baseline_masks({
            **record,
            "features": record["symbolic_features"],
        })
        for record in records
    ]
    for record_index, baseline in enumerate(baselines):
        selected = []
        for attack_index, baseline_mask in enumerate(baseline):
            counts = Counter(
                predictions[record_index][attack_index]
                for predictions in prediction_sets
            )
            mask, votes = counts.most_common(1)[0]
            selected.append(
                mask if votes > len(prediction_sets) / 2 else baseline_mask
            )
        majority.append(selected)
    return majority


def _confidence_gated_records(records, confidences, threshold):
    gated = []
    for record, margins in zip(records, confidences):
        existing = record.get(
            "learned_lane_allowed",
            [True] * len(margins),
        )
        gated.append({
            **record,
            "learned_lane_allowed": [
                bool(allowed and margin >= threshold)
                for allowed, margin in zip(existing, margins)
            ],
        })
    return gated


def _mean_confidences(confidence_sets):
    return [
        np.mean(
            np.asarray([
                confidence_set[record_index]
                for confidence_set in confidence_sets
            ]),
            axis=0,
        ).tolist()
        for record_index in range(len(confidence_sets[0]))
    ]


def _phrase_rank_predictions(records, lane_min, lane_max):
    predictions = []
    for record in records:
        baseline = _baseline_masks({
            **record,
            "features": record["symbolic_features"],
        })
        supported = record.get(
            "learned_lane_allowed",
            [True] * len(baseline),
        )
        masks = []
        for baseline_mask, symbolic, features, allowed in zip(
            baseline,
            record["symbolic_features"],
            record["features"],
            supported,
        ):
            polyphony = max(1, min(5, round(float(symbolic[3]) * 5)))
            if allowed and polyphony == 1:
                rank = float(features[-2])
                lane = int(round(
                    lane_min + rank * (lane_max - lane_min)
                ))
                masks.append(1 << lane)
            else:
                masks.append(baseline_mask)
        predictions.append(masks)
    return predictions


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
    attack_scores=None,
    learned_lane_allowed=None,
):
    import torch

    scores = (
        np.asarray(attack_scores)
        if attack_scores is not None
        else torch.sigmoid(outputs["attack"][0]).cpu().numpy()
    )
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
        "learned_lane_allowed": (
            list(learned_lane_allowed)
            if learned_lane_allowed is not None
            else [True] * len(peaks)
        ),
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
        matched = exact = polyphony_exact = 0
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
            polyphony_exact += int(
                len(reference_lanes) == len(predicted_lanes)
            )
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
            "matched_exact_polyphony": (
                polyphony_exact / max(1, matched)
            ),
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
            "matched_exact_polyphony",
            "matched_lane_jaccard",
            "joint_exact_precision",
            "joint_exact_recall",
            "joint_exact_f1",
        )
    } | {
        "songs": rows,
    }


def _train_lane_models(
    records,
    seed,
    epochs,
    device,
    single_only_loss=False,
    lane_seeds=None,
    movement_loss_weight=0.0,
):
    import torch
    import torch.nn.functional as functional

    train = [record for record in records if record["split"] == "train"]
    stacked = np.concatenate([record["features"] for record in train])
    mean = stacked.mean(axis=0)
    scale = np.maximum(stacked.std(axis=0), 1e-5)
    class_polyphony = torch.tensor(
        [len(_lanes(mask)) for mask in MASKS],
        device=device,
    )
    lane_models = []
    for lane_seed in list(lane_seeds or [seed]):
        random.seed(lane_seed)
        np.random.seed(lane_seed)
        torch.manual_seed(lane_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(lane_seed)
        lane_model = _build_model(
            stacked.shape[1],
            movement_head=movement_loss_weight > 0,
        ).to(device)
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
                targets = torch.from_numpy(
                    record["targets"][None, :]
                ).to(device)
                encoded = lane_model.encode(features)
                logits = lane_model.output(encoded)
                if single_only_loss:
                    single = class_polyphony[targets] == 1
                    loss = functional.cross_entropy(
                        logits[single],
                        targets[single],
                    )
                else:
                    allowed = (
                        class_polyphony[targets].unsqueeze(-1)
                        == class_polyphony.reshape(1, 1, -1)
                    )
                    loss = functional.cross_entropy(
                        logits.masked_fill(
                            ~allowed,
                            -1e9,
                        ).reshape(-1, len(MASKS)),
                        targets.reshape(-1),
                    )
                if movement_loss_weight > 0 and len(record["targets"]) > 1:
                    centers = torch.tensor(
                        [
                            float(np.mean(_lanes(MASKS[int(target)])))
                            for target in record["targets"]
                        ],
                        device=device,
                    )
                    deltas = centers[1:] - centers[:-1]
                    movement_targets = (
                        torch.sign(deltas).long() + 1
                    )
                    close = torch.tensor(
                        np.diff(record["times"]) < 1.0,
                        device=device,
                    )
                    if bool(torch.any(close)):
                        movement_loss = functional.cross_entropy(
                            lane_model.movement_logits(encoded)[0][close],
                            movement_targets[close],
                        )
                        loss = (
                            loss
                            + movement_loss_weight * movement_loss
                        )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        lane_models.append(lane_model)
    return lane_models, mean, scale


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
    single_only_loss=False,
    distilled_students=None,
    distilled_weight=0.0,
    attack_threshold=None,
    teacher_supported_lanes=False,
    lane_seeds=None,
    movement_loss_weight=0.0,
    calibrate_lane_confidence=False,
    lane_ensemble="majority",
    phrase_rank_decoder=False,
    calibrate_lane_transition=False,
    evaluate_test=True,
):
    import torch

    if relative_pitch_only and not pitch_contour_features:
        raise ValueError("relative pitch features require pitch contour features")
    if calibrate_lane_confidence and not teacher_supported_lanes:
        raise ValueError(
            "lane confidence calibration requires teacher-supported lanes"
        )
    if lane_ensemble not in ("majority", "logit"):
        raise ValueError("lane ensemble must be majority or logit")
    if calibrate_lane_transition and len(lane_seeds or [seed]) != 1:
        raise ValueError(
            "lane transition calibration requires exactly one lane seed"
        )
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
    lane_seeds = list(lane_seeds or [seed])
    lane_models, mean, scale = _train_lane_models(
        records,
        seed,
        epochs,
        device,
        single_only_loss=single_only_loss,
        lane_seeds=lane_seeds,
        movement_loss_weight=movement_loss_weight,
    )
    structure_model, structure_configuration = _load_model(
        structure_checkpoint,
        device,
    )
    distilled_students = list(distilled_students or [])
    student_models = [
        _load_student(path, device)[0]
        for path in distilled_students
    ]
    if distilled_weight and not student_models:
        raise ValueError("distilled weight requires student checkpoints")
    if not 0 <= distilled_weight <= 1:
        raise ValueError("distilled weight must be between zero and one")
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
            selected_attack_scores = None
            teacher_scores = torch.sigmoid(
                outputs["attack"][0]
            ).cpu().numpy()
            if student_models:
                short_features = _short_primary_features(
                    _audio_path(item),
                    arrays,
                    short_delta=True,
                )
                short_tensor = torch.from_numpy(
                    short_features
                ).unsqueeze(0).to(device)
                selected_attack_scores = _blend_attack_scores(
                    teacher_scores,
                    [
                        torch.sigmoid(
                            model(short_tensor)["attack"][0]
                        ).cpu().numpy()
                        for model in student_models
                    ],
                    distilled_weight,
                )
            selected_threshold = (
                float(attack_threshold)
                if attack_threshold is not None
                else float(structure_configuration["attack_threshold"])
            )
            learned_lane_allowed = None
            if teacher_supported_lanes:
                selected_scores = (
                    selected_attack_scores
                    if selected_attack_scores is not None
                    else teacher_scores
                )
                selected_peaks = _attack_peaks(
                    selected_scores,
                    selected_threshold,
                )
                teacher_peaks = _attack_peaks(
                    teacher_scores,
                    float(structure_configuration["attack_threshold"]),
                )
                learned_lane_allowed = [
                    bool(np.any(np.abs(teacher_peaks - peak) <= 1))
                    for peak in selected_peaks
                ]
            predicted.append(_predicted_record(
                item,
                arrays,
                outputs,
                selected_threshold,
                context_features=context_features,
                pitch_contour_features=pitch_contour_features,
                relative_pitch_only=relative_pitch_only,
                attack_scores=selected_attack_scores,
                learned_lane_allowed=learned_lane_allowed,
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
    if lane_ensemble == "logit":
        validation_learned, validation_confidences = (
            _predict_logit_ensemble(
                lane_models,
                validation_predicted,
                mean,
                scale,
                device,
            )
        )
    else:
        validation_prediction_sets = []
        validation_confidence_sets = []
        for lane_model in lane_models:
            masks, margins = _predict_with_confidence(
                lane_model,
                validation_predicted,
                mean,
                scale,
                device,
            )
            validation_prediction_sets.append(masks)
            validation_confidence_sets.append(margins)
        validation_learned = _majority_predictions(
            validation_predicted,
            validation_prediction_sets,
        )
        validation_confidences = _mean_confidences(
            validation_confidence_sets
        )
    validation_baseline_masks = [
        _baseline_masks({
            **record,
            "features": record["symbolic_features"],
        })
        for record in validation_predicted
    ]
    stage = (
        _supported_staged_predictions
        if teacher_supported_lanes
        else _staged_predictions
    )
    validation_staged = stage(validation_predicted, validation_learned)
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
    phrase_rank_candidates = []
    selected_phrase_rank_span = None
    if phrase_rank_decoder:
        for lane_min, lane_max in ((0, 4), (1, 3)):
            masks = _phrase_rank_predictions(
                validation_predicted,
                lane_min,
                lane_max,
            )
            metrics = _timing_lane_metrics(
                references,
                validation_predicted,
                stage(validation_predicted, masks),
            )
            phrase_rank_candidates.append({
                "lane_min": lane_min,
                "lane_max": lane_max,
                "metrics": metrics,
            })
        selected_phrase_rank = max(
            phrase_rank_candidates,
            key=lambda row: (
                row["metrics"]["joint_exact_f1"],
                row["metrics"]["matched_exact_lanes"],
                row["metrics"]["matched_lane_jaccard"],
            ),
        )
        selected_phrase_rank_span = [
            selected_phrase_rank["lane_min"],
            selected_phrase_rank["lane_max"],
        ]
        validation_learned = _phrase_rank_predictions(
            validation_predicted,
            *selected_phrase_rank_span,
        )
        staged_metrics = selected_phrase_rank["metrics"]
    confidence_candidates = []
    confidence_threshold = None
    if calibrate_lane_confidence:
        finite_margins = np.asarray([
            margin
            for margins in validation_confidences
            for margin in margins
            if np.isfinite(margin)
        ])
        for quantile in np.linspace(0.0, 0.9, 10):
            threshold = float(np.quantile(finite_margins, quantile))
            gated_records = _confidence_gated_records(
                validation_predicted,
                validation_confidences,
                threshold,
            )
            metrics = _timing_lane_metrics(
                references,
                gated_records,
                stage(gated_records, validation_learned),
            )
            confidence_candidates.append({
                "quantile": float(quantile),
                "threshold": threshold,
                "metrics": metrics,
            })
        selected_confidence = max(
            confidence_candidates,
            key=lambda row: (
                row["metrics"]["joint_exact_f1"],
                row["metrics"]["matched_exact_lanes"],
                row["metrics"]["matched_lane_jaccard"],
                -row["threshold"],
            ),
        )
        confidence_threshold = selected_confidence["threshold"]
        staged_metrics = selected_confidence["metrics"]
    transition_candidates = []
    selected_transition_weight = None
    if calibrate_lane_transition:
        transition = _lane_transition_log_probabilities(train)
        for weight in (0.05, 0.1, 0.25, 0.5, 1.0):
            masks = _predict_viterbi(
                lane_models[0],
                validation_predicted,
                mean,
                scale,
                device,
                transition,
                weight,
            )
            metrics = _timing_lane_metrics(
                references,
                validation_predicted,
                stage(validation_predicted, masks),
            )
            transition_candidates.append({
                "weight": weight,
                "metrics": metrics,
            })
        selected_transition = max(
            transition_candidates,
            key=lambda row: (
                row["metrics"]["joint_exact_f1"],
                row["metrics"]["matched_exact_lanes"],
                row["metrics"]["matched_lane_jaccard"],
            ),
        )
        selected_transition_weight = selected_transition["weight"]
        validation_learned = _predict_viterbi(
            lane_models[0],
            validation_predicted,
            mean,
            scale,
            device,
            transition,
            selected_transition_weight,
        )
        staged_metrics = selected_transition["metrics"]
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
        "lane_seeds": lane_seeds,
        "epochs": epochs,
        "context_features": context_features,
        "pitch_contour_features": pitch_contour_features,
        "relative_pitch_only": relative_pitch_only,
        "single_only_loss": single_only_loss,
        "movement_loss_weight": movement_loss_weight,
        "calibrate_lane_confidence": calibrate_lane_confidence,
        "lane_confidence_threshold": confidence_threshold,
        "lane_ensemble": lane_ensemble,
        "phrase_rank_decoder": phrase_rank_decoder,
        "selected_phrase_rank_span": selected_phrase_rank_span,
        "calibrate_lane_transition": calibrate_lane_transition,
        "lane_transition_weight": selected_transition_weight,
        "distilled_student_sha256": [
            _hash_file(path)
            for path in distilled_students
        ],
        "distilled_weight": distilled_weight,
        "attack_threshold": (
            float(attack_threshold)
            if attack_threshold is not None
            else float(structure_configuration["attack_threshold"])
        ),
        "teacher_supported_lanes": teacher_supported_lanes,
        "guard_passed": guard_passed,
        "test_evaluated": bool(
            evaluate_test and guard_passed and test_predicted
        ),
        "feature_index_sha256": _hash_file(feature_index),
        "authoring_index_sha256": _hash_file(authoring_index),
        "structure_checkpoint_sha256": _hash_file(structure_checkpoint),
        "validation": {
            "fixed_polyphony_baseline": baseline_metrics,
            "staged_audio_candidate": staged_metrics,
        },
    }
    if confidence_candidates:
        report["validation"]["ungated_audio_candidate"] = (
            _timing_lane_metrics(
                references,
                validation_predicted,
                validation_staged,
            )
        )
        report["validation"]["confidence_candidates"] = (
            confidence_candidates
        )
    if phrase_rank_candidates:
        report["validation"]["phrase_rank_candidates"] = (
            phrase_rank_candidates
        )
    if transition_candidates:
        report["validation"]["transition_candidates"] = (
            transition_candidates
        )
    if evaluate_test and guard_passed and test_predicted:
        if lane_ensemble == "logit":
            test_learned, test_confidences = _predict_logit_ensemble(
                lane_models,
                test_predicted,
                mean,
                scale,
                device,
            )
        else:
            test_prediction_sets = []
            test_confidence_sets = []
            for lane_model in lane_models:
                masks, margins = _predict_with_confidence(
                    lane_model,
                    test_predicted,
                    mean,
                    scale,
                    device,
                )
                test_prediction_sets.append(masks)
                test_confidence_sets.append(margins)
            test_learned = _majority_predictions(
                test_predicted,
                test_prediction_sets,
            )
            test_confidences = _mean_confidences(
                test_confidence_sets
            )
        if selected_phrase_rank_span is not None:
            test_learned = _phrase_rank_predictions(
                test_predicted,
                *selected_phrase_rank_span,
            )
        if selected_transition_weight is not None:
            test_learned = _predict_viterbi(
                lane_models[0],
                test_predicted,
                mean,
                scale,
                device,
                transition,
                selected_transition_weight,
            )
        evaluated_test_records = test_predicted
        if confidence_threshold is not None:
            evaluated_test_records = _confidence_gated_records(
                test_predicted,
                test_confidences,
                confidence_threshold,
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
                evaluated_test_records,
                stage(evaluated_test_records, test_learned),
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
    parser.add_argument("--single-only-loss", action="store_true")
    parser.add_argument("--movement-loss-weight", type=float, default=0.0)
    parser.add_argument("--calibrate-lane-confidence", action="store_true")
    parser.add_argument(
        "--lane-ensemble",
        choices=["majority", "logit"],
        default="majority",
    )
    parser.add_argument("--phrase-rank-decoder", action="store_true")
    parser.add_argument("--calibrate-lane-transition", action="store_true")
    parser.add_argument(
        "--distilled-student",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--distilled-weight", type=float, default=0.0)
    parser.add_argument("--attack-threshold", type=float)
    parser.add_argument("--teacher-supported-lanes", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument(
        "--lane-seed",
        type=int,
        action="append",
        default=[],
    )
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
        single_only_loss=args.single_only_loss,
        movement_loss_weight=args.movement_loss_weight,
        calibrate_lane_confidence=args.calibrate_lane_confidence,
        lane_ensemble=args.lane_ensemble,
        phrase_rank_decoder=args.phrase_rank_decoder,
        calibrate_lane_transition=args.calibrate_lane_transition,
        distilled_students=[
            path.resolve()
            for path in args.distilled_student
        ],
        distilled_weight=args.distilled_weight,
        attack_threshold=args.attack_threshold,
        teacher_supported_lanes=args.teacher_supported_lanes,
        lane_seeds=args.lane_seed,
        evaluate_test=not args.skip_test,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
