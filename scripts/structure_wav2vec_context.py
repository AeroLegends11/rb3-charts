"""Evaluate frozen raw-audio embeddings as an authorized chartability branch."""

import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

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


EMBEDDING_SAMPLE_RATE = 16000
EMBEDDING_STRIDE = 320
EMBEDDING_RECEPTIVE_FIELD = 400


def _embedding_times(sample_start, frame_count):
    centers = (
        EMBEDDING_RECEPTIVE_FIELD - 1
    ) / 2 + np.arange(frame_count) * EMBEDDING_STRIDE
    return (sample_start + centers) / EMBEDDING_SAMPLE_RATE


def _embedding_targets(times, attack_frames):
    target = np.zeros(len(times), dtype=np.float32)
    attack_times = (
        np.asarray(attack_frames)
        * FEATURE_HOP_LENGTH
        / FEATURE_SAMPLE_RATE
    )
    indices = np.searchsorted(times, attack_times)
    indices = np.clip(indices, 0, max(0, len(times) - 1))
    previous = np.maximum(indices - 1, 0)
    use_previous = (
        np.abs(times[previous] - attack_times)
        < np.abs(times[indices] - attack_times)
    )
    indices[use_previous] = previous[use_previous]
    target[indices] = 1.0
    return target


def _extract_embeddings(audio_path, model, device, layer):
    import librosa
    import torch

    audio, _ = librosa.load(
        audio_path,
        sr=EMBEDDING_SAMPLE_RATE,
        mono=True,
    )
    core_samples = 20 * EMBEDDING_SAMPLE_RATE
    context_samples = EMBEDDING_SAMPLE_RATE
    embeddings = []
    times = []
    with torch.no_grad():
        for core_start in range(0, len(audio), core_samples):
            core_end = min(len(audio), core_start + core_samples)
            input_start = max(0, core_start - context_samples)
            input_end = min(len(audio), core_end + context_samples)
            waveform = torch.from_numpy(
                audio[input_start:input_end]
            ).unsqueeze(0).to(device)
            features = model.extract_features(
                waveform,
                num_layers=layer,
            )[0][-1][0].cpu().numpy()
            feature_times = _embedding_times(input_start, len(features))
            keep = (
                (feature_times >= core_start / EMBEDDING_SAMPLE_RATE)
                & (feature_times < core_end / EMBEDDING_SAMPLE_RATE)
            )
            embeddings.append(features[keep])
            times.append(feature_times[keep])
    return (
        np.concatenate(embeddings).astype(np.float16),
        np.concatenate(times),
    )


def _load_or_extract(item, cache_dir, model, device, layer):
    cache_path = cache_dir / f"{item['song_id']}-layer{layer}.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        return cached["embeddings"], cached["times"]
    embeddings, times = _extract_embeddings(
        _audio_path(item),
        model,
        device,
        layer,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        embeddings=embeddings,
        times=times,
    )
    return embeddings, times


def _normalization(records):
    total = 0
    feature_sum = None
    squared_sum = None
    for record in records:
        values = record["embeddings"].astype(np.float64)
        total += len(values)
        if feature_sum is None:
            feature_sum = values.sum(axis=0)
            squared_sum = np.square(values).sum(axis=0)
        else:
            feature_sum += values.sum(axis=0)
            squared_sum += np.square(values).sum(axis=0)
    mean = feature_sum / total
    variance = np.maximum(squared_sum / total - np.square(mean), 1e-6)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def _train(records, mean, scale, seed, epochs, device):
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    class ContextHead(nn.Module):
        def __init__(self, input_features):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv1d(input_features, 64, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv1d(64, 1, kernel_size=1),
            )

        def forward(self, values):
            return self.layers(values.transpose(1, 2)).squeeze(1)

    model = ContextHead(len(mean)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )
    positives = sum(float(record["targets"].sum()) for record in records)
    total = sum(len(record["targets"]) for record in records)
    positive_weight = min(12.0, (total - positives) / max(1, positives))
    losses = []
    for epoch in range(epochs):
        epoch_losses = []
        order = list(range(len(records)))
        random.Random(seed + epoch).shuffle(order)
        model.train()
        for index in order:
            record = records[index]
            features = torch.from_numpy(
                (
                    (record["embeddings"].astype(np.float32) - mean)
                    / scale
                )[None]
            ).to(device)
            targets = torch.from_numpy(
                record["targets"][None]
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = functional.binary_cross_entropy_with_logits(
                logits,
                targets,
                pos_weight=torch.tensor(
                    positive_weight,
                    device=device,
                ),
            )
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
    return model, losses, positive_weight


def _context_scores(model, record, mean, scale, output_frames, device):
    import torch

    model.eval()
    with torch.no_grad():
        features = torch.from_numpy(
            (
                (record["embeddings"].astype(np.float32) - mean)
                / scale
            )[None]
        ).to(device)
        scores = torch.sigmoid(model(features)[0]).cpu().numpy()
    output_times = (
        np.arange(output_frames)
        * FEATURE_HOP_LENGTH
        / FEATURE_SAMPLE_RATE
    )
    return np.interp(
        output_times,
        record["times"],
        scores,
        left=float(scores[0]),
        right=float(scores[-1]),
    )


def _chartability_gate_scores(base_scores, context_scores, weight):
    from scipy.ndimage import gaussian_filter1d

    base = np.clip(base_scores, 1e-5, 1 - 1e-5)
    context = np.clip(context_scores, 1e-5, 1 - 1e-5)
    context_logit = np.log(context / (1 - context))
    sigma = (
        0.25 * FEATURE_SAMPLE_RATE / FEATURE_HOP_LENGTH
    )
    context_logit = gaussian_filter1d(context_logit, sigma=sigma)
    context_logit = (
        context_logit - np.mean(context_logit)
    ) / max(1e-6, float(np.std(context_logit)))
    base_logit = np.log(base / (1 - base))
    fused_logit = base_logit + weight * context_logit
    return 1 / (1 + np.exp(-fused_logit))


def _gated_records(records, weight):
    return [
        {
            **record,
            "context_scores": _chartability_gate_scores(
                record["context_scores"],
                record["local_scores"],
                weight,
            ),
        }
        for record in records
    ]


def run(
    index_path,
    teacher_path,
    student_paths,
    cache_dir,
    output,
    seed=0,
    epochs=8,
    layer=6,
    device="auto",
    torch_home=None,
):
    if torch_home is not None:
        os.environ["TORCH_HOME"] = str(torch_home)
    import torch
    import torchaudio

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    embedding_model = bundle.get_model().to(device).eval()
    teacher, _ = _load_model(teacher_path, device)
    students = [
        _load_student(path, device)[0]
        for path in student_paths
    ]
    _, items = _load_feature_index(index_path)
    records = []
    with torch.no_grad():
        for item in items:
            arrays, _ = _load_feature_artifact(item["path"])
            embeddings, times = _load_or_extract(
                item,
                cache_dir,
                embedding_model,
                device,
                layer,
            )
            actual = np.flatnonzero(arrays["attack_target"] > 0)
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
            records.append({
                "song_id": item["song_id"],
                "split": item["split"],
                "embeddings": embeddings,
                "times": times,
                "targets": _embedding_targets(times, actual),
                "actual": actual,
                "context_scores": (
                    0.8 * teacher_scores + 0.2 * student_scores
                ),
            })
    del embedding_model
    torch.cuda.empty_cache()
    train = [record for record in records if record["split"] == "train"]
    validation = [
        record for record in records if record["split"] == "validation"
    ]
    test = [record for record in records if record["split"] == "test"]
    mean, scale = _normalization(train)
    model, losses, positive_weight = _train(
        train,
        mean,
        scale,
        seed,
        epochs,
        device,
    )
    for record in records:
        record["local_scores"] = _context_scores(
            model,
            record,
            mean,
            scale,
            len(record["context_scores"]),
            device,
        )
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
    for weight in (0.05, 0.1, 0.25, 0.5, 1.0):
        gated = _gated_records(validation, weight)
        for threshold in np.linspace(0.05, 0.95, 37):
            candidates.append({
                "chartability_weight": weight,
                **_summarize(
                    gated,
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
        "schema": "rb3charts-structure-wav2vec-context",
        "schema_version": 1,
        "scope": (
            "offline authorized-data evaluation using frozen LibriSpeech "
            "Wav2Vec2 embeddings; test remains sealed unless validation passes"
        ),
        "seed": seed,
        "epochs": epochs,
        "layer": layer,
        "device": device,
        "epoch_losses": losses,
        "positive_weight": positive_weight,
        "feature_index_sha256": _hash_file(index_path),
        "teacher_sha256": _hash_file(teacher_path),
        "student_sha256": [
            _hash_file(path) for path in student_paths
        ],
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
            _gated_records(test, selected["chartability_weight"]),
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
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("students", type=Path, nargs="+")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument("--torch-home", type=Path)
    args = parser.parse_args()
    report = run(
        args.index.resolve(),
        args.teacher.resolve(),
        [path.resolve() for path in args.students],
        args.cache.resolve(),
        args.output.resolve(),
        seed=args.seed,
        epochs=args.epochs,
        layer=args.layer,
        device=args.device,
        torch_home=(
            args.torch_home.resolve()
            if args.torch_home is not None else None
        ),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
