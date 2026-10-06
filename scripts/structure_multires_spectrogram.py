"""Train attack structure from aligned short/medium/long spectrograms."""

import argparse
import json
import os
from pathlib import Path
import random

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from rb3charts.model import save_json
from rb3charts.structure_model import build_structure_model
from rb3charts.training import (
    FEATURE_HOP_LENGTH,
    FEATURE_SAMPLE_RATE,
    _hash_file,
    _load_feature_artifact,
    _load_feature_index,
    _onset_metrics,
)


FFT_SIZES = (512, 2048, 8192)
MELS_PER_SCALE = 48
SHORT_MELS = 64


def _audio_path(item):
    manifest_path = Path(item["manifest_path"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    matches = [
        row for row in payload["items"]
        if str(row["song_id"]) == str(item["song_id"])
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Manifest does not uniquely identify {item['song_id']}"
        )
    path = (manifest_path.parent / matches[0]["audio"]).resolve()
    if not path.is_file():
        raise ValueError(f"Authorized audio is missing: {path}")
    if _hash_file(path) != item["metadata"]["audio_sha256"]:
        raise ValueError(f"Authorized audio hash mismatch: {path}")
    return path


def _multires_features(audio_path, beat_phase, expected_frames):
    import librosa

    y, _ = librosa.load(
        audio_path,
        sr=FEATURE_SAMPLE_RATE,
        mono=True,
    )
    scales = []
    for fft_size in FFT_SIZES:
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=FEATURE_SAMPLE_RATE,
            n_fft=fft_size,
            hop_length=FEATURE_HOP_LENGTH,
            n_mels=MELS_PER_SCALE,
            fmin=30,
            fmax=FEATURE_SAMPLE_RATE / 2,
            power=2,
        )
        scale = np.clip(
            (
                librosa.power_to_db(
                    mel,
                    ref=1.0,
                    top_db=80,
                )
                + 80
            )
            / 80,
            0,
            1,
        ).T.astype(np.float32)
        if len(scale) != expected_frames:
            raise ValueError(
                f"Feature frame mismatch for {audio_path}: "
                f"{len(scale)} != {expected_frames}"
            )
        scales.append(scale)
    return np.concatenate(
        scales + [beat_phase.astype(np.float32)],
        axis=1,
    )


def _short_augmented_features(audio_path, arrays):
    import librosa

    y, _ = librosa.load(
        audio_path,
        sr=FEATURE_SAMPLE_RATE,
        mono=True,
    )
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=FEATURE_SAMPLE_RATE,
        n_fft=512,
        hop_length=FEATURE_HOP_LENGTH,
        n_mels=SHORT_MELS,
        fmin=30,
        fmax=FEATURE_SAMPLE_RATE / 2,
        power=2,
    )
    short_log_mel = np.clip(
        (
            librosa.power_to_db(
                mel,
                ref=1.0,
                top_db=80,
            )
            + 80
        )
        / 80,
        0,
        1,
    ).T.astype(np.float32)
    if len(short_log_mel) != len(arrays["attack_target"]):
        raise ValueError(
            f"Short feature frame mismatch for {audio_path}"
        )
    return np.concatenate((
        arrays["log_mel"],
        arrays["mel_delta"],
        short_log_mel,
        arrays["beat_phase"],
    ), axis=1).astype(np.float32)


def _short_primary_features(
    audio_path,
    arrays,
    short_delta=False,
):
    import librosa

    y, _ = librosa.load(
        audio_path,
        sr=FEATURE_SAMPLE_RATE,
        mono=True,
    )
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=FEATURE_SAMPLE_RATE,
        n_fft=512,
        hop_length=FEATURE_HOP_LENGTH,
        n_mels=128,
        fmin=30,
        fmax=FEATURE_SAMPLE_RATE / 2,
        power=2,
    )
    short_log_mel = np.clip(
        (
            librosa.power_to_db(mel, ref=1.0, top_db=80)
            + 80
        )
        / 80,
        0,
        1,
    ).T.astype(np.float32)
    if len(short_log_mel) != len(arrays["attack_target"]):
        raise ValueError(
            f"Short feature frame mismatch for {audio_path}"
        )
    delta = (
        np.maximum(
            np.diff(
                short_log_mel,
                axis=0,
                prepend=short_log_mel[:1],
            ),
            0,
        ).astype(np.float32)
        if short_delta else arrays["mel_delta"]
    )
    return np.concatenate((
        short_log_mel,
        delta,
        arrays["beat_phase"],
    ), axis=1).astype(np.float32)


def _dual_branch_features(audio_path, arrays):
    short = _short_primary_features(
        audio_path,
        arrays,
        short_delta=True,
    )
    return np.concatenate((
        arrays["log_mel"],
        arrays["mel_delta"],
        short[:, :256],
        arrays["beat_phase"],
    ), axis=1).astype(np.float32)


def _build_dual_branch(hidden_features=64):
    import torch
    import torch.nn as nn

    class DualBranchModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.long_projection = nn.Conv1d(
                258,
                hidden_features,
                kernel_size=1,
            )
            self.long_encoder = nn.Sequential(
                nn.Conv1d(
                    hidden_features,
                    hidden_features,
                    kernel_size=5,
                    padding=2,
                ),
                nn.ReLU(),
                nn.Conv1d(
                    hidden_features,
                    hidden_features,
                    kernel_size=5,
                    padding=2,
                ),
                nn.ReLU(),
            )
            self.context_pools = nn.ModuleList([
                nn.AvgPool1d(
                    kernel_size=kernel,
                    stride=1,
                    padding=kernel // 2,
                    count_include_pad=False,
                )
                for kernel in (33, 129, 257)
            ])
            self.context_fusion = nn.Sequential(
                nn.Conv1d(
                    hidden_features * 4,
                    hidden_features,
                    kernel_size=1,
                ),
                nn.ReLU(),
            )
            self.short_projection = nn.Conv1d(
                258,
                hidden_features,
                kernel_size=1,
            )
            self.short_encoder = nn.Sequential(
                nn.Conv1d(
                    hidden_features,
                    hidden_features,
                    kernel_size=5,
                    padding=2,
                ),
                nn.ReLU(),
                nn.Conv1d(
                    hidden_features,
                    hidden_features,
                    kernel_size=5,
                    padding=2,
                ),
                nn.ReLU(),
            )
            self.fusion = nn.Sequential(
                nn.Conv1d(
                    hidden_features * 2,
                    hidden_features,
                    kernel_size=1,
                ),
                nn.ReLU(),
            )
            self.attack = nn.Conv1d(
                hidden_features,
                1,
                kernel_size=1,
            )
            self.polyphony = nn.Conv1d(
                hidden_features,
                7,
                kernel_size=1,
            )
            self.sustain = nn.Conv1d(
                hidden_features,
                7,
                kernel_size=1,
            )

        def forward(self, features):
            long_features = torch.cat((
                features[:, :, :256],
                features[:, :, 512:514],
            ), dim=-1).transpose(1, 2)
            short_features = features[:, :, 256:514].transpose(1, 2)
            long_encoded = self.long_encoder(
                self.long_projection(long_features)
            )
            context = self.context_fusion(torch.cat(
                [long_encoded] + [
                    pool(long_encoded) for pool in self.context_pools
                ],
                dim=1,
            ))
            short_encoded = self.short_encoder(
                self.short_projection(short_features)
            )
            fused = self.fusion(torch.cat(
                (short_encoded, context),
                dim=1,
            ))
            return {
                "attack": self.attack(fused).squeeze(1),
                "polyphony": self.polyphony(fused).transpose(1, 2),
                "sustain": self.sustain(fused).transpose(1, 2),
            }

    return DualBranchModel()


def _macro_metrics(named_records, threshold, tolerance_frames):
    by_song = {
        song_id: _onset_metrics(
            [(scores, actual)],
            threshold,
            tolerance_frames,
        )
        for song_id, scores, actual in named_records
    }
    aggregate = _onset_metrics(
        [
            (scores, actual)
            for _, scores, actual in named_records
        ],
        threshold,
        tolerance_frames,
    )
    aggregate.update({
        "macro_precision": float(np.mean([
            row["precision"] for row in by_song.values()
        ])),
        "macro_recall": float(np.mean([
            row["recall"] for row in by_song.values()
        ])),
        "macro_f1": float(np.mean([
            row["f1"] for row in by_song.values()
        ])),
        "macro_median_timing_error_ms": float(np.mean([
            row["median_timing_error_ms"]
            for row in by_song.values()
            if row["median_timing_error_ms"] is not None
        ])),
        "macro_p95_timing_error_ms": float(np.mean([
            row["p95_timing_error_ms"]
            for row in by_song.values()
            if row["p95_timing_error_ms"] is not None
        ])),
        "by_song": by_song,
    })
    return aggregate


def run(
    index_path,
    output,
    seed=0,
    epochs=20,
    device="auto",
    representation="multires",
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
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")

    index_path, items = _load_feature_index(index_path)
    records = []
    for item in items:
        arrays, metadata = _load_feature_artifact(item["path"])
        audio_path = _audio_path(item)
        if representation == "multires":
            features = _multires_features(
                audio_path,
                arrays["beat_phase"],
                len(arrays["attack_target"]),
            )
        elif representation == "short-augment":
            features = _short_augmented_features(audio_path, arrays)
        elif representation == "dual-branch":
            features = _dual_branch_features(audio_path, arrays)
        else:
            features = _short_primary_features(
                audio_path,
                arrays,
                short_delta=representation == "short-consistent",
            )
        records.append({
            "song_id": item["song_id"],
            "split": item["split"],
            "features": features,
            "attack": arrays["attack_target"].astype(np.float32),
            "polyphony": arrays["polyphony_target"].astype(np.int64),
            "sustain": arrays["sustain_target"].astype(np.int64),
            "targets": set(metadata.get("targets") or (
                "attack",
                "polyphony",
                "sustain",
            )),
        })
    train = [record for record in records if record["split"] == "train"]
    validation = [
        record for record in records if record["split"] == "validation"
    ]
    test = [record for record in records if record["split"] == "test"]
    positives = sum(float(record["attack"].sum()) for record in train)
    frames = sum(len(record["attack"]) for record in train)
    positive_weight = min(
        50.0,
        max(1.0, (frames - positives) / positives),
    )
    model = (
        _build_dual_branch()
        if representation == "dual-branch"
        else build_structure_model(
            "multi-scale",
            input_features=train[0]["features"].shape[1],
            hidden_features=64,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )
    epoch_losses = []
    for _ in range(epochs):
        model.train()
        total = 0.0
        for record in train:
            features = torch.from_numpy(
                record["features"]
            ).unsqueeze(0).to(device)
            attacks = torch.from_numpy(
                record["attack"]
            ).unsqueeze(0).to(device)
            polyphony = torch.from_numpy(
                record["polyphony"]
            ).unsqueeze(0).to(device)
            sustain = torch.from_numpy(
                record["sustain"]
            ).unsqueeze(0).to(device)
            outputs = model(features)
            attack_loss = functional.binary_cross_entropy_with_logits(
                outputs["attack"],
                attacks,
                pos_weight=torch.tensor(positive_weight, device=device),
            )
            attack_mask = attacks > 0.5
            polyphony_loss = (
                functional.cross_entropy(
                    outputs["polyphony"][attack_mask],
                    polyphony[attack_mask],
                )
                if "polyphony" in record["targets"]
                else outputs["polyphony"].sum() * 0
            )
            sustain_loss = (
                functional.cross_entropy(
                    outputs["sustain"].reshape(-1, 7),
                    sustain.reshape(-1),
                )
                if "sustain" in record["targets"]
                else outputs["sustain"].sum() * 0
            )
            loss = (
                attack_loss
                + 0.5 * polyphony_loss
                + 0.25 * sustain_loss
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
        epoch_losses.append(total / len(train))

    def predict(selected):
        model.eval()
        result = []
        with torch.no_grad():
            for record in selected:
                outputs = model(
                    torch.from_numpy(
                        record["features"]
                    ).unsqueeze(0).to(device)
                )
                result.append((
                    record["song_id"],
                    torch.sigmoid(
                        outputs["attack"][0]
                    ).cpu().numpy(),
                    np.flatnonzero(record["attack"] > 0).tolist(),
                ))
        return result

    tolerance_frames = max(
        1,
        round(0.05 * FEATURE_SAMPLE_RATE / FEATURE_HOP_LENGTH),
    )
    validation_predictions = predict(validation)
    candidates = [
        _macro_metrics(
            validation_predictions,
            threshold,
            tolerance_frames,
        )
        for threshold in np.linspace(0.05, 0.95, 37)
    ]
    selected = max(
        candidates,
        key=lambda row: (
            row["macro_f1"],
            row["macro_precision"],
            row["f1"],
            row["threshold"],
        ),
    )
    baseline = {
        "macro_f1": 0.759665373617493,
        "macro_recall": 0.718132099284813,
        "macro_median_timing_error_ms": 14.9271137026239,
        "macro_p95_timing_error_ms": 38.14706835115,
    }
    guard_passed = bool(
        selected["macro_f1"] >= baseline["macro_f1"]
        and selected["macro_recall"] >= baseline["macro_recall"]
        and selected["macro_median_timing_error_ms"]
        < baseline["macro_median_timing_error_ms"]
        and selected["macro_p95_timing_error_ms"]
        <= baseline["macro_p95_timing_error_ms"]
    )
    report = {
        "schema": "rb3charts-structure-multires-spectrogram",
        "schema_version": 1,
        "scope": "offline authorized-data evaluation",
        "seed": seed,
        "epochs": epochs,
        "device": device,
        "representation": {
            "mode": representation,
            "fft_sizes": (
                list(FFT_SIZES)
                if representation == "multires"
                else (
                    [2048, 512]
                    if representation in (
                        "short-augment",
                        "dual-branch",
                    )
                    else [512]
                )
            ),
            "mels_per_scale": (
                MELS_PER_SCALE
                if representation == "multires"
                else (
                    [128, SHORT_MELS]
                    if representation == "short-augment"
                    else (
                        [128, 128]
                        if representation == "dual-branch"
                        else [128]
                    )
                )
            ),
            "hop_length": FEATURE_HOP_LENGTH,
            "sample_rate": FEATURE_SAMPLE_RATE,
            "beat_phase_features": 2,
            "positive_mel_delta_bins": (
                0
                if representation == "multires"
                else (
                    256
                    if representation == "dual-branch"
                    else 128
                )
            ),
            "positive_mel_delta_fft": (
                512
                if representation == "short-consistent"
                else 2048
            ),
        },
        "positive_weight": positive_weight,
        "epoch_losses": epoch_losses,
        "feature_index_sha256": _hash_file(index_path),
        "validation_baseline": baseline,
        "validation": selected,
        "guard_passed": guard_passed,
        "test_evaluated": guard_passed and bool(test),
    }
    if guard_passed and test:
        report["test"] = _macro_metrics(
            predict(test),
            selected["threshold"],
            tolerance_frames,
        )
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": report["schema"],
        "schema_version": report["schema_version"],
        "model_state": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "configuration": report["representation"],
        "training_report": report,
    }, output)
    save_json(output.with_suffix(".json"), report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--representation",
        choices=[
            "multires",
            "short-augment",
            "short-primary",
            "short-consistent",
            "dual-branch",
        ],
        default="multires",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    args = parser.parse_args()
    report = run(
        args.index,
        args.output,
        seed=args.seed,
        epochs=args.epochs,
        device=args.device,
        representation=args.representation,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
