from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np


PERFORMERS = tuple(f"{index:02d}" for index in range(6))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a deterministic FretNet pilot on rights-cleared GuitarSet audio."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--fretnet-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holdout", choices=PERFORMERS, default="05")
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--validation-examples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _seed_everything(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _dataset_fingerprint(dataset: Path) -> str:
    digest = hashlib.sha256()
    annotation_dir = dataset / "annotations"
    for path in sorted(annotation_dir.glob("*.jams")):
        stat = path.stat()
        digest.update(path.name.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        with path.open("rb") as source:
            digest.update(source.read(4096))
    return digest.hexdigest()


def _git_revision(repository: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _mean_losses(
    model: Any,
    dataset: Any,
    count: int,
    seed: int,
    torch: Any,
    data_loader: Any,
) -> dict[str, float]:
    _seed_everything(seed, torch)
    loader = data_loader(dataset, batch_size=1, shuffle=False, num_workers=0)
    totals: dict[str, float] = {}
    evaluated = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            output = model.run_on_batch(batch)
            for key, value in output["loss"].items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
            evaluated += 1
            if evaluated >= count:
                break
    if evaluated == 0:
        raise RuntimeError("The held-out GuitarSet split did not yield any examples.")
    return {key: value / evaluated for key, value in sorted(totals.items())}


def _train_steps(
    model: Any,
    loader: Any,
    optimizer: Any,
    steps: int,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    tracked = 0
    batches = iter(loader)
    model.train()
    for step in range(steps):
        try:
            batch = next(batches)
        except StopIteration:
            batches = iter(loader)
            batch = next(batches)
        optimizer.zero_grad()
        output = model.run_on_batch(batch)
        losses = output["loss"]
        losses["loss_total"].backward()
        optimizer.step()
        if step >= max(0, steps - 100):
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
            tracked += 1
    return {key: value / tracked for key, value in sorted(totals.items())}


def main() -> int:
    args = _parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.validation_examples < 1:
        raise ValueError("Steps, batch size, and validation examples must be positive.")

    dataset = args.dataset.resolve()
    fretnet_repo = args.fretnet_repo.resolve()
    output = args.output.resolve()
    if not (dataset / "annotations").is_dir():
        raise FileNotFoundError(f"GuitarSet annotations not found under {dataset}")
    if not (dataset / "audio-mic").is_dir():
        raise FileNotFoundError(f"GuitarSet mic audio not found under {dataset}")
    if not (fretnet_repo / "guitar_transcription_continuous").is_dir():
        raise FileNotFoundError(f"FretNet source not found under {fretnet_repo}")

    sys.path.insert(0, str(fretnet_repo))
    try:
        import librosa
        import torch
        from amt_tools import tools
        from amt_tools.features import HCQT
        from guitar_transcription_continuous.datasets import GuitarSetPlus
        from guitar_transcription_continuous.models import FretNet
        from torch.utils.data import DataLoader
    except ImportError as error:
        raise RuntimeError(
            "FretNet pilot dependencies are missing. Use its isolated research environment."
        ) from error

    _seed_everything(args.seed, torch)
    output.mkdir(parents=True, exist_ok=True)

    profile = tools.GuitarProfile(num_frets=19)
    features = HCQT(
        sample_rate=22050,
        hop_length=512,
        fmin=librosa.note_to_hz("E2"),
        harmonics=[0.5, 1, 2, 3, 4, 5],
        n_bins=144,
        bins_per_octave=36,
    )
    common_dataset_args = {
        "base_dir": str(dataset),
        "hop_length": 512,
        "sample_rate": 22050,
        "num_frames": 200,
        "data_proc": features,
        "profile": profile,
        "save_data": False,
        "store_data": False,
        "augment": False,
        "evaluation_extras": False,
    }
    train_dataset = GuitarSetPlus(
        splits=[split for split in PERFORMERS if split != args.holdout],
        **common_dataset_args,
    )
    heldout_dataset = GuitarSetPlus(splits=[args.holdout], **common_dataset_args)
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        generator=loader_generator,
    )
    model = FretNet(
        dim_in=features.get_feature_size(),
        profile=profile,
        in_channels=features.get_num_channels(),
        estimate_onsets=True,
        device=args.device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    initial_losses = _mean_losses(
        model,
        heldout_dataset,
        args.validation_examples,
        args.seed,
        torch,
        DataLoader,
    )
    _seed_everything(args.seed, torch)
    final_training_losses = _train_steps(
        model,
        train_loader,
        optimizer,
        args.steps,
    )
    final_losses = _mean_losses(
        model,
        heldout_dataset,
        args.validation_examples,
        args.seed,
        torch,
        DataLoader,
    )

    checkpoint = output / "fretnet-pilot-state.pt"
    torch.save(model.state_dict(), checkpoint)
    report = {
        "seed": args.seed,
        "device": args.device,
        "optimizer_steps": args.steps,
        "batch_size": args.batch_size,
        "training_performers": [
            split for split in PERFORMERS if split != args.holdout
        ],
        "holdout_performer": args.holdout,
        "validation_examples": args.validation_examples,
        "dataset_fingerprint": _dataset_fingerprint(dataset),
        "fretnet_revision": _git_revision(fretnet_repo),
        "checkpoint": checkpoint.name,
        "initial_losses": initial_losses,
        "final_training_losses": final_training_losses,
        "final_losses": final_losses,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "amt_tools": _package_version("amt-tools"),
            "guitar_transcription_continuous": _package_version(
                "guitar-transcription-continuous"
            ),
        },
        "environment": {
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
