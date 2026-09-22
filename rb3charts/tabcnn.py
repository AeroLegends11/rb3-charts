"""Offline inference for the CC BY 4.0 EGSet12 TabCNN checkpoint."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .model import save_json


SAMPLE_RATE = 22050
HOP_LENGTH = 512
CQT_BINS = 192
BINS_PER_OCTAVE = 24
FRAME_WIDTH = 9
MODEL_SOURCE = "https://zenodo.org/records/11406378"
MODEL_LICENSE = "CC BY 4.0"
MODEL_ATTRIBUTION = "Hegel Pedroza, Wallace Abreu, Ryan M. Corey, and Iran R. Roman"


class TabCNNError(ValueError):
    pass


def _hash_file(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _group_attacks(notes, tolerance=0.04):
    attacks = []
    for note in sorted(notes, key=lambda row: (row["start"], row["pitch"], row["string"])):
        if attacks and note["start"] - attacks[-1]["start"] <= tolerance:
            attacks[-1]["notes"].append(note)
        else:
            attacks.append({"start": note["start"], "notes": [note]})
    result = []
    for index, attack in enumerate(attacks):
        rows = attack["notes"]
        pitches = sorted({int(row["pitch"]) for row in rows})
        result.append({
            "source_id": f"tabcnn-attack-{index}",
            "start": float(min(row["start"] for row in rows)),
            "end": float(max(row["end"] for row in rows)),
            "pitches": pitches,
            "polyphony": len(pitches),
            "is_chord": len(pitches) > 1,
            "strings": sorted({int(row["string"]) for row in rows}),
            "confidence": float(np.mean([row["confidence"] for row in rows])),
        })
    return result


def _decode_notes(states, probabilities, times, tuning, silence_class):
    notes = []
    hop_seconds = HOP_LENGTH / SAMPLE_RATE
    for string in range(states.shape[1]):
        start = 0
        while start < len(states):
            fret = int(states[start, string])
            if fret == silence_class:
                start += 1
                continue
            end = start + 1
            while end < len(states) and int(states[end, string]) == fret:
                end += 1
            notes.append({
                "source_id": f"tabcnn-string-{string}-frame-{start}",
                "start": float(times[start]),
                "end": float(
                    times[end] if end < len(times) else times[-1] + hop_seconds
                ),
                "pitch": int(tuning[string] + fret),
                "string": string,
                "fret": fret,
                "confidence": float(np.mean(probabilities[start:end, string, fret])),
            })
            start = end
    return notes


def _sample_attack_states(probabilities, times, attack_rows, tuning, radius=2):
    silence_class = probabilities.shape[-1] - 1
    sampled = []
    for index, attack in enumerate(attack_rows):
        start = float(attack["start"])
        center = int(np.clip(np.searchsorted(times, start), 0, len(times) - 1))
        candidates = range(max(0, center - radius), min(len(times), center + radius + 1))

        def frame_score(frame):
            classes = np.argmax(probabilities[frame], axis=-1)
            margins = [
                probabilities[frame, string, fret] - probabilities[frame, string, silence_class]
                for string, fret in enumerate(classes)
                if fret != silence_class
            ]
            return (sum(max(0.0, float(value)) for value in margins), -abs(times[frame] - start))

        frame = max(candidates, key=frame_score)
        classes = np.argmax(probabilities[frame], axis=-1)
        strings = []
        for string, fret in enumerate(classes):
            fret = int(fret)
            if fret == silence_class:
                continue
            strings.append({
                "string": string,
                "fret": fret,
                "pitch": int(tuning[string] + fret),
                "confidence": float(probabilities[frame, string, fret]),
            })
        pitches = sorted({row["pitch"] for row in strings})
        sampled.append({
            "source_id": str(attack.get("source_id") or f"structure-{index}"),
            "start": start,
            "tabcnn_frame_time": float(times[frame]),
            "pitches": pitches,
            "polyphony": len(pitches),
            "is_chord": len(pitches) > 1,
            "strings": strings,
            "attack_probability": attack.get("attack_probability"),
        })
    return sampled


def predict_tabcnn(checkpoint_path, audio_path, output, attacks_path=None, device="auto"):
    """Write offline string/fret and polyphony evidence without changing a chart."""
    checkpoint_path = Path(checkpoint_path).resolve()
    audio_path = Path(audio_path).resolve()
    output = Path(output).resolve()
    if output.suffix.lower() != ".json":
        raise TabCNNError("TabCNN prediction output must use the .json suffix")
    if not checkpoint_path.is_file():
        raise TabCNNError(f"TabCNN checkpoint does not exist: {checkpoint_path}")
    if not audio_path.is_file():
        raise TabCNNError(f"TabCNN audio does not exist: {audio_path}")

    try:
        import librosa
        import torch
        from amt_tools.features import CQT
    except (ImportError, OSError) as exc:
        raise TabCNNError(
            "TabCNN inference requires the upstream AMT-Tools package from "
            "https://github.com/robust-guitar-tabs/code"
        ) from exc

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in ("cpu", "cuda"):
        raise TabCNNError("TabCNN device must be auto, cpu, or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise TabCNNError("CUDA TabCNN inference was requested but is unavailable")
    torch_device = torch.device(device)

    try:
        model = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError, ModuleNotFoundError) as exc:
        raise TabCNNError(f"Cannot read TabCNN checkpoint: {exc}") from exc
    output_layer = getattr(getattr(model, "dense", [None])[-1], "num_groups", None)
    class_count = getattr(getattr(model, "dense", [None])[-1], "num_classes", None)
    if (
        getattr(model, "dim_in", None) != CQT_BINS
        or getattr(model, "frame_width", None) != FRAME_WIDTH
        or output_layer != 6
        or class_count != 21
    ):
        raise TabCNNError("TabCNN checkpoint has incompatible input or output dimensions")

    try:
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    except (OSError, ValueError) as exc:
        raise TabCNNError(f"Cannot decode TabCNN audio: {exc}") from exc
    if not len(audio):
        raise TabCNNError("TabCNN audio is empty")
    extractor = CQT(
        sample_rate=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        n_bins=CQT_BINS,
        bins_per_octave=BINS_PER_OCTAVE,
    )
    features = extractor.process_audio(audio).astype(np.float32, copy=False)
    times = np.arange(features.shape[-1], dtype=np.float64) * HOP_LENGTH / SAMPLE_RATE
    padding = FRAME_WIDTH // 2
    padded = np.pad(features, ((0, 0), (0, 0), (padding, padding)))
    windows = np.lib.stride_tricks.sliding_window_view(
        padded,
        FRAME_WIDTH,
        axis=-1,
    )
    windows = np.moveaxis(windows, 2, 0)

    model.change_device(torch_device)
    model.eval()
    logits = []
    with torch.no_grad():
        for start in range(0, len(windows), 2048):
            batch = torch.from_numpy(windows[start:start + 2048].copy()).to(torch_device)
            rows = model(batch.unsqueeze(0))["tablature"][0]
            logits.append(rows.cpu())
    logits = torch.cat(logits, dim=0)
    probabilities = torch.softmax(logits.reshape(-1, 6, 21), dim=-1).numpy()
    states = np.argmax(probabilities, axis=-1)
    tuning = np.asarray(model.profile.get_midi_tuning(), dtype=int)
    notes = _decode_notes(states, probabilities, times, tuning, silence_class=20)
    attacks = _group_attacks(notes)

    sampled = None
    attacks_sha256 = None
    if attacks_path is not None:
        attacks_path = Path(attacks_path).resolve()
        try:
            attack_payload = json.loads(attacks_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TabCNNError(f"Cannot read attack predictions: {exc}") from exc
        attack_rows = attack_payload.get("events")
        if not isinstance(attack_rows, list):
            raise TabCNNError("Attack predictions require an events array")
        sampled = _sample_attack_states(probabilities, times, attack_rows, tuning)
        attacks_sha256 = _hash_file(attacks_path)

    result = {
        "schema": "rb3charts-tabcnn-predictions",
        "schema_version": 1,
        "scope": "offline pitch/polyphony evaluation; not applied to chart generation",
        "model": {
            "path": checkpoint_path.name,
            "sha256": _hash_file(checkpoint_path),
            "source": MODEL_SOURCE,
            "license": MODEL_LICENSE,
            "attribution": MODEL_ATTRIBUTION,
        },
        "audio": {
            "path": audio_path.name,
            "sha256": _hash_file(audio_path),
            "duration": float(len(audio) / SAMPLE_RATE),
        },
        "configuration": {
            "sample_rate": SAMPLE_RATE,
            "hop_length": HOP_LENGTH,
            "cqt_bins": CQT_BINS,
            "bins_per_octave": BINS_PER_OCTAVE,
            "frame_width": FRAME_WIDTH,
            "device": device,
        },
        "attack_input": (
            {"path": Path(attacks_path).name, "sha256": attacks_sha256}
            if attacks_path is not None else None
        ),
        "summary": {
            "frames": len(times),
            "notes": len(notes),
            "attacks": len(attacks),
            "chord_attacks": sum(row["is_chord"] for row in attacks),
            "sampled_attacks": len(sampled) if sampled is not None else None,
            "sampled_with_pitch": (
                sum(bool(row["pitches"]) for row in sampled)
                if sampled is not None else None
            ),
        },
        "notes": notes,
        "attacks": attacks,
        "sampled_attacks": sampled,
    }
    save_json(output, result)
    return result
