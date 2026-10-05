from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


OPEN_STRING_MIDI = (40, 45, 50, 55, 59, 64)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write lane-independent FretNet note evidence without changing a chart."
    )
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fretnet-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _note_evidence(
    stacked_notes: dict[int, tuple[Any, Any]], offset: float
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for string_index in sorted(stacked_notes):
        pitches, intervals = stacked_notes[string_index]
        for note_index, (pitch, interval) in enumerate(zip(pitches, intervals)):
            midi_pitch = float(pitch)
            evidence.append(
                {
                    "source_id": f"fretnet-s{string_index}-n{note_index}",
                    "onset_s": round(float(interval[0]) + offset, 6),
                    "offset_s": round(float(interval[1]) + offset, 6),
                    "midi_pitch": round(midi_pitch, 4),
                    "string_index": string_index,
                    "fret": round(midi_pitch - OPEN_STRING_MIDI[string_index], 4),
                }
            )
    return sorted(evidence, key=lambda note: (note["onset_s"], note["string_index"]))


def _pitch_runs(
    tablature: Any, times: Any, offset: float, hop_seconds: float
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for string_index, frets in enumerate(tablature):
        start = 0
        while start < len(frets):
            fret = int(frets[start])
            end = start + 1
            while end < len(frets) and int(frets[end]) == fret:
                end += 1
            if fret >= 0:
                runs.append(
                    {
                        "source_id": f"fretnet-frame-s{string_index}-f{start}",
                        "onset_s": round(float(times[start]) + offset, 6),
                        "offset_s": round(
                            float(times[end - 1]) + offset + hop_seconds, 6
                        ),
                        "midi_pitch": OPEN_STRING_MIDI[string_index] + fret,
                        "string_index": string_index,
                        "fret": fret,
                        "frame_count": end - start,
                    }
                )
            start = end
    return sorted(runs, key=lambda run: (run["onset_s"], run["string_index"]))


def main() -> int:
    args = _parse_args()
    audio_path = args.audio.resolve()
    checkpoint = args.checkpoint.resolve()
    fretnet_repo = args.fretnet_repo.resolve()
    output = args.output.resolve()
    if args.offset < 0 or (args.duration is not None and args.duration <= 0):
        raise ValueError("Offset must be non-negative and duration must be positive.")
    for path in (audio_path, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not (fretnet_repo / "guitar_transcription_continuous").is_dir():
        raise FileNotFoundError(f"FretNet source not found under {fretnet_repo}")

    sys.path.insert(0, str(fretnet_repo))
    try:
        import librosa
        import torch
        import amt_tools.tools as tools
        import guitar_transcription_continuous.utils as utils
        from amt_tools.features import HCQT
        from amt_tools.inference import run_offline
        from amt_tools.transcribe import (
            ComboEstimator,
            StackedNoteTranscriber,
            StackedOffsetsWrapper,
            TablatureWrapper,
        )
        from guitar_transcription_continuous.estimators import (
            StackedPitchListTablatureWrapper,
        )
        from guitar_transcription_continuous.models import FretNet
    except ImportError as error:
        raise RuntimeError(
            "FretNet shadow dependencies are missing. Use its isolated research environment."
        ) from error

    sample_rate = 22050
    hop_length = 512
    profile = tools.GuitarProfile(num_frets=19)
    feature_extractor = HCQT(
        sample_rate=sample_rate,
        hop_length=hop_length,
        fmin=librosa.note_to_hz("E2"),
        harmonics=[0.5, 1, 2, 3, 4, 5],
        n_bins=144,
        bins_per_octave=36,
    )
    model = FretNet(
        dim_in=feature_extractor.get_feature_size(),
        profile=profile,
        in_channels=feature_extractor.get_num_channels(),
        estimate_onsets=True,
        device=args.device,
    )
    model.load_state_dict(
        torch.load(checkpoint, map_location=args.device, weights_only=True)
    )
    model.eval()

    audio, _ = librosa.load(
        str(audio_path),
        sr=sample_rate,
        mono=True,
        offset=args.offset,
        duration=args.duration,
    )
    if not len(audio):
        raise ValueError("The requested audio range is empty.")
    peak = float(abs(audio).max())
    if peak:
        audio = audio / peak
    features = {
        tools.KEY_FEATS: feature_extractor.process_audio(audio),
        tools.KEY_TIMES: feature_extractor.get_times(audio),
    }
    estimator = ComboEstimator(
        [
            TablatureWrapper(profile=profile),
            StackedOffsetsWrapper(profile=profile),
            StackedNoteTranscriber(profile=profile),
            StackedPitchListTablatureWrapper(
                profile=profile,
                multi_pitch_key=tools.KEY_TABLATURE,
                multi_pitch_rel_key=utils.KEY_TABLATURE_REL,
            ),
        ]
    )
    with torch.no_grad():
        predictions = run_offline(features, model, estimator)
    notes = _note_evidence(predictions[tools.KEY_NOTES], args.offset)
    pitch_runs = _pitch_runs(
        predictions[tools.KEY_TABLATURE],
        predictions[tools.KEY_TIMES],
        args.offset,
        hop_length / sample_rate,
    )
    active_strings = sorted({run["string_index"] for run in pitch_runs})
    report = {
        "schema": 1,
        "kind": "fretnet_shadow_evidence",
        "audio": {
            "path": str(audio_path),
            "sha256": _sha256(audio_path),
            "offset_s": args.offset,
            "duration_s": round(len(audio) / sample_rate, 6),
        },
        "model": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "device": args.device,
        },
        "analysis": {
            "sample_rate": sample_rate,
            "hop_length": hop_length,
            "note_count": len(notes),
            "pitch_run_count": len(pitch_runs),
            "active_frame_count": sum(run["frame_count"] for run in pitch_runs),
            "active_strings": active_strings,
        },
        "notes": notes,
        "pitch_runs": pitch_runs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote {len(notes)} onset notes and {len(pitch_runs)} pitch runs across "
        f"{len(active_strings)} strings to {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
