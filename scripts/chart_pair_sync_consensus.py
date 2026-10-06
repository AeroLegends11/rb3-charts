"""Corroborate chart/audio offset with independent onset representations."""

import argparse
import json
from pathlib import Path

import numpy as np

from chart_pair_sync_audit import (
    _distinct_attacks,
    _offset_scan,
)
from rb3charts.model import save_json


def _group_event_times(events, key, tolerance=0.03):
    values = sorted(
        float(event[key])
        for event in events
        if event.get(key) is not None
    )
    groups = []
    for value in values:
        if groups and value - groups[-1][-1] <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return np.asarray([
        min(group) for group in groups
    ], dtype=np.float64)


def _event_offset_scan(
    attack_times,
    event_times,
    limit=0.1,
    step=0.001,
):
    offsets = np.arange(-limit, limit + step / 2, step)
    scores = []
    for offset in offsets:
        shifted = np.asarray(attack_times) + offset
        insertions = np.searchsorted(event_times, shifted)
        right = np.clip(insertions, 0, len(event_times) - 1)
        left = np.maximum(right - 1, 0)
        distance = np.minimum(
            np.abs(event_times[right] - shifted),
            np.abs(event_times[left] - shifted),
        )
        scores.append(float(np.mean(np.exp(
            -0.5 * np.square(distance / 0.02)
        ))))
    best = int(np.argmax(scores))
    return {
        "best_offset_seconds": round(float(offsets[best]), 6),
        "best_score": round(float(scores[best]), 6),
        "zero_score": round(
            float(scores[int(np.argmin(np.abs(offsets)))]),
            6,
        ),
    }


def _envelopes(y, sr, hop):
    import librosa

    onset = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=hop,
        center=True,
    )
    high_frequency = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=hop,
        center=True,
        fmin=700,
        fmax=8000,
        n_mels=96,
    )
    spectrum = np.abs(librosa.stft(
        y,
        n_fft=1024,
        hop_length=hop,
        center=True,
    ))
    flux = np.maximum(0, np.diff(
        spectrum,
        axis=1,
        prepend=spectrum[:, :1],
    )).mean(axis=0)
    rms = librosa.feature.rms(
        y=y,
        frame_length=1024,
        hop_length=hop,
        center=True,
    )[0]
    rms_rise = np.maximum(0, np.diff(rms, prepend=rms[:1]))
    return {
        "mel_onset": onset,
        "high_frequency_onset": high_frequency,
        "spectral_flux": flux,
        "rms_rise": rms_rise,
    }


def run(audio_path, chart_path, event_path, output):
    import librosa

    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    attacks = _distinct_attacks(chart_path)
    hop = 128
    audio_methods = {
        name: _offset_scan(
            envelope,
            attacks,
            sr,
            hop,
            limit=0.1,
        )
        for name, envelope in _envelopes(y, sr, hop).items()
    }
    events = json.loads(event_path.read_text(encoding="utf-8"))
    event_methods = {}
    for key in ("start", "model_start"):
        event_times = _group_event_times(events, key)
        event_methods[key] = {
            "event_groups": len(event_times),
            **_event_offset_scan(attacks, event_times),
        }
    audio_offsets = np.asarray([
        row["best_offset_seconds"]
        for row in audio_methods.values()
    ])
    all_offsets = np.asarray([
        *audio_offsets.tolist(),
        *[
            row["best_offset_seconds"]
            for row in event_methods.values()
        ],
    ])
    consensus = float(np.median(audio_offsets))
    report = {
        "schema": "rb3charts-chart-pair-sync-consensus",
        "schema_version": 1,
        "scope": (
            "diagnostic synchronization corroboration; no model selection "
            "or manifest mutation"
        ),
        "audio": str(audio_path),
        "chart": str(chart_path),
        "events": str(event_path),
        "attacks": len(attacks),
        "audio_methods": audio_methods,
        "event_methods": event_methods,
        "consensus_offset_seconds": round(consensus, 6),
        "audio_method_spread_ms": round(
            float(np.ptp(audio_offsets) * 1000),
            3,
        ),
        "all_method_spread_ms": round(
            float(np.ptp(all_offsets) * 1000),
            3,
        ),
        "corroborated": bool(
            np.ptp(audio_offsets) <= 0.015
            and all(
                abs(
                    row["best_offset_seconds"] - consensus
                ) <= 0.025
                for row in event_methods.values()
            )
        ),
    }
    save_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("chart", type=Path)
    parser.add_argument("events", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = run(
        args.audio.resolve(),
        args.chart.resolve(),
        args.events.resolve(),
        args.output.resolve(),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
