"""Audit chart/audio synchronization before admitting authorized pairs."""

import argparse
import json
from pathlib import Path

import numpy as np

from rb3charts.datasets import _read_expert_gems
from rb3charts.model import save_json


def _distinct_attacks(chart):
    return sorted({
        round(float(note["start"]), 6)
        for note in _read_expert_gems(
            chart,
            "frets_on_fire_midi",
            "expert",
        )
    })


def _sample_score(envelope, frames):
    valid = frames[(frames >= 0) & (frames < len(envelope))]
    if not len(valid):
        return float("-inf")
    neighborhoods = np.stack([
        envelope[np.clip(valid + shift, 0, len(envelope) - 1)]
        for shift in (-1, 0, 1)
    ])
    return float(np.mean(np.max(neighborhoods, axis=0)))


def _offset_scan(envelope, attack_times, sr, hop, limit=0.5):
    import librosa

    frame_limit = int(round(limit * sr / hop))
    offsets = np.arange(-frame_limit, frame_limit + 1) * hop / sr
    scores = []
    for offset in offsets:
        frames = librosa.time_to_frames(
            np.asarray(attack_times) + offset,
            sr=sr,
            hop_length=hop,
        )
        scores.append(_sample_score(envelope, frames))
    best = int(np.argmax(scores))
    zero = int(np.argmin(np.abs(offsets)))
    return {
        "best_offset_seconds": round(float(offsets[best]), 6),
        "best_score": round(float(scores[best]), 6),
        "zero_score": round(float(scores[zero]), 6),
        "zero_to_best_ratio": round(
            float(scores[zero] / max(scores[best], 1e-12)),
            6,
        ),
    }


def _sync_flags(global_scan, windows, nearest):
    offsets = [row["best_offset_seconds"] for row in windows]
    within_25 = float(np.mean(np.asarray(nearest) <= 0.025))
    within_50 = float(np.mean(np.asarray(nearest) <= 0.05))
    criteria = {
        "global_offset_within_50ms": bool(
            abs(global_scan["best_offset_seconds"]) <= 0.05
        ),
        "all_window_offsets_within_100ms": bool(all(
            abs(offset) <= 0.1 for offset in offsets
        )),
        "window_offset_spread_within_80ms": bool(
            max(offsets) - min(offsets) <= 0.08
        ),
        "at_least_40pct_attacks_near_onset": bool(
            within_50 >= 0.4
        ),
    }
    near_one_to_one = {
        "global_offset_within_25ms": bool(
            abs(global_scan["best_offset_seconds"]) <= 0.025
        ),
        "all_window_offsets_within_50ms": bool(all(
            abs(offset) <= 0.05 for offset in offsets
        )),
        "at_least_40pct_attacks_within_25ms": bool(
            within_25 >= 0.4
        ),
    }
    return criteria, near_one_to_one


def _corrected_sync_flags(global_scan, windows, nearest):
    offset = float(global_scan["best_offset_seconds"])
    residuals = [
        float(row["best_offset_seconds"]) - offset
        for row in windows
    ]
    within_25 = float(np.mean(np.asarray(nearest) <= 0.025))
    within_50 = float(np.mean(np.asarray(nearest) <= 0.05))
    criteria = {
        "recommended_offset_within_100ms": abs(offset) <= 0.1,
        "all_window_residuals_within_60ms": all(
            abs(residual) <= 0.06 for residual in residuals
        ),
        "window_residual_spread_within_80ms": (
            max(residuals) - min(residuals) <= 0.08
        ),
        "at_least_40pct_corrected_attacks_near_onset": within_50 >= 0.4,
    }
    near_one_to_one = {
        "all_window_residuals_within_50ms": all(
            abs(residual) <= 0.05 for residual in residuals
        ),
        "at_least_40pct_corrected_attacks_within_25ms": within_25 >= 0.4,
    }
    return residuals, criteria, near_one_to_one


def audit_pair(guitar_audio, chart):
    import librosa
    from scipy import signal

    y, sr = librosa.load(guitar_audio, sr=22050, mono=True)
    hop = 256
    envelope = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=hop,
        center=True,
    )
    scale = float(np.percentile(envelope, 95))
    if scale > 0:
        envelope = np.clip(envelope / scale, 0, 4)
    attacks = _distinct_attacks(chart)
    global_scan = _offset_scan(envelope, attacks, sr, hop)
    duration = len(y) / sr
    windows = []
    for name, left, right in (
        ("early", 0.0, duration / 3),
        ("middle", duration / 3, 2 * duration / 3),
        ("late", 2 * duration / 3, duration),
    ):
        selected = [time for time in attacks if left <= time < right]
        windows.append({
            "name": name,
            "start_seconds": round(left, 6),
            "end_seconds": round(right, 6),
            "attacks": len(selected),
            **_offset_scan(envelope, selected, sr, hop, limit=0.1),
        })
    rolling = []
    start = 0.0
    while start < duration:
        selected = [
            time for time in attacks
            if start <= time < min(duration, start + 45.0)
        ]
        if len(selected) >= 32:
            rolling.append({
                "start_seconds": round(start, 6),
                "end_seconds": round(min(duration, start + 45.0), 6),
                "attacks": len(selected),
                **_offset_scan(envelope, selected, sr, hop, limit=0.1),
            })
        start += 30.0
    peaks, _ = signal.find_peaks(
        envelope,
        distance=2,
        prominence=max(1e-8, float(np.max(envelope)) * 0.01),
    )
    peak_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop)
    nearest = [
        float(np.min(np.abs(peak_times - time)))
        for time in attacks
    ]
    corrected_attacks = np.asarray(attacks) + float(
        global_scan["best_offset_seconds"]
    )
    corrected_nearest = [
        float(np.min(np.abs(peak_times - time)))
        for time in corrected_attacks
    ]
    criteria, _ = _sync_flags(
        global_scan,
        windows,
        nearest,
    )
    residuals, corrected_criteria, corrected_near_one_to_one = (
        _corrected_sync_flags(
            global_scan,
            windows,
            corrected_nearest,
        )
    )
    return {
        "audio": str(guitar_audio),
        "chart": str(chart),
        "audio_duration_seconds": round(duration, 6),
        "attacks": len(attacks),
        "global": global_scan,
        "windows": windows,
        "rolling_45s_windows": rolling,
        "attack_onset_support": {
            "within_25ms": round(
                float(np.mean(np.asarray(nearest) <= 0.025)),
                6,
            ),
            "within_50ms": round(
                float(np.mean(np.asarray(nearest) <= 0.05)),
                6,
            ),
            "median_nearest_onset_ms": round(
                float(np.median(nearest) * 1000),
                3,
            ),
        },
        "recommended_audio_offset_seconds": global_scan[
            "best_offset_seconds"
        ],
        "corrected_window_residuals_seconds": [
            round(float(residual), 6) for residual in residuals
        ],
        "corrected_attack_onset_support": {
            "within_25ms": round(
                float(np.mean(np.asarray(corrected_nearest) <= 0.025)),
                6,
            ),
            "within_50ms": round(
                float(np.mean(np.asarray(corrected_nearest) <= 0.05)),
                6,
            ),
            "median_nearest_onset_ms": round(
                float(np.median(corrected_nearest) * 1000),
                3,
            ),
        },
        "criteria": criteria,
        "automatic_sync_pass": all(criteria.values()),
        "correctable_sync_criteria": corrected_criteria,
        "correctable_sync_pass": all(corrected_criteria.values()),
        "near_one_to_one_timing_criteria": corrected_near_one_to_one,
        "near_one_to_one_timing_pass": all(
            corrected_near_one_to_one.values()
        ),
    }


def run(intake_path, output):
    root = intake_path.parent
    intake = json.loads(intake_path.read_text(encoding="utf-8"))
    pairs = []
    for pack in intake["packs"]:
        for song in pack["songs"]:
            folder = root / song["folder"]
            row = audit_pair(folder / "guitar.ogg", folder / "notes.mid")
            row.update({
                "pack": pack["id"],
                "song": song["metadata"]["name"],
                "chart_sha256": song["chart_sha256"],
                "guitar_audio_sha256": next(
                    item["sha256"]
                    for item in song["audio"]
                    if item["file"] == "guitar.ogg"
                ),
            })
            pairs.append(row)
    report = {
        "schema": "rb3charts-chart-pair-sync-audit",
        "schema_version": 1,
        "method": (
            "Center-compensated Librosa onset envelope from the shipped "
            "isolated guitar stem; "
            "zero-offset chart attacks compared with a +/-500 ms scan globally "
            "and independently over duration thirds"
        ),
        "admission_rule": (
            "Passing global and regional checks verifies the shipped zero-offset "
            "alignment for local training; listening review remains recommended "
            "before publishing or redistributing derived artifacts"
        ),
        "near_one_to_one_timing_rule": (
            "The stricter timing gate additionally requires global alignment "
            "within 25 ms, every duration-third offset within 50 ms, and at "
            "least 40 percent of chart attacks within 25 ms of an audio onset"
        ),
        "pairs": pairs,
        "summary": {
            "pairs": len(pairs),
            "automatic_sync_pass": sum(
                row["automatic_sync_pass"] for row in pairs
            ),
            "correctable_sync_pass": sum(
                row["correctable_sync_pass"] for row in pairs
            ),
            "near_one_to_one_timing_pass": sum(
                row["near_one_to_one_timing_pass"] for row in pairs
            ),
        },
    }
    save_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("intake", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = run(args.intake.resolve(), args.output.resolve())
    print(json.dumps({
        "summary": report["summary"],
        "pairs": [
            {
                "pack": row["pack"],
                "song": row["song"],
                "automatic_sync_pass": row["automatic_sync_pass"],
                "correctable_sync_pass": row["correctable_sync_pass"],
                "near_one_to_one_timing_pass": (
                    row["near_one_to_one_timing_pass"]
                ),
                "global": row["global"],
                "recommended_audio_offset_seconds": row[
                    "recommended_audio_offset_seconds"
                ],
                "window_offsets": [
                    window["best_offset_seconds"]
                    for window in row["windows"]
                ],
                "within_50ms": row["attack_onset_support"]["within_50ms"],
            }
            for row in report["pairs"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
