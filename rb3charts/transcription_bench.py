"""Deterministic audio-level guitar transcription benchmark with synthetic stem bleed.

A plucked-guitar melody is rendered with a leaked vocal line and drum clicks. The
raw and bleed-suppressed stems are transcribed with the production Basic Pitch
path and scored against the known notes. Synthetic audio only: it checks the
cleanup direction, not real-song quality.
"""
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 22050


def _pluck(midi, seconds):
    t = np.arange(int(SR * seconds)) / SR
    f = 440 * 2 ** ((midi - 69) / 12)
    y = sum(np.sin(2 * np.pi * f * h * t) / h for h in (1, 2, 3, 4)) * np.exp(-t * 2.5)
    return y * np.minimum(1, t * 300)


def _line(notes, length, gain=1.0):
    out = np.zeros(int(SR * length))
    for start, midi, dur in notes:
        a = int(start * SR)
        y = _pluck(midi, dur)[:len(out) - a]
        out[a:a + len(y)] += y * gain
    return out


def render(seed=7, bars=8, bpm=100):
    rng = np.random.default_rng(seed)
    beat = 60 / bpm
    scale = [52, 55, 57, 59, 62, 64]
    truth = [(round(i * beat / 2, 4), int(rng.choice(scale))) for i in range(bars * 8)]
    length = bars * 4 * beat + 1.0
    guitar = _line([(s, p, beat / 2) for s, p in truth], length)
    vocal_scale = [64, 67, 69, 71, 74]
    vocals = _line([(i * beat * 1.5, int(rng.choice(vocal_scale)), beat * 1.4) for i in range(int(bars * 4 / 1.5))], length, 1.0)
    drums = np.zeros_like(guitar)
    for i in range(bars * 4):
        a = int(i * beat * SR)
        n = min(2000, len(drums) - a)
        drums[a:a + n] += rng.normal(0, 1, n) * np.exp(-np.arange(n) / 300)
    scale_g = guitar / np.max(np.abs(guitar))
    return truth, scale_g, vocals / np.max(np.abs(vocals)), drums / np.max(np.abs(drums))


def score(truth, notes, tolerance=.08):
    """Precision/recall with one-to-one onset (+-tolerance) and exact-pitch matching."""
    used, hits = set(), 0
    for start, pitch in truth:
        for index, note in enumerate(notes):
            if index not in used and note["pitch"] == pitch and abs(note["start"] - start) <= tolerance:
                used.add(index)
                hits += 1
                break
    precision = hits / len(notes) if notes else 0.0
    recall = hits / len(truth) if truth else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3), "notes": len(notes)}


def run_transcription_benchmark(leak=.6, seed=7, folder=None):
    from .audio import clean_guitar_stem
    from .transcribe import pitched
    truth, guitar, vocals, drums = render(seed)
    leaky = guitar + leak * vocals + .3 * drums
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(folder or scratch)
        paths = {}
        for name, data in (("guitar", leaky), ("vocals", vocals), ("drums", drums), ("bass", np.zeros_like(guitar))):
            paths[name] = root / f"{name}.wav"
            sf.write(paths[name], np.stack([data, data]).T, SR, subtype="FLOAT")
        results = {}
        cleaned = clean_guitar_stem(paths, lambda message: None)
        for label, path in (("raw", paths["guitar"]), ("cleaned", cleaned["guitar"])):
            notes = pitched(path, "guitar", .5, lambda message: None)
            results[label] = score(truth, notes)
    return results
