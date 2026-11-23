import numpy as np
import soundfile as sf
from rb3charts.audio import clean_guitar_stem
from rb3charts.stem_cleanup import suppress_bleed

SR = 22050


def tone(freq, seconds=2.0, start=0.0):
    t = np.arange(int(SR * seconds)) / SR
    y = sum(np.sin(2 * np.pi * freq * h * t) / h for h in (1, 2, 3))
    y[t < start] = 0
    return y / np.sqrt(np.mean(y[t >= start] ** 2))


def sdr(reference, estimate):
    return 10 * np.log10(np.sum(reference ** 2) / np.sum((estimate - reference) ** 2))


def stereo(x):
    return np.stack([x, x])


def test_bleed_suppression_improves_guitar_estimate_and_keeps_shape():
    guitar, vocal = tone(196), tone(587, start=1.0)
    leaky = guitar + .5 * vocal
    cleaned = suppress_bleed(stereo(leaky), [stereo(vocal)])
    assert cleaned.shape == (2, len(leaky))
    assert sdr(guitar, cleaned[0]) > sdr(guitar, leaky) + 3
    assert np.dot(cleaned[0], guitar) / np.dot(guitar, guitar) > .85


def test_clean_guitar_stem_leaves_raw_stem_and_reuses_output(tmp_path):
    guitar, vocal = tone(196), tone(587, start=1.0)
    paths = {}
    for name, data in (("guitar", guitar + .5 * vocal), ("vocals", vocal), ("drums", np.zeros_like(vocal))):
        paths[name] = tmp_path / f"{name}.wav"
        sf.write(paths[name], stereo(data).T, SR, subtype="FLOAT")
    raw_bytes = paths["guitar"].read_bytes()
    stems = clean_guitar_stem(paths, lambda message: None)
    assert stems["guitar"].name == "guitar-clean.wav" and stems["guitar_raw"] == paths["guitar"]
    assert paths["guitar"].read_bytes() == raw_bytes
    modified = stems["guitar"].stat().st_mtime_ns
    assert clean_guitar_stem(paths, lambda message: None)["guitar"].stat().st_mtime_ns == modified


def test_clean_guitar_stem_skips_mismatched_stems(tmp_path):
    sf.write(tmp_path / "guitar.wav", stereo(tone(196)).T, SR)
    sf.write(tmp_path / "vocals.wav", stereo(tone(587, seconds=1.0)).T, SR)
    stems = {"guitar": tmp_path / "guitar.wav", "vocals": tmp_path / "vocals.wav"}
    assert clean_guitar_stem(stems, lambda message: None) == stems


def test_cleanup_raises_transcription_precision_without_losing_recall():
    import pytest
    pytest.importorskip("basic_pitch")
    from rb3charts.transcription_bench import run_transcription_benchmark
    result = run_transcription_benchmark(leak=.6)
    assert result["cleaned"]["precision"] > result["raw"]["precision"] + .1
    assert result["cleaned"]["recall"] >= result["raw"]["recall"] - .02


def test_lead_focus_keeps_confident_octave_candidates():
    import pytest
    pytest.importorskip("basic_pitch")
    from rb3charts.transcription_bench import run_transcription_benchmark
    result = run_transcription_benchmark(leak=.6)
    assert result["cleaned_focused"]["recall"] >= result["cleaned"]["recall"] - .03
    assert result["cleaned_focused"]["precision"] > result["cleaned"]["precision"]
