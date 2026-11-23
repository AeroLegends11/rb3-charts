"""Whole-pipeline check on synthetic stems: needs Basic Pitch and ffmpeg, no real songs."""
import json
import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("basic_pitch")
pytest.importorskip("imageio_ffmpeg")


def test_transposed_riff_keeps_pitches_and_fingering(tmp_path):
    from rb3charts import transcription_bench as tb
    from rb3charts.model import Options
    from rb3charts.pipeline import generate
    beat, bars, riff = .6, 16, [0, 3, 5, 7, 5, 3, 7, 10]
    truth = [(bar * 4 * beat + i * beat / 2, 52 + (0 if bar % 4 < 2 else 5) + step)
             for bar in range(bars) for i, step in enumerate(riff)]
    length = bars * 4 * beat + 1.5
    guitar = tb._line([(s, p, beat / 2) for s, p in truth], length)
    guitar /= np.max(np.abs(guitar))
    drums = np.zeros_like(guitar)
    for i in range(int(length / beat)):
        a = int(i * beat * tb.SR)
        n = min(1500, len(drums) - a)
        drums[a:a + n] += np.random.default_rng(i).normal(0, 1, n) * np.exp(-np.arange(n) / 250)
    drums /= np.max(np.abs(drums))
    stems = tmp_path / "stems"
    stems.mkdir()
    for name, data in (("guitar", guitar), ("drums", drums), ("vocals", 0 * guitar), ("bass", 0 * guitar)):
        sf.write(stems / f"{name}.wav", np.stack([data, data]).T, tb.SR, subtype="FLOAT")
    mix = guitar * .8 + drums * .4
    sf.write(tmp_path / "song.wav", np.stack([mix, mix]).T, tb.SR, subtype="FLOAT")
    result = generate(Options(source=tmp_path / "song.wav", output=tmp_path / "out", title="Synth", artist="Test",
                              instruments=("guitar",), chart_profile="band", stems=stems, compile=False, bpm=100),
                      progress=lambda message: None)
    folder = next((tmp_path / "out").iterdir())
    expert = json.loads((folder / "analysis" / "charts.json").read_text())["guitar"]["expert"]
    exact = sum(any(abs(n["start"] - s) <= .06 and n["pitch"] == p for n in expert) for s, p in truth)
    assert exact >= .95 * len(truth)
    lanes = {}
    for n in expert:
        lanes.setdefault(int((n["start"] + .15) // (4 * beat)), []).append(n["lanes"][0])
    # Bars repeat the riff; the later bars of each four-bar group are the same riff transposed
    assert lanes[1] == lanes[5] == lanes[9] == lanes[13]
    assert lanes[3] == lanes[7] == lanes[11] == lanes[15]
    charts = json.loads((folder / "analysis" / "charts.json").read_text())["guitar"]
    for difficulty in ("hard", "medium"):
        rows = charts[difficulty]
        repeated = [1 for a, b in zip(rows, rows[1:])
                    if len(a["lanes"]) == 1 == len(b["lanes"]) and a["lanes"] == b["lanes"]
                    and a["pitch"] != b["pitch"] and b["start"] - a["start"] <= .8]
        assert not repeated, difficulty
