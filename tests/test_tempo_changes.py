import numpy as np
import pytest
from rb3charts.timing import _sustained_tempo_change, analyze_tempo


def test_continuous_tempo_change_requires_active_supported_windows():
    trajectory=np.repeat([70,76,84,92,103,115,125],800)
    assert _sustained_tempo_change(trajectory,np.ones(len(trajectory)),.01)
    assert not _sustained_tempo_change(trajectory,np.zeros(len(trajectory)),.01)


def test_constant_tempo_jitter_and_half_time_switch_do_not_trigger_drift():
    for values in ([120,121,119,120,122,120], [120,120,60,60,120,120]):
        trajectory=np.repeat(values,800)
        assert not _sustained_tempo_change(trajectory,np.ones(len(trajectory)),.01)


def test_audio_acceleration_overrides_studio_preference_but_not_manual_bpm():
    sr=8000
    duration=40
    y=np.zeros(sr*duration,dtype=np.float32)
    time=.2
    while time<duration-.1:
        index=int(time*sr)
        y[index:index+80]=np.hanning(80)
        time+=60/(75+50*min(time/30,1))
    tempo=analyze_tempo(y,sr,duration,prefer_constant=True)
    intervals=np.diff(tempo.beats)
    assert np.median(intervals[:8]) > np.median(intervals[-12:])*1.3
    assert any('Sustained tempo change' in warning for warning in tempo.warnings)
    manual=analyze_tempo(y,sr,duration,bpm=120,prefer_constant=True)
    assert np.allclose(np.diff(manual.beats),.5)


def test_studio_preference_produces_stable_grid_for_constant_band_audio():
    sr=8000
    duration=24
    y=np.zeros(sr*duration,dtype=np.float32)
    for time in np.arange(.4,duration-.1,.6):
        index=int(time*sr)
        y[index:index+80]=np.hanning(80)

    tempo=analyze_tempo(y,sr,duration,prefer_constant=True)

    assert np.std(np.diff(tempo.beats)) < 1e-8
    assert tempo.bpm == pytest.approx(100,abs=.2)
    assert any("stable studio grid" in warning for warning in tempo.warnings)
