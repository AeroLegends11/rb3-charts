import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import structure_score_ensemble as ensemble
import structure_timing_snap as timing_snap


def test_timing_snap_uses_local_maximum_and_deduplicates():
    scores = np.asarray([0.0, 0.2, 0.9, 0.1, 0.8, 0.0])

    assert timing_snap._snap([1, 3], scores, radius=1) == [2]
    assert timing_snap._snap([1, 4], scores, radius=1) == [2, 4]


def test_timing_metrics_report_frame_error_in_milliseconds():
    metrics = timing_snap._metrics(
        predicted=[10, 20],
        actual=np.asarray([11, 20]),
        tolerance_frames=1,
    )
    frame_ms = (
        timing_snap.FEATURE_HOP_LENGTH
        / timing_snap.FEATURE_SAMPLE_RATE
        * 1000
    )

    assert metrics["f1"] == 1
    assert metrics["median_timing_error_ms"] == pytest.approx(frame_ms / 2)
    assert metrics["p95_timing_error_ms"] == pytest.approx(frame_ms * 0.95)


def test_score_ensemble_blends_scores_before_peak_selection():
    records = [{
        "song_id": "fixture",
        "local_scores": np.asarray([0.0, 0.9, 0.0, 0.1, 0.0]),
        "context_scores": np.asarray([0.0, 0.1, 0.0, 0.9, 0.0]),
        "actual": np.asarray([1]),
    }]

    local = ensemble._summarize(
        records,
        local_weight=1.0,
        threshold=0.5,
        tolerance_frames=1,
    )
    context = ensemble._summarize(
        records,
        local_weight=0.0,
        threshold=0.5,
        tolerance_frames=1,
    )

    assert local["macro_f1"] == 1
    assert context["macro_f1"] == 0
