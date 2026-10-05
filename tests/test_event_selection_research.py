import json
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import event_selection_ensemble as ensemble
import event_selection_pilot as pilot
import phrase_voice_ownership_pilot as voice


def _event(start, pitch, score, accepted=False, source_id="source"):
    return {
        "start": start,
        "end": start + 0.1,
        "pitch": pitch,
        "confidence": score,
        "accepted": accepted,
        "source_id": source_id,
        "evidence": {
            "evidence_score": score,
            "attack_strength": score,
            "relative_level": score,
            "rms": score,
        },
    }


def test_candidate_groups_exclude_derived_and_compound_sources(tmp_path):
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({
        "events": [
            _event(1.0, 48, 0.8, accepted=True),
            _event(1.02, 52, 0.7),
            {
                **_event(1.01, 55, 1.0),
                "classification": "derived",
            },
            _event(2.0, 57, 1.0, source_id="left+right"),
        ],
    }), encoding="utf-8")

    groups = pilot._candidate_groups(trace)

    assert len(groups) == 1
    assert groups[0]["time"] == pytest.approx(1.01)
    assert groups[0]["accepted"] is True
    assert groups[0]["features"][9] == 2


def test_match_targets_assigns_each_reference_once():
    groups = [
        {"time": 1.0, "events": [_event(1.0, 48, 0.4)]},
        {"time": 1.04, "events": [_event(1.04, 50, 0.9)]},
    ]

    labels, offsets = pilot._match_targets(groups, [1.02])

    assert labels.tolist() == [0.0, 1.0]
    assert offsets[1] == pytest.approx(-0.02)


def test_timing_only_metrics_preserve_selection_when_matches_do_not_change():
    record = {
        "song_id": "fixture",
        "times": np.asarray([1.0, 2.0], dtype=np.float32),
        "scores": np.asarray([1.0, 0.0], dtype=np.float32),
        "corrections": np.asarray([0.02, 0.05], dtype=np.float32),
        "reference": [1.03, 2.0],
    }

    baseline = pilot._metrics([record], 0.5, False)
    corrected = pilot._metrics([record], 0.5, True)

    for key in ("macro_precision", "macro_recall", "macro_f1"):
        assert corrected[key] == baseline[key]
    assert corrected["songs"][0]["selected"] == baseline["songs"][0]["selected"]
    assert (
        corrected["macro_median_timing_error_ms"]
        < baseline["macro_median_timing_error_ms"]
    )


def test_ensemble_rejects_unsupported_checkpoint():
    with pytest.raises(ValueError, match="Unsupported event-selection checkpoint"):
        ensemble._score([], [{"schema": "wrong"}])


def test_voice_path_resets_after_a_rest():
    groups = [
        {"time": 0.0, "events": [_event(0.0, 40, 1.0), _event(0.0, 70, 0.1)]},
        {"time": 0.2, "events": [_event(0.2, 42, 1.0), _event(0.2, 72, 0.1)]},
        {"time": 2.0, "events": [_event(2.0, 70, 1.0), _event(2.0, 40, 0.1)]},
    ]

    selected = voice._select_path(
        groups,
        evidence_weight=1.0,
        jump_weight=1.0,
        octave_penalty=1.0,
    )

    assert selected == [40, 42, 70]
