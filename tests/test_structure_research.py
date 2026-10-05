import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import structure_score_ensemble as ensemble
import structure_timing_snap as timing_snap
import structure_local_peak_rerank as local_peak_rerank
import structure_candidate_classifier as candidate_classifier
import structure_candidate_sequence as candidate_sequence
import structure_cell_detector as cell_detector
import predicted_authoring_sequence as predicted_authoring
import predicted_authoring_self_train as self_trained_authoring
import audio_authoring_sequence as audio_authoring
import symbolic_authoring_sequence as symbolic_authoring


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


def test_local_peak_rerank_preserves_peak_times_and_samples_context_radius():
    local = np.asarray([0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0])
    context = np.asarray([0.0, 0.1, 0.9, 0.0, 0.0, 0.0, 0.2, 0.0])

    scores = local_peak_rerank._candidate_scores(
        local,
        context,
        local_weight=0.5,
        radius=1,
    )

    assert np.flatnonzero(scores).tolist() == [1, 6]
    assert scores[1] == pytest.approx(0.85)
    assert scores[6] == pytest.approx(0.45)


def test_local_peak_rerank_rejects_candidates_without_timing_matches():
    context = {
        "macro_f1": 0.7,
        "macro_recall": 0.7,
        "macro_median_timing_error_ms": 20.0,
    }
    local = {"macro_median_timing_error_ms": 10.0}
    candidates = [{
        "macro_f1": 0.8,
        "macro_precision": 0.9,
        "macro_recall": 0.8,
        "macro_median_timing_error_ms": None,
        "candidate_local_weight": 0.4,
        "context_radius_frames": 2,
    }]

    selected, guard_passed = (
        local_peak_rerank._select_validation_candidate(
            candidates,
            context,
            local,
        )
    )

    assert selected["candidate_local_weight"] == 0.4
    assert selected["context_radius_frames"] == 2
    assert guard_passed is False


def test_local_peak_rerank_does_not_evaluate_test_when_guard_fails():
    results = local_peak_rerank._test_evaluation(
        test=None,
        selected=None,
        context_threshold=None,
        local_threshold=None,
        tolerance_frames=None,
        guard_passed=False,
    )

    assert results == {}


def test_candidate_targets_match_each_reference_only_once():
    peaks = np.asarray([10, 14, 30])
    actual = np.asarray([12, 31])

    targets = candidate_classifier._candidate_targets(
        peaks,
        actual,
        tolerance_frames=4,
    )

    assert targets.tolist() == [1.0, 0.0, 1.0]


def test_candidate_features_keep_fixed_peak_rows():
    arrays = {
        "log_mel": np.zeros((12, 128), dtype=np.float32),
        "mel_delta": np.zeros((12, 128), dtype=np.float32),
        "beat_phase": np.zeros((12, 2), dtype=np.float32),
    }
    local = np.linspace(0.0, 1.0, 12, dtype=np.float32)
    context = local[::-1].copy()

    features = candidate_classifier._candidate_features(
        arrays,
        local,
        context,
        np.asarray([2, 9]),
    )

    assert features.shape == (2, 280)
    assert features[0, 4] == local[2]
    assert features[1, 13] == context[9]


def test_candidate_sequence_model_preserves_sequence_length():
    torch = pytest.importorskip("torch")
    model = candidate_sequence._build_model(input_features=8)

    logits = model(torch.zeros((2, 5, 8)))

    assert logits.shape == (2, 5)


def test_cell_detector_encodes_objectness_and_within_cell_phase():
    attacks = np.zeros(12, dtype=np.float32)
    attacks[[1, 7, 10]] = 1

    objectness, phase = cell_detector._cell_targets(attacks)

    assert objectness.tolist() == [1.0, 1.0, 1.0]
    assert phase.tolist() == [1, 3, 2]


def test_cell_detector_decodes_phase_and_suppresses_close_cells():
    frames = cell_detector._decode_cells(
        objectness=np.asarray([0.8, 0.9, 0.7]),
        phase=np.asarray([3, 0, 2]),
        threshold=0.5,
    )

    assert frames == [4, 10]


def test_cell_detector_model_outputs_one_row_per_four_frames():
    torch = pytest.importorskip("torch")
    model = cell_detector._build_model(input_features=6)

    outputs = model(torch.zeros((2, 11, 6)))

    assert outputs["objectness"].shape == (2, 3)
    assert outputs["phase"].shape == (2, 3, 4)


def test_cell_detector_packs_adjacent_frames_within_each_cell():
    torch = pytest.importorskip("torch")
    encoded = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)

    packed = cell_detector._pack_cells(encoded)

    assert packed[0, :, 0].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert packed[0, :, 1].tolist() == [4.0, 5.0, 6.0, 7.0]


def test_predicted_authoring_metrics_require_timing_and_lane_agreement():
    references = {
        "fixture": {
            "times": [1.0, 2.0],
            "masks": [1 << 1, (1 << 1) | (1 << 3)],
        },
    }
    predictions = [{
        "song_id": "fixture",
        "times": [1.01, 2.02, 3.0],
    }]

    metrics = predicted_authoring._timing_lane_metrics(
        references,
        predictions,
        [[1 << 1, (1 << 0) | (1 << 4), 1 << 2]],
    )

    assert metrics["attack_precision"] == pytest.approx(2 / 3)
    assert metrics["attack_recall"] == 1
    assert metrics["matched_exact_lanes"] == 0.5
    assert metrics["joint_exact_precision"] == pytest.approx(1 / 3)
    assert metrics["joint_exact_recall"] == 0.5


def test_predicted_authoring_context_features_summarize_fixed_windows():
    arrays = {
        "log_mel": np.tile(
            np.arange(10, dtype=np.float32)[:, None],
            (1, 128),
        ),
        "mel_delta": np.tile(
            np.arange(10, dtype=np.float32)[:, None],
            (1, 128),
        ),
    }

    features = predicted_authoring._context_audio_features(
        arrays,
        np.asarray([0, 5]),
    )

    assert features.shape == (2, 768)
    assert features[0, 0] == 0
    assert features[0, 256] == 2
    assert features[0, 384] == 4
    assert features[1, 512] == pytest.approx(4.5)
    assert features[1, 640] == 9


def test_predicted_authoring_pitch_contour_resets_after_rest():
    arrays = {
        "log_mel": np.full((4, 128), -20.0, dtype=np.float32),
        "mel_delta": np.zeros((4, 128), dtype=np.float32),
    }
    arrays["log_mel"][0, 10] = 0
    arrays["log_mel"][1, 30] = 0
    arrays["log_mel"][2, 50] = 0
    arrays["log_mel"][3, 70] = 0

    features = predicted_authoring._pitch_contour_features(
        arrays,
        np.asarray([0, 1, 2, 3]),
        [0.0, 0.2, 2.0, 2.2],
    )

    assert features.shape == (4, 5)
    assert features[1, 2] > 0
    assert features[2, 2] == 0
    assert features[:, 3].tolist() == [0.0, 1.0, 0.0, 1.0]


def test_self_trained_authoring_targets_only_matched_single_notes():
    reference = {
        "times": [1.0, 2.0, 3.0],
        "masks": [1 << 1, (1 << 1) | (1 << 3), 1 << 4],
    }
    predicted = {
        "times": [1.01, 2.01, 3.2],
        "symbolic_features": np.asarray([
            [0, 0, 0, 0.2, 0, 0],
            [0, 0, 0, 0.4, 0, 0],
            [0, 0, 0, 0.2, 0, 0],
        ], dtype=np.float32),
    }

    targets = self_trained_authoring._matched_single_targets(
        reference,
        predicted,
    )

    assert targets.tolist() == [
        symbolic_authoring.MASK_TO_CLASS[1 << 1],
        -1,
        -1,
    ]


def test_staged_authoring_changes_single_notes_but_preserves_chords():
    record = {
        "symbolic_features": np.asarray([
            [0, 0, 0, 0.2, 0, 0],
            [0, 0, 0, 0.4, 0, 0],
        ], dtype=np.float32),
        "features": np.asarray([
            [0, 0, 0, 0.2, 0, 0],
            [0, 0, 0, 0.4, 0, 0],
        ], dtype=np.float32),
    }

    staged = audio_authoring._staged_predictions(
        [record],
        [[1 << 4, (1 << 0) | (1 << 4)]],
    )[0]

    assert staged[0] == 1 << 4
    assert staged[1] == (1 << 1) | (1 << 3)


def test_authoring_metrics_separate_single_and_chord_accuracy():
    record = {
        "song_id": "fixture",
        "masks": [1 << 2, (1 << 1) | (1 << 3)],
        "times": [0.0, 0.5],
    }

    metrics = symbolic_authoring._metrics(
        [record],
        [[1 << 2, (1 << 0) | (1 << 4)]],
    )

    assert metrics["macro_single_exact"] == 1
    assert metrics["macro_chord_exact"] == 0
