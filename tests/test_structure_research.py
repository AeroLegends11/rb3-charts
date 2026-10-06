import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import structure_score_ensemble as ensemble
import structure_timing_snap as timing_snap
import structure_local_peak_rerank as local_peak_rerank
import structure_candidate_classifier as candidate_classifier
import structure_candidate_sequence as candidate_sequence
import structure_cell_detector as cell_detector
import structure_beat_grid as beat_grid
import structure_beat_finetune as beat_finetune
import structure_beat_occupancy as beat_occupancy
import structure_multires_spectrogram as multires_spectrogram
import structure_short_distillation as short_distillation
import structure_distill_ensemble as distill_ensemble
import structure_highres_timing as highres_timing
import structure_distill_variance as distill_variance
import structure_timing_bias_diagnostic as timing_bias
import structure_hop128_finetune as hop128
import structure_polyphony_refit as polyphony_refit
import predicted_authoring_sequence as predicted_authoring
import predicted_authoring_shadow as authoring_shadow
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


def test_authoring_shadow_applies_fixed_offset_and_decodes_masks():
    attacks = authoring_shadow._predicted_attacks(
        {"times": [0.5, 1.5]},
        [1, 20],
        offset=-1.0,
    )

    assert attacks == [{
        "time": 0.5,
        "duration": 0.1,
        "lanes": [2, 4],
    }]


def test_authoring_shadow_reports_fixed_tolerances():
    reference = [{
        "time": 1.0,
        "duration": 0.1,
        "lanes": [2],
    }]
    candidate = [{
        "time": 1.04,
        "duration": 0.1,
        "lanes": [2],
    }]

    metrics = authoring_shadow._metrics(reference, candidate)

    assert metrics["25"]["attack"]["matched"] == 0
    assert metrics["50"]["attack"]["matched"] == 1


def test_highres_timing_matches_each_reference_once():
    pairs = highres_timing._match_times(
        reference=[1.0, 1.1],
        predicted=[1.02, 1.04, 1.11],
        tolerance=0.05,
    )

    assert pairs == [(0, 0), (2, 1)]


def test_highres_timing_patches_center_on_requested_times():
    feature_map = np.tile(
        np.arange(100, dtype=np.float32),
        (highres_timing.HIGHRES_MELS * 2, 1),
    )
    center = 20
    time = (
        center
        * highres_timing.HIGHRES_HOP_LENGTH
        / highres_timing.FEATURE_SAMPLE_RATE
    )

    patches = highres_timing._timing_patches(feature_map, [time])

    assert patches.shape == (
        1,
        highres_timing.HIGHRES_MELS * 2,
        2 * highres_timing.PATCH_RADIUS + 1,
    )
    assert patches[0, 0, highres_timing.PATCH_RADIUS] == center


def test_highres_timing_metrics_use_exact_seconds():
    metrics = highres_timing._timing_metrics([{
        "song_id": "fixture",
        "reference": np.asarray([1.0]),
        "predicted": np.asarray([1.01]),
    }])

    assert metrics["macro_f1"] == 1
    assert metrics["macro_median_timing_error_ms"] == pytest.approx(10)


def test_audio_authoring_movement_head_preserves_mask_logits():
    import torch

    model = audio_authoring._build_model(12, movement_head=True)
    features = torch.zeros((1, 4, 12))
    encoded = model.encode(features)

    assert model(features).shape == (1, 4, len(symbolic_authoring.MASKS))
    assert model.movement_logits(encoded).shape == (1, 3, 3)


def test_lane_confidence_gate_preserves_existing_support():
    records = [{
        "learned_lane_allowed": [True, False, True],
    }]

    gated = predicted_authoring._confidence_gated_records(
        records,
        [[0.8, 0.9, 0.2]],
        threshold=0.5,
    )

    assert gated[0]["learned_lane_allowed"] == [True, False, False]


def test_lane_confidence_mean_handles_variable_song_lengths():
    averaged = predicted_authoring._mean_confidences([
        [[1.0, 3.0], [2.0]],
        [[3.0, 5.0], [4.0]],
    ])

    assert averaged == [[2.0, 4.0], [3.0]]


def test_logit_lane_ensemble_averages_before_decoding():
    import torch

    class Fixed(torch.nn.Module):
        def __init__(self, lane, strength):
            super().__init__()
            self.lane = lane
            self.strength = strength

        def forward(self, features):
            logits = torch.zeros(
                (1, features.shape[1], len(symbolic_authoring.MASKS)),
                device=features.device,
            )
            logits[:, :, self.lane] = self.strength
            return logits

    record = {
        "features": np.zeros((1, 2), dtype=np.float32),
        "symbolic_features": np.asarray(
            [[0, 0, 0, 0.2, 0, 0]],
            dtype=np.float32,
        ),
    }
    masks, margins = audio_authoring._predict_logit_ensemble(
        [Fixed(0, 1), Fixed(1, 3)],
        [record],
        np.zeros(2),
        np.ones(2),
        "cpu",
    )

    assert masks == [[2]]
    assert margins[0][0] == pytest.approx(1)


def test_phrase_rank_decoder_preserves_chords_and_support_gate():
    records = [{
        "symbolic_features": np.asarray([
            [0, 0, 0, 0.2, 0, 0],
            [0, 0, 0, 0.2, 0, 0],
            [0, 0, 0, 0.4, 0, 0],
        ], dtype=np.float32),
        "features": np.asarray([
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.5],
        ], dtype=np.float32),
        "learned_lane_allowed": [True, False, True],
    }]

    masks = predicted_authoring._phrase_rank_predictions(
        records,
        0,
        4,
    )

    assert masks == [[1, 4, 10]]


def test_distill_variance_gate_uses_teacher_on_disagreement():
    teacher = np.asarray([0.2, 0.8])
    students = [
        np.asarray([0.4, 0.1]),
        np.asarray([0.4, 0.9]),
        np.asarray([0.4, 0.5]),
    ]

    scores = distill_variance._variance_gated_scores(
        teacher,
        students,
        student_weight=0.2,
        variance_threshold=0.001,
    )

    assert scores[0] == pytest.approx(0.24)
    assert scores[1] == pytest.approx(0.8)


def test_timing_bias_diagnostic_preserves_error_sign():
    metrics = timing_bias._signed_metrics({
        "reference": np.asarray([1.01, 1.98]),
        "predicted": np.asarray([1.0, 2.0]),
    })

    assert metrics["matches"] == 2
    assert metrics["signed_mean_ms"] == pytest.approx(-5)
    assert metrics["median_absolute_ms"] == pytest.approx(15)


def test_hop128_targets_use_exact_nearest_frame():
    times = [
        10.4 * hop128.HIGHRES_HOP_LENGTH / hop128.FEATURE_SAMPLE_RATE,
        20.6 * hop128.HIGHRES_HOP_LENGTH / hop128.FEATURE_SAMPLE_RATE,
    ]

    frames = hop128._target_frames(times, frames=100)

    assert frames.tolist() == [10, 21]


def test_hop128_metrics_use_configured_time_resolution():
    records = [{
        "song_id": "fixture",
        "scores": np.asarray([0.0, 0.9, 0.0]),
        "reference": np.asarray([
            1.5 * hop128.HIGHRES_HOP_LENGTH / hop128.FEATURE_SAMPLE_RATE,
        ]),
        "hop_length": hop128.HIGHRES_HOP_LENGTH,
    }]

    metrics = hop128._score_metrics(records, threshold=0.5)

    assert metrics["macro_f1"] == 1
    assert metrics["macro_median_timing_error_ms"] == pytest.approx(
        0.5
        * hop128.HIGHRES_HOP_LENGTH
        / hop128.FEATURE_SAMPLE_RATE
        * 1000
    )


def test_hop128_resolution_blend_interpolates_coarse_scores():
    records = [{
        "scores": np.asarray([0.0, 0.0, 0.0]),
        "coarse_scores": np.asarray([0.0, 1.0]),
    }]

    blended = hop128._blend_resolution_records(
        records,
        highres_weight=0.5,
    )

    assert blended[0]["scores"].tolist() == [0.0, 0.25, 0.5]


def test_timing_lane_metrics_report_polyphony_separately():
    references = {
        "fixture": {
            "times": [1.0, 2.0],
            "masks": [1, 3],
        },
    }
    predicted = [{
        "song_id": "fixture",
        "times": [1.0, 2.0],
    }]

    metrics = predicted_authoring._timing_lane_metrics(
        references,
        predicted,
        [[2, 5]],
    )

    assert metrics["matched_exact_lanes"] == 0
    assert metrics["matched_exact_polyphony"] == 1


def test_polyphony_refit_metrics_report_confusion():
    import torch

    class Fixed(torch.nn.Module):
        def forward(self, features):
            logits = torch.zeros((1, 2, 7))
            logits[0, 0, 1] = 2
            logits[0, 1, 2] = 2
            return {"polyphony": logits}

    metrics = polyphony_refit._polyphony_metrics(
        Fixed(),
        [{
            "features": np.zeros((2, 3), dtype=np.float32),
            "attack": np.ones(2, dtype=np.float32),
            "polyphony": np.asarray([1, 3]),
        }],
        "cpu",
    )

    assert metrics["accuracy"] == 0.5
    assert metrics["confusion"] == {"1->1": 1, "3->2": 1}


def test_lane_viterbi_uses_transition_evidence():
    unary = np.asarray([
        [2.0, 1.9, 0.0, 0.0, 0.0],
        [1.9, 2.0, 0.0, 0.0, 0.0],
    ])
    transition = np.full((5, 5), -10.0)
    np.fill_diagonal(transition, 0.0)

    lanes = audio_authoring._viterbi_single_lanes(
        unary,
        transition,
        weight=1.0,
    )

    assert lanes == [0, 0]


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


def test_beat_grid_interpolates_twelve_positions_per_beat():
    times = beat_grid._grid_times([0.0, 0.6, 1.2])

    assert len(times) == 25
    assert times[3] == pytest.approx(0.15)
    assert times[18] == pytest.approx(0.9)


def test_beat_grid_targets_preserve_exact_grid_attacks():
    grid = beat_grid._grid_times([0.0, 0.6, 1.2])
    targets, errors, indices = beat_grid._grid_targets(
        grid,
        [0.15, 0.9],
    )

    assert np.flatnonzero(targets).tolist() == [3, 18]
    assert indices.tolist() == [3, 18]
    assert errors.tolist() == pytest.approx([0.0, 0.0])


def test_beat_grid_selection_suppresses_adjacent_subdivisions():
    probabilities = np.asarray([0.0, 0.8, 0.9, 0.0, 0.7])
    times = np.asarray([0.0, 0.03, 0.06, 0.09, 0.12])

    selected = beat_grid._select_predictions(
        probabilities,
        times,
        threshold=0.5,
    )

    assert selected == pytest.approx([0.06, 0.12])


def test_beat_finetune_samples_primary_model_at_grid_frames():
    torch = pytest.importorskip("torch")

    class FixtureModel(torch.nn.Module):
        def forward(self, features):
            frames = features.shape[1]
            logits = torch.arange(
                frames,
                dtype=torch.float32,
                device=features.device,
            )[None, :]
            return {
                "attack": logits,
                "polyphony": torch.zeros((1, frames, 7)),
                "sustain": torch.zeros((1, frames, 7)),
            }

    records = [{
        "song_id": "fixture",
        "frame_features": np.zeros((5, 2), dtype=np.float32),
        "grid_frames": np.asarray([1, 3]),
    }]
    predicted = beat_finetune._predict(
        FixtureModel(),
        records,
        "cpu",
    )

    assert predicted[0]["probabilities"].tolist() == pytest.approx([
        1 / (1 + np.exp(-1)),
        1 / (1 + np.exp(-3)),
    ])


def test_beat_occupancy_encodes_counts_and_subdivisions():
    occupancy, errors = beat_occupancy._beat_targets(
        [0.0, 0.6, 1.2],
        [0.15, 0.3, 0.9],
    )

    assert np.flatnonzero(occupancy[0]).tolist() == [3, 6]
    assert np.flatnonzero(occupancy[1]).tolist() == [6]
    assert errors.tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_beat_occupancy_decoder_uses_predicted_count():
    count_probabilities = np.asarray([
        [0.0, 0.1, 0.9, 0.0, 0.0],
    ])
    subdivision_scores = np.zeros((1, 12), dtype=np.float32)
    subdivision_scores[0, [3, 7]] = [0.8, 0.9]

    predicted = beat_occupancy._decode_beats(
        count_probabilities,
        subdivision_scores,
        np.asarray([0.0, 0.6]),
    )

    assert predicted == pytest.approx([0.15, 0.35])


def test_multires_spectrogram_preserves_expected_frame_count(
    tmp_path,
):
    samples = max(multires_spectrogram.FFT_SIZES)
    sf.write(
        tmp_path / "fixture.wav",
        np.zeros(samples, dtype=np.float32),
        multires_spectrogram.FEATURE_SAMPLE_RATE,
    )
    expected_frames = (
        1 + samples // multires_spectrogram.FEATURE_HOP_LENGTH
    )
    features = multires_spectrogram._multires_features(
        tmp_path / "fixture.wav",
        np.zeros((expected_frames, 2), dtype=np.float32),
        expected_frames,
    )

    assert features.shape == (
        expected_frames,
        len(multires_spectrogram.FFT_SIZES)
        * multires_spectrogram.MELS_PER_SCALE
        + 2,
    )


def test_short_spectrogram_augments_existing_representation(tmp_path):
    samples = np.zeros(4096, dtype=np.float32)
    sf.write(
        tmp_path / "fixture.wav",
        samples,
        multires_spectrogram.FEATURE_SAMPLE_RATE,
    )
    frames = 1 + len(samples) // multires_spectrogram.FEATURE_HOP_LENGTH
    arrays = {
        "log_mel": np.zeros((frames, 128), dtype=np.float32),
        "mel_delta": np.zeros((frames, 128), dtype=np.float32),
        "beat_phase": np.zeros((frames, 2), dtype=np.float32),
        "attack_target": np.zeros(frames, dtype=np.float32),
    }

    features = multires_spectrogram._short_augmented_features(
        tmp_path / "fixture.wav",
        arrays,
    )

    assert features.shape == (
        frames,
        128 + 128 + multires_spectrogram.SHORT_MELS + 2,
    )


def test_short_primary_replaces_long_window_log_mel(tmp_path):
    samples = np.zeros(4096, dtype=np.float32)
    sf.write(
        tmp_path / "fixture.wav",
        samples,
        multires_spectrogram.FEATURE_SAMPLE_RATE,
    )
    frames = 1 + len(samples) // multires_spectrogram.FEATURE_HOP_LENGTH
    arrays = {
        "mel_delta": np.zeros((frames, 128), dtype=np.float32),
        "beat_phase": np.zeros((frames, 2), dtype=np.float32),
        "attack_target": np.zeros(frames, dtype=np.float32),
    }

    features = multires_spectrogram._short_primary_features(
        tmp_path / "fixture.wav",
        arrays,
    )

    assert features.shape == (frames, 258)


def test_short_consistent_representation_derives_short_delta(tmp_path):
    samples = np.zeros(4096, dtype=np.float32)
    samples[2048:] = 0.5
    sf.write(
        tmp_path / "fixture.wav",
        samples,
        multires_spectrogram.FEATURE_SAMPLE_RATE,
    )
    frames = 1 + len(samples) // multires_spectrogram.FEATURE_HOP_LENGTH
    arrays = {
        "mel_delta": np.zeros((frames, 128), dtype=np.float32),
        "beat_phase": np.zeros((frames, 2), dtype=np.float32),
        "attack_target": np.zeros(frames, dtype=np.float32),
    }

    features = multires_spectrogram._short_primary_features(
        tmp_path / "fixture.wav",
        arrays,
        short_delta=True,
    )

    assert np.max(features[:, 128:256]) > 0


def test_dual_branch_model_separates_short_and_long_features():
    torch = pytest.importorskip("torch")
    model = multires_spectrogram._build_dual_branch()
    outputs = model(torch.zeros((2, 10, 514)))

    assert outputs["attack"].shape == (2, 10)
    assert outputs["polyphony"].shape == (2, 10, 7)


def test_short_distillation_uses_short_primary_representation(tmp_path):
    samples = np.zeros(4096, dtype=np.float32)
    sf.write(
        tmp_path / "fixture.wav",
        samples,
        short_distillation.FEATURE_SAMPLE_RATE,
    )
    frames = 1 + len(samples) // short_distillation.FEATURE_HOP_LENGTH
    arrays = {
        "mel_delta": np.zeros((frames, 128), dtype=np.float32),
        "beat_phase": np.zeros((frames, 2), dtype=np.float32),
        "attack_target": np.zeros(frames, dtype=np.float32),
    }

    features = short_distillation._short_primary_features(
        tmp_path / "fixture.wav",
        arrays,
        short_delta=True,
    )

    assert features.shape == (frames, 258)


def test_multires_macro_metrics_handle_threshold_without_matches():
    metrics = multires_spectrogram._macro_metrics(
        [("fixture", np.asarray([0.0, 0.1, 0.0]), [1])],
        threshold=0.9,
        tolerance_frames=1,
    )

    assert metrics["macro_f1"] == 0
    assert metrics["macro_median_timing_error_ms"] is None
    assert metrics["macro_p95_timing_error_ms"] is None


def test_distill_ensemble_rejects_wrong_checkpoint_schema(tmp_path):
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "wrong.pt"
    torch.save({"schema": "wrong", "schema_version": 1}, checkpoint)

    with pytest.raises(ValueError, match="Unsupported distilled checkpoint"):
        distill_ensemble._load_student(checkpoint, "cpu")


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


def test_predicted_authoring_blends_mean_student_attack_scores():
    scores = predicted_authoring._blend_attack_scores(
        teacher_scores=np.asarray([0.2, 0.8]),
        student_scores=[
            np.asarray([0.6, 0.4]),
            np.asarray([0.8, 0.2]),
        ],
        student_weight=0.25,
    )

    assert scores == pytest.approx([0.325, 0.675])


def test_predicted_authoring_preserves_baseline_on_unsupported_lane():
    records = [{
        "symbolic_features": np.asarray([
            [0, 0, 0, 0.2, 0, 0],
            [0, 0, 0, 0.2, 0, 0],
        ], dtype=np.float32),
        "features": np.asarray([
            [0, 0, 0, 0.2, 0, 0],
            [0, 0, 0, 0.2, 0, 0],
        ], dtype=np.float32),
        "learned_lane_allowed": [True, False],
    }]

    staged = predicted_authoring._supported_staged_predictions(
        records,
        [[1 << 4, 1 << 4]],
    )

    assert staged == [[1 << 4, 1 << 2]]


def test_predicted_authoring_majority_uses_baseline_for_three_way_tie():
    records = [{
        "symbolic_features": np.asarray([
            [0, 0, 0, 0.2, 0, 0],
            [0, 0, 0, 0.2, 0, 0],
        ], dtype=np.float32),
    }]

    majority = predicted_authoring._majority_predictions(
        records,
        [
            [[1 << 0, 1 << 0]],
            [[1 << 0, 1 << 1]],
            [[1 << 4, 1 << 4]],
        ],
    )

    assert majority == [[1 << 0, 1 << 2]]


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
