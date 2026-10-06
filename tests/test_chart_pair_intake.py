import sys
from pathlib import Path

import numpy as np
import pretty_midi
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import chart_pair_sync_audit as sync_audit
import chart_pair_sync_consensus as sync_consensus
import extend_authorized_features as features
import prepare_debian_chart_pairs as debian_pairs


def test_offset_scan_recovers_known_frame_shift():
    envelope = np.zeros(40, dtype=float)
    envelope[[12, 22, 32]] = 1
    attacks = [1.0, 2.0, 3.0]

    result = sync_audit._offset_scan(
        envelope,
        attacks,
        sr=10,
        hop=1,
        limit=0.5,
    )

    assert result["best_offset_seconds"] == pytest.approx(0.1)
    assert result["zero_to_best_ratio"] == pytest.approx(0.0)


def test_near_one_to_one_sync_gate_is_stricter_than_admission():
    global_scan = {"best_offset_seconds": -0.046}
    windows = [
        {"best_offset_seconds": -0.058},
        {"best_offset_seconds": -0.046},
        {"best_offset_seconds": -0.046},
    ]
    nearest = np.asarray([0.01] * 50 + [0.04] * 50)

    criteria, near_one_to_one = sync_audit._sync_flags(
        global_scan,
        windows,
        nearest,
    )

    assert all(criteria.values())
    assert not all(near_one_to_one.values())
    assert all(isinstance(value, bool) for value in criteria.values())


def test_sync_consensus_groups_simultaneous_pitch_events():
    events = [
        {"start": 1.0},
        {"start": 1.02},
        {"start": 1.2},
    ]

    grouped = sync_consensus._group_event_times(events, "start")

    assert grouped.tolist() == [1.0, 1.2]


def test_sync_consensus_recovers_event_offset():
    scan = sync_consensus._event_offset_scan(
        [1.0, 2.0, 3.0],
        np.asarray([0.96, 1.96, 2.96]),
    )

    assert scan["best_offset_seconds"] == pytest.approx(-0.04)


def test_corrected_sync_gate_uses_regional_residuals():
    residuals, criteria, near_one_to_one = (
        sync_audit._corrected_sync_flags(
            {"best_offset_seconds": 0.07},
            [
                {"best_offset_seconds": 0.07},
                {"best_offset_seconds": 0.08},
                {"best_offset_seconds": 0.06},
            ],
            np.asarray([0.01] * 50),
        )
    )

    assert residuals == pytest.approx([0.0, 0.01, -0.01])
    assert all(criteria.values())
    assert all(near_one_to_one.values())


def test_debian_pack_assignments_keep_authors_whole_and_disjoint():
    assert debian_pairs.PACKS["sectoid"]["split"] == "train"
    assert debian_pairs.PACKS["muldjord"]["split"] == "validation"
    assert (
        debian_pairs.PACKS["sectoid"]["chart_author_id"]
        != debian_pairs.PACKS["muldjord"]["chart_author_id"]
    )


def test_authoring_example_converts_to_lane_independent_structure(tmp_path):
    midi_path = tmp_path / "notes.mid"
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    instrument = pretty_midi.Instrument(program=0)
    instrument.notes.append(pretty_midi.Note(
        velocity=100,
        pitch=60,
        start=0.0,
        end=2.0,
    ))
    midi.instruments.append(instrument)
    midi.write(str(midi_path))
    example = {
        "song_id": "fixture",
        "split_keys": ["artist:a", "chart-author:b"],
        "conditioning": [{
            "attack_id": "fixture:attack:0",
            "time_seconds": 0.5,
            "polyphony": 2,
        }],
        "expert_targets": [{
            "attack_id": "fixture:attack:0",
            "lanes": [1, 4],
            "lane_durations_seconds": [0.1, 0.3],
        }],
    }
    manifest = {
        "dataset": {
            "id": "fixture",
            "version": "1",
            "source_url": "https://example.invalid",
            "audio_license": "fixture",
            "annotation_license": "fixture",
            "commercial_use": "allowed",
            "attribution": "fixture",
        },
    }
    item = {
        "annotations": "charts/fixture.mid",
        "annotations_sha256": "a" * 64,
        "synchronization": {"audio_offset_seconds": 0.04},
    }

    labels = features._structure_labels(
        example,
        manifest,
        item,
        midi_path,
    )

    assert labels["targets"] == ["attack", "polyphony", "sustain"]
    assert [note["pitch"] for note in labels["notes"]] == [49, 52]
    assert labels["notes"][0]["start"] == pytest.approx(0.5)
    assert labels["notes"][1]["end"] == pytest.approx(0.8)
    assert labels["beats"][0]["time"] == pytest.approx(0.04)
    assert labels["attacks"][0]["polyphony"] == 2
    assert labels["split_keys"] == ["artist:a", "chart-author:b"]
