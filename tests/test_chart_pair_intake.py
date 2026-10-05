import sys
from pathlib import Path

import numpy as np
import pretty_midi
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import chart_pair_sync_audit as sync_audit
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
    }

    labels = features._structure_labels(
        example,
        manifest,
        item,
        midi_path,
    )

    assert labels["targets"] == ["attack", "polyphony", "sustain"]
    assert [note["pitch"] for note in labels["notes"]] == [49, 52]
    assert labels["notes"][1]["end"] == pytest.approx(0.8)
    assert labels["attacks"][0]["polyphony"] == 2
    assert labels["split_keys"] == ["artist:a", "chart-author:b"]
