import numpy as np

from rb3charts.tabcnn import _decode_notes, _group_attacks, _sample_attack_states


def test_tabcnn_decodes_string_states_and_groups_chords():
    states = np.full((4, 6), 20, dtype=np.int64)
    states[1:3, 0] = 3
    states[1:3, 1] = 2
    probabilities = np.zeros((4, 6, 21), dtype=np.float32)
    probabilities[..., 20] = 1
    probabilities[1:3, 0, 3] = 0.9
    probabilities[1:3, 1, 2] = 0.8
    times = np.arange(4) * 512 / 22050
    notes = _decode_notes(
        states,
        probabilities,
        times,
        np.asarray([40, 45, 50, 55, 59, 64]),
        silence_class=20,
    )
    attacks = _group_attacks(notes)
    assert len(notes) == 2
    assert len(attacks) == 1
    assert attacks[0]["pitches"] == [43, 47]
    assert attacks[0]["polyphony"] == 2
    assert notes[0]["end"] == times[3]


def test_tabcnn_samples_nearby_frame_at_structure_attack():
    probabilities = np.zeros((5, 6, 21), dtype=np.float32)
    probabilities[..., 20] = 0.8
    probabilities[3, 0, 5] = 0.95
    probabilities[3, 0, 20] = 0.05
    probabilities[3, 1, 4] = 0.9
    probabilities[3, 1, 20] = 0.1
    rows = _sample_attack_states(
        probabilities,
        np.arange(5) * 0.02,
        [{"source_id": "attack-1", "start": 0.04, "attack_probability": 0.8}],
        np.asarray([40, 45, 50, 55, 59, 64]),
        radius=2,
    )
    assert rows[0]["source_id"] == "attack-1"
    assert rows[0]["tabcnn_frame_time"] == 0.06
    assert rows[0]["pitches"] == [45, 49]
    assert rows[0]["polyphony"] == 2
