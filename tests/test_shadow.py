import pytest

from rb3charts.model import Note
from rb3charts.shadow import apply_fretnet_note_shadow


def _payload(runs):
    return {
        "schema": 1,
        "kind": "fretnet_shadow_evidence",
        "pitch_runs": runs,
    }


def _run(source_id, onset, offset, pitch):
    return {
        "source_id": source_id,
        "onset_s": onset,
        "offset_s": offset,
        "midi_pitch": pitch,
    }


def test_fretnet_shadow_uses_upper_median_on_single_note_events():
    notes = [
        Note(index * 0.25, index * 0.25 + 0.1, (2,), source_id=str(index), pitch=48)
        for index in range(12)
    ]
    payload = _payload([
        shadow
        for index in range(12)
        for shadow in (
            _run(f"low-{index}", index * 0.25, index * 0.25 + 0.1, 55),
            _run(f"high-{index}", index * 0.25, index * 0.25 + 0.1, 59),
        )
    ])

    result, report = apply_fretnet_note_shadow(notes, payload)

    assert result[0].pitch == 59
    assert result[0].lanes == (2,)
    assert report["assignments"][0]["match"] == "onset_change"
    assert report["assignments"][0]["original_pitch"] == 48
    assert report["changed"] == 12
    assert report["repetition_supported"] == 12


def test_fretnet_shadow_preserves_polyphonic_source_events():
    notes = [Note(1.0, 1.1, (1, 3), source_id="a+b", pitch=48)]

    result, report = apply_fretnet_note_shadow(
        notes,
        _payload([_run("shadow", 0.9, 1.2, 60)]),
    )

    assert result[0].pitch == 48
    assert result[0].lanes == (1, 3)
    assert report["chords_preserved"] == 1
    assert report["changed"] == 0


def test_fretnet_shadow_rejects_unknown_schema():
    with pytest.raises(ValueError, match="Unsupported"):
        apply_fretnet_note_shadow([], {"schema": 2, "pitch_runs": []})
