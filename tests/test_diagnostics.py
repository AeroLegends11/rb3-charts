from copy import deepcopy
from dataclasses import asdict
import json
import pytest
from rb3charts.authoring import author_chart
from rb3charts.chart import (enforce_playability, conform_expert_to_reference,
                            refine_band_guitar, stabilize_repeated_phrases, reductions)
from rb3charts.phrases import author_guitar_phrases
from rb3charts.diagnostics import StageAudit, changes, repeat_families, repeat_metrics
from rb3charts.model import Note, TempoMap, save_json
from rb3charts.replay import replay


@pytest.mark.parametrize("instrument", ["guitar", "bass", "keys", "drums"])
@pytest.mark.parametrize("profile", ["band", "electronic"])
def test_shared_authoring_preserves_original_pipeline_order(instrument, profile):
    tempo = TempoMap([i * .5 for i in range(50)])
    notes = [Note(i * .25, i * .25 + .1, (i % 4,), .8, str(i), 48 + i % 4)
             for i in range(48)]
    notes[11].lanes = (0, 4)
    original = deepcopy(notes)
    # Frozen pre-extraction order: integration safeguard, not a second production path.
    expected, _ = enforce_playability(deepcopy(notes), instrument, tempo, profile,
                                     phrase_aware=instrument == "guitar")
    if profile == "band" and instrument in ("guitar", "bass", "keys"):
        expected, _ = conform_expert_to_reference(expected, tempo, instrument)
    if profile == "band" and instrument == "guitar":
        expected = refine_band_guitar(expected, tempo)
        expected, _ = stabilize_repeated_phrases(expected, tempo, instrument)
    if instrument == "guitar":
        expected, _ = author_guitar_phrases(expected, tempo)
    expected = reductions(expected, instrument, tempo)
    actual, _, audit = author_chart(notes, instrument, tempo, profile)
    assert actual == expected
    assert notes == original
    assert audit["stages"][0]["notes"] == [asdict(n) for n in original]
    assert audit["stages"][-1]["notes"] == [asdict(n) for n in actual["easy"]]


def test_snapshots_attribute_changes_without_mutation_or_identity_loss():
    tempo = TempoMap([0, .5, 1, 1.5])
    note = Note(.5, .6, (0,), source_id="a", pitch=48)
    audit = StageAudit(tempo, "guitar")
    audit.capture("mapped", [note])
    note.lanes = (2,)
    audit.capture("later", [note])
    report = audit.report()
    assert report["stages"][0]["notes"][0]["lanes"] == (0,)
    assert report["stages"][1]["changes"][0]["fields"] == {"lanes": {"before": (0,), "after": (2,)}}
    row = asdict(note)
    assert changes([row, row], [row])[0]["kind"] == "ambiguous_identity"
    assert changes([row], [])[0]["kind"] == "removed"


def test_repeat_metrics_distinguish_transposition_fret_shift_and_missing_notes():
    tempo = TempoMap([i * .5 for i in range(50)])
    rows = [asdict(Note(i * .25, i * .25 + .065, (i % 4,), source_id=str(i),
                        pitch=48 + i % 4 + (5 if i >= 8 else 0))) for i in range(12)]
    families = repeat_families(rows, tempo)
    perfect = repeat_metrics(rows, families)
    assert perfect["comparisons"] > 0
    assert perfect["absolute_lane_agreement"] == 1
    shifted = deepcopy(rows)
    for row in shifted[8:]:
        row["lanes"] = (row["lanes"][0] + 1,)
    result = repeat_metrics(shifted, families)
    assert result["absolute_lane_agreement"] < 1
    assert result["relative_shape_agreement"] == 1
    missing = repeat_metrics(rows[:3], families)
    assert missing["retained_occurrences"] == 0
    assert missing["absolute_lane_agreement"] is None
    assert missing["covered_attacks"] == 0


def test_cached_replay_is_deterministic_and_keeps_full_song_context(tmp_path):
    source = tmp_path / "saved"
    tempo = TempoMap([i * .5 for i in range(50)])
    raw = [{"start": i * .25, "end": i * .25 + .1, "pitch": 48 + i % 4,
            "confidence": .8, "source_id": str(i)} for i in range(40)]
    save_json(source / "analysis/guitar-transcription.json", raw)
    save_json(source / "analysis/tempo.json", tempo)
    save_json(source / "metadata/song.json", {"source_seconds": 15, "chart_profile": "band"})
    first = replay(source, tmp_path / "runs", end=3)
    second = replay(source, tmp_path / "runs")
    from pathlib import Path
    a, b = Path(first["folder"]), Path(second["folder"])
    assert a != b
    assert (a / "charts.json").read_bytes() == (b / "charts.json").read_bytes()
    stages = json.loads((a / "authoring-stages.json").read_text())
    assert stages["stages"][0]["metrics"]["attacks"] == 40
    assert stages["stages"][0]["region_metrics"]["attacks"] == 12
    provenance = json.loads((a / "provenance.json").read_text())
    assert provenance["input_hashes"]["events"]
    assert provenance["mode"] == "cached_post_voice_selection_authoring"
    assert json.loads((source / "analysis/guitar-transcription.json").read_text()) == raw
    with pytest.raises(ValueError):
        replay(source, tmp_path, start=4, end=3)
