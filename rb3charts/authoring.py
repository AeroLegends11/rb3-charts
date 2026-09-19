"""Shared chart-authoring order for generation and controlled cached replay."""
from dataclasses import replace
from .chart import (enforce_playability, conform_expert_to_reference,
                    refine_band_guitar, stabilize_repeated_phrases, reductions)
from .phrases import author_guitar_phrases
from .diagnostics import StageAudit


def author_chart(notes, instrument, tempo, profile):
    """Author already mapped, duration-clipped notes; retain immutable snapshots.

    This is the existing pipeline order. Transcription, voice selection, initial
    mapping and final drum activation alignment are outside this function.
    """
    notes = [replace(note) for note in notes]
    audit = StageAudit(tempo, instrument)
    audit.capture("mapped", notes)
    reports = {}
    notes, reports["playability"] = enforce_playability(
        notes, instrument, tempo, profile, phrase_aware=instrument == "guitar")
    audit.capture("playability", notes)
    if profile == "band" and instrument in ("guitar", "bass", "keys"):
        notes, reports["reference_conformance"] = conform_expert_to_reference(notes, tempo, instrument)
        audit.capture("reference_conformance", notes)
    if profile == "band" and instrument == "guitar":
        notes = refine_band_guitar(notes, tempo)
        audit.capture("band_refinement", notes)
        notes, reports["pattern_stabilization"] = stabilize_repeated_phrases(notes, tempo, instrument)
        audit.capture("pattern_stabilization", notes)
    if instrument == "guitar":
        notes, reports["phrase_authoring"] = author_guitar_phrases(notes, tempo)
        audit.capture("phrase_authoring", notes)
    charts = reductions(notes, instrument, tempo)
    for difficulty in ("expert", "hard", "medium", "easy"):
        audit.capture(difficulty, charts[difficulty])
    return charts, reports, audit.report()
