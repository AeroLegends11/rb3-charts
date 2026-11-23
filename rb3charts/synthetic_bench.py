"""Deterministic synthetic benchmark for guitar phrase authoring.

It uses generated melodies only, so it measures lane-authoring behavior and says
nothing about transcription, musical quality or playability on real songs.
"""
import random

import numpy as np

from .model import Note, TempoMap
from .phrases import author_guitar_phrases

RAPID_SECONDS = .8


def _melody(rng, trial, pitch_count):
    base = rng.choice([40, 52, 60])
    scale = sorted(rng.sample(range(19), pitch_count))
    t, notes = 1.0, []
    for i in range(rng.randint(8, 40)):
        t += rng.choice([.125, .25, .5])
        notes.append(Note(t, t + .1, (2,), .8, f"{trial}-{i}", base + rng.choice(scale)))
    return notes


def _repeated_motifs(rng, trial):
    """The same rhythm/contour played at several starting pitches, separated by rests."""
    shape = [rng.choice([-2, -1, 1, 2, 3]) for _ in range(5)]
    steps = [0]
    for step in shape:
        steps.append(steps[-1] + step)
    notes, groups, t = [], [], 1.0
    for repeat in range(4):
        root = 55 + rng.choice([0, 2, 5, 7])
        group = [Note(t + i * .125, t + i * .125 + .08, (2,), .8, f"{trial}-{repeat}-{i}", root + step)
                 for i, step in enumerate(steps)]
        notes += group
        groups.append(group)
        t += 3.0
    return notes, groups


def run_benchmark(seed=7, trials=300, pitch_range=(6, 9)):
    """Return counts for same-color pitch changes, reversals, big hops and repeat agreement."""
    rng = random.Random(seed)
    tempo = TempoMap(np.arange(0, 400, .5).tolist())
    totals = dict(notes=0, rapid_pairs=0, collapsed=0, reversed=0, large_hops=0,
                  repeat_groups=0, repeat_groups_identical=0, out_of_range=0, attacks_changed=0)
    for trial in range(trials):
        notes = _melody(rng, trial, rng.randint(*pitch_range))
        result, _ = author_guitar_phrases(notes, tempo)
        totals["notes"] += len(notes)
        totals["attacks_changed"] += sum((a.start, a.source_id, a.pitch) != (b.start, b.source_id, b.pitch)
                                         for a, b in zip(notes, result))
        totals["out_of_range"] += sum(not 0 <= lane <= 4 for note in result for lane in note.lanes)
        for a, b in zip(result, result[1:]):
            if b.start - a.start >= RAPID_SECONDS:
                continue
            totals["rapid_pairs"] += 1
            if a.pitch != b.pitch:
                totals["collapsed"] += a.lanes == b.lanes
                totals["reversed"] += (b.pitch - a.pitch) * (b.lanes[0] - a.lanes[0]) < 0
                totals["large_hops"] += abs(b.lanes[0] - a.lanes[0]) >= 3 and abs(b.pitch - a.pitch) <= 3
    for trial in range(trials // 3):
        notes, groups = _repeated_motifs(rng, trial)
        result, _ = author_guitar_phrases(notes, tempo)
        by_id = {note.source_id: note.lanes for note in result}
        shapes = [tuple(by_id[note.source_id] for note in group) for group in groups]
        totals["repeat_groups"] += len(shapes)
        totals["repeat_groups_identical"] += sum(shape == shapes[0] for shape in shapes)
    return totals
