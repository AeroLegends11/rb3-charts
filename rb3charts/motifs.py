from dataclasses import dataclass
from typing import NamedTuple
import numpy as np


MOTIF_VERSION = 1


@dataclass(frozen=True)
class MotifOccurrence:
    phrase_index: int
    source_ids: tuple[str, ...]
    start: float
    root_pitch: int
    relative_pitches: tuple[int, ...]
    beat_offsets: tuple[int, ...]
    contour: tuple[int, ...]
    confidences: tuple[float, ...]
    quality: float


class MotifMatch(NamedTuple):
    matched: bool
    reason: str
    score: float
    transposition: int
    alignment: tuple[tuple[int, int], ...]


def _contour(values):
    return tuple(0 if b == a else int(np.sign(b - a)) for a, b in zip(values, values[1:]))


def phrase_occurrence(phrase_index, notes, tempo):
    if len(notes) < 3 or any(len(note.lanes) != 1 for note in notes):
        return None
    base = int(round(float(notes[0].pitch)))
    relative = tuple(int(round(float(note.pitch))) - base for note in notes)
    beats = tuple(int(round((tempo.beat_at(note.start) - tempo.beat_at(notes[0].start)) * 12)) for note in notes)
    return MotifOccurrence(
        phrase_index=phrase_index,
        source_ids=tuple(note.source_id for note in notes),
        start=notes[0].start,
        root_pitch=base,
        relative_pitches=relative,
        beat_offsets=beats,
        contour=_contour(relative),
        confidences=tuple(float(note.confidence) for note in notes),
        quality=float(np.mean([note.confidence for note in notes])),
    )


def match_occurrences(a, b, max_insertions=1):
    if abs(len(a.relative_pitches) - len(b.relative_pitches)) > max_insertions:
        return MotifMatch(False, "length_delta", float("inf"), 0, ())
    ins = 2.25
    n, m = len(a.relative_pitches), len(b.relative_pitches)
    costs = np.full((n + 1, m + 1), float("inf"))
    links = [[None] * (m + 1) for _ in range(n + 1)]
    costs[0, 0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            cost = costs[i, j]
            if not np.isfinite(cost):
                continue
            if i < n and j < m:
                pitch_error = abs(a.relative_pitches[i] - b.relative_pitches[j]) * 0.8
                rhythm_error = abs(a.beat_offsets[i] - b.beat_offsets[j]) * 0.35
                step = cost + pitch_error + rhythm_error
                if step < costs[i + 1, j + 1]:
                    costs[i + 1, j + 1] = step
                    links[i + 1][j + 1] = (i, j, "diag")
            if i < n and cost + ins < costs[i + 1, j]:
                costs[i + 1, j] = cost + ins
                links[i + 1][j] = (i, j, "up")
            if j < m and cost + ins < costs[i, j + 1]:
                costs[i, j + 1] = cost + ins
                links[i][j + 1] = (i, j, "left")
    i, j = n, m
    alignment = []
    while i or j:
        link = links[i][j]
        if link is None:
            break
        pi, pj, move = link
        if move == "diag":
            alignment.append((pi, pj))
        i, j = pi, pj
    alignment = tuple(reversed(alignment))
    if len(alignment) < 3:
        return MotifMatch(False, "insufficient_overlap", float("inf"), 0, alignment)
    coverage = min(len(alignment) / max(1, n), len(alignment) / max(1, m))
    if coverage < 0.75:
        return MotifMatch(False, "coverage", float("inf"), 0, alignment)
    contour_match = 0
    contour_total = 0
    for (ai, aj), (bi, bj) in zip(alignment, alignment[1:]):
        first = 0 if a.relative_pitches[bi] == a.relative_pitches[ai] else int(np.sign(a.relative_pitches[bi] - a.relative_pitches[ai]))
        second = 0 if b.relative_pitches[bj] == b.relative_pitches[aj] else int(np.sign(b.relative_pitches[bj] - b.relative_pitches[aj]))
        contour_total += 1
        contour_match += first == second
    contour_total = max(1, contour_total)
    if contour_match / contour_total < 0.75:
        return MotifMatch(False, "contour", float("inf"), 0, alignment)
    strong_divergence = [
        (i, j)
        for i, j in alignment
        if abs(a.relative_pitches[i] - b.relative_pitches[j]) >= 3
        and min(a.confidences[i], b.confidences[j]) >= 0.75
    ]
    isolated_pitch_variation = (
        len(strong_divergence) == 1
        and len(alignment) >= 4
        and abs(
            a.relative_pitches[strong_divergence[0][0]]
            - b.relative_pitches[strong_divergence[0][1]]
        ) <= 4
    )
    if strong_divergence and not isolated_pitch_variation:
        return MotifMatch(False, "strong_variation", float("inf"), 0, alignment)
    score = float(costs[n, m] / len(alignment))
    if score > 1.2:
        return MotifMatch(False, "score", score, 0, alignment)
    return MotifMatch(
        True,
        "matched_motif_with_pitch_variation" if isolated_pitch_variation else "matched_relative_motif",
        score,
        int(round(b.root_pitch - a.root_pitch)),
        alignment,
    )


def discover_phrase_motifs(phrases, tempo):
    occurrences = {}
    for phrase_index, notes in enumerate(phrases):
        occurrence = phrase_occurrence(phrase_index, notes, tempo)
        if occurrence is not None:
            occurrences[phrase_index] = occurrence
    if len(occurrences) < 2:
        return {}
    parents = {index: index for index in occurrences}
    links = {}

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(a_index, b_index):
        ra, rb = find(a_index), find(b_index)
        if ra != rb:
            if ra < rb:
                parents[rb] = ra
            else:
                parents[ra] = rb

    indices = sorted(occurrences)
    for offset, a_index in enumerate(indices):
        for b_index in indices[offset + 1:]:
            match = match_occurrences(occurrences[a_index], occurrences[b_index])
            if not match.matched:
                continue
            links[(a_index, b_index)] = match
            links[(b_index, a_index)] = MotifMatch(True, match.reason, match.score, -match.transposition,
                                                   tuple((j, i) for i, j in match.alignment))
            union(a_index, b_index)
    families = {}
    for index in indices:
        families.setdefault(find(index), []).append(index)
    assignments = {}
    family_counter = 0
    for member_indices in (sorted(members) for members in families.values() if len(members) >= 2):
        family_counter += 1
        canonical = max(
            member_indices,
            key=lambda idx: (
                occurrences[idx].quality,
                len(occurrences[idx].relative_pitches),
                -occurrences[idx].start,
                -idx,
            ),
        )
        for index in member_indices:
            if index == canonical:
                alignment = tuple((i, i) for i in range(len(occurrences[index].relative_pitches)))
                reason = "canonical"
                score = 0.0
                transposition = 0
            else:
                match = links.get((canonical, index))
                if match is None or not match.matched:
                    continue
                alignment = match.alignment
                reason = match.reason
                score = match.score
                transposition = match.transposition
            assignments[index] = {
                "version": MOTIF_VERSION,
                "family_id": family_counter,
                "canonical_phrase": canonical,
                "alignment": alignment,
                "reason": reason,
                "score": score,
                "transposition": transposition,
                "source_ids": occurrences[index].source_ids,
            }
    return assignments
