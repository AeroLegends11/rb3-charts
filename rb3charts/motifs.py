from dataclasses import dataclass
from typing import NamedTuple
import numpy as np


MOTIF_VERSION = 2


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


@dataclass(frozen=True)
class MotifWindow:
    phrase_index: int
    start_index: int
    stop_index: int
    occurrence: MotifOccurrence


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
    if strong_divergence:
        return MotifMatch(False, "strong_variation", float("inf"), 0, alignment)
    score = float(costs[n, m] / len(alignment))
    if score > 1.2:
        return MotifMatch(False, "score", score, 0, alignment)
    return MotifMatch(
        True,
        "matched_relative_motif",
        score,
        int(round(b.root_pitch - a.root_pitch)),
        alignment,
    )


def _window_candidate_keys(window):
    pitches = window.occurrence.relative_pitches
    beats = window.occurrence.beat_offsets
    representations = [(pitches, beats)]
    if len(pitches) > 3:
        representations.extend(
            (
                pitches[:index] + pitches[index + 1:],
                beats[:index] + beats[index + 1:],
            )
            for index in range(len(pitches))
        )
    keys = set()
    for values, offsets in representations:
        contour = _contour(values)
        contours = [contour]
        if len(values) == 5:
            contours.extend(
                contour[:index] + (None,) + contour[index + 1:]
                for index in range(len(contour))
            )
        span = max(values) - min(values)
        endpoint = values[-1] - values[0]
        gaps = tuple(b - a for a, b in zip(offsets, offsets[1:]))
        for pitch_offset in (0, 2):
            span_bucket = (span + pitch_offset) // 4
            endpoint_bucket = (endpoint + pitch_offset) // 4
            for rhythm_offset in (0, 2):
                rhythm = tuple((gap + rhythm_offset) // 4 for gap in gaps)
                for candidate_contour in contours:
                    keys.add((
                        len(values),
                        candidate_contour,
                        pitch_offset,
                        span_bucket,
                        endpoint_bucket,
                        rhythm_offset,
                        rhythm,
                    ))
    return keys


def discover_window_motifs(phrases, tempo, min_window=3, max_window=5):
    """Find bounded repeated figures inside phrases of unequal length."""
    windows = []
    for phrase_index, notes in enumerate(phrases):
        for start in range(max(0, len(notes) - min_window + 1)):
            for stop in range(start + min_window, min(len(notes), start + max_window) + 1):
                occurrence = phrase_occurrence(phrase_index, notes[start:stop], tempo)
                if occurrence is None:
                    continue
                gaps = tuple(
                    b - a for a, b in zip(occurrence.beat_offsets, occurrence.beat_offsets[1:])
                )
                if any(gap <= 0 or gap > 24 for gap in gaps):
                    continue
                windows.append(MotifWindow(phrase_index, start, stop, occurrence))

    candidate_index = {}
    candidate_pairs = set()
    for index, window in enumerate(windows):
        keys = _window_candidate_keys(window)
        for key in keys:
            candidate_pairs.update((old_index, index) for old_index in candidate_index.get(key, ()))
        for key in keys:
            candidate_index.setdefault(key, []).append(index)

    rejection_counts = {}
    candidates = {}
    comparisons = 0
    full_occurrences = {
        phrase_index: phrase_occurrence(phrase_index, notes, tempo)
        for phrase_index, notes in enumerate(phrases)
    }
    full_matches = {}
    for left_index, right_index in sorted(candidate_pairs):
        left, right = windows[left_index], windows[right_index]
        left_length = left.stop_index - left.start_index
        left_gaps = tuple(
            b - a
            for a, b in zip(left.occurrence.beat_offsets, left.occurrence.beat_offsets[1:])
        )
        right_length = right.stop_index - right.start_index
        if abs(left_length - right_length) > 1:
            continue
        if left.phrase_index == right.phrase_index:
            if max(left.start_index, right.start_index) < min(left.stop_index, right.stop_index):
                continue
        right_gaps = tuple(
            b - a
            for a, b in zip(right.occurrence.beat_offsets, right.occurrence.beat_offsets[1:])
        )
        if abs(float(np.median(left_gaps)) - float(np.median(right_gaps))) > 6:
            continue
        comparisons += 1
        match = match_occurrences(left.occurrence, right.occurrence)
        if not match.matched:
            rejection_counts[match.reason] = rejection_counts.get(match.reason, 0) + 1
            continue
        contour_pairs = list(zip(match.alignment, match.alignment[1:]))
        contour_hits = sum(
            (
                0 if left.occurrence.relative_pitches[after_left]
                == left.occurrence.relative_pitches[before_left]
                else int(np.sign(
                    left.occurrence.relative_pitches[after_left]
                    - left.occurrence.relative_pitches[before_left]
                ))
            )
            == (
                0 if right.occurrence.relative_pitches[after_right]
                == right.occurrence.relative_pitches[before_right]
                else int(np.sign(
                    right.occurrence.relative_pitches[after_right]
                    - right.occurrence.relative_pitches[before_right]
                ))
            )
            for ((before_left, before_right), (after_left, after_right))
            in contour_pairs
        )
        contour_agreement = contour_hits / max(1, len(contour_pairs))
        phrase_pair = tuple(sorted((left.phrase_index, right.phrase_index)))
        if left.phrase_index != right.phrase_index:
            if phrase_pair not in full_matches:
                full_left = full_occurrences[left.phrase_index]
                full_right = full_occurrences[right.phrase_index]
                full_matches[phrase_pair] = (
                    match_occurrences(full_left, full_right)
                    if full_left is not None and full_right is not None else None
                )
            full_match = full_matches[phrase_pair]
            if full_match is not None:
                if full_match.reason == "strong_variation" and len(match.alignment) < 4:
                    rejection_counts["strong_phrase_variation"] = (
                        rejection_counts.get("strong_phrase_variation", 0) + 1
                    )
                    continue
        row = {
            "left_phrase": left.phrase_index,
            "left_start": left.start_index,
            "left_stop": left.stop_index,
            "left_time": left.occurrence.start,
            "left_quality": left.occurrence.quality,
            "left_source_ids": left.occurrence.source_ids,
            "right_phrase": right.phrase_index,
            "right_start": right.start_index,
            "right_stop": right.stop_index,
            "right_time": right.occurrence.start,
            "right_quality": right.occurrence.quality,
            "right_source_ids": right.occurrence.source_ids,
            "alignment": match.alignment,
            "contour_agreement": contour_agreement,
            "score": match.score,
            "transposition": match.transposition,
            "reason": match.reason,
        }
        candidates.setdefault(phrase_pair, []).append(row)

    accepted = []
    for rows in candidates.values():
        kept = []
        for row in sorted(
            rows,
            key=lambda item: (
                -item["contour_agreement"],
                -len(item["alignment"]),
                item["score"],
                -(item["left_stop"] - item["left_start"]
                  + item["right_stop"] - item["right_start"]),
                item["left_time"],
                item["right_time"],
            ),
        ):
            duplicate = any(
                max(row["left_start"], old["left_start"])
                < min(row["left_stop"], old["left_stop"])
                and max(row["right_start"], old["right_start"])
                < min(row["right_stop"], old["right_stop"])
                for old in kept
            )
            if not duplicate:
                kept.append(row)
        accepted.extend(kept)
    accepted.sort(
        key=lambda row: (
            row["left_time"],
            row["right_time"],
            row["left_start"],
            row["right_start"],
        )
    )
    accepted_limit = 250
    return accepted, {
        "version": MOTIF_VERSION,
        "window_size": [min_window, max_window],
        "windows_considered": len(windows),
        "candidate_comparisons": comparisons,
        "matches_accepted": len(accepted),
        "rejection_counts": rejection_counts,
        "accepted": accepted[:accepted_limit],
        "accepted_truncated": max(0, len(accepted) - accepted_limit),
    }


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
