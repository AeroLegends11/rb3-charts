"""Phrase-level lead selection and five-fret authoring, without reference charts.

This is an explicit musical/ergonomic heuristic, not a model of human judgment.
It never creates an audio event or replaces a detected pitch with a fret number.
"""
from dataclasses import replace
import numpy as np
from .chart import BAND_GUITAR_SUSTAIN_SECONDS, _motif_groups
from .motifs import discover_phrase_motifs, discover_window_motifs

MOTIF_CONSTRAINT_WEIGHT = 2.0


def select_lead_phrases(source_events):
    """Choose a continuous voice using neighboring candidates in both directions.

    A small source/register continuity cost breaks ambiguous local choices;
    sufficiently strong evidence still permits a real voice or register change.
    """
    events = sorted((dict(e) for rows in source_events for e in rows),
                    key=lambda e: (e["start"], e["pitch"], e["source_id"]))
    groups = []
    for event in events:
        if groups and event["start"] - groups[-1][0]["start"] <= .03:
            groups[-1].append(event)
        else:
            groups.append([event])
    phrases = []
    for group in groups:
        if not phrases or group[0]["start"] - phrases[-1][-1][0]["start"] > .8:
            phrases.append([])
        phrases[-1].append(group)
    result = []
    changes = 0
    for phrase in phrases:
        costs, parents = [], []
        for i, group in enumerate(phrase):
            emissions = [-float(e["confidence"]) - .08 * float(e.get("model_onset_strength") or 0)
                         - (.04 if e.get("source") in ("other", "piano") else 0)
                         + (.15 if e["end"] - e["start"] < .075 else 0) for e in group]
            if not i:
                costs.append(emissions)
                parents.append([-1] * len(group))
                continue
            scores, links = [], []
            for j, event in enumerate(group):
                transitions = [costs[-1][k]
                               + (.16 if event.get("source") != old.get("source") else 0)
                               + min(.3, abs(event["pitch"] - old["pitch"]) * .012)
                               for k, old in enumerate(phrase[i - 1])]
                best = int(np.argmin(transitions))
                scores.append(emissions[j] + transitions[best])
                links.append(best)
            costs.append(scores)
            parents.append(links)
        index = int(np.argmin(costs[-1]))
        chosen = []
        for i in range(len(phrase) - 1, -1, -1):
            chosen.append(phrase[i][index])
            index = parents[i][index]
        for group, event in zip(phrase, reversed(chosen)):
            changes += event is not max(group, key=lambda e: e["confidence"])
            result.append(event)
            # Preserve only clear sustained polyphony, not a pitch plus its
            # octave. Ordinary melodic attacks remain single-note decisions.
            chord = [event]
            if event["confidence"] >= .75 and event["end"] - event["start"] >= .4:
                for extra in sorted(group, key=lambda e: -e["confidence"]):
                    if (extra is not event and len(chord) < 3
                            and extra.get("source") == event.get("source")
                            and extra["confidence"] >= .75
                            and float(extra.get("model_onset_strength") or 0) >= .55
                            and extra["end"] - extra["start"] >= .4
                            and all(abs(extra["pitch"] - old["pitch"]) >= 3
                                    and extra["pitch"] % 12 != old["pitch"] % 12 for old in chord)):
                        chord.append(extra)
                        result.append(extra)
    ordered=sorted(result, key=lambda e: (e["start"], e["pitch"]))
    rejected=[]
    for i in range(1,len(ordered)-1):
        before,event,after=ordered[i-1:i+2]
        if (event["confidence"]<.55 and after["start"]-before["start"]<.8
                and abs(before["pitch"]-after["pitch"])<=5
                and min(abs(event["pitch"]-before["pitch"]),abs(event["pitch"]-after["pitch"]))>=9
                and event["start"]-before["start"]>.03 and after["start"]-event["start"]>.03):
            rejected.append(event["source_id"])
    rejected_ids=set(rejected)
    result=[event for event in ordered if event["source_id"] not in rejected_ids]
    return result, {
        "candidates": len(events), "onset_groups": len(groups), "phrases": len(phrases),
        "contextual_choices": changes, "retained": len(result),
        "unsupported_register_spikes":rejected,
        "method": "sequence confidence with source/register continuity; conservative sustained chords"}


def _transition_cost(interval, old, new, gap):
    step=new-old
    musical=(8 if step else 0) if not interval else (5 if not step else 8 if step*interval<0 else 0)
    ergonomic=max(0,abs(step)-1)**2*(.25+max(0,1-gap/.5))
    return musical+ergonomic


def _constraint_cost(lanes, preferred):
    if preferred is None:
        return 0.0
    return float(
        sum(abs(lane - target) for lane, target in zip(lanes, preferred) if target is not None)
    )


def _lane_run_quality(phrases):
    runs = []
    lane_counts = {lane: 0 for lane in range(5)}
    attacks = 0
    for rows in phrases:
        current = []

        def finish():
            if len(current) < 5:
                return
            pitches = [note.pitch for note in current]
            distinct = len(set(pitches))
            span = max(pitches) - min(pitches)
            pitch_changes = sum(
                left != right for left, right in zip(pitches, pitches[1:])
            )
            runs.append({
                "start": current[0].start,
                "end": current[-1].start,
                "lane": current[0].lanes[0],
                "attacks": len(current),
                "distinct_pitches": distinct,
                "pitch_span": span,
                "pitch_changes": pitch_changes,
                "collapsed_contour": (
                    distinct >= 3 and span >= 4 and pitch_changes >= 3
                ),
            })

        for note in rows:
            attacks += 1
            lane_counts[note.lanes[0]] += 1
            if current and note.lanes == current[-1].lanes:
                current.append(note)
            else:
                finish()
                current = [note]
        finish()
    long_runs = [run for run in runs if run["attacks"] >= 8]
    collapsed = [run for run in runs if run["collapsed_contour"]]
    return {
        "single_note_attacks": attacks,
        "lane_counts": lane_counts,
        "dominant_lane_ratio": round(
            max(lane_counts.values(), default=0) / max(1, attacks), 4
        ),
        "max_same_lane_run": max(
            (run["attacks"] for run in runs), default=0
        ),
        "long_same_lane_runs": len(long_runs),
        "collapsed_contour_runs": len(collapsed),
        "runs": sorted(
            runs,
            key=lambda run: (
                not run["collapsed_contour"],
                -run["attacks"],
                run["start"],
            ),
        )[:20],
    }


def _guitar_single_note_phrases(notes, tempo):
    phrases = []
    current = []
    for note in sorted(notes, key=lambda row: row.start):
        if len(note.lanes) != 1:
            if current:
                phrases.append(current)
                current = []
            continue
        if current:
            beat = tempo.beat_at(note.start)
            period = tempo.seconds_at(beat + 1) - tempo.seconds_at(beat)
            rest = note.start - current[-1].end
            if rest > period * .75:
                phrases.append(current)
                current = []
        current.append(note)
    if current:
        phrases.append(current)
    return phrases


def analyze_guitar_lane_quality(notes, tempo):
    return _lane_run_quality(_guitar_single_note_phrases(notes,tempo))


def _preserve_supported_lane_span(rows, lanes):
    incoming = [note.lanes[0] for note in rows]
    duration = rows[-1].start - rows[0].start
    if (
        len(rows) < 32
        or duration <= 0
        or len(rows) / duration < 4
        or len(set(lanes)) < 2
    ):
        return lanes, False
    incoming_min, incoming_max = min(incoming), max(incoming)
    if incoming_max - incoming_min < 3:
        return lanes, False
    edge_support = min(incoming.count(incoming_min), incoming.count(incoming_max))
    pitches = np.asarray([note.pitch for note in rows], dtype=float)
    correlation = np.corrcoef(pitches, np.asarray(incoming, dtype=float))[0, 1]
    if edge_support < 2 or not np.isfinite(correlation) or correlation < .8:
        return lanes, False
    pitch_lanes = {}
    for note, lane in zip(rows, incoming):
        pitch_lanes.setdefault(note.pitch, []).append(lane)
    stable = [
        max(set(pitch_lanes[note.pitch]), key=lambda lane: (
            pitch_lanes[note.pitch].count(lane),
            -lane,
        ))
        for note in rows
    ]
    if max(stable) - min(stable) < 3:
        return lanes, False
    directions = [
        (right.pitch - left.pitch) * (right_lane - left_lane)
        for left, right, left_lane, right_lane in zip(rows, rows[1:], stable, stable[1:])
        if right.pitch != left.pitch
    ]
    agreement = sum(direction > 0 for direction in directions) / max(1, len(directions))
    reversals = sum(direction < 0 for direction in directions) / max(1, len(directions))
    if agreement < .85 or reversals > .08:
        return lanes, False
    stable_quality = _lane_run_quality([[
        replace(note, lanes=(lane,))
        for note, lane in zip(rows, stable)
    ]])
    baseline_quality = _lane_run_quality([[
        replace(note, lanes=(lane,))
        for note, lane in zip(rows, lanes)
    ]])
    if (
        stable_quality["collapsed_contour_runs"]
        > baseline_quality["collapsed_contour_runs"]
        or stable_quality["long_same_lane_runs"]
        > baseline_quality["long_same_lane_runs"]
        or stable_quality["max_same_lane_run"]
        > baseline_quality["max_same_lane_run"]
    ):
        return lanes, False
    return stable, stable != lanes


def _spread_supported_pitch_contour(rows, lanes):
    incoming = np.asarray([note.lanes[0] for note in rows], dtype=float)
    pitches = np.asarray([note.pitch for note in rows], dtype=float)
    duration = rows[-1].start - rows[0].start
    if (
        len(rows) < 32
        or duration <= 0
        or len(rows) / duration < 4
        or len(set(pitches)) < 5
        or np.ptp(pitches) < 10
        or np.ptp(incoming) < 3
    ):
        return lanes, False
    correlation = np.corrcoef(pitches, incoming)[0, 1]
    if not np.isfinite(correlation) or correlation < .9:
        return lanes, False
    incoming_min, incoming_max = min(incoming), max(incoming)
    if (
        sum(incoming == incoming_min) < 2
        or sum(incoming == incoming_max) < 2
    ):
        return lanes, False
    pitch_lanes = {}
    for pitch, lane in zip(pitches, incoming):
        pitch_lanes.setdefault(pitch, []).append(lane)
    stable = np.asarray([
        max(set(pitch_lanes[pitch]), key=lambda lane: (
            pitch_lanes[pitch].count(lane),
            -lane,
        ))
        for pitch in pitches
    ])
    directions = [
        (right_pitch - left_pitch) * (right_lane - left_lane)
        for left_pitch, right_pitch, left_lane, right_lane
        in zip(pitches, pitches[1:], stable, stable[1:])
        if right_pitch != left_pitch
    ]
    agreement = sum(value > 0 for value in directions) / max(1, len(directions))
    reversals = sum(value < 0 for value in directions) / max(1, len(directions))
    if agreement < .78 or reversals:
        return lanes, False
    unique_pitches = sorted(set(pitches))
    spread_map = {
        pitch: int(round(index * 4 / (len(unique_pitches) - 1)))
        for index, pitch in enumerate(unique_pitches)
    }
    spread = [spread_map[pitch] for pitch in pitches]
    if set(spread) != set(range(5)):
        return lanes, False
    return spread, spread != lanes


def _stabilize_repeated_pitch_lanes(rows, lanes):
    proposals = {}
    for pitch in {note.pitch for note in rows}:
        pitch_indices = [
            index for index, note in enumerate(rows) if note.pitch == pitch
        ]
        if len(pitch_indices) < 4:
            continue
        counts = {
            lane: sum(lanes[index] == lane for index in pitch_indices)
            for lane in {lanes[index] for index in pitch_indices}
        }
        anchor = max(counts, key=lambda lane: (counts[lane], -lane))
        if counts[anchor] / len(pitch_indices) < .6:
            continue
        proposals.update({
            index: anchor
            for index in pitch_indices
            if lanes[index] != anchor
        })
    candidate = [
        proposals.get(index, lane)
        for index, lane in enumerate(lanes)
    ]
    protected = set()
    for left in range(len(rows) - 1):
        right = left + 1
        pitch_step = rows[right].pitch - rows[left].pitch
        if not pitch_step or not ({left, right} & proposals.keys()):
            continue
        original_direction = pitch_step * (lanes[right] - lanes[left])
        candidate_direction = pitch_step * (
            candidate[right] - candidate[left]
        )
        if original_direction > 0 and candidate_direction <= 0:
            protected.update({left, right} & proposals.keys())
    stabilized = [
        lane if index in protected else candidate[index]
        for index, lane in enumerate(lanes)
    ]
    return stabilized, len(proposals) - len(protected), len(protected)


def _stabilize_repeat_fingerings(notes, tempo, quality_phrases=None):
    ordered, groups = _motif_groups(notes, tempo, "guitar")
    by_source = {note.source_id: note for note in ordered}
    changed = set()

    def exact_groups(motif_groups):
        for _, _, rows in motif_groups:
            exact = {}
            for index, phrase in rows:
                if any(len(note.lanes) != 1 for note in phrase):
                    continue
                key = tuple(
                    round(float(note.pitch), 3)
                    for note in phrase
                )
                exact.setdefault(key, []).append((index, phrase))
            yield from exact.values()

    def select_target(available, shapes):
        counts = {
            shape: shapes.count(shape)
            for shape in set(shapes)
        }

        def boundary_cost(shape):
            cost = 0.0
            for index, phrase in available:
                if index > 0 and len(ordered[index - 1].lanes) == 1:
                    previous = ordered[index - 1]
                    cost += _transition_cost(
                        phrase[0].pitch - previous.pitch,
                        previous.lanes[0],
                        shape[0],
                        phrase[0].start - previous.start,
                    )
                following_index = index + len(phrase)
                if (
                    following_index < len(ordered)
                    and len(ordered[following_index].lanes) == 1
                ):
                    following = ordered[following_index]
                    cost += _transition_cost(
                        following.pitch - phrase[-1].pitch,
                        shape[-1],
                        following.lanes[0],
                        following.start - phrase[-1].start,
                    )
            return cost

        target = min(
            counts,
            key=lambda shape: (
                -counts[shape],
                boundary_cost(shape),
                shapes.index(shape),
                shape,
            ),
        )
        return target, counts

    offset_plans = []
    for repeated in exact_groups(groups):
        available = [
            (index, [by_source[note.source_id] for note in phrase])
            for index, phrase in repeated
            if all(note.source_id in by_source for note in phrase)
        ]
        if len(available) < 2:
            continue
        shapes = [
            tuple(note.lanes[0] for note in phrase)
            for _, phrase in available
        ]
        relative_shapes = {
            tuple(lane - shape[0] for lane in shape)
            for shape in shapes
        }
        if len(relative_shapes) != 1 or len(set(shapes)) == 1:
            continue
        target, _ = select_target(available, shapes)
        assignments = {
            note.source_id: lane
            for (_, phrase), shape in zip(available, shapes)
            if shape != target
            for note, lane in zip(phrase, target)
            if note.lanes != (lane,)
        }
        if assignments:
            offset_plans.append((assignments, len(available)))
    proposed = {}
    for assignments, _ in offset_plans:
        for source_id, lane in assignments.items():
            proposed.setdefault(source_id, set()).add(lane)
    accepted_offsets = [
        (assignments, count)
        for assignments, count in offset_plans
        if all(len(proposed[source_id]) == 1 for source_id in assignments)
    ]
    rejected_collapsed_contours=0
    applied_offsets=[]
    for assignments, count in accepted_offsets:
        before={
            source_id:by_source[source_id].lanes
            for source_id in assignments
        }
        collapsed_before=(
            _lane_run_quality(quality_phrases)["collapsed_contour_runs"]
            if quality_phrases is not None else 0
        )
        for source_id, lane in assignments.items():
            by_source[source_id].lanes = (lane,)
        collapsed_after=(
            _lane_run_quality(quality_phrases)["collapsed_contour_runs"]
            if quality_phrases is not None else collapsed_before
        )
        if collapsed_after>collapsed_before:
            for source_id,lanes in before.items():
                by_source[source_id].lanes=lanes
            rejected_collapsed_contours+=1
            continue
        changed.update(assignments)
        applied_offsets.append((assignments,count))

    return {
        "families": len(applied_offsets),
        "occurrences": sum(
            count for _, count in applied_offsets
        ),
        "offset_families": len(applied_offsets),
        "notes_changed": len(changed),
        "rejected_collapsed_contours":rejected_collapsed_contours,
    }


def _fingering(rows, previous=None, preferred=None):
    pitches = [note.pitch for note in rows]
    unique = sorted(set(pitches))
    if len(unique) <= 5:
        # Distinct local pitches get distinct adjacent frets; one rare solo
        # pitch elsewhere cannot flatten this riff into a single color.
        options=[[offset+unique.index(pitch) for pitch in pitches] for offset in range(6-len(unique))]
        if previous is None:
            if preferred is not None:
                best = min(
                    options,
                    key=lambda lanes: (
                        _constraint_cost(lanes, preferred),
                        abs(np.mean(lanes) - 2),
                    ),
                )
                if _constraint_cost(best, preferred) == 0:
                    return best
            else:
                return options[(5-len(unique))//2]
        else:
            best=min(options,key=lambda lanes:(
                     _transition_cost(pitches[0]-previous.pitch,
                                      previous.lanes[0],lanes[0],rows[0].start-previous.start)
                     + MOTIF_CONSTRAINT_WEIGHT * _constraint_cost(lanes, preferred)))
            if (_transition_cost(pitches[0]-previous.pitch,previous.lanes[0],
                                 best[0],rows[0].start-previous.start)<5
                    and (preferred is None or _constraint_cost(best, preferred) == 0)):
                return best
        # No rigid pitch dictionary joins this boundary naturally. Let the
        # sequence optimizer preserve the incoming note and choose a later reset.
    target = np.interp(pitches, unique, np.linspace(0, 4, len(unique)))
    costs = .2 * (np.arange(5) - target[0])**2
    if preferred is not None and preferred[0] is not None:
        costs += MOTIF_CONSTRAINT_WEIGHT * np.abs(np.arange(5) - preferred[0])
    if previous is not None:
        costs+=np.array([_transition_cost(pitches[0]-previous.pitch,previous.lanes[0],lane,
                                         rows[0].start-previous.start) for lane in range(5)])
    links = []
    for i in range(1, len(rows)):
        interval = pitches[i] - pitches[i - 1]
        matrix = np.zeros((5, 5))
        for old in range(5):
            for new in range(5):
                matrix[old, new] = (
                    costs[old]
                    + _transition_cost(interval,old,new,rows[i].start-rows[i-1].start)
                    + .2 * (new - target[i])**2
                    + (MOTIF_CONSTRAINT_WEIGHT * abs(new - preferred[i])
                       if preferred is not None and preferred[i] is not None else 0)
                )
        links.append(np.argmin(matrix, axis=0))
        costs = np.min(matrix, axis=0)
    lane = int(np.argmin(costs))
    output = [lane]
    for link in reversed(links):
        lane = int(link[lane])
        output.append(lane)
    return list(reversed(output))


def _window_lane_constraints(local_lanes, matches, phrases, tempo):
    proposals = {}
    closure_proposals = {}
    for match in matches:
        left_rank = (
            match["left_quality"],
            match["left_stop"] - match["left_start"],
            -match["left_time"],
        )
        right_rank = (
            match["right_quality"],
            match["right_stop"] - match["right_start"],
            -match["right_time"],
        )
        if left_rank >= right_rank:
            source_phrase = match["left_phrase"]
            source_start = match["left_start"]
            target_phrase = match["right_phrase"]
            target_start = match["right_start"]
            alignment = match["alignment"]
            quality = match["left_quality"]
        else:
            source_phrase = match["right_phrase"]
            source_start = match["right_start"]
            target_phrase = match["left_phrase"]
            target_start = match["left_start"]
            alignment = tuple((right, left) for left, right in match["alignment"])
            quality = match["right_quality"]
        source = [
            local_lanes[source_phrase][source_start + source_index]
            for source_index, _ in alignment
        ]
        target = [
            local_lanes[target_phrase][target_start + target_index]
            for _, target_index in alignment
        ]
        shifts = range(-min(source), 5 - max(source))
        shift = min(
            shifts,
            key=lambda value: (
                sum(abs(lane + value - expected) for lane, expected in zip(source, target)),
                abs(value),
            ),
        )
        weight = len(alignment) * quality / (1 + match["score"])
        for lane, (_, target_index) in zip(source, alignment):
            key = (target_phrase, target_start + target_index)
            projected = lane + shift
            proposals.setdefault(key, {}).setdefault(projected, 0.0)
            proposals[key][projected] += weight
        source_offsets = [source_index for source_index, _ in alignment]
        target_offsets = [target_index for _, target_index in alignment]
        if (
            len(alignment) == 4
            and source_offsets == list(range(source_offsets[0], source_offsets[0] + len(alignment)))
            and target_offsets == list(range(target_offsets[0], target_offsets[0] + len(alignment)))
        ):
            source_first = source_start + source_offsets[0]
            source_previous = source_start + source_offsets[-1]
            source_next = source_previous + 1
            target_previous = target_start + target_offsets[-1]
            target_next = target_previous + 1
            source_rows = phrases[source_phrase]
            target_rows = phrases[target_phrase]
            if source_next < len(source_rows) and target_next < len(target_rows):
                source_note = source_rows[source_next]
                target_note = target_rows[target_next]
                source_gap = (
                    tempo.beat_at(source_note.start)
                    - tempo.beat_at(source_rows[source_previous].start)
                )
                target_gap = (
                    tempo.beat_at(target_note.start)
                    - tempo.beat_at(target_rows[target_previous].start)
                )
                if (
                    source_first == 0
                    and source_next == 4
                    and source_next == len(source_rows) - 1
                    and target_start + target_offsets[0] == 0
                    and target_next == 4
                    and abs(source_note.pitch - source_rows[source_first].pitch) <= 1
                    and target_note.confidence < .55
                    and source_note.confidence >= target_note.confidence + .04
                    and abs(source_gap - target_gap) <= .2
                ):
                    projected = local_lanes[source_phrase][source_next] + shift
                    if 0 <= projected <= 4:
                        key = (target_phrase, target_next)
                        closure_proposals.setdefault(key, []).append(
                            (weight, source_phrase, source_first, source_next, target_phrase, target_start)
                        )
    constraints = {}
    for key, lanes in proposals.items():
        local = local_lanes[key[0]][key[1]]
        constraints[key] = min(
            lanes,
            key=lambda lane: (-lanes[lane], abs(lane - local), lane),
        )
    closures = {
        key: max(rows, key=lambda row: (row[0], -row[1], -row[2]))
        for key, rows in closure_proposals.items()
    }
    return constraints, closures


def _has_contiguous_constraints(preferred, minimum=3):
    run = 0
    for target in preferred:
        run = run + 1 if target is not None else 0
        if run >= minimum:
            return True
    return False


def _continuity_anchor(notes, index, start, tempo):
    if not index:
        return None, "none", 0
    beat = tempo.beat_at(start)
    period = tempo.seconds_at(beat + 1) - start
    previous = notes[index - 1]
    previous_time = previous.start if len(previous.lanes) == 1 else previous.end
    if start - previous_time > period * .75:
        return None, "none", 0
    if len(previous.lanes) == 1:
        return previous, "adjacent_single", 0
    barrier_chords = 0
    for cursor in range(index - 1, -1, -1):
        note = notes[cursor]
        if cursor < index - 1 and notes[cursor + 1].start - note.end > period * .75:
            return None, "none", barrier_chords
        if len(note.lanes) == 1:
            barrier_start = notes[cursor + 1].start
            if (barrier_chords
                    and beat - tempo.beat_at(barrier_start) <= 2
                    and beat - tempo.beat_at(note.start) <= 2):
                return note, "through_chord", barrier_chords
            return None, "none", barrier_chords
        barrier_chords += 1
    return None, "none", barrier_chords


def _apply_band_chord_sections(notes, tempo, evidence, min_proposals=50):
    report = {
        "method": "repeated fifth-evidence sections with locally matched power-chord attacks",
        "gate_passed": False,
        "candidate_seeds": 0,
        "octave_seeds": 0,
        "candidate_notes": 0,
        "propagation_gate_passed": False,
        "direct_notes_changed": 0,
        "propagated_notes": 0,
        "notes_changed": 0,
        "regions": [],
    }
    if not notes or not evidence:
        return report
    focus = evidence.get("focus") or {}
    onset_groups = int(focus.get("onset_groups", 0))
    if onset_groups <= 0:
        return report
    correction_rate = float(focus.get("pitch_corrections", 0)) / onset_groups
    octave_anchor_rate = float(focus.get("octave_anchors", 0)) / onset_groups
    report["correction_rate"] = round(correction_rate, 4)
    report["octave_anchor_rate"] = round(octave_anchor_rate, 4)
    report["gate_passed"] = True
    candidates = sorted(
        (dict(event) for event in evidence.get("candidates", ()) if event.get("accepted", True)),
        key=lambda event: (event["start"], event["pitch"]),
    )
    by_source = {
        str(event.get("source_id")): event
        for event in candidates
        if event.get("source_id")
    }
    groups = []
    for event in candidates:
        if groups and event["start"] - groups[-1][0]["start"] <= .045:
            groups[-1].append(event)
        else:
            groups.append([event])
    seed_rows = []
    octave_seed_count = 0
    for group in groups:
        supported = False
        octave_supported = False
        for index, left in enumerate(group):
            for right in group[index + 1:]:
                interval = abs(int(left["pitch"]) - int(right["pitch"]))
                ratio = min(left["confidence"], right["confidence"]) / max(
                    left["confidence"], right["confidence"]
                )
                onset = min(
                    float(left.get("model_onset_strength") or 0),
                    float(right.get("model_onset_strength") or 0),
                )
                if interval in (5, 7) and ratio >= .5 and onset >= .5:
                    supported = True
                if interval == 12 and ratio >= .5 and onset >= .5:
                    octave_supported = True
        if octave_supported:
            octave_seed_count += 1
        if supported:
            seed_rows.append((group[0]["start"], tempo.beat_at(group[0]["start"])))
    report["candidate_seeds"] = len(seed_rows)
    report["octave_seeds"] = octave_seed_count
    if len(seed_rows) < 3:
        return report
    seed_rows.sort(key=lambda row: row[1])
    clusters = []
    cursor = 0
    while cursor < len(seed_rows):
        cluster = [seed_rows[cursor]]
        stop = cursor + 1
        while (
            stop < len(seed_rows)
            and seed_rows[stop][1] - cluster[-1][1] <= 10
            and seed_rows[stop][1] - cluster[0][1] <= 16
        ):
            cluster.append(seed_rows[stop])
            stop += 1
        clusters.append(cluster)
        cursor = stop if len(cluster) >= 3 else cursor + 1
    proposals = set()
    support_indices = set()
    for cluster in clusters:
        if len(cluster) < 3:
            continue
        for seed_time, _ in cluster:
            candidates = [
                index for index, note in enumerate(notes)
                if abs(note.start - seed_time) <= .075
            ]
            if candidates:
                index = min(candidates, key=lambda row: abs(notes[row].start - seed_time))
                support_indices.add(index)
                if len(notes[index].lanes) == 1:
                    proposals.add(index)
    report["candidate_notes"] = len(proposals)
    changed = []
    changed_indices = set()
    runs = []
    for index in sorted(support_indices):
        if (
            runs
            and tempo.beat_at(notes[index].start)
            - tempo.beat_at(notes[runs[-1][-1]].start) <= 2.25
        ):
            runs[-1].append(index)
        else:
            runs.append([index])
    for run in runs:
        if len(run) < 3:
            continue
        seed_pitches = [notes[idx].pitch for idx in run]
        seed_counts = {p: sum(notes[idx].pitch == p for idx in run) for p in set(seed_pitches)}
        if not any(count >= 2 for count in seed_counts.values()):
            continue
        span_indices = list(range(run[0], run[-1] + 1))
        pitch_counts = {
            pitch: sum(notes[index].pitch == pitch for index in span_indices)
            for pitch in {notes[index].pitch for index in span_indices}
        }
        repeated_roots = {
            pitch for pitch, count in pitch_counts.items() if count >= 2
        }
        for index in run:
            if index not in proposals or notes[index].pitch not in repeated_roots:
                continue
            note = notes[index]
            root = note.lanes[0]
            note.lanes = (root, root + 2) if root <= 2 else (root - 2, root)
            changed.append(note)
            changed_indices.add(index)
    report["direct_notes_changed"] = len(changed)
    propagation_gate = (
        len(proposals) >= min_proposals
        and len(changed) > 0
        and len(seed_rows) / max(1, octave_seed_count) >= 1.5
    )
    report["propagation_gate_passed"] = propagation_gate
    if propagation_gate:
        for run in runs:
            if len(run) < 3:
                continue
            seed_pitches = [notes[idx].pitch for idx in run]
            seed_counts = {p: sum(notes[idx].pitch == p for idx in run) for p in set(seed_pitches)}
            if not any(count >= 2 for count in seed_counts.values()):
                continue
            span_indices = list(range(run[0], run[-1] + 1))
            pitch_counts = {
                pitch: sum(notes[index].pitch == pitch for index in span_indices)
                for pitch in {notes[index].pitch for index in span_indices}
            }
            repeated_roots = {
                pitch for pitch, count in pitch_counts.items() if count >= 2
            }
            root_lanes = {}
            for index in span_indices:
                pitch = notes[index].pitch
                if pitch in repeated_roots and pitch not in root_lanes:
                    root = notes[index].lanes[0]
                    root_lanes[pitch] = (root, root + 2) if root <= 2 else (root - 2, root)
            for index in span_indices:
                note = notes[index]
                if (
                    index in changed_indices
                    or len(note.lanes) != 1
                ):
                    continue
                cand_pitches = [
                    float(by_source[s]["pitch"])
                    for s in str(note.source_id).split("+")
                    if s in by_source and by_source[s].get("pitch") is not None
                ]
                all_pitches = [note.pitch] + cand_pitches
                matched_root = None
                for p in all_pitches:
                    if p in repeated_roots:
                        matched_root = p
                        break
                    close = [r for r in repeated_roots if abs(p - r) <= 1]
                    if close:
                        matched_root = close[0]
                        break
                if matched_root is not None:
                    if matched_root in root_lanes:
                        note.lanes = root_lanes[matched_root]
                    else:
                        root = note.lanes[0]
                        note.lanes = (root, root + 2) if root <= 2 else (root - 2, root)
                    changed.append(note)
                    changed_indices.add(index)
                    report["propagated_notes"] += 1
                else:
                    harmonics = []
                    for p in all_pitches:
                        h = [
                            r for r in repeated_roots
                            if abs(int(round(p - r))) in (5, 7, 12, 19)
                            and r in root_lanes
                        ]
                        if h:
                            harmonics = h
                            break
                    if harmonics:
                        source_events = [
                            by_source[s] for s in str(note.source_id).split("+")
                            if s in by_source
                        ]
                        single_attack = max(
                            (float(e.get("evidence", {}).get("attack_strength", 0)) for e in source_events),
                            default=0.0,
                        )
                        is_weak_reattack = "reattack" in str(note.source_id) and single_attack < 0.05
                        if not is_weak_reattack:
                            matched_root = min(
                                harmonics,
                                key=lambda r: min(
                                    abs(note.start - notes[i].start)
                                    for i in span_indices if notes[i].pitch == r
                                ),
                            )
                            note.lanes = root_lanes[matched_root]
                            changed.append(note)
                            changed_indices.add(index)
                            report["propagated_notes"] += 1
    if not changed:
        return report
    report["notes_changed"] = len(changed)
    changed_regions = []
    for note in sorted(changed, key=lambda row: row.start):
        if (
            not changed_regions
            or tempo.beat_at(note.start) - tempo.beat_at(changed_regions[-1][-1].start) > 2.25
        ):
            changed_regions.append([note])
        else:
            changed_regions[-1].append(note)
    report["regions"] = [
        {"start": region[0].start, "end": region[-1].end, "notes": len(region)}
        for region in changed_regions
    ]
    return report


def _remove_pre_chord_voice_duplicates(notes, evidence):
    report = {
        "method": "continuing source voice duplicated immediately before chord",
        "notes_removed": 0,
        "events": [],
    }
    candidates = sorted(
        (
            dict(event)
            for event in (evidence or {}).get("candidates", ())
            if event.get("accepted", True)
        ),
        key=lambda event: (event["start"], event["pitch"]),
    )
    if len(notes) < 2 or not candidates:
        return notes, report
    by_source = {
        str(event.get("source_id")): event
        for event in candidates
        if event.get("source_id")
    }
    groups = []
    for event in candidates:
        if groups and event["start"] - groups[-1][0]["start"] <= .045:
            groups[-1].append(event)
        else:
            groups.append([event])
    removed = set()
    for i in range(1, len(notes) - 1):
        prev_chord = notes[i - 1]
        single = notes[i]
        next_chord = notes[i + 1]
        is_tight_sandwich = (
            len(single.lanes) == 1
            and len(prev_chord.lanes) >= 2
            and len(next_chord.lanes) >= 2
            and single.start - prev_chord.start <= .12
            and next_chord.start - single.start <= .12
        )
        is_identical_chord_sandwich = (
            len(single.lanes) == 1
            and len(prev_chord.lanes) >= 2
            and len(next_chord.lanes) >= 2
            and prev_chord.lanes == next_chord.lanes
            and single.start - prev_chord.start <= .32
            and next_chord.start - single.start <= .32
        )
        if is_tight_sandwich or is_identical_chord_sandwich:
            source_events = [
                by_source[source_id]
                for source_id in str(single.source_id).split("+")
                if source_id in by_source
            ]
            single_attack = max(
                (float(event.get("evidence", {}).get("attack_strength", 0)) for event in source_events),
                default=0.0,
            )
            is_weak_reattack = "reattack" in str(single.source_id)
            if is_tight_sandwich or single_attack < .045 or is_weak_reattack:
                removed.add(id(single))
                report["events"].append({
                    "source_id": single.source_id,
                    "start": single.start,
                    "pitch": single.pitch,
                    "prev_chord_start": prev_chord.start,
                    "next_chord_start": next_chord.start,
                    "single_attack_strength": round(single_attack, 6),
                    "reason": "tight single note sandwiched between chords" if is_tight_sandwich else "weak single note sandwiched between identical chords",
                })
    for single, chord in zip(notes, notes[1:]):
        gap = chord.start - single.start
        if (
            len(single.lanes) != 1
            or len(chord.lanes) < 2
            or gap <= 0
            or gap > .18
        ):
            continue
        source_events_all = [
            by_source[source_id]
            for source_id in str(single.source_id).split("+")
            if source_id in by_source
        ]
        single_attack_any = max(
            (float(event.get("evidence", {}).get("attack_strength", 0)) for event in source_events_all),
            default=0.0,
        )
        if gap <= .085 and single_attack_any < .035:
            removed.add(id(single))
            report["events"].append({
                "source_id": single.source_id,
                "start": single.start,
                "pitch": single.pitch,
                "following_chord_source_id": chord.source_id,
                "following_chord_start": chord.start,
                "attack_gap_ms": round(gap * 1000, 3),
                "single_attack_strength": round(single_attack_any, 6),
                "reason": "weak strum flam or rake immediately before chord",
            })
            continue
        chord_group = min(
            groups,
            key=lambda rows: abs(rows[0]["start"] - chord.start),
        )
        if abs(chord_group[0]["start"] - chord.start) > .045:
            continue
        chord_pitches = {int(event["pitch"]) for event in chord_group}
        pitch_diffs = {abs(int(single.pitch) - p) for p in chord_pitches}
        matches_pitch = int(single.pitch) in chord_pitches
        matches_harmonic = any(diff in (7, 12, 19) for diff in pitch_diffs)
        if not (matches_pitch or matches_harmonic):
            continue
        source_events = [
            by_source[source_id]
            for source_id in str(single.source_id).split("+")
            if source_id in by_source
            and (
                int(by_source[source_id]["pitch"]) == int(single.pitch)
                or abs(int(by_source[source_id]["pitch"]) - int(single.pitch)) in (7, 12, 19)
            )
        ]
        if (
            not source_events
            or max(float(event["end"]) for event in source_events)
            < chord.start - .035
        ):
            continue
        single_attack = max(
            float(event.get("evidence", {}).get("attack_strength", 0))
            for event in source_events
        )
        chord_attack = max(
            float(event.get("evidence", {}).get("attack_strength", 0))
            for event in chord_group
        )
        if matches_pitch:
            if (
                single_attack >= .03
                or chord_attack < .04
                or chord_attack < single_attack * 2
            ):
                continue
        else:
            if (
                single_attack >= .035
                or chord_attack < .04
                or chord_attack < single_attack * 1.5
            ):
                continue
        removed.add(id(single))
        report["events"].append({
            "source_id": single.source_id,
            "start": single.start,
            "pitch": single.pitch,
            "following_chord_source_id": chord.source_id,
            "following_chord_start": chord.start,
            "attack_gap_ms": round(gap * 1000, 3),
            "single_attack_strength": round(single_attack, 6),
            "chord_attack_strength": round(chord_attack, 6),
            "reason": "same source pitch or overtone continues into immediate chord evidence",
        })
    report["notes_removed"] = len(removed)
    return [note for note in notes if id(note) not in removed], report


def _collapse_isolated_wide_chords(notes, tempo):
    report = {
        "method": "final isolated wide-chord cleanup",
        "notes_changed": 0,
        "events": [],
    }
    beats = [tempo.beat_at(note.start) for note in notes]
    for index, note in enumerate(notes):
        if (
            len(note.lanes) < 2
            or note.lanes[-1] - note.lanes[0] < 3
            or note.confidence >= .75
        ):
            continue
        neighbors = [
            row
            for row, beat in zip(notes, beats)
            if abs(beat - beats[index]) <= 4
        ]
        local_chords = sum(len(row.lanes) > 1 for row in neighbors)
        if (
            len(neighbors) < 8
            or local_chords > max(1, int(np.ceil(len(neighbors) * .15)))
        ):
            continue
        before = note.lanes
        note.lanes = (note.lanes[0],)
        report["events"].append({
            "source_id": note.source_id,
            "start": note.start,
            "confidence": note.confidence,
            "lanes_before": list(before),
            "lanes_after": list(note.lanes),
        })
    report["notes_changed"] = len(report["events"])
    return report


def _apply_audio_repeat_consistency(notes, tempo, evidence):
    report = {
        "method": "audio self-similarity with local source support for asymmetric repeats",
        "regions_considered": 0,
        "regions_compatible": 0,
        "regions_applied": 0,
        "notes_changed": 0,
        "conflicts": 0,
        "rejected_quality": False,
        "regions": [],
        "region_audits": [],
    }
    regions = (evidence or {}).get("regions", ())
    if not notes or not regions:
        return report
    report["regions_considered"] = len(regions)

    def has_chord_evidence(note):
        return len(note.source_id.split("+")) > 1

    def region_notes(start, end):
        return [note for note in notes if start <= note.start < end]

    def shape_quality(rows):
        pairs = [
            (left, right)
            for left, right in zip(rows, rows[1:])
            if len(left.lanes) == len(right.lanes) == 1
            and left.pitch != right.pitch
        ]
        directional = sum(
            (right.pitch - left.pitch)
            * (right.lanes[0] - left.lanes[0]) > 0
            for left, right in pairs
        )
        flattened = sum(left.lanes == right.lanes for left, right in pairs)
        diversity = len({tuple(note.lanes) for note in rows})
        confidence = float(np.mean([note.confidence for note in rows]))
        chord_rate = sum(len(note.lanes) > 1 for note in rows) / len(rows)
        return (
            confidence,
            chord_rate,
            directional - flattened,
            diversity,
        )

    proposals = {}
    accepted_regions = []
    for region in regions:
        left = region_notes(region["left_start"], region["left_end"])
        right = region_notes(region["right_start"], region["right_end"])
        if min(len(left), len(right)) < 4 or max(len(left), len(right)) > 64:
            continue
        left_times = np.asarray([
            note.start - region["left_start"] for note in left
        ])
        right_times = np.asarray([
            note.start - region["right_start"] for note in right
        ])
        shifts = [
            float(right_time - left_time)
            for left_time in left_times
            for right_time in right_times
            if abs(right_time - left_time) <= .5
        ]
        if not shifts:
            continue
        shift = min(
            shifts,
            key=lambda value: (
                -sum(
                    np.min(np.abs(right_times - (left_time + value))) <= .09
                    for left_time in left_times
                ),
                abs(value),
            ),
        )
        alignment = []
        right_cursor = 0
        for left_index, left_time in enumerate(left_times):
            candidates = [
                right_index
                for right_index in range(right_cursor, len(right_times))
                if abs(right_times[right_index] - (left_time + shift)) <= .09
            ]
            if not candidates:
                continue
            right_index = min(
                candidates,
                key=lambda index: abs(
                    right_times[index] - (left_time + shift)
                ),
            )
            alignment.append((left_index, right_index))
            right_cursor = right_index + 1
        coverage = len(alignment) / max(len(left), len(right))
        high_audio_match = (
            region["similarity"] >= .83
            and region.get("duration", 0) >= 8
            and region.get("anchors", 0) >= 20
            and coverage >= .65
        )
        if len(alignment) < 4 or (coverage < .7 and not high_audio_match):
            continue
        aligned_left = [left[i] for i, _ in alignment]
        aligned_right = [right[j] for _, j in alignment]
        polyphony_agreement = sum(
            len(a.lanes) == len(b.lanes)
            for a, b in zip(aligned_left, aligned_right)
        ) / len(alignment)
        chord_density_diff = abs(
            sum(len(a.lanes) > 1 for a in aligned_left) / len(aligned_left)
            - sum(len(b.lanes) > 1 for b in aligned_right) / len(aligned_right)
        )
        asymmetric_evidence = sum(
            len(a.lanes) != len(b.lanes)
            and (has_chord_evidence(a) or has_chord_evidence(b))
            for a, b in zip(aligned_left, aligned_right)
        )
        chord_asymmetry_match = (
            chord_density_diff >= .25
            and (high_audio_match or asymmetric_evidence >= 1)
        )
        report["region_audits"].append({
            "left_start": region["left_start"],
            "right_start": region["right_start"],
            "coverage": round(coverage, 4),
            "polyphony_agreement": round(polyphony_agreement, 4),
            "chord_density_diff": round(chord_density_diff, 4),
            "asymmetric_evidence": asymmetric_evidence,
            "high_audio_match": high_audio_match,
            "chord_asymmetry_match": chord_asymmetry_match,
        })
        if (
            polyphony_agreement < .75
            and not high_audio_match
            and not chord_asymmetry_match
        ):
            continue
        pitch_delta = np.asarray([
            right_note.pitch - left_note.pitch
            for left_note, right_note in zip(aligned_left, aligned_right)
        ])
        transposition = float(np.median(pitch_delta))
        if (
            float(np.max(np.abs(pitch_delta - transposition))) > 2
            and not high_audio_match
            and not chord_asymmetry_match
        ):
            continue
        pitch_steps = [
            (
                int(np.sign(b.pitch - a.pitch)),
                int(np.sign(d.pitch - c.pitch)),
            )
            for a, b, c, d in zip(
                aligned_left,
                aligned_left[1:],
                aligned_right,
                aligned_right[1:],
            )
            if b.pitch != a.pitch or d.pitch != c.pitch
        ]
        if (
            pitch_steps
            and sum(a == b for a, b in pitch_steps) / len(pitch_steps) < .8
            and not high_audio_match
            and not chord_asymmetry_match
        ):
            continue
        report["regions_compatible"] += 1
        source, target = (
            (aligned_left, aligned_right)
            if shape_quality(aligned_left) >= shape_quality(aligned_right)
            else (aligned_right, aligned_left)
        )
        flat_source = [lane for note in source for lane in note.lanes]
        shifts = range(-min(flat_source), 5 - max(flat_source))
        shift = min(
            shifts,
            key=lambda value: (
                sum(
                    abs(
                        float(np.mean(source_note.lanes)) + value
                        - float(np.mean(target_note.lanes))
                    )
                    for source_note, target_note in zip(source, target)
                ),
                abs(value),
            ),
        )
        changed = 0
        for source_note, target_note in zip(source, target):
            if (
                len(source_note.lanes) != len(target_note.lanes)
                and not high_audio_match
            ):
                single_note = (
                    source_note if len(source_note.lanes) == 1 else target_note
                )
                if len(single_note.lanes) == 1 and not has_chord_evidence(single_note):
                    continue
            lanes = tuple(lane + shift for lane in source_note.lanes)
            if lanes == target_note.lanes:
                continue
            proposals.setdefault(target_note.source_id, set()).add(lanes)
            changed += 1
        if changed:
            accepted_regions.append({
                "left_start": region["left_start"],
                "right_start": region["right_start"],
                "matched_attacks": len(alignment),
                "coverage": round(coverage, 4),
                "high_audio_match": high_audio_match,
                "proposed_changes": changed,
                "similarity": region["similarity"],
            })
    report["conflicts"] = sum(len(rows) > 1 for rows in proposals.values())
    assignments = {
        source_id: next(iter(rows))
        for source_id, rows in proposals.items()
        if len(rows) == 1
    }
    if not assignments:
        return report
    by_source = {note.source_id: note for note in notes}
    before = {
        source_id: by_source[source_id].lanes
        for source_id in assignments
        if source_id in by_source
    }
    baseline_quality = _lane_run_quality(_guitar_single_note_phrases(notes, tempo))
    for source_id, lanes in assignments.items():
        if source_id in by_source:
            by_source[source_id].lanes = lanes
    candidate_quality = _lane_run_quality(_guitar_single_note_phrases(notes, tempo))
    if (
        candidate_quality["collapsed_contour_runs"]
        > baseline_quality["collapsed_contour_runs"]
        or candidate_quality["long_same_lane_runs"]
        > baseline_quality["long_same_lane_runs"]
        or candidate_quality["max_same_lane_run"]
        > baseline_quality["max_same_lane_run"]
    ):
        for source_id, lanes in before.items():
            by_source[source_id].lanes = lanes
        report["rejected_quality"] = True
        return report
    report["regions_applied"] = len(accepted_regions)
    report["notes_changed"] = len(before)
    report["regions"] = accepted_regions[:24]
    return report


def _apply_rhythmic_riff_consistency(notes, tempo):
    report = {
        "method": "three/four-beat attack alignment with pitch-contour agreement",
        "families_considered": 0,
        "families_applied": 0,
        "notes_changed": 0,
        "conflicts": 0,
        "rejected_quality": False,
        "families": [],
    }
    if len(notes) < 24:
        return report

    def align(left, right, cycle_start_left, cycle_start_right):
        left_beats = [
            tempo.beat_at(note.start) - cycle_start_left for note in left
        ]
        right_beats = [
            tempo.beat_at(note.start) - cycle_start_right for note in right
        ]
        pairs = []
        right_cursor = 0
        for left_index, left_beat in enumerate(left_beats):
            candidates = [
                right_index
                for right_index in range(right_cursor, len(right_beats))
                if abs(right_beats[right_index] - left_beat) <= .25
            ]
            if not candidates:
                continue
            right_index = min(
                candidates,
                key=lambda index: abs(right_beats[index] - left_beat),
            )
            pairs.append((left_index, right_index))
            right_cursor = right_index + 1
        return pairs

    proposals = {}
    accepted = []
    first_beat = tempo.beat_at(notes[0].start)
    last_beat = tempo.beat_at(notes[-1].start)
    for cycle_beats in (3, 4):
        first_cycle = int(np.floor(first_beat / cycle_beats))
        last_cycle = int(np.floor(last_beat / cycle_beats))
        windows = {}
        for cycle in range(first_cycle, last_cycle + 1):
            start = cycle * cycle_beats
            rows = [
                note
                for note in notes
                if start <= tempo.beat_at(note.start) < start + cycle_beats
            ]
            if 8 <= len(rows) <= 14:
                windows[cycle] = rows
        if len(windows) < 3:
            continue
        parents = {cycle: cycle for cycle in windows}
        links = {}

        def find(cycle):
            while parents[cycle] != cycle:
                parents[cycle] = parents[parents[cycle]]
                cycle = parents[cycle]
            return cycle

        def union(left_cycle, right_cycle):
            left_root, right_root = find(left_cycle), find(right_cycle)
            if left_root != right_root:
                parents[max(left_root, right_root)] = min(left_root, right_root)

        cycles = sorted(windows)
        for offset, left_cycle in enumerate(cycles):
            left = windows[left_cycle]
            for right_cycle in cycles[offset + 1:]:
                if right_cycle != left_cycle + 1:
                    continue
                right = windows[right_cycle]
                pairs = align(
                    left,
                    right,
                    left_cycle * cycle_beats,
                    right_cycle * cycle_beats,
                )
                coverage = len(pairs) / max(len(left), len(right))
                if coverage < .8:
                    continue
                pitch_steps = []
                for (a, c), (b, d) in zip(pairs, pairs[1:]):
                    left_diff = left[b].pitch - left[a].pitch
                    right_diff = right[d].pitch - right[c].pitch
                    if left_diff != 0 or right_diff != 0:
                        left_step = int(np.sign(left_diff))
                        right_step = int(np.sign(right_diff))
                        if left_step == 0 and abs(left_diff) % 12 == 0:
                            left_step = right_step
                        elif right_step == 0 and abs(right_diff) % 12 == 0:
                            right_step = left_step
                        pitch_steps.append((left_step, right_step))
                contour_agreement = (
                    sum(a == b for a, b in pitch_steps) / len(pitch_steps)
                    if pitch_steps else 1.0
                )
                if contour_agreement < .5:
                    continue
                links[(left_cycle, right_cycle)] = (
                    pairs,
                    coverage,
                    contour_agreement,
                )
                union(left_cycle, right_cycle)
        families = {}
        for cycle in cycles:
            families.setdefault(find(cycle), []).append(cycle)
        for family_cycles in families.values():
            if len(family_cycles) < 3:
                continue
            report["families_considered"] += 1
            canonical_cycle = max(
                family_cycles,
                key=lambda cycle: (
                    len({
                        lane
                        for note in windows[cycle]
                        for lane in note.lanes
                    }),
                    sum(
                        lane in (0, 4)
                        for note in windows[cycle]
                        for lane in note.lanes
                    ),
                    sum(
                        a.lanes != b.lanes
                        for a, b in zip(windows[cycle], windows[cycle][1:])
                    ),
                    float(np.mean([
                        note.confidence for note in windows[cycle]
                    ])),
                    -cycle,
                ),
            )
            canonical = windows[canonical_cycle]
            family_changes = 0
            matched_cycles = 0
            for target_cycle in family_cycles:
                if target_cycle == canonical_cycle:
                    continue
                pair_key = tuple(sorted((canonical_cycle, target_cycle)))
                match = links.get(pair_key)
                if match is None:
                    continue
                pairs = match[0]
                if canonical_cycle > target_cycle:
                    pairs = [(right, left) for left, right in pairs]
                target = windows[target_cycle]
                matched_cycles += 1
                suggestions = []
                for source_index, target_index in pairs:
                    source_note = canonical[source_index]
                    target_note = target[target_index]
                    if len(source_note.lanes) != 1 or len(target_note.lanes) != 1:
                        continue
                    suggestions.append((target_note, source_note.lanes))
                by_pitch = {}
                for target_note, lanes in suggestions:
                    by_pitch.setdefault(int(round(target_note.pitch)), []).append(
                        (target_note, lanes)
                    )
                stabilized = []
                for rows in by_pitch.values():
                    lane_counts = {}
                    for _, lanes in rows:
                        lane_counts[lanes] = lane_counts.get(lanes, 0) + 1
                    majority = max(
                        lane_counts,
                        key=lambda lanes: (
                            lane_counts[lanes],
                            sum(note.lanes == lanes for note, _ in rows),
                            -abs(lanes[0] - 2),
                        ),
                    )
                    stabilized.extend((note, majority) for note, _ in rows)
                for target_note, lanes in stabilized:
                    if lanes == target_note.lanes:
                        continue
                    proposals.setdefault(target_note.source_id, set()).add(lanes)
                    family_changes += 1
            if family_changes:
                accepted.append({
                    "cycle_beats": cycle_beats,
                    "cycles": family_cycles,
                    "canonical_cycle": canonical_cycle,
                    "matched_cycles": matched_cycles,
                    "proposed_changes": family_changes,
                })
    report["conflicts"] = sum(len(rows) > 1 for rows in proposals.values())
    assignments = {
        source_id: next(iter(rows))
        for source_id, rows in proposals.items()
        if len(rows) == 1
    }
    if not assignments:
        return report
    by_source = {note.source_id: note for note in notes}
    before = {
        source_id: by_source[source_id].lanes
        for source_id in assignments
        if source_id in by_source
    }
    baseline_quality = _lane_run_quality(_guitar_single_note_phrases(notes, tempo))
    for source_id, lanes in assignments.items():
        if source_id in by_source:
            by_source[source_id].lanes = lanes
    candidate_quality = _lane_run_quality(_guitar_single_note_phrases(notes, tempo))
    if (
        candidate_quality["collapsed_contour_runs"]
        > baseline_quality["collapsed_contour_runs"]
        or candidate_quality["long_same_lane_runs"]
        > baseline_quality["long_same_lane_runs"]
        or candidate_quality["max_same_lane_run"]
        > baseline_quality["max_same_lane_run"]
    ):
        for source_id, lanes in before.items():
            by_source[source_id].lanes = lanes
        report["rejected_quality"] = True
        return report
    report["families_applied"] = len(accepted)
    report["notes_changed"] = len(before)
    report["families"] = accepted[:24]
    return report


def author_guitar_phrases(
    notes, tempo, chord_evidence=None, audio_repetitions=None,
    sustain_threshold=BAND_GUITAR_SUSTAIN_SECONDS,
):
    """Sight-reading pass: phrase contour, repeatable fingering, short tails.

    Preserve retained event identity, pitch and attack timing. Chords are
    barriers to single-note fingering; their already-mapped voicing is retained.
    """
    result = [replace(n) for n in sorted(notes, key=lambda n: n.start)]
    phrases = _guitar_single_note_phrases(result,tempo)
    vocabulary = {}
    phrase_report = []
    indices={id(note):i for i,note in enumerate(result)}
    local_lanes = {}
    motif_assignments = discover_phrase_motifs(phrases, tempo)
    window_matches, window_diagnostics = discover_window_motifs(phrases, tempo)
    baseline = [replace(n) for n in result]
    continuity_summary = {"adjacent_single": 0, "through_chord": 0, "none": 0}
    pitch_anchor_changes = 0
    pitch_anchor_contours_protected = 0
    supported_span_preservations = 0
    supported_contour_spreads = 0
    supported_contour_spread_rejections = 0
    for phrase_index, rows in enumerate(phrases):
        index = indices[id(rows[0])]
        previous, _, _ = _continuity_anchor(baseline, index, rows[0].start, tempo)
        span = baseline[index:index + len(rows)]
        lanes = _fingering(span, previous)
        local_lanes[phrase_index] = lanes
        for note, lane in zip(span, lanes):
            note.lanes = (lane,)
    window_constraints, low_confidence_closures = _window_lane_constraints(
        local_lanes, window_matches, phrases, tempo
    )
    family_lanes = {}
    for phrase_index, assignment in motif_assignments.items():
        canonical = assignment["canonical_phrase"]
        if phrase_index != canonical:
            continue
        family_lanes[assignment["family_id"]] = local_lanes[canonical]
    for phrase_index, rows in enumerate(phrases):
        index=indices[id(rows[0])]
        previous, continuity_source, barrier_chords = _continuity_anchor(result, index, rows[0].start, tempo)
        continuity_summary[continuity_source] += 1
        root = rows[0].pitch
        signature = (tuple(n.pitch - root for n in rows),
                     tuple(round((tempo.beat_at(b.start) - tempo.beat_at(a.start)) * 12)
                           for a, b in zip(rows, rows[1:])))
        reused = signature in vocabulary
        lanes = vocabulary.get(signature)
        shift=0
        if lanes is not None and continuity_source == "adjacent_single" and previous is not None:
            shift=min(range(-min(lanes),5-max(lanes)),key=lambda s:
                      _transition_cost(rows[0].pitch-previous.pitch,previous.lanes[0],lanes[0]+s,
                                       rows[0].start-previous.start)+.01*abs(s))
            lanes=[lane+shift for lane in lanes]
            if _transition_cost(rows[0].pitch-previous.pitch,previous.lanes[0],lanes[0],rows[0].start-previous.start)>=5:
                lanes=None
                reused=False
        motif_info = motif_assignments.get(phrase_index)
        motif_reused = False
        motif_reason = "not_matched"
        preferred = [
            window_constraints.get((phrase_index, note_index))
            for note_index in range(len(rows))
        ]
        window_constrained = _has_contiguous_constraints(preferred)
        if not window_constrained:
            preferred = None
        if motif_info is not None:
            motif_reason = motif_info["reason"]
            canonical = family_lanes.get(motif_info["family_id"])
            if canonical is not None:
                projected = list(local_lanes[phrase_index])
                for source, target in motif_info["alignment"]:
                    if source < len(canonical) and target < len(projected):
                        projected[target] = canonical[source]
                if projected != local_lanes[phrase_index]:
                    motif_reused = True
                lanes = projected
        if preferred is not None and not reused and motif_info is None:
            candidate = _fingering(rows, previous, preferred)
            if lanes is None or _constraint_cost(candidate, preferred) < _constraint_cost(lanes, preferred):
                lanes = candidate
                reused = False
                motif_reused = False
        if lanes is None:
            lanes = _fingering(rows, previous, preferred)
        vocabulary.setdefault(signature,lanes)
        if continuity_source == "adjacent_single" and previous is not None and previous.pitch == rows[0].pitch and len(previous.lanes) == 1:
            boundary_delta = previous.lanes[0] - lanes[0]
            shifted = [lane + boundary_delta for lane in lanes]
            if (all(0 <= lane <= 4 for lane in shifted)
                    and _constraint_cost(shifted, preferred) <= _constraint_cost(lanes, preferred)):
                lanes = [lane + boundary_delta for lane in lanes]
        lanes, changes, protected = _stabilize_repeated_pitch_lanes(rows, lanes)
        pitch_anchor_changes += changes
        pitch_anchor_contours_protected += protected
        lanes, span_preserved = _preserve_supported_lane_span(rows, lanes)
        supported_span_preservations += int(span_preserved)
        if not span_preserved:
            spread_lanes, contour_spread = _spread_supported_pitch_contour(
                rows, lanes
            )
            if contour_spread:
                baseline_quality = _lane_run_quality([[
                    replace(note, lanes=(lane,))
                    for note, lane in zip(rows, lanes)
                ]])
                spread_quality = _lane_run_quality([[
                    replace(note, lanes=(lane,))
                    for note, lane in zip(rows, spread_lanes)
                ]])
                if (
                    spread_quality["collapsed_contour_runs"]
                    > baseline_quality["collapsed_contour_runs"]
                    or spread_quality["long_same_lane_runs"]
                    > baseline_quality["long_same_lane_runs"]
                    or spread_quality["max_same_lane_run"]
                    > baseline_quality["max_same_lane_run"]
                ):
                    contour_spread = False
                    supported_contour_spread_rejections += 1
                else:
                    lanes = spread_lanes
            supported_contour_spreads += int(contour_spread)
        else:
            contour_spread = False
        before = [n.lanes[0] for n in rows]
        for note, lane in zip(rows, lanes):
            note.lanes = (lane,)
        phrase_report.append({"start": rows[0].start, "end": rows[-1].end,
                              "source_ids": [n.source_id for n in rows],
                              "pitches": [n.pitch for n in rows], "before": before,
                              "after": lanes, "reused_fingering": reused,
                              "boundary_fret_shift":shift,"joined_previous_phrase":previous is not None,
                              "continuity_source": continuity_source,
                              "barrier_chords": barrier_chords,
                              "motif_family": motif_info["family_id"] if motif_info else None,
                              "motif_reused": motif_reused,
                              "motif_reason": motif_reason,
                              "supported_span_preserved": span_preserved,
                              "supported_contour_spread": contour_spread,
                              "window_motif_constrained": window_constrained})
    for (phrase_index, note_index), closure in low_confidence_closures.items():
        if phrase_report[phrase_index]["supported_span_preserved"]:
            continue
        _, source_phrase, source_first, source_next, _, target_first = closure
        shift = (
            phrases[phrase_index][target_first].lanes[0]
            - phrases[source_phrase][source_first].lanes[0]
        )
        lane = int(np.clip(phrases[source_phrase][source_next].lanes[0] + shift, 0, 4))
        phrases[phrase_index][note_index].lanes = (lane,)
        phrase_report[phrase_index]["after"][note_index] = lane
    tails = []
    for i, note in enumerate(result):
        duration = note.end - note.start
        if .065 < duration < sustain_threshold:
            end = note.start + .065
            if i + 1 < len(result):
                end = min(end, result[i + 1].start - .015)
            if end > note.start:
                tails.append({"source_id": note.source_id, "before": note.end, "after": end})
                note.end = end
    def contour(rows):
        pairs = [(a, b) for a, b in zip(rows, rows[1:])
                 if len(a.lanes) == len(b.lanes) == 1 and b.start - a.start < .8]
        return {"collapsed_pitch_changes": sum(a.pitch != b.pitch and a.lanes == b.lanes for a,b in pairs),
                "reversed_pitch_changes": sum((b.pitch-a.pitch)*(b.lanes[0]-a.lanes[0]) < 0 for a,b in pairs)}
    chord_sections = _apply_band_chord_sections(result, tempo, chord_evidence)
    repeat_fingerings = _stabilize_repeat_fingerings(
        result, tempo, quality_phrases=phrases
    )
    audio_repeat_consistency = _apply_audio_repeat_consistency(
        result, tempo, audio_repetitions
    )
    rhythmic_riff_consistency = _apply_rhythmic_riff_consistency(result, tempo)
    isolated_wide_chord_cleanup = _collapse_isolated_wide_chords(result, tempo)
    result, pre_chord_cleanup = _remove_pre_chord_voice_duplicates(
        result, chord_evidence
    )
    lane_quality = _lane_run_quality(_guitar_single_note_phrases(result, tempo))
    for phrase, rows in zip(phrase_report, phrases):
        phrase["final_lanes"] = [list(note.lanes) for note in rows]
    return result, {"method": "phrase-local fingering with whole-sequence ergonomic costs and bounded motif windows",
                    "before": contour(sorted(notes, key=lambda n:n.start)), "after": contour(result),
                    "continuity_summary": continuity_summary,
                    "pitch_anchor_changes": pitch_anchor_changes,
                    "pitch_anchor_contours_protected": pitch_anchor_contours_protected,
                    "supported_span_preservations": supported_span_preservations,
                    "supported_contour_spreads": supported_contour_spreads,
                    "supported_contour_spread_rejections":
                    supported_contour_spread_rejections,
                    "lane_quality": lane_quality,
                    "repeat_fingerings": repeat_fingerings,
                    "sustain_threshold_seconds": sustain_threshold,
                    "phrases": phrase_report, "short_tails_removed": tails,
                    "window_motif_matching": window_diagnostics,
                    "window_motif_constrained_phrases": sum(
                        phrase["window_motif_constrained"] for phrase in phrase_report
                    ),
                    "window_motif_low_confidence_closures": len(low_confidence_closures),
                    "chord_sections": chord_sections,
                    "pre_chord_cleanup": pre_chord_cleanup,
                    "audio_repeat_consistency": audio_repeat_consistency,
                    "rhythmic_riff_consistency": rhythmic_riff_consistency,
                    "isolated_wide_chord_cleanup": isolated_wide_chord_cleanup,
                    "notes_added": 0,
                    "notes_removed": pre_chord_cleanup["notes_removed"]}
