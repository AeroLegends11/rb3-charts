"""Phrase-level lead selection and five-fret authoring, without reference charts.

This is an explicit musical/ergonomic heuristic, not a model of human judgment.
It never creates an audio event or replaces a detected pitch with a fret number.
"""
from dataclasses import replace
import numpy as np
from .motifs import discover_phrase_motifs, discover_window_motifs

MOTIF_CONSTRAINT_WEIGHT = 2.0
MAX_INFERRED_CHORD_RATE = .4


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


def _apply_band_chord_sections(notes, tempo, evidence):
    report = {
        "method": "bounded fifth-evidence sections with repeated root power-chord attacks",
        "gate_passed": False,
        "candidate_seeds": 0,
        "candidate_notes": 0,
        "chord_budget": 0,
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
    if correction_rate > .15 or octave_anchor_rate > .3:
        return report
    report["gate_passed"] = True
    candidates = sorted(
        (dict(event) for event in evidence.get("candidates", ()) if event.get("accepted", True)),
        key=lambda event: (event["start"], event["pitch"]),
    )
    groups = []
    for event in candidates:
        if groups and event["start"] - groups[-1][0]["start"] <= .045:
            groups[-1].append(event)
        else:
            groups.append([event])
    seed_beats = []
    for group in groups:
        supported = False
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
                    break
            if supported:
                break
        if supported:
            seed_beats.append(tempo.beat_at(group[0]["start"]))
    report["candidate_seeds"] = len(seed_beats)
    if len(seed_beats) < 3:
        return report
    seed_beats.sort()
    clusters = []
    cursor = 0
    while cursor < len(seed_beats):
        cluster = [seed_beats[cursor]]
        stop = cursor + 1
        while (
            stop < len(seed_beats)
            and seed_beats[stop] - cluster[-1] <= 10
            and seed_beats[stop] - cluster[0] <= 16
        ):
            cluster.append(seed_beats[stop])
            stop += 1
        clusters.append(cluster)
        cursor = stop if len(cluster) >= 3 else cursor + 1
    beats = [tempo.beat_at(note.start) for note in notes]
    region_proposals = []
    regions = []
    for cluster in clusters:
        if len(cluster) < 3:
            continue
        region_indices = [
            index for index, beat in enumerate(beats)
            if cluster[0] <= beat <= cluster[-1]
        ]
        contiguous = []
        for index in region_indices:
            if contiguous and beats[index] - beats[contiguous[-1]] > 2.25:
                if len(contiguous) >= 4:
                    regions.append(contiguous)
                contiguous = []
            contiguous.append(index)
        if len(contiguous) >= 4:
            regions.append(contiguous)
    for region in regions:
        pitch_counts = {}
        for index in region:
            pitch = notes[index].pitch
            pitch_counts[pitch] = pitch_counts.get(pitch, 0) + 1
        pitch_limit = float(np.quantile([notes[index].pitch for index in region], .6))
        proposals = {}
        for index in region:
            note = notes[index]
            if (len(note.lanes) != 1
                    or pitch_counts[note.pitch] < 3
                    or note.pitch > pitch_limit):
                continue
            proposals[index] = (pitch_counts[note.pitch], note.confidence, -note.start, index)
        region_proposals.append((region, list(proposals.values())))
    existing_chords = sum(len(note.lanes) > 1 for note in notes)
    chord_budget = max(0, int(np.ceil(len(notes) * MAX_INFERRED_CHORD_RATE)) - existing_chords)
    report["candidate_notes"] = sum(len(proposals) for _, proposals in region_proposals)
    report["chord_budget"] = chord_budget
    changed = []
    for region, proposals in region_proposals:
        existing_region_chords = sum(len(notes[index].lanes) > 1 for index in region)
        region_budget = max(
            0,
            int(np.ceil(len(region) * MAX_INFERRED_CHORD_RATE)) - existing_region_chords,
        )
        selected = sorted(proposals, reverse=True)[:min(region_budget, chord_budget - len(changed))]
        selected.sort(key=lambda row: notes[row[-1]].start)
        runs = []
        for proposal in selected:
            note = notes[proposal[-1]]
            if (
                runs
                and tempo.beat_at(note.start) - tempo.beat_at(notes[runs[-1][-1][-1]].start) <= 2.25
            ):
                runs[-1].append(proposal)
            else:
                runs.append([proposal])
        for run in runs:
            if len(run) < 3:
                continue
            for _, _, _, index in run:
                note = notes[index]
                root = note.lanes[0]
                note.lanes = (root, root + 2) if root <= 2 else (root - 2, root)
                changed.append(note)
        if len(changed) >= chord_budget:
            break
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


def author_guitar_phrases(notes, tempo, chord_evidence=None):
    """Sight-reading pass: phrase contour, repeatable fingering, short tails.

    Preserve event identity, pitch, attack timing and count. Chords are barriers
    to single-note fingering; their already-mapped voicing is retained.
    """
    result = [replace(n) for n in sorted(notes, key=lambda n: n.start)]
    phrases = []
    current = []
    for note in result:
        if len(note.lanes) != 1:
            if current:
                phrases.append(current)
                current = []
            continue
        if current:
            beat = tempo.beat_at(note.start)
            period = tempo.seconds_at(beat + 1) - tempo.seconds_at(beat)
            rest = note.start - current[-1].end
            if rest > period * .75 or beat - tempo.beat_at(current[0].start) >= 8:
                phrases.append(current)
                current = []
        current.append(note)
    if current:
        phrases.append(current)
    vocabulary = {}
    phrase_report = []
    indices={id(note):i for i,note in enumerate(result)}
    local_lanes = {}
    motif_assignments = discover_phrase_motifs(phrases, tempo)
    window_matches, window_diagnostics = discover_window_motifs(phrases, tempo)
    baseline = [replace(n) for n in result]
    continuity_summary = {"adjacent_single": 0, "through_chord": 0, "none": 0}
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
        if motif_info is not None and previous is None:
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
                              "window_motif_constrained": window_constrained})
    for (phrase_index, note_index), closure in low_confidence_closures.items():
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
        if .065 < duration < .22:
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
    for phrase, rows in zip(phrase_report, phrases):
        phrase["final_lanes"] = [list(note.lanes) for note in rows]
    return result, {"method": "phrase-local fingering with whole-sequence ergonomic costs and bounded motif windows",
                    "before": contour(sorted(notes, key=lambda n:n.start)), "after": contour(result),
                    "continuity_summary": continuity_summary,
                    "phrases": phrase_report, "short_tails_removed": tails,
                    "window_motif_matching": window_diagnostics,
                    "window_motif_constrained_phrases": sum(
                        phrase["window_motif_constrained"] for phrase in phrase_report
                    ),
                    "window_motif_low_confidence_closures": len(low_confidence_closures),
                    "chord_sections": chord_sections,
                    "notes_added": 0, "notes_removed": 0}
