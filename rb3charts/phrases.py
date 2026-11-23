"""Phrase-level lead selection and five-fret authoring, without reference charts.

This is an explicit musical/ergonomic heuristic, not a model of human judgment.
It never creates an audio event or replaces a detected pitch with a fret number.
"""
from dataclasses import replace
import numpy as np
from .motifs import discover_phrase_motifs


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


def _fingering(rows, previous=None):
    pitches = [note.pitch for note in rows]
    unique = sorted(set(pitches))
    if len(unique) <= 5:
        # Distinct local pitches get distinct adjacent frets; one rare solo
        # pitch elsewhere cannot flatten this riff into a single color.
        options=[[offset+unique.index(pitch) for pitch in pitches] for offset in range(6-len(unique))]
        if previous is None:
            return options[(5-len(unique))//2]
        best=min(options,key=lambda lanes:_transition_cost(pitches[0]-previous.pitch,
                 previous.lanes[0],lanes[0],rows[0].start-previous.start))
        if _transition_cost(pitches[0]-previous.pitch,previous.lanes[0],best[0],rows[0].start-previous.start)<5:
            return best
        # No rigid pitch dictionary joins this boundary naturally. Let the
        # sequence optimizer preserve the incoming note and choose a later reset.
    return _hand_window_lanes(pitches, unique, rows, previous)


def _hand_window_lanes(pitches, unique, rows, previous):
    """Fret by pitch rank inside a five-fret hand window.

    Rising runs that would pass orange reposition to green; falling runs that
    would pass green reposition to orange. Distinct nearby pitches keep distinct
    adjacent frets, so only the repositioning notes jump.
    """
    ranks = [unique.index(pitch) for pitch in pitches]
    reset_cost = 10
    states = {}
    for base in range(ranks[0] - 4, ranks[0] + 1):
        lane = ranks[0] - base
        cost = 0
        if previous is not None:
            cost += _transition_cost(pitches[0] - previous.pitch, previous.lanes[0], lane, rows[0].start - previous.start)
        states[base] = (cost, [lane])
    for i in range(1, len(ranks)):
        rank, step = ranks[i], ranks[i] - ranks[i - 1]
        nxt = {}
        for base, (cost, path) in states.items():
            if base <= rank <= base + 4:
                options = [(base, cost)]
            else:
                # Prefer green after rising, orange after falling.
                prefer_green = rank > base
                options = [(rank, cost + reset_cost + .01 * (len(ranks) - i) + (0 if prefer_green else 3)),
                           (rank - 4, cost + reset_cost + .01 * (len(ranks) - i) + (3 if prefer_green else 0))]
            for new_base, new_cost in options:
                lane = rank - new_base
                if lane == path[-1] and pitches[i] != pitches[i - 1]:
                    new_cost += 25  # A real pitch change must not repeat the previous color.
                if new_base == base:
                    new_cost += _transition_cost(pitches[i] - pitches[i - 1], path[-1], lane, rows[i].start - rows[i - 1].start)
                if new_base not in nxt or new_cost < nxt[new_base][0]:
                    nxt[new_base] = (new_cost, path + [lane])
        states = nxt
    return min(states.values(), key=lambda item: item[0])[1]


def _collapsed_changes(pitches, lanes):
    return sum(a != b and x == y for a, b, x, y in zip(pitches, pitches[1:], lanes, lanes[1:]))


def _separate_from_previous(lanes, rows, previous):
    """Move a phrase one fret when its first note repeats the previous color on a new pitch."""
    if (previous is None or len(previous.lanes) != 1 or rows[0].start - previous.start >= .8
            or previous.pitch == rows[0].pitch or lanes[0] != previous.lanes[0]):
        return lanes
    direction = 1 if rows[0].pitch > previous.pitch else -1
    for shift in (direction, -direction):
        moved = [lane + shift for lane in lanes]
        if min(moved) >= 0 and max(moved) <= 4:
            return moved
    return lanes


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


def author_guitar_phrases(notes, tempo):
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
    rank_vocabulary = {}
    phrase_report = []
    indices={id(note):i for i,note in enumerate(result)}
    local_lanes = {}
    motif_assignments = discover_phrase_motifs(phrases, tempo)
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
        # Same contour and rhythm in another scale/key also counts as the same pattern.
        unique_pitches = sorted({n.pitch for n in rows})
        rank_signature = (tuple(unique_pitches.index(n.pitch) for n in rows), signature[1])
        lanes = vocabulary.get(signature)
        intervals = [b.pitch - a.pitch for a, b in zip(rows, rows[1:])]
        if lanes is None and len(rows) >= 4:
            for known_intervals, known_lanes in rank_vocabulary.get(rank_signature, ()):
                if all(abs(x - y) <= 1 for x, y in zip(intervals, known_intervals)):
                    lanes = known_lanes
                    break
        reused = lanes is not None
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
        if motif_info is not None and previous is None:
            motif_reason = motif_info["reason"]
            canonical = family_lanes.get(motif_info["family_id"])
            if canonical is not None:
                projected = list(local_lanes[phrase_index])
                for source, target in motif_info["alignment"]:
                    if source < len(canonical) and target < len(projected):
                        projected[target] = canonical[source]
                pitch_list = [n.pitch for n in rows]
                if (_collapsed_changes(pitch_list, projected)
                        <= _collapsed_changes(pitch_list, local_lanes[phrase_index])):
                    if projected != local_lanes[phrase_index]:
                        motif_reused = True
                    lanes = projected
        pitch_list = [n.pitch for n in rows]
        if lanes is None or _collapsed_changes(pitch_list, lanes) > _collapsed_changes(pitch_list, local_lanes[phrase_index]):
            lanes = local_lanes[phrase_index]
            reused = False
        vocabulary.setdefault(signature,lanes)
        rank_vocabulary.setdefault(rank_signature,[]).append((intervals,lanes))
        if continuity_source == "adjacent_single" and previous is not None and previous.pitch == rows[0].pitch and len(previous.lanes) == 1:
            boundary_delta = previous.lanes[0] - lanes[0]
            if all(0 <= lane + boundary_delta <= 4 for lane in lanes):
                lanes = [lane + boundary_delta for lane in lanes]
        lanes = _separate_from_previous(lanes, rows, result[index - 1] if index else None)
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
                              "motif_reason": motif_reason})
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
    return result, {"method": "phrase-local fingering with whole-sequence ergonomic costs",
                    "before": contour(sorted(notes, key=lambda n:n.start)), "after": contour(result),
                    "continuity_summary": continuity_summary,
                    "phrases": phrase_report, "short_tails_removed": tails,
                    "notes_added": 0, "notes_removed": 0}
