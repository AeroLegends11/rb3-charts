"""Phrase-level lead selection and five-fret authoring, without reference charts.

This is an explicit musical/ergonomic heuristic, not a model of human judgment.
It never creates an audio event or replaces a detected pitch with a fret number.
"""
from dataclasses import replace
import numpy as np


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


def _phrase_signature(rows, tempo):
    root = rows[0].pitch
    return (tuple(note.pitch - root for note in rows),
            tuple(round((tempo.beat_at(b.start) - tempo.beat_at(a.start)) * 12)
                  for a, b in zip(rows, rows[1:])))


def _motif_constraints(notes, tempo, window=4):
    ordered = [note for note in sorted(notes, key=lambda note: note.start) if len(note.lanes) == 1]
    repeated = {}
    for index in range(len(ordered) - window + 1):
        rows = ordered[index:index + window]
        gaps = [tempo.beat_at(b.start) - tempo.beat_at(a.start) for a, b in zip(rows, rows[1:])]
        if any(gap <= 0 or gap > 2 for gap in gaps):
            continue
        signature = (tuple(int(round(b.pitch - a.pitch)) for a, b in zip(rows, rows[1:])),
                     tuple(round(gap * 4) / 4 for gap in gaps))
        repeated.setdefault(signature, []).append((index, rows))
    constraints = {}
    for occurrences in repeated.values():
        distinct = []
        for occurrence in occurrences:
            if not distinct or occurrence[0] - distinct[-1][0] >= window:
                distinct.append(occurrence)
        if len(distinct) < 2:
            continue
        for _, rows in distinct:
            for note in rows:
                constraints[id(note)] = note.lanes[0]
    return constraints


def _constraint_cost(lanes, preferred):
    if preferred is None:
        return 0
    return float(sum(abs(lane - target) for lane, target in zip(lanes, preferred) if target is not None))


def _fingering(rows, previous=None, preferred=None):
    pitches = [note.pitch for note in rows]
    unique = sorted(set(pitches))
    if len(unique) <= 5:
        # Distinct local pitches get distinct adjacent frets; one rare solo
        # pitch elsewhere cannot flatten this riff into a single color.
        options=[[offset+unique.index(pitch) for pitch in pitches] for offset in range(6-len(unique))]
        if previous is None:
            if preferred is None:
                return options[(5-len(unique))//2]
            center = np.mean([target for target in preferred if target is not None])
            return min(options, key=lambda lanes: (_constraint_cost(lanes, preferred),
                                                 abs(np.mean(lanes) - center)))
        best=min(options,key=lambda lanes:(_transition_cost(pitches[0]-previous.pitch,
                 previous.lanes[0],lanes[0],rows[0].start-previous.start)
                 + .6 * _constraint_cost(lanes, preferred)))
        if _transition_cost(pitches[0]-previous.pitch,previous.lanes[0],best[0],rows[0].start-previous.start)<5:
            return best
        # No rigid pitch dictionary joins this boundary naturally. Let the
        # sequence optimizer preserve the incoming note and choose a later reset.
    target = np.interp(pitches, unique, np.linspace(0, 4, len(unique)))
    costs = .2 * (np.arange(5) - target[0])**2
    if preferred is not None:
        if preferred[0] is not None:
            costs += .6 * np.abs(np.arange(5) - preferred[0])
    if previous is not None:
        costs+=np.array([_transition_cost(pitches[0]-previous.pitch,previous.lanes[0],lane,
                                         rows[0].start-previous.start) for lane in range(5)])
    links = []
    for i in range(1, len(rows)):
        interval = pitches[i] - pitches[i - 1]
        matrix = np.zeros((5, 5))
        for old in range(5):
            for new in range(5):
                matrix[old, new] = (costs[old]
                                    + _transition_cost(interval,old,new,rows[i].start-rows[i-1].start)
                                    + .2 * (new - target[i])**2
                                    + (.6 * abs(new - preferred[i]) if preferred is not None and preferred[i] is not None else 0))
        links.append(np.argmin(matrix, axis=0))
        costs = np.min(matrix, axis=0)
    lane = int(np.argmin(costs))
    output = [lane]
    for link in reversed(links):
        lane = int(link[lane])
        output.append(lane)
    return list(reversed(output))


def author_guitar_phrases(notes, tempo):
    """Sight-reading pass: phrase contour, repeatable fingering, short tails.

    Preserve event identity, pitch, attack timing and count. Chords are barriers
    to single-note fingering; their already-mapped voicing is retained. Upstream
    repeated-phrase stabilization can contribute soft lane templates, but this
    pass owns the final Expert fret assignment.
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
    motif_constraints = _motif_constraints(result, tempo)
    vocabulary = {}
    constrained_vocabulary = {}
    phrase_report = []
    indices={id(note):i for i,note in enumerate(result)}
    constrained = 0
    for rows in phrases:
        index=indices[id(rows[0])]
        previous=result[index-1] if index else None
        period=tempo.seconds_at(tempo.beat_at(rows[0].start)+1)-rows[0].start
        if previous is not None and (len(previous.lanes)!=1 or rows[0].start-previous.start>period*.75):
            previous=None
        signature = _phrase_signature(rows, tempo)
        preferred = [motif_constraints.get(id(note)) for note in rows]
        constrained_phrase = all(target is not None for target in preferred)
        preferred = preferred if constrained_phrase else None
        constraint_key = (signature, tuple(preferred)) if preferred is not None else None
        constrained += constrained_phrase
        lanes = constrained_vocabulary.get(constraint_key) if constraint_key is not None else None
        reused = lanes is not None
        if lanes is None:
            lanes = vocabulary.get(signature)
            reused = lanes is not None
        if lanes is not None and preferred is not None and _constraint_cost(lanes, preferred):
            candidate = _fingering(rows, previous, preferred=preferred)
            if _constraint_cost(candidate, preferred) < _constraint_cost(lanes, preferred):
                lanes = candidate
                reused = False
                constrained_vocabulary[constraint_key] = lanes
        shift=0
        if lanes is not None and previous is not None:
            shift=min(range(-min(lanes),5-max(lanes)),key=lambda s:
                      _transition_cost(rows[0].pitch-previous.pitch,previous.lanes[0],lanes[0]+s,
                                       rows[0].start-previous.start)
                      + .6 * _constraint_cost([lane+s for lane in lanes], preferred)
                      + .01*abs(s))
            lanes=[lane+shift for lane in lanes]
            if _transition_cost(rows[0].pitch-previous.pitch,previous.lanes[0],lanes[0],rows[0].start-previous.start)>=5:
                lanes=None
                reused=False
        if lanes is None:
            lanes = _fingering(rows,previous,preferred=preferred)
            if constraint_key is not None:
                constrained_vocabulary[constraint_key] = lanes
            else:
                vocabulary.setdefault(signature,lanes)
        before = [n.lanes[0] for n in rows]
        for note, lane in zip(rows, lanes):
            note.lanes = (lane,)
        phrase_report.append({"start": rows[0].start, "end": rows[-1].end,
                              "source_ids": [n.source_id for n in rows],
                              "pitches": [n.pitch for n in rows], "before": before,
                              "after": lanes, "reused_fingering": reused,
                              "boundary_fret_shift":shift,"joined_previous_phrase":previous is not None,
                              "motif_constrained": constrained_phrase})
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
    return result, {"method": "final phrase authoring with bounded repeated-phrase lane constraints",
                    "before": contour(sorted(notes, key=lambda n:n.start)), "after": contour(result),
                    "phrases": phrase_report, "short_tails_removed": tails,
                    "motif_constrained_phrases": constrained,
                    "notes_added": 0, "notes_removed": 0}
