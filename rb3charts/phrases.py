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


def _motif_windows(phrases, tempo, min_window=3, max_window=5):
    windows = []
    for phrase_index, rows in enumerate(phrases):
        for start in range(len(rows) - min_window + 1):
            stop = min(len(rows), start + max_window)
            for end in range(start + min_window, stop + 1):
                window = rows[start:end]
                beat_gaps = [tempo.beat_at(b.start) - tempo.beat_at(a.start) for a, b in zip(window, window[1:])]
                if any(gap <= 0 or gap > 2 for gap in beat_gaps):
                    continue
                windows.append({
                    "phrase_index": phrase_index,
                    "start": start,
                    "rows": window,
                    "beat_gaps": beat_gaps,
                    "start_time": window[0].start,
                })
    return sorted(windows, key=lambda row: (row["start_time"], row["phrase_index"], row["start"]))


def _motif_alignment(a, b, max_edits=1):
    rows_a, rows_b = a["rows"], b["rows"]
    if abs(len(rows_a) - len(rows_b)) > max_edits:
        return {"accepted": False, "reason": "length_gap_too_large"}
    transpose_seed = [rows_b[0].pitch - rows_a[0].pitch]
    overlap = min(len(rows_a), len(rows_b))
    transpose_seed.append(float(np.median([rows_b[i].pitch - rows_a[i].pitch for i in range(overlap)])))
    transpose_seed.extend(float(rows_b[j].pitch - rows_a[i].pitch)
                          for i in range(len(rows_a)) for j in range(len(rows_b)))
    transposes = sorted(set(int(round(seed)) for seed in transpose_seed))
    best = None
    for transpose in transposes:
        n, m = len(rows_a), len(rows_b)
        costs = np.full((n + 1, m + 1), float("inf"))
        links = [[None] * (m + 1) for _ in range(n + 1)]
        costs[0, 0] = 0
        for i in range(1, n + 1):
            costs[i, 0] = i
            links[i][0] = (i - 1, 0, "delete", 1)
        for j in range(1, m + 1):
            costs[0, j] = j
            links[0][j] = (0, j - 1, "insert", 1)
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                pitch_error = abs((rows_b[j - 1].pitch - rows_a[i - 1].pitch) - transpose)
                gap_a = a["beat_gaps"][i - 2] if i > 1 else None
                gap_b = b["beat_gaps"][j - 2] if j > 1 else None
                rhythm_error = abs(gap_a - gap_b) if gap_a is not None and gap_b is not None else 0.0
                contour_a = int(np.sign(rows_a[i - 1].pitch - rows_a[i - 2].pitch)) if i > 1 else 0
                contour_b = int(np.sign(rows_b[j - 1].pitch - rows_b[j - 2].pitch)) if j > 1 else 0
                lane_a = int(np.sign(rows_a[i - 1].lanes[0] - rows_a[i - 2].lanes[0])) if i > 1 else 0
                lane_b = int(np.sign(rows_b[j - 1].lanes[0] - rows_b[j - 2].lanes[0])) if j > 1 else 0
                contour_ok = contour_a == 0 or contour_b == 0 or contour_a == contour_b
                lane_ok = lane_a == 0 or lane_b == 0 or lane_a == lane_b
                substitute = 0 if pitch_error <= 1 and rhythm_error <= .25 and contour_ok else 1
                options = (
                    (costs[i - 1, j] + 1, (i - 1, j, "delete", 1)),
                    (costs[i, j - 1] + 1, (i, j - 1, "insert", 1)),
                    (costs[i - 1, j - 1] + substitute, (i - 1, j - 1, "match", substitute)),
                )
                value, link = min(options, key=lambda row: (row[0], row[1][2] != "match"))
                costs[i, j] = value
                links[i][j] = link
        total = int(costs[n, m])
        i, j = n, m
        aligned = []
        pitch_hits = rhythm_hits = contour_hits = lane_hits = 0
        edits = 0
        max_pitch_error = 0
        while i or j:
            prev_i, prev_j, action, penalty = links[i][j]
            if action == "match":
                aligned.append((prev_i, prev_j))
                edits += penalty
                aa, bb = rows_a[prev_i], rows_b[prev_j]
                pitch_error = abs((bb.pitch - aa.pitch) - transpose)
                max_pitch_error = max(max_pitch_error, pitch_error)
                pitch_hits += pitch_error <= 1
                if prev_i and prev_j:
                    gap_error = abs(a["beat_gaps"][prev_i - 1] - b["beat_gaps"][prev_j - 1])
                    rhythm_hits += gap_error <= .25
                    contour_a = int(np.sign(aa.pitch - rows_a[prev_i - 1].pitch))
                    contour_b = int(np.sign(bb.pitch - rows_b[prev_j - 1].pitch))
                    lane_a = int(np.sign(aa.lanes[0] - rows_a[prev_i - 1].lanes[0]))
                    lane_b = int(np.sign(bb.lanes[0] - rows_b[prev_j - 1].lanes[0]))
                    contour_hits += contour_a == 0 or contour_b == 0 or contour_a == contour_b
                    lane_hits += lane_a == 0 or lane_b == 0 or lane_a == lane_b
            else:
                edits += penalty
            i, j = prev_i, prev_j
        aligned.reverse()
        compared = max(1, len(aligned))
        linked = max(1, sum(1 for ia, ib in aligned if ia > 0 and ib > 0))
        pitch_score = pitch_hits / compared
        rhythm_score = rhythm_hits / linked
        contour_score = contour_hits / linked
        lane_score = lane_hits / linked
        confidence = (.45 * pitch_score + .25 * rhythm_score + .2 * contour_score + .1 * lane_score)
        accepted = (total <= max_edits and len(aligned) >= 3 and pitch_score >= .6
                    and confidence >= .72 and max_pitch_error <= 5)
        candidate = {
            "accepted": accepted,
            "transpose": transpose,
            "edits": total,
            "pitch_score": round(pitch_score, 3),
            "rhythm_score": round(rhythm_score, 3),
            "contour_score": round(contour_score, 3),
            "lane_shape_score": round(lane_score, 3),
            "confidence": round(confidence, 3),
            "max_pitch_error": round(float(max_pitch_error), 3),
            "alignment": aligned,
            "reason": "accepted" if accepted else ("confidence_below_threshold" if total <= max_edits else "edit_distance_exceeded"),
        }
        if best is None or candidate["confidence"] > best["confidence"] or (
                candidate["confidence"] == best["confidence"] and candidate["edits"] < best["edits"]):
            best = candidate
    return best or {"accepted": False, "reason": "no_alignment"}


def _project_shape(reference_rows, alignment, transpose):
    if not alignment:
        return {}
    aligned_reference = [reference_rows[source] for source, _ in alignment]
    lower = -min(note.lanes[0] for note in aligned_reference)
    upper = 4 - max(note.lanes[0] for note in aligned_reference)
    shift = int(np.clip(round(float(transpose) / 5.0), lower, upper))
    if shift == 0 and abs(transpose) >= 4:
        shift = int(np.clip(int(np.sign(transpose)), lower, upper))
    lanes = {}
    for source, target in alignment:
        lanes[target] = int(np.clip(reference_rows[source].lanes[0] + shift, 0, 4))
    return lanes


def _motif_constraints(phrases, tempo):
    windows = _motif_windows(phrases, tempo)
    evidence = {}
    diagnostics = {"accepted": [], "rejected": []}
    for i in range(len(windows)):
        for j in range(i + 1, len(windows)):
            left, right = windows[i], windows[j]
            if abs(len(left["rows"]) - len(right["rows"])) > 1:
                continue
            if left["phrase_index"] == right["phrase_index"]:
                left_end = left["start"] + len(left["rows"])
                right_end = right["start"] + len(right["rows"])
                if max(left["start"], right["start"]) < min(left_end, right_end):
                    continue
            rhythm_delta = abs(np.median(left["beat_gaps"]) - np.median(right["beat_gaps"]))
            if rhythm_delta > .5:
                diagnostics["rejected"].append({
                    "left_start": left["start_time"],
                    "right_start": right["start_time"],
                    "reason": "rhythm_gap_too_large",
                    "rhythm_delta": round(float(rhythm_delta), 3),
                })
                continue
            alignment = _motif_alignment(left, right)
            row = {
                "left_start": left["start_time"],
                "right_start": right["start_time"],
                "left_source_ids": [note.source_id for note in left["rows"]],
                "right_source_ids": [note.source_id for note in right["rows"]],
                "edits": alignment.get("edits"),
                "confidence": alignment.get("confidence"),
                "transpose": alignment.get("transpose"),
                "alignment": alignment.get("alignment", []),
                "reason": alignment.get("reason"),
            }
            if not alignment["accepted"]:
                diagnostics["rejected"].append(row)
                continue
            diagnostics["accepted"].append(row)
            projected = _project_shape(left["rows"], alignment["alignment"], alignment["transpose"])
            for source, target in alignment["alignment"]:
                score = alignment["confidence"]
                candidate = (left["rows"][source].lanes[0], score)
                note_id = id(left["rows"][source])
                if note_id not in evidence or candidate[1] > evidence[note_id][1]:
                    evidence[note_id] = candidate
                lane = projected.get(target)
                if lane is None:
                    continue
                candidate = (lane, score)
                note_id = id(right["rows"][target])
                if note_id not in evidence or candidate[1] > evidence[note_id][1]:
                    evidence[note_id] = candidate
    constraints = {note_id: lane for note_id, (lane, _) in evidence.items()}
    diagnostics.update({
        "windows_considered": len(windows),
        "matches_accepted": len(diagnostics["accepted"]),
        "matches_rejected": len(diagnostics["rejected"]),
    })
    return constraints, diagnostics


def _constraint_cost(lanes, preferred):
    if preferred is None:
        return 0
    return float(sum(abs(lane - target) for lane, target in zip(lanes, preferred) if target is not None))


def _is_phrase_constrained(preferred):
    """Require full coverage, except one-note local edits in longer phrases."""
    covered = [index for index, target in enumerate(preferred) if target is not None]
    contiguous = bool(covered) and covered[-1] - covered[0] + 1 == len(covered)
    return (len(covered) == len(preferred) or
            (len(preferred) >= 5 and len(covered) == len(preferred) - 1 and contiguous))


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
    motif_constraints, motif_diagnostics = _motif_constraints(phrases, tempo)
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
        constrained_phrase = _is_phrase_constrained(preferred)
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
                    "motif_matching": motif_diagnostics,
                    "notes_added": 0, "notes_removed": 0}
