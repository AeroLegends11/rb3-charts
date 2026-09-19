"""Lane-independent evidence for pitched transcription candidates.

Scores are transparent heuristics, not calibrated probabilities. Beat alignment
and repetition can support an audible event, but cannot create one.
"""
from collections import Counter, defaultdict
import numpy as np


def interpret_events(events, y, sr, tempo=None, threshold=0.35):
    """Return accepted copies and an audit containing *all* supplied candidates.

    No lane, density cap, quantization, pitch correction or timing offset is
    applied here. Model confidence and model onset evidence remain separate.
    """
    rows = sorted((dict(e) for e in events), key=lambda e: (e["start"], e["pitch"]))
    hop = max(1, round(sr * .01))
    # RMS frames avoid repeatedly scanning an entire stem for every candidate.
    padded = np.pad(np.asarray(y, dtype=float), (0, (-len(y)) % hop))
    rms = np.sqrt(np.mean(padded.reshape(-1, hop)**2, axis=1)) if len(padded) else np.zeros(1)

    def level(start, end):
        a = max(0, int(np.floor(start * sr / hop)))
        b = min(len(rms), int(np.ceil(end * sr / hop)))
        return float(np.median(rms[a:b])) if b > a else 0.0

    for event in rows:
        start, end = event["start"], event["end"]
        before = level(start - .06, start - .01)
        after = level(start, min(end, start + .06))
        local = level(start - .5, end + .5)
        attack = float(np.clip((after - before) / max(after, before, 1e-8), 0, 1))
        beat = tempo.beat_at(start) if tempo else None
        alignment = None
        if tempo:
            # Straight sixteenths and eighth-note triplets, measured in seconds
            # against the piecewise map rather than a song-wide BPM.
            distance = min(abs(tempo.seconds_at(round(beat * grid) / grid) - start)
                           for grid in (4, 3))
            alignment = float(np.clip(1 - distance / .04, 0, 1))
        event["evidence"] = {
            "model_confidence": float(event["confidence"]),
            "model_onset_strength": event.get("model_onset_strength"),
            "attack_strength": attack,
            "rms": after,
            "relative_level": after / max(local, 1e-8),
            "duration_seconds": float(end - start),
            "beat_position": beat,
            "beat_alignment": alignment,
            "repeated_phrase": False,
        }

    # Group simultaneous pitches before recognizing patterns so a chord cannot
    # masquerade as a repeated rapid phrase. Repetition must occur independently.
    groups = []
    for i, event in enumerate(rows):
        if groups and event["start"] - rows[groups[-1][0]]["start"] <= .03:
            groups[-1].append(i)
        else:
            groups.append([i])
    motifs = defaultdict(list)
    if tempo:
        for i in range(len(groups) - 3):
            window = groups[i:i + 4]
            if any(max(rows[j]["evidence"]["rms"] for j in g) < 1e-5 for g in window):
                continue
            beats = [tempo.beat_at(rows[g[0]]["start"]) for g in window]
            gaps = np.diff(beats)
            if min(gaps) < .12 or max(gaps) > 2:
                continue
            pitches = tuple(tuple(sorted(rows[j]["pitch"] for j in g)) for g in window)
            key = (pitches, tuple(int(round(gap * 12)) for gap in gaps))
            motifs[key].append(i)
        for occurrences in motifs.values():
            independent = []
            for i in occurrences:
                if not independent or i >= independent[-1] + 4:
                    independent.append(i)
            if len(independent) >= 2:
                for i in occurrences:
                    for group in groups[i:i + 4]:
                        for j in group:
                            rows[j]["evidence"]["repeated_phrase"] = True

    accepted = []
    for event in rows:
        evidence = event["evidence"]
        model = evidence["model_confidence"]
        onset = evidence["model_onset_strength"]
        attack = evidence["attack_strength"]
        support = max(attack, onset or 0)
        audible = evidence["rms"] >= 1e-5
        valid = 0 <= event["start"] < len(y) / sr and evidence["duration_seconds"] >= .055
        # A confident sustained/legato note need not have a sharp broadband
        # attack. Reject only weak candidates lacking independent support.
        ordinary = model >= threshold and (model >= .65 or support >= .25)
        rescued = (model >= max(.15, threshold * .75) and support >= .4
                   and evidence["repeated_phrase"])
        keep = valid and audible and (ordinary or rescued)
        if not valid:
            classification, reason = "artifact", "invalid timing or candidate shorter than 55 ms"
        elif not audible:
            classification, reason = "artifact", "no measurable stem energy at candidate"
        elif keep and model >= .75 and support >= .55:
            classification, reason = "definite", "strong pitch confidence and attack evidence"
        elif keep:
            classification = "probable"
            reason = ("audible repeated phrase supports weak candidate" if rescued and not ordinary
                      else "model confidence supported by attack or sustained pitch activation")
        elif model < threshold:
            classification, reason = "uncertain", "below confidence threshold without supported repetition"
        else:
            classification, reason = "uncertain", "weak pitch activation without a supported new attack"
        # Keep original backend confidence for the existing mapper; record the
        # combined evidence score separately to avoid silently retuning it.
        evidence["evidence_score"] = float(np.clip(.65 * model + .30 * support
                                                  + .05 * evidence["repeated_phrase"], 0, 1))
        event.update(accepted=keep, classification=classification, reason=reason)
        if keep:
            accepted.append(event)
    report = {"version": 1, "score_kind": "heuristic_not_probability",
              "scope": "pitched model candidates before role selection and lane mapping",
              "candidate_count": len(rows), "accepted_count": len(accepted),
              "rejected_count": len(rows) - len(accepted),
              "classifications": dict(Counter(e["classification"] for e in rows)),
              "events": rows}
    return accepted, report


def write_event_trace(folder, instrument, selected, expert):
    """Join pre-mapping evidence to selected roles and final Expert ancestry.

    JSON retains complete features; CSV offers a compact chronological audit.
    Later gameplay changes are explicitly distinguished from detection rejects.
    """
    import csv
    import json
    from pathlib import Path
    from .model import save_json
    folder = Path(folder)
    selected_ids = {event["source_id"] for event in selected}
    final = defaultdict(list)
    for note in expert:
        for source_id in note.source_id.split("+"):
            final[source_id].append({"start": note.start, "end": note.end,
                                     "lanes": list(note.lanes), "pitch": note.pitch})
    rows = []
    for path in sorted(folder.glob(f"{instrument}-*-events.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8"))["events"])
    detected_ids = {event["source_id"] for event in rows}
    # Gameplay-generated pulses are not mislabeled as audio detections.
    for event in selected:
        if event["source_id"] not in detected_ids:
            rows.append(dict(event, accepted=True, classification="derived",
                             reason="created by role interpretation, not a detected candidate"))
    for event in rows:
        source_id = event["source_id"]
        event["expert_notes"] = final.get(source_id, [])
        event["outcome"] = ("expert" if event["expert_notes"] else
                            "detection_rejected" if not event["accepted"] else
                            "role_filtered" if source_id not in selected_ids else
                            "mapping_or_playability_filtered")
    rows.sort(key=lambda event: (event["start"], event["pitch"], event["source_id"]))
    save_json(folder / f"{instrument}-event-trace.json", {
        "score_kind": "heuristic_not_probability",
        "outcomes": dict(Counter(event["outcome"] for event in rows)), "events": rows})
    fields = ["source_id", "start", "end", "pitch", "confidence", "classification",
              "reason", "outcome", "model_onset_strength", "attack_strength",
              "rms", "beat_alignment", "repeated_phrase", "evidence_score", "expert_notes"]
    with (folder / f"{instrument}-event-trace.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for event in rows:
            writer.writerow(dict(event, **event.get("evidence", {}),
                                 expert_notes=json.dumps(event["expert_notes"])))
