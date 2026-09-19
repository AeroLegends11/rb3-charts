"""Source-ID stage differences; metrics describe detected events, not audio truth."""
from collections import Counter, defaultdict
from dataclasses import asdict
from copy import deepcopy


def changes(before, after):
    """Join by full source ID, refusing ambiguous identities instead of guessing."""
    left, right = defaultdict(list), defaultdict(list)
    for row in before:
        left[row["source_id"]].append(row)
    for row in after:
        right[row["source_id"]].append(row)
    result = []
    for identity in sorted(left.keys() | right.keys()):
        a, b = left[identity], right[identity]
        if not identity or len(a) > 1 or len(b) > 1:
            result.append({"source_id": identity, "kind": "ambiguous_identity",
                           "before": a, "after": b})
        elif not a or not b:
            result.append({"source_id": identity, "kind": "added" if b else "removed",
                           "before": a, "after": b})
        else:
            fields = {key: {"before": a[0][key], "after": b[0][key]}
                      for key in a[0] if a[0][key] != b[0][key]}
            if fields:
                result.append({"source_id": identity, "kind": "changed", "fields": fields,
                               "start": b[0]["start"]})
    return result


def metrics(rows):
    ordered = sorted(rows, key=lambda row: row["start"])
    pairs = [(a, b) for a, b in zip(ordered, ordered[1:])
             if len(a["lanes"]) == len(b["lanes"]) == 1 and b["start"] - a["start"] < .8]
    return {"attacks": len(rows),
            "chord_rate": sum(len(r["lanes"]) > 1 for r in rows) / len(rows) if rows else None,
            "hold_rate": sum(r["end"] - r["start"] >= .22 for r in rows) / len(rows) if rows else None,
            "contour_pairs": len(pairs),
            "collapsed_pitch_changes": sum(a["pitch"] != b["pitch"] and a["lanes"] == b["lanes"] for a, b in pairs),
            "reversed_pitch_changes": sum((b["pitch"] - a["pitch"]) * (b["lanes"][0] - a["lanes"][0]) < 0 for a, b in pairs)}


def repeat_families(rows, tempo):
    """Fixed four-event reference windows from mapped input, across the whole song.

    Exact relative pitch and 1/12-beat gap signatures only. No fuzzy matching,
    audio similarity, motif locking or inferred missing attacks is claimed.
    """
    families = defaultdict(list)
    counts = Counter(r["source_id"] for r in rows)
    for i in range(len(rows) - 3):
        window = rows[i:i + 4]
        if any(len(r["lanes"]) != 1 or not r["source_id"] or counts[r["source_id"]] != 1 for r in window):
            continue
        gaps = [tempo.beat_at(b["start"]) - tempo.beat_at(a["start"]) for a, b in zip(window, window[1:])]
        if min(gaps) < .12 or max(gaps) > 2:
            continue
        key = (tuple(r["pitch"] - window[0]["pitch"] for r in window),
               tuple(round(gap * 12) for gap in gaps))
        occurrences = families[key]
        if not occurrences or i >= occurrences[-1]["index"] + 4:
            occurrences.append({"index": i, "start": window[0]["start"],
                                "source_ids": [r["source_id"] for r in window]})
    return [v for _, v in sorted(families.items()) if len(v) >= 2]


def repeat_metrics(rows, families):
    counts = Counter(r["source_id"] for r in rows)
    by_id = {r["source_id"]: r for r in rows if counts[r["source_id"]] == 1}
    compared = absolute = shape = 0
    retained = 0
    covered = set()
    mismatches = []
    for family in families:
        observed = []
        for occurrence in family:
            ids = occurrence["source_ids"]
            if not all(identity in by_id and len(by_id[identity]["lanes"]) == 1 for identity in ids):
                continue
            lanes = [by_id[identity]["lanes"][0] for identity in ids]
            observed.append((occurrence, lanes))
            retained += 1
            covered.update(ids)
        # Compare each retained occurrence with the first, O(n), not all pairs.
        for b, lanes_b in observed[1:]:
            a, lanes_a = observed[0]
            compared += 1
            same = lanes_a == lanes_b
            same_shape = [x - lanes_a[0] for x in lanes_a] == [x - lanes_b[0] for x in lanes_b]
            absolute += same
            shape += same_shape
            if not same:
                mismatches.append({"first_start": a["start"], "start": b["start"],
                                   "first_lanes": lanes_a, "lanes": lanes_b,
                                   "same_shape": same_shape})
    return {"families": len(families), "input_occurrences": sum(map(len, families)),
            "retained_occurrences": retained, "covered_attacks": len(covered),
            "comparisons": compared,
            "absolute_lane_agreement": absolute / compared if compared else None,
            "relative_shape_agreement": shape / compared if compared else None,
            "mismatches": mismatches}


class StageAudit:
    def __init__(self, tempo, instrument):
        self.tempo = tempo
        self.instrument = instrument
        self.stages = []
        self.families = []

    def capture(self, name, notes):
        rows = [asdict(n) for n in notes]
        if not self.stages and self.instrument != "drums":
            self.families = repeat_families(rows, self.tempo)
        previous = self.stages[-1]["notes"] if self.stages else rows
        self.stages.append({"stage": name, "notes": rows,
                            "metrics": metrics(rows),
                            "repetition": repeat_metrics(rows, self.families),
                            "changes": changes(previous, rows)})

    def report(self):
        return deepcopy({"schema": 1, "instrument": self.instrument,
                         "scope": "post-voice-selection authoring; before drum activation alignment",
                         "confidence": "heuristic event signatures, not audio similarity or calibrated musical quality",
                         "hold_threshold_seconds": .22,
                         "stages": self.stages})
