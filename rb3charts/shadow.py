from __future__ import annotations

from dataclasses import replace


def apply_fretnet_note_shadow(notes, payload, onset_tolerance=0.08):
    if payload.get("schema") != 1 or payload.get("kind") != "fretnet_shadow_evidence":
        raise ValueError("Unsupported FretNet shadow evidence schema.")
    pitch_runs = payload.get("pitch_runs")
    if not isinstance(pitch_runs, list):
        raise ValueError("FretNet shadow evidence requires pitch_runs.")
    runs = sorted(
        (
            {
                "onset": float(row["onset_s"]),
                "offset": float(row["offset_s"]),
                "pitch": float(row["midi_pitch"]),
                "source_id": str(row["source_id"]),
            }
            for row in pitch_runs
        ),
        key=lambda row: (row["onset"], row["pitch"]),
    )
    report = {
        "provider": "fretnet",
        "strategy": "upper_median_pitch",
        "application_stage": "after_author_guitar_phrases",
        "onset_tolerance_seconds": float(onset_tolerance),
        "notes": len(notes),
        "single_notes": 0,
        "chords_preserved": 0,
        "matched": 0,
        "repetition_supported": 0,
        "unsupported_matches": 0,
        "changed": 0,
        "unchanged": 0,
        "assignments": [],
    }
    ordered = sorted(notes, key=lambda note: note.start)
    proposals = []
    for note in ordered:
        if len(note.lanes) != 1:
            report["chords_preserved"] += 1
            proposals.append(None)
            continue
        report["single_notes"] += 1
        changed = [
            row for row in runs
            if abs(row["onset"] - note.start) <= onset_tolerance
        ]
        active = [
            row for row in runs
            if row["onset"] <= note.start < row["offset"]
        ]
        matches = changed or active
        if not matches:
            report["unchanged"] += 1
            proposals.append(None)
            continue
        matches.sort(key=lambda row: (row["pitch"], row["source_id"]))
        selected = matches[len(matches) // 2]
        report["matched"] += 1
        proposals.append({
            "source_id": note.source_id,
            "start": round(float(note.start), 6),
            "match": "onset_change" if changed else "active_frame",
            "shadow_source_ids": [row["source_id"] for row in matches],
            "pitch_candidates": [row["pitch"] for row in matches],
            "original_pitch": float(note.pitch),
            "selected_pitch": selected["pitch"],
        })
    families = {}
    for index in range(len(ordered) - 3):
        window = ordered[index:index + 4]
        selected = proposals[index:index + 4]
        if any(row is None for row in selected):
            continue
        gaps = [b.start - a.start for a, b in zip(window, window[1:])]
        if any(gap <= 0 or gap > 0.75 for gap in gaps):
            continue
        pitches = [row["selected_pitch"] for row in selected]
        signature = (
            tuple(round(pitch - pitches[0], 3) for pitch in pitches),
            tuple(round(gap / 0.025) for gap in gaps),
        )
        families.setdefault(signature, []).append(index)
    supported = set()
    for occurrences in families.values():
        nonoverlapping = []
        for index in occurrences:
            if not nonoverlapping or index >= nonoverlapping[-1] + 4:
                nonoverlapping.append(index)
        if len(nonoverlapping) < 3:
            continue
        for index in nonoverlapping:
            supported.update(range(index, index + 4))
    result = []
    report["unsupported_matches"] = report["matched"] - len(supported)
    for index, (note, proposal) in enumerate(zip(ordered, proposals)):
        if proposal is None or index not in supported:
            result.append(replace(note))
            continue
        report["repetition_supported"] += 1
        report["assignments"].append(proposal)
        if proposal["selected_pitch"] != note.pitch:
            report["changed"] += 1
        else:
            report["unchanged"] += 1
        result.append(replace(note, pitch=proposal["selected_pitch"]))
    return result, report
