"""Structural validation is separate from musical accuracy and game testing."""
import math
import mido
from .model import DIFFICULTIES, TRACKS

def validate(charts, tempo, duration, midi_path=None):
    errors, warnings = [], list(tempo.warnings)
    counts = {}
    for instrument, difficulties in charts.items():
        counts[instrument] = {d:len(n) for d,n in difficulties.items()}
        if set(difficulties) != set(DIFFICULTIES):
            errors.append(f"{instrument}: missing difficulty")
            continue
        previous = None
        for difficulty in ("expert","hard","medium","easy"):
            notes = difficulties[difficulty]
            if not notes:
                errors.append(f"{instrument}/{difficulty}: empty chart")
            ancestry = {n.source_id for n in previous} if previous is not None else None
            prior = None
            for n in notes:
                tag = f"{instrument}/{difficulty} at {n.start:.3f}s"
                if not all(math.isfinite(x) for x in (n.start,n.end,n.confidence)):
                    errors.append(tag+": non-finite value")
                if n.start < 0 or n.end <= n.start or n.start >= duration or n.end > duration+0.1:
                    errors.append(tag+": invalid timing")
                if not n.lanes or len(set(n.lanes)) != len(n.lanes) or any(l not in range(5) for l in n.lanes):
                    errors.append(tag+": invalid lanes")
                if instrument == "drums" and len([l for l in n.lanes if l != 0]) > 2:
                    errors.append(tag+": more than two simultaneous hands")
                if instrument != "drums" and len(n.lanes)>3:
                    errors.append(tag+": more than three fret buttons")
                if ancestry is not None and n.source_id not in ancestry:
                    errors.append(tag+": reduction lost source ancestry")
                if prior and (n.start <= prior.start or n.start < prior.end):
                    errors.append(tag+": overlap or unsorted notes")
                prior = n
            if previous is not None and len(notes)>len(previous):
                errors.append(f"{instrument}/{difficulty}: denser than parent")
            previous = notes
        if counts[instrument]["expert"] == counts[instrument]["easy"]:
            warnings.append(f"{instrument}: Easy and Expert have the same event count; listen for appropriate simplicity.")
    if not charts:
        errors.append("No playable instruments generated")
    if midi_path:
        midi = mido.MidiFile(midi_path)
        names = {t.name for t in midi.tracks}
        for required in ["TEMPO", "EVENTS", "BEAT"] + [TRACKS[x] for x in charts]:
            if required not in names:
                errors.append("Missing MIDI track: "+required)
        event_texts = {m.text for t in midi.tracks for m in t if m.type=="text"}
        for text in ("[music_start]","[music_end]","[end]"):
            if text not in event_texts:
                errors.append("Missing event: "+text)
        for tr in midi.tracks:
            active=set()
            allowed = {b+l for b in DIFFICULTIES.values() for l in range(7)} | {103,110,111,112,116,120,121,122,123,124}
            for msg in tr:
                if msg.type=="note_on" and msg.velocity:
                    if tr.name.startswith("PART") and msg.note not in allowed:
                        errors.append(f"{tr.name}: invalid MIDI pitch {msg.note}")
                    if msg.note in active:
                        errors.append(f"{tr.name}: overlapping MIDI note {msg.note}")
                    active.add(msg.note)
                elif msg.type=="note_off" or (msg.type=="note_on" and not msg.velocity):
                    if msg.note not in active:
                        errors.append(f"{tr.name}: orphan note-off {msg.note}")
                    active.discard(msg.note)
            if active:
                errors.append(f"{tr.name}: missing note-offs")
    warnings.append("Automatic transcription needs listening review. Structural validation does not certify musical accuracy or Xbox compatibility.")
    return {"passed":not errors,"errors":errors,"warnings":warnings,"counts":counts}

def report_text(report):
    return "STRUCTURAL VALIDATION: "+("PASS" if report["passed"] else "FAIL")+"\n\n"+"\n".join("ERROR: "+x for x in report["errors"])+"\n"+"\n".join("REVIEW: "+x for x in report["warnings"])+"\n\n"+"\n".join(f"{i}: {counts}" for i,counts in report["counts"].items())+"\n"
