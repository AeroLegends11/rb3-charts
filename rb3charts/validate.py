"""Structural validation is separate from musical accuracy and game testing."""
import math
import mido
from .model import DIFFICULTIES, TRACKS
from .midi import authored_tick

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
                if (
                    instrument == "guitar"
                    and difficulty == "expert"
                    and len(n.lanes) == 3
                    and 0 in n.lanes
                    and 4 in n.lanes
                ):
                    errors.append(tag+": Expert chord spans green through orange")
                if (
                    instrument == "guitar"
                    and difficulty != "expert"
                    and (len(n.lanes) > 2 or (0 in n.lanes and 4 in n.lanes))
                ):
                    errors.append(tag+": reduced chord violates fret-hand limits")
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
        if midi.type != 1:
            errors.append(f"MIDI type must be 1, found {midi.type}")
        if midi.ticks_per_beat != 960:
            errors.append(
                f"MIDI ticks per beat must be 960, found {midi.ticks_per_beat}"
            )
        tracks_by_name = {}
        for track in midi.tracks:
            tracks_by_name.setdefault(track.name, []).append(track)
        names = set(tracks_by_name)
        for required in ["TEMPO", "EVENTS", "BEAT"] + [TRACKS[x] for x in charts]:
            if required not in names:
                errors.append("Missing MIDI track: "+required)
            elif len(tracks_by_name[required]) != 1:
                errors.append("Duplicate MIDI track: "+required)
        event_tracks = tracks_by_name.get("EVENTS", [])
        event_text_rows = [
            message.text
            for track in event_tracks
            for message in track
            if message.type=="text"
        ]
        event_texts=set(event_text_rows)
        for text in ("[music_start]","[music_end]","[end]"):
            if text not in event_texts:
                errors.append("Missing EVENTS marker: "+text)
            elif event_text_rows.count(text)!=1:
                errors.append(
                    "Duplicate EVENTS marker: "+text
                )
        music_start_tick = None
        event_text_ticks = {}
        if event_tracks:
            absolute = 0
            for message in event_tracks[0]:
                absolute += message.time
                if message.type=="text":
                    event_text_ticks[message.text]=absolute
                    if message.text=="[music_start]":
                        music_start_tick=absolute
        if music_start_tick is not None:
            offset_tick = music_start_tick - round(
                tempo.beat_at(0)*midi.ticks_per_beat
            )
            offset_beats=round(offset_tick/midi.ticks_per_beat)
            expected_event_ticks={
                "[music_start]":music_start_tick,
                "[music_end]":authored_tick(
                    tempo,duration,offset_beats,midi.ticks_per_beat
                ),
                "[end]":round(
                    (
                        tempo.beat_at(duration+2)+offset_beats
                    )*midi.ticks_per_beat
                ),
            }
            for text,expected_tick in expected_event_ticks.items():
                actual_tick=event_text_ticks.get(text)
                if actual_tick is not None and actual_tick!=expected_tick:
                    errors.append(
                        f"EVENTS marker {text} is at tick {actual_tick}, "
                        f"expected {expected_tick}"
                    )
        tempo_tracks = tracks_by_name.get("TEMPO", [])
        if tempo_tracks and not any(
            message.type=="set_tempo"
            for message in tempo_tracks[0]
        ):
            errors.append("TEMPO track has no tempo events")
        if tempo_tracks and music_start_tick is not None:
            actual_tempos = []
            actual_signatures = []
            absolute = 0
            for message in tempo_tracks[0]:
                absolute += message.time
                if message.type=="set_tempo":
                    actual_tempos.append((absolute,message.tempo))
                elif message.type=="time_signature":
                    actual_signatures.append((
                        absolute,
                        message.numerator,
                        message.denominator,
                    ))
            offset_tick = music_start_tick - round(
                tempo.beat_at(0)*midi.ticks_per_beat
            )
            expected_tempos=[(
                0,
                round((tempo.beats[1]-tempo.beats[0])*1_000_000),
            )]
            expected_tempos.extend(
                (
                    offset_tick+i*midi.ticks_per_beat,
                    round(
                        (tempo.beats[i+1]-tempo.beats[i])*1_000_000
                    ),
                )
                for i in range(len(tempo.beats)-1)
            )
            if actual_tempos!=expected_tempos:
                errors.append(
                    "TEMPO events differ from the authored tempo map"
                )
            if actual_signatures!=[(0,tempo.numerator,4)]:
                errors.append(
                    "TEMPO time signature differs from the authored meter"
                )
        beat_tracks = tracks_by_name.get("BEAT", [])
        if beat_tracks:
            beat_pitches = {
                message.note
                for message in beat_tracks[0]
                if message.type=="note_on" and message.velocity
            }
            if not beat_pitches:
                errors.append("BEAT track has no beat notes")
            elif not beat_pitches <= {12,13}:
                errors.append(
                    "BEAT track contains pitches other than 12 and 13"
                )
            if music_start_tick is not None:
                offset_tick = music_start_tick - round(
                    tempo.beat_at(0)*midi.ticks_per_beat
                )
                offset_beats=round(offset_tick/midi.ticks_per_beat)
                end_beat=tempo.beat_at(duration+2)+offset_beats
                expected_beats=[
                    (
                        beat*midi.ticks_per_beat,
                        beat*midi.ticks_per_beat+120,
                        12
                        if (beat-offset_beats)%tempo.numerator==0
                        else 13,
                    )
                    for beat in range(math.floor(end_beat))
                ]
                actual_beats=[]
                active={}
                absolute=0
                for message in beat_tracks[0]:
                    absolute+=message.time
                    if (
                        message.type=="note_on"
                        and message.velocity
                    ):
                        active[message.note]=absolute
                    elif (
                        message.type=="note_off"
                        or (
                            message.type=="note_on"
                            and not message.velocity
                        )
                    ) and message.note in active:
                        actual_beats.append((
                            active.pop(message.note),
                            absolute,
                            message.note,
                        ))
                if actual_beats!=expected_beats:
                    errors.append(
                        "BEAT track timing differs from the authored beat grid"
                    )
        for tr in midi.tracks:
            active=set()
            instrument = next(
                (
                    name for name,track_name in TRACKS.items()
                    if track_name==tr.name
                ),
                None,
            )
            allowed = {
                base+lane
                for base in DIFFICULTIES.values()
                for lane in range(5)
            } | {116}
            if instrument=="drums":
                allowed |= {109,110,111,112,120,121,122,123,124}
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
        for instrument,difficulties in charts.items():
            matching = tracks_by_name.get(TRACKS[instrument], [])
            if len(matching) != 1:
                continue
            track = matching[0]
            if music_start_tick is not None:
                text_ticks={}
                text_counts={}
                absolute=0
                for message in track:
                    absolute+=message.time
                    if message.type=="text":
                        text_ticks[message.text]=absolute
                        text_counts[message.text]=(
                            text_counts.get(message.text,0)+1
                        )
                idle_tick=authored_tick(
                    tempo,duration,offset_beats,midi.ticks_per_beat
                )
                for text,expected_tick in (
                    ("[play]",music_start_tick),
                    ("[idle]",idle_tick),
                ):
                    if (
                        text_counts.get(text)!=1
                        or text_ticks.get(text)!=expected_tick
                    ):
                        errors.append(
                            f"{TRACKS[instrument]}: marker {text} must occur "
                            f"exactly once at tick {expected_tick}"
                        )
                if instrument=="drums":
                    for index in range(len(DIFFICULTIES)):
                        text=f"[mix {index} drums0]"
                        if (
                            text_counts.get(text)!=1
                            or text_ticks.get(text)!=0
                        ):
                            errors.append(
                                f"{TRACKS[instrument]}: marker {text} must "
                                "occur exactly once at tick 0"
                            )
            for difficulty,base in DIFFICULTIES.items():
                actual = sum(
                    message.type=="note_on"
                    and bool(message.velocity)
                    and base<=message.note<=base+4
                    for message in track
                )
                expected = sum(
                    len(note.lanes)
                    for note in difficulties[difficulty]
                )
                if actual != expected:
                    errors.append(
                        f"{TRACKS[instrument]}/{difficulty}: MIDI has "
                        f"{actual} gems, expected {expected}"
                    )
                if music_start_tick is None:
                    continue
                offset_tick = music_start_tick - round(
                    tempo.beat_at(0)*midi.ticks_per_beat
                )
                expected_intervals = sorted(
                    (
                        authored_tick(
                            tempo,note.start,offset_beats,
                            midi.ticks_per_beat,
                        ),
                        max(
                            authored_tick(
                                tempo,note.start,offset_beats,
                                midi.ticks_per_beat,
                            )
                            + 1,
                            authored_tick(
                                tempo,note.end,offset_beats,
                                midi.ticks_per_beat,
                            ),
                        ),
                        base+lane,
                    )
                    for note in difficulties[difficulty]
                    for lane in note.lanes
                )
                active = {}
                actual_intervals = []
                absolute = 0
                for message in track:
                    absolute += message.time
                    if not hasattr(message,"note"):
                        continue
                    if (
                        message.type=="note_on"
                        and message.velocity
                        and base<=message.note<=base+4
                    ):
                        active[message.note]=absolute
                    elif (
                        message.type=="note_off"
                        or (
                            message.type=="note_on"
                            and not message.velocity
                        )
                    ) and message.note in active:
                        actual_intervals.append(
                            (
                                active.pop(message.note),
                                absolute,
                                message.note,
                            )
                        )
                actual_intervals.sort()
                if actual_intervals != expected_intervals:
                    mismatch = next(
                        (
                            (expected_item,actual_item)
                            for expected_item,actual_item
                            in zip(expected_intervals,actual_intervals)
                            if expected_item!=actual_item
                        ),
                        (
                            expected_intervals[len(actual_intervals)]
                            if len(expected_intervals)>len(actual_intervals)
                            else None,
                            actual_intervals[len(expected_intervals)]
                            if len(actual_intervals)>len(expected_intervals)
                            else None,
                        ),
                    )
                    errors.append(
                        f"{TRACKS[instrument]}/{difficulty}: MIDI timing "
                        f"differs from authored chart; expected {mismatch[0]}, "
                        f"found {mismatch[1]}"
                    )
    warnings.append("Automatic transcription needs listening review. Structural validation does not certify musical accuracy or Xbox compatibility.")
    return {"passed":not errors,"errors":errors,"warnings":warnings,"counts":counts}

def report_text(report):
    return "STRUCTURAL VALIDATION: "+("PASS" if report["passed"] else "FAIL")+"\n\n"+"\n".join("ERROR: "+x for x in report["errors"])+"\n"+"\n".join("REVIEW: "+x for x in report["warnings"])+"\n\n"+"\n".join(f"{i}: {counts}" for i,counts in report["counts"].items())+"\n"
