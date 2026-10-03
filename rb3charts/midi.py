from pathlib import Path
import math
import mido
from .model import DIFFICULTIES, TRACKS

PPQ = 960

def authored_tick(tempo, seconds, offset_beats, ppq=PPQ):
    return max(
        0,round((tempo.beat_at(seconds)+offset_beats)*ppq)
    )

def write_midi(path, tempo, instruments, duration, phrases):
    # Two full measures before the first mapped beat. Negative pickup beats
    # receive extra lead, so every audible attack remains after 2.5 seconds.
    first_interval = tempo.beats[1]-tempo.beats[0]
    offset_beats = max(tempo.numerator*2, math.ceil((2.5+tempo.beats[0])/first_interval))
    lead = offset_beats*first_interval-tempo.beats[0]
    end_beat = tempo.beat_at(duration+2.0)+offset_beats
    tick = lambda seconds: authored_tick(tempo,seconds,offset_beats)
    midi = mido.MidiFile(type=1, ticks_per_beat=PPQ)
    def track(name, events):
        tr = mido.MidiTrack([mido.MetaMessage("track_name",name=name,time=0)])
        last = 0
        for t, order, event in sorted(events, key=lambda e:(e[0],e[1])):
            tr.append(event.copy(time=t-last))
            last = t
        tr.append(mido.MetaMessage("end_of_track",time=max(0,round(end_beat*PPQ)-last)))
        midi.tracks.append(tr)
    def gem(events, a, b, pitch):
        events.extend([(a,2,mido.Message("note_on",note=pitch,velocity=96)),(max(a+1,b),1,mido.Message("note_off",note=pitch,velocity=0))])
    temp = [(0,0,mido.MetaMessage("time_signature",numerator=tempo.numerator,denominator=4)),(0,0,mido.MetaMessage("set_tempo",tempo=round(first_interval*1_000_000)))]
    for i in range(len(tempo.beats)-1):
        us = round((tempo.beats[i+1]-tempo.beats[i])*1_000_000)
        if not 1 <= us <= 0xffffff:
            raise ValueError("Beat interval is outside MIDI tempo limits")
        temp.append((round((i+offset_beats)*PPQ),0,mido.MetaMessage("set_tempo",tempo=us)))
    track("TEMPO",temp)
    events = [(tick(0),0,mido.MetaMessage("text",text="[music_start]")),(tick(duration),0,mido.MetaMessage("text",text="[music_end]")),(round(end_beat*PPQ),0,mido.MetaMessage("text",text="[end]"))]
    # Honest neutral sections; do not pretend automatic verse labels are known.
    for i,b in enumerate(range(0,max(1,int(tempo.beat_at(duration))),tempo.numerator*8)):
        events.append((round((b+offset_beats)*PPQ),0,mido.MetaMessage("text",text=f"[section section_{i+1}]")))
    track("EVENTS",events)
    beats = []
    for b in range(math.floor(end_beat)):
        gem(beats,b*PPQ,b*PPQ+120,12 if (b-offset_beats)%tempo.numerator == 0 else 13)
    track("BEAT",beats)
    for instrument, charts in instruments.items():
        events = []
        for difficulty, notes in charts.items():
            base = DIFFICULTIES[difficulty]
            for n in notes:
                for lane in n.lanes:
                    gem(events,tick(n.start),tick(n.end),base+lane)
            if instrument == "drums":
                events.append((0,0,mido.MetaMessage("text",text=f"[mix {list(DIFFICULTIES).index(difficulty)} drums0]")))
        for a,b in phrases.get(instrument,[]):
            gem(events,tick(a),tick(b),116)
        if instrument == "drums":
            # One-measure activation fills at phrase opportunities, leaving a
            # terminal cymbal and never overlapping an overdrive phrase.
            for a,b in phrases.get(instrument,[]):
                end_beat_fill=math.ceil(tempo.beat_at(b)/tempo.numerator)*tempo.numerator+tempo.numerator
                nominal_end=tempo.seconds_at(end_beat_fill)
                fill_start=tempo.seconds_at(end_beat_fill-tempo.numerator)
                common=set(n.start for n in charts["expert"])
                for notes in charts.values():common.intersection_update(n.start for n in notes)
                nearby=[t for t in common if abs(t-nominal_end)<0.000001]
                if not nearby:continue
                # Magma requires the activation gem and fill end to coincide.
                # Match the actual attack rather than shifting it onto the grid.
                fill_end=min(nearby,key=lambda t:abs(t-nominal_end))
                if fill_end < duration-2 and fill_end>fill_start:
                    for pitch in range(120,125):
                        gem(events,tick(fill_start),tick(fill_end),pitch)
            # Tom markers from ADTOF classes or supplied GM transcription.
            for n in charts["expert"]:
                for lane in n.toms:
                    gem(events,tick(n.start),tick(n.end),108+lane)
        events += [(tick(0),0,mido.MetaMessage("text",text="[play]")),(tick(duration),0,mido.MetaMessage("text",text="[idle]"))]
        track(TRACKS[instrument],events)
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    midi.save(path)
    return lead

def note_times(path):
    """Independent MIDI-time readback used to measure serialization drift."""
    time=0.0
    result=[]
    for message in mido.MidiFile(path):
        time += message.time
        if message.type == "note_on" and message.velocity:
            result.append((time,message.note))
    return result
