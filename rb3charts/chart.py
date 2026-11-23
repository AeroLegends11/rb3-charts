import dataclasses
import json
from collections import Counter
from pathlib import Path
import numpy as np
from .model import Note

ELECTRONIC_SUSTAIN_SECONDS = 0.09
BAND_SUSTAIN_SECONDS = 0.22

def _reference_style(key):
    path=Path(__file__).resolve().parents[1]/"reference"/"style-profile.json"
    if not path.is_file():return {}
    try:return json.loads(path.read_text(encoding="utf-8")).get("styles",{}).get(key,{})
    except (OSError,ValueError):return {}

def map_pitched(events, instrument, style="range", sustain_threshold=None):
    if not events:
        return []
    # A stable song-wide pitch dictionary makes recurring riffs deterministic.
    pitches = sorted(set(n["pitch"] for n in events))
    if style=="electronic_lead":
        weights={p:sum(n["confidence"]*max(0.06,min(0.5,n["end"]-n["start"])) for n in events if n["pitch"]==p) for p in pitches}
        total=sum(weights.values())
        running=0.0
        center=pitches[-1]
        for pitch in pitches:
            running+=weights[pitch]
            if running>=total/2:
                center=pitch
                break
        lower=[p for p in pitches if p<center]
        higher=[p for p in pitches if p>center]
        mapping={center:2}
        mapping.update({p:(0 if len(lower)==1 else round(i/max(1,len(lower)-1))) for i,p in enumerate(lower)})
        mapping.update({p:(3 if len(higher)==1 else 3+round(i/max(1,len(higher)-1))) for i,p in enumerate(higher)})
    else:
        if len(pitches)<=5:
            mapping={p:round(i*4/max(1,len(pitches)-1)) for i,p in enumerate(pitches)}
        else:
            # Map the registers that carry most of the performance across the
            # fretboard. A min/max scale lets one isolated solo or octave error
            # compress an entire lower riff into a single lane.
            mass={p:sum(n["confidence"]*max(0.06,min(0.5,n["end"]-n["start"])) for n in events if n["pitch"]==p) for p in pitches}
            cumulative=np.cumsum([mass[p] for p in pitches])
            boundaries=[]
            for fraction in (.2,.4,.6,.8):
                boundaries.append(pitches[int(np.searchsorted(cumulative,cumulative[-1]*fraction,side="left"))])
            mapping={p:int(np.searchsorted(boundaries,p,side="left")) for p in pitches}
    sustain_threshold=sustain_threshold if sustain_threshold is not None else (ELECTRONIC_SUSTAIN_SECONDS if style=="electronic_lead" else BAND_SUSTAIN_SECONDS)
    groups = []
    for event in events:
        if groups and event["start"]-groups[-1][0]["start"] <= 0.030:
            groups[-1].append(event)
        else:
            groups.append([event])
    notes = []
    for group in groups:
        if instrument == "bass":
            # Harmonic overtones should not become bass chords.
            strongest = max(e["confidence"] for e in group)
            group = [min((e for e in group if e["confidence"] >= strongest*0.75), key=lambda e:e["pitch"])]
        else:
            group = sorted(group, key=lambda e:e["confidence"], reverse=True)[:3]
        lanes = tuple(sorted(set(mapping[e["pitch"]] for e in group)))
        start = min(e["start"] for e in group)
        end = min(e["end"] for e in group)
        # Preserve brief audible electronic decays. Only normalize events too
        # short to draw or play as a meaningful sustain.
        if end-start < sustain_threshold:
            end = start+0.065
        notes.append(Note(start, max(start+0.04,end), lanes, float(np.mean([e["confidence"] for e in group])), "+".join(e["source_id"] for e in group), min(e["pitch"] for e in group)))
    notes=clean(notes,instrument)
    # Cleanup can shorten a real sustain to make room for the following attack.
    # If that leaves only a tiny tail, render it as a tap rather than cluttering
    # the highway with a non-musical partial sustain.
    for index,note in enumerate(notes):
        if note.end-note.start<sustain_threshold:
            next_start=notes[index+1].start if index+1<len(notes) else float("inf")
            note.end=min(note.start+0.065,next_start-0.015)
    return notes

def clean(notes, instrument):
    result = []
    for n in sorted(notes, key=lambda n:n.start):
        if not n.lanes:
            continue
        n = dataclasses.replace(n)
        if result and n.start-result[-1].start < 0.025:
            old = result[-1]
            old.lanes = tuple(sorted(set(old.lanes+n.lanes)))
            if instrument == "drums":
                hands = [x for x in old.lanes if x != 0][:2]
                old.lanes = tuple(([0] if 0 in old.lanes else [])+hands)
            else:
                old.lanes = old.lanes[:3]
            old.toms = tuple(sorted(set(old.toms+n.toms)))
            old.source_id += "+"+n.source_id
            continue
        if result:
            result[-1].end = min(result[-1].end, n.start-0.015)
        result.append(n)
    return result

def _motif_groups(notes, tempo, instrument, window=4):
    """Find repeated short figures by musical shape and quantized beat spacing."""
    ordered=sorted(notes,key=lambda note:note.start)
    candidates={}
    for i in range(len(ordered)-window+1):
        phrase=ordered[i:i+window]
        beat_gaps=[tempo.beat_at(b.start)-tempo.beat_at(a.start) for a,b in zip(phrase,phrase[1:])]
        if any(gap<=0 or gap>2.0 for gap in beat_gaps):continue
        rhythm=tuple(round(gap*4)/4 for gap in beat_gaps)
        if instrument=="drums":
            shape=tuple(tuple(note.lanes) for note in phrase)
        else:
            intervals=tuple(int(np.clip(round(b.pitch-a.pitch),-12,12)) for a,b in zip(phrase,phrase[1:]))
            shape=(intervals,tuple(len(note.lanes) for note in phrase))
        candidates.setdefault((shape,rhythm),[]).append((i,phrase))
    groups=[]
    for (signature,rhythm),occurrences in candidates.items():
        distinct=[]
        for index,phrase in occurrences:
            if not distinct or index-distinct[-1][0]>=window:
                distinct.append((index,phrase))
        if len(distinct)>=2:groups.append((signature,rhythm,distinct))
    return ordered,sorted(groups,key=lambda row:(-len(row[2]),row[2][0][1][0].start))

def analyze_patterns(notes, tempo, instrument):
    """Summarize riff/groove repetition and pitched lane-contour fidelity."""
    ordered,groups=_motif_groups(notes,tempo,instrument)
    supported={note.source_id for _,_,occurrences in groups for _,phrase in occurrences for note in phrase}
    lanes={lane:0 for lane in range(5)}
    for note in ordered:
        for lane in note.lanes:lanes[lane]=lanes.get(lane,0)+1
    report={"attacks":len(ordered),"lane_counts":lanes,
            "recurring_pattern_families":len(groups),
            "recurring_pattern_occurrences":sum(len(row[2]) for row in groups),
            "pattern_supported_attacks":len(supported),"top_patterns":[]}
    for signature,rhythm,occurrences in groups[:12]:
        shape=signature if instrument=="drums" else signature[0]
        report["top_patterns"].append({"shape":shape,"beat_gaps":rhythm,"occurrences":len(occurrences),
                                       "source_times":[round(phrase[0].start,3) for _,phrase in occurrences[:8]]})
    if instrument!="drums":
        comparisons=agreements=flattened=0
        motions=[]
        for a,b in zip(ordered,ordered[1:]):
            if tempo.beat_at(b.start)-tempo.beat_at(a.start)>2 or b.pitch==a.pitch:continue
            pitch_direction=np.sign(b.pitch-a.pitch)
            lane_direction=np.sign(np.mean(b.lanes)-np.mean(a.lanes))
            comparisons+=1
            if lane_direction==0:flattened+=1
            elif lane_direction==pitch_direction:agreements+=1
        singles=[note for note in ordered if len(note.lanes)==1]
        for a,b in zip(singles,singles[1:]):
            if tempo.beat_at(b.start)-tempo.beat_at(a.start)<=2:
                motions.append(abs(b.lanes[0]-a.lanes[0]))
        report.update({"contour_comparisons":comparisons,"contour_agreements":agreements,
                       "flattened_pitch_changes":flattened,
                       "contour_agreement_ratio":round(agreements/max(1,comparisons-flattened),3),
                       "flattened_change_ratio":round(flattened/max(1,comparisons),3),
                       "lane_motion_comparisons":len(motions),
                       "adjacent_motion_rate":round(sum(value<=1 for value in motions)/max(1,len(motions)),3),
                       "wide_jump_rate":round(sum(value>=3 for value in motions)/max(1,len(motions)),3)})
    return report


def stabilize_repeated_phrases(notes,tempo,instrument):
    """Reuse one ergonomic lane shape for strongly repeated monophonic riffs."""
    if instrument!="guitar" or len(notes)<12:
        return notes,{"families":0,"occurrences":0,"notes_changed":0,"patterns":[]}
    ordered=sorted(notes,key=lambda note:note.start)
    candidates={}
    window=4
    for index in range(len(ordered)-window+1):
        phrase=ordered[index:index+window]
        gaps=tuple(tempo.beat_at(b.start)-tempo.beat_at(a.start) for a,b in zip(phrase,phrase[1:]))
        if any(gap<=0 or gap>2 for gap in gaps) or any(len(note.lanes)!=1 for note in phrase):continue
        # Absolute pitch identity prevents unrelated transposed strumming from
        # being forced onto one fret. Rhythm is checked with tolerance below,
        # so small onset jitter does not split an otherwise repeated riff.
        key=tuple(note.pitch for note in phrase)
        candidates.setdefault(key,[]).append((index,phrase,gaps))
    groups=[]
    for pitches,occurrences in candidates.items():
        distinct=[]
        for row in occurrences:
            if not distinct or row[0]-distinct[-1][0]>=window:distinct.append(row)
        if len(distinct)<3:continue
        median_gaps=np.median([row[2] for row in distinct],axis=0)
        coherent=[row for row in distinct if max(abs(np.asarray(row[2])-median_gaps))<=.35]
        if len(coherent)>=3:groups.append((pitches,tuple(round(value*4)/4 for value in median_gaps),coherent))
    groups.sort(key=lambda row:(-len(row[2]),row[2][0][1][0].start))
    result=[dataclasses.replace(note) for note in ordered]
    by_source={note.source_id:note for note in result}
    locked=set()
    report={"families":0,"occurrences":0,"notes_changed":0,"transposed_occurrences":0,
            "inferred_notes":0,"propagated_notes":0,"patterns":[]}
    templates=[]
    for pitches,rhythm,occurrences in groups:
        intervals=tuple(b-a for a,b in zip(pitches,pitches[1:]))
        available=[]
        for _,phrase,_ in occurrences:
            mapped=[by_source.get(note.source_id) for note in phrase]
            if all(mapped) and not any(id(note) in locked for note in mapped):available.append(mapped)
        if len(available)<3:continue
        phrase_pitches=[0]
        for interval in intervals:phrase_pitches.append(phrase_pitches[-1]+interval)
        levels=sorted(set(phrase_pitches))
        level_lanes={levels[0]:0}
        for lower,higher in zip(levels,levels[1:]):
            gap=higher-lower
            level_lanes[higher]=level_lanes[lower]+(1 if gap<=5 else (2 if gap<12 else 3))
        relative=[level_lanes[pitch] for pitch in phrase_pitches]
        span=max(relative)-min(relative)
        if span>4:
            relative=[round((value-min(relative))*4/span) for value in relative]
        desired=float(np.median([np.mean([note.lanes[0] for note in phrase]) for phrase in available]))
        low=-min(relative);high=4-max(relative)
        shift=int(np.clip(round(desired-np.mean(relative)),low,high))
        shape=tuple(int(value+shift) for value in relative)
        changed=0
        for phrase in available:
            for note,lane in zip(phrase,shape):
                if note.lanes!=(lane,):note.lanes=(lane,);changed+=1
                locked.add(id(note))
        report["families"]+=1
        report["occurrences"]+=len(available)
        report["notes_changed"]+=changed
        report["patterns"].append({"lanes":shape,"beat_gaps":rhythm,"occurrences":len(available),
                                   "source_times":[round(phrase[0].start,3) for phrase in available[:8]]})
        if len(set(pitches))>1:
            templates.append((pitches,shape,rhythm,[tempo.beat_at(phrase[0].start) for phrase in available]))
    # Extend a well-established riff through nearby key/root changes. Three of
    # four pitches must agree under one transposition; only a low-confidence
    # fourth pitch may be inferred. This repairs isolated octave/harmonic errors
    # without turning unrelated song sections into copies of the riff.
    projected=[]
    for index in range(len(result)-3):
        phrase=result[index:index+4]
        if any(len(note.lanes)!=1 or id(note) in locked for note in phrase):continue
        gaps=tuple(tempo.beat_at(b.start)-tempo.beat_at(a.start) for a,b in zip(phrase,phrase[1:]))
        if any(gap<=0 or gap>2 for gap in gaps):continue
        for pitches,shape,rhythm,anchors in templates:
            differences=np.asarray([note.pitch-pitch for note,pitch in zip(phrase,pitches)])
            transpose=int(round(float(np.median(differences))))
            residual=np.abs(differences-transpose)
            matches=int(sum(residual==0))
            misses=[position for position,value in enumerate(residual) if value!=0]
            rhythm_error=float(max(abs(np.asarray(gaps)-np.asarray(rhythm))))
            start_beat=tempo.beat_at(phrase[0].start)
            local_steps=[b-a for a,b in zip(anchors,anchors[1:]) if 1<=b-a<=4]
            cadence=float(np.median(local_steps)) if local_steps else None
            phase_error=(min(abs((start_beat-anchor)-round((start_beat-anchor)/cadence)*cadence)
                             for anchor in anchors if start_beat>anchor)
                         if cadence and any(start_beat>anchor for anchor in anchors) else float("inf"))
            cadence_aligned=phase_error<=.30
            near_established=any(0<start_beat-anchor<=8 for anchor in anchors) and cadence_aligned
            strong_match=matches>=3 and near_established and (2<=abs(transpose)<=10 or not misses)
            weak_continuation=matches==2 and near_established and all(phrase[position].confidence<.50 for position in misses)
            # Half a beat of tolerance can consume the first note of the next
            # riff as the missing tail of this one. Keep projection within one
            # sixteenth-note subdivision of the learned phrase rhythm.
            if not (strong_match or weak_continuation) or rhythm_error>.25:continue
            if misses and any(phrase[position].confidence>=.60 for position in misses):continue
            projected.append((-matches,rhythm_error,phrase[0].start,phrase,pitches,shape,transpose,misses))
    projections=[]
    for _,_,_,phrase,pitches,shape,transpose,misses in sorted(projected,key=lambda row:(row[0],row[2],row[1])):
        if any(id(note) in locked for note in phrase):continue
        lane_shift=int(np.sign(transpose)*max(1,round(abs(transpose)/5)))
        lane_shift=int(np.clip(lane_shift,-min(shape),4-max(shape)))
        target=tuple(lane+lane_shift for lane in shape)
        changed=0
        for position,(note,lane,pitch) in enumerate(zip(phrase,target,pitches)):
            if note.lanes!=(lane,):note.lanes=(lane,);changed+=1
            note.pitch=pitch+transpose
            locked.add(id(note))
        report["transposed_occurrences"]+=1
        report["inferred_notes"]+=len(misses)
        report["notes_changed"]+=changed
        report["patterns"].append({"lanes":target,"transposition":transpose,"inferred_notes":len(misses),
                                   "source_times":[round(phrase[0].start,3)]})
        projections.append((tempo.beat_at(phrase[0].start),{pitch+transpose:lane for pitch,lane in zip(pitches,target)}))
    # A confirmed transposition establishes the local fret vocabulary for the
    # following two measures. Otherwise the next occurrence can immediately
    # fall back to a section-wide harmonic estimate and break the riff. Fold a
    # low-confidence nearby harmonic into that vocabulary as well; this also
    # collapses an uncertain chord produced by overlapping harmonic estimates.
    for note in result:
        if id(note) in locked:continue
        note_beat=tempo.beat_at(note.start)
        nearby=[row for row in projections if -2.5<=note_beat-row[0]<=8]
        if not nearby:continue
        preceding=[row for row in nearby if row[0]<=note_beat]
        _,mapping=(max(preceding,key=lambda row:row[0]) if preceding else
                   min(nearby,key=lambda row:row[0]-note_beat))
        target_pitch=note.pitch if note.pitch in mapping else None
        if target_pitch is None and note.confidence<.50:
            nearest=min(mapping,key=lambda pitch:abs(pitch-note.pitch))
            if abs(nearest-note.pitch)<=5:target_pitch=nearest
        if target_pitch is None:continue
        target=(mapping[target_pitch],)
        if note.lanes!=target or note.pitch!=target_pitch:
            note.lanes=target
            note.pitch=target_pitch
            report["notes_changed"]+=1
            report["propagated_notes"]+=1
    return clean(result,instrument),report


def refine_band_guitar(notes,tempo):
    """Apply five-button authoring conventions to the Band guitar path."""
    working=[dataclasses.replace(note) for note in notes]
    sections={}
    section_beats=tempo.numerator*8
    for note in working:
        sections.setdefault(int(np.floor(tempo.beat_at(note.start)/section_beats)),[]).append(note)
    # Distorted-guitar transcription often alternates between a fundamental
    # and its octave harmonic. Within a section, fold only weaker octave
    # variants into the established form; comparable octave lines stay intact.
    for rows in sections.values():
        original_mass={}
        for note in rows:
            original_mass[note.pitch]=original_mass.get(note.pitch,0)+note.confidence*max(.06,min(.5,note.end-note.start))
        by_class={}
        for pitch in original_mass:by_class.setdefault(pitch%12,[]).append(pitch)
        representatives={}
        for variants in by_class.values():
            if len(variants)<2:continue
            representative=max(variants,key=lambda pitch:original_mass[pitch])
            for pitch in variants:
                if pitch!=representative and original_mass[pitch]<=original_mass[representative]*.65:
                    representatives[pitch]=representative
        # A real octave line elsewhere in the section is not a harmonic flip:
        # fold only beside the established form within a beat and a half, and only when
        # it is clearly stronger or the variant is two or more octaves away.
        beats=[(tempo.beat_at(note.start),note.pitch,note.confidence) for note in rows]
        for note in rows:
            target=representatives.get(note.pitch)
            if target is None:continue
            here=tempo.beat_at(note.start)
            if any(pitch==target and abs(beat-here)<=1.5 and (confidence>=note.confidence+.05 or abs(note.pitch-target)>=24) for beat,pitch,confidence in beats):note.pitch=target
    section_maps={}
    section_pitch_counts={section:dict(Counter(note.pitch for note in rows)) for section,rows in sections.items()}
    for section,rows in sections.items():
        mass={}
        for note in rows:
            mass[note.pitch]=mass.get(note.pitch,0)+note.confidence*max(.06,min(.5,note.end-note.start))
        pitches=sorted(mass)
        if len(pitches)<2:
            section_maps[section]={pitch:2 for pitch in pitches}
            continue
        peak=max(mass.values())
        anchors=[pitch for pitch in pitches if mass[pitch]>=peak*.12]
        if len(anchors)>5:
            anchors=sorted(sorted(anchors,key=lambda pitch:(-mass[pitch],pitch))[:5])
        mapping={}
        for pitch in pitches:
            if pitch<anchors[0]:lane=0
            elif pitch>anchors[-1]:lane=min(4,len(anchors))
            else:lane=min(range(len(anchors)),key=lambda index:abs(anchors[index]-pitch))
            mapping[pitch]=lane
        section_maps[section]=mapping
    refined=[]
    for source in sorted(working,key=lambda note:note.start):
        note=dataclasses.replace(source)
        duration=note.end-note.start
        beat=tempo.beat_at(note.start)
        grids=(round(beat*4)/4,round(beat*3)/3)
        target=min((tempo.seconds_at(grid) for grid in grids),key=lambda value:abs(value-note.start))
        # Correct small performance/model jitter while leaving deliberate loose
        # attacks and unusual tuplets at their detected times.
        if abs(target-note.start)<=0.028 and (not refined or target-refined[-1].start>=0.04):
            note.start=target
            note.end=target+duration
        section=int(np.floor(tempo.beat_at(note.start)/section_beats))
        local_lane=section_maps.get(section,{}).get(note.pitch,note.lanes[0])
        offsets=[lane-min(note.lanes) for lane in note.lanes]
        lanes=[local_lane+offset for offset in offsets]
        overflow=max(lanes)-4
        if overflow>0:lanes=[lane-overflow for lane in lanes]
        underflow=-min(lanes)
        if underflow>0:lanes=[lane+underflow for lane in lanes]
        if 0 in lanes and 4 in lanes:
            # Three-note G/O spans are illegal. Two-note G/O is legal only on
            # Expert but is intentionally rare; use the adjacent 1-4 shape.
            if len(lanes)==2:
                lanes=[0,3]
            elif 3 not in lanes:
                lanes=[3 if lane==4 else lane for lane in lanes]
            else:
                lanes=[1 if lane==0 else lane for lane in lanes]
        stable_pitch=section_pitch_counts.get(section,{}).get(note.pitch,0)>1
        if not stable_pitch and len(lanes)==1 and refined and len(refined[-1].lanes)==1:
            previous=refined[-1]
            gap=tempo.beat_at(note.start)-tempo.beat_at(previous.start)
            interval=abs(note.pitch-previous.pitch)
            max_step=1 if interval<=5 else (2 if interval<12 else 4)
            movement=lanes[0]-previous.lanes[0]
            if gap<=2 and abs(movement)>max_step:
                lanes=[int(np.clip(previous.lanes[0]+np.sign(movement)*max_step,0,4))]
        note.lanes=tuple(sorted(set(lanes)))
        refined.append(note)
    # Compare the single-note melody as a continuous phrase even when a chord
    # sits between two notes. This matches the same measurement used by the
    # reference analyzer and catches chord-to-single cleanup boundaries.
    previous_single=None
    for note in refined:
        if len(note.lanes)!=1:continue
        section=int(np.floor(tempo.beat_at(note.start)/section_beats))
        stable_pitch=section_pitch_counts.get(section,{}).get(note.pitch,0)>1
        if previous_single is not None and not stable_pitch:
            gap=tempo.beat_at(note.start)-tempo.beat_at(previous_single.start)
            interval=abs(note.pitch-previous_single.pitch)
            max_step=1 if interval<=5 else (2 if interval<12 else 4)
            movement=note.lanes[0]-previous_single.lanes[0]
            if gap<=2 and abs(movement)>max_step:
                note.lanes=(int(np.clip(previous_single.lanes[0]+np.sign(movement)*max_step,0,4)),)
        previous_single=note
    return clean(refined,"guitar")

def _reduce_fast_hard_eighths(notes,tempo):
    """At fast tempos, remove every fourth attack in long eighth-note runs."""
    if tempo.bpm<160 or len(notes)<8:return notes
    result=[]
    run=[]
    def flush():
        if len(run)>=8:
            result.extend(note for index,note in enumerate(run) if (index+1)%4!=0)
        else:
            result.extend(run)
        run.clear()
    for note in notes:
        if run:
            gap=tempo.beat_at(note.start)-tempo.beat_at(run[-1].start)
            if not 0.36<=gap<=0.64:flush()
        run.append(note)
    flush()
    return result

def conform_expert_to_reference(notes,tempo,instrument):
    """Bring ambiguous generated details inside finished-chart style bands."""
    style=_reference_style(f"{instrument}.expert")
    if not style or not notes or instrument=="drums":return notes,{"applied":False}
    result=[dataclasses.replace(note) for note in notes]
    phrase_chords_removed=0
    if instrument=="guitar":
        # A lone chord inside a busy single-note riff is usually two harmonics
        # from one attack rather than an authored chord. Preserve chords when
        # the surrounding four-beat phrase establishes a chord texture.
        beats=[tempo.beat_at(note.start) for note in result]
        for index,note in enumerate(result):
            if len(note.lanes)<2 or note.confidence>=.70:continue
            neighbors=[row for row,row_beat in zip(result,beats) if abs(row_beat-beats[index])<=4]
            local_chords=sum(len(row.lanes)>1 for row in neighbors)
            if len(neighbors)>=8 and local_chords<=max(1,int(np.ceil(len(neighbors)*.15))):
                note.lanes=(note.lanes[0],)
                phrase_chords_removed+=1
    chord_band=style.get("song_chord_rate",{})
    # Source-separation harmonics frequently look like simultaneous notes. Aim
    # generated chords at the corpus median; p90 is only an outlier boundary and
    # allowed visibly over-chorded machine transcriptions through unchanged.
    chord_limit=chord_band.get("median",chord_band.get("p90",style.get("chord_rate",1)))
    triad_limit=style.get("song_three_note_chord_rate",{}).get("p90",style.get("three_note_chord_rate",1))
    hold_limit=style.get("song_hold_rate",{}).get("p90",style.get("hold_rate",1))
    chord_target=int(np.ceil(len(result)*chord_limit))
    triad_target=int(np.ceil(len(result)*triad_limit))
    chords=[note for note in result if len(note.lanes)>1]
    triads=[note for note in chords if len(note.lanes)>2]
    # Three-note detections are most vulnerable to distorted-guitar harmonics.
    for note in sorted(triads,key=lambda row:(row.confidence,row.end-row.start))[:max(0,len(triads)-triad_target)]:
        note.lanes=(note.lanes[0],note.lanes[-1])
    chords=[note for note in result if len(note.lanes)>1]
    for note in sorted(chords,key=lambda row:(row.confidence,row.end-row.start))[:max(0,len(chords)-chord_target)]:
        note.lanes=(round(float(np.mean(note.lanes))),)
    holds=[note for note in result if note.end-note.start>=BAND_SUSTAIN_SECONDS]
    hold_target=int(np.ceil(len(result)*hold_limit))
    for note in sorted(holds,key=lambda row:(row.end-row.start,row.confidence))[:max(0,len(holds)-hold_target)]:
        note.end=note.start+0.065
    return clean(result,instrument),{"applied":True,"reference_songs":style.get("songs",0),
        "chord_rate_target":chord_limit,"three_note_chord_rate_p90":triad_limit,"hold_rate_p90":hold_limit,
        "isolated_phrase_chords_removed":phrase_chords_removed,
        "chords_before":len(chords),"chords_after":sum(len(note.lanes)>1 for note in result),
        "triads_before":len(triads),"triads_after":sum(len(note.lanes)>2 for note in result),
        "holds_before":len(holds),"holds_after":sum(note.end-note.start>=BAND_SUSTAIN_SECONDS for note in result)}

def enforce_playability(notes, instrument, tempo, profile="band", phrase_aware=False):
    """Apply controller-oriented density limits before deriving difficulties.

    Limits count attack events rather than chord gems. Electronic pitched roles
    allow two attacks per quarter note; conventional band parts allow four.
    Drums retain four attacks because simultaneous pads are already one event.
    Confidence, grid alignment and detected sustain duration choose survivors.
    """
    if not notes:
        return [],{"before":0,"after":0,"removed":0,"holds":0,"peak_four_second_nps":0.0,"minimum_gap_ms":None}
    beat_period=float(np.median(np.diff(tempo.beats)))
    pitched=instrument in ("guitar","bass","keys")
    if profile=="electronic" and pitched:
        per_beat=2
        minimum_gap=beat_period/4
    elif profile=="band" and pitched:
        per_beat=3
        minimum_gap=beat_period/6
    else:
        per_beat=4
        minimum_gap=beat_period/8
    sustain_threshold=ELECTRONIC_SUSTAIN_SECONDS if profile=="electronic" else BAND_SUSTAIN_SECONDS
    _,motif_groups=_motif_groups(notes,tempo,instrument)
    motif_bonus={}
    for _,_,occurrences in motif_groups:
        bonus=min(0.16,0.04*(len(occurrences)-1))
        for _,phrase in occurrences:
            for note in phrase:motif_bonus[note.source_id]=max(motif_bonus.get(note.source_id,0),bonus)
    def priority(note):
        beat=tempo.beat_at(note.start)
        grid=1-abs(beat*4-round(beat*4))/.5
        duration=note.end-note.start
        return note.confidence+max(0,grid)*0.10+(0.18 if duration>=sustain_threshold else 0)+(0.08 if duration>=0.35 else 0)+motif_bonus.get(note.source_id,0)
    bins={}
    for note in notes:
        bins.setdefault(int(np.floor(tempo.beat_at(note.start))),[]).append(note)
    retained=[]
    for group in bins.values():
        ranked=sorted(group,key=priority,reverse=True)
        chosen=ranked[:per_beat]
        # A clearly recurring fast riff may use the fourth sixteenth-note
        # attack. Isolated candidates do not receive that exception.
        if profile=="band" and pitched and len(ranked)>per_beat:
            patterned=next((note for note in ranked[per_beat:] if note.source_id in motif_bonus),None)
            if patterned is not None:chosen.append(patterned)
        if phrase_aware and instrument=="guitar" and profile=="electronic":
            # Expert repeated figures may contain four real attacks per beat.
            # Preserve supported members rather than replacing the whole rule
            # with an unrestricted density increase.
            chosen.extend(note for note in ranked[per_beat:]
                          if note.source_id in motif_bonus and len(chosen)<4)
        retained.extend(chosen)
    retained.sort(key=lambda note:note.start)
    spaced=[]
    for note in retained:
        recurring_pair=(spaced and note.source_id in motif_bonus and spaced[-1].source_id in motif_bonus)
        required_gap=beat_period/8 if profile=="band" and pitched and recurring_pair else minimum_gap
        if phrase_aware and instrument=="guitar" and profile=="electronic" and recurring_pair:
            beat=tempo.beat_at(note.start)
            local_period=tempo.seconds_at(beat+1)-tempo.seconds_at(beat)
            required_gap=local_period/6
        if spaced and note.start-spaced[-1].start<required_gap:
            if priority(note)>priority(spaced[-1]):spaced[-1]=note
            continue
        spaced.append(note)
    result=clean(spaced,instrument)
    starts=np.asarray([n.start for n in result])
    gaps=np.diff(starts)
    peak=0
    right=0
    for left,start in enumerate(starts):
        right=max(right,left)
        while right<len(starts) and starts[right]<start+4:right+=1
        peak=max(peak,right-left)
    report={"before":len(notes),"after":len(result),"removed":len(notes)-len(result),
            "max_attacks_per_beat":per_beat,"minimum_allowed_gap_ms":round(minimum_gap*1000,3),
            "recurring_riff_fourth_attack_allowed":bool(profile=="band" and pitched),
            "minimum_gap_ms":round(float(gaps.min()*1000),3) if len(gaps) else None,
            "peak_four_second_nps":round(peak/4,3),
            "recurring_pattern_families":len(motif_groups),
            "pattern_supported_candidates":len(motif_bonus),
            "sustain_threshold_ms":round(sustain_threshold*1000),
            "holds":sum(n.end-n.start>=sustain_threshold for n in result),
            "longest_hold_seconds":round(max((n.end-n.start for n in result),default=0),3)}
    report["phrase_aware_guitar"]=bool(phrase_aware and instrument=="guitar")
    report["supported_phrase_attacks_per_beat"]=4 if phrase_aware and instrument=="guitar" else per_beat
    return result,report

def _separate_reduced_lane(note, prev_note, prev_lane, lanes, max_lanes, beats):
    """Keep a real pitch change visible when five source lanes are squeezed into fewer.

    The straight rounding is kept unless it repeats the previous color on a
    different pitch; the note then moves one lane in the direction of the pitch change.
    """
    if note.pitch == prev_note.pitch or beats > 2 or lanes[0] != prev_lane:
        return lanes
    direction = 1 if note.pitch > prev_note.pitch else -1
    for candidate in (lanes[0] + direction, lanes[0] - direction):
        if 0 <= candidate < max_lanes:
            return (candidate,)
    return lanes


def reductions(expert, instrument, tempo):
    charts = {"expert": expert}
    parent = expert
    for difficulty, spacing, max_lanes, chord_size in [("hard",0.46,5,2), ("medium",0.95,4,2), ("easy",1.90,3,1)]:
        candidates = []
        last_beat = -100.0
        previous = None
        for n in parent:
            beat = tempo.beat_at(n.start)
            phase = abs(beat-round(beat))
            strong = phase < 0.12
            if beat-last_beat < spacing:
                continue
            if difficulty == "easy" and not strong and candidates and beat-last_beat < 1.8:
                continue
            if instrument == "drums":
                lanes = n.lanes
                if difficulty == "medium":
                    # Prefer kick/snare anchors over accompaniment.
                    anchors = tuple(x for x in lanes if x in (0,1))
                    lanes = anchors or lanes[:1]
                elif difficulty == "easy":
                    lanes = (1,) if 1 in lanes else ((0,) if 0 in lanes else lanes[:1])
            else:
                # Compress lane vocabulary consistently, retaining contour.
                source_lanes = 5 if difficulty == "hard" else (5 if difficulty == "medium" else 4)
                lanes = tuple(sorted(set(round(x*(max_lanes-1)/(source_lanes-1)) for x in n.lanes)))[:chord_size]
                if len(n.lanes) == 1 and previous is not None and len(previous[1]) == 1:
                    lanes = _separate_reduced_lane(n, previous[0], previous[1][0], lanes, max_lanes, beat - last_beat)
                previous = (n, lanes)
            candidates.append(dataclasses.replace(n, lanes=lanes, toms=tuple(t for t in n.toms if t in lanes)))
            last_beat = beat
        if difficulty=="hard" and instrument in ("guitar","bass","keys"):
            candidates=_reduce_fast_hard_eighths(candidates,tempo)
            normalized=[]
            for note in candidates:
                lanes=list(note.lanes)
                if instrument=="guitar" and len(lanes)==1 and normalized and len(normalized[-1].lanes)==1:
                    gap=tempo.beat_at(note.start)-tempo.beat_at(normalized[-1].start)
                    prior=normalized[-1].lanes[0]
                    if gap<=.65 and prior==0 and lanes[0]==4:lanes=[3]
                    elif gap<=.65 and prior==4 and lanes[0]==0:lanes=[1]
                lanes=tuple(lanes)
                if lanes==(0,4):lanes=(0,3)
                normalized.append(dataclasses.replace(note,lanes=lanes))
            if instrument=="guitar":
                # The edge shifts above can recreate a repeated color on a pitch change.
                for index in range(1,len(normalized)):
                    before,note=normalized[index-1],normalized[index]
                    if len(before.lanes)==1==len(note.lanes):
                        beats=tempo.beat_at(note.start)-tempo.beat_at(before.start)
                        lanes=_separate_reduced_lane(note,before,before.lanes[0],note.lanes,5,beats)
                        if lanes!=note.lanes:normalized[index]=dataclasses.replace(note,lanes=lanes)
            candidates=normalized
        parent = clean(candidates, instrument)
        charts[difficulty] = parent
    return charts

def overdrive(charts, tempo, duration):
    """Common phrase opportunities, only where all difficulties have notes."""
    result = []
    measure = tempo.numerator
    for beat in range(measure*8, int(tempo.beat_at(duration))-measure*3, measure*8):
        a, b = tempo.seconds_at(beat), tempo.seconds_at(beat+measure*2)
        if all(any(a <= n.start < b for n in notes) for notes in charts.values()):
            result.append((a,b))
    return result

def align_drum_activations(charts, tempo, phrases, duration):
    """Magma treats fill endpoints as beat-aligned, even with off-grid markers.

    Only align an existing shared attack when <=12ms of adjustment suffices.
    Record each change. Never create a drum hit or shift an entire phrase.
    """
    import math
    corrections=[]
    for a,b in phrases:
        end_beat=math.ceil(tempo.beat_at(b)/tempo.numerator)*tempo.numerator+tempo.numerator
        target=tempo.seconds_at(end_beat)
        if target>=duration-2:continue
        common=set(n.start for n in charts["expert"])
        for notes in charts.values():common.intersection_update(n.start for n in notes)
        nearby=[t for t in common if abs(t-target)<=0.012]
        if not nearby:continue
        original=min(nearby,key=lambda t:abs(t-target))
        # No neighboring note can become overlapped by the constrained shift.
        if any(any(n.start!=original and abs(n.start-target)<0.08 for n in notes) for notes in charts.values()):continue
        for notes in charts.values():
            for n in notes:
                if n.start==original:
                    length=n.end-n.start
                    n.start=target
                    n.end=target+length
        corrections.append({"original_seconds":original,"authored_seconds":target,"change_ms":(target-original)*1000,"reason":"Magma drum-fill activation alignment"})
    return corrections
