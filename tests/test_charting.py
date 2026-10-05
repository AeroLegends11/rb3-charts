import numpy as np
import pytest
import mido
from rb3charts.model import Note,TempoMap
from rb3charts.chart import reductions,map_pitched,clean,enforce_playability,analyze_patterns,refine_band_guitar,conform_expert_to_reference,stabilize_repeated_phrases
from rb3charts.midi import authored_tick, write_midi,note_times
from rb3charts.validate import validate
from rb3charts.transcribe import analyze_band_guitar_voice,electronic_lead_events,electronic_bass_events,_refine_pitched_articulations,_focus_band_guitar_candidates,_filter_sequential_lead_outliers,_resolve_supported_register_voice_groups,_repeated_candidate_preferences

def test_tempo_map_roundtrip_drifting_and_pickups():
    tempo=TempoMap([0.7,1.2,1.69,2.17,2.8,3.4])
    for seconds in np.linspace(0,5,501):
        assert tempo.seconds_at(tempo.beat_at(seconds))==pytest.approx(seconds,abs=1e-10)


def test_repeated_candidate_preferences_reuse_transposed_source_shape():
    groups = []
    shape = [0, 4, 2, 7]
    distractors = [[10, -1, 6, -4], [7, 2, -3, 12], [11, -2, -6, 5]]
    for occurrence, root in enumerate([48, 50, 53]):
        for index, interval in enumerate(shape):
            start = occurrence * 2 + index * .12
            groups.append([
                {
                    "start": start,
                    "pitch": root + interval,
                    "confidence": .9,
                    "source_id": f"motif-{occurrence}-{index}",
                },
                {
                    "start": start,
                    "pitch": root + distractors[occurrence][index],
                    "confidence": .95,
                    "source_id": f"distractor-{occurrence}-{index}",
                },
            ])

    preferences = _repeated_candidate_preferences(groups, .48)

    assert len(preferences) == 12
    assert set(preferences.values()) == {
        f"motif-{occurrence}-{index}"
        for occurrence in range(3)
        for index in range(4)
    }


def test_repeated_candidate_preferences_ignore_overlapping_windows():
    groups = [[{
        "start": index * .12,
        "pitch": 48 + index,
        "confidence": .9,
        "source_id": f"candidate-{index}",
    }, {
        "start": index * .12,
        "pitch": 60 + index,
        "confidence": .95,
        "source_id": f"distractor-{index}",
    }] for index in range(11)]

    assert _repeated_candidate_preferences(groups, .48) == {}


def test_midi_serialization_preserves_drifting_onsets(tmp_path):
    tempo=TempoMap([0.3,0.8,1.29,1.77,2.4,3.0,3.4,4.1,4.7,5.2])
    original=[0.0,0.19,0.8,1.23,2.39,3.44,4.95,6.8]
    notes=[Note(t,t+0.06,(i%5,),source_id=str(i)) for i,t in enumerate(original)]
    charts={"guitar":reductions(notes,"guitar",tempo)}
    path=tmp_path/"notes.mid"
    lead=write_midi(path,tempo,charts,7.0,{})
    actual=[t-lead for t,n in note_times(path) if 96<=n<=100]
    assert actual==pytest.approx(original,abs=0.001)

def test_reductions_preserve_ancestry_and_reduce_density():
    tempo=TempoMap(np.arange(0,20,0.5).tolist())
    notes=[Note(i/8,i/8+0.06,(i%5,),source_id=str(i)) for i in range(100)]
    charts=reductions(notes,"guitar",tempo)
    assert len(charts["expert"])>len(charts["hard"])>len(charts["medium"])>len(charts["easy"])
    for p,c in [("expert","hard"),("hard","medium"),("medium","easy")]:
        assert {n.start for n in charts[c]} <= {n.start for n in charts[p]}
        assert {n.source_id for n in charts[c]} <= {n.source_id for n in charts[p]}
    assert max(max(n.lanes) for n in charts["easy"])<=2

def test_hard_reduction_preserves_playable_point_four_beat_pattern():
    tempo=TempoMap([beat*.5 for beat in range(20)])
    expert=[
        Note(index*.2,index*.2+.065,(index%5,),source_id=str(index))
        for index in range(20)
    ]
    hard=reductions(expert,"guitar",tempo)["hard"]
    assert len(hard)==len(expert)
    drum_hard=reductions(expert,"drums",tempo)["hard"]
    assert len(drum_hard)<len(expert)

def test_repeated_riff_has_identical_mapping():
    events=[{"start":i*0.5,"end":i*0.5+0.3,"pitch":p,"confidence":0.9,"source_id":str(i)} for i,p in enumerate([48,52,55,50]*3)]
    notes=map_pitched(events,"guitar")
    assert [n.lanes for n in notes[:4]]==[n.lanes for n in notes[4:8]]==[n.lanes for n in notes[8:]]

def test_validator_rejects_three_handed_drums():
    tempo=TempoMap([0,0.5,1,1.5,2])
    charts={d:[Note(0.5,0.6,(0,1,2,3),source_id="a")] for d in ("expert","hard","medium","easy")}
    result=validate({"drums":charts},tempo,2)
    assert not result["passed"]
    assert any("hands" in e for e in result["errors"])

def test_invalid_beat_map_rejected():
    with pytest.raises(ValueError):TempoMap([0,0.5,0.4])

def test_cleanup_cuts_sustains_before_next_attack():
    notes=clean([Note(0,2,(0,)),Note(0.5,1,(1,))],"guitar")
    assert notes[0].end<notes[1].start

def test_validator_rejects_new_reduction_note():
    tempo=TempoMap([0,0.5,1,1.5,2])
    charts=reductions([Note(0.5,0.6,(0,),source_id="original")],"guitar",tempo)
    charts["easy"][0].source_id="invented"
    assert not validate({"guitar":charts},tempo,2)["passed"]

def test_validator_checks_track_specific_midi_pitches(tmp_path):
    tempo=TempoMap([0,.5,1,1.5,2,2.5,3])
    charts=reductions(
        [Note(.5,.6,(0,),source_id="original")],"guitar",tempo
    )
    path=tmp_path/"notes.mid"
    write_midi(path,tempo,{"guitar":charts},2,{})
    midi=mido.MidiFile(path)
    guitar=next(track for track in midi.tracks if track.name=="PART GUITAR")
    guitar.insert(1,mido.Message("note_on",note=120,velocity=96,time=0))
    guitar.insert(2,mido.Message("note_off",note=120,velocity=0,time=1))
    midi.save(path)
    result=validate({"guitar":charts},tempo,2,path)
    assert not result["passed"]
    assert any("invalid MIDI pitch 120" in error for error in result["errors"])

def test_validator_checks_serialized_gem_counts(tmp_path):
    tempo=TempoMap([0,.5,1,1.5,2,2.5,3])
    charts=reductions(
        [Note(.5,.6,(0,),source_id="original")],"guitar",tempo
    )
    path=tmp_path/"notes.mid"
    write_midi(path,tempo,{"guitar":charts},2,{})
    midi=mido.MidiFile(path)
    guitar=next(track for track in midi.tracks if track.name=="PART GUITAR")
    guitar[:]=[
        message for message in guitar
        if not (
            message.type in ("note_on","note_off")
            and message.note==96
        )
    ]
    midi.save(path)
    result=validate({"guitar":charts},tempo,2,path)
    assert not result["passed"]
    assert any(
        "PART GUITAR/expert: MIDI has 0 gems, expected 1" in error
        for error in result["errors"]
    )

def test_validator_checks_serialized_gem_timing(tmp_path):
    tempo=TempoMap([0,.5,1,1.5,2,2.5,3])
    charts=reductions(
        [Note(.5,.6,(0,),source_id="original")],"guitar",tempo
    )
    path=tmp_path/"notes.mid"
    write_midi(path,tempo,{"guitar":charts},2,{})
    midi=mido.MidiFile(path)
    guitar=next(track for track in midi.tracks if track.name=="PART GUITAR")
    note_on=next(
        message for message in guitar
        if message.type=="note_on" and message.velocity and message.note==96
    )
    note_off=next(
        message for message in guitar
        if message.type=="note_off" and message.note==96
    )
    note_on.time+=10
    note_off.time-=10
    midi.save(path)
    result=validate({"guitar":charts},tempo,2,path)
    assert not result["passed"]
    assert any(
        "PART GUITAR/expert: MIDI timing differs" in error
        for error in result["errors"]
    )


def test_authored_tick_rounds_after_lead_offset():
    class EdgeTempo:
        def beat_at(self, seconds):
            return 10.419270833333334

    tick=authored_tick(EdgeTempo(),0,8)
    assert tick==17683
    assert tick!=8*960+round(10.419270833333334*960)

def test_validator_checks_serialized_tempo_map(tmp_path):
    tempo=TempoMap([0,.5,1.01,1.53,2.06,2.6,3.15])
    charts=reductions(
        [Note(.5,.6,(0,),source_id="original")],"guitar",tempo
    )
    path=tmp_path/"notes.mid"
    write_midi(path,tempo,{"guitar":charts},2,{})
    midi=mido.MidiFile(path)
    tempo_track=next(track for track in midi.tracks if track.name=="TEMPO")
    tempo_event=next(
        message for message in tempo_track
        if message.type=="set_tempo"
    )
    tempo_event.tempo+=1000
    midi.save(path)
    result=validate({"guitar":charts},tempo,2,path)
    assert not result["passed"]
    assert any(
        "TEMPO events differ from the authored tempo map" in error
        for error in result["errors"]
    )

def test_validator_checks_serialized_beat_grid(tmp_path):
    tempo=TempoMap([0,.5,1.01,1.53,2.06,2.6,3.15])
    charts=reductions(
        [Note(.5,.6,(0,),source_id="original")],"guitar",tempo
    )
    path=tmp_path/"notes.mid"
    write_midi(path,tempo,{"guitar":charts},2,{})
    midi=mido.MidiFile(path)
    beat_track=next(track for track in midi.tracks if track.name=="BEAT")
    beat_on=next(
        message for message in beat_track
        if message.type=="note_on" and message.velocity
    )
    beat_off=next(
        message for message in beat_track
        if message.type=="note_off"
    )
    beat_on.time+=5
    beat_off.time-=5
    midi.save(path)
    result=validate({"guitar":charts},tempo,2,path)
    assert not result["passed"]
    assert any(
        "BEAT track timing differs from the authored beat grid" in error
        for error in result["errors"]
    )

def test_validator_checks_serialized_lifecycle_markers(tmp_path):
    tempo=TempoMap([0,.5,1,1.5,2,2.5,3])
    charts=reductions(
        [Note(.5,.6,(0,),source_id="original")],"guitar",tempo
    )
    path=tmp_path/"notes.mid"
    write_midi(path,tempo,{"guitar":charts},2,{})
    midi=mido.MidiFile(path)
    events=next(track for track in midi.tracks if track.name=="EVENTS")
    music_end=next(
        message for message in events
        if message.type=="text" and message.text=="[music_end]"
    )
    music_end.time+=1
    guitar=next(track for track in midi.tracks if track.name=="PART GUITAR")
    idle=next(
        message for message in guitar
        if message.type=="text" and message.text=="[idle]"
    )
    idle.text="[idle-corrupt]"
    midi.save(path)
    result=validate({"guitar":charts},tempo,2,path)
    assert not result["passed"]
    assert any(
        "EVENTS marker [music_end]" in error
        for error in result["errors"]
    )
    assert any(
        "PART GUITAR: marker [idle]" in error
        for error in result["errors"]
    )

def test_validator_rejects_duplicate_lifecycle_markers(tmp_path):
    tempo=TempoMap([0,.5,1,1.5,2,2.5,3])
    charts=reductions(
        [Note(.5,.6,(0,),source_id="original")],"guitar",tempo
    )
    path=tmp_path/"notes.mid"
    write_midi(path,tempo,{"guitar":charts},2,{})
    midi=mido.MidiFile(path)
    events=next(track for track in midi.tracks if track.name=="EVENTS")
    events.insert(
        1,mido.MetaMessage("text",text="[music_start]",time=0)
    )
    guitar=next(track for track in midi.tracks if track.name=="PART GUITAR")
    guitar.insert(1,mido.MetaMessage("text",text="[play]",time=0))
    midi.save(path)
    result=validate({"guitar":charts},tempo,2,path)
    assert not result["passed"]
    assert "Duplicate EVENTS marker: [music_start]" in result["errors"]
    assert any(
        "marker [play] must occur exactly once" in error
        for error in result["errors"]
    )

def test_electronic_lead_prefers_high_hook_and_deduplicates_stems():
    guitar=[{"start":1.0,"end":1.2,"pitch":60,"confidence":.9,"source_id":"guitar-0","source":"guitar"}]
    synth=[{"start":1.01,"end":1.3,"pitch":72,"confidence":.84,"source_id":"other-0","source":"other"}]
    result=electronic_lead_events([guitar,synth])
    assert len(result)==1
    assert result[0]["pitch"]==72

def test_electronic_bass_retriggers_sustained_tone_on_kicks():
    bass=[[{"start":0.0,"end":1.2,"pitch":40,"confidence":.9,"source_id":"bass-0","source":"bass"}]]
    drums=[Note(.5,.56,(0,),source_id="kick"),Note(1.5,1.56,(0,),source_id="late")]
    result=electronic_bass_events(bass,drums)
    assert [e["start"] for e in result]==[0.0,.5]
    assert result[1]["pitch"]==40
    assert result[1]["source"]=="kick"

def test_electronic_lead_consolidates_recurring_octave_harmonics():
    events=[]
    for i,pitch in enumerate([68,80,92]*4):
        events.append({"start":i*.12,"end":i*.12+.2,"pitch":pitch,"confidence":.7 if pitch==80 else .5,"source_id":f"other-{i}","source":"other"})
    result=electronic_lead_events([events],beat_period=.47)
    assert {event["pitch"] for event in result}=={80}

def test_electronic_lead_removes_weak_short_texture_events():
    events=[{"start":i*.15,"end":i*.15+(.05 if i%2 else .2),"pitch":60+i%3,"confidence":.41 if i%2 else .8,"source_id":f"other-{i}","source":"other"} for i in range(24)]
    result=electronic_lead_events([events],beat_period=.5)
    assert len(result)==12
    assert all(event["confidence"]==.8 for event in result)

def test_electronic_mapping_anchors_dominant_pitch_in_center_lane():
    events=[]
    for i,pitch in enumerate([68,75,80,80,80,80,84,92]):
        events.append({"start":i*.2,"end":i*.2+.15,"pitch":pitch,"confidence":.8,"source_id":str(i)})
    notes=map_pitched(events,"guitar","electronic_lead")
    by_pitch={note.pitch:note.lanes[0] for note in notes}
    assert by_pitch[80]==2
    assert by_pitch[75]==1 and by_pitch[84]==3
    assert by_pitch[68]==0 and by_pitch[92]==4

def test_electronic_lead_caps_texture_bursts_near_sixteenth_notes():
    events=[{"start":i*.07,"end":i*.07+.065,"pitch":80,"confidence":.7,"source_id":f"other-{i}","source":"other"} for i in range(12)]
    result=electronic_lead_events([events],beat_period=60/130)
    assert all(b["start"]-a["start"]>=.1 for a,b in zip(result,result[1:]))

def test_playability_gate_caps_electronic_roles_at_two_attacks_per_beat():
    tempo=TempoMap(np.arange(0,8,.5).tolist())
    notes=[Note(1+i*.07,1+i*.07+.065,(i%5,),confidence=.6+i*.01,source_id=str(i)) for i in range(12)]
    result,report=enforce_playability(notes,"guitar",tempo,"electronic")
    assert report["removed"]>0
    assert report["max_attacks_per_beat"]==2
    assert all(b.start-a.start>=.125-1e-9 for a,b in zip(result,result[1:]))

def test_playability_gate_prefers_legitimate_holds():
    tempo=TempoMap(np.arange(0,8,.5).tolist())
    notes=[Note(1+i*.04,1+i*.04+.065,(i,),confidence=.8,source_id=str(i)) for i in range(3)]
    notes.append(Note(1.08,1.55,(2,),confidence=.7,source_id="hold"))
    result,report=enforce_playability(notes,"guitar",tempo,"electronic")
    assert any(n.source_id=="hold" and n.end-n.start>=.15 for n in result)
    assert report["holds"]==1

def test_band_profile_allows_clear_sixteenth_note_part():
    tempo=TempoMap(np.arange(0,8,.5).tolist())
    notes=[
        Note(1+i*.125,1+i*.125+.065,(i,),source_id=str(i),pitch=pitch)
        for i,pitch in enumerate((48,55,61,72))
    ]
    result,report=enforce_playability(notes,"guitar",tempo,"band")
    assert len(result)==len(notes)
    assert report["removed"]==0

def test_band_profile_reduces_unpatterned_crowding():
    tempo=TempoMap(np.arange(0,8,.5).tolist())
    notes=[Note(1+i*.045,1+i*.045+.065,(i%5,),confidence=.8-i*.01,source_id=str(i),pitch=48+i) for i in range(4)]
    result,report=enforce_playability(notes,"guitar",tempo,"band")
    assert len(result)<len(notes)
    assert report["max_attacks_per_beat"]==4
    reasons=report["removal_attribution"]
    assert reasons["spacing_rejected_later"]["count"] > 0
    assert sum(row["count"] for row in reasons.values()) == report["removed"]

def test_playability_report_distinguishes_cap_and_spacing_replacement():
    tempo=TempoMap(np.arange(0,8,.5).tolist())
    notes=[
        Note(1.01,1.075,(0,),confidence=.95,source_id="a",pitch=48),
        Note(1.12,1.185,(1,),confidence=.90,source_id="b",pitch=50),
        Note(1.23,1.295,(2,),confidence=.85,source_id="c",pitch=52),
        Note(1.45,1.515,(4,),confidence=.40,source_id="cap",pitch=56),
        Note(1.49,1.555,(0,),confidence=.50,source_id="earlier",pitch=58),
        Note(1.51,1.575,(1,),confidence=.99,source_id="replacement",pitch=60),
    ]
    _,report=enforce_playability(notes,"guitar",tempo,"band")
    reasons=report["removal_attribution"]
    assert reasons["per_beat_cap"]["count"] == 1
    assert reasons["spacing_replaced_earlier"]["count"] == 1
    assert reasons["spacing_replaced_earlier"]["samples"][0]["source_id"]=="earlier"
    assert reasons["spacing_replaced_earlier"]["samples"][0]["replaced_by_source_id"]=="replacement"
    assert sum(row["count"] for row in reasons.values()) == report["removed"]

def test_pitch_specific_reattack_splits_hold_and_preserves_releases():
    from basic_pitch.note_creation import model_frames_to_time,MIDI_OFFSET
    sr=22050
    y=np.zeros(sr*2,dtype=np.float32)
    for start,length in ((.50,.28),(.90,.38)):
        a=int(start*sr); n=int(length*sr); t=np.arange(n)/sr
        y[a:a+n]+=np.sin(2*np.pi*261.63*t)*np.exp(-t*8)
    times=model_frames_to_time(180)
    output={"onset":np.zeros((180,88),dtype=np.float32)}
    frame=int(np.argmin(abs(times-.90)))
    output["onset"][frame,60-MIDI_OFFSET]=.8
    events=[{"start":.50,"end":1.35,"pitch":60,"confidence":.8,"source_id":"other-0","source":"other"}]
    result=_refine_pitched_articulations(events,output,y,sr,40,100,"other")
    assert len(result)==2
    assert result[0]["end"]<result[1]["start"]
    assert result[1]["source_id"].startswith("other-reattack-")
    assert all(event["end"]-event["start"]>=.065 for event in result)

def test_map_pitched_preserves_brief_electronic_decay_as_sustain():
    events=[{"start":1.0,"end":1.105,"pitch":68,"confidence":.8,"source_id":"piano-1"}]
    note=map_pitched(events,"guitar","electronic_lead")[0]
    assert note.end-note.start==pytest.approx(.105)

def test_band_mapping_turns_brief_articulation_into_tap():
    events=[{"start":1.0,"end":1.105,"pitch":48,"confidence":.8,"source_id":"guitar-1"}]
    note=map_pitched(events,"guitar","range")[0]
    assert note.end-note.start==pytest.approx(.065)


def test_band_guitar_mapping_turns_short_sustain_into_tap():
    events=[{"start":1.0,"end":1.32,"pitch":55,"confidence":.8,"source_id":"guitar-1"}]
    note=map_pitched(events,"guitar","range")[0]
    assert note.end-note.start==pytest.approx(.065)


def test_band_guitar_mapping_drops_sustain_from_lead_corrected_pitch():
    events=[{"start":1.0,"end":1.8,"pitch":55,"confidence":.8,
             "source_id":"guitar-1","lead_focus_corrected":True}]
    note=map_pitched(events,"guitar","range")[0]
    assert note.end-note.start==pytest.approx(.065)

def test_band_mapping_uses_common_register_despite_high_outlier():
    events=[]
    for index,pitch in enumerate([48,48,48,50,51,53,55,58,60,87]):
        events.append({"start":index*.2,"end":index*.2+.1,"pitch":pitch,"confidence":.8,"source_id":str(index)})
    notes=map_pitched(events,"guitar","range")
    riff_lanes={note.lanes[0] for note in notes[:-1]}
    assert len(riff_lanes)>=4
    assert notes[0].lanes!=notes[4].lanes

def test_pattern_analysis_finds_repeated_riff_and_checks_contour():
    tempo=TempoMap(np.arange(0,8,.5).tolist())
    notes=[]
    for offset in (0.0,2.0):
        for step,pitch in enumerate((48,50,53,51)):
            lane=(0,1,3,2)[step]
            notes.append(Note(offset+step*.25,offset+step*.25+.065,(lane,),source_id=f"{offset}-{step}",pitch=pitch))
    report=analyze_patterns(notes,tempo,"guitar")
    assert report["recurring_pattern_families"]>=1
    assert report["pattern_supported_attacks"]==8
    assert report["contour_agreement_ratio"]==1
    assert report["adjacent_motion_rate"]==pytest.approx(2/3,abs=.001)
    assert report["wide_jump_rate"]==0
    assert report["top_patterns"][0]["lane_variants"]==1
    assert report["top_patterns"][0]["exact_pitch_repetitions"]==2
    assert report["top_patterns"][0]["lane_mismatch_times"]==[]
    assert report["repeatable_fingering_occurrences"]==2
    assert report["repeatable_fingering_ratio"]==1

def test_pattern_analysis_reports_repeated_source_shape_lane_mismatches():
    tempo=TempoMap(np.arange(0,8,.5).tolist())
    notes=[]
    for offset,lanes in ((0.0,(0,1,3,2)),(2.0,(0,1,2,3))):
        for step,pitch in enumerate((48,50,53,51)):
            notes.append(Note(
                offset+step*.25,
                offset+step*.25+.065,
                (lanes[step],),
                source_id=f"{offset}-{step}",
                pitch=pitch,
            ))
    report=analyze_patterns(notes,tempo,"guitar")
    pattern=report["top_patterns"][0]
    assert pattern["exact_pitch_repetitions"]==2
    assert pattern["lane_variants"]==2
    assert pattern["lane_mismatch_times"]==[2.0]
    assert report["repeatable_fingering_occurrences"]==2
    assert report["repeatable_fingering_ratio"]==.5

def test_pattern_analysis_does_not_flag_transposed_lane_variants():
    tempo=TempoMap(np.arange(0,8,.5).tolist())
    notes=[]
    for offset,pitches,lanes in (
        (0.0,(48,50,53,51),(0,1,3,2)),
        (2.0,(55,57,60,58),(1,2,4,3)),
    ):
        for step,pitch in enumerate(pitches):
            notes.append(Note(
                offset+step*.25,
                offset+step*.25+.065,
                (lanes[step],),
                source_id=f"{offset}-{step}",
                pitch=pitch,
            ))
    pattern=analyze_patterns(notes,tempo,"guitar")["top_patterns"][0]
    assert pattern["exact_pitch_repetitions"]==0
    assert pattern["lane_variants"]==1
    assert pattern["lane_mismatch_times"]==[]

def test_band_guitar_refinement_snaps_jitter_and_removes_wide_shapes():
    tempo=TempoMap(np.arange(0,4,.5).tolist())
    notes=[Note(.49,.79,(0,4),source_id="wide",pitch=48),
           Note(1.01,1.31,(0,2,4),source_id="illegal",pitch=55)]
    result=refine_band_guitar(notes,tempo)
    assert result[0].start==pytest.approx(.5)
    assert result[0].lanes==(0,3)
    assert all(not (len(note.lanes)==3 and 0 in note.lanes and 4 in note.lanes) for note in result)

def test_band_guitar_uses_local_register_without_octave_compression():
    tempo=TempoMap(np.arange(0,24,.5).tolist())
    pitches=[48,48,48,50,51,53,55,48,50,51,87]
    notes=[Note(1+i*.3,1+i*.3+.065,(1,),confidence=.8,source_id=str(i),pitch=pitch) for i,pitch in enumerate(pitches)]
    result=refine_band_guitar(notes,tempo)
    common={}
    for note in result:
        if note.pitch<60:common.setdefault(note.pitch,note.lanes[0])
    assert len({note.lanes[0] for note in result if note.pitch<60})>=4
    assert common[48]<common[50]<=common[51]<common[53]<=common[55]
    assert result[-1].pitch==51
    assert result[-1].lanes==result[4].lanes

def test_band_guitar_wraps_small_intervals_to_adjacent_frets():
    tempo=TempoMap(np.arange(0,12,.5).tolist())
    notes=[Note(1,1.065,(0,),source_id="a",pitch=48),Note(1.25,1.315,(4,),source_id="b",pitch=51)]
    result=refine_band_guitar(notes,tempo)
    assert abs(result[1].lanes[0]-result[0].lanes[0])<=1


def test_band_guitar_keeps_repeated_riff_pitch_on_same_fret():
    tempo=TempoMap(np.arange(0,12,.5).tolist())
    pitches=(48,48,50,51,48,48,50,51)
    notes=[Note(1+i*.2,1+i*.2+.065,(0,),source_id=str(i),pitch=pitch) for i,pitch in enumerate(pitches)]
    result=refine_band_guitar(notes,tempo)
    lanes_by_pitch={pitch:{note.lanes[0] for note in result if note.pitch==pitch} for pitch in set(pitches)}
    assert all(len(lanes)==1 for lanes in lanes_by_pitch.values())
    assert [result[i].lanes[0] for i in range(4)]==[0,0,1,2]


def test_repeated_phrase_stabilizer_reuses_one_playable_shape():
    tempo=TempoMap(np.arange(0,16,.5).tolist())
    notes=[]
    lane_variants=((0,0,1,2),(1,1,3,4),(0,0,2,3))
    for occurrence,(offset,lanes) in enumerate(zip((0.,2.,4.),lane_variants)):
        for step,(pitch,lane) in enumerate(zip((48,48,50,51),lanes)):
            notes.append(Note(offset+step*.25,offset+step*.25+.065,(lane,),confidence=.8,
                              source_id=f"{occurrence}-{step}",pitch=pitch))
    result,report=stabilize_repeated_phrases(notes,tempo,"guitar")
    shapes=[tuple(note.lanes[0] for note in result[index:index+4]) for index in (0,4,8)]
    assert shapes[0]==shapes[1]==shapes[2]
    assert tuple(b-a for a,b in zip(shapes[0],shapes[0][1:]))==(0,1,1)
    assert report["families"]>=1 and report["occurrences"]>=3


def test_repeated_phrase_stabilizer_returns_to_same_root_fret():
    tempo=TempoMap(np.arange(0,16,.5).tolist())
    notes=[]
    for occurrence,offset in enumerate((0.,2.,4.)):
        for step,(pitch,lane) in enumerate(zip((48,50,51,48),(0,2,4,1))):
            notes.append(Note(offset+step*.25,offset+step*.25+.065,(lane,),confidence=.8,
                              source_id=f"return-{occurrence}-{step}",pitch=pitch))
    result,report=stabilize_repeated_phrases(notes,tempo,"guitar")
    for index in (0,4,8):assert result[index].lanes==result[index+3].lanes
    assert report["families"]>=1


def test_repeated_phrase_stabilizer_projects_transposed_partial_match():
    tempo=TempoMap(np.arange(0,20,.5).tolist())
    notes=[]
    for occurrence,offset in enumerate((0.,2.,4.)):
        for step,pitch in enumerate((48,48,50,51)):
            notes.append(Note(offset+step*.25,offset+step*.25+.065,(step%3,),confidence=.8,
                              source_id=f"base-{occurrence}-{step}",pitch=pitch))
    for step,pitch in enumerate((53,53,55,48)):
        notes.append(Note(6+step*.25,6+step*.25+.065,(3,),confidence=.4 if step==3 else .8,
                          source_id=f"shift-{step}",pitch=pitch))
    notes.append(Note(7.25,7.315,(4,),confidence=.8,source_id="shift-follow",pitch=53))
    result,report=stabilize_repeated_phrases(notes,tempo,"guitar")
    shifted=result[-5:-1]
    assert [note.pitch for note in shifted]==[53,53,55,56]
    assert [note.lanes[0] for note in shifted]==[1,1,2,3]
    assert result[-1].lanes==(1,)
    assert report["transposed_occurrences"]==1 and report["inferred_notes"]==1
    assert report["propagated_notes"]==1


def test_repeated_phrase_stabilizer_repairs_adjacent_weak_continuation():
    tempo=TempoMap(np.arange(0,20,.5).tolist())
    notes=[]
    for occurrence,offset in enumerate((0.,2.,4.)):
        for step,pitch in enumerate((48,48,50,51)):
            notes.append(Note(offset+step*.25,offset+step*.25+.065,(step%3,),confidence=.8,
                              source_id=f"clean-{occurrence}-{step}",pitch=pitch))
    for step,(pitch,confidence) in enumerate(zip((48,51,44,51),(.8,.4,.4,.8))):
        notes.append(Note(6+step*.25,6+step*.25+.065,(4,),confidence=confidence,
                          source_id=f"noisy-{step}",pitch=pitch))
    result,report=stabilize_repeated_phrases(notes,tempo,"guitar")
    repaired=result[-4:]
    assert [note.pitch for note in repaired]==[48,48,50,51]
    assert [note.lanes[0] for note in repaired]==[0,0,1,2]
    assert report["inferred_notes"]==2


def test_repeated_phrase_stabilizer_does_not_reach_across_rest():
    tempo=TempoMap(np.arange(0,20,.5).tolist())
    notes=[]
    for occurrence,offset in enumerate((0.,2.,4.)):
        for step,pitch in enumerate((48,48,50,51)):
            notes.append(Note(offset+step*.25,offset+step*.25+.065,(step%3,),confidence=.8,
                              source_id=f"clean-{occurrence}-{step}",pitch=pitch))
    for step,(time,pitch) in enumerate(zip((6.,6.25,6.5,7.),(53,53,55,48))):
        notes.append(Note(time,time+.065,(3,),confidence=.4 if step==3 else .8,
                          source_id=f"partial-{step}",pitch=pitch))
    result,report=stabilize_repeated_phrases(notes,tempo,"guitar")
    assert [note.pitch for note in result[-4:]]==[53,53,55,48]
    assert report["transposed_occurrences"]==0


def test_confirmed_transposition_colors_preceding_partial_response():
    tempo=TempoMap(np.arange(0,20,.5).tolist())
    notes=[]
    for occurrence,offset in enumerate((0.,2.,4.)):
        for step,pitch in enumerate((48,48,50,51)):
            notes.append(Note(offset+step*.25,offset+step*.25+.065,(step%3,),confidence=.8,
                              source_id=f"clean-{occurrence}-{step}",pitch=pitch))
    for step,pitch in enumerate((53,53,55)):
        notes.append(Note(5.25+step*.25,5.25+step*.25+.065,(4,),confidence=.8,
                          source_id=f"partial-{step}",pitch=pitch))
    for step,pitch in enumerate((53,53,55,48)):
        notes.append(Note(6+step*.25,6+step*.25+.065,(4,),confidence=.4 if step==3 else .8,
                          source_id=f"shift-{step}",pitch=pitch))
    result,report=stabilize_repeated_phrases(notes,tempo,"guitar")
    partial=result[-7:-4]
    assert [note.lanes[0] for note in partial]==[1,1,2]
    assert report["transposed_occurrences"]==1


def test_transposed_riff_folds_nearby_weak_harmonic_and_chord():
    tempo=TempoMap(np.arange(0,20,.5).tolist())
    notes=[]
    for occurrence,offset in enumerate((0.,2.,4.)):
        for step,pitch in enumerate((48,48,50,51)):
            notes.append(Note(offset+step*.25,offset+step*.25+.065,(step%3,),confidence=.8,
                              source_id=f"base-{occurrence}-{step}",pitch=pitch))
    for step,pitch in enumerate((53,53,55,48)):
        notes.append(Note(6+step*.25,6+step*.25+.065,(3,),confidence=.4 if step==3 else .8,
                          source_id=f"shift-{step}",pitch=pitch))
    notes.append(Note(7.25,7.315,(0,2),confidence=.45,source_id="weak-harmonic",pitch=48))
    result,report=stabilize_repeated_phrases(notes,tempo,"guitar")
    assert result[-1].pitch==53
    assert result[-1].lanes==(1,)
    assert report["propagated_notes"]==1


def test_band_guitar_lead_focus_uses_monophonic_guide_for_weak_harmonics():
    times=np.arange(0,1,.01)
    midi=np.full_like(times,48.0)
    voiced=np.ones_like(times,dtype=bool)
    events=[
        {"start":.10,"end":.25,"pitch":48,"confidence":.62,"source_id":"root","source":"guitar"},
        {"start":.30,"end":.45,"pitch":51,"confidence":.44,"source_id":"weak","source":"guitar"},
        {"start":.50,"end":.65,"pitch":48,"confidence":.50,"source_id":"fundamental","source":"guitar"},
        {"start":.50,"end":.65,"pitch":55,"confidence":.47,"source_id":"harmonic","source":"guitar"},
    ]
    result,report=_focus_band_guitar_candidates(events,times,midi,voiced,.375)
    assert [event["pitch"] for event in result]==[48,48,48]
    assert result[1]["lead_focus_corrected"] is True
    assert report["pitch_corrections"]==1
    assert report["groups_resolved"]==1


def test_band_guitar_lead_focus_preserves_strong_chord():
    times=np.arange(0,1,.01)
    midi=np.full_like(times,48.0)
    voiced=np.ones_like(times,dtype=bool)
    events=[
        {"start":.20,"end":.50,"pitch":48,"confidence":.70,"source_id":"a","source":"guitar"},
        {"start":.20,"end":.50,"pitch":55,"confidence":.64,"source_id":"b","source":"guitar"},
    ]
    result,report=_focus_band_guitar_candidates(events,times,midi,voiced,.375)
    assert [event["pitch"] for event in result]==[48,55]
    assert report["groups_resolved"]==0


def test_band_guitar_lead_focus_prefers_near_equal_root_in_fifth_dominant_audio():
    times=np.arange(0,12,.01)
    midi=np.full_like(times,np.nan)
    voiced=np.zeros_like(times,dtype=bool)
    events=[]
    for index in range(50):
        start=.2+index*.2
        events.extend([
            {"start":start,"end":start+.1,"pitch":48,"confidence":.57,
             "model_onset_strength":.7,"source_id":f"root-{index}","source":"guitar"},
            {"start":start,"end":start+.1,"pitch":55,"confidence":.59,
             "model_onset_strength":.7,"source_id":f"fifth-{index}","source":"guitar"},
        ])

    result,report=_focus_band_guitar_candidates(
        events,times,midi,voiced,.375
    )

    assert report["fifth_root_bias"]
    assert report["fifth_groups"]==50
    assert report["octave_groups"]==0
    assert {event["pitch"] for event in result}=={48}


def test_supported_register_run_preserves_root_containing_group():
    events=[
        {"start":1.0,"end":1.2,"pitch":51,"confidence":.70,
         "source_id":"rhythm","source":"guitar"},
        {"start":1.0,"end":1.2,"pitch":75,"confidence":.68,
         "source_id":"lead","source":"guitar"},
    ]
    proposals=[
        {
            "event":{"start":start},
            "pitch":pitch,
            "side":-1,
            "anchor":72,
        }
        for start,pitch in ((.7,51),(.85,58),(1.15,51),(1.3,58))
    ]

    result,resolved=_resolve_supported_register_voice_groups(
        events,proposals,.7
    )

    assert [event["pitch"] for event in result]==[51,75]
    assert [event["source_id"] for event in result]==[
        "rhythm","lead"
    ]
    assert resolved==[]


def test_displaced_register_run_separates_wide_harmonic_group():
    events=[
        {"start":1.0,"end":1.2,"pitch":70,"confidence":.70,
         "source_id":"lower-harmonic","source":"guitar"},
        {"start":1.0,"end":1.2,"pitch":82,"confidence":.68,
         "source_id":"upper-harmonic","source":"guitar"},
    ]
    proposals=[
        {
            "event":{"start":start},
            "pitch":pitch,
            "side":-1,
            "anchor":72,
        }
        for start,pitch in ((.7,51),(.85,58),(1.15,51),(1.3,58))
    ]

    result,resolved=_resolve_supported_register_voice_groups(
        events,proposals,.7
    )

    assert [event["pitch"] for event in result]==[70]
    assert result[0]["source_id"]=="lower-harmonic+upper-harmonic"
    assert result[0]["register_voice_selected"] is True
    assert len(resolved)==1
    assert resolved[0]["proposal_support"]==4


def test_sparse_register_evidence_preserves_wide_strong_group():
    events=[
        {"start":1.0,"end":1.2,"pitch":51,"confidence":.70,
         "source_id":"rhythm","source":"guitar"},
        {"start":1.0,"end":1.2,"pitch":75,"confidence":.68,
         "source_id":"lead","source":"guitar"},
    ]
    proposals=[
        {
            "event":{"start":.8},
            "pitch":51,
            "side":-1,
            "anchor":72,
        },
    ]

    result,resolved=_resolve_supported_register_voice_groups(
        events,proposals,.7
    )

    assert [event["pitch"] for event in result]==[51,75]
    assert resolved==[]


def test_band_guitar_lead_focus_anchors_isolated_octave_to_recent_register():
    times=np.arange(0,1.5,.01)
    midi=np.where(times<.70,48.0,41.0)
    voiced=np.ones_like(times,dtype=bool)
    events=[]
    for index,pitch in enumerate((48,48,50,51)):
        start=.10+index*.14
        events.append({"start":start,"end":start+.10,"pitch":pitch,"confidence":.7,
                       "source_id":f"intro-{index}","source":"guitar"})
    events.extend([
        {"start":.75,"end":.90,"pitch":53,"confidence":.50,"source_id":"upper","source":"guitar"},
        {"start":1.00,"end":1.15,"pitch":41,"confidence":.52,"source_id":"weak-low","source":"guitar"},
    ])
    result,report=_focus_band_guitar_candidates(events,times,midi,voiced,.375)
    assert [event["pitch"] for event in result[-2:]]==[53,53]
    assert report["octave_anchors"]>=2
    assert report["candidate_supported_registers"]==0
    assert report["guided_onset_groups"]>=2
    assert any(
        row["raw_pitch"]==41
        and row["anchored_pitch"]==53
        and row["selected_pitches"]==[53]
        for row in report["raw_guide_samples"]
    )


def test_band_guitar_lead_focus_preserves_repeated_register_evidence():
    times=np.arange(0,2,.01)
    midi=np.full_like(times,70.0)
    voiced=np.ones_like(times,dtype=bool)
    events=[]
    pitches=(70,58,70,51,70,58,70,51,70)
    for index,pitch in enumerate(pitches):
        start=.1+index*.18
        midi[(times>=start)&(times<=start+.13)]=pitch
        events.append({
            "start":start,
            "end":start+.13,
            "pitch":pitch,
            "confidence":.5,
            "source_id":str(index),
            "source":"guitar",
        })

    focused,report=_focus_band_guitar_candidates(
        events,times,midi,voiced,.5
    )
    result,outlier_report=_filter_sequential_lead_outliers(focused,.5)

    assert [event["pitch"] for event in result]==list(pitches)
    assert report["candidate_supported_registers"]==2
    assert outlier_report["candidate_supported_registers"]==2
    assert [
        row["pitch"]
        for row in outlier_report["accepted_register_proposals"]
    ]==[58,58]


def test_band_guitar_lead_focus_keeps_two_supported_repeated_attacks():
    times=np.arange(0,1,.01)
    midi=np.full_like(times,48.0)
    voiced=np.ones_like(times,dtype=bool)
    events=[
        {"start":.20,"end":.42,"pitch":48,"confidence":.60,"source_id":"first","source":"guitar"},
        {"start":.34,"end":.53,"pitch":51,"confidence":.48,"source_id":"second","source":"guitar"},
    ]
    result,_=_focus_band_guitar_candidates(events,times,midi,voiced,.375)
    assert [event["pitch"] for event in result]==[48,48]


def test_band_guitar_voice_filter_removes_rapid_harmonic_spike():
    events=[]
    for index,pitch in enumerate((48,48,55,48,48)):
        events.append({"start":index*.14,"end":index*.14+.10,"pitch":pitch,
                       "confidence":.48,"source_id":str(index),"source":"guitar"})
    result,report=_filter_sequential_lead_outliers(events,.375)
    assert [event["pitch"] for event in result]==[48,48,48,48]
    assert report["sequential_outliers_removed"]==1


def test_band_guitar_voice_filter_preserves_repeated_pedal_riff():
    events=[]
    for index,pitch in enumerate((70,51,70,58,70,51,70,58,70)):
        events.append({
            "start":index*.12,
            "end":index*.12+.10,
            "pitch":pitch,
            "confidence":.48,
            "source_id":str(index),
            "source":"guitar",
        })

    result,report=_filter_sequential_lead_outliers(events,.5)

    assert [event["pitch"] for event in result]==[
        70,51,70,58,70,51,70,58,70
    ]
    assert report["sequential_outliers_removed"]==0
    assert report["repeated_riff_spikes_preserved"]==4


def test_band_guitar_voice_filter_requires_repeated_register_proposals():
    def proposed_events(count):
        events=[]
        proposal_pitches=(51,58,51,58)
        proposal_index=0
        for index in range(9):
            event={
                "start":index*.12,
                "end":index*.12+.10,
                "pitch":70,
                "confidence":.7,
                "source_id":str(index),
                "source":"guitar",
            }
            if index%2 and proposal_index<count:
                pitch=proposal_pitches[proposal_index]
                event["_lead_focus_register_proposal"]={
                    "event":dict(event),
                    "pitch":pitch,
                    "source_id":str(index),
                    "side":-1,
                    "anchor":70,
                    "replaced_correction":False,
                }
                proposal_index+=1
            events.append(event)
        return events

    repeated,accepted=_filter_sequential_lead_outliers(
        proposed_events(4),.5
    )
    sparse,rejected=_filter_sequential_lead_outliers(
        proposed_events(2),.5
    )

    assert [event["pitch"] for event in repeated]==[
        70,51,70,58,70,51,70,58,70
    ]
    assert accepted["candidate_supported_registers"]==4
    assert [event["pitch"] for event in sparse]==[70]*9
    assert rejected["candidate_supported_registers"]==0
    assert rejected["accepted_register_proposals"]==[]
    assert rejected["unsupported_register_recoveries_reverted"]==2
    assert all(
        "_lead_focus_register_proposal" not in event
        for event in repeated+sparse
    )


def test_band_guitar_voice_filter_keeps_coherent_high_run():
    events=[]
    for index,pitch in enumerate((48,48,60,62,64,48)):
        events.append({"start":index*.14,"end":index*.14+.10,"pitch":pitch,
                       "confidence":.5,"source_id":str(index),"source":"guitar"})
    result,report=_filter_sequential_lead_outliers(events,.375)
    assert [event["pitch"] for event in result]==[48,48,60,62,64,48]
    assert report["sequential_outliers_removed"]==0


def test_band_guitar_voice_analysis_flags_dominant_mixed_register_window():
    tempo=TempoMap(np.arange(0,24,.5).tolist())
    events=[]
    for index,pitch in enumerate((48,72,50,74,52,76,53,77,55,79,57,81)):
        confidence=.78 if pitch>=70 else .42
        events.append({"start":1+index*.2,"end":1+index*.2+.08,"pitch":pitch,
                       "confidence":confidence,"source_id":str(index),"source":"guitar"})

    report=analyze_band_guitar_voice(events,tempo)

    assert report["mixed_register_windows"]>=1
    assert report["candidate_windows"]>=1
    assert report["windows"][0]["dominant_register"]=="upper"


def test_hard_wraps_quick_green_orange_jump_inward():
    tempo=TempoMap(np.arange(0,8,.5).tolist())
    expert=[Note(1,1.065,(0,),source_id="low",pitch=48),Note(1.25,1.315,(4,),source_id="high",pitch=60)]
    hard=reductions(expert,"guitar",tempo)["hard"]
    assert [note.lanes for note in hard]==[(0,),(3,)]

def test_reference_conformance_reduces_ambiguous_chords_and_holds(monkeypatch):
    style={"songs":20,"song_chord_rate":{"p90":.25},"song_three_note_chord_rate":{"p90":.05},"song_hold_rate":{"p90":.20}}
    monkeypatch.setattr("rb3charts.chart._reference_style",lambda key:style)
    tempo=TempoMap(np.arange(0,16,.5).tolist())
    notes=[]
    for i in range(20):
        lanes=(0,2,3) if i<8 else ((0,2) if i<12 else (1,))
        notes.append(Note(i*.3,i*.3+(.4 if i<10 else .065),lanes,confidence=.5+i*.01,source_id=str(i),pitch=48+i%5))
    result,report=conform_expert_to_reference(notes,tempo,"guitar")
    assert sum(len(note.lanes)>1 for note in result)<=5
    assert sum(len(note.lanes)>2 for note in result)<=1
    assert sum(note.end-note.start>=.22 for note in result)<=4
    assert report["reference_songs"]==20

def test_fast_hard_reduction_breaks_long_eighth_note_runs():
    tempo=TempoMap(np.arange(0,8,.375).tolist())
    expert=[Note(i*.1875,i*.1875+.065,(i%5,),source_id=str(i),pitch=48+i%5) for i in range(16)]
    hard=reductions(expert,"guitar",tempo)["hard"]
    assert len(hard)==12
    assert all(note.lanes!=(0,4) for note in hard)


def test_hard_reduction_preserves_omitted_contour_anchors():
    tempo=TempoMap(np.arange(0,8,.5).tolist())
    expert=[]
    for index in range(18):
        lane=2 if index%2==0 else (3 if index<9 else 1)
        pitch=(70,77,73,75,58,73)[index%6]
        start=1+(index//2)*.35+(index%2)*.12
        expert.append(Note(
            start,start+.065,(lane,),
            confidence=.5+(index%3)*.05,
            source_id=str(index),
            pitch=pitch,
        ))

    hard=reductions(expert,"guitar",tempo)["hard"]

    assert len(hard)==9
    assert len({note.lanes for note in hard})>=2
    assert {note.source_id for note in hard}<={note.source_id for note in expert}
    assert min(
        tempo.beat_at(b.start)-tempo.beat_at(a.start)
        for a,b in zip(hard,hard[1:])
    )>=.32


def test_hard_reduction_rejects_weak_contour_replacement():
    tempo=TempoMap(np.arange(0,8,.5).tolist())
    expert=[]
    for index in range(12):
        selected=index%2==0
        start=1+(index//2)*.35+(index%2)*.12
        expert.append(Note(
            start,start+.065,
            (2,) if selected else (3,),
            confidence=.8 if selected else .5,
            source_id=str(index),
            pitch=(58,77,63,75,70,73)[index%6],
        ))

    hard=reductions(expert,"guitar",tempo)["hard"]

    assert {note.source_id for note in hard}=={
        str(index) for index in range(0,12,2)
    }
