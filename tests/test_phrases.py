import numpy as np
from rb3charts.model import Note, TempoMap
from rb3charts.phrases import author_guitar_phrases, select_lead_phrases
from rb3charts.chart import enforce_playability, reductions


def grid():
    return TempoMap(np.arange(0, 20, .5).tolist())


def test_phrase_restores_distinct_pitches_and_repeated_shape_without_new_attacks():
    notes = [Note(t+i*.125,t+i*.125+.11,(1,),.8,f"{t}-{i}",pitch)
             for t in (1,4) for i,pitch in enumerate([60,62,64,65,64,62,60])]
    result, report = author_guitar_phrases(notes,grid())
    assert [n.lanes for n in result[:7]] == [n.lanes for n in result[7:]]
    assert len(set(n.lanes for n in result[:4])) == 4
    assert report["after"]["collapsed_pitch_changes"] == 0
    assert report["phrases"][1]["reused_fingering"]
    assert [(n.start,n.source_id,n.pitch) for n in result] == [(n.start,n.source_id,n.pitch) for n in notes]
    assert all(n.lanes == (1,) for n in notes)  # No mutation of detector/mapping inputs.
    assert all(abs(n.end-n.start-.065)<1e-9 for n in result)


def test_hammer_on_like_phrase_reuses_repeated_fingering_without_new_attacks():
    notes = [Note(t+i*.125,t+i*.125+.09,(4,),.85,f"{t}-{i}",pitch)
             for t in (1,3) for i,pitch in enumerate([60,62,64,62,60])]
    result, report = author_guitar_phrases(notes,grid())
    assert [n.lanes for n in result[:5]] == [n.lanes for n in result[5:]]
    assert report["phrases"][1]["reused_fingering"]
    assert report["notes_added"] == report["notes_removed"] == 0


def test_phrase_preserves_long_hold_chord_and_repeat_pitch():
    notes=[Note(1,2,(0,2),.9,"chord",60), Note(2.2,2.8,(0,),.9,"hold",64),
           Note(3,3.07,(4,),.8,"repeat",64)]
    result,_=author_guitar_phrases(notes,grid())
    assert result[0] == notes[0]
    assert result[1].end == 2.8
    assert result[1].lanes == result[2].lanes


def test_sequence_fingering_can_handle_more_than_five_pitches_without_out_of_range():
    notes=[Note(1+i*.125,1+i*.125+.065,(1,),.9,str(i),p)
           for i,p in enumerate([60,62,64,65,67,69,71,72,71,69,67,65,64,62,60])]
    result,report=author_guitar_phrases(notes,grid())
    assert all(0<=n.lanes[0]<=4 for n in result)
    assert report["after"]["collapsed_pitch_changes"] < report["before"]["collapsed_pitch_changes"]
    assert reductions(result,"guitar",grid())["expert"] == result


def event(t,p,c,source):
    return dict(start=t,end=t+.2,pitch=p,confidence=c,source=source,source_id=f"{source}-{t}")


def test_voice_context_avoids_isolated_stronger_wrong_register():
    lead=[event(1+i*.2,p,.8,"other") for i,p in enumerate([72,74,76,74,72])]
    distractor=[event(1.4,48,.88,"guitar")]
    result,report=select_lead_phrases([lead,distractor])
    assert [e["pitch"] for e in result] == [72,74,76,74,72]
    assert report["contextual_choices"] == 1


def test_voice_context_allows_sustained_real_change_and_supported_chord():
    low=[event(1+i*.2,48,.95,"guitar") for i in range(5)]
    high=[event(1+i*.2,76,.45,"other") for i in range(5)]
    result,_=select_lead_phrases([low,high])
    assert {e["pitch"] for e in result} == {48}
    chord=[dict(event(3,p,.9,"other"),end=4,model_onset_strength=.8) for p in [60,64,67,72]]
    result,_=select_lead_phrases([chord])
    assert len(result)==3 and len({e['pitch']%12 for e in result})==3


def test_expert_gate_preserves_supported_fast_riff_but_not_unrelated_burst():
    notes=[Note(1+i*.125,1+i*.125+.065,(i%4,),.8,str(i),[60,62,64,65][i%4]) for i in range(16)]
    old,_=enforce_playability(notes,"guitar",grid(),"electronic")
    new,_=enforce_playability(notes,"guitar",grid(),"electronic",phrase_aware=True)
    assert len(new)==16 and len(old)<len(new)
    burst=[Note(1+i*.045,1+i*.045+.04,(i%5,),.5,str(i),40+i*7) for i in range(7)]
    filtered,_=enforce_playability(burst,"guitar",grid(),"electronic",phrase_aware=True)
    assert len(filtered)<len(burst)


def test_empty_phrase_inputs():
    assert author_guitar_phrases([],grid())[0]==[]
    assert select_lead_phrases([])[0]==[]


def test_voice_rejects_weak_isolated_register_spike_but_keeps_strong_jump():
    events=[event(1,60,.8,"other"),event(1.2,48,.41,"other"),event(1.4,62,.8,"other")]
    result,report=select_lead_phrases([events])
    assert len(result)==2 and report["unsupported_register_spikes"]
    events[1]["confidence"]=.9
    assert len(select_lead_phrases([events])[0])==3


def test_artificial_phrase_boundary_does_not_recolor_repeated_pitch():
    pitches=[60,62,64,65]*4+[65,64,62,60]*2
    notes=[Note(1+i*.25,1+i*.25+.065,(1,),.8,str(i),p) for i,p in enumerate(pitches)]
    result,report=author_guitar_phrases(notes,grid())
    assert result[15].pitch==result[16].pitch
    assert result[15].lanes==result[16].lanes
    assert report['phrases'][1]['joined_previous_phrase']


def test_real_rest_allows_new_hand_position():
    notes=[Note(1,1.065,(4,),.9,'a',60),Note(4,4.065,(0,),.9,'b',72)]
    _,report=author_guitar_phrases(notes,grid())
    assert not report['phrases'][1]['joined_previous_phrase']
    assert report["phrases"][1]["continuity_source"] == "none"


def test_short_chord_barrier_preserves_returning_riff_shape():
    notes = []
    for offset,prefix,pitches in ((1.0,"a",(60,62,64,65)),(2.5,"b",(60,62,64,65))):
        for step,pitch in enumerate(pitches):
            notes.append(Note(offset+step*.25,offset+step*.25+.09,(4,),.9,f"{prefix}-{step}",pitch))
    notes[4:4] = [
        Note(2.0,2.12,(0,2),.9,"chord-0",57),
        Note(2.25,2.37,(1,3),.9,"chord-1",60),
    ]
    result,report=author_guitar_phrases(notes,grid())
    assert [n.lanes for n in result[:4]] == [n.lanes for n in result[6:10]]
    assert report["phrases"][1]["continuity_source"] == "through_chord"
    assert report["phrases"][1]["barrier_chords"] == 2
    assert report["continuity_summary"]["through_chord"] == 1


def test_transposed_return_after_chords_reuses_relative_shape():
    notes = []
    for offset,prefix,pitches in ((1.0,"a",(60,62,64,65)),(2.25,"b",(67,69,71,72))):
        for step,pitch in enumerate(pitches):
            notes.append(Note(offset+step*.25,offset+step*.25+.09,(0,),.9,f"{prefix}-{step}",pitch))
    notes[4:4] = [Note(2.0,2.1,(0,2),.9,"chord",55)]
    result,report=author_guitar_phrases(notes,grid())
    assert [n.lanes for n in result[:4]] == [n.lanes for n in result[5:9]]
    assert report["phrases"][1]["continuity_source"] == "through_chord"
    assert report["phrases"][1]["reused_fingering"]


def test_short_chord_barrier_does_not_force_strong_variation():
    notes = []
    for offset,prefix,pitches in ((1.0,"a",(60,62,64,65)),(2.25,"b",(60,66,67,70))):
        for step,pitch in enumerate(pitches):
            notes.append(Note(offset+step*.25,offset+step*.25+.09,(2,),.9,f"{prefix}-{step}",pitch))
    notes[4:4] = [Note(2.0,2.1,(0,2),.9,"chord",55)]
    result,report=author_guitar_phrases(notes,grid())
    assert [n.lanes for n in result[:4]] != [n.lanes for n in result[5:9]]
    assert report["phrases"][1]["continuity_source"] == "through_chord"
    assert not report["phrases"][1]["reused_fingering"]


def test_long_sustain_before_chord_does_not_fake_nearby_continuity():
    notes = [
        Note(1.0,2.9,(0,),.9,"hold",60),
        Note(3.0,3.1,(0,2),.9,"chord",55),
        Note(3.25,3.34,(4,),.9,"return-0",60),
        Note(3.5,3.59,(4,),.9,"return-1",62),
        Note(3.75,3.84,(4,),.9,"return-2",64),
    ]
    _,report=author_guitar_phrases(notes,grid())
    assert report["phrases"][1]["continuity_source"] == "none"
    assert not report["phrases"][1]["joined_previous_phrase"]


def test_phrase_motif_reuses_transposed_shape_without_pitch_edits():
    notes=[]
    for offset,pitches in ((1.0,(60,62,64,65)),(4.0,(67,69,71,72))):
        for step,pitch in enumerate(pitches):
            notes.append(Note(offset+step*.25,offset+step*.25+.11,(4,),.9,f"{offset}-{step}",pitch))
    result,report=author_guitar_phrases(notes,grid())
    assert [n.pitch for n in result]==[n.pitch for n in notes]
    assert [n.lanes for n in result[:4]]==[n.lanes for n in result[4:]]
    assert report["phrases"][0]["motif_family"]==report["phrases"][1]["motif_family"]


def test_phrase_motif_keeps_strong_variation_and_rhythm_only_coincidence_distinct():
    notes=[]
    for offset,pitches in (
            (1.0,(60,62,64,65)),
            (4.0,(60,66,67,70)),
            (7.0,(72,60,75,59))):
        for step,pitch in enumerate(pitches):
            notes.append(Note(offset+step*.25,offset+step*.25+.11,(2,),.9,f"{offset}-{step}",pitch))
    _,report=author_guitar_phrases(notes,grid())
    families=[phrase["motif_family"] for phrase in report["phrases"]]
    assert families[1] is None and families[2] is None
    assert not any(phrase["motif_reused"] for phrase in report["phrases"])


def test_phrase_motif_late_confirmation_labels_earlier_weak_occurrence():
    notes=[]
    for offset,pitches,conf in (
            (1.0,(60,63,64,65),(.9,.4,.9,.9)),
            (4.0,(60,62,64,65),(.9,.9,.9,.9)),
            (7.0,(67,69,71,72),(.9,.9,.9,.9))):
        for step,(pitch,c) in enumerate(zip(pitches,conf)):
            notes.append(Note(offset+step*.25,offset+step*.25+.11,(3,),c,f"{offset}-{step}",pitch))
    _,report=author_guitar_phrases(notes,grid())
    families=[phrase["motif_family"] for phrase in report["phrases"]]
    assert families[0] == families[1] == families[2]


def test_phrase_motif_matching_is_deterministic():
    notes=[Note(1+i*.25,1+i*.25+.11,(4,),.9,str(i),p)
           for i,p in enumerate((60,62,64,65,67,69,71,72,60,62,64,65))]
    first,first_report=author_guitar_phrases(notes,grid())
    second,second_report=author_guitar_phrases(notes,grid())
    assert [(n.lanes,n.pitch,n.end) for n in first]==[(n.lanes,n.pitch,n.end) for n in second]
    assert first_report["phrases"]==second_report["phrases"]
