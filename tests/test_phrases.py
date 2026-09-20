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


def test_motif_constraints_do_not_flatten_strong_variation():
    notes = []
    for offset in (1, 4):
        for step, (pitch, lane) in enumerate(zip((60, 60, 62, 63), (0, 0, 1, 2))):
            notes.append(Note(offset + step * .125, offset + step * .125 + .11, (lane,), .8,
                              f"repeat-{offset}-{step}", pitch))
    for step, (pitch, lane) in enumerate(zip((60, 61, 62, 72), (0, 0, 1, 2))):
        notes.append(Note(7 + step * .125, 7 + step * .125 + .11, (lane,), .8,
                          f"variation-{step}", pitch))

    result, report = author_guitar_phrases(notes, grid())

    first = [note.lanes[0] for note in result[:4]]
    second = [note.lanes[0] for note in result[4:8]]
    variation = [note.lanes[0] for note in result[8:]]
    assert first == second == [0, 0, 1, 2]
    assert variation == [0, 1, 2, 3]
    assert report["motif_constrained_phrases"] == 2
    assert not report["phrases"][2]["motif_constrained"]
    assert report["motif_matching"]["matches_rejected"] > 0


def test_related_riff_with_small_pitch_change_reuses_lane_shape_and_reports_match():
    notes = []
    for offset, pitches, lanes, tag in (
            (1.0, (60, 62, 64, 65, 64), (0, 1, 2, 3, 2), "a"),
            (4.0, (63, 65, 67, 68, 70), (1, 2, 3, 4, 4), "b")):
        for step, (pitch, lane) in enumerate(zip(pitches, lanes)):
            notes.append(Note(offset + step * .125, offset + step * .125 + .11, (lane,), .8,
                              f"{tag}-{step}", pitch))
    result, report = author_guitar_phrases(notes, grid())
    first = [note.lanes[0] for note in result[:5]]
    second = [note.lanes[0] for note in result[5:]]
    assert [b - a for a, b in zip(first[:4], first[1:4])] == [b - a for a, b in zip(second[:4], second[1:4])]
    assert report["phrases"][1]["motif_constrained"]
    assert any(row["reason"] == "accepted" and row["confidence"] >= .72
               for row in report["motif_matching"]["accepted"])


def test_three_note_related_riff_gets_motif_constraint_from_short_window():
    notes = []
    for offset, pitches, lanes, tag in (
            (1.0, (60, 62, 64), (0, 1, 2), "a"),
            (4.0, (63, 65, 68), (4, 4, 4), "b")):
        for step, (pitch, lane) in enumerate(zip(pitches, lanes)):
            notes.append(Note(offset + step * .125, offset + step * .125 + .11, (lane,), .8,
                              f"{tag}-{step}", pitch))
    result, report = author_guitar_phrases(notes, grid())
    first = [note.lanes[0] for note in result[:3]]
    second = [note.lanes[0] for note in result[3:]]
    assert [b - a for a, b in zip(first[:2], first[1:])] == [b - a for a, b in zip(second[:2], second[1:])]
    assert report["phrases"][1]["motif_constrained"]
    assert report["motif_matching"]["windows_considered"] >= 2
    assert any(row["reason"] == "accepted" and len(row["alignment"]) == 3
               for row in report["motif_matching"]["accepted"])


def test_offset_insert_alignment_constrains_shared_subsequence_without_crossing_boundaries():
    notes = []
    for step, (pitch, lane) in enumerate(zip((60, 62, 64, 65, 64), (0, 1, 2, 3, 2))):
        notes.append(Note(1 + step * .125, 1 + step * .125 + .11, (lane,), .8, f"base-{step}", pitch))
    for step, (pitch, lane) in enumerate(zip((58, 60, 62, 64, 65, 64), (4, 4, 3, 2, 1, 0))):
        notes.append(Note(3 + step * .125, 3 + step * .125 + .11, (lane,), .8, f"shift-{step}", pitch))
    original = [(note.start, note.source_id, note.pitch) for note in notes]
    result, report = author_guitar_phrases(notes, grid())
    assert [(note.start, note.source_id, note.pitch) for note in result] == original
    assert len(report["phrases"]) == 2
    inserted = [note.lanes[0] for note in result if note.source_id.startswith("shift-")]
    assert inserted[1:5] == [1, 2, 3, 4]
    assert any(row["edits"] == 1 and row["reason"] == "accepted" for row in report["motif_matching"]["accepted"])
