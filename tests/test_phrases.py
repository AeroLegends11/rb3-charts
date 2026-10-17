import numpy as np
from rb3charts.model import Note, TempoMap
from rb3charts.phrases import (
    _apply_audio_repeat_consistency,
    _apply_band_chord_sections,
    _apply_rhythmic_riff_consistency,
    _collapse_isolated_wide_chords,
    _lane_run_quality,
    _preserve_supported_lane_span,
    _remove_pre_chord_voice_duplicates,
    _spread_supported_pitch_contour,
    _stabilize_repeat_fingerings,
    _stabilize_repeated_pitch_lanes,
    author_guitar_phrases,
    select_lead_phrases,
)
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
    assert "pitch_anchor_contours_protected" in report
    assert report["phrases"][1]["reused_fingering"]
    assert [(n.start,n.source_id,n.pitch) for n in result] == [(n.start,n.source_id,n.pitch) for n in notes]
    assert all(n.lanes == (1,) for n in notes)  # No mutation of detector/mapping inputs.
    assert all(abs(n.end-n.start-.065)<1e-9 for n in result)


def test_rhythmic_riff_reuses_widest_pitch_stable_cycle():
    pitches = (51, 44, 56, 51, 47, 56, 46, 54, 42, 51)
    offsets = (.4, .75, 1.02, 1.21, 1.48, 1.8, 2.02, 2.54, 2.76, 2.97)
    canonical_lanes = (3, 2, 3, 2, 1, 2, 1, 3, 0, 2)
    notes = []
    for cycle in range(3):
        for index, (pitch, offset) in enumerate(zip(pitches, offsets)):
            beat = cycle * 3 + offset
            lane = canonical_lanes[index] if cycle == 1 else 2
            notes.append(Note(
                beat * .5,
                beat * .5 + .065,
                (lane,),
                .7,
                f"{cycle}-{index}",
                pitch,
            ))

    report = _apply_rhythmic_riff_consistency(notes, grid())

    expected = [2, 2, 2, 2, 1, 2, 1, 3, 0, 2]
    assert [note.lanes[0] for note in notes[:10]] == expected
    assert [note.lanes[0] for note in notes[10:20]] == list(canonical_lanes)
    assert [note.lanes[0] for note in notes[20:]] == expected
    assert report["families_applied"] == 1
    assert report["notes_changed"] == 8


def test_final_cleanup_collapses_only_isolated_wide_chord():
    notes = [
        Note(i * .2, i * .2 + .065, (1, 4) if i == 5 else (2,),
             confidence=.706 if i == 5 else .8, source_id=str(i), pitch=48 + i)
        for i in range(11)
    ]

    report = _collapse_isolated_wide_chords(notes, grid())

    assert notes[5].lanes == (1,)
    assert report["notes_changed"] == 1


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


def test_phrase_removes_partial_short_guitar_sustain():
    result,report=author_guitar_phrases(
        [Note(1,1.3,(1,),.8,"short-hold",60)],grid()
    )
    assert np.isclose(result[0].end-result[0].start,.065)
    assert report["short_tails_removed"][0]["source_id"]=="short-hold"


def test_phrase_preserves_sustain_at_electronic_threshold():
    result,report=author_guitar_phrases(
        [Note(1,1.284,(1,),.8,"electronic-hold",60)],grid(),
        sustain_threshold=.09,
    )
    assert np.isclose(result[0].end-result[0].start,.284)
    assert report["short_tails_removed"]==[]
    assert report["sustain_threshold_seconds"]==.09


def test_sequence_fingering_can_handle_more_than_five_pitches_without_out_of_range():
    notes=[Note(1+i*.125,1+i*.125+.065,(1,),.9,str(i),p)
           for i,p in enumerate([60,62,64,65,67,69,71,72,71,69,67,65,64,62,60])]
    result,report=author_guitar_phrases(notes,grid())
    assert all(0<=n.lanes[0]<=4 for n in result)
    assert report["after"]["collapsed_pitch_changes"] < report["before"]["collapsed_pitch_changes"]
    assert reductions(result,"guitar",grid())["expert"] == result


def test_lane_run_quality_distinguishes_flat_contour_from_repeated_pitch():
    collapsed=[
        Note(i*.125,i*.125+.065,(2,),pitch=pitch)
        for i,pitch in enumerate((60,62,64,65,67,69,71,72))
    ]
    repeated=[
        Note(2+i*.125,2+i*.125+.065,(1,),pitch=60)
        for i in range(8)
    ]
    repeated_with_tail=[
        Note(4+i*.125,4+i*.125+.065,(3,),pitch=pitch)
        for i,pitch in enumerate((70,70,70,70,70,72,68))
    ]

    report=_lane_run_quality([collapsed,repeated,repeated_with_tail])

    assert report["max_same_lane_run"]==8
    assert report["long_same_lane_runs"]==2
    assert report["collapsed_contour_runs"]==1
    assert report["runs"][0]["pitch_changes"]==7
    assert report["lane_counts"]=={0:0,1:8,2:8,3:7,4:0}


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


def test_band_gate_preserves_supported_six_attack_picking():
    notes = [
        Note(1+i/12,1+i/12+.065,(i%5,),.8,str(i),(60,62,64,65,67,69)[i%6])
        for i in range(24)
    ]

    ordinary,_ = enforce_playability(notes,"guitar",grid(),"band")
    supported,report = enforce_playability(notes,"guitar",grid(),"band",phrase_aware=True)

    assert len(supported) > len(ordinary)
    assert len(supported) == len(notes)
    assert report["supported_phrase_attacks_per_beat"] == 6


def test_phrase_fingering_preserves_repeated_supported_lane_span():
    pitches = (48, 55, 60, 67) * 8
    incoming = list((0, 1, 3, 4) * 8)
    incoming[4] = 1
    rows = [
        Note(i * .125, i * .125 + .065, (lane,), pitch=pitch, source_id=str(i))
        for i, (pitch, lane) in enumerate(zip(pitches, incoming))
    ]

    lanes, changed = _preserve_supported_lane_span(rows, [1, 1, 2, 3] * 8)

    assert changed
    assert lanes == [0, 1, 3, 4] * 8


def test_phrase_fingering_spreads_supported_no_reversal_contour():
    pitches = (48, 50, 52, 55, 60) * 7
    incoming = (0, 0, 1, 2, 3) * 7
    rows = [
        Note(i * .125, i * .125 + .065, (lane,), pitch=pitch, source_id=str(i))
        for i, (pitch, lane) in enumerate(zip(pitches, incoming))
    ]

    lanes, changed = _spread_supported_pitch_contour(
        rows, [1, 1, 2, 2, 3] * 7
    )

    assert changed
    assert set(lanes) == set(range(5))
    assert lanes == lanes[:5] * 7


def test_pre_chord_cleanup_removes_continuing_voice_duplicate():
    notes = [
        Note(1.0, 1.065, (4,), .58, "single", 54),
        Note(1.125, 1.19, (1, 3), .62, "low+continued", 47),
        Note(1.5, 1.565, (2,), .7, "tail", 52),
    ]
    evidence = {"candidates": [
        {
            "start": 1.0, "end": 1.11, "pitch": 54, "confidence": .58,
            "source_id": "single", "accepted": True,
            "evidence": {"attack_strength": .01},
        },
        {
            "start": 1.125, "end": 1.2, "pitch": 47, "confidence": .62,
            "source_id": "low", "accepted": True,
            "evidence": {"attack_strength": .08},
        },
        {
            "start": 1.125, "end": 1.2, "pitch": 54, "confidence": .6,
            "source_id": "continued", "accepted": True,
            "evidence": {"attack_strength": .08},
        },
    ]}

    result, report = _remove_pre_chord_voice_duplicates(notes, evidence)

    assert [note.source_id for note in result] == ["low+continued", "tail"]
    assert report["notes_removed"] == 1
    assert report["events"][0]["attack_gap_ms"] == 125


def test_pre_chord_cleanup_preserves_separate_grace_note():
    notes = [
        Note(1.0, 1.065, (4,), .7, "grace", 54),
        Note(1.125, 1.19, (1, 3), .62, "low+high", 47),
    ]
    evidence = {"candidates": [
        {
            "start": 1.0, "end": 1.065, "pitch": 54, "confidence": .7,
            "source_id": "grace", "accepted": True,
        },
        {
            "start": 1.125, "end": 1.2, "pitch": 47, "confidence": .62,
            "source_id": "low", "accepted": True,
        },
        {
            "start": 1.125, "end": 1.2, "pitch": 55, "confidence": .6,
            "source_id": "high", "accepted": True,
        },
    ]}

    result, report = _remove_pre_chord_voice_duplicates(notes, evidence)

    assert result == notes
    assert report["notes_removed"] == 0


def test_pre_chord_cleanup_preserves_strong_separate_attack():
    notes = [
        Note(1.0, 1.065, (4,), .7, "single", 54),
        Note(1.125, 1.19, (1, 3), .62, "low+continued", 47),
    ]
    evidence = {"candidates": [
        {
            "start": 1.0, "end": 1.11, "pitch": 54, "confidence": .7,
            "source_id": "single", "accepted": True,
            "evidence": {"attack_strength": .07},
        },
        {
            "start": 1.125, "end": 1.2, "pitch": 47, "confidence": .62,
            "source_id": "low", "accepted": True,
            "evidence": {"attack_strength": .08},
        },
        {
            "start": 1.125, "end": 1.2, "pitch": 54, "confidence": .6,
            "source_id": "continued", "accepted": True,
            "evidence": {"attack_strength": .08},
        },
    ]}

    result, report = _remove_pre_chord_voice_duplicates(notes, evidence)

    assert result == notes
    assert report["notes_removed"] == 0


def test_pre_chord_cleanup_removes_tight_sandwiched_single_note():
    notes = [
        Note(1.0, 1.065, (1, 3), .7, "prev_chord", 42),
        Note(1.060, 1.100, (4,), .5, "sandwiched_flam", 58),
        Note(1.130, 1.195, (2, 4), .75, "next_chord", 49),
    ]
    evidence = {"candidates": [
        {"start": 1.0, "end": 1.065, "pitch": 42, "confidence": .7, "source_id": "prev_chord", "accepted": True},
        {"start": 1.060, "end": 1.100, "pitch": 58, "confidence": .5, "source_id": "sandwiched_flam", "accepted": True},
        {"start": 1.130, "end": 1.195, "pitch": 49, "confidence": .75, "source_id": "next_chord", "accepted": True},
    ]}
    result, report = _remove_pre_chord_voice_duplicates(notes, evidence)
    assert [n.source_id for n in result] == ["prev_chord", "next_chord"]
    assert report["notes_removed"] == 1


def test_pre_chord_cleanup_removes_weak_strum_flam():
    notes = [
        Note(1.050, 1.090, (2,), .5, "flam", 56),
        Note(1.120, 1.185, (1, 3), .75, "chord", 40),
    ]
    evidence = {"candidates": [
        {"start": 1.050, "end": 1.090, "pitch": 56, "confidence": .5, "source_id": "flam", "accepted": True, "evidence": {"attack_strength": 0.0}},
        {"start": 1.120, "end": 1.185, "pitch": 40, "confidence": .75, "source_id": "chord", "accepted": True, "evidence": {"attack_strength": 0.08}},
    ]}
    result, report = _remove_pre_chord_voice_duplicates(notes, evidence)
    assert [n.source_id for n in result] == ["chord"]
    assert report["notes_removed"] == 1


def test_repeated_exact_riff_preserves_structural_variation():
    notes = []
    for occurrence, lanes in enumerate((
        (0, 1, 2, 1),
        (0, 1, 2, 1),
        (0, 1, 2, 1),
        (1, 2, 2, 2),
    )):
        start = 1 + occurrence * 2
        notes.extend(
            Note(
                start + index * .125,
                start + index * .125 + .065,
                (lane,),
                pitch=pitch,
                source_id=f"{occurrence}-{index}",
            )
            for index, (pitch, lane)
            in enumerate(zip((60, 62, 64, 62), lanes))
        )

    before = [note.lanes for note in notes]

    report = _stabilize_repeat_fingerings(notes, grid())

    assert [note.lanes for note in notes] == before
    assert report["notes_changed"] == 0


def test_repeated_exact_riff_reuses_relative_fingering_offset():
    notes = [
        Note(
            1 + occurrence * 2 + index * .125,
            1 + occurrence * 2 + index * .125 + .065,
            (lane,),
            pitch=pitch,
            source_id=f"{occurrence}-{index}",
        )
        for occurrence, lanes in enumerate((
            (0, 1, 2, 1),
            (1, 2, 3, 2),
        ))
        for index, (pitch, lane)
        in enumerate(zip((60, 62, 64, 62), lanes))
    ]

    report = _stabilize_repeat_fingerings(notes, grid())

    first = [note.lanes[0] for note in notes[:4]]
    second = [note.lanes[0] for note in notes[4:]]
    assert first == second
    assert first in ([0, 1, 2, 1], [1, 2, 3, 2])
    assert report["offset_families"] == 1
    assert report["notes_changed"] == 4


def test_repeat_fingering_rejects_new_boundary_contour_collapse(monkeypatch):
    occurrences=[]
    notes=[]
    for occurrence,lanes in enumerate(((0,1),(0,1),(0,1),(1,2))):
        phrase=[
            Note(
                occurrence*2+index*.125,
                occurrence*2+index*.125+.065,
                (lane,),
                pitch=pitch,
                source_id=f"{occurrence}-{index}",
            )
            for index,(pitch,lane) in enumerate(zip((56,61),lanes))
        ]
        occurrences.append((len(notes),phrase))
        notes.extend(phrase)
    tail=[
        Note(6.25+i*.125,6.315+i*.125,(1,),pitch=pitch,source_id=f"tail-{i}")
        for i,pitch in enumerate((61,54,61,56))
    ]
    notes.extend(tail)
    monkeypatch.setattr(
        "rb3charts.phrases._motif_groups",
        lambda *_:(notes,[(None,None,occurrences)]),
    )
    before=[note.lanes for note in notes]

    report=_stabilize_repeat_fingerings(
        notes,grid(),quality_phrases=[notes]
    )

    assert [note.lanes for note in notes]==before
    assert report["rejected_collapsed_contours"]==1
    assert report["notes_changed"]==0


def test_repeated_riff_requires_three_example_strict_majority():
    for lane_shapes in (
        ((0, 1, 2, 1), (0, 1, 2, 1), (1, 2, 2, 2)),
        (
            (0, 1, 2, 1),
            (0, 1, 2, 1),
            (0, 1, 2, 1),
            (1, 2, 2, 2),
            (1, 2, 2, 2),
            (1, 2, 2, 2),
        ),
    ):
        notes = [
            Note(
                1 + occurrence * 2 + index * .125,
                1 + occurrence * 2 + index * .125 + .065,
                (lane,),
                pitch=pitch,
                source_id=f"{occurrence}-{index}",
            )
            for occurrence, lanes in enumerate(lane_shapes)
            for index, (pitch, lane)
            in enumerate(zip((60, 62, 64, 62), lanes))
        ]
        before = [note.lanes for note in notes]

        report = _stabilize_repeat_fingerings(notes, grid())

        assert [note.lanes for note in notes] == before
        assert report["notes_changed"] == 0


def test_phrase_authoring_keeps_supported_pitch_lanes_consistent():
    pitches = (48, 55, 60, 67) * 8
    incoming = list((0, 1, 3, 4) * 8)
    incoming[4] = 1
    notes = [
        Note(i * .125, i * .125 + .065, (lane,), pitch=pitch, source_id=str(i))
        for i, (pitch, lane) in enumerate(zip(pitches, incoming))
    ]

    result,report = author_guitar_phrases(notes,grid())

    assert report["supported_span_preservations"] == 1
    assert all(
        len({note.lanes for note in result if note.pitch == pitch}) == 1
        for pitch in set(pitches)
    )


def test_pre_chord_cleanup_removes_sandwiched_weak_single_note_between_identical_chords():
    notes = [
        Note(21.15, 21.25, (0, 2), .8, "chord-1", 42),
        Note(21.30, 21.36, (4,), .5, "guitar-reattack-1844-54", 54),
        Note(21.50, 21.60, (0, 2), .8, "chord-2", 42),
    ]
    evidence = {"candidates": [
        {
            "start": 21.15, "end": 21.25, "pitch": 42, "confidence": .8,
            "source_id": "chord-1", "accepted": True,
            "evidence": {"attack_strength": .08},
        },
        {
            "start": 21.30, "end": 21.36, "pitch": 54, "confidence": .5,
            "source_id": "guitar-reattack-1844-54", "accepted": True,
            "evidence": {"attack_strength": .02},
        },
        {
            "start": 21.50, "end": 21.60, "pitch": 42, "confidence": .8,
            "source_id": "chord-2", "accepted": True,
            "evidence": {"attack_strength": .08},
        },
    ]}

    result, report = _remove_pre_chord_voice_duplicates(notes, evidence)

    assert [note.source_id for note in result] == ["chord-1", "chord-2"]
    assert report["notes_removed"] == 1
    assert report["events"][0]["reason"] == "weak single note sandwiched between identical chords"


def test_band_chord_sections_propagates_power_chord_harmonic_overtones():
    notes = [
        Note(24.0, 24.1, (0,), .8, "seed-v1", 60),
        Note(24.2, 24.3, (0,), .8, "seed-v2", 61),
        Note(24.35, 24.50, (0,), .8, "chord-1", 42),
        Note(24.6, 24.65, (0,), .8, "seed-v3", 62),
        Note(24.70, 24.85, (4,), .7, "overtone-octave", 54),
        Note(24.8, 24.85, (0,), .8, "seed-v4", 63),
        Note(24.95, 25.10, (3,), .7, "overtone-fifth", 49),
        Note(25.35, 25.50, (0,), .8, "chord-2", 42),
        Note(25.70, 25.85, (0,), .8, "chord-3", 42),
    ]
    candidates = [
        {"start": 24.0, "pitch": 60, "confidence": .8, "source_id": "seed-v1", "accepted": True, "model_onset_strength": .8},
        {"start": 24.0, "pitch": 67, "confidence": .8, "source_id": "seed-v1-5th", "accepted": True, "model_onset_strength": .8},
        {"start": 24.2, "pitch": 61, "confidence": .8, "source_id": "seed-v2", "accepted": True, "model_onset_strength": .8},
        {"start": 24.2, "pitch": 68, "confidence": .8, "source_id": "seed-v2-5th", "accepted": True, "model_onset_strength": .8},
        {"start": 24.35, "pitch": 42, "confidence": .8, "source_id": "chord-1", "accepted": True, "model_onset_strength": .8, "evidence": {"attack_strength": .08}},
        {"start": 24.35, "pitch": 49, "confidence": .8, "source_id": "chord-1-5th", "accepted": True, "model_onset_strength": .8, "evidence": {"attack_strength": .08}},
        {"start": 24.6, "pitch": 62, "confidence": .8, "source_id": "seed-v3", "accepted": True, "model_onset_strength": .8},
        {"start": 24.6, "pitch": 69, "confidence": .8, "source_id": "seed-v3-5th", "accepted": True, "model_onset_strength": .8},
        {"start": 24.70, "pitch": 54, "confidence": .7, "source_id": "overtone-octave", "accepted": True, "evidence": {"attack_strength": .07}},
        {"start": 24.8, "pitch": 63, "confidence": .8, "source_id": "seed-v4", "accepted": True, "model_onset_strength": .8},
        {"start": 24.8, "pitch": 70, "confidence": .8, "source_id": "seed-v4-5th", "accepted": True, "model_onset_strength": .8},
        {"start": 24.95, "pitch": 49, "confidence": .7, "source_id": "overtone-fifth", "accepted": True, "evidence": {"attack_strength": .07}},
        {"start": 25.35, "pitch": 42, "confidence": .8, "source_id": "chord-2", "accepted": True, "model_onset_strength": .8, "evidence": {"attack_strength": .08}},
        {"start": 25.35, "pitch": 49, "confidence": .8, "source_id": "chord-2-5th", "accepted": True, "model_onset_strength": .8, "evidence": {"attack_strength": .08}},
        {"start": 25.70, "pitch": 42, "confidence": .8, "source_id": "chord-3", "accepted": True, "model_onset_strength": .8, "evidence": {"attack_strength": .08}},
        {"start": 25.70, "pitch": 49, "confidence": .8, "source_id": "chord-3-5th", "accepted": True, "model_onset_strength": .8, "evidence": {"attack_strength": .08}},
    ]
    evidence = {
        "focus": {"onset_groups": 10, "pitch_corrections": 1, "octave_anchors": 0},
        "candidates": candidates,
    }
    tempo = TempoMap(np.arange(0, 30, .5).tolist())
    report = _apply_band_chord_sections(notes, tempo, evidence, min_proposals=5)
    assert report["propagation_gate_passed"]
    assert report["propagated_notes"] >= 2
    by_id = {note.source_id: note for note in notes}
    assert by_id["overtone-octave"].lanes == (0, 2)
    assert by_id["overtone-fifth"].lanes == (0, 2)
    assert by_id["chord-1"].lanes == (0, 2)
    assert by_id["chord-2"].lanes == (0, 2)
    assert by_id["chord-3"].lanes == (0, 2)


def test_band_chord_sections_propagates_two_strum_chord_progressions():
    notes = [
        Note(10.0, 10.1, (0,), .8, "seed-1", 60),
        Note(10.2, 10.3, (0,), .8, "seed-2", 61),
        Note(10.4, 10.5, (0,), .8, "chord-fsharp-1", 42),
        Note(10.6, 10.7, (0,), .8, "chord-fsharp-2", 42),
        Note(10.8, 10.9, (1,), .8, "chord-gsharp-1", 44),
        Note(11.0, 11.1, (1,), .8, "chord-gsharp-2", 44),
        Note(11.2, 11.3, (0,), .8, "seed-3", 62),
    ]
    candidates = [
        {"start": 10.0, "pitch": 60, "confidence": .8, "source_id": "seed-1", "accepted": True, "model_onset_strength": .8},
        {"start": 10.0, "pitch": 67, "confidence": .8, "source_id": "seed-1-5th", "accepted": True, "model_onset_strength": .8},
        {"start": 10.2, "pitch": 61, "confidence": .8, "source_id": "seed-2", "accepted": True, "model_onset_strength": .8},
        {"start": 10.2, "pitch": 68, "confidence": .8, "source_id": "seed-2-5th", "accepted": True, "model_onset_strength": .8},
        {"start": 10.4, "pitch": 42, "confidence": .8, "source_id": "chord-fsharp-1", "accepted": True, "model_onset_strength": .8, "evidence": {"attack_strength": .08}},
        {"start": 10.4, "pitch": 49, "confidence": .8, "source_id": "chord-fsharp-1-5th", "accepted": True, "model_onset_strength": .8, "evidence": {"attack_strength": .08}},
        {"start": 10.6, "pitch": 42, "confidence": .8, "source_id": "chord-fsharp-2", "accepted": True, "model_onset_strength": .8, "evidence": {"attack_strength": .08}},
        {"start": 10.6, "pitch": 49, "confidence": .8, "source_id": "chord-fsharp-2-5th", "accepted": True, "model_onset_strength": .8, "evidence": {"attack_strength": .08}},
        {"start": 10.8, "pitch": 44, "confidence": .8, "source_id": "chord-gsharp-1", "accepted": True, "model_onset_strength": .8, "evidence": {"attack_strength": .08}},
        {"start": 10.8, "pitch": 51, "confidence": .8, "source_id": "chord-gsharp-1-5th", "accepted": True, "model_onset_strength": .8, "evidence": {"attack_strength": .08}},
        {"start": 11.0, "pitch": 44, "confidence": .8, "source_id": "chord-gsharp-2", "accepted": True, "model_onset_strength": .8, "evidence": {"attack_strength": .08}},
        {"start": 11.0, "pitch": 51, "confidence": .8, "source_id": "chord-gsharp-2-5th", "accepted": True, "model_onset_strength": .8, "evidence": {"attack_strength": .08}},
        {"start": 11.2, "pitch": 62, "confidence": .8, "source_id": "seed-3", "accepted": True, "model_onset_strength": .8},
        {"start": 11.2, "pitch": 69, "confidence": .8, "source_id": "seed-3-5th", "accepted": True, "model_onset_strength": .8},
    ]
    evidence = {
        "focus": {"onset_groups": 10, "pitch_corrections": 1, "octave_anchors": 0},
        "candidates": candidates,
    }
    tempo = TempoMap(np.arange(0, 20, .5).tolist())
    report = _apply_band_chord_sections(notes, tempo, evidence, min_proposals=3)
    assert report["propagation_gate_passed"]
    by_id = {note.source_id: note for note in notes}
    assert by_id["chord-fsharp-1"].lanes == (0, 2)
    assert by_id["chord-fsharp-2"].lanes == (0, 2)
    assert by_id["chord-gsharp-1"].lanes == (1, 3)
    assert by_id["chord-gsharp-2"].lanes == (1, 3)



def test_phrase_authoring_stabilizes_broad_four_note_per_second_run():
    pitches = (48, 55, 60, 67) * 8
    notes = [
        Note(i * .25, i * .25 + .065, (lane,), pitch=pitch, source_id=str(i))
        for i, (pitch, lane) in enumerate(zip(pitches, (0, 1, 3, 4) * 8))
    ]

    result,report = author_guitar_phrases(notes,grid())

    assert report["supported_span_preservations"] == 1
    assert len({note.lanes[0] for note in result}) == 4


def test_phrase_span_rejects_locally_reversed_mapping():
    pitches = (48, 55, 60, 67) * 8
    rows = [
        Note(i * .125, i * .125 + .065, (lane,), pitch=pitch, source_id=str(i))
        for i, (pitch, lane) in enumerate(zip(pitches, (0, 2, 1, 4) * 8))
    ]
    authored = [1, 1, 2, 3] * 8

    lanes,changed = _preserve_supported_lane_span(rows, authored)

    assert not changed
    assert lanes == authored


def test_phrase_span_rejects_collapsed_contour_regression():
    # Pitches that climb over 12 semitones, but incoming has collapsed higher notes onto lane 4
    pitches = [48, 50, 52, 55, 57, 60, 62, 65] * 4
    # Incoming maps top 4 pitches all to lane 4, creating collapsed contour
    incoming = [0, 1, 2, 3, 4, 4, 4, 4] * 4
    rows = [
        Note(i * .125, i * .125 + .065, (lane,), pitch=pitch, source_id=str(i))
        for i, (pitch, lane) in enumerate(zip(pitches, incoming))
    ]
    # Authored correctly distributes notes across frets without collapsed contours
    authored = [0, 0, 1, 1, 2, 2, 3, 4] * 4

    lanes, changed = _preserve_supported_lane_span(rows, authored)

    assert not changed
    assert lanes == authored



def test_repeated_pitch_stability_does_not_flatten_supported_contour():
    rows = [
        Note(i * .25, i * .25 + .065, (lane,), pitch=pitch, source_id=str(i))
        for i, (pitch, lane) in enumerate(zip(
            (60, 62, 60, 62, 60, 58, 60),
            (1, 2, 1, 2, 1, 2, 3),
        ))
    ]

    lanes, changed, protected = _stabilize_repeated_pitch_lanes(
        rows, [note.lanes[0] for note in rows]
    )

    assert lanes == [1, 2, 1, 2, 1, 2, 3]
    assert changed == 0
    assert protected == 1


def test_repeated_pitch_stability_keeps_safe_anchor_change():
    rows = [
        Note(i * .25, i * .25 + .065, (lane,), pitch=pitch, source_id=str(i))
        for i, (pitch, lane) in enumerate(zip(
            (60, 62, 60, 62, 60, 62, 60),
            (1, 2, 1, 2, 1, 2, 2),
        ))
    ]

    lanes, changed, protected = _stabilize_repeated_pitch_lanes(
        rows, [note.lanes[0] for note in rows]
    )

    assert lanes == [1, 2, 1, 2, 1, 2, 1]
    assert changed == 1
    assert protected == 0


def test_empty_phrase_inputs():
    assert author_guitar_phrases([],grid())[0]==[]
    assert select_lead_phrases([])[0]==[]


def test_band_chord_sections_expand_repeated_fifths_as_power_chords():
    notes=[
        Note(1+i*.25,1+i*.25+.09,(2,),.8,f"note-{i}",60 if i in (0,3,6) else 64)
        for i in range(8)
    ]
    candidates=[]
    for index,start in enumerate((1.0,1.75,2.5)):
        candidates.extend([
            {"start":start,"end":start+.1,"pitch":48,"confidence":.7,
             "model_onset_strength":.7,"source_id":f"low-{index}","accepted":True},
            {"start":start,"end":start+.1,"pitch":55,"confidence":.5,
             "model_onset_strength":.65,"source_id":f"high-{index}","accepted":True},
        ])
    evidence={
        "candidates":candidates,
        "focus":{"onset_groups":20,"pitch_corrections":9,"octave_anchors":12},
    }
    baseline,_=author_guitar_phrases(notes,grid())
    result,report=author_guitar_phrases(notes,grid(),chord_evidence=evidence)
    assert sum(len(note.lanes)>1 for note in result)==3
    assert any(len(note.lanes)==1 for note in result)
    assert all(note.lanes[1]-note.lanes[0]==2 for note in result if len(note.lanes)==2)
    assert [(note.start,note.end,note.source_id,note.pitch) for note in result]==[
        (note.start,note.end,note.source_id,note.pitch) for note in baseline
    ]
    assert report["chord_sections"]["notes_changed"]==3


def test_band_chord_sections_preserve_existing_voicings():
    notes=[
        Note(1+i*.25,1+i*.25+.09,(0,4) if i==0 else (2,),.8,f"note-{i}",60)
        for i in range(10)
    ]
    candidates=[]
    for index,start in enumerate((1.0,2.0,3.0)):
        candidates.extend([
            {"start":start,"end":start+.1,"pitch":48,"confidence":.7,
             "model_onset_strength":.7,"source_id":f"low-{index}","accepted":True},
            {"start":start,"end":start+.1,"pitch":55,"confidence":.5,
             "model_onset_strength":.65,"source_id":f"high-{index}","accepted":True},
        ])
    result,report=author_guitar_phrases(
        notes,grid(),chord_evidence={
            "candidates":candidates,
            "focus":{"onset_groups":20,"pitch_corrections":2,"octave_anchors":3},
        }
    )
    assert result[0].lanes==(0,4)
    assert sum(len(note.lanes)>1 for note in result)==3
    assert report["chord_sections"]["notes_changed"]==2


def test_band_chord_sections_reject_sparse_evidence_independent_of_guide_stability():
    notes=[
        Note(1+i*.25,1+i*.25+.09,(2,),.8,f"note-{i}",60+i%2)
        for i in range(5)
    ]
    candidates=[]
    for index,start in enumerate((1.0,1.5)):
        candidates.extend([
            {"start":start,"end":start+.1,"pitch":48,"confidence":.7,
             "model_onset_strength":.7,"source_id":f"low-{index}","accepted":True},
            {"start":start,"end":start+.1,"pitch":55,"confidence":.5,
             "model_onset_strength":.65,"source_id":f"high-{index}","accepted":True},
        ])
    sparse, sparse_report=author_guitar_phrases(
        notes,grid(),chord_evidence={
            "candidates":candidates,
            "focus":{"onset_groups":20,"pitch_corrections":2,"octave_anchors":3},
        }
    )
    unstable, unstable_report=author_guitar_phrases(
        notes,grid(),chord_evidence={
            "candidates":candidates*2,
            "focus":{"onset_groups":10,"pitch_corrections":3,"octave_anchors":6},
        }
    )
    assert all(len(note.lanes)==1 for note in sparse+unstable)
    assert sparse_report["chord_sections"]["candidate_seeds"]==2
    assert unstable_report["chord_sections"]["gate_passed"]


def test_band_chord_sections_propagate_only_when_fifths_dominate_octaves():
    supported=set(index for index in range(75) if index%3)
    repeated_supported=set(sorted(supported)[:10])
    notes=[
        Note(
            1+index*.25,
            1+index*.25+.09,
            (2,),
            .8,
            f"note-{index}",
            60 if index not in supported or index in repeated_supported else 70+index,
        )
        for index in range(75)
    ]
    candidates=[]
    for index in supported:
        start=notes[index].start
        candidates.extend([
            {"start":start,"end":start+.1,"pitch":48,"confidence":.7,
             "model_onset_strength":.7,"source_id":f"low-{index}","accepted":True},
            {"start":start,"end":start+.1,"pitch":55,"confidence":.5,
             "model_onset_strength":.65,"source_id":f"high-{index}","accepted":True},
        ])
    result,report=author_guitar_phrases(
        notes,grid(),chord_evidence={
            "candidates":candidates,
            "focus":{"onset_groups":75,"pitch_corrections":3,"octave_anchors":5},
        }
    )
    chord_report=report["chord_sections"]
    assert chord_report["propagation_gate_passed"]
    assert chord_report["candidate_notes"]>=50
    assert chord_report["direct_notes_changed"]==10
    assert chord_report["propagated_notes"]>0
    assert sum(len(note.lanes)>1 for note in result)>10


def test_voice_rejects_weak_isolated_register_spike_but_keeps_strong_jump():
    events=[event(1,60,.8,"other"),event(1.2,48,.41,"other"),event(1.4,62,.8,"other")]
    result,report=select_lead_phrases([events])
    assert len(result)==2 and report["unsupported_register_spikes"]
    events[1]["confidence"]=.9
    assert len(select_lead_phrases([events])[0])==3


def test_continuous_riff_does_not_split_or_recolor_repeated_pitch():
    pitches=[60,62,64,65]*4+[65,64,62,60]*2
    notes=[Note(1+i*.25,1+i*.25+.065,(1,),.8,str(i),p) for i,p in enumerate(pitches)]
    result,report=author_guitar_phrases(notes,grid())
    assert result[15].pitch==result[16].pitch
    assert result[15].lanes==result[16].lanes
    assert len({note.lanes for note in result if note.pitch==60})==1
    assert len(report["phrases"])==1


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


def test_phrase_motif_reuses_shape_across_chord_continuity_anchor():
    notes=[
        *[
            Note(1+step*.25,1+step*.25+.11,(4,),.9,f"left-{step}",pitch)
            for step,pitch in enumerate((60,62,64,65))
        ],
        Note(2,2.1,(0,2),.9,"barrier",60),
        *[
            Note(2.25+step*.25,2.25+step*.25+.11,(4,),.9,f"right-{step}",pitch)
            for step,pitch in enumerate((67,69,71,72))
        ],
    ]

    _,report=author_guitar_phrases(notes,grid())

    assert report["phrases"][1]["continuity_source"]=="through_chord"
    assert report["phrases"][1]["motif_family"] is not None
    assert report["phrases"][1]["motif_reason"]!="not_matched"


def test_window_motif_extends_weak_root_return_after_four_matched_notes():
    notes=[]
    for offset,prefix,pitches,confidences in (
        (1.0,"clear",(47,47,54,51,47),(.8,.8,.8,.8,.65)),
        (4.0,"weak",(46,46,53,50,51),(.8,.8,.8,.8,.5)),
    ):
        for step,(pitch,confidence) in enumerate(zip(pitches,confidences)):
            notes.append(Note(
                offset+step*.25,offset+step*.25+.09,(2,),confidence,
                f"{prefix}-{step}",pitch,
            ))
    result,report=author_guitar_phrases(notes,grid())
    assert [note.lanes for note in result[:5]]==[note.lanes for note in result[5:]]
    assert report["window_motif_low_confidence_closures"]>=1


def test_phrase_motif_constrains_related_windows_inside_unequal_phrases():
    notes = []
    phrases = (
        (1.0, (47, 47, 54, 51, 47), "opening"),
        (4.0, (46, 46, 54, 49, 51, 51, 44, 44, 51, 47, 46, 54, 42), "long"),
    )
    for offset, pitches, prefix in phrases:
        for step, pitch in enumerate(pitches):
            notes.append(
                Note(
                    offset + step * .125,
                    offset + step * .125 + .065,
                    (4,),
                    .7,
                    f"{prefix}-{step}",
                    pitch,
                )
            )

    result, report = author_guitar_phrases(notes, grid())

    opening = [note.lanes[0] for note in result[:4]]
    related = [note.lanes[0] for note in result[5:9]]
    assert [b - a for a, b in zip(opening, opening[1:])] == [
        b - a for a, b in zip(related, related[1:])
    ]
    assert report["window_motif_matching"]["matches_accepted"] > 0
    assert report["phrases"][1]["window_motif_constrained"]
    assert all(
        len(phrase["after"]) == len(phrase["source_ids"])
        for phrase in report["phrases"]
    )


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


def test_audio_repeat_consistency_reuses_supported_lane_and_chord_shape():
    notes = []
    for occurrence, (offset, confidence, lanes) in enumerate((
        (1.0, .9, ((0, 2), (1,), (2, 4), (3,))),
        (5.0, .6, ((1, 3), (1,), (1, 3), (1,))),
    )):
        for step, (pitch, gems) in enumerate(zip((60, 62, 64, 65), lanes)):
            notes.append(Note(
                offset + step * .25,
                offset + step * .25 + .1,
                gems,
                confidence,
                f"{occurrence}-{step}",
                pitch,
            ))
    pitches = [note.pitch for note in notes]
    starts = [note.start for note in notes]

    report = _apply_audio_repeat_consistency(
        notes,
        grid(),
        {"regions": [{
            "left_start": 1.0,
            "left_end": 2.0,
            "right_start": 5.0,
            "right_end": 6.0,
            "similarity": .95,
        }]},
    )

    assert [note.pitch for note in notes] == pitches
    assert [note.start for note in notes] == starts
    assert [note.lanes for note in notes[:4]] == [note.lanes for note in notes[4:]]
    assert report["regions_applied"] == 1
    assert report["notes_changed"] == 3


def test_audio_repeat_consistency_propagates_chords_to_single_note_region():
    notes = [
        # Left region (has chords)
        Note(1.0, 1.1, (0, 2), .9, "left-0", 42),
        Note(1.25, 1.35, (0, 2), .9, "left-1", 42),
        Note(1.5, 1.6, (1, 3), .9, "left-2", 44),
        Note(1.75, 1.85, (1, 3), .9, "left-3", 44),
        # Right region (missed chords, only singles)
        Note(5.0, 5.1, (0,), .6, "right-0", 42),
        Note(5.25, 5.35, (0,), .6, "right-1", 42),
        Note(5.5, 5.6, (1,), .6, "right-2", 44),
        Note(5.75, 5.85, (1,), .6, "right-3", 44),
    ]

    report = _apply_audio_repeat_consistency(
        notes,
        grid(),
        {"regions": [{
            "left_start": 1.0,
            "left_end": 2.0,
            "right_start": 5.0,
            "right_end": 6.0,
            "similarity": .80,
        }]},
    )

    assert report["regions_applied"] == 0
    assert report["notes_changed"] == 0
    by_id = {note.source_id: note for note in notes}
    assert [by_id[f"right-{index}"].lanes for index in range(4)] == [
        (0,), (0,), (1,), (1,)
    ]


def test_audio_repeat_consistency_only_copies_locally_supported_chords():
    notes = [
        Note(1.0, 1.1, (0, 2), .9, "left-0", 42),
        Note(1.25, 1.35, (0, 2), .9, "left-1", 42),
        Note(1.5, 1.6, (1, 3), .9, "left-2", 44),
        Note(1.75, 1.85, (1, 3), .9, "left-3", 44),
        Note(5.0, 5.1, (0,), .6, "right-0+candidate", 42),
        Note(5.25, 5.35, (0,), .6, "right-1", 42),
        Note(5.5, 5.6, (1,), .6, "right-2+candidate", 44),
        Note(5.75, 5.85, (1,), .6, "right-3", 44),
    ]

    report = _apply_audio_repeat_consistency(
        notes,
        grid(),
        {"regions": [{
            "left_start": 1.0,
            "left_end": 2.0,
            "right_start": 5.0,
            "right_end": 6.0,
            "similarity": .80,
        }]},
    )

    assert report["regions_applied"] == 1
    assert report["notes_changed"] == 2
    by_id = {note.source_id: note for note in notes}
    assert len(by_id["right-0+candidate"].lanes) == 2
    assert by_id["right-1"].lanes == (0,)
    assert len(by_id["right-2+candidate"].lanes) == 2
    assert by_id["right-3"].lanes == (1,)


def test_audio_repeat_consistency_propagates_chords_with_strong_audio_match():
    notes = []
    for index in range(16):
        pitch = 42 + 2 * (index % 2)
        offset = index * .5
        notes.extend((
            Note(
                1.0 + offset,
                1.1 + offset,
                (0, 2) if index % 2 == 0 else (1, 3),
                .9,
                f"left-{index}",
                pitch,
            ),
            Note(
                20.0 + offset,
                20.1 + offset,
                (0,) if index % 2 == 0 else (1,),
                .6,
                f"right-{index}",
                pitch,
            ),
        ))

    report = _apply_audio_repeat_consistency(
        notes,
        grid(),
        {"regions": [{
            "left_start": 1.0,
            "left_end": 9.0,
            "right_start": 20.0,
            "right_end": 28.0,
            "duration": 8.0,
            "similarity": .92,
            "anchors": 20,
        }]},
    )

    assert report["regions_applied"] == 1
    assert report["notes_changed"] == 16
    by_id = {note.source_id: note for note in notes}
    assert all(len(by_id[f"right-{index}"].lanes) == 2 for index in range(16))


def test_rhythmic_riff_uses_transitions_and_octave_tolerance_in_canonical_scoring():
    pitches = (51, 44, 56, 51, 47, 56, 46, 54, 42, 51)
    offsets = (.4, .75, 1.02, 1.21, 1.48, 1.8, 2.02, 2.54, 2.76, 2.97)
    canonical_lanes = (3, 2, 3, 2, 1, 2, 1, 3, 0, 2)
    flat_lanes = (2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
    notes = []
    for cycle in range(3):
        cycle_pitches = (
            [p + 12 if idx in (0, 9) else p for idx, p in enumerate(pitches)]
            if cycle == 2 else pitches
        )
        for index, (pitch, offset) in enumerate(zip(cycle_pitches, offsets)):
            beat = cycle * 3 + offset
            lane = canonical_lanes[index] if cycle == 1 else flat_lanes[index]
            notes.append(Note(
                beat * .5,
                beat * .5 + .065,
                (lane,),
                .75,
                f"{cycle}-{index}",
                pitch,
            ))

    report = _apply_rhythmic_riff_consistency(notes, grid())

    expected = [2, 2, 2, 2, 1, 2, 1, 3, 0, 2]
    assert report["families_applied"] == 1
    assert [note.lanes[0] for note in notes[:10]] == expected
    assert [note.lanes[0] for note in notes[10:20]] == list(canonical_lanes)
    assert [note.lanes[0] for note in notes[20:]] == expected
