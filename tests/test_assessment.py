from rb3charts.assessment import (
    _guitar_difficulty,
    analyze_structure_diagnostics,
    assess_chart,
    write_batch_review,
    write_song_review_notes,
)
from rb3charts.model import Note,TempoMap


def chart(count=40,lanes=(0,1,2,3,4)):
    notes=[Note(i*.25,i*.25+.065,(lanes[i%len(lanes)],),source_id=str(i),pitch=48+i%5) for i in range(count)]
    return {"guitar":{difficulty:notes for difficulty in ("expert","hard","medium","easy")}}


def test_assessment_flags_single_lane_chart():
    charts=chart(lanes=(1,))
    patterns={"guitar":{"lane_counts":{1:40},"recurring_pattern_families":2,"contour_agreement_ratio":1,"flattened_change_ratio":0}}
    playability={"guitar":{"before":40,"removed":0,"holds":0,"peak_four_second_nps":4}}
    result=assess_chart(charts,patterns,playability,"band")
    assert result["status"]=="needs_work"
    assert "only 1 lanes" in result["severe"][0]


def test_batch_review_lists_song_rating(tmp_path):
    path=tmp_path/"batch.md"
    write_batch_review(path,[{"status":"draft_ready","title":"Test Song","folder":str(tmp_path),
                              "review":str(tmp_path/"review.html"),"chart_profile":"band",
                              "assessment":{"label":"Close enough to review","summary":["No flags."],
                                            "structure":{"activity_blocks":3,
                                                         "repeated_drum_groove_families":2,
                                                         "drum_groove_supported_attack_ratio":.625}}}])
    text=path.read_text(encoding="utf-8")
    assert "Test Song" in text and "Close enough to review" in text and "Open review" in text
    assert "3 blocks; 2 drum grooves (62%)" in text


def test_structure_diagnostics_reports_neutral_sections_and_repeated_drum_grooves():
    tempo=TempoMap([beat*.5 for beat in range(81)])
    guitar=[
        Note(1+i*.5,1+i*.5+.1,(i%5,),source_id=f"g-{i}")
        for i in range(24)
    ]
    drums=[]
    for measure in (0,1,4,5):
        start=measure*2
        drums.extend([
            Note(start,start+.05,(0,),source_id=f"d-{measure}-0"),
            Note(start+.5,start+.55,(1,),source_id=f"d-{measure}-1"),
            Note(start+1,start+1.05,(0,),source_id=f"d-{measure}-2"),
            Note(start+1.5,start+1.55,(2,),source_id=f"d-{measure}-3"),
        ])
    charts={
        "guitar":{"expert":guitar},
        "drums":{"expert":drums},
    }
    report=analyze_structure_diagnostics(charts,tempo,40)
    assert report["scope"].startswith("post-authoring neutral")
    assert report["section_beats"]==32
    assert len(report["sections"])==3
    assert report["sections"][0]["start_seconds"]==0
    assert report["sections"][-1]["end_seconds"]==40
    assert report["strongest_activity_boundaries"]
    assert report["drum_grooves"]["repeated_families"]==1
    assert report["drum_grooves"]["supported_attacks"]==16
    assert report["drum_grooves"]["supported_attack_ratio"]==1


def test_assessment_exposes_structure_diagnostics_in_review_notes(tmp_path):
    charts=chart()
    patterns={"guitar":{
        "lane_counts":{lane:8 for lane in range(5)},
        "recurring_pattern_families":2,
        "contour_agreement_ratio":1,
        "flattened_change_ratio":0,
    }}
    playability={"guitar":{
        "before":40,
        "removed":0,
        "holds":0,
        "peak_four_second_nps":4,
    }}
    structure={
        "sections":[{},{}],
        "strongest_activity_boundaries":[{}],
        "drum_grooves":{
            "repeated_families":3,
            "supported_attack_ratio":.625,
        },
    }
    result=assess_chart(
        charts,patterns,playability,"band",
        structure_diagnostics=structure,
    )
    assert result["structure"]["activity_blocks"]==2
    path=tmp_path/"review_notes.md"
    write_song_review_notes(path,"Demo","Artist",result)
    text=path.read_text(encoding="utf-8")
    assert "Difficulty reductions" in text
    assert "| Guitar | 40 | 40 | 40 | 40 | 100.0%" in text
    assert "Neutral structure diagnostics" in text
    assert "62.5%" in text


def test_assessment_rejects_forbidden_hard_chord():
    charts=chart()
    charts["guitar"]["hard"]=[Note(0,.065,(0,4),source_id="wide",pitch=48)]
    patterns={"guitar":{"lane_counts":{lane:8 for lane in range(5)},"recurring_pattern_families":2,"contour_agreement_ratio":1,"flattened_change_ratio":0}}
    playability={"guitar":{"before":40,"removed":0,"holds":0,"peak_four_second_nps":4}}
    result=assess_chart(charts,patterns,playability,"band")
    assert result["status"]=="needs_work"
    assert result["instruments"]["guitar"]["hard_forbidden_chords"]==1


def test_assessment_flags_motion_outside_reference_range(monkeypatch):
    charts=chart()
    patterns={"guitar":{"lane_counts":{lane:8 for lane in range(5)},"recurring_pattern_families":2,
                        "contour_agreement_ratio":1,"flattened_change_ratio":0,
                        "lane_motion_comparisons":39,"adjacent_motion_rate":.5,"wide_jump_rate":.3}}
    playability={"guitar":{"before":40,"removed":0,"holds":0,"peak_four_second_nps":4}}
    monkeypatch.setattr("rb3charts.assessment._reference_style",lambda key:{
        "song_adjacent_motion_rate":{"p10":.7},"song_wide_jump_rate":{"p90":.15}})
    result=assess_chart(charts,patterns,playability,"band")
    assert any("adjacent fret motion" in concern for concern in result["concerns"])
    assert any("wide fret jumps" in concern for concern in result["concerns"])


def test_assessment_flags_inconsistent_repeated_riff_fingering():
    charts=chart()
    patterns={"guitar":{
        "lane_counts":{lane:8 for lane in range(5)},
        "recurring_pattern_families":4,
        "contour_agreement_ratio":1,
        "flattened_change_ratio":0,
        "repeatable_fingering_occurrences":20,
        "repeatable_fingering_ratio":.75,
    }}
    playability={"guitar":{
        "before":40,
        "removed":0,
        "holds":0,
        "peak_four_second_nps":4,
    }}

    result=assess_chart(charts,patterns,playability,"band")

    assert any(
        "repeated riffs use inconsistent fingerings at 75% agreement"
        in concern
        for concern in result["concerns"]
    )


def test_assessment_flags_hard_reduction_outside_reference_range(monkeypatch):
    charts=chart()
    patterns={"guitar":{
        "lane_counts":{lane:8 for lane in range(5)},
        "recurring_pattern_families":2,
        "contour_agreement_ratio":1,
        "flattened_change_ratio":0,
    }}
    playability={"guitar":{
        "before":40,
        "removed":0,
        "holds":0,
        "peak_four_second_nps":4,
    }}
    monkeypatch.setattr(
        "rb3charts.assessment._reference_retention",
        lambda instrument:{"p10":.5,"p90":.8},
    )
    result=assess_chart(charts,patterns,playability,"band")
    assert result["instruments"]["guitar"]["hard_attack_retention"]==1
    assert any("above the reference p90" in item for item in result["concerns"])


def test_guitar_difficulty_estimate_keeps_fast_repeatable_chart_challenging():
    score,tier,rank=_guitar_difficulty(900,.23,.02,7.0,{"pattern_supported_attacks":190})
    assert 4<=score<=5.5
    assert tier==5 and 333<=rank<=408
