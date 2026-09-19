from rb3charts.assessment import assess_chart,write_batch_review,_guitar_difficulty
from rb3charts.model import Note


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
                              "assessment":{"label":"Close enough to review","summary":["No flags."]}}])
    text=path.read_text(encoding="utf-8")
    assert "Test Song" in text and "Close enough to review" in text and "Open review" in text


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


def test_guitar_difficulty_estimate_keeps_fast_repeatable_chart_challenging():
    score,tier,rank=_guitar_difficulty(900,.23,.02,7.0,{"pattern_supported_attacks":190})
    assert 4<=score<=5.5
    assert tier==5 and 333<=rank<=408
