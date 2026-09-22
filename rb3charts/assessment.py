"""Consistent musical-review triage for generated drafts and batches."""
import json
from pathlib import Path
import numpy as np
from .chart import BAND_GUITAR_SUSTAIN_SECONDS


LABELS={"close_enough":"Close enough to review","review_closely":"Review closely","needs_work":"Needs work"}
DIFFICULTY_LABELS=("Warmup","Apprentice","Solid","Moderate","Challenging","Nightmare","Impossible")


def _guitar_difficulty(attacks,chord_ratio,triad_ratio,peak,pattern):
    # RB3 stores a human-authored rank rather than calculating one. This
    # estimate uses the hardest four-second density, chord complexity and the
    # amount of learnable repetition, then maps the result to RB3's 0-6 scale.
    knots=((2,0),(3,1),(4,2),(5,3),(6,4),(7.5,5),(9,6))
    base=0.0
    for (x0,y0),(x1,y1) in zip(knots,knots[1:]):
        if peak<=x1:
            base=y0+(y1-y0)*(peak-x0)/(x1-x0);break
    else:base=6.0
    repeat_ratio=pattern.get("pattern_supported_attacks",0)/max(1,attacks)
    score=max(0,min(6,base+min(.5,chord_ratio*1.5)+min(.25,triad_ratio*3)-min(.2,repeat_ratio*.5)))
    tier=int(np.clip(round(score),0,6))
    rank_midpoints=(70,157,198,244,300,370,455)
    return round(score,2),tier,rank_midpoints[tier]


def _reference_style(key):
    path=Path(__file__).resolve().parents[1]/"reference"/"style-profile.json"
    if not path.is_file():return {}
    try:return json.loads(path.read_text(encoding="utf-8")).get("styles",{}).get(key,{})
    except (OSError,ValueError):return {}


def assess_chart(charts,patterns,playability,profile,missing=()):
    concerns=[]
    severe=[]
    instruments={}
    for name,parts in charts.items():
        notes=parts["expert"]
        pattern=patterns.get(name,{})
        gate=playability.get(name,{})
        attacks=len(notes)
        lane_counts=pattern.get("lane_counts",{})
        lane_coverage=sum(int(count)>0 for count in lane_counts.values())
        sustain_threshold = (
            .09
            if profile == "electronic"
            else BAND_GUITAR_SUSTAIN_SECONDS
            if name == "guitar"
            else .22
        )
        hold_ratio=sum(note.end-note.start>=sustain_threshold for note in notes)/max(1,attacks)
        removed_ratio=gate.get("removed",0)/max(1,gate.get("before",attacks))
        chord_ratio=sum(len(note.lanes)>1 for note in notes)/max(1,attacks) if name!="drums" else 0
        triad_ratio=sum(len(note.lanes)>2 for note in notes)/max(1,attacks) if name!="drums" else 0
        peak=float(gate.get("peak_four_second_nps",0))
        item={"attacks":attacks,"lane_coverage":lane_coverage,"hold_ratio":round(hold_ratio,3),
              "chord_ratio":round(chord_ratio,3),"removed_ratio":round(removed_ratio,3),
              "peak_four_second_nps":peak,
              "recurring_pattern_families":pattern.get("recurring_pattern_families",0)}
        if name=="guitar":
            difficulty_score,difficulty_tier,difficulty_rank=_guitar_difficulty(
                attacks,chord_ratio,triad_ratio,peak,pattern)
            item.update({"difficulty_score_0_6":difficulty_score,"difficulty_tier":difficulty_tier,
                         "difficulty_label":DIFFICULTY_LABELS[difficulty_tier],
                         "suggested_rb3_rank":difficulty_rank})
            expert_forbidden=sum(len(note.lanes)==3 and 0 in note.lanes and 4 in note.lanes for note in notes)
            hard=parts.get("hard",[])
            hard_forbidden=sum(len(note.lanes)>2 or (0 in note.lanes and 4 in note.lanes) for note in hard)
            item["expert_forbidden_chords"]=expert_forbidden
            item["hard_forbidden_chords"]=hard_forbidden
            if expert_forbidden:severe.append(f"guitar: {expert_forbidden} Expert chords span green through orange")
            if hard_forbidden:severe.append(f"guitar: {hard_forbidden} Hard chords violate reduced chord-shape rules")
        if "contour_agreement_ratio" in pattern:
            item["contour_agreement_ratio"]=pattern["contour_agreement_ratio"]
            item["flattened_change_ratio"]=pattern["flattened_change_ratio"]
            item["adjacent_motion_rate"]=pattern.get("adjacent_motion_rate",0)
            item["wide_jump_rate"]=pattern.get("wide_jump_rate",0)
        instruments[name]=item
        if name!="drums" and attacks>=30 and lane_coverage<3:
            severe.append(f"{name}: only {lane_coverage} lanes are used")
        if name!="drums" and pattern.get("contour_agreement_ratio",1)<.75:
            severe.append(f"{name}: lane direction often conflicts with pitch direction")
        if name!="drums" and pattern.get("flattened_change_ratio",0)>.40:
            concerns.append(f"{name}: many detected pitch changes remain on one lane")
        if profile=="band" and name=="guitar" and pattern.get("lane_motion_comparisons",0)>=20:
            reference=_reference_style("guitar.expert")
            adjacent_floor=reference.get("song_adjacent_motion_rate",{}).get("p10")
            wide_ceiling=reference.get("song_wide_jump_rate",{}).get("p90")
            if adjacent_floor is not None and pattern.get("adjacent_motion_rate",1)<adjacent_floor:
                concerns.append(f"guitar: adjacent fret motion is below the reference range at {pattern['adjacent_motion_rate']:.0%}")
            if wide_ceiling is not None and pattern.get("wide_jump_rate",0)>wide_ceiling:
                concerns.append(f"guitar: wide fret jumps exceed the reference range at {pattern['wide_jump_rate']:.0%}")
        peak_limit=10 if name=="drums" else (6 if profile=="electronic" else (8 if name=="guitar" else 7))
        if peak>peak_limit:
            concerns.append(f"{name}: peak density is {peak:.2f}/s")
        if removed_ratio>.50:
            severe.append(f"{name}: over half of candidates were rejected as congestion")
        elif removed_ratio>.30:
            concerns.append(f"{name}: {removed_ratio:.0%} of candidates were rejected as congestion")
        if name=="guitar" and chord_ratio>.65:
            severe.append(f"{name}: {chord_ratio:.0%} of attacks are chords")
        elif name=="guitar" and chord_ratio>.45:
            concerns.append(f"{name}: chord frequency is high at {chord_ratio:.0%}")
        hold_limit=.97 if profile=="electronic" else (.35 if name=="guitar" else .60)
        if name!="drums" and hold_ratio>hold_limit:
            concerns.append(f"{name}: hold frequency is high at {hold_ratio:.0%}")
        if attacks>=200 and pattern.get("recurring_pattern_families",0)==0:
            concerns.append(f"{name}: no recurring four-note patterns were detected")
    for name in missing:
        severe.append(f"{name}: requested instrument is missing")
    status="needs_work" if severe else ("review_closely" if concerns else "close_enough")
    return {"status":status,"label":LABELS[status],"profile":profile,
            "summary":severe+concerns or ["No automatic congestion, lane, contour, chord or sustain flags."],
            "severe":severe,"concerns":concerns,"instruments":instruments,
            "human_review_required":True}


def write_song_review_notes(path,title,artist,assessment):
    lines=[f"# Chart review notes — {title}","",f"Artist: {artist}",
           f"Automatic assessment: **{assessment['label']}**",""]
    lines += [f"- {note}" for note in assessment["summary"]]
    lines += ["","## Listening verdict","","- [ ] Close enough","- [ ] Review closely","- [ ] Needs work",
              "","## Timestamp notes","","- `00:00` — ","","## Instrument notes","",
              "- Guitar: ","- Bass: ","- Drums: ","- Keys: ",""]
    Path(path).write_text("\n".join(lines),encoding="utf-8")


def write_batch_review(path,results):
    lines=["# Batch chart review","",
           "Use the automatic assessment to choose where to listen first. Record your own verdict and timestamp notes in each song folder's `review_notes.md`.","",
           "| Song | Automatic assessment | Profile | Review | Main note |",
           "|---|---|---|---|---|"]
    for result in results:
        if result.get("status")=="failed":
            lines.append(f"| {result.get('source','Unknown')} | Needs work | — | — | Generation failed: {result.get('error','unknown error')} |")
            continue
        assessment=result.get("assessment",{})
        folder=result.get("folder","")
        title=result.get("title") or (Path(folder).name if folder else "Untitled result")
        review=Path(result["review"]).as_posix() if result.get("review") else ""
        review_link=f"[Open review]({review})" if review else "—"
        note=(assessment.get("summary") or ["Listening review required."])[0].replace("|","/")
        lines.append(f"| {title} | {assessment.get('label','Unrated')} | {result.get('chart_profile','—')} | {review_link} | {note} |")
    lines += ["","## Batch observations","","- Closest charts: ","- Charts needing the most work: ",
              "- Repeated problem across songs: ","- Useful timestamps: ",""]
    Path(path).write_text("\n".join(lines),encoding="utf-8")
