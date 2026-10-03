"""Consistent musical-review triage for generated drafts and batches."""
import json
from pathlib import Path
import numpy as np
from .chart import BAND_GUITAR_SUSTAIN_SECONDS


LABELS={"close_enough":"Close enough to review","review_closely":"Review closely","needs_work":"Needs work"}
DIFFICULTY_LABELS=("Warmup","Apprentice","Solid","Moderate","Challenging","Nightmare","Impossible")


def analyze_structure_diagnostics(charts,tempo,duration):
    """Summarize neutral activity sections and repeated drum measures."""
    section_beats=tempo.numerator*8
    end_beat=max(1,int(np.ceil(tempo.beat_at(duration))))
    sections=[]
    for start_beat in range(0,end_beat,section_beats):
        end_section_beat=min(end_beat,start_beat+section_beats)
        start=(
            0.0
            if start_beat==0
            else max(0.0,float(tempo.seconds_at(start_beat)))
        )
        end=(
            float(duration)
            if end_section_beat>=end_beat
            else min(float(duration),float(tempo.seconds_at(end_section_beat)))
        )
        if end<=start:
            continue
        instruments={}
        for instrument,parts in charts.items():
            notes=[
                note for note in parts["expert"]
                if start<=note.start<end
            ]
            instruments[instrument]={
                "attacks":len(notes),
                "attacks_per_second":round(len(notes)/(end-start),3),
                "chord_rate":round(
                    sum(len(note.lanes)>1 for note in notes)/max(1,len(notes)),4
                ) if instrument!="drums" else 0.0,
                "hold_rate":round(
                    sum(note.end-note.start>=.22 for note in notes)/max(1,len(notes)),4
                ) if instrument!="drums" else 0.0,
                "lane_counts":{
                    str(lane):sum(lane in note.lanes for note in notes)
                    for lane in range(5)
                },
            }
        sections.append({
            "index":len(sections)+1,
            "start_beat":start_beat,
            "end_beat":end_section_beat,
            "start_seconds":round(start,3),
            "end_seconds":round(end,3),
            "instruments":instruments,
        })
    boundaries=[]
    names=sorted(charts)
    for left,right in zip(sections,sections[1:]):
        left_rates=np.asarray([
            left["instruments"][name]["attacks_per_second"] for name in names
        ])
        right_rates=np.asarray([
            right["instruments"][name]["attacks_per_second"] for name in names
        ])
        change=float(np.sum(np.abs(right_rates-left_rates))/max(.001,np.sum(left_rates+right_rates)))
        boundaries.append({
            "seconds":right["start_seconds"],
            "beat":right["start_beat"],
            "activity_change":round(change,4),
        })
    grooves={}
    if "drums" in charts:
        measures={}
        for note in charts["drums"]["expert"]:
            beat=float(tempo.beat_at(note.start))
            measure=int(np.floor(beat/tempo.numerator))
            relative=beat-measure*tempo.numerator
            slot=int(round(relative*4))
            measures.setdefault(measure,[]).append((slot,tuple(note.lanes)))
        signatures={}
        for measure,events in measures.items():
            signature=tuple(sorted(events))
            if len(signature)>=2:
                signatures.setdefault(signature,[]).append(measure)
        repeated=[
            (signature,occurrences)
            for signature,occurrences in signatures.items()
            if len(occurrences)>=2
        ]
        repeated.sort(key=lambda row:(-len(row[1]),row[1][0],row[0]))
        repeated_measures={
            measure for _,occurrences in repeated for measure in occurrences
        }
        drum_notes=charts["drums"]["expert"]
        supported=sum(
            int(np.floor(tempo.beat_at(note.start)/tempo.numerator))
            in repeated_measures
            for note in drum_notes
        )
        grooves={
            "analyzed_measures":len(measures),
            "repeated_families":len(repeated),
            "supported_attacks":supported,
            "supported_attack_ratio":round(supported/max(1,len(drum_notes)),4),
            "top_families":[{
                "occurrences":len(occurrences),
                "measure_indices":occurrences[:16],
                "events":[{
                    "sixteenth_slot":slot,
                    "lanes":list(lanes),
                } for slot,lanes in signature],
            } for signature,occurrences in repeated[:12]],
        }
    return {
        "schema_version":1,
        "scope":"post-authoring neutral diagnostics; no automatic section labels",
        "section_beats":section_beats,
        "sections":sections,
        "strongest_activity_boundaries":sorted(
            boundaries,key=lambda row:(-row["activity_change"],row["seconds"])
        )[:12],
        "drum_grooves":grooves,
    }


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


def _reference_retention(instrument):
    path=Path(__file__).resolve().parents[1]/"reference"/"style-profile.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")).get(
            "expert_to_hard_retention",{}
        ).get(instrument,{})
    except (OSError,ValueError):
        return {}


def assess_chart(
    charts,
    patterns,
    playability,
    profile,
    missing=(),
    structure_diagnostics=None,
):
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
        hard_retention=len(parts.get("hard",[]))/max(1,attacks)
        item["hard_attack_retention"]=round(hard_retention,4)
        retention_reference=_reference_retention(name)
        item["reference_hard_retention_p10"]=retention_reference.get("p10")
        item["reference_hard_retention_p90"]=retention_reference.get("p90")
        if (
            attacks>=30
            and retention_reference.get("p90") is not None
            and hard_retention>retention_reference["p90"]
        ):
            concerns.append(
                f"{name}: Hard retains {hard_retention:.0%} of Expert attacks, "
                f"above the reference p90 of "
                f"{retention_reference['p90']:.0%}"
            )
        elif (
            attacks>=30
            and retention_reference.get("p10") is not None
            and hard_retention<retention_reference["p10"]
        ):
            concerns.append(
                f"{name}: Hard retains {hard_retention:.0%} of Expert attacks, "
                f"below the reference p10 of "
                f"{retention_reference['p10']:.0%}"
            )
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
    structure={}
    if structure_diagnostics:
        grooves=structure_diagnostics.get("drum_grooves",{})
        structure={
            "activity_blocks":len(structure_diagnostics.get("sections",[])),
            "activity_boundaries":len(
                structure_diagnostics.get(
                    "strongest_activity_boundaries",[]
                )
            ),
            "repeated_drum_groove_families":grooves.get(
                "repeated_families",0
            ),
            "drum_groove_supported_attack_ratio":grooves.get(
                "supported_attack_ratio",0
            ),
        }
    return {"status":status,"label":LABELS[status],"profile":profile,
            "summary":severe+concerns or ["No automatic congestion, lane, contour, chord or sustain flags."],
            "severe":severe,"concerns":concerns,"instruments":instruments,
            "structure":structure,
            "human_review_required":True}


def write_song_review_notes(path,title,artist,assessment):
    lines=[f"# Chart review notes — {title}","",f"Artist: {artist}",
           f"Automatic assessment: **{assessment['label']}**",""]
    lines += [f"- {note}" for note in assessment["summary"]]
    structure=assessment.get("structure",{})
    if structure:
        lines += [
            "",
            "## Neutral structure diagnostics",
            "",
            f"- Activity blocks: {structure['activity_blocks']}",
            f"- Ranked activity boundaries: {structure['activity_boundaries']}",
            "- Repeated drum-groove families: "
            f"{structure['repeated_drum_groove_families']}",
            "- Expert drum attacks in repeated groove measures: "
            f"{structure['drum_groove_supported_attack_ratio']:.1%}",
        ]
    lines += ["","## Listening verdict","","- [ ] Close enough","- [ ] Review closely","- [ ] Needs work",
              "","## Timestamp notes","","- `00:00` — ","","## Instrument notes","",
              "- Guitar: ","- Bass: ","- Drums: ","- Keys: ",""]
    Path(path).write_text("\n".join(lines),encoding="utf-8")


def write_batch_review(path,results):
    lines=["# Batch chart review","",
           "Use the automatic assessment to choose where to listen first. Record your own verdict and timestamp notes in each song folder's `review_notes.md`.","",
           "| Song | Automatic assessment | Profile | Structure evidence | Review | Main note |",
           "|---|---|---|---|---|---|"]
    for result in results:
        if result.get("status")=="failed":
            lines.append(f"| {result.get('source','Unknown')} | Needs work | — | — | — | Generation failed: {result.get('error','unknown error')} |")
            continue
        assessment=result.get("assessment",{})
        folder=result.get("folder","")
        title=result.get("title") or (Path(folder).name if folder else "Untitled result")
        review=Path(result["review"]).as_posix() if result.get("review") else ""
        review_link=f"[Open review]({review})" if review else "—"
        note=(assessment.get("summary") or ["Listening review required."])[0].replace("|","/")
        structure=assessment.get("structure",{})
        structure_note=(
            f"{structure.get('activity_blocks',0)} blocks; "
            f"{structure.get('repeated_drum_groove_families',0)} drum grooves "
            f"({structure.get('drum_groove_supported_attack_ratio',0):.0%})"
            if structure else "—"
        )
        lines.append(f"| {title} | {assessment.get('label','Unrated')} | {result.get('chart_profile','—')} | {structure_note} | {review_link} | {note} |")
    lines += ["","## Batch observations","","- Closest charts: ","- Charts needing the most work: ",
              "- Repeated problem across songs: ","- Useful timestamps: ",""]
    Path(path).write_text("\n".join(lines),encoding="utf-8")
