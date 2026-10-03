"""Learn five-lane authoring statistics from finished Rock Band MIDI charts."""
from collections import Counter,defaultdict
from pathlib import Path
import json
import numpy as np
import mido

BASES={"easy":60,"medium":72,"hard":84,"expert":96}
TRACKS={"PART GUITAR":"guitar","PART BASS":"bass","PART DRUMS":"drums","PART KEYS":"keys"}


def read_five_lane(path):
    midi=mido.MidiFile(path)
    result={}
    for track in midi.tracks:
        instrument=TRACKS.get(track.name)
        if not instrument:continue
        absolute=0
        active={}
        rows=defaultdict(list)
        for message in track:
            absolute+=message.time
            if message.type=="note_on" and message.velocity:
                active[message.note]=absolute
            elif message.type in ("note_off","note_on") and (message.type=="note_off" or not message.velocity):
                start=active.pop(message.note,None)
                if start is not None:
                    for difficulty,base in BASES.items():
                        if base<=message.note<=base+4:
                            rows[difficulty].append((start,absolute,message.note-base))
        result[instrument]={difficulty:_group(rows[difficulty],midi.ticks_per_beat) for difficulty in BASES}
    return result


def _group(rows,ticks_per_beat):
    attacks=[]
    for start,end,lane in sorted(rows):
        if attacks and attacks[-1]["tick"]==start:
            attacks[-1]["lanes"].append(lane)
            attacks[-1]["duration_beats"]=min(attacks[-1]["duration_beats"],(end-start)/ticks_per_beat)
        else:
            attacks.append({"tick":start,"beat":start/ticks_per_beat,"lanes":[lane],"duration_beats":(end-start)/ticks_per_beat})
    return attacks


def _metrics(attacks):
    if not attacks:return None
    gaps=np.diff([row["beat"] for row in attacks])
    chords=[tuple(row["lanes"]) for row in attacks if len(row["lanes"])>1]
    singles=[row for row in attacks if len(row["lanes"])==1]
    motions=[]
    for a,b in zip(singles,singles[1:]):
        if b["beat"]-a["beat"]<=2:motions.append(abs(b["lanes"][0]-a["lanes"][0]))
    holds=[row["duration_beats"] for row in attacks if row["duration_beats"]>.26]
    peak=0;right=0
    for left,row in enumerate(attacks):
        right=max(right,left)
        while right<len(attacks) and attacks[right]["beat"]<row["beat"]+4:right+=1
        peak=max(peak,right-left)
    return {"attacks":len(attacks),"chords":len(chords),"three_note_chords":sum(len(shape)>=3 for shape in chords),
            "green_orange_chords":sum(0 in shape and 4 in shape for shape in chords),
            "holds":len(holds),"hold_beats":holds,"gaps":gaps.tolist(),"motions":motions,
            "lane_counts":dict(Counter(lane for row in attacks for lane in row["lanes"])),
            "chord_shapes":dict(Counter("".join(map(str,shape)) for shape in chords)),
            "peak_attacks_per_four_beats":peak}


def _percentiles(values):
    if not values:return {"p10":0,"median":0,"p90":0}
    return {"p10":round(float(np.quantile(values,.1)),4),"median":round(float(np.median(values)),4),"p90":round(float(np.quantile(values,.9)),4)}


def analyze_reference_corpus(folder,output=None,full_output=None):
    folder=Path(folder).resolve()
    if (
        output
        and full_output
        and Path(output).resolve()==Path(full_output).resolve()
    ):
        raise ValueError(
            "Compact profile and full report require different paths"
        )
    discovered=sorted(folder.rglob("*.mid"))+sorted(folder.rglob("*.midi"))
    # Repositories often keep RB2, RB3-update or conversion variants beside the
    # canonical chart. Treat one song folder as one sample to avoid bias.
    by_folder=defaultdict(list)
    for path in discovered:by_folder[path.parent].append(path)
    files=[]
    for paths in by_folder.values():
        canonical=next((path for path in paths if path.name.lower()=="notes.mid"),None)
        files.append(canonical or paths[0])
    files=sorted(files)
    songs=[]
    aggregate=defaultdict(list)
    for path in files:
        try:charts=read_five_lane(path)
        except Exception as exc:
            songs.append({"path":str(path),"error":str(exc)});continue
        entry={"path":str(path),"tracks":{}}
        for instrument,difficulties in charts.items():
            entry["tracks"][instrument]={}
            for difficulty,attacks in difficulties.items():
                metrics=_metrics(attacks)
                if metrics:
                    entry["tracks"][instrument][difficulty]=metrics
                    aggregate[(instrument,difficulty)].append(metrics)
        songs.append(entry)
    styles={}
    for (instrument,difficulty),items in aggregate.items():
        key=f"{instrument}.{difficulty}"
        attacks=sum(item["attacks"] for item in items)
        chords=sum(item["chords"] for item in items)
        holds=sum(item["holds"] for item in items)
        motions=[value for item in items for value in item["motions"]]
        chord_shapes=Counter()
        lanes=Counter()
        for item in items:
            chord_shapes.update(item["chord_shapes"]);lanes.update({int(k):v for k,v in item["lane_counts"].items()})
        styles[key]={"songs":len(items),"attacks":attacks,"attacks_per_song":_percentiles([item["attacks"] for item in items]),
                     "chord_rate":round(chords/max(1,attacks),4),"three_note_chord_rate":round(sum(item["three_note_chords"] for item in items)/max(1,attacks),4),
                     "green_orange_chord_rate":round(sum(item["green_orange_chords"] for item in items)/max(1,attacks),4),
                     "hold_rate":round(holds/max(1,attacks),4),"hold_beats":_percentiles([v for item in items for v in item["hold_beats"]]),
                     "song_chord_rate":_percentiles([item["chords"]/max(1,item["attacks"]) for item in items]),
                     "song_three_note_chord_rate":_percentiles([item["three_note_chords"]/max(1,item["attacks"]) for item in items]),
                     "song_hold_rate":_percentiles([item["holds"]/max(1,item["attacks"]) for item in items]),
                     "inter_attack_beats":_percentiles([v for item in items for v in item["gaps"] if v>0]),
                     "peak_attacks_per_four_beats":_percentiles([item["peak_attacks_per_four_beats"] for item in items]),
                     "adjacent_motion_rate":round(sum(value<=1 for value in motions)/max(1,len(motions)),4),
                     "wide_jump_rate":round(sum(value>=3 for value in motions)/max(1,len(motions)),4),
                     "song_adjacent_motion_rate":_percentiles([sum(value<=1 for value in item["motions"])/max(1,len(item["motions"])) for item in items]),
                     "song_wide_jump_rate":_percentiles([sum(value>=3 for value in item["motions"])/max(1,len(item["motions"])) for item in items]),
                     "lane_rates":{str(lane):round(count/max(1,sum(lanes.values())),4) for lane,count in sorted(lanes.items())},
                     "chord_shapes":dict(chord_shapes.most_common())}
    retention={}
    retention_by_difficulty={}
    for instrument in ("guitar","bass","drums","keys"):
        instrument_retention={}
        for difficulty in ("hard","medium","easy"):
            ratios=[]
            for song in songs:
                track=song.get("tracks",{}).get(instrument,{})
                if track.get("expert") and track.get(difficulty):
                    ratios.append(
                        track[difficulty]["attacks"]
                        / max(1,track["expert"]["attacks"])
                    )
            if ratios:
                instrument_retention[difficulty]=_percentiles(ratios)
        if instrument_retention:
            retention_by_difficulty[instrument]=instrument_retention
        if "hard" in instrument_retention:
            retention[instrument]=instrument_retention["hard"]
    report={"schema":2,"source_folder":str(folder),"midi_files":len(files),
            "analyzed_files":sum("error" not in song for song in songs),"failed_files":sum("error" in song for song in songs),
            "styles":styles,"expert_to_hard_retention":retention,
            "expert_to_difficulty_retention":retention_by_difficulty,
            "songs":songs}
    if output:
        Path(output).parent.mkdir(parents=True,exist_ok=True)
        compact={
            key:value for key,value in report.items()
            if key not in ("source_folder","songs")
        }
        Path(output).write_text(
            json.dumps(compact,indent=2),encoding="utf-8"
        )
    if full_output:
        Path(full_output).parent.mkdir(parents=True,exist_ok=True)
        Path(full_output).write_text(
            json.dumps(report,indent=2),encoding="utf-8"
        )
    return report
