"""Offline scrolling chart preview, plus an audible onset-check WAV."""
import html
import json
import os
from pathlib import Path
import numpy as np
import soundfile as sf


def _review_diagnostics(
    charts,lead,profile,patterns,structure_diagnostics
):
    diagnostics = {}
    for instrument, parts in charts.items():
        notes = sorted(parts["expert"], key=lambda note: note.start)
        low_confidence = sorted(
            [
                {
                    "time": note.start + lead,
                    "kind": "low confidence",
                    "detail": f"{note.confidence:.2f}",
                }
                for note in notes
                if note.confidence < 0.45
            ],
            key=lambda marker: (float(marker["detail"]), marker["time"]),
        )[:20]
        markers = sorted(low_confidence, key=lambda marker: marker["time"])
        if instrument != "drums":
            markers.extend(
                {
                    "time": note.start + lead,
                    "kind": "unusual chord",
                    "detail": "/".join(str(lane) for lane in note.lanes),
                }
                for note in notes
                if len(note.lanes) >= 3 and 0 in note.lanes and 4 in note.lanes
            )
        density_limit = (
            10
            if instrument == "drums"
            else 6
            if profile == "electronic" and instrument == "guitar"
            else 8
            if instrument == "guitar"
            else 7
        )
        dense = []
        right = 0
        for left, note in enumerate(notes):
            right = max(right, left)
            while right < len(notes) and notes[right].start < note.start + 4:
                right += 1
            if right - left > density_limit * 4:
                start = note.start + lead
                end = notes[right - 1].start + lead
                if dense and start <= dense[-1]["end"]:
                    dense[-1]["end"] = max(dense[-1]["end"], end)
                    dense[-1]["peak"] = max(dense[-1]["peak"], (right - left) / 4)
                else:
                    dense.append({"start": start, "end": end, "peak": (right - left) / 4})
        markers.extend(
            {
                "time": region["start"],
                "kind": "high density",
                "detail": f"{region['peak']:.2f}/s through {region['end']:.2f}s",
            }
            for region in dense
        )
        motif_mismatches = []
        motif_occurrences = []
        for pattern_index,pattern in enumerate(
            (patterns or {}).get(instrument, {}).get("top_patterns", [])
        ):
            motif_mismatches.extend(
                {
                    "time": float(start) + lead,
                    "kind": "motif lane mismatch",
                    "detail": f"{int(pattern.get('lane_variants', 1))} lane variants",
                }
                for start in pattern.get("lane_mismatch_times", [])
            )
            if pattern_index<4:
                motif_occurrences.extend(
                    {
                        "time":float(start)+lead,
                        "kind":"repeated motif candidate",
                        "detail":(
                            f"family {pattern_index+1}; "
                            f"{int(pattern.get('occurrences',0))} "
                            f"occurrences; instance {index+1}"
                        ),
                    }
                    for index,start in enumerate(
                        pattern.get("source_times",[])[:4]
                    )
                )
        markers.extend(sorted(motif_mismatches, key=lambda marker: marker["time"])[:20])
        markers.extend(sorted(
            motif_occurrences,key=lambda marker:marker["time"]
        )[:16])
        markers.extend({
            "time":float(boundary["seconds"])+lead,
            "kind":"activity boundary",
            "detail":f"{float(boundary['activity_change']):.2f} change",
        } for boundary in (
            structure_diagnostics or {}
        ).get("strongest_activity_boundaries",[])[:8])
        if markers:
            unique = {
                (round(marker["time"], 3), marker["kind"]): marker
                for marker in markers
            }
            diagnostics[instrument] = sorted(
                unique.values(), key=lambda marker: marker["time"]
            )
    return diagnostics


def create_preview(
    folder,
    charts,
    tempo,
    lead,
    metadata,
    voice_continuity=None,
    profile="band",
    patterns=None,
    structure_diagnostics=None,
):
    y,sr = sf.read(folder/"audio"/"song.wav",dtype="float32",always_2d=True)
    for instrument, parts in charts.items():
        clicks = np.zeros(len(y),dtype="float32")
        frequency={"guitar":880,"bass":440,"drums":1320,"keys":660}[instrument]
        t=np.arange(int(0.025*sr))/sr
        pulse=np.sin(2*np.pi*frequency*t)*np.exp(-t*150)*0.18
        for note in parts["expert"]:
            start=round((note.start+lead)*sr)
            end=min(len(clicks),start+len(pulse))
            if start>=0 and end>start:
                clicks[start:end]+=pulse[:end-start]
        audit=y*0.62+clicks[:,None]
        audit/=max(1.0,float(np.max(np.abs(audit)))/0.98)
        sf.write(folder/"audio"/f"timing-audit-{instrument}.wav",audit,sr,subtype="PCM_16")
    data={i:{d:[{"t":n.start+lead,"end":n.end+lead,"lanes":n.lanes,"c":n.confidence} for n in notes] for d,notes in diffs.items()} for i,diffs in charts.items()}
    overlays={}
    for instrument, report in (voice_continuity or {}).items():
        regions=[
            {
                "start":float(window["start_seconds"])+lead,
                "end":float(window["end_seconds"])+lead,
                "register_separation":float(window["register_separation"]),
                "dominant_evidence_share":float(window["dominant_evidence_share"]),
                "register_switches":int(window["register_switches"]),
            }
            for window in report.get("windows",[])
            if window.get("candidate_for_tracking")
        ]
        if regions:
            overlays[instrument]=regions
    payload=json.dumps({
        "charts":data,
        "beats":[beat+lead for beat in tempo.beats],
        "meter":tempo.numerator,
        "lead":lead,
        "overlays":overlays,
        "diagnostics":_review_diagnostics(
            charts,lead,profile,patterns,structure_diagnostics
        ),
        "difficulty_counts":{
            instrument:{
                difficulty:len(notes)
                for difficulty,notes in parts.items()
            }
            for instrument,parts in charts.items()
        },
    }).replace("<","\\u003c")
    title=html.escape(metadata["title"])
    artist=html.escape(metadata["artist"])
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Chart review</title>
<style>body{background:#0b111c;color:#e4ebf5;font:16px system-ui;margin:32px auto;max-width:1000px}h1{margin-bottom:4px}p{color:#a8b8cd}button,select{font:inherit;padding:8px 14px;background:#24344b;color:white;border:1px solid #526780;border-radius:8px}button{cursor:pointer}canvas{width:100%;height:auto;aspect-ratio:11/7;background:linear-gradient(#10192a,#080d16);border:1px solid #2b3d57;border-radius:12px;margin:18px 0;box-shadow:0 18px 50px #0008}audio{width:100%;margin-top:12px}.badge{color:#75dac5}.controls{display:flex;gap:9px;align-items:center;flex-wrap:wrap}label{margin-right:12px}#diagnostics{display:flex;gap:6px;flex-wrap:wrap;margin:12px 0}#diagnostics button{font-size:13px;padding:5px 8px;background:#3d2d46;border-color:#8c668f}</style>
<h1>__TITLE__</h1><p>__ARTIST__ · Draft chart review</p><p class="badge">Original timing preserved · Expert → Hard → Medium → Easy</p>
<div class="controls"><label>Instrument <select id="inst"></select></label><label>Difficulty <select id="diff"><option>expert</option><option>hard</option><option>medium</option><option>easy</option></select></label><label>Audio <select id="mode"><option value="song">Song</option><option value="audit">Song + selected-instrument Expert clicks</option></select></label><label>Highway spacing <select id="zoom"><option value="2.6">Spread out</option><option value="3.5">Normal</option><option value="4.5">Far view</option></select></label><button id="back">−5 seconds</button><button id="toggle">Play / Pause</button><label>Speed <select id="speed"><option value="1">1×</option><option value=".75">0.75×</option><option value=".5">0.5×</option></select></label></div>
<canvas id="chart" width="1100" height="700"></canvas><audio id="audio" controls src="audio/song.wav"></audio>
<p>Notes travel down the highway and cross the white strike line when played. Bright beat lines show measure starts. Amber regions mark conflicting guitar-register evidence for focused listening; they do not change the chart. Review markers jump to low-confidence notes, unusual chords, dense regions, exact-pitch motif lane mismatches, and strong neutral activity boundaries. Switch to click audio to hear Expert onsets, slow playback for dense passages, and scrub or jump back to repeat a section.</p><p id="difficultySummary" class="badge"></p><div id="diagnostics"></div><p id="stats"></p>
<script>const payload=__DATA__,data=payload.charts,beats=payload.beats,meter=payload.meter,lead=payload.lead,overlays=payload.overlays||{},difficultyCounts=payload.difficulty_counts||{};const inst=document.getElementById('inst'),diff=document.getElementById('diff'),audio=document.getElementById('audio'),canvas=document.getElementById('chart'),ctx=canvas.getContext('2d'),mode=document.getElementById('mode'),speed=document.getElementById('speed'),zoom=document.getElementById('zoom');Object.keys(data).forEach(i=>{const o=document.createElement('option');o.textContent=i;inst.append(o)});function setAudio(){const t=audio.currentTime,playing=!audio.paused;audio.src='audio/'+(mode.value==='audit'?'timing-audit-'+inst.value+'.wav':'song.wav');audio.onloadedmetadata=()=>{audio.currentTime=t;audio.playbackRate=+speed.value;if(playing)audio.play()}}mode.onchange=setAudio;inst.onchange=()=>{if(mode.value==='audit')setAudio()};speed.onchange=()=>audio.playbackRate=+speed.value;zoom.onchange=()=>lookAhead=+zoom.value;document.getElementById('back').onclick=()=>audio.currentTime=Math.max(0,audio.currentTime-5);document.getElementById('toggle').onclick=()=>audio.paused?audio.play():audio.pause();document.addEventListener('keydown',e=>{if(e.target.tagName==='SELECT')return;if(e.code==='Space'){e.preventDefault();audio.paused?audio.play():audio.pause()}if(e.code==='ArrowLeft')audio.currentTime=Math.max(0,audio.currentTime-5);if(e.code==='ArrowRight')audio.currentTime=Math.min(audio.duration||Infinity,audio.currentTime+5)});
const diagnostics=payload.diagnostics||{},diagnosticBox=document.getElementById('diagnostics'),difficultySummary=document.getElementById('difficultySummary');function showDiagnostics(){diagnosticBox.replaceChildren();for(const marker of diagnostics[inst.value]||[]){const button=document.createElement('button');button.textContent=marker.time.toFixed(2)+'s · '+marker.kind+' · '+marker.detail;button.onclick=()=>audio.currentTime=marker.time;diagnosticBox.append(button)}const counts=difficultyCounts[inst.value]||{};difficultySummary.textContent='Difficulty attacks · Expert '+(counts.expert||0)+' · Hard '+(counts.hard||0)+' · Medium '+(counts.medium||0)+' · Easy '+(counts.easy||0)}inst.addEventListener('change',showDiagnostics);showDiagnostics();
const colors=['#65df7b','#ff5864','#f8d45d','#4c9dff','#ff973d'],W=1100,H=700,horizon=48,strike=590,topWidth=300,bottomWidth=820;let lookAhead=+zoom.value;function geom(y){const p=Math.max(0,Math.min(1,(y-horizon)/(strike-horizon))),width=topWidth+(bottomWidth-topWidth)*p;return {left:(W-width)/2,width}}function yFor(t,time){return strike-(t-time)/lookAhead*(strike-horizon)}function laneX(lane,y){const g=geom(y);return g.left+(lane+.5)*g.width/5}function lineAt(y,color,width=1){const g=geom(y);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(g.left,y);ctx.lineTo(g.left+g.width,y);ctx.stroke()}function draw(){const time=audio.currentTime,notes=data[inst.value][diff.value],regions=overlays[inst.value]||[];ctx.clearRect(0,0,W,H);const floor=ctx.createLinearGradient(0,horizon,0,strike);floor.addColorStop(0,'#111d31');floor.addColorStop(1,'#17283d');const bottom=geom(strike),top=geom(horizon);ctx.fillStyle=floor;ctx.beginPath();ctx.moveTo(top.left,horizon);ctx.lineTo(top.left+top.width,horizon);ctx.lineTo(bottom.left+bottom.width,strike);ctx.lineTo(bottom.left,strike);ctx.closePath();ctx.fill();for(const region of regions){const y1=Math.max(horizon,Math.min(strike,yFor(region.start,time))),y2=Math.max(horizon,Math.min(strike,yFor(region.end,time)));if(yFor(region.start,time)<horizon||yFor(region.end,time)>strike)continue;const high=geom(Math.min(y1,y2)),low=geom(Math.max(y1,y2));ctx.globalAlpha=.18;ctx.fillStyle='#ffb454';ctx.beginPath();ctx.moveTo(high.left,Math.min(y1,y2));ctx.lineTo(high.left+high.width,Math.min(y1,y2));ctx.lineTo(low.left+low.width,Math.max(y1,y2));ctx.lineTo(low.left,Math.max(y1,y2));ctx.closePath();ctx.fill()}ctx.globalAlpha=1;for(let lane=0;lane<=5;lane++){ctx.strokeStyle='#52647d';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(top.left+lane*top.width/5,horizon);ctx.lineTo(bottom.left+lane*bottom.width/5,strike);ctx.stroke()}beats.forEach((beat,index)=>{const y=yFor(beat,time);if(y<horizon||y>strike+20)return;lineAt(y,index%meter===0?'#95aac5':'#344760',index%meter===0?3:1)});lineAt(strike,'#ffffff',6);ctx.shadowBlur=0;const labels=inst.value==='drums'?['Kick','Snare','Hi-hat','Blue','Green']:['Green','Red','Yellow','Blue','Orange'];ctx.font='16px system-ui';ctx.textAlign='center';for(let lane=0;lane<5;lane++){const x=laneX(lane,strike);ctx.fillStyle=colors[lane];ctx.beginPath();ctx.arc(x,strike,22,0,Math.PI*2);ctx.fill();ctx.fillStyle='#dfe9f5';ctx.fillText(labels[lane],x,642)}for(const n of notes){const rawStart=yFor(n.t,time),rawEnd=yFor(n.end,time);if(rawStart<horizon-30||rawEnd>strike+45)continue;const y=Math.min(strike,rawStart),endY=Math.max(horizon,rawEnd);for(const lane of n.lanes){if(n.end-n.t>.15&&endY<y){const x1=laneX(lane,y),x2=laneX(lane,endY),w1=5+10*(y-horizon)/(strike-horizon),w2=5+10*(endY-horizon)/(strike-horizon);ctx.globalAlpha=.25+.5*n.c;ctx.fillStyle=colors[lane];ctx.beginPath();ctx.moveTo(x1-w1,y);ctx.lineTo(x1+w1,y);ctx.lineTo(x2+w2,endY);ctx.lineTo(x2-w2,endY);ctx.closePath();ctx.fill()}const x=laneX(lane,y),size=7+13*Math.max(0,Math.min(1,(y-horizon)/(strike-horizon)));ctx.globalAlpha=.38+.62*n.c;ctx.fillStyle=colors[lane];ctx.strokeStyle='#f7fbff';ctx.lineWidth=2;if(Math.abs(n.t-time)<.07||(n.t<time&&n.end>time)){ctx.shadowColor=colors[lane];ctx.shadowBlur=28}ctx.beginPath();ctx.roundRect(x-size*1.35,y-size*.62,size*2.7,size*1.24,size*.55);ctx.fill();ctx.stroke();ctx.shadowBlur=0}}ctx.globalAlpha=1;ctx.textAlign='left';ctx.fillStyle='#b9c8db';ctx.fillText('NOTES APPROACH THIS LINE',bottom.left+8,strike+38);const sourceTime=Math.max(0,time-lead),local=notes.filter(n=>Math.abs(n.t-time)<=2).length/4,active=regions.find(region=>region.start<=time&&time<=region.end),voice=active?' · voice conflict '+active.register_separation.toFixed(1)+' semitones · '+active.register_switches+' switches':'';document.getElementById('stats').textContent=notes.length+' notes · player '+time.toFixed(2)+'s · song '+sourceTime.toFixed(2)+'s · local density '+local.toFixed(1)+'/s · '+notes.filter(n=>n.end-n.t>=.15).length+' holds'+voice+' · Space: play/pause · Arrows: ±5s';requestAnimationFrame(draw)}draw();</script></html>'''
    page=page.replace("n.end-n.t>.15","n.end-n.t>=.09").replace("n.end-n.t>=.15","n.end-n.t>=.09")
    (folder/"review.html").write_text(page.replace("__TITLE__",title).replace("__ARTIST__",artist).replace("__DATA__",payload),encoding="utf-8")


def create_comparison_preview(baseline, candidate, output):
    """Write a synchronized local A/B page for two completed draft runs."""
    baseline=Path(baseline).resolve()
    candidate=Path(candidate).resolve()
    output=Path(output).resolve()

    def load_run(folder):
        required={
            "charts":folder/"analysis"/"charts.json",
            "tempo":folder/"analysis"/"tempo.json",
            "metadata":folder/"metadata"/"song.json",
            "source":folder/"source"/"source.json",
            "audio":folder/"audio"/"song.wav",
        }
        missing=[str(path) for path in required.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Comparison run is incomplete: "+", ".join(missing)
            )
        return {
            key:(
                json.loads(path.read_text(encoding="utf-8"))
                if key!="audio" else path
            )
            for key,path in required.items()
        }

    runs={"baseline":load_run(baseline),"candidate":load_run(candidate)}
    baseline_meta=runs["baseline"]["metadata"]
    candidate_meta=runs["candidate"]["metadata"]
    identity=("song_id","title","artist")
    if any(
        baseline_meta.get(key)!=candidate_meta.get(key)
        for key in identity
    ):
        raise ValueError("A/B review requires matching song identity")
    if runs["baseline"]["source"].get("sha256") != runs["candidate"]["source"].get("sha256"):
        raise ValueError("A/B review requires the same source-audio hash")
    if runs["baseline"]["tempo"] != runs["candidate"]["tempo"]:
        raise ValueError("A/B review requires the same tempo map")
    baseline_lead=float(baseline_meta.get("lead_seconds",0))
    candidate_lead=float(candidate_meta.get("lead_seconds",0))
    if abs(baseline_lead-candidate_lead)>1e-9:
        raise ValueError("A/B review requires matching lead-in timing")

    instruments=sorted(
        set(runs["baseline"]["charts"]) & set(runs["candidate"]["charts"])
    )
    if not instruments:
        raise ValueError("A/B review runs have no shared instruments")

    def note_key(note):
        source_id=str(note.get("source_id",""))
        return (
            ("source",source_id)
            if source_id
            else (
                "position",
                round(float(note["start"]),6),
                tuple(note["lanes"]),
            )
        )

    charts={}
    counts={}
    focus={}
    for version,run in runs.items():
        charts[version]={}
        counts[version]={}
        for instrument in instruments:
            shared_difficulties=sorted(
                set(runs["baseline"]["charts"][instrument])
                & set(runs["candidate"]["charts"][instrument]),
                key=("expert","hard","medium","easy").index,
            )
            charts[version][instrument]={}
            counts[version][instrument]={}
            for difficulty in shared_difficulties:
                other=(
                    runs["candidate"]["charts"][instrument][difficulty]
                    if version=="baseline"
                    else runs["baseline"]["charts"][instrument][difficulty]
                )
                other_keys={note_key(note) for note in other}
                notes=[]
                for note in run["charts"][instrument][difficulty]:
                    notes.append({
                        "t":float(note["start"])+baseline_lead,
                        "end":float(note["end"])+baseline_lead,
                        "lanes":[int(lane) for lane in note["lanes"]],
                        "confidence":float(note.get("confidence",1)),
                        "different":note_key(note) not in other_keys,
                    })
                charts[version][instrument][difficulty]=notes
                counts[version][instrument][difficulty]=len(notes)

    for instrument in instruments:
        focus[instrument]={}
        for difficulty in charts["candidate"][instrument]:
            added=[
                note["t"]
                for note in charts["candidate"][instrument][difficulty]
                if note["different"]
            ]
            windows={}
            for timestamp in added:
                start=int(timestamp//8)*8
                windows[start]=windows.get(start,0)+1
            focus[instrument][difficulty]=[
                {"time":start,"count":count}
                for start,count in sorted(
                    windows.items(),key=lambda item:(-item[1],item[0])
                )[:8]
            ]

    output.parent.mkdir(parents=True,exist_ok=True)
    audio=os.path.relpath(
        runs["candidate"]["audio"],output.parent
    ).replace("\\","/")
    tempo=runs["candidate"]["tempo"]
    payload=json.dumps({
        "charts":charts,
        "counts":counts,
        "focus":focus,
        "beats":[float(beat)+baseline_lead for beat in tempo["beats"]],
        "meter":int(tempo.get("numerator",4)),
        "lead":baseline_lead,
        "audio":audio,
    }).replace("<","\\u003c")
    title=html.escape(str(candidate_meta["title"]))
    artist=html.escape(str(candidate_meta["artist"]))
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>A/B chart review</title>
<style>body{background:#0b111c;color:#e4ebf5;font:16px system-ui;margin:28px auto;max-width:1050px}h1{margin-bottom:4px}p{color:#a8b8cd}.badge{color:#75dac5}.controls{display:flex;gap:9px;align-items:center;flex-wrap:wrap}button,select{font:inherit;padding:8px 13px;background:#24344b;color:white;border:1px solid #526780;border-radius:8px}button{cursor:pointer}.active{background:#176c65;border-color:#75dac5}canvas{width:100%;height:auto;aspect-ratio:11/7;background:#10192a;border:1px solid #2b3d57;border-radius:12px;margin:16px 0}audio{width:100%}#focus{display:flex;gap:6px;flex-wrap:wrap;margin:12px 0}#focus button{font-size:13px;padding:5px 8px;background:#3d2d46}</style>
<h1>__TITLE__</h1><p>__ARTIST__ · synchronized A/B chart review</p>
<p class="badge">Baseline and candidate use the same source audio, tempo map, lead-in, and transport. Cyan outlines identify attacks unique to the selected version.</p>
<p><strong>Review question:</strong> does B preserve the intended rhythm more naturally than A without becoming tiring or over-dense? Switch with the A/B keys while playback continues at the same timestamp.</p>
<div class="controls"><button id="baseline">A · Baseline</button><button id="candidate">B · Candidate</button><label>Instrument <select id="inst"></select></label><label>Difficulty <select id="diff"></select></label><label>Speed <select id="speed"><option value="1">1×</option><option value=".75">0.75×</option><option value=".5">0.5×</option></select></label><button id="back">−5 seconds</button><button id="toggle">Play / Pause</button></div>
<canvas id="chart" width="1100" height="700"></canvas><audio id="audio" controls></audio>
<p id="summary" class="badge"></p><div id="focus"></div><p id="stats"></p>
<script>const payload=__DATA__,charts=payload.charts,counts=payload.counts,beats=payload.beats,meter=payload.meter;let version='candidate';const audio=document.getElementById('audio'),inst=document.getElementById('inst'),diff=document.getElementById('diff'),canvas=document.getElementById('chart'),ctx=canvas.getContext('2d'),focus=document.getElementById('focus');audio.src=payload.audio;for(const name of Object.keys(charts.baseline)){const option=document.createElement('option');option.textContent=name;inst.append(option)}if(charts.baseline.guitar)inst.value='guitar';function loadDiffs(){const selected=diff.value;diff.replaceChildren();for(const name of Object.keys(charts.baseline[inst.value])){const option=document.createElement('option');option.textContent=name;option.selected=name===(selected||'hard');diff.append(option)}update()}function selectVersion(name){version=name;document.getElementById('baseline').classList.toggle('active',name==='baseline');document.getElementById('candidate').classList.toggle('active',name==='candidate');update()}document.getElementById('baseline').onclick=()=>selectVersion('baseline');document.getElementById('candidate').onclick=()=>selectVersion('candidate');inst.onchange=loadDiffs;diff.onchange=update;document.getElementById('speed').onchange=e=>audio.playbackRate=+e.target.value;document.getElementById('back').onclick=()=>audio.currentTime=Math.max(0,audio.currentTime-5);document.getElementById('toggle').onclick=()=>audio.paused?audio.play():audio.pause();document.addEventListener('keydown',e=>{if(e.target.tagName==='SELECT')return;if(e.code==='Space'){e.preventDefault();audio.paused?audio.play():audio.pause()}if(e.key.toLowerCase()==='a')selectVersion('baseline');if(e.key.toLowerCase()==='b')selectVersion('candidate');if(e.code==='ArrowLeft')audio.currentTime=Math.max(0,audio.currentTime-5);if(e.code==='ArrowRight')audio.currentTime=Math.min(audio.duration||Infinity,audio.currentTime+5)});function update(){if(!inst.value||!diff.value)return;const a=counts.baseline[inst.value][diff.value],b=counts.candidate[inst.value][diff.value],delta=b-a;document.getElementById('summary').textContent='A '+a+' attacks · B '+b+' attacks · delta '+(delta>=0?'+':'')+delta;focus.replaceChildren();for(const marker of payload.focus[inst.value][diff.value]||[]){const button=document.createElement('button');button.textContent=marker.time.toFixed(0)+'s · '+marker.count+' added attacks';button.onclick=()=>audio.currentTime=marker.time;focus.append(button)}}const colors=['#65df7b','#ff5864','#f8d45d','#4c9dff','#ff973d'],W=1100,H=700,horizon=48,strike=590,topWidth=300,bottomWidth=820,lookAhead=3.5;function geom(y){const p=Math.max(0,Math.min(1,(y-horizon)/(strike-horizon))),width=topWidth+(bottomWidth-topWidth)*p;return {left:(W-width)/2,width}}function yFor(t,time){return strike-(t-time)/lookAhead*(strike-horizon)}function laneX(lane,y){const g=geom(y);return g.left+(lane+.5)*g.width/5}function lineAt(y,color,width=1){const g=geom(y);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(g.left,y);ctx.lineTo(g.left+g.width,y);ctx.stroke()}function draw(){const time=audio.currentTime,notes=charts[version][inst.value][diff.value];ctx.clearRect(0,0,W,H);const floor=ctx.createLinearGradient(0,horizon,0,strike);floor.addColorStop(0,'#111d31');floor.addColorStop(1,'#17283d');const bottom=geom(strike),top=geom(horizon);ctx.fillStyle=floor;ctx.beginPath();ctx.moveTo(top.left,horizon);ctx.lineTo(top.left+top.width,horizon);ctx.lineTo(bottom.left+bottom.width,strike);ctx.lineTo(bottom.left,strike);ctx.closePath();ctx.fill();for(let lane=0;lane<=5;lane++){ctx.strokeStyle='#52647d';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(top.left+lane*top.width/5,horizon);ctx.lineTo(bottom.left+lane*bottom.width/5,strike);ctx.stroke()}beats.forEach((beat,index)=>{const y=yFor(beat,time);if(y>=horizon&&y<=strike+20)lineAt(y,index%meter===0?'#95aac5':'#344760',index%meter===0?3:1)});lineAt(strike,'#fff',6);ctx.font='16px system-ui';ctx.textAlign='center';for(let lane=0;lane<5;lane++){const x=laneX(lane,strike);ctx.fillStyle=colors[lane];ctx.beginPath();ctx.arc(x,strike,22,0,Math.PI*2);ctx.fill();ctx.fillStyle='#dfe9f5';ctx.fillText((inst.value==='drums'?['Kick','Snare','Hi-hat','Blue','Green']:['Green','Red','Yellow','Blue','Orange'])[lane],x,642)}ctx.textAlign='left';ctx.fillStyle='#b9c8db';ctx.fillText('NOTES HIT THIS LINE',bottom.left+8,strike+38);for(const n of notes){const rawY=yFor(n.t,time);if(rawY<horizon-30||rawY>strike+2)continue;const y=Math.min(strike,rawY);for(const lane of n.lanes){const x=laneX(lane,y),size=7+13*Math.max(0,Math.min(1,(y-horizon)/(strike-horizon)));ctx.globalAlpha=.38+.62*n.confidence;ctx.fillStyle=colors[lane];ctx.strokeStyle=n.different?'#69fff0':'#f7fbff';ctx.lineWidth=n.different?5:2;ctx.beginPath();ctx.roundRect(x-size*1.35,y-size*.62,size*2.7,size*1.24,size*.55);ctx.fill();ctx.stroke()}}ctx.globalAlpha=1;ctx.fillStyle='#b9c8db';ctx.font='16px system-ui';ctx.fillText((version==='baseline'?'A · BASELINE':'B · CANDIDATE')+' · '+notes.length+' attacks · player '+time.toFixed(2)+'s · song '+Math.max(0,time-payload.lead).toFixed(2)+'s · A/B keys switch versions · Space plays',50,665);requestAnimationFrame(draw)}selectVersion('candidate');loadDiffs();draw();</script></html>'''
    output.write_text(
        page.replace("__TITLE__",title)
        .replace("__ARTIST__",artist)
        .replace("__DATA__",payload),
        encoding="utf-8",
    )
    return {
        "output":str(output),
        "title":candidate_meta["title"],
        "artist":candidate_meta["artist"],
        "source_sha256":runs["candidate"]["source"]["sha256"],
        "counts":counts,
    }
