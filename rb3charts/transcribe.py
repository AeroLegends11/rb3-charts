"""Audio transcription backends. Confidence values are heuristic, not probabilities."""
from pathlib import Path
import numpy as np
from scipy import signal
from .audio import read_mono
from .model import Note
from .timing import refine_onset

def _refine_pitched_articulations(events, model_output, y, sr, low, high, source_label):
    """Recover repeated attacks and set releases from pitch-band energy.

    A release is the first 58ms run below both -24dB from the local peak and
    3dB above the preceding noise floor. Strong model onsets can split a long
    activation or restore a same-pitch echo omitted by frame thresholding.
    """
    import librosa
    from basic_pitch.note_creation import model_frames_to_time, MIDI_OFFSET
    if not events:return events
    times=model_frames_to_time(model_output["onset"].shape[0])
    additions=[]
    by_pitch={pitch:sorted([e for e in events if e["pitch"]==pitch],key=lambda e:e["start"]) for pitch in range(low,high+1)}
    for pitch,rows in by_pitch.items():
        if not rows:continue
        activation=model_output["onset"][:,pitch-MIDI_OFFSET]
        peaks,properties=signal.find_peaks(activation,distance=7,prominence=0.12)
        for index,frame in enumerate(peaks):
            t=float(times[frame]); strength=float(activation[frame]); prominence=float(properties["prominences"][index])
            if any(abs(e["start"]-t)<0.09 for e in rows+additions):continue
            containing=next((e for e in rows if e["start"]+0.10<t<e["end"]-0.04),None)
            near_same=min((abs(e["start"]-t) for e in rows),default=99)<1.25
            if not ((containing and strength>=0.42 and prominence>=0.20) or (near_same and strength>=0.62 and prominence>=0.30)):
                continue
            additions.append({"start":t,"end":t+0.12,"pitch":pitch,"confidence":strength,"source_id":f"{source_label}-reattack-{frame}-{pitch}","source":source_label,"model_onset_strength":strength})
    all_events=sorted(events+additions,key=lambda e:(e["start"],e["pitch"]))
    spectrum=np.abs(librosa.stft(y,n_fft=1024,hop_length=256))**2
    frequencies=librosa.fft_frequencies(sr=sr,n_fft=1024)
    audio_times=librosa.frames_to_time(np.arange(spectrum.shape[1]),sr=sr,hop_length=256)
    envelopes={}
    for pitch in set(e["pitch"] for e in all_events):
        hz=float(librosa.midi_to_hz(pitch))
        # Electronic tones often drift slightly in pitch and carry more decay
        # in their harmonics than at the fundamental.
        width=1.25
        mask=(frequencies>=hz*2**(-width/12))&(frequencies<=hz*2**(width/12))
        harmonic=(frequencies>=2*hz*2**(-width/12))&(frequencies<=2*hz*2**(width/12))
        third=(frequencies>=3*hz*2**(-width/12))&(frequencies<=3*hz*2**(width/12))
        power=np.sum(spectrum[mask],axis=0)
        if np.any(harmonic):power+=0.35*np.sum(spectrum[harmonic],axis=0)
        if np.any(third):power+=0.18*np.sum(spectrum[third],axis=0)
        envelopes[pitch]=10*np.log10(np.convolve(power,np.ones(3)/3,mode="same")+1e-12)
    for pitch in set(e["pitch"] for e in all_events):
        rows=sorted([e for e in all_events if e["pitch"]==pitch],key=lambda e:e["start"])
        db=envelopes[pitch]
        for i,event in enumerate(rows):
            next_start=rows[i+1]["start"] if i+1<len(rows) else event["start"]+3.0
            a=int(np.searchsorted(audio_times,event["start"])); peak_end=int(np.searchsorted(audio_times,event["start"]+0.10))
            limit=int(np.searchsorted(audio_times,min(event["start"]+3.0,next_start-0.015)))
            if peak_end<=a or limit<=peak_end:continue
            peak=float(np.max(db[a:peak_end]))
            noise_a=int(np.searchsorted(audio_times,max(0,event["start"]-0.20))); noise_b=int(np.searchsorted(audio_times,max(0,event["start"]-0.05)))
            noise=float(np.median(db[noise_a:noise_b])) if noise_b>noise_a else peak-30
            release_level=max(peak-24.0,noise+3.0)
            below=db[peak_end:limit]<release_level
            release=None
            for j in range(max(0,len(below)-4)):
                if np.all(below[j:j+5]):
                    release=float(audio_times[peak_end+j]);break
            if release is None:release=min(float(event["end"]),next_start-0.015)
            event["end"]=max(event["start"]+0.065,min(release,next_start-0.015))
    return sorted(all_events,key=lambda e:(e["start"],e["pitch"]))

def pitched(path, instrument, threshold, log, source_label=None, refine_articulations=False,
            tempo=None, analysis_folder=None):
    from basic_pitch.inference import predict
    from basic_pitch import build_icassp_2022_model_path, FilenameSuffix
    log(f"Transcribing {instrument} with Basic Pitch ONNX")
    low, high = {"bass": (30, 84), "guitar": (40, 100), "keys": (36, 108)}[instrument]
    from librosa import midi_to_hz
    model_output, _, events = predict(str(path), model_or_model_path=build_icassp_2022_model_path(FilenameSuffix.onnx), onset_threshold=0.55, frame_threshold=0.3, minimum_note_length=70, minimum_frequency=float(midi_to_hz(low)), maximum_frequency=float(midi_to_hz(high)), multiple_pitch_bends=False, melodia_trick=False)
    y, sr = read_mono(path)
    result = []
    label=source_label or instrument
    from basic_pitch.note_creation import model_frames_to_time, MIDI_OFFSET
    onset_times=model_frames_to_time(model_output["onset"].shape[0])
    for index, (start, end, pitch, amplitude, _) in enumerate(events):
        model_start=float(start)
        start = refine_onset(y, sr, float(start), radius=0.035)
        a=int(np.searchsorted(onset_times,model_start-.04))
        b=int(np.searchsorted(onset_times,model_start+.04,side="right"))
        channel=int(pitch)-MIDI_OFFSET
        onset=float(np.max(model_output["onset"][a:b,channel])) if b>a else 0.0
        result.append({"start": start, "end": float(end), "pitch": int(pitch), "confidence": float(amplitude), "source_id": f"{label}-{index}", "source": label,
                       "model_start":model_start,"model_onset_strength":onset})
    if refine_articulations:
        result=_refine_pitched_articulations(result,model_output,y,sr,low,high,label)
    from .events import interpret_events
    result,report=interpret_events(result,y,sr,tempo,threshold)
    if analysis_folder is not None:
        from .model import save_json
        save_json(Path(analysis_folder)/f"{instrument}-{label}-events.json",report)
    log(f"{instrument}/{label}: accepted {report['accepted_count']} of {report['candidate_count']} pitched candidates from event evidence")
    return result


def _focus_band_guitar_candidates(events,times,midi_f0,voiced,beat_period=None):
    """Resolve ambiguous guitar candidates against a stable monophonic contour."""
    rows=sorted((dict(event) for event in events),key=lambda event:(event["start"],event["pitch"]))
    groups=[]
    for event in rows:
        if groups and event["start"]-groups[-1][0]["start"]<=.045:
            groups[-1].append(event)
        else:
            groups.append([event])
    selected=[]
    report={"candidates":len(rows),"onset_groups":len(groups),"ambiguous_groups":0,
            "groups_resolved":0,"pitch_corrections":0,"octave_anchors":0,
            "rapid_duplicates_removed":0}
    for group in groups:
        start=min(event["start"] for event in group)
        end=min(max(event["end"] for event in group),start+.115)
        mask=(times>=start+.012)&(times<=end)&voiced&np.isfinite(midi_f0)
        values=midi_f0[mask]
        guide=None
        if len(values)>=3:
            median=float(np.median(values))
            dispersion=float(np.median(np.abs(values-median)))
            if dispersion<=.75:guide=int(round(median))
        if guide is not None and selected:
            recent=[event["pitch"] for event in selected[-16:]]
            center=float(np.median(recent))
            octaves=[guide+12*offset for offset in range(-4,5) if 40<=guide+12*offset<=84]
            anchored=min(octaves,key=lambda pitch:abs(pitch-center))
            if anchored!=guide:
                guide=anchored
                report["octave_anchors"]+=1
        if len(group)>1:
            report["ambiguous_groups"]+=1
            # Preserve a strongly supported real chord. Distorted harmonics in
            # the stem usually arrive as a cluster of weaker candidates.
            strong=[event for event in group if event["confidence"]>=.58]
            if len(strong)>=2:
                selected.extend(group)
                continue
            if guide is not None:
                chosen=min(group,key=lambda event:(abs(event["pitch"]-guide),-event["confidence"]))
            else:
                chosen=max(group,key=lambda event:event["confidence"])
            chosen=dict(chosen)
            chosen["source_id"]="+".join(event["source_id"] for event in group)
            if guide is not None and chosen["pitch"]!=guide and (abs(chosen["pitch"]-guide)<=5 or abs(abs(chosen["pitch"]-guide)-12)<=1):
                chosen["pitch"]=guide
                chosen["lead_focus_corrected"]=True
                report["pitch_corrections"]+=1
            selected.append(chosen)
            report["groups_resolved"]+=1
            continue
        chosen=dict(group[0])
        if guide is not None:
            delta=abs(chosen["pitch"]-guide)
            correct=(delta<=3) or (chosen["confidence"]<.55 and delta<=5) or (chosen["confidence"]<.56 and abs(delta-12)<=1)
            if correct and delta:
                chosen["pitch"]=guide
                chosen["lead_focus_corrected"]=True
                report["pitch_corrections"]+=1
        selected.append(chosen)
    # Consolidate two estimates of the same attack after harmonic correction.
    # The limit stays below a sixteenth note at common tempos, retaining fast
    # authored strumming while removing split model activations.
    min_gap=float(np.clip((beat_period or .45)/4.8,.075,.095))
    result=[]
    for event in sorted(selected,key=lambda row:(row["start"],row["pitch"])):
        corrected_overlap=(result and (event.get("lead_focus_corrected") or result[-1].get("lead_focus_corrected")) and
                           event["start"]<result[-1]["end"]-.02 and event["start"]-result[-1]["start"]<.12)
        duplicate=(result and event["pitch"]==result[-1]["pitch"] and
                   (event["start"]-result[-1]["start"]<min_gap or corrected_overlap))
        if duplicate:
            if event["confidence"]>result[-1]["confidence"] and not corrected_overlap:
                result[-1]=event
            report["rapid_duplicates_removed"]+=1
            continue
        result.append(event)
    report["retained"]=len(result)
    return result,report


def _filter_sequential_lead_outliers(events,beat_period=None):
    """Remove unsupported pitch spikes while retaining coherent lead runs."""
    rows=sorted((dict(event) for event in events),key=lambda event:(event["start"],event["pitch"]))
    removed=set()
    neighborhood=float(np.clip((beat_period or .45)*1.2,.32,.52))
    for index,event in enumerate(rows):
        previous=rows[index-1] if index else None
        following=rows[index+1] if index+1<len(rows) else None
        rapid_spike=False
        if previous and following and event["confidence"]<.62:
            rapid_spike=(event["start"]-previous["start"]<=.24 and
                         following["start"]-event["start"]<=.24 and
                         abs(previous["pitch"]-following["pitch"])<=4 and
                         abs(event["pitch"]-previous["pitch"])>=6 and
                         abs(event["pitch"]-following["pitch"])>=6)
        if rapid_spike:removed.add(index)
    return [event for index,event in enumerate(rows) if index not in removed],{
        "sequential_outliers_removed":len(removed)}


def band_guitar_lead_events(path,events,beat_period=None):
    """Use pYIN as a monophonic guide for ambiguous Band guitar detections."""
    import librosa
    if not events:return events,{"candidates":0,"retained":0}
    y,sr=librosa.load(path,sr=22050,mono=True)
    f0,voiced,_=librosa.pyin(y,fmin=librosa.midi_to_hz(40),fmax=librosa.midi_to_hz(84),
                             sr=sr,frame_length=2048,hop_length=256,fill_na=np.nan)
    times=librosa.times_like(f0,sr=sr,hop_length=256)
    midi_f0=librosa.hz_to_midi(f0)
    focused,report=_focus_band_guitar_candidates(events,times,midi_f0,voiced,beat_period)
    focused,outlier_report=_filter_sequential_lead_outliers(focused,beat_period)
    report.update(outlier_report)
    report["retained"]=len(focused)
    return focused,report

def electronic_lead_events(source_events, beat_period=None):
    """Choose a playable lead line from synth, piano and guitar candidates.

    Basic Pitch can report the same hook in several Demucs stems. Nearby attacks
    are therefore treated as one musical decision. `other` and `piano` receive a
    small preference because lead synths commonly land there, while confidence
    still decides between clearly stronger candidates.
    """
    weights={"other":1.10,"piano":1.06,"guitar":1.0}
    events=[]
    for group in source_events:
        rows=[dict(e) for e in group]
        # Large `other` stems contain quiet pads, reverb and percussive pitch
        # guesses. Keep locally meaningful notes without imposing one loudness
        # threshold on the whole song.
        if len(rows)>20:
            confidences=np.asarray([e["confidence"] for e in rows])
            floor=max(0.40,float(np.quantile(confidences,0.25)))
            rows=[e for e in rows if e["confidence"]>=floor and
                  (e["end"]-e["start"]>=0.08 or e["confidence"]>=0.65)]
        events.extend(rows)
    # Electronic timbres often make one note appear at its fundamental and
    # octave harmonics. Consolidate recurring octave families from `other` so
    # a single hook does not jump across the fretboard as its spectrum changes.
    other=[e for e in events if e.get("source")=="other"]
    for pitch_class in range(12):
        family=[e for e in other if e["pitch"]%12==pitch_class]
        pitches=sorted(set(e["pitch"] for e in family))
        if len(family)>=8 and pitches and pitches[-1]-pitches[0]>=12:
            scores={pitch:sum(e["confidence"]*min(0.5,e["end"]-e["start"]+0.05) for e in family if e["pitch"]==pitch) for pitch in pitches}
            canonical=max(scores,key=scores.get)
            for event in family:event["pitch"]=canonical
    events.sort(key=lambda e:(e["start"],e["pitch"]))
    groups=[]
    for event in events:
        if groups and event["start"]-groups[-1][0]["start"]<=0.035:
            groups[-1].append(event)
        else:
            groups.append([event])
    selected=[]
    for group in groups:
        def score(event):
            source=event.get("source",event.get("source_id","").split("-",1)[0])
            return event["confidence"]*weights.get(source,1.0)
        best_score=max(score(e) for e in group)
        contenders=[e for e in group if score(e)>=best_score*0.92]
        # The upper voice is generally the audible hook when a synth chord and
        # melody arrive together.
        chosen=max(contenders,key=lambda e:(e["pitch"],score(e)))
        selected.append(chosen)
    # A five-button controller rarely benefits from spectral attacks faster
    # than an approximately sixteenth-note pulse. Retain slight timing jitter,
    # but reject the sub-100ms texture bursts that made dense EDM sections look
    # like stacked duplicate gems.
    min_gap=float(np.clip((beat_period or 0.45)/4.6,0.085,0.12))
    result=[]
    for event in selected:
        if result and event["start"]-result[-1]["start"]<min_gap:
            if score(event)>score(result[-1])*1.15:
                result[-1]=event
            continue
        result.append(event)
    return result

def electronic_bass_events(source_events, drum_notes=()):
    """Merge low synth/bass candidates and retrigger sustained bass on kicks."""
    events=sorted((dict(e) for group in source_events for e in group if e["pitch"]<=67),key=lambda e:(e["start"],e["pitch"]))
    merged=[]
    for event in events:
        if merged and event["start"]-merged[-1]["start"]<=0.035:
            old=merged[-1]
            # Prefer the lower fundamental unless it is much less confident.
            if event["pitch"]<old["pitch"] and event["confidence"]>=old["confidence"]*0.72:
                merged[-1]=event
            elif event["confidence"]>old["confidence"]:
                merged[-1]=event
        else:
            merged.append(event)
    additions=[]
    for kick in (n for n in drum_notes if 0 in n.lanes):
        if any(abs(e["start"]-kick.start)<=0.11 for e in merged):
            continue
        sounding=[e for e in merged if e["start"]<kick.start and (e["end"]>=kick.start-0.04 or kick.start-e["start"]<=1.0)]
        if not sounding:
            continue
        source=max(sounding,key=lambda e:e["start"])
        additions.append({"start":kick.start,"end":kick.start+0.075,"pitch":source["pitch"],"confidence":min(0.62,max(0.38,source["confidence"]*0.72)),"source_id":f"kick-bass-{kick.source_id}","source":"kick"})
    return sorted(merged+additions,key=lambda e:(e["start"],e["pitch"]))

def drums(path, threshold, log):
    """Conservative multi-band onset baseline: kick, snare, cymbal.

    Simultaneous bands remain separate events before grouping, so kick+snare
    is possible. This is NOT reliable tom/ride classification; no pro markers
    are claimed. An imported GM drum MIDI can supersede this backend.
    """
    import librosa
    log("Transcribing drums with multi-band onset detector (review required)")
    y, sr = read_mono(path)
    hop = 110
    spec = np.abs(librosa.stft(y, n_fft=1024, hop_length=hop))
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=1024)
    events = []
    for lane, low, high in [(0, 35, 150), (1, 180, 2200), (2, 5500, 10500)]:
        band = spec[(frequencies >= low) & (frequencies <= high)]
        energy = np.mean(band, axis=0)
        flux = np.maximum(0, np.diff(np.log1p(band*10), axis=1)).mean(axis=0)
        flux = np.pad(flux, (1, 0))
        if np.max(flux) < 1e-6:
            continue
        # Local adaptive threshold prevents loud choruses setting verse thresholds.
        baseline = signal.medfilt(flux, kernel_size=101)
        scale = max(float(np.quantile(flux, 0.98)), 1e-7)
        peaks, _ = signal.find_peaks(flux, height=np.maximum(baseline*2.8, scale*0.16), distance=max(1, round(0.075*sr/hop)), prominence=scale*0.09)
        for i in peaks:
            confidence = float(np.clip(flux[i]/scale, 0, 1))
            if confidence < threshold or energy[i] < np.max(energy)*0.025:
                continue
            t = refine_onset(y, sr, i*hop/sr, radius=0.018)
            events.append(Note(t, t+0.065, (lane,), confidence, f"drums-{lane}-{i}"))
    # Group attacks within 20ms without spreading one hit into several kicks.
    events.sort(key=lambda n: n.start)
    grouped = []
    for n in events:
        if grouped and n.start-grouped[-1].start < 0.020:
            old = grouped[-1]
            old.lanes = tuple(sorted(set(old.lanes+n.lanes)))
            old.confidence = max(old.confidence, n.confidence)
            old.source_id += "+"+n.source_id
        else:
            grouped.append(n)
    return grouped

def import_gm_drums(path):
    """Use a user-corrected or externally transcribed General MIDI drum part."""
    import pretty_midi
    midi = pretty_midi.PrettyMIDI(str(path))
    mapping = {35:0, 36:0, 37:1, 38:1, 40:1, 42:2, 44:2, 46:2, 49:4, 57:4, 51:3, 59:3, 53:3, 41:4, 43:4, 45:3, 47:3, 48:2, 50:2}
    tom_pitches = {41,43,45,47,48,50}
    notes = []
    for part in midi.instruments:
        for n in part.notes:
            if n.pitch in mapping:
                lane = mapping[n.pitch]
                notes.append(Note(n.start, n.start+0.065, (lane,), 1.0, f"gm-{len(notes)}", toms=(lane,) if n.pitch in tom_pitches else ()))
    return sorted(notes, key=lambda n:n.start)

def neural_drums(path, threshold, device, log, analysis_folder):
    """ADTOF's trained five-class model, with bounded inference windows.

    Context is retained around windows, then cropped so no seam duplicates
    are introduced. Pad labels are never taken from uninitialized weights.
    """
    import torch
    from adtof_pytorch import ADTOFFrameRNN, calculate_n_bins, get_default_weights_path, load_audio_for_model, PeakPicker, LABELS_5, FRAME_RNN_THRESHOLDS
    weights_path=Path(get_default_weights_path())
    if not weights_path.is_file():
        raise FileNotFoundError("ADTOF trained weights missing. Rerun Setup.ps1.")
    if device=="auto":
        device="cuda" if torch.cuda.is_available() else "cpu"
    log(f"Transcribing drums with trained ADTOF model on {device}")
    model=ADTOFFrameRNN(n_bins=calculate_n_bins())
    weights=torch.load(weights_path,map_location="cpu",weights_only=True)
    model.load_state_dict(weights.get("model_weights",weights),strict=True)
    model.eval().to(device)
    x=load_audio_for_model(str(path))
    predictions=[]
    window,context=3000,200
    with torch.inference_mode():
        for start in range(0,x.shape[1],window):
            end=min(x.shape[1],start+window)
            left=max(0,start-context)
            right=min(x.shape[1],end+context)
            pred=model(x[:,left:right].to(device)).cpu().numpy()[0]
            predictions.append(pred[start-left:end-left])
            log(f"Drum analysis: {end/100:.0f}s / {x.shape[1]/100:.0f}s")
    pred=np.concatenate(predictions)
    np.savez_compressed(analysis_folder/"drum-activations.npz",activations=pred,fps=100)
    # Retain published class-specific thresholds, scaled by user conservatism.
    thresholds=[t*max(0.7,threshold/0.35) for t in FRAME_RNN_THRESHOLDS]
    peaks=PeakPicker(thresholds=thresholds,fps=100).pick(pred,labels=LABELS_5)[0]
    y,sr=read_mono(path)
    lane_map={35:0,38:1,47:3,42:2,49:4}
    notes=[]
    for label,times in peaks.items():
        klass=LABELS_5.index(label)
        for t in times:
            confidence=float(pred[min(len(pred)-1,round(t*100)),klass])
            if confidence<threshold:
                continue
            a,b=max(0,int((t-0.02)*sr)),min(len(y),int((t+0.06)*sr))
            if b<=a or float(np.sqrt(np.mean(y[a:b]**2)))<0.0015:
                continue
            onset=refine_onset(y,sr,t,radius=0.015)
            lane=lane_map[label]
            notes.append(Note(onset,onset+0.065,(lane,),confidence,f"adtof-{label}-{round(t*100)}",toms=(lane,) if label==47 else ()))
    del model
    if device=="cuda":torch.cuda.empty_cache()
    return sorted(notes,key=lambda n:n.start)
