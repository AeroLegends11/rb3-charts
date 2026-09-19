import json
import numpy as np
from .model import TempoMap

def repair_beat_gaps(beats, duration):
    """Fill isolated tracking gaps using neighboring tempo, not a global BPM.

    Sparse percussion can cause the tracker to skip several beats. Preserve the
    detected anchor times and subdivide only isolated long intervals. Discard
    sub-200ms double detections to keep the map within Magma's 300 BPM limit.
    """
    beats=np.asarray(beats,dtype=float)
    beats=beats[beats<duration]
    intervals=np.diff(beats)
    repaired=[float(beats[0])]
    inserted=discarded=0
    for i,end in enumerate(beats[1:]):
        interval=float(end-repaired[-1])
        if interval<0.2:
            discarded+=1
            continue
        neighborhood=intervals[max(0,i-8):min(len(intervals),i+9)]
        usual=float(np.median(neighborhood))
        count=max(1,round(interval/usual)) if interval>usual*1.8 else 1
        count=max(count,int(np.ceil(interval/1.5)))
        start=repaired[-1]
        repaired.extend(start+interval*j/count for j in range(1,count+1))
        inserted+=count-1
    return repaired,inserted,discarded

def _grid_phase(onset, sr, hop, bpm):
    period=60.0/bpm
    times=np.arange(len(onset))*hop/sr
    candidates=np.arange(0,period,0.002)
    scores=[]
    for phase in candidates:
        grid=np.arange(phase,times[-1],period)
        indices=np.clip(np.searchsorted(times,grid),0,len(onset)-1)
        scores.append(float(np.mean(onset[indices])))
    return float(candidates[int(np.argmax(scores))])

def _refine_studio_grid(onset, sr, hop, initial_bpm):
    """Resolve the coarse tempo-bin estimate against the entire recording."""
    times=np.arange(len(onset))*hop/sr
    kernel=np.exp(-0.5*(np.arange(-3,4)/1.5)**2)
    smooth=np.convolve(onset,kernel/kernel.sum(),mode="same")
    def evaluate(bpm_values,phase_step):
        best=(-np.inf,float(initial_bpm),0.0)
        for bpm in bpm_values:
            period=60.0/bpm
            for phase in np.arange(0,period,phase_step):
                values=np.interp(np.arange(phase,times[-1],period),times,smooth)
                if not len(values):continue
                # Ignore the quietest third of grid points so breakdowns do not
                # outweigh alignment across active sections.
                score=float(np.mean(np.sort(values)[int(len(values)*0.35):]))
                if score>best[0]:best=(score,float(bpm),float(phase))
        return best
    span=max(2.0,float(initial_bpm)*0.025)
    coarse=evaluate(np.arange(initial_bpm-span,initial_bpm+span+0.01,0.02),0.01)
    fine=evaluate(np.arange(coarse[1]-0.04,coarse[1]+0.0401,0.002),0.002)
    return fine[1],fine[2]

def _sustained_tempo_change(dynamic, onset, frame_seconds):
    """Detect supported tempo drift; isolated half/double-time guesses don't count.

    Require three active local windows on a continuous tempo trajectory. This
    intentionally does not claim to resolve abrupt tempo or meter changes.
    """
    width=max(1,round(8/frame_seconds))
    active_floor=max(1e-8,float(np.quantile(onset,.95))*.08)
    run=[]
    for start in range(0,len(dynamic),width):
        values=dynamic[start:start+width]
        energy=onset[start:start+width]
        if len(values)<width/2 or float(np.mean(energy))<active_floor:
            run=[]
            continue
        bpm=float(np.median(values))
        if not np.isfinite(bpm) or bpm<=0:
            run=[]
            continue
        if run and abs(np.log(bpm/run[-1]))>.18:
            run=[]
        run.append(bpm)
        if len(run)>=3 and max(run)/min(run)>1.20:
            return True
    return False


def analyze_tempo(y, sr, duration, bpm=None, first_beat=0.0, meter=4, beats_file=None, bpm_hint=None, prefer_constant=False):
    import librosa
    if beats_file:
        data = json.loads(beats_file.read_text(encoding="utf-8"))
        return TempoMap(data["beats"], data.get("numerator", meter), 1.0, ["Using supplied beat map; synchronization still requires listening."])
    if bpm:
        if not 30 <= bpm <= 300:
            raise ValueError("BPM must be between 30 and 300")
        return TempoMap(np.arange(first_beat, duration+60/bpm, 60/bpm).tolist(), meter, 1.0, ["Using manual constant BPM; unsuitable for drifting performances."])
    hop = 220  # 10 ms at 22.05kHz, finer than the common 512-sample grid.
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    if np.max(onset) < 1e-5:
        raise ValueError("No measurable rhythmic content. Supply a BPM or beat map.")
    aggregate=float(np.asarray(librosa.feature.tempo(onset_envelope=onset,sr=sr,hop_length=hop,aggregate=np.median)).flat[0])
    dynamic=librosa.feature.tempo(onset_envelope=onset,sr=sr,hop_length=hop,aggregate=None,std_bpm=2.0)
    changing=_sustained_tempo_change(dynamic,onset,hop/sr)
    if bpm_hint and not changing:
        candidates=[float(bpm_hint)*factor for factor in (0.5,1,2,4) if 30<=float(bpm_hint)*factor<=300]
        chosen=min(candidates,key=lambda value:abs(np.log(value/aggregate)))
        if abs(np.log(chosen/aggregate))<0.10:
            phase=_grid_phase(onset,sr,hop,chosen)
            beats=np.arange(phase,duration+60/chosen,60/chosen).tolist()
            return TempoMap(beats,meter,0.9,[f"Using embedded BPM hint {bpm_hint:g}, interpreted as {chosen:g} BPM from audio evidence.",f"Quarter-note grid phase estimated at {phase:.3f}s; confirm the measure/downbeat phase by listening."])
    if prefer_constant and not changing:
        chosen=aggregate
        while chosen<75:chosen*=2
        while chosen>190:chosen/=2
        chosen,phase=_refine_studio_grid(onset,sr,hop,chosen)
        beats=np.arange(phase,duration+60/chosen,60/chosen).tolist()
        return TempoMap(beats,meter,0.82,[f"Electronic profile inferred a stable studio grid at {chosen:.3f} BPM.",f"Quarter-note grid phase estimated at {phase:.3f}s; confirm the measure/downbeat phase by listening."])
    _, frames = librosa.beat.beat_track(onset_envelope=onset, sr=sr, hop_length=hop, bpm=dynamic, trim=False)
    beats = librosa.frames_to_time(frames, sr=sr, hop_length=hop)
    if len(beats) < 4:
        raise ValueError("Could not establish enough beats. Supply BPM and first-beat time.")
    beats,inserted,discarded=repair_beat_gaps(beats,duration)
    if changing:
        # Use acoustic attacks to remove the onset-envelope's small analysis
        # delay. No external chart or song-specific tempo curve is consulted.
        refined=[refine_onset(y,sr,beat,radius=.03) for beat in beats]
        if np.all(np.diff(refined)>0):beats=refined
    intervals = np.diff(beats)
    variability = float(np.std(intervals) / np.mean(intervals))
    confidence = float(np.clip(1-variability*3, 0, 0.85))
    warnings = [f"Meter assumed {meter}/4; downbeat phase and half/double tempo need listening review.", "Tempo confidence measures consistency, not musical correctness."]
    if changing:
        warnings.append("Sustained tempo change detected in active audio; used a variable beat map instead of a constant studio grid or BPM tag.")
    if inserted or discarded:
        warnings.append(f"Beat tracking repair: inserted {inserted} beats inside tracking gaps; discarded {discarded} double detections. Review these inferred beats.")
    if variability > 0.12:
        warnings.append("Unstable beat intervals: inspect beat map before compilation.")
    return TempoMap(list(beats), meter, confidence, warnings)

def refine_onset(y, sr, start, radius=0.04):
    """Find local attack rise without snapping notes to a beat grid."""
    a, b = max(0, int((start-radius)*sr)), min(len(y), int((start+radius)*sr))
    if b-a < 16:
        return max(0.0, start)
    window = max(3, round(sr*0.002))
    energy = np.convolve(np.abs(y[a:b]), np.ones(window)/window, mode="same")
    rise = np.maximum(0, energy - np.roll(energy, window))
    rise[:window] = 0
    if float(rise.max()) < 1e-5:
        return max(0.0, start)
    return (a+int(np.argmax(rise)))/sr
