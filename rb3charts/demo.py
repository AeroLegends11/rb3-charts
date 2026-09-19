"""Original synthetic audio with known note times, no copyrighted test assets."""
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfilt
import soundfile as sf
from .model import save_json

def make_demo(folder):
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    stems=folder/"stems"
    stems.mkdir(exist_ok=True)
    sr=44100
    duration=34.0
    rng=np.random.default_rng(7)
    bass=np.zeros(round(duration*sr),dtype="float32")
    guitar=bass.copy()
    drums=bass.copy()
    expected=[]
    for i,t in enumerate(np.arange(1,32,0.5)):
        for buffer,pitch,length,amp in [(bass,[40,40,43,45,40,47,45,43][i%8],0.35,0.19),(guitar,[52,55,59,57,52,55,59,64][i%8],0.27,0.15)]:
            x=np.arange(round(length*sr))/sr
            f=440*2**((pitch-69)/12)
            tone=np.sin(2*np.pi*f*x)+0.28*np.sin(4*np.pi*f*x)
            tone*=np.minimum(1,x/0.005)*np.exp(-x*5)*amp
            start=round(t*sr)
            buffer[start:start+len(tone)]+=tone.astype("float32")
        x=np.arange(round(0.14*sr))/sr
        if i%2==0:
            pulse=np.sin(2*np.pi*(55*x+4*(1-np.exp(-x*25))))*np.exp(-x*30)*0.5
            kind="kick"
        else:
            pulse=sosfilt(butter(3,[500,3500],fs=sr,btype="bandpass",output="sos"),rng.normal(size=len(x)))*np.exp(-x*30)*0.3
            kind="snare"
        start=round(t*sr)
        drums[start:start+len(pulse)]+=pulse.astype("float32")
        expected.append({"time":float(t),"kind":kind})
        for htime in (t,t+0.25):
            x=np.arange(round(0.06*sr))/sr
            noise=sosfilt(butter(3,6500,fs=sr,btype="highpass",output="sos"),rng.normal(size=len(x)))*np.exp(-x*80)*0.10
            start=round(htime*sr)
            drums[start:start+len(noise)]+=noise.astype("float32")
    for name,buffer in (("guitar",guitar),("bass",bass),("drums",drums)):
        sf.write(stems/f"{name}.wav",np.column_stack([buffer,buffer]),sr,subtype="PCM_24")
    mix=bass+guitar+drums
    sf.write(folder/"Precision Demo.wav",np.column_stack([mix,mix]),sr,subtype="PCM_24")
    save_json(folder/"ground-truth.json",expected)
    return folder/"Precision Demo.wav",stems
