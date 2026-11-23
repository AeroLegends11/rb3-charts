from pathlib import Path
import hashlib
import os
import subprocess
import numpy as np
import soundfile as sf
from .model import save_json

def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def run_process(args, log, cwd=None):
    log("Running " + Path(str(args[0])).name)
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with subprocess.Popen([str(a) for a in args], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", creationflags=flags) as proc:
        for line in proc.stdout:
            if line.strip():
                log(line.rstrip())
        if proc.wait() != 0:
            raise RuntimeError(f"{Path(str(args[0])).name} failed with exit code {proc.returncode}; see generation.log")

def decode(source, destination, log):
    import imageio_ffmpeg
    run_process([imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", source, "-vn", "-ar", "44100", "-ac", "2", "-c:a", "pcm_f32le", destination], log)
    info = sf.info(destination)
    if info.duration < 2:
        raise ValueError("Audio must be at least two seconds long.")
    return info.duration

def read_mono(path, sr=22050):
    import librosa
    y, native = sf.read(path, dtype="float32", always_2d=True)
    y = np.mean(y, axis=1)
    if native != sr:
        y = librosa.resample(y, orig_sr=native, target_sr=sr)
    return y, sr

def separate(wav, cache, quality, device, log):
    """Use model API and soundfile to avoid Windows torchaudio codec problems."""
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    key = fingerprint(wav)[:20] + "-" + quality + "-v1"
    folder = Path(cache) / key
    manifest = folder / "complete.json"
    if manifest.exists() and all((folder / f"{s}.wav").exists() for s in ("drums", "bass", "guitar", "piano")):
        log("Reusing cached stems")
        return {s: folder/f"{s}.wav" for s in ("drums", "bass", "guitar", "piano", "vocals", "other")}
    folder.mkdir(parents=True, exist_ok=True)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Choose CPU or install the CUDA runtime using Setup.ps1.")
    torch.manual_seed(17)
    import random
    random.seed(17)
    y, sr = sf.read(wav, dtype="float32", always_2d=True)
    mix = torch.from_numpy(y.T.copy())
    ref = mix.mean(0)
    mean, std = ref.mean(), ref.std().clamp(min=1e-8)
    normalized = (mix-mean)/std
    models = ["htdemucs_6s"] + (["htdemucs_ft"] if quality == "precision" else [])
    paths = {}
    for name in models:
        log(f"Separating {name} on {device}; first run downloads the model weights")
        model = get_model(name)
        if model.samplerate != sr:
            raise ValueError("Unexpected separation sample rate")
        model.eval()
        with torch.inference_mode():
            result = apply_model(model, normalized[None], device=device, shifts=2 if quality == "precision" else 1, split=True, overlap=0.5 if quality == "precision" else 0.25, progress=True)[0].cpu()
        result = result*std + mean
        for i, name_source in enumerate(model.sources):
            # Six-stem guitar/keys plus fine-tuned drums/bass/vocals.
            if name == "htdemucs_ft" and name_source == "other":
                continue
            target = folder / f"{name_source}.wav"
            sf.write(target, result[i].numpy().T, sr, subtype="FLOAT")
            paths[name_source] = target
        del model, result
        if device == "cuda":
            torch.cuda.empty_cache()
    save_json(manifest, {"models": models, "device": device, "source": fingerprint(wav)})
    return paths

def clean_guitar_stem(stems, log, floor_ratio=.35):
    """Write guitar-clean-<settings>.wav beside the cached guitar stem and return updated stems.

    The raw cached stem is untouched and stays available as `guitar_raw`. The
    cleaned file is rejected, keeping the raw stem, if it loses most of the energy.
    """
    from .stem_cleanup import suppress_bleed, SETTINGS_TAG
    guitar = stems.get("guitar")
    names = [name for name in ("vocals", "drums", "bass") if stems.get(name)]
    if not guitar or not names:
        return stems
    guitar = Path(guitar)
    target = guitar.with_name(f"guitar-clean-{SETTINGS_TAG}.wav")
    # The tag changes with the suppression settings; the mtime check rejects a
    # file left from a different raw stem written to the same folder.
    if not target.exists() or target.stat().st_mtime_ns < guitar.stat().st_mtime_ns:
        log("Suppressing vocal/drum/bass bleed in the guitar stem")
        raw, sr = sf.read(guitar, dtype="float32", always_2d=True)
        others = []
        for name in names:
            data, other_sr = sf.read(stems[name], dtype="float32", always_2d=True)
            if other_sr != sr or data.shape != raw.shape:
                log(f"Skipping guitar bleed cleanup: {name} stem does not match the guitar stem")
                return stems
            others.append(data.T)
        cleaned = suppress_bleed(raw.T, others).T
        rms = lambda x: float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
        if rms(cleaned) < floor_ratio * max(rms(raw), 1e-12):
            log("Guitar bleed cleanup removed too much energy; using the raw stem")
            return stems
        sf.write(target, cleaned, sr, subtype="FLOAT")
        log(f"Guitar bleed cleanup kept {rms(cleaned)/max(rms(raw),1e-12):.0%} of the raw stem's RMS; compare {target.name} with {guitar.name}")
    return {**stems, "guitar": target, "guitar_raw": guitar}


def write_padded_audio(wav, output, lead, tail=3.0):
    y, sr = sf.read(wav, dtype="float32", always_2d=True)
    padded = np.pad(y, ((round(lead*sr), round(tail*sr)), (0, 0)))
    peak = float(np.max(np.abs(padded)))
    gain = min(1.0, 0.98 / max(peak, 1e-8))
    sf.write(output, padded*gain, sr, subtype="PCM_24")
    return gain

def detect_chart_profile(stems, genre=""):
    """Choose literal-band or role-based electronic charting from stems."""
    electronic_words=("electronic","electronica","edm","dance","dubstep","house","techno","trance","synth","industrial")
    genre_text=str(genre or "").lower()
    def rms(path):
        if not path or not Path(path).is_file():return 0.0
        total=count=0
        with sf.SoundFile(path) as stream:
            for block in stream.blocks(blocksize=262144,dtype="float32",always_2d=True):
                total+=float(np.sum(block.astype(np.float64)**2))
                count+=block.size
        return float(np.sqrt(total/max(1,count)))
    energies={name:rms(stems.get(name)) for name in ("guitar","piano","other")}
    alternate=max(energies["other"],energies["piano"])
    ratio=energies["guitar"]/max(alternate,1e-8)
    genre_match=next((word for word in electronic_words if word in genre_text),None)
    if energies["guitar"]>=0.003 and ratio>1.8:
        resolved="band"
        reason="isolated guitar energy strongly exceeds synth/other"
    elif alternate>=0.003 and ratio<0.55:
        resolved="electronic"
        reason="synth/other stem energy substantially exceeds guitar"
    elif genre_match:
        resolved="electronic"
        reason=f"ambiguous stem balance resolved by genre '{genre_match}'"
    else:
        resolved="band"
        reason="ambiguous stem balance; conservative literal-instrument fallback"
    return resolved,{"resolved":resolved,"reason":reason,"genre":genre or "","stem_rms":{k:round(v,7) for k,v in energies.items()},"guitar_to_alternate_ratio":round(ratio,4)}
