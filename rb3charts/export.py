from pathlib import Path
import configparser
import shutil
import yaml
from .audio import run_process

ROOT = Path(__file__).resolve().parents[1]

def find_onyx():
    return next((ROOT/"tools"/"onyx").rglob("onyx.exe"), None) or (Path(shutil.which("onyx")) if shutil.which("onyx") else None)

def write_import(folder, midi, wav, metadata, instruments, difficulty_tiers=None):
    folder.mkdir(parents=True,exist_ok=True)
    shutil.copy2(midi,folder/"notes.mid")
    shutil.copy2(wav,folder/"song.wav")
    ini = configparser.ConfigParser(interpolation=None)
    tiers=difficulty_tiers or {}
    ini["song"] = {"name":metadata["title"],"artist":metadata["artist"],"album":metadata["album"],"year":metadata["year"] or "2000","genre":metadata["genre"],"charter":"RB3 Charts (automatic draft)","diff_guitar":str(tiers.get("guitar",2)) if "guitar" in instruments else "-1","diff_bass":str(tiers.get("bass",2)) if "bass" in instruments else "-1","diff_drums":str(tiers.get("drums",2)) if "drums" in instruments else "-1","diff_keys":str(tiers.get("keys",2)) if "keys" in instruments else "-1","diff_vocals":"-1","pro_drums":"True","preview_start_time":str(metadata["preview_ms"])}
    with (folder/"song.ini").open("w",encoding="utf-8") as f:
        ini.write(f)

def compile_rb3(import_folder, run_folder, song_id, onyx, log):
    if not onyx or not Path(onyx).exists():
        raise FileNotFoundError("Onyx CLI not found. Run Setup.ps1 or choose onyx.exe.")
    project = run_folder/"onyx-project"
    run_process([onyx,"import",import_folder,"--to",project],log)
    yaml_path = project/"song.yml"
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    # Use Onyx's own importer for its internal schema, changing only a verified target.
    data.setdefault("targets",{})["rb3"] = {"game":"rb3","song-id":song_id}
    yaml_path.write_text(yaml.safe_dump(data,allow_unicode=True,sort_keys=False),encoding="utf-8")
    target = run_folder/"rb3"/f"{song_id}.con"
    target.parent.mkdir(exist_ok=True)
    run_process([onyx,"build",yaml_path,"--target","rb3","--to",target],log)
    if not target.exists() or target.stat().st_size<1024:
        raise RuntimeError("Onyx did not produce a complete package")
    with target.open("rb") as f:
        if f.read(4) not in (b"CON ",b"LIVE",b"PIRS"):
            raise RuntimeError("Output is not an Xbox STFS package")
    return target
