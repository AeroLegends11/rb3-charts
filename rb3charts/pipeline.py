from pathlib import Path
from dataclasses import asdict
from datetime import datetime, timezone
import json
import re
import shutil
import traceback
from .model import Options, save_json
from .audio import decode, fingerprint, separate, read_mono, write_padded_audio, detect_chart_profile
from .timing import analyze_tempo
from .transcribe import analyze_band_guitar_voice, pitched, neural_drums, import_gm_drums, electronic_bass_events, band_guitar_lead_events
from .chart import ELECTRONIC_SUSTAIN_SECONDS, BAND_GUITAR_SUSTAIN_SECONDS, map_pitched, clean, reductions, overdrive, align_drum_activations, enforce_playability, analyze_patterns, conform_expert_to_reference
from .midi import write_midi
from .validate import validate, report_text
from .export import write_import, compile_rb3, find_onyx
from .preview import create_preview
from .assessment import (
    analyze_structure_diagnostics,
    assess_chart,
    write_song_review_notes,
    write_batch_review,
)
from .events import write_event_trace
from .phrases import select_lead_phrases, author_guitar_phrases
from .regression import (
    compute_regression_metrics,
    git_state,
    hash_file,
    reductions_entry,
    runtime_versions,
    stage_entry,
    summarize_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
SUPPORTED = {".mp3",".wav",".flac",".ogg",".m4a",".aiff",".aif",".wma"}


def _new_output_folder(base_output: Path, artist: str, title: str):
    slug=re.sub(r"[^a-zA-Z0-9]+","_",artist+"_"+title).strip("_")[:48] or "song"
    folder=base_output.resolve()/f"{slug}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    for part in ("source","audio","analysis","midi","metadata","logs","rb3"):
        (folder/part).mkdir(parents=True,exist_ok=True)
    return folder


def _record_stage_snapshot(store: dict, instrument: str, stage: str, notes):
    payload=store.setdefault(instrument,{"instrument":instrument,"source_join":"source_id","stages":[]})
    payload["stages"].append(stage_entry(stage,notes))


def _record_reduction_snapshot(store: dict, instrument: str, charts: dict):
    payload=store.setdefault(instrument,{"instrument":instrument,"source_join":"source_id","stages":[]})
    payload["reductions"]=reductions_entry("after_reductions",charts)


def _tempo_from_payload(payload: dict):
    from .model import TempoMap

    return TempoMap(payload["beats"],payload.get("numerator",4),payload.get("confidence",0.0),payload.get("warnings",[]))


def _notes_from_payload(rows):
    from .model import Note

    return [
        Note(
            start=float(row["start"]),
            end=float(row["end"]),
            lanes=tuple(int(lane) for lane in row["lanes"]),
            confidence=float(row.get("confidence",1.0)),
            source_id=str(row.get("source_id","")),
            pitch=(float(row["pitch"]) if row.get("pitch") is not None else None),
            toms=tuple(int(item) for item in row.get("toms",())),
        )
        for row in rows
    ]


def _load_accepted_event_groups(analysis_folder: Path, instrument: str):
    groups=[]
    for report_path in sorted(analysis_folder.glob(f"{instrument}-*-events.json")):
        payload=json.loads(report_path.read_text(encoding="utf-8"))
        events=payload.get("events",[])
        if not isinstance(events,list):
            continue
        accepted=[dict(row) for row in events if isinstance(row,dict) and row.get("accepted")]
        if accepted:
            groups.append(accepted)
    return groups


def _replay_provenance(source_folder: Path, consumed_inputs: list[Path], options_payload: dict, profile_payload: dict):
    return {
        "mode":"cached_replay",
        "replay_source_folder":str(source_folder),
        "consumed_inputs":[{"path":str(path),"sha256":hash_file(path)} for path in sorted(consumed_inputs,key=lambda row:str(row))],
        "settings":options_payload,
        "chart_profile":profile_payload,
        "git":git_state(ROOT),
        "runtime_versions":runtime_versions(),
        "timestamp_utc":datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00","Z"),
    }


def _write_package_input_manifest(folder: Path):
    folder=Path(folder).resolve()
    paths=(
        folder/"import"/"notes.mid",
        folder/"import"/"song.wav",
        folder/"import"/"song.ini",
    )
    save_json(folder/"analysis"/"package-inputs.json",{
        "schema_version":1,
        "files":[
            {
                "path":str(path.relative_to(folder)).replace("\\","/"),
                "sha256":hash_file(path),
            }
            for path in paths
        ],
    })


def _verify_package_input_manifest(folder: Path):
    folder=Path(folder).resolve()
    path=folder/"analysis"/"package-inputs.json"
    if not path.is_file():
        return False
    payload=json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version")!=1 or not payload.get("files"):
        raise RuntimeError("Package-input manifest is malformed")
    expected_paths={
        "import/notes.mid",
        "import/song.wav",
        "import/song.ini",
    }
    manifest_paths={
        str(item.get("path","")).replace("\\","/")
        for item in payload["files"]
    }
    if manifest_paths!=expected_paths or len(payload["files"])!=3:
        raise RuntimeError("Package-input manifest is incomplete")
    for item in payload["files"]:
        relative=Path(str(item.get("path","")))
        candidate=(folder/relative).resolve()
        if folder not in candidate.parents:
            raise RuntimeError("Package-input manifest path escapes the run folder")
        if not candidate.is_file():
            raise RuntimeError(f"Reviewed package input is missing: {relative}")
        if hash_file(candidate)!=item.get("sha256"):
            raise RuntimeError(
                f"Reviewed package input changed after validation: {relative}"
            )
    return True

def generate(options: Options, progress=print):
    source=options.source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Audio file not found: {source}")
    if not options.instruments or any(i not in ("guitar","bass","drums","keys") for i in options.instruments):
        raise ValueError("Select at least one supported instrument")
    if options.chart_profile not in ("auto","band","electronic"):
        raise ValueError("Chart profile must be 'auto', 'band' or 'electronic'")
    metadata={k:getattr(options,k) for k in ("title","artist","album","year","genre")}
    metadata["chart_profile"]=options.chart_profile
    bpm_hint=None
    try:
        import mutagen
        tags=mutagen.File(source,easy=True)
        if tags:
            for k in ("title","artist","album","year","genre"):
                metadata[k]=metadata[k] or (tags.get("date" if k=="year" else k,[""])[0])
            try:bpm_hint=float(tags.get("bpm",[0])[0]) or None
            except (TypeError,ValueError):bpm_hint=None
    except Exception:
        pass
    metadata["title"]=metadata["title"] or source.stem
    metadata["artist"]=metadata["artist"] or "Unknown Artist"
    metadata["genre"]=metadata["genre"] or "Other"
    digest=fingerprint(source)
    song_id="rb3c_"+digest[:12]
    folder=_new_output_folder(options.output,metadata["artist"],metadata["title"])
    logfile=folder/"logs"/"generation.log"
    def log(message):
        text=str(message)
        with logfile.open("a",encoding="utf-8") as f:
            f.write(text+"\n")
        try:
            progress(text)
        except UnicodeEncodeError:
            # Library callers may supply `print` backed by a legacy Windows
            # code page. Keep the UTF-8 log intact and degrade display only.
            progress(text.encode("ascii",errors="replace").decode("ascii"))
    save_json(folder/"source"/"source.json",{"path":str(source),"sha256":digest})
    save_json(folder/"metadata"/"options.json",asdict(options))
    try:
        log("Decoding source audio")
        wav=folder/"source"/"decoded.wav"
        duration=decode(source,wav,log)
        if options.stems:
            stems={p.stem:p for p in options.stems.resolve().glob("*.wav")}
            log("Using supplied stems (must share source timing)")
        else:
            stems=separate(wav,ROOT/"cache"/"stems",options.quality,options.device,log)
        save_json(folder/"analysis"/"stems.json",stems)
        if options.chart_profile=="auto":
            resolved_profile,profile_evidence=detect_chart_profile(stems,metadata["genre"])
            log(f"Automatic charting style: {resolved_profile} ({profile_evidence['reason']})")
        else:
            resolved_profile=options.chart_profile
            profile_evidence={"resolved":resolved_profile,"reason":"manual override","genre":metadata["genre"]}
        metadata["chart_profile_requested"]=options.chart_profile
        metadata["chart_profile"]=resolved_profile
        save_json(folder/"analysis"/"chart-profile.json",profile_evidence)
        log("Analyzing tempo and beat positions")
        y,sr=read_mono(stems.get("drums",wav))
        tempo=analyze_tempo(
            y, sr, duration, options.bpm, options.first_beat, options.meter,
            options.beats_file, bpm_hint, prefer_constant=True,
        )
        save_json(folder/"analysis"/"tempo.json",asdict(tempo))
        charts={}
        playability={}
        reference_conformance={}
        pattern_stabilization={}
        lead_focus={}
        voice_continuity={}
        phrase_authoring={}
        stage_snapshots={}
        warnings=[]
        ordered=list(options.instruments)
        if resolved_profile=="electronic" and "drums" in ordered:
            ordered.remove("drums")
            ordered.insert(0,"drums")
        for instrument in ordered:
            log(f"Generating {instrument} Expert chart")
            chord_evidence=None
            stem=stems.get("piano" if instrument=="keys" else instrument)
            if resolved_profile=="electronic" and instrument in ("guitar","bass"):
                role_sources=("other","piano","guitar") if instrument=="guitar" else ("bass","other")
                stem=next((stems.get(name) for name in role_sources if stems.get(name)),None)
            if not stem:
                warnings.append(f"{instrument}: stem unavailable; instrument omitted")
                continue
            import soundfile as sf
            if abs(sf.info(stem).duration-duration)>0.15:
                raise ValueError(f"{instrument} stem length differs from source; use stems aligned to the original audio")
            if instrument=="drums":
                gm=options.stems/"drums.mid" if options.stems else None
                if gm and gm.exists():
                    notes=clean(import_gm_drums(gm),instrument)
                else:
                    notes=clean(neural_drums(stem,options.confidence,options.device,log,folder/"analysis"),instrument)
                    warnings.append("ADTOF distinguishes kick, snare, tom, hi-hat and cymbal. Tom height and ride versus crash remain simplified; Pro Drum markers need listening review.")
                _record_stage_snapshot(
                    stage_snapshots,instrument,"after_clean_drums",notes
                )
            else:
                if resolved_profile=="electronic" and instrument in ("guitar","bass"):
                    source_names=("other","piano","guitar") if instrument=="guitar" else ("bass","other")
                    candidates=[]
                    for source_name in source_names:
                        candidate_stem=stems.get(source_name)
                        if candidate_stem:
                            log(f"Analyzing {source_name} stem for electronic {instrument} role")
                            candidates.append(pitched(candidate_stem,instrument,options.confidence,log,source_name,True,
                                                      tempo=tempo,analysis_folder=folder/"analysis"))
                    candidate_count=sum(len(group) for group in candidates)
                    if instrument=="guitar":
                        raw,lead_focus[instrument]=select_lead_phrases(candidates)
                    else:
                        raw=electronic_bass_events(candidates,charts.get("drums",{}).get("expert",[]))
                    if instrument=="guitar":
                        warnings.append(f"Electronic lead focus retained {len(raw)} of {candidate_count} pitched candidates using phrase voice continuity.")
                else:
                    raw=pitched(stem,instrument,options.confidence,log,
                                refine_articulations=(
                                    resolved_profile=="band" and instrument=="guitar"
                                ),
                                tempo=tempo,analysis_folder=folder/"analysis")
                    if resolved_profile=="band" and instrument=="guitar":
                        log("Selecting the lead-guitar contour from ambiguous pitched candidates")
                        chord_candidates=[dict(event) for event in raw]
                        raw,lead_focus[instrument]=band_guitar_lead_events(stem,raw,60/tempo.bpm)
                        chord_evidence={
                            "candidates":chord_candidates,
                            "focus":lead_focus[instrument],
                        }
                        focus=lead_focus[instrument]
                        warnings.append(
                            f"Guitar lead focus resolved {focus.get('groups_resolved',0)} ambiguous onset groups "
                            f"and removed {focus.get('sequential_outliers_removed',0)} unsupported sequential pitch spikes."
                        )
                save_json(folder/"analysis"/f"{instrument}-transcription.json",raw)
                if resolved_profile=="band" and instrument=="guitar":
                    voice_continuity[instrument]=analyze_band_guitar_voice(raw,tempo)
                    save_json(folder/"analysis"/"guitar-voice-continuity.json",
                              voice_continuity[instrument])
                notes=map_pitched(raw,instrument,
                                  "electronic_lead" if resolved_profile=="electronic" and instrument=="guitar" else "range",
                                  0.09 if resolved_profile=="electronic" and instrument!="guitar"
                                  else BAND_GUITAR_SUSTAIN_SECONDS
                                  if resolved_profile=="band" and instrument=="guitar"
                                  else 0.22)
                _record_stage_snapshot(
                    stage_snapshots,instrument,"after_map_pitched",notes
                )
            notes=[n for n in notes if n.start<duration and n.end>n.start]
            for n in notes:
                n.end=min(duration,n.end)
            if not notes:
                if instrument!="drums":
                    write_event_trace(folder/"analysis",instrument,raw,[])
                warnings.append(f"{instrument}: no confident notes; omitted")
                continue
            notes,playability[instrument]=enforce_playability(notes,instrument,tempo,resolved_profile,
                                                           phrase_aware=instrument=="guitar")
            _record_stage_snapshot(
                stage_snapshots,instrument,"after_enforce_playability",notes
            )
            if resolved_profile=="band" and instrument in ("guitar","bass","keys"):
                notes,reference_conformance[instrument]=conform_expert_to_reference(notes,tempo,instrument)
                _record_stage_snapshot(
                    stage_snapshots,
                    instrument,
                    "after_conform_expert_to_reference",
                    notes,
                )
            if instrument=="guitar":
                notes,phrase_authoring[instrument]=author_guitar_phrases(
                    notes,tempo,chord_evidence=chord_evidence,
                    sustain_threshold=(
                        ELECTRONIC_SUSTAIN_SECONDS
                        if resolved_profile=="electronic"
                        else BAND_GUITAR_SUSTAIN_SECONDS
                    ),
                )
                changed=phrase_authoring[instrument]["chord_sections"]["notes_changed"]
                if changed:
                    warnings.append(
                        f"Guitar phrase authoring inferred {changed} power-chord attacks "
                        "inside repeated, source-supported texture sections."
                    )
                _record_stage_snapshot(stage_snapshots,instrument,"after_author_guitar_phrases",notes)
            removed=playability[instrument]["removed"]
            if removed:
                warnings.append(f"{instrument}: playability gate removed {removed} crowded attacks; retained {playability[instrument]['after']} with a peak four-second density of {playability[instrument]['peak_four_second_nps']:.2f}/s.")
            log(f"Reducing {instrument}: Expert → Hard → Medium → Easy")
            charts[instrument]=reductions(notes,instrument,tempo)
            _record_reduction_snapshot(
                stage_snapshots,instrument,charts[instrument]
            )
            if instrument!="drums":
                write_event_trace(folder/"analysis",instrument,raw,charts[instrument]["expert"])
        phrases={i:overdrive(parts,tempo,duration) for i,parts in charts.items()}
        patterns={instrument:analyze_patterns(parts["expert"],tempo,instrument) for instrument,parts in charts.items()}
        for instrument,details in pattern_stabilization.items():
            patterns.setdefault(instrument,{})["stabilization"]=details
        for instrument,pattern in patterns.items():
            families=pattern["recurring_pattern_families"]
            if families:
                warnings.append(f"{instrument}: detected {families} recurring four-note pattern families and favored their attacks during playability filtering.")
        if resolved_profile=="electronic":
            warnings.append("Electronic profile maps lead synth/piano hooks to Guitar and low synth plus kick-supported pulses to Bass. Review role choices by listening.")
        corrections=align_drum_activations(charts["drums"],tempo,phrases.get("drums",[]),duration) if "drums" in charts else []
        structure_diagnostics=analyze_structure_diagnostics(
            charts,tempo,duration
        )
        save_json(folder/"analysis"/"timing-adjustments.json",corrections)
        save_json(
            folder/"analysis"/"structure-diagnostics.json",
            structure_diagnostics,
        )
        save_json(folder/"analysis"/"playability.json",playability)
        save_json(folder/"analysis"/"patterns.json",patterns)
        save_json(folder/"analysis"/"reference-conformance.json",reference_conformance)
        save_json(folder/"analysis"/"pattern-stabilization.json",pattern_stabilization)
        save_json(folder/"analysis"/"lead-focus.json",lead_focus)
        save_json(folder/"analysis"/"phrase-authoring.json",phrase_authoring)
        save_json(folder/"analysis"/"charts.json",charts)
        save_json(folder/"analysis"/"overdrive.json",phrases)
        for instrument,payload in stage_snapshots.items():
            save_json(folder/"analysis"/f"{instrument}-stage-snapshots.json",payload)
        log("Writing tempo-aware Rock Band MIDI")
        midi=folder/"midi"/"notes.mid"
        lead=write_midi(midi,tempo,charts,duration,phrases)
        gain=write_padded_audio(wav,folder/"audio"/"song.wav",lead)
        metadata.update({"song_id":song_id,"source_seconds":duration,"lead_seconds":lead,"gain":gain,
                         "preview_ms":round((lead+min(30,duration*0.25))*1000),
                         "analysis_mode":"complete_audio_reanalysis"})
        save_json(folder/"metadata"/"song.json",metadata)
        report=validate(charts,tempo,duration,midi)
        report["warnings"].extend(warnings)
        (folder/"validation_report.txt").write_text(report_text(report),encoding="utf-8")
        save_json(folder/"validation_report.json",report)
        if not report["passed"]:
            raise RuntimeError("Chart validation failed. See validation_report.txt")
        log("Creating audible timing check and chart review")
        create_preview(
            folder,
            charts,
            tempo,
            lead,
            metadata,
            voice_continuity,
            resolved_profile,
            patterns,
            structure_diagnostics,
        )
        missing=[i for i in options.instruments if i not in charts]
        assessment=assess_chart(
            charts,
            patterns,
            playability,
            resolved_profile,
            missing,
            structure_diagnostics,
        )
        difficulty_tiers={name:item["difficulty_tier"] for name,item in assessment["instruments"].items() if "difficulty_tier" in item}
        write_import(folder/"import",midi,folder/"audio"/"song.wav",metadata,charts,difficulty_tiers)
        _write_package_input_manifest(folder)
        save_json(folder/"analysis"/"quality-assessment.json",assessment)
        write_song_review_notes(folder/"review_notes.md",metadata["title"],metadata["artist"],assessment)
        result={"status":"draft_ready","folder":str(folder),"review":str(folder/"review.html"),"review_notes":str(folder/"review_notes.md"),"title":metadata["title"],"artist":metadata["artist"],"chart_profile_requested":options.chart_profile,"chart_profile":resolved_profile,"analysis_mode":"complete_audio_reanalysis","assessment":assessment,"warnings":report["warnings"],"counts":report["counts"],"missing_instruments":missing,"package":None,"xbox_tested":False}
        if options.compile:
            log("Compiling with Onyx and its Magma toolchain")
            try:
                result["package"]=str(compile_rb3(folder/"import",folder,song_id,options.onyx or find_onyx(),log))
                result["status"]="compiled_needs_playtest"
            except Exception as exc:
                log(f"Compilation failed: {exc}")
                result["status"]="compile_failed"
                result["warnings"].append(str(exc))
        save_json(folder/"result.json",result)
        log(f"Finished: {result['status']} — {folder}")
        return result
    except Exception:
        log(traceback.format_exc())
        save_json(folder/"result.json",{"status":"failed","folder":str(folder)})
        raise

def package_existing(folder, progress=print, onyx=None):
    """Compile an already generated and reviewed run without re-analysis."""
    folder=Path(folder).resolve()
    result_path=folder/"result.json"
    metadata_path=folder/"metadata"/"song.json"
    validation_path=folder/"validation_report.json"
    required=(result_path,metadata_path,validation_path,folder/"import"/"notes.mid",folder/"import"/"song.wav",folder/"import"/"song.ini")
    missing=[str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Reviewed chart is incomplete: "+", ".join(missing))
    result=json.loads(result_path.read_text(encoding="utf-8"))
    report=json.loads(validation_path.read_text(encoding="utf-8"))
    if not report.get("passed"):
        raise RuntimeError("Chart validation did not pass; packaging was stopped")
    metadata=json.loads(metadata_path.read_text(encoding="utf-8"))
    logfile=folder/"logs"/"generation.log"
    def log(message):
        text=str(message)
        with logfile.open("a",encoding="utf-8") as stream:stream.write(text+"\n")
        try:progress(text)
        except UnicodeEncodeError:progress(text.encode("ascii",errors="replace").decode("ascii"))
    recheck_inputs=(
        folder/"analysis"/"charts.json",
        folder/"analysis"/"tempo.json",
    )
    current_recheck=all(path.is_file() for path in recheck_inputs)
    if current_recheck:
        recheck=validate_existing_run(folder)
        if not recheck["passed"]:
            raise RuntimeError(
                "Current chart revalidation failed; packaging was stopped"
            )
        log("Current structural and package-input revalidation passed")
        result["validation_recheck"]="passed"
        result["validation_recheck_report"]=str(
            folder/"validation_recheck_report.json"
        )
    elif not _verify_package_input_manifest(folder):
        warning=(
            "Legacy run has no package-input manifest; package inputs could "
            "not be checked for changes after validation."
        )
        result.setdefault("warnings",[]).append(warning)
        log(warning)
        result["validation_recheck"]="legacy_unavailable"
        result["validation_recheck_report"]=None
    else:
        result["validation_recheck"]="legacy_manifest_only"
        result["validation_recheck_report"]=None
    log("Packaging reviewed chart with Onyx and Magma")
    try:
        package=compile_rb3(folder/"import",folder,metadata["song_id"],onyx or find_onyx(),log)
        result.update({"status":"compiled_needs_playtest","package":str(package),"xbox_tested":False})
        log(f"Reviewed chart package created: {package}")
    except Exception as exc:
        result["status"]="compile_failed"
        result.setdefault("warnings",[]).append(str(exc))
        log(f"Packaging failed: {exc}")
    save_json(result_path,result)
    return result


def validate_existing_run(folder):
    """Recheck an existing draft without changing its authored artifacts."""
    folder=Path(folder).resolve()
    charts_path=folder/"analysis"/"charts.json"
    tempo_path=folder/"analysis"/"tempo.json"
    metadata_path=folder/"metadata"/"song.json"
    midi_path=folder/"import"/"notes.mid"
    required=(charts_path,tempo_path,metadata_path,midi_path)
    missing=[str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Run validation inputs are incomplete: "+", ".join(missing)
        )
    chart_payload=json.loads(charts_path.read_text(encoding="utf-8"))
    charts={
        instrument:{
            difficulty:_notes_from_payload(notes)
            for difficulty,notes in difficulties.items()
        }
        for instrument,difficulties in chart_payload.items()
    }
    tempo=_tempo_from_payload(
        json.loads(tempo_path.read_text(encoding="utf-8"))
    )
    metadata=json.loads(metadata_path.read_text(encoding="utf-8"))
    duration=float(metadata.get("source_seconds",0))
    if duration<=0:
        raise ValueError(
            "Run metadata requires a positive source_seconds value"
        )
    report=validate(charts,tempo,duration,midi_path)
    try:
        manifest_verified=_verify_package_input_manifest(folder)
    except RuntimeError as exc:
        manifest_verified=False
        report["errors"].append(str(exc))
    if manifest_verified:
        report["package_input_manifest"]="verified"
    else:
        report["package_input_manifest"]="missing_or_invalid"
        if not any(
            "package input" in error.lower()
            for error in report["errors"]
        ):
            report["warnings"].append(
                "Package-input manifest is missing; audio and song.ini "
                "post-validation integrity could not be verified."
            )
    report["passed"]=not report["errors"]
    (folder/"validation_recheck_report.txt").write_text(
        report_text(report),encoding="utf-8"
    )
    save_json(folder/"validation_recheck_report.json",report)
    return report


def replay_cached(folder, output=None, progress=print, compile=False, onyx=None):
    """Replay authoring from cached analysis artifacts without new audio inference."""
    source_folder=Path(folder).resolve()
    analysis_source=source_folder/"analysis"
    metadata_source=source_folder/"metadata"
    required=[analysis_source/"tempo.json",metadata_source/"song.json"]
    missing=[str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Replay source is incomplete: "+", ".join(missing))
    consumed_inputs=set()
    song_path=metadata_source/"song.json"
    song_metadata=json.loads(song_path.read_text(encoding="utf-8"))
    consumed_inputs.add(song_path)
    options_path=metadata_source/"options.json"
    options_payload=(json.loads(options_path.read_text(encoding="utf-8")) if options_path.is_file() else {})
    if options_path.is_file():
        consumed_inputs.add(options_path)
    title=song_metadata.get("title") or "Replay"
    artist=song_metadata.get("artist") or "Unknown Artist"
    folder=_new_output_folder(Path(output) if output else source_folder.parent,artist,title)
    logfile=folder/"logs"/"generation.log"

    def log(message):
        text=str(message)
        with logfile.open("a",encoding="utf-8") as stream:
            stream.write(text+"\n")
        try:
            progress(text)
        except UnicodeEncodeError:
            progress(text.encode("ascii",errors="replace").decode("ascii"))

    log("Starting cached replay (no source decoding, stem separation, or model inference)")
    analysis_folder=folder/"analysis"
    copied_candidates=[]
    for path in sorted(analysis_source.glob("*-events.json"))+sorted(analysis_source.glob("*-transcription.json")):
        shutil.copy2(path,analysis_folder/path.name)
        copied_candidates.append(path)
    for shared in ("chart-profile.json","tempo.json","stems.json"):
        shared_path=analysis_source/shared
        if shared_path.is_file():
            shutil.copy2(shared_path,analysis_folder/shared)
            copied_candidates.append(shared_path)
    source_json=source_folder/"source"/"source.json"
    decoded_wav=source_folder/"source"/"decoded.wav"
    if source_json.is_file():
        shutil.copy2(source_json,folder/"source"/"source.json")
        consumed_inputs.add(source_json)
    if not decoded_wav.is_file():
        raise FileNotFoundError(f"Replay requires cached decoded audio: {decoded_wav}")
    shutil.copy2(decoded_wav,folder/"source"/"decoded.wav")
    tempo_path=analysis_source/"tempo.json"
    tempo=_tempo_from_payload(json.loads(tempo_path.read_text(encoding="utf-8")))
    consumed_inputs.add(tempo_path)
    profile_path=analysis_source/"chart-profile.json"
    profile_payload=(json.loads(profile_path.read_text(encoding="utf-8")) if profile_path.is_file() else {})
    if profile_path.is_file():
        consumed_inputs.add(profile_path)
    previous_charts_path=analysis_source/"charts.json"
    previous_charts=(json.loads(previous_charts_path.read_text(encoding="utf-8")) if previous_charts_path.is_file() else {})
    if previous_charts_path.is_file():
        consumed_inputs.add(previous_charts_path)
    resolved_profile=profile_payload.get("resolved") or song_metadata.get("chart_profile") or "band"
    duration=float(song_metadata.get("source_seconds",0))
    if duration<=0:
        import soundfile as sf

        duration=float(sf.info(decoded_wav).duration)
    available_transcriptions=[
        path.name.replace("-transcription.json","")
        for path in analysis_source.glob("*-transcription.json")
    ]
    instruments=tuple(options_payload.get("instruments") or sorted(set(previous_charts)|set(available_transcriptions)) or ("guitar","bass","drums"))
    ordered=list(instruments)
    if resolved_profile=="electronic" and "drums" in ordered:
        ordered.remove("drums")
        ordered.insert(0,"drums")
    charts={}
    playability={}
    reference_conformance={}
    pattern_stabilization={}
    lead_focus={}
    voice_continuity={}
    phrase_authoring={}
    stage_snapshots={}
    warnings=[]
    for instrument in ordered:
        chord_evidence=None
        if instrument=="drums":
            if instrument not in previous_charts:
                warnings.append("drums: prior chart unavailable in replay source; instrument omitted")
                continue
            expert=_notes_from_payload(
                previous_charts[instrument]["expert"]
            )
            charts[instrument]=reductions(expert,instrument,tempo)
            _record_stage_snapshot(
                stage_snapshots,
                instrument,
                "replay_source_expert",
                expert,
            )
            _record_reduction_snapshot(
                stage_snapshots,instrument,charts[instrument]
            )
            continue
        if resolved_profile=="electronic" and instrument in ("guitar","bass"):
            report_paths=sorted(analysis_source.glob(f"{instrument}-*-events.json"))
            consumed_inputs.update(report_paths)
            groups=_load_accepted_event_groups(analysis_source,instrument)
            if not groups:
                raise FileNotFoundError(f"Cached replay for electronic {instrument} requires {instrument}-*-events.json event reports")
            if instrument=="guitar":
                raw,lead_focus[instrument]=select_lead_phrases(groups)
                save_json(analysis_folder/f"{instrument}-replay-selected.json",raw)
                log(f"Replayed electronic lead selection for {instrument} from cached event reports")
            else:
                raw=electronic_bass_events(groups,charts.get("drums",{}).get("expert",[]))
                save_json(analysis_folder/f"{instrument}-replay-selected.json",raw)
                log("Replayed electronic bass role selection from cached event reports")
        else:
            transcription_path=analysis_source/f"{instrument}-transcription.json"
            if not transcription_path.is_file():
                warnings.append(f"{instrument}: cached transcription missing; instrument omitted")
                continue
            raw=json.loads(transcription_path.read_text(encoding="utf-8"))
            consumed_inputs.add(transcription_path)
            if resolved_profile=="band" and instrument=="guitar":
                report_paths=sorted(analysis_source.glob("guitar-*-events.json"))
                candidate_groups=_load_accepted_event_groups(analysis_source,instrument)
                focus_path=analysis_source/"lead-focus.json"
                focus_payload=json.loads(focus_path.read_text(encoding="utf-8")) if focus_path.is_file() else {}
                focus_report=focus_payload.get("guitar",{})
                if candidate_groups and focus_report:
                    chord_evidence={
                        "candidates":[event for group in candidate_groups for event in group],
                        "focus":focus_report,
                    }
                    lead_focus[instrument]=focus_report
                    consumed_inputs.update(report_paths)
                    consumed_inputs.add(focus_path)
                else:
                    warnings.append(
                        "Guitar chord-section evidence unavailable in cached replay; "
                        "section texture authoring was skipped."
                    )
            save_json(analysis_folder/f"{instrument}-transcription.json",raw)
        if resolved_profile=="band" and instrument=="guitar":
            voice_continuity[instrument]=analyze_band_guitar_voice(raw,tempo)
            save_json(analysis_folder/"guitar-voice-continuity.json",
                      voice_continuity[instrument])
        notes=map_pitched(raw,instrument,
                          "electronic_lead" if resolved_profile=="electronic" and instrument=="guitar" else "range",
                          0.09 if resolved_profile=="electronic" and instrument!="guitar"
                          else BAND_GUITAR_SUSTAIN_SECONDS
                          if resolved_profile=="band" and instrument=="guitar"
                          else 0.22)
        _record_stage_snapshot(
            stage_snapshots,instrument,"after_map_pitched",notes
        )
        notes=[note for note in notes if note.start<duration and note.end>note.start]
        for note in notes:
            note.end=min(duration,note.end)
        if not notes:
            warnings.append(f"{instrument}: no replay notes after mapping; instrument omitted")
            continue
        notes,playability[instrument]=enforce_playability(notes,instrument,tempo,resolved_profile,phrase_aware=instrument=="guitar")
        _record_stage_snapshot(
            stage_snapshots,instrument,"after_enforce_playability",notes
        )
        if resolved_profile=="band" and instrument in ("guitar","bass","keys"):
            notes,reference_conformance[instrument]=conform_expert_to_reference(notes,tempo,instrument)
            _record_stage_snapshot(
                stage_snapshots,
                instrument,
                "after_conform_expert_to_reference",
                notes,
            )
        if instrument=="guitar":
            notes,phrase_authoring[instrument]=author_guitar_phrases(
                notes,tempo,chord_evidence=chord_evidence,
                sustain_threshold=(
                    ELECTRONIC_SUSTAIN_SECONDS
                    if resolved_profile=="electronic"
                    else BAND_GUITAR_SUSTAIN_SECONDS
                ),
            )
            changed=phrase_authoring[instrument]["chord_sections"]["notes_changed"]
            if changed:
                warnings.append(
                    f"Guitar phrase authoring inferred {changed} power-chord attacks "
                    "inside repeated, source-supported texture sections."
                )
            _record_stage_snapshot(stage_snapshots,instrument,"after_author_guitar_phrases",notes)
        charts[instrument]=reductions(notes,instrument,tempo)
        _record_reduction_snapshot(
            stage_snapshots,instrument,charts[instrument]
        )
        write_event_trace(analysis_folder,instrument,raw,charts[instrument]["expert"])
    phrases={instrument:overdrive(parts,tempo,duration) for instrument,parts in charts.items()}
    patterns={instrument:analyze_patterns(parts["expert"],tempo,instrument) for instrument,parts in charts.items()}
    for instrument,details in pattern_stabilization.items():
        patterns.setdefault(instrument,{})["stabilization"]=details
    if resolved_profile=="electronic":
        warnings.append("Cached replay used electronic role selection from saved event reports.")
    # Replay preserves the already-aligned drum Expert chart from charts.json.
    # Reapplying activation alignment would shift drums twice.
    corrections=[]
    structure_diagnostics=analyze_structure_diagnostics(
        charts,tempo,duration
    )
    save_json(analysis_folder/"timing-adjustments.json",corrections)
    save_json(
        analysis_folder/"structure-diagnostics.json",
        structure_diagnostics,
    )
    save_json(analysis_folder/"playability.json",playability)
    save_json(analysis_folder/"patterns.json",patterns)
    save_json(analysis_folder/"reference-conformance.json",reference_conformance)
    save_json(analysis_folder/"pattern-stabilization.json",pattern_stabilization)
    save_json(analysis_folder/"lead-focus.json",lead_focus)
    save_json(analysis_folder/"phrase-authoring.json",phrase_authoring)
    save_json(analysis_folder/"charts.json",charts)
    save_json(analysis_folder/"overdrive.json",phrases)
    for instrument,payload in stage_snapshots.items():
        save_json(analysis_folder/f"{instrument}-stage-snapshots.json",payload)
    metrics_by_instrument={
        instrument:compute_regression_metrics(payload,tempo)
        for instrument,payload in stage_snapshots.items()
    }
    save_json(
        analysis_folder/"regression-metrics-by-instrument.json",
        {
            "schema_version":1,
            "instruments":metrics_by_instrument,
        },
    )
    for instrument,metrics in metrics_by_instrument.items():
        save_json(
            analysis_folder/f"{instrument}-regression-metrics.json",metrics
        )
        log(f"{instrument}: {summarize_metrics(metrics)}")
    guitar_metrics=metrics_by_instrument.get(
        "guitar",{"instrument":"guitar","stage_count":0,"stage_rates":{},"transitions":[],"reductions":{}}
    )
    save_json(analysis_folder/"regression-metrics.json",guitar_metrics)
    save_json(analysis_folder/"replay-provenance.json",_replay_provenance(source_folder,sorted(consumed_inputs|set(copied_candidates),key=lambda row:str(row)),options_payload,profile_payload))
    log("Writing tempo-aware Rock Band MIDI from cached replay")
    midi=folder/"midi"/"notes.mid"
    lead=write_midi(midi,tempo,charts,duration,phrases)
    gain=write_padded_audio(decoded_wav,folder/"audio"/"song.wav",lead)
    metadata=dict(song_metadata)
    metadata.update({
        "source_seconds":duration,
        "lead_seconds":lead,
        "gain":gain,
        "preview_ms":round((lead+min(30,duration*0.25))*1000),
        "analysis_mode":"cached_replay",
        "replay_source_folder":str(source_folder),
    })
    metadata.setdefault("album","")
    metadata.setdefault("year","")
    metadata.setdefault("genre","Other")
    save_json(folder/"metadata"/"song.json",metadata)
    save_json(folder/"metadata"/"options.json",options_payload)
    report=validate(charts,tempo,duration,midi)
    report["warnings"].extend(warnings)
    (folder/"validation_report.txt").write_text(report_text(report),encoding="utf-8")
    save_json(folder/"validation_report.json",report)
    if not report["passed"]:
        raise RuntimeError("Cached replay validation failed. See validation_report.txt")
    create_preview(
        folder,
        charts,
        tempo,
        lead,
        metadata,
        voice_continuity,
        resolved_profile,
        patterns,
        structure_diagnostics,
    )
    missing=[instrument for instrument in instruments if instrument not in charts]
    assessment=assess_chart(
        charts,
        patterns,
        playability,
        resolved_profile,
        missing,
        structure_diagnostics,
    )
    difficulty_tiers={name:item["difficulty_tier"] for name,item in assessment["instruments"].items() if "difficulty_tier" in item}
    write_import(folder/"import",midi,folder/"audio"/"song.wav",metadata,charts,difficulty_tiers)
    _write_package_input_manifest(folder)
    save_json(analysis_folder/"quality-assessment.json",assessment)
    write_song_review_notes(folder/"review_notes.md",metadata["title"],metadata["artist"],assessment)
    result={"status":"draft_ready","folder":str(folder),"review":str(folder/"review.html"),
            "review_notes":str(folder/"review_notes.md"),"title":metadata["title"],"artist":metadata["artist"],
            "chart_profile_requested":song_metadata.get("chart_profile_requested",resolved_profile),
            "chart_profile":resolved_profile,"analysis_mode":"cached_replay",
            "replay_source_folder":str(source_folder),"assessment":assessment,"warnings":report["warnings"],
            "counts":report["counts"],"missing_instruments":missing,"package":None,"xbox_tested":False}
    if compile:
        log("Compiling cached replay with Onyx and its Magma toolchain")
        try:
            result["package"]=str(compile_rb3(folder/"import",folder,metadata["song_id"],onyx or find_onyx(),log))
            result["status"]="compiled_needs_playtest"
        except Exception as exc:
            result["status"]="compile_failed"
            result["warnings"].append(str(exc))
            log(f"Compilation failed: {exc}")
    save_json(folder/"result.json",result)
    log(f"Finished cached replay: {result['status']} — {folder}")
    return result

def batch(options, progress=print):
    from dataclasses import replace
    files=sorted(p for p in options.source.iterdir() if p.suffix.lower() in SUPPORTED)
    if not files:
        raise ValueError("No supported audio files in selected folder")
    results=[]
    for file in files:
        try:
            results.append(generate(replace(options,source=file,title="",artist=""),progress))
        except Exception as exc:
            results.append({"source":str(file),"status":"failed","error":str(exc)})
    options.output.mkdir(parents=True,exist_ok=True)
    stamp=f"{datetime.now():%Y%m%d_%H%M%S}"
    save_json(options.output/f"batch_report_{stamp}.json",results)
    review_path=options.output/f"batch_review_{stamp}.md"
    write_batch_review(review_path,results)
    progress(f"Batch assessment written: {review_path}")
    return results
