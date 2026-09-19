from pathlib import Path
from dataclasses import asdict
from datetime import datetime
import json
import re
import shutil
import traceback
from .model import Options, save_json
from .audio import decode, fingerprint, separate, read_mono, write_padded_audio, detect_chart_profile
from .timing import analyze_tempo
from .transcribe import pitched, neural_drums, import_gm_drums, electronic_bass_events, band_guitar_lead_events
from .chart import map_pitched, clean, reductions, overdrive, align_drum_activations, enforce_playability, analyze_patterns, refine_band_guitar, conform_expert_to_reference, stabilize_repeated_phrases
from .midi import write_midi
from .validate import validate, report_text
from .export import write_import, compile_rb3, find_onyx
from .preview import create_preview
from .assessment import assess_chart, write_song_review_notes, write_batch_review
from .events import write_event_trace
from .phrases import select_lead_phrases, author_guitar_phrases

ROOT = Path(__file__).resolve().parents[1]
SUPPORTED = {".mp3",".wav",".flac",".ogg",".m4a",".aiff",".aif",".wma"}

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
    slug=re.sub(r"[^a-zA-Z0-9]+","_",metadata["artist"]+"_"+metadata["title"]).strip("_")[:48] or "song"
    song_id="rb3c_"+digest[:12]
    folder=options.output.resolve()/f"{slug}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    for part in ("source","audio","analysis","midi","metadata","logs","rb3"):
        (folder/part).mkdir(parents=True,exist_ok=True)
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
        tempo=analyze_tempo(y,sr,duration,options.bpm,options.first_beat,options.meter,options.beats_file,bpm_hint,resolved_profile=="electronic")
        save_json(folder/"analysis"/"tempo.json",asdict(tempo))
        charts={}
        playability={}
        reference_conformance={}
        pattern_stabilization={}
        lead_focus={}
        phrase_authoring={}
        warnings=[]
        ordered=list(options.instruments)
        if resolved_profile=="electronic" and "drums" in ordered:
            ordered.remove("drums")
            ordered.insert(0,"drums")
        for instrument in ordered:
            log(f"Generating {instrument} Expert chart")
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
                                tempo=tempo,analysis_folder=folder/"analysis")
                    if resolved_profile=="band" and instrument=="guitar":
                        log("Selecting the lead-guitar contour from ambiguous pitched candidates")
                        raw,lead_focus[instrument]=band_guitar_lead_events(stem,raw,60/tempo.bpm)
                        focus=lead_focus[instrument]
                        warnings.append(
                            f"Guitar lead focus resolved {focus.get('groups_resolved',0)} ambiguous onset groups "
                            f"and removed {focus.get('sequential_outliers_removed',0)} unsupported sequential pitch spikes."
                        )
                save_json(folder/"analysis"/f"{instrument}-transcription.json",raw)
                notes=map_pitched(raw,instrument,
                                  "electronic_lead" if resolved_profile=="electronic" and instrument=="guitar" else "range",
                                  0.09 if resolved_profile=="electronic" and instrument!="guitar" else 0.22)
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
            if resolved_profile=="band" and instrument in ("guitar","bass","keys"):
                notes,reference_conformance[instrument]=conform_expert_to_reference(notes,tempo,instrument)
            if resolved_profile=="band" and instrument=="guitar":
                # Smooth the final playable shapes after corpus conformance;
                # chord simplification can itself create new single-note moves.
                notes=refine_band_guitar(notes,tempo)
                notes,pattern_stabilization[instrument]=stabilize_repeated_phrases(notes,tempo,instrument)
            if instrument=="guitar":
                notes,phrase_authoring[instrument]=author_guitar_phrases(notes,tempo)
            removed=playability[instrument]["removed"]
            if removed:
                warnings.append(f"{instrument}: playability gate removed {removed} crowded attacks; retained {playability[instrument]['after']} with a peak four-second density of {playability[instrument]['peak_four_second_nps']:.2f}/s.")
            log(f"Reducing {instrument}: Expert → Hard → Medium → Easy")
            charts[instrument]=reductions(notes,instrument,tempo)
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
        save_json(folder/"analysis"/"timing-adjustments.json",corrections)
        save_json(folder/"analysis"/"playability.json",playability)
        save_json(folder/"analysis"/"patterns.json",patterns)
        save_json(folder/"analysis"/"reference-conformance.json",reference_conformance)
        save_json(folder/"analysis"/"pattern-stabilization.json",pattern_stabilization)
        save_json(folder/"analysis"/"lead-focus.json",lead_focus)
        save_json(folder/"analysis"/"phrase-authoring.json",phrase_authoring)
        save_json(folder/"analysis"/"charts.json",charts)
        save_json(folder/"analysis"/"overdrive.json",phrases)
        log("Writing tempo-aware Rock Band MIDI")
        midi=folder/"midi"/"notes.mid"
        lead=write_midi(midi,tempo,charts,duration,phrases)
        gain=write_padded_audio(wav,folder/"audio"/"song.wav",lead)
        metadata.update({"song_id":song_id,"source_seconds":duration,"lead_seconds":lead,"gain":gain,"preview_ms":round((lead+min(30,duration*0.25))*1000)})
        save_json(folder/"metadata"/"song.json",metadata)
        report=validate(charts,tempo,duration,midi)
        report["warnings"].extend(warnings)
        (folder/"validation_report.txt").write_text(report_text(report),encoding="utf-8")
        save_json(folder/"validation_report.json",report)
        if not report["passed"]:
            raise RuntimeError("Chart validation failed. See validation_report.txt")
        log("Creating audible timing check and chart review")
        create_preview(folder,charts,tempo,lead,metadata)
        missing=[i for i in options.instruments if i not in charts]
        assessment=assess_chart(charts,patterns,playability,resolved_profile,missing)
        difficulty_tiers={name:item["difficulty_tier"] for name,item in assessment["instruments"].items() if "difficulty_tier" in item}
        write_import(folder/"import",midi,folder/"audio"/"song.wav",metadata,charts,difficulty_tiers)
        save_json(folder/"analysis"/"quality-assessment.json",assessment)
        write_song_review_notes(folder/"review_notes.md",metadata["title"],metadata["artist"],assessment)
        result={"status":"draft_ready","folder":str(folder),"review":str(folder/"review.html"),"review_notes":str(folder/"review_notes.md"),"title":metadata["title"],"artist":metadata["artist"],"chart_profile_requested":options.chart_profile,"chart_profile":resolved_profile,"assessment":assessment,"warnings":report["warnings"],"counts":report["counts"],"missing_instruments":missing,"package":None,"xbox_tested":False}
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
