import argparse
import json
from pathlib import Path
import sys
from .model import Options

def main():
    if sys.stdout is None or sys.stderr is None:
        log_folder=Path(__file__).resolve().parents[1]/"logs"
        log_folder.mkdir(exist_ok=True)
        stream=(log_folder/"desktop.log").open("a",encoding="utf-8",buffering=1)
        if sys.stdout is None:sys.stdout=stream
        if sys.stderr is None:sys.stderr=stream
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream,"reconfigure"):
            stream.reconfigure(encoding="utf-8",errors="replace")
    parser=argparse.ArgumentParser(description="Generate local Rock Band 3 draft charts")
    sub=parser.add_subparsers(dest="command")
    sub.add_parser("gui")
    share=sub.add_parser("share-chatgpt",help="Export project context and optional chart diagnostics locally")
    share.add_argument("--folder",type=Path,help="Individual chart output folder containing result.json")
    share.add_argument("--output",type=Path,required=True,help="New .txt file to review and transfer to your phone")
    demo=sub.add_parser("demo")
    demo.add_argument("--compile",action="store_true")
    demo.add_argument("--output",type=Path,default=Path("output"))
    package=sub.add_parser("package",help="Compile an already reviewed output folder without reanalysis")
    package.add_argument("folder",type=Path)
    package.add_argument("--onyx",type=Path)
    replay=sub.add_parser("replay",help="Cached replay: rerun authoring from saved analysis (no new audio reanalysis)")
    replay.add_argument("folder",type=Path,help="Existing output run folder with analysis candidates and tempo")
    replay.add_argument("--output",type=Path,default=None,help="Destination root for the replay run folder")
    replay.add_argument("--onyx",type=Path)
    replay.add_argument("--compile",action="store_true",help="Compile replay output with Onyx/Magma after validation")
    reference=sub.add_parser("analyze-reference",help="Learn style statistics from finished Rock Band MIDI charts")
    reference.add_argument("folder",type=Path)
    reference.add_argument("--output",type=Path,default=Path("reference")/"style-profile.json")
    normalize=sub.add_parser("normalize-audio2chart",help="Normalize saved audio2chart token chunks; does not run model inference")
    normalize.add_argument("source",type=Path,help="The exact WAV supplied to audio2chart")
    normalize.add_argument("tokens",type=Path,help="JSON list of token sequences returned by the pinned upstream engine")
    normalize.add_argument("output",type=Path,help="New normalized event-bundle JSON path")
    normalize.add_argument("--pad-token-id",type=int,required=True,help="pad_token_id from the loaded checkpoint config")
    normalize.add_argument("--grid-ms",type=int,required=True,help="grid_ms from the loaded checkpoint config")
    normalize.add_argument("--model-id",default="3podi/charter-v1.0-40-M-best-acc")
    normalize.add_argument("--model-revision")
    normalize.add_argument("--weights-sha256")
    normalize.add_argument("--device",default="unknown")
    normalize.add_argument("--seed",type=int)
    normalize.add_argument("--temperature",type=float)
    normalize.add_argument("--top-k",type=int)
    normalize.add_argument("--runtime-seconds",type=float)
    normalize.add_argument("--tempo-file",type=Path,help="JSON with beats_seconds, numerator and provenance")
    normalize_chart=sub.add_parser("normalize-audio2chart-chart",help="Normalize an audio2chart ExpertSingle .chart file; does not run inference")
    normalize_chart.add_argument("source",type=Path,help="The exact source WAV supplied to audio2chart")
    normalize_chart.add_argument("chart",type=Path,help="The .chart file written by audio2chart")
    normalize_chart.add_argument("output",type=Path,help="New normalized event-bundle JSON path")
    normalize_chart.add_argument("--tempo-file",type=Path,required=True,help="Shared source-time tempo JSON with beats_seconds, numerator and provenance")
    normalize_chart.add_argument("--model-id",default="3podi/charter-v1.0-40-M-best-acc")
    normalize_chart.add_argument("--model-revision")
    normalize_chart.add_argument("--weights-sha256")
    normalize_chart.add_argument("--device",default="unknown")
    normalize_chart.add_argument("--seed",type=int)
    normalize_chart.add_argument("--temperature",type=float)
    normalize_chart.add_argument("--top-k",type=int)
    normalize_chart.add_argument("--runtime-seconds",type=float)
    import_events=sub.add_parser("import-pretrained-events",help="Create a review-only MIDI/preview from normalized Expert events")
    import_events.add_argument("source",type=Path,help="Exact source WAV used for inference")
    import_events.add_argument("events",type=Path,help="Versioned normalized event bundle")
    import_events.add_argument("output",type=Path,help="New output folder; existing review drafts are never overwritten")
    import_events.add_argument("--title")
    import_events.add_argument("--artist")
    compare=sub.add_parser("compare-pretrained-events",help="Compare event bundles against a held-out rights-attested reference")
    compare.add_argument("reference",type=Path)
    compare.add_argument("predictions",type=Path,nargs="+")
    compare.add_argument("--rights-manifest",type=Path,required=True)
    compare.add_argument("--output",type=Path,required=True)
    compare.add_argument("--tolerance-ms",type=float,default=50.0)
    for command in ("generate","batch"):
        p=sub.add_parser(command)
        p.add_argument("source",type=Path)
        p.add_argument("--output",type=Path,default=Path("output"))
        for name in ("title","artist","album","year"):
            p.add_argument("--"+name,default="")
        p.add_argument("--genre",default="")
        p.add_argument("--chart-profile",choices=["auto","band","electronic"],default="auto")
        p.add_argument("--instruments",nargs="+",choices=["guitar","bass","drums","keys"],default=["guitar","bass","drums"])
        p.add_argument("--quality",choices=["precision","balanced"],default="precision")
        p.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
        p.add_argument("--stems",type=Path)
        p.add_argument("--beats-file",type=Path)
        p.add_argument("--bpm",type=float)
        p.add_argument("--first-beat",type=float,default=0)
        p.add_argument("--meter",type=int,default=4)
        p.add_argument("--confidence",type=float,default=0.35)
        p.add_argument("--onyx",type=Path)
        p.add_argument("--no-compile",action="store_true")
    args=parser.parse_args()
    if args.command in (None,"gui"):
        from .gui import launch
        return launch()
    if args.command=="share-chatgpt":
        from .sharing import export_chatgpt
        try:
            destination=export_chatgpt(args.output,args.folder)
        except (OSError,ValueError) as exc:
            parser.error(str(exc))
        print(f"Saved {destination}. Review before sharing; nothing was uploaded.")
        return 0
    from .pipeline import generate,batch,package_existing,replay_cached
    if args.command=="normalize-audio2chart":
        import librosa
        import soundfile as sf
        from .pretrained import audio2chart_tokens_to_bundle,sha256_file
        if args.output.exists() or args.output.is_symlink():
            parser.error(f"Refusing to overwrite normalized event bundle: {args.output}")
        if not args.source.is_file() or not args.tokens.is_file():
            parser.error("Source WAV and token JSON must both exist")
        if args.source.suffix.lower() != ".wav":
            parser.error("audio2chart normalization requires the exact source WAV")
        source_info=sf.info(args.source)
        model_audio,_=librosa.load(args.source,sr=24000,mono=True)
        tempo=(json.loads(args.tempo_file.read_text(encoding="utf-8")) if args.tempo_file else None)
        payload=audio2chart_tokens_to_bundle(
            json.loads(args.tokens.read_text(encoding="utf-8")),
            total_samples=len(model_audio),
            source_sha256=sha256_file(args.source),
            source_sample_rate=source_info.samplerate,
            source_duration_seconds=source_info.frames/source_info.samplerate,
            pad_token_id=args.pad_token_id,
            grid_ms=args.grid_ms,
            model_id=args.model_id,
            model_revision=args.model_revision,
            weights_sha256=args.weights_sha256,
            device=args.device,
            seed=args.seed,
            temperature=args.temperature,
            top_k=args.top_k,
            runtime_seconds=args.runtime_seconds,
            tempo=tempo,
        )
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open("x",encoding="utf-8") as stream:
            json.dump(payload,stream,indent=2)
        print(f"Normalized {len(payload['events'])} audio2chart token events -> {args.output}")
        return 0
    if args.command=="normalize-audio2chart-chart":
        import soundfile as sf
        from .pretrained import audio2chart_chart_to_bundle,sha256_file
        if args.output.exists() or args.output.is_symlink():
            parser.error(f"Refusing to overwrite normalized event bundle: {args.output}")
        if not args.source.is_file() or not args.chart.is_file() or args.source.suffix.lower()!=".wav":
            parser.error("An existing source WAV and audio2chart .chart file are required")
        source_info=sf.info(args.source)
        tempo=json.loads(args.tempo_file.read_text(encoding="utf-8"))
        payload=audio2chart_chart_to_bundle(
            args.chart,
            source_sha256=sha256_file(args.source),
            source_sample_rate=source_info.samplerate,
            source_duration_seconds=source_info.frames/source_info.samplerate,
            tempo=tempo,
            model_id=args.model_id,
            model_revision=args.model_revision,
            weights_sha256=args.weights_sha256,
            device=args.device,
            seed=args.seed,
            temperature=args.temperature,
            top_k=args.top_k,
            runtime_seconds=args.runtime_seconds,
        )
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open("x",encoding="utf-8") as stream:
            json.dump(payload,stream,indent=2)
        print(f"Normalized {len(payload['events'])} audio2chart .chart events -> {args.output}")
        return 0
    if args.command=="import-pretrained-events":
        from .pretrained import import_event_bundle
        result=import_event_bundle(args.source,args.events,args.output,title=args.title,artist=args.artist)
        print(f"Review-only pretrained draft: {result['folder']}")
        return 0
    if args.command=="compare-pretrained-events":
        from .pretrained import compare_bundles,load_bundle,sha256_file,validate_rights_manifest
        if args.output.exists() or args.output.is_symlink():
            parser.error(f"Refusing to overwrite evaluation report: {args.output}")
        reference=load_bundle(args.reference)
        manifest=json.loads(args.rights_manifest.read_text(encoding="utf-8"))
        report=[]
        for path in args.predictions:
            prediction=load_bundle(path)
            validate_rights_manifest(manifest,prediction,args.reference)
            report.append({
                "backend":prediction["backend"]["name"],
                "prediction_bundle_sha256":sha256_file(path),
                **compare_bundles(prediction,reference,args.tolerance_ms),
            })
        result={
            "evaluation_version":1,
            "rights_manifest_sha256":sha256_file(args.rights_manifest),
            "reference_bundle_sha256":sha256_file(args.reference),
            "results":report,
            "limitations":[
                "Event metrics are not a human-quality score or listening/playtest result.",
                "No inference is performed by this command.",
            ],
        }
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open("x",encoding="utf-8") as stream:
            json.dump(result,stream,indent=2)
        print(f"Saved held-out event comparison -> {args.output}")
        return 0
    if args.command=="analyze-reference":
        from .reference import analyze_reference_corpus
        report=analyze_reference_corpus(args.folder,args.output)
        print(f"Analyzed {report['analyzed_files']}/{report['midi_files']} reference charts -> {args.output}")
        return 1 if report["failed_files"] else 0
    if args.command=="package":
        result=package_existing(args.folder,onyx=args.onyx)
    elif args.command=="replay":
        result=replay_cached(args.folder,output=args.output,onyx=args.onyx,compile=args.compile)
    elif args.command=="demo":
        from .demo import make_demo
        source,stems=make_demo(Path("input")/"demo")
        result=generate(Options(source=source,stems=stems,title="Precision Demo",artist="RB3 Charts",output=args.output,bpm=120,first_beat=1,compile=args.compile))
    else:
        values=vars(args).copy()
        command=values.pop("command")
        values["compile"]=not values.pop("no_compile")
        result=(batch if command=="batch" else generate)(Options(**values))
    results=result if isinstance(result,list) else [result]
    return 1 if any(r["status"] in ("failed","compile_failed") for r in results) else 0

if __name__=="__main__":
    sys.exit(main())
