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
    benchmark=sub.add_parser("benchmark-replays",help="Compare guitar metrics across two replay corpora")
    benchmark.add_argument("baseline",type=Path)
    benchmark.add_argument("candidate",type=Path)
    benchmark.add_argument("--output",type=Path,default=Path("output")/"replay-benchmark.json")
    reference=sub.add_parser("analyze-reference",help="Learn style statistics from finished Rock Band MIDI charts")
    reference.add_argument("folder",type=Path)
    reference.add_argument("--output",type=Path,default=Path("reference")/"style-profile.json")
    dataset=sub.add_parser("validate-dataset",help="Validate a rights-traceable training dataset manifest")
    dataset.add_argument("manifest",type=Path)
    dataset.add_argument("--verify-hashes",action="store_true",help="Read every audio/annotation file and verify SHA-256")
    chart_pairs=sub.add_parser("validate-chart-pairs",help="Validate authorized synchronized audio/chart training pairs")
    chart_pairs.add_argument("manifest",type=Path)
    chart_pairs.add_argument("--verify-hashes",action="store_true",help="Verify audio, chart, and rights-evidence hashes")
    convert_pairs=sub.add_parser("convert-chart-pairs",help="Create Stage B examples from authorized chart pairs")
    convert_pairs.add_argument("manifest",type=Path)
    convert_pairs.add_argument("output",type=Path)
    guitarset=sub.add_parser("convert-guitarset",help="Convert GuitarSet JAMS to lane-independent structure labels")
    guitarset.add_argument("annotation",type=Path)
    guitarset.add_argument("output",type=Path)
    slakh=sub.add_parser("convert-slakh",help="Convert one Slakh track to lane-independent structure labels")
    slakh.add_argument("track",type=Path)
    slakh.add_argument("output",type=Path)
    slakh_mix=sub.add_parser("mix-slakh-guitar",help="Sum aligned Guitar-class Slakh stems")
    slakh_mix.add_argument("track",type=Path)
    slakh_mix.add_argument("output",type=Path)
    guitar_techs=sub.add_parser("convert-guitar-techs",help="Convert Guitar-TECHS MIDI to structure labels")
    guitar_techs.add_argument("midi",type=Path)
    guitar_techs.add_argument("audio",type=Path)
    guitar_techs.add_argument("output",type=Path)
    guitar_techs.add_argument("--performer-id",default="P3")
    features=sub.add_parser("extract-structure-features",help="Create model-ready audio features and structure targets")
    features.add_argument("audio",type=Path)
    features.add_argument("labels",type=Path)
    features.add_argument("output",type=Path)
    training=sub.add_parser("train-structure-baseline",help="Train the first attack/polyphony/sustain baseline")
    training.add_argument("index",type=Path)
    training.add_argument("output",type=Path)
    training.add_argument("--epochs",type=int,default=1)
    training.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    training.add_argument("--seed",type=int,default=1337)
    prediction=sub.add_parser("predict-structure",help="Run lane-independent model evaluation without changing charts")
    prediction.add_argument("model",type=Path)
    prediction.add_argument("audio",type=Path)
    prediction.add_argument("output",type=Path)
    prediction.add_argument("--beats",type=Path)
    prediction.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    tabcnn=sub.add_parser("predict-tabcnn",help="Run offline CC BY 4.0 TabCNN pitch/polyphony evaluation")
    tabcnn.add_argument("checkpoint",type=Path)
    tabcnn.add_argument("audio",type=Path)
    tabcnn.add_argument("output",type=Path)
    tabcnn.add_argument("--attacks",type=Path,help="Optional structure-predictions JSON to sample")
    tabcnn.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
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
    if args.command=="analyze-reference":
        from .reference import analyze_reference_corpus
        report=analyze_reference_corpus(args.folder,args.output)
        print(f"Analyzed {report['analyzed_files']}/{report['midi_files']} reference charts -> {args.output}")
        return 1 if report["failed_files"] else 0
    if args.command=="validate-dataset":
        from .datasets import DatasetManifestError, validate_dataset_manifest
        try:
            report=validate_dataset_manifest(args.manifest,args.verify_hashes)
        except DatasetManifestError as exc:
            print(f"Dataset manifest is invalid: {exc}",file=sys.stderr)
            return 1
        print(json.dumps(report,indent=2))
        return 0
    if args.command=="validate-chart-pairs":
        from .datasets import DatasetManifestError, validate_chart_pair_manifest
        try:
            report=validate_chart_pair_manifest(args.manifest,args.verify_hashes)
        except DatasetManifestError as exc:
            print(f"Chart-pair manifest is invalid: {exc}",file=sys.stderr)
            return 1
        print(json.dumps(report,indent=2))
        return 0
    if args.command=="convert-chart-pairs":
        from .training import TrainingDataError, convert_authorized_chart_pairs
        try:
            report=convert_authorized_chart_pairs(args.manifest,args.output)
        except TrainingDataError as exc:
            print(f"Chart-pair conversion failed: {exc}",file=sys.stderr)
            return 1
        print(json.dumps({
            "dataset_id": report["dataset_id"],
            "examples": len(report["examples"]),
            "output": str(args.output),
        },indent=2))
        return 0
    if args.command=="benchmark-replays":
        from .regression import benchmark_replays
        try:
            report=benchmark_replays(args.baseline,args.candidate,args.output)
        except (FileNotFoundError,ValueError) as exc:
            print(f"Replay benchmark failed: {exc}",file=sys.stderr)
            return 1
        print(json.dumps(report["macro"],indent=2))
        print(f"Compared {report['songs_matched']} songs -> {args.output}")
        return 0
    if args.command=="convert-guitarset":
        from .training import TrainingDataError, convert_guitarset_jams
        try:
            report=convert_guitarset_jams(args.annotation,args.output)
        except TrainingDataError as exc:
            print(f"GuitarSet conversion failed: {exc}",file=sys.stderr)
            return 1
        print(json.dumps(report["summary"],indent=2))
        return 0
    if args.command=="convert-slakh":
        from .training import TrainingDataError, convert_slakh_track
        try:
            report=convert_slakh_track(args.track,args.output)
        except TrainingDataError as exc:
            print(f"Slakh conversion failed: {exc}",file=sys.stderr)
            return 1
        print(json.dumps(report["summary"],indent=2))
        return 0
    if args.command=="mix-slakh-guitar":
        from .training import TrainingDataError, mix_slakh_guitar_stems
        try:
            report=mix_slakh_guitar_stems(args.track,args.output)
        except TrainingDataError as exc:
            print(f"Slakh guitar mixing failed: {exc}",file=sys.stderr)
            return 1
        print(json.dumps(report,indent=2))
        return 0
    if args.command=="convert-guitar-techs":
        from .training import TrainingDataError, convert_guitar_techs_midi
        try:
            report=convert_guitar_techs_midi(
                args.midi,args.audio,args.output,args.performer_id
            )
        except TrainingDataError as exc:
            print(f"Guitar-TECHS conversion failed: {exc}",file=sys.stderr)
            return 1
        print(json.dumps(report["summary"],indent=2))
        return 0
    if args.command=="extract-structure-features":
        from .training import TrainingDataError, extract_structure_features
        try:
            report=extract_structure_features(args.audio,args.labels,args.output)
        except TrainingDataError as exc:
            print(f"Structure feature extraction failed: {exc}",file=sys.stderr)
            return 1
        print(json.dumps(report,indent=2))
        return 0
    if args.command=="train-structure-baseline":
        from .training import TrainingDataError, train_structure_baseline
        try:
            report=train_structure_baseline(args.index,args.output,args.epochs,args.device,args.seed)
        except TrainingDataError as exc:
            print(f"Structure baseline training failed: {exc}",file=sys.stderr)
            return 1
        print(json.dumps(report,indent=2))
        return 0
    if args.command=="predict-structure":
        from .training import TrainingDataError, predict_structure
        try:
            report=predict_structure(args.model,args.audio,args.output,args.beats,args.device)
        except TrainingDataError as exc:
            print(f"Structure inference failed: {exc}",file=sys.stderr)
            return 1
        print(json.dumps(report["summary"],indent=2))
        return 0
    if args.command=="predict-tabcnn":
        from .tabcnn import TabCNNError, predict_tabcnn
        try:
            report=predict_tabcnn(
                args.checkpoint,args.audio,args.output,args.attacks,args.device
            )
        except TabCNNError as exc:
            print(f"TabCNN inference failed: {exc}",file=sys.stderr)
            return 1
        print(json.dumps(report["summary"],indent=2))
        return 0
    from .pipeline import generate,batch,package_existing,replay_cached
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
