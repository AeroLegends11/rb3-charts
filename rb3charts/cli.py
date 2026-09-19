import argparse
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
    reference=sub.add_parser("analyze-reference",help="Learn style statistics from finished Rock Band MIDI charts")
    reference.add_argument("folder",type=Path)
    reference.add_argument("--output",type=Path,default=Path("reference")/"style-profile.json")
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
    from .pipeline import generate,batch,package_existing,replay_cached
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
