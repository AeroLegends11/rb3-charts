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
    sub.add_parser("compare-gui",help="Open the folder-and-video comparison app")
    replay_parser=sub.add_parser("replay",help="Diagnose authoring from cached post-voice-selection pitched events")
    replay_parser.add_argument("folder",type=Path)
    replay_parser.add_argument("--output",type=Path,default=Path("output"))
    replay_parser.add_argument("--instrument",choices=["guitar","bass","keys"],default="guitar")
    replay_parser.add_argument("--start",type=float,default=0)
    replay_parser.add_argument("--end",type=float)
    compare_parser=sub.add_parser("compare",help="Compare a generated chart with a finished five-lane reference MIDI")
    compare_parser.add_argument("generated",type=Path,help="Generated notes.mid or output folder")
    compare_parser.add_argument("reference",type=Path,help="Reference notes.mid or folder")
    compare_parser.add_argument("--output",type=Path,default=Path("comparisons"))
    compare_parser.add_argument("--instrument",choices=["guitar","bass","drums","keys"],default="guitar")
    compare_parser.add_argument("--difficulty",choices=["easy","medium","hard","expert"],default="expert")
    compare_parser.add_argument("--tolerance-beats",type=float,default=0.125)
    compare_parser.add_argument("--offset-beats",type=float,help="Manual generated-to-reference beat offset")
    video_parser=sub.add_parser("compare-video",help="Compare a review folder beside an official YouTube gameplay video")
    video_parser.add_argument("folder",type=Path,help="RB3 Charts output folder containing review.html")
    video_parser.add_argument("video_url")
    video_parser.add_argument("--output",type=Path,default=Path("comparisons"))
    video_parser.add_argument("--no-open",action="store_true")
    encore_parser=sub.add_parser("compare-encore",help="Find a pinned Chorus Encore reference and compare automatically")
    encore_parser.add_argument("folder",type=Path,help="RB3 Charts output folder")
    encore_parser.add_argument("--output",type=Path,default=Path("comparisons"))
    encore_parser.add_argument("--instrument",choices=["guitar","bass","drums","keys"],default="guitar")
    encore_parser.add_argument("--difficulty",choices=["easy","medium","hard","expert"],default="expert")
    encore_parser.add_argument("--refresh",action="store_true",help="Refresh the pinned catalog entry from Encore")
    demo=sub.add_parser("demo")
    demo.add_argument("--compile",action="store_true")
    demo.add_argument("--output",type=Path,default=Path("output"))
    package=sub.add_parser("package",help="Compile an already reviewed output folder")
    package.add_argument("folder",type=Path)
    package.add_argument("--onyx",type=Path)
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
    if args.command=="compare-gui":
        from .compare_gui import launch
        return launch()
    if args.command=="replay":
        from .replay import replay
        result=replay(args.folder,args.output,args.instrument,args.start,args.end)
        print(result["report"])
        return 0
    if args.command=="compare":
        from .compare import compare_charts
        report=compare_charts(args.generated,args.reference,args.output,args.instrument,args.difficulty,args.tolerance_beats,args.offset_beats)
        print(report["output"]["html"])
        return 0
    if args.command=="compare-video":
        from .video_compare import create_video_comparison,start_comparison_server
        result=create_video_comparison(args.folder,args.video_url,args.output,False)
        print(result["html"])
        if not args.no_open:
            import webbrowser
            server,url=start_comparison_server(result)
            print(f"Serving {url} - press Ctrl+C when finished")
            webbrowser.open(url)
            try:
                from threading import Event
                Event().wait()
            except KeyboardInterrupt:server.shutdown()
        return 0
    if args.command=="compare-encore":
        from .encore import compare_with_encore
        report=compare_with_encore(args.folder,args.output,refresh=args.refresh,instrument=args.instrument,difficulty=args.difficulty)
        reference=report["encore_reference"]
        print(f"{reference['charter']} chart #{reference['chart_id']} -> {report['output']['html']}")
        return 0
    from .pipeline import generate,batch,package_existing
    if args.command=="analyze-reference":
        from .reference import analyze_reference_corpus
        report=analyze_reference_corpus(args.folder,args.output)
        print(f"Analyzed {report['analyzed_files']}/{report['midi_files']} reference charts -> {args.output}")
        return 1 if report["failed_files"] else 0
    if args.command=="package":
        result=package_existing(args.folder,onyx=args.onyx)
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
