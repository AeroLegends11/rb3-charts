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
    validate_run=sub.add_parser(
        "validate-run",
        help="Recheck an existing draft without rerunning authoring",
    )
    validate_run.add_argument("folder",type=Path)
    validate_runs=sub.add_parser(
        "validate-runs",
        help="Recheck every existing draft below a folder",
    )
    validate_runs.add_argument("folder",type=Path)
    validate_runs.add_argument("--output",type=Path)
    replay=sub.add_parser("replay",help="Cached replay: rerun authoring from saved analysis (no new audio reanalysis)")
    replay.add_argument("folder",type=Path,help="Existing output run folder with analysis candidates and tempo")
    replay.add_argument("--output",type=Path,default=None,help="Destination root for the replay run folder")
    replay.add_argument("--onyx",type=Path)
    replay.add_argument("--compile",action="store_true",help="Compile replay output with Onyx/Magma after validation")
    replay.add_argument(
        "--fretnet-shadow",
        type=Path,
        help="Optional FretNet evidence for a diagnostic-only replay snapshot",
    )
    benchmark=sub.add_parser(
        "benchmark-replays",
        help="Compare one instrument across two replay corpora",
    )
    benchmark.add_argument("baseline",type=Path)
    benchmark.add_argument("candidate",type=Path)
    benchmark.add_argument("--output",type=Path)
    benchmark.add_argument(
        "--instrument",
        choices=["guitar","bass","drums","keys","all"],
        default="guitar",
    )
    review_compare=sub.add_parser(
        "compare-reviews",
        help="Create a synchronized A/B page for two completed drafts",
    )
    review_compare.add_argument("baseline",type=Path)
    review_compare.add_argument("candidate",type=Path)
    review_compare.add_argument("--output",type=Path)
    midi_review=sub.add_parser(
        "review-midi-comparison",
        help="Create a synchronized generated-versus-reference review page",
    )
    midi_review.add_argument("generated",type=Path)
    midi_review.add_argument("reference",type=Path)
    midi_review.add_argument("--output",type=Path)
    midi_review.add_argument(
        "--difficulty",
        choices=["expert","hard","medium","easy"],
        default="hard",
    )
    midi_review.add_argument(
        "--reference-difficulty",
        choices=["expert","hard","medium","easy"],
        default="expert",
    )
    midi_review.add_argument("--offset",type=float,required=True)
    abc_review=sub.add_parser(
        "review-expert-hard-reference",
        help="Create a synchronized project Expert/Hard/reference Expert review",
    )
    abc_review.add_argument("generated",type=Path)
    abc_review.add_argument("reference",type=Path)
    abc_review.add_argument("--output",type=Path)
    abc_review.add_argument("--offset",type=float,required=True)
    human_compare=sub.add_parser("compare-human-chart",help="Compare generated guitar with an authorized held-out chart")
    human_compare.add_argument("manifest",type=Path)
    human_compare.add_argument("song_id")
    human_compare.add_argument("generated",type=Path)
    human_compare.add_argument("--output",type=Path)
    human_compare.add_argument(
        "--source-agreement-report",
        type=Path,
        help="Attach external or directly covered paired-source evidence",
    )
    midi_compare=sub.add_parser(
        "compare-midi",
        help="Evaluate a completed guitar chart against reference MIDI",
    )
    midi_compare.add_argument("generated",type=Path)
    midi_compare.add_argument("reference",type=Path)
    midi_compare.add_argument("--output",type=Path)
    midi_compare.add_argument(
        "--difficulty",
        choices=["expert","hard","medium","easy"],
        default="expert",
    )
    midi_compare.add_argument(
        "--reference-difficulty",
        choices=["expert","hard","medium","easy"],
        help="Reference difficulty; defaults to the generated difficulty",
    )
    midi_compare.add_argument("--tolerance-ms",type=int,default=80)
    midi_compare.add_argument("--offset",type=float)
    comparison_index=sub.add_parser(
        "index-comparisons",
        help="Build a sanitized local corpus index of comparison reports",
    )
    comparison_index.add_argument("root",type=Path)
    comparison_index.add_argument("--output",type=Path)
    comparison_trend=sub.add_parser(
        "compare-comparison-indexes",
        help="Compare compatible sanitized comparison corpus indexes",
    )
    comparison_trend.add_argument("baseline",type=Path)
    comparison_trend.add_argument("candidate",type=Path)
    comparison_trend.add_argument("--output",type=Path)
    release_gate=sub.add_parser(
        "assess-official-quality",
        help="Assess the authorized corpus against the 80%% near-official gate",
    )
    release_gate.add_argument("index",type=Path)
    release_gate.add_argument("--output",type=Path)
    release_gate.add_argument("--target-pass-rate",type=float,default=.8)
    release_gate.add_argument("--minimum-songs",type=int,default=20)
    release_gate.add_argument("--minimum-artists",type=int,default=5)
    evidence_audit=sub.add_parser(
        "audit-playability-evidence",
        help="Join playability removals to post-generation source evidence",
    )
    evidence_audit.add_argument("folder",type=Path)
    evidence_audit.add_argument("--output",type=Path)
    evidence_audit.add_argument("--structure-predictions",type=Path)
    evidence_audit.add_argument("--reference",type=Path)
    evidence_audit.add_argument(
        "--reference-format",
        choices=["rock_band_midi","frets_on_fire_midi","clone_hero_chart"],
        default="rock_band_midi",
    )
    evidence_audit.add_argument("--tolerance-ms",type=int,default=80)
    evidence_audit.add_argument("--offset",type=float)
    reference=sub.add_parser("analyze-reference",help="Learn style statistics from finished Rock Band MIDI charts")
    reference.add_argument("folder",type=Path)
    reference.add_argument("--output",type=Path,default=Path("reference")/"style-profile.json")
    reference.add_argument(
        "--full-output",
        type=Path,
        help="Optional local per-song report; do not stage this file",
    )
    dataset=sub.add_parser("validate-dataset",help="Validate a rights-traceable training dataset manifest")
    dataset.add_argument("manifest",type=Path)
    dataset.add_argument("--verify-hashes",action="store_true",help="Read every audio/annotation file and verify SHA-256")
    chart_pairs=sub.add_parser("validate-chart-pairs",help="Validate authorized synchronized audio/chart training pairs")
    chart_pairs.add_argument("manifest",type=Path)
    chart_pairs.add_argument("--verify-hashes",action="store_true",help="Verify audio, chart, and rights-evidence hashes")
    chart_pairs.add_argument(
        "--source-agreement-report",
        type=Path,
        help="Attach a validated offline paired-source evaluation summary",
    )
    convert_pairs=sub.add_parser("convert-chart-pairs",help="Create Stage B examples from authorized chart pairs")
    convert_pairs.add_argument("manifest",type=Path)
    convert_pairs.add_argument("output",type=Path)
    guitarset=sub.add_parser("convert-guitarset",help="Convert GuitarSet JAMS to lane-independent structure labels")
    guitarset.add_argument("annotation",type=Path)
    guitarset.add_argument("output",type=Path)
    egfxset=sub.add_parser("convert-egfxset",help="Convert one EGFxSet WAV to attack-only structure labels")
    egfxset.add_argument("audio",type=Path)
    egfxset.add_argument("output",type=Path)
    egfx_phrase=sub.add_parser(
        "synthesize-egfxset-phrase",
        help="Build a deterministic multi-attack phrase from EGFxSet WAVs",
    )
    egfx_phrase.add_argument("source",type=Path)
    egfx_phrase.add_argument("audio",type=Path)
    egfx_phrase.add_argument("labels",type=Path)
    egfx_phrase.add_argument("--seed",type=int,default=1337)
    egfx_phrase.add_argument("--sequence",type=int,default=0)
    egfx_phrase.add_argument("--duration",type=float,default=10.0)
    procedural_phrase=sub.add_parser(
        "synthesize-procedural-guitar",
        help="Render rights-clean guitar-like audio and structure labels",
    )
    procedural_phrase.add_argument("audio",type=Path)
    procedural_phrase.add_argument("labels",type=Path)
    procedural_phrase.add_argument("--seed",type=int,default=1337)
    procedural_phrase.add_argument("--sequence",type=int,default=0)
    procedural_phrase.add_argument("--duration",type=float,default=12.0)
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
    training.add_argument(
        "--attack-loss",
        choices=["weighted-bce","focal"],
        default="weighted-bce",
    )
    training.add_argument("--focal-gamma",type=float,default=2.0)
    training.add_argument(
        "--calibration-dataset",
        help="Select the global threshold on validation items from one dataset",
    )
    training.add_argument(
        "--architecture",
        choices=[
            "conv1d",
            "dilated-tcn",
            "multi-scale",
            "multi-scale-local",
            "multi-scale-offset",
            "multi-scale-gated",
        ],
        default="conv1d",
    )
    training.add_argument(
        "--skip-test",
        action="store_true",
        help="Train and select on validation without evaluating test items",
    )
    training.add_argument(
        "--localization-weight",
        type=float,
        default=0.0,
        help="Weight a local attack-peak ranking objective",
    )
    training.add_argument(
        "--onset-feature",
        action="store_true",
        help="Append positive spectral flux as an explicit timing feature",
    )
    training.add_argument(
        "--initialize-model",
        type=Path,
        help="Initialize compatible weights from an existing checkpoint",
    )
    training.add_argument(
        "--offset-only",
        action="store_true",
        help="Freeze initialized selection weights and train only attack offsets",
    )
    prediction=sub.add_parser("predict-structure",help="Run lane-independent model evaluation without changing charts")
    prediction.add_argument("model",type=Path)
    prediction.add_argument("audio",type=Path)
    prediction.add_argument("output",type=Path)
    prediction.add_argument("--beats",type=Path)
    prediction.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    ranking=sub.add_parser("evaluate-structure-ranking",help="Score an offline candidate rank against structure labels")
    ranking.add_argument("predictions",type=Path)
    ranking.add_argument("labels",type=Path)
    ranking.add_argument("--output",type=Path)
    ranking.add_argument("--rank-field",default="periodic_width_attack_rank_percentile")
    ranking.add_argument("--selection-count",type=int)
    ranking.add_argument("--tolerance-ms",type=int,default=50)
    agreement=sub.add_parser(
        "evaluate-source-agreement",
        help="Evaluate aligned rights-cleared recording channels",
    )
    agreement.add_argument("manifest",type=Path)
    agreement.add_argument("--output",type=Path)
    agreement.add_argument("--primary-channel",default="micamp")
    agreement.add_argument("--alternate-channel",default="directinput")
    agreement.add_argument("--tolerance-ms",type=int,default=50)
    pin_agreement=sub.add_parser(
        "pin-source-agreement",
        help="Hash-pin aligned alternate recordings into a dataset manifest",
    )
    pin_agreement.add_argument("manifest",type=Path)
    pin_agreement.add_argument("alternate_root",type=Path)
    pin_agreement.add_argument("output",type=Path)
    pin_agreement.add_argument("--primary-suffix",required=True)
    pin_agreement.add_argument("--alternate-suffix",required=True)
    pin_agreement.add_argument("--archive",type=Path)
    pin_agreement.add_argument("--published-md5")
    pin_agreement.add_argument("--release-url")
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
        report=analyze_reference_corpus(
            args.folder,args.output,args.full_output
        )
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
            report=validate_chart_pair_manifest(
                args.manifest,
                args.verify_hashes,
                args.source_agreement_report,
            )
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
        from .regression import benchmark_all_replays, benchmark_replays
        output=(
            args.output
            or Path("output")/(
                "replay-benchmark.json"
                if args.instrument=="guitar"
                else "all-instruments-replay-benchmark.json"
                if args.instrument=="all"
                else f"{args.instrument}-replay-benchmark.json"
            )
        )
        try:
            if args.instrument=="all":
                report=benchmark_all_replays(
                    args.baseline,args.candidate,output
                )
            else:
                report=benchmark_replays(
                    args.baseline,
                    args.candidate,
                    output,
                    instrument=args.instrument,
                )
        except (FileNotFoundError,ValueError) as exc:
            print(f"Replay benchmark failed: {exc}",file=sys.stderr)
            return 1
        if args.instrument=="all":
            print(json.dumps({
                instrument:item["macro"]
                for instrument,item in report["instruments"].items()
            },indent=2))
            print(
                f"Compared {len(report['instruments'])} instruments "
                f"-> {output}"
            )
        else:
            print(json.dumps(report["macro"],indent=2))
            print(
                f"Compared {report['songs_matched']} songs -> {output}"
            )
        return 0
    if args.command=="compare-reviews":
        from .preview import create_comparison_preview
        output=args.output or Path("output")/"comparison-review.html"
        try:
            report=create_comparison_preview(
                args.baseline,args.candidate,output
            )
        except (
            FileNotFoundError,KeyError,TypeError,ValueError,json.JSONDecodeError
        ) as exc:
            print(f"Review comparison failed: {exc}",file=sys.stderr)
            return 1
        print(json.dumps(report,indent=2))
        return 0
    if args.command=="review-midi-comparison":
        from .preview import create_midi_comparison_preview
        output=args.output or Path("output")/"midi-comparison-review.html"
        try:
            report=create_midi_comparison_preview(
                args.generated,
                args.reference,
                output,
                args.difficulty,
                args.reference_difficulty,
                args.offset,
            )
        except (
            FileNotFoundError,KeyError,OSError,TypeError,ValueError,
            json.JSONDecodeError,
        ) as exc:
            print(f"MIDI review creation failed: {exc}",file=sys.stderr)
            return 1
        print(json.dumps(report,indent=2))
        return 0
    if args.command=="review-expert-hard-reference":
        from .preview import create_expert_hard_reference_preview
        output=args.output or Path("output")/"expert-hard-reference-review.html"
        try:
            report=create_expert_hard_reference_preview(
                args.generated,args.reference,output,args.offset
            )
        except (
            FileNotFoundError,KeyError,OSError,TypeError,ValueError,
            json.JSONDecodeError,
        ) as exc:
            print(f"Three-way review creation failed: {exc}",file=sys.stderr)
            return 1
        print(json.dumps(report,indent=2))
        return 0
    if args.command=="compare-human-chart":
        from .datasets import DatasetManifestError
        from .regression import compare_authorized_chart
        try:
            report=compare_authorized_chart(
                args.manifest,
                args.song_id,
                args.generated,
                args.output,
                args.source_agreement_report,
            )
        except (DatasetManifestError,FileNotFoundError,ValueError) as exc:
            print(f"Human-chart comparison failed: {exc}",file=sys.stderr)
            return 1
        print(json.dumps(report["difficulties"]["expert"],indent=2))
        return 0
    if args.command=="compare-midi":
        from .datasets import DatasetManifestError
        from .regression import compare_reference_midi
        try:
            report=compare_reference_midi(
                args.generated,
                args.reference,
                args.output,
                difficulty=args.difficulty,
                reference_difficulty=args.reference_difficulty,
                tolerance=args.tolerance_ms/1000,
                offset=args.offset,
            )
        except (
            DatasetManifestError,FileNotFoundError,KeyError,TypeError,ValueError
        ) as exc:
            print(f"MIDI comparison failed: {exc}",file=sys.stderr)
            return 1
        print(json.dumps({
            "alignment_offset_seconds":
                report["provenance"]["alignment_offset_seconds"],
            **report["metrics"]["attack"],
            "worst_reference_recall":
                report["metrics"]["attack_windows"]
                ["worst_reference_recall"],
        },indent=2))
        return 0
    if args.command=="index-comparisons":
        from .regression import build_comparison_index
        try:
            report=build_comparison_index(args.root,args.output)
        except (FileNotFoundError,KeyError,TypeError,ValueError) as exc:
            print(f"Comparison indexing failed: {exc}",file=sys.stderr)
            return 1
        print(json.dumps({
            "comparison_schema":report["comparison_schema"],
            "runs":len(report["entries"]),
            "available":sum(
                row["comparison_status"]=="available"
                for row in report["entries"]
            ),
        },indent=2))
        return 0
    if args.command=="compare-comparison-indexes":
        from .regression import compare_comparison_indexes
        try:
            report=compare_comparison_indexes(
                args.baseline,args.candidate,args.output
            )
        except (FileNotFoundError,KeyError,TypeError,ValueError) as exc:
            print(f"Comparison trend failed: {exc}",file=sys.stderr)
            return 1
        print(json.dumps({
            "paired_available":report["coverage"]["paired_available"],
            "shared_missing":report["coverage"]["shared_missing"],
            "baseline_only":report["coverage"]["baseline_only"],
            "candidate_only":report["coverage"]["candidate_only"],
            "attack_f1":report["macro"]["attack_f1"],
        },indent=2))
        return 0
    if args.command=="assess-official-quality":
        from .regression import assess_official_quality_release
        try:
            report=assess_official_quality_release(
                args.index,
                args.output,
                target_pass_rate=args.target_pass_rate,
                minimum_songs=args.minimum_songs,
                minimum_artists=args.minimum_artists,
            )
        except (FileNotFoundError,KeyError,TypeError,ValueError) as exc:
            print(f"Official-quality assessment failed: {exc}",file=sys.stderr)
            return 1
        print(json.dumps({
            "release_ready":report["release_ready"],
            "songs_passed":report["songs_passed"],
            "songs_evaluated":report["songs_evaluated"],
            "pass_rate":report["pass_rate"],
            "coverage_ready":report["coverage_ready"],
        },indent=2))
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
    if args.command=="convert-egfxset":
        from .training import TrainingDataError, convert_egfxset_audio
        try:
            report=convert_egfxset_audio(args.audio,args.output)
        except TrainingDataError as exc:
            print(f"EGFxSet conversion failed: {exc}",file=sys.stderr)
            return 1
        print(json.dumps(report["summary"],indent=2))
        return 0
    if args.command=="synthesize-egfxset-phrase":
        from .training import TrainingDataError, synthesize_egfxset_phrase
        try:
            report=synthesize_egfxset_phrase(
                args.source,
                args.audio,
                args.labels,
                args.seed,
                args.sequence,
                args.duration,
            )
        except TrainingDataError as exc:
            print(f"EGFxSet phrase synthesis failed: {exc}",file=sys.stderr)
            return 1
        print(json.dumps(report["summary"],indent=2))
        return 0
    if args.command=="synthesize-procedural-guitar":
        from .training import TrainingDataError, synthesize_procedural_guitar_phrase
        try:
            report=synthesize_procedural_guitar_phrase(
                args.audio,
                args.labels,
                args.seed,
                args.sequence,
                args.duration,
            )
        except TrainingDataError as exc:
            print(f"Procedural guitar synthesis failed: {exc}",file=sys.stderr)
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
            report=train_structure_baseline(
                args.index,
                args.output,
                args.epochs,
                args.device,
                args.seed,
                args.attack_loss,
                args.focal_gamma,
                args.calibration_dataset,
                args.architecture,
                not args.skip_test,
                args.localization_weight,
                args.onset_feature,
                args.initialize_model,
                args.offset_only,
            )
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
    if args.command=="evaluate-structure-ranking":
        from .training import TrainingDataError, evaluate_structure_ranking
        try:
            report=evaluate_structure_ranking(
                args.predictions,
                args.labels,
                args.output,
                args.rank_field,
                args.selection_count,
                args.tolerance_ms,
            )
        except TrainingDataError as exc:
            print(f"Structure ranking evaluation failed: {exc}",file=sys.stderr)
            return 1
        print(json.dumps(report,indent=2))
        return 0
    if args.command=="evaluate-source-agreement":
        from .training import TrainingDataError, evaluate_source_agreement
        try:
            report=evaluate_source_agreement(
                args.manifest,
                args.output,
                args.primary_channel,
                args.alternate_channel,
                args.tolerance_ms,
            )
        except TrainingDataError as exc:
            print(f"Source agreement evaluation failed: {exc}",file=sys.stderr)
            return 1
        print(json.dumps(report["splits"],indent=2))
        return 0
    if args.command=="pin-source-agreement":
        from .training import TrainingDataError, pin_source_agreement_manifest
        try:
            report=pin_source_agreement_manifest(
                args.manifest,
                args.alternate_root,
                args.output,
                args.primary_suffix,
                args.alternate_suffix,
                args.archive,
                args.published_md5,
                args.release_url,
            )
        except TrainingDataError as exc:
            print(f"Source agreement pinning failed: {exc}",file=sys.stderr)
            return 1
        print(json.dumps(report,indent=2))
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
    if args.command=="audit-playability-evidence":
        from .regression import audit_playability_evidence
        try:
            report=audit_playability_evidence(
                args.folder,
                args.output,
                args.structure_predictions,
                args.reference,
                args.reference_format,
                args.tolerance_ms/1000,
                args.offset,
            )
        except (OSError,ValueError,KeyError,json.JSONDecodeError) as exc:
            print(f"Playability evidence audit failed: {exc}",file=sys.stderr)
            return 1
        print(json.dumps(report["summary"],indent=2))
        return 0
    from .pipeline import (
        batch,
        generate,
        package_existing,
        replay_cached,
        validate_existing_run,
    )
    if args.command=="validate-run":
        try:
            report=validate_existing_run(args.folder)
        except (
            FileNotFoundError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            print(
                f"Existing run validation failed: {exc}",
                file=sys.stderr,
            )
            return 1
        print(
            "Existing run validation: "
            + ("PASS" if report["passed"] else "FAIL")
        )
        return 0 if report["passed"] else 1
    if args.command=="validate-runs":
        from .pipeline import validate_existing_runs
        try:
            report=validate_existing_runs(args.folder,args.output)
        except (FileNotFoundError,ValueError) as exc:
            print(f"Batch run validation failed: {exc}",file=sys.stderr)
            return 1
        print(json.dumps({
            "runs_total":report["runs_total"],
            "runs_passed":report["runs_passed"],
            "runs_failed":report["runs_failed"],
            "output":report["output"],
        },indent=2))
        return 1 if report["runs_failed"] else 0
    if args.command=="package":
        try:
            result=package_existing(args.folder,onyx=args.onyx)
        except (FileNotFoundError,RuntimeError,ValueError) as exc:
            print(f"Packaging failed: {exc}",file=sys.stderr)
            return 1
    elif args.command=="replay":
        try:
            result=replay_cached(
                args.folder,
                output=args.output,
                onyx=args.onyx,
                compile=args.compile,
                fretnet_shadow=args.fretnet_shadow,
            )
        except (
            FileNotFoundError,KeyError,RuntimeError,TypeError,ValueError
        ) as exc:
            print(f"Cached replay failed: {exc}",file=sys.stderr)
            return 1
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
