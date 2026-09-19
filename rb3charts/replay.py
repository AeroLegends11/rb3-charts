"""Offline authoring replay from saved post-voice-selection pitched events."""
from dataclasses import asdict
from datetime import datetime
from hashlib import sha256
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import json
import math
import subprocess
from .authoring import author_chart
from .chart import map_pitched
from .diagnostics import metrics, changes, repeat_families, repeat_metrics
from .model import TempoMap, save_json

ROOT = Path(__file__).resolve().parents[1]


def _hash(path):
    return sha256(path.read_bytes()).hexdigest()


def _git(*args):
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def replay(folder, output=Path("output"), instrument="guitar", start=0.0, end=None):
    """Replay the whole song; start/end restrict reporting only, never context.

    Produces diagnostics and chart JSON, not a package or an audio reanalysis.
    Each invocation creates a new directory, preserving source runs and reviews.
    """
    if instrument not in ("guitar", "bass", "keys"):
        raise ValueError("Replay supports pitched instruments: guitar, bass, keys")
    if not math.isfinite(start) or start < 0 or (end is not None and (not math.isfinite(end) or end <= start)):
        raise ValueError("Report interval must be finite, nonnegative and increasing")
    folder = Path(folder).resolve()
    inputs = {"events": folder / "analysis" / f"{instrument}-transcription.json",
              "tempo": folder / "analysis" / "tempo.json",
              "metadata": folder / "metadata" / "song.json"}
    data = {key: json.loads(path.read_text(encoding="utf-8")) for key, path in inputs.items()}
    tempo = TempoMap(**data["tempo"])
    metadata = data["metadata"]
    profile = metadata["chart_profile"]
    if profile not in ("band", "electronic"):
        raise ValueError("Replay requires a saved resolved Band/Electronic profile")
    duration = float(metadata["source_seconds"])
    if not math.isfinite(duration) or duration <= start:
        raise ValueError("Report start must be before the finite positive song duration")
    stop = min(duration, end) if end is not None else duration
    raw = data["events"]
    notes = map_pitched(raw, instrument,
                        "electronic_lead" if profile == "electronic" and instrument == "guitar" else "range",
                        .09 if profile == "electronic" and instrument != "guitar" else .22)
    notes = [n for n in notes if n.start < duration and n.end > n.start]
    for note in notes:
        note.end = min(duration, note.end)
    charts, reports, audit = author_chart(notes, instrument, tempo, profile)
    region = lambda rows: [r for r in rows if start <= r["start"] < stop]
    families = repeat_families(region(audit["stages"][0]["notes"]), tempo)
    for stage in audit["stages"]:
        rows = region(stage["notes"])
        stage["region_metrics"] = metrics(rows)
        stage["region_repetition"] = repeat_metrics(rows, families)
    audit["report_interval_source_seconds"] = [start, stop]

    old_charts = folder / "analysis" / "charts.json"
    baseline = None
    if old_charts.is_file():
        inputs["historical_charts"] = old_charts
        previous = json.loads(old_charts.read_text(encoding="utf-8")).get(instrument, {}).get("expert")
        if previous is not None:
            current = json.loads(json.dumps([asdict(n) for n in charts["expert"]]))
            baseline = {"label": "historical saved chart; original code/dependencies may differ",
                        "before": metrics(region(previous)), "after": metrics(region(current)),
                        "changes": changes(region(previous), region(current))}
    options_path = folder / "metadata" / "options.json"
    source_path = folder / "source" / "source.json"
    for key, path in (("saved_options", options_path), ("saved_source", source_path)):
        if path.is_file():
            inputs[key] = path
    packages = {}
    for package in ("numpy", "scipy", "librosa", "mido", "basic-pitch", "demucs", "adtof-pytorch"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    profile_path = ROOT / "reference" / "style-profile.json"
    status = _git("status", "--porcelain")
    provenance = {"schema": 1, "mode": "cached_post_voice_selection_authoring",
                  "source_run": folder.name, "instrument": instrument, "profile": profile,
                  "git_revision": _git("rev-parse", "HEAD"),
                  "git_dirty": bool(status) if status is not None else None,
                  "code_hashes": {p.name: _hash(p) for p in sorted((ROOT / "rb3charts").glob("*.py"))},
                  "input_hashes": {key: _hash(path) for key, path in inputs.items()},
                  "profile_sha256": _hash(profile_path) if profile_path.is_file() else None,
                  "installed_packages": packages,
                  "saved_options": json.loads(options_path.read_text(encoding="utf-8")) if options_path.is_file() else None,
                  "model_weights": "not rerun; original weight hashes are not recorded in legacy runs",
                  "limits": "Does not test source separation, event detection, voice selection, MIDI or export. No random choices in authoring."}
    destination = Path(output).resolve() / f"authoring_replay_{instrument}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    destination.mkdir(parents=True, exist_ok=False)
    save_json(destination / "provenance.json", provenance)
    save_json(destination / "authoring-stages.json", audit)
    save_json(destination / "charts.json", charts)
    save_json(destination / "stage-reports.json", reports)
    save_json(destination / "historical-comparison.json", baseline)
    lines = ["# Cached authoring replay", "", f"Input run: `{folder.name}`",
             f"Instrument/profile: {instrument}/{profile}. Source seconds {start:g}–{stop:g}.", "",
             "Whole-song context was processed. This is a diagnostic replay, not audio reanalysis or proof of musical improvement.", "",
             "| Stage | Attacks | Collapsed pitch changes | Reversed contour | Repeat comparisons | Same lanes | Same shape |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    def percent(value):
        return "n/a" if value is None else f"{100 * value:.1f}%"
    for stage in audit["stages"]:
        m, r = stage["region_metrics"], stage["region_repetition"]
        lines.append(f"| {stage['stage']} | {m['attacks']} | {m['collapsed_pitch_changes']} | {m['reversed_pitch_changes']} | {r['comparisons']} | {percent(r['absolute_lane_agreement'])} | {percent(r['relative_shape_agreement'])} |")
    lines += ["", "Repetition uses fixed four-event pitch/rhythm signatures from mapped input. Missing occurrences reduce coverage; n/a means no comparisons, not perfect agreement.",
              "", "Inspect `authoring-stages.json` for every note snapshot, source-ID change, coverage and mismatch timestamp. Historical comparison is separately labeled because the earlier generator may differ."]
    (destination / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"status": "diagnostics_ready", "folder": str(destination), "report": str(destination / "report.md")}
