"""Local user feedback: save reports, find recurring problems and draft issues.

Reports never change charting rules automatically. They become an improvement
queue and GitHub issue drafts so each fix can be tested, measured and approved.
"""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
import json
import re
import subprocess
import uuid

from . import __version__
from .sharing import _redact

ROOT = Path(__file__).resolve().parents[1]
ISSUE_URL = "https://github.com/AeroLegends11/rb3-charts/issues/new"
MAX_DESCRIPTION = 4000
MAX_URL = 7000
RECURRING_THRESHOLD = 2

CATEGORIES = {
    "missing_notes": "Missing notes — the music plays something the chart skips",
    "extra_notes": "Extra notes — notes where the music is quiet or different",
    "timing": "Timing — notes early, late or drifting",
    "wrong_lane": "Wrong lane or color — melody moves but the frets do not match",
    "repeated_riff": "Repeated riff charted differently each time",
    "sustain": "Sustain length — held notes too long or too short",
    "difficulty": "Difficulty feels too hard or too easy",
    "drums": "Drum pattern — kicks, cymbals or fills",
    "packaging": "Packaging or Xbox problem",
    "app": "App problem — buttons, wording, crash or confusing step",
    "other": "Something else",
}
INSTRUMENTS = ("any", "guitar", "bass", "drums", "keys")
DIFFICULTIES = ("any", "expert", "hard", "medium", "easy")
SEVERITIES = {
    "minor": "Minor — noticeable but playable",
    "major": "Major — makes a section feel wrong",
    "blocking": "Blocking — unplayable, crash or cannot package",
}
CHART_CATEGORIES = {"missing_notes", "extra_notes", "timing", "wrong_lane",
                    "repeated_riff", "sustain", "difficulty", "drums"}


def parse_time(text):
    """Return seconds from '83.5', '1:23' or '1:02:03'; None when blank."""
    text = (text or "").strip()
    if not text:
        return None
    if not re.fullmatch(r"\d+(?::\d{1,2}){0,2}(?:\.\d+)?", text):
        raise ValueError(f"Use a time like 1:23 or 83.5 seconds, not '{text}'.")
    seconds = 0.0
    for part in text.split(":"):
        seconds = seconds * 60 + float(part)
    return round(seconds, 3)


def format_time(seconds):
    if seconds is None:
        return ""
    minutes, rest = divmod(float(seconds), 60)
    return f"{int(minutes)}:{rest:06.3f}".rstrip("0").rstrip(".")


def make_report(category, description, *, instrument="any", difficulty="any",
                severity="minor", start="", end="", run_folder=None):
    if category not in CATEGORIES:
        raise ValueError("Choose what kind of problem this is.")
    if instrument not in INSTRUMENTS or difficulty not in DIFFICULTIES:
        raise ValueError("Choose a listed instrument and difficulty.")
    if severity not in SEVERITIES:
        raise ValueError("Choose how serious the problem is.")
    description = (description or "").strip()
    if len(description) < 5:
        raise ValueError("Describe what you noticed in a few words.")
    if len(description) > MAX_DESCRIPTION:
        raise ValueError(f"Keep the description under {MAX_DESCRIPTION} characters.")
    start_s, end_s = parse_time(start), parse_time(end)
    if start_s is None and end_s is not None:
        raise ValueError("Enter a start time before the end time.")
    if start_s is not None and end_s is not None and end_s < start_s:
        raise ValueError("The end time must be after the start time.")
    created = datetime.now(timezone.utc)
    return {
        "schema": 1,
        "id": created.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6],
        "created": created.isoformat(timespec="seconds"),
        "category": category,
        "instrument": instrument,
        "difficulty": difficulty,
        "severity": severity,
        "start_seconds": start_s,
        "end_seconds": end_s,
        "description": description,
        "status": "new",
        "app_version": __version__,
        "app_revision": _revision(),
        "run": _run_context(run_folder),
    }


def _revision():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
                              ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _run_context(run_folder):
    if not run_folder:
        return None
    folder = Path(run_folder)
    context = {"folder_name": folder.name}
    try:
        result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return context
    context.update({key: result.get(key) for key in ("status", "title", "artist", "chart_profile")})
    assessment = result.get("assessment")
    if isinstance(assessment, dict):
        context["assessment"] = assessment.get("status")
    return context


def feedback_root(root=ROOT):
    return Path(root) / "output" / "feedback"


def save_report(report, *, run_folder=None, root=ROOT):
    """Store the report with its chart run and in the shared local index."""
    folder = feedback_root(root)
    folder.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    saved = folder / f"{report['id']}.json"
    saved.write_text(text, encoding="utf-8")
    if run_folder and (Path(run_folder) / "result.json").is_file():
        run_feedback = Path(run_folder) / "feedback"
        run_feedback.mkdir(exist_ok=True)
        (run_feedback / saved.name).write_text(text, encoding="utf-8")
    with (folder / "index.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False) + "\n")
    write_improvement_queue(root)
    return saved


def load_reports(root=ROOT):
    index = feedback_root(root) / "index.jsonl"
    reports = []
    if not index.is_file():
        return reports
    for line in index.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict) and item.get("category") in CATEGORIES:
            reports.append(item)
    return reports


def summarize(reports):
    groups = Counter((r["category"], r.get("instrument", "any")) for r in reports)
    recurring = [
        {"category": category, "instrument": instrument, "count": count,
         "label": CATEGORIES[category],
         "blocking": sum(r.get("severity") == "blocking" for r in reports
                         if (r["category"], r.get("instrument", "any")) == (category, instrument))}
        for (category, instrument), count in groups.most_common()
        if count >= RECURRING_THRESHOLD
    ]
    return {
        "total": len(reports),
        "by_category": dict(Counter(r["category"] for r in reports).most_common()),
        "by_severity": dict(Counter(r.get("severity", "minor") for r in reports)),
        "recurring": recurring,
    }


def write_improvement_queue(root=ROOT):
    reports = load_reports(root)
    summary = summarize(reports)
    lines = [
        "# Improvement queue from user feedback",
        "",
        "Generated locally from saved reports. The app does not change its charting",
        "rules by itself: each recurring problem should become a GitHub issue with a",
        "regression check, then a reviewed fix measured before and after.",
        "",
        f"Reports: {summary['total']}",
        "",
        "## Recurring problems (reported at least twice)",
        "",
    ]
    if summary["recurring"]:
        for item in summary["recurring"]:
            lines.append(f"- {item['count']}× {item['label']} · instrument: {item['instrument']}"
                         + (f" · {item['blocking']} blocking" if item["blocking"] else ""))
    else:
        lines.append("- None yet.")
    lines += ["", "## Latest reports", ""]
    for report in reports[-10:][::-1]:
        lines.append(f"- {report['created']} · {CATEGORIES[report['category']]} · "
                     f"{report.get('instrument', 'any')} · {report.get('severity', 'minor')}")
    path = feedback_root(root) / "improvement-queue.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def issue_title(report):
    run = report.get("run") or {}
    song = " — ".join(x for x in (run.get("artist"), run.get("title")) if x)
    label = CATEGORIES[report["category"]].split(" — ")[0]
    return _redact(f"[Feedback] {label}" + (f": {song}" if song else ""))[:200]


def issue_body(report):
    """Markdown matching .github/ISSUE_TEMPLATE/chart-quality.md, without local paths."""
    run = report.get("run") or {}
    start, end = format_time(report.get("start_seconds")), format_time(report.get("end_seconds"))
    where = f"{start}–{end}" if start and end else start or "not given"
    acceptance = ("A listening comparison or regression replay of this section shows the "
                  "problem is gone without making other sections worse."
                  if report["category"] in CHART_CATEGORIES else
                  "The described step works as expected on a fresh launch.")
    body = f"""## What I hear / want

{report['description']}

Category: {CATEGORIES[report['category']]}
Severity: {SEVERITIES[report.get('severity', 'minor')]}

## Where

Source-song timestamp or range: {where}
Instrument and difficulty: {report.get('instrument', 'any')} / {report.get('difficulty', 'any')}
Chart profile: {run.get('chart_profile') or 'unknown'}
Local output folder name: {run.get('folder_name') or 'not attached'}
Draft status / self-assessment: {run.get('status') or 'unknown'} / {run.get('assessment') or 'unknown'}
App: {report.get('app_version')} ({report.get('app_revision') or 'unknown revision'})
Report ID: {report['id']}

## Acceptance check

{acceptance}

Submitted from the in-app feedback form. No audio, MIDI or chart files are attached.
"""
    return _redact(body)


def issue_url(report):
    query = {"title": issue_title(report), "body": issue_body(report), "labels": "user-feedback"}
    url = ISSUE_URL + "?" + urlencode(query)
    if len(url) > MAX_URL:
        query["body"] = query["body"][:1500] + "\n\n(Trimmed; full text was copied by the app.)"
        url = ISSUE_URL + "?" + urlencode(query)
    return url
