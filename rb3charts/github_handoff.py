"""Create and submit privacy-safe chart-quality tasks for GitHub Copilot."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import subprocess


def _percent(value) -> str:
    return "n/a" if value is None else f"{float(value):.1%}"


def _number(value, suffix: str = "") -> str:
    return "n/a" if value is None else f"{float(value):.4f}{suffix}"


def _problem_score(row: dict) -> float:
    # Timing omissions/extras dominate. Lane disagreement breaks ties without
    # exposing the reference chart's individual note events.
    matched = int(row.get("matched_attacks", 0))
    lane_disagreement = matched * (1.0 - float(row.get("exact_lane_rate", 0.0)))
    return int(row.get("extra_attacks", 0)) + int(row.get("missing_attacks", 0)) + lane_disagreement


def _git_value(*arguments: str) -> str:
    completed = subprocess.run(["git", *arguments], cwd=Path(__file__).resolve().parents[1],
                               text=True, capture_output=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else ""


def create_github_handoff(report: dict, reference: dict, destination: Path | str,
                          title: str, artist: str, region_limit: int = 8) -> Path:
    """Write a report summary suitable for committing or posting to GitHub.

    It intentionally excludes local paths, audio, per-note rows, and cached
    reference-chart contents. Hashes keep comparisons traceable.
    """
    folder = Path(destination).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    provenance, summary = report["provenance"], report["summary"]
    coordinate = provenance.get("alignment_coordinate", "second")
    start_key, end_key = f"start_{coordinate}", f"end_{coordinate}"
    worst = sorted(report.get("sections", []), key=_problem_score, reverse=True)[:region_limit]
    instrument = provenance.get("instrument", "guitar").title()
    difficulty = provenance.get("difficulty", "expert").title()
    issue_title = f"Improve {instrument} chart generation from {artist} - {title} comparison"
    base_branch = _git_value("branch", "--show-current") or "main"

    rows = []
    for row in worst:
        rows.append(
            f"| {row.get(start_key, 0):g}-{row.get(end_key, 0):g} {coordinate}s | "
            f"{row.get('generated_attacks', 0)} | {row.get('reference_attacks', 0)} | "
            f"{row.get('extra_attacks', 0)} | {row.get('missing_attacks', 0)} | "
            f"{_percent(row.get('exact_lane_rate'))} |"
        )

    timing_key = f"median_timing_error_{coordinate}s" if coordinate == "second" else "median_timing_error_beats"
    p95_key = f"p95_timing_error_{coordinate}s" if coordinate == "second" else "p95_timing_error_beats"
    sustain_key = f"sustain_mae_{coordinate}s" if coordinate == "second" else "sustain_mae_beats"
    lines = [
        f"# {issue_title}", "",
        "## Reproducible comparison summary", "",
        f"- Song metadata: **{artist} - {title}**",
        f"- Track: **{difficulty} {instrument}**",
        f"- Generated MIDI SHA-256: `{provenance.get('generated_sha256', 'unknown')}`",
        f"- Reference: `{reference.get('provider', 'unknown')}` chart `{reference.get('chart_id', 'unknown')}` by `{reference.get('charter', 'unknown')}`",
        f"- Reference package MD5: `{reference.get('md5', 'unknown')}`",
        f"- Reference chart hash: `{reference.get('chart_hash', 'unknown')}`",
        f"- Reference tempo-map hash: `{reference.get('tempo_map_hash', 'unknown')}`",
        f"- Reference MIDI SHA-256: `{provenance.get('reference_sha256', 'unknown')}`",
        f"- GitHub base branch: `{base_branch}`",
        f"- Alignment: `{provenance.get('alignment', 'unknown')}`, offset `{provenance.get(f'alignment_offset_{coordinate}s' if coordinate == 'second' else 'alignment_offset_beats', 0):+.4f} {coordinate}s`",
        "", "## Current result", "",
        f"- Attacks: {summary.get('generated_attacks', 0)} generated / {summary.get('reference_attacks', 0)} reference; {summary.get('extra_attacks', 0)} extra and {summary.get('missing_attacks', 0)} missing",
        f"- Attack precision / recall / F1: {_percent(summary.get('attack_precision'))} / {_percent(summary.get('attack_recall'))} / {_percent(summary.get('attack_f1'))}",
        f"- Median / p95 timing error: {_number(summary.get(timing_key))} / {_number(summary.get(p95_key))} {coordinate}s",
        f"- Exact lanes / relative shape / chord class: {_percent(summary.get('exact_lane_rate'))} / {_percent(summary.get('relative_shape_rate'))} / {_percent(summary.get('chord_class_rate'))}",
        f"- Sustain MAE: {_number(summary.get(sustain_key))} {coordinate}s",
        f"- Recurring-pattern coverage: {_percent(summary.get('recurring_pattern_coverage'))}",
        "", "## Highest-error regions", "",
        f"| Source time | Generated | Reference | Extra | Missing | Exact lanes |",
        "|---|---:|---:|---:|---:|---:|",
        *rows,
        "", "## Coding task", "",
        "Use this comparison as evaluation evidence. Diagnose which general charting subsystem most plausibly causes the attack-density, phrase-stability, lane-shape, or sustain errors shown above. Make one scoped, deterministic improvement that applies across songs, and add or update a focused test that demonstrates the intended behavior.",
        "", "Follow these constraints:", "",
        "- Do not special-case this song, artist, timestamp, title, or reference chart ID.",
        "- Do not use reference-chart notes as generation inputs or copy its authored note sequence.",
        "- Preserve supported musical variations and rests; repetition is evidence, not permission to invent attacks.",
        "- Keep source IDs and source-audio timing traceable. Derive lower difficulties from canonical Expert.",
        "- Keep commercial audio, cached reference MIDI, full comparison data, and generated outputs out of Git.",
        "- Read `AGENTS.md` and `docs/DEVELOPMENT_PLAN.md` before changing chart logic.",
        "", "## Acceptance checks", "",
        "- The focused regression test passes and represents a general musical rule.",
        "- The full test suite passes.",
        "- Report the subsystem changed, the expected musical effect, and all tests run.",
        "- If the same local cached evidence is available, report before/after metrics using identical inputs and settings. Otherwise state clearly that musical improvement still needs local replay and listening review.",
        "", "This packet contains aggregate evaluation statistics only. It does not contain the song audio, private local paths, or the reference chart's note events.", "",
    ]
    path = folder / "github-handoff.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    metadata = {
        "schema": 1, "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "issue_title": issue_title, "body_file": path.name, "base_branch": base_branch,
    }
    (folder / "github-handoff.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path


def submit_to_github_copilot(handoff: Path | str, repository: str | None = None) -> str:
    """Create one GitHub issue assigned to Copilot and return its URL."""
    handoff_path = Path(handoff).expanduser().resolve()
    folder = handoff_path.parent
    submitted_path = folder / "github-submission.json"
    if submitted_path.is_file():
        saved = json.loads(submitted_path.read_text(encoding="utf-8"))
        if saved.get("issue_url"):
            return str(saved["issue_url"])
    metadata = json.loads((folder / "github-handoff.json").read_text(encoding="utf-8"))
    repository = repository or _git_value("config", "--get", "remote.origin.url")
    if repository.endswith(".git"):
        repository = repository[:-4]
    if repository.startswith("git@github.com:"):
        repository = repository.split(":", 1)[1]
    elif "github.com/" in repository:
        repository = repository.split("github.com/", 1)[1]
    repository = repository.strip("/")
    if not repository or "/" not in repository:
        raise RuntimeError("Could not determine the GitHub owner/repository from origin")
    base_branch = metadata.get("base_branch", "main")
    published = subprocess.run(["git", "ls-remote", "--exit-code", "--heads", "origin", base_branch],
                               cwd=Path(__file__).resolve().parents[1], text=True,
                               capture_output=True, check=False)
    if published.returncode:
        raise RuntimeError(f"The base branch '{base_branch}' has not been pushed to GitHub yet")
    payload = {
        "title": metadata["issue_title"], "body": handoff_path.read_text(encoding="utf-8"),
        "assignees": ["copilot-swe-agent[bot]"],
        "agent_assignment": {
            "target_repo": repository, "base_branch": base_branch,
            "custom_instructions": "Follow the issue body and repository Copilot instructions. Keep the change scoped and open a pull request.",
            "custom_agent": "", "model": "",
        },
    }
    command = ["gh", "api", "--method", "POST", "-H", "Accept: application/vnd.github+json",
               "-H", "X-GitHub-Api-Version: 2022-11-28", f"repos/{repository}/issues", "--input", "-"]
    completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], text=True,
                               input=json.dumps(payload), capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError((completed.stderr or completed.stdout or "GitHub issue creation failed").strip())
    try:
        issue_url = json.loads(completed.stdout)["html_url"]
    except (json.JSONDecodeError, KeyError):
        issue_url = completed.stdout.strip().splitlines()[-1]
    submitted_path.write_text(json.dumps({"schema": 1, "issue_url": issue_url,
                                           "submitted_at": datetime.now().astimezone().isoformat(timespec="seconds")}, indent=2) + "\n", encoding="utf-8")
    return issue_url
