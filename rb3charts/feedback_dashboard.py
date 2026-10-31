"""Deterministic remote feedback triage; never evaluates user-supplied text."""
from collections import Counter
import json
import os
from pathlib import Path

from . import feedback
from .feedback_remote import REPOSITORY, _api, report_metadata

DASHBOARD_MARKER = "<!-- rb3-feedback-dashboard -->"
DASHBOARD_NUMBER = 18


def _label_names(issue):
    return {label["name"] if isinstance(label, dict) else label
            for label in issue.get("labels", [])}


def triage_labels(metadata):
    return {
        "user-feedback",
        "feedback:" + metadata["category"],
        "instrument:" + metadata["instrument"],
        "severity:" + metadata["severity"],
    }


def is_dashboard(issue):
    return ((issue.get("body") or "").startswith(DASHBOARD_MARKER + "\n")
            and "feedback-dashboard" in _label_names(issue))


def render_dashboard(issues):
    reports = []
    seen = set()
    for issue in issues:
        if "pull_request" in issue or "feedback-test" in _label_names(issue):
            continue
        metadata = report_metadata(issue)
        if metadata and metadata["id"] not in seen:
            seen.add(metadata["id"])
            reports.append((issue, metadata))
    open_reports = [(issue, meta) for issue, meta in reports if issue["state"] == "open"]
    groups = Counter((meta["category"], meta["instrument"]) for _, meta in open_reports)
    blocking = sum(meta["severity"] == "blocking" for _, meta in open_reports)
    reasons = Counter(issue.get("state_reason") or "unspecified"
                      for issue, _ in reports if issue["state"] == "closed")
    lines = [
        DASHBOARD_MARKER,
        "# Shared user feedback inbox",
        "",
        "Automatically grouped from in-app reports. Update the original issues to discuss, "
        "assign or resolve problems; this dashboard follows their state.",
        "",
        f"**Reports:** {len(reports)} total | {len(open_reports)} open | "
        f"{len(reports)-len(open_reports)} closed | {blocking} blocking and open",
        "",
        "Closed reasons: " + (", ".join(f"{reason}: {count}" for reason, count in sorted(reasons.items()))
                             or "none yet"),
        "",
        "## Open recurring problems",
        "",
        "| Problem | Instrument | Open reports |",
        "|---|---|---|",
    ]
    recurring = [(group, count) for group, count in groups.most_common() if count >= 2]
    for (category, instrument), count in recurring:
        lines.append(f"| {feedback.CATEGORIES[category]} | {instrument} | {count} |")
    if not recurring:
        lines.append("| None reported twice yet | - | 0 |")
    lines += ["", "## Open reports (blocking first)", ""]
    severity_order = {"blocking": 0, "major": 1, "minor": 2}
    ordered = sorted(open_reports, key=lambda row: (
        severity_order[row[1]["severity"]], -row[0]["number"]))
    for issue, metadata in ordered[:100]:
        lines.append(
            f"- #{issue['number']} - {feedback.CATEGORIES[metadata['category']]} "
            f"/ {metadata['instrument']} / {metadata['severity']}")
    if not ordered:
        lines.append("No open in-app reports.")
    if len(ordered) > 100:
        lines.append("Showing 100; the Issues view contains the full inbox.")
    lines += [
        "",
        "Reports are observations, not proof of chart quality. Fixes still require tests "
        "and listening/regression review. Closed does not necessarily mean fixed; use "
        "the original issue's resolution reason.",
        "",
        "Do not post audio, commercial charts, credentials or personal details. "
        "This public dashboard includes only categories, counts and issue numbers.",
    ]
    return "\n".join(lines) + "\n"


def update_dashboard():
    dashboard = _api(f"repos/{REPOSITORY}/issues/{DASHBOARD_NUMBER}")
    if not is_dashboard(dashboard):
        raise ValueError("The configured dashboard issue is missing its identifying label or marker.")
    pages = _api(f"repos/{REPOSITORY}/issues?state=all&per_page=100", paginate=True)
    issues = [issue for page in pages for issue in page if "pull_request" not in issue]
    # Ensure only fixed, allowlisted labels; never create labels from descriptions.
    existing = _api(f"repos/{REPOSITORY}/labels?per_page=100", paginate=True)
    names = {label["name"] for page in existing for label in page}
    needed = {"feedback-dashboard"}
    for issue in issues:
        metadata = report_metadata(issue)
        if metadata:
            needed.update(triage_labels(metadata))
    for name in sorted(needed - names):
        _api(f"repos/{REPOSITORY}/labels", method="POST", payload={
            "name": name, "color": "75e2c0", "description": "Automated feedback intake"})
    managed = ({"user-feedback"}
               | {"feedback:" + key for key in feedback.CATEGORIES}
               | {"instrument:" + key for key in feedback.INSTRUMENTS}
               | {"severity:" + key for key in feedback.SEVERITIES})
    for issue in issues:
        metadata = report_metadata(issue)
        if not metadata:
            continue
        old = _label_names(issue)
        expected = triage_labels(metadata)
        updated = (old - managed) | expected
        if old != updated:
            _api(f"repos/{REPOSITORY}/issues/{issue['number']}/labels",
                 method="PUT", payload={"labels": sorted(updated)})
    body = render_dashboard(issues)
    if dashboard.get("body") != body:
        _api(f"repos/{REPOSITORY}/issues/{DASHBOARD_NUMBER}", method="PATCH",
             payload={"body": body})
    return dashboard["html_url"]


def main():
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if event_path:
        event = json.loads(Path(event_path).read_text(encoding="utf-8"))
        issue = event.get("issue") or {}
        if is_dashboard(issue):
            return
    print(update_dashboard())


if __name__ == "__main__":
    main()
