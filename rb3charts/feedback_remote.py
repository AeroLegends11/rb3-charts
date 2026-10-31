"""Consent-based GitHub issue delivery using the user's GitHub CLI sign-in."""
import json
from datetime import datetime, timezone
import re
import subprocess

from . import feedback
from .sharing import _redact

REPOSITORY = "AeroLegends11/rb3-charts"
INBOX_URL = f"https://github.com/{REPOSITORY}/issues?q=label%3Auser-feedback"
REPORT_ID = re.compile(r"\d{8}T\d{6}Z-[a-f0-9]{6}")
MARKER = re.compile(
    r"^<!-- rb3-feedback:(\d{8}T\d{6}Z-[a-f0-9]{6}) "
    r"category=(\w+) instrument=(\w+) severity=(\w+) -->$", re.MULTILINE
)


def public_payload(report):
    """Build the exact public text the user reviews; omit song and folder names."""
    report_id = report["id"]
    if not REPORT_ID.fullmatch(report_id):
        raise ValueError("Invalid feedback report ID.")
    category, instrument, severity = (
        report["category"], report["instrument"], report["severity"]
    )
    if (category not in feedback.CATEGORIES or instrument not in feedback.INSTRUMENTS
            or severity not in feedback.SEVERITIES):
        raise ValueError("Invalid feedback category, instrument or severity.")
    safe = dict(report)
    run = report.get("run") or {}
    safe["run"] = {key: run.get(key) for key in ("status", "chart_profile", "assessment")}
    body = feedback.issue_body(safe)
    body = body.replace("Local output folder name: not attached\n", "")
    marker = (f"<!-- rb3-feedback:{report_id} category={category} "
              f"instrument={instrument} severity={severity} -->")
    # The authoritative marker is generated here, not accepted from free text.
    body = MARKER.sub("[report marker removed]", body)
    return {
        "title": _redact(f"[Feedback] {feedback.CATEGORIES[category].split(' — ')[0]}"),
        "body": f"{marker}\n\n{body}",
    }


def report_metadata(issue):
    matches = MARKER.findall(issue.get("body") or "")
    if len(matches) != 1:
        return None
    report_id, category, instrument, severity = matches[0]
    if (category not in feedback.CATEGORIES or instrument not in feedback.INSTRUMENTS
            or severity not in feedback.SEVERITIES):
        return None
    return {"id": report_id, "category": category,
            "instrument": instrument, "severity": severity}


def _delivery_path(report_id, root):
    if not REPORT_ID.fullmatch(report_id):
        raise ValueError("Invalid feedback report ID.")
    return feedback.feedback_root(root) / "delivery" / f"{report_id}.json"


def _write_delivery(path, delivery):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(delivery, indent=2), encoding="utf-8")
    temporary.replace(path)


def authorize(report, *, root=feedback.ROOT):
    """Queue only the public payload explicitly approved by the user."""
    path = _delivery_path(report["id"], root)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    delivery = {"report_id": report["id"], "status": "pending",
                "payload": public_payload(report)}
    _write_delivery(path, delivery)
    return delivery


def _api(endpoint, *, method="GET", payload=None, paginate=False):
    command = ["gh", "api", "--hostname", "github.com", "--method", method, endpoint]
    if paginate:
        command += ["--paginate", "--slurp"]
    if payload is not None:
        command += ["--input", "-"]
    try:
        result = subprocess.run(
            command, input=json.dumps(payload) if payload is not None else None,
            capture_output=True, text=True, encoding="utf-8", timeout=60,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except FileNotFoundError as exc:
        raise ValueError(
            "Install GitHub CLI (cli.github.com), then run 'gh auth login' once. "
            "Your report is still saved locally; you can also use the browser button."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise ValueError("GitHub did not respond in time. The report remains queued; retry later.") from exc
    if result.returncode:
        raise ValueError(
            "GitHub could not complete the request. Check your connection and run "
            "'gh auth status' (or 'gh auth login' to sign in).\n" + _redact(result.stderr)[:1500]
        )
    try:
        return json.loads(result.stdout)
    except ValueError as exc:
        raise ValueError("GitHub returned an invalid response; no delivery was confirmed.") from exc


def _receipt(issue):
    if not isinstance(issue, dict):
        raise ValueError("GitHub returned an invalid issue; no delivery was confirmed.")
    number, state, url = issue.get("number"), issue.get("state"), issue.get("html_url")
    if (not isinstance(number, int) or isinstance(number, bool) or number < 1
            or state not in ("open", "closed")
            or url != f"https://github.com/{REPOSITORY}/issues/{number}"):
        raise ValueError("GitHub returned an unexpected issue; no delivery was confirmed.")
    return {"number": number, "state": state, "url": url,
            "state_reason": issue.get("state_reason")}


def submit(report_id, *, root=feedback.ROOT):
    """Send an approved payload, recovering previous deliveries before retrying."""
    path = _delivery_path(report_id, root)
    if not path.is_file():
        raise ValueError("Review and approve public sharing before sending this report.")
    delivery = json.loads(path.read_text(encoding="utf-8"))
    if delivery.get("status") == "sent":
        return delivery
    pages = _api(f"repos/{REPOSITORY}/issues?state=all&per_page=100", paginate=True)
    if not isinstance(pages, list) or not all(isinstance(page, list) for page in pages):
        raise ValueError("Could not check existing issues; the report was not sent.")
    issue = None
    for page in pages:
        for candidate in page:
            if not isinstance(candidate, dict):
                raise ValueError("Could not read existing issues; the report was not sent.")
            if "pull_request" in candidate:
                continue
            metadata = report_metadata(candidate)
            if metadata and metadata["id"] == report_id:
                issue = candidate
                break
        if issue:
            break
    if issue is None:
        if delivery.get("status") == "uncertain":
            elapsed = (datetime.now(timezone.utc)
                       - datetime.fromisoformat(delivery["attempted_at"])).total_seconds()
            if elapsed < 120:
                raise ValueError(
                    "The previous submission was not confirmed. Wait two minutes before "
                    "retrying so GitHub can make it visible; this avoids duplicate issues.")
        delivery.update(status="uncertain", attempted_at=datetime.now(timezone.utc).isoformat())
        _write_delivery(path, delivery)
        issue = _api(f"repos/{REPOSITORY}/issues", method="POST", payload=delivery["payload"])
    receipt = _receipt(issue)
    delivery.update(status="sent", issue=receipt)
    _write_delivery(path, delivery)
    return delivery


def sync(*, root=feedback.ROOT):
    """Retry approved pending reports and refresh previously sent issue states."""
    directory = feedback.feedback_root(root) / "delivery"
    deliveries = []
    for path in sorted(directory.glob("*.json")):
        delivery = json.loads(path.read_text(encoding="utf-8"))
        if delivery.get("status") in ("pending", "uncertain"):
            delivery = submit(delivery["report_id"], root=root)
        elif delivery.get("status") == "sent":
            issue = _api(f"repos/{REPOSITORY}/issues/{delivery['issue']['number']}")
            delivery["issue"] = _receipt(issue)
            _write_delivery(path, delivery)
        else:
            raise ValueError(f"Unknown delivery status in {path.name}; nothing was sent.")
        deliveries.append(delivery)
    return deliveries
