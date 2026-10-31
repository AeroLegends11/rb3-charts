from rb3charts import feedback, feedback_dashboard as dashboard, feedback_remote as remote
import pytest


def issue(number, *, instrument="drums", severity="minor", state="open", reason=None):
    report = feedback.make_report("timing", "Notes are late", instrument=instrument,
                                  severity=severity)
    return {"number": number, "state": state, "state_reason": reason,
            "body": remote.public_payload(report)["body"], "labels": []}


def test_dashboard_groups_open_reports_and_preserves_closed_reason():
    first = issue(1, severity="blocking")
    second = issue(2)
    closed = issue(3, state="closed", reason="not_planned")
    output = dashboard.render_dashboard([first, second, closed])
    assert "3 total | 2 open | 1 closed | 1 blocking" in output
    assert "not_planned: 1" in output
    assert "| drums | 2 |" in output
    assert output.index("- #1") < output.index("- #2")
    assert "- #3" not in output
    assert "Notes are late" not in output


def test_dashboard_ignores_pull_requests_test_reports_and_duplicate_ids():
    first = issue(1)
    duplicate = {**first, "number": 2}
    pull = {**issue(3), "pull_request": {}}
    test = {**issue(4), "labels": [{"name": "feedback-test"}]}
    assert "1 total" in dashboard.render_dashboard([first, duplicate, pull, test])


def test_label_classification_is_fixed_and_not_user_text():
    labels = dashboard.triage_labels(remote.report_metadata(issue(1, severity="major")))
    assert labels == {"user-feedback", "feedback:timing", "instrument:drums", "severity:major"}


def test_updater_preserves_unrelated_labels_and_issues(monkeypatch):
    first = issue(1)
    first["labels"] = [{"name": "help wanted"}, {"name": "severity:blocking"}]
    unrelated = {"number": 2, "body": "Not an app report", "labels": [{"name": "user-feedback"}]}
    calls = []
    saved = {"number": dashboard.DASHBOARD_NUMBER, "body": dashboard.DASHBOARD_MARKER + "\nOld",
             "labels": ["feedback-dashboard"], "html_url": "https://github.com/example/dashboard"}
    def api(endpoint, **kwargs):
        calls.append((endpoint, kwargs))
        if endpoint.endswith(f"/issues/{dashboard.DASHBOARD_NUMBER}") and not kwargs:
            return saved
        if "state=all" in endpoint:
            return [[first, unrelated]]
        if "labels?per_page" in endpoint:
            return [[]]
        return {}
    monkeypatch.setattr(dashboard, "_api", api)
    dashboard.update_dashboard()
    update = next(kwargs for endpoint, kwargs in calls if endpoint.endswith("/1/labels"))
    assert "help wanted" in update["payload"]["labels"]
    assert "severity:blocking" not in update["payload"]["labels"]
    assert not any(endpoint.endswith("/2/labels") for endpoint, _ in calls)


def test_existing_dashboard_does_not_generate_repeated_updates(monkeypatch):
    first = issue(1)
    first["labels"] = sorted(dashboard.triage_labels(remote.report_metadata(first)))
    saved = {"number": 7, "body": dashboard.render_dashboard([first]),
             "labels": ["feedback-dashboard"], "html_url": "https://github.com/example/7"}
    def api(endpoint, **kwargs):
        if endpoint.endswith(f"/issues/{dashboard.DASHBOARD_NUMBER}") and not kwargs:
            return saved
        if "state=all" in endpoint:
            return [[first, saved]]
        if "labels?per_page" in endpoint:
            return [[{"name": name} for name in first["labels"] + ["feedback-dashboard"]]]
        raise AssertionError("No write should be needed")
    monkeypatch.setattr(dashboard, "_api", api)
    assert dashboard.update_dashboard() == saved["html_url"]


def test_missing_dashboard_marker_stops_without_creating_another_issue(monkeypatch):
    calls = []
    def api(endpoint, **kwargs):
        calls.append((endpoint, kwargs))
        return {"body": "Unrelated issue", "labels": []}
    monkeypatch.setattr(dashboard, "_api", api)
    with pytest.raises(ValueError, match="identifying"):
        dashboard.update_dashboard()
    assert len(calls) == 1
