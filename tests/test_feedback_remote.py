import json
import subprocess

import pytest

from rb3charts import feedback, feedback_remote as remote


def report():
    return feedback.make_report("timing", "Notes are late at /root/private/song.wav",
                                instrument="drums", severity="major")


def issue(report_id, *, state="open", number=42, body=None):
    return {"number": number, "state": state, "state_reason": None,
            "html_url": f"https://github.com/{remote.REPOSITORY}/issues/{number}",
            "body": body or f"<!-- rb3-feedback:{report_id} category=timing instrument=drums severity=major -->"}


def test_public_payload_omits_local_song_metadata_and_paths():
    item = report()
    item["run"] = {"title": "Private title", "artist": "Private artist",
                   "folder_name": "private-folder", "chart_profile": "band"}
    payload = remote.public_payload(item)
    assert "Private" not in json.dumps(payload)
    assert "private-folder" not in payload["body"]
    assert "/root/private" not in payload["body"]
    assert "Chart profile: band" in payload["body"]
    assert remote.report_metadata({"body": payload["body"]})["id"] == item["id"]


def test_description_cannot_inject_report_metadata():
    item = report()
    fake = f"<!-- rb3-feedback:{item['id']} category=app instrument=any severity=blocking -->"
    item["description"] += "\n" + fake
    assert remote.report_metadata({"body": remote.public_payload(item)["body"]})["category"] == "timing"
    assert remote.report_metadata({"body": fake + "\n" + fake}) is None
    assert remote.report_metadata({"body": fake.replace("category=app", "category=bad")}) is None


def test_no_consent_means_no_network(monkeypatch, tmp_path):
    monkeypatch.setattr(remote, "_api", lambda *a, **k: pytest.fail("Unexpected network request"))
    with pytest.raises(ValueError, match="approve"):
        remote.submit(report()["id"], root=tmp_path)
    assert remote.sync(root=tmp_path) == []


def test_submit_stores_receipt_and_never_resends(monkeypatch, tmp_path):
    item = report()
    remote.authorize(item, root=tmp_path)
    calls = []

    def api(endpoint, **kwargs):
        calls.append((endpoint, kwargs))
        return [[]] if kwargs.get("paginate") else issue(item["id"])

    monkeypatch.setattr(remote, "_api", api)
    result = remote.submit(item["id"], root=tmp_path)
    assert result["status"] == "sent"
    assert len(calls) == 2 and calls[1][1]["method"] == "POST"
    assert calls[1][1]["payload"] == remote.public_payload(item)
    assert remote.submit(item["id"], root=tmp_path) == result
    assert len(calls) == 2


def test_timeout_retry_recovers_issue_without_duplicate(monkeypatch, tmp_path):
    item = report()
    remote.authorize(item, root=tmp_path)
    calls = []

    def failing_api(endpoint, **kwargs):
        return [[]] if kwargs.get("paginate") else (_ for _ in ()).throw(ValueError("timeout"))

    monkeypatch.setattr(remote, "_api", failing_api)
    with pytest.raises(ValueError, match="timeout"):
        remote.submit(item["id"], root=tmp_path)

    def recovered_api(endpoint, **kwargs):
        calls.append(endpoint)
        return [[issue(item["id"])]]

    monkeypatch.setattr(remote, "_api", recovered_api)
    assert remote.submit(item["id"], root=tmp_path)["status"] == "sent"
    assert len(calls) == 1


def test_sync_refreshes_resolution_and_does_not_send_local_only_reports(monkeypatch, tmp_path):
    item = report()
    feedback.save_report(report(), root=tmp_path)
    remote.authorize(item, root=tmp_path)
    monkeypatch.setattr(remote, "_api", lambda endpoint, **kwargs:
                        [[]] if kwargs.get("paginate") else issue(item["id"]))
    remote.submit(item["id"], root=tmp_path)
    closed = {**issue(item["id"], state="closed"), "state_reason": "not_planned"}
    monkeypatch.setattr(remote, "_api", lambda *a, **k: closed)
    result = remote.sync(root=tmp_path)
    assert len(result) == 1
    assert result[0]["issue"]["state"] == "closed"
    assert result[0]["issue"]["state_reason"] == "not_planned"


@pytest.mark.parametrize("failure", [FileNotFoundError(), subprocess.TimeoutExpired("gh", 60)])
def test_missing_cli_and_timeout_are_actionable(monkeypatch, failure):
    def run(*args, **kwargs):
        raise failure
    monkeypatch.setattr(remote.subprocess, "run", run)
    with pytest.raises(ValueError):
        remote._api("repos/example/issues")


def test_api_error_keeps_report_queued_and_redacts_credentials(monkeypatch, tmp_path):
    item = report()
    remote.authorize(item, root=tmp_path)
    monkeypatch.setattr(remote.subprocess, "run", lambda *a, **k:
                        subprocess.CompletedProcess([], 1, "", "password=secret-value"))
    with pytest.raises(ValueError) as exc:
        remote.submit(item["id"], root=tmp_path)
    assert "secret-value" not in str(exc.value)
    path = remote._delivery_path(item["id"], tmp_path)
    assert json.loads(path.read_text())["status"] == "pending"


@pytest.mark.parametrize("response", [
    {"number": 42, "state": "open", "html_url": "https://evil.example"},
    {"number": True, "state": "open", "html_url": "https://github.com/example"},
])
def test_invalid_receipt_is_not_success(response):
    with pytest.raises(ValueError):
        remote._receipt(response)


def test_approved_payload_is_immutable(tmp_path):
    item = report()
    first = remote.authorize(item, root=tmp_path)
    item["description"] = "Completely different"
    assert remote.authorize(item, root=tmp_path) == first


def test_recent_uncertain_submission_waits_for_github_visibility(monkeypatch, tmp_path):
    from datetime import datetime, timezone
    item = report()
    delivery = remote.authorize(item, root=tmp_path)
    delivery.update(status="uncertain", attempted_at=datetime.now(timezone.utc).isoformat())
    remote._write_delivery(remote._delivery_path(item["id"], tmp_path), delivery)
    calls = []
    def api(endpoint, **kwargs):
        calls.append(kwargs)
        assert kwargs.get("paginate"), "No second POST should be made"
        return [[]]
    monkeypatch.setattr(remote, "_api", api)
    with pytest.raises(ValueError, match="Wait two minutes"):
        remote.submit(item["id"], root=tmp_path)
    assert len(calls) == 1
