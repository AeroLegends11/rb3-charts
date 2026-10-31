import json

import pytest

from rb3charts import feedback


def _report(category="timing", instrument="drums", **kwargs):
    return feedback.make_report(category, "Notes drift late in the bridge.", instrument=instrument, **kwargs)


@pytest.mark.parametrize("text,seconds", [("", None), ("83.5", 83.5), ("1:23", 83), ("1:02:03", 3723)])
def test_parse_time(text, seconds):
    assert feedback.parse_time(text) == seconds


@pytest.mark.parametrize("kwargs", [
    {"category": "nope"}, {"start": "abc"}, {"start": "", "end": "1:00"},
    {"start": "2:00", "end": "1:00"}, {"severity": "huge"},
])
def test_invalid_reports_are_rejected(kwargs):
    args = {"category": "timing", **kwargs}
    with pytest.raises(ValueError):
        feedback.make_report(args.pop("category"), "Something is wrong here", **args)


def test_short_description_rejected():
    with pytest.raises(ValueError):
        feedback.make_report("timing", "  ")


def test_save_links_run_and_builds_recurring_queue(tmp_path):
    run = tmp_path / "output" / "run1"
    run.mkdir(parents=True)
    (run / "result.json").write_text(json.dumps({"status": "draft_ready", "title": "T",
                                                  "assessment": {"status": "review_closely"}}))
    first = _report(run_folder=run)
    assert first["run"] == {"folder_name": "run1", "status": "draft_ready", "title": "T",
                            "artist": None, "chart_profile": None, "assessment": "review_closely"}
    feedback.save_report(first, run_folder=run, root=tmp_path)
    feedback.save_report(_report(severity="blocking"), root=tmp_path)
    feedback.save_report(_report("app", "any"), root=tmp_path)
    assert (run / "feedback" / f"{first['id']}.json").is_file()
    reports = feedback.load_reports(tmp_path)
    summary = feedback.summarize(reports)
    assert summary["total"] == 3
    assert summary["recurring"] == [{"category": "timing", "instrument": "drums", "count": 2,
                                     "label": feedback.CATEGORIES["timing"], "blocking": 1}]
    queue = (tmp_path / "output" / "feedback" / "improvement-queue.md").read_text(encoding="utf-8")
    assert "2× Timing" in queue and "1 blocking" in queue


def test_corrupt_index_lines_are_skipped(tmp_path):
    folder = feedback.feedback_root(tmp_path)
    folder.mkdir(parents=True)
    (folder / "index.jsonl").write_text('not json\n{"category": "bogus"}\n' + json.dumps(_report()) + "\n")
    assert len(feedback.load_reports(tmp_path)) == 1


def test_issue_draft_redacts_paths_and_trims_long_urls(tmp_path):
    report = feedback.make_report("app", "Crash opening C:\\Users\\me\\song.mp3 " + "x" * 3900)
    body = feedback.issue_body(report)
    assert "C:\\Users" not in body and "[LOCAL PATH]" in body
    assert "No audio, MIDI or chart files are attached" in body
    assert len(feedback.issue_url(report)) <= feedback.MAX_URL + 500
