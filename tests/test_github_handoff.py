import json

from rb3charts.github_handoff import create_github_handoff, submit_to_github_copilot


def sample_report():
    return {
        "provenance": {
            "generated_midi": r"C:\\Private\\Song\\notes.mid",
            "reference_midi": r"C:\\Private\\Reference\\notes.mid",
            "generated_sha256": "a" * 64, "reference_sha256": "b" * 64,
            "instrument": "guitar", "difficulty": "expert",
            "alignment_coordinate": "second", "alignment": "automatic-onsets-only",
            "alignment_offset_seconds": .125,
        },
        "summary": {
            "generated_attacks": 100, "reference_attacks": 90, "extra_attacks": 20,
            "missing_attacks": 10, "attack_precision": .8, "attack_recall": .7,
            "attack_f1": .7467, "median_timing_error_seconds": .03,
            "p95_timing_error_seconds": .09, "exact_lane_rate": .4,
            "relative_shape_rate": .6, "chord_class_rate": .8,
            "sustain_mae_seconds": .12, "recurring_pattern_coverage": .5,
        },
        "sections": [
            {"start_second": 0, "end_second": 10, "generated_attacks": 10, "reference_attacks": 10, "matched_attacks": 9, "extra_attacks": 1, "missing_attacks": 1, "exact_lane_rate": .8},
            {"start_second": 50, "end_second": 60, "generated_attacks": 30, "reference_attacks": 10, "matched_attacks": 8, "extra_attacks": 22, "missing_attacks": 2, "exact_lane_rate": .1},
        ],
    }


def test_handoff_is_safe_and_prioritizes_worst_region(tmp_path):
    reference = {"provider": "chorus-encore", "chart_id": 123, "charter": "Harmonix",
                 "md5": "c" * 32, "chart_hash": "d" * 64, "tempo_map_hash": "e" * 64}
    path = create_github_handoff(sample_report(), reference, tmp_path, "Example", "Artist")
    text = path.read_text(encoding="utf-8")
    assert "Artist - Example" in text
    assert "50-60 seconds" in text
    assert text.index("50-60 seconds") < text.index("0-10 seconds")
    assert r"C:\\Private" not in text
    assert "Do not special-case" in text
    assert json.loads((tmp_path / "github-handoff.json").read_text())["issue_title"].startswith("Improve Guitar")


def test_submission_is_idempotent(tmp_path, monkeypatch):
    reference = {"chart_id": 1}
    handoff = create_github_handoff(sample_report(), reference, tmp_path, "Example", "Artist")
    calls = []
    class Completed:
        returncode = 0; stdout = '{"html_url":"https://github.com/example/repo/issues/7"}\n'; stderr = ""
    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command[:3] == ["git", "config", "--get"]:
            result = Completed(); result.stdout = "https://github.com/example/repo.git\n"; return result
        if command[:2] == ["git", "ls-remote"]:
            return Completed()
        return Completed()
    monkeypatch.setattr("rb3charts.github_handoff.subprocess.run", run)
    assert submit_to_github_copilot(handoff) == "https://github.com/example/repo/issues/7"
    assert submit_to_github_copilot(handoff) == "https://github.com/example/repo/issues/7"
    api_calls = [call for call in calls if call[0][:2] == ["gh", "api"]]
    assert len(api_calls) == 1
    payload = json.loads(api_calls[0][1]["input"])
    assert payload["assignees"] == ["copilot-swe-agent[bot]"]
    assert payload["agent_assignment"]["base_branch"]
