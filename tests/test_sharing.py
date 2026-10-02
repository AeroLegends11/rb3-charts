import json
import sys
import pytest
from rb3charts.sharing import export_chatgpt, MAX_FILE_BYTES


def test_export_includes_project_and_diagnostics_but_not_private_files(tmp_path):
    project=tmp_path/"project"
    (project/"rb3charts").mkdir(parents=True)
    (project/"README.md").write_text("Project overview",encoding="utf-8")
    (project/"rb3charts"/"model.py").write_text("class Note: pass",encoding="utf-8")
    (project/".env").write_text("EXCLUDED_PRIVATE",encoding="utf-8")
    run=tmp_path/"run"
    (run/"analysis").mkdir(parents=True)
    (run/"result.json").write_text(json.dumps({
        "status":"draft_ready","folder":"/home/private/song",
        "source":r"C:\Users\Private\song.wav","api_key":"sensitive-value",
    }),encoding="utf-8")
    (run/"analysis"/"quality-assessment.json").write_text(
        '{"status":"review_closely"}',encoding="utf-8")
    (run/"analysis"/"charts.json").write_text("EXCLUDED_NOTES",encoding="utf-8")
    (run/"song.wav").write_bytes(b"EXCLUDED_AUDIO")
    destination=tmp_path/"share.txt"
    export_chatgpt(destination,run,project_root=project)
    text=destination.read_text(encoding="utf-8")
    for expected in ("Project overview","class Note","draft_ready","review_closely",
                     "SUGGESTED PROMPT","Not available.","[LOCAL PATH]","[REDACTED]"):
        assert expected in text
    for excluded in ("EXCLUDED_","sensitive-value","/home/private",r"C:\\Users"):
        assert excluded not in text


def test_missing_oversized_invalid_and_linked_files_are_explicit(tmp_path):
    project=tmp_path/"project"
    project.mkdir()
    (project/"README.md").write_bytes(b"x"*(MAX_FILE_BYTES+1))
    outside=tmp_path/"private.txt"
    outside.write_text("PRIVATE_TARGET",encoding="utf-8")
    (project/"pyproject.toml").symlink_to(outside)
    run=tmp_path/"run"
    run.mkdir()
    (run/"result.json").write_text("invalid json",encoding="utf-8")
    destination=tmp_path/"share.txt"
    export_chatgpt(destination,run,project_root=project)
    text=destination.read_text(encoding="utf-8")
    assert "exceeds" in text and "link outside" in text and "invalid text/JSON" in text
    assert "PRIVATE_TARGET" not in text


def test_export_does_not_overwrite_existing_file(tmp_path):
    destination=tmp_path/"review.txt"
    destination.write_text("Keep user review",encoding="utf-8")
    with pytest.raises(FileExistsError):
        export_chatgpt(destination,project_root=tmp_path)
    assert destination.read_text(encoding="utf-8")=="Keep user review"


def test_requires_individual_run_and_supports_project_only(tmp_path):
    with pytest.raises(ValueError,match="individual chart"):
        export_chatgpt(tmp_path/"invalid.txt",tmp_path)
    assert not (tmp_path/"invalid.txt").exists()
    export_chatgpt(tmp_path/"project.txt",project_root=tmp_path)
    assert "No chart run selected" in (tmp_path/"project.txt").read_text(encoding="utf-8")


def test_cli_export_does_not_import_pipeline(tmp_path,monkeypatch):
    from rb3charts.cli import main
    monkeypatch.setitem(sys.modules,"rb3charts.pipeline",None)
    destination=tmp_path/"share.txt"
    monkeypatch.setattr(sys,"argv",["rb3-charts","share-chatgpt","--output",str(destination)])
    assert main()==0
    assert destination.is_file()
