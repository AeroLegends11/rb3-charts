import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QBoxLayout, QGroupBox, QLineEdit
import pytest

from rb3charts.gui import ACCENT, BACKGROUND, MUTED, SURFACE, TEXT, Window


@pytest.fixture(scope="session")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app):
    widget = Window()
    widget.show()
    app.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()
    app.processEvents()


def _luminance(color):
    components = QColor(color).getRgbF()[:3]
    linear = [
        value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
        for value in components
    ]
    return sum(value * weight for value, weight in zip(linear, (0.2126, 0.7152, 0.0722)))


@pytest.mark.parametrize(
    "foreground,background",
    [(TEXT, BACKGROUND), (TEXT, SURFACE), (MUTED, BACKGROUND),
     (MUTED, SURFACE), (ACCENT, SURFACE), ("#10241e", ACCENT)],
)
def test_theme_text_has_accessible_contrast(foreground, background):
    light, dark = sorted((_luminance(foreground), _luminance(background)), reverse=True)
    assert (light + 0.05) / (dark + 0.05) >= 4.5


def test_readable_fields_and_review_first_defaults(window):
    assert window.font().pointSize() >= 11
    for field in (*window.findChildren(QLineEdit), window.log):
        assert field.palette().color(QPalette.ColorRole.PlaceholderText) == QColor(MUTED)
    assert window.profile.currentIndex() == 0
    assert window.quality.currentIndex() == 0
    assert window.device.currentIndex() == 0
    assert not window.compile.isChecked()
    assert window.first.text() == "0"
    assert [name for name, box in window.instruments.items() if box.isChecked()] == [
        "guitar", "bass", "drums",
    ]
    assert not window.review.isEnabled()
    assert not window.package.isEnabled()
    assert (Path(__file__).parents[1] / "assets" / "check.svg").is_file()


def test_panels_stack_and_advanced_settings_remain_accessible(window):
    app = QApplication.instance()
    assert window.setup_panels.direction() == QBoxLayout.Direction.LeftToRight
    window.resize(820, 720)
    app.processEvents()
    assert window.setup_panels.direction() == QBoxLayout.Direction.TopToBottom
    advanced = next(group for group in window.findChildren(QGroupBox) if group.isCheckable())
    assert not advanced.isChecked()
    assert not window.bpm.isVisible()
    advanced.setChecked(True)
    app.processEvents()
    assert window.bpm.isVisible()
    assert window.quality.isEnabled()
    window.resize(1100, 960)
    app.processEvents()
    assert window.setup_panels.direction() == QBoxLayout.Direction.LeftToRight


def test_song_selection_preserves_metadata_and_batch_toggle(window, monkeypatch, tmp_path):
    from rb3charts import gui

    song = str(tmp_path / "song.wav")
    monkeypatch.setattr(gui.QFileDialog, "getOpenFileName", lambda *args: (song, "Audio"))
    monkeypatch.setattr(gui.QFileDialog, "getExistingDirectory", lambda *args: str(tmp_path))
    window.title.setText("My title")
    window.select_folder()
    assert window.batch.isChecked()
    assert window.source.text() == str(tmp_path)
    window.select_song()
    assert not window.batch.isChecked()
    assert window.source.text() == song
    assert window.title.text() == "My title"


def test_generate_forwards_settings_without_changing_engine(window, monkeypatch, tmp_path):
    from PySide6.QtCore import QObject, Signal
    from rb3charts import gui

    class StubWorker(QObject):
        message = Signal(str)
        result = Signal(object)
        failed = Signal(str)
        finished = Signal()

        def __init__(self, payload, mode):
            super().__init__()
            self.payload, self.mode = payload, mode
            self.started = False

        def start(self):
            self.started = True

        def isRunning(self):
            return False

    monkeypatch.setattr(gui, "Worker", StubWorker)
    window.source.setText(str(tmp_path / "song.wav"))
    window.artist.setText("Fixture artist")
    window.profile.setCurrentIndex(2)
    window.instruments["keys"].setChecked(True)
    window.generate.click()
    assert window.worker.started
    assert window.worker.mode == "generate"
    assert window.worker.payload.chart_profile == "electronic"
    assert window.worker.payload.artist == "Fixture artist"
    assert window.worker.payload.instruments == ("guitar", "bass", "drums", "keys")
    assert not window.worker.payload.compile
    assert not window.generate.isEnabled()
    window.worker.failed.emit("Fixture failure")
    assert "Fixture failure" in window.log.toPlainText()
    window.worker.finished.emit()
    assert window.generate.isEnabled()


def test_help_opens_local_guides_without_changing_options(window, monkeypatch):
    from PySide6.QtWidgets import QTabWidget, QTextBrowser
    from rb3charts import gui

    def inspect_dialog(dialog):
        tabs = dialog.findChild(QTabWidget)
        assert tabs.count() == 2
        assert tabs.tabText(0) == "Getting started"
        assert tabs.tabText(1) == "Playing on Xbox 360"
        browsers = dialog.findChildren(QTextBrowser)
        assert all(not browser.openLinks() for browser in browsers)
        assert "not a chart editor" in browsers[0].toPlainText()
        assert "does not install songs on a console" in browsers[1].toPlainText()
        assert "model downloads" in browsers[0].toPlainText()
        return 0

    monkeypatch.setattr(gui.QDialog, "exec", inspect_dialog)
    window.help.click()
    assert window.profile.currentIndex() == 0
    assert not window.compile.isChecked()
    assert window.worker is None


def test_missing_help_guide_reports_error(window, monkeypatch, tmp_path):
    from rb3charts import gui

    warnings = []
    monkeypatch.setattr(gui, "ROOT", tmp_path)
    monkeypatch.setattr(gui.QMessageBox, "warning", lambda *args: warnings.append(args))
    window.open_help()
    assert len(warnings) == 1
    assert warnings[0][1] == "Help unavailable"
    assert "Could not open the local guide" in warnings[0][2]


def test_review_button_has_no_mnemonic_underscore(window):
    assert "&" not in window.review.text()


def test_feedback_dialog_saves_report_and_drafts_issue(app, monkeypatch, tmp_path):
    import json
    from urllib.parse import parse_qs, urlsplit
    from rb3charts import gui

    run = tmp_path / "output" / "Artist - Song - 20250101"
    run.mkdir(parents=True)
    (run / "result.json").write_text(json.dumps(
        {"status": "draft_ready", "title": "Song", "artist": "Artist", "chart_profile": "band"}))
    opened = []
    monkeypatch.setattr(gui.QDesktopServices, "openUrl", lambda url: opened.append(url.toString()))
    dialog = gui.FeedbackDialog(None, run, root=tmp_path)
    assert not dialog.issue.isEnabled()
    dialog.category.setCurrentIndex(dialog.category.findData("repeated_riff"))
    dialog.instrument.setCurrentIndex(dialog.instrument.findData("guitar"))
    dialog.start.setText("1:23")
    dialog.description.setPlainText("Chorus riff uses different frets on each repeat.")
    dialog.save.click()
    assert dialog.saved_report["start_seconds"] == 83
    assert len(list((run / "feedback").glob("*.json"))) == 1
    dialog.issue.click()
    query = parse_qs(urlsplit(opened[-1]).query)
    assert "Artist" not in query["title"][0]
    assert str(tmp_path) not in query["body"][0]
    assert "1:23" in query["body"][0]
    assert app.clipboard().text().startswith("<!-- rb3-feedback:")
    dialog.deleteLater()


def test_feedback_dialog_rejects_empty_description(app, monkeypatch, tmp_path):
    from rb3charts import gui

    warnings = []
    monkeypatch.setattr(gui.QMessageBox, "warning", lambda *args: warnings.append(args))
    dialog = gui.FeedbackDialog(None, None, root=tmp_path)
    dialog.save.click()
    assert warnings and dialog.saved_report is None
    assert not (tmp_path / "output" / "feedback" / "index.jsonl").exists()
    dialog.deleteLater()


def test_remote_submission_is_opt_in_and_cancellable(app, monkeypatch, tmp_path):
    from rb3charts import gui, feedback_remote

    monkeypatch.setattr(feedback_remote, "authorize",
                        lambda *a, **k: pytest.fail("No consent was given"))
    dialog = gui.FeedbackDialog(root=tmp_path)
    assert not dialog.auto_submit.isChecked()
    dialog.description.setPlainText("The app button is confusing.")
    dialog.save_report()
    assert dialog.remote_worker is None
    monkeypatch.setattr(gui.QDialog, "exec", lambda self: gui.QDialog.DialogCode.Rejected)
    dialog.submit_report()
    assert dialog.remote_worker is None
    dialog.deleteLater()


def test_auto_submit_runs_only_after_save_and_public_approval(app, monkeypatch, tmp_path):
    from rb3charts import gui, feedback_remote

    dialog = gui.FeedbackDialog(root=tmp_path)
    dialog.auto_submit.setChecked(True)
    dialog.description.setPlainText("The app button is confusing.")
    started = []
    monkeypatch.setattr(gui.QDialog, "exec", lambda self: gui.QDialog.DialogCode.Accepted)
    monkeypatch.setattr(dialog, "start_remote", lambda report_id: started.append(report_id))
    dialog.save_report()
    assert started == [dialog.saved_report["id"]]
    delivery = feedback_remote._delivery_path(started[0], tmp_path)
    assert delivery.exists()
    assert not dialog.description.isEnabled()
    dialog.deleteLater()


def test_report_worker_surfaces_auth_error_and_never_hides_failure(app, monkeypatch, tmp_path):
    from rb3charts import gui, feedback_remote

    def fail(*args, **kwargs):
        raise ValueError("Sign in with gh auth login")

    monkeypatch.setattr(feedback_remote, "sync", fail)
    worker = gui.FeedbackSyncWorker(tmp_path)
    results, errors = [], []
    worker.result.connect(results.append)
    worker.failed.connect(errors.append)
    worker.run()
    assert not results
    assert errors == ["Sign in with gh auth login"]
