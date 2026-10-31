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
