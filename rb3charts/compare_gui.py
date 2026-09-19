"""Small drag-and-drop launcher for video chart comparisons."""
from pathlib import Path
import webbrowser

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton, QVBoxLayout, QWidget

from .compare import compare_charts
from .encore import compare_with_encore
from .video_compare import create_video_comparison, start_comparison_server


class FolderEdit(QLineEdit):
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setPlaceholderText("Drop an RB3 Charts review folder here")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            path = paths[0]
            self.setText(str(path.parent if path.name.lower() == "review.html" else path))


class ComparisonWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RB3 Chart Comparison")
        self.resize(760, 330)
        self.servers = []
        root = QWidget(); layout = QVBoxLayout(root)
        intro = QLabel("Drop a review folder and click Start to find a matching Chorus Encore reference and compare it automatically. You can override it with a local MIDI, or use a YouTube link for guided visual review.")
        intro.setWordWrap(True); layout.addWidget(intro)
        form = QFormLayout(); self.folder = FolderEdit(); self.reference = QLineEdit(); self.url = QLineEdit()
        self.reference.setPlaceholderText("Optional notes.mid for automatic measurements")
        self.url.setPlaceholderText("https://www.youtube.com/watch?v=…")
        row = QHBoxLayout(); row.addWidget(self.folder); browse = QPushButton("Browse…"); browse.clicked.connect(self._browse); row.addWidget(browse)
        reference_row = QHBoxLayout(); reference_row.addWidget(self.reference); reference_browse = QPushButton("Browse…"); reference_browse.clicked.connect(self._browse_reference); reference_row.addWidget(reference_browse)
        form.addRow("Generated review folder", row); form.addRow("Reference MIDI (automatic)", reference_row); form.addRow("YouTube video (guided)", self.url); layout.addLayout(form)
        note = QLabel("Leave both reference fields blank for automatic Chorus Encore lookup. Only notes.mid is retrieved from its SNG package; audio is not downloaded."); note.setWordWrap(True); layout.addWidget(note)
        create = QPushButton("Start comparison"); create.clicked.connect(self._create); layout.addWidget(create, alignment=Qt.AlignRight)
        self.setCentralWidget(root)

    def _browse(self):
        selected = QFileDialog.getExistingDirectory(self, "Choose RB3 Charts review folder")
        if selected:
            self.folder.setText(selected)

    def _browse_reference(self):
        selected, _ = QFileDialog.getOpenFileName(self, "Choose reference chart MIDI", filter="MIDI files (*.mid *.midi)")
        if selected:
            self.reference.setText(selected)

    def _create(self):
        try:
            if self.reference.text().strip():
                result = compare_charts(self.folder.text(), self.reference.text())
                webbrowser.open(Path(result["output"]["html"]).as_uri())
                QMessageBox.information(self, "Automatic comparison complete", f"Report:\n{result['output']['html']}")
                return
            if not self.url.text().strip():
                result = compare_with_encore(self.folder.text())
                webbrowser.open(Path(result["output"]["html"]).as_uri())
                reference = result["encore_reference"]
                QMessageBox.information(self, "Encore comparison complete", f"Compared against {reference['charter']} chart #{reference['chart_id']}.\n\nReport:\n{result['output']['html']}")
                return
            result = create_video_comparison(self.folder.text(), self.url.text())
            server, url = start_comparison_server(result)
            self.servers.append(server)
            webbrowser.open(result["video_url"])
            webbrowser.open(url)
        except Exception as exc:
            QMessageBox.critical(self, "Could not create comparison", str(exc)); return
        QMessageBox.information(self, "Guided comparison created", "The normal YouTube page and generated review were opened separately. Keep this launcher open while reviewing.")


def launch():
    app = QApplication.instance() or QApplication([])
    window = ComparisonWindow(); window.show()
    return app.exec()
