"""Small drag-and-drop launcher for video chart comparisons."""
from pathlib import Path
import webbrowser

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton, QVBoxLayout, QWidget

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
        self.resize(720, 230)
        self.servers = []
        root = QWidget(); layout = QVBoxLayout(root)
        intro = QLabel("Compare a generated review with an official YouTube gameplay chart.\nThe generated chart and reference video remain separate.")
        intro.setWordWrap(True); layout.addWidget(intro)
        form = QFormLayout(); self.folder = FolderEdit(); self.url = QLineEdit()
        self.url.setPlaceholderText("https://www.youtube.com/watch?v=…")
        row = QHBoxLayout(); row.addWidget(self.folder); browse = QPushButton("Browse…"); browse.clicked.connect(self._browse); row.addWidget(browse)
        form.addRow("Review folder", row); form.addRow("Official chart video", self.url); layout.addLayout(form)
        create = QPushButton("Open comparison workspace"); create.clicked.connect(self._create); layout.addWidget(create, alignment=Qt.AlignRight)
        self.setCentralWidget(root)

    def _browse(self):
        selected = QFileDialog.getExistingDirectory(self, "Choose RB3 Charts review folder")
        if selected:
            self.folder.setText(selected)

    def _create(self):
        try:
            result = create_video_comparison(self.folder.text(), self.url.text())
            server, url = start_comparison_server(result)
            self.servers.append(server)
            webbrowser.open(url)
        except Exception as exc:
            QMessageBox.critical(self, "Could not create comparison", str(exc)); return
        QMessageBox.information(self, "Comparison created", f"Opened:\n{url}\n\nKeep this launcher open while reviewing.")


def launch():
    app = QApplication.instance() or QApplication([])
    window = ComparisonWindow(); window.show()
    return app.exec()
