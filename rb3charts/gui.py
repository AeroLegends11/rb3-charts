from pathlib import Path
import sys
import traceback
from datetime import datetime
from PySide6.QtCore import Qt, QThread, Signal, QUrl
from PySide6.QtGui import QDesktopServices, QFontDatabase, QIcon
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QFileDialog, QFormLayout, QComboBox, QCheckBox, QPlainTextEdit, QProgressBar, QGroupBox, QMessageBox, QScrollArea
from .model import Options
from . import __version__

ROOT=Path(__file__).resolve().parents[1]

class Worker(QThread):
    message=Signal(str)
    result=Signal(object)
    failed=Signal(str)
    def __init__(self,payload,mode):
        super().__init__()
        self.payload,self.mode=payload,mode
    def run(self):
        try:
            from .pipeline import generate,batch,package_existing
            action={"generate":generate,"batch":batch,"package":package_existing}[self.mode]
            self.result.emit(action(self.payload,self.message.emit))
        except Exception:
            self.failed.emit(traceback.format_exc())

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        # Explicit font registration also supports Qt's headless/offscreen renderer.
        for font in ("segoeui.ttf","segoeuib.ttf"):
            path=Path("C:/Windows/Fonts")/font
            if path.exists():QFontDatabase.addApplicationFont(str(path))
        self.setWindowTitle(f"RB3 Charts • Local authoring • v{__version__}")
        icon=ROOT/"assets"/"rb3-chart-maker.ico"
        if icon.exists():self.setWindowIcon(QIcon(str(icon)))
        self.resize(930,860)
        self.setAcceptDrops(True)
        self.worker=None
        self.last_folder=None
        self.last_result=None
        self.review_path=None
        central=QWidget()
        scroll=QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(central)
        self.setCentralWidget(scroll)
        layout=QVBoxLayout(central)
        layout.setContentsMargins(24,18,24,18)
        layout.setSpacing(10)
        heading=QLabel("RB3 CHARTS")
        heading.setObjectName("heading")
        layout.addWidget(heading)
        build_info=QLabel(f"v{__version__} · Launched {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        build_info.setObjectName("build_info")
        layout.addWidget(build_info)
        subtitle=QLabel("Your songs. Four difficulties. A better first draft.")
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)
        self.source=QLineEdit()
        self.source.setPlaceholderText("Drop an audio file here, or select a song…")
        row=QHBoxLayout()
        row.addWidget(self.source)
        select=QPushButton("Select song")
        select.clicked.connect(self.select_song)
        row.addWidget(select)
        batch_button=QPushButton("Select folder")
        batch_button.clicked.connect(self.select_folder)
        row.addWidget(batch_button)
        layout.addLayout(row)
        self.batch=QCheckBox("Process every audio file in the selected folder")
        layout.addWidget(self.batch)
        fields=QFormLayout()
        fields.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        fields.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.title=QLineEdit()
        self.artist=QLineEdit()
        self.album=QLineEdit()
        self.year=QLineEdit()
        for label,field in [("Song title",self.title),("Artist",self.artist),("Album",self.album),("Year",self.year)]:
            fields.addRow(label,field)
        layout.addLayout(fields)
        profile_row=QFormLayout()
        self.profile=QComboBox()
        self.profile.addItems(["Auto — detect instruments and musical roles","Band — guitar and bass instruments","Electronic — lead synth on Guitar, low synth/pulses on Bass"])
        profile_row.addRow("Charting style",self.profile)
        layout.addLayout(profile_row)
        instrument_group=QGroupBox("Instruments · Expert, Hard, Medium and Easy included")
        instruments=QHBoxLayout(instrument_group)
        self.instruments={}
        for name in ("Guitar","Bass","Drums","Keys"):
            box=QCheckBox(name if name!="Keys" else "Keys (experimental)")
            box.setChecked(name!="Keys")
            self.instruments[name.lower()]=box
            instruments.addWidget(box)
        layout.addWidget(instrument_group)
        advanced=QGroupBox("Analysis & timing · Precision mode selected")
        advanced.setCheckable(True)
        advanced.setChecked(False)
        advanced_body=QWidget()
        advanced_layout=QVBoxLayout(advanced)
        advanced_layout.addWidget(advanced_body)
        form=QFormLayout(advanced_body)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        advanced.toggled.connect(advanced_body.setVisible)
        advanced_body.setVisible(False)
        self.quality=QComboBox()
        self.quality.addItems(["Precision — two separation models, slower","Balanced — one separation model"])
        form.addRow("Quality",self.quality)
        self.device=QComboBox()
        self.device.addItems(["Automatic GPU / CPU","NVIDIA GPU","CPU"])
        form.addRow("Compute",self.device)
        self.bpm=QLineEdit()
        self.bpm.setPlaceholderText("Automatic variable tempo; enter BPM only if known")
        form.addRow("BPM override",self.bpm)
        self.first=QLineEdit("0")
        form.addRow("First beat (seconds; manual BPM)",self.first)
        self.stems=QLineEdit()
        self.stems.setPlaceholderText("Optional folder with aligned guitar.wav, bass.wav, drums.wav")
        form.addRow("Existing stems",self.stems)
        self.compile=QCheckBox("Compile immediately (skip the review-first workflow)")
        self.compile.setChecked(False)
        form.addRow(self.compile)
        self.clean_guitar=QCheckBox("Suppress vocal/drum/bass bleed in the guitar stem")
        self.clean_guitar.setChecked(True)
        form.addRow(self.clean_guitar)
        layout.addWidget(advanced)
        note=QLabel("Precision preserves note timing. Trained drum and pitch models still need listening review.\nPro Drums and keys are experimental. Vocals are not available in this version.")
        note.setWordWrap(True)
        note.setObjectName("hint")
        layout.addWidget(note)
        self.generate=QPushButton("Generate visual review")
        self.generate.setObjectName("primary")
        self.generate.clicked.connect(self.start)
        layout.addWidget(self.generate)
        self.progress=QProgressBar()
        self.progress.setRange(0,1)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        layout.addWidget(self.progress)
        self.status=QLabel("Ready · Drop in a song to begin")
        layout.addWidget(self.status)
        self.log=QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1500)
        self.log.setMinimumHeight(90)
        self.log.setMaximumHeight(110)
        layout.addWidget(self.log,1)
        row=QHBoxLayout()
        self.review=QPushButton("Listen & review chart")
        self.review.setEnabled(False)
        self.review.clicked.connect(self.open_review)
        row.addWidget(self.review)
        self.package=QPushButton("Package reviewed chart for Xbox")
        self.package.setEnabled(False)
        self.package.clicked.connect(self.package_reviewed)
        row.addWidget(self.package)
        self.open_output=QPushButton("Open output folder")
        self.open_output.clicked.connect(lambda:QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_folder or ROOT/"output"))))
        row.addWidget(self.open_output)
        layout.addLayout(row)
        self.share=QPushButton("Export for ChatGPT")
        self.share.clicked.connect(self.export_for_chatgpt)
        layout.addWidget(self.share)
        self.setStyleSheet('''QMainWindow,QWidget{background:#111a28;color:#e5edf8;font:14px "Segoe UI"}QLabel#heading{font-size:28px;font-weight:700;letter-spacing:3px;color:#6ee7c5}QLabel#build_info{font-size:11px;color:#7c8ea8;letter-spacing:1px}QLabel#subtitle{font-size:18px;color:#b1c2d9}QLabel#hint{color:#a6b6ce;font-size:12px}QLineEdit,QComboBox,QPlainTextEdit{background:#1b293b;border:1px solid #34475e;border-radius:6px;padding:7px}QPushButton{background:#263a52;border:1px solid #435e7a;border-radius:7px;padding:9px 16px}QPushButton:hover{background:#33516e}QPushButton:disabled{color:#697b90}QPushButton#primary{background:#51c9a5;color:#10241e;font-weight:700;font-size:16px;padding:12px}QGroupBox{border:1px solid #34475e;border-radius:8px;margin-top:8px;padding-top:16px}QGroupBox::title{subcontrol-origin:margin;left:10px;padding:0 4px}''')

    def export_for_chatgpt(self):
        from .sharing import export_chatgpt
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self,"Export for ChatGPT","Wait until generation or packaging finishes.")
            return
        answer=QMessageBox.question(
            self,"Include chart diagnostics?",
            "The export includes project code and development progress. Include a chart run too?\n"
            "Only text diagnostics are included, not audio or MIDI. Nothing is uploaded automatically.")
        run=None
        if answer==QMessageBox.StandardButton.Yes:
            folder=QFileDialog.getExistingDirectory(
                self,"Select one chart run containing result.json",str(self.last_folder or ROOT/"output"))
            if not folder:return
            run=Path(folder)
        name,_=QFileDialog.getSaveFileName(
            self,"Save ChatGPT context",str(ROOT/"rb3-chatgpt-context.txt"),"Text (*.txt)")
        if not name:return
        try:
            export_chatgpt(Path(name),run)
        except (OSError,ValueError) as exc:
            QMessageBox.warning(self,"Export failed",str(exc))
            return
        QMessageBox.information(
            self,"Export ready",
            "Review the text for private information first; redaction is best-effort.\n"
            "Transfer it to your phone (for example via OneDrive), attach it in ChatGPT, "
            "and use the suggested prompt at the top. Nothing was uploaded.")

    def select_song(self):
        name,_=QFileDialog.getOpenFileName(self,"Select audio",str(ROOT),"Audio (*.mp3 *.wav *.flac *.ogg *.m4a *.aiff *.wma)")
        if name:
            self.source.setText(name)
            self.batch.setChecked(False)
    def select_folder(self):
        name=QFileDialog.getExistingDirectory(self,"Select audio folder",str(ROOT))
        if name:
            self.source.setText(name)
            self.batch.setChecked(True)
    def dragEnterEvent(self,event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    def dropEvent(self,event):
        file=event.mimeData().urls()[0].toLocalFile()
        self.source.setText(file)
        self.batch.setChecked(Path(file).is_dir())
    def start(self):
        try:
            if not self.source.text().strip():
                raise ValueError("Select a song first.")
            options=Options(source=Path(self.source.text()),output=ROOT/"output",title=self.title.text(),artist=self.artist.text(),album=self.album.text(),year=self.year.text(),instruments=tuple(i for i,b in self.instruments.items() if b.isChecked()),chart_profile=["auto","band","electronic"][self.profile.currentIndex()],quality="precision" if self.quality.currentIndex()==0 else "balanced",device=["auto","cuda","cpu"][self.device.currentIndex()],bpm=float(self.bpm.text()) if self.bpm.text().strip() else None,first_beat=float(self.first.text()),stems=Path(self.stems.text()) if self.stems.text().strip() else None,compile=self.compile.isChecked(),clean_guitar_stem=self.clean_guitar.isChecked())
        except ValueError as exc:
            QMessageBox.warning(self,"Check input",str(exc))
            return
        self.generate.setEnabled(False)
        self.review.setEnabled(False)
        self.package.setEnabled(False)
        self.progress.setRange(0,0)
        self.log.clear()
        self.worker=Worker(options,"batch" if self.batch.isChecked() else "generate")
        self.worker.message.connect(self.log.appendPlainText)
        self.worker.result.connect(self.finished)
        self.worker.failed.connect(self.failed)
        self.worker.finished.connect(self.reset)
        self.status.setText("Working · Model downloads can take several minutes on first use")
        self.worker.start()
    def finished(self,result):
        if isinstance(result,list):
            good=sum(r["status"] in ("draft_ready","compiled_needs_playtest") for r in result)
            ratings={key:sum(r.get("assessment",{}).get("status")==key for r in result) for key in ("close_enough","review_closely","needs_work")}
            self.status.setText(f"Batch finished · {good}/{len(result)} ready · {ratings['close_enough']} close · {ratings['review_closely']} review closely · {ratings['needs_work']} need work")
            self.last_folder=ROOT/"output"
            reports=sorted((ROOT/"output").glob("batch_review_*.md"),key=lambda path:path.stat().st_mtime,reverse=True)
            self.review_path=reports[0] if reports else None
            self.review.setText("Open batch review notes")
            self.review.setEnabled(self.review_path is not None)
            for item in result:
                label=item.get("assessment",{}).get("label","Generation failed")
                self.log.appendPlainText(f"{item.get('title',Path(item.get('source','Unknown')).stem)}: {label}")
        else:
            self.last_folder=Path(result["folder"])
            self.last_result=result
            self.review_path=Path(result["review"])
            self.review.setText("Listen & review chart")
            self.status.setText({"compiled_needs_playtest":"Package compiled · Listening review and Xbox playtest needed","draft_ready":"Draft ready · Review timing before compilation","compile_failed":"Charts ready, compilation failed · See log"}[result["status"]])
            self.log.appendPlainText(f"Charting style used: {result.get('chart_profile','unknown')}")
            if result.get("missing_instruments"):
                self.status.setText(self.status.text()+" · Omitted: "+", ".join(result["missing_instruments"]))
            for name,counts in result["counts"].items():
                self.log.appendPlainText(f"{name}: {counts}")
            for warning in result["warnings"]:
                self.log.appendPlainText("Review: "+warning)
            self.review.setEnabled(True)
            self.package.setEnabled(result["status"] in ("draft_ready","compile_failed"))
    def open_review(self):
        if self.review_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.review_path)))
    def package_reviewed(self):
        if not self.last_folder:return
        self.generate.setEnabled(False)
        self.package.setEnabled(False)
        self.progress.setRange(0,0)
        self.status.setText("Packaging reviewed chart · Analysis will not run again")
        self.worker=Worker(self.last_folder,"package")
        self.worker.message.connect(self.log.appendPlainText)
        self.worker.result.connect(self.finished)
        self.worker.failed.connect(self.failed)
        self.worker.finished.connect(self.reset)
        self.worker.start()
    def failed(self,error):
        self.status.setText("Generation failed · See details below")
        self.log.appendPlainText(error)
    def reset(self):
        self.progress.setRange(0,1)
        self.progress.setValue(1)
        self.generate.setEnabled(True)
        if self.last_result:
            self.package.setEnabled(self.last_result.get("status") in ("draft_ready","compile_failed"))
    def closeEvent(self,event):
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self,"Generation in progress","Please leave this window open until generation finishes.")
            event.ignore()
        else:
            event.accept()

def launch():
    app=QApplication.instance() or QApplication(sys.argv)
    window=Window()
    window.show()
    return app.exec()
