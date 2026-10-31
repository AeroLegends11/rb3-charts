from pathlib import Path
import sys
import traceback
from datetime import datetime
from PySide6.QtCore import Qt, QThread, Signal, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QFont, QFontDatabase, QGuiApplication, QIcon, QPalette
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QBoxLayout, QLabel, QPushButton, QLineEdit, QFileDialog, QFormLayout, QComboBox, QCheckBox, QPlainTextEdit, QProgressBar, QGroupBox, QMessageBox, QScrollArea, QDialog, QDialogButtonBox, QTabWidget, QTextBrowser
from .model import Options
from . import feedback
from . import __version__

ROOT=Path(__file__).resolve().parents[1]

BACKGROUND="#101722"
SURFACE="#192434"
TEXT="#f1f5fb"
MUTED="#bac8dc"
ACCENT="#75e2c0"

STYLE = """
QMainWindow, QDialog, QScrollArea, QWidget#canvas { background: #101722; }
QWidget { color: #f1f5fb; }
QScrollArea { border: none; }
QLabel#heading { font-size: 30pt; font-weight: 700; color: #f1f5fb; }
QLabel#subtitle { font-size: 13pt; color: #bac8dc; }
QLabel#build_info, QLabel#hint { color: #bac8dc; }
QLabel#build_info { font-size: 10pt; }
QLabel#status { font-weight: 600; }
QGroupBox {
    background: #192434; border: 1px solid #3c506a; border-radius: 12px;
    margin-top: 14px; padding: 22px 16px 16px; font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin; left: 16px; padding: 0 6px; color: #75e2c0;
}
QLineEdit, QComboBox, QPlainTextEdit, QTextBrowser {
    background: #101b2a; color: #f1f5fb; border: 1px solid #516782;
    border-radius: 8px; padding: 9px 12px; selection-background-color: #75e2c0;
    selection-color: #101722;
}
QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus { border: 2px solid #75e2c0; }
QComboBox { padding-right: 28px; }
QComboBox QAbstractItemView {
    background: #192434; color: #f1f5fb; border: 1px solid #75e2c0;
    selection-background-color: #304b64; selection-color: #ffffff;
}
QPushButton {
    background: #2b4058; color: #f1f5fb; border: 1px solid #627b98;
    border-radius: 8px; padding: 10px 16px; font-weight: 600;
}
QPushButton:hover { background: #395775; border-color: #75e2c0; }
QPushButton:pressed { background: #1d3047; }
QPushButton:focus { border: 2px solid #75e2c0; }
QPushButton:disabled {
    background: #202d3e; color: #a7b6ca; border-color: #45566e;
}
QPushButton#primary {
    background: #75e2c0; color: #10241e; border-color: #75e2c0;
    font-size: 13pt; padding: 14px;
}
QPushButton#primary:hover { background: #9aeed4; }
QPushButton#primary:pressed { background: #55c9a6; }
QPushButton#primary:disabled { background: #334d49; color: #c2d3cf; }
QPushButton#primary:focus { border: 2px solid #ffffff; }
QCheckBox { spacing: 9px; }
QCheckBox::indicator, QGroupBox::indicator {
    width: 18px; height: 18px; border: 1px solid #8095b0;
    border-radius: 4px; background: #101b2a;
}
QCheckBox::indicator:checked, QGroupBox::indicator:checked {
    background: #75e2c0; border-color: #75e2c0; image: url("__CHECK_ICON__");
}
QCheckBox::indicator:checked:disabled { background: #69887f; }
QCheckBox:focus { color: #75e2c0; }
QProgressBar {
    background: #101b2a; border: 1px solid #516782; border-radius: 5px;
    min-height: 8px; max-height: 8px;
}
QProgressBar::chunk { background: #75e2c0; border-radius: 4px; }
QToolTip { background: #24364c; color: #f1f5fb; border: 1px solid #75e2c0; }
QTabWidget::pane { border: 1px solid #516782; }
QTabBar::tab { background: #192434; color: #bac8dc; padding: 10px 16px; }
QTabBar::tab:selected { background: #2b4058; color: #f1f5fb; }
"""

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
        self.setFont(QFont("Segoe UI",11))
        palette=self.palette()
        for role,color in (
            (QPalette.ColorRole.Window,BACKGROUND),
            (QPalette.ColorRole.WindowText,TEXT),
            (QPalette.ColorRole.Base,BACKGROUND),
            (QPalette.ColorRole.AlternateBase,SURFACE),
            (QPalette.ColorRole.Text,TEXT),
            (QPalette.ColorRole.Button,SURFACE),
            (QPalette.ColorRole.ButtonText,TEXT),
            (QPalette.ColorRole.PlaceholderText,MUTED),
            (QPalette.ColorRole.Highlight,ACCENT),
            (QPalette.ColorRole.HighlightedText,BACKGROUND),
        ):
            palette.setColor(role,QColor(color))
        self.setPalette(palette)
        self.resize(1100,960)
        self.setAcceptDrops(True)
        self.worker=None
        self.last_folder=None
        self.last_result=None
        self.review_path=None
        central=QWidget()
        central.setObjectName("canvas")
        scroll=QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(central)
        self.setCentralWidget(scroll)
        layout=QVBoxLayout(central)
        layout.setContentsMargins(28,24,28,24)
        layout.setSpacing(16)
        header=QVBoxLayout()
        header.setSpacing(4)
        heading=QLabel("RB3 CHARTS")
        heading.setObjectName("heading")
        heading_row=QHBoxLayout()
        heading_row.addWidget(heading,1)
        self.help=QPushButton("Getting started")
        self.help.setToolTip("Open the offline beginner guide and Xbox packaging information.")
        self.help.clicked.connect(self.open_help)
        heading_row.addWidget(self.help)
        header.addLayout(heading_row)
        build_info=QLabel(f"v{__version__} · Launched {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        build_info.setObjectName("build_info")
        subtitle=QLabel(
            "Turn a music file into draft note charts for Rock Band 3. "
            "Review the result before creating an Xbox 360 package.")
        subtitle.setWordWrap(True)
        subtitle.setObjectName("subtitle")
        header.addWidget(subtitle)
        header.addWidget(build_info)
        layout.addLayout(header)
        self.setup_panels=QBoxLayout(QBoxLayout.Direction.LeftToRight)
        self.setup_panels.setSpacing(16)
        layout.addLayout(self.setup_panels)
        source_group=QGroupBox("01  /  Choose your music")
        source_layout=QVBoxLayout(source_group)
        source_layout.setSpacing(12)
        self.source=QLineEdit()
        self.source.setPlaceholderText("Drop an audio file here, or select a song…")
        row=QHBoxLayout()
        row.addWidget(self.source)
        source_layout.addLayout(row)
        row=QHBoxLayout()
        select=QPushButton("Select song")
        select.clicked.connect(self.select_song)
        row.addWidget(select)
        batch_button=QPushButton("Select folder")
        batch_button.clicked.connect(self.select_folder)
        row.addWidget(batch_button)
        source_layout.addLayout(row)
        self.batch=QCheckBox("Process every audio file in the selected folder")
        source_layout.addWidget(self.batch)
        source_hint=QLabel(
            "Choose a music file saved on your computer, such as MP3, WAV or FLAC. "
            "Streaming links are not supported. Your music stays on this computer.")
        source_hint.setWordWrap(True)
        source_hint.setObjectName("hint")
        source_layout.addWidget(source_hint)
        metadata_hint=QLabel(
            "Optional song details for display in the game. Add the title, artist, "
            "album and year for a more polished song listing; they are not required "
            "to generate a chart.")
        metadata_hint.setWordWrap(True)
        metadata_hint.setObjectName("hint")
        source_layout.addWidget(metadata_hint)
        fields=QFormLayout()
        fields.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        fields.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        fields.setVerticalSpacing(12)
        fields.setHorizontalSpacing(20)
        self.title=QLineEdit()
        self.artist=QLineEdit()
        self.album=QLineEdit()
        self.year=QLineEdit()
        for label,field in [("Song title",self.title),("Artist",self.artist),("Album",self.album),("Year",self.year)]:
            field.setPlaceholderText(label+" (optional)")
            fields.addRow(label,field)
        source_layout.addLayout(fields)
        self.setup_panels.addWidget(source_group,1)
        instrument_group=QGroupBox("02  /  Chart setup")
        setup_layout=QVBoxLayout(instrument_group)
        setup_layout.setSpacing(12)
        setup_hint=QLabel("Not sure what to choose? Leave Auto selected and keep the default settings.")
        setup_hint.setWordWrap(True)
        setup_hint.setObjectName("hint")
        setup_layout.addWidget(setup_hint)
        profile_row=QFormLayout()
        profile_row.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.profile=QComboBox()
        self.profile.addItems(["Auto — detect instruments and musical roles","Band — guitar and bass instruments","Electronic — lead synth on Guitar, low synth/pulses on Bass"])
        profile_row.addRow("Charting style",self.profile)
        self.profile.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.profile.setMinimumContentsLength(24)
        self.profile.setToolTip(self.profile.currentText())
        self.profile.currentTextChanged.connect(self.profile.setToolTip)
        setup_layout.addLayout(profile_row)
        instruments=QHBoxLayout()
        self.instruments={}
        for name in ("Guitar","Bass","Drums","Keys"):
            box=QCheckBox(name if name!="Keys" else "Keys (experimental)")
            box.setChecked(name!="Keys")
            self.instruments[name.lower()]=box
            instruments.addWidget(box)
        setup_layout.addLayout(instruments)
        difficulties=QLabel("Expert, Hard, Medium and Easy are included for every selected instrument.")
        difficulties.setWordWrap(True)
        difficulties.setObjectName("hint")
        setup_layout.addWidget(difficulties)
        self.setup_panels.addWidget(instrument_group,1)
        advanced=QGroupBox("Advanced settings (optional)")
        advanced.setCheckable(True)
        advanced.setChecked(False)
        advanced_body=QWidget()
        advanced_layout=QVBoxLayout(advanced)
        advanced_hint=QLabel("Leave these unchanged unless you know you need them.")
        advanced_hint.setWordWrap(True)
        advanced_hint.setObjectName("hint")
        advanced_layout.addWidget(advanced_hint)
        advanced_layout.addWidget(advanced_body)
        form=QFormLayout(advanced_body)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        form.setVerticalSpacing(12)
        advanced.toggled.connect(advanced_body.setVisible)
        advanced_body.setVisible(False)
        self.quality=QComboBox()
        self.quality.addItems(["Precision — two separation models, slower","Balanced — one separation model"])
        self.quality.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.quality.setMinimumContentsLength(24)
        self.quality.setToolTip(self.quality.currentText())
        self.quality.currentTextChanged.connect(self.quality.setToolTip)
        form.addRow("Quality",self.quality)
        self.device=QComboBox()
        self.device.addItems(["Automatic GPU / CPU","NVIDIA GPU","CPU"])
        self.device.setToolTip("Auto chooses an available graphics processor or uses the CPU. CPU analysis can take longer.")
        form.addRow("Compute",self.device)
        self.bpm=QLineEdit()
        self.bpm.setPlaceholderText("Automatic variable tempo; enter BPM only if known")
        self.bpm.setToolTip("BPM means beats per minute. Leave blank for automatic timing.")
        form.addRow("BPM override",self.bpm)
        self.first=QLineEdit("0")
        form.addRow("First beat (seconds; manual BPM)",self.first)
        self.stems=QLineEdit()
        self.stems.setPlaceholderText("Optional folder with aligned guitar.wav, bass.wav, drums.wav")
        self.stems.setToolTip("Stems are separate instrument recordings. They must share the original song's start time and length.")
        form.addRow("Existing stems",self.stems)
        self.compile=QCheckBox("Compile immediately (skip the review-first workflow)")
        self.compile.setChecked(False)
        form.addRow(self.compile)
        setup_layout.addWidget(advanced)
        note=QLabel("Precision preserves note timing. Trained drum and pitch models still need listening review.\nPro Drums and keys are experimental. Vocals are not available in this version.")
        note.setWordWrap(True)
        note.setObjectName("hint")
        setup_layout.addWidget(note)
        review_group=QGroupBox("03  /  Generate and review")
        review_layout=QVBoxLayout(review_group)
        review_layout.setSpacing(12)
        layout.addWidget(review_group)
        review_hint=QLabel(
            "Generate a draft, then listen and check it. Automatic notes may need corrections. "
            "First use downloads analysis models and can take longer; progress appears below.")
        review_hint.setWordWrap(True)
        review_hint.setObjectName("hint")
        review_layout.addWidget(review_hint)
        self.generate=QPushButton("Generate visual review")
        self.generate.setObjectName("primary")
        self.generate.clicked.connect(self.start)
        review_layout.addWidget(self.generate)
        self.progress=QProgressBar()
        self.progress.setRange(0,1)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        review_layout.addWidget(self.progress)
        self.status=QLabel("Ready · Drop in a song to begin")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        review_layout.addWidget(self.status)
        self.log=QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1500)
        self.log.setPlaceholderText("Analysis progress and review notes will appear here.")
        self.log.setMinimumHeight(100)
        self.log.setMaximumHeight(120)
        review_layout.addWidget(self.log,1)
        row=QHBoxLayout()
        self.review=QPushButton("Listen and review chart")
        self.review.setEnabled(False)
        self.review.clicked.connect(self.open_review)
        self.review.setToolTip("Check timing, repeated riffs, extra notes and missing notes against the music.")
        row.addWidget(self.review)
        self.package=QPushButton("Package reviewed chart for Xbox")
        self.package.setEnabled(False)
        self.package.clicked.connect(self.package_reviewed)
        self.package.setToolTip(
            "Builds an Xbox 360 game package from the existing draft without analyzing "
            "the song again. Does not transfer it to your console.")
        row.addWidget(self.package)
        review_layout.addLayout(row)
        next_steps=QLabel(
            "Listen for notes that match the music, consistent riffs and quiet sections "
            "without extra notes. Packaging reuses this draft; it does not transfer "
            "the song to your console.")
        next_steps.setWordWrap(True)
        next_steps.setObjectName("hint")
        review_layout.addWidget(next_steps)
        row=QHBoxLayout()
        self.open_output=QPushButton("Open output folder")
        self.open_output.clicked.connect(lambda:QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_folder or ROOT/"output"))))
        row.addWidget(self.open_output)
        self.share=QPushButton("Export for ChatGPT")
        self.share.clicked.connect(self.export_for_chatgpt)
        self.share.setToolTip("Saves a text file for you to review and share manually. Nothing is uploaded automatically.")
        row.addWidget(self.share)
        self.report=QPushButton("Report a problem")
        self.report.setToolTip(
            "Tell the project what sounded wrong or what was confusing. Reports are saved "
            "on this computer and grouped into an improvement list; nothing is uploaded "
            "unless you choose to open a GitHub issue.")
        self.report.clicked.connect(self.open_feedback)
        row.addWidget(self.report)
        review_layout.addLayout(row)
        layout.addStretch()
        self.setStyleSheet(STYLE.replace("__CHECK_ICON__",(ROOT/"assets"/"check.svg").as_posix()))
        for field in (*self.findChildren(QLineEdit),self.log):
            field_palette=field.palette()
            field_palette.setColor(QPalette.ColorRole.PlaceholderText,QColor(MUTED))
            field.setPalette(field_palette)

    def resizeEvent(self,event):
        self.setup_panels.setDirection(
            QBoxLayout.Direction.TopToBottom if event.size().width()<1000
            else QBoxLayout.Direction.LeftToRight)
        super().resizeEvent(event)

    def open_help(self):
        guides=[]
        try:
            for title,filename in (
                ("Getting started","GETTING_STARTED.md"),
                ("Playing on Xbox 360","PLAYING_ON_XBOX.md"),
            ):
                guides.append((title,(ROOT/"docs"/filename).read_text(encoding="utf-8")))
        except OSError as exc:
            QMessageBox.warning(self,"Help unavailable",f"Could not open the local guide:\n{exc}")
            return
        dialog=QDialog(self)
        dialog.setWindowTitle("RB3 Charts · Help")
        dialog.resize(720,640)
        layout=QVBoxLayout(dialog)
        tabs=QTabWidget()
        for title,text in guides:
            browser=QTextBrowser()
            browser.setOpenLinks(False)
            browser.setMarkdown(text)
            tabs.addTab(browser,title)
        layout.addWidget(tabs)
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()
        dialog.deleteLater()

    def open_feedback(self):
        dialog=FeedbackDialog(self,self.last_folder if self.last_result else None)
        dialog.exec()
        dialog.deleteLater()

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
            options=Options(source=Path(self.source.text()),output=ROOT/"output",title=self.title.text(),artist=self.artist.text(),album=self.album.text(),year=self.year.text(),instruments=tuple(i for i,b in self.instruments.items() if b.isChecked()),chart_profile=["auto","band","electronic"][self.profile.currentIndex()],quality="precision" if self.quality.currentIndex()==0 else "balanced",device=["auto","cuda","cpu"][self.device.currentIndex()],bpm=float(self.bpm.text()) if self.bpm.text().strip() else None,first_beat=float(self.first.text()),stems=Path(self.stems.text()) if self.stems.text().strip() else None,compile=self.compile.isChecked())
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
            self.review.setText("Listen and review chart")
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

class FeedbackDialog(QDialog):
    """Collects one report; storage and summaries live in rb3charts.feedback."""
    def __init__(self,parent=None,run_folder=None,root=ROOT):
        super().__init__(parent)
        self.run_folder=run_folder
        self.root=root
        self.saved_report=None
        self.setWindowTitle("RB3 Charts · Report a problem")
        self.resize(640,620)
        layout=QVBoxLayout(self)
        intro=QLabel(
            "Describe one problem per report. Reports are saved on this computer, linked to "
            "the chart you just made, and grouped so problems many people notice are fixed "
            "first. Fixes are tested before they change how charts are made.")
        intro.setWordWrap(True)
        intro.setObjectName("hint")
        layout.addWidget(intro)
        form=QFormLayout()
        self.category=QComboBox()
        for key,label in feedback.CATEGORIES.items():
            self.category.addItem(label,key)
        form.addRow("What kind of problem?",self.category)
        self.instrument=QComboBox()
        for key in feedback.INSTRUMENTS:
            self.instrument.addItem("Any / not sure" if key=="any" else key.title(),key)
        form.addRow("Instrument",self.instrument)
        self.difficulty=QComboBox()
        for key in feedback.DIFFICULTIES:
            self.difficulty.addItem("Any / not sure" if key=="any" else key.title(),key)
        form.addRow("Difficulty",self.difficulty)
        self.severity=QComboBox()
        for key,label in feedback.SEVERITIES.items():
            self.severity.addItem(label,key)
        form.addRow("How serious?",self.severity)
        times=QHBoxLayout()
        self.start=QLineEdit()
        self.start.setPlaceholderText("Start, e.g. 1:23")
        self.end=QLineEdit()
        self.end.setPlaceholderText("End (optional)")
        times.addWidget(self.start)
        times.addWidget(self.end)
        form.addRow("Where in the song?",times)
        self.description=QPlainTextEdit()
        self.description.setPlaceholderText(
            "What happened, and what did you expect? For example: the guitar riff at "
            "the chorus repeats four times in the song but the chart uses different frets each time.")
        self.description.setMinimumHeight(120)
        form.addRow("Details",self.description)
        layout.addLayout(form)
        attached=QLabel(
            f"Linked chart: {Path(run_folder).name}" if run_folder else
            "No chart linked. Make a draft first to attach its settings, or report an app problem.")
        attached.setWordWrap(True)
        attached.setObjectName("hint")
        layout.addWidget(attached)
        self.summary=QLabel()
        self.summary.setWordWrap(True)
        self.summary.setObjectName("hint")
        layout.addWidget(self.summary)
        self.refresh_summary()
        buttons=QDialogButtonBox()
        self.save=buttons.addButton("Save report",QDialogButtonBox.ButtonRole.AcceptRole)
        self.issue=buttons.addButton("Send to project on GitHub",QDialogButtonBox.ButtonRole.ActionRole)
        self.issue.setToolTip("Opens a pre-filled GitHub issue in your browser. You review it before posting.")
        self.issue.setEnabled(False)
        self.queue=buttons.addButton("Open improvement list",QDialogButtonBox.ButtonRole.ActionRole)
        close=buttons.addButton(QDialogButtonBox.StandardButton.Close)
        self.save.clicked.connect(self.save_report)
        self.issue.clicked.connect(self.open_issue)
        self.queue.clicked.connect(self.open_queue)
        close.clicked.connect(self.reject)
        layout.addWidget(buttons)

    def refresh_summary(self):
        summary=feedback.summarize(feedback.load_reports(self.root))
        if not summary["total"]:
            self.summary.setText("No reports saved on this computer yet.")
            return
        top=summary["recurring"][:3]
        text=f"{summary['total']} report(s) saved on this computer."
        if top:
            text+=" Most common: "+"; ".join(
                f"{item['label'].split(' — ')[0]} ({item['instrument']}, {item['count']}×)" for item in top)
        self.summary.setText(text)

    def save_report(self):
        try:
            report=feedback.make_report(
                self.category.currentData(),self.description.toPlainText(),
                instrument=self.instrument.currentData(),difficulty=self.difficulty.currentData(),
                severity=self.severity.currentData(),start=self.start.text(),end=self.end.text(),
                run_folder=self.run_folder)
            feedback.save_report(report,run_folder=self.run_folder,root=self.root)
        except (OSError,ValueError) as exc:
            QMessageBox.warning(self,"Check the report",str(exc))
            return
        self.saved_report=report
        self.issue.setEnabled(True)
        self.save.setText("Saved ✓")
        self.save.setEnabled(False)
        self.refresh_summary()

    def open_issue(self):
        if not self.saved_report:return
        QGuiApplication.clipboard().setText(feedback.issue_body(self.saved_report))
        QDesktopServices.openUrl(QUrl(feedback.issue_url(self.saved_report)))

    def open_queue(self):
        path=feedback.write_improvement_queue(self.root)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

def launch():
    app=QApplication.instance() or QApplication(sys.argv)
    window=Window()
    window.show()
    return app.exec()
