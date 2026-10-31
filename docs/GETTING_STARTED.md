# Getting started with RB3 Charts

RB3 Charts turns a music file into a **draft chart**: the colored notes you play
in Rock Band 3. It is a local Windows tool, not an official Harmonix product.
Automatic charts can contain mistakes and need listening review.

## 1. Choose your music

Once setup is complete, open **Launch.cmd**. Click **Select song** or drop a
music file into the window. Supported files include MP3, WAV, FLAC, OGG, M4A,
AIFF and WMA. Use music you have permission to use. Streaming links are not
supported; the app does not download songs from streaming services.

**Select folder** processes the audio files in a folder. Start with one song
while learning the tool.

Song title, artist, album and year are optional details for the in-game song
listing. Filling them in makes the listing more polished; it does not improve
the generated notes or make the custom an official release.

## 2. Keep the defaults

Leave charting style on **Auto** unless you want a specific interpretation:
**Band** follows guitar and bass instruments; **Electronic** puts lead synth
parts on Guitar and low synth or pulses on Bass.

Guitar, Bass and Drums are selected by default. Each selected instrument gets
Expert, Hard, Medium and Easy. Keys and Pro Drums are experimental; vocals
are not available.

You do not need to open **Advanced settings**. Precision is the default analysis
mode; it uses more separation models and takes longer than Balanced.
Automatic compute uses an available GPU (graphics processor), or the CPU.
CPU analysis can take longer. BPM means beats per minute: leave it blank for
automatic timing. Stems are separate instrument recordings; only supply them
if they share the original song's start time and length.

## 3. Generate a draft

Click **Generate visual review**. The app analyzes the audio and creates notes.
First use downloads analysis models and can take several minutes or longer.
The moving progress bar means the task is running, not a percentage complete.
Keep the window open and watch the progress messages.

Your source music is not uploaded by the generation workflow. Internet access
is needed for setup and model downloads. Generation creates new output files;
it does not edit the original song.

## 4. Listen before packaging

When the draft is ready, click **Listen & review chart**. The preview opens in
your browser and shows a note highway alongside the music. For a batch, this
button opens batch review notes instead.

Check whether attacks match the music, notes are missing or extra, repeated
riffs feel consistent, and quiet sections remain quiet. Look for awkward chords
or notes held after the sound ends. Check the lower difficulties too.

The preview helps you find problems; it is not a chart editor. If you find an
issue, record the instrument and timestamp. Corrections may require different
analysis settings, a chart editor, or help from an experienced charter.
A "draft ready" message or passing automatic checks does not certify accuracy,
fun or console compatibility.

## 5. Create the game package

After review, click **Package reviewed chart for Xbox**. This builds the package
from the existing draft without analyzing the song again. It does **not**
transfer it to your console. Read the **Playing on Xbox 360** help tab before
trying to play it.

**Open output folder** shows the generated files. The per-song folder contains
the preview, review notes, diagnostics and, after successful packaging, the
Xbox package. Keep this folder so you can review or package again.

## If something goes wrong

- **No song selected:** choose a local audio file before generating.
- **Long first run:** model downloads and CPU analysis can take time. Read the
  progress messages; no fixed completion time is guaranteed.
- **Compilation failed:** the draft may still be reviewable. Read the error
  details before retrying; Onyx/Magma packaging tools must be installed.
- **Need help:** note what you clicked and the exact error. **Export for ChatGPT**
  saves project context and optional chart diagnostics to a text file. Review
  it for private information before sharing manually; redaction is best-effort.
  Exporting does not automatically upload anything.
- **Something sounds wrong or is confusing:** click **Report a problem**. Pick
  what kind of problem it is, the instrument, roughly where in the song (for
  example `1:23`), and describe it in your own words. The report is saved on this
  computer with the chart you just made. **Send to project on GitHub** opens a
  pre-filled issue in your browser that you can read before posting.

Setup on a new computer is a separate step: follow the README's Windows
installation instructions before launching.
