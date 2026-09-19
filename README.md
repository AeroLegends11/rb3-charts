# RB3 Charts

Development: read [AGENTS.md](AGENTS.md) for commands and branch rules,
[the ordered development plan](docs/DEVELOPMENT_PLAN.md) for the next milestones,
and [the repository review](docs/REPOSITORY_REVIEW.md) before the first GitHub upload.

A local Windows application that turns audio into draft Rock Band 3 guitar,
bass and drum charts at **Expert, Hard, Medium and Easy**, with optional
five-lane keys. It compiles Xbox 360 CON packages using Onyx and Magma.

The priority is timing precision and musical consistency. Automatic transcription
is imperfect: a package that compiles is not proof that its chart is accurate or
fun. The current build has compiled both generated test audio and the supplied
“Empty Walls” by Serj Tankian recording. Console playability and musical accuracy
still require a listening review and Xbox playtest.

## Start

On this computer dependencies are installed in `.venv`. Double-click
**Launch.cmd**, select or drop a song, enter metadata, and click **Generate charts**.

- Precision mode uses six-stem Demucs for guitar/keys and fine-tuned Demucs for
  drums/bass/vocals. It is slower and caches its results.
- Balanced mode uses the six-stem model only.
- NVIDIA GPU is selected automatically when available; CPU can be selected.
- All four difficulties are always generated. Keys are optional and experimental.
- **Auto** charting style is the default. It compares separated guitar, piano and
  other-stem energy plus genre metadata. Prominent guitar uses literal Band
  tracking; synth-led material uses high-toned lead roles on Guitar and low synth
  or bass pulses on Bass. Manual Band and Electronic overrides remain available.
- Generation now stops at a reviewable draft by default. Use **Listen & review
  chart** to watch a five-lane perspective note highway with the music, inspect
  measure lines and hear separate guitar, bass, or drum onset clicks. Highway
  spacing can be spread out for dense charts, and pitched-instrument sustains
  remain visible at the strike line until their release.
- After review, **Package reviewed chart for Xbox** runs Onyx/Magma without
  repeating separation or transcription. Immediate compilation remains an
  advanced opt-in.
- **Open output folder** exposes the package, MIDI, diagnostics and logs.

Supported input: WAV, FLAC, MP3, OGG, M4A, AIFF and WMA through bundled FFmpeg.
Audio stays on this machine. Internet is used for installing tools and downloading
model weights; source songs are not uploaded. Spotify links/downloads are not input files.

## Install on another Windows machine

Install Python 3.11 or 3.12 and Git, then run:

```powershell
.\Setup.ps1 -Python 'C:\Path\To\python.exe'
# Without an NVIDIA GPU:
.\Setup.ps1 -Python 'C:\Path\To\python.exe' -CpuOnly
```

The setup installs an isolated environment and the official Onyx 20251011 CLI.
The CUDA PyTorch download is approximately 3.5 GB. Precision separation downloads
multiple model checkpoints on first use and can require several GB of free space.
Keep the workspace and its `.venv` locally available if stored in OneDrive.

Basic Pitch is deliberately installed without its obsolete TensorFlow dependency:
this application explicitly uses its supplied ONNX model. `pip check` may report
the uninstalled TensorFlow dependency; the actual ONNX inference path is tested.
Setuptools is pinned below version 81 because ADTOF's resampling dependency still
uses the removed `pkg_resources` compatibility module.

## Command line

```powershell
.\.venv\Scripts\python.exe -m rb3charts.cli generate 'C:\Music\Empty Walls.flac' --artist 'Serj Tankian' --title 'Empty Walls'
.\.venv\Scripts\python.exe -m rb3charts.cli generate 'C:\Music\EDM Track.flac' --chart-profile electronic
.\.venv\Scripts\python.exe -m rb3charts.cli batch 'C:\Music\To Chart'
.\.venv\Scripts\python.exe -m rb3charts.cli package 'C:\Path\To\Reviewed_Output'
.\.venv\Scripts\python.exe -m rb3charts.cli replay 'C:\Path\To\Existing_Output'
.\.venv\Scripts\python.exe -m rb3charts.cli demo --compile
.\.venv\Scripts\python.exe -m pytest -q
```

Useful options:

- `--device cpu` / `--device cuda`
- `--quality precision` / `--quality balanced`
- `--chart-profile electronic` for EDM, dubstep and synth-led pop
- `--chart-profile auto` is the default; each decision is recorded in
  `analysis/chart-profile.json`
- `--instruments guitar bass drums keys`
- `--no-compile` for review/export only
- `--bpm 120 --first-beat 0.75` for a known constant tempo
- `--beats-file beats.json` for an edited variable tempo map
- `--stems 'C:\Music\MyStems'` to reuse properly aligned stems
- `--confidence 0.45` to omit more uncertain notes
- `--onyx 'C:\Tools\onyx.exe'` to select another Onyx CLI

Supplied stems must use the same start time and length as the original audio and
be named `guitar.wav`, `bass.wav`, `drums.wav`, and optionally `piano.wav`. They
must not be trimmed independently. Optional `drums.mid` in that folder replaces
automatic drum transcription with General MIDI drum events at source-audio times.
Only supply a dedicated drum MIDI, not a mixed-instrument MIDI.

A beat file is JSON: `{"beats": [0.75, 1.25, 1.76, 2.28], "numerator": 4}`.
Each entry is the source-audio time of a consecutive quarter note. Include the
whole song for accurate variable timing. The first entry defines downbeat phase.

### Electronic charting style

Electronic mode treats Rock Band tracks as playable musical roles. Guitar follows
a monophonic lead selected from the separated `other`, piano and guitar stems.
Bass combines the isolated bass stem with low synth notes from `other`. When a
sustained low note spans a kick without a new bass attack, the chart may retrigger
that pitch on the kick so the lane follows an audible dance pulse. These generated
pulses are provenance-labelled in the transcription analysis. Drums continue to
use the trained percussion model, including programmed percussion.

This mode does not claim that a synth is physically a guitar or bass. It aims to
assign the hook, low end and beat to the three controllers in a musical way.
Electronic Guitar now chooses a voice over a sequence of candidate groups,
balancing model confidence with source and register continuity. Strong sustained
polyphony can survive as chords; weak isolated register spikes are rejected.
Electronic recordings currently use a stable studio-tempo grid
by default. A whole-song refinement removes the coarse tempo-bin error that can
otherwise accumulate across a track; note attacks retain their measured
source-audio timing.
Initial electronic lead lanes are centered on the weighted middle of the hook;
a subsequent phrase-authoring pass replaces that global mapping with local
melodic fingering so the global range does not flatten a local riff.

Before difficulty reduction, every Expert chart passes a beat-aware playability
gate. Electronic pitched roles normally permit two attacks per quarter note;
Expert Guitar now permits up to four for detected recurring figures and relaxes
spacing within those supported figures. Confidence, beat alignment and detected
sustains determine which events survive. The decisions and hold counts are saved
in `analysis/playability.json`; the gate never creates hold notes.

For electronic pitched roles, the transcription onset channel is also checked
inside long activations and around repeated same-pitch effects. A strong second
attack splits the event into separate gems. Sustain releases use pitch-band
energy: the hold ends after 58 ms below both 24 dB beneath its local peak and
3 dB above the preceding noise floor. Brief audible decays of at least 90 ms
are retained as sustains on electronic bass. Guitar uses the 220 ms authoring
threshold to avoid decorating rapid phrases with tiny holds. This prevents one model activation from
swallowing repeated synth hits or holding long after the audible tone decays.

Band parts use a separate 220 ms sustain threshold. Short guitar and bass
articulations remain taps, and a sustain shortened below that threshold by the
next attack is converted back to a tap so small tails do not congest the highway.

Band guitar and bass lanes use weighted pitch percentiles from the commonly
played register. This preserves riff contour across the fretboard without a
single high solo note or octave error compressing lower passages into one lane.

Band Guitar also compares Basic Pitch's polyphonic candidates with a high-
resolution monophonic pYIN contour from the isolated guitar stem. The contour
chooses among weak simultaneous fundamentals and harmonics, anchors ambiguous
octaves to the recent lead register, and corrects stable low-confidence pitch
substitutions before frets are assigned. Two strongly supported simultaneous
pitches remain available as a real chord. A sequential voice pass then removes
rapid unsupported pitch spikes between neighboring notes that agree, while
retaining coherent high-note runs and repeated attacks. Every run records
candidate, correction, duplicate, and sequential-rejection counts in
`analysis/lead-focus.json`.

Before crowded notes are removed, the playability gate searches for recurring
four-note riffs and drum grooves by interval or pad shape plus beat spacing.
Repeated figures receive a small retention preference so the density limiter
keeps recognizable musical patterns instead of choosing isolated high-confidence
artifacts. Each run writes pattern counts, occurrence times, lane use and melodic
contour checks to `analysis/patterns.json`.

After chord cleanup, Band Guitar also performs conservative phrase stabilization.
When the same four-note pitch sequence occurs at least three times with compatible
beat spacing, the program derives one compact fingering and reuses it for those
occurrences. Equal pitches always receive equal frets inside the shape. The rule
can recognize the same established riff after a root or key change when three
pitches agree, and it can repair two low-confidence pitch errors in a repetition
immediately beside a confirmed occurrence. A confirmed transposition supplies a
local two-measure pitch-to-fret vocabulary: weak nearby octave or harmonic errors
fold into that vocabulary, including uncertain overlapping harmonics that would
otherwise appear as a chord. Strong unmatched notes, distant sections, one-off
fills and chord-based riffs remain independent. Its exact changes are saved to
`analysis/pattern-stabilization.json`.

Ordinary Band guitar, bass and keys passages are limited to three attacks per
quarter beat with a minimum spacing guard. A fourth attack is retained only when
it belongs to a detected recurring fast riff, allowing deliberate sixteenth-note
figures while removing one-off transcription congestion.

Band Guitar receives a dedicated authoring pass before difficulty reduction.
Detected attacks within 28 ms of a sixteenth-note or triplet grid are aligned,
green/orange-spanning chord shapes are rewritten to standard playable shapes,
and Hard removes every fourth attack from long eighth-note runs at 160 BPM or
faster. Hard also replaces green/orange chords and keeps at most two-note chords.
Each eight-measure section maps its active guitar register independently using
the pitches carrying meaningful local weight. Common pitches move through
adjacent frets, while rare low or high events use the remaining edge lanes.
Narrow riffs therefore keep natural motion without stray detections compressing
the section. During a phrase, pitch changes of four semitones or less move at
most one fret and intervals below an octave move at most two. Wide jumps remain
available for octave-scale changes and phrase breaks. Hard also wraps quick
green/orange jumps inward.

Every song now receives `analysis/quality-assessment.json` and an editable
`review_notes.md`. Automatic triage checks congestion, hold and chord frequency,
lane coverage, recurring patterns and pitch-to-lane contour. Folder batches also
produce `batch_review_*.md`, which sorts the listening work into Close enough to
review, Review closely and Needs work while keeping a place for timestamped
human feedback.

Guitar assessment also estimates Rock Band's 0–6 difficulty tier from the
hardest four-second density, chord complexity and repeated-pattern support. The
result includes the tier number, label and a suggested numeric RB3 rank; generated
`song.ini` files use that tier instead of a fixed Guitar difficulty. This is a
chart-based estimate for sorting and metadata, not a substitute for playing it.

When `reference/style-profile.json` is present, Band Guitar, Bass and Keys also
conform uncertain details to the 90th-percentile ranges learned from finished
reference charts. Excess low-confidence chords, three-note chords and short
sustains are simplified before lower difficulties are produced. Each decision
is recorded in `analysis/reference-conformance.json`.

## What precision means here

1. Isolate instruments before pitch transcription.
2. Use Basic Pitch ONNX for pitched instruments and trained ADTOF for drums.
3. Retain detected timing, with bounded local onset refinement. Do not force all
   notes onto a constant-tempo quantization grid.
4. Convert audio seconds to beats and back using one shared piecewise tempo map.
   MIDI serialization is independently tested against original note times.
5. Map pitches deterministically so repeated riffs keep consistent fingering.
6. Derive lower difficulties from the previous chart, with source-event ancestry,
   reduced density, smaller chords and fewer fret lanes.
7. Validate structure and compile with the actual external game toolchain.

When a file contains a BPM tag, the app compares it with audio analysis and can
interpret common half-time tags (for example, 80 as 160 BPM). It then estimates
the quarter-note phase from percussion. The review still asks the user to confirm
the measure/downbeat phase.

Drum-fill activation is a narrow exception: Magma requires its ending hit on a
beat boundary. Existing hits may move by at most 12 ms, consistently across all
difficulties, and every adjustment is recorded in `analysis/timing-adjustments.json`.
If that is insufficient, that fill is omitted instead of distorting the rhythm.

ADTOF distinguishes kick, snare, toms, hi-hat and cymbals. Tom heights are currently
collapsed onto blue, general cymbals onto green, with experimental tom markers.
It cannot reliably distinguish ride from crash. Guitar chord and sustain detection
can also be wrong, especially on distorted or densely mixed tracks.

## Output

Each run gets a new timestamped folder, so existing charts are preserved:

```text
output/Artist_Song_TIMESTAMP/
  source/              decoded audio and source fingerprint
  analysis/            beat map, note events, model activations, stems manifest
  midi/notes.mid       generated authoring MIDI
  audio/song.wav       source mix with the same lead-in as MIDI
  audio/timing-audit.wav
  metadata/            song and generation options
  import/              song.ini, MIDI, audio for Onyx import
  onyx-project/        imported project and compilation intermediates
  rb3/*.con            compiled package when successful
  review.html          offline scrolling chart preview
  validation_report.txt
  result.json          explicit completion / failure / playtest status
  logs/generation.log
```

`cache/stems` is keyed by decoded-audio hash and quality settings. Source/stems
are retained, which uses disk space but permits troubleshooting without repeating
separation. Original files are never overwritten. Batch processing continues
after a failed song and writes a report. Cached stems are referenced in the
manifest rather than copied into every run.

The package currently uses the original stereo mix as backing audio. Instrument
audio does not mute independently when the player misses. This preserves source
quality without separation artifacts or misleading stem-mix gains.

## Current limits and next validation

- “Empty Walls” has compiled successfully; listening and Xbox testing are still needed.
- Automatic tempo can choose half/double tempo or the wrong bar phase. Meter
  defaults to 4/4; use manual overrides for other meters or difficult timing.
- Practice sections use neutral numbered labels; verse/chorus/solo recognition
  is not claimed. No automatic solo markers yet.
- Vocals, lyric alignment, harmonies and pro keys/guitar/bass are not implemented.
- Album art and advanced metadata editing are not yet exposed in the GUI.
- Difficulty tiers in the song list are provisional, not calibrated ratings.
- Compiler validation covers RB3 structure; listening and console testing remain
  the acceptance check. Generated venue/animation work is delegated to Magma.

If generation fails, inspect `logs/generation.log` and `result.json`. CUDA errors
can be retried with CPU. If all notes disappear, review the stem and threshold;
the app does not manufacture notes to pass validation. A failed compilation is
reported as `compile_failed`, while keeping the authoring files and review page.

## Code and research

- `audio.py`: decoding, cached Demucs separation, shared audio lead-in.
- `timing.py`: tempo analysis and bounded onset refinement.
- `transcribe.py`: Basic Pitch, ADTOF and supplied GM drum transcription.
- `chart.py`: pitch-to-lane mapping, reductions and overdrive opportunities.
- `midi.py`: sourced RB MIDI encodings and precise tempo serialization.
- `validate.py`: structural checks independent of compiler execution.
- `export.py`: Onyx import/build integration.
- `preview.py`, `gui.py`, `cli.py`: local review and entry points.
- `events.py`, `phrases.py`: candidate evidence, tracing and phrase fingering.
- `pipeline.py`: engine orchestration, independent of the GUI.
- `assessment.py`, `reference.py`, `regression.py`: quality triage, replay
  regression snapshots/metrics and reference statistics.
- `tests/`: synthetic regression fixtures; no commercial audio required.
- `reference/style-profile.json`: compact runtime statistics; full local reference
  reports and downloaded chart repositories are excluded from Git.

`input/`, `cache/`, `output/`, `tools/`, `.research/` and `Ready for Xbox/` remain
local. Git exclusions do not prevent normal reading, generation or packaging.
`main` holds the tested baseline; use a `codex/<topic>` branch for experiments
and review before merging. A clean clone must run setup and supply local songs;
it does not contain the existing machine's audio, models or generated drafts.

New transcription backends should return source-timed events with confidence and
stable source IDs, then use the existing mapping/reduction/export stages. Never
introduce backend-specific offsets without an independently measured timing test.

### Pitched-event evidence and diagnosis

Guitar, bass and keys now pass through a lane-independent event-evidence stage
before role selection and mapping. Each Basic Pitch candidate retains its original
model timing, refined timing, pitch confidence, model onset strength, local RMS,
relative level, amplitude rise, beat alignment and repeated-phrase support.
Quiet audible attacks no longer disappear solely because they fall below the old
fixed RMS cutoff. Confident legato notes can survive without a sharp attack;
weak candidates require attack support. A weaker candidate can be retained when
an audible repeated four-event phrase supports it. Beat alignment alone never
creates or accepts a note, and this stage never snaps timing or assigns lanes.

Inspect `analysis/<instrument>-<stem>-events.json` for accepted **and rejected**
candidates, with definite/probable/uncertain/artifact classifications and reasons.
Scores are heuristic evidence, not calibrated probabilities. These reports begin
with candidates returned by Basic Pitch; they cannot expose notes that the model
never proposed. Phrase matching currently uses recurring pitch sets and beat
spacing, not verse/chorus recognition.

`analysis/<instrument>-event-trace.csv` and the corresponding JSON join those
candidates to the final Expert notes by source ID. They distinguish detection
rejections, role-selection filtering, later mapping/playability filtering, and
surviving notes with their final timing and lanes. Role-generated bass pulses are
explicitly marked as derived rather than audio detections. Drum transcription
continues to use its existing separate backend and diagnostics.

### Guitar phrase authoring

Both Band and Electronic Guitar now receive a phrase pass after playability and
before lower difficulties are derived. It groups single-note passages at rests,
chords and eight-beat boundaries. Phrases with up to five distinct pitches use
distinct adjacent frets. Larger pitch ranges use a whole-sequence cost balancing
melodic direction, repeated pitches, hand movement and local register. Identical
relative-pitch/rhythm phrases reuse their fingering, including transpositions.
Chords and meaningful long holds remain; short tails become taps. Every attack's
time, source ID and detected pitch are preserved by this pass.

`analysis/phrase-authoring.json` records each phrase's pitches, old/new frets,
reused fingering, tail changes and contour diagnostics. This is a heuristic
authoring layer, not a trained imitation of professional charters. It cannot
recover notes that transcription missed, and ranges wider than five frets may
still need hand-position resets. Electronic tempo acceleration detection remains
unfinished: the constant-tempo assumption still needs correction for songs such
as Less Talk More Rokk. Reference charts are used for validation, never copied
into the generated output.

See [docs/RESEARCH.md](docs/RESEARCH.md) for sources, decisions and tool licensing.
