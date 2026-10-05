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
If you want one-click testing of the latest local branch state, use
**UpdateAndLaunch.cmd** (runs `git fetch` + `git pull --ff-only` for the current
branch, then starts the GUI).

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
  measure lines and hear separate guitar, bass, or drum onset clicks. All review
  highways provide Normal plus Hyper 1-5 visual speeds independently of audio
  playback speed. Tap heads cross the strike line once at their authored onset;
  active pitched-instrument sustains keep only their tail at the line until
  release. A/B reviews open on the shared difficulty with the most changed
  attacks. Difficulty-specific review markers identify long same-lane runs and
  distinguish repeated detected pitch from multiple pitches collapsed onto one
  color. Cyan A/B outlines include added/removed attacks plus same-source
  onset and lane changes. These are review prompts, not automatic
  correctness failures. Highway speeds are consistent look-ahead levels, not a
  claim of pixel-for-pixel calibration to a specific Rock Band display.
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
.\.venv\Scripts\python.exe -m rb3charts.cli validate-run 'C:\Path\To\Existing_Output'
.\.venv\Scripts\python.exe -m rb3charts.cli validate-runs 'C:\Path\To\Output_Root'
.\.venv\Scripts\python.exe -m rb3charts.cli replay 'C:\Path\To\Existing_Output'
.\.venv\Scripts\python.exe -m rb3charts.cli benchmark-replays 'C:\Path\To\Baseline_Replays' 'C:\Path\To\Candidate_Replays'
.\.venv\Scripts\python.exe -m rb3charts.cli compare-reviews 'C:\Path\To\Baseline_Draft' 'C:\Path\To\Candidate_Draft'
.\.venv\Scripts\python.exe -m rb3charts.cli review-midi-comparison 'C:\Path\To\Generated_Draft' 'C:\Path\To\Reference\notes.mid' --difficulty hard --reference-difficulty expert --offset -1.1
.\.venv\Scripts\python.exe -m rb3charts.cli review-expert-hard-reference 'C:\Path\To\Generated_Draft' 'C:\Path\To\Reference\notes.mid' --offset -1.1
.\.venv\Scripts\python.exe -m rb3charts.cli benchmark-replays 'C:\Path\To\Baseline_Replays' 'C:\Path\To\Candidate_Replays' --instrument drums
.\.venv\Scripts\python.exe -m rb3charts.cli benchmark-replays 'C:\Path\To\Baseline_Replays' 'C:\Path\To\Candidate_Replays' --instrument all
.\.venv\Scripts\python.exe -m rb3charts.cli compare-human-chart 'C:\Path\To\ChartPairs\manifest.json' song-id 'C:\Path\To\Generated_Output' --source-agreement-report 'C:\Path\To\agreement.json'
.\.venv\Scripts\python.exe -m rb3charts.cli compare-midi 'C:\Path\To\Generated_Output' 'C:\Path\To\Evaluation\notes.mid'
.\.venv\Scripts\python.exe -m rb3charts.cli compare-midi 'C:\Path\To\Generated_Output' 'C:\Path\To\Evaluation\notes.mid' --difficulty hard --reference-difficulty expert
.\.venv\Scripts\python.exe -m rb3charts.cli index-comparisons 'C:\Path\To\Generated_Corpus'
.\.venv\Scripts\python.exe -m rb3charts.cli compare-comparison-indexes 'C:\Path\To\Baseline\comparison-index.json' 'C:\Path\To\Candidate\comparison-index.json'
.\.venv\Scripts\python.exe -m rb3charts.cli audit-playability-evidence 'C:\Path\To\Generated_Output' --structure-predictions 'C:\Path\To\predictions.json'
.\.venv\Scripts\python.exe -m rb3charts.cli analyze-reference 'C:\Path\To\Reference_MIDI_Corpus'
.\.venv\Scripts\python.exe -m rb3charts.cli validate-dataset 'C:\Path\To\Dataset\manifest.json' --verify-hashes
.\.venv\Scripts\python.exe -m rb3charts.cli validate-chart-pairs 'C:\Path\To\ChartPairs\manifest.json' --verify-hashes --source-agreement-report 'C:\Path\To\agreement.json'
.\.venv\Scripts\python.exe -m rb3charts.cli convert-chart-pairs 'C:\Path\To\ChartPairs\manifest.json' 'C:\Path\To\AuthoringExamples'
.\.venv\Scripts\python.exe -m rb3charts.cli convert-guitarset 'C:\Path\To\annotation.jams' 'C:\Path\To\labels.json'
.\.venv\Scripts\python.exe -m rb3charts.cli convert-egfxset 'C:\Path\To\Clean\Neck\6-0.wav' 'C:\Path\To\labels.json'
.\.venv\Scripts\python.exe -m rb3charts.cli synthesize-egfxset-phrase 'C:\Path\To\Selected_EGFxSet' 'C:\Path\To\phrase.wav' 'C:\Path\To\phrase.json' --sequence 0
.\.venv\Scripts\python.exe -m rb3charts.cli convert-slakh 'C:\Path\To\Track00001' 'C:\Path\To\labels.json'
.\.venv\Scripts\python.exe -m rb3charts.cli convert-guitar-techs 'C:\Path\To\midi.mid' 'C:\Path\To\micamp.wav' 'C:\Path\To\labels.json'
.\.venv\Scripts\python.exe -m rb3charts.cli extract-structure-features 'C:\Path\To\audio.wav' 'C:\Path\To\labels.json' 'C:\Path\To\features.npz'
.\.venv\Scripts\python.exe -m rb3charts.cli train-structure-baseline 'C:\Path\To\feature-index.json' 'C:\Path\To\baseline.pt' --device cuda
.\.venv\Scripts\python.exe -m rb3charts.cli predict-structure 'C:\Path\To\baseline.pt' 'C:\Path\To\guitar.wav' 'C:\Path\To\predictions.json' --beats 'C:\Path\To\tempo.json'
.\.venv\Scripts\python.exe -m rb3charts.cli evaluate-structure-ranking 'C:\Path\To\predictions.json' 'C:\Path\To\labels.json'
.\.venv\Scripts\python.exe -m rb3charts.cli pin-source-agreement 'C:\Path\To\manifest.json' 'C:\Path\To\alternate-audio' 'C:\Path\To\paired-manifest.json' --primary-suffix _mic.wav --alternate-suffix _mix.wav
.\.venv\Scripts\python.exe -m rb3charts.cli evaluate-source-agreement 'C:\Path\To\paired-dataset-manifest.json' --output 'C:\Path\To\agreement.json'
.\.venv\Scripts\python.exe -m rb3charts.cli predict-tabcnn 'C:\Path\To\TabCNN.pt' 'C:\Path\To\guitar.wav' 'C:\Path\To\tabcnn.json' --attacks 'C:\Path\To\structure-predictions.json'
.\.venv\Scripts\python.exe -m rb3charts.cli assess-official-quality 'C:\Path\To\comparison-index.json'
.\.venv\Scripts\python.exe -m rb3charts.cli demo --compile
.\.venv\Scripts\python.exe -m pytest -q
```

`benchmark-replays` defaults to guitar and also supports bass, drums, keys, or
all available instruments in one combined report.
Its macro report includes difficulty-retention totals; simultaneous drum hits
are reported separately rather than classified as chords. When both replay
roots contain structure diagnostics, the report also compares mean activity
blocks, boundary strengths, repeated drum-groove families, and groove-supported
attack coverage. Complete reduction snapshots add mean/max four-second density
for Expert, Hard, Medium, and Easy. Pattern artifacts add song-mean recurring
family/occurrence counts, supported-attack coverage, and pitched lane-motion
metrics. Guitar reports also measure exact repeated-riff fingering agreement;
automatic review flags charts below 80% when at least twelve repeated
occurrences are available. Final authoring can normalize a repeated riff's
absolute fretboard offset only when every occurrence has identical relative
lane steps and all overlapping proposals agree.
Each root must contain at most one run for a given artist/title; ambiguous
duplicate song keys and missing artist/title metadata are rejected instead of
selecting or matching a run implicitly.

`assess-official-quality` turns an authorized comparison index into the release
scorecard. Release requires at least 20 songs from five artists and at least 80%
of songs must independently clear attack, chord, sustain, lane, movement,
weakest-window and repeated-motif thresholds. Missing metrics fail rather than
silently passing. The scorecard is evaluation-only: human chart notes never
enter generation or runtime feature selection.
Expanded multi-instrument, reduction-density, structure, and pattern reports use
benchmark schema 2; guitar remains the default instrument. Without `--output`,
guitar writes `output/replay-benchmark.json` and other instruments write
`output/<instrument>-replay-benchmark.json`. The all-instrument mode writes
`output/all-instruments-replay-benchmark.json` and lists instruments that were
unavailable in either replay root.

`compare-reviews` creates one local synchronized A/B review page for two
completed drafts of the same source recording. It verifies song identity,
source-audio hash, tempo map and lead-in timing before comparing them. One audio
transport is shared across both versions; use the A/B buttons or keyboard keys
to switch at the same timestamp. Candidate-only and baseline-only attacks are
outlined, and the densest eight-second candidate-addition windows become jump
markers. This is a review surface only and never changes either draft.

`review-midi-comparison` creates the same synchronized A/B highway for a
generated difficulty and an evaluation-only reference difficulty. The required
offset must come from a prior aligned comparison. The command materializes
aligned reference markers in a local companion folder and hard-links the
generated review audio when supported; neither chart is modified or used as
generation input.

`review-expert-hard-reference` creates one A/B/C highway for project Expert,
project Hard and aligned reference Expert. It starts on project Hard and keeps
one audio transport while A/B/C keys switch interpretations. This is useful
when a generated reduction exposes a cleaner structural backbone than generated
Expert even though reference metrics remain evaluation-only.

`compare-midi` is an evaluation-only cross-check for a completed generated
guitar chart and a local reference MIDI. It automatically estimates a global
onset offset unless `--offset` is supplied, records hashes and alignment
provenance, reports unmatched attacks and weakest four-second windows, and
attributes metrics to saved authoring stages when snapshots are present. It
writes JSON plus a Markdown summary; reference notes are loaded only after
generation and are never runtime inputs. The Markdown includes content hashes
but omits local paths and individual reference-note rows.
`--reference-difficulty` permits explicit cross-difficulty review, such as
generated Hard against reference Expert; it defaults to `--difficulty`.

`index-comparisons` scans generated runs below one local corpus root and writes
`comparison-index.json` plus Markdown. It indexes either authorized-chart or
reference-MIDI comparisons, never a mixture. Entries contain display identity,
run basename, comparison/validation status, compact attack/chord/sustain/lane/
motif/weak-window summaries, paired-source status and relative local links.
Runs with no comparison remain explicitly `missing`; duplicate song IDs,
mixed schemas and malformed metadata/reports are rejected before either index
is written. Available reports must match the current generated MIDI hash and,
for authorized comparisons, the metadata song ID; stale comparisons are
rejected. Source paths, unmatched/reference note rows and detailed authorized
records are excluded. A custom index output must remain directly inside the
corpus root so every relative report link can be revalidated without escaping
the index boundary.

`compare-comparison-indexes` writes deterministic JSON plus concise Markdown for
two compatible sanitized indexes. Song IDs are paired only when display
identity also matches; baseline-only, candidate-only and shared-missing coverage
remain explicit. Per-song and macro deltas cover the safe aggregate F1,
lane/motif and weakest-window fields, with ranked regressions/improvements.
Null metrics remain unavailable rather than becoming zero. The command rejects
duplicate identities, mixed comparison schemas, changed metric semantics,
modified linked reports and changed generated MIDI. Trend artifacts contain
index hashes but no local paths, report links, events or reference notes.

`audit-playability-evidence` is also post-generation only. It joins attributed
playability removals to cached transcription evidence, optional offline
structure ranks, source-mix and guitar-stem onset height/prominence/width ranks,
and voice-continuity windows. An optional `--reference` labels whether a removed
attack could cover an otherwise unmatched evaluation onset and reports
top-quartile feature separability; those labels never enter generation.

New review drafts record hashes for `import/notes.mid`, `song.wav`, and
`song.ini`. The `package` command rejects any of those inputs if they changed
after validation; legacy drafts without a manifest remain packageable with an
explicit warning.

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

Structure-model training datasets use an item-level JSON manifest. Each item
records relative audio and annotation paths, SHA-256 hashes, song and artist IDs,
and its train/validation/test split. Dataset metadata records audio and annotation
licenses separately, attribution, source/version, and whether commercial use is
allowed, noncommercial, or unknown. `validate-dataset` rejects missing files,
path traversal, duplicate songs/files, malformed hashes, and artists crossing
splits; `--verify-hashes` also reads and verifies every file.

Human-authored chart training is an opt-in pilot, not a license assumption.
`validate-chart-pairs` requires separate per-song evidence that the audio rights
holder and chart rights holder allow model training. It also requires verified
audio/chart synchronization, Rock Band MIDI with a substantive Expert guitar
part, file hashes, and artist- plus chart-author-disjoint splits. A chart being
downloadable from a community site does not by itself authorize model training.
See `docs/AUTHORIZED_CHART_PILOT.md`.

`--source-agreement-report` additively attaches an existing
`evaluate-source-agreement` result to the chart-pair validation report. The
summary re-hashes the evaluation report, paired manifest, primary recordings,
labels and pinned alternates, verifies exact evaluation-item identity and
release lineage, then records tolerance, splits, shared precision/recall/F1,
the best individual source and deltas. Missing reports are explicitly
`unavailable`; broken lineage is `invalid`. A valid GuitarSet or Guitar-TECHS
result is external
rights-cleared calibration, not evidence that authorized chart-pair songs have
paired captures.

The same option on `compare-human-chart` writes the evidence into
`human-chart-comparison.json` and a deterministic companion Markdown report.
The Markdown labels external calibration separately from generated-song
evidence, reports explicit corpus coverage, omits local manifest/report paths,
and suppresses all metrics when lineage is unavailable or invalid.

`convert-guitarset` extracts note attacks, pitches, polyphony, sustain endpoints,
and beat positions from GuitarSet JAMS annotations. Its output deliberately has
no Rock Band lanes or reference-chart sequence: those remain the responsibility
of deterministic authoring after the learned structure stage.

`convert-slakh` reads `metadata.yaml`, selects only rendered Guitar-class source
MIDI, preserves source identities, combines simultaneous guitar sources into
polyphonic attacks, and derives beat times from the MIDI tempo map. Use the
duplicate-free Slakh2100-redux splits; do not train from its `omitted` directory.

`convert-guitar-techs` converts synchronized six-string MIDI from the CC BY 4.0
Guitar-TECHS corpus. Its labels retain the performer identity as an authoritative
split key; all excerpts by one player must remain in the same data split.

`convert-egfxset` converts one CC BY 4.0 EGFxSet `string-fret.wav` recording.
All effect variants of one pickup/string/fret source share a group key, and the
unknown performer key keeps the entire corpus in one split. EGFxSet's fixed
five-second effect intervals are accepted only as attack and single-note
polyphony supervision; they are never treated as musical sustain labels.
`synthesize-egfxset-phrase` deterministically combines short, faded excerpts
into a ten-second multi-attack training phrase. The seed, sequence number,
source hashes and augmentation parameters remain in the labels, which still
expose only attack and single-note polyphony targets.

`synthesize-procedural-guitar` creates deterministic additive-pluck audio with
single notes, chords, sustains, beat positions, and aligned lane-independent
labels without reading any source recording or annotation. Every artifact
records its generator version, seed, sequence, style, tempo, and sample rate.
All generated examples share a generator split key so this synthetic domain
cannot leak across train, validation, and test partitions.

`extract-structure-features` writes fixed-reference 128-bin log-mel audio,
128 positive spectral-change bins, source-time frame positions, variable-tempo
beat phase, and the aligned targets declared by each label artifact. The compressed
artifact includes audio/label hashes and fixed feature parameters so training
runs can reject stale or incompatible caches.

`train-structure-baseline` trains a small convolutional network with separate
attack, polyphony, and sustain heads. Its feature index requires relative artifact
paths, explicit train/validation/test splits, a `group_id`, and a validated
dataset-manifest reference for every item. Training verifies source file hashes,
converted-label lineage, dataset licensing metadata, and manifest splits before
reading features. Authoritative artifact split keys prevent either a musical
progression or performer identity from crossing splits. Losses are applied only
to targets declared by each artifact, allowing attack-only corpora without
inventing sustain labels. Validation calibrates
the attack threshold by macro song-level F1 after applying the same local-peak
suppression used by inference; the untouched test split uses that threshold
without retuning. `--attack-loss focal --focal-gamma 2` enables deterministic
unweighted focal attack training while the existing weighted BCE remains the
default for comparable baseline runs. `--calibration-dataset DATASET_ID`
restricts global threshold selection to the intended validation domain without
using test labels. Reports include per-song validation and test metrics so long
recordings cannot hide regressions elsewhere in the split.
`--architecture dilated-tcn` selects a versioned residual temporal model with
about three seconds of receptive context; checkpoints record and restore the
architecture explicitly.
`--architecture multi-scale` combines local convolutions with pooled
approximately 0.4, 1.5 and 3.0 second context, while
`--architecture multi-scale-local` adds a separate local attack head for
controlled timing experiments. Both remain offline research architectures;
validation and frozen-test timing gates still apply before runtime use.
`--skip-test` omits test evaluation during candidate selection while retaining
the indexed test count in the report; rerun the selected deterministic
configuration without that flag only after it wins on validation.
Checkpoints retain manifest and feature hashes, the feature configuration, seed,
runtime versions, loss history, attack metrics, and polyphony accuracy. This
baseline is an engineering measurement, not yet a chart-quality claim or runtime
model.

`predict-structure` performs offline, lane-independent evaluation and writes
auditable attack/polyphony predictions plus the complete ranked set of local
attack candidates. It does not alter a transcription or generated chart.
Supplying a generation run's `analysis/tempo.json` reproduces beat-phase
conditioning; without it the model uses audio features alone. Candidate rows
also expose raw, symmetric-periodicity, peak-width, and combined rank
percentiles, plus the sustain head's raw active-run duration and a version
bounded by the next predicted attack. These are offline evidence and are not
runtime attack-recovery or articulation instructions.

`evaluate-structure-ranking` makes the offline gate reproducible: it selects a
fixed top-K candidate set by a named rank field and performs one-to-one attack
matching against lane-independent labels at a fixed tolerance. By default K is
the number of distinct labeled attacks and the accepted combined rank is used.
The report records all matching parameters and can be saved with `--output`.

`evaluate-source-agreement` validates a rights-cleared manifest and its hashes,
then compares aligned recording channels such as Guitar-TECHS mic/amp and
direct-input audio or GuitarSet microphone and pickup-mix audio. Each channel
contributes the same deterministic number of
prominence-ranked onset candidates as the authoritative label count. Reports
separate shared, primary-only, and alternate-only precision/recall, retain
group/performer split keys and both audio hashes, and reject missing, duplicated,
duration-mismatched, or provenance-mismatched pairs. It is offline evidence and
does not change generation. Every paired-manifest item must explicitly provide
`alternate_audio` and `alternate_audio_sha256`; evaluation rejects unpinned
alternates and alternate content identical to any primary recording.

`pin-source-agreement` reproducibly creates the explicit paired manifest. It
maps a filename suffix into an aligned alternate-audio directory, pins every
alternate SHA-256, preserves all existing artist/performer splits, and can
verify and record a release archive's published MD5, byte size and source
record. The source manifest, alternate recordings, archive and output must stay
inside one ignored dataset root.

`predict-tabcnn` evaluates the CC BY 4.0 augmented TabCNN checkpoint published
with EGSet12. It writes string/fret, pitch and polyphony evidence and can sample
that evidence at independently predicted structure attacks. The command is
offline-only, records model/audio/input hashes and attribution, and never alters
a generated chart. It requires the upstream AMT-Tools package from
`robust-guitar-tabs/code`; that research dependency is intentionally not part of
the ordinary RB3 Charts installation.

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

After chord cleanup, one final Band Guitar phrase author owns Expert fingering.
It compares whole phrases and bounded three-to-five-note windows, reuses compact
lane shapes only for supported motif matches, and keeps strong variations
independent. Its decisions and rejected matches are saved to
`analysis/phrase-authoring.json`; earlier passes no longer rewrite lanes before
this authoritative stage.

Ordinary Band guitar, bass and keys passages are limited to three attacks per
quarter beat with a minimum spacing guard. A fourth attack is retained only when
it belongs to a detected recurring fast riff, allowing deliberate sixteenth-note
figures while removing one-off transcription congestion.

Band Guitar receives its authoritative phrase authoring pass before difficulty
reduction. Hard removes every fourth attack from long eighth-note runs at 160
BPM or faster, replaces green/orange chords and keeps at most two-note chords.

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
  review.html          offline scrolling chart preview with diagnostic overlays
  validation_report.txt
  result.json          explicit completion / failure / playtest status
  logs/generation.log
```

The review page includes clickable, non-destructive markers for up to the 20
weakest low-confidence Expert attacks per instrument, forbidden green-to-orange
triads, merged four-second density spikes, and up to 20 repeated-source motifs
whose final lane pattern differs from the family majority. Marker thresholds
follow the selected Band or Electronic chart profile; they guide listening and
never alter the authored chart.

`analysis/structure-diagnostics.json` records neutral eight-measure activity
blocks, strongest unlabeled activity boundaries and repeated measure-level drum
grooves. Review pages link the eight strongest boundaries. These reports do not
invent verse/chorus names or rewrite notes.

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
- `scripts/fretnet_pilot.py`: optional, offline FretNet training pilot on
  rights-cleared GuitarSet audio. It is not a runtime dependency and does not
  feed predictions into chart generation.
- `scripts/fretnet_shadow.py`: exports source-time, per-string FretNet note and
  frame-pitch evidence for an audio file without assigning chart lanes.
- `tests/`: synthetic regression fixtures; no commercial audio required.
- `reference/style-profile.json`: compact runtime statistics; full local reference
  reports and downloaded chart repositories are excluded from Git.

Newly analyzed reference profiles include aggregate per-song
Expert-to-Hard/Medium/Easy retention percentiles. The committed schema-1 profile
continues to provide its existing Hard prior until its excluded source corpus is
available for legitimate regeneration.
`analyze-reference` writes only compact aggregate statistics to `--output`.
Use `--full-output` for an optional per-song report with source paths, and keep
that report local.

The FretNet pilot requires an isolated environment containing the external
MIT-licensed FretNet repository and its research dependencies. A deterministic
held-out-performer run is:

```powershell
.\.research\fretnet-venv\Scripts\python.exe scripts\fretnet_pilot.py `
  --dataset .research\guitarset `
  --fretnet-repo .research\fretnet `
  --output .research\fretnet-pilot `
  --holdout 05 --steps 2500 --seed 0
```

The local report records the dataset fingerprint, FretNet revision, dependency
versions, initial/final held-out losses and checkpoint name. Checkpoints and
reports remain under `.research/`; training success alone does not authorize
runtime integration.

`fretnet_shadow.py` can then write lane-independent evidence for a separated
guitar stem. Supplying that JSON to `replay --fretnet-shadow` records a
`fretnet_shadow_diagnostic` stage snapshot and proposal report. The replay's
actual chart remains unchanged; this option exists to measure transfer before
any model output can influence authoring.

```powershell
.\.research\fretnet-venv\Scripts\python.exe scripts\fretnet_shadow.py `
  --audio output\review\audio\timing-audit-guitar.wav `
  --checkpoint .research\fretnet-pilot\fretnet-pilot-state.pt `
  --fretnet-repo .research\fretnet `
  --output .research\fretnet-shadow\review.json

.\.venv\Scripts\python.exe -m rb3charts.cli replay output\review `
  --output output\shadow-replay `
  --fretnet-shadow .research\fretnet-shadow\review.json
```

`validate-run` rechecks an existing draft's package MIDI against its saved
charts and tempo map, verifies the package-input manifest when present, and
writes separate `validation_recheck_report.json`/`.txt` files without rerunning
authoring or overwriting the original validation report.
`package` automatically performs this current recheck when the saved charts and
tempo map are available; legacy folders without those artifacts retain the
documented compatibility fallback. The packaged `result.json` records whether
the current recheck passed, only a legacy manifest was available, or validation
artifacts were unavailable, and links the recheck report when present.

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
chords and eight-beat boundaries. Brief chord barriers still preserve nearby
single-note continuity for returning riffs, but they do not create or infer new
attacks. Phrases with up to five distinct pitches use distinct adjacent frets.
Larger pitch ranges use a whole-sequence cost balancing melodic direction,
repeated pitches, hand movement and local register. Identical relative-pitch/
rhythm phrases reuse their fingering, including transpositions. Chords and
meaningful long holds remain; short tails become taps. Every attack's time,
source ID and detected pitch are preserved by this pass.

`analysis/phrase-authoring.json` records each phrase's pitches, old/new frets,
reused fingering, continuity source, tail changes and contour diagnostics. This
is a heuristic authoring layer, not a trained imitation of professional
charters. It cannot recover notes that transcription missed, and ranges wider
than five frets may still need hand-position resets. Electronic tempo
acceleration detection remains unfinished: the constant-tempo assumption still
needs correction for songs such as Less Talk More Rokk. Reference charts are
used for validation, never copied into the generated output.

See [docs/RESEARCH.md](docs/RESEARCH.md) for sources, decisions and tool licensing.
