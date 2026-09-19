# Ordered development plan

Based on the shared conversation and source inspection, 2026-09-19.

## 1. Recoverable repository baseline — completed locally

Preserve the existing engine, test it, exclude local music/research/build output,
provide agent instructions and checkpoint on main. GitHub publication is a
separate approval step. Existing Setup.ps1, Launch.cmd and README already cover
installation; a clean-machine installation has not yet been verified.

## 2. Establish measured chart regressions — next implementation

Build a deterministic replay command that consumes saved transcription candidates
and tempo, runs the existing authoring stages, and writes a fresh output directory.
Record input hashes, Git revision, dirty state, settings, profile hash, model and
dependency versions. Compare identical inputs before/after, and clearly label
cached replay separately from a complete audio reanalysis.

Retain snapshots after mapping, playability, reference conformance, Band refinement,
stabilization, final phrase authoring and difficulty reduction. Join each snapshot
by source ID so a lane, pitch, duration or attack change names its responsible
stage. Existing event traces identify broad filtering stages, but do not retain
all these intermediate chart states.

Start with synthetic fixtures: repeated four-note runs, transposed returns,
near repeats with genuine strong variations, missing attacks/rests, chords as
boundaries, unrelated matching rhythms, and a late high-confidence repetition.
Add variable tempo and phrase-boundary fixtures. Tests must not download music.

Local listening benchmarks: Empty Walls source seconds 0–35, Less Talk More Rokk
accelerating intro, and Fineshrine repeated attacks/decays. Keep audio and charts
local; identify exact source hashes and cached run IDs before comparing. Existing
historical reports are evidence of past runs, not proof of present improvement.

Report attack retention, matched timing error, contour collapses/reversals,
chord/hold rates and repeated-phrase lane agreement, per stage and time region.
Keep absolute-lane agreement distinct from relative-shape agreement. Report
coverage and false-lock rate: a system that labels nothing a motif cannot earn
a perfect consistency score. Human listening remains the acceptance gate.

## 3. Consolidate motif understanding — architecture proposal

The project already has musical events and several repetition mechanisms:

- `events.interpret_events`: four-onset pitch-set/rhythm evidence before mapping.
- `chart.analyze_patterns` and playability: repeated figures influence retention.
- `chart.stabilize_repeated_phrases`: Band-only four-note templates/projections.
- `phrases.author_guitar_phrases`: final fingering dictionary over phrases cut at
  chords, rests or eight beats; signatures require exact relative pitch and
  quantized rhythm sequences.

`pipeline.generate` currently maps notes, limits playability, applies reference
conformance, refines Band guitar, stabilizes repeated phrases, then authors guitar
phrases again. The final pass can overwrite earlier lane decisions. Its signature
changes if detection adds/drops a note or if an eight-beat cut shifts the phrase
boundary. Reuse can be shifted or rejected to accommodate the previous phrase.
The first encountered fingering becomes the dictionary entry; there is no
whole-song best-evidence canonical selection. These are code-level failure
mechanisms to test, not a verified diagnosis of the current Empty Walls recording.

Introduce a versioned lane-independent motif representation between role/voice
selection and mapping, initially for Guitar only. An occurrence should record:
source IDs, beat/second bounds, instrument/role, relative pitches and rhythm,
durations, original evidence, family ID, alignment, transposition, unmatched
events, confidence components and rejection reasons. Preserve candidate evidence
separately from corrected interpretations. Confidence scores are heuristics until
calibrated against labeled examples.

Compare candidate phrases across the entire song using rhythm/contour indexing
and bounded alignment. Begin with exact and transposed matches, then gated near
matches. Penalize insertions/deletions and rhythm disagreement; do not bridge
rests or chord boundaries without evidence. Bound candidate comparisons to avoid
an unrestricted quadratic all-pairs audio matrix on long tracks.

Select canonical occurrences after the whole song has been considered, using
evidence quality and deterministic ties. Later clearer evidence may inform an
earlier weak repetition. High-confidence novelty must remain a variation; weak
outliers may receive a consistency prior, with the full reason recorded.

Give one final Expert authoring stage ownership of fingering and explicit motif
constraints. Measure boundary playability against repeated-shape consistency;
record any necessary hand reset. Do not simply append another postprocessing
lock after the current stack. Retire overlapping old heuristics only after
equivalence and new regression checks. Leave drums/bass/keys unchanged initially.

Files likely affected: a new `motifs.py`, `phrases.py`, the relevant Guitar paths
of `chart.py`, `pipeline.py`, optional model/schema types and focused tests.
`transcribe.py` should retain raw evidence; MIDI/export should not need changes.

## 4. Review and anomaly tools

After motif metrics are trustworthy, annotate the existing preview with low
confidence, unexpected density/chords, and similar-event/dissimilar-chart regions.
Add generated-versus-human comparisons with explicit time alignment and provenance.
Never substitute reference notes into generated output. Distinguish event-level
similarity from audio similarity; the latter needs independent audio features.

The first independent comparison workflow is complete: `Compare Charts.cmd`
accepts a generated review folder and YouTube gameplay-chart URL, presents both
players with an adjustable synchronization offset, and exports timestamped human
observations. The CLI also performs read-only, onset-aligned MIDI comparisons when
an official chart MIDI is locally available. Automatic note extraction from video
is a later measured computer-vision task; the current video mode does not claim
machine-derived note accuracy.

## 5. Tempo and broader musical structure

Address accelerating electronic tempo independently from motif fingering. The
existing constant-tempo preference is a known limitation for Less Talk More Rokk.
Then explore sections, phrase vocabulary, drum grooves and richer human priors.
Validate one subsystem at a time rather than rewriting the entire engine.

## 6. Remote work and provider flexibility

After the first-upload review, create a private GitHub backup and use feature
branches plus the issue template as the phone idea inbox. Keep local audio/model
work on the home machine; a remote checkout alone does not include ignored assets.
Confirm current supported Codex Remote setup and account/device requirements
before changing access or power settings. Remote access is not configured by
this repository setup. Subscription changes, API billing and purchases remain
user decisions; the shared conversation's pricing claims were not adopted as
verified requirements. No additional provider is needed to use this repository.
