# Ordered development plan

Based on the shared conversation and source inspection, 2026-09-19.

## 1. Recoverable repository baseline — completed locally

Preserve the existing engine, test it, exclude local music/research/build output,
provide agent instructions and checkpoint on main. GitHub publication is a
separate approval step. Existing Setup.ps1, Launch.cmd and README already cover
installation; a clean-machine installation has not yet been verified.

## 2. Establish measured chart regressions — implemented

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

Status update (2026-09-19):

- Added a deterministic cached replay command (`rb3charts.cli replay`) and
  `pipeline.replay_cached` path that reruns authoring from saved analysis
  artifacts without rerunning decode/separation/model inference.
- Added `analysis/replay-provenance.json` with consumed input hashes, git
  revision/dirty state, replay settings/profile context, and runtime versions.
- Added additive guitar stage snapshots in
  `analysis/guitar-stage-snapshots.json`.
- Added replay-time `analysis/regression-metrics.json` plus concise log summary.
- Added `benchmark-replays` corpus comparison with per-song and macro attack,
  density, chord/hold, lane-coverage and contour metrics. This is an automated
  regression gate only; listening review and Xbox playtesting remain separate.
- Added synthetic regression fixtures/tests for repeated runs, transposed
  returns, strong near-variation, missing rests/attacks, chord boundaries,
  rhythm-only coincidences, late transposition confirmation behavior, variable
  tempo, and phrase-boundary splitting.

Deferred:

- Snapshot/metric expansion beyond guitar remains future work.

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

Status update (2026-09-20):

- Added versioned whole-phrase relative-motif discovery and canonical fingering
  reuse without changing source pitches, attacks or timing.
- Added indexed 3-5-note window matching for related figures embedded in unequal
  phrases. Accepted windows act as bounded final-authoring constraints and defer
  to exact phrase reuse, whole-phrase motifs and strong-variation rejection.
- Cached replay now records window matching coverage, rejection counts and a
  capped accepted-match sample in `analysis/phrase-authoring.json`.
- Band playability now permits four well-spaced attacks per beat even when motif
  matching has not yet recognized the phrase. The minimum-gap filter still
  rejects implausibly crowded bursts. A bounded phrase-aware exception retains
  up to six attacks per beat only when recurring-riff evidence supports each
  attack above the ordinary four. On cached held-out Hail to the King evidence,
  the combined density changes recovered 118 Expert attacks overall and nine
  attacks in the opening phrase.
- Three user-supplied screenshots of the held-out official Hail to the King
  Expert chart showed that its rapid intro uses broad five-lane single-note
  motion rather than lead-guitar power chords. They were used only to identify
  an aggregate contour mismatch, never as generation input. The final phrase
  pass had collapsed an already pitch-ordered five-lane mapping into three
  middle lanes. Long rapid phrases now retain a broad incoming mapping only
  when both boundary lanes recur, pitch/lane correlation is strong and at least
  85% of local pitch changes move in the same lane direction. Repeated pitches
  use their majority supported lane rather than inheriting note-level mapping
  noise, and later window closure cannot overwrite that stronger evidence.
  A cached replay over all 13 discovered held-out songs rejected a looser version
  that increased contour reversals, then accepted the direction-gated version.
  Hail's collapsed/reversed changes improved from 86/21 to 50/10 and Cliffs of
  Dover improved from 109/8 to 70/7. These are automated replay measurements,
  not listening review or Xbox validation; human review remains the acceptance
  gate.
- Whole-song canonical family selection and calibration against broader listening
  examples remain in progress; this does not complete Milestone 3.
- Added non-destructive Band-guitar voice-continuity diagnostics over overlapping
  eight-beat windows. The frozen 14-song baseline contains many mixed-register
  windows without a dominant evidence cluster, so confidence-only register
  selection is not enabled; additional lane-independent source evidence is
  required before phrase-level voice tracking can alter notes.

## 4. Review and anomaly tools

After motif metrics are trustworthy, annotate the existing preview with low
confidence, unexpected density/chords, and similar-event/dissimilar-chart regions.
Add generated-versus-human comparisons with explicit time alignment and provenance.
Never substitute reference notes into generated output. Distinguish event-level
similarity from audio similarity; the latter needs independent audio features.

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

## 7. Learned lane-independent structure

Prototype an optional model for attack, polyphony, sustain and section evidence,
then retain deterministic Expert fingering and difficulty reductions. Training
data must have item-level audio and annotation provenance, explicit license and
commercial-use status, hashes, and artist-disjoint splits. Official/reference
charts remain post-generation evaluation evidence only.

Status update (2026-09-22):

- Began an authorized human-chart pilot for the weak Expert authoring stage
  without replacing the working timing, export, validation, replay or reduction
  pipeline. The new manifest gate requires separate hashed training permission
  for audio and charts, verified synchronization, artist- and
  chart-author-disjoint splits, and a substantive in-range Expert guitar part.
  No RhythmVerse or official-chart data has been collected for training; actual
  model work remains blocked until lawful opt-in pairs are available.
- Added deterministic conversion from validated pairs into versioned Stage B
  examples. Each artifact keeps aligned lane-independent attack, polyphony and
  sustain conditioning separate from Expert lane/duration targets, preserves
  artist and chart-author split keys, and re-verifies every source and
  permission-evidence hash. The chart-derived conditioning is a pipeline proxy
  only; held-out generation must use independently predicted audio structure.
- Added a versioned dataset-manifest validator with separate audio/annotation
  licenses, attribution, commercial-use status, file hashes and path checks.
- Artist leakage across train, validation and test splits is rejected.
- Added GuitarSet JAMS conversion into lane-independent note, attack, polyphony,
  sustain and beat labels without adding a JAMS runtime dependency.
- Added deterministic log-mel, beat-phase and frame-target extraction with source
  hashes and versioned feature parameters.
- Added Slakh2100-redux conversion that selects rendered Guitar-class source MIDI,
  preserves stem provenance, groups polyphony and derives the variable tempo map.
- Added a deterministic CUDA-capable convolutional baseline with separate attack,
  polyphony and sustain heads, group-disjoint feature indexes, checkpoint
  provenance and validation metrics.
- GuitarSet artifacts now carry both progression and performer split keys.
  Validation threshold calibration is separate from untouched test evaluation.
- A local CC BY 4.0 pilot used 160 training, 10 validation and 10 test recordings
  with both progressions and performers disjoint. At a 50 ms onset tolerance, the
  untouched acoustic-guitar test measured 0.368 precision, 0.401 recall and 0.384
  F1; polyphony accuracy was 0.900. These are automated structure metrics, not
  listening review, RB lane agreement or Xbox playability.
- Direct application to 13 readable commercial full mixes was rejected as a
  domain mismatch: most songs produced very few predicted attacks and no chord
  attacks. Newly separated guitar stems showed the same failure, so runtime
  integration is rejected. Those local songs remain evaluation-only.
- Positive spectral-change features raised the BabySlakh untouched-test attack
  F1 from 0.136 to 0.507. A combined GuitarSet and BabySlakh checkpoint measured
  0.437 F1 and 0.601 polyphony accuracy on its disjoint untouched test. Both
  checkpoints still under-detected held-out commercial guitar stems at their
  cleared validation thresholds, so direct thresholded integration remains
  rejected.
- A ranking-only evaluation selected each model's highest-scoring stem peaks at
  the existing generated chart's attack count; it did not use the reference
  attack count. Across seven local commercial songs with official evaluation
  references, macro attack F1 increased from 0.6889 for existing charts to
  0.7071 for ranked model peaks, improving or tying six songs and regressing one.
  This is evidence that onset ranking transfers better than calibration. It is
  not evidence of lane, chord, sustain, listening-review or Xbox quality, and
  reference notes remain evaluation-only. No learned output is wired into
  generation.
- Feature-index schema 2 now requires every feature item to link to a
  hash-verified dataset-manifest item. Training rejects mismatched audio,
  annotation lineage, split, or licensing provenance. Threshold calibration now
  applies the same local-peak suppression as inference. Earlier checkpoints lack
  this manifest linkage and are retained only as historical evaluation evidence;
  they are not loadable by the current inference command.
- Added CC BY 4.0 Guitar-TECHS conversion with performer-disjoint split keys.
  Twelve amplifier-microphone excerpts from performer P3 were added to training
  only, leaving the existing 13-item validation and 13-item test sets unchanged.
  Relative to the preceding 20-epoch checkpoint, untouched test attack F1 rose
  from 0.7311 to 0.7982 and matched-attack polyphony accuracy rose from 0.6009 to
  0.8194.
- On the separate CC BY 4.0 EGSet12 electric-guitar evaluation, the same
  checkpoint measured 0.7994 attack F1 and 0.6737 polyphony accuracy. Attack F1
  was effectively unchanged from the preceding checkpoint's 0.8013, while
  polyphony improved from 0.4132. This supports further offline polyphony work,
  not runtime adoption.
- Added an offline provider for the CC BY 4.0 augmented TabCNN checkpoint
  published with EGSet12. Sampling TabCNN at independently predicted attacks
  measured 0.7184 polyphony accuracy, 0.5987 exact pitch-set agreement and
  0.8855 agreement on at least one pitch on EGSet12. Replacing the current pitch
  evidence wholesale was rejected because a Jonas replay slightly improved
  attack F1 but reduced official-reference chord-class agreement.
- The Guitar-TECHS checkpoint's validation-selected 0.725 attack threshold
  transferred poorly to 13 held-out commercial guitar stems: predicted/chart
  density ratios ranged from 0.029 to 0.729, including 0.376 on Jonas. Thresholded
  attack integration is therefore still rejected. No learned output is wired
  into normal generation.
- Structure prediction artifacts now record per-song attack-peak rank percentiles
  and score quantiles so offline fusion experiments do not assume cleared-data
  probabilities are calibrated on commercial stems. A 14-song shadow run still
  showed inverted rank behavior on several songs, and TabCNN phrase-path trials
  reduced some contour reversals while damaging supported intro lane range and
  increasing collapsed changes. Learned attack and pitch evidence therefore
  remain diagnostic-only.
- Evaluated 56 additional Guitar-TECHS P1/P2 chord recordings as cleared
  training data. Seven source MIDI files omitted final note-off events; conversion
  now closes those notes at the authoritative track end and records the repair
  count in label provenance. A 30 ms attack-grouping window incorrectly split
  performed strums, while 100 ms grouping recovered the intended full voicings.
  Neither the full expansion nor a performer-balanced 12-recording subset
  generalized: the best existing P3 checkpoint retained higher untouched-test
  attack and single-versus-chord metrics. These chord-drill checkpoints are
  rejected, remain offline research artifacts and are not used by generation.
- These are automated structure and post-generation comparison measurements.
  The new checkpoint has not received listening review or Xbox playtesting.
