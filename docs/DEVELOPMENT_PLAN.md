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

Status update (2026-10-02):

- Stage snapshots now cover every generated/replayed instrument. Pitched
  instruments record mapping, playability, optional Band reference conformance,
  and reductions; drums record their cleaned/replay-source Expert chart and
  reductions.
- Cached replay writes per-instrument regression metrics plus an aggregate
  `regression-metrics-by-instrument.json`. The existing guitar
  `regression-metrics.json` remains as a compatibility alias.
- Cached drum replay preserves the already-aligned Expert chart but now
  rederives Hard/Medium/Easy, so reduction experiments no longer reuse stale
  cached difficulties. It still does not rerun drum activation alignment.
- Per-instrument metrics now audit every difficulty reduction for attack
  retention, retained/orphan source IDs, lane changes from Expert, and timing
  error. These are observability checks; intended lane simplification remains
  part of reduction behavior.
- Assessments compare generated Expert-to-Hard attack retention with the
  existing aggregate human-chart p10/p90 range for each instrument. This is a
  review flag only. Hail measured 0.53 guitar retention (below the 0.60 p10)
  and 0.97 drum retention (above the 0.82 p90); changing reduction behavior
  remains gated on broader replay evidence.
- Reference analyzer schema 2 now emits per-instrument Expert-to-Hard, Medium,
  and Easy retention percentiles while retaining the old Hard-only key as a
  compatibility alias. The committed schema-1 profile intentionally lacks
  per-song records and was not reverse-engineered or overwritten.
- The analyzer now writes a compact, path-free aggregate profile by default.
  Full per-song records and source paths require a distinct explicit
  `--full-output`, preventing the tracked runtime profile from accidentally
  receiving excluded local reference details.
  - A drum-specific grid-consistent accompaniment thinning candidate moved
    Hail/Less Talk Hard retention from 0.967/0.773 to 0.798/0.704 and retained
    46%/52% of accompaniment-only attacks. It was rejected from runtime because
    only two drum charts were available and no lawful authoring labels or
    listening evidence establish that the changed grooves are musically better.
- A six-song cached-replay sweep then reduced guitar Hard minimum spacing from
  0.46 to 0.38 beats. Retention moved from 0.531-0.724 to 0.606-0.810, placing
  every song inside the aggregate guitar p10-p90 range (0.599-0.854). Worst
  four-second Hard density rose only from 4.25 to 4.50 attacks/s; assessment
  status, Expert notes, timing, ancestry, and structural validation were
  unchanged. Hail Hard is a listening-review candidate, not an Expert rewrite.
- Corpus `benchmark-replays` comparisons now select guitar, bass, drums, or
  keys and aggregate Expert-to-Hard/Medium/Easy retention. Simultaneous drum
  attacks are reported independently from pitched-instrument chords. Two
  independent Hail replays produced exact zero bass and drum deltas, including
  drum reduction retention of 0.9672/0.5546/0.2991.
- Reduction macro totals now include only songs whose baseline and candidate
  both contain complete reduction snapshots. Asymmetric legacy coverage reports
  unavailable retention instead of comparing different song sets.
- Corpus discovery rejects duplicate artist/title keys within either root and
  reports both conflicting paths. This prevents broad experiment folders from
  silently selecting an arbitrary run as the baseline.
- The expanded replay benchmark contract is schema 2. It remains additive for
  the original guitar macro/song fields while explicitly versioning instrument,
  reduction-density, structure, and pattern sections.
- Complete reduction pairs also aggregate mean/max four-second density for all
  four difficulties. The retained Hail guitar Hard change is now reproducible
  directly in `benchmark-replays`: retention 0.5312 to 0.6056 and max density
  3.75 to 4.25 attacks/s.
- Structural MIDI validation now enforces type 1, 960 PPQ, unique required
  tracks, EVENTS-local lifecycle markers, tempo/beat content, track-specific
  pitches, and exact serialized gem counts for every instrument/difficulty.
  It also reads every serialized gem interval back in tick space and compares
  its attack, release, and pitch with the authored chart. A production-sized
  cached Hail replay passed this export/readback audit across all instruments
  and difficulties.
- Validation now compares every serialized tempo event and the time signature
  with the authored `TempoMap`, closing the gap where correct gem ticks could
  still play at incorrect wall-clock times. A variable-tempo corruption test
  and a fresh Hail replay pass this gate.
- The BEAT track is also read back as paired note intervals and compared with
  the expected full grid, including onset ticks, 120-tick duration, downbeat
  pitches, meter phase, and terminal extent.
- EVENTS lifecycle marker ticks, per-instrument `[play]`/`[idle]` ticks, and all
  four drum mix markers are now validated against authored duration/lead. A
  corruption test and production-sized Hail replay pass this gate.
- Those required lifecycle/mix markers must each occur exactly once; duplicates
  can no longer be hidden by set/dictionary-based marker lookup.
- New drafts hash the three reviewed package inputs after validation.
  Packaging rejects post-validation changes to MIDI, audio, or `song.ini`;
  legacy drafts without a manifest remain supported with an explicit warning.
- `validate-run` now applies the current structural/readback validator and
  package-input manifest check to an existing draft without replaying authoring.
  It writes separate recheck reports, preserving the original validation record.
- Packaging now invokes that current recheck automatically for complete run
  folders and stops before compilation on any failure. Older folders missing
  charts/tempo retain the explicit legacy manifest/report fallback.
- Added evaluation-only `compare-midi` for arbitrary local reference MIDI
  cross-checks after generation. It records input hashes, Git state, alignment,
  exact unmatched events, weakest four-second windows, and per-authoring-stage
  metrics. Reference notes remain excluded from generation and training.
- Added post-generation playability-evidence audits that join each attributed
  removal to raw onset/level evidence, optional offline structure ranks,
  source-mix versus guitar-stem onset prominence/width, and voice-continuity
  windows. Optional reference labels measure separability only after generation
  and remain excluded from authoring.

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
  required before phrase-level voice tracking can alter notes. Candidate windows
  now appear as amber regions in the offline review highway, with register
  separation and switch counts shown during playback so timestamped feedback
  maps directly to the responsible source-evidence region.

## 4. Review and anomaly tools

After motif metrics are trustworthy, annotate the existing preview with low
confidence, unexpected density/chords, and similar-event/dissimilar-chart regions.
Add generated-versus-human comparisons with explicit time alignment and provenance.
Never substitute reference notes into generated output. Distinguish event-level
similarity from audio similarity; the latter needs independent audio features.

The preview now exposes clickable, non-destructive markers for up to the twenty
weakest low-confidence Expert attacks per instrument, forbidden green-to-orange
triads, merged four-second density spikes using profile-specific limits, and up
to twenty repeated source-event motifs whose final lane pattern differs from
the family majority. These diagnostics remain listening evidence only.
An automatic exact-pitch majority-lane normalization was selected only on the
three rights-cleared development songs and rejected: it was neutral on Certain
Death and reduced both exact-lane and movement agreement on Dirtbag and There's
Something Wrong. The markers do not authorize lane rewriting.

## 5. Tempo and broader musical structure

The accelerating electronic-tempo milestone is complete independently from
motif fingering. A fresh production-path analysis of Less Talk More Rokk now
detects the sustained acceleration and retains a 598-beat variable map instead
of the historical constant 124.998 BPM grid. Against the aligned, evaluation-only
RB tempo map, one-to-one 50 ms beat matching improved from 493/576 to 567/576
over the full song. In the critical first 70 seconds it improved from 25/108 to
107/108, with one missed and four extra generated beats. The exported MIDI
contains 565 tempo changes and begins near 70.18 BPM, confirming that the
variable analysis survives export. The draft still requires listening review;
these timing results do not validate guitar authorship.

The subsequent Less Talk review rejected Guitar even with the corrected tempo.
The opening 284 ms source-supported electronic sustain was incorrectly shortened
by the shared Band tail rule; final phrase authoring now uses the electronic
90 ms threshold and preserves that hold. The scaling riff remains blocked
upstream: lead selection retained 11/15 evaluation attacks in the first reviewed
phrase and playability reduced both reviewed phrases to eight attacks. Directly
reusing the flexible motif matcher before playability was rejected because it
accepted 21,172 full-song windows yet found no safe strict match between the
actual riff phrases. Guitar-only and ungated multi-stem variants also traded a
minor first-phrase improvement for a worse second phrase and substantially more
full-song attacks. Do not request another listening review until lane-independent
voice evidence produces a stable repeated attack/pitch sequence.

Next explore sections, phrase vocabulary, drum grooves and richer human priors.
Validate one subsystem at a time rather than rewriting the entire engine.

Status update (2026-10-02):

- Added post-authoring `structure-diagnostics.json` with neutral eight-measure
  activity blocks, ranked unlabeled activity boundaries, and repeated
  measure-level drum-groove families. The review page links the eight strongest
  activity boundaries for focused listening. These diagnostics never name
  verse/chorus sections and do not alter charts.
- Review pages now expose up to 16 capped jump markers per instrument for
  repeated motif candidates, including family occurrence count and instance
  number. These are A/B listening links, not correctness claims or generation
  inputs.
- Corpus replay benchmarks now compare final pattern vocabulary: recurring
  families/occurrences, pattern-supported attack ratio, and pitched contour,
  flattening, adjacent-motion, and wide-jump rates. Independent Hail roots
  matched exactly at 158 families, 442 occurrences, and 0.5619 supported-attack
  ratio.
- On the cached Hail replay, 19 repeated drum-groove families covered 62.99% of
  Expert drum attacks. This is descriptive evidence for future groove work, not
  permission to normalize drum patterns.
- Song assessments and `review_notes.md` now surface neutral activity-block,
  ranked-boundary, and repeated-groove counts plus supported drum-attack
  coverage. This changes review visibility only, not assessment status or notes.
- Review notes now include an always-visible per-instrument Expert/Hard/Medium/
  Easy attack table, generated Hard retention, and aggregate human p10-p90
  range. Hail currently exposes guitar 1600/969/466/236, bass
  493/400/280/166, and drums 916/885/508/274 for targeted difficulty review.
- The visual review page now shows those four attack counts for the selected
  instrument and updates them immediately when the instrument selector changes.
- Corpus replay benchmarks now aggregate those neutral structure metrics when
  both roots provide them. Independent Hail replay roots matched exactly:
  19 blocks, 12 ranked boundaries, 19 repeated groove families, and 0.6299
  supported drum-attack ratio, with zero deltas across all structure fields.

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
- Rejected an offline nearest-neighbor symbolic lane baseline trained only on
  the three authorized Brad Sucks charts. Across 247 independently
  pitch-aligned single-note attacks in artist- and charter-disjoint Persephone,
  its best configuration reached 0.190 exact-lane and 0.418 movement-direction
  agreement, below the deterministic author's 0.265 and 0.515. A public-data
  audit also rejected a nonexistent search result, the proprietary Frets on
  Fire bundle and an Encore sample whose "Courtesy of" media credit does not
  clearly grant training rights. Do not fit a lane model until the authorized
  corpus has more artist and charter diversity.
- Built a local, commercially reusable four-song five-lane seed from explicit
  CC BY 4.0 and CC BY-SA 3.0 sources: 3,240 Expert attacks from two artists and
  three chart authors. Three Brad Sucks charts are training-only and Zep
  Hurme's Persephone is held out by both artist and charter. The source files,
  manifests and permission evidence remain local under `.research/`.
- Expanded the local authorized seed with 24 Fervi-authored Frets on Fire songs
  from `IntinteDAO/FretsOnFire-Songs-GuitarMaster` at commit
  `afc3f9520d5976bd526e2b48203faa974563337d`. Every admitted folder contains
  synchronized audio, Expert MIDI, named chart attribution, and a per-folder
  CC BY 3.0/4.0 or CC BY-SA 3.0 license. Five Zep Hurme songs were excluded to
  preserve artist and chart-author split isolation from Persephone; one chart
  with missing fretter attribution was also excluded. Full validation now
  covers 28 items, 12 artists, four chart authors, and 12,335 Expert attacks,
  with all 56 audio/chart rights-evidence hashes verified. Deterministic Stage B
  conversion succeeds for all 28 items. The corpus remains local under
  `.research/`; independent audio pitch events were extracted before a new
  symbolic lane baseline was fit. Basic Pitch extraction completed
  for all 24 songs. A validation-selected kNN trained on 3,625 aligned
  single-note events regressed held-out Persephone exact lanes from 0.2915 to
  0.2794 and movement from 0.4253 to 0.3801 versus the matched original-data
  baseline. Chart-author-balanced sampling was worse at 0.1943/0.3665. The
  symbolic lane fit remains blocked on additional chart-author diversity; the
  new audio/chart pairs can still supply lane-independent attack supervision.
- Added strict ingestion for legacy Frets on Fire MIDI and Clone Hero `.chart`
  files without weakening the Rock Band `PART GUITAR` requirement. The local
  seed's Basic Pitch attack recall within 80 ms ranged from 0.570 to 0.959 on
  training material but measured only 0.477 on the held-out song. This is enough
  to validate the data path, not enough to train or request musical-quality
  judgments; expand artist/charter diversity and improve held-out audio
  structure first.
- Enabled stable studio-grid inference for Band as well as electronic profiles
  whenever no sustained tempo change is detected. On the held-out Persephone
  audio this replaced a 78-231 BPM variable map with a 99.996 BPM constant map
  against the chart's evaluation-only 100 BPM tempo, raising attack F1 from
  0.5055 to 0.5142.
- Enabled the existing pitch-specific reattack/release refiner for Band guitar.
  Held-out Persephone attack F1 rose further to 0.5553, recall rose from 0.4528
  to 0.5603, and generated attacks moved from 486 to 625 against 614 human
  attacks. Across the four-song open corpus the audio-only candidate measured
  0.7181 macro attack F1 and 0.7602 recall. Chord and sustain style still vary
  substantially by chart and remain the main blockers; broad chord expansion
  was rejected because multi-pitch evidence overlapped human single notes more
  often than human chords.
- Added a reusable held-out human-chart benchmark that verifies authorized
  provenance before comparing attacks, timing, chords, sustains, lane movement,
  density, rest boundaries, repeated motifs and difficulty retention. The first
  Persephone candidate measured 0.5553 attack F1 with 18.2 ms median matched
  timing error and a 1.018 attack-count ratio, but zero matched chords, 0.0217
  sustain F1, 0.0513 phrase-boundary F1 and only 0.069 relative motif-lane
  agreement. This confirms that timing/density are ready for preliminary review
  while phrase, chord, sustain and lane authoring remain the primary work.
- Removed the two intermediate Band Guitar lane-rewriting passes from fresh
  generation and cached replay so corpus conformance now feeds the authoritative
  phrase author directly. Against the local evaluation-only Hail reference,
  0-40 second movement agreement rose from 0.256 to 0.353 and motif lane-shape
  agreement from 0.000 to 0.022. On authorized held-out Persephone, movement
  agreement rose from 0.389 to 0.518 and motif lane-shape agreement from 0.069
  to 0.290 while attack F1 remained effectively level (0.555 to 0.554).
  Listening review found the same Hail failure despite the metric gains. The
  aligned evaluation exposed a localized source-time 20-24 second attack-recall
  collapse that was hidden by the aggregate 0-40 second score. Broad focus
  bypass, octave folding, and recurrence-aware spike-filter variants were
  rejected because they regressed lane or motif quality on Hail or held-out
  Persephone. Human-chart comparisons now report fixed four-second attack
  windows before another transcription candidate is accepted.
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
- Verified the 8,970-file EGFxSet release directly against Zenodo record
  7044411: dataset audio and annotations are CC BY 4.0, and the downloaded clean
  archive matches its published MD5. Added deterministic `string-fret.wav`
  conversion with effect/pickup provenance and leakage-resistant group keys.
  EGFxSet's fixed five-second intervals are limited to attack and single-note
  polyphony targets; target-aware training losses prevent effect tails from
  becoming false sustain supervision. A real `Clean/Neck/6-0.wav` pilot
  produced one MIDI-40 attack and a 431-frame feature artifact.
  Training pilots did not pass the untouched-set gate. Adding 30 clean examples
  lowered test attack F1 from 0.7982 to 0.6374; a bounded six-example run reached
  0.7017. Seeded per-epoch shuffling produced a 0.7799 control but the matched
  six-example candidate reached only 0.7276. EGFxSet therefore remains an
  available research converter, not accepted runtime training data, and
  `hybrid-guitar-techs-v1` remains the selected checkpoint.
- Added a manifest- and hash-validated aligned-source agreement evaluator.
  Guitar-TECHS direct-input and mic/amp recordings provide two genuinely
  independent captures of each labeled performance while preserving performer
  split keys. Shared onsets are measured separately from channel-only onsets;
  the evaluator remains offline and does not alter generation.
- Verified GuitarSet v1.1.0 against its authoritative Zenodo CC BY 4.0 record
  and published checksums. Added deterministic paired-manifest pinning for its
  aligned microphone and mono pickup-mix recordings, including alternate
  SHA-256 hashes and release-archive MD5/size evidence. Across the disjoint
  train/validation/test performer splits at 50 ms, shared-onset precision was
  0.8527/0.8996/0.9260 versus best-source 0.8244/0.8518/0.8919. The direction
  persisted at 30 and 80 ms. This completes a second lawful independent-capture
  calibration without changing runtime recovery.
- Authorized chart-pair validation can now attach those existing source-
  agreement artifacts through an additive provenance summary. It distinguishes
  available, unavailable and invalid lineage; exposes report/manifest/release
  hashes, tolerance and per-split shared/best-source metrics; and separately
  reports that the current chart-pair corpus has no independent paired capture.
  No evaluation metric is recomputed from audio and generation remains unchanged.
- Reframed EGFxSet as deterministic multi-attack phrase augmentation rather
  than one five-second optimizer item per isolated note. Twelve ten-second
  phrases contain 236 known attacks, use 280 ms source excerpts with 25 ms
  fades, retain every source hash, and remain attack/polyphony-only. The
  committed `synthesize-egfxset-phrase` command exactly reproduces the measured
  sequence audio, attack times and pitches from seed 1337.
  - Untouched hybrid test attack F1 improved from 0.7982 to 0.8471 and
    polyphony accuracy from 0.8194 to 0.8250.
  - External EGSet12 attack F1 improved from 0.7994 to 0.8346 and polyphony
    accuracy from 0.6737 to 0.6995.
  - Authorized Persephone threshold attack F1 improved from 0.5235 to 0.5430;
    equal-count ranking F1 improved from 0.4536 to 0.4697.
  - Across 14 commercial stems, equal-count top-peak support rose from 0.6409
    to 0.6515, improving eight songs and tying one.
  - Hail full-song equal-count ranking F1 improved from 0.3098 to 0.3251, but
    the reviewed 20-24 second window fell from 0.5000 to 0.3860. The checkpoint
    is therefore the strongest offline research candidate, but remains blocked
    from runtime generation and does not justify a new listening review.
  - A matched phrase set with synthetic dyads raised untouched polyphony
    accuracy to 0.8441 and Persephone threshold F1 to 0.5533, but reduced
    untouched attack F1 to 0.8270, EGSet12 F1 to 0.8221, and Persephone ranking
    F1 to 0.4649. It produced only one Hail chord prediction and none on
    Persephone, while Hail's reviewed-window ranking remained below baseline at
    0.4590. The dyad variant is rejected; synthetic mixing is not yet a
    substitute for real phrase-level chord recordings.
  - Faster synthetic timing did not repair the protected Hail window. A
    six-ordinary/six-rapid mix reached 0.8163 untouched attack F1 and 0.4211 in
    Hail's 20-24 second ranking window; reducing the rapid dose to two of twelve
    phrases reached only 0.8128 and 0.4068. The latter also measured 0.8245
    polyphony accuracy and 0.3135 Hail full-song ranking F1. Both variants are
    rejected, and synthetic rapid-phrase proportion is no longer an active
    tuning direction.
  - Repeating short three- and four-note source motifs in an equal-size
    ordinary-tempo synthetic set improved EGSet12 ranking to 0.8537, but
    untouched attack F1 was only 0.8257 and Persephone lost most of the
    nonrepeating candidate's gain. Its protected Hail-window match count fell
    below both the selected model and the nonrepeating synthetic candidate, so
    it is rejected.
  - Oversampling the twelve densest real GuitarSet/Guitar-TECHS training
    recordings produced the strongest untouched attack F1 so far (0.8511) and
    EGSet12 F1 of 0.8689. It also improved Hail full-song ranking and tied the
    nonrepeating synthetic candidate on Persephone, but reduced matched-attack
    polyphony accuracy to 0.8066 and still missed the protected Hail gate.
    Halving the dose collapsed untouched F1 to 0.7536, exposing fixed-order
    sensitivity. Direct 1.5x loss weighting of rapid labeled attacks likewise
    regressed untouched F1 to 0.7639 and polyphony to 0.7517. Item duplication
    and dense-positive weighting are therefore rejected. A matched
    deterministic per-epoch shuffle then produced 0.7654 attack F1 for the
    twelve-recording dense candidate versus 0.7799 for its shuffled control, so
    the fixed-order gain is not reproducible. Future work needs a more robust
    sampling/training design before revisiting real rapid phrase emphasis.
  - The two-layer baseline's receptive field is only nine frames (about
    104 ms), so temporal-context controls were tested on the unchanged corpus.
    A three-layer dilated model spanning about 337 ms improved untouched attack
    F1 to 0.8221, polyphony to 0.8296, and EGSet12 to 0.8815, but regressed
    Persephone and both Hail ranking gates. A same-parameter 151 ms dilation
    then reduced untouched attack F1 to 0.7232 and polyphony to 0.7986.
    Both architecture controls are rejected; the original 104 ms network is
    restored.
  - A validation-selected 5% symmetric-periodicity prior now accompanies raw
    attack ranks in offline prediction artifacts. It reranks existing peaks
    only: no attack or timestamp is invented. Cleared validation equal-count
    F1 improved from 0.8533 to 0.8584; EGSet12 improved from approximately
    0.7994 to 0.8172; Persephone tied at 274/614; Hail full-song matches rose
    from 540 to 542 while its protected 20-24 second window held at 18/31.
    Fourteen-song raw-event support was neutral-to-positive (0.64091 to
    0.64116; five wins, four ties, five losses). The prior remains offline
    evidence because it also strongly supports a known false Hail reattack;
    independent `model_start` evidence is still mandatory for recovery.
    A conservative recovery audit requiring periodic rank at least 0.90,
    periodic support at least 0.50, and independent `model_start` evidence
    found 9/11 Hail proposals but only 2/4 Persephone proposals in the
    authorized chart. Stricter thresholds isolated a Persephone false
    positive, so periodic ranking is not a safe post-authoring recovery gate.
    Weighting support by exact interval symmetry fell below raw validation
    ranking, while requiring a four-peak run improved raw validation ranking
    but underperformed the retained three-peak prior. Neither refinement is
    retained.
  - Peak-width rank provides independent score-curve evidence without adding or
    moving candidates. A validation-selected blend of 90% periodic score and
    10% peak-width rank improved cleared equal-count matches from 2027/2366 to
    2040/2366, with six wins, six ties, and one loss across thirteen
    recordings. Frozen evaluation improved Hail by three matched attacks at
    both tested selection counts while preserving its protected 20-24 second
    window, improved Persephone by five, and improved every EGSet12 recording
    (638 to 657 total matches). Fourteen-song commercial raw-event support rose
    from 0.64124 to 0.64682 with twelve wins, one tie, and one loss. Prediction
    artifacts expose the width and combined ranks for offline analysis;
    generation and attack recovery remain unchanged.
  - Deterministic multi-recording gradient accumulation was tested as a remedy
    for fixed-order sensitivity. Four-recording steps raised untouched test F1
    from 0.7982 to 0.8278, Hail ranking by 44-61 matches, its protected window
    by two, and EGSet12 from 657 to 701 matches. It nevertheless reduced
    fourteen-song macro support from 0.64682 to 0.64456, including large
    regressions on When You Were Young and Empty Walls. Two-recording steps
    reduced validation F1 to 0.7957 and polyphony accuracy to 0.4958. Gradient
    accumulation and dense-real duplication under accumulation are rejected;
    the original per-recording optimizer remains selected.
  - The 24 newly authorized GuitarMaster full mixes were converted into
    lane-independent structure features and interleaved through the unchanged
    186-recording training corpus. Recomputing class weight from the expanded,
    sparse full-mix corpus regressed validation/test attack F1 to 0.8124/0.7924
    and polyphony to 0.5055/0.6218. Holding the selected baseline's 26.25
    positive weight improved validation/test F1 to 0.8417/0.8511 and raised
    EGSet12 combined-rank matches from 658 to 720, but broad commercial support
    fell from 0.63536 to 0.62898 with seven losses. Large regressions on Beast
    and the Harlot, Less Talk, and When You Were Young reject this checkpoint
    and the experimental weight override. The licensed full mixes remain
    research evidence rather than selected training inputs.
  - Low-rate one-, two-, and four-epoch refits of the strongest full-mix
    checkpoint back onto the original corpus restored cleared attack metrics.
    The four-epoch result reached validation/test F1 of 0.8325/0.8455 and kept
    gains on EGSet12, Persephone, and Hail. It nevertheless reduced the broad
    commercial proxy from 0.63536 to 0.62576 with eleven losses, including 60
    fewer supported Less Talk attacks. Warm-start refitting and its temporary
    trainer controls are rejected.
  - A 25% adjacent-frame soft attack target reduced untouched attack F1 to
    0.7395 and was removed. A fixed eight-beat phrase bound appeared to improve
    Persephone against an older baseline, but an isolated comparison against
    the current no-bound replay showed no such gain. It also regressed Hail
    exact lanes from 0.1761 to 0.1472 and protected-window exact lanes from
    0.2857 to 0.1905.
  - Phrase bounds of 8, 12, 16, and 24 beats were then selected only on the
    three rights-cleared development songs. Twelve beats gave the strongest
    mean movement agreement (0.4521 to 0.4956) and motif shape (0.1850 to
    0.1948), though mean exact lanes fell from 0.1548 to 0.1037. Frozen
    evaluation preserved Persephone movement and motifs but lowered exact
    lanes from 0.2624 to 0.2595. Hail movement rose only from 0.3877 to 0.3936
    while exact lanes fell from 0.1761 to 0.1544, motif shape from 0.0238 to
    0.0060, and protected-window exact lanes from 0.2857 to 0.2381. Fixed
    phrase bounds are rejected; rest-based boundaries remain selected.
  - Variable-tempo grid-supported playability recovered two attacks in Less
    Talk's first reviewed riff and one in the second with only 17 additions
    song-wide, but lane-sequence edit distance changed from 7+6 to 7+7.
    Hard exact internal motif reuse made the repeated detected pitches
    consistent but increased collapsed changes and did not repair the
    evaluation pattern. Both chart-authoring experiments are rejected.
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
