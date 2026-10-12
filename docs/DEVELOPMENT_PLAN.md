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
- A later 22-song sweep rejected a further reduction from 0.38 to 0.32 beats.
  Official Hard attack F1 improved on all seven songs, but movement agreement
  regressed on Hail and When You Were Young. Across 15 authorized held-out
  charts, macro Hard F1 improved 0.3187->0.3269 and movement 0.3620->0.3669,
  but exact lanes fell 0.1482->0.1454 and mean interval error worsened
  1.6408->1.6865. The two low-retention failures were especially unsafe:
  Unbutton movement fell 0.6667->0.5333 and Astronaut exact lanes fell
  0.1071->0.0690. Adaptive use only on low-retention songs is therefore also
  rejected; they require stronger selection evidence rather than more notes.
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
- `compare-midi` can explicitly compare different generated and reference
  difficulties. A Hail generated-Hard versus official-Expert A/B check, using
  one Expert-derived frozen alignment, showed the denser Hard candidate improved
  attack F1 0.4750→0.5182 and matched 74 of 119 added attacks while exact lanes
  declined 0.1474→0.1330. This supports attack density/timing but not the current
  lane pattern; the result remains post-generation evaluation only.
- Direct `midi\notes.mid` comparisons now recover `lead_seconds` from the
  containing completed run just like run-folder comparisons. This removed a
  silent 3.844-second Hail evaluation mismatch: the corrected 50 ms,
  -0.70-second-offset Expert comparison is 623/1,150 matched attacks, precision
  0.3848, recall 0.5417 and F1 0.4500. The earlier apparent 4–8 second collapse
  was a comparison artifact; the current 20–24 second window matches 18/31
  reference attacks. Reference notes remain post-generation evaluation only.
- Automatic onset-only reference alignment now preserves its top eight offset
  candidates and emits an explicit repeated-phase warning in JSON, Markdown and
  CLI output. Hail's onset-only optimum was -1.10 seconds, while independent
  guitar-stem support selected -0.735 seconds, within 35 ms of the manually
  reviewed -0.70-second overlay; the -0.735-second comparison reached 682/1,150
  matches and F1 0.4926. Naive stem-onset selection was rejected as an automatic
  replacement: it chose the accepted manual phase on only one of seven
  official-reference songs. Manual synchronized review remains authoritative.
- Human A/B/C review reframed that result: generated Hard may expose a cleaner
  attack-selection backbone for canonical Expert rather than merely an easier
  chart. On Hail, project Expert charts 1.3913 times the official Expert attack
  count with 0.5288 precision; project Hard charts 0.8426 times the count with
  0.5666 precision. Both remain near 0.133 exact-lane agreement. Before any
  runtime change, test whether reduction-retained versus Expert-only attacks
  separate on source-derived evidence across rights-cleared songs. Official
  notes may evaluate the frozen result but cannot select that evidence or rule.
  Human review does not support treating intro density alone as the defect:
  generated Expert can approach reference attack density while choosing the
  wrong attacks and lane progression. Optimize source-supported selection and
  phrase continuity, not a blanket reduction in intro note count.
- Saved-stage tracing of Hail's repeated-yellow windows identifies two separate
  defects. The player 16.7-18.6 second run is dominated by repeated detected
  pitch 70, but the player 31.8-33.0 second Hard run maps detected pitches
  63/73/73/75/77 entirely to yellow. Expert still has one red and one blue in
  that latter window; spacing-based Hard reduction drops them. Add diagnostics
  for collapsed nonzero pitch intervals and evaluate contour-aware reduction
  on rights-cleared material before changing either pitch selection or lane
  authoring.
- Expert repeated-pitch anchoring now protects an adjacent interval when the
  pre-anchor fingering already moves in the same direction as source pitch.
  Isolated cached replay reduced collapsed Expert transitions from 22 to 3,
  32 to 8, 56 to 50 and 25 to 11 on the three development songs plus held-out
  Persephone, without changing attacks or timing. Held-out movement agreement
  improved 0.5178 to 0.5330 while exact color agreement shifted 0.2624 to
  0.2536; Hail post-generation evaluation improved exact lanes 0.1336 to
  0.1430 and movement 0.4071 to 0.4390. Retain this source-derived contour
  safeguard, but do not treat it as a fix for repeated same-pitch voice collapse
  or difficulty-reduction sampling.
- Band lead selection now defers alternate-register recovery until a compact
  recurring pedal pattern is established. A raw pYIN register must agree with
  a Basic Pitch candidate, oppose both neighboring anchored pitches, recur at
  least four times within three beats against the same local register, and
  survive the sequential-outlier check. Unsupported proposals are restored
  before downstream charting; held-out Persephone is event-for-event identical
  to the prior lead transcription. On the three development songs attack F1
  changed +0.0008/+0.0002/+0.0013, while Hail's reviewed player 17.65-18.62
  second window recovered alternating 51/58 pedal notes instead of a repeated
  pitch-70 line. This is source-derived voice evidence, not permission to use
  reference-chart notes for register selection.
- Inside those independently supported alternate-register runs, simultaneous
  strong candidates are now treated as competing voices only when the entire
  group remains at least one octave from the supported family. This removes a
  displaced harmonic layer without collapsing a root-containing chord. The
  rule changes four Hail groups but is event-for-event inert on held-out
  Persephone, three other rights-cleared replays, Mr. Brightside, Cliffs of
  Dover and Joker and the Thief. Frozen Hail Expert evaluation changed attack
  F1 0.6091 to 0.6096, exact lanes 0.1791 to 0.1801 and motif coverage 0.2772
  to 0.2857; overall movement shifted 0.4317 to 0.4309 while the reviewed
  handoff window improved 0.6078 to 0.6275. Keep the saved A/B as the deciding
  listening check rather than broadening this gate from one song's result.
- Hard guitar reduction now checks only five-plus-note same-lane runs whose
  retained source pitches span at least four semitones. It may swap up to three
  retained attacks for omitted Expert contour anchors when the resulting Hard
  chart keeps at least 0.32 beats between attacks and the anchor is no more than
  0.05 less confident than the attack it replaces. It never adds attacks or
  invents lanes/source IDs. The rule was an exact no-op on all four
  rights-cleared replay songs, including held-out Persephone. On Hail it
  replaced three yellow selections with existing blue Expert anchors, retained
  the recovered low pedal notes, and removed the player-17 collapsed-contour
  review marker. Frozen post-generation Hard-versus-Expert timing F1 remained
  0.5212; movement improved 0.3692 to 0.3728 while exact lanes shifted 0.1588
  to 0.1570. Keep the saved A/B as a required feel review rather than treating
  the official chart as runtime evidence.
- Added post-generation playability-evidence audits that join each attributed
  removal to raw onset/level evidence, optional offline structure ranks,
  source-mix versus guitar-stem onset prominence/width, and voice-continuity
  windows. Optional reference labels measure separability only after generation
  and remain excluded from authoring.
- Added synchronized local A/B review pages for completed drafts. They reject
  mismatched song identity, source hash, tempo map or lead-in before presenting
  one shared audio transport, instant same-timestamp version switching,
  version-only attack highlighting and densest-change jump markers. This closes
  the gap between measured replay selection and an actionable listening review
  without making either draft an input to the other.

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
- Exact repeated pitch-and-rhythm riffs now report a song-level repeatable
  fingering ratio. Final authoring may normalize the absolute lane offset of an
  exact four-note repeat only when every occurrence has the same relative lane
  steps. Overlapping windows are collected before mutation and applied only
  when all proposals agree for every affected source note; structurally
  different variants remain untouched. Automatic review flags guitar charts
  below 80% agreement when at least twelve repeated occurrences exist.
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
- A first strict-majority reuse check improved exact-repeat agreement but left
  Hail unchanged and slightly reduced frozen Cliffs movement, so it was
  superseded. Conflict-free offset consensus raises Hail from 0.7527 to 0.8011,
  Cliffs of Dover from 0.6437 to 0.6667, Mr. Brightside from 0.8049 to 0.8537
  and held-out Persephone from 0.5833 to 0.8750. Frozen Hail movement remains
  0.4496 while interval error improves 1.6657 to 1.6597. Frozen Cliffs movement
  remains 0.3440, exact lanes improve 0.1744 to 0.1816 and interval error
  improves 1.7352 to 1.7336. These metrics are diagnostics and post-generation
  evaluation, not reference-fed selection.
- Added an evidence-aware final-lane quality gate. Five-plus-note same-lane
  runs fail automatic assessment only when they contain at least three detected
  pitches spanning four semitones and at least three actual pitch-change
  transitions; eight-plus-note repeated-pitch runs instead request listening
  review because recoloring them would invent unsupported motion. The added
  transition gate reclassified only one seven-note authorized run whose first
  five attacks repeat pitch 70 before a two-note tail; Unbutton retains its
  separate evidence-backed collapsed contour. Current-code Hail has two
  repeated-pitch review runs and no collapsed contour runs. Fifteen cached
  authorized replays originally exposed six collapsed-contour runs across 31
  long runs. A guarded local-fingering fallback activated zero times on Hail and
  all 15 authorized runs, so it was rejected rather than adding an unproven lane
  rewrite. Attacks, source IDs and timing were unchanged.
- Repeat-fingering normalization now rejects an individual offset plan when
  applying it would create a new evidence-backed collapsed contour at an
  occurrence boundary. Across 15 authorized and seven official-reference
  cached replays, four plans were rejected: three authorized songs and
  Mr. Brightside each lost one collapsed-contour failure. Authorized aggregate
  collapsed pitch changes fell 340→335; official-reference changes fell
  215→214; reversals, attacks, timing, chords and all difficulty counts were
  unchanged. Held-out attack, exact-lane, movement and motif metrics were
  unchanged; one unmatched-note mean interval error shifted 1.4222→1.4370.
  Hail was unchanged. When You Were Young's remaining collapse predates phrase
  authoring and includes a 32-semitone source-pitch outlier, so this guard
  correctly does not invent a lane repair there. Follow-up source inspection
  found that the pitch-98 event has no local pYIN support, but removing it would
  still leave a five-note top-lane run spanning pitches 66-75. The actual
  collapse is introduced when the long-run supported-span pass restores the
  globally correlated incoming pitch mapping across a 127-note phrase. Two
  source-only safety variants were rejected: rolling back the whole phrase
  removed the failure but reduced held-out movement-direction agreement
  0.3587->0.3478, while rolling back only the collapsed window preserved
  movement and interval error but reduced exact-lane agreement 0.1008->0.0982.
  Attack F1 remained 0.5505. The existing `needs_work` diagnostic remains the
  safe outcome until stronger source evidence can justify a lane change.
- Guitar mapping now legalizes three-note Expert chords that span green through
  orange by moving one outer gem inward while preserving the detected attack,
  triad, timing and source IDs. Structural validation also rejects forbidden
  Expert and reduced-difficulty guitar shapes before packaging. Seven official
  and 15 authorized cached replays changed only the affected chord lanes:
  Beast and the Harlot's single illegal triad and all 17 illegal triads across
  four authorized songs were removed. Aggregate attacks, chords, sustains,
  density, contour changes and difficulty counts were unchanged; four songs
  improved from `needs_work` to `review_closely`. Beast's official attack,
  exact-lane, movement and interval metrics were unchanged. Authorized attack,
  chord, movement, interval and motif metrics were unchanged; Your Day's exact
  lane agreement shifted 0.0674->0.0638 because its held-out chart contains the
  matched shape the Xbox format forbids, so structural playability takes
  precedence over that diagnostic.
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
- Hail listening feedback identified the unexpected player-time 15-second
  descent inside one of these windows: its upper register has 76% of weighted
  evidence, 78% of events are weak, and the detector briefly drops
  73→68→63 before returning upward. A post-mapping minority-register filter was
  tested so it could not perturb the song-wide pitch dictionary. It was rejected
  on rights-cleared selection before Hail evaluation: Persephone Expert attack
  F1 fell 0.5537→0.5363 and exact lanes 0.2624→0.2523. The diagnosis is retained;
  confidence/register heuristics remain disabled.
- Band-guitar lead-focus reports now preserve non-mutating raw pYIN lineage for
  up to 240 guided onset groups: the raw guide, octave-anchored guide, selected
  pitches, and selected source IDs. Hail exposes recurring selected pitches
  nineteen semitones above its raw 51/58 family, but a runtime correction was
  rejected. Even after strict confidence, family-span, and source-lineage
  gates made the rule an exact chart no-op on Cliffs of Dover, Mr. Brightside,
  Joker and the Thief, and held-out Persephone, Hail movement-direction
  agreement fell 0.4496→0.4318, mean interval error worsened 1.6597→1.6920,
  and attack F1 fell 0.6147→0.6139. The diagnostic remains evidence for a
  future richer voice-role model; source-pitch rewriting stays disabled. A
  direct non-mutating lane constraint was also rejected: nine local Hail lane
  changes reduced exact agreement 0.1528→0.1492, movement-direction agreement
  0.4496→0.4254, and increased interval error 1.6597→1.6999.
- Band-guitar chord sections now distinguish fifth/fourth-dominant harmonic
  evidence from generic octave doubling before propagating a repeatedly
  supported root to intervening attacks in the same bounded evidence run. The
  gate is an exact chart no-op on three rights-cleared songs plus Cliffs of
  Dover, Hail to the King, Mr. Brightside, Joker and the Thief, and Persephone.
  Frozen Jonas evaluation improves chords 72→144 and chord F1 0.1538→0.2678,
  with exact lanes 0.1047→0.1279 and unchanged motif coverage. Propagated
  texture remains explicitly review-gated rather than receiving an automatic
  close-enough label.
- Phrase-level motif projection now remains available across chord continuity
  anchors rather than being silently skipped whenever a preceding single-note
  anchor exists. The four-song authorized check has no metric regressions and
  improves Certain Death mean interval error 1.2667→1.2000; frozen Jonas and
  Hail are unchanged. Expanded authorized evaluation also adds dense Things
  Change as a counterexample for future chord classifiers: count-calibrated and
  active-overlap rules were rejected because they selected the wrong attacks.
- Band voice focus now uses the same source-only fifth-versus-octave dominance
  evidence to prefer a near-equal lower root only when the monophonic pYIN guide
  is unavailable. A 95%-confidence floor avoids the movement regression from
  broader root selection. The gate is inactive on four authorized songs and
  the eight-song commercial safety set; frozen Jonas improves chord F1
  0.2678→0.2957 and exact lanes 0.1279→0.1341 with movement and motif coverage
  unchanged.
- Fifth-dominant Band passages can now reuse a four-attack candidate contour
  only when the same direction, span, and endpoint family has at least three
  non-overlapping source occurrences and each selected candidate remains
  within 80% of the strongest local confidence. Rights-cleared Things Change
  improves attack F1 0.4893→0.4902, chord F1 0.2031→0.2050, exact lanes
  0.1421→0.1550, movement 0.2864→0.2941, mean interval error
  1.8644→1.8312, and motif coverage 0.2336→0.2617. The source gate is inactive on
  Certain Death, Dirtbag, There's Something Wrong, Hail, Cliffs of Dover,
  Mr. Brightside, Joker and the Thief, and Persephone. Jonas passes the
  fifth-dominance gate but has no qualifying three-occurrence candidate family,
  so its generated MIDI and frozen metrics remain unchanged.
- Frozen Jonas motif diagnosis confirms that final fingering is not the main
  blocker: among 90 covered official repeated single-note windows, the selected
  source-pitch contour agrees with the official lane contour only four times.
  Accepted pre-focus candidates contain a contour-compatible path in 48 of the
  90 windows, but most do not form a stable three-occurrence source family.
  Future work should therefore improve transferable voice ownership rather
  than relax lane or motif matching globally.
- Evaluation stage metrics now report source-pitch contour agreement for
  covered repeated reference windows. This keeps the voice-selection ceiling
  distinct from final lane-shape agreement: frozen Jonas reports 4/90
  (`0.0444`) at the final authoring stage, while the reference chart remains
  post-generation evaluation only.

## 4. Review and anomaly tools

The final quality target is now executable rather than aspirational.
`assess-official-quality` requires an authorized comparison corpus with at
least 20 songs and five artists; at least 80% of songs must independently pass
near-official attack, chord, sustain, exact-lane, movement, weakest-window and
repeated-motif thresholds. Commercial references remain frozen post-generation
evaluation only. The first coverage-complete scorecard measures 20
rights-cleared songs from
14 artists and passes 0/20. Coverage is no longer the release blocker. Across
the corpus, threshold pass counts are 2/20 for attack F1, 2/20 for chord F1,
0/19 for applicable sustain F1, 0/20 for exact lanes, 0/20 for movement,
0/14 for applicable motif coverage and shape, and 1/20 for weakest-window
recall. Median attack, exact-lane, movement and sustain F1 values are 0.415,
0.104, 0.382 and 0.032 respectively. This baseline confirms that the remaining
gap is model/authoring quality, not missing evaluation breadth; thresholds must
not be relaxed to manufacture release readiness.

The coverage-complete corpus also rejects two tempting attack-density fixes.
Increasing the existing transcription confidence threshold monotonically
reduced artist-disjoint macro attack F1, and a deterministic linear classifier
over confidence, onset strength, attack strength, evidence score, level, RMS,
polyphony, duration, beat alignment, repetition and local gaps changed held-out
macro F1 from 0.3524 to 0.3482. Current event heuristics do not distinguish
chart-worthy attacks. The validation-selected 0.70 full-mix structure model was
then frozen and tested on four authorized songs excluded from its full-mix
training. Direct attack F1 was mixed: Certain Death 0.7160 to 0.7100, Dirtbag
0.4613 to 0.5630, There's Something Wrong 0.7570 to 0.8478, and Persephone
0.4558 to 0.1584. Requiring agreement between existing and structure attacks
regressed three of four songs. Neither confidence pruning nor structure gating
is accepted into generation.

A phrase-context bidirectional-GRU pilot grouped raw guitar candidates within
30 ms and learned attack selection plus bounded 80 ms timing correction using
only the strict authorized train/validation split. A guarded three-seed
ensemble improved validation attack F1 from 0.6882 to 0.7021 while preserving
recall and weakest-window recall, but the frozen Persephone test changed F1
from 0.60137 to 0.60068 and precision from 0.51891 to 0.51452. A Hail
post-generation shadow also lost raw-event recall, from 0.7678 to 0.4322, and
reduced the weakest four-second recall to zero. Direct selection is rejected.
Applying only the ensemble timing head to the unchanged accepted event mask
improved frozen Persephone median timing error from 28.55 to 26.49 ms, but
validation timing worsened from 22.63 to 22.65 ms and validation F1 also
declined because corrected attacks crossed the fixed matching boundary. Timing
correction is therefore rejected from runtime as well.

An offline phrase-level source-pitch path then tested evidence scores, bounded
pitch-jump costs, octave penalties and one-second rest resets. Authorized
validation lane-direction agreement changed from 0.5889 to 0.5852 micro
(0.3967 to 0.3946 macro), so the candidate was rejected before frozen-test or
Hail evaluation. These negative results reinforce that continuity alone cannot
identify chart-worthy attacks or transferable guitar voice ownership; no
selector, timing correction or voice path from these pilots is wired into
normal generation.

Band sustain failure is not a simple authoring-threshold problem. Replaying
pre-authoring durations with sustain cutoffs from 65 to 350 ms produced
effectively identical artist-disjoint results because source durations are
mostly shorter than 65 ms or longer than 450 ms. Structure prediction artifacts
now expose the sustain head's active-run duration and a next-attack-bounded
duration for controlled evaluation. At the frozen 0.70 attack threshold,
bounded sustain F1 on four chart-author-held-out songs was 0.0000, 0.0187,
0.5299 and 0.0000. The isolated There's Something Wrong gain is insufficient;
learned sustain remains diagnostic-only and is not applied to charts.

An exhaustive public-source search found no additional synchronized five-lane
guitar/audio corpus with explicit commercial-use rights for both sides and
named chart attribution. GitHub mirrors either duplicate the existing
IntinteDAO/Fervi pack, omit chart or audio rights, use noncommercial licenses,
or contain commercial songs. The next lawful route to chart-author diversity
is opt-in commissioning or direct permission from additional charters for
already CC-licensed audio; provenance requirements are not relaxed.

The existing 28 pairs nevertheless support one strict charter-disjoint
structure experiment: 24 Fervi GuitarMaster songs trained the model, the three
Brad Sucks songs (MB1Nightmare/Greninjo and JRabes) formed validation, and
Valerio Bozzolan's Persephone remained untouched test. The chart-only model
reached validation F1 0.5916 but only 0.4405 on Persephone, with 0.2844
precision and 0.9772 recall. Adding the same held-out songs to the broader
GuitarSet/Guitar-TECHS/Slakh experiment reduced aggregate validation/test
attack F1 to 0.4588/0.5056 versus the selected diverse baseline's
0.8417/0.8511. Both checkpoints are rejected. The evidence now shows that
single-charter full-mix supervision does not transfer safely even when artist,
audio, and chart-author boundaries are enforced.

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
- An initial unweighted focal-loss result appeared to raise validation/test
  attack F1 from 0.4588/0.5056 to 0.6872/0.7099, but those were micro-aggregates
  across mixed transcription datasets and concealed the actual chart-domain
  result. Reproduction with per-song reporting scored untouched Persephone at
  only 0.1123 when the threshold was selected broadly. Selecting one global
  threshold solely on the three authorized validation charts recovered
  Persephone to 0.4268, still below the 0.4405 chart-only baseline, while
  overpredicting the broad test split. The focal checkpoint is rejected.
  The trainer retains focal loss as a reproducible experiment option, supports
  validation-only dataset calibration, and now reports per-song, per-dataset,
  micro and macro results to prevent aggregate masking. No learned output is
  wired into normal generation.
- A six-layer residual dilated temporal convolution expanded receptive context
  from about 104 ms to about three seconds. Under the same focal loss, seed,
  epochs and authorized-only threshold calibration, authorized validation
  macro F1 fell from 0.6933 to 0.4741 and untouched Persephone F1 fell from
  0.4268 to 0.2825. This TCN is rejected. Architecture identifiers remain in
  the offline trainer and checkpoints so future temporal experiments are
  auditable and prediction restores the exact trained architecture.
- The exhaustive lawful-data search found no additional ready-made corpus at
  meaningful scale with commercially usable rights for both synchronized audio
  and five-lane charts. Added deterministic procedural guitar synthesis as a
  train-only acquisition path: it renders additive plucks, repeated riffs,
  chords and sustains with no source recording or annotation, records complete
  generation provenance, and uses one generator split key to prevent synthetic
  leakage into validation or test. This data remains an offline experiment and
  is not evidence of human chart-author diversity.
- A 24-phrase procedural pilot initially raised authorized validation macro
  attack F1 from 0.6933 to 0.8640 at seed 0 and earned one frozen Persephone
  evaluation, which improved from 0.4405 to 0.5407. Paired validation-only
  repeats showed that the gain was not robust: procedural-minus-baseline macro
  F1 was +0.1707, -0.2249, and -0.1400 for seeds 0, 1, and 2. The procedural
  batch is rejected as a selected training recipe. The generator remains
  available for future augmentation research, but its output is not included
  in normal training or runtime generation.
- Added eight independently credited Debian-pack pairs after verifying the
  Sectoid MIT and Muldjord CC BY-SA 3.0 notices, all 16 source audio/chart
  hashes, and synchronization against isolated guitar stems. The initial audit
  used an uncentered 2048-sample onset window and therefore biased every offset
  by about -46 ms. A center-compensated reaudit measured actual per-song offsets
  from +23.22 to +69.66 ms. Applying those offsets to authoring examples,
  structure attacks and beat labels gives all eight a corrected admission pass
  and seven a strict near-1:1 pass; Internal Degeneration retains a 58.05 ms
  middle-third residual. The resulting local corpus has 36 pairs, 14 artists
  and six chart-author groups. Sectoid remains train-only, Muldjord
  validation-only, and Persephone remains untouched test. Muldjord checkpoint
  distribution still requires a separate ShareAlike assessment.
- Three matched weighted-BCE seeds rejected immediate use of the expanded
  corpus. Common Brad validation macro F1 changed 0.5651 to 0.6573 at seed 0,
  but 0.7147 to 0.5806 and 0.8346 to 0.5518 at seeds 1 and 2. Matched focal
  runs were also mixed and substantially weaker, with expanded common-domain
  macro F1 values of 0.2230, 0.1947 and 0.3783. No expanded-corpus checkpoint
  earned frozen Persephone evaluation and nothing is wired into generation.
- A multi-scale temporal model combined the existing local encoder with pooled
  approximately 0.4, 1.5 and 3.0 second context. Across three expanded-corpus
  seeds it improved validation macro attack F1 from 0.6351/0.6144/0.6122 to
  0.7597/0.7008/0.7221 while raising aggregate precision from about 0.48 to
  0.80. Validation-selected seed 0 improved frozen Persephone attack F1 from
  0.3559 to 0.5616 (precision 0.2219 to 0.4323), but median timing error
  worsened from 11.61 to 34.83 ms and P95 from 34.83 to 46.44 ms. A separate
  local attack head did not match the context model's validation F1. Snapping
  context-selected peaks to local maxima worsened validation F1/timing, and a
  guarded score blend restored Persephone timing only by collapsing F1 to
  0.2534. The context model is retained offline as a substantial attack-ranking
  result, but timing prevents runtime integration.
- Adding positive spectral flux as an explicit framewise onset channel to the
  multi-scale model reduced seed-0 validation macro F1 from 0.7597 to 0.7025
  and worsened median timing from 14.93 to 23.22 ms. The regression is decisive;
  additional seeds and untouched-test evaluation were skipped. The optional
  feature remains available for auditable research checkpoints but is disabled
  by default and not used in generation.
- A materially distinct joint timing representation added a bounded
  four-frame residual-offset head to the multi-scale encoder. Supervision maps
  every frame within about 46 ms of an authorized attack directly to the
  continuous displacement required to reach that attack, rather than snapping,
  blending scores, reranking fixed candidates, adding localization loss, or
  exposing spectral flux. End-to-end joint training decisively regressed
  validation macro F1/recall from 0.7597/0.7181 to 0.4919/0.4405 and macro
  median timing from 14.93 to 28.20 ms. A guarded second trial initialized the
  accepted seed-0 selector, froze every existing weight, and trained only the
  offset head. It exactly preserved validation macro F1 0.759665, recall
  0.718132, macro median 14.927 ms and macro P95 38.147 ms: learned rounded
  offsets were behaviorally a no-op. The predeclared gate required non-regressed
  F1/recall/P95 and a strict median-timing improvement, so untouched Persephone
  was not evaluated. The offset architecture remains only to make the negative
  result reproducible; it is rejected from runtime.
- A second primary-representation experiment replaced framewise peak
  classification with 46.44 ms non-overlapping event cells. One head predicts
  whether a cell contains a chart attack and a second predicts its exact
  four-way frame phase, so attack selection and timing are supervised as an
  object-detection problem rather than inferred from score-peak location. The
  best seed-0 validation run collapsed to the weighted class-prior solution:
  macro precision 0.2260, recall 1.0000, F1 0.3644, macro median timing
  24.879 ms and macro P95 46.440 ms. Reducing phase loss tenfold, removing it,
  and training new cell heads on the frozen 0.7597-F1 multi-scale encoder did
  not recover discrimination. The predeclared validation gate failed, so
  untouched Persephone was not evaluated. This rules out the tested fixed-grid
  cell representation without spending frozen-test evidence or weakening the
  artist/chart-author-disjoint split.
- Beat-synchronous supervision then represented every song on a 12-position
  grid per beat, with beat-domain context spanning approximately 0.75, 3 and
  8 beats. This was motivated by an authorized-validation oracle: all 7,131
  attacks mapped within 50 ms, with effectively zero median and P95
  quantization error. Training from scratch with the existing attack,
  polyphony and sustain supervision reached only 0.4835 macro F1 despite
  1.0000 recall. Fine-tuning the selected 0.7597-F1 primary multi-scale model
  directly against grid positions improved macro timing to 12.058 ms median
  and 27.578 ms P95, but precision/F1 remained only 0.3215/0.4818 because
  nearby grid positions were not suppressed. This was direct primary-model
  supervision, not post-hoc snapping; the validation F1 gate failed and
  untouched Persephone was not evaluated. The grid oracle remains evidence
  that beat timing can represent these charts, but not that the tested audio
  model can select the correct grid positions.
- A factorized frame model next separated timing from chartability inside the
  primary representation: a local encoder produced the only sharp attack
  probability, while pooled 0.4/1.5/3.0-second context produced a separate
  gate smoothed over roughly 197 ms. Multiplying those probabilities was
  intended to let context suppress non-chartable local attacks without moving
  their peaks. Seed-0 validation reached 0.7158 macro F1, 0.6261 recall,
  19.90 ms macro median timing and 36.49 ms macro P95. It preserved useful
  precision (0.8273 micro) and improved P95 relative to the selected model,
  but missed both F1/recall and median-timing gates. Untouched Persephone was
  not evaluated. The architecture is retained only as an auditable negative
  result and is not used by generation.
- A hierarchical beat model directly predicted each beat's attack count
  (zero through four) and ranked its 12 subdivisions, eliminating adjacent
  grid overprediction by construction rather than peak suppression. Validation
  macro precision/recall/F1 reached 0.7249/0.7300/0.7166 after validation-only
  count calibration. For matched attacks, macro median timing was effectively
  zero and P95 0.105 ms, confirming that the synchronized authorized charts are
  overwhelmingly beat-grid authored. However, whole-song selection remained
  below the 0.7597 F1 gate, especially on one Brad validation song, so frozen
  Persephone was not evaluated. This is the strongest timing representation
  tested in this work window but is rejected from runtime until beat occupancy
  generalizes without losing attack selection.
  Inverse-frequency count supervision was also tested and was effectively
  identical at 0.7167 F1; count imbalance was not the limiting factor.
- A learnable transient pyramid added signed per-frequency center-surround
  filters over approximately 35, 105 and 383 ms, initialized as temporal
  contrast operators but optimized jointly with local and multi-scale context.
  Unlike the rejected scalar spectral-flux channel, this retained 128-band
  signed transient structure at every scale. Seed-0 validation nevertheless
  regressed sharply to 0.5428 macro F1, 0.8481 micro recall, 34.83 ms median
  timing and 46.44 ms P95. It strongly overpredicted attacks and failed every
  selection/timing gate, so further seeds and untouched Persephone evaluation
  were skipped. The result rejects this transient representation as trained;
  it is not a runtime feature.
- A causal/local-context fusion then replaced the symmetric local encoder with
  two strictly left-looking convolutions and predicted attacks jointly from
  causal local evidence plus symmetric pooled context. Seed-0 validation
  reached only 0.5913 macro F1, 0.5292 recall, 24.88 ms macro median timing and
  39.81 ms macro P95. Causality did not preserve the selected model's attack
  ranking and did not improve timing. Untouched Persephone was not evaluated;
  the architecture is rejected from runtime.
- Multi-resolution source features then addressed the fixed 2048-sample
  analysis window directly. Aligned 512/2048/8192 FFT mel representations
  improved validation macro timing to 11.61 ms median and 29.85 ms P95, but
  macro F1/recall fell to 0.6803/0.5908. Replacing only the primary log-mel
  input with a 512-sample window raised F1 to 0.7274 and recall to 0.8003, but
  timing was 14.93/41.46 ms. Deriving both log-mel and per-band positive
  deltas from the short window produced 0.7277 F1 and 0.9004 recall with the
  same timing. A separate short-local/long-context dual branch scored 0.6597
  F1 and 21.56/41.46 ms. These results establish a real timing/selection
  tradeoff: short and multi-resolution windows transfer useful timing, but the
  tested joint models lose too much chartability precision. None passed the
  validation gate, untouched Persephone was not evaluated, and no feature is
  integrated into generation.
- Distilling multi-scale chartability probabilities into the precise
  short-window student improved its best seed-0 validation macro F1 to 0.7403
  at teacher weight 0.5 while retaining 11.61 ms median timing. Weights
  0.1/0.25/1.0 reached 0.7317/0.7372/0.7258; the strongest teacher weight also
  worsened median timing to 13.27 ms. None preserved the multi-scale baseline's
  0.7597 F1 while strictly improving timing, so untouched Persephone was not
  evaluated and short-context distillation is rejected.
- Blending the distilled short-window scores back into the multi-scale teacher
  succeeds where the earlier unrelated local/context blend failed. At student
  weight 0.15, seeds 0 and 2 passed the strict validation gate: seed 0 reached
  0.7631 macro F1 and 9.95 ms median timing, while seed 2 reached 0.7731 and
  13.27 ms; seed 1 narrowly missed the strict timing improvement. The
  corresponding untouched Persephone F1 values improved from the teacher's
  0.5616 to 0.6070 and 0.5781, with frozen timing unchanged at 34.83 ms. An
  average of all three students passed validation at 0.7712 F1/12.44 ms and
  reached 0.5951 on Persephone. This is the strongest attack-selection family
  so far, but seed-sensitive frozen gains and unchanged frozen timing keep it
  offline pending broader author-disjoint evidence and end-to-end evaluation.
- The seed-stable three-student mean was then carried through predicted-input
  lane authoring. Applying learned single-note lanes only where the original
  teacher independently supported the attack preserved deterministic baseline
  masks on newly recovered attacks. All three lane seeds improved validation;
  validation selected seed 2. On untouched Persephone, attack F1 improved from
  0.6073 to 0.6274, joint timing-plus-exact-lane F1 from 0.1210 to 0.1279,
  matched exact lanes from 0.1992 to 0.2039, and lane overlap from 0.2387 to
  0.2473. This is the strongest lawful end-to-end offline result so far, but
  remains below runtime quality and does not resolve frozen timing. A
  deterministic majority vote across lane seeds 0-2 reduced validation joint
  exact F1 from the selected seed's 0.1323 to 0.1271 and Persephone from 0.1279
  to 0.1257, so seed 2 remains selected and the vote is rejected.
- The same frozen distilled/seed-2 configuration was applied unchanged to the
  Hail guitar stem as a post-generation shadow, using the existing fixed
  -1.1-second alignment and never exposing reference notes to training,
  threshold selection, or generation. At 25 ms it reached 0.3732 attack F1,
  barely above the old runtime chart's 0.3702 but below the long-context
  teacher's 0.4046. At 50 ms it reached 0.5200 versus 0.5556 for the runtime
  chart and 0.5337 for the teacher. The seed-stable blend therefore does not
  transfer to Hail and remains rejected from runtime despite its authorized
  validation and Persephone gains. The reusable shadow evaluator records attack,
  timing, lane, motif, and weakest-window metrics at fixed 25/50 ms tolerances.
- A 2.90-ms-hop local timing refiner then preserved the blended attack identities
  and learned bounded corrections from 35-frame, 64-channel short-FFT patches.
  Its seed-0 training loss remained near random 35-class entropy. The smallest
  validation-selected correction reduced macro median error from 10.87 to
  10.70 ms, but also reduced F1 from 0.7687 to 0.7625 and worsened P95 from
  33.46 to 34.04 ms. It failed the predeclared gate, Persephone remained sealed,
  and additional seeds are not justified without a more informative timing
  representation.
- Three lane-ownership refinements were also rejected. An auxiliary phrase-motion
  loss at weights 0.1 and 0.025 either regressed or merely tied validation before
  reducing Persephone joint exact F1. Validation-calibrated lane confidence
  improved joint F1 from 0.13230 to 0.13279 but reversed on Persephone from
  0.1279 to 0.1257. Averaging logits across seeds 0-2 preserved Persephone exact
  lanes but reduced validation joint F1 to 0.1292, while a deterministic
  within-phrase pitch-rank decoder reached only 0.1123. Seed 2 without these
  additions remains selected.
- A signed-residual audit showed that the selected model's Persephone timing
  errors were 98 percent positive with a +30.11 ms median, unlike validation
  medians of -6.23 to +5.68 ms. The first follow-up incorrectly attributed this
  to a -46.44 ms pair offset: the synchronization audit used an uncentered
  2048-sample onset window whose half-window delay is itself about 46 ms.
  Center-compensated remeasurement confirms Persephone at zero global offset,
  early/middle/late offsets of -11.61/0/0 ms, 97.72% onset support within
  25 ms, and an 8.16 ms median nearest-onset distance. Persephone sub-25-ms
  timing is valid; the one-sided residual is genuine model localization error.
  The same correction exposed +23.22 to +69.66 ms offsets in the eight new
  pairs. Their manifest offsets and derived labels are now corrected.
- Retraining the fixed multi-scale seed-0 recipe on those corrected labels
  slightly changed validation macro F1 from 0.7597 to 0.7566, while untouched
  Persephone improved from 0.5616 to 0.6452 attack F1, 0.8013 to 0.9153 recall,
  34.83 to 23.22 ms median timing and 46.44 to 34.83 ms P95. Seeds 1-2 remained
  validation-inferior with test evaluation disabled. Rebuilt short-context
  students improved validation F1/recall when blended but did not improve
  median timing, so the corrected teacher remains selected. Carrying the fixed
  seed-2 lane recipe end-to-end reached 0.6498 Persephone attack F1, 0.9218
  recall, 11.61 ms matched timing and 0.1251 joint exact-lane F1. Corrected
  validation again selected lane seed 2; seeds 0-1 remained sealed.
- Rechecked the closest timing-sensitive rejected candidates after correcting
  the Muldjord labels. A teacher-only compact peak ranker raised validation F1
  only from 0.7566 to 0.7583 while lowering recall and leaving 14.93 ms timing
  unchanged. Frozen Wav2Vec2 layer-12 context lowered F1 to 0.7584 and worsened
  timing to 16.59 ms. Both gates stayed closed and Persephone remained sealed.
- Added an offline, audio-only Basic Pitch lane feature using source pitch,
  confidence, event distance, pitch motion and within-phrase rank. All three
  validation-only recurrent lane seeds regressed below the selected 0.1256
  joint exact-lane F1. A deterministic three-lane phrase-rank decoder improved
  validation to 0.1320, legitimately opening one held-out evaluation, but
  failed to transfer: Persephone exact lanes fell from 0.1926 to 0.1413 and
  joint exact-lane F1 from 0.1251 to 0.0918. The feature family is rejected;
  the corrected pitch-contour seed-2 recipe remains selected.
- Retrained the local-onset architecture and the frozen-selector residual
  offset head on corrected labels. Nonzero local-snap radii worsened macro
  median timing from 14.93 ms to at least 19.90 ms. The offset-only model
  exactly preserved validation F1, aggregate and macro median timing, P95 and
  every selected peak: rounded offsets remained a behavioral no-op. Both
  timing paths remain rejected. The timing-snap evaluator now keeps the held-out
  split sealed unless a nonzero radius preserves F1 and strictly improves
  validation timing.
- A 0.05 auxiliary chart-lane movement loss improved corrected validation
  exact lanes from 0.1754 to 0.1776, overlap from 0.2486 to 0.2522 and joint
  exact-lane F1 from 0.1256 to 0.1272; weights 0.1 and 0.25 were weaker. The
  validation-selected candidate earned one held-out evaluation but left
  Persephone exact lanes and joint F1 unchanged at 0.1926 and 0.1251 while
  slightly reducing overlap from 0.2341 to 0.2332. It is rejected, and the
  no-auxiliary corrected seed-2 lane model remains selected.
- Human Hail review rejected the automatic -1.10 second reference alignment.
  The first audible guitar attack near player 8.61 seconds corresponds to the
  reference's first 4.068-second attack, establishing an initial offset near
  -0.70 seconds. The A/B/C review now supports 50 ms human reference nudges and
  exports the effective offset. At -0.70 seconds, the unchanged generated chart
  reaches only 0.3702 attack F1 at 50 ms and 0.1520 at 25 ms; the worst
  four-second reference window has 31 attacks and one generated candidate.
  Previous -1.10-second Hail metrics are invalid for timing conclusions. This
  confirms that the current Hail output is far from the requested quality and
  that its missing/extra attacks must not be masked by automatic alignment.
- A full-context 128-sample-hop experiment then doubled the parameter-free
  context pools to preserve the selected teacher's receptive field in seconds.
  The unchanged initialization reached 0.7477 validation F1 and 13.20 ms exact
  median timing. Attack-only fine-tuning collapsed to 0.5545 F1, and blending
  high-resolution scores back into the selected coarse blend peaked at 0.7669
  F1 while worsening timing from 10.87 to 12.20 ms. The representation is
  rejected; simply doubling temporal resolution does not preserve the teacher's
  learned attack localization.
- Separating matched polyphony from exact color shows that the selected
  end-to-end model has 0.6906 validation and 0.8014 Persephone polyphony
  accuracy, versus only 0.1794/0.2039 exact lanes. Certain Death is an extreme
  domain gap: 763 of 926 attacks are triple chords, while the original training
  split contains only ten triple-chord attacks, and its matched polyphony
  accuracy is 0.07. A fixed final-fit experiment froze every attack parameter
  and refit only the polyphony head on all 35 non-test songs, exposing 959
  triple-chord attacks. Persephone nevertheless remained unchanged: all 500
  single notes were correct and all 114 dyads were still predicted as singles.
  The refit is rejected. More class weighting or validation reuse cannot supply
  the missing transferable chord evidence.
- A first-order lane-transition decoder learned sub-second single-note color
  transitions from authorized training phrases and calibrated its Viterbi weight
  on validation. It raised validation joint exact F1 from 0.1323 to 0.1338, but
  reduced Persephone joint exact F1 from 0.1279 to 0.1246 and matched exact
  lanes from 0.2039 to 0.1986. It is rejected as another validation-specific
  structural prior; the seed-2 framewise pitch-contour model remains selected.
- A train-only ridge predictor then attempted to infer each song's attack rate
  from the selected score distribution and peak counts. Its least aggressive
  validation calibration raised precision but reduced recall, lowering macro F1
  from 0.7712 to 0.7574. It failed before test evaluation and rules out global
  density correction as a substitute for local chartability.
- Frozen Wav2Vec2 raw-audio embeddings supplied a materially different
  pretrained context representation. A layer-6 attack head introduced too many
  peaks when blended directly; a smoothed log-odds gate reached only 0.7321
  validation F1. The final contextual layer was closer at 0.7621 but still
  regressed recall and timing. Both remained below the sealed-test gate; a
  speech-pretrained representation does not transfer enough guitar
  chartability to justify runtime weight or dependency changes.
- Independent Basic Pitch events were extracted for all eight new
  Sectoid/Muldjord songs and evaluated as positive attack evidence. They raised
  recall on sparse Muldjord songs but flooded the Brad songs, reducing
  validation F1 to 0.7095. A compact ranker over the selected teacher/student
  peaks was more promising: score agreement, prominence and rank raised
  validation F1 to 0.7757 while preserving timing, but missed the baseline
  recall floor by 0.0019 across three deterministic seeds. Adding Basic Pitch
  proximity raised F1/precision further to 0.7779/0.8200 but reduced recall to
  0.7650. Recall-weighted fitting did not repair the ordering. All variants are
  rejected without test evaluation; auxiliary evidence can prune false peaks
  but does not yet recover the charted attacks the selected model misses.
- A fixed-timestamp local-peak reranker used the short-context model only to
  propose attack times and the multi-scale model only to rank their
  chartability. Its best validation candidate retained useful recall at 0.7751,
  but macro F1 fell from the context baseline's 0.7597 to 0.6811 and median
  timing worsened from 14.93 to 27.37 ms. It failed the predeclared validation
  gate, so untouched Persephone was not evaluated and the candidate is rejected.
  The next attack experiment must train directly over local onset candidates
  rather than combine independently calibrated frame scores.
- That direct candidate experiment trained a deterministic audio/context MLP
  over 78,622 fixed short-context peaks, including 11,453 one-to-one matched
  positives. Seed 0 reached only 0.5731 validation macro F1, 0.7445 recall and
  24.88 ms median timing versus the multi-scale baseline's 0.7597, 0.7549 and
  14.93 ms. The failure is large enough that additional seeds cannot satisfy
  the predeclared gate; untouched Persephone was not evaluated. This candidate
  classifier is rejected, and further attack work needs better transferable
  candidate representations or additional independently authored supervision,
  not threshold or seed search.
- A bidirectional candidate-sequence ranker then added candidate-to-candidate
  rhythm gaps and whole-song context over the same fixed local peaks. Seed 0
  reached only 0.5925 validation macro F1 with 26.54 ms median timing, versus
  the multi-scale baseline's 0.7597 and 14.93 ms. This is a decisive rejection:
  independent, reranked, and sequence-modeled fixed local candidates have all
  failed the same gate. Further timing work must improve the primary learned
  representation or supervision rather than add another selector over these
  peaks.
- Explicit attack-localization losses at weights 0.25 and 0.05 required each
  labeled frame to outrank negatives within about 46 ms. The stronger weight
  reduced seed-0 validation F1 from 0.7597 to 0.6631; the lighter weight changed
  the three seed results to 0.7370/0.7004/0.7096 and did not robustly improve the
  selection gate. Neither variant earned another frozen-test evaluation.
- A rhythm-only bidirectional symbolic authoring model regressed authorized
  exact lanes, overlap, chords and movement, confirming that rhythm/polyphony
  alone cannot infer five-lane voice ownership. Adding local log-mel and
  spectral-change features improved validation single-note exactness from
  0.2162 to 0.2856 at the selected seed. A staged decoder applied learned masks
  only to single notes and retained the fixed polyphony-aware chord mask:
  validation exact lanes improved 0.1837 to 0.2152 and overlap 0.2567 to
  0.2882; frozen Persephone improved exact lanes 0.2280 to 0.2329 and overlap
  0.2448 to 0.2497 with chord accuracy unchanged. This remains offline because
  its original evaluation used oracle authorized attack/polyphony conditioning.
- A subsequent timing-aware evaluation replaced oracle inference inputs with
  the multi-scale model's predicted attacks and polyphony and removed chart
  duration from the lane model. Across all three deterministic lane seeds,
  validation joint timing-plus-exact-lane F1 improved from 0.1064 to
  0.1141/0.1132/0.1200. Validation-selected seed 2 improved untouched
  Persephone joint exact F1 from 0.0947 to 0.1187, matched exact lanes from
  0.1560 to 0.1955, and lane overlap from 0.2199 to 0.2312 without changing
  attack selection. This is the first lane gain to survive predicted
  attack/polyphony inference, but remains offline because the upstream
  multi-scale checkpoint still fails its timing gate and checkpoint
  distribution needs the existing license assessment.
- Training lane ownership directly on predicted sequences provided only 3,684
  matched single-note targets and failed validation at seed 0: joint exact F1
  changed 0.1064 to 0.1061 while overlap regressed 0.2265 to 0.2108. Pairing
  the oracle-trained lane model with the better-timed local attack checkpoint
  improved its own lane baseline but reached only 0.0696 Persephone joint exact
  F1, versus 0.1187 for multi-scale attacks. Both alternatives are rejected;
  the selected offline bridge remains multi-scale attacks plus seed-2 staged
  audio lanes.
- Expanding lane inputs from one attack frame to 46/186 ms log-mel means and
  spectral-delta maxima improved validation at all three seeds. The
  validation-selected seed 2 raised joint exact F1 from 0.1064 to 0.1273,
  matched exact lanes from 0.1510 to 0.1766 and overlap from 0.2265 to 0.2474.
  It nevertheless regressed untouched Persephone joint exact F1 from 0.0947 to
  0.0856 and overlap from 0.2199 to 0.1805. Seeds 0-1 did not beat seed 2 on
  the frozen validation selection rule, so their different test behavior
  cannot be used to choose a checkpoint. The richer-context variant is
  rejected as domain overfit.
- Compact audio-derived pitch contour generalized better. Spectral centroid,
  peak bin, rest-reset delta, within-phrase pitch rank and change strength
  improved validation joint exact F1 at all three seeds; validation selected
  seed 2 at 0.1247 versus the prior 0.1200. The same checkpoint improved
  untouched Persephone from 0.1187 to 0.1210, matched exact lanes from 0.1955
  to 0.1992 and overlap from 0.2312 to 0.2387. This replaces the prior selected
  offline lane bridge, but is still far below runtime-quality lane ownership.
  Combining contour with the broader spectral summaries raised selected
  validation F1 to 0.1295 but regressed Persephone to 0.0970; keeping only
  transposition-relative contour reduced validation and returned Persephone to
  the prior 0.1187. Both variants are rejected.
- Restricting training loss to the single-note masks that the staged decoder
  can replace also failed. With the selected pitch-contour inputs, seed 2
  validation joint exact F1 regressed from 0.1247 to 0.1078 and lane overlap
  from 0.2453 to 0.2190. Frozen test evaluation was skipped; chord examples
  remain useful sequence context even when their masks are preserved at
  inference.
- The two external acquisition issues still had no comments on 2026-10-05, and
  no timestamped Hail listening-feedback export was present locally. These are
  genuine external gates, not reasons to weaken automated acceptance criteria.
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
- Verified the MIT-licensed FretNet continuous guitar transcription stack in an
  isolated Python 3.12 environment and processed a real GuitarSet microphone
  example into its six-channel HCQT and per-string targets. Added
  `scripts/fretnet_pilot.py` to run exact deterministic optimizer steps, compare
  fixed held-out-performer losses before and after training, save a local state
  dictionary and record input/model/dependency provenance. A repeated one-step
  smoke run produced identical initial, training and final losses. FretNet
  remains an offline research candidate until a meaningful held-out pilot and
  source-contour shadow comparison beat the existing evidence.
- Completed the first 2,500-step performer-disjoint FretNet pilot with seed 0.
  On 12 fixed GuitarSet performer-05 windows, held-out total loss fell from
  450.9486 to 2.6288; exact per-string active-pitch F1 reached 0.6146 and strict
  frame/pitch onset F1 reached 0.2088, so the result is not an all-silence
  collapse. The local checkpoint, hashes and versions remain under `.research/`.
- Added a lane-independent FretNet shadow exporter. Its onset head transfers
  poorly to separated/full-song audio, but frame-level tablature retains useful
  contour evidence. On four authorized generated runs, unrestricted
  upper-median frame pitch improved weighted source-contour agreement from
  0.3382 to 0.4928 across 207 covered motif occurrences. The same frozen rule
  raised Jonas from 0.0444 to 0.1000.
- Rejected using those pitches directly in chart authoring. On the strongest
  authorized test, changing raw pitches reduced exact-lane agreement from 0.32
  to 0.13 and movement agreement from 0.64 to 0.52. Applying only exact
  three-occurrence repeated four-note contours still reduced exact lanes to
  0.28 and motif-shape agreement from 0.51 to 0.46. `replay
  --fretnet-shadow` therefore records a diagnostic proposal and stage snapshot
  only; verified attack, chord, lane, movement and motif chart metrics remain
  byte-for-byte equivalent to the baseline replay.
- Completed a bounded six-clip EG-IPT articulation assessment without expanding
  the 23.76 GB archive. Basic Pitch missed the pull-off clip, emitted an extra
  high-octave event on hammer-on and split the slide into two pitches; spectral
  flux also exposed multiple internal bend/slide peaks. The publisher's
  GPL-3.0 classifier recognized the coarse muted, bend and legato categories,
  but its dataset has one performer, its split varies pickup configuration and
  its labels provide neither note counts nor onset times. This is useful
  diagnostic evidence for missed legato and false re-attacks, not supervised
  attack truth. Full-archive download, runtime integration and chart-quality
  claims are rejected until an independent-performer or full-song transfer
  check with authoritative onset labels is available.
- Completed a bounded AG-PT-set onset audit over 12 hash-verified recordings,
  three players and 440 publisher labels. Centered spectral flux placed 96.36%
  of labels within 25 ms and 98.64% within 50 ms, confirming alignment. The
  unchanged production guitar path reached precision/recall/F1
  0.9151/0.9795/0.9462 at 25 ms and 0.9214/0.9864/0.9528 at 50 ms. Pick over
  soundhole dominated extras (29 at 25 ms, 26 at 50 ms); palm mute accounted
  for all six remaining 50 ms misses. Bends and hammer-ons are not labeled, so
  the pilot remains isolated and does not justify runtime or corpus changes.
- Built a local Onyx listening review for Sevish's 14-EDO electronic track
  `Sleep Deprived Cooked Alive`, charted by Onyxite, after verifying all 13
  source WAV content hashes. The pinned `song.yml` routing, +2.5 dB gain and
  2.817-second pad were preserved. The routed guitar mix passed beginning,
  middle and end synchronization checks at +46.44, +34.83 and +34.83 ms,
  with 98.56% of 347 Expert attacks within 50 ms of an audio onset. This is an
  electronic-instrument authoring example, not real-guitar transcription data;
  it remains outside the 36-pair corpus pending human review and ShareAlike
  admission analysis.
- Audited `Keytoyze/Mug-Diffusion` as a possible chart-model reference. The
  released checkpoint is fixed to four lanes (`x_channels=16`, four channels
  per lane), its README restricts weights and generated charts to
  non-commercial use, and the maintainer confirmed in issue 9 that roughly
  30,000 osu!/Malody charts were downloaded without individual mapper
  permission. It does not perform source separation and cannot be converted
  into a five-lane RB3 model without retraining. Opened upstream issue 26 to
  ask whether the announced 4-8K work has an explicit weight/output license,
  rights-cleared provenance or lane-generic architecture guidance. No
  Mug-Diffusion code, data or weights are used by generation.
- Run-local review now retains exact separation WAVs under `source/stems` and
  exposes them in the synchronized review player. It also writes honest
  `guitar-left-channel` and `guitar-right-channel` diagnostic views without
  claiming model-level lead/rhythm separation. Mr. Brightside's Demucs guitar
  has left/right correlation 0.045 and side-to-mid RMS 0.956, confirming that
  its layered guitars contain substantial stereo-position evidence.
- Completed a diagnostic-only Mel-Band RoFormer guitar pilot on Mr. Brightside.
  Code commit `fad9aac7e9da7e60ba91e4dc7892a6af0713ed04` and its release are
  MIT-labeled, but the checkpoint was trained on CC BY-NC-SA 4.0 MoisesDB and
  is not eligible to become a bundled/default dependency. Published checkpoint
  SHA-256 `1a81a87b46eed215c051579ac5cff3b139bfdfb9fa3bf4666df059ef87404d14`
  was verified before local inference. RoFormer removed the one collapsed
  contour and reduced long same-lane runs from four to three, but official
  attack F1 regressed 0.75 to 0.71, movement agreement 0.30 to 0.27, mean
  interval error 1.80 to 1.85 and motif coverage 0.38 to 0.27. Wholesale
  separator replacement is rejected. Any follow-up must use RoFormer only as
  diagnostic corroboration, preserve Demucs attacks/timing, remain five-lane,
  and pass held-out gates before runtime consideration.
- Transcribed the left and right Demucs guitar channels independently on all
  nine corpus songs with left/right correlation below 0.4. Wholesale channel
  replacement is rejected: although attack F1 improved on most of the five
  official-chart songs, channel charts commonly added 11-30% more Expert
  attacks and lane/motif effects were inconsistent. In particular, When You
  Were Young and Hail to the King regressed exact-lane, movement and interval
  metrics despite higher attack F1.
- Added a narrowly gated Precision fallback for severe Band guitar contour
  collapses. It evaluates both panned channels only after the full-stem chart
  fails the existing contour diagnostic, and accepts a channel only if it
  removes a collapse, remains within 2% of baseline Expert attacks, and does
  not increase either long or maximum same-lane runs. The decision and all
  rejection reasons are persisted in `guitar-stereo-authoring.json`.
- A fresh Mr. Brightside run selected the left channel: Expert attacks changed
  1,225 to 1,233, collapsed contours 1 to 0, long same-lane runs 4 to 3 and
  maximum run 10 to 9. Official attack F1 improved 0.7521 to 0.7660, exact
  lanes 0.2209 to 0.2275, movement 0.3030 to 0.3129, mean interval error
  1.7993 to 1.7474 and motif coverage 0.3765 to 0.4407. A fresh Bury the Light
  run retained its 2,300-attack baseline: left increased long/max runs and
  right increased the maximum run from 10 to 15. Both integrated runs passed
  structural validation; unresolved Bury uncertainty remains visible.
