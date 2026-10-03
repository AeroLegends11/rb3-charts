# Overnight Progress

Window: 2026-10-02 22:03 through 2026-10-03 07:00 America/Chicago.

## Operating constraints

- Continue between experiments and completed subtasks; save listening questions
  and review candidates for the morning.
- Preserve the current dirty worktree and existing review drafts. Do not push,
  merge, publish, or use reference-chart notes as generation inputs.
- Select model/ranking changes on rights-cleared validation data, then freeze
  them before checking EGSet12, authorized Persephone, evaluation-only Hail, and
  the local commercial support corpus.
- Keep the rejected fixed eight-/twelve-beat phrase bounds removed. Retain the
  validated 5% periodic rerank as offline evidence only; runtime recovery stays
  disabled.

## Prioritized queue

1. Improve lane-independent attack ranking without evaluation-song tuning.
2. Expand or improve the rights-cleared authorized-data path if ranking stalls.
3. Improve phrase-level guitar voice/lane evidence without fixed phrase bounds
   or reference-fed rules.
4. Revisit Less Talk attack/pattern recovery only with new independent evidence.
5. Run focused and full validation for every retained source change; record
   rejected experiments without leaving runtime hooks.

## Checkpoints

### 2026-10-02 22:03 - Baseline reconciled

- Branch: `codex/authorized-chart-pilot`.
- Worktree already contains the retained multi-session source, test, and
  documentation changes plus local untracked `comparisons/`.
- Full suite immediately before this overnight request: 164 passed with six
  existing dependency/deprecation warnings.
- Final focused suite after rejecting phrase bounds: 56 passed.
- No review candidate is currently strong enough for morning listening.
- Next experiment: validation-only ranking features that distinguish a weak
  missing middle attack from generic periodic peaks; evaluation songs remain
  unseen until a variant wins on cleared validation.

### 2026-10-02 22:06 - Periodic-context count rejected

- Cleared validation tested flank support, repeated symmetric contexts, mean
  support, and weak-middle emphasis over fixed rank weights.
- Best result was a 5% repeated-context score: 2028/2366 equal-count matches
  versus 2027 for the retained three-peak prior and 2016 for raw rank. It won
  six, tied five, and lost two of thirteen validation recordings.
- Frozen evaluation gained several Persephone matches under the local scorer
  but lost two Hail matches relative to the retained prior. The one-match
  validation margin is not sufficient to accept that regression.
- No source change retained. Continue with validation-only metrical-alignment
  evidence derived from the existing beat-phase features.

### 2026-10-02 22:08 - Metrical ranking rejected

- A conservative 3% eighth-note alignment prior improved cleared validation
  equal-count matches from 2027 to 2039, with seven wins, six ties, and no
  losses across the thirteen validation recordings.
- Frozen evaluation rejected it. Depending on the fixed count/tolerance, Hail
  lost 3-5 matched attacks and Persephone lost four at reference-count
  selection. Hail's protected 20-24 second count tied, but aggregate regression
  is sufficient to reject the feature.
- No runtime or prediction-artifact change retained.
- Next experiment: validation-only peak prominence/local contrast, using only
  the model score curve rather than beat-grid assumptions.

### 2026-10-02 22:10 - Peak-width ranking retained offline

- Validation selected a conservative blend of 90% retained periodic score and
  10% peak-width rank: 2040/2366 matches versus 2027/2366, with six wins, six
  ties, and one loss across thirteen recordings.
- Frozen Hail evaluation gained three matches at both tested selection counts;
  the protected 20-24 second window tied.
- Frozen Persephone evaluation gained five matches.
- EGSet12 improved from 638 to 657 matches, with all twelve recordings winning.
- Fourteen-song commercial raw-event support improved from 0.64124 to 0.64682:
  twelve wins, one tie, one loss. Less Talk support improved by sixteen events.
- Retained new offline prediction fields for width rank and the combined rank.
  Runtime generation and attack recovery remain disabled.
- Next: audit whether the stronger rank can safely narrow rejected recovery
  proposals; this is evaluation only unless all independent gates pass.

### 2026-10-02 22:11 - Width-ranked recovery rejected

- Replacing periodic rank with the stronger combined rank in the already-fixed
  conservative recovery audit produced 7/9 reference-supported Hail proposals
  and only 2/5 on authorized Persephone.
- Runtime recovery remains disabled; width ranking is evidence, not permission
  to invent or restore chart attacks.
- Next experiment: select an attack-to-attack phrase-gap boundary on the three
  rights-cleared development songs. This tests whether overlong detected
  sustains hide musical rests without returning to rejected fixed phrase spans.

### 2026-10-02 22:12 - Attack-gap phrase boundary rejected

- Tested one-, 1.5-, and two-beat attack-gap boundaries on the three
  rights-cleared development songs.
- The 1.5- and two-beat boundaries were exact no-ops versus rest-based
  segmentation. One beat reduced mean exact lanes 0.1548 to 0.1528, movement
  0.4521 to 0.4409, and motif shape 0.1850 to 0.1768.
- Removed the experimental parameter without evaluating on Hail or Persephone.
- Next: measure peak width at alternate conventional relative heights using
  cleared validation only; retain 50% height unless another setting wins
  robustly before frozen external evaluation.

### 2026-10-02 22:13 - Broader peak-width candidate rejected

- Validation favored 75%-height width at a 15% blend (2053/2366 versus 2040
  for the retained 50%-height/10% form).
- Frozen Persephone gained another five to six matches and EGSet12 gained
  seventeen, but Hail lost seven 50 ms matches at reference-count selection
  and two 80 ms matches. Its protected window tied.
- Retained the already accepted 50%-height/10% width rank rather than selecting
  a setting using evaluation results.
- Next experiment: reduce the structure trainer's fixed-order sensitivity with
  deterministic multi-recording gradient accumulation. Compare a matched
  baseline and dense-real emphasis under the same optimizer-step grouping.

### 2026-10-02 22:15 - Four-recording accumulation rejected

- Matched four-recording accumulation improved untouched test F1 from 0.7982
  to 0.8278 and polyphony to 0.8254. Dense-real duplication tied rather than
  beating the accumulated control, confirming duplication itself adds no value.
- Frozen Hail ranking improved by 44-61 matches and its protected window by two.
  EGSet12 improved from 657 to 701 matches. Persephone was mixed but near level.
- Fourteen-song support failed the aggregate gate: 0.64682 to 0.64456, with
  nine wins and five losses. When You Were Young fell by 0.1262 and Empty Walls
  by 0.0614, so the checkpoint is rejected.
- One final conservative two-recording accumulation control will be selected
  only on cleared validation/test metrics. If it is not competitive on both,
  remove the experimental trainer parameter.

### 2026-10-02 22:16 - Accumulation mechanism removed

- Two-recording accumulation reached test F1 0.8182 but validation F1 fell to
  0.7957 and validation polyphony accuracy collapsed to 0.4958.
- Removed the accumulation parameter and restored the original trainer. Local
  checkpoints remain research artifacts only.
- Next: improve the accepted peak-width evidence implementation and benchmark
  coverage, then move to a different rights-cleared authoring/data experiment.

### 2026-10-02 22:17 - Band sustain threshold retained

- Replayed the three rights-cleared development songs at 200, 275, 350, and
  425 ms Band Guitar short-tail thresholds.
- 200, 275, and 350 ms produced identical sustain precision/recall/F1 and
  matched-duration errors on all three songs. Raising to 425 ms reduced mean
  sustain F1 from 0.0913 to 0.0843.
- Retained 350 ms. Electronic Guitar's separate 90 ms threshold was untouched.
- Next: search for additional explicitly licensed synchronized audio/chart
  pairs so authoring/model decisions are not selected on this four-song seed.

### 2026-10-02 22:19 - Authorized corpus materially expanded

- Found `IntinteDAO/FretsOnFire-Songs-GuitarMaster` at commit
  `afc3f9520d5976bd526e2b48203faa974563337d`.
- Audited 30 folders with synchronized `song.ogg`, `notes.mid`, `song.ini`, and
  per-folder CC BY/CC BY-SA evidence. All 30 Expert charts parse; together they
  contain 11,626 attacks.
- Excluded five Zep Hurme songs to avoid artist/chart-author split leakage into
  held-out Persephone. Excluded `Coffee` because `song.ini` does not name a
  fretter.
- Imported 24 training songs into the local ignored pilot. The combined
  manifest passes full validation: 28 items, 12 artists, four chart authors,
  12,335 Expert attacks, and 56 verified rights-evidence hashes.
- Deterministic Stage B conversion produced 28 examples.
- Basic Pitch extraction for the 24 new licensed full mixes is running in the
  background. No symbolic lane model will be fit until those independent
  contours exist.

### 2026-10-02 22:26 - Expanded symbolic lane fits rejected

- Basic Pitch extraction completed for all 24 new licensed songs.
- Artist-disjoint cross-validation selected a 63-neighbor sequence model over
  normalized pitch, adjacent pitch motion, timing gaps, and beat phase.
- On untouched Persephone, adding the expanded corpus reduced exact lanes from
  0.2915 to 0.2794 and movement from 0.4253 to 0.3801 versus the matched
  original-data model.
- Chart-author-balanced leave-one-author-out selection was worse: 0.1943 exact
  lanes and 0.3665 movement.
- Rejected both fits. The corpus supplies audio/attack diversity, but 24 charts
  from one fretter do not solve lane-style diversity.
- Next: convert the new pairs into lane-independent structure features and test
  full-mix attack supervision without using their lane targets.

### 2026-10-02 22:30 - Authorized full-mix structure candidate rejected

- Converted all 24 new synchronized songs into beat-aware, lane-independent
  attack/polyphony/sustain artifacts and interleaved them deterministically
  through the original 186 training recordings. The original thirteen-song
  validation and test splits were unchanged.
- Corpus-derived positive weighting rose from 26.25 to 32.34. That matched
  candidate reduced validation F1 from 0.8181 to 0.8124 and test F1 from
  0.7982 to 0.7924, while validation/test polyphony fell to 0.5055/0.6218.
- A controlled candidate fixed the attack-positive weight at the selected
  baseline's 26.25. It materially improved validation F1 to 0.8417 and
  untouched test F1 to 0.8511, although polyphony remained lower at
  0.5769/0.7658.
- Frozen evaluation confirmed real attack-transfer gains: EGSet12 combined-rank
  matches improved from 658 to 720, Persephone from 280 to 297, and Hail's
  strict reference-count match from 314 to 330. Hail's protected 20-24 second
  window improved from 10 to 13 under the same 50 ms scorer.
- The broad commercial gate rejected the checkpoint. The same-score proxy fell
  from 0.63536 to 0.62898 with six wins, one tie, and seven losses; Beast and
  the Harlot lost 51 supported events, Less Talk lost 58, and When You Were
  Young lost 41.
- Removed the experimental positive-weight override and retained neither
  checkpoint. The authorized corpus remains useful evidence, but sparse
  full-mix chart supervision needs a less domain-sensitive training strategy.

### 2026-10-02 22:33 - Full-mix dose controls rejected

- Tested deterministic evenly spaced six- and twelve-song subsets with the
  unmodified trainer, original item order interleaved, seed 1337, and unchanged
  validation/test splits.
- The six-song dose produced validation/test F1 of 0.7265/0.6668. The
  twelve-song dose produced 0.7825/0.7165. Both failed before any frozen
  external evaluation.
- A follow-up repository audit found no additional admissible chart-author
  source. Two indexed copies of the API demo song explicitly name an unknown
  fretter. Other discovered repositories bundle commercial recordings without
  per-item audio/chart reuse evidence and were excluded.
- Simple full-mix dosage is exhausted. More lane-authoring training still
  requires independently licensed charts from additional identified authors.

### 2026-10-02 22:35 - Structure ranking gate made reproducible

- Added `evaluate-structure-ranking`, an offline evaluator for prediction and
  lane-independent label artifacts.
- It records the exact rank field, fixed top-K selection count, tolerance, and
  one-to-one match totals. The default protocol selects as many candidates as
  distinct labeled attacks using the accepted combined periodic/width rank.
- Added CLI exposure, malformed-input checks, documentation, and focused test
  coverage. This does not alter generation or model predictions.

### 2026-10-02 22:38 - Review anomaly markers added

- Added non-destructive, clickable review-page timestamps for up to the twenty
  weakest low-confidence Expert attacks per instrument, forbidden
  green-to-orange triads, and merged four-second density spikes.
- Density limits follow the selected Band/Electronic profile and instrument.
  The diagnostics do not add, remove, move, or relane notes.
- Existing amber voice-continuity overlays remain unchanged. Focused preview
  and training coverage passes.

### 2026-10-02 22:40 - Motif lane-mismatch markers added

- Extended the existing repeated source-shape analysis to record how many final
  lane patterns each motif family uses and the non-majority occurrence times.
- Review pages expose up to twenty clickable mismatch timestamps per instrument.
  Real cached replays found mismatches in both Hail and Less Talk without
  changing either chart or its regression metrics.

### 2026-10-02 22:43 - Exact-pitch motif normalization rejected

- Restricted motif mismatch diagnostics to repeated occurrences with the same
  absolute pitch sequence, avoiding false flags for legitimately transposed
  riffs. Hail guitar markers fell from 45 to four.
- Tested a strict majority-lane normalization only on the three rights-cleared
  development songs. It changed 71/12/65 notes on Certain Death, Dirtbag, and
  There's Something Wrong.
- Certain Death's lane metrics tied. Dirtbag exact/movement agreement fell from
  0.1038/0.3558 to 0.1000/0.3497; There's Something Wrong fell from
  0.3453/0.5839 to 0.3387/0.5705.
- Rejected the authoring change without checking evaluation-song references.
  Exact-pitch mismatch markers remain non-destructive listening diagnostics.

### 2026-10-02 22:48 - Full-mix warm-start refit rejected

- Fine-tuned the strongest authorized full-mix checkpoint for one, two, and
  four low-rate epochs on the original 186-item corpus. Four epochs gave the
  best balanced cleared result: validation F1/polyphony 0.8325/0.6340 and
  untouched test F1/polyphony 0.8455/0.7411.
- Frozen ranking retained narrow gains: EGSet12 658 to 712, Persephone 280 to
  303, Hail 314 to 329, and Hail's protected window 10 to 12 under the same
  strict 50 ms scorer.
- The broad commercial proxy regressed further from 0.63536 to 0.62576, with
  three wins and eleven losses. Less Talk lost 60 supported attacks.
- Rejected all refits and removed the experimental warm-start/learning-rate
  trainer parameters. The selected checkpoint and runtime remain unchanged.

## Morning summary / early safe stop

Checkpoint time: 2026-10-02 22:45 America/Chicago.

- Retained the validation-selected offline peak-width rank and reproducible
  `evaluate-structure-ranking` command. Neither changes generated attacks.
- Expanded the fully validated local authorized corpus to 28 songs, then
  rejected symbolic lane fits, full-dose/lower-dose structure training, and
  warm-start refits because independent or broad-song gates regressed.
- Added clickable, capped review diagnostics for low-confidence attacks, density
  spikes, forbidden green-to-orange triads, and exact-pitch motif lane
  mismatches. A proposed automatic motif rewrite regressed cleared development
  charts and was removed.
- Diagnostic-only cached reviews are saved under
  `comparisons/review-diagnostics-v5` (Hail) and
  `comparisons/review-diagnostics-v3` (Less Talk). Their authored charts are
  unchanged, so they are not stronger listening candidates and no new listening
  verdict is requested.
- Remaining blockers are external or evidentiary: more independently licensed
  charts from identified additional chart authors; independent source evidence
  for safe attack recovery; and lawful section/drum labels before changing those
  subsystems.
- Final retained validation: 168 tests passed; CLI help passed; `git diff
  --check` passed apart from existing line-ending warnings.
- No commit, push, merge, publication, runtime-model replacement, or review-draft
  overwrite was performed.

## Extended autonomous shift through 12:00 PM

### 2026-10-02 23:49 - Multi-instrument regression coverage retained

- Expanded stage snapshots from guitar-only to every generated/replayed
  instrument. Pitched parts record mapping, playability, optional Band reference
  conformance, and reductions; drums retain cleaned/replay-source Expert and
  reduction snapshots.
- Cached replay now writes per-instrument metrics and
  `regression-metrics-by-instrument.json`, while preserving the existing guitar
  report as a compatibility alias.
- A real Hail cached replay recorded three guitar transitions at 0.975 mean
  retention, two bass transitions at 0.984, and an auditable drum source plus
  reductions. No chart behavior changed.

### 2026-10-02 23:52 - Difficulty-reduction integrity metrics retained

- Added Expert-to-Hard/Medium/Easy attack retention, retained/orphan source IDs,
  lane-change counts, and timing-error metrics for every instrument.
- Focused replay coverage passes and verifies no orphan IDs or timing changes.

### 2026-10-02 23:56 - Neutral section and drum-groove diagnostics retained

- Added final-chart `structure-diagnostics.json` with neutral eight-measure
  activity blocks, ranked unlabeled boundaries, and repeated measure-level drum
  groove families.
- Review pages expose the eight strongest activity boundaries as clickable
  listening markers. No verse/chorus label is inferred and no note is changed.
- Real Hail replay found 19 repeated drum-groove families covering 62.99% of
  Expert drum attacks. Less Talk correctly made no drum-groove claim.

### 2026-10-03 00:01 - MIDI structural validation hardened

- Added checks for type-1/960-PPQ output, unique required tracks, EVENTS-local
  lifecycle markers, tempo/beat content, instrument-specific pitches, and exact
  serialized gem counts per difficulty.
- Synthetic corruption tests prove invalid track pitches and missing serialized
  Expert gems fail validation. Existing integration output remains valid.

### 2026-10-03 00:08 - Multi-instrument corpus benchmarks retained

- Expanded `benchmark-replays` from guitar-only comparison to explicit guitar,
  bass, drums, or keys selection while preserving guitar as the default.
- Added macro Expert-to-Hard/Medium/Easy attack retention so corpus comparisons
  cannot hide reduction regressions behind unchanged Expert output.
- Separated simultaneous drum attacks from pitched-instrument chords. A real
  Hail comparison reports a 0.5895 simultaneous-attack rate and zero chord rate,
  instead of the former misleading 0.5895 drum chord rate.
- Independent Hail replay roots produced exact zero deltas for bass and drums.
  Drum retention was 0.9672 Hard, 0.5546 Medium, and 0.2991 Easy in both roots.
- Legacy snapshots without reduction payloads report unavailable retention and
  an explicit zero-song coverage count rather than a misleading 0% result.
- Focused benchmark validation: 15 tests passed; command help exposed all four
  supported instruments.

### 2026-10-03 00:15 - MIDI export/readback timing audit retained

- Structural validation now pairs serialized note-on/note-off events and
  verifies each playable gem's absolute tick, release tick, and pitch against
  the authored chart for every instrument and difficulty.
- A corruption test shifts an Expert guitar attack without changing its gem
  count and proves the validator rejects it.
- A fresh production-sized cached Hail replay passed the new audit across all
  instruments and difficulties. Focused charting/integration validation passed
  64 tests.

### 2026-10-03 00:21 - Reviewed package-input integrity retained

- New generation and replay drafts record SHA-256 hashes for the exact
  `notes.mid`, `song.wav`, and `song.ini` inputs that will be compiled.
- `package` rejects an input changed after validation, preventing a stale green
  validation report from authorizing modified content. Existing drafts without
  the additive manifest remain packageable with an explicit warning.
- A real cached Hail replay wrote and verified the manifest. Synthetic packaging
  coverage passes, including deliberate post-validation MIDI mutation.

### 2026-10-03 00:26 - Structure summaries exposed for review

- Added neutral activity-block, ranked-boundary, repeated drum-groove family,
  and groove-supported attack summaries to assessments and `review_notes.md`.
- These fields do not change chart notes or automatic quality status.
- A fresh Hail replay exposed 19 activity blocks, 19 repeated groove families,
  and 63.0% supported Expert drum attacks directly in its review notes.
- Focused assessment/integration validation passed 21 tests.

### 2026-10-03 00:31 - Corpus structure benchmark retained

- Extended replay-corpus reports with additive song-mean comparisons for
  neutral activity blocks, boundary count/strength, repeated drum-groove
  families, and groove-supported attack coverage when both roots provide
  structure diagnostics.
- Independent Hail roots matched exactly at 19 blocks, 12 ranked boundaries,
  mean/max boundary change 0.1544/0.3088, 19 repeated groove families, and
  0.6299 supported drum-attack ratio. Every structure delta was zero.
- Focused replay benchmark validation passed 15 tests.

### 2026-10-03 00:35 - Human-prior reduction triage retained

- Added assessment-only Expert-to-Hard retention checks against the existing
  aggregate human-chart p10/p90 range for each instrument.
- A real Hail replay now flags guitar retention of 53% below the 60% reference
  p10 and drum retention of 97% above the 82% reference p90. Bass remained
  inside its aggregate range.
- No reduction behavior changed. These outliers need broader replay evidence
  before any spacing or density thresholds are adjusted.
- Focused assessment validation passed 8 tests.

### 2026-10-03 00:44 - Guitar Hard spacing calibrated and retained

- Swept Hard minimum attack spacing across six distinct cached songs. The
  existing 0.46-beat spacing left Hail, Certain Death, and There's Something
  Wrong below the aggregate human guitar p10 retention.
- Rejected removal of the separate fast-eighth reducer because it changed only
  the 160 BPM song and did not address the broad result.
- A 0.40-beat control improved every song but left Hail below p10 at 0.581.
- Retained 0.38 beats. Per-song Expert-to-Hard retention changed:
  - Hail: 0.5312 -> 0.6056; peak Hard density 3.75 -> 4.25/s.
  - Less Talk: 0.6242 -> 0.6776; 3.75 -> 4.00/s.
  - Certain Death: 0.5875 -> 0.7125; 3.75 -> 4.50/s.
  - Dirtbag: 0.7244 -> 0.8099; 4.25 -> 4.50/s.
  - There's Something Wrong: 0.5809 -> 0.6940; 3.75 -> 4.50/s.
  - Persephone: 0.6160 -> 0.6576; 3.00 -> 3.25/s.
- All six now fall inside the aggregate guitar p10-p90 range of 0.5992-0.8543.
  Assessment statuses were unchanged. Expert notes, source ancestry, timing,
  MIDI validation, and package-input integrity remained intact.
- Hail's Hard chart is saved as a morning listening candidate under the local
  ignored `.research/hard-spacing-38` replay root. Work continues elsewhere.
- Immediately after the checkpoint, the shared reduction loop was audited and
  the 0.38 spacing was explicitly scoped to guitar. Bass, drums, and keys retain
  their prior 0.46 spacing; a regression test prevents cross-instrument drift.
  A fresh Hail replay confirmed guitar retention 0.6056 and unchanged drum
  retention 0.9672.

### 2026-10-03 01:18 - Drum reduction candidate measured and rejected

- Found that cached replay copied prior drum difficulties verbatim, preventing
  drum reduction experiments from exercising current code. Replay now preserves
  the already-aligned Expert drum chart but rederives Hard/Medium/Easy; it still
  does not rerun activation alignment.
- A global drum spacing increase was rejected because retention fell sharply to
  about 0.56 on both Hail and Less Talk, below the aggregate p10.
- An aggressive nearby-accompaniment filter was rejected because it retained
  only 3.3%/17.2% of accompaniment-only attacks and would hollow out grooves.
- A bounded measure-grid variant moved Hail/Less Talk Hard retention from
  0.9672/0.7728 to 0.7980/0.7037, retained 46.0%/52.3% of accompaniment-only
  attacks, and held peak Hard density to 3.75/s.
- The bounded variant is also rejected from runtime: two songs and aggregate
  retention do not establish musical groove quality, and lawful drum-authoring
  labels/listening evidence remain unavailable. Local candidates are saved
  under ignored `.research/drum-hard-grid`; the runtime rule was removed.
- Focused replay/chart validation passed 69 tests before rejection; the retained
  replay-harness change has dedicated stale-reduction coverage.

### 2026-10-03 01:24 - Review UX and tempo readback hardened

- Batch review indexes now expose neutral activity-block, repeated drum-groove,
  and supported-attack coverage beside each song. Review-page guidance now
  names motif-lane and activity-boundary markers.
- MIDI validation now compares the full serialized tempo-event sequence and
  time signature with the authored `TempoMap`; matching gem ticks alone can no
  longer hide incorrect playback timing.
- A variable-tempo corruption test proves a changed tempo value fails. A fresh
  production-sized Hail replay passed the complete tempo/readback gate.
- Focused assessment/integration tests passed 22; focused charting/integration
  tests passed 67.
- Extended the same audit to every BEAT interval: onset, release, downbeat
  pitch, meter phase, and total grid extent. A deliberate five-tick shift fails,
  while a fresh Hail replay passes. Focused coverage increased to 68 tests.
- Added exact EVENTS `[music_start]`/`[music_end]`/`[end]`, instrument
  `[play]`/`[idle]`, and drum mix-marker validation. Corrupted timing/text fails;
  a fresh Hail replay passes. Focused coverage increased to 69 tests.
- Corrected corpus reduction aggregation to use only matched songs with complete
  snapshots on both sides. Asymmetric legacy coverage now reports unavailable
  retention/count deltas rather than a misleading comparison. Focused replay
  benchmark coverage passes 16 tests.

### 2026-10-03 01:32 - Difficulty review table retained

- Added an always-visible per-instrument Expert/Hard/Medium/Easy attack table to
  `review_notes.md`, including generated Hard retention and the aggregate human
  p10-p90 range.
- A fresh Hail replay exposes guitar 1600/969/466/236 (60.6%), bass
  493/400/280/166 (81.1%), and drums 916/885/508/274 (96.6%) directly to the
  morning reviewer.
- Focused assessment/integration coverage passes 22 tests.
- Added the selected instrument's four difficulty counts directly to
  `review.html`; switching instruments updates the summary without leaving the
  listening page. Focused preview/assessment coverage remains 22 tests.
- Added mean/max four-second density for every difficulty to corpus replay
  benchmarks. The retained Hail Hard change now reproduces directly in the
  standard report: retention 0.5312 -> 0.6056 and max density 3.75 -> 4.25/s.
  Focused replay benchmark coverage passes 16 tests.

### 2026-10-03 01:38 - Repeated motif review links retained

- Added up to 16 capped repeated-motif candidate markers per instrument, with
  occurrence count and instance number, so reviewers can jump among candidate
  phrase returns for direct A/B listening.
- Markers are descriptive only and do not normalize or rewrite lanes.
- Fresh Hail and Less Talk reviews expose the capped markers; focused integration
  coverage passes 14 tests.
- Added corpus-level pattern vocabulary comparisons for recurring family and
  occurrence counts, supported-attack coverage, contour agreement, flattening,
  adjacent motion, and wide jumps.
- Independent Hail roots matched exactly at 158 families, 442 occurrences,
  0.5619 supported-attack ratio, 0.993 contour agreement, and zero deltas across
  every pattern metric. Focused replay benchmark coverage passes 16 tests.

### 2026-10-03 01:46 - Future difficulty priors made reproducible

- Confirmed the committed compact style profile intentionally omits per-song
  records, so Medium/Easy percentile ranges cannot be reconstructed safely from
  aggregate totals.
- Updated the reference analyzer to schema 2 so a legitimate future corpus run
  emits Expert-to-Hard/Medium/Easy p10/median/p90 for every instrument, while
  preserving the existing Hard-only compatibility key.
- Did not overwrite the committed schema-1 profile or invent runtime priors.
  Focused reference/integration validation passed 15 tests.
- Versioned the expanded replay benchmark report as schema 2 now that it carries
  instrument, paired-reduction density, structure, and pattern-vocabulary
  sections. Original guitar macro/song fields remain additive-compatible.
- Prevented instrument runs from overwriting the default guitar report: bass,
  drums, and keys now default to instrument-specific output filenames. A real
  bass comparison wrote `output/bass-replay-benchmark.json`.
- Hardened reference data exclusion: default analyzer output is now compact and
  omits `source_folder` plus per-song records. An explicit distinct
  `--full-output` writes the local detailed report and is documented as
  unstageable. Focused reference tests and CLI help pass.

### 2026-10-03 01:58 - Existing-run revalidation retained

- Added `validate-run` to apply the current MIDI structure, tempo, beat,
  lifecycle, gem timing/count, and package-input integrity checks to an existing
  draft without rerunning inference or authoring.
- Recheck results are written separately as `validation_recheck_report.json`
  and `.txt`; original validation and review drafts remain untouched.
- A real Hail motif-review draft passed with a verified manifest and zero errors.
  A synthetic post-validation `song.ini` change fails as expected.
- Focused integration coverage passes 15 tests; CLI help passes.
- Complete run folders now perform the current recheck automatically at the
  packaging boundary. A valid synthetic reviewed run compiles through the fake
  Onyx harness; changing `song.ini` afterward stops packaging before compile.
  Legacy folders without charts/tempo retain their compatibility fallback.
- Package results now persist `validation_recheck` provenance and the recheck
  report path. Legacy manifest-only and unavailable states are explicit rather
  than inferable only from logs.
- The current Less Talk motif-review draft also passes `validate-run`.
- Required EVENTS lifecycle, instrument play/idle, and drum mix markers now
  require exact uniqueness as well as correct ticks. Duplicate corruption is
  rejected; focused charting/integration coverage passes 71 tests.

### 2026-10-03 02:05 - Fresh morning reviews saved

- Generated new non-overwriting current-code drafts:
  - `comparisons/morning-review/Avenged_sevenfold_Hail_To_The_King_20261003_003813_937125`
  - `comparisons/morning-review/Freezepop_Less_Talk_More_Rokk_20261003_003814_021279`
- Both pass `validate-run` with verified package-input manifests and the full
  tempo/beat/lifecycle/gem readback audit.
- Hail's listening question is limited to Hard guitar: does the retained
  0.38-beat spacing feel more appropriately dense than the prior 0.46-beat
  reduction without becoming tiring? Expert is not a new musical candidate.
- Less Talk remains an informational current review. The scaling-riff Expert
  attack/pattern problem remains upstream and no rejected recovery hook was
  re-enabled.
- A retained-change review found malformed package manifests could escape as
  JSON/type exceptions. Manifest parsing now normalizes truncated JSON and
  invalid entry types into failed validation reports, so `validate-run` and
  packaging fail cleanly instead of producing a traceback. Focused integration
  coverage passes 15 tests.
- `validate-run` now converts incomplete or malformed saved chart/tempo metadata
  into a concise exit-1 message. A missing-run probe returned the expected error
  without a traceback.
- `package` now provides the same concise nonzero CLI failure for incomplete,
  invalid, or failed-revalidation folders; a missing-run probe also produced no
  traceback.
- Replay benchmarking now accepts `--instrument all`, writes one combined
  artifact for every instrument available in both roots, and records unavailable
  instruments explicitly. A real comparison exercised guitar, bass, and drums
  in one invocation; the focused regression suite passes 19 tests.
- Replay corpus discovery now rejects duplicate artist/title keys and names
  both conflicting paths, preventing accidental arbitrary baseline selection.
  Missing artist/title identity is also rejected. Focused benchmark coverage
  passes 18 tests.
