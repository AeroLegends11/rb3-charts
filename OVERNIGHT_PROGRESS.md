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
- Added `validate-runs` for non-destructive corpus rechecks. It discovers draft
  runs by saved charts, continues after malformed runs, preserves each run's
  errors/warnings, returns nonzero when any fail, and writes one aggregate JSON
  report. One invocation revalidated both morning-review drafts successfully;
  the focused integration suite passes 15 tests.
- Replay corpus discovery now rejects duplicate artist/title keys and names
  both conflicting paths, preventing accidental arbitrary baseline selection.
  Missing artist/title identity is also rejected. Focused benchmark coverage
  passes 18 tests.

### 2026-10-03 06:17 - General reference cross-check restored

- Audited the earlier standalone chart comparison tool against the current
  provenance-gated authorized evaluator. The current engine already had richer
  difficulty, motif, phrase-boundary, density, and four-second metrics, but had
  lost arbitrary local MIDI comparison, automatic alignment, unmatched-event
  detail, and authoring-stage attribution.
- Added evaluation-only `compare-midi` for completed output folders or MIDI
  files. It reads reference notes only after generation, hashes both MIDIs,
  records Git/alignment provenance, estimates or accepts a global offset, and
  writes JSON plus Markdown.
- Reports now preserve exact unmatched attacks, weakest four-second windows,
  and comparison metrics after each saved guitar authoring stage. A real
  Certain Death cross-check recovered a -10 ms offset and measured final Expert
  attack F1 0.7987. Mapping matched 853 attacks; playability matched 849 while
  improving overall F1 from 0.75 to 0.80 by removing unsupported extras.
- Focused regression coverage passes 21 tests. The command does not alter
  generation and does not make reference charts training inputs.
- Replayed the seven-song official-reference evaluation corpus through current
  code. Six passed immediately; Jonas exposed a one-tick serialized guitar
  mismatch on a floating-point half-tick boundary in its variable tempo map.
  MIDI writing rounded after adding lead beats while validation rounded before
  adding them. Both now use the shared `authored_tick` conversion. The real
  Jonas replay passes, and 56 focused charting tests pass.
- Current-code replays across seven official-reference evaluation songs improve
  or preserve final attack F1 versus mapped notes on five songs. Hail and Cliffs
  remain playability-loss outliers: the pass removes 115/91 attacks and 62/41
  reference-matched attacks respectively. A rights-cleared selection experiment
  allowed one additional >=0.80-confidence sixteenth-grid fifth attack per beat;
  it was an exact no-op on Certain Death, Dirtbag, and There's Something Wrong,
  so the candidate was removed without consulting official results.
- A second rights-cleared gate reduced the minimum gap from one eighth to one
  tenth beat only for adjacent >=0.75-confidence, sixteenth-grid-aligned Band
  Guitar attacks. It was also an exact three-song no-op and was removed. These
  results indicate the official-song losses are not recoverable through another
  generic density exception without stronger source evidence.
- Cached replay now reports incomplete inputs and validation failures as concise
  exit-1 CLI errors rather than raw tracebacks, matching `validate-run` and
  packaging behavior.

### 2026-10-03 06:35 - Playability removal attribution retained

- Traced every current-code Band Guitar attack removed by
  `enforce_playability` into four non-overlapping causes: per-beat cap, later
  attack rejected by spacing, earlier attack replaced by a higher-priority
  neighbor, and cleanup merge/drop. Reports preserve full source IDs/times plus
  confidence, priority, sixteenth-grid alignment, sustain and motif evidence.
  Chart behavior is unchanged.
- On the rights-cleared selection, removals were:
  - Certain Death: 33 cap, 52 later-spacing, 68 earlier-replaced (153 total).
  - Dirtbag: 0 cap, 1 later-spacing, 2 earlier-replaced (3 total).
  - There's Something Wrong: 5 cap, 34 later-spacing, 16 earlier-replaced
    (55 total).
  - Frozen Persephone: 6 cap, 10 later-spacing, 13 earlier-replaced (29 total).
  - Cleanup removed zero attacks on all four.
- Across those four lawful gates, 209/240 removals were the same weak/plain
  observable class seen in the official failures: confidence below 0.75, no
  sustain evidence and no recurring-motif support. Only one lawful removal had
  confidence at least 0.75; 30 had motif support and 27 had at least 0.80
  sixteenth-grid alignment. This confirms that relaxing the broad weak/plain
  class would also affect cleared songs substantially.
- Evaluation-only attribution of mapped attacks that matched the frozen
  official references found 192 removed candidates: 118 cap, 27 later-spacing,
  47 earlier-replaced, zero cleanup. Of these, 177 were weak/plain, none had
  confidence at least 0.75, none had sustain evidence, only 15 had motif
  support, and 28 had at least 0.80 grid alignment. Hail accounted for
  56/7/17 and Cliffs for 22/12/15 respectively. These are removed mapped
  matches, not net recall losses: a neighboring retained attack can still match
  the same reference onset.
- The seven frozen stage gates remained behavior-identical:
  - Beast and the Harlot: 1234->1195 attacks, 771->756 matches,
    F1 0.5392->0.5360.
  - Hail to the King: 1496->1381, 808->746, F1 0.6107->0.5895.
  - Cliffs of Dover: 1141->1050, 733->692, F1 0.6157->0.6044.
  - Empty Walls: 870->867, 619->619, F1 0.7240->0.7252.
  - Mr. Brightside: 1025->1004, 777->770, F1 0.7446->0.7454.
  - When You Were Young: 907->848, 400->389, F1 0.5461->0.5533.
  - My Name Is Jonas: 884->869, 604->601, F1 0.6975->0.7001.
- No runtime recovery rule was retained. The official losses do not expose a
  source-derived confidence/grid/sustain/motif class absent from the cleared
  removals, and the two already-rejected high-confidence density exceptions
  remain exact no-ops on the cleared selection. The next safe step is to seek
  an independent attack/voice-continuity feature rather than relax density.

### 2026-10-03 06:55 - Independent removal evidence audited

- Added post-generation `audit-playability-evidence`. It joins every attributed
  removal to cached model-onset, attack-strength, relative-level and evidence
  scores; optional structure-model probability/periodic-width ranks; source-mix
  and guitar-stem onset height/prominence/width ranks; and overlapping
  voice-continuity windows. Optional reference labels and feature-separability
  summaries are evaluation-only and never enter generation.
- Threshold selection used only Certain Death, Dirtbag, and There's Something
  Wrong. Their current charts left 10 otherwise-unmatched authorized attacks
  among 211 removals. Top-quartile feature precision was weak:
  confidence 3/53, model onset 4/53, attack strength 2/53, evidence score 4/53,
  structure rank 1/53, structure peak width 2/53, onset prominence 1/53, and
  onset width 2/53. The best voice-continuity subset recovered 3/17. No simple
  one-to-three-feature conjunction supported at least two songs above 8.6%
  precision.
- The cleared source audio and saved guitar stem are byte-identical for all
  three development songs, so no independent stem-agreement threshold can be
  selected from them. The audit records this explicitly instead of treating
  duplicate audio as corroboration.
- Frozen Persephone contained four otherwise-unmatched authorized attacks among
  29 removals. Development thresholds recovered zero with confidence, model
  onset, attack strength, evidence score, or onset prominence; structure width
  recovered 1/2 selected, while the broad voice candidate recovered 4/22 and
  stem/mix agreement 2/15. This did not justify a runtime rule.
- Only after rejection, the seven official-reference audits showed 139
  otherwise-unmatched labeled attacks among 343 removals. Development
  thresholds appeared substantially more selective there (model onset 46/88,
  structure rank 43/73, onset prominence 37/62, attack strength 6/8), confirming
  a domain-separation problem rather than providing lawful threshold-selection
  evidence. Broad official-only stem agreement was 101/240 and voice continuity
  61/128.
- No attack was restored and no generation threshold changed. The retained
  tooling makes this lack of cross-domain separability reproducible and prevents
  future official-only correlations from being mistaken for selectable runtime
  evidence.

### 2026-10-03 07:10 - Rights-clean source agreement calibrated

- Audited the implemented/local dataset paths. GuitarSet has 360 licensed mic
  recordings and JAMS labels locally, but not its hex-pickup channel, so it
  cannot currently measure independent capture agreement. Slakh provides
  rendered source stems and authoritative MIDI, but its guitar mix is a
  deterministic sum and separate Guitar-class stems are different arranged
  parts, not independent captures of one attack. EGFxSet pickup/effect files
  are separate isolated-note performances without aligned paired onsets.
  Authorized chart pairs contain full mixes plus separator-derived stems, not
  independent recordings.
- Guitar-TECHS is suitable: the local CC BY 4.0 release contains synchronized
  direct-input and mic/amp recordings plus MIDI for the same performance.
  Added deterministic `evaluate-source-agreement`, which verifies the existing
  manifest and primary hashes, supports explicit alternate paths/hashes, rejects
  duplicate or duration-mismatched pairs, preserves label group/performer split
  keys, and selects one prominence-ranked onset per labeled attack in each
  channel. Local paired manifests now pin all 68 alternate recording hashes.
- Across 56 P1/P2 chord recordings and 4,885 labeled attacks at 50 ms, mic/amp
  precision was 0.5431, direct input 0.5294, and 4,445 shared onsets reached
  0.5591. Mic-only onsets measured 0.3818 and DI-only 0.2614. P1 shared
  precision was 0.6514 versus 0.6130/0.6001; P2 was 0.4887 versus
  0.4871/0.4727.
- Across 12 performer-P3 music excerpts and 1,550 labeled attacks, mic/amp was
  0.8161, direct input 0.8077, and 1,511 shared onsets reached 0.8226.
  Mic-only precision was 0.4872 and DI-only 0.2308.
- The direction was stable at 30/50/80 ms. Chord shared precision was
  0.4518/0.5591/0.6919 versus best-source 0.4311/0.5431/0.6680. Music shared
  precision was 0.8020/0.8226/0.8405 versus 0.7923/0.8161/0.8297.
- This establishes lawful evidence that agreement between independent captures
  modestly raises attack precision and that channel-only peaks are much weaker.
  It does not supply paired captures for the cleared chart songs, so no runtime
  recovery rule or threshold changed.

### 2026-10-03 07:35 - GuitarSet pickup agreement verified

- Authoritative Zenodo record `3371780` identifies GuitarSet v1.1.0 as open
  access under CC BY 4.0. Existing `annotation.zip` and `audio_mono-mic.zip`
  matched published MD5 values `b39b78e63d3446f2e54ddb7a54df9b10` and
  `275966d6610ac34999b58426beb119c3`. The full original/debleeded hex archives
  are 3,210,735,945 and 3,607,349,578 bytes; the sufficient official mono
  pickup mix is 683,145,360 bytes.
- Downloaded that archive only into ignored `.research/guitarset/`. Its local
  MD5 exactly matched published value `aecce79f425a44e2055e46f680e10f6a`.
  Added `pin-source-agreement` to deterministically map aligned filenames, pin
  all 180 alternate SHA-256 hashes, preserve the existing performer split, and
  verify/record archive MD5, byte size and release URL. The resulting ignored
  manifest SHA-256 is
  `5b2bbdd30284f12bc4aa3f42c37a6a3a8f880a61d7f7e6c1386ec7b2bbc21c2f`.
- At 30 ms, shared precision versus best individual source was:
  train 0.8120 vs 0.7845, validation 0.8435 vs 0.7991, and untouched test
  0.9062 vs 0.8770. Shared lifts were +0.0275/+0.0444/+0.0292.
- At 50 ms, train (15,124 labels) was mic 0.8141, pickup 0.8244 and shared
  0.8527; validation (1,140) was 0.8360/0.8518/0.8996; untouched test (1,203)
  was 0.8845/0.8919/0.9260. Shared lifts were +0.0283/+0.0478/+0.0341.
  Channel-only precision was much weaker: train mic/pickup 0.4464/0.5938,
  validation 0.3269/0.4712, and test 0.5474/0.7053.
- At 80 ms, shared precision versus best source was train 0.8596 vs 0.8319,
  validation 0.9040 vs 0.8588, and test 0.9306 vs 0.8969. Shared lifts were
  +0.0277/+0.0452/+0.0337.
- Agreement therefore gives stable lawful attack evidence across untouched
  performers and fixed tolerances. It still does not create independent paired
  captures for Certain Death, Dirtbag, There's Something Wrong or Persephone.
  No chart-generation or playability-recovery rule changed.

### 2026-10-03 06:59 - Paired-source evidence exposed in corpus reports

- Extended `validate-chart-pairs` with optional
  `--source-agreement-report`. It consumes the existing
  `evaluate-source-agreement` artifact rather than re-running onset selection or
  matching, and emits additive `rb3charts-paired-source-evidence` schema v1.
- The summary distinguishes `available`, `unavailable` and `invalid` lineage.
  Available evidence includes evaluation-report SHA-256, paired-manifest
  SHA-256, release archive published MD5/size, tolerance, channels, split,
  shared precision/recall/F1, best-source baseline and deltas.
- The current authorized 28-song manifest plus GuitarSet 50 ms report validates
  as `external_rights_cleared_calibration`. Target coverage is explicitly
  unavailable: 0/28 chart pairs have an independently captured alternate
  channel. The tool does not infer agreement from separator stems.
- GuitarSet shared-versus-best-source F1 deltas are train -0.0045, validation
  +0.0048 and untouched test -0.0040 despite precision lifts of
  +0.0283/+0.0478/+0.0341. This makes the precision/coverage tradeoff visible
  rather than presenting agreement as an unconditional recall gain.
- No runtime generation, playability threshold or chart logic changed.
- Validation: 16 dataset/reporting tests passed; paired chart/training selection
  passed 12 tests; full suite passed 193 tests. CLI help and the real 28-song
  corpus plus GuitarSet report were also exercised successfully.

### 2026-10-03 07:02 - Paired-source evidence added to review Markdown

- `compare-human-chart` now accepts `--source-agreement-report`, carries the
  already-validated additive summary into its JSON, and writes deterministic
  `human-chart-comparison.md` beside the existing JSON.
- Available external evidence is prominently labeled rights-cleared calibration
  rather than generated-song evidence. The report prints explicit paired corpus
  coverage, tolerance, report/manifest/release hashes and compact split metrics.
  It does not print local manifest/report paths.
- Unavailable lineage states that absence does not imply a paired capture.
  Invalid lineage omits every supplied metric and renders only the failure
  reason, preventing stale or forged values from appearing authoritative.
- Rendering tests cover external calibration, unavailable lineage, invalid
  lineage, path omission and the authorized comparison wrapper. Runtime
  generation and reference loading remain unchanged.
- Validation: four focused comparison/lineage tests passed and the full suite
  passed 196 tests.

### 2026-10-03 07:08 - Reference Markdown disclosure minimized

- As the next independent evaluation cleanup, `compare-midi` Markdown now shows
  only MIDI basenames and content hashes, not local generated/reference paths.
  It explicitly states that individual reference-note rows are not embedded.
- JSON remains the local auditable artifact with existing provenance and
  unmatched-event detail; generation and comparison metrics are unchanged.
- Added a rendering regression test covering path omission, hash disclosure and
  summary-only labeling.
- Validation: four focused comparison Markdown tests passed; full suite passed
  197 tests.

### 2026-10-03 07:18 - Sanitized comparison corpus indexes added

- Added `index-comparisons`, which discovers generated runs from
  `metadata/song.json` and writes deterministic JSON plus Markdown for one
  comparison schema. Current comparison writers now identify authorized and
  reference reports with explicit additive schema names.
- Safe entries contain song display identity, generated run basename,
  comparison and validation status, aggregate attack/chord/sustain/lane/motif
  and weakest-window metrics, paired-source status, and relative local report
  links. Missing comparisons remain visible rather than disappearing.
- Indexes omit source/generated absolute paths, unmatched/reference note rows,
  detailed authorized records and per-event data. Markdown escapes display
  delimiters and URL-encodes local links.
- Duplicate song IDs, mixed authorized/reference corpora, malformed metadata,
  malformed validation reports and unknown comparison schemas abort before
  outputs are written.
- Focused tests cover sanitization, missing comparisons, duplicate identities,
  mixed schemas and malformed schemas. Runtime generation remains unchanged.
- Validation: seven focused comparison/reporting tests passed; full suite
  passed 200 tests; CLI help was exercised.

### 2026-10-03 07:24 - Stale comparison lineage rejected

- Began the next independent quality task by binding corpus index entries to
  the current generated MIDI. Authorized comparison artifacts now record that
  MIDI SHA-256; reference comparisons already did.
- Indexing verifies the current `midi/notes.mid` or `import/notes.mid` against
  the comparison hash and verifies authorized comparison `song_id` against
  generated metadata. Changed MIDI or mismatched identity aborts before index
  output, preventing stale metrics from being presented as current.
- Added a focused stale-lineage regression test. Runtime generation and metric
  calculation remain unchanged.
- Validation: four focused index tests passed; full suite passed 201 tests.

### 2026-10-03 07:35 - Sanitized comparison trends added

- Added `compare-comparison-indexes` for deterministic JSON and Markdown deltas
  between compatible sanitized corpus indexes. Compatibility requires the same
  authorized/reference schema and explicit chart-comparison metric semantics.
- Pairing requires matching song IDs and exact display identities. Coverage
  separately records paired available comparisons, shared missing comparisons,
  baseline-only identities and candidate-only identities; asymmetric sets are
  never represented as fully paired.
- Safe per-song and macro deltas cover attack/chord/sustain F1, exact lane
  agreement, motif coverage/shape agreement and weakest-window recall. JSON
  ranks improvements and regressions per metric; Markdown concisely displays
  macro changes and the largest attack-F1 movements.
- Null metrics remain explicitly unavailable and are excluded from only that
  metric's macro denominator. No null is coerced to zero.
- Trend loading revalidates linked comparison-report SHA-256 and current
  generated-MIDI SHA-256, rejects duplicate display/song identities, malformed
  metric values, incompatible schemas/semantics and stale lineage, and emits no
  paths, links, events or reference-note rows.
- Focused tests cover paired/asymmetric coverage, missing comparisons, null
  metrics, incompatible schemas, duplicate identities, stale lineage and output
  sanitization. Runtime generation remains unchanged.
- Validation: eight focused index/trend tests passed; full suite passed 205
  tests; CLI help was exercised.

### 2026-10-03 16:00 - Extended autonomous checkpoint

- Restored and expanded post-generation reference evaluation without allowing
  reference notes into generation. `compare-midi` now provides automatic/manual
  alignment, hashes, Git provenance, unmatched-event details, weakest windows,
  per-authoring-stage attribution and sanitized Markdown.
- Replayed the seven-song official-reference corpus through current code.
  Playability improves or preserves final attack F1 on five songs but remains a
  loss point on Hail and Cliffs. Two independently selected density/spacing
  recovery gates were exact no-ops on the three rights-cleared development
  songs and were removed before official evaluation.
- Retained diagnostic attribution for per-beat-cap removals, spacing rejects,
  replacements and cleanup. Rights-cleared evidence showed the same weak/plain
  candidate class as the official matched removals, so no unsafe runtime
  recovery rule was accepted.
- Retained `audit-playability-evidence`. Cleared-song feature conjunctions
  reached at most 8.6% precision and did not transfer safely to Persephone.
  Stronger official-domain correlations remain evaluation-only and rejected for
  threshold selection.
- Established lawful independent-capture calibration. Guitar-TECHS mic/DI and
  GuitarSet mic/pickup agreement consistently raised attack precision across
  fixed tolerances and untouched splits. The 28 authorized chart pairs still
  contain zero independent paired captures, so this evidence is external
  calibration rather than per-song runtime permission.
- Added provenance-validated source-agreement pinning/evaluation, additive
  chart-pair lineage states, sanitized review Markdown, corpus comparison
  indexes, stale-lineage rejection and paired index trend reports. Real
  authorized trend generation correctly reports the rejected spacing candidate
  as zero delta across all three development songs.
- Fixed one-tick authored-versus-serialized MIDI disagreement on floating-point
  half-tick boundaries by sharing the post-lead tick conversion. The real Jonas
  replay now validates. Cached replay failures also return concise exit-1
  messages rather than tracebacks.
- Retained commits from this checkpoint:
  `c494ce0`, `5af24ed`, `5e93528`, `43ebc79`, `27ed594`, `f329a72`,
  `18da809`, `ce46916`, `6a79f57`, `c9e1fc6`, `97687f1`, `ccc5c57`,
  and `01fb097`.
- Final automated gate: 207 tests passed with six existing dependency/
  deprecation warnings. The branch is synchronized with origin and only the
  intentionally local `comparisons/` tree is untracked.
- Human review remains limited to Hail Hard guitar density/feel in the saved
  morning review. Less Talk remains informational: its Expert scaling-riff
  attack/pitch sequence is still upstream-blocked, and no rejected recovery
  hook was restored.

### 2026-10-03 16:10 - Comparison lineage hardening retained

- Exercising sanitized trends on the real three-song authorized development
  corpus exposed that manifest `song_id` and generated package `song_id` are
  distinct identities. Authorized reports now bind both: the manifest identity
  remains the corpus key while generated package ID, artist/title and MIDI hash
  establish current-run lineage. The real rejected spacing experiment reports
  three paired songs and exactly zero attack-F1 delta.
- Index output is restricted to the corpus root so relative report links cannot
  escape the revalidation boundary. Trend loading rejects escaping links,
  input-index overwrite, altered linked reports, generated MIDI changes, and
  index summaries that no longer equal the hashed report's sanitized summary.
- Fixed index Markdown rendering for explicitly unavailable/null metrics rather
  than formatting them as numbers.
- Focused index/trend validation passes 11 tests; the full suite passes 209
  tests with six existing dependency/deprecation warnings.

### 2026-10-03 16:20 - Independent-source provenance tightened

- Source-agreement evaluation now requires every alternate recording path and
  SHA-256 to be explicit in the validated paired manifest. The prior fallback
  could derive an alternate path and compute its hash without making that input
  part of the predeclared manifest lineage.
- Pinning and evaluation both reject alternate audio whose bytes equal any
  primary recording in the manifest. Identical files can no longer appear as
  perfect agreement between supposedly independent captures.
- Existing GuitarSet and Guitar-TECHS measurements already use explicit pinned
  alternates and remain unchanged. Focused provenance tests and the 209-test
  full suite pass.
- Chart-pair evidence attachment now independently re-hashes the paired
  manifest's primary audio, labels and alternate recordings rather than trusting
  the evaluation report's verification count. Duplicate, missing or extra
  evaluation item identities and changed alternate bytes produce `invalid`
  lineage without exposing their metrics.

### 2026-10-03 16:30 - Final hygiene gate

- Revalidated both saved morning-review drafts with the current validator and
  package-input manifests: 2/2 passed.
- `pip check` reports only Basic Pitch 0.4.0's declared TensorFlow dependency.
  This is the documented intentional exception: Setup installs Basic Pitch with
  `--no-deps`, and RB3 Charts executes its bundled ONNX model through
  ONNX Runtime. Installing TensorFlow would add an unused heavyweight runtime.
- The branch is synchronized with `origin/codex/authorized-chart-pilot`; only
  the intentionally local `comparisons/` tree remains untracked.
- Hail voice/attack recovery is blocked pending independent per-song source
  evidence that separates true weak attacks from the same weak/plain class on
  cleared songs. The only current human question is the saved Hard-guitar
  density review.
- Less Talk Expert scaling-riff recovery is blocked pending stronger
  lane-independent attack/pitch/voice evidence. Existing grid, motif, stem,
  TabCNN and structure-fusion candidates were rejected and remain absent from
  runtime generation.

### 2026-10-03 16:11 - Hail Hard synchronized A/B review

- Added `compare-reviews` for two completed drafts of the same source. It
  verifies song identity, source SHA-256, tempo map and lead-in, then writes one
  local page with a shared transport and instant same-timestamp A/B switching.
- Version-only attacks receive cyan outlines. The densest eight-second
  candidate-addition windows become jump buttons so the reviewer need not search
  the whole song manually.
- Generated
  `comparisons/morning-review/Hail_To_The_King_Hard_AB.html` from the exact
  retained 0.46 and 0.38 spacing drafts. It defaults to Guitar/Hard/Candidate:
  A has 850 attacks and B has 969; Expert is identical at 1,600 attacks.
- The review question is now concrete: does B preserve the intended Hard rhythm
  more naturally than A without becoming tiring or over-dense? A/B keys switch
  versions while playback continues.
- Full validation passes: 210 tests, with the six existing dependency and
  deprecation warnings.

### 2026-10-03 16:23 - Hail review feedback traced

- Human review marked the current Hail draft as needs-work: Hard still appeared
  under-dense; Expert remained over-dense in the intro; and a lane-pattern
  descent near player 15 seconds was musically interesting but inconsistent
  with the established opening phrase. The generated chart remains substantially
  different from the official evaluation chart.
- Fixed an independent A/B review bug discovered during that review. The page
  had omitted colored strike targets and allowed note heads to continue below
  the strike line. It now draws explicit targets, clamps attacks to the hit line,
  removes passed attacks immediately, and displays both player and source time.
- Player 15.0 seconds maps to source 11.16 seconds. The source-derived voice
  diagnostic covering 8.36-12.43 seconds has 14.24-semitone register separation,
  76.46% upper-register evidence, 78.38% weak events and 12 register switches.
  The detector briefly reports 73→68→63; final phrase authoring expands that to
  lanes 2→1→0.
- Tested a post-mapping dominant-voice filter so removal could not alter the
  song-wide pitch dictionary. It required at least 70% dominant evidence,
  14-semitone separation, 70% weak events, sub-0.65 note confidence, no sustain,
  and unanimous overlapping-window classification.
- Rejected the filter on rights-cleared data before Hail evaluation. Certain
  Death and There's Something Wrong were no-ops. Dirtbag removed three attacks
  and slightly reduced movement agreement. Persephone removed 30 attacks,
  lowering Expert attack F1 0.5537→0.5363 and exact lanes 0.2624→0.2523,
  despite motif shape 0.2903→0.3103. No runtime music logic remains.

### 2026-10-03 16:36 - Hail generated Hard compared with official Expert

- Extended evaluation-only `compare-midi` with an explicit
  `--reference-difficulty`; generation and reference difficulties no longer
  need to match. Non-Expert generated comparisons correctly omit Expert-only
  authoring-stage attribution.
- Froze alignment at -1.10 seconds, independently recovered from both identical
  generated Expert charts, then compared the 0.46- and 0.38-spacing generated
  Hard charts against official Expert.
- Baseline A: 850 generated attacks, 475/1,150 official Expert attacks matched,
  precision 0.5588, recall 0.4130 and F1 0.4750.
- Candidate B: 969 generated attacks, 549 matches, precision 0.5666, recall
  0.4774 and F1 0.5182. The 119 added attacks produced 74 additional matches;
  precision increased by 0.0078 rather than falling.
- In every four-second window from source 4-24 seconds, all generated Hard
  attacks matched an official Expert onset within 80 ms. Candidate B raised
  window recall from 0.3548-0.4375 to 0.3871-0.4688 while retaining 1.0 timing
  precision in those windows.
- This supports the reviewer's observation that the denser Hard chart resembles
  official Expert attack placement. It does not solve pattern authoring:
  exact-lane agreement fell 0.1474→0.1330, although movement direction improved
  0.2769→0.3176. Relative motif-shape agreement remained zero.
- Added `review-midi-comparison` and generated
  `comparisons/morning-review/Hail_Generated_Hard_vs_Official_Expert.html`.
  Its shared transport switches A (969-note generated Hard) and B (1,150-note
  aligned official Expert) at the same timestamp. The official markers are
  materialized only in the ignored local companion folder and never enter
  generation.

### 2026-10-03 16:47 - Three-way Expert/Hard/official review

- Clarified the human observation: project Hard is not being proposed as a
  difficulty match for official Expert. It appears to be a cleaner version of
  what project Expert should have interpreted, while project Expert's event
  choices and play pattern remain substantially wrong.
- Generated
  `comparisons/morning-review/Hail_Project_Expert_vs_Hard_vs_Official_Expert.html`.
  Its shared transport switches A (1,600-note project Expert), B (969-note
  project Hard) and C (1,150-note aligned official Expert); it starts on B.
- A has higher official-Expert recall/F1 (0.7357/0.6153) by charting 39.1% more
  attacks than C. B has lower recall/F1 (0.4774/0.5182), but its total count is
  much closer to C and timing precision is higher (0.5666 versus 0.5288).
- Exact-lane agreement is essentially unchanged and poor (A 0.1336, B 0.1330).
  B therefore exposes a useful de-noised attack backbone, not a solved lane
  pattern. A safe future experiment is to measure whether reduction-retained
  attacks carry stronger source evidence than Expert-only attacks across the
  rights-cleared corpus, then use that evidence to improve canonical Expert
  selection without deriving Expert from Hard or using reference notes.
- Follow-up visual inspection softened the earlier intro-density claim: raw
  density alone may be reasonably close to the official intro. The actionable
  defect is the selection, clustering, continuity and lane/play pattern of
  those attacks. Future experiments must not treat a blanket density reduction
  as the target.

### 2026-10-03 17:09 - Traced repeated single-yellow runs

- Traced the reported player-time windows near 17, 27, 30 and 32 seconds
  (source time is player time minus the 3.844-second review lead-in).
- The failure has more than one cause. From player 16.7-18.6 seconds, Expert
  has 13 attacks and seven yellows; the five yellow attacks retained by Hard
  all carry detected pitch 70. This is primarily repeated-pitch transcription
  combined with same-pitch lane stability.
- From player 31.8-33.0 seconds, Expert has eight attacks (six yellow, one red,
  one blue), while Hard retains five attacks and every one is yellow. Those
  five attacks carry detected pitches 63, 73, 73, 75 and 77, proving that this
  run is not caused by absent pitch variation. Phrase authoring collapses the
  distinct pitches onto yellow, then time-based Hard reduction removes the
  remaining color anchors and exposes a monochrome line.
- Hard inherits Expert lanes except for limited edge-jump normalization; it
  does not re-author a coherent reduced fingering pattern. Future work should
  distinguish same-pitch repetition from collapsed nonzero pitch intervals and
  make reduction preserve supported contour anchors rather than merely sampling
  attacks by spacing.

### 2026-10-03 17:20 - Protected Expert contour from pitch-anchor flattening

- Kept the implementation scoped to canonical Expert authoring. Repeated-pitch
  lane stabilization may still make recurring pitches consistent, but it can
  no longer overwrite an already-correct adjacent pitch/lane direction into a
  flat or reversed move.
- Isolated cached baseline versus candidate replays changed no attacks or
  timing. Collapsed Expert pitch changes fell:
  - Certain Death: 22 to 3.
  - There's Something Wrong: 32 to 8.
  - Dirtbag: 56 to 50.
  - Held-out Persephone: 25 to 11.
  - Hail evaluation: 139 to 35.
- Authorized lane evaluation:
  - Certain Death exact lanes 0.0153 to 0.0165; movement unchanged at 0.4167.
  - There's Something Wrong exact lanes 0.3453 to 0.3373; movement
    0.5839 to 0.6510; motif shape 0.5294 to 0.5343.
  - Dirtbag was unchanged.
  - Held-out Persephone exact lanes 0.2624 to 0.2536; movement
    0.5178 to 0.5330; mean interval error 1.1429 to 1.1357.
- Frozen post-generation Hail evaluation also improved exact lanes
  0.1336 to 0.1430 and movement 0.4071 to 0.4390. The player 32-second Expert
  window now preserves the large 63-to-73 pitch rise instead of flattening both
  attacks to yellow.
- Retained because the source-derived contour metric improves broadly and the
  held-out movement gain outweighs the small absolute-color shift. This does
  not solve repeated same-pitch transcription at player 17 seconds or the
  contour-blind Hard reduction; those remain separate tasks.

### 2026-10-03 18:20 - Recovered compact repeated-register riffs

- Traced the player 17-second Hail failure through both voice-selection passes.
  pYIN and Basic Pitch independently supported recurring pitches 51 and 58,
  but recent-register octave anchoring moved them near pitch 70 and the
  sequential-outlier filter then treated the remaining pedal notes as isolated
  harmonic spikes.
- Rejected four broad recovery variants before retention. They restored the
  local riff but changed too many attacks and slightly regressed held-out
  Persephone or whole-song movement. The broadest retained 72 extra Hail
  attacks and reduced frozen attack F1 0.6153 to 0.6130.
- The retained rule stages rather than immediately applies alternate-register
  proposals. Evidence must agree between raw pYIN and a Basic Pitch candidate,
  oppose both adjacent anchored pitches by at least six semitones, recur at
  least four times within three beats, stay within a seven-semitone pedal
  register and alternate against one local anchor register. Sequential cleanup
  applies the same compact-family requirement.
- Speculative proposals no longer affect duplicate consolidation or ordinary
  spike decisions. Unsupported proposals restore the exact anchored event
  before mapping and authoring. Held-out Persephone considered eight proposals,
  accepted none, and produced a byte-for-byte identical 660-event lead
  transcription and identical Expert evaluation.
- Rights-cleared Expert results versus the committed contour baseline:
  - Certain Death: attacks 1200 to 1203, attack F1 0.7996 to 0.8004;
    movement unchanged at 0.4167.
  - There's Something Wrong: attacks 964 to 966, attack F1 0.8228 to
    0.8230, movement 0.5839 to 0.6510 and motif shape 0.5294 to 0.5343;
    exact lanes shifted 0.3453 to 0.3356.
  - Dirtbag: attacks 831 to 834, attack F1 0.6220 to 0.6233, exact lanes
    0.1038 to 0.1111 and motif shape 0.0256 to 0.0476; movement stayed
    0.3558.
  - Held-out Persephone remained exactly unchanged at 625 attacks, 0.5537
    attack F1, 0.2624 exact lanes and 0.5178 movement.
- Hail Expert now has 1,618 rather than 1,600 attacks. In the reviewed player
  17.65-18.62 second span, recovered pitches 51/58 create red/yellow/blue
  motion instead of the prior pitch-70 yellow line. Using the same frozen
  -1.10-second post-generation comparison, attack F1 changed 0.6087 to 0.6091,
  exact lanes 0.1768 to 0.1791, movement 0.4230 to 0.4317 and mean interval
  error 1.7622 to 1.7295.
- Added cached Band replay of accepted candidate evidence plus the saved guitar
  stem, avoiding source decode, Demucs and Basic Pitch inference while still
  recomputing deterministic pYIN lead selection. Generated
  `comparisons/morning-review/Hail_Expert_Repeated_Riff_Recovery_AB.html`
  for synchronized Expert listening review.

### 2026-10-03 18:40 - Review strike geometry corrected

- Fixed the normal review highway retaining tap heads on the white strike line
  after their authored onset. Tap heads now cross once and disappear; during a
  pitched sustain, only the active tail remains pinned to the target until
  release. Chart and audio timestamps are unchanged.
- Synchronized A/B reviews now select the shared difficulty with the most
  changed attacks for the chosen instrument. The Hail repeated-riff review
  opens on Expert, while the earlier Hard-density comparison still opens on
  Hard.
- Regenerated
  `comparisons/morning-review/Hail_Expert_Repeated_Riff_Recovery_AB.html`
  with Guitar/Expert as its default review surface.

### 2026-10-03 18:50 - Same-lane review markers added

- Added difficulty-specific, non-destructive review markers for five or more
  consecutive single-lane attacks no more than 0.75 beats apart. A marker
  distinguishes a repeated detected pitch from a collapsed pitch contour when
  at least three pitches span four or more semitones on one lane.
- Markers change only review navigation. They do not reject notes, alter lanes
  or classify a generated chart as correct/incorrect.
- A current cached Hail replay produced 13 Expert and eight Hard guitar markers.
  Hard now exposes the reported player 17-second defect directly:
  `9 yellow attacks; pitches 51-70; through 19.28s`. The current Expert does
  not contain that marker after repeated-register recovery, isolating the
  remaining failure to difficulty reduction rather than lead selection.
- Generated the current normal review at
  `comparisons/review-markers/hail/Avenged_sevenfold_Hail_To_The_King_20261003_183319_643726/review.html`.

### 2026-10-03 19:00 - Hard contour anchors preserved

- The player-17 Hard marker traced to greedy spacing selection, not lane
  remapping: Hard retained nine yellow Expert attacks with pitches 51-70 while
  skipping the interleaved blue/red Expert anchors.
- Added a bounded Hard-guitar reduction safeguard only for five-plus-note
  same-lane runs with at least three retained pitches spanning four semitones.
  It swaps retained attacks for omitted Expert attacks of another lane, keeps
  attack count fixed, preserves source IDs/times, allows at most one swap per
  three collapsed attacks and enforces a 0.32-beat local spacing floor.
- Confidence-first selection was rejected because it removed recovered pitch
  51. The retained selector maximizes pitch-span preservation before lane
  contrast and confidence. Hail keeps pitches 51/58 and now charts blue anchors
  at player 17.12, 17.48 and 18.81 seconds; no collapsed-contour marker remains
  in that reviewed Hard span.
- The candidate changed zero Hard attacks on Certain Death, There's Something
  Wrong, Dirtbag and held-out Persephone compared with the current committed
  replay baseline.
- Frozen post-generation Hail Hard-versus-official-Expert evaluation kept 976
  attacks, 554 matches and 0.5212 attack F1. Movement improved 0.3692 to 0.3728;
  exact lanes shifted 0.1588 to 0.1570 and mean interval error 1.6302 to
  1.6350. Reference notes did not select the rule.
- Generated
  `comparisons/morning-review/Hail_Hard_Contour_Anchors_AB.html`; it opens on
  Guitar/Hard and highlights the three same-count replacements.
