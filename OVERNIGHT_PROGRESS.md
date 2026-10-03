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
