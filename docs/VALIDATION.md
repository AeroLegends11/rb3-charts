# Validation record

## Riff-cycle boundary correction — 2026-09-15

The review draft in
`output/Serj_Tankian_Empty_Walls_20260915_112234_533510/` fixes a phrase-tracking
error exposed by the previous lead-filtered chart. A partial three-note response
had been allowed to reach across a rest and consume the first note of the next
riff as its fourth note. Transposed-pattern projection now requires every timing
gap to remain within one sixteenth-note subdivision of the learned phrase. A
confirmed full cycle may apply its fret vocabulary backward only to the
immediately preceding partial response, keeping identical pitches on identical
colors without inventing an attack.

The opening retains its G, G, R, Y statements and G, R, Y partial response. The
raised passage at 8.655 seconds is R, R, Y; the complete cycles at 9.405 and
10.155 seconds are R, R, Y, B. The lead pass resolved 334 ambiguous onset groups
and conservatively removed 69 rapid between-note spikes. Guitar Expert contains
843 attacks, a 6.25 attacks/second peak, 0.5% chords, and 11.3% wide jumps. All
62 automated tests pass, including regressions for rests between cycles and
backward coloring of a preceding partial response. No Xbox package was created.

## Guitar-stem secondary voice separation — 2026-09-15

The app already separates the source mix into guitar, bass, drums, piano, vocals,
and other stems before transcription. The review draft in
`output/Serj_Tankian_Empty_Walls_20260915_110942_083017/` adds a second stage for
the guitar stem: after monophonic contour selection, it rejects unsupported
sequential pitch spikes that behave like an interleaved harmonic or secondary
guitar voice. A note is removed only when nearby lead notes support a different
voice and it is either a rapid spike between agreeing neighbors or an isolated
local extreme. Coherent high-register runs and strong chords remain eligible.

This pass removed 87 sequential outliers after resolving 334 ambiguous onset
groups. Guitar Expert fell from 900 to 830 attacks, peak four-second density fell
from 7.0 to 6.25 attacks/second, and wide-jump frequency fell from 17.0% to 11.4%.
The opening G, G, R, Y statements and the intentional three-note response remain
unchanged. Guitar chord frequency is 0.5%, holds are 15.5%, and difficulty is
3.99, tier 4 (Challenging). All 60 automated tests pass. No Xbox package was
created.

## Empty Walls lead-guitar reader — 2026-09-15

The Auto precision draft in
`output/Serj_Tankian_Empty_Walls_20260915_110253_640644/` adds a lead-selection
stage before fret mapping. It compares Basic Pitch's polyphonic candidates with
a 22.05 kHz pYIN contour from the isolated guitar stem. Of 1,448 candidates in
985 onset groups, it resolved 334 ambiguous groups, made 130 pitch corrections,
anchored 224 octave decisions to the recent lead register, removed 47 split
activations, and retained 965 lead events before later playability filtering.
Strongly supported simultaneous pitches remain eligible as authored chords.

The resulting Expert intro holds G, G, R, Y for its first three statements. The
next response at 5.093 seconds is G, R, Y; onset analysis of the isolated stem
finds attacks at 5.085 and 5.294 seconds but no attack in the predicted 4.905-
second slot, so the program preserves the rest instead of inventing a gem. It
then returns to four consecutive G, G, R, Y statements before the register change
at 8.655 seconds begins R, R, Y, B. Guitar Expert contains 900 attacks, 0.3%
chords, 15.6% holds, and a 7.0 attacks/second peak. The estimated difficulty is
4.47, tier 4 (Challenging), within the requested 4–5 range. All 58 tests pass.
No Xbox package was created.

## Empty Walls transposed-riff continuation and difficulty — 2026-09-15

The latest Auto precision draft is
`output/Serj_Tankian_Empty_Walls_20260915_104915_204188/`. The stabilizer keeps
the final original-root statement at 7.905 seconds as G, G, R, Y after correcting
two weak harmonic/octave estimates. At 8.655 seconds it recognizes the five-
semitone lift as the related compact R, R, Y, B shape. For the following two
measures, the most recent confirmed transposition owns the local fret vocabulary;
weak harmonic estimates no longer revert to green or create overlapping chords.
Locked phrase notes cannot be overwritten by an earlier overlapping template.

The pattern pass found three exact families and fourteen exact occurrences. It
accepted eight conservative projections, inferred nine low-confidence pitches,
and propagated the local vocabulary to 44 notes. Guitar Expert has 903 attacks,
21.8% chords, 15.5% holds, and a 7.0 attacks/second peak over four seconds. Its
difficulty estimate is 4.93, which maps to tier 5, Nightmare, and suggested RB3
rank 370; `import/song.ini` contains `diff_guitar = 5`. The assessment still asks
for close listening because adjacent motion is 62.8% and wide jumps are 16.6%.
The reported 0:56 congestion and 1:20 articulation/hold passages remain explicit
human review points. All 54 automated tests pass. No Xbox package was created.

## Confidence-gated repeated-phrase fingering — 2026-09-15

Band Guitar now distinguishes repeated riffs from ordinary song content. It
groups the same four-note pitch sequence only after at least three occurrences
with compatible beat spacing, derives a compact lane shape, and applies it only
to those occurrences. Equal pitches return to the same fret. One-off fills,
unmatched passages and chord riffs remain unchanged. This captures the fast but
learnable pattern principle requested from My Name Is Jonas and Less Talk More
Rokk without forcing repetition across every song.

The Auto run in `output/Serj_Tankian_Empty_Walls_20260915_005606_110484/`
recognized pitch sequence 48, 48, 50, 51 six times and stabilized it as G, G, R,
Y. The matches begin at 2.655, 3.405, 4.196, 7.204, 207.530 and 208.951 seconds.
Only six notes changed; all other material passed through the existing Band
authoring rules. All 50 tests pass. No Xbox package was created.

## Auto/Band repeated-riff stability — 2026-09-15

The report that Auto produced too many different notes in the Empty Walls intro
was reproduced in the prior draft: the same detected pitch could change colors
after a nearby note, and isolated simultaneous harmonics could survive as chords.
Band Guitar now keeps pitches that recur within an eight-measure section on a
stable fret. Generated chord frequency targets the finished-chart corpus median
instead of its permissive 90th percentile, and low-confidence isolated chords in
an otherwise single-note four-beat neighborhood collapse to their root note.

The Auto review-only run in
`output/Serj_Tankian_Empty_Walls_20260915_004946_219589/` selected Band from the
stem evidence. Expert begins G, G, R, Y at 2.655 seconds and repeats G, G, R, Y
for the next two complete statements. The earlier G+R chord at 7.343 seconds is
now one green note. Corpus conformance removed 44 isolated phrase chords and
reduced total Guitar chord frequency from 32.1% to 22.9%. All 48 tests pass. No
Xbox package was created.

## Harmonix reference-corpus conformance — 2026-09-14

The local chart-only reference set contains 21 finished Harmonix-authored songs:
five Rock Band 3 songs and sixteen Rock Band 1/2 songs from the public Onyx source
repository. No song audio was downloaded. `reference/style-profile.json` records
per-difficulty distributions for attack spacing, peak density, chord shapes,
three-note chords, sustains, lane use, fret movement and Expert-to-Hard retention.
Band generation now applies the corpus's song-level 90th-percentile limits to
ambiguous chords, triads and holds, then checks Guitar motion against the corpus's
10th-to-90th-percentile range.

The final review-only Empty Walls run is
`output/Serj_Tankian_Empty_Walls_20260914_175118_471795/`. Guitar Expert contains
903 attacks, 32.1% chords, 1.9% triads and 15.5% holds. Its adjacent-fret motion
is 72.9% and its wide-jump rate is 8.9%, compared with reference song bands of
70.0–90.8% and 2.7–14.5%. Hard contains 645 attacks, retains 71.4% of Expert,
contains no triads or green/orange chords, and measures 73.6% adjacent motion and
9.9% wide jumps. The automatic assessment reports no congestion, lane, contour,
chord or sustain flags. All 47 tests pass. The run remains a review draft with no
Xbox package.

## Empty Walls Expert and Hard Guitar authoring pass — 2026-09-14

The review-only run in
`output/Serj_Tankian_Empty_Walls_20260914_151143_883571/` adds a Band Guitar
authoring pass based on the Rock Band Network conventions. Each eight-measure
section now assigns its strongest local pitches to adjacent frets, small onset
jitter within 28 ms is aligned to a sixteenth or triplet grid, Expert three-note
green/orange spans are removed and Hard contains neither three-note nor
green/orange chords. At 160 BPM and above, long Hard eighth-note sequences drop
every fourth attack, and quick green/orange jumps wrap inward.

The resulting Guitar charts contain 903 Expert and 645 Hard attacks. Expert and
Hard each contain zero prohibited wide chord shapes; Expert hold frequency is
19.6%, and melodic contour agreement is 98.0%. The opening common pitches now
follow the adjacent green, red, yellow, blue and orange sequence instead of
collapsing or making semitone-sized wide jumps. All 43 tests pass. The run is a
review draft with no Xbox package.

## Batch musical-review notes — 2026-09-14

Every generated song now receives `analysis/quality-assessment.json` and an
editable `review_notes.md`. The assessment classifies the draft as Close enough
to review, Review closely or Needs work from lane coverage, peak density,
candidate rejection, hold and chord frequency, recurring patterns and melodic
contour. Batch runs additionally create `batch_review_*.md` with one row per
song, direct review links and space for cross-song observations and timestamps.
Failed or partial songs remain in the report without preventing later songs from
finishing.

The full regression run in
`output/Serj_Tankian_Empty_Walls_20260914_150539_528120/` produced both files
and classified the cleaned chart as Close enough to review. Its recorded Guitar
metrics are 904 attacks, five-lane coverage, 19.6% holds, 32.0% chords, 12.4%
candidate rejection, 7.0/s peak density and 93.7% contour agreement. All 43
automated tests pass.

## Empty Walls congestion and profile-specific sustains — 2026-09-14

The review-only run in
`output/Serj_Tankian_Empty_Walls_20260914_124148_728506/` separates Band sustain
and density behavior from the Electronic rules used for Fineshrine. Band pitched
parts now require 220 ms of supported duration for a hold; shortened fragments
become taps. Ordinary Band passages permit three attacks per beat, with a fourth
allowed only for a detected recurring fast riff.

Compared with the congested Empty Walls draft, Guitar Expert fell from 975 to
904 attacks, peak four-second density fell from 8.0 to 7.0 attacks/second and
visible holds fell from 681 to 177. In the first 45 seconds, attacks fell from
172 to 163 and holds from 138 to 33 while all five lanes remain in use. All 43
automated tests pass. The result remains `draft_ready` with no Xbox package.

## Recurring riff and groove recognition — 2026-09-14

The charting pipeline now detects recurring four-attack patterns from musical
shape and quantized beat spacing before applying the density limit. For Guitar,
Bass and Keys, shape means pitch intervals plus chord size; for Drums it means
the pad sequence. Pattern-supported attacks receive a bounded preference during
crowding decisions. Each run writes `analysis/patterns.json` with recurring
families, occurrence times, lane use, supported-attack counts and pitched
contour checks.

The Band regression in
`output/Serj_Tankian_Empty_Walls_20260914_123733_786080/` finds 16 recurring
Guitar pattern families, with 97 final attacks participating in them. Its
non-flat Guitar lane motion agrees with pitch direction 94% of the time. The
Electronic regression in
`output/Purity_Ring_Fineshrine_20260914_123752_265086/` finds 72 recurring lead
pattern families, with 421 final attacks participating and 100% directional
agreement. Both runs are review-only drafts with no Xbox package. All 43 tests
pass.

## Empty Walls Band lane-contour pass — 2026-09-14

The review-only run in
`output/Serj_Tankian_Empty_Walls_20260914_123514_908173/` verifies that isolated
high pitches no longer compress the intro riff into one fret. Band mapping now
uses weighted pitch percentiles from the commonly played register. In the first
45 seconds, Guitar Expert uses all five lanes: 59 green, 68 red, 53 yellow,
15 blue and 14 orange gems. The opening pattern now follows pitches 48, 50, 51,
53, 55, 56 and 60 across the lane contour. The chart remains `draft_ready`
with no Xbox package.

All 43 automated tests pass, including a regression where a high octave outlier
must not flatten a lower guitar riff into one lane.

## Fineshrine repeated-attack and decay pass — 2026-09-14

The review-only run in
`output/Purity_Ring_Fineshrine_20260914_123245_901317/` verifies the reported
passage near source time 1:20. Guitar Expert contains separate green attacks at
78.211 and 78.905 seconds. Pitch-band energy places their releases at 78.477
and 79.091 seconds, producing audible 266 ms and 186 ms sustains without joining
the two attacks. The release detector follows energy down to 24 dB beneath the
local peak, stops above the preceding noise floor and always caps a sustain
before the next same-pitch attack.

The run selected Electronic automatically at 130.001 BPM. Its Guitar chart has
606 Expert attacks, a 115.5 ms minimum interval and a peak four-second density
of 4.75 attacks/second. It remains `draft_ready` with no Xbox package. All 43
automated tests pass, including recovery of repeated same-pitch attacks and
preservation of brief envelope-supported sustains.

## Automatic charting-style selection — 2026-09-14

Auto is now the default and compares genre metadata with RMS energy from the
separated guitar, piano and `other` stems. On the two commercial regression
recordings it resolved Empty Walls as Band (guitar-to-alternate ratio 4.0686)
and Fineshrine as Electronic (ratio 0.0312). The selected profile, reason, stem
energies and ratio are written to `analysis/chart-profile.json`; manual Band and
Electronic choices remain overrides for ambiguous mixes.

The full Auto Fineshrine run is
`output/Purity_Ring_Fineshrine_20260914_004258_132955/`. Its MP3 genre is
Synthpop and its stem evidence independently selects Electronic. It generated
the same playability-limited 461/358/238/140 Guitar difficulties and stopped at
`draft_ready` with no package.

43 automated tests pass, including guitar-led, synth-led and genre-signaled
automatic selection.

## Fineshrine electronic realism pass — 2026-09-13

The supplied Purity Ring recording exposed two electronic-charting problems. The
broad `other` stem contributed quiet textures and octave harmonics as if each were
a separate lead note, and the coarse tempo estimator reported 130.731 BPM with an
unstable-looking earlier map. The updated pipeline now filters short weak texture
events, consolidates recurring octave families, suppresses implausibly close
attacks based on beat period, centers fret mapping on the weighted hook pitch and
fits a constant studio grid across the entire recording.

The latest review-only regression run is
`output/Purity_Ring_Fineshrine_20260914_003834_079010/`. It inferred 130.001 BPM
with a 0.224-second phase. After lead focusing, the general playability gate
removed another 70 crowded attacks and produced Guitar difficulty counts of
461/358/238/140. Expert peak four-second density is 4.5 notes/second and its
minimum interval is 115.5 ms. At source time 56 seconds, the four-second window
fell from the original 27 notes to 13 with a 116 ms minimum interval. The gate
retained 233 Guitar and 151 Bass Expert holds supported by transcription
durations. `analysis/playability.json` records the automatic decision. This run
intentionally has no CON package pending listening approval.

43 automated tests pass, including electronic studio-tempo refinement, harmonic
consolidation, focused texture filtering, centered lead-lane mapping and MIDI
hold-duration preservation. They also cover automatic per-beat crowding limits,
minimum spacing, preservation of supported holds and the higher density allowed
for conventional band parts. The review defaults to a 2.6-second spread view and
renders active sustain trails through their release point.

## Electronic profile — 2026-09-13

The electronic role-mapping path completed trained ADTOF and Basic Pitch ONNX
inference on the 34-second aligned-stem fixture. It generated drums, lead-role
Guitar and low-end-role Bass at all four difficulties. Onyx/Magma validation
then passed and produced `rb3c_ef9e7d6050a7.con` in
`output/RB3_Charts_Electronic_Profile_Demo_20260913_123140_125535/rb3/`.

The run also exposed two environment defects that are now fixed: setuptools 81+
removed `pkg_resources`, which resampy 0.4.2 still imports, and direct library
calls could send Unicode progress text to a legacy Windows console. Setup now
pins setuptools 80.9.0 and progress display falls back safely while preserving
the complete UTF-8 log.

## Real-song run — 2026-09-13

The supplied 320 kbps, 44.1 kHz “Empty Walls” MP3 (about 229.3 seconds) was
processed with precision separation and the trained transcription models.
All three requested instruments were generated, with no omissions. Magma
validation and Xbox CON compilation succeeded.

| Instrument | Expert | Hard | Medium | Easy |
|---|---:|---:|---:|---:|
| Guitar | 1032 | 643 | 389 | 209 |
| Bass | 528 | 413 | 280 | 174 |
| Drums | 931 | 789 | 460 | 237 |

Counts are note onsets, with simultaneous chords counted once. The final package
MIDI retained every onset count across all 12 charts. The maximum nearest-onset
timing difference from authoring MIDI was **0.391 ms** (rounded up). This measures
export timing preservation, not correctness of the original transcription.

The first compilation exposed missed beats in sparse sections: intervals reached
approximately 19 BPM, triggering compiler tempo conversion and a drum-fill
mismatch. Gap repair fixed that compiler failure. A later timing pass reconciled
the file's embedded 80 BPM tag with the audio evidence as a 160 BPM quarter-note
grid and estimated its phase at 0.030 seconds. This stable map compiled
successfully. Its measure/downbeat phase still needs listening review.

43 automated tests pass, including regression checks for retained beat anchors,
half-time BPM-tag reconciliation, electronic lead selection and kick-supported
bass retriggers, review controls and packaging reviewed drafts without repeating
analysis. Xbox testing is still outstanding.

Successful output: `output/Serj_Tankian_Empty_Walls_20260913_122013_983911/`.
Its `analysis/compiled-timing.json` contains all 12 timing comparisons.

## Original prototype checks — 2026-09-10

## Verified

- 12 automated tests pass: variable-tempo round trips, MIDI onset preservation,
  repeat-pattern mapping, difficulty ancestry/density, invalid charts, beat
  detection from clicks, matching audio lead-in, drum-fill endpoints and batch
  failure isolation.
- The NVIDIA RTX 5070 runs PyTorch 2.8.0 with CUDA 12.8 successfully.
- Both six-stem and fine-tuned Demucs separation have actually run on this GPU.
- Basic Pitch ONNX and ADTOF trained weights have both performed inference.
- The 34-second original synthetic demo, with supplied source-aligned stems,
  produced guitar, bass and drum charts at all four difficulties. Onyx/Magma
  validation completed and produced a CON package, including drum-fill authoring.
- The package's final MIDI retains all 61 guitar, 62 bass and 124 drum Expert
  onsets. Maximum nearest-onset change from authoring MIDI to final package MIDI
  was approximately **0.521 ms** for each instrument. This is serialization
  precision, not a real-world transcription accuracy guarantee.
- The full automatic mixed-audio path, including separation and automatic tempo,
  also compiled. Demucs did not identify the synthetic guitar-like tone well
  enough for a guitar chart; the result correctly reported that omission.
- The native GUI was rendered and visually inspected. A screenshot is saved as
  `gui-preview.png`.

`demo-measurements.json` records onset matching for the isolated-stem synthetic
fixture; `compiled-timing.json` records the final MIDI comparison. Neither measures
pitch, pad-label or human gameplay quality. Simple generated audio is not a
substitute for evaluation on a real commercial mix.

## Fixes found by execution

- Windows console encoding could not print Unicode arrows; CLI streams now use
  UTF-8 and pythonw launches redirect logs to a file.
- A difficulty-reduction test found Hard retained an entire fast passage; spacing
  rules were adjusted while preserving source-note ancestry.
- The native offscreen Qt renderer did not load system fonts; the GUI explicitly
  registers Segoe UI and uses a scrollable layout.
- Magma requires beat-aligned drum activation hits even when the fill's MIDI
  endpoint is placed off-grid. Only an existing shared hit within 12 ms can be
  adjusted; the change is recorded. Otherwise that fill is omitted.

## Not yet verified

- The initial prototype had no real-song audio; this was supplied and processed on September 13, as recorded above.
- Actual Xbox 360 loading, audio synchronization, instrument behavior and
  playability have not been tested.
- CPU fallback is implemented but the complete separation pipeline was tested
  on CUDA, not benchmarked end-to-end on CPU.
- Vocals/harmonies and advanced authoring features listed as unavailable in the
  README are not implemented.
