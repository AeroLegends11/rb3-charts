# Review checklist: `codex/hand-window-pattern-reuse` (PR #21)

Ask the assistant: "Pull `codex/hand-window-pattern-reuse` and walk me through the review checklist."

## Setup

```powershell
git fetch origin
git checkout codex/hand-window-pattern-reuse
.\.venv\Scripts\python.exe -m pytest -q
```

Expect everything to pass. These changes were only tested on synthetic data.

## What changed

1. **Hand window** (`rb3charts/phrases.py`): rising past orange resets to green, falling past green resets to orange.
2. **Same pattern, other key**: same contour and rhythm reuses fingering (intervals within 1 semitone).
3. **No same color on a real pitch change**: phrase resets, motif reuse, phrase boundaries, and Hard/Medium reductions (`rb3charts/chart.py`).
4. **Guitar stem bleed cleanup** (`rb3charts/stem_cleanup.py`, `audio.clean_guitar_stem`): writes `guitar-clean-*.wav`; raw stem kept as `guitar_raw`; `--raw-guitar-stem` or the GUI checkbox "Suppress vocal/drum/bass bleed in the guitar stem" disables it.
5. **Tempo**: synthetic acceleration tests only; no behavior change.

## Update log

- Added the GUI checkbox for the guitar bleed cleanup (default on).
- The run log now reports how much RMS the cleanup kept (roughly 70-95% is plausible; under 50% suggests over-suppression).
- Basic Pitch ONNX now runs in the test env, so there is an audio-level benchmark (`rb3charts/transcription_bench.py`): synthetic guitar + leaked vocals/drums, raw vs cleaned stem, scored against known notes. Precision rose from ~0.54-0.59 to ~0.77-0.80 with recall unchanged (~0.98).
- Cleanup made stronger from that sweep (weight 2, floor .03, power 2). Synthetic only: on real songs listen for thin/watery guitar; if so, `--raw-guitar-stem` or ease the constants in `stem_cleanup.py`.
- Lead-focus fix (`transcribe._focus_band_guitar_candidates`): it was rewriting a confident true note to the contour's octave (recall 0.98 -> 0.91 after focus). Now octave-equivalent candidates tie-break on confidence and the octave rewrite only applies to low-confidence picks. Cleaned-stem benchmark after focus: precision 0.88 -> 0.96, recall 0.91 -> 0.98. Check real riffs for wrong-octave notes.
- Chord benchmark (`run_chord_benchmark`): lead focus was dropping power chords (recall 0.885 -> 0.677). Moderate-confidence fifth/octave stacks are now kept as a unit (recall 0.875). Listen for real chords being thinned to single notes, or bleed pitches kept as fake chord tones.
- New whole-pipeline synthetic test (`tests/test_end_to_end_synthetic.py`, needs Basic Pitch + ffmpeg): a transposed riff goes through `generate`. It exposed `refine_band_guitar` folding genuine octave notes onto each other across a 32-beat section (exact pitches 104/127 -> 127/127 after the fix). Octave folding now needs the established form within 1.5 beats and either clearly higher confidence or a 2+ octave gap. Check real riffs that jump an octave stay on the right register/lane, and that harmonic flips on distorted guitar are still folded.
- Hard guitar: the green/orange edge shift in `reductions` could recreate the same color on a pitch change (11 cases in the end-to-end run, 0 now). The end-to-end test asserts this for Hard and Medium.
- Steady quarter-note single-note lines (common in slower songs) were each treated as their own phrase because short gems leave a long rest, so a rising 52,55,57,59 line came out as 2,3,2,3. Phrases now also join on onset spacing (<=1.1 beat), giving 0,1,2,3. Cost: more hand resets are reachable in long lines (synthetic `reversed`/`large_hops` rose, `collapsed` stays 0). Check slower melodic parts by eye.
- Chord-rate cap (`conform_expert_to_reference`): identical chords (same pitch and shape) are now demoted together, not one at a time by confidence, so a repeated chord stays consistent. Check strummed songs for chords flickering between chord and single gem.
- Whole-pipeline sweep on random melodies (70-170 BPM, mixed 16th/8th/quarter/half notes, 5 seeds): exact pitch 88-99% of attacks, 0 same-color pitch changes on any difficulty, healthy lane spread. Misses were only the note at t=0 (Basic Pitch emits nothing there) and 16ths at 170 BPM (density limit). Synthetic only.
- The cleaned stem is now `guitar-clean-<settings>.wav`, rebuilt if older than the raw stem, so earlier cleaned files with old settings are never silently reused.

## Things to check by ear/eye (needs real songs)

- Mr. Brightside: compare `guitar-clean-*.wav` vs the raw guitar stem in the review Audio selector. Is the blended/faded sound reduced? Any watery/thin artifacts? Are vocal/drum notes gone from the guitar chart?
- Same song regenerated with `--raw-guitar-stem` for a before/after chart comparison.
- Fast runs: do rapid different pitches get different colors? Do long rising runs reset to green sensibly?
- Repeated riffs, including transposed ones: same colors?
- Hard and Medium: any same-color runs on changing pitches?
- Electronic songs: tempo map follows acceleration (Less Talk More Rokk).

## Known limits

- Reversed lane movement increased on purpose (hand resets).
- Constant-tempo grids measured ~15 ms late on synthetic bursts; not corrected.
- Bass/keys do not use the phrase pass.
- Benchmark: `from rb3charts.synthetic_bench import run_benchmark; run_benchmark()`.
