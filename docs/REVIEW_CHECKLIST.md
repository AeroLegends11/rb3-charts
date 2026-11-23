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
4. **Guitar stem bleed cleanup** (`rb3charts/stem_cleanup.py`, `audio.clean_guitar_stem`): writes `guitar-clean.wav`; raw stem kept as `guitar_raw`; `--raw-guitar-stem` or the GUI checkbox "Suppress vocal/drum/bass bleed in the guitar stem" disables it.
5. **Tempo**: synthetic acceleration tests only; no behavior change.

## Update log

- Added the GUI checkbox for the guitar bleed cleanup (default on).
- The run log now reports how much RMS the cleanup kept (roughly 70-95% is plausible; under 50% suggests over-suppression).
- Basic Pitch ONNX now runs in the test env, so there is an audio-level benchmark (`rb3charts/transcription_bench.py`): synthetic guitar + leaked vocals/drums, raw vs cleaned stem, scored against known notes. Precision rose from ~0.54-0.59 to ~0.77-0.80 with recall unchanged (~0.98).
- Cleanup made stronger from that sweep (weight 2, floor .03, power 2). Synthetic only: on real songs listen for thin/watery guitar; if so, `--raw-guitar-stem` or ease the constants in `stem_cleanup.py`.
- Lead-focus fix (`transcribe._focus_band_guitar_candidates`): it was rewriting a confident true note to the contour's octave (recall 0.98 -> 0.91 after focus). Now octave-equivalent candidates tie-break on confidence and the octave rewrite only applies to low-confidence picks. Cleaned-stem benchmark after focus: precision 0.88 -> 0.96, recall 0.91 -> 0.98. Check real riffs for wrong-octave notes.

## Things to check by ear/eye (needs real songs)

- Mr. Brightside: compare `guitar-clean.wav` vs the raw guitar stem in the review Audio selector. Is the blended/faded sound reduced? Any watery/thin artifacts? Are vocal/drum notes gone from the guitar chart?
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
