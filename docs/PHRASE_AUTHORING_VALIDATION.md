# Guitar phrase-authoring validation — 2026-09-18

Implemented sequence-based electronic lead selection, conservative sustained
chord retention, supported fast-riff allowances, and a final guitar phrase pass
for both profiles. The final pass assigns local fingering before difficulties
are reduced and records its changes in `analysis/phrase-authoring.json`.

Validation: **78 tests passed**. Coverage includes local pitch separation,
repeated/transposed phrase vocabulary, immutability of source attacks, large
pitch ranges, long holds, chords, coherent voice selection, weak isolated
register errors, genuine register changes, supported fast figures, and unrelated
bursts. One existing dependency-deprecation warning remains.

## Controlled Less Talk More Rokk replay

New review:
`output/Freezepop_Less_Talk_More_Rokk_Phrase_Review_20260918_192303_058381/review.html`

The replay reused the accepted candidate events, drum notes and tempo map from
`Freezepop_Less_Talk_More_Rokk_20260918_121409_694063`. It recomputed lead selection,
mapping, playability, phrase authoring, lower difficulties, MIDI and preview.
Structural validation passed. No reference notes were used as generation inputs.
The new run records these conditions in `analysis/replay-provenance.json`.

- Original Expert guitar: 904 attacks. New candidate: 1,236 attacks. More notes
  alone do not establish improved musical accuracy.
- Within the new candidate, the final phrase pass reduced changing pitches
  assigned to the same fret from 389 to 30.
- That metric has a tradeoff: opposite-direction pitch/fret transitions rose
  from 0 to 16. Phrase boundaries and five-fret resets can cause these; they
  remain listening/playtest review targets rather than being hidden.
- There are 138 authored phrases; 23 reuse an established phrase fingering.
- 55 notes have tails longer than the preview's 150 ms threshold, compared with
  202 in the earlier draft. Meaningful long-hold detection is still limited by
  the input events.

Source/song time 25.420–28.299 s now has 9 attacks (`YRGYRGRYB`), compared with
the old 6 (`RRRRRB`) and the GH2 reference's 15. Source/song time
28.299–31.102 s now has 8 (`YYBYRYBO`), compared with 7 Reds previously and 13
reference attacks. These differences show that local fingering is no longer
flattened to Red, but they **do not show recovery of the correct full melody**.

The initial accelerating tempo is still represented by the previous constant
grid in this controlled replay. Missing/misidentified lead events, tempo changes,
and phrase-boundary hand resets remain unresolved. This draft is ready for
comparison in the listening preview; it is not certified as matching the official
chart and was not packaged or tested on Xbox.
