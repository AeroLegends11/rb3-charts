# Less Talk More Rokk — reference cross-check

Reviewed 2026-09-18. Target: `output/Freezepop_Less_Talk_More_Rokk_20260918_121409_694063`, Expert difficulty. No chart-generation code or existing chart was changed during this comparison.

The current chart loses the recognizable lead phrase. This is not merely a preference for different fret colors: meaningful attacks disappear before mapping, different pitches collapse to one fret, and the electronic playability gate removes additional notes. The constant-tempo assumption also contradicts the recording's accelerating introduction.

## Reference and alignment

- Guitar: [SlowHero's rendering of the Guitar Hero II Xbox 360 Expert chart](https://slowhero.moto-coda.org/gh2-x360/expert/lesstalkmorerokk.blank.png). This is a community-hosted rendering of the game chart, not a newly authored custom. It is explicitly the **GH2** guitar reference; an original RB guitar MIDI was not obtained, so this report does not claim a full exact RB guitar comparison.
- Tempo and drums: [Rock Band 3 update MIDI preserved in Onyx](https://github.com/mtolly/onyx/blob/master/haskell/resources/rb3-updates/lesstalkmorerokk/lesstalkmorerokk_update.mid). This file contains a tempo track and PART DRUMS, not guitar or bass.
- Corroborating structure: [Rock Band Expert per-measure scoring data](https://rockband.yajags.com/pathstats/lesstalkmorerokk_expert.txt). Scores are not treated as direct note counts.

The median offset of the first 28 reference kick attacks against the generated drum attacks is **6.626698 seconds** (reference time minus source-audio time). The maximum residual for those pairs is **0.228 ms**, supporting the recording alignment without time stretching. This is an alignment check, not an overall accuracy claim.

GH2's depicted measures correspond to the RB timeline with two additional opening measures: GH2 measure 9 corresponds to RB measure 11. The depicted tempo progression and opening structure support this alignment. Guitar phrase windows below use that correspondence and the RB tempo map; there is no original GH2 MIDI in this audit. The two displayed guitar sequences were read from the chart image.

The local review adds **3.364062 seconds** of lead-in. Its status line shows both player and song time; use song time for the source timestamps below.

## Concrete guitar examples

G/R/Y/B/O = Green/Red/Yellow/Blue/Orange. Sequences show attacks, not sustain lengths.

| Source/song time | Player time | GH2 Expert reference | Current Expert | Events entering mapper |
|---|---|---|---|---|
| 25.420–28.299 s | 28.784–31.663 s | 15: `RGRYBGRYBGRYBOB` | 6: `RRRRRB` | 8 |
| 28.299–31.102 s | 31.663–34.466 s | 13: `RGRYBGRYBGRYB` | 7: `RRRRRRR` | 8 |

These are GH2 measures 9–10, during the accelerating lead riff. The reference has a recognizable repeated descending shape with a phrase-ending variation. Our chart has largely lost both its rhythm and contour.

In the first window, the surviving pitches are MIDI 60, 61, 58, 59, 60, 64, mapped to R/R/R/R/R/B. The contributing sources jump between guitar, piano and other stems. That proves a mapping collapse for different detected pitches; source switching is also a plausible contributor to an incoherent voice, though it does not by itself prove each selected pitch is wrong.

Only eight events enter mapping in each window, compared with 15 and 13 reference attacks. Detection and/or role selection has already lost substantial phrase detail. Later mapping/playability reduces those eight to six and seven. Merely lifting the density cap cannot recover the entire phrase.

## Findings and causes

1. **The tempo map is wrong during the introduction.** The generated map is constant approximately 124.998 BPM. The RB reference starts the audible passage at approximately 70 BPM and reaches 125 BPM at source time 70.065 s. The detected opening kicks independently show the acceleration: intervals start around 0.85 s and shrink. Electronic mode currently requests a constant studio grid. That affects beat-based filtering, reductions and measure display even though retained note attacks keep their source timing.

2. **The Expert density cap discards legitimate musical vocabulary.** The guitar playability gate removes **389 of 1,293** mapped attacks, leaving **904** (30.1% removed). Electronic mode permits only two attacks per quarter note, while the reference uses recurring sixteenth-note runs. The detector/role stage also uses minimum-spacing and confidence filters, so this is not the only source of missing attacks. The correct remedy is to recognize supported fast phrases, not to keep all candidates indiscriminately.

3. **Lane mapping compresses the riff.** The electronic mapping assigns the lower pitch range to only G/R, a central pitch to Y, and the upper range to B/O. Several distinct pitches in these phrases all become Red. A phrase-local pitch vocabulary should preserve ordering, repeated notes, and an established fingering across repetitions. Whole-song pitch range alone is insufficient.

4. **Chord and sustain interpretation differs substantially.** The generated Expert guitar has **zero chords**, whereas the reference starts with a long G/R/Y chord and later contains sustained multi-note shapes. The electronic lead selector selects one pitch per onset group, making those chords unavailable to the mapper. Our longest guitar note is only **0.589 s**. Separately, **898/904 notes last at least 90 ms**, the electronic sustain threshold, and the median duration is about 114 ms. This is mostly tiny tails, not long musical holds: only **202** exceed the preview's 150 ms drawing threshold. The earlier playability hold count should therefore not be read as 898 visible long sustains.

5. **Source-role choices need phrase continuity.** Strongest-event selection across guitar/piano/other stems can choose unrelated voices from one attack to the next. The measured example changes stems repeatedly. A coherent multi-note lead hypothesis should be scored before lane assignment, while retaining justified lead changes and real chords.

## Drum cross-check

The RB3 update has **1,413 distinct Expert attack times**, versus **1,050** generated attack times. With the single alignment offset above and an order-preserving one-to-one 50 ms matching window:

- 881 attacks match in time.
- 532 reference attacks have no generated match.
- 169 generated attacks have no reference match.
- Median absolute timing difference among matches is approximately 0.40 ms.

This comparison ignores pad identity, simultaneous gem count, cymbal/tom status and fills. Unmatched attacks are disagreements with this reference, not independently verified audio mistakes. It suggests that retained timing is often close while event coverage and instrument classification still need work. Bass was not compared against a complete original bass chart in this audit.

## Implementation priorities informed by this benchmark

1. Detect changing tempo before choosing the electronic constant-tempo path; retain a variable map for supported acceleration.
2. Evaluate a lead voice over a phrase, with pitch-specific attack evidence and consistent source selection. Audit missing attacks before tweaking fret patterns.
3. Replace unconditional Expert density limits with evidence-based acceptance of recurring fast phrases.
4. Use phrase-local, contour-preserving fingering and repeated-phrase consistency for electronic parts too.
5. Distinguish clipped articulations from meaningful holds and allow well-supported electronic chords.
6. Re-run this benchmark and several unrelated songs, reserving songs that were not used to tune rules. Improvements should not depend on this title or hard-coded timestamps.

The existing event-evidence addition supplies useful diagnostics, but this cross-check demonstrates that it has not yet solved lead interpretation or human-style authoring.

Machine-readable measurements and sample notes are saved in the reviewed run's `analysis/reference-crosscheck.json`. Retrieved reference material is retained under `.research/less-talk-more-rokk/`.
