# Authorized human-chart pilot

This pilot tests whether high-quality human authoring can improve the musical
core without replacing the working generation, review, validation, and export
pipeline.

No community chart or paired song audio enters training merely because it is
downloadable. Each pair needs explicit, auditable model-training rights for both
the audio and the chart. Official charts and unlicensed commercial recordings
remain evaluation-only.

The current local manifest contains 36 validated pairs from 14 artists and six
chart authors, totaling 19,411 Expert attacks. It combines the original Brad
Sucks/Zep Hurme seed, 24 Fervi-authored GuitarMaster songs, four
Carlos Viola Iborra/Sectoid songs, and four El Senator/Muldjord songs. The
Sectoid MIT and Muldjord CC BY-SA 3.0 notices explicitly cover the songs,
performances, and fret files. All eight added pairs passed a shipped-stem
synchronization audit globally and over early/middle/late regions before
admission. Songs that would leak the held-out Persephone artist/chart-author
split, and a chart without explicit fretter attribution, are excluded. Audio,
charts, permissions, hashes, synchronization reports, and converted artifacts
remain local under `.research/`.

The added creator groups were assigned before model evaluation: Sectoid remains
train-only and Muldjord validation-only; Persephone remains the untouched test.
Muldjord-derived model distribution requires a separate ShareAlike assessment.
Local training eligibility does not by itself authorize publishing a checkpoint.

For near-1:1 timing evaluation, the synchronization audit additionally requires
global alignment within 25 ms, every early/middle/late regional offset within
50 ms, and at least 40 percent of chart attacks within 25 ms of an isolated
guitar onset. A subsequent diagnostic found that Persephone's declared
zero-offset pair aligns best near -46 ms, so its existing sub-25-ms timing
results are not a valid localization gate pending independently corrected
synchronization provenance. Its artist- and author-disjoint assignment is
unchanged, and coarse 50 ms attack-selection comparisons remain evaluation-only.

## Stages

1. Collect a small opt-in set from chart authors and audio rights holders.
2. Validate provenance, hashes, synchronization, Expert guitar coverage, and
   artist/chart-author-disjoint splits.
3. Convert authorized Expert guitar into lane-independent phrase structure plus
   a separate five-lane authoring target.
4. Train a Stage B authoring pilot without changing normal generation.
5. Compare held-out artists and chart authors, then use the frozen 14-song
   commercial collection only for post-generation evaluation.

## Required manifest fields

The ordinary dataset fields and hashes remain required. A chart-pair manifest
also uses:

```json
{
  "schema_version": 1,
  "dataset": {
    "id": "authorized-chart-pilot",
    "version": "1",
    "purpose": "authorized_human_chart_pairs",
    "source_url": "https://collection.example/",
    "audio_license": "Per-item permission",
    "annotation_license": "Per-item permission",
    "commercial_use": "unknown",
    "attribution": "Per-item rights holders"
  },
  "items": [
    {
      "song_id": "stable-song-id",
      "artist_id": "stable-artist-id",
      "chart_author_id": "stable-chart-author-id",
      "split": "train",
      "audio": "audio/song.wav",
      "audio_sha256": "64-lowercase-hex-characters",
      "annotations": "charts/notes.mid",
      "annotations_sha256": "64-lowercase-hex-characters",
      "chart_format": "rock_band_midi",
      "synchronization": {
        "method": "manual_alignment",
        "audio_offset_seconds": 0.0,
        "verified": true
      },
      "audio_rights": {
        "basis": "explicit_permission",
        "license": "Pilot training permission",
        "attribution": "Audio rights holder",
        "training_allowed": true,
        "redistribution_allowed": false,
        "evidence": "permissions/song-audio.txt",
        "evidence_sha256": "64-lowercase-hex-characters"
      },
      "chart_rights": {
        "basis": "explicit_permission",
        "license": "Pilot training permission",
        "attribution": "Chart author",
        "training_allowed": true,
        "redistribution_allowed": false,
        "evidence": "permissions/song-chart.txt",
        "evidence_sha256": "64-lowercase-hex-characters"
      }
    }
  ]
}
```

Rights bases are `creator_owned`, `explicit_permission`, or
`compatible_license`. Synchronization methods are `manual_alignment`,
`shared_render`, or `embedded_audio_offset`.
`chart_format` may be `rock_band_midi`, `frets_on_fire_midi`, or
`clone_hero_chart`; legacy MIDI is accepted only when exactly one track contains
Expert five-lane gems, and `.chart` input must contain resolution, tempo events,
and an `ExpertSingle` section.

Permission evidence stays local with the dataset. The validator records its path
and hash but does not copy its contents into model artifacts.

## Validation

```powershell
.\.venv\Scripts\python.exe -m rb3charts.cli validate-chart-pairs `
  'C:\Path\To\ChartPairs\manifest.json' --verify-hashes `
  --source-agreement-report 'C:\Path\To\source-agreement.json'
```

The command rejects missing training permission, malformed or mismatched hashes,
duplicate audio/charts, artist or chart-author split leakage, unverified
synchronization, missing `PART GUITAR`, fewer than 32 Expert guitar gems, fewer
than three used lanes, and chart timing outside the paired audio.

The source-agreement report is optional and never changes chart-pair
eligibility or generation. When supplied, its report hash, paired-manifest hash,
published release checksum, tolerance and per-split shared/best-source metrics
are exposed under `paired_source_agreement`. Status is `available` only when
lineage validates, `unavailable` when no report is supplied, and `invalid` for
missing or mismatched lineage. The current 36 authorized chart pairs have zero
independent paired captures; GuitarSet and Guitar-TECHS results therefore appear
only as external rights-cleared calibration with target coverage unavailable.

After validation, create deterministic Stage B authoring examples:

```powershell
.\.venv\Scripts\python.exe -m rb3charts.cli convert-chart-pairs `
  'C:\Path\To\ChartPairs\manifest.json' `
  'C:\Path\To\AuthoringExamples'
```

Conversion always verifies the audio, chart, and permission-evidence hashes. It
writes aligned lane-independent attack/polyphony/sustain conditioning beside
the authorized Expert lane targets, with artist and chart-author split keys.
The conditioning is explicitly marked as a chart-derived training proxy. At
held-out inference it must be replaced by independently predicted audio
structure; chart targets are never available to the generator.

## Held-out comparison

Generate a song from audio first, then compare the completed output with its
authorized human chart:

```powershell
.\.venv\Scripts\python.exe -m rb3charts.cli compare-human-chart `
  'C:\Path\To\ChartPairs\manifest.json' song-id `
  'C:\Path\To\GeneratedOutput' `
  --source-agreement-report 'C:\Path\To\source-agreement.json'
```

The command revalidates the complete manifest and writes
`analysis/human-chart-comparison.json` plus a deterministic Markdown companion.
It reports attack precision/recall and timing error, chord and sustain agreement,
lane movement, four-second density, rest-delimited phrase boundaries,
repeated-motif lane shapes, and Expert-to-lower-difficulty retention. Optional
paired-source evidence is clearly labeled external calibration unless the
generated-song audio hash has direct covered lineage. Unavailable or invalid
lineage never renders agreement metrics. Human notes are loaded only after
generation and must never become runtime inputs.
