# Authorized human-chart pilot

This pilot tests whether high-quality human authoring can improve the musical
core without replacing the working generation, review, validation, and export
pipeline.

No community chart or paired song audio enters training merely because it is
downloadable. Each pair needs explicit, auditable model-training rights for both
the audio and the chart. Official charts and unlicensed commercial recordings
remain evaluation-only.

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

Permission evidence stays local with the dataset. The validator records its path
and hash but does not copy its contents into model artifacts.

## Validation

```powershell
.\.venv\Scripts\python.exe -m rb3charts.cli validate-chart-pairs `
  'C:\Path\To\ChartPairs\manifest.json' --verify-hashes
```

The command rejects missing training permission, malformed or mismatched hashes,
duplicate audio/charts, artist or chart-author split leakage, unverified
synchronization, missing `PART GUITAR`, fewer than 32 Expert guitar gems, fewer
than three used lanes, and chart timing outside the paired audio.
