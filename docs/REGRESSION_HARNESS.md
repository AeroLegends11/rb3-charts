# Regression harness

Milestone 2 adds a deterministic cached replay path that reruns authoring from
saved analysis artifacts.

```powershell
.\.venv\Scripts\python.exe -m rb3charts.cli replay `
  'C:\Path\To\output\Artist_Song_YYYYMMDD_HHMMSS_xxxxxx'
```

Optional flags:

- `--output <folder>`: destination root for the replay run folder.
- `--compile`: compile replay output with Onyx/Magma after validation.

Replay does **not** run source decode, stem separation, or model inference. It
reuses cached analysis and writes a new timestamped output folder.

## Replay labeling

- Replay outputs set `analysis_mode: "cached_replay"` in `metadata/song.json`
  and `result.json`.
- Full generation outputs set `analysis_mode: "complete_audio_reanalysis"`.

## New replay analysis files

- `analysis/replay-provenance.json`
  - replay source path
  - consumed replay inputs + SHA-256 hashes
  - settings/profile payload
  - git revision + dirty flag
  - runtime dependency/model versions
- `analysis/guitar-stage-snapshots.json`
  - source-id keyed snapshots after:
    - map_pitched
    - enforce_playability
    - conform_expert_to_reference (band)
    - refine_band_guitar (band)
    - stabilize_repeated_phrases (band)
    - author_guitar_phrases
    - reductions (per difficulty)
- `analysis/regression-metrics.json`
  - attack retention
  - matched timing error
  - contour collapse/reversal deltas
  - chord/hold rates
  - repeated phrase absolute-lane vs relative-shape agreement
  - motif coverage and false-lock rate

## Coverage and false-lock caveat

Coverage uses an explicit denominator (eligible motif windows), so labeling
nothing cannot appear perfectly consistent. False-lock rate is measured over
detected motif windows and captures motif matches that fail both absolute-lane
and relative-shape checks.

## Corpus comparison

Compare two replay roots after running the same cached songs through baseline
and candidate authoring:

```powershell
.\.venv\Scripts\python.exe -m rb3charts.cli benchmark-replays `
  'C:\Path\To\Baseline_Replays' `
  'C:\Path\To\Candidate_Replays' `
  --output 'output\replay-benchmark.json'
```

Runs are matched by artist and title metadata. The JSON report includes
per-song and macro guitar attack counts, chord/hold rates, peak four-second
density, lane coverage, and collapsed/reversed pitch-motion counts. Unmatched
runs are listed separately. These automated measurements support broad
regression decisions; they do not replace listening review or Xbox playtesting.
