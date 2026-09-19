# Local reference material

`style-profile.json` is a compact aggregate of the existing 21-song reference
analysis. Its `styles` values are exactly those used before repository setup;
no charting thresholds were changed. The source corpus is described in
`docs/RESEARCH.md` and `docs/VALIDATION.md`.

Per-song note rows and absolute computer paths are omitted from this versioned
profile. The original full report was retained locally under
`.research/repository-baseline/style-profile.full.json`. Downloaded reference
repositories, MIDI, cover art and other game assets remain local and ignored.

The reference analyzer can generate a full local report with:

```powershell
.\.venv\Scripts\python.exe -m rb3charts.cli analyze-reference 'PATH_TO_LOCAL_CHARTS' --output .research/reference-analysis.json
```

Do not overwrite the versioned profile with a full report. Updating the runtime
aggregate is a chart-behavior change: compare metrics and review it separately.

## Chorus Encore comparison catalog

`catalog.json` contains small, versioned identities for official comparison
charts. It records provider IDs and hashes, but contains no note events, audio,
artwork or video. `compare-encore` uses it to retrieve only `notes.mid` byte ranges
from the indexed SNG file. Retrieved MIDIs and their local provenance manifests
are stored in ignored `reference/encore-cache/` and can be rebuilt on any machine
with network access.

An automatic lookup is pinned into the catalog the first time an exact Harmonix
or Neversoft artist/title match is selected. Use `--refresh` only when deliberately
reviewing a newer provider version.
