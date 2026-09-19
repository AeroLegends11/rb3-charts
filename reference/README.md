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
