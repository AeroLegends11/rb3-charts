# Working on RB3 Charts

## Run and validate

Windows, Python 3.11/3.12. Work from the repository root. `Setup.ps1` installs
the audio/model dependencies and external Onyx toolchain; use `-CpuOnly` on a
machine without an NVIDIA GPU. Do not reinstall a working environment routinely.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m rb3charts.cli --help
.\.venv\Scripts\python.exe -m rb3charts.cli gui
```

`Launch.cmd` opens the GUI. Tests synthesize their own small fixtures and do not
need commercial songs, model downloads or Onyx. Full generation does need models.
CLI `generate` and `batch` compile by default: pass `--no-compile` for review
drafts. The GUI defaults to review first. Package only a reviewed output when
requested. A passing test suite does not certify musical quality or Xbox playability.

## Architecture and invariants

- `audio.py`, `timing.py`, `transcribe.py`: source audio, beats, model candidates.
- `events.py`: lane-independent evidence, accepted/rejected candidates and tracing.
- `phrases.py`, `chart.py`: voice selection, Expert fingering, playability and
  reductions. `pipeline.py` orders the passes; inspect it before adding a pass.
- `model.py`: source-time notes, shared tempo map, metadata and options.
- `midi.py`, `validate.py`, `export.py`: MIDI, structural checks, Onyx/Magma.
- `assessment.py`, `preview.py`: heuristic triage and listening review.
- `gui.py` and `cli.py` call the engine; keep musical logic out of GUI callbacks.

Keep source IDs and source-audio timing traceable. Never use reference chart
notes as generation inputs or special-case a song title. Preserve supported
variations and rests; repetition is evidence, not permission to invent attacks.
Derive Hard, Medium and Easy from canonical Expert. Keep timing conversion in
the shared TempoMap. Any new timing correction needs measured validation.
Preserve unrelated instruments and existing behavior unless requested otherwise.

## Files and workflow

Edit source, tests, setup and documentation. Audio in `input/`, stems in `cache/`,
runs in `output/`, `Ready for Xbox/`, `.research/`, downloaded `tools/`, and local
reference repositories stay local. Do not manually edit generated charts or
analysis to make a test pass. Do not delete/rewrite existing user review drafts.
`reference/style-profile.json` contains aggregate runtime statistics; full
reference reports and chart files must not be staged.

`main` is the baseline. Use one `codex/<topic>` branch per substantial experiment.
Check `git status` first, preserve user changes, and checkpoint before major edits.
Do not merge to main without the user's instruction. The first GitHub repository
creation/push awaits the user's review, as requested in the shared conversation.
Never force-push or rewrite existing history as routine cleanup.

Change one subsystem at a time. Run relevant tests plus the existing suite for
engine changes. Report before/after metrics with the same inputs/settings and
separate inference, heuristic checks, compiler success, and human playtest results.
Keep changes deterministic; record any random seed and dependency/model versions.
Read `docs/DEVELOPMENT_PLAN.md` before motif or tempo work. Update its progress
when a milestone is actually completed. Keep expensive inference out of ordinary
unit-test iteration; use cached candidate replay for controlled comparisons.
