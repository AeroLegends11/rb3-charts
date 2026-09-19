# First GitHub upload review

Prepared 2026-09-19. No remote repository has been created and nothing has been
pushed. The shared conversation explicitly requests review before either action.

## Proposed destination

- Account: the currently authenticated GitHub account, AeroLegends11.
- Proposed name: `rb3-charts` (availability has not been checked).
- Visibility: **private**, to start with an off-PC backup under your control.
- Local main baseline plus a `codex/motif-diagnostics` development branch.
- Local Git author uses the authenticated account's login and GitHub no-reply
  address; no global Git identity settings are changed.

## Included

About **1.64 MiB uncompressed**, dominated by the app icon. Exact committed file
inventory is available with `git ls-tree -r --name-only main`.

- `rb3charts/`: the existing Python engine, GUI, CLI and preview.
- `tests/`: existing synthetic unit and integration tests.
- `assets/`: application PNG and ICO.
- `docs/`: research, historical validation, small metric summaries, development
  plan and this review; the generated GUI screenshot is excluded.
- `reference/README.md` and `reference/style-profile.json`: aggregate statistics.
- README, AGENTS, Setup.ps1, Launch.cmd, pyproject.toml, tested requirements,
  .gitignore and .github agent instructions/issue template.

## Local only

`input/`, `cache/`, `output/`, `Ready for Xbox/`, `.venv/`, `.research/`, `tools/`,
`logs/`, Python/test/build caches, environment/credential files, downloaded
reference repositories, song audio, MIDI, game packages and model weights.
These files were not deleted or relocated. Ignore rules affect version tracking
only. The existing app still has access to them.

The old 7,765,853-byte reference report contained per-song note rows and absolute
user paths. A full backup remains at
`.research/repository-baseline/style-profile.full.json`. The versioned 27,367-byte
profile preserves all 16 runtime `styles` entries exactly and retains aggregate
retention statistics and counts, omitting per-song data and local source paths.
Both runtime readers consume only `styles`; no engine code was changed.

## Inspection and validation

- Git 2.55.0 and GitHub CLI are installed; GitHub CLI is authenticated.
- Existing Git directory had no commits or remotes; no history was overwritten.
- Baseline: **83 tests passed**, with one existing resampy/pkg_resources
  deprecation warning. Tests exercise real audio/MIDI libraries using synthetic
  fixtures, without downloading models or invoking Onyx/Magma.
- Candidate source/configuration/docs were scanned for common credential formats,
  quoted secret assignments and absolute user paths; no matches remained.
  The app artwork was visually inspected. This is a focused inspection, not a
  guarantee that every possible secret format can be detected.
- No commercial audio/chart/game assets or large generated files are proposed
  for upload. Largest included file is the 1.27 MB application PNG.
- No fresh dependency installation, full-song model inference, compiler build,
  listening session or Xbox playtest was performed during repository setup.

## After approval

Create a private repository without template README/license, attach it as origin,
and push the reviewed main baseline. Push development branches as needed. Do not
merge experimental chart changes automatically. Git history is a code backup;
it does not back up ignored local songs, generated drafts or model caches.

Remote phone access and provider/account configuration remain separate work.
See `DEVELOPMENT_PLAN.md` for the ordered chart-quality milestones.
