# Pretrained charting evaluation checkpoint

## Decision and scope

The existing `pipeline.generate` path remains the default and is unchanged.
This checkpoint adds a versioned event interchange, an audio2chart token
normalizer, a source-hash-checked review MIDI/preview import, and rights-gated
event comparison. It does **not** run either pretrained model, add a selectable
generator backend, or add a hybrid pass. No real-song quality claim or model
winner is supported by the evidence below.

## Evidence matrix

| Evidence | audio2chart | CloneCharter |
| --- | --- | --- |
| Source inspected | `3podi/audio2chart` at `b6abaafecd3d39a11793ca64538431263dcd99ad`; paths below were retrieved from raw GitHub URLs pinned to that revision. | The user-supplied model-card URL and claimed implementation link could not be fetched from this environment. |
| Official weights | `generate.py` names `3podi/charter-v1.0-40-M-best-acc`; the default loader downloads `config.json` and `pytorch_model.bin` through Hugging Face Hub. No model file was downloaded and no HF revision or SHA-256 was verified. | Unknown: model card, file tree, weight revision and checksum were not reachable. |
| Inference implementation | Present in upstream `inference/engine.py`; not executed here. It loads Encodec separately and generates 30-second chunks in one batch. | Unknown: implementation link, loader and inference code could not be verified. |
| Lanes / difficulty | `chart/tokenizer.py` maps combinations of five lanes (0–4) and a separate open-note token. `generate.py` writes `[ExpertSingle]` only. Open notes are rejected by this adapter. | The card claims Guitar/Bass/Drums and Easy through Expert+++, but those claims and token semantics remain unverified. |
| Timing / durations | Grid timestamps use checkpoint `grid_ms`. The upstream CLI's default BPM 200 and resolution 480 are chart serialization parameters, not audio tempo analysis. `convert_notes_to_ticks` emits zero sustain ticks, so offsets/confidence are recorded unknown; no sustains are invented. | Unknown: tempo, timing grid, event duration, difficulty and conversion semantics could not be inspected. |
| Input / output | Upstream loads mono audio at 24 kHz and rejects clips shorter than 30 seconds. Its script accepts common audio paths and writes a Clone Hero `.chart` file. This adapter accepts a WAV plus returned token sequences and imports a normalized bundle into a review-only RB3 MIDI/preview. | Unknown: card claims WAV/model inference and chart output, but exact implementation behavior is unverified. |
| Local runtime / hardware | Environment has no `torch`, `transformers` or Hugging Face client. No CUDA/GPU inference was attempted. | No model runtime or hardware execution was attempted. |
| License / commercial use | The pinned GitHub `LICENSE` path returned 404 and the HF model card was inaccessible. Code, checkpoint and Encodec licensing/redistribution/commercial permissions are therefore unknown. `Charter.from_pretrained` calls `torch.load` without an explicit `weights_only` argument and does not pin the HF revision. Do not execute/download or make commercial-use claims until licensing and safe immutable loading are addressed. | Unknown. A model-card tag alone would not establish rights in weights, MERT, audio or charts. |
| Dataset / training rights | Not used. | The problem statement describes `CloneHeroDatasetCharts` as MIT-tagged with about 43.7k rows, but that card/files/license could not be independently fetched. Audio/chart rights and commercial use remain unknown. No dataset was downloaded or used. |
| Dependencies | Upstream `requirements.txt` pins Torch 2.9.1, Transformers 4.57.1, NumPy 2.3.4 and other research dependencies; this project pins NumPy 1.26.4 and Torch 2.8.0 in its tested setup. They were not installed into the project environment. | Unknown. No install was attempted. |

The requested external references are [audio2chart's official weight card](https://huggingface.co/3podi/charter-v1.0-40-M-best-acc),
[CloneCharter](https://huggingface.co/thejorseman/CloneCharter), its
[claimed implementation](https://github.com/thejorseman/auto-charter), and the
[CloneHeroDatasetCharts card](https://huggingface.co/datasets/thejorseman/CloneHeroDatasetCharts).
The HF host could not be resolved here; GitHub API requests timed out or returned
403, and direct raw requests for the claimed auto-charter `main` README/LICENSE
returned 404. These failures leave code, model files, revisions and licenses
unverified; claims in the issue are not treated as independently confirmed.

### audio2chart source references

All code links below point to the inspected immutable source revision:

- [`generate.py`](https://github.com/3podi/audio2chart/blob/b6abaafecd3d39a11793ca64538431263dcd99ad/generate.py): model ID, sampling defaults (`temperature=0.5`, `top_k=32`), `bpm=200`, `resolution=480`, token concatenation and `.chart` output.
- [`inference/engine.py`](https://github.com/3podi/audio2chart/blob/b6abaafecd3d39a11793ca64538431263dcd99ad/inference/engine.py): Hugging Face downloads, Encodec loader, audio resampling, 30-second minimum/chunks and generated-token slicing.
- [`chart/tokenizer.py`](https://github.com/3podi/audio2chart/blob/b6abaafecd3d39a11793ca64538431263dcd99ad/chart/tokenizer.py): lane-combination mapping and open-note handling.
- [`chart/time_conversion.py`](https://github.com/3podi/audio2chart/blob/b6abaafecd3d39a11793ca64538431263dcd99ad/chart/time_conversion.py): grid-to-tick conversion; generated note durations are zero.
- [`chart/chart_writer.py`](https://github.com/3podi/audio2chart/blob/b6abaafecd3d39a11793ca64538431263dcd99ad/chart/chart_writer.py): ExpertSingle `.chart` serialization.
- [`requirements.txt`](https://github.com/3podi/audio2chart/blob/b6abaafecd3d39a11793ca64538431263dcd99ad/requirements.txt): upstream runtime pins.

The upstream chunk algorithm moves its last 30-second window backward for
non-multiple durations, then appends the entire last window. Naively concatenating
tokens repeats overlapping audio. `audio2chart_tokens_to_bundle` reconstructs
the exact upstream starts and drops tokens whose source-time positions were
already covered. Its tests use synthetic token arrays only; they do not validate
model quality or model inference.

## Normalized bundle and commands

Schema version 1 records upstream/model/weight identifiers and known hashes,
source WAV hash/rate/duration, device, seed, sampling parameters, runtime,
source-time grid/offset, instrument/difficulty, optional shared tempo provenance,
and stable event IDs. Unknown model revision, weight hash, confidence and note
offset remain explicit `null` values. Unsupported open notes or token IDs, unknown
modifiers, invalid times/lanes, duplicate or unsorted attacks and malformed tempo
maps fail instead of being silently remapped.

`normalize-audio2chart` consumes token sequences saved by an external inference
runner; it does **not** execute or download audio2chart:

```powershell
.\.venv\Scripts\python.exe -m rb3charts.cli normalize-audio2chart `
  'C:\authorized\song.wav' `
  'C:\local\audio2chart-token-sequences.json' `
  '.research\song-audio2chart-events.json' `
  --pad-token-id 34 --grid-ms 20 `
  --temperature 0.5 --top-k 32 `
  --tempo-file '.research\shared-tempo.json'
```

Read `pad_token_id` and `grid_ms` from the loaded checkpoint config; the example
values are not defaults guaranteed for every checkpoint. The token file must be
a JSON list of the per-chunk integer sequences returned by the upstream engine.
Hugging Face weight revisions/hashes are not known here; do not fill them with
guesses.

When the separately executed upstream script has written its documented
`notes.chart`, `normalize-audio2chart-chart` can parse that file directly without
requiring token extraction from upstream internals:

```powershell
.\.venv\Scripts\python.exe -m rb3charts.cli normalize-audio2chart-chart `
  'C:\authorized\song.wav' `
  'C:\local\audio2chart-run\notes.chart' `
  '.research\song-audio2chart-events.json' `
  --tempo-file '.research\shared-tempo.json'
```

It converts the chart's ticks through its own SyncTrack BPM and resolution to
source seconds, groups same-tick lanes as chords, preserves any explicitly
serialized sustain, and rejects nonzero chart offsets, unsupported open notes,
HOPO/tap markers and inconsistent chord sustains. The upstream writer's BPM is
serialization metadata; the separate `--tempo-file` remains the shared audio
tempo map for comparison and RB3 MIDI output.

To produce an **uncompiled review draft**, the bundle must describe canonical
Expert events and include a shared source-time tempo map:

```powershell
.\.venv\Scripts\python.exe -m rb3charts.cli import-pretrained-events `
  'C:\authorized\song.wav' `
  '.research\song-audio2chart-events.json' `
  'output\Pretrained_Song_Review'
```

This importer verifies the exact WAV SHA-256, writes MIDI through the existing
writer, validates its structure, and creates the existing visual and audible
onset preview. It refuses to overwrite an existing output. The existing MIDI
writer needs a positive note duration; if a backend supplies no duration, one
MIDI tick is used strictly as serialization for a tap and metadata explicitly
states that this is **not** a predicted sustain. No model confidence is
fabricated: the existing `Note` compatibility field receives a documented zero
sentinel, while the normalized event retains `null`.

For real held-out comparisons, prepare a local manifest binding rights claims
and content hashes; the tool refuses a missing attestation, non-held-out split,
or mismatched source/reference hash. Do not include a reference chart as an
inference input.

```json
{
  "manifest_version": 1,
  "evaluation_split": "held_out",
  "rights_attested": true,
  "audio_rights": "Describe the permission that covers this audio and evaluation.",
  "reference_chart_rights": "Describe the permission that covers this reference.",
  "audio_sha256": "64-character SHA-256 of the source WAV",
  "reference_bundle_sha256": "64-character SHA-256 of the reference event JSON"
}
```

```powershell
.\.venv\Scripts\python.exe -m rb3charts.cli compare-pretrained-events `
  '.research\authorized-reference-events.json' `
  '.research\audio2chart-events.json' '.research\clonecharter-events.json' `
  --rights-manifest '.research\held-out-rights.json' `
  --output '.research\pretrained-comparison.json' --tolerance-ms 50
```

Matching first maximizes the number of one-to-one onset matches within tolerance,
then minimizes their total absolute onset error. The report includes onset
precision/recall/F1, timing error, coverage, exact lane-set agreement on matched
events, known-duration error when both sides provide offsets, density and
descriptive lane transitions. It reports unsupported duration and
playability/human-quality metrics as unavailable, not as perfect. It does not
calculate independent audio contour or human-authored quality, and is not a
model winner score.

Local PowerShell helpers are:

```powershell
.\scripts\Test-PretrainedEnvironment.ps1
.\scripts\Normalize-Audio2ChartChart.ps1 -SourceWav 'C:\authorized\song.wav' `
  -ChartFile 'C:\local\audio2chart-run\notes.chart' `
  -TempoFile '.research\shared-tempo.json' -OutputBundle '.research\song-events.json'
.\scripts\Import-PretrainedEvents.ps1 -SourceWav 'C:\authorized\song.wav' `
  -EventBundle '.research\song-events.json' -OutputFolder 'output\Song_Review'
.\scripts\Compare-PretrainedEvents.ps1 -ReferenceBundle '.research\reference.json' `
  -PredictionBundles '.research\audio2chart.json','.research\clonecharter.json' `
  -RightsManifest '.research\rights.json' -OutputReport '.research\comparison.json'
```

The diagnostic reports the machine running PowerShell; it cannot report the
agent's or another machine's GPU. There are intentionally no model-install or
weight-download scripts at this checkpoint: neither model's immutable weight
source and permissions have been independently verified. The import and compare
helpers only operate on already-created local JSON bundles.

## Baseline and execution record

The actual checkout at baseline was `4aef7cc536cbdae60f2414f6f101c872294ebdf7`,
whose parent is the prompt's `008e298891ac3ba1271974d388eeb539a5f68476`.
The task branch was already `copilot/codexpretrained-charting-evaluation`;
it was retained rather than switching branches.

- Baseline Python: 3.12.3. The repository environment was absent, so the suite
  ran in an isolated `/tmp` virtual environment with the repository's pinned
  existing runtime dependencies; no project environment was changed.
- Baseline `pytest -q`: 108 passed, 2 warnings.
- Baseline `rb3charts.cli --help`: passed.
- Baseline synthetic cached replay: `draft_ready`, structural validation passed,
  compile disabled. It used the existing `tests/test_regression_replay.py`
  fixture (8-second silent WAV and scripted guitar event data), not model output.
- Full output tree hashes and regression metrics are preserved locally, ignored
  by Git, in `.research/pretrained-charting-evaluation/baseline/`. Source WAV
  SHA-256: `1918846628d4d25823f9b32c75503e1167c680c3cc0821dd950d41b9c8fdd0c7`.
- Current full `pytest -q`: 134 passed, 2 existing dependency deprecation
  warnings; this includes 26 focused tests. `compileall`, main CLI help and new
  CLI help passed.
- Focused tests cover synthetic chunks at 30, 30.02, 45, 60 and 61 seconds,
  overlap trimming, 20/40 ms grids, `.chart` tempo/chord/sustain parsing,
  unsupported token/open-note failures, schema validation, one-to-one metrics,
  held-out rights checks, source-hash mismatch and no-overwrite behavior.
- Both CLI paths were manually smoke-tested using only synthetic fixtures:
  token JSON and an `.chart` file each normalized to events and imported to a
  structurally valid review MIDI/preview. A deliberately identical synthetic
  prediction/reference yielded F1 1.0; this is only plumbing verification, not
  an inference or musical result.
- PowerShell 7.6.6 ran the GPU/runtime diagnostic and normalize/import/compare
  wrappers. This runner reports Linux, no `torch`, no `nvidia-smi` and no CUDA;
  it is not the user's Windows/GPU environment.

The network in this environment could not resolve `huggingface.co`; GitHub API
requests timed out or returned 403. No CloneCharter files, model weights, audio
or dataset were downloaded. No commercial or held-out real-song evaluation,
GPU run, Onyx compilation, human listen, or Xbox playtest was performed. The
CloneCharter adapter, its dependencies/download scripts, GUI selection and any
hybrid generation pass are deferred until its code, immutable weights, licensing
and token/time semantics can be independently verified. The existing generator
remains the only automatic generation backend.
