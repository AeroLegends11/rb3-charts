# Initial research intake

Prepared 2026-10-31 from repository evidence. This is the initial baseline for
the researcher, not a new chart corpus or an executed training experiment.
Future findings and decisions belong in the `Chart quality research tracker`
issue described in [RECURRING_RESEARCH.md](RECURRING_RESEARCH.md).

## Known evidence

| Source | What is established | Rights / limitations | Next action |
| --- | --- | --- | --- |
| `reference/README.md`, `docs/VALIDATION.md`, `reference/style-profile.json` | The repository documents a 21-song Harmonix-authored, chart-only analysis and an aggregate runtime profile. | Chart publication is not proof of training permission. Full reference reports and chart files remain local; no source audio was downloaded for this analysis. | Inventory the local corpus and its provenance before proposing expanded use. Do not alter the runtime profile as a research-only change. |
| `docs/REGRESSION_HARNESS.md`, `docs/DEVELOPMENT_PLAN.md` | Cached authoring replay, source-ID snapshots and regression metrics exist; generated-versus-human comparisons are planned. | Replay cannot establish improvements to source separation or transcription. Current checks are not calibrated expert-quality scores. | Propose aligned reference comparisons with missed/extra attacks and relative phrase shape, using permission-cleared held-out examples. |
| `docs/RESEARCH.md` | Onyx/Magma, Demucs, Basic Pitch and ADTOF are already part of the documented toolchain, with pinned versions/provenance. | Code and model/data licenses differ. ADTOF material has noncommercial/share-alike restrictions described in the research notes. Recheck upstream terms before new uses. | Research changes with reproducible evidence over these baselines, rather than suggesting their first installation. |
| `docs/DEVELOPMENT_PLAN.md` | Motif ownership, generated-versus-human review and accelerating tempo are identified gaps. Guitar snapshots are implemented; broader snapshot coverage is deferred. | Historical listening notes are not current-run acceptance results. Local source assets are not present in a remote checkout. | Select one subsystem per experiment and preserve baseline replay provenance. |

## Candidate-source admission

Before calling a chart a trusted reference, record its author, original URL,
revision/hash, instrument/difficulty coverage, and the specific evidence of
expert review. A custom's popularity or an official label alone does not settle
whether it is a suitable ground truth for the targeted task.

Record audio-to-chart alignment, lead-in/tempo handling, alternate arrangements,
intentional gameplay simplifications and any known mistakes. Song-level aggregate
statistics do not substitute for event-level alignment.

Verify evaluation permission separately from model-training permission and
redistribution permission. If permission is unknown, retain only the source link
and proposed permission request; do not acquire the corpus automatically.

For a proposed benchmark, declare the training/tuning/held-out split, matching
tolerance and minimum evaluated coverage before comparing runs. Hold out songs
and authors where feasible to avoid measuring memorization of one charter's
style. Start with a small permission-cleared set across contrasting charting
situations, not an indiscriminate download of every available chart.

## First research target

Find a practical route to an independently reviewed Guitar Expert benchmark:
permission-cleared paired recordings and charts, or expert annotations of
recordings the owner is allowed to use. Include repeated riffs, transposed
returns, genuine variations, rests, lead/rhythm overlap and tempo changes.

Deliver a reproducible comparison plan with attack precision/recall, alignment
error, phrase-shape agreement and coverage. Treat fingering alternatives as
potentially valid rather than requiring exact colors everywhere. Describe how
an experienced charter can review a small set of ambiguous phrases so the owner
is not the sole musical-quality judge.
