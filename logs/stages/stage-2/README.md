# stage-2 — Offline warm-up data collection

Pipeline stage **2**: prepare the offline training data the initialized student is taught with (AGENTS.md 4.4).

Generated from [`../index.json`](../index.json). It records the
evidence behind every claim below; a path in a code span is a
pointer to its canonical owner, which is where the revisions,
counts, hashes and configs live. Stage membership is read from
repository facts — a config's stage field, the
`configs/stage<N>/` directory it lives in, a declared `stage_id`,
or an experiment's own subject and product — never guessed from a
name.

## Purpose

Prepare the offline data the initialized student is taught with. **The product of this stage is data.** It trains nothing, and no training run is invented to make the log tree look symmetrical.

## Inputs

What this stage receives from the pipeline before it.

* public instruction, QA, tool-calling, code, math and long-context datasets, each pinned by revision in the manifest
* no teacher-generated targets in v0 or v1 — the manifest records that explicitly; the teacher-generated corpus is a separate pilot

## Data

* `data/stage2/stage2_offline_v0.manifest.json` — mixture v0: the first grouped build, from public sources only
* `data/stage2_v1/stage2_offline_v1.manifest.json` — mixture v1: the approved ~4.5x train scale-up for data-limited Stage-3 recovery. The current mixture
* `data/stage3_pilot/manifest.json` — the teacher-generated pilot: a control/treatment 2x2 over one shared accepted prompt subset, which is what Stage 3 actually trains on today

Groups, as the manifests name them:

| group | what it is |
| --- | --- |
| `instruction` | general instruction following |
| `rag_evidence` | answering from supplied evidence |
| `multihop_qa` | questions needing more than one hop |
| `tool_calling` | structured / function-call outputs |
| `refusal_uncertainty` | unanswerable and under-specified prompts |
| `code_math` | programs and grade-school to competition math |
| `short_realtime` | short conversational turns |
| `long_context` | long documents |

Every group is split three ways:

* **train** — what recovery optimizes on
* **val** — held out from train, for in-training validation
* **calib** — a small per-group sample reserved for calibration and quantization-sensitivity work, never trained on

Source families. The manifests own each one's dataset
id, revision, license and sample count:

* OpenAssistant oasst2, smol-smoltalk, everyday-conversations (instruction, short_realtime)
* SQuAD v2 (rag_evidence, refusal_uncertainty), HotpotQA (multihop_qa)
* Glaive function-calling v2 and xLAM function-calling 60k (tool_calling)
* GSM8K, OpenMathInstruct-2, Magicoder-OSS-Instruct (code_math)
* FineWeb-Edu (long_context)

## Outputs

* `data/stage2_v1` — the v1 mixture on disk: `train/`, `val/` and `calib/` per group. Untracked; rebuildable from the manifest and its builder

## Pipeline activity

The stage's own work, and measurements supporting it. Not
studies *of* it.

| what | kind | logs | status |
| --- | --- | --- | --- |
| Offline warm-up / distillation mixtures v0 and v1 | `pipeline-activity` | none | complete — v0 and v1 built |

## Experiments

**None.** No experiment has taken this stage as its subject,
so there are no experiment-run logs. That is not the same as
an empty stage: the material above is this stage's real
output, and no run, experiment or manifest is invented to
fill the gap. If experiment logs appear later, they get
directories here.

## Runs

**0** run(s) are registered for this stage's
experiments. An experiment's plans, analyses, results, history,
validations and runs are all inside its own directory; the
canonical run list, across every stage, is
[`../../index.json`](../../index.json).

## Canonical configs

`configs/` is the source of truth. A run's manifest records
the config path and hash it ran under.

* `data/stage2_v1/stage2_offline_v1.manifest.json`

## Canonical data and artifact manifests

A dataset manifest lives beside the data it describes, and
an artifact lives outside git with its manifest. Neither is
copied here.

* `data/stage2/stage2_offline_v0.manifest.json`
* `data/stage2_v1/stage2_offline_v1.manifest.json`
* `data/stage3_pilot/manifest.json`
* `data/stage2_v1`

## Decisions

* logs/budget/decisions.md — 2026-07-21: Stage 2 offline mixture v0, grouped public sources, no teacher-generated data
* logs/budget/decisions.md — 2026-07-21: offline data rendering, assistant-span masking, empty-think targets
* logs/budget/decisions.md — 2026-07-26: Stage 2 mixture v1, approved 4x scale-up, carry-plus-fresh design, named val sets
* logs/budget/decisions.md — 2026-08-01: difficulty-aware mixture replaces equal four-way balance

## Current status

**Complete — v0 and v1 built, and the teacher-target pilot exists.** Every group's dataset revision, license, sample count and content hash is owned by the manifests above; the split rule and the deduplication and holdout-exclusion rules are recorded there too. Which groups a given recipe should actually train on is a capability-scope decision (AGENTS.md P3, P10.1), not a completeness checklist — a group existing here does not mean it enters a mixture.

Money, authorizations and what is running right now are not
here: they belong to [`../../budget/`](../../budget/) and
[`../../state/`](../../state/).
