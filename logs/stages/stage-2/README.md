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

* `data/stages/stage-2/stage2/stage2_offline_v0.manifest.json` — mixture v0: the first grouped build, from public sources only
* `data/stages/stage-2/stage2_v1/stage2_offline_v1.manifest.json` — mixture v1: the approved ~4.5x train scale-up for data-limited Stage-3 recovery. The current mixture
* `data/stages/stage-3/stage3_pilot/manifest.json` — the teacher-generated pilot: a control/treatment 2x2 over one shared accepted prompt subset, which is what Stage 3 actually trains on today

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

* `data/stages/stage-2/stage2_v1` — the v1 mixture on disk: `train/`, `val/` and `calib/` per group. Untracked; rebuildable from the manifest and its builder

## Pipeline activity

The stage's own work, and measurements supporting it. Not
studies *of* it.

| what | kind | logs | status |
| --- | --- | --- | --- |
| Offline warm-up / distillation mixtures v0 and v1 | `pipeline-activity` | none | complete — v0 and v1 built |

## Experiments

What each one asked of this stage.

| experiment | what it asked | logs | status |
| --- | --- | --- | --- |
| `reasoning_data` — Stage-2 verified teacher-native reasoning data, measured through Stage-3 controlled recovery | Under one fixed initialization and the fixed historical recovery recipe, does verified teacher-native reasoning data lift a 0.6B student off the fluid-reasoning floor where corpus_v2 left it, and how does that scale over a nested 0.86M-5.50M ladder? | [`reasoning_data/`](reasoning_data/) | H1 P1 EXECUTED 2026-10-22 UNDER THE MAINTAINER'S GRANT ($3.33 of $8.02); P1v STOPPED AT FOUR SHORT CELLS; @v2 DOES NOT BUILD ON THE FORMAL LADDER (closed infeasible). THE MAINTAINER CHOSE stage2.reasoning_master@v3 FOR B (2026-10-23): @v1's capability shares, analogy from the project generator alone. P1t PREPARED AT $0, NOT AUTHORIZED: the application's 59 unused tasks (25 analogy + 34 Reasoning Gym composition) frozen in plans/p1t/ under their own launch identity, $1.01 expected / $2.60 hard cap inside the absolute $666.17 cap; the two-session merge refuses a shared task, sets the 0.0 cells' units aside by name and packs only on a full formal build. B's pack not built; @v1 unchanged. Fixed: D4 Top-1, the recovery objective, M1, eval-set-1, the 150k reserve. 34 pods / $58.33 unattributed. No student trained |

## Runs

**1** run(s) are registered for this stage's
experiments. An experiment's plans, analyses, results, history,
validations and runs are all inside its own directory; the
canonical run list, across every stage, is
[`../../index.json`](../../index.json).

## Canonical configs

`configs/` is the source of truth. A run's manifest records
the config path and hash it ran under.

* `configs/stages/stage-2/reasoning_data/corpus_spec.json`
* `data/stages/stage-2/stage2_v1/stage2_offline_v1.manifest.json`

## Canonical data and artifact manifests

A dataset manifest lives beside the data it describes, and
an artifact lives outside git with its manifest. Neither is
copied here.

* `data/stages/stage-2/stage2/stage2_offline_v0.manifest.json`
* `data/stages/stage-2/stage2_v1/stage2_offline_v1.manifest.json`
* `data/stages/stage-3/stage3_pilot/manifest.json`
* `data/stages/stage-2/stage2_v1`

## Decisions

* logs/budget/decisions.md — 2026-07-21: Stage 2 offline mixture v0, grouped public sources, no teacher-generated data
* logs/budget/decisions.md — 2026-07-21: offline data rendering, assistant-span masking, empty-think targets
* logs/budget/decisions.md — 2026-07-26: Stage 2 mixture v1, approved 4x scale-up, carry-plus-fresh design, named val sets
* logs/budget/decisions.md — 2026-08-01: difficulty-aware mixture replaces equal four-way balance
* logs/budget/decisions.md -- 2026-10-21: Stage-2/3 reasoning-data round opened at $0: audit, contracts and a PROPOSED design
* logs/budget/decisions.md -- 2026-10-21: Maintainer: H1 fixes D4 Top-1; budget and durability first; TIP as an H3 candidate
* logs/budget/decisions.md -- 2026-10-22: Independent review of H1 preparation: eight targeted corrections at $0, no paid authorization
* logs/budget/decisions.md -- 2026-10-22: Maintainer: H1 design choices (arm B = stage2.reasoning_master@v1, interactive exploratory); two corrections and the P1 + P1v application at $0
* logs/budget/decisions.md -- 2026-10-22: Maintainer: P1 must be launchable; comparators, completion-time saving and launch wiring at $0, final P1 + P1v application
* logs/budget/decisions.md -- 2026-10-22: Maintainer: two launch-boundary checks before any P1 grant
* logs/budget/decisions.md -- 2026-10-22: Maintainer: H1 P1 + P1v formally authorized; P1 executed ($3.33), P1v stopped at four short cells
* logs/budget/decisions.md -- 2026-10-23: Maintainer: a $0 corpus revision before any top-up; @v2 infeasible, plan 2 applied for
* logs/budget/decisions.md -- 2026-10-23: Maintainer: B's corpus is @v3; P1t prepared at $0 for a separate grant

## Current status

**Complete — v0 and v1 built, and the teacher-target pilot exists.** Every group's dataset revision, license, sample count and content hash is owned by the manifests above; the split rule and the deduplication and holdout-exclusion rules are recorded there too. Which groups a given recipe should actually train on is a capability-scope decision (AGENTS.md P3, P10.1), not a completeness checklist — a group existing here does not mean it enters a mixture.

Money, authorizations and what is running right now are not
here: they belong to [`../../budget/`](../../budget/) and
[`../../state/`](../../state/).
