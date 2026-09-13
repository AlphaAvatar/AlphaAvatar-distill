# stage-1 — Projection and structural initialization

Pipeline stage **1**: build a student checkpoint from teacher structure and teacher activations rather than random weights (AGENTS.md 4.3).

Generated from [`../index.json`](../index.json). It records the
evidence behind every claim below; a path in a code span is a
pointer to its canonical owner, which is where the revisions,
counts, hashes and configs live. Stage membership is read from
repository facts — a config's stage field, the
`configs/stage<N>/` directory it lives in, a declared `stage_id`,
or an experiment's own subject and product — never guessed from a
name.

## Purpose

Build a complete student checkpoint from teacher structure and teacher activations instead of random weights, and decide which initialization operators and operator order to use.

## Inputs

What this stage receives from the pipeline before it.

* Stage 0's statistics cache, named by the recipe as `stats_dir`
* the same pinned teacher checkpoint
* a declared student geometry — 1024 hidden, 28 layers, FFN 3072, 16 query and 8 key/value heads, tied embeddings — which the recipe states and this page does not own

## Data

* `data/warmup/holdout_v1.manifest.json` — purpose: "Held-out perplexity eval for Stage 1 gate"; the gate compares the initialized student against a random baseline

## How it runs

Only what helps to understand the stage; the config is the
source of truth.

* embedding and lm-head by activation PCA, attention by sandwich initialization, FFN width by activation-importance selection and depth by a teacher-span map — the recipe is the source of truth for which operators a given checkpoint used
* the initialization is deterministic from its config and seed, which is what lets a regenerated Stage-0 cache be checked by hash rather than by rerunning the comparison

## Outputs

* `artifacts/stage1/qwen3_0p6b_init_v0` — the pinned init checkpoint every Stage-3 recovery run forks from, plus the random baseline saved beside it for comparison
* `artifacts/stage1/state_eval_v1` — the state-evaluation suite the AutoInitializer search scores candidate initializations with
* `artifacts/stage1/e8_contribution_init_v1` — the contribution-guided depth variant, built for E8

## Pipeline activity

The stage's own work, and measurements supporting it. Not
studies *of* it.

| what | kind | logs | status |
| --- | --- | --- | --- |
| PCA / sandwich structural initialization of the 0.6B student | `pipeline-activity` | none | complete |
| Causal-depth runtime measurement — pricing a Stage-1 operator | `engineering-measurement` | [`measurement/`](measurement/) | complete |
| AutoInit program analyses and harness validations, spanning the Stage-1 experiments | `experiment-spanning` | logs/shared/ (BLOCKED — pinned by frozen sources) | pinned in place |

`autoinit_program_material` is stage-1 material that stays where it is: scripts/pod/autoinit_phase_b_driver.py and scripts/pod/autoinit_continuation_b_driver.py read these exact paths and are named with a digest by consumed Phase-B and continuation-B authorizations, now under each phase's own history/superseded_authorizations/. Moving it would mean editing frozen-set members to tidy a directory. Declared rather than left looking stage-neutral.

## Experiments

What each one asked of this stage.

| experiment | what it asked | logs | status |
| --- | --- | --- | --- |
| `phase_a` — AutoInitializer Phase A — greedy search over initialization operator paths | Under one fixed calibration distribution, which initialization operator path gives the best behavioural starting point? | [`phase_a/`](phase_a/) | complete |
| `phase_b` — AutoInitializer Phase B — does the preferred composition change with the calibration distribution? | Does the preferred composition change when the calibration distribution is allowed to vary? | [`phase_b/`](phase_b/) | complete |
| `continuation_b` — Continuation B — resolving the Phase-B behavioural selection at rung 2 | Which Phase-B candidate wins once the behavioural comparison is carried to rung 2? | [`continuation_b/`](continuation_b/) | complete |
| `recovery_continuation` — Recovery continuation — finishing Phase A's search under the repaired harness | Finished under the repaired harness, which of Phase A's searched states survive? | [`recovery_continuation/`](recovery_continuation/) | complete |
| `phase_c1` — Phase C1 — fixed-path ATTENTION isolation | Does a replacement ATTENTION operator beat the frozen incumbent with every other operator held on a fixed path? | [`phase_c1/`](phase_c1/) | authorized — not launched |

## Runs

**44** run(s) are registered for this stage's
experiments. An experiment's plans, analyses, results, history,
validations and runs are all inside its own directory; the
canonical run list, across every stage, is
[`../../index.json`](../../index.json).

## Canonical configs

`configs/` is the source of truth. A run's manifest records
the config path and hash it ran under.

* `configs/experiments/phase_a/source_sets.json`
* `configs/experiments/phase_c1/authorization.json`
* `configs/stage1/qwen3_0p6b_from_4b_thinking.json`
* `configs/stage1/qwen3_0p6b_from_4b_thinking_contribution.json`

## Canonical data and artifact manifests

A dataset manifest lives beside the data it describes, and
an artifact lives outside git with its manifest. Neither is
copied here.

* `data/warmup/holdout_v1.manifest.json`
* `artifacts/stage1/qwen3_0p6b_init_v0`
* `artifacts/stage1/state_eval_v1`
* `artifacts/stage1/e8_contribution_init_v1`

## Current status

**Complete for the canonical recipe; the operator search is the open work.** `qwen3_0p6b_init_v0` exists and is pinned. Phase A ended `unresolved_equivalence` with no winner; Phase B resolved one, but by a margin its own record puts at about 3.6% of a single correct answer, and its winner is not distinguishable from Phase A's leader. Phase C1 — the ATTENTION isolation that would give the first operator-level causal claim — is authorized and has never been executed. The figures are in `logs/state/experiment_index.md`, which owns them.

Money, authorizations and what is running right now are not
here: they belong to [`../../budget/`](../../budget/) and
[`../../state/`](../../state/).
