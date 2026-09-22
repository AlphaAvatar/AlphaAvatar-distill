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

## This stage's own areas

Material that belongs to the stage rather than to one of its
experiments.

| area | what it holds |
| --- | --- |
| [`history/`](history/) | records spanning this stage's experiments, kept verbatim |

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
| `phase_c2_full_search` — Phase C2 full joint re-search — all four operator kinds, Top-5 | After ATTENTION was promoted by C1, which JOINT assignment of DEPTH, FFN, RESIDUAL_WIDTH and ATTENTION -- including their order and each step's calibration mixture -- produces the best initialization on the frozen state_eval metrics? Search-1 could not answer it: it fixed three operators at the incumbent's assignment and varied one, so an ATTENTION change that moved the best DEPTH or WIDTH was invisible to it. The answer is a preregistered Top-5 candidate set, not an incumbent: which candidate becomes the incumbent is a BEHAVIOURAL question a separate session asks. | [`phase_c2_full_search/`](phase_c2_full_search/) | chain built — NOT AUTHORIZED, no grant approved |
| `phase_c2_replay` — Phase C2 replay-only artifact reconstruction — attempt 3's Top-5 checkpoints | Can the five checkpoints behind the frozen Top-5 be rebuilt byte-for-byte from the committed evidence? The full joint re-search committed its selection and then lost the weights when the pod was torn down, so the ranking survives and the artifacts it ranked do not. This session replays each selected path with EVERY step pinned to the artifact digest attempt 3 recorded, and either reproduces each leaf's exact identity or stops. It decides nothing: no beam, no ranking, no selection, no selection-bearing evaluation, no control comparison and no behavioural work. A digest mismatch would be a scientific finding, not a retryable engineering failure. | [`phase_c2_replay/`](phase_c2_replay/) | built and gated — $4.77 all-in derived, inside the original $5.00; chain not yet consumed |
| `phase_c2_behavioural` — Phase C2 behavioural selection — 12 probes over the frozen Top-5 | Does any of the five reconstructed Full-Search candidates beat the behavioural incumbent B, and by enough to take its place? Cheap-metric order is not behavioural order, so the search's own ranking cannot promote anything. Six screening probes rank the five candidates against B on one preregistered seed; exactly one advances; six confirmation probes test that one against B on three disjoint paired seeds under C1's frozen decision rule. Only the confirmation rung may name a C2 incumbent, and NO_GO and INCONCLUSIVE are results. | [`phase_c2_behavioural/`](phase_c2_behavioural/) | PROPOSED — not authorized; no grant, readiness, authorization or bundle |
| `phase_c2` — Phase C2 — ATTENTION-aware composition/order re-search | With that ATTENTION operator now fixed, does re-optimizing the operator order and ATTENTION's calibration profile beat the frozen C1 treatment? | [`phase_c2/`](phase_c2/) | planned — space implemented and priced, NOT authorized |
| `phase_c2_baseline_completion` — Phase C2 baseline completion — the B side of B→C | What does the frozen C1 treatment baseline B score on the same state_eval suite the five selected Search-1 candidates were measured on -- so that the B→C comparison Attempt 4 collected its ranking to ask can finally be computed? | [`phase_c2_baseline_completion/`](phase_c2_baseline_completion/) | granted — attempt5 grant committed, NOT yet authorized |

## Runs

**83** run(s) are registered for this stage's
experiments. An experiment's plans, analyses, results, history,
validations and runs are all inside its own directory; the
canonical run list, across every stage, is
[`../../index.json`](../../index.json).

## Canonical configs

`configs/` is the source of truth. A run's manifest records
the config path and hash it ran under.

* `configs/autoinit/c2_replay_artifacts.json`
* `configs/experiments/phase_a/source_sets.json`
* `configs/experiments/phase_c1/authorization.json`
* `configs/experiments/phase_c2/baseline_completion_authorization.json`
* `configs/experiments/phase_c2/full_search_authorization.json`
* `configs/stage1/qwen3_0p6b_from_4b_thinking.json`
* `configs/stage1/qwen3_0p6b_from_4b_thinking_contribution.json`
* `configs/validation/c2_full_search_cuda.json`
* `configs/validation/c2_full_search_performance.json`
* `configs/validation/c2_state_eval_certification.json`
* `logs/stages/stage-1/phase_c2/plans/phase_c2_full_search_protocol.json`

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
