# stage-0 — Initialization warm-up data collection

Pipeline stage **0**: collect the teacher signals a teacher-aware student initialization needs (AGENTS.md 4.2).

Generated from [`../index.json`](../index.json). It records the
evidence behind every claim below; a path in a code span is a
pointer to its canonical owner, which is where the revisions,
counts, hashes and configs live. Stage membership is read from
repository facts — a config's stage field, the
`configs/stage<N>/` directory it lives in, a declared `stage_id`,
or an experiment's own subject and product — never guessed from a
name.

## Purpose

Collect the teacher signals a teacher-aware student initialization needs. Stage 0 trains nothing and produces no model — its product is statistics.

## Inputs

What this stage consumes, and from where.

* the teacher checkpoint `Qwen/Qwen3-4B-Thinking-2507`, pinned by revision in the collection config
* a small, deliberately diverse warm-up corpus — general text, instructions, grade-school math and short programs — so the statistics are not fitted to one distribution

## Data

* `data/stages/stage-0/warmup/warmup_v1.jsonl` — the warm-up corpus itself: five families — FineWeb-Edu, Dolly-15k, GSM8K, MBPP and a small project-authored set — deduplicated on content hash
* `data/stages/stage-0/warmup/warmup_v1.manifest.json` — owns the per-source dataset id, revision, license, sample count and output hash. Read it rather than this page for any number
* `data/stages/stage-0/warmup/holdout_v1.jsonl` — held-out FineWeb-Edu documents, disjoint from the warm-up corpus, kept for the Stage-1 perplexity gate

## How it runs

Only what helps to understand the stage; the config is the
source of truth.

* the teacher runs in BF16 at a 1024-token sequence length, under a logged activation-cache budget
* the collector caches streaming **sufficient statistics**, not raw activations (decision 2026-07-12) — which is why a cache of this size can stand in for a corpus pass

## Outputs

* `artifacts/stages/stage-0/qwen3_4b_thinking_v1` — the statistics cache Stage 1 initializes from. Outside git; regenerable from the config and the corpus

## Pipeline activity

The stage's own work, and measurements supporting it. Not
studies *of* it.

| what | kind | logs | status |
| --- | --- | --- | --- |
| Teacher activation / statistics collection for student initialization | `pipeline-activity` | none | complete |

## Experiments

What each one asked of this stage.

| experiment | what it asked | logs | status |
| --- | --- | --- | --- |
| `calibration_volume` — Stage-0 calibration volume dose-response (first Stage-0 study) | With teacher, geometry, the forward-baseline method and the recovery recipe fixed, does more independent, capability-aligned Stage-0 calibration data stabilize the AutoInitializer's structural selection (H1), improve the recovered student's reasoning on unseen rule families (H2), and saturate (H3)? | [`calibration_volume/`](calibration_volume/) | CLOSED 2026-10-21: EXECUTION COMPLETE / DOWNSTREAM REASONING FLOOR / VOLUME EFFECT ON REASONING NOT RESOLVED. Formal study d1/d2/d4 (one D1 full search and one fixed recovery each) and the Top-4 extension (24 students) measured on all of eval-set-1: fluid correct at most 3/570 static and 0/144 episodes, so every preregistered contrast sits at the floor; static usable 40.0-71.4%, moved more by the recovery seed than by the candidate. Volumes retired; budget reconciled against the provider's billing. Mixture ratio DEFERRED (never run) |

## Runs

**43** run(s) are registered for this stage's
experiments. An experiment's plans, analyses, results, history,
validations and runs are all inside its own directory; the
canonical run list, across every stage, is
[`../../index.json`](../../index.json).

## Canonical configs

`configs/` is the source of truth. A run's manifest records
the config path and hash it ran under.

* `configs/stages/stage-0/calibration_volume/pilot_design.json`
* `configs/stages/stage-0/qwen3_4b_thinking.json`
* `configs/stages/stage-0/qwen3_4b_thinking_v1.json`

## Canonical data and artifact manifests

A dataset manifest lives beside the data it describes, and
an artifact lives outside git with its manifest. Neither is
copied here.

* `data/stages/stage-0/warmup/warmup_v1.manifest.json`
* `artifacts/stages/stage-0/qwen3_4b_thinking_v1`

## Decisions

* logs/budget/decisions.md — 2026-07-12: Stage 0 caches streaming sufficient statistics, not raw activations
* logs/budget/decisions.md — 2026-08-10: the Stage 0 activation cache was lost; regenerate it and prove the initialization is unchanged
* logs/budget/decisions.md — 2026-08-10: RESOLVED, the Stage 0 / initialization hash gate passed
* logs/budget/decisions.md -- 2026-10-18: Stage-0 calibration-volume programme started; battery valid, students at floor, root-level pilot; pilot attempt 1 OOM repaired on an 80 GB card; programme terminal; permanent evaluation registry; resolution attempts 1-2 repaired; design amended to one draw per dose, still NOT RUN
* logs/budget/decisions.md -- 2026-10-19: $90 new-spend authority, eval-set-1 frozen, formal study preregistered and run; 2026-10-20: Top-4 extension (+$30, +$6, two sessions at once); 2026-10-21: final closeout at $0

## Current status

**Complete.** The v0 cache was lost and v1 was regenerated; the initialization hash gate then passed, so the Stage-1 result is unchanged by the regeneration (decisions 2026-08-10). No experiment ran on top of the collection, so there are no experiment-run logs — and none are invented.

Money, authorizations and what is running right now are not
here: they belong to [`../../budget/`](../../budget/) and
[`../../state/`](../../state/).
