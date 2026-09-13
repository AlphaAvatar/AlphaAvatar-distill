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

* `data/warmup/warmup_v1.jsonl` — the warm-up corpus itself: five families — FineWeb-Edu, Dolly-15k, GSM8K, MBPP and a small project-authored set — deduplicated on content hash
* `data/warmup/warmup_v1.manifest.json` — owns the per-source dataset id, revision, license, sample count and output hash. Read it rather than this page for any number
* `data/warmup/holdout_v1.jsonl` — held-out FineWeb-Edu documents, disjoint from the warm-up corpus, kept for the Stage-1 perplexity gate

## How it runs

Only what helps to understand the stage; the config is the
source of truth.

* the teacher runs in BF16 at a 1024-token sequence length, under a logged activation-cache budget
* the collector caches streaming **sufficient statistics**, not raw activations (decision 2026-07-12) — which is why a cache of this size can stand in for a corpus pass

## Outputs

* `artifacts/stage0/qwen3_4b_thinking_v1` — the statistics cache Stage 1 initializes from. Outside git; regenerable from the config and the corpus

## Pipeline activity

The stage's own work, and measurements supporting it. Not
studies *of* it.

| what | kind | logs | status |
| --- | --- | --- | --- |
| Teacher activation / statistics collection for student initialization | `pipeline-activity` | none | complete |

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

* `configs/stage0/qwen3_4b_thinking.json`
* `configs/stage0/qwen3_4b_thinking_v1.json`

## Canonical data and artifact manifests

A dataset manifest lives beside the data it describes, and
an artifact lives outside git with its manifest. Neither is
copied here.

* `data/warmup/warmup_v1.manifest.json`
* `artifacts/stage0/qwen3_4b_thinking_v1`

## Decisions

* logs/budget/decisions.md — 2026-07-12: Stage 0 caches streaming sufficient statistics, not raw activations
* logs/budget/decisions.md — 2026-08-10: the Stage 0 activation cache was lost; regenerate it and prove the initialization is unchanged
* logs/budget/decisions.md — 2026-08-10: RESOLVED, the Stage 0 / initialization hash gate passed

## Current status

**Complete.** The v0 cache was lost and v1 was regenerated; the initialization hash gate then passed, so the Stage-1 result is unchanged by the regeneration (decisions 2026-08-10). No experiment ran on top of the collection, so there are no experiment-run logs — and none are invented.

Money, authorizations and what is running right now are not
here: they belong to [`../../budget/`](../../budget/) and
[`../../state/`](../../state/).
