# stage-6 — Recurrent depth

Pipeline stage **6**: study shared-parameter repeated execution -- a block executed again and again on its own output -- and the training it needs (AGENTS.md 4.8, maintainer 2026-11-13).

Generated from [`../index.json`](../index.json). It records the
evidence behind every claim below; a path in a code span is a
pointer to its canonical owner, which is where the revisions,
counts, hashes and configs live. Stage membership is read from
repository facts — a config's stage field, the
`configs/stage<N>/` directory it lives in, a declared `stage_id`,
or an experiment's own subject and product — never guessed from a
name.

## Purpose

Study recurrent forward execution: a block whose parameters are shared and executed repeatedly, one iteration after another, and the training such a student needs (AGENTS.md 4.8, maintainer decision 2026-11-13). Whether repetition recovers depth -- and with it reasoning -- a compressed student lost, at the parameter count the budget allows, and at what compute and latency cost.

## Inputs

What this stage receives from the pipeline before it.

* a training-pre initialization of a depth-compressed student and the teacher lineage it came from
* a recovery data set and recipe each Stage-6 design registers for itself (nothing inherited by default)

## How it runs

Only what helps to understand the stage; the config is the
source of truth.

* the shared blocks, how often each runs, the initialization, the data and packing, the objective and the compute accounting are recorded as the design's identity before its first run
* parameter-matched, FLOPs-matched and latency-matched comparisons are different comparisons, reported apart

## Outputs

* recovered recurrent students' weights on the relay, their sealed eval-set-1 sets on the dev box

## Experiments

What each one asked of this stage.

| experiment | what it asked | logs | status |
| --- | --- | --- | --- |
| `depth_recurrence` — Depth recovery, variants 2 and 3: recurrent execution of the kept blocks, and per-iteration hidden-state distillation (DMR1-R, DMR1-RH) | Does executing each kept block once per teacher layer of its span leave more static fluid reasoning after the same recovery than executing it once, from the same initialization; and does a per-iteration hidden-state loss change what survives? | [`depth_recurrence/`](depth_recurrence/) | PREPARED at $0 (DMR1 variants 2 and 3, READY_FOR_REVIEW 2026-11-26; no paid run authorized): SOB1's 16 kept blocks executed on the 36-logical-layer schedule from the selected training-pre initialization, with and without a per-iteration hidden-state term; estimated, not proposed |

## Runs

**0** run(s) are registered for this stage's
experiments. An experiment's plans, analyses, results, history,
validations and runs are all inside its own directory; the
canonical run list, across every stage, is
[`../../index.json`](../../index.json).

## Canonical configs

`configs/` is the source of truth. A run's manifest records
the config path and hash it ran under.

* `configs/stages/stage-6/depth_recurrence/design.json`

## Canonical data and artifact manifests

None: this stage's material is code and logs.

## Decisions

* the maintainer's brief of 2026-11-26 (DMR1: variants 2 and 3 prepared, not authorized)

## Current status

Prepared at $0 (DMR1, 2026-11-26): no Stage-6 run exists; nothing is authorized.

Money, authorizations and what is running right now are not
here: they belong to [`../../budget/`](../../budget/) and
[`../../state/`](../../state/).
