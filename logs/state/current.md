# Current state

**Updated:** 2026-09-13. The human view. Every number here has an owner named
beside it, and this file restates none of them from memory — a second
hand-maintained copy of a cost or a status is how two documents come to
disagree.

Start at [`README.md`](../README.md) if you do not know which document you want.

## Right now

**Nothing is running. Nothing is billing. No pod exists. Nothing is prepared
for launch.** The only process is a `$0` read-only capacity watch.

| | | owner |
| --- | --- | --- |
| phase | C1 — fixed-path ATTENTION isolation | [`stages/stage-1/phase_c1/`](../stages/stage-1/phase_c1/) |
| replay | **MEASURED — 2/2 PASS** (attempt 9) | [`index.json`](../index.json) |
| treatment, endpoint | **UNMEASURED** — zero probes trained. Attempt 9 is **NO DECISION**: a pre-treatment infrastructure abort, not a frozen-rule result | [`phase_c1/history/operational_history.md`](../stages/stage-1/phase_c1/history/operational_history.md) |
| launch chain | attempt 13's grant is committed; **nothing else is prepared**, and the chain is **paused** pending this cleanup | [`phase_c1/runs/attempt13/`](../stages/stage-1/phase_c1/runs/attempt13/) |
| blocker | provider capacity for secure L40S at `$1.09/h` | below |
| spend | `$268.2958` of `$320.0000` | [`budget/ledger.md`](../budget/ledger.md) |

## Readiness

<!-- readiness:begin -->

| readiness | | owner |
| --- | --- | --- |
| latest sweep | **diagnostic — FAIL** (3736 passed, 3 failed), swept at `30f6ceed`; **does not describe the current tree** | [`c1_pod_environment_verification.json`](../stages/stage-1/phase_c1/analyses/c1_pod_environment_verification.json) |
| launch-bound for the next session | **not prepared** — a launch-bound sweep on the final clean pre-authorization tree is owed | this file's launch-chain section |
| last launch-bound failure | swept at `82745981` on 2026-09-12 — kept as history, not a current state | [`readiness_history.json`](../stages/stage-1/phase_c1/history/readiness_history.json) |

*Generated from the record by `scripts/consolidate/render_log_navigation.py`; do not edit by hand — it went stale within hours when it was prose.*

<!-- readiness:end -->

## Budget — four limits that do not transfer

Derived by [`scripts/consolidate/derive_budget.py`](../../scripts/consolidate/derive_budget.py)
from the approved package and each session's own closeout. **Do not restate
these by hand; run the deriver.**

| limit | remaining |
| --- | --- |
| formal sessions | `$45.0465` of `$45.4425` |
| GPU engineering | `$6.0000` of `$6.0000` |
| package | `$51.0465` of `$51.4425` |
| project cap | `$51.7042` of `$320.0000` |

**Full-ceiling sessions the FORMAL allowance funds: 2.** Not three. Three
ceilings cost `$45.4425` and the formal allowance has `$45.0465` — short by
`$0.3960`. The package balance is `$51.0465` and dividing *that* by the ceiling
gives three, which is the error: the engineering allowance cannot pay for a
formal probe. This file, `current_state.json`, the ledger and attempt 13's
grant all carried the wrong figure until 2026-09-12.

Remaining balance is not permission.

## The blocker: capacity, not configuration

Attempt 12's create was refused — *"no longer any instances available with the
requested specifications"* — at the accepted `$1.09/h`, after all 14
pre-provider gates passed twice. Nothing was created and nothing billed.

Host draws do not address it: a draw replaces a host that was *created* and
never became usable, and creation itself was refused. A create-attempt loop
would ask the same market again, which is stock chasing and is not authorized.

**A stock label is not the gate.** Attempt 10 acquired at `Medium`, attempt 11
failed at `Low`, attempt 12 was refused at `Low`; the label predicted none of
them. `Low` is not `none`, and a `null` reading is *unknown* — neither available
nor permanently unavailable. What gates a launch is the live `securePrice` quote
at or below the accepted rate plus every pre-provider gate.

## The launch chain, and where it stopped

One authorization funds one launcher session: up to three acquisition draws
inside it, never two billing resources, all sharing one `$15.1475` ceiling.

1. this run's **grant**, committed on a clean tree — **done**, attempt 13
2. a **`launch_bound` sweep** on that clean pre-authorization tree — **failed**
   2026-09-12 and is owed again
3. the one-use **authorization**, issued from the grant — not started
4. the exact-session **bundle** — not started
5. a live quote, every pre-provider gate, the single launch

Step 2 must follow step 1: `verify_record` permits exactly two tracked paths to
differ after a sweep — the readiness record and the issued authorization — so a
grant committed after a sweep invalidates it.

Attempt 12's chain is **consumed** and authorizes nothing. It is never reused.

Full terms — attempt counting, the six retry conditions, the stop list:
`execution_package` in
[`../configs/experiments/phase_c1/authorization.json`](../../configs/experiments/phase_c1/authorization.json).

## What ends a round

A complete `GO`, `NO-GO` **or** `INCONCLUSIVE` all end it. `INCONCLUSIVE` is a
result and is never re-run in pursuit of a `GO`.

Stop and report if: the first probe has started training, or whether it started
cannot be confirmed; a real replay mismatch at stage D or E; an input-identity
conflict; a limit reached; or a resource whose billing state is unknown.

## The log tree: stage-first, and every experiment attributed

`logs/` is **Stage → Experiment → Run**. Four stages have repository evidence
and therefore exist — stage-0 and stage-2 as pipeline activity with no
experiment-run logs, stage-1 and stage-3 with both.

Every historical experiment has a stage, rebuilt from repository facts and
checked rather than asserted: [`stages/index.json`](../stages/index.json) is the
stage index. It carries the evidence for each assignment and the rules that
decided it, and it also says what each stage is *for*, what it consumes and
produces, and where its canonical configs, data manifests and artifacts live —
so `logs/stages/` is the pipeline's stage-level entry point rather than a run
container. **20 experiments, 20 in one stage, 0 genuinely cross-stage, 0
unresolved.** There is no `cross-stage/` directory, because no experiment takes
several pipeline stages as its subject; a run whose stage nobody declared is
refused rather than shelved.

Five Stage-3 experiments ran and produced no log files of their own — `ttb`,
`p0_real`, `d0`, `p0`, `p2`. They are listed in
[`stages/stage-3/`](../stages/stage-3/) with their configs, artifacts and index
sections, and given no directory: an empty one would assert material that does
not exist. `e2` now has one, holding the single document it left behind.

`migrations/` and `archive/` are gone. A superseded document is deleted, since
git history holds it; a historical document that is still part of the record —
a preregistration, a consumed authorization, the pre-layout experiment
chronology — sits under the experiment or stage that owns it. The few old paths
current tooling must still resolve are a flat table in
[`index.json`](../index.json)`.historical_paths`.

One declared exception: `shared/analyses/autoinit_*` and four
`shared/validations/` directories are **Stage-1 material, not stage-neutral**.
They stay where they are because frozen pod drivers read those exact paths —
recorded in the stage index with its blocker rather than left looking ownerless.

This changed no experiment result, no authorization and no frozen evidence. The
launch chain is unaffected and still paused.

## Engineering, not results

The initialization migration and the CUDA stage-F validation are engineering
records. The stage-F device repair is **CONFIRMED ON REAL CUDA** at execution
SHA `7027a8f4`. Neither is a C1 result and neither authorizes anything:
[`maintenance/source-relocations/initialization-core/v1/`](../maintenance/source-relocations/initialization-core/v1/) ·
[`phase_c1/validations/cuda-stage-f/v1/`](../stages/stage-1/phase_c1/validations/cuda-stage-f/v1/)

## History

This file holds the current state only. The narrative lives with what it is
about, and is unedited:

* [`phase_c1/history/operational_history.md`](../stages/stage-1/phase_c1/history/operational_history.md)
  — every C1 session, its cost, failure and repair
* [`stages/stage-3/history/EXPERIMENTS.md`](../stages/stage-3/history/EXPERIMENTS.md)
  — the pre-layout experiment chronology, and the only record several Stage-3
  experiments have
* [`experiment_index.md`](experiment_index.md) — what each experiment proved,
  what it does **not** support, and which conclusions still bind
* [`phase_index.md`](phase_index.md) — the same history organized by phase
* [`decisions.md`](../budget/decisions.md) — decision records
* [`budget/ledger.md`](../budget/ledger.md) — every cost, per session
