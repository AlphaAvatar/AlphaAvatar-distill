# phase_c1 — fixed-path ATTENTION isolation

Runs of the Phase-C1 experiment: a two-arm isolation that changes **one
operator** of a frozen initialization path and measures whether it helps.

* **arms** — incumbent `attention.weight_proxy_v0` vs treatment
  `attention.activation_importance_v1`, on a shared
  `DEPTH → FFN → RESIDUAL_WIDTH` prefix
* **instrument** — 2 arms × 3 fresh paired seeds = 6 `E1_KD_HEAVY_0860K`
  recovery probes, each evaluated once on the frozen 950/850 confirmation
  battery
* **decision** — the frozen three-way `GO` / `NO-GO` / `INCONCLUSIVE` rule

It is **not** formal recovery evidence and establishes no recovered-model
capability.

## Where the facts live

| what | owner |
| --- | --- |
| the frozen protocol | `logs/phase_c0_preregistration.json` |
| what a session executes, and its bound identities | `logs/experiments/phase_c1/execution_preregistration.json` |
| the approved budget, attempt and retry policy | `configs/experiments/phase_c1/authorization.json` → `execution_package` |
| spend, per attempt and cumulative | `logs/BUDGET_LEDGER.md` |
| current status | `logs/current_state.json`, `logs/STATE.md` |
| every run, both layouts | `logs/runs/index.json` |

This file deliberately restates none of them. A second hand-maintained copy of a
cost, a commit or an authorization status is how the two disagree.

## Earlier attempts — three locations, not one

This said *"attempts 1–12 remain at `logs/runs/phase_c1/<run_id>/`"*, and only
three of them are there. The real layout, derived from
[`../../index.json`](../../index.json) rather than restated:

| attempts | where |
| --- | --- |
| 1–9 | `logs/autoinit_c1_attempt<N>/`, with several also holding a flat `logs/autoinit_c1_attempt<N>_grant.json` |
| 10–12 | `logs/runs/phase_c1/<run_id>/` |
| 13– | here |

They are **found, not relocated.** Attempts 1–9 predate the run directory
entirely; 10–12 predate the stage grouping. Their grants, consumed
authorizations and closeouts name the paths they are at, so moving one breaks
the lineage that makes it evidence — and for the flat grants that binding is
explicit: a consumed authorization records the grant's path *and* its hash.

The index covers all three locations. Query it rather than guessing from a
path.
