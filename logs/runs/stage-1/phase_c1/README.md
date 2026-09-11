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
| what a session executes, and its bound identities | `logs/phase_c1_execution_preregistration.json` |
| the approved budget, attempt and retry policy | `configs/experiments/phase_c1/authorization.json` → `execution_package` |
| spend, per attempt and cumulative | `logs/BUDGET_LEDGER.md` |
| current status | `logs/current_state.json`, `logs/STATE.md` |
| every run, both layouts | `logs/runs/index.json` |

This file deliberately restates none of them. A second hand-maintained copy of a
cost, a commit or an authorization status is how the two disagree.

## Earlier attempts

Attempts 1–12 ran before this grouping existed and remain at
`logs/runs/phase_c1/<run_id>/`. They are indexed from the same `index.json`;
they are not moved, because their grants and closeouts name those paths.
