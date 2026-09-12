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

## Every attempt is here

All 14 Phase-C1 runs live under this one directory. They were in three
layouts — `logs/autoinit_c1_attempt<N>/` for 1–9, `logs/runs/phase_c1/<run>/`
for 10–12, and this one for 13 — and the log-layout-v1 migration brought them
together without changing a byte.

Old paths still appear inside consumed authorizations and closed manifests.
That is not staleness: each states where the object was when that payload was
written, and
[`../../../migrations/log-layout-v1/manifest.json`](../../../migrations/log-layout-v1/manifest.json)
maps every one forward.

Stage 1 because `configs/experiments/phase_c1/authorization.json` DECLARES
`stage_id: "1"` — the only experiment in this repository that declares one.
