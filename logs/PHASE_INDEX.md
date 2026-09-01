# Phase index — the AutoInit initialization-selection programme

**Start here** if you want the scientific history and do not know the filenames.

[`CATALOG.md`](CATALOG.md) organizes logs by *class* — who owns which fact, what
is current, what is superseded. This file organizes the same evidence by
**phase → experiment → attempt**, which is how the science actually happened. It
duplicates no facts: every row is a status label and a link.

**Terminal states at a glance**

| phase | status | outcome |
| --- | --- | --- |
| **Phase A** | COMPLETE | **`unresolved_equivalence`** — no winner. Numerical leader `cca699c93f34` |
| **Phase B** | COMPLETE | **`resolved`** — winner `fe9683e6a9c7`, but the separation is razor-thin |
| **Phase C0** | **COMPLETE / APPROVED / FROZEN** 2026-09-01 | protocol [`phase_c0_preregistration.json`](phase_c0_preregistration.json), sizing [`phase_c0_sizing_evidence.json`](phase_c0_sizing_evidence.json). N=850 scorable / 950 total, 3 fresh paired seeds |
| **Phase C1** | **IMPLEMENTED / NOT EXECUTED / NOT PRICED / NOT AUTHORIZED** | fixed-path ATTENTION isolation using short 0.86M recovery probes. CPU machinery and the `c1_confirmation_v1` battery exist; no probe has run and no arm has been evaluated. Structure in [`phase_c_roadmap.md`](phase_c_roadmap.md) |
| **Phase C2** | **NOT STARTED** | runs only if C1 finds a worthwhile ATTENTION formulation |

> The two phases' best candidates are **not distinguishable**: `16/510` vs
> `15/510`, one correct answer apart. See
> [`phase_a_vs_phase_b_comparison.md`](phase_a_vs_phase_b_comparison.md).

---

## Phase A — search under a fixed calibration

Question: under `calib.domain_balanced@v1` alone, which initialization and
operator order gives the best behavioural starting point?

### Search and infrastructure attempts

| attempt | status | record |
| --- | --- | --- |
| 1–5 | failed before the current session architecture | [`decisions.md`](decisions.md) |
| 6 | failed | [`autoinit_phase_a_attempt6/`](autoinit_phase_a_attempt6/) |
| 7 | failed | [`autoinit_phase_a_attempt7/`](autoinit_phase_a_attempt7/) |
| 8 | failed, `$0.1900` | [`autoinit_phase_a_attempt8/`](autoinit_phase_a_attempt8/) |
| 9 | failed, `$0.3400` | [`autoinit_phase_a_attempt9/`](autoinit_phase_a_attempt9/) |
| 10 | failed, `$11.4300` | [`autoinit_phase_a_attempt10/`](autoinit_phase_a_attempt10/) |
| 11 | search completed; five selected leaves lost at teardown | [`autoinit_phase_a_attempt11/`](autoinit_phase_a_attempt11/) |
| 12 | OOM | [`autoinit_phase_a_attempt12/`](autoinit_phase_a_attempt12/) |
| summary | — | [`autoinit_phase_a_attempts/`](autoinit_phase_a_attempts/) |

### Behavioural selection — recovery continuation

| attempt | status | record |
| --- | --- | --- |
| 1 | `$0.01`, launcher died 27 s in | [`autoinit_recovery_continuation_attempt1/`](autoinit_recovery_continuation_attempt1/) |
| 2–6 | failed | [`autoinit_recovery_continuation_attempt2/`](autoinit_recovery_continuation_attempt2/) … [`…attempt6/`](autoinit_recovery_continuation_attempt6/) |
| **7** | **`ALL_DONE` — the Phase-A terminal result**, `$12.8587` | [`autoinit_recovery_continuation_attempt7/`](autoinit_recovery_continuation_attempt7/) |

### Phase-A terminal result

**`unresolved_equivalence`, `winner: None`.** `cca699c93f34` `15/510 = 0.029412`
(usable `0.6561`) against `85bde4ded2c3` `10/510 = 0.019608` — margin `0.009804`,
**inside** the `0.011695` interval. Both beat the canonical control (`0.008824`).

* result → [`autoinit_recovery_continuation_attempt7/phase_a_result.json`](autoinit_recovery_continuation_attempt7/phase_a_result.json)
* preregistration → [`autoinit_phase_a_preregistration.json`](autoinit_phase_a_preregistration.json)
* frozen science plan → [`autoinit_phase_a_recovery_plan_frozen.json`](autoinit_phase_a_recovery_plan_frozen.json)
* permanent Stage-3 controls → [`autoinit_permanent_controls/`](autoinit_permanent_controls/), [`autoinit_stage3_complete/`](autoinit_stage3_complete/)

---

## Phase B — joint `P=2` search over two calibration profiles

Question: does the preferred composition change when the calibration
distribution is allowed to vary?

### P=2 search attempts

| attempt | status | record |
| --- | --- | --- |
| 1 | aborted at the pod test gate, `$0.15` | [`autoinit_phase_b_attempt1.json`](autoinit_phase_b_attempt1.json) |
| 2 | aborted at authorization binding, `$0.23` | [`autoinit_phase_b_attempt2.json`](autoinit_phase_b_attempt2.json) |
| 3 | search deadline exhausted, `$9.50` | [`autoinit_phase_b_attempt3.json`](autoinit_phase_b_attempt3.json) · [`autoinit_phase_b_attempt3/`](autoinit_phase_b_attempt3/) |
| 4 | search completed, raised writing its summary, `$8.17` | [`autoinit_phase_b_attempt4.json`](autoinit_phase_b_attempt4.json) · [`autoinit_phase_b_attempt4/`](autoinit_phase_b_attempt4/) |
| **5** | **Stage 1 COMPLETE — authoritative Top-5**, then Stage 2 failed. `$11.97` | [`autoinit_phase_b_attempt5.json`](autoinit_phase_b_attempt5.json) · [`autoinit_phase_b_attempt5/`](autoinit_phase_b_attempt5/) |

### Stage-1 Top-5 → six-candidate universe → rung 1

| step | artifact |
| --- | --- |
| authoritative Top-5 and operator paths | [`autoinit_phase_b_attempt5/stage1_selection.json`](autoinit_phase_b_attempt5/stage1_selection.json) |
| identity collapse, six-candidate universe, rung-1 selection | [`autoinit_phase_b_identity_collapse_amendment.json`](autoinit_phase_b_identity_collapse_amendment.json) |
| post-freeze executable drift, declared | [`autoinit_phase_b_post_freeze_changes.json`](autoinit_phase_b_post_freeze_changes.json) |

Two Top-5 leaves reproduced retained Phase-A finalists byte-for-byte, so the
universe collapses to **six distinct candidates**; rung 1 advanced
`fe9683e6a9c7`, `85bde4ded2c3` and the control.

### Behavioural continuation — five attempts

| attempt | status | reached | record |
| --- | --- | --- | --- |
| 1 | failed, `$0.2513` | pod test gate | [`autoinit_continuation_b_attempt1.json`](autoinit_continuation_b_attempt1.json) · [`…attempt1/`](autoinit_continuation_b_attempt1/) |
| 2 | failed, `$0.3146` | driver stage 0 | [`autoinit_continuation_b_attempt2.json`](autoinit_continuation_b_attempt2.json) · [`…attempt2/`](autoinit_continuation_b_attempt2/) |
| 3 | failed, `$0.2275` | stage 1 | [`autoinit_continuation_b_attempt3.json`](autoinit_continuation_b_attempt3.json) · [`…attempt3/`](autoinit_continuation_b_attempt3/) |
| 4 | `ALL_DONE`, `$1.4680` — **decision WITHDRAWN**, probe retained | [`autoinit_continuation_b_attempt4.json`](autoinit_continuation_b_attempt4.json) · [`…attempt4/`](autoinit_continuation_b_attempt4/) |
| **5** | **`ALL_DONE` — the Phase-B terminal result**, `$1.5433` | [`autoinit_continuation_b_attempt5.json`](autoinit_continuation_b_attempt5.json) · [`…attempt5/`](autoinit_continuation_b_attempt5/) |

> ### ⚠ Attempt 4's result is WITHDRAWN
>
> Attempt 4 reported `resolved / winner=fe9683e6a9c7`. **That decision is not
> accepted.** The inherited pooling admitted the *imported* `85bde4ded2c3/sc`
> into a rung-2 comparison, making it `sa+sb+sc` (n=570) against `sa+sb` (n=380)
> for the others — not a same-rung comparison, and its `0.012745` margin is not a
> quantity the frozen rule is defined over.
>
> The probe it bought, `fe9683e6a9c7/sb`, **is** valid and is retained.
>
> The accepted rung-2 decision is
> [`autoinit_continuation_b_corrected_rung2.json`](autoinit_continuation_b_corrected_rung2.json):
> `sa+sb` only → **`tie_pending`**, candidates `{fe9683e6a9c7, 85bde4ded2c3}`.

### Reuse records — all three, all authorization-bound

| record | admits | verifier |
| --- | --- | --- |
| [`autoinit_historical_probe_reuse.json`](autoinit_historical_probe_reuse.json) | 8 historical probes | `scripts/autoinit/verify_historical_probe_reuse.py` |
| [`autoinit_attempt5_probe_reuse.json`](autoinit_attempt5_probe_reuse.json) | 3 fresh `sa` | `scripts/autoinit/verify_attempt5_probe_reuse.py` |
| [`autoinit_attempt4_probe_reuse.json`](autoinit_attempt4_probe_reuse.json) | `fe9683e6a9c7/sb` | `scripts/autoinit/verify_attempt4_probe_reuse.py` |

### Phase-B terminal result

**`resolved`, winner `fe9683e6a9c783bbc6fe276a78c851c6`**, `tie_break_ran: true`.

| final pooled | correct | `correct_overall` | `usable_rollout` |
| --- | --- | --- | --- |
| `fe9683e6a9c7` | 16/510 | **0.031373** | 0.6842 |
| `85bde4ded2c3` | 10/510 | 0.019608 | 0.5456 |
| control | 3/340 | 0.008824 | 0.4947 |

> **The margin clears the equivalence interval by `0.000070`** — about 3.6% of
> one correct sample. At 15 correct instead of 16 the result would have been
> `unresolved_equivalence`. Protocol-resolved; **not** strong evidence of
> intrinsic superiority.

* result → [`autoinit_continuation_b_attempt5/phase_a_result.json`](autoinit_continuation_b_attempt5/phase_a_result.json)
* preregistration → [`autoinit_continuation_b_preregistration.json`](autoinit_continuation_b_preregistration.json)
* pricing → [`autoinit_behavioural_continuation_pricing.json`](autoinit_behavioural_continuation_pricing.json)
* grant → [`autoinit_continuation_b_grant.json`](autoinit_continuation_b_grant.json)
* consumed authorizations → [`superseded/`](superseded/)

---

## Phase A vs Phase B

**→ [`phase_a_vs_phase_b_comparison.md`](phase_a_vs_phase_b_comparison.md)** — the
scientific handoff into Phase C. Both phases end to end, the operator-level
evidence classification, the recommended incumbent and its caveats, and the
Phase-C starting point.

---

## Diagnostics and infrastructure — no scientific claim

| what | record |
| --- | --- |
| E1–E8 experiment series | [`EXPERIMENT_INDEX.md`](EXPERIMENT_INDEX.md), [`EXPERIMENTS.md`](EXPERIMENTS.md) |
| bounded measurement sessions | [`autoinit_measurement_attempt1/`](autoinit_measurement_attempt1/) … [`…attempt3/`](autoinit_measurement_attempt3/) |
| device canary — TERMINATED | [`autoinit_device_canary_attempt1/`](autoinit_device_canary_attempt1/), [`…attempt2/`](autoinit_device_canary_attempt2/) |
| micro-preflight | [`autoinit_preflight_run4/`](autoinit_preflight_run4/) |
| causal-depth backend equivalence | [`autoinit_depth_backend_equivalence.json`](autoinit_depth_backend_equivalence.json) |
| capacity / transport findings | [`autoinit_continuation_b_capacity.json`](autoinit_continuation_b_capacity.json), [`autoinit_leaf_transport_quota_finding.json`](autoinit_leaf_transport_quota_finding.json) |
| scratch inventory | [`scratch_inventory_20260829.json`](scratch_inventory_20260829.json) |

## Operational lessons

Every paid failure and what closed it: [`decisions.md`](decisions.md). The
continuation's four failures are also summarized in the attempt table above —
each closed a distinct defect class, and all four were found on hardware after
`$0` checks had passed.

## Budget

[`BUDGET_LEDGER.md`](BUDGET_LEDGER.md) owns all spend, caps and authorization
status. Nothing else does.
