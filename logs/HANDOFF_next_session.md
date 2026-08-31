# Handoff — Phase B closed, Phase C not started

Two handoffs, one frozen factual state. **A** is for the ChatGPT scientific-review
session; **B** is for the Claude Code execution session. Both describe the same
repository; neither carries a fact that is not committed and discoverable.

---

# A. ChatGPT — scientific review session

## Your role

You are the **independent scientific reviewer and GO / NO-GO authority**. You do
not launch GPUs, issue authorizations, or run compute. The execution session
implements and executes; you decide whether it may.

## Frozen state

| | |
| --- | --- |
| repository HEAD | see `git rev-parse origin/main`; recorded in [`current_state.json`](current_state.json) |
| spend | **`$263.8597`** of a **`$283.7600`** cap · headroom **`$19.9003`** |
| authorizations | **none.** All five behavioural-continuation grants retired |
| pods / orchestration | **zero** |
| Phase C | **NOT STARTED / NOT DESIGNED / NOT PRICED / NOT AUTHORIZED** |

## Phase A — COMPLETE

**`unresolved_equivalence`, `winner: None`.** Search under
`calib.domain_balanced@v1` alone.

| | correct | `correct_overall` | `usable_rollout` |
| --- | --- | --- | --- |
| `cca699c93f34` (numerical leader) | 15/510 | 0.029412 | 0.6561 |
| `85bde4ded2c3` | 10/510 | 0.019608 | 0.5456 |
| control | 3/340 | 0.008824 | 0.4947 |

Margin `0.009804`, **inside** the `0.011695` equivalence interval — the two were
never separated. Phase A selected nothing.

## Phase B — COMPLETE

**`resolved`, winner `fe9683e6a9c783bbc6fe276a78c851c6`.** Joint `P=2` search
over `calib.domain_balanced@v1` and `calib.reasoning_heavy@v2`.

| | correct | `correct_overall` | `usable_rollout` |
| --- | --- | --- | --- |
| **`fe9683e6a9c7`** | 16/510 | **0.031373** | 0.6842 |
| `85bde4ded2c3` | 10/510 | 0.019608 | 0.5456 |
| control | 3/340 | 0.008824 | 0.4947 |

> **The margin clears the interval by `0.000070`** — about **3.6% of one correct
> sample**. At 15 correct instead of 16 the result would have been
> `unresolved_equivalence`. Protocol-resolved; **not** evidence of intrinsic
> superiority.

**Attempt 4's originally reported winner is WITHDRAWN** (its pooling admitted an
imported `sc` into a rung-2 comparison). Its purchased probe is valid and
retained. The accepted terminal result is Attempt 5's.

## The comparison — read this before deciding anything

**[`phase_a_vs_phase_b_comparison.md`](phase_a_vs_phase_b_comparison.md)**

Three conclusions that should govern Phase C:

1. **The two phases' best candidates are not distinguishable.** `16/510` vs
   `15/510` — one answer, `0.001961`, about one sixth of the interval. And
   `cca699c93f34` was cut at Phase-B rung 1 on **single-seed** evidence by a gap
   *inside* the interval, so they were never tested head-to-head.
2. **Search-side KL did not predict behaviour.** The winner sat on Pareto
   **front 3**; `cca699` was front 0.
3. **ATTENTION took `calib.none@v1` in every competitive path**, including one
   that placed it first. Supports only the narrow claim that *the current
   operator and its search formulation* gave no competitive positive
   transformation — the Phase-C motivation.

**No operator-level causal claim is supported by either phase.** Neither varied
one operator with the rest held fixed.

## Frozen Phase-C incumbent

`fe9683e6a9c783bbc6fe276a78c851c6` — artifact digest `c313d1b4081b…`, retained at
`/home/ecs-user/aad-artifacts/autoinit/phase_a/fe9683e6a9c783bbc6fe276a78c851c6`,
with complete three-seed behavioural evidence.

Recommended **with** its caveats: the margin is `0.000070`; it is not
distinguishable from Phase A's leader; it is not a strong benchmark (16/510, near
the floor); **it is not recovered** — no Stage-2/3 training has ever been run;
and none of its component operators is individually validated.

## Roadmap — [`phase_c_roadmap.md`](phase_c_roadmap.md)

* **C0 — protocol/power design, before any probe.** The effects that decided both
  phases are 1–6 correct answers out of 510 and the interval is 6 samples. A
  similarly powered ATTENTION experiment will most likely return
  `unresolved_equivalence` regardless of whether the operator helps. C0 must
  register the minimum effect worth detecting, whether n=510 can detect it, what
  the interval means for an isolation experiment, and priced options if power is
  inadequate.
* **C1 — fixed-path ATTENTION isolation.** Freeze `fe9683`; vary only ATTENTION.
* **C2 — ATTENTION-aware joint re-search.** Only if C1 finds something; must keep
  three anchors so operator improvement stays separable from re-optimized
  composition.

Then FFN, then RESIDUAL_WIDTH; joint confirmation search later; canonical Stage-1
NLL only once the initialization is uniquely selected. **Formal Stage-2/3
recovery remains deferred.**

## Storage

Repo 8.27 GiB · canonical `aad-artifacts` 94.69 GiB · scratch 0.24 GiB · free
10.14 GiB. See [`storage_closeout_20260831.json`](storage_closeout_20260831.json).
One open maintainer decision: two duplicate checkpoint pairs totalling 2.92 GiB,
deliberately not deleted — reasoning in that file.

---

# B. Claude Code — execution session

## Your role

You **implement and execute**. The ChatGPT session is the reviewer and GO / NO-GO
authority. **No Phase-C implementation or compute begins until that review
approves the design.**

## Roots — the only three

| purpose | path |
| --- | --- |
| git repository | `/home/ecs-user/AlphaAvatar-distill` |
| canonical scientific artifacts | `/home/ecs-user/aad-artifacts` |
| scratch | `/home/ecs-user/aad-scratch/sessions/<session-id>` |

**No other AlphaAvatar-distill working root.** The five `$HOME/phase_b_*_scr`
directories were deleted on 2026-08-31 after every byte was proven redundant;
they must not return.

## Entry points

| you want | read |
| --- | --- |
| the scientific history, by phase | [`PHASE_INDEX.md`](PHASE_INDEX.md) — **start here** |
| what Phase A and B concluded | [`phase_a_vs_phase_b_comparison.md`](phase_a_vs_phase_b_comparison.md) |
| the Phase-C structure | [`phase_c_roadmap.md`](phase_c_roadmap.md) |
| live facts, machine-readable | [`current_state.json`](current_state.json) |
| live facts, prose | [`STATE.md`](STATE.md) |
| which log owns which fact | [`CATALOG.md`](CATALOG.md) |
| spend, caps, authorizations | [`BUDGET_LEDGER.md`](BUDGET_LEDGER.md) |
| why things are as they are | [`decisions.md`](decisions.md) |
| where code lives | [`../docs/REPO_LAYOUT.md`](../docs/REPO_LAYOUT.md) |
| how a paid session is specified | [`../docs/SESSION_ARCHITECTURE.md`](../docs/SESSION_ARCHITECTURE.md) |
| the binding working contract | [`../AGENTS.md`](../AGENTS.md) |

## Verify before trusting any of this

```
git rev-parse HEAD && git rev-parse origin/main && git status --porcelain
.venv/bin/python -m pytest tests -q                      # ~26 min
PYTHONPATH=src .venv/bin/python scripts/autoinit/verify_frozen_assets.py
PYTHONPATH=src .venv/bin/python scripts/autoinit/verify_historical_probe_reuse.py
PYTHONPATH=src .venv/bin/python scripts/autoinit/verify_attempt5_probe_reuse.py
PYTHONPATH=src .venv/bin/python scripts/autoinit/verify_attempt4_probe_reuse.py
```

Checkpoints load correctly **only** in the repo `.venv` — a different
`transformers` misreads `rope_theta` by 500× and silently reports a wrong NLL.

## What must not happen without review

* no Phase-C compute, pod, or authorization;
* no rerun of Phase A or Phase B;
* no formal Stage-2/3 recovery training;
* no change to the equivalence interval, ranking metric, science plan, candidate
  universe, finalist set, reuse contracts, or any frozen digest;
* no promotion of `usable_rollout` to a ranking metric without its own decision
  record.

## If a paid session is ever approved

The existing contract is not optional and has caught real defects five times:
preregistration → one-use authorization bound to an executable digest →
authorization-only launch commit (differing from its base by exactly that one
path) → all pre-provider gates green → launch → teardown confirmed by polling the
provider, not by a 200.

Four of the five continuation attempts failed on hardware **after** every `$0`
check had passed. Each failure was a contract inherited from proven machinery
without checking what that machinery required. Read
[`decisions.md`](decisions.md) before assuming a derived session is safe.

## Known open items, none blocking

* two duplicate checkpoint pairs, 2.92 GiB — maintainer decision, see
  [`storage_closeout_20260831.json`](storage_closeout_20260831.json);
* `tests/pod/session_specs.py::SESSION_LAUNCHERS` omits the Phase-B and
  continuation launchers, so neither is covered by the simulator/pod ignore-list
  pin. Recorded as infrastructure debt; the pin also assumes every session shares
  one ignore list, which is no longer true;
* the issuer crashes on a cosmetic `relative_to` when `--out` points outside the
  repository, *after* correctly writing the file.
