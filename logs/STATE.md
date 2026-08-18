**Updated:** 2026-08-18 · branch `main` · structural baseline

# Current state

The **human view of [`current_state.json`](current_state.json)**. That file owns
the live facts; this one says the same things in prose and adds nothing it does
not carry. If the two disagree, a structural test fails.

**Nothing is running. Nothing is billing. Nothing is authorized. Nothing is
prepared for launch.**

## Budget

```
cumulative spend   $194.0530
approved cap       $219.00
remaining          $24.9470   NOT permission
```

Every authorization issued is **consumed** — each one's lineage gate refuses the
current HEAD by construction. A new paid action needs a new artifact. Detail in
[`BUDGET_LEDGER.md`](BUDGET_LEDGER.md).

## Frozen — do not change without an explicit decision

| | identity |
| --- | --- |
| science plan | `02be33b9a7a8e26bc8bfb75795351e8cdc9ffd441b47066cc81887cfc511b55c` |
| session plan | `9377a2dc61f21790dd111d72a5de0e039ea1d31afef2d09e18c98a0b0cc2a0aa` |
| Stage-3 evaluation protocol | `250f72efbd43b86a475e8dda293b45f07ee61a4d858e147f4a5bd7681c32c2e4` |
| equivalence interval | `0.011695296982299022` |
| feasibility floor | `0.3` |
| seeds | sa `20260726`, sb `20260801`, sc `20260813` (conditional); **no fourth** |
| calibration | `calib.domain_balanced@v1 (67 items, 59,763 positions)` |
| runtime comparability | `generation_runtime_comparability@v2` |

Also frozen: recovery design, selection rules, pooled_counts@v2, Stage-3 artifacts and thresholds, the operator ledger's declared semantics.

## Complete

* Stage 3 (both permanent controls, thresholds materialized)
* the Phase-A harness, stages 0-5 executing for real at $0
* the Stage-1 device audit (autoinit.stage1_device_contract@v1)
* Stage 0 on hardware, passed three times

## Latest verification

After making the session manifest authoritative for relay science-input staging.
CPU only — no checkpoint was loaded and no metric measured:

* full suite **1872 passed, 11 skipped, 0 errors in 19:20 (the 1842 baseline plus the 30 new relay-contract gates)**.
* pod simulator **1832 passed, 22 skipped in 2:53; artifact tree restored EXACTLY - 1591 entries, identical type/size/path before and after**.
* frozen-asset verifier **passed**, no problems.
* Phase-A pricing reproduces **$17.8933 / $22.7183 / $23.0483** exactly, with
  both soft-stop reserves at their derived minutes — the cheapest proof the
  change preserved behaviour.

**The canonical full-suite baseline is `1842 passed, 11 skipped, 0 errors`**
under the repo `.venv`. The `1837 passed, 5 errors` recorded before was the same
1842 tests under the sibling AlphaAvatar venv, whose transformers 4.57.1 raises
on a toy tokenizer in `tmp_path` before any repository artifact is read.

## What failed, and why

All five are the same two classes, and both are now closed by construction:
GPU-only device code (fixed by `autoinit.stage1_device_contract@v1`) and a
contract owned by inherited machinery (fixed by the session specification).

| run | cost | died at | cause |
| --- | ---: | --- | --- |
| Phase A 1-5 | $1.6321 | setup / stage 0 | five distinct fail-closed gates; all fixed |
| Phase A 6 | $0.3552 | stage 1 | _validate probe on config.device, child on the host |
| Phase A 7 | $0.3955 | stage 1 | ActivationStatsCollector accumulators unplaced |
| canary 1 | $0.0603 | before setup | wrapper missing 3 inherited self.a attributes |
| canary 2 | $0.0637 | in setup | wrapper set LOCAL_ASSETS = (); shared setup copies them |

**The pattern:** every one was code no $0 path could execute: a GPU-only device, or a contract owned by inherited machinery. Twice the symptom was generalized
instead of the cause, and it was paid for twice. Full diagnoses in
[`decisions.md`](decisions.md); per-run evidence in the directories
[`CATALOG.md`](CATALOG.md) lists.

## Canonical infrastructure

* **the session specification**: one immutable `SessionSpec` per session, one
  runner, no inheritance and no module-global retargeting
  ([`../docs/SESSION_ARCHITECTURE.md`](../docs/SESSION_ARCHITECTURE.md))
* the shared pod setup is manifest-driven on **both** sides: it installs the
  local assets a session declares and stages the relay science inputs a session
  declares, and names no asset, relay path, destination or digest of its own.
  `RelayInput` carries source, destination, digest and mirror; `dest=None` means
  only "the driver stages this", never "the shell knows"
* the Phase-A authorization schema carries no grant; the issuer requires one
* autoinit.stage1_device_contract@v1
* full tracebacks for unexpected in-process driver exceptions
* plan_session soft_stop_reserves, applied BEFORE the soft stop
* the pre-flight rehearsal ignored in both the pod gate and the simulator, with the ignore lists pinned equal
* the pod simulator restores exactly and refuses concurrent sweeps

## Abandoned / terminated — do not revive

* recovery_search_v1 (INVALID before first use)
* Phase-A pricing bases $20.0126 and $22.4508
* student-prefix recovery (E5)
* any fourth seed
* the paid device-canary session path: STRATEGICALLY TERMINATED 2026-08-18. Two authorized sessions, $0.1240, zero canary runs; both died in the wrapper's inherited contracts. Its evidence and its generic lesson are kept; no further canary is prepared.

## Next starting point

**Nothing is authorized and nothing is prepared.** The maintainer chooses.

The session architecture is **implemented** (2026-08-18), so the defect class
that cost three paid pods is gone by construction rather than by care. What is
left is a paid decision:

1. **Phase A, attempt 8.** $23.0484 hard, unchanged — the price reproduces
   exactly after both the session-architecture refactor and the relay-staging
   fix. Stage-1 failures now come home with a traceback, and the
   inherited-contract failures cannot recur on either the local-asset or the
   relay side.

   **The setup-contract readiness question is closed.** A seven-property audit on
   2026-08-18 asked whether the shared setup was really driven by the session
   declarations; it failed on all seven, the narrow fix landed at $0, and every
   new gate is mutation-verified ([`decisions.md`](decisions.md)). The harness
   digest moved, which invalidates nothing — every authorization was already
   consumed. What remains is the GO / NO-GO decision.
2. the device canary stays TERMINATED. Its launcher is converted, which is what
   makes the point: it declares no local assets and now gets none. Reviving it
   would be a new decision, not a resumption.

A paid run needs the full chain, and it now starts one step earlier: **write a
grant document** -> pre-auth base commit -> issue against that grant -> auth-only
commit -> bundle upload -> round-trip verify -> launch. There is no grant
document, so `issue_phase_a_authorization.py` refuses.

## Where else to look

| you want | read |
| --- | --- |
| which log owns which fact | [`CATALOG.md`](CATALOG.md) |
| what storage exists, where, and how much | [`storage_measurements.json`](storage_measurements.json) |
| which checkpoints exist, in which of the three stores | [`checkpoint_registry.json`](checkpoint_registry.json) |
| what was deleted, and how to get it back | [`checkpoint_tombstones.json`](checkpoint_tombstones.json) |
| where code lives | [`../docs/REPO_LAYOUT.md`](../docs/REPO_LAYOUT.md) |
| how a pod session is specified and run | [`../docs/SESSION_ARCHITECTURE.md`](../docs/SESSION_ARCHITECTURE.md) |
| which pod script is live, historical or terminated | [`../docs/POD_SCRIPTS.md`](../docs/POD_SCRIPTS.md) |
| AutoInitializer binding rules and pinned assets | [`../docs/AUTOINIT_REFERENCE.md`](../docs/AUTOINIT_REFERENCE.md) |
| why things are the way they are | [`decisions.md`](decisions.md) |
| what each experiment proved | [`EXPERIMENT_INDEX.md`](EXPERIMENT_INDEX.md) |
| the working contract for agents | [`../AGENTS.md`](../AGENTS.md) |
