**Updated:** 2026-08-18 · branch `main` · after Phase-A attempt 9

# Current state

The **human view of [`current_state.json`](current_state.json)**. That file owns
the live facts; this one says the same things in prose and adds nothing it does
not carry. If the two disagree, a structural test fails.

**Nothing is running. Nothing is billing. Nothing is authorized. Nothing is
prepared for launch.**

Phase-A attempt 9 ran on 2026-08-18: **Stage 0 passed and attested, Stage 1
failed**, $0.34, pod deleted and provider-confirmed gone. Its grant is spent.

## Budget

```
cumulative spend   $194.5830
approved cap       $219.00
remaining          $24.4170   NOT permission
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
* Stage 0 on hardware, passed four times — latest attempt 9, attesting
  protocol `250f72ef`, identity `70a26e0b`, science plan `02be33b9`

## Latest verification

After the $0 fix for the two assertions that failed Phase-A attempt 8's setup
test gate. CPU only — no checkpoint was loaded and no metric measured:

* full suite **1874 passed, 11 skipped, 0 errors** in 19:32.
* pod simulator **1834 passed, 22 skipped**; artifact tree restored **exactly**
  — 1598 entries, identical type/size/path before and after.
* focused docs/storage/session **395 passed, 5 skipped**.
* frozen-asset verifier **passed**, no problems.
* Phase-A pricing reproduces **$17.8933 / $22.7183 / $23.0483** exactly, reserves
  at 147.7683 and 36.2158 min.
* **the pod's own condition reproduced**: with `artifacts/stage3/ladder_uniform`
  present, `test_no_tombstoned_path_is_still_on_disk` now passes. The directory
  was removed again afterwards.

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
| Phase A 8 | $0.1900 | setup / test gate | two dev-box-environment tests that cannot pass in a container |
| Phase A 9 | $0.3400 | stage 1 | `stream_projection`'s `avg` allocated with no device (`project.py:57`) |

**The pattern, through attempt 7:** code no $0 path could execute — a GPU-only
device, or a contract owned by inherited machinery. **Attempt 8 is a new
pattern:** a $0 path that *does* run on the dev box and asserts **dev-box
filesystem state**, so the pod simulator passes it for the wrong reason.
Details in [`autoinit_phase_a_attempt8/`](autoinit_phase_a_attempt8/). Twice the symptom was generalized
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
* the layout test partitions its references: repository-relative paths must
  exist, declared host-local storage roots are verified where present and
  **skipped where absent**, and an undeclared absolute path still fails
* an active tombstone may not name a declared session staging or mirror
  destination — checked against the four `SessionSpec`s, touching no path, so the
  dev box, the simulator and a pod all give the same answer
* `checkpoint_tombstones.json` owns the active-tombstone counts and bytes
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

**STOPPED FOR REVIEW.** Attempt 9's grant covered one launch and is spent; its
lineage gate refuses the current HEAD. **No attempt 10 is authorized, funded or
prepared**, and none may be inferred from the $24.4170 of unused headroom.

**What worked.** Setup, the manifest-driven relay staging, the pod-side
frozen-asset gate, and — for the first time — the blocking test gate that killed
attempt 8. Stage 0 passed and attested all three frozen identities. Artifact
collection, the teardown gate and provider-confirmed teardown all worked, and the
stage-1 traceback came home, which is what made this diagnosis free.

**What failed.** `src/aadistill/init/project.py:57` allocates the projection
accumulator with a dtype and **no device**, so it lands on CPU while
`uncentered_moment` follows `state` — CPU on the dev box, `cuda:0` on a pod:

```
RuntimeError: Expected all tensors to be on the same device,
              but found at least two devices, cuda:0 and cpu!
    project.py:60   avg += w * (m / m.trace())
```

**It is the third of its kind.** Attempt 6 ($0.3552) the `_validate` probe,
attempt 7 ($0.3955) the `ActivationStatsCollector` accumulators, attempt 9
($0.3400) this. `autoinit.stage1_device_contract@v1` closed the first two and did
not reach an accumulator two call levels below the operator. Every one is a
tensor allocated without a device in a path only a GPU executes, and **a CPU
rehearsal cannot see the class by construction** — on CPU both operands agree and
the arithmetic is right.

**The defect is not fixed.** Whether to change one line or audit every allocation
on the Stage-1 path, whether the device contract should assert *placement* rather
than arithmetic, and whether a $0 GPU-free placement check (meta device,
fake-tensor pass) is possible at all, are the maintainer's decisions.
Details in [`autoinit_phase_a_attempt9/`](autoinit_phase_a_attempt9/).

A paid run needs the full chain: **write a grant document** -> pre-auth base
commit -> issue against that grant -> authorization-only commit -> bundle upload
-> round-trip verify -> launch.

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
