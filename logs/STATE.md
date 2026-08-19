**Updated:** 2026-08-19 · branch `main` · after Phase-A attempt 10

# Current state

The **human view of [`current_state.json`](current_state.json)**. That file owns
the live facts; this one says the same things in prose and adds nothing it does
not carry. If the two disagree, a structural test fails.

**Nothing is running. Nothing is billing. Nothing is authorized. Nothing is
prepared for launch.**

Phase-A attempt 10 ran 2026-08-18/19 and was **stopped on maintainer
instruction**: Stage 0 passed, Stage 1's third operator expansion ran 10 h 47 m
without finishing while the paid L40S sat at 0-1 %. **$11.43, incomplete,
operator runtime-cost failure — not a scientific result and not a Stage-1
selection result.**

## Budget

```
cumulative spend   $206.0130
approved cap       $219.00
remaining          $12.9870   NOT permission

**$12.9870 is not enough for another full Phase-A attempt** at the $23.0484
per-launch ceiling. A retry requires a separate budget/cap decision after the
runtime fix is reviewed. Stopping attempt 10 early preserved ~$11.6 of the
ceiling it was authorized to spend.
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

After the bounded Stage-1 device-allocation audit. CPU only — no checkpoint was
loaded and no metric measured:

* full suite **1881 passed, 11 skipped, 0 errors** in 19:48.
* `tests/init/` + `tests/autoinit/` **335 passed**.
* pod simulator **1841 passed, 22 skipped**; artifact tree restored **exactly** —
  1609 entries identical before and after.
* frozen-asset verifier **passed**, no problems.
* Phase-A pricing **$17.8933 / $22.7183 / $23.0483**, reserves 147.7683 and
  36.2158 min — unchanged.
* harness digest **`24d89b9f…` unchanged, 16 files, deliberately not expanded**.

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
| Phase A 10 | $11.4300 | stage 1, 3rd expansion | `depth.causal_kl_greedy_v1`: 260×67 full-vocabulary softmax/KL **on the CPU**, unbounded and unpriced |

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
* autoinit.stage1_device_contract@v1, **category 5 added 2026-08-19**: a fresh
  tensor factory on the Stage-1 path either names a device derived from what it
  meets, or is host-only on purpose and must not be mechanically moved
* placement asserted by intercepting factory calls
  (`tests/autoinit/factory_placement.py`) — the dual of the `HostCacheTensor`
  split; neither instrument can see the other's class
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

**STOPPED FOR REVIEW.** Attempt 10's grant is spent. **Attempt 11 is not
authorized**, and the remaining $12.9870 is both not permission and not enough.

**What is now settled.** The Stage-1 device fixes work on CUDA: the composite
expansion completed and wrote a state, so `stream_projection`'s `avg`, the
orthonormality `eye` and `_head_rows` all placed correctly. Stage 0 has now
passed five times. Setup, staging, the frozen-asset gate, the test gate, artifact
collection, the teardown gate and provider-confirmed teardown all work.

**What failed is a new class.** Attempts 6/7/9 were *placement* — a tensor
allocated without a device. Attempt 8 was a $0 path asserting dev-box filesystem
state. Attempt 10 had correct placement and correct numerics, and failed on
**runtime cost that nothing priced or bounded**:

* `depth.causal_kl_greedy_v1` runs `greedy_removal(36, 8)` = **260 evaluations ×
  67 items = 17,420 forward+distortion pairs**;
* each copies the logits device→host and runs a **151,936-vocabulary softmax/KL
  on the CPU** — ~86 TiB of CPU traffic over the expansion, ~8.6 TiB copied off
  the device. The `.cpu()` is deliberate and documented as memory-spike
  avoidance; its CPU cost was never priced;
* the driver ran **192 threads on a 13-vCPU cgroup** — the setup script computes
  that budget and applies `taskset` **only to the test suite**;
* `--search-minutes 180.0` is an **affordability precheck**, never a runtime
  deadline. One expansion exceeded the whole search budget by ≥3.6×.

How far through the 260 evaluations it reached is **unknown** — `greedy_removal`
journals only on completion.

**The reviewer decides** whether the reduction may move to the device (**not** by
deleting the `.cpu()` calls: the causal-depth scoring and removal rule is frozen
science, so numerical equivalence and identical removal decisions must be shown
first); whether the driver should inherit the cgroup-aware thread budget; whether
`search_minutes` becomes a real deadline checked *inside* an expansion; whether
the cost model should charge the scoring reduction; and the budget question.

**Do not resume** the two attempt-10 search states against a changed numerical
path without demonstrated compatibility. Their weights are gone with the pod;
only specs and hashes survive.

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
