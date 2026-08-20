**Updated:** 2026-08-20 · branch `main` · after the Stage-1 deadline/pricing alignment

# Current state

The **human view of [`current_state.json`](current_state.json)**. That file owns
the live facts; this one says the same things in prose and adds nothing it does
not carry. If the two disagree, a structural test fails.

**Nothing is running. Nothing is billing. Nothing is authorized. Nothing is
prepared for launch.**

**Measurement Attempt 3 COMPLETED on 2026-08-20 for $0.2077.** `ALL_DONE`, both
fail-closed conditions passed, pod deleted with provider confirmation. The
repaired causal-depth port reaches **12.07 weighted evaluations/min** against
E8a's frozen **12.0/min** anchor, and agrees with E8a **exactly** — per-item KL
delta 0.0 at both |skip|=1 and |skip|=8. **These values authorize nothing.**

Phase-A attempt 10 ran 2026-08-18/19 and was **stopped on maintainer
instruction**: Stage 0 passed, Stage 1's third operator expansion ran 10 h 47 m
without finishing while the paid L40S sat at 0-1 %. **$11.43, incomplete,
operator runtime-cost failure — not a scientific result and not a Stage-1
selection result.**

## Budget

```
cumulative spend   $206.4741
approved cap       $231.00    RAISED AND APPROVED 2026-08-20
remaining          $24.5259   affordability, NOT permission

**The cap was raised from $219.00 to $231.00 on 2026-08-20**, as a *cumulative
project ceiling only*: it does not change the $23.0484 per-launch hard ceiling,
does not authorize any subsequent attempt, and does not authorize spend outside
the next explicitly approved session. One full hard-ceiling attempt reaches
$229.5225, leaving $1.4775 of margin. It funds **one** Phase-A Attempt 11 and
nothing else — Attempt 12 is not funded, authorized, prepared or implied.
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

After moving the measurement entrypoint behind an executable seam and separating
teardown truncation from a missing artifact. CPU only — no checkpoint loaded, no
metric measured, no GPU used:

* full suite **1958 passed, 11 skipped, 0 errors** in 16:51.
* pod simulator **1918 passed, 22 skipped**; artifact tree restored **exactly** —
  1167 entries, identical listing hash before and after, hide directory empty.
* frozen-asset verifier **passed**, and was **not** weakened or rescoped.
* entrypoint seam mutation-verified **8 ways** across the seam, the loader and
  the resolver. Two mutations initially *passed* and exposed the seam's own blind
  spot — its injection points — which is how the calibration-path defect was
  found; `load_teacher` and `resolve_calibration` now have their own tests.
  `Hardware`'s three `torch.cuda.*` calls stay $0-uncoverable and are recorded
  as such.
* teardown gate mutation-verified **3 ways**, including that a caller supplying
  no evidence still gets the strict fail-closed rule.

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
| measurement 1 | $0.0700 | setup / frozen-asset gate | declared `LOCAL_ASSETS = ()`; the shared setup verifies both frozen roots unconditionally |
| measurement 2 | $0.1834 | driver entrypoint, after `SETUP_RC=0` | `main()` imported `as_operator_items` from the wrong module — and **no $0 test called `main()`** |

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

**Phase-A Attempt 11 is granted and being prepared.** The maintainer's GO of
2026-08-20 covers exactly one launch through the frozen Stage 0-5 protocol,
against reviewed implementation `16e382f`, following the approved cap raise to
$231.00. Grant:
[`autoinit_phase_a_attempt11_grant.json`](autoinit_phase_a_attempt11_grant.json).

**At this commit nothing is authorized and nothing is launchable**: the
authorization does not exist yet. It is issued against this tree in the next
commit, which the lineage gate requires to differ from this one in nothing but
the authorization artifact. That is why this snapshot still reads `authorized:
false` — it describes the pre-authorization base, which is what a pod checks out.

### What attempt 10 left open, and how it was closed

| blocker | closure |
| --- | --- |
| causal-depth scored on the host: $11.43, GPU at 0–1%, no expansion finished in 10 h 47 m | reduction returned to the accelerator, CPU equivalence proved at $0, then **measured**: 12.07 eval/min vs the 12.0 anchor, per-item KL delta **0.0**, GPU 98.3% |
| the runtime deadline was the 180-min base while the priced envelope is 363.9841 min | deadline now **derived** from the same `BudgetPlan` as the dollar figures; mutation-verified 7 ways; pricing and identities unchanged |

The cost model is **measured-confirmed, not replaced** — the causal-depth
operator is not repriced, and beam width, search space, operators, calibration,
seeds and the recovery/selection rules are untouched.

### Pricing and bounds for this launch

```
expected            $17.8933      soft stop    $22.7183
priced hard         $23.0483      ceiling      $23.0484
stage-1 deadline    363.9841 min  = 180.0000 + 36.2158 + 147.7683
```

Worst case takes the project to $229.5225 of $231.00. **The cap raise funds this
attempt and nothing after it**: Attempt 12 is not funded, authorized, prepared or
implied, and must not be inferred from whatever headroom remains.

**Process rule, standing:** "my intended next decision is GO" is not executable
permission, and a recommendation is not an approval.

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
