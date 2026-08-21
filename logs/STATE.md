**Updated:** 2026-08-21 · branch `main` · Stage 1 is now importable; the continuation is implemented at $0

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
cumulative spend   $213.4714
approved cap       $234.00    RAISED AND APPROVED 2026-08-21
remaining          $20.5286   $2.5198 SHORT of one more full attempt

**The cap was raised from $219.00 to $231.00 on 2026-08-20** to fund exactly one
Phase-A attempt. That attempt has run and cost **$3.2101** — far under its
$23.0484 ceiling, because it stopped at Stage 2 rather than training nine probes.
**$21.3158 remains, which is $1.7326 short of another full-ceiling attempt**, and
the approval says in terms that it "does not authorize any subsequent attempt".
A cap of **$234.00** is **recommended and NOT approved** — one more full-ceiling
launch reaches $232.7326, so $234.00 leaves $1.2674 of margin. That is a project
ceiling only: the $23.0484 per-launch ceiling is unchanged, Attempt 12 still
needs its own one-use grant and authorization, and $234.00 would not authorize an
Attempt 13. Attempt 12 is not funded, authorized, prepared or implied.
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

**Stage 1 is a result to import, not work to repeat.** Attempts 11 and 12 agree
byte for byte, and attempt 12's five leaves are preserved and digest-verified. A
recovery continuation starting at Stage 2 is implemented and verified at $0.
**Nothing is running, billing, authorized or prepared.**

### What was built

| | |
| --- | --- |
| **device handoff** | records **allocated / reserved / driver-free** either side of a release, and derives the verdict from `allocated` — high reserved with low allocated is "allocator reservation, not a model leak". Headroom is checked against the **driver's** free bytes, since a sibling cannot use the parent's cached blocks. Placed **after** durability, so a failure costs a diagnostic and never a completed search |
| **strict Stage-1 import** | binds the config hash; requires the five ids **in order**; re-identifies every checkpoint **from bytes** against both digests; requires target geometry and an artifact-bound evaluation; rebuilds the control from its frozen hash; feeds the same `admit_leaves` a live search feeds |
| **no deserializer** | there is deliberately no `InitializationState.from_dict`, and a test asserts none appears. The journal is evidence, not a trusted format |
| **derived budget** | `904.44` min, **`$14.9233` expected / `$16.7456` hard**, from the Phase-A `BudgetSpec` minus the Stage-1 phase and both Stage-1-only reserves. No dollar figure is written in the code |

### One open item, before any GO

**The control comes back unmeasured, and cannot be otherwise.** Its Stage-1
evaluation is not in the persisted evidence — no `retained_canonical` row in the
journal, and `search_result["control"]` carries identity only — because a live
search measures it on the GPU. A continuation must measure it on the suite, so
the teacher and suite still have to be resident. One evaluation against a
203-minute search, but not free. The admission gate refuses the control until it
is measured, which is what makes that unskippable rather than forgotten.

### The orchestration caught what the fast tests could not

The first handoff dropped `found` and then read `found.summary` four lines below
— a `NameError` **after** a 203-minute search had already succeeded, at exactly
the boundary added to protect it. Every focused suite passed; only the real
Stage-0→5 run reaches that line. A test now asserts **zero** reads of `found`
after the release, so the bug fails in 0.25 s instead of 21 minutes.

### Frozen science untouched

Session `9377a2dc…`, science `02be33b9…`, recovery fingerprint `ab0d8cfd…`,
tokenizer `7781771a…`, Stage-1 deadline `363.9841 min`, Phase-A price
`$17.8933 / $22.7183 / $23.0483`. Nothing was rewritten to pretend the session
always began at Stage 2.

### Budget

Cap **$234.00**, spent **$213.4714**, remaining **$20.5286**. The continuation's
`$16.7456` hard fits with **$3.78** to spare — no cap increase is needed, and
none should be requested to rerun a completed search.

**The next review is a GO/NO-GO for the recovery continuation, not for another
Phase-A search.**

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
