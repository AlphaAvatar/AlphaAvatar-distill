**Updated:** 2026-08-21 · branch `main` · after the derived-cache storage cleanup

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
cumulative spend   $209.6842
approved cap       $234.00    RAISED AND APPROVED 2026-08-21
remaining          $24.3158   enough for ONE attempt, $1.2674 margin

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

**The selected-leaf closure is now actually off-pod.** The first version staged
the five leaves on the pod and verified them there, which was useful staging and
not durability: the artifact specs never named `selected_leaves`, the durability
report was not fetched, and `fetch_finalists` returns immediately when Stage 2
did not pass. A Stage-2 failure could still have deleted all five — the exact
class the closure exists to prevent, with every other check green.

What closes it:

| | |
| --- | --- |
| the durability report | now a **fetched report**, ordered before the products that read it |
| the Stage-1 leaf fetch | **not gated on Stage 2** — it runs whenever Stage 1 staged leaves |
| destination | `/home/ecs-user/aad-artifacts/autoinit/phase_a/<state_id>`, the existing store, by the **product `scp` path** |
| not the tarball | the collector keeps archive *and* extracted copy while verifying; five incompressible 1.11 GiB safetensors would roughly double the temporary footprint |
| verification | re-identified **on the dev box** from local bytes; `arch_signature` and `num_parameters` come from the record because no file carries them, everything else is recomputed |
| teardown | `required_products_secured` is in `GATE_ORDER`, and an unreported check counts as False — so it **fails closed** |

`checkpoint_hashes_matched` could not do this job: it is `all([])` and therefore
vacuously true when the fetch returned nothing at all.

### The $0 capacity gate refuses today, correctly

```
destination free   8.30 GiB
five leaves        5.55 GiB   (measured, 5 × 1.110)
+ working room     6.00 GiB   (the suite alone needs ~5 GB of scratch)
= required        11.55 GiB   ->  REFUSED before a pod exists
```

The driver's own headroom check runs on the **pod** and proves only that the pod
can stage them. This one asks whether the destination can hold them, at $0,
before anything is paid for.

### Storage: cleared, and no checkpoint was deleted

```
free   7.61 GiB  ->  22.25 GiB      used 96%  ->  84.5%
reclaimed 14.9107 GiB, entirely from DERIVED caches
```

| removed | GiB | reconstruction |
| --- | ---: | --- |
| `~/.cache/uv` | 7.4181 | PyPI via `uv sync` from the committed lockfile |
| 3 × Qwen3-4B-Thinking weight blobs | 7.4924 | the Hub at revision `768f209d…` |

**The checkpoint inventory had nothing safely deletable.** All ten `review` units
(23.09 GiB) are `local only (no second copy)` and need **paid GPU** to
reconstruct, including the two optimizer states; `aad-artifacts/e5` is
additionally `protected` and unclassified and is **retained pending explicit
classification**. Sixteen tombstones had already retired 3.64 GiB in an earlier
pass.

Safety evidence rather than assertion: of **26,363** venv files, **zero** are
hardlinked to the uv cache (same device, so links were possible); each evicted
blob's sha256 was recomputed from its bytes before removal and matched its cache
name; only one revision is cached; and the tokenizer, configs and shard index
were kept, verified afterwards under `HF_HUB_OFFLINE=1` to still hash
`7781771acc3798ee…`.

Recorded in [`derived_cache_cleanup.json`](derived_cache_cleanup.json) as
**cleanup events, not tombstones** — both caches are recreated by routine
commands, and an active tombstone on such a path is the semantics defect this
project has already fixed twice. The tombstone file is untouched at 16 active /
3.6406 GiB.

The storage inventory now reports filesystem free space. It previously measured
only what the project occupies, so a reclamation outside all four areas moved no
figure at all — and free space is exactly what the new capacity gate decides on.

Untouched: the sibling AlphaAvatar project's Hub caches (~3.8 GB).

### Two things stand between here and Attempt 12

1. ~~Storage~~ — **cleared**: 22.25 GiB free and the capacity gate now passes.
2. Nothing else. The cap is approved at **$234.00**, which funds one attempt with
   $1.2674 of margin and nothing after it.

**Attempt 12 is not prepared, granted or authorized.** It still needs its own
one-use grant and authorization.

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
