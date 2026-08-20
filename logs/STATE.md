**Updated:** 2026-08-20 · branch `main` · after Phase-A attempt 11 — **Stage 1 PASSED**

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
approved cap       $231.00    RAISED AND APPROVED 2026-08-20
remaining          $21.3158   $1.7326 SHORT of one more full attempt

**The cap was raised from $219.00 to $231.00 on 2026-08-20** to fund exactly one
Phase-A attempt. That attempt has run and cost **$3.2101** — far under its
$23.0484 ceiling, because it stopped at Stage 2 rather than training nine probes.
**$21.3158 remains, which is $1.7326 short of another full-ceiling attempt**, and
the approval says in terms that it "does not authorize any subsequent attempt".
Attempt 12 is not funded, authorized, prepared or implied.
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

**Phase-A attempt 11 ran: Stage 0 and Stage 1 PASSED, Stage 2 failed closed.**
$3.2101, 194.5 min, pod deleted with provider confirmation. Grant and
authorization **consumed**. Full record:
[`autoinit_phase_a_attempt11/`](autoinit_phase_a_attempt11/).

### The first completed AutoInit search

```
43 states · 4 levels · 7 complete leaves · 18 pruned · 180.3 min
5 leaves selected, each 596,049,920 parameters
```

Each leaf is a distinct four-operator composition. **The existing
`COMPOSITE_STAGE1` recipe lands on Pareto front 4 and is not selected** — four
search-discovered orderings dominate it. That is the first evidence this project
has that operator *ordering* carries signal, which is the question Phase A was
built to ask. The canonical control was injected and verified against its frozen
hash.

**The leaf weights are gone.** Finalists are fetched after Stage-5 selection,
which never ran, so the five checkpoints died with the pod. The *record* survives
— `search_result.json`, `search_states_reduced.jsonl`, and the 25 MB full journal
out of tree — but regenerating the weights costs another ~180 min of search.

### Two findings from the run itself

**The reference cache fell back all four times.** 16.9 GiB does not fit in 66% of
the **20.3 GiB** free *inside* the search; Measurement Attempt 3 saw 36.42 GiB
free standalone. Four causal-depth invocations ran at 6.96–10.79 eval/min against
the standalone 12.07, taking 122.1 min — **68% of the whole search**. The
measurement was not wrong; it measured a different memory regime, and the cached
path may simply not be reachable inside the search at this teacher size.

**The deadline fix was load-bearing by 17 seconds.** Stage 1 took 180.283 min
against the 180.0 bound it would have been killed at before `16e382f`. That
commit is the only reason this search produced a result.

### The open defect, diagnosed at $0 and NOT fixed

```
train_stage3.py:173  ValueError: teacher and student tokenizers differ
```

`Qwen3Adapter.save()` calls `save_pretrained()`, which writes weights and config
and **no tokenizer files** — correct for the search, which consumes pre-tokenized
items. Stage 2 then points a probe at that directory, and
`AutoTokenizer.from_pretrained()` **does not raise**: it returns a **one-token
vocabulary**. The teacher/student equality guard caught it; without that guard the
probe would have trained against a 1-token tokenizer and produced numbers.

Reproduced exactly at $0. The canonical init's own tokenizer is fine — identical
hash to the teacher, 151,669 tokens, zero differences. What is missing is any
step that carries those files into a *searched* leaf.

**This is the `control_sb` class again**: identity gates all pass while the
checkpoint cannot be used, because the gates check what the producer needs rather
than what the consumer needs. Every leaf passed `materialize → reload → hash →
validate` and reached `MEASURED`; none of those asks whether a trainer can load a
tokenizer.

**No fix is applied.** The run stopped for review, as the grant requires.

### Two decisions owed, and nothing is prepared

1. **How tokenizer files reach a searched leaf** — and whether
   `AutoTokenizer`'s silent 1-token fallback should be refused at the producer,
   the consumer, or both.
2. **Budget.** $21.3158 remains, **$1.7326 short** of another full-ceiling
   attempt. The cap raise funded one attempt, it has run, and the approval says
   it "does not authorize any subsequent attempt".

**No Attempt 12 is prepared, granted, funded or implied.**

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
