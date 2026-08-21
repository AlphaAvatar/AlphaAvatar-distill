**Updated:** 2026-08-21 · branch `main` · the recovery continuation is an executable, separately authorized session

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

```

The cap went **$219.00 → $231.00** (2026-08-20, to fund exactly one Phase-A
attempt) **→ $234.00** (2026-08-21). Attempt 12 has since **run**, costing
**$3.7872** — far under its $23.0484 ceiling, because it stopped at Stage 2
rather than training nine probes. **$20.5286 remains, $2.5198 short of another
full-ceiling attempt.** The recovery continuation's derived **$16.7456** ceiling
does fit, with $3.78 to spare.

Each raise is a **project ceiling only**. It authorizes nothing: the next paid
action needs its own one-use grant and its own authorization artifact, and
remaining balance has never been permission.

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

After giving the recovery continuation its own authorization type, harness digest
and issuer, and closing the three **pod-side** consumers still wired to the
Phase-A artifact. CPU only — no checkpoint loaded, no metric measured, no GPU
used:

* full suite **2098 passed, 11 skipped, 0 errors** in 18:48.
* pod simulator **2057 passed, 22 skipped**; artifact tree restored **exactly** —
  1168 entries, listing hash `c1726a62…` before and after.
* frozen-asset verifier **passed**, and was **not** weakened or rescoped.
* **13 mutations**, each a passing state made to fail: the four executables the
  new digest must cover, the search whose identity it must *not* follow, the
  schema refusal, the file-set substitution, the derived ceiling, the grant type,
  the driver's artifact, the setup branch and `SESSION_KIND`.
* the setup branch is verified by **running the real shell block** with a real
  artifact — text-matching it proves only that it was typed.

**Run it in the repo `.venv`** (transformers 5.13.1). The AlphaAvatar venv's
4.57.1 fails six tokenizer tests on this tree and is not the canonical
environment; a run there reads as seven failures that are not real.

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
* one session, one authorization **type**, refused across by schema — and one
  harness set naming exactly what that session executes. Full Phase A and the
  recovery continuation are distinct operational harnesses, measured
  independently; a shared executable belongs to both sets by derivation, never by
  a second hand-maintained copy
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

**The recovery continuation is now an executable session.** The previous commit
shipped only the primitives — the production path still priced with `budget()`,
launched `--stage all` against the full driver, and ran the search
unconditionally. Authorizing it would have rerun the 203-minute search the work
existed to avoid. **Nothing is running, billing, authorized or prepared.**

### What changed

| | |
| --- | --- |
| **search unreachable** | the continuation driver never imports `phase_a_search`; the frozen identities moved to `phase_a_frozen.py` (same values, re-exported) so they can be bound without the search in reach. `stage1` is overridden and never delegates; no `--stage` value searches. Three AST tests enforce it |
| **priced by the derivation** | `continuation_budget(args)` — **904.44 min, `$14.9233` expected, `$16.7456` hard**, no Stage-1 phase, no Stage-1 reserves. Restoring `budget()` breaks ten tests |
| **five leaves as inputs** | declared by state id **in the selected order** with attempt 12's digests, read from the committed record (a test forbids hard-coding them), staged via `SESSION_ASSETS`, re-identified **from bytes** by a `$0` precheck before a pod exists |
| **the strict importer is used** | `import_stage1_result()` on the staged bytes; a test forbids a second reconstruction in the driver |
| **the control is measured** | once, on the frozen suite, through the same evaluator and primed teacher; attached hash-bound, persisted, and put through the same `admit_leaves` gate |
| **handoff before Stage 2** | teacher and evaluator dropped explicitly, `release_to_subprocess` measures what that freed, headroom required. It costs nothing here because this session never held the search |
| **its own authorization** | schema `recovery_continuation_authorization/v1`, harness digest over **22** files — the Phase-A set minus the unreachable search, plus this launcher, driver, `stage1_import`, `device_handoff`, `leaf_durability`. Refused across from Phase A **by schema**. Ceiling derived from `continuation_budget()`, never written |

### The authorization measured the wrong executable

`PHASE_A_HARNESS_SOURCE_FILES_V1` contains **neither continuation file**, so
issuing with `--out logs/autoinit_recovery_continuation_…` would have produced a
green digest over code this session does not run, while the launcher, driver and
strict importer it *does* run went unmeasured — and carried the search's
`$23.0484` ceiling into a session priced at `$16.7456`. The two are now distinct
operational harnesses, independently measured: the continuation set is **derived**
from the Phase-A set (minus search, plus its own), so the fourteen shared paths
cannot drift between two hand-maintained copies. Phase A's set was **not**
broadened.

**Three pod-side consumers were still wired to the Phase-A artifact**, found by
asking who else loads it — none of them reachable from the launcher tests:

* `PhaseADriver.__init__` loaded a hard-coded
  `logs/autoinit_phase_a_authorization.json`. That file **is committed**, holding
  attempt 12's consumed `$23.0484` authorization, so the continuation would not
  have crashed — it would have enforced `require_within_cap` against the search's
  ceiling, **38% too high**, and recorded the wrong grant as what authorized the
  run. The artifact and type are now class attributes; the continuation overrides
  both.
* the shared setup script selects its loader on `SESSION_KIND`, and the
  continuation **declared none** — falling to `spend`, whose `SpendAuthorization`
  refuses any artifact asserting `phase_a_authorized`. The session would have
  died at setup, **exit 98**, before any work. A third branch now loads the
  continuation type; the session declares the kind and requires it.
* `required_env` omitted `SESSION_KIND` entirely.

**Mutation-verified 13 ways.** Two initially *passed*: the schema-refusal test was
satisfied one check later by a different refusal, so it was not measuring what it
named; and `as_dict` wrote `allows_beam_search` as a literal rather than from the
property. The setup branch is exercised by **running the real shell block**, not
by matching its text.

### The entrypoint is exercised, not just its helpers

Which is the whole point of this correction, so the tests run the real
`stage1()` on the real preserved bytes: five leaves imported in the ranking's
order, all `MEASURED`, control measured and admitted, handoff recorded, all three
evidence files written. A second test symlinks one leaf's bytes under another's
id and requires the entrypoint to refuse.

**One hole found by mutation and closed:** the preserved-leaf precheck was tested
as a function but never asserted to be *in* `spec.precheck` — dropping it passed
every other test. The same helper-versus-wiring gap that produced this
correction, one level down in my own tests.

### Frozen science untouched

Session `9377a2dc…`, science `02be33b9…`, recovery fingerprint `ab0d8cfd…`,
tokenizer `7781771a…`. The continuation carries the **same** `plan_hash` with a
distinct operational `session_id`; nothing was rewritten to pretend Phase A
always began at Stage 2. Full Phase-A pricing unchanged at `$17.8933 / $22.7183 /
$23.0483`.

### Budget

Cap **$234.00**, spent **$213.4714**, remaining **$20.5286**. The continuation's
`$16.7456` hard fits with **$3.78** to spare. No cap increase requested.

**The next review is a GO/NO-GO for one recovery continuation under the derived
`$16.7456` ceiling — not another Phase-A search.**

**Process rule, standing:** helpers existing is not the same as the path using
them, and "my intended next decision is GO" is not executable permission.

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
