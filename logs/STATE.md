**Updated:** 2026-09-02 · branch `main` · **PHASE B CLOSED · PHASE C0 FROZEN ·
PHASE C1 RUNNABLE + PRICED, NOT EXECUTED, NOT AUTHORIZED**

# Current state

The **human view of [`current_state.json`](current_state.json)**. That file owns
the live facts; this one says the same things in prose and adds nothing it does
not carry. If the two disagree, a structural test fails.

**Nothing is running. Nothing is billing. Nothing is prepared for launch. Nothing
is authorized.** Zero pods, zero orchestration, all five behavioural-continuation
grants retired. Spend is `$263.8597` of the `$283.7600` cap, `$19.9003`
remaining.

| phase | status |
| --- | --- |
| Phase A | **COMPLETE / FROZEN** |
| Phase B | **COMPLETE / RESOLVED / FROZEN** |
| Phase C0 | **COMPLETE / APPROVED / FROZEN** |
| Phase C1 | **IMPLEMENTED / PRICED / NOT EXECUTED / NOT AUTHORIZED** |
| Phase C2 | **NOT STARTED** |
| formal recovery evidence | **NONE** |

**Phase C1 is PRICED and UNAUTHORIZED, and has consumed no compute.** Floor
**$12.2070** · expected **$13.4277** · enforceable one-attempt ceiling
**$13.7578**, against **$19.9003** headroom — **$6.1425** would remain if the
whole ceiling were consumed. **Priced is not authorized: no grant exists.**

## Phase C1 — RUNNABLE, PRICED and PREREGISTERED, NOT EXECUTED

> **Both 2026-09-02 blockers are closed.** The driver is now standalone — it
> neither subclasses nor imports the Phase-A driver, owns stages B–I, binds the
> C1 plan hash, writes only C1 paths, keeps training and evaluation physically
> separate, and enforces the CUDA release/headroom gate before probe 1. And the
> scoring blocker is resolved by **option 3**: `recovery_search_scoring@v2`
> cannot run on this battery, so C1 declares `c1_confirmation_scoring@v1` and
> leaves both frozen assets untouched. That the numbers did not move is
> demonstrated, not asserted — 15 retained probes scored through both paths,
> **IDENTICAL, 0 differences**. See
> [`decisions.md`](decisions.md) and
> [`phase_c1_scoring_equivalence.json`](phase_c1_scoring_equivalence.json).

Design and implementation proceeded at `$0`; **execution did not**. What exists:

| built | where |
| --- | --- |
| `attention.activation_importance_v1` | `autoinit/operators/attention_activation.py` — its **own** module. **Import is inert**; a consumer calls `register()`. Staying out of `V1_IMPLEMENTATIONS` is not enough on its own, because an unrestricted beam enumerates the whole registry |
| fixed-path executor with the fail-stop digest gate | `autoinit/fixed_path.py` |
| two-arm plan, seed rule, paired bootstrap, 3-way decision | `autoinit/c1_isolation.py` |
| confirmation battery `c1_confirmation_v1`, 950/850 | out of tree; identity in [`phase_c1_battery.json`](phase_c1_battery.json) |
| teacher shard binding | [`phase_c1_teacher_binding.json`](phase_c1_teacher_binding.json) |

| ten-stage session contract + the two fail-stop replay gates | `autoinit/c1_session.py` |
| execution preregistration | [`phase_c1_execution_preregistration.json`](phase_c1_execution_preregistration.json) · `54d858ed1dfe1939…` at `head_commit 913a88a` · supersedes `bc48515d…` ← `63ca7c24…` ← `58015cd3…` ← `48beff49…` |
| price bound | [`phase_c1_pricing.json`](phase_c1_pricing.json) · floor **$12.2070** · expected **$13.4277** · **ceiling $13.7578** |
| evidence declaration | [`configs/autoinit/c1_artifacts.json`](../configs/autoinit/c1_artifacts.json) + [`_failed`](../configs/autoinit/c1_artifacts_failed.json) — inside the measured harness, so editing what survives teardown moves the digest a grant binds |
| standalone session | `scripts/pod/autoinit_c1_launch.py` + `autoinit_c1_driver.py` · `SESSION_KIND=c1` · `C1Authorization` · 8 pre-provider gates · no Phase-A driver or launcher in the closure |
| Stage-H admission | no probe is scored unless the protocol observed from **its own** raw summaries is comparable to the attested one; on drift the session stops `C1_INCOMPLETE` before scoring, and no later probe is evaluated |
| C1 scoring binding | `c1_confirmation_scoring@v1` · `77507935f21f83eb…` over 11 files · parent `recovery_search_scoring@v2` `808080a7…`, unchanged · equivalence **IDENTICAL / 15 cases / 0 differences** |

The three fresh seeds are **`1635674081`, `1656475568`, `696460635`**, derived
from the frozen C0 digest and confirmed independently three ways.

**No grant exists.** The only C1 authorization artifact is a *candidate* at a
scratch path outside the repository, structurally valid so the real launcher,
loader, `BudgetSpec` and pre-provider gates can run at `$0`. It is not live and
is never committed.

Still absent, both requiring GPU or network: the pre-ATTENTION parent
`b8820f41d062` rebuild and the ~8 GB teacher weights. **No model has been
evaluated on the C1 battery, and no probe has been trained.**

> **Why the new operator lives in its own module.** `operators/attention.py` and
> its package `__init__` are both members of `CONTINUATION_SOURCE_FILES_V2`, the
> executable source set Phase B's closed preregistration binds to
> `a5ce6311789e…`. Adding a class to either moved that digest, leaving a frozen
> historical document describing code that did not exist when it ran. Since a
> frozen record must never be regenerated to match new code, the operator moved
> instead. **No pre-existing tracked source implementation file was modified; all
> C1 source implementation was added in new files.** All six Phase-A/B operator
> signature hashes are unchanged.

## Phase C0 — frozen 2026-09-01

The protocol governing Phase C1 is
[`phase_c0_preregistration.json`](phase_c0_preregistration.json)
(`aadistill.autoinit.phase_c0_protocol/v1`); the power evidence behind its
battery size is
[`phase_c0_sizing_evidence.json`](phase_c0_sizing_evidence.json).

| | |
| --- | --- |
| C1 confirmation battery | **850 scorable / 950 total** |
| C1 confirmation seeds | **exactly 3 fresh paired fixed blocks** |
| primary endpoint | `correct_overall` over the 850 scorable |
| SESOI / design alternative | `+0.010` boundary · `+0.015` at `P(GO) = 0.8379` |

C1 is a **fixed-path ATTENTION isolation experiment using short 0.86M recovery
probes** — 2 arms × 3 fresh seeds = **6 `E1_KD_HEAVY_0860K` probes**. It is *not*
formal recovery evidence and establishes no recovered-model capability.

### Verification run with the C0 freeze

`1057 passed / 11 skipped / 0 failed` (20m40s) across `tests/docs`,
`tests/autoinit/test_state.py`, `tests/pod` and `tests/test_usable_rollout.py` —
the suites that own the state, schema, catalog, link and session-contract
invariants this commit touches. All four frozen-asset verifiers PASS:
`verify_frozen_assets`, `verify_historical_probe_reuse`,
`verify_attempt5_probe_reuse`, `verify_attempt4_probe_reuse`. The verifiers
rewrite only their own `generated_utc`; every digest was unchanged and that
churn was reverted, so no frozen evidence file is touched by this commit.

**No runtime or source implementation file changed** — the diff is `logs/` plus
one test whose premise expired (it required the snapshot to call Phase C "NOT
DESIGNED", which C0 has now made false).

## Where to start

| you want | read |
| --- | --- |
| the scientific history, by phase | [`PHASE_INDEX.md`](PHASE_INDEX.md) |
| the frozen C1 protocol | [`phase_c0_preregistration.json`](phase_c0_preregistration.json) |
| the power evidence behind N=850 | [`phase_c0_sizing_evidence.json`](phase_c0_sizing_evidence.json) |
| what Phase A and B concluded | [`phase_a_vs_phase_b_comparison.md`](phase_a_vs_phase_b_comparison.md) |
| the Phase-C structure | [`phase_c_roadmap.md`](phase_c_roadmap.md) |
| the next-session handoffs | [`HANDOFF_next_session.md`](HANDOFF_next_session.md) |
| storage and what was cleaned | [`storage_closeout_20260831.json`](storage_closeout_20260831.json) |

## Filesystem policy — three roots, no others

| purpose | path |
| --- | --- |
| git repository | `/home/ecs-user/AlphaAvatar-distill` |
| canonical scientific artifacts | `/home/ecs-user/aad-artifacts` |
| scratch | `/home/ecs-user/aad-scratch/sessions/<session-id>` |

The five `$HOME/phase_b_*_scr` directories were deleted on 2026-08-31 after every
one of their 200 files was proven to have a canonical copy — including hashing
all 128 members of the three remaining tarballs. **140 MiB** reclaimed, nothing
unique lost. Before deleting, 89 files of sole-copy Attempt-5 raw evidence were
promoted to `aad-artifacts`, and 60 session journals to the per-attempt log
directories.

> **That closeout covered those five `$HOME` roots ONLY.** It did **not** mean
> that all `/home/ecs-user/aad-scratch` scientific evidence had been removed, and
> it had not been. A 2026-09-01 audit found **11 of the 12** retained Phase-A
> per-sample row files, plus 90 raw generation files, still sole copies under
> `aad-scratch` session directories outside the declared convention.

### Preserved evidence — `preserved_scratch_20260901`

`/home/ecs-user/aad-artifacts/autoinit/preserved_scratch_20260901/` — **359
files, 12.95 MiB**, byte-for-byte copies of every sole-copy Phase-A/B per-sample
row file, raw generation, training event stream and scored probe result found
under a session `store/extracted/` tree. `MANIFEST.json`
(`aadistill.promoted_raw_evidence/v1`, sha256
`3e1a72e2ade2fde610b95e41e147f7f93d37fd4418a4a2d9f61ae58d27cca0b7`) records
origin path, size, sha256 and scientific role for every file.

All 359 were re-hashed from the manifest — 0 missing, 0 mismatched — and the
preserved copy **alone** re-derives the frozen Phase-A/B pooled figures. **Copy,
not move: the `aad-scratch` originals are retained and must not be deleted in
this phase.** The 140.66 MiB of session transport bundles need no C0 decision.

---

# PHASE B IS CLOSED — resolved, $1.5433

All **8** pre-provider gates passed. The pod recomputed the corrected rung-2
decision independently and **reproduced it exactly** — `11/340`, `9/340`, `3/340`
→ `tie_pending`, candidates `{fe9683e6a9c7, 85bde4ded2c3}` — which is the
strongest available confirmation that the stage-aware pooling repair works in the
real runtime. Stage 4 then bought **exactly one** probe,
`autoinit.v1.phase_a.rung3.fe9683e6a9c7.sc`, and stage 5 resolved.

## The result

**Winner: `fe9683e6a9c783bbc6fe276a78c851c6`.** Not the control.
`decision_status: resolved`, `tie_break_ran: true`, report `8c8842b84fe85cec`.

| final pooled | seeds | correct | correct_overall | usable_rollout |
| --- | --- | --- | --- | --- |
| **`fe9683e6a9c7`** | sa+sb+sc | 16/510 | **0.031373** | 0.6842 |
| `85bde4ded2c3` | sa+sb+sc | 10/510 | 0.019608 | 0.5456 |
| control | sa+sb | 3/340 | 0.008824 | 0.4947 |

The control has fewer observations **by design** — it is outside the equivalence
interval and correctly never advanced to `sc`. At `0.008824` it cannot affect the
outcome.

## Read the margin carefully

`0.011765` against an interval of `0.011695`. It clears by **`0.000070`**.

One correct answer is `1/510 = 0.001961`, so the separation is about **3.6% of a
single correct sample**. Had `fe9683e6a9c7` scored 15 instead of 16, the margin
would be `0.009804` and the result `unresolved_equivalence`.

The frozen rule was applied exactly as preregistered and the decision is what it
returns. It is **not** a comfortable separation, and nothing downstream should
treat it as one.

Correctness remains near the floor in every arm — 16 of 510 for the winner. This
is **selection** evidence under a fixed 0.86M probe, not recovered-model
capability. `correct_overall` is the ranking metric; `usable_rollout` does not
rank, though here it agrees.

## What it authorizes

**Nothing.** Not formal recovery, not a canonical Stage-1 NLL, and not Phase C
execution. The Phase A vs Phase B scientific comparison
([`phase_a_vs_phase_b_comparison.md`](phase_a_vs_phase_b_comparison.md)) is
**COMPLETE and accepted**; the next research task is **Phase C0 — protocol,
statistical power and experimental design review**, which must precede any Phase
C execution.

## Cost across five launches

| attempt | $ | reached | outcome |
| --- | --- | --- | --- |
| 1 | 0.2513 | pod test gate | closed the product-contract / test-scope defect |
| 2 | 0.3146 | driver stage 0 | closed the constructor / authorization defect |
| 3 | 0.2275 | stage 1 | closed the dev-box finalist-path defect |
| 4 | 1.4680 | ALL_DONE | bought `fe9683/sb`; decision **withdrawn** (pooling defect) |
| 5 | 1.5433 | **ALL_DONE** | bought `fe9683/sc`; **RESOLVED** |

**$3.8047** total. Records: [`autoinit_continuation_b_attempt5.json`](autoinit_continuation_b_attempt5.json)
and the per-attempt files beside it.

---

# Superseded: the withdrawn attempt-4 decision

# THE ATTEMPT-4 DECISION IS WITHDRAWN — the state is TIE_PENDING

Attempt 4 reported `resolved / winner=fe9683e6a9c7`. **That decision is not
accepted.** The probe it bought is valid and retained; the decision it computed
was formed over evidence the frozen rule is not defined on.

## What went wrong

The continuation **imports** completed evidence before stage 3, and that includes
`85bde4ded2c3/sc`. The inherited `pooled_over_rungs` pools *every completed
rung* — correct in Phase A, where rung 3 could only exist after rung 2 had
decided. Here it produced an asymmetric comparison:

| withdrawn | rungs | n | correct_overall |
| --- | --- | --- | --- |
| `fe9683e6a9c7` | sa+sb | 380 | 0.032353 |
| `85bde4ded2c3` | **sa+sb+sc** | **570** | 0.019608 |
| control | sa+sb | 380 | 0.008824 |

The `0.012745` margin that "resolved" the session is not a same-rung quantity.

## The corrected decision

Recomputed at `$0` from the same retained journals with `sa+sb` only, using the
driver's own pooling and the **real** frozen `select_final_winner`:

| pooled `sa+sb` | correct_overall | usable_rollout |
| --- | --- | --- |
| `fe9683e6a9c7` | **0.032353** (11/340) | 0.7158 |
| `85bde4ded2c3` | **0.026471** (9/340) | 0.5632 |
| control | **0.008824** (3/340) | 0.4947 |

`margin = 0.005882` — **inside** the `0.011695` equivalence interval.

> **`decision_status: tie_pending`, `winner: None`**
> tie candidates `{fe9683e6a9c7, 85bde4ded2c3}`; control is outside the interval
> and does not advance.

Record: [`autoinit_continuation_b_corrected_rung2.json`](autoinit_continuation_b_corrected_rung2.json).

## A wording correction

Earlier reports here called `usable_rollout` the primary axis and leaned on it to
describe the winner. **`correct_overall` is the frozen ranking and equivalence
metric**; `usable_rollout` is a feasibility/behaviour axis reported alongside it
and does **not** rank. The driver's own `axes` field already said so — the code
was right and the prose was not. That `fe9683e6a9c7` also leads on
`usable_rollout` is supporting evidence, not what selects it.

## What is still owed — and the contract that enforces it

**Exactly one observation: `fe9683e6a9c7/sc`.** Every `sa`, every `sb` — including
the one Attempt 4 purchased — and `85bde4ded2c3/sc` are retained.

A dollar ceiling does not encode that: `$5.4784` funds one probe of *any* kind,
so a session that bought a replacement `sb` instead of the owed `sc` would stay
inside budget and report success. Three independent statements of the scope must
agree, and the executable is narrowed at the act of buying:

| where | what it says |
| --- | --- |
| `ContinuationDriver.PURCHASABLE` | `(("fe9683e6a9c7", 3),)` — a whitelist, since a *count* is satisfied by buying the wrong probe |
| `probe_config()` | the purchase seam. The inherited `run_probe` reaches it **iff** `restore_probe` returned nothing, and it is the only call site in the codebase — so a citable descriptor never arrives, and this can only stop a purchase, never suppress reuse |
| launcher | `rung2_probes=0`, `tie_break_probes=1` |
| `workload_scope_gate` | priced `hard_probes`, booked probes and the whitelist must agree — and a right-sized whitelist naming the **wrong candidate** is refused |

Every `sa`, every `sb` and `85bde4ded2c3/sc` are **reuse-only**: a missing or
non-binding one **fails closed** and is never repurchased, because a replacement
is a different measurement from the one the corrected rung-2 decision was
computed over.

**Attempt 4's probe is authorization-bound.** `attempt4_reuse_probes_dir_digest`
is the seventh `BOUND_EVIDENCE` entry, produced by `observed_evidence()`, carried
by the issuer, recorded in the preregistration's reuse rule, and re-checked at
stage 0 and by the launcher's evidence gate. `reuse_verified` inside the record
is **not** equivalent — it says what was true when the record was written, and
nothing stops the record moving between issuance and execution.

**Session plan v2 → v3.** Version 2 described "one missing `sb` and at most two
conditional `sc`", which would authorize work now forbidden. The frozen recovery
science plan is untouched.

**9 mutations, no survivors** — dropping the purchase seam, permitting
everything, widening the whitelist to a replacement `sb`, pointing it at the
wrong candidate, unbinding the attempt-4 digest, dropping it from observed
evidence, booking a second probe, dropping the workload gate, reverting the plan
to v2.

## Re-priced for what is owed

| | was | now |
| --- | --- | --- |
| floor | `$5.4784` | **`$4.1830`** |
| hard ceiling | `$8.0691` | **`$5.4784`** |
| max new probes | 3 | **1** |

`sc` is priced for the **tie candidates** lacking a verified one, not for every
advancing candidate. Over-booking was harmless before a rung-2 decision existed;
it is not harmless now that one does.

## The repair

Pooling is stage-aware, local to `ContinuationDriver`: rung 2 admits rungs
`(1, 2)` only, and the final decision admits `sc` for exactly the tie candidates.
The withheld record is named in the run log rather than silently dropped.

**10 tests from the real retained journals; 5 mutations, no survivors.** One
survived the first pass — removing the attempt-4 import source killed nothing,
because the reuse record proved the probe *citable* while nothing proved the
driver *cites* it. A future session would have silently re-bought ~72 min of
L40S. Closed by driving the real import.

---

# Superseded: what attempt 4 reported

The full withdrawn result, its asymmetric table and the four-launch cost history
are retained in
[`autoinit_continuation_b_attempt4.json`](autoinit_continuation_b_attempt4.json)
and the per-attempt records beside it.

---

# REPAIRED — finalist bytes come from the pod staging contract

`build_finalist_states` no longer reads the amendment's `checkpoint_path`. A new
`staged_checkpoint()` derives the pod location:

| candidate | resolved path |
| --- | --- |
| searched finalists | `REPO / "artifacts/autoinit/phase_a_selected" / state_id` |
| canonical control | `REPO / f"artifacts/stage1/{CONTROL_ID}/checkpoint"` |

The control path is **derived from `CONTROL_ID`** so the staged location and the
control's identity cannot drift apart. Both constants are **restated, not
imported** from `autoinit_phase_b_driver`: borrowing them would pull the Phase-B
search driver into this session's import closure, which `no_search_gate` measures
and `CONTINUATION_SOURCE_FILES_V2` pins.

**The amendment is unchanged and stays frozen** — bound identity
`df413bd99119dab7`, verified intact. Its dev-box paths remain correct provenance
for the machine that materialized those bytes.

**Identity verification is untouched.** Whatever is found at the staged path is
still re-identified and required to equal the amendment's bound digest; a
mismatch still raises `IdentityCollapseError`. The amendment still decides
whether the bytes are the right ones — it no longer decides where to look.

## Verification

| check | result |
| --- | --- |
| `$0` stage-1 reproduction | amendment `checkpoint_path` points at a **nonexistent** dev-box directory (asserted absent), the valid checkpoint exists **only** under the staged location, `build_finalist_states` **succeeds** |
| launcher ↔ driver contract | the launcher's relay `dest` and the driver's derived path pinned to the same string, from both ends |
| fail-closed check | a staged checkpoint whose identity moved still raises |
| mutations | **5, no survivors** — restoring `Path(c.checkpoint_path)`, returning the amendment path, moving either constant, dropping the digest check |
| full suite | **2499 passed, 12 skipped, 0 failed** |
| frozen verifier | clean |
| frozen identities | all hold; the rest of the preregistration is **byte-identical** |

**Two things the verification itself caught**, both worth keeping:

* The first version of these tests built at the real frozen target geometry — a
  ~2.4 GiB float32 checkpoint written twice per run — and six mutation-harness
  runs took this box to **100% disk**. They now use the toy geometry: 2.4 s
  instead of 31.8 s, no measurable disk. That matters beyond the dev box, since
  they also run inside the **pod's setup gate**, on a 2700 s timeout whose expiry
  exits 90 and kills a paid session.
* The full suite went **red** on the first pass — 11 failures — because the
  whole-function test still staged its checkpoints where the driver no longer
  looks. Targeted runs of the changed files passed; only the full suite exercised
  the interaction. The test now depends on the staging contract, which is exactly
  the property whose absence let attempt 3 buy the failure.

**Nothing is issued and nothing is launched.**

---

# ATTEMPT 3 PASSED STAGE 0 — and stopped at stage 1 · $0.2275

The furthest any continuation attempt has reached, and **the first to pass a
scientific stage**. All 7 pre-provider gates passed, including
`session_commit_and_lineage`; setup completed; the driver loaded its own
`ContinuationAuthorization`; and **stage 0 PASSED** — every cited identity bound,
the collapsed universe `e94f15d2e648…` and Stage-1 selection `84fd64968519…`
verified, the evaluation protocol attested (interval `0.011695`, floor `0.3000`,
plan `02be33b9a7a8…`), and **all eleven probe journals imported**. The attempt-2
constructor defect is hardware-closed.

Stage 1 then refused, in the same second:

```
85bde4ded2c31953f802e39cf2252c87 is an ADVANCING finalist but is not staged at
/home/ecs-user/aad-artifacts/autoinit/phase_a/85bde4ded2c31953f802e39cf2252c87
```

**A dev-box path used on the pod.** `build_finalist_states` resolves each
finalist as `Path(c.checkpoint_path)`, and `checkpoint_path` comes from the
frozen identity-collapse amendment, which records dev-box absolute paths under
`/home/ecs-user/aad-artifacts/autoinit/phase_a/`. On the pod the finalists are
staged at `artifacts/autoinit/phase_a_selected/<state_id>`.

**Phase B already resolves exactly that location.**
`autoinit_phase_b_driver.py:110` defines
`STAGED_FINALISTS = REPO / "artifacts/autoinit/phase_a_selected"` and line 362
uses `STAGED_FINALISTS / canonical_id`. Same constant, same staging directory,
different resolution — the fourth instance of the same inheritance class, and the
second time a dev-box-only path has reached a pod.

**The amendment must not move.** Its hash `df413bd99119dab7` is bound by the
authorization and the preregistration, and its dev-box paths are correct
provenance for the machine that holds those bytes. The repair belongs in the
**driver**: map `state_id` to the pod staging directory.

**Why no `$0` check caught it.** The whole-function test builds toy checkpoints in
`tmp_path` and points the amendment at them, so `c.checkpoint_path` always exists
there. Its assertion — that an advancing finalist must be staged — is right, and
is what fired on the pod. What nothing asserts is that the path the driver
*derives* is one the pod will have.

**No probe was bought.** No `sb`, no `sc`, no pooled decision, no final selection,
no search reachable, no evidence moved, permanent controls untouched.

Twenty artifacts across six classes were collected and the teardown gate allowed
before deletion. Record:
[`autoinit_continuation_b_attempt3.json`](autoinit_continuation_b_attempt3.json),
evidence in [`autoinit_continuation_b_attempt3/`](autoinit_continuation_b_attempt3/).

**A diagnosis hazard worth recording.** The reused scratch `store/extracted` still
held attempt 2's `stage0_traceback.log`, dated `2026-08-29T20:50:21Z`. This run
produced none, because stage 0 passed. Reading the reused directory would have
reported the previous attempt's failure as this one's; the diagnosis was taken
from a clean extraction of this run's own archive.

**The repair is not applied.** A fourth attempt needs a maintainer decision.

---

# Previous attempt (2) — constructor defect, now closed

# ATTEMPT 2 CLEARED THE TEST GATE AND DIED AT DRIVER STAGE 0 — $0.3146

**The attempt-1 repair worked.** The CPU test gate **passed** on hardware,
`SETUP_RC=0`, `AUTHORIZATION_OK` and `SETUP_DONE` were reached, and the driver
detached and was confirmed by descriptor probe. The calibration/test-scope defect
is closed and did not recur.

Three minutes into the driver, `stage_bind` raised:

```
AttributeError: 'PhaseAAuthorization' object has no attribute 'require_evidence'
```

**No scientific stage ran** — no probe trained or evaluated, no `sb`, no `sc`, no
pooled decision, no final selection, no search reachable. No evidence moved.

## REPAIRED — ContinuationDriver owns its constructor

`ContinuationDriver` now declares `AUTHORIZATION_TYPE = ContinuationAuthorization`,
`AUTHORIZATION_PATH = logs/autoinit_continuation_b_authorization.json` and
`PLAN = CONTINUATION_PLAN_V1`, and **does not call `super().__init__`**. Setting
the two attributes alone would not have been enough: `PhaseADriver.__init__`
binds three things this session must not inherit — the Phase-A type and path,
`PHASE_A_PLAN_V1`, and the Phase-A evidence schema and scope — and its closing
`require_plan(PHASE_A_PLAN_V1.plan_hash)` is refused by a
`ContinuationAuthorization`. The failure would simply have moved one line.

The shape is `PhaseBDriver`'s, kept **local**: `PhaseADriver` is untouched and no
framework was extracted to remove the constructor duplication. The evidence
**pathname is unchanged** — the collection contract names it — but its content is
now the continuation's, not a Phase-A schema and scope inherited along with the
writer. Nothing read `ev["phase_a"]` or `ev["scope"]`; checked before removing.

**The test that could not have caught this now can.** The whole-function test's
`driver.auth = make_auth(...)` overwrite is gone: it writes the scenario grant to
disk and points the seam at it, so the constructor performs the same load the pod
does. In the mutation run, *dropping `AUTHORIZATION_TYPE` is now killed by that
test itself*.

[`test_driver_authorization_wiring.py`](../tests/pod/test_driver_authorization_wiring.py)
asserts on `driver.auth` **as the constructor left it**, never substituting one:
the type, that the object carries the identity of the file at
`AUTHORIZATION_PATH`, that the continuation plan is accepted and a Phase-A plan
refused, that the evidence envelope is continuation-owned, and that no attribute
`PhaseADriver.__init__` establishes is left unset — derived from the parent's
**bytecode**, so a future parent addition cannot silently leave this driver
short. It pins the same seam for all four `PhaseADriver`-derived drivers.

**Five mutations, no survivors:** drop either override, point the path back at
Phase A, validate against Phase A's plan, restore the Phase-A evidence schema.

## The seam was built for this subclass, and this subclass ignored it

`PhaseADriver` exposes `AUTHORIZATION_TYPE` and `AUTHORIZATION_PATH` as class
attributes. Its own comment on them reads:

> the continuation subclasses this driver: with the path fixed here, the
> continuation would have loaded attempt 12's CONSUMED Phase-A authorization on
> the pod — a real, committed file … Not a crash; a silently wrong ceiling, and
> evidence naming the wrong grant.

`ContinuationDriver` overrides **neither**, so on the pod it loaded
`logs/autoinit_phase_a_authorization.json` as a `PhaseAAuthorization`. It is the
**only** `PhaseADriver` subclass that sets neither — `autoinit_phase_b_driver.py`
and `autoinit_recovery_continuation_driver.py` both do. It crashed rather than
running silently on a wrong ceiling only because `require_evidence` does not
exist on the Phase-A type, which is the lucky version of this bug.

**The repair is not one line.** Setting the two attributes moves the failure to
the next statement: `PhaseADriver.__init__` ends with
`self.auth.require_plan(PHASE_A_PLAN_V1.plan_hash)`, and a
`ContinuationAuthorization` refuses it — verified at `$0`: *"this authorization
binds plan `a2ef4cd68a4b` and the session runs `9377a2dc61f2`"*. The parent also
stamps Phase A's evidence schema and scope. `PhaseBDriver` solved exactly this by
**not** calling `super().__init__` and reimplementing it with its own `TYPE`,
`PATH` and `PLAN`; `ContinuationDriver` calls `super().__init__` and inherits all
four wrong things.

## Why no `$0` check caught it

The whole-function test runs the real `__init__` — which loaded the wrong
authorization there too — and then discards it in the next line:

```python
driver = ContinuationDriver.__new__(ContinuationDriver)
ContinuationDriver.__init__(driver, Args())
driver.auth = make_auth(...)          # overwrites what __init__ loaded
```

Every later assertion ran against a correctly-typed authorization the test
supplied. The seam's wrong default was loaded and masked in the same three lines
— seam injection points are the blind spot, again.

The launcher's `evidence_binding_gate` asked a **different object**: it calls
`ContinuationDriver.observed_evidence()` and then `ctx.auth.require_evidence()`
on the *launcher's* correctly-loaded grant. It proved the evidence matches; it
never exercised the driver's own load path.

Record: [`autoinit_continuation_b_attempt2.json`](autoinit_continuation_b_attempt2.json),
evidence in [`autoinit_continuation_b_attempt2/`](autoinit_continuation_b_attempt2/).

## Verification of the repair

| check | result |
| --- | --- |
| full suite | **2495 passed, 12 skipped, 0 failed** (27:19) |
| frozen verifier | clean, no problems |
| no-search / whole-function / calibration-boundary tests | green |
| pre-provider gates | **6 of 6** against a candidate authorization built to a scratch path **outside** the repo |
| `session_commit_and_lineage` | **defers** — it structurally requires the authorization *committed at the session commit*, which cannot exist before issuance |
| executable digest | `746b9d68…` → `1bf97d7da6d183ec…` |
| preregistration | `8335d149…` → `68cb7e898552388d…` |
| the rest of the preregistration | **byte-identical** — it moved only because it binds the new executable |
| frozen science / evidence / pricing | all unchanged, compared mechanically against `8398108` |
| `194657Z` | permanently **CONSUMED** |

**Nothing is issued and nothing is launched.** The repaired frozen commit is
reported for review; the replacement authorization is minted only on the
reviewer's go-ahead.

Two minor findings, neither a blocker: `issue_continuation_b_authorization.py`
crashes on a cosmetic `relative_to` when `--out` points outside the repo, *after*
correctly writing the file; and `SESSION_LAUNCHERS` still omits the Phase-B and
continuation launchers, as previously recorded.

# ATTEMPT 1 REACHED THE POD AND STOPPED AT THE TEST GATE — $0.2513

The behavioural continuation launched `2026-08-29T16:05:11Z` at commit
`8df1bad` on an L40S. All seven pre-provider gates passed **inside the real
launcher**, the bundle round-tripped, and setup reached `VLLM_READY`,
`TEACHER_READY` and `ROPE_OK`. The CPU test gate then returned **4 failed / 2300
passed / 95 skipped** and the session aborted after 15.2 minutes.

**No scientific stage ran.** No probe was trained or evaluated, no `sb` or `sc`
was bought, no pooled decision or final selection was computed, and no search was
reachable. No evidence moved.

## The gate caught a real missing input

The continuation launcher stages **neither calibration mixture**. It declares

```
relay_inputs = (*CANONICAL_INIT, *RECOVERY_LADDER, *continuation_inputs())
local_assets = PHASE_A_LOCAL_ASSETS
```

so Phase B's two calibration declarations are both absent:

| asset | path | travels by | Phase B | continuation |
| --- | --- | --- | --- | --- |
| `calib.domain_balanced@v1` | `artifacts/stage1/e8_calibration_v1/items.jsonl` | relay | declared | **omitted** |
| `calib.reasoning_heavy@v2` | `artifacts/stage1/reasoning_heavy_v2/items.jsonl` | dev-box local asset — built here and never uploaded | declared | **omitted** |

**This is not merely a test problem.** `calib.domain_balanced@v1` is the fixed
distribution every probe is measured under, and **both** identities are bound
into this session's authorization and preregistration. The session asserts the
mixtures identify it while never staging their bytes. The gate stopped it at
`$0.2513`; the next failure point would have been a probe stage.

**Same class as the product contract repaired earlier today** — derived from
Phase B's launcher, but taking some declarations from Phase A. Reasoning from
what the session *needed* instead of what the machinery *requires*. Not caused by
that repair: both omissions predate it, and the previous attempt never reached a
pod because `ckpt_store_capacity_gate` refused first.

## Reproduced at $0, exactly

Hiding both directories on the dev box and re-running the suite under the pod's
exact ignore set fails **the same four tests and only those four**. Hiding
`reasoning_heavy_v2` alone reproduces exactly two of them, which is what
separates the two missing assets. The fourth name was recovered this way rather
than from the log: the pod's setup probe relays a bounded tail, so only three of
the four survived in `launcher.out` — a gap worth closing.

Full record: [`autoinit_continuation_b_attempt1.json`](autoinit_continuation_b_attempt1.json),
journals in [`autoinit_continuation_b_attempt1/`](autoinit_continuation_b_attempt1/).

## The repair: BIND IS NOT CONSUME — the ignore set, not the asset list

The reviewer traced the runtime path and corrected my framing: **neither mixture
should be staged.** They are Phase-B *search* inputs. The paid behavioural probes
train from `artifacts/stage3/ladder_uniform_probe` and are scored on the
`artifacts/stage3/recovery_search_v2` battery.

**Verified rather than accepted**, because excluding tests that read a genuine
dependency would hide it and buy another pod:

* the continuation driver contains **no calibration reference at all** — not
  `DOMAIN_BALANCED`, `REASONING_HEAVY`, `e8_calibration_v1`, `reasoning_heavy_v2`
  or `resolve_calibration`;
* its stage map is `{0: stage_bind, 1: stage_import, 3, 4, 5}` — no search;
* the only calibration mention on the inherited path is a **comment** inside the
  `PhaseADriver.stage1` this session overrides with a raise and never binds.

So the two failing modules exercise machinery this session is structurally unable
to run, and both are added to `CONTINUATION_TEST_IGNORES`:

| module | what it needs that this session has no reason to have |
| --- | --- |
| `tests/autoinit/test_causal_depth_measurement_job.py` | drives the causal-depth job through the real resolver, which loads `e8_calibration_v1/items.jsonl` |
| `tests/pod/test_phase_b_driver_and_launcher.py` | builds the **Phase-B** spec and requires both profiles plus `reasoning_heavy_v2` staging |

Both keep running in full on the dev box, where the bytes live. **The check
moves; it does not disappear** — as `test_phase_b_reuse_hostlocal` already does.

The identities stay bound. They name the distribution the imported Stage-1 result
was produced under, and a grant that names a mixture must be able to refuse a
different one. `require_calibration`'s docstring said the probe trainer consumes
the mixture; it does not, and the wording is corrected.

**Proof, at `$0`.** With `artifacts/stage1/e8_calibration_v1` **and**
`artifacts/stage1/reasoning_heavy_v2` moved off the filesystem simultaneously,
the exact pod command `python -m pytest tests/ -q $SESSION_TEST_IGNORES` returns
**2288 passed, 23 skipped, 0 failed**.

Five new tests pin it, including one that fails if the spec ever grows a relay
input or local asset for either mixture, and one that fails if the driver ever
gains a calibration reference.

## A gap this exposed, not repaired here

`tests/pod/session_specs.py::SESSION_LAUNCHERS` says *"keep this list complete: a
session missing from it is a session no structural check covers"* — and it omits
both `autoinit_phase_b_launch` and `autoinit_continuation_b_launch`. Neither is
covered by the simulator/pod ignore-list pin or the setup-env forwarding checks,
which is why this divergence had no `$0` guard. That pin also asserts every
session's ignore list **equals** the simulator's two-entry list, which Phase B
(4) and the continuation (7) both legitimately exceed, so adding them requires
reworking the pin. Larger than the approved repair; recorded, not done.

The bundle round-tripped before any provider resource existed: a fresh clone of
the relay bytes checks out `8df1bad216aa`, its 62 executable files re-derive to
`96c346ffcf6a…`, and it carries this exact authorization. The earlier
`autoinit.continuation_b.20260829T115028Z` grant bound the pre-repair digest
`1682cd7d` and is retired to `superseded/` **UNUSED**.

# The product contract is REPAIRED — the last blocker is closed

`ckpt_store_capacity_gate` was refusing every launch, and it was right to refuse
the thing it was asked: it demanded **11.55 GiB** of dev-box space for "five
stage-1 selected leaves" plus 6 GiB of working room, on `--ckpt-store`. **This
session produces no leaves.** It runs no search; its three finalists already
exist locally and on the relay. It was asking for roughly a thousand times the
session's need, on the wrong filesystem, for a product that does not exist.

The same inheritance set `fetch_products = fetch_selected_leaves` and
`products_secured = selected_leaves_secured`. Both are Phase B's, and both exist
to rescue **search output** — the attempt-11 fix, where five measured leaves were
lost because the product fetch returned early.

**What those two actually did, driven rather than assumed.** They were **inert,
not fatal**. `selected_leaf_records` reads
`<scr>/store/selected_leaf_durability.json`; the continuation driver never writes
one and does not fetch one, so both returned empty and the secured check passed
vacuously with `"stage 1 did not stage any selected leaves"`. The earlier reading
here — that at teardown they would hunt the pod for five directories and judge
the session unsecured — **overstated it**, and the correction is pinned by a test
rather than left in prose. They still had to go: a declaration that is wrong but
currently harmless is exactly the kind that survives review, and it stays
harmless only until something downstream starts writing that report.

**The repaired contract.** No leaf fetch and no leaf-secured check — the
`ArtifactPolicy` defaults, which answer "this session owes no off-pod products"
explicitly rather than skipping the gate. The durable science travels where it
already did, in the artifact archive: probe journal, probe configs, generations,
per-sample rows, the pooled decision, the report and the session/audit evidence.
Temporary probe training checkpoints are **not** promoted to permanent products;
nothing downstream reads them.

**The replacement gate measures what the collector writes.**
`SessionRunner.collect_and_teardown` scp's the archive to `<args.scr>/store` and
extracts a second full copy beside it, so that is the filesystem checked. Its
requirement is measured, not guessed: `du -sb` over every retained session store
on this box, restricted to the **search-free** ones, because every store above
13.7 MB here is dominated by a 26–55 MB `states.jsonl` this session cannot write.
Largest comparable **13,641,956 B**, ×4, plus 1 GiB of working room — **1076
MiB**, against a closest structural analogue (recovery continuation attempt 7:
eleven probes, full generations, `ALL_DONE`) of 11.53 MiB. Derivation in
[`autoinit_continuation_b_capacity.json`](autoinit_continuation_b_capacity.json).

**11 tests, 15 mutations, no survivors** — reconnecting either callable,
restoring the five-leaf gate, pointing the gate at `--ckpt-store`, silently
moving the measured bound or the safety factor, dropping the probe journal or the
pooled decision from either archive, promoting probe checkpoints to products,
adding a search call site, and moving the 6→3 boundary all fail.

**What moved, and what did not.** The executable digest `1682cd7d…` → `96c346ff…`
and the preregistration `808390d1…` → `21e69909…`, both by design. The
preregistration diff is exactly those two lines. Science, evidence and pricing
are untouched: plan `a2ef4cd68a4b…`, Stage-1 selection, the amendment, the
six-candidate universe, both reuse digests, the rung-1 result, the calibration
identities, the three finalists, floor `$5.4784`, ceiling `$8.0691`, cap
`$283.76`. The authorization issued against the old digest is retired to
`superseded/` **UNUSED**.

**Scratch no longer lands in `$HOME`, and the five old roots are gone.** The
launcher defaults to `/home/ecs-user/aad-scratch/sessions/<session-id>`, so no
sixth `phase_b_*_scr` is created.

The 2026-08-29 inventory found that **`phase_b_a5_scr` held the only surviving
raw generations, per-sample rows and train logs for Attempt 5's three fresh `sa`
probes** — including `fe9683e6a9c7`. That is why it was not deleted then. It has
since been **resolved rather than retired**: on 2026-08-31 those 89 sole-copy
files were promoted into `aad-artifacts`, and 60 session journals into the
per-attempt log directories, every file hash-verified, before any deletion. The
five roots were then removed, reclaiming **140 MiB**. `verify_attempt5_probe_reuse`
now passes reading only repo journals and `aad-artifacts`, which is the evidence
that nothing load-bearing went with them.

Earlier, only the five git bundles had been deleted — each proven rebuildable
from a commit that is an ancestor of `origin/main` — reclaiming **31.26 MiB**.
Records: [`scratch_inventory_20260829.json`](scratch_inventory_20260829.json) and
[`storage_closeout_20260831.json`](storage_closeout_20260831.json).

# The launch blocker is REPAIRED

The behavioural continuation was approved and then could not launch, because its
executable identity was computed in a format the launch gate cannot reproduce.

`continuation_source_digest` used `sha256_json(entries)`. `session_commit_gate` —
the **first** pre-provider precheck, written after continuation attempt 5 died at
`$0.1369` on a stale binding — does not read that value: it **re-derives** the
digest from the git blobs at the launch commit, as `sha256` over sorted
`path:sha256` lines. That is what Phase A's `harness_source_digest` and
`phase_b_source_digest` already produce. Mine was the only one that differed, so
the two could never agree and the gate refused every launch.

**Why every `$0` check missed it.** All of them compared the digest against code
that computes it the same way — `continuation_source_gate` and `require_source`
share one implementation — so they agreed under either formula. Nothing exercised
the one consumer that re-derives it independently. A matching attribute name is
not a matching contract.

**The repair.** The producer adapts to the shared convention; the gate is
untouched and no authorization-specific workaround exists. The formula had been
inlined in **seven** places, including inside the consumer, which is how an
eighth copy came to disagree, so
`aadistill.infrastructure.source_identity.canonical_source_digest` now holds one
implementation for a new session to import. It deliberately does **not** rewrite
the existing seven: each lives inside a frozen source set, and changing them
would move Phase-A and Phase-B digests for a reason unrelated to their science. A
test asserts all three producers return the same value for the same files.

Set version **v2 → v3** records an *algorithm* change, not a content change: the
same files under the two schemes give different digests and would otherwise read
as an edit. `source_identity.py` joined the source set — caught by the closure
test, not by inspection — taking it from 61 to **62** files.

**The regressions run the real consumer.** Seven tests drive the actual
`session_commit_gate` against the actual authorization shape and require
acceptance; five mutations (the old `sha256_json`, an unsorted set, a dropped
separator, a changed separator, a silently reverted version) all fail through the
gate. No science, evidence or pricing identity moved: plan `a2ef4cd68a4b…`,
Stage-1 selection, amendment, six-candidate universe, both reuse digests, rung-1
result, calibration identities, finalists, floor `$5.4784`, ceiling `$8.0691` and
cap `$283.76` all held.

# PHASE A IS COMPLETE — as a SELECTION experiment

**What Phase A is.** A low-budget **initialization-selection** experiment: it
searches initialization and operator order under the fixed
`calib.domain_balanced@v1` calibration distribution, and ranks candidates using
matched **0.86M selection probes**. It is **not** the project's formal recovery
training. Formal recovery remains later Stage 2 / Stage 3 work, and nothing here
estimates recovered-model capability.

**Two different things are called "the control", and they stay distinct:**

* `preflight_ctl_r0860k_{sa,sb}` — the **historical permanent Stage-3 controls**
  that materialized the equivalence interval and feasibility floor. **Imported,
  never retrained**; Phase A does not touch them.
* `qwen3_0p6b_init_v0` — the **Phase-A canonical initialization control**,
  hash-injected, which correctly received matched fresh selection probes
  alongside the searched leaves.

**Recovery continuation attempt 7 ran 2026-08-23/24 and finished: `ALL_DONE`,
$12.8587, all six stages, eleven probes across three seeds.** Pod
`3c1g6e01kdu1ya`, 779.3 min, deleted with provider confirmation. Full record:
[`autoinit_recovery_continuation_attempt7/`](autoinit_recovery_continuation_attempt7/).

## The result

```
decision_status   unresolved_equivalence
winner            None
tie_break_ran     True
```

Pooled over seeds, primary metric `correct_overall`, interval **0.011695**:

| candidate | correct_overall | usable_rollout_rate |
| --- | ---: | ---: |
| `cca699c93f34` | **0.029412** | 0.6561 |
| `85bde4ded2c3` | 0.019608 | 0.5456 |
| `control-qwen` (canonical init) | 0.008824 | 0.4947 |

* `cca699c9` − `85bde4de` = **0.009804**, inside the interval → tied, unresolved
  even after seed sc. **No fourth seed follows** — that is the frozen rule, and
  `unresolved_equivalence` is a result, not a condition to resolve.
* `cca699c9` − control = **0.020588**, outside → separated, leaf better, and
  ahead on the behaviour axis too.
* `85bde4de` − control = **0.010784**, inside → not separated.

**No searched leaf is the resolved Phase-A winner.** `cca699c93f34…` is the
**numerical/provisional leader** and is separated from the canonical
initialization under the frozen equivalence rule; it and `85bde4ded2c3…` were
**not separated from each other** after the full frozen sa/sb/sc procedure, so
the plan's winner rule returned `None`.

**Do not read the small absolute 0.86M correctness rates as final
recovered-model capability.** They sit near the floor in every arm including the
control — the same property the Stage-3 controls show at `correct_overall`
0.0118. These are selection probes at equal budget, nothing more.

**Comparability holds:** `comparable_identity 70a26e0b…` live equals historical,
`bound_to_stage3_thresholds: true`. The run's raw protocol hash differs from
`250f72ef` only by driver patch, which `generation_runtime_comparability@v2`
declares non-material.

**Timing:** eleven probes at **61.0–61.1 min** training against **61.55** priced —
attempt 6's 71.9 min was host variance. Both carried-in repairs held: handoff
freed to 0.01 GiB, and the evaluation tokenizer materialized before every battery.

## The next scientific sequence — NONE of it started

1. **Phase B** — calibration/data-distribution sensitivity:
   `calib.reasoning_heavy@v2` against `calib.domain_balanced@v1`, searched
   **jointly at P=2**.

   **PHASE-B STAGE 1 IS COMPLETE.** Attempt 5 ran Stage 0 and the joint P=2
   search to completion, emitted an authoritative Top-5 and a durable Stage-1
   selection artifact, and paid for three rung-1 `sa` probes. It then died in
   stage 2 on `duplicate seeds` — the pooling guard was right, and what was wrong
   was the candidate universe it was asked to pool. The identity-collapse
   amendment resolves that: two "searched" leaves are byte-identical to two
   retained Phase-A finalists, so the universe is **6 distinct candidates**, not
   the 8 the preregistration assumed. **Rung 1 was then completed at `$0`** by the
   frozen selection code from retained evidence.

   Stage 1 is **bought, retained and authoritative**. It is never re-purchased.
   The identities below are therefore **historical evidence, not current launch
   identities** — the `$35.6660` ceiling in particular prices a 16.5 h search that
   already exists.

   | historical — full Phase B | |
   | --- | --- |
   | session plan | `23f23649…` (`autoinit.v1.phase_b_session`, 6 stages) |
   | preregistration | `53f1347f6865…`, self-verifying, frozen before any result |
   | executable-source digest | `3ad6a6e3edf1…` over **59** files, set version 5 |
   | science plan | `02be33b9…`, **reused unchanged** |
   | `calib.domain_balanced@v1` | spec `11f36a88…` · content `d65c1f40…` |
   | `calib.reasoning_heavy@v2` | spec `6c67b8df…` · content `cdb28389…` |
   | Stage-1 selection | `84fd6496851995ae…`, durable, authoritative |
   | amendment / universe | `df413bd99119dab7…` / `e94f15d2e648…`, **6** distinct |
   | planning floor / ceiling | **$16.4555** / **$35.6660** — **HISTORICAL ONLY** |

1b. **The behavioural continuation** — the only paid work Phase B still owes,
   and a **separate session** rather than a flag on the existing one.

   **What remains is small.** One missing `sb` and at most two conditional `sc`:
   **1 to 3 probes**, against the ten a full Phase B books.

   ```
   candidate        sa       sb        sc
   fe9683e6a9c7     cited    MISSING   MISSING
   85bde4ded2c3     cited    cited     cited
   control-qwen     cited    cited     MISSING
   ```

   **Six candidates are evidence; three are the workload.** The collapsed
   universe of 6 carries the completed `sa` evidence and the identity-collapse
   result. Rung 1 is COMPLETE and frozen, and only its survivors plus the
   auto-advancing control — `fe9683e6a9c7`, `85bde4ded2c3`, `control-qwen3_0p6b_init_v0`
   — may enter `sb`, the pooled decision, `sc` and the final selection.

   The other three — `ab7632b00788`, `bf5ae3b6ae00`, `cca699c93f34` — are
   **searched non-survivors**: eligible leaves that ranked below the top-2 cut.
   They keep their citable `sa` observations and are never materialized or probed.
   That is a different thing from a **gate exclusion**, and the preregistration now
   records the two separately: `gate_exclusions` (feasibility and catastrophic
   capability) are **both empty** for this result, because every searched leaf was
   eligible and the cut was made by rank. The non-survivor list is derived
   mechanically as the authoritative Top-5 minus the two frozen survivors — never
   from behavioural scores, since a list read off a measurement would let the
   measurement decide who had been eligible.

   | continuation — CURRENT | |
   | --- | --- |
   | session plan | `6d6f971364868d84…` **v3** (`autoinit.v1.phase_b_behavioural_continuation`) |
   | stages | `0, 1, 3, 4, 5` — **no stage 2**, because Phase A's stage 2 is rung 1 |
   | preregistration | `bd827d12b660…` |
   | executable-source digest | `a5ce6311789e…` over **62** files, set version 3 |
   | relay assets | `1643145376ba…`, `fe9683e6a9c7` round-trip verified |
   | floor / ceiling | **$4.1830** (0 probes) / **$5.4784** (1 probe) |
   | purchasable | exactly `fe9683e6a9c7/sc` |

   The `$8.0691` ceiling and the v2 plan below priced three probes and are
   **superseded**. Figures in the historical sections that follow record what was
   true at the time and are left alone.

   **The search is unreachable, and that is checked five ways.** The plan declares
   no search stage; `stage1()` and `run_search()` raise; the stage map never binds
   them; `ContinuationAuthorization.runs_search` is `False` **by type**, with no
   field to set; and the whole-function test drives the real stage map with
   `BeamSearch` and `run_phase_a_search` replaced by detonators, on both the
   resolved and the tie path, touching neither.

   **Loaded is not the same claim as permitted.** The executable-source digest is
   derived from the REAL import closure, so it **includes** `search.py`,
   `ranking.py` and every operator module — the `aadistill.autoinit` package
   `__init__` loads them for every consumer. Excluding them would let loaded files
   change under a grant that claimed to pin the executable. What may *execute* is
   a separate contract, enforced separately and more strongly. Exactly one loaded
   file holds a search call site — `PhaseADriver`, whose `stage1` is overridden
   with a raise — and a call site appearing anywhere else fails the `$0` gate.

   **One frozen digest moved, and it is recorded rather than re-frozen.**
   `autoinit_preflight_setup.sh` is the single `SESSION_KIND` dispatcher for every
   launchable session AND a member of three frozen source sets, so the
   continuation could not become launchable without moving them. Phase B's
   executable digest went `3d3b5d07…` → `a043e2c7…`. The Phase-B preregistration
   is **not** rewritten — that would destroy the record of what attempt 5 executed
   — and the gate is not deleted. Instead `aadistill.autoinit.post_freeze` accepts
   drift only when it is declared in
   [`autoinit_phase_b_post_freeze_changes.json`](autoinit_phase_b_post_freeze_changes.json),
   is additive with zero lines removed, and leaves **every pre-existing dispatch
   branch byte-identical** — re-derived from the script, not read from the note.
   The launcher gate and its test share that one implementation. Seven mutations
   of the note, plus deleting it, are all refused. Phase A's harness set and the
   recovery continuation's contain the same file; both sessions are complete and
   their authorizations consumed, so no live gate binds them, and the blast radius
   is recorded rather than left to be discovered.

   **Building it found five defects that every cheaper gate had passed**, each
   guaranteed to fire on a paid pod: `stage_bind` never ran the inherited stage 0,
   so `self.plan` and `self.evaluation_protocol` were unset when stage 3 read
   them; `enter()` ordered stages against Phase A's plan, whose stage 1 is the
   search; `ContinuationAuthorization` lacked `require_science_plan`, which the
   inherited stage 0 calls unconditionally; `restore_probe` was inherited strict
   and would have **re-bought all eight citable probes** at roughly nine times the
   ceiling while reporting success; and the universe builder demanded all six
   checkpoints be staged to read three. A sixth, `SESSION_KIND=continuation_b`
   having no bash dispatch branch, was caught automatically by the guard written
   after attempt 2 paid `$0.2300` for the same defect.

   **Why reuse works at all.** The eight citable probes carry **two** raw
   `evaluation_protocol_hash` values (`7327e880…`, `250f72ef…`) that differ by host
   NVIDIA driver patch alone. `generation_runtime_comparability@v2` declares both
   the patch and the raw hash non-material, and all of them share the comparable
   identity `70a26e0b…`. The continuation binds imported evidence on student
   identity and seed and defers comparability to that rule.

   **Authorized, not launched.** The grant document
   [`autoinit_continuation_b_grant.json`](autoinit_continuation_b_grant.json) and
   the one-use authorization `autoinit.continuation_b.20260829T115028Z` both
   exist; no cap increase was requested, and the `$8.0691` ceiling fits the
   existing `$23.7050` headroom. The authorization is **unused** — the launch is
   blocked at `ckpt_store_capacity_gate`, above.

   **Comparability is a terminate condition, not a bigger run.** If Stage 0 finds
   the runtime not comparable under `generation_runtime_comparability@v2`, the
   session **stops before any search or probe** — the frozen feasibility floor and
   equivalence interval were materialized under Phase A's runtime and would not
   describe anything this session could produce. The 14-probe no-reuse figure
   (`$33.3529`) is a **rejected counterfactual**, priced so the rejection is on
   the record, and is **not** the authorization ceiling.

   **The candidate set is closed**: the Phase-B Top-5, the two retained Phase-A
   finalists and the canonical control — 8 at `sa`. The three Phase-A
   non-finalists stay excluded even though their `sa` probes are already paid for
   and verified; zero marginal cost is not an admission criterion.

   **Phase A's harness identity is untouched.** Phase B declares its own
   executable-source set covering what a paid *search* executes; the probe path is
   accounted for by the existing trainer, generation and scoring digests rather
   than duplicated.

   **The executables exist and have been repaired.**
   `scripts/pod/autoinit_phase_b_{driver,launch}.py`. A reviewer pass found three
   defects in the *paid execution path*, all now fixed and mutation-tested:

   * the inherited stage 2 built **six** candidates while the preregistration
     froze **eight** — journal seeding alone could not fix it, because a citation
     is only consulted if a descriptor exists for its candidate. A
     `candidate_universe()` seam now feeds stages 2/3/4 and the retention record,
     and the two retained Phase-A finalists are injected by digest and measured
     inside the search, cited rather than retrained;
   * the launcher inherited a **200 GiB** disk against a 244.87 GiB working set —
     now **300** with a fail-closed gate;
   * `fetch_products` returned id strings while `products_secured` expected
     transfer records — a mismatch that could only bite **after** a successful
     run. The proven record-returning fetcher is wired, and the secured gate now
     refuses a wrong shape instead of raising.

   **Attempt 1 aborted in the pod's own setup test gate**, 8.8 min, **$0.1500**,
   no scientific stage entered; the pod is deleted and the provider confirms it.
   Two defects, both introduced by this session: `verify_historical_probe_reuse`
   reconstructs Phase-A citations from `/home/ecs-user/aad-artifacts`, a dev-box
   artifact store no pod has, and its two tests ran inside the pod's gate; and a
   log directory created for the invalidated authorization was left out of
   `CATALOG.md`, so the structural test was **already red at the launch commit**.

   **Both are repaired.** The strict reconstruction now lives in a host-local test
   module excluded by a Phase-B-specific ignore list — Phase A's historical
   contract is untouched — and the same check runs on the dev box *before a pod
   exists* as `historical_reuse_reconstruction_gate`, over all 11 probes, the
   per-probe digest re-derivation, the evidence-set digest and the eight citations
   the ten-probe budget assumes. **The verifier itself was not weakened**; the
   question moved to the machine that can answer it. Nine mutations, all killed.

   Identities: executable source **`686d43aa…`** (57 files, set version 3, nothing
   uncovered — the verifier joined the set because it now decides whether a pod is
   created), preregistration **`3e466574…`**. Science unchanged: session plan
   `23f23649…`, science plan `02be33b9…`, both calibration spec **and** content
   identities, the closed 8-candidate set, thresholds, seeds and reuse rules all
   verified unmoved. Cost unchanged: floor **$13.0800**, ceiling **$26.8049**.

   **Attempt 2 then aborted one step past the repair**, 13.9 min, **$0.2300**, no
   scientific stage entered; the pod is deleted and the provider confirms it. The
   **pod test gate passed** — 2207 passed — so the attempt-1 defect is closed.
   Setup failed at `AUTHORIZATION_MISMATCH`: `autoinit_preflight_setup.sh`
   dispatches the authorization check on `SESSION_KIND` and has branches for
   `phase_a`, `recovery_continuation` and a `spend` default only, so a
   `PhaseBAuthorization` was loaded by `SpendAuthorization.load`, which reads a
   `preflight_plan_hash` the Phase-B schema does not carry.

   **ATTEMPT 3 ENTERED THE SCIENCE AND DID NOT COMPLETE — `EXECUTION_INCOMPLETE / NO_SCIENTIFIC_RESULT`.** Pod
   `wkausr939ts7vv`, 575.9 min, **$9.50**. **Stage 0 PASSED**: comparability held
   under `generation_runtime_comparability@v2`, both mixtures and the executable
   digest rebound on the pod, and the 8 historical citations imported — so the
   ten-probe budget stood and the rejected 14-probe path was never reachable.
   **Stage 1 hit its 544 min search deadline** and stopped fail-closed rather than
   running to the `$23.72` backstop. **No Top-5, no `sa`/`sb`/`sc` rungs, no
   selection.** **No calibration hypothesis was answered**, because the
   procedure that would answer it did not run. This is **not** a scientific null:
   Phase A's `unresolved_equivalence` is a completed outcome — the procedure ran
   to its end and the candidates were indistinguishable inside the frozen
   interval — and conflating the two would let an execution failure be cited as
   evidence about the calibration distributions.

   **The P=2 search was underpriced at the top of its own range** — 1.91–7.51 h
   priced, 9.08 h actual and unfinished. `depth.causal_kl_greedy_v1` took
   **388.2 min, 71.3% of the search**, over 12 expansions averaging 32.3 min.

   **The 2026-08-27 `$0` pass repaired all three.** *Journal retention:* Phase-B
   artifact specs now collect `phase_b_search/states.jsonl`, derived from the
   writer by test; the failure path stays optional so a search that died before
   writing cannot hold a pod open. *Speed, science-preserving:* the reference
   cache admits the first k mixture items that fit under the **unchanged** 0.66
   fraction instead of refusing 79% that did fit — 41.8% less causal-depth forward
   work, with cached/partial/recomputed proven to produce identical removal
   orders, score tables and metrics; and the activation-stats cache holds one
   entry per active profile, ending a P=2 thrash that cost four passes where two
   would do. *Telemetry:* per-expansion load/operator/materialize/identify/reload/
   validate/evaluate timings, plus the causal-depth reference breakdown, written
   beside the journal and never into state identity.

   **The cost model is re-derived and the ceiling has moved.** Search planning
   range **$5.2705–$9.5066**; search hard bound **16.461 h**; Stage-1 deadline
   **987.6348 min**; session floor **$16.4555** (was `$13.0800`); **session hard
   ceiling $35.6660**, implying a cumulative cap of **$275.5810** at the current
   spend. `children_max × mean` is gone as the ceiling — an average cannot bound a
   beam that retains expensive parents.

   **A second pricing pass closed three accounting gaps**, two of which were
   cancelling: `composite.stage1_sandwich_v0` reached the branching counts and was
   priced at zero; FFN/WIDTH statistics were charged per operator rather than once
   per `(parent, profile)` as `StatsCache` actually shares them; and the non-FLOP
   path — save, hash, canonical reload, validate — was covered by a TFLOP/s anchor
   measured on forward compute. The root is priced as unable to share, because
   `_stats_key` returns `None` without an artifact digest: **six** collections at
   level 0 against two per level-1 parent. The bound moved only 994.4 → 987.6 min,
   which is why a check on the total alone would have passed against both defects.

   **Approved 2026-08-27** (historical, for the search that has since been
   bought): session hard ceiling **$35.6660**, cumulative cap **$275.59**. The
   corrected pricing model, P1 and P2 are frozen.

   **ATTEMPT 4 RAN AND GOT FURTHER THAN ANY BEFORE IT.** Pod `zjwpsurs2dyvw8`,
   495.2 min, **$8.17**. Stage 0 passed in 2.2 min. **Stage 1 completed its
   search**, measured the canonical control and both imported finalists, and
   computed its ranking — then raised `AttributeError: 'Qwen3Config' object has no
   attribute 'run_id'` while building the summary record. Stages 2–5 never ran, so
   the outcome is again **`EXECUTION_INCOMPLETE / NO_SCIENTIFIC_RESULT`**. Pod
   deleted, provider-confirmed.

   **The defect is one line and Phase-B-only.** `scripts/autoinit/phase_a_search.py`
   holds the `SearchConfig` in a local named `config`, and the
   `for entry in retained_candidates:` loop rebinds it with
   `AutoConfig.from_pretrained(...)`. That loop body executes **only when retained
   candidates are passed, which is Phase B alone**, so no Phase-A run and no test
   ever reached it. Not fixed: it is a new material blocker and the standing
   instruction is to stop and report.

   **Both repairs from the previous pass are confirmed on hardware.** The search
   journal came home — 53 MB, retained out-of-tree — where attempt 3 lost it. P1
   ran in `partial` mode throughout (*"caching 52/67 items (78%, 13.2 GiB)"*), and
   the same 12 causal-depth expansions cost **287.3 min against attempt 3's
   388.2 — a 26% reduction**, measured rather than modelled.

   **Correction to this record:** an earlier version said the journal held "5
   searched + 2 composite + 2 imported finalists". Mechanically it holds **9
   BeamSearch leaves, all `provenance="search"`** — 2 composite and 7 decomposed —
   and **no imported finalists at all**, because `make_retained_state` builds them
   outside BeamSearch and nothing appends them to the store. Two of the seven
   carry the Phase-A finalists' state ids because ids are content-derived and the
   same composition was rediscovered, which is the reproducibility contract
   working rather than the finalists appearing.

   **The telemetry's first real reading accounts for 98.7% of Stage 1** and names
   the next bottleneck: **state evaluation is 31.2%** (147.2 min over 82
   expansions) against a cost model that prices it at seconds per child, while the
   materialization overhead I added is *over*priced at 4.8 min actual. Recorded,
   not acted on — pricing research is closed.

   **Repaired at `$0` on 2026-08-28.** The shadowing is fixed by lifecycle naming
   (`search_config` / `control_model_config` / `retained_model_config`); a
   whole-function CPU test now executes the Phase-B-only Stage-1 path — P=2, two
   real mixtures, two real retained candidates — and reverting the fix inside it
   reproduces attempt 4's failure exactly; and a completed search is committed to
   an atomic hash-bound `stage1_selection.json` **before** the control, the
   retained candidates or the summary, with the failed-run collector required to
   secure those five checkpoints when it exists. Executable source
   **`3ad6a6e3edf1…`** (59 files, set version 5), preregistration
   **`53f1347f6865…`**; pricing verified unmoved.

   **ATTEMPT 5 REACHED THE SCIENCE.** Pod `37aah10zvqk4lo`, 725.7 min, **$11.97**.
   Stage 0 passed; **Stage 1 completed the P=2 search in 464 min of its 987.6 min
   allowance and emitted an authoritative Top-5** — the attempt-4 shadowing fix
   held and `search_result.json` was written. Stage 2 ran **three new `sa`
   probes** and then failed:
   `ValueError: duplicate seeds in [20260726, 20260726]`.

   **The cause is a scientific finding, not a bug.** Two of the five searched
   leaves are **byte-identical** to two retained Phase-A finalists — `cca699c93f34`
   and `85bde4ded2c3` — verified by re-deriving `identify_checkpoint` on both the
   retained directory and the transferred copy. State ids are content-derived, so
   the larger P=2 search rediscovered the same compositions from the same root and
   produced the same bytes. The frozen candidate universe assumes 5 searched + 2
   imported + 1 control are **distinct**; in fact there were **6 distinct
   candidates**, and two ids carried both a cited historical `sa` probe and a fresh
   one at the same seed. The pooling guard refused, correctly.

   **The project state is now `PHASE_B_STAGE1_COMPLETE /
   BEHAVIOURAL_SELECTION_INCOMPLETE`** — not a null result and not a winner. Stage
   0, the joint P=2 search, the authoritative Top-5 and the Stage-1 selection
   artifact are all done and retained, and **all five Top-5 checkpoints are on the
   dev box with re-derived identities**. What is missing is the behavioural
   cross-phase selection.

   **Identity collapse (`df413bd9…`), by materialized identity only.** Collapse
   requires the state id **and** the re-derived artifact digest to agree; same id
   with different bytes is a refusal, not a merge; names, prefixes, `sa` scores and
   behavioural outcomes are never inputs. The searched role is primary; the
   imported role is an evidence alias. **The Top-5 is unchanged — no rank 6/7
   backfill.** The universe is **6 distinct candidates** (identity `e94f15d2…`),
   not the 8 the preregistration assumed, and the amendment records that the
   assumption was falsified rather than pretending it always said 6.

   **`POOLED_COUNTS_V2` is untouched** — its refusal was correct. The fix is one
   observation per `(initialization, seed)`.

   **Rung 1 was computed at `$0` by the frozen selection code**, on six candidates
   whose `sa` all reconstruct strictly: survivors `fe9683e6a9c7` (0.0471) and
   `85bde4ded2c3` (0.0412), control auto-advancing, no feasibility or catastrophic
   exclusions. Only `fe9683e6a9c7` lacks `sb`, and at most `fe9683e6a9c7` and the
   control lack `sc`.

   **So the next paid session is a behavioural continuation, not another Phase B**:
   floor **$5.4784** (1 probe), **ceiling $8.0691** (3 probes) — against the
   `$35.6660` that books a search already bought. **Current headroom `$23.7050`
   already covers it; no cap increase is needed.**

   **APPROVED for one paid behavioural-continuation session** at ceiling
   `$8.0691`, floor `$5.4784`, cumulative cap `$283.76` — no cap increase. Current
   launch identities are the CONTINUATION's, listed in the table above; the
   Attempt-5 figures (`$35.6660` ceiling, `$16.4555` floor, `987.6348 min` Stage-1
   deadline, executable `3ad6a6e3edf1…`, preregistration `53f1347f6865…`) are
   **historical evidence of a completed search** and are not launch identities.
   Detail: [`autoinit_phase_b_attempt4.json`](autoinit_phase_b_attempt4.json).

   **One further repair closed before attempt 4.** Phase B inherited Phase A's
   `--poll-limit-min 1320` — 22 h — against a corrected envelope that terminates
   at **1925.87 min (32.10 h)**, so the launcher would have stopped polling **606
   min** before the session reached its own hard bound, leaving a pod alive and
   billing. The Phase-B default is now **derived from the priced plan**:
   `hard_terminate_minutes` plus one poll interval, the checkpoint-fetch limit and
   a teardown time measured from attempt 3 — **1967.87 min (32.80 h)**, 42 min of
   slack. Phase A's 1320 is untouched, and `poll_lifetime_gate` refuses before a
   pod exists if the two ever drift apart again.
   Detail: [`autoinit_phase_b_attempt3.json`](autoinit_phase_b_attempt3.json).

   **Repaired by reviewer decision (Option A):** `autoinit_preflight_setup.sh` now
   has a dedicated `phase_b` branch that loads `PhaseBAuthorization`, binds the
   session plan, and asserts Phase B allowed / Phase A refused / no automatic
   follow-on — symmetric with the `phase_a` and `recovery_continuation` branches.
   Phase B is **not** routed through `SpendAuthorization` and `expected_usd` is not
   reintroduced: that loader falls back to `HARNESS_SOURCE_FILES_V1`, so it would
   have passed while binding Phase B to Phase A's file list. The exact branch body
   was executed against the real Phase-B artifact on the dev box.

   **A static completeness gate now prevents the class**
   (`tests/pod/test_session_kind_dispatch.py`): it builds every launcher's real
   `SessionSpec`, reads the `SESSION_KIND` it exports and the
   `authorization_loader` it uses, parses the shell branches, and requires the two
   to name the same class. Seven mutations, all killed — including a branch that
   exists but loads the wrong type, and a launcher declaring a kind with no branch.

   Identities after the repair: executable source **`45f6bb8c93ff…`** (57 files,
   set version 3), preregistration **`e167feaa502f…`**. Only those two moved;
   science plan, session plan, both calibration identities, candidate set,
   thresholds, seeds, floor and ceiling verified unchanged.
   Detail: [`autoinit_phase_b_reconstruction.md`](autoinit_phase_b_reconstruction.md),
   [`autoinit_phase_b_preregistration.json`](autoinit_phase_b_preregistration.json).
2. **Complete final initialization selection**, after Phase B. Phase A's
   `unresolved_equivalence` is not a final selection.
3. **The canonical Stage-1 NLL diagnostic**, run on the finally selected
   initialization under the established Stage-1 measurement contract.
   **Diagnostic only — never a promotion criterion**, and **not** the
   `state_eval@v1` search-side NLL fields, which are a different measurement and
   must not be substituted for it. This measurement has never been run.
4. **Only then**, the project's formal **Stage 2 / Stage 3 recovery training**.

**Nothing in this list is implemented or started**, and none of it is funded.
What IS funded is exactly one **behavioural-continuation** session, bounded by
the hard ceiling **`$8.0691`** against headroom `$23.7050`. `$5.4784` is a
planning floor, not an expected spend. The `$35.6750` / `$35.6660` full-Phase-B
allowance is **spent and closed** — Stage 1 was bought and retained by attempt 5.
Unused headroom is **not** authorization for any later experiment.

## The launcher defect — cosmetic here, must be fixed before any future session

`session_runner.py` computes
`all(f.get("rc") == 0 for f in fetched)`, but Phase A's `fetch_products` is
`finalists_to_fetch`, which returns **`canonical_id` strings**. `AttributeError`
— on a line only a **successful** Phase A reaches, which is why seven attempts
never hit it.

It fired **after** collection: 9 reports fetched, `local_hash_problems: []`, the
archive extracted with all 11 probe trees, and nothing owed off-pod — the two
retained finalists are *initializations* already preserved at 1.2 GiB each and
mirrored in transport. Verified present.

Consequence: the session reads `INCOMPLETE` / `passed: false` despite
`DRIVER_EXITED:0`, and teardown ran as an emergency delete. **A successful Phase A
currently cannot be recorded as successful.**

**Recovery continuation attempt 6 ran on 2026-08-23 and went one step further
than any attempt: $1.4926.** Pod `ifp8feyil1gp7v`, 90.5 min, deleted with
provider confirmation. Full record:
[`autoinit_recovery_continuation_attempt6/`](autoinit_recovery_continuation_attempt6/).

**The attempt-5 repair is confirmed on hardware.** The probe trained,
`trained_model_dir()` resolved its checkpoint with no `FileNotFoundError`, and
execution entered `battery()` — past the exact line that ended attempt 5. Setup
cost **$0.13** and TCP 22 came up in **0.2 min**, both the best yet.

**Generation then failed**, 50 s after `PROBE_TRAINED`:

> `uncapped_eval.py` → `tok.apply_chat_template(...)` →
> `ValueError: tokenizer.chat_template is not set`.
>
> `Trainer.save_checkpoint` writes `save_pretrained(ckpt_dir/"model")` — weights
> and config, **no tokenizer** — and `battery()` passes `--model <that dir>` with
> no `--tokenizer`, whose default is *"the checkpoint's own tokenizer"*.

Every previously proven caller passed `--model CANONICAL_INIT`, which *is* a full
checkpoint with `tokenizer*.json` and `chat_template.jinja`. The Phase-A battery
is the first ever to point `--model` at a **trainer-written** checkpoint, so the
default had never met one. Third appearance of the checkpoint-without-a-tokenizer
class.

**The obvious one-line fix is not protocol-neutral, and must not be applied
blind.** Stage 0 of this run attested `tokenizer_source: "the evaluated
checkpoint"` and `tokenizer_sha256: c1db93c8…` under the frozen protocol
`250f72ef…`. `RecoveryEvaluationProtocol.identity()` returns every declared
field, and `generation_runtime_comparability@v2` declares material *"every field
the protocol already declares except `runtime_digest`,
`evaluation_protocol_hash`, `generation_protocol_fingerprint`"* — so
`tokenizer_source` **is material**. Passing `--tokenizer` would change it to
`"external: …"` and make every probe incomparable to the Stage-3 controls that
materialized the thresholds. The protocol-neutral shape is to make *"the
evaluated checkpoint"* true instead. That is a maintainer decision.

**Recorded, not diagnosed:** the probe took **71.9 min** against attempt 5's 61.7
and the 61.55 priced — **+16.6%** on a different host. One observation is not a
trend, but nine probes at 71.9 would move the envelope. The trained probe was
**again lost with the pod**: 71.9 paid minutes, for the second time.

### The repair, approved and applied 2026-08-23

**The frozen rule is preserved by making it true, not by changing it.** After
`trained_model_dir()` resolves a trained probe and before `battery()` runs,
`materialize_eval_tokenizer()` copies `tokenizer.json`, `tokenizer_config.json`
and `chat_template.jinja` from the already-attested `CANONICAL_INIT` into the
checkpoint's `model/` directory. `battery()` still invokes `uncapped_eval.py`
with `--model <trained dir>` and **no `--tokenizer`**, so `tokenizer_source`
remains *"the evaluated checkpoint"* and the bytes are the ones Stage 0 attested.

Fail-closed and idempotent: a missing source sidecar raises; an absent
destination is copied and re-hashed; an identical destination is accepted; a
**differing** destination is refused rather than overwritten, because which
tokenizer a probe was scored against must stay recoverable.

`uncapped_eval.py`, the trainer, `Trainer.save_checkpoint`, the frozen recipe,
comparability v2, the Stage-3 attestation and every frozen threshold are
**untouched** — `uncapped_eval.py` is part of the generation source digest, so
editing it would have turned an infrastructure repair into a protocol change.

The continuation harness digest moves **`8b56fc7b…` → `b824441c…`**.

**Recovery continuation attempt 5 ran on 2026-08-22 and bought the most of any
attempt: $1.3511, both memory repairs verified on hardware, and the first
recovery probe TRAINED.** Pod `9jxov5bjtiy2xu`, 81.9 min, deleted with provider
confirmation. Full record:
[`autoinit_recovery_continuation_attempt5/`](autoinit_recovery_continuation_attempt5/).

**The memory repairs are confirmed**, against attempt 4's identical `before`:

| | attempt 4 | attempt 5 |
| --- | ---: | ---: |
| `freed_allocated_bytes` | **0** | **8,101,709,824** |
| allocated after | 7.55 GiB | **0.008 GiB** |
| free after | 36.32 GiB | **43.87 GiB** |
| `live_retention` | **true** | **false** |

The caller-owned release freed **7.54 GiB** attempt 4 could not free at all, and
`require_headroom` passed on **43.87 GiB against 43.65 required** — the figure the
repair predicted before the run, to two decimals.

**Then the first recovery probe trained**: `MARKER:PROBE_TRAINED` after **61.7
minutes**, against the 61.55 the budget is priced from.

**Stage 2 then failed reading that probe's checkpoint** — a writer/consumer gap
with the writer right and one of two consumers wrong:

> `train.py:1208` writes `out_dir/checkpoints/latest.txt` and
> `checkpoints/<tag>/model`. `train_stage3.py`'s resume path reads exactly that.
> `autoinit_phase_a_driver.py:736-738` reads `out_dir/latest.txt` and
> `out_dir/<tag>/model` — **dropping the `checkpoints/` component in both**.

No `$0` gate could catch it: the line is reached only after a real 62-minute
probe completes, and the simulator and rehearsal stub the training subprocess.
**The probe's own artifacts were lost with the pod** — the fetch spec collects
finalists, and a trained-but-unscored probe is not one.

### The repair, approved and applied 2026-08-23

`trained_model_dir(out_dir)` resolves `out_dir/checkpoints/latest.txt` →
`out_dir/checkpoints/<tag>/model` — what `Trainer.save_checkpoint` writes — and
`run_probe` calls it and does no path arithmetic of its own. Both failure modes
are named rather than raising a bare `FileNotFoundError`, because by then a probe
has been paid for and "wrote no checkpoint" and "index names a missing tag" are
different diagnoses.

Per the maintainer's instruction this is **option (a) only**: a driver-local
helper, no cross-module checkpoint abstraction and no harness expansion for a
hypothetical third consumer. The writer and `train_stage3.py`'s resume consumer
already agree.

**Retention was deliberately not implemented.** `restore_probe()` resumes only
completed *scored* journal entries, so making a trained-only probe reusable would
need a new bound journal state, cross-session staging, evaluation-only resume
semantics and a large-checkpoint transfer policy — disproportionate to this
blocker, and it would enlarge the final paid attempt's change surface.
**Attempt 5's trained probe is not resumable scientific work: attempt 6 starts
recovery normally from the preserved Stage-1 inputs and retrains it.**

**The `$0` harness had been simulating a layout the trainer never writes.**
`tests/pod/test_phase_a_stages1_5_execute.py` *does* execute `run_probe` end to
end — but its fake trainer wrote `out/latest.txt` and `out/step000/model`, the
layout the **driver** wrongly expected. Two artifacts agreed with each other and
both disagreed with `Trainer.save_checkpoint`, which
`tests/training/test_train.py` has always asserted correctly. The fake now writes
the real layout, and with the driver defect reintroduced that harness fails
**1 + 14** — so it would have caught attempt 5.

The continuation harness digest moves **`95cf336d…` → `0dbf1272…`**.

**Recovery continuation attempt 4 ran on 2026-08-22 and got further than any
attempt before it: $0.4112, Stages 0 and 1 PASSED on hardware, Stage 2 OOM.**
Pod `k1mgu38q0y6sei`, 24.9 min, deleted with provider confirmation. Full record:
[`autoinit_recovery_continuation_attempt4/`](autoinit_recovery_continuation_attempt4/).

**What it bought, and it is real:**

* **Stage 0** attested the frozen protocol on hardware — interval `0.011695`,
  floor `0.3000`, plan `02be33b9…`;
* **Stage 1 imported the five Attempt-12 leaves in the frozen selected order and
  re-identified every one of them from the bytes that arrived on the pod**,
  matching the Stage-1 artifact and shard digests. `config_hash 567d32789ba6…`.
  No search was run, and none was reachable;
* **the canonical control was measured once** on the frozen `state_eval@v1`
  suite (74022 positions), producing `artifact_digest dc9500d3…`;
* admission accepted the five leaves plus the measured control.

That is the whole import path — transport, staging, strict byte re-identification,
control measurement, admission — demonstrated end to end on a paid pod.

**Then Stage 2's first rung-1 probe hit a CUDA OOM**, and two defects compounded:

> **The release freed nothing.** `release_to_subprocess(drop=[teacher, evaluator])`
> reports `freed_allocated_bytes: 0` — allocated was 8,110,229,504 B before *and*
> after; only reserved cache came back. The handoff diagnosed it correctly:
> `live_retention: true`, *"a genuine retention, not allocator caching"*.
>
> **The headroom gate's estimate of the trainer is ~14 GiB too low.**
> `require_headroom` demands `RECOVERY_TRAINER_BYTES + margin` = 22 + 2 = **24.00
> GiB** and saw **36.32 GiB** free, so it passed with 12.32 GiB of apparent slack.
> The probe then used **36.30 GiB** and OOM'd asking for 298 MiB more.

Had the release worked, ~43.87 GiB would have been free and the probe would have
fit. Had the threshold matched the real trainer, the gate would have refused at
`$0.36` with a diagnosis instead of an OOM. **This is attempt 12's class, and the
gate written to stop it — whose refusal message names attempt 12 — did not fire,
because it was calibrated against a trainer footprint that was never measured.**

### Both repairs, approved and applied 2026-08-23

**(a) The handoff lifecycle.** The old API was the defect: it copied the caller's
sequence into a callee-local list and cleared *that*, and it measured `after`
before the caller's `del` ran. `complete_release(before)` now **accepts no
objects** — the caller snapshots, deletes its own names, then measures — applied
identically to the continuation's teacher/evaluator path and Phase A's `found`
path. `require_released` makes the existing `live_retention` verdict **block**
rather than merely be recorded, above the same 1 GiB limit, and is kept separate
from `require_headroom` so a caller can tell which failed.

**(b) The trainer requirement, measured.** `RECOVERY_TRAINER_BYTES` was `22 GiB`
— **17.79 GiB below the trainer's measured peak**. Attempt 4's 36.30 GiB is only
a lower bound, since it died before `backward()` and the first `AdamW.step()`.
The permanent controls `preflight_ctl_r0860k_{sa,sb}` report
`max_memory_allocated()` of **39.79 GiB**, identically, each over a completed
1023-step run on an **L40S**, running every memory-relevant field of the frozen
recipe. Plus 1.35 GiB reserved slack and 0.51 GiB non-PyTorch overhead — both
read off attempt 4's own OOM decomposition — gives **41.65 GiB**. Basis recorded
in [`autoinit_recovery_trainer_memory_basis.json`](autoinit_recovery_trainer_memory_basis.json)
and pinned by a test. **No calibration launch was bought.**

The margin is unchanged, and the recipe was not touched. **The tightness is now
visible:** 41.65 + 2.00 = **43.65 GiB required against a 44.39 GiB card**. A
correctly released card offers ~43.87 GiB — attempt 4's own figures put the
driver's non-allocated overhead at 0.52 GiB — so it clears by **~0.22 GiB**. This
recipe genuinely almost fills an L40S, which is why a retention of any size is
fatal.

**The continuation harness digest moves `162c09ed…` → `95cf336d…`**, because
unlike attempt 3's repair these files are inside the 22-file set.

**Recovery continuation attempt 3 ran on 2026-08-22 and bought nothing: $0.2011,
no stage executed.** Pod `ku8vcn5mu8hp9i`, 12.2 min, deleted with provider
confirmation. Full record:
[`autoinit_recovery_continuation_attempt3/`](autoinit_recovery_continuation_attempt3/).

**The transport premise is now proven on a paid pod, which is the result worth
keeping.** The pre-provider gate read *"25 relay inputs (10 from
AlphaAvatar/aadistill-artifacts, 15 from AlphaAvatar/aadistill-transport), 2 local
assets"* — attempt 2's read *"10 relay inputs, 7 local assets"* — and the pod then
reached `ASSETS_STAGED`, `ASSETS_READY` and `VLLM_READY`. The setup script runs
under `set -euo pipefail` and marks strictly in order, so those markers prove all
25 declared inputs were fetched from their own declared repositories and every
declared sha256 verified at every landing site, **including 5.5513 GiB of Stage-1
leaves pulled from the transport repo at hub speed**.

**Then the setup CPU test gate failed**, and the cause is a dev-box path in a
dev-box tool:

> `scripts/autoinit/publish_selected_leaves.py:199` calls
> `tempfile.mkdtemp(prefix="leaf-roundtrip-", dir="/home/ecs-user/aad-scratch")`.
> That directory does not exist on a pod, so `mkdtemp` raises and the five tests
> in `tests/autoinit/test_leaf_transport_publish.py` that reach `verify()` fail.

**Reproduced at $0 rather than inferred**, by running the real module in a mount
namespace holding the repo and the interpreter but no `/home/ecs-user/aad-scratch`:
**5 failed**, matching the pod's count exactly, and recovering the two failure
names the 40-line `setup.log` tail did not transport.

**This is attempt 8's class, one step out.** Attempt 8 was a `$0` test *asserting*
dev-box filesystem state. This is a `$0` test *executing production code that
requires* it. The layout test skips a **declared** host-local root where absent —
correctly — and nothing connects "this path is host-local" to "code requiring it
must not run on a pod". The pod simulator simulates the pod's repository tree, not
its host filesystem. `publish_selected_leaves.py` is deliberately **not** in the
22-file harness, but its test module is not in `TEST_IGNORES` either, so the pod
ran it.

A retry needs a **new grant and a new authorization**: attempt 3's are spent, and
although a fix to that file would not move the harness digest, it would move the
session commit, which the lineage gate constrains to differ from its base in
exactly one path.

### The repair, approved and applied 2026-08-22

`scratch_dir()` returns a **preference, never a requirement**: the configured
`AAD_SCRATCH` if it names an existing directory, else the dev box's own root if
present, else `None` — which lets `tempfile` choose. Dev-box behaviour is
unchanged.

**Running it under real pod conditions found a second instance of the same defect
one line further down.** `token()` read `~/.cache/huggingface/token`, which a pod
does not have — every pod-side script in the setup reads `HF_TOKEN`, which the
setup exports *before* the test gate runs. It is evaluated as an **argument** to
`hf_hub_download`, so it executes even in tests that patch the download away, and
it only surfaced once `mkdtemp` stopped failing first. `token()` now reads the
environment first and falls back to the file.

A third test, `test_the_before_manifest_refuses_a_drifted_canonical_shard`, drove
`build_before()` against the **real 5.55 GiB canonical store**, so on any host
without it the run exited at "canonical leaf missing" and the drift assertion
never executed. It now builds a synthetic store and record. This is not a
weakening: deleting the drift check still fails it.

Per the maintainer's instruction the module was **not** added to `TEST_IGNORES`
and **no** generic hard-coded-host-root guard was added.

**Recovery continuation attempt 2 ran on 2026-08-21 and bought nothing: $0.2389,
no stage executed.** Pod `7hthdteyc25xgx`, 14.5 min, deleted with provider
confirmation. **The provider-resilience closure worked** — the readiness poll
that killed attempt 1 reached TCP 22 in 3.7 min and the image identity was
confirmed. It then failed staging the first Stage-1 leaf, and the reason is
arithmetic rather than luck:

> `SessionRunner` scps each declared `LOCAL_ASSET` with
> `subprocess.run(…, timeout=600)`, which **raises**. One leaf is **1.110 GiB**,
> so fitting 600 s needs **1.99 MB/s sustained**. This session's own bundle
> upload minutes earlier ran at **0.44 MB/s**; the recorded dev-box uplink is
> **0.72 MB/s**. One leaf needs 28–45 min against a 10-minute timeout — over by
> **3–4.5×** — and four more leaves would have followed.

**SUPERSEDED.** At the time, the five leaves had no route to a pod: scp was
infeasible above, and the relay alternative reported **1.60 GiB** of headroom
against **5.55 GiB** of leaves. That was resolved by the transport mirror, and
attempts 3 and 4 have since staged all five on paid hardware. Full record:
[`autoinit_recovery_continuation_attempt2/`](autoinit_recovery_continuation_attempt2/).

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
cumulative spend   $260.0550  incl. Phase-B attempts 1-5 at $30.0200
approved cap       $283.7600  no increase requested or required
remaining          $23.7050   headroom — COVERS the $8.0691 continuation ceiling

```

The cap went **$219.00 → $231.00** (2026-08-20, to fund exactly one Phase-A
attempt) **→ $234.00** (2026-08-21). Since then attempt 12 **ran** for $3.7872 —
far under its $23.0484 ceiling, because it stopped at Stage 2 rather than
training nine probes — and the four recovery continuations spent $0.0100,
$0.2389, $0.2011 and $0.4112 — the last of which passed Stages 0 and 1. A
continuation's derived **$16.7456** ceiling still fits inside the remaining
$19.6674, with $2.9218 to spare; a full Phase-A attempt does not. **Every grant
issued so far is spent.**

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
* Stage 0 on hardware, passed five times — attempt 10 attesting protocol
  `250f72ef`, identity `70a26e0b`, science plan `02be33b9`
* Phase-A **Stage 1**, passed twice with byte-identical results; attempt 12's
  five selected leaves are preserved off-pod and verify locally
* the continuation's own authorization type, harness digest and issuer

## Latest verification

**2026-08-29 — after the v3 source-digest repair and the pre-provider gate run.**
CPU only, repo `.venv`. No checkpoint trained, no GPU, no pod, `$0` spent.

* **full suite: 2468 passed, 12 skipped, 0 failed** (27:51)
* frozen-asset verifier clean; both probe-reuse verifiers re-run with unchanged
  `probes_dir_digest` values
* continuation: **42 tests**, seven of which drive the **real** shared
  `session_commit_gate` rather than comparing the digest against itself
* **17 mutations, no survivors** — 12 behavioural, 5 against the digest contract
  (old `sha256_json`, unsorted, dropped separator, changed separator, reverted
  version)
* pre-provider gates at the launch commit: **6 of 7 green**, including
  `session_commit_gate` with `harness_matches`, `commit_carries_this_authorization`
  and `lineage.ok` all true. `ckpt_store_capacity_gate` **BLOCKS** — see above
* an earlier run of this suite reported 14 failures; every one was the disk-full
  event, not code. One tracked config had been truncated to 0 bytes and committed
  by an unchecked `git add -A`; restored from `49861a5`, and no other tracked
  file is zero-length

## What failed, and why

The early runs are two classes, both closed by construction: GPU-only device
code (fixed by `autoinit.stage1_device_contract@v1`) and a contract owned by
inherited machinery (fixed by the session specification). **Continuation 1 is a
third:** an unguarded transport call on a path that only runs while a pod is
already billing.

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
| Phase A 12 | $3.7872 | stage 2, first rung-1 probe | CUDA OOM: the driver still held the search's ~24.05 GiB. **Stages 0-1 passed; five leaves preserved off-pod** |
| continuation 1 | $0.0100 | launcher readiness poll, 27 s in | `wait_endpoint` calls `provider._gql` uncaught against an endpoint measured at **25% transport failure**. Every gate passed; no stage ran |
| continuation 2 | $0.2389 | LOCAL_ASSET staging, 10.5 min in | scp of one 1.110 GiB leaf against a hard-coded 600 s timeout: needs 1.99 MB/s, dev box gives ≤0.79. **Closed by the transport mirror** |
| continuation 3 | $0.2011 | setup CPU test gate, 12.2 min in | `publish_selected_leaves.verify()` mkdtemps into `/home/ecs-user/aad-scratch`, absent on a pod. **The 25-input transport staging PASSED first** |
| continuation 4 | $0.4112 | stage 2, first rung-1 probe | CUDA OOM: the release freed 0 allocated bytes and the headroom gate demands 24 GiB for a probe that needs ≳36.6. **Stages 0 and 1 PASSED on hardware** |
| continuation 5 | $1.3511 | stage 2, after the probe trained | `latest.txt` read without the `checkpoints/` component the trainer writes. **Both memory repairs verified; the first probe TRAINED in 61.7 min** |

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

## The continuation ran, and bought nothing — $0.01

**Attempt 1, 2026-08-21.** Authorization `autoinit.recovery_continuation.2026-08-21T1642Z`,
base `b1ebbb6`, session commit `8c7c42e`, bundle `aad_autoinit_8c7c42e1.bundle`.
Pod `dckc72mtoe9ijw`, L40S, **0.7 min, $0.01, provider confirms gone**. Full
record: [`autoinit_recovery_continuation_attempt1/`](autoinit_recovery_continuation_attempt1/).

**The whole chain worked.** One-use grant, clean base, continuation-specific
issuance, an authorization-only commit differing from its base in exactly one
path, a bundle whose bytes and checkout round-tripped and whose **harness digest
recomputed from the relay checkout** matched, and all four pre-provider `$0`
gates green — including all five preserved leaves verifying locally.

**Then the launcher died 27 seconds after creating the pod**, on
`URLError: SSL UNEXPECTED_EOF` in its readiness poll. **No stage ran. No leaf was
read on the pod. No science changed.**

### Root cause, measured at $0 afterwards

The RunPod GraphQL endpoint was failing **5 of 20 requests — 25%** (SSL EOF,
`ECONNRESET`, `RemoteDisconnected`). `session_runner.wait_endpoint()` polls it
every 10 s for up to 15 minutes — **up to 90 calls** — via `provider._gql()`
**directly**, and catches nothing. `provider.get()` is the wrapper built for
exactly this, carrying the comment *"Never raises. A watchdog that dies on a
transient 502 is not a backstop."* The launcher deserves the same tolerance and
did not have it. At 25% loss, surviving five polls is ~24%: **relaunching
unchanged would repeat, not gamble.**

Blast radius is bounded. Three `_gql` sites are uncaught, all pre-driver:
`check_gpu_offered` (1 call, $0), `wait_endpoint` (≤90, billing ← this),
`read_image_digest` (1). **The 15-hour main poll uses `get()`** and is not
exposed.

### Closed — all three sites, not only the one that failed

`SessionRunner` had **three** direct `provider._gql()` calls, and two of them run
while a pod is billing. Fixing only the endpoint poll would have moved the same
failure one step later, into `read_image_digest()`, after SSH was already up.

The classification now lives in one place. `provider.TRANSIENT_TRANSPORT` names
what `get()` has always caught — `URLError`, `OSError`, `ValueError`
(`JSONDecodeError` is a truncated body), `TimeoutError` — and
`provider.observe()` applies it to *any* query, returning an `Observation` that
**never raises** and reports a declined answer as **unknown**, never as "no".

| site | billing? | behaviour now |
| --- | --- | --- |
| `check_gpu_offered` | no, `$0` | retries on the existing `--create-attempts` × `--create-retry-seconds`, then aborts cleanly. **No pod is created and nothing propagates through the launcher.** An unanswered price query is unknown, not "GPU not offered" |
| `wait_endpoint` | **yes** | keeps polling under the **caller's** `startup_limit_min` deadline. A failed observation costs exactly one poll interval and is never read as no-ports or as gone |
| `read_image_digest` | **yes** | retries under the *same* deadline, then **fails closed** with `ImageIdentityUnavailable` → `no_image_identity` → teardown before setup runs. It no longer falls back to `self.a.image`, which is what we *asked for*, not what is running |

`setup_on_draw` now owns **one** startup deadline for both billing observations,
so the operator's `startup_limit_min` cannot silently cover two windows. No new
timeout constant was added and no deadline value changed.

**`no_image_identity` is deliberately not redrawable**: the control plane failed,
not the host, so redrawing onto another machine would just pay again for the same
unanswered question.

**14 mutations.** One initially *passed* and is worth naming: deleting the
`continue` after a failed observation still loops and still recovers, so every
outcome-level assertion held — but the failure then falls into the port scan with
no data, sleeps a **second** time and advances the progress counter, spending the
startup deadline at twice the intended rate. The tests now pin the sleep
sequence, not just the outcome.

**The class cannot come back quietly.** A test walks `session_runner.py`'s syntax
tree and fails on **any** `_gql` attribute access, so a future single-shot in a
paid path — precisely attempt 1 — breaks the suite instead of a pod.

**This invalidates the spent authorization, by design.** The continuation harness
digest moved `f2ea4332…` → `e5a7183a…`, and the consumed artifact now refuses it.
That is the mechanism working, not a problem to route around.

### Why this stopped instead of retrying

The grant says *"Consumed by exactly one issuance."* Fixing the launcher moves
the continuation harness digest, which invalidates the authorization **by
design** and would need a second issuance from a spent one-use grant. That is a
maintainer decision.

**Since applied, and broadened** — see *Closed — all three sites* above. Attempt
1 spent **$0.01** and is a closed record; attempt 2 is the run in flight.

### The continuation is now covered by the shared structural contracts

`tests/pod/session_specs.py` defines the set of real sessions the generic
`SessionSpec`/setup/staging checks run against, and the continuation was absent
from it — so the session about to be paid for was the one session those contracts
did not cover. It is now in `SESSION_LAUNCHERS`, and it passes all of them,
including the **executed** staging block: its real manifest, through the real
shell code, landing the real destinations.

Adding it required separating two identities the uniqueness test had conflated.
A `plan_hash` names *what science is being run*; a status file, log,
authorization path and job id name *which run is running*. The invariant now
requires **`session_id`, `schema`, `status_path`, `run_log_path`,
`authorization_path` and `driver_job_id`** to be distinct across operational
sessions, and no longer requires `plan_hash` to be globally unique — because the
continuation runs the frozen Phase-A plan from Stage 2, and demanding uniqueness
there would have forced a **frozen scientific identity** to change to satisfy a
test about file names.

Dropping that field removes a check, so it is replaced by a stronger, specific
one: the continuation must share the **full** Phase-A `plan_hash`
(`9377a2dc…`, asserted literally) while differing in all six operational fields,
loading `RecoveryContinuationAuthorization` rather than Phase A's, carrying a
harness set that excludes the search, pricing with no `stage1_beam_search` phase
and no Stage-1 reserves to a hard `$16.7456`, and declaring
`runs_a_search == False`.

**Mutation-verified 9 ways**, including that removing the continuation from the
covered set fails, that each of the six uniqueness fields fails on a real
collision, and that rewriting the continuation's `plan_hash` — the thing this
change exists *not* to do — fails.

## Leaf transport is solved and verified

**The five Attempt-12 leaves are mirrored** in the private
`AlphaAvatar/aadistill-transport` repo — 15 files, **5.5513 GiB**, in the frozen
selected order. The manifest
[`autoinit_selected_leaf_transport_manifest.json`](autoinit_selected_leaf_transport_manifest.json)
is marked `verified: true`, and the continuation's `$0` gate accepts it.

Verified three independent ways, because a copy that is *present* but *wrong*
would be discovered on a billing pod:

| check | result |
| --- | --- |
| remote size + hub LFS sha256 OID, no bytes moved | 15/15 |
| round-trip download, re-hashed locally | 15/15 |
| `verify_transferred_leaf` re-identification | **5/5 `matched` and `shard_matched`** |
| `artifact_digest` reproduces the attempt-12 record | **5/5** |

The session now declares **25** relay inputs — 10 from the main relay, 15 from
transport — and only **2** small artifacts remain on the scp path.

### What made it possible

Retiring **8.4752 GiB** of remote copies, an explicit maintainer decision after
the account-wide finding. These are **remote retention changes, not retirements
of science**: every checkpoint still exists canonically on local disk with
unchanged identity and hashes.
Record: [`autoinit_relay_retention_20260822.json`](autoinit_relay_retention_20260822.json).

Quota: inventory **92.7330 → 84.2578 GiB** (8.8722 free), then the leaves
uploaded to **89.8091 GiB**, leaving **3.3209 GiB** of headroom.

**The reclaim was capped by evidence, not effort.** 32 objects totalling
**69.671 GiB** have no byte-identical local copy and are the only surviving
copies — `e1_scaling_20260801` alone is 42.19 GiB across 19 objects, the largest
and most obviously obsolete group by name, untouchable for exactly that reason.
8.47 GiB was all the duplicated bytes there were, and **this lever is now spent**.

**A near-miss:** the two wheelhouses (7.4 GiB) are fetched by `snapshot_download`
inside the setup script, *not* through the `RelayInput` contract, so "delete what
no manifest declares" would have flagged them — putting PyPI back on the paid
critical path, the failure that cost $2.07. Protected by prefix, re-derived
inside the deletion script.

**Rates, for the record:** leaves 3–5 uploaded at **0.76–0.79 MB/s** — the same
rate that made scp impossible. Nothing about the dev box improved; the push now
happens once, at `$0`, off the paid path, and the pod pulls from the hub instead.
Leaves 1–2 went at 164 and 241 MB/s because the hub deduplicated them, which
incidentally showed the quota-blocked run had transferred leaf 2's bytes and only
had its commit refused.

## How the second-repo premise was refuted (historical)

The maintainer directed that the five leaves reach a pod through a second
private Hugging Face repo, on the strength of a 2026-08-13 note: *"a 1 MiB write
to a different private repo succeeded, so the limit binds per-repo, not
account-wide."* **That conclusion is refuted.**

A dedicated private repo `AlphaAvatar/aadistill-transport` was created and the
leaves uploaded in the frozen selected order. It accepted **exactly one leaf —
1.110 GiB** — and then refused:

> `BadRequestError: Private repository storage limit reached`

| repo | files | size |
| --- | ---: | ---: |
| `aadistill-transport` (**brand new**) | 6 | **1.1103 GiB** |
| `aadistill-artifacts` | 1162 | 92.1687 GiB |
| **combined** | | **93.279 GiB** vs the ~**93.13 GiB** recorded limit |

A fresh repository got **no allowance of its own** — it consumed the account's
remaining slack and stopped. The limit is **account-wide**.

**Why the earlier inference failed, which is the reusable part.** A 1 MiB write
succeeding is equally consistent with an account-wide limit that happens to have
≥1 MiB of slack — which is exactly what it had. A token-sized probe cannot
distinguish the two hypotheses; only a write at **representative size** can. Any
replacement transport must be tested at ~5.55 GiB, not with a small file.
Measurements: [`autoinit_leaf_transport_quota_finding.json`](autoinit_leaf_transport_quota_finding.json).

**Also measured, and it confirms the attempt-2 diagnosis independently:** leaf 1
uploaded at **0.69 MB/s (28.8 min)**. The scp path needs 1.99 MB/s to fit the
600 s per-asset timeout — 2.9× over.

### What was built anyway, and is worth keeping

The multi-repo contract is **transport-agnostic** and mutation-verified, so
whatever route is chosen, a session can declare where its bytes come from:

* `RelayInput` carries `repo`, defaulting to the main relay — nothing existing
  moved, and the ten Phase-A science inputs are unchanged;
* `SESSION_RELAY_INPUTS` serializes it, so the session record preserves it;
* the shared setup fetches each item from **its** declared repo and now names no
  repository of its own in the staging block; an item with no repo fails closed;
* the `$0` precheck groups declared inputs **by repository** and lists every one,
  so a leaf in the wrong repo, a changed repo id, or one missing remote file all
  refuse before a provider call.

**Mutation-verified 14 ways** across the transport publisher (4 verification
layers plus canonical-drift and unverified-manifest refusal) and the multi-repo
contract (single-repo revert, repo dropped from the env, the shell naming a repo
again, the default moved), plus an **executed** two-repo staging test.

The continuation now declares the leaves as relay inputs and **refuses at `$0`**
while no verified transport manifest exists — it does not silently declare inputs
that would 404 on a pod.

**Nothing was deleted.** The 1.110 GiB in the transport repo is a re-creatable
copy whose canonical original is intact and verifying; removing it would reclaim
account quota, but that is the maintainer's call.

## How the continuation was built

**The recovery continuation is an executable session.** An earlier commit
shipped only the primitives — the production path still priced with `budget()`,
launched `--stage all` against the full driver, and ran the search
unconditionally. Authorizing that would have rerun the 203-minute search the work
existed to avoid.

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

Cap **$234.00**, spent **$213.4814**, remaining **$20.5186** — attempt 1 cost
**$0.01**. A further continuation's `$16.7456` still fits with **$3.77** to
spare. No cap increase requested; **remaining balance is not authorization**.

**The next review is a GO/NO-GO for Recovery Continuation Attempt 2 under the
same derived `$16.7456` ceiling. It needs a NEW one-use grant: the previous one
is spent, and the fix moved the harness it was bound to.**

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
