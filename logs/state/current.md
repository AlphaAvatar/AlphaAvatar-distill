# Current state

**Updated:** 2026-09-16. The human view. Every number here has an owner named
beside it, and this file restates none of them from memory — a second
hand-maintained copy of a cost or a status is how two documents come to
disagree.

Start at [`README.md`](../README.md) if you do not know which document you want.

## Right now

**A PAID SESSION IS RUNNING AND BILLING.** Phase-C2 Search-1 `attempt4`
launched 2026-09-16 at 18:58 UTC on pod `vqwg6o4ftpda4b`, NVIDIA L40S at
`$1.09/h`, under a `$15.0446` ceiling with a `$14.4996` soft stop and an
independent watchdog at 828 min. All nine `$0` pre-provider gates passed.
Live evidence: `/home/ecs-user/aad-scratch/c2_attempt4/` (launcher, an
independent read-only cost poller, the watchdog journal). The beam envelope is
635.96 min, so the session may run ~11 h; it trains nothing.

Attempt 4's chain is complete and **consumed**: grant `e76e42b` → launch-bound
readiness `b852082` → authorization `74cfe3e` → bundle `aad_autoinit_74cfe3e3`
(`ac7634cd85e5…`). Attempts 1–3 are historical and immutable and none of their
artifacts is reused.

Attempts 2 and 3 both launched and both aborted before the beam search —
`$0.0552` and `$0.1674`, `$0.2226` together, **no measurement of any kind**.
Neither is a Search-1 null, partial or scientific result. Attempt 3's pod was
deleted 9.2 min after creation and the provider confirmed it gone.

Attempt 2's root cause is repaired *and confirmed on real hardware*: attempt 3's
setup passed in 2 min 33 s, the frozen-asset step verified C2's own expectation
document, and the vLLM environment was never built. Attempt 3 then died one
second into stage `bind_identities` on an **empty calibration registry** — the
driver imported the application bootstrap inside stage B while stage A already
called `get_profile`. Repaired at module scope, where an import cannot be
ordered after a stage. The same abort exposed two defects in the failure path,
both repaired: both C2 artifact specs declared a lifecycle that does not exist,
which made them unloadable — **including the success spec, where a completed
ten-hour run would have held a `$1.09/h` pod until the 828-minute watchdog** —
and a session that declares no event streams could not satisfy the emergency
teardown gate's naming rule once its manifest was gone.

The **2026-09-16 maintainer campaign decision** authorizes fresh formal chains
after an ordinary pre-science repair without a further approval, inside a
`$16.20` envelope measured from `$290.5174` and *including* attempt 2's
`$0.0552`. Every formal launcher keeps the unchanged `$15.0446` ceiling and the
`<= $1.09/h` L40S basis, and a launch requires that a full ceiling still fit:
`0.2226 + 15.0446 = 15.2672 <= 16.20`. Engineering GPU validation has spent
`$0.0000` of its `$0.25`, and none is planned: the real stage A now executes in
a fresh interpreter off-pod and verifies the frozen B spec `3a233a90…`.

**If attempt 4 aborts before the beam search begins**, the campaign decision's
flow applies without asking again: preserve evidence, provider-confirmed
teardown, reconcile, diagnose, minimal repair, fresh identity, continue — while
a full ceiling still fits. **Once the beam search has begun, no automatic
formal retry is authorized**; a failure after that point is preserved, torn
down, reconciled and reported.

**C1 is COMPLETE.** Attempt 18 executed the whole frozen protocol — both replay
gates, both arms, six probes trained, six evaluated on the frozen battery — and
the frozen Stage-I rule returned **`GO`** at `$10.2018`, under its own planning
floor. A complete valid verdict ends the round.

| | | owner |
| --- | --- | --- |
| phase | C1 — fixed-path ATTENTION isolation, **CLOSED by a verdict**, and its execution preregistration is now **frozen to the binding attempt 18 ran under**. C2 Search-1 is **EXECUTABLE and priced, and NOT AUTHORIZED**: space, baseline rule, B→C comparison record, driver, launcher, evidence contract, a `$15.0446` ceiling, a CPU preflight of its own, a derived executable closure, a readiness contract, an authorization issuer, a bundle transport, run ownership of the whole execution and an enforceable provider-resource scope all exist, with the beam and the baseline rebuild on separate clocks; no grant, readiness record, authorization or bundle does | [`phase_c2/plans/phase_c2_search1_plan.md`](../stages/stage-1/phase_c2/plans/phase_c2_search1_plan.md) · [`phase_c1/plans/phase_c_roadmap.md`](../stages/stage-1/phase_c1/plans/phase_c_roadmap.md) |
| replay | **MEASURED — 2/2 PASS**, for the third time (attempts 9, 17, 18). Passing replay is not a result: 9 and 17 are **NO DECISION**, pre-treatment aborts that measured no endpoint. Attempt 18 is the only attempt that decided anything | [`attempt18/closeout/outcome.json`](../stages/stage-1/phase_c1/runs/attempt18/closeout/outcome.json) |
| treatment, endpoint | **MEASURED** — six probes trained and six evaluated on the frozen battery; the frozen Stage-I rule returned **`GO`**. Figures in the block below | [`attempt18/evidence/c1_decision.json`](../stages/stage-1/phase_c1/runs/attempt18/evidence/c1_decision.json) |
| launch chain | every C2 chain so far is **consumed** — attempts 1, 2 and 3 — and **nothing is prepared**. No further C1 attempt is authorized or prepared either; a complete verdict ended that round | [`phase_c2/runs/attempt3/governance/`](../stages/stage-1/phase_c2/runs/attempt3/governance/) |
| last attempt | **C2 Search-1 attempt 3 — ABORTED IN STAGE A, `$0.1674`, pre-science, nothing measured.** 9/9 `$0` gates and setup passed; the driver died one second in on an empty calibration registry. Pod deleted and provider-confirmed gone. The last complete scientific execution remains **C1 attempt 18** (`$10.2018`, `ALL_DONE`, verdict `GO`) | [`c2 attempt3`](../stages/stage-1/phase_c2/runs/attempt3/closeout/outcome.json) · [`c1 attempt18`](../stages/stage-1/phase_c1/runs/attempt18/closeout/outcome.json) |
| blocker | **none blocking attempt 4**: the campaign decision authorizes a fresh chain after an ordinary pre-science repair, the repair is done and a full `$15.0446` ceiling still fits. C1 is finished. One **resource** decision remains the maintainer's: durable large-artifact capacity, see the defects block below | [`budget/decisions.md`](../budget/decisions.md) |
| spend | owned by the budget block below | [`budget/ledger.md`](../budget/ledger.md) |

## Readiness

<!-- readiness:begin -->

| readiness | | owner |
| --- | --- | --- |
| latest sweep | **launch_bound — PASS**, swept at `e80eb60b`; **does not describe the current tree** | [`c1_pod_environment_verification.json`](../stages/stage-1/phase_c1/analyses/c1_pod_environment_verification.json) |
| launch-bound for the next session | **not prepared** — a launch-bound sweep on the final clean pre-authorization tree is owed | this file's launch-chain section |
| last launch-bound failure | swept at `82745981` on 2026-09-12 — kept as history, not a current state | [`readiness_history.json`](../stages/stage-1/phase_c1/history/readiness_history.json) |

*Generated from the record by `scripts/consolidate/render_log_navigation.py`; do not edit by hand — it went stale within hours when it was prose.*

<!-- readiness:end -->

## Budget — four limits that do not transfer

Derived by [`scripts/consolidate/derive_budget.py`](../../scripts/consolidate/derive_budget.py)
from the approved package and each session's own closeout. **Do not restate
these by hand; run the deriver.**

<!-- budget:begin -->

| limit | remaining |
| --- | --- |
| formal sessions | `$22.8249` of `$45.4425` |
| GPU engineering | `$6.0000` of `$6.0000` |
| package | `$28.8249` of `$51.4425` |
| project cap | `$290.7400` spent of `$320.0000`, leaving `$29.2600` |

**Full-ceiling sessions the FORMAL allowance funds: 1.** 2 ceilings cost `$30.2950` and the formal allowance has `$22.8249`. Dividing the PACKAGE balance instead gives 1, which is the error: the engineering allowance cannot pay for a formal probe.

*Generated by `scripts/consolidate/render_log_navigation.py` from `derive_budget.py`; do not edit by hand.*

<!-- budget:end -->

Remaining balance is not permission.

## The C1 result

The verdict and every figure behind it live in
[`runs/attempt18/evidence/c1_decision.json`](../stages/stage-1/phase_c1/runs/attempt18/evidence/c1_decision.json).
Cited, not restated from memory:

| | |
| --- | --- |
| verdict | **`GO`** |
| delta, `correct_overall` | `0.013725` |
| one-sided LCB | `0.005490` — above zero |
| two-sided CI | `[0.004314, 0.023137]` |
| SESOI | `0.010` — the point estimate clears it |
| seed robustness | 3 of 3 positive, 2 required |
| guardrails | passed, **no vetoes** |
| per-seed delta | `0.01529`, `0.00941`, `0.01647` |

**Read it with care, and read the record.** Absolute correctness is low on both
arms: of 850 scorable prompts the incumbent scores 18 / 26 / 26 and the treatment
31 / 34 / 40. A 1.37-point delta on a ~2.7% base is a large relative change over
a small absolute one. `usable_rollout` covers 555–605 prompts of 850 for the
incumbent and 582–605 for the treatment, so roughly a third of the battery
produces no usable rollout on either arm.

The decision record states its own claim boundary: *prompt-distribution
uncertainty conditional on the three preregistered fresh recovery-seed
checkpoint pairs — not a CI over hypothetical future recovery seeds.* This is
selection evidence about one operator under a fixed 0.86M-token recovery budget.
It is not a capability claim and not a statement about a trained model, and
**nothing has been added to the README Optim record**: an official record needs
the full §3.8 package and maintainer approval.

## Two engineering defects attempt 18 exposed

Neither affects the C1 result, and both are held separately from it.

**1. The relay stream copy of `c1_evidence.json` arrived corrupted — REPAIRED.**
The driver rewrites that document on every state change; the relay mirrors files
by byte offset because its other streams are append-only. It had synced 11,343
bytes of an early version, the driver replaced the file with a 23,425-byte one,
and `tail -c +11344` appended the new document's tail to the old document's
head. Exactly the right size, and not JSON.

`RelaySpec` now carries `whole_file`, and such a spec is written by temp file
plus atomic `os.replace` — the local copy becomes exactly the new bytes or is
left alone — and refuses rather than writing a document truncated at the chunk
cap. The evidence document is the one spec that declares it; the event streams
are still appended, because re-reading a growing train log every poll is what
the offset scheme exists to avoid.

Two regressions cover it, and both were confirmed by mutation: a long document
replaced by a shorter one leaves exactly the shorter one and parses; and a
`whole_file` spec ignores a *stored* offset rather than merely never writing
one — the first version of the fix passed every other test with that guard
removed, and an offsets file written by the attempt-18 relay carries `11343`
for exactly this path.

The closeout also noted that **the collector still prefers the stream copy**.
That is no longer a defect and needs no change: the preference was only
dangerous because the preferred copy could be corrupt. The runner performs a
final `sync_once` after the terminal marker, which for a whole-file spec is a
complete re-read, so the stream copy is now the finished document.

**2. The probe-durability mechanism preserved nothing — NOT repaired, and not
mine to repair.** Added before this run so completed probes survive a later
failure, it ran on all six and every upload was refused: *"Private repository
storage limit reached"*, 2.22 GiB per probe. The mechanism behaved correctly —
never raised, disturbed no stage, recorded each probe's identity, seed, config
hash, content hash and the exact reason — but the bytes are gone with the pod.
It cost this run nothing because the run succeeded; had stage H failed again,
six probes would have been lost a second time.

What it needs is a durable large-artifact backend with capacity. The private
quota is account-wide, and freeing it means permanently deleting historical LFS
objects or changing a paid plan — a maintainer decision either way. The
requirement is recorded for future long experiments in AGENTS.md P8.2.1.

**It does not block C2 Search-1.** Search-1 trains no probes and exports no
checkpoint: its candidates are measured on the pod and their identities come
home in the search journal, which is kilobytes. The capacity decision becomes a
precondition only for a later behavioural-confirmation experiment, which would
produce six 2.22 GiB probes and is separately authorized.

## The launch chain — nothing owed, because no session is pending

**C1 owes no step.** Attempt 18's chain is consumed and the round is ended by a
verdict. This section is kept because the chain is the mechanism any *future*
authorized paid session uses, not because a session is pending. The readiness
block above says the latest sweep does not describe the current tree; that is
correct and is not a debt — a `launch_bound` sweep describes the tree a launch
will use, so it is run once, when a launch is actually imminent (AGENTS.md P8.3).

One authorization funds one launcher session: up to three acquisition draws
inside it, never two billing resources, all sharing that session's single
ceiling. The ordering constraints, which cost real money to learn:

1. the run's **grant**, committed on a clean tree
2. a **`launch_bound` sweep** on that clean pre-authorization tree
3. commit **ONLY the readiness record**
4. the one-use **authorization**, issued against that clean commit. The issuer
   refuses a dirty tree by default since attempt 15 skipped step 3
5. commit **ONLY the authorization artifact**
6. the exact-session **bundle**, staged with `--run-id`
7. a live quote, every pre-provider gate, the single launch

Steps 3 and 5 are separate commits because `session_commit_and_lineage` permits
exactly one tracked path to differ between the authorized base and the session
commit. Attempt 15 combined them and was refused at `$0`.

Step 2 must follow step 1: `verify_record` permits exactly two tracked paths to
differ after a sweep — the readiness record and the issued authorization — so a
grant committed after a sweep invalidates it.

A chain is consumed by the launcher's invocation, whether or not a provider
resource followed, and is never reused. A launcher invocation that aborts before
formal training is an engineering subrun: it is closed, repaired and retried
under a fresh chain without a further approval (AGENTS.md P12.1). That exception
governs retries **before** measurement and never after a complete verdict.

Full terms — attempt counting, the six retry conditions, the stop list:
`execution_package` in
[`../configs/experiments/phase_c1/authorization.json`](../../configs/experiments/phase_c1/authorization.json).
Those terms are C1's. A C2 session would need its own grant and its own ceiling;
neither the project headroom above nor C1's unused formal allowance is
authorization for one.

## What ends a round

A complete `GO`, `NO-GO` **or** `INCONCLUSIVE` all end it. `INCONCLUSIVE` is a
result and is never re-run in pursuit of a `GO`.

Stop and report if: the first probe has started training, or whether it started
cannot be confirmed; a real replay mismatch at stage D or E; an input-identity
conflict; a limit reached; or a resource whose billing state is unknown.

## The log tree: stage-first, and every experiment attributed

`logs/` is **Stage → Experiment → Run**. Four stages have repository evidence
and therefore exist — stage-0 and stage-2 as pipeline activity with no
experiment-run logs, stage-1 and stage-3 with both.

Every historical experiment has a stage, rebuilt from repository facts and
checked rather than asserted: [`stages/index.json`](../stages/index.json) is the
stage index. It carries the evidence for each assignment and the rules that
decided it, and it also says what each stage is *for*, what it consumes and
produces, and where its canonical configs, data manifests and artifacts live —
so `logs/stages/` is the pipeline's stage-level entry point rather than a run
container. **20 experiments, 20 in one stage, 0 genuinely cross-stage, 0
unresolved.** There is no `cross-stage/` directory, because no experiment takes
several pipeline stages as its subject; a run whose stage nobody declared is
refused rather than shelved.

Five Stage-3 experiments ran and produced no log files of their own — `ttb`,
`p0_real`, `d0`, `p0`, `p2`. They are listed in
[`stages/stage-3/`](../stages/stage-3/) with their configs, artifacts and index
sections, and given no directory: an empty one would assert material that does
not exist. `e2` now has one, holding the single document it left behind.

`migrations/` and `archive/` are gone. A superseded document is deleted, since
git history holds it; a historical document that is still part of the record —
a preregistration, a consumed authorization, the pre-layout experiment
chronology — sits under the experiment or stage that owns it. The few old paths
current tooling must still resolve are a flat table in
[`index.json`](../index.json)`.historical_paths`.

One declared exception: `shared/analyses/autoinit_*` and four
`shared/validations/` directories are **Stage-1 material, not stage-neutral**.
They stay where they are because frozen pod drivers read those exact paths —
recorded in the stage index with its blocker rather than left looking ownerless.

This changed no experiment result, no authorization and no frozen evidence. The
launch chain is unaffected and still paused.

## Engineering, not results

The initialization migration and the CUDA stage-F validation are engineering
records. The stage-F device repair is **CONFIRMED ON REAL CUDA** at execution
SHA `7027a8f4`. Neither is a C1 result and neither authorizes anything:
[`maintenance/source-relocations/initialization-core/v1/`](../maintenance/source-relocations/initialization-core/v1/) ·
[`phase_c1/validations/cuda-stage-f/v1/`](../stages/stage-1/phase_c1/validations/cuda-stage-f/v1/)

## History

This file holds the current state only. The narrative lives with what it is
about, and is unedited:

* [`phase_c1/history/operational_history.md`](../stages/stage-1/phase_c1/history/operational_history.md)
  — every C1 session, its cost, failure and repair
* [`stages/stage-3/history/EXPERIMENTS.md`](../stages/stage-3/history/EXPERIMENTS.md)
  — the pre-layout experiment chronology, and the only record several Stage-3
  experiments have
* [`experiment_index.md`](experiment_index.md) — what each experiment proved,
  what it does **not** support, and which conclusions still bind
* [`phase_index.md`](phase_index.md) — the same history organized by phase
* [`decisions.md`](../budget/decisions.md) — decision records
* [`budget/ledger.md`](../budget/ledger.md) — every cost, per session
