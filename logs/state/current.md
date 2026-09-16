# Current state

**Updated:** 2026-09-16. The human view. Every number here has an owner named
beside it, and this file restates none of them from memory — a second
hand-maintained copy of a cost or a status is how two documents come to
disagree.

Start at [`README.md`](../README.md) if you do not know which document you want.

## Right now

**Nothing is running. Nothing is billing. No pod exists.** Attempt 6's pod
`v0h4f5at4112g2` was deleted after 2.27 min, the provider confirms it gone, and
an independent query returns `ZERO_PODS` with `pod(v0h4f5at4112g2) = null`.
Every C2 chain is consumed and **nothing is prepared for launch**.

**The C2 beam search RAN TO COMPLETION and the comparison it was collected for
did not run.** Attempt 4 (2026-09-16, `$6.0785`) passed 9/9 `$0` gates, setup,
and stage A; the beam ran 4 h 56 min against its 635.96-min envelope, produced
**7 complete leaves** and committed **5** to `stage1_selection.json` at
00:00:17Z with its journal, telemetry and hashes. B was absent from those
leaves, so the single conditional rebuild started — and hit its **27.66-min
allowance** inside `depth.causal_kl_greedy_v1` at round 5 candidate 26/31, 196
evaluations in.

So there is **no `c2_baseline_comparison.json` and Search-1 is INCOMPLETE**. B
has no surviving checkpoint bytes and its `state_eval` number has never been
measured anywhere, so the five selected leaves have objective values with
nothing to compare against. **This is not a null result, not a partial answer,
and not evidence that any candidate beats or loses to B.**

**Spend was never the constraint** — the session ended at 40.4% of its
`$15.0446` ceiling. The 27.665-min reserve was. The frozen C1 treatment path's
first operator runs its own greedy layer search at the observed 7.05
evaluations/min, and this repository's Phase-B measurements put a DEPTH
expansion at roughly 32 min on an L40S: the reserve could not fund the path's
first operator, let alone its other three. Nothing crashed; the deadline
mechanism stopped the work instead of running to the cost backstop, which is
its purpose.

**THE BASELINE COMPLETION WAS AUTHORIZED, RAN TWICE, AND B IS STILL NOT
MEASURED.** The maintainer decision of 2026-09-17 closed the science for
implementation purposes and authorized exactly two formal sessions to establish
B, measure it once on the same frozen suite, and compute the preregistered B→C
comparison against the five frozen candidate measurements. **Both are consumed.
Neither reached a rebuild.**

| attempt | where it stopped | cost |
| --- | --- | --- |
| [5](../stages/stage-1/phase_c2_baseline_completion/runs/attempt5/closeout/outcome.json) | the launcher's **first statement**: `claim_output_root` took the stage id positionally where the signature takes `outputs` by keyword. No gate ran, no price was queried, **no provider resource existed**. The chain was consumed anyway — its one-use rule counts the invocation | `$0.0000` |
| [6](../stages/stage-1/phase_c2_baseline_completion/runs/attempt6/closeout/outcome.json) | **10/10 `$0` gates passed** and setup refused at **`ROPE_OK`**, which globs `artifacts/stage1/*/checkpoint/config.json`. This session stages no checkpoint — it rebuilds B on the pod — so the step had nothing to look at. `SETUP_RC=1`, no driver stage | `$0.0412` |

Both are the same class, and it is the one this repository keeps paying for: **an
inherited declaration rather than an inherited need.** C1 attempt 2 died on that
same `ROPE_OK` line for `$0.1013`.

The repair does not drop the marker's job. `transformers` 4.x reads the flat
`rope_theta` where 5.x records the nested one and the two disagree by 500×, so a
loader taking the wrong field would give B a different positional basis and a
silently wrong `state_eval` — the one number the session exists to produce. The
guard **moved to where the artifact exists**: onto the rebuilt B, in the
interpreter that measures it, after materialization and before the measurement,
through the same two helpers the setup step uses. Its reading is carried into the
durable measurement block.

The six identities the decision froze by value are **unchanged**, derived through
the production assembler and now asserted by a regression — the completion
closure moved, as that decision said it would, so a moved closure is no longer
the only visible difference between a permitted repair and a change to closed
science:

| artifact | identity |
| --- | --- |
| frozen candidate side of B→C | [`c2_frozen_comparison_inputs.json`](../stages/stage-1/phase_c2/runs/attempt4/evidence/c2_frozen_comparison_inputs.json) · `55f6677067392fd0…` |
| completion protocol | [`phase_c2_baseline_completion_protocol.json`](../stages/stage-1/phase_c2/plans/phase_c2_baseline_completion_protocol.json) · `9f566eb6f8d71d57…` |
| completion pricing | [`phase_c2_baseline_completion_pricing.json`](../stages/stage-1/phase_c2/plans/phase_c2_baseline_completion_pricing.json) · `dcc64bcf9b3dc9fb…` |
| the reserve defect | [`c2_baseline_reserve_defect.json`](../stages/stage-1/phase_c2/analyses/c2_baseline_reserve_defect.json) · `f389350cca7782ca…` |
| completion executable closure | [`c2_baseline_completion_closure.json`](../stages/stage-1/phase_c2/analyses/c2_baseline_completion_closure.json) — **derived live and expected to move; the digest is pinned there, not restated here** |

The **driver and a thin formal launcher** were repaired against seven execution
defects that would each have surfaced only after B had been rebuilt and measured
— the suite root, an unprimed evaluator, a second teacher, a caller-supplied
search identity, the wrong ranking citation, a nested verdict key, and a record
mutated after its own hash — then against six authorized enforcement repairs,
then against the two failures above. An eighth, a missing `SESSION_KIND` branch
in the shared setup script, would have exited 98 on a billing pod. Every one is
held by a mutation-verified regression, and the chain machinery has now been
executed end to end at `$0`: entry path, all ten gates, closeout.

The authorization is a **distinct type** reporting `authorizes_c2_search1 =
False`, and the two loaders refuse each other's schemas, so a completion grant
cannot buy a beam. The derived closure contains no Search-1 module.

The completion is priced at **`$0.7212` expected and a `$1.1950` hard ceiling**
(39.70 / 65.78 min at `$1.09/h`) — derived from attempt 4's own telemetry, not
from the reserve that failed. **No budget increase is requested**, the rate
boundary is unchanged, and the old Search-1 pricing document is preserved
exactly as the authorization basis attempts 2–4 ran under.

**A THIRD SESSION NEEDS A MAINTAINER DECISION.** The 2026-09-17 message
authorized two and made attempt 6 the last automatically authorized one, so its
own failure rule ends here: teardown, preserve, reconcile, diagnose, repair,
regress, **report** — not another chain. `$2.3488` of the `$2.3900` retry
sub-envelope is unspent and that is not permission.

**One open question belongs to the maintainer.** Cross-session numerical
comparability is preserved structurally — every checkpoint is scored
independently against the original teacher under `RECOMPUTE`, with no
candidate normalized against another — but its *magnitude* is unmeasured: no
state was ever measured twice anywhere in this project, the per-measurement
`runtime` block is empty, and the image name does not pin the host driver
(attempt 3 saw `595.91.07`, attempt 4 `580.126.09`). The five candidates are
separated by 78–1273 epsilons, so the front structure is not balanced at that
scale; the protocol therefore pre-registers a disclosure rule, before B exists,
for any B→C margin at or below the tightest observed gap of `0.007782`.

Attempts 2 and 3 aborted **before** the beam for `$0.0552` and `$0.1674` and
measured nothing; both root causes are repaired and both repairs were confirmed
on real hardware by attempt 4 — setup and stage A passed, and the artifact
collection that raised `ArtifactError` in attempt 3 returned `manifest rc=0`
with all 7 files and `missing: []`.

**C1 is COMPLETE.** Attempt 18 executed the whole frozen protocol — both replay
gates, both arms, six probes trained, six evaluated on the frozen battery — and
the frozen Stage-I rule returned **`GO`** at `$10.2018`, under its own planning
floor. A complete valid verdict ends the round.

| | | owner |
| --- | --- | --- |
| phase | C1 — fixed-path ATTENTION isolation, **CLOSED by a verdict**, and its execution preregistration is now **frozen to the binding attempt 18 ran under**. C2 Search-1 is **INCOMPLETE**: its beam ran to completion and the B→C comparison it was collected for has never been computed, because the baseline completion authorized to supply B was consumed by two pre-measurement aborts. The completion package itself is **executable, priced, repaired and NOT AUTHORIZED** — protocol, pricing, frozen candidate side, driver, thin launcher, ten `$0` gates, a CPU preflight of its own, a derived closure, a readiness contract, an authorization type and issuer, a bundle transport, run ownership and an enforceable resource scope all exist and have all now executed at `$0` | [`phase_c2/plans/phase_c2_search1_plan.md`](../stages/stage-1/phase_c2/plans/phase_c2_search1_plan.md) · [`phase_c1/plans/phase_c_roadmap.md`](../stages/stage-1/phase_c1/plans/phase_c_roadmap.md) |
| replay | **MEASURED — 2/2 PASS**, for the third time (attempts 9, 17, 18). Passing replay is not a result: 9 and 17 are **NO DECISION**, pre-treatment aborts that measured no endpoint. Attempt 18 is the only attempt that decided anything | [`attempt18/closeout/outcome.json`](../stages/stage-1/phase_c1/runs/attempt18/closeout/outcome.json) |
| treatment, endpoint | **MEASURED** — six probes trained and six evaluated on the frozen battery; the frozen Stage-I rule returned **`GO`**. Figures in the block below | [`attempt18/evidence/c1_decision.json`](../stages/stage-1/phase_c1/runs/attempt18/evidence/c1_decision.json) |
| launch chain | every C2 chain is **consumed** — Search-1 attempts 1–4, baseline completion attempts 5 and 6 — and **nothing is prepared**. No further C1 attempt is authorized or prepared either; a complete verdict ended that round | [`phase_c2_baseline_completion/runs/attempt6/governance/`](../stages/stage-1/phase_c2_baseline_completion/runs/attempt6/governance/) |
| last attempt | **baseline completion attempt 6 — ABORTED AT SETUP, `$0.0412`.** Ten `$0` gates passed, the pod billed 2.27 min, setup refused at `ROPE_OK`, and no rebuild or measurement happened. Before it, attempt 5 consumed its chain at `$0.0000` without creating a resource. The last complete scientific execution remains **C1 attempt 18** (`$10.2018`, `ALL_DONE`, verdict `GO`) | [`attempt6/closeout/outcome.json`](../stages/stage-1/phase_c2_baseline_completion/runs/attempt6/closeout/outcome.json) · [`phase_c1/runs/attempt18/closeout/outcome.json`](../stages/stage-1/phase_c1/runs/attempt18/closeout/outcome.json) |
| blocker | **A MAINTAINER DECISION IS REQUIRED.** Both sessions the 2026-09-17 message authorized are consumed and **B still carries no measurement**, so the B→C comparison remains uncomputed and Search-1 stays incomplete. Both root causes are repaired and regressed at `$0`; `$2.3488` of the retry sub-envelope is unspent and is not permission. Durable large-artifact capacity remains a separate open decision | [`attempt6/closeout/outcome.json`](../stages/stage-1/phase_c2_baseline_completion/runs/attempt6/closeout/outcome.json) · [`budget/decisions.md`](../budget/decisions.md) |
| spend | owned by the budget block below | [`budget/ledger.md`](../budget/ledger.md) |

## Readiness

<!-- readiness:begin -->

| readiness | | owner |
| --- | --- | --- |
| latest POINTED-TO sweep — C1 attempt18 | **launch_bound — PASS**, swept at `e80eb60b`; **does not describe the current tree** | [`c1_pod_environment_verification.json`](../stages/stage-1/phase_c1/analyses/c1_pod_environment_verification.json) |
| every other experiment's readiness | **run-owned and not pointed at from here** — one record per attempt, under that attempt's `governance/readiness.json`, so a later sweep cannot overwrite what an earlier one launched under | [`stages/stage-1/`](../stages/stage-1/) |
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
| project cap | `$296.8597` spent of `$320.0000`, leaving `$23.1403` |

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
