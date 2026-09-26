# Current state

**Updated:** 2026-09-23. The human view. Every number here has an owner named
beside it, and this file restates none of them from memory — a second
hand-maintained copy of a cost or a status is how two documents come to
disagree.

Start at [`README.md`](../README.md) if you do not know which document you want.

## Right now

**Nothing is running and nothing is billing.** Zero pods and zero network
volumes; both `59qt99zeg5` and `a0zqgxsm7p` are deleted.

**The C3 batch-invariance root-cause investigation is MEASURED and its finding
is derived**, on branch `review/c3-operator-batching`, which is **not merged
into `main`**. In one line: a batched forward is not the same computation as a
solo one because a GEMM whose reduction is deep relative to its output width
reduces in a shape-dependent order, and in bf16 that moves an FFN top-k in most
layers. See
[C3 — the refactor is on review](#c3--the-refactor-is-on-review-and-c3-has-not-started)
below. **Nothing was changed in response** — no default, no identity semantics,
no operator definition. **C3 itself remains NOT STARTED and its `$25.00` stage
envelope is untouched.**

## Stage ladder

```text
C0  COMPLETE
C1  COMPLETE / GO
C2  CLOSED WITHOUT PROMOTION            <- maintainer decision, 2026-09-24
      full joint search                 completed
      screening                         completed
      behavioural confirmation          executed
      historical observations           pointed NO_GO (attempt13, attempt14)
      final confirmation evidence       MIXED evaluation-protocol identities
      canonical C2 promotion verdict    NOT claimed
      new incumbent named by C2         NONE
      accepted incumbent after C2       B = frozen C1 treatment
C3  NOT STARTED                         <- next scientific stage
```

**B stands because no valid C2 challenger displaced it** — not because a clean
canonical NO_GO experiment ruled against the candidate. The search and
screening did produce a candidate; the behavioural confirmation did not produce
a sufficiently clean, protocol-consistent result to replace the incumbent.

**C2 is not described here as a canonical NO_GO**, because its decision archive
does not pass the protocol-consistency gate added in the same repair. The six
confirmation probes share one battery, one scoring contract and one metric
contract, and carry **three distinct generation protocol fingerprints** — with
the third seed's pair spanning two of them, so the confound sits inside the
pair. Owner:
[`c2_behavioural_verdict_20260923.md`](../stages/stage-1/phase_c2_behavioural/analyses/c2_behavioural_verdict_20260923.md).

**Both historical observations stand, unaveraged.** attempt13 measured
`delta −0.008235…`; attempt14's reconstruction measured `delta −0.009019…`; the
difference is two prompts of 850 on probe 11. Neither is selected as more
correct. Protocol/runtime drift is a live alternative explanation and cannot be
separated from any candidate-versus-B effect at that seed using this evidence.
The earlier attribution to greedy-decoding nondeterminism was asserted, not
established, and has been withdrawn.

**Further C2 scientific spend is not authorized.** No re-evaluation of the six
checkpoints, no uniform six-probe replay, no new seeds, no attempt15.

**Artifacts follow consumers, not campaigns** — AGENTS.md **P8.4**, adopted
2026-09-23 as a standing rule for C2, C3, C4, Stage 2/3 and every later stage.
Never move a large artifact onto an execution resource because it exists or
belongs to the same campaign; ask what exact downstream operation will read the
bytes, and if there is none, do not move them. A completed and validly scored
probe contributes evidence; its checkpoint is archival. For this campaign the
working set is **8.1 MiB against 22.21 GiB — 0.035% of the bytes**. Owner:
[`consumer_derived_working_set_20260923.md`](../stages/stage-1/phase_c2_behavioural/analyses/consumer_derived_working_set_20260923.md).

**The project cumulative was under-reporting by `$1.9299`.** `project_balance`
summed run closeouts alone, and an engineering campaign is not a run — so four
CUDA validations and the durable staging reached their packages' books and
nothing else. The repair made the campaign term a reported, derived component
of the project balance, so every provider dollar counts against the cap
whichever book authorized it. The total itself is **not restated here** — it is
derived on every render; read the `project cap` row of the summary table.
Owner: `derive_budget.py :: project_balance`.

**C2's execute-to-completion authorization is SPENT and CLOSED.** It was
granted on 2026-09-23 — a `$25.00` all-in stage envelope for all remaining C2
work, managed internally rather than per category, under a `$42.0000`
cumulative behavioural campaign ceiling. That window is over: the maintainer
closed C2 without promotion on 2026-09-24 and **no further C2 scientific spend
is authorized**. Remaining allowance under either figure is not permission and
must not be spent. The `$370` project envelope never moved.

**C2 owes no probes.** All twelve were trained and scored — six screening, six
confirmation. What the stage did *not* produce is a promotion verdict: the six
confirmation probes do not form one uniform evaluation-protocol field, so no
canonical NO_GO is claimed and no new incumbent is named. **B, the frozen C1
treatment, remains the accepted incumbent by absence of a valid challenger.**
Owner:
[`c2_behavioural_verdict_20260923.md`](../stages/stage-1/phase_c2_behavioural/analyses/c2_behavioural_verdict_20260923.md).

**behavioural attempt8 was retired at `$0` without acquiring a machine.** All
**ten** pre-provider gates passed — the first chain to clear every one — and
then eight consecutive create calls over 35 minutes were refused with "no
longer any instances available with the requested specifications". The session
had pinned itself to **EU-NL-1** in order to attach the network volume, and
`volume_gate` had just reported that no probe needs its weights on the pod. It
narrowed its own hardware supply to one datacenter to attach a volume nothing
was going to read. No pod id was ever returned, so nothing billed. The
repair derives the attachment from need and clears volume, mount and
datacenter together when no remaining operation reads a pre-staged
checkpoint — P8.4 applied to the acquisition constraint, not just to the
bytes. Owner:
[`attempt8/closeout/outcome.json`](../stages/stage-1/phase_c2_behavioural/runs/attempt8/closeout/outcome.json).

**The durable backend is a network volume, not an object store.** The
object-store route was built, tested and never run: every store reachable from
this environment needs an account creation and payment step only a maintainer
can complete, and the Hugging Face account that exists has **2.62 GiB** of
private headroom — bisected at `$0` through the LFS batch endpoint — against
22.21 GiB needed. A volume needs no credential anywhere, and attaching it
removes the transfer from the billed session entirely. **It is then not
attached at all**, because P8.4 left nothing for it to carry: the reserve is
**10 minutes** for re-reading 8.1 MiB of evidence, and a session that attaches
the volume anyway buys a one-datacenter draw for no consumer — which is how
attempt8 failed. The volume is a restore *source* only; a newly trained probe
is preserved the other way, by `_fetch_and_verify` pulling it to the launcher
host and re-identifying it there. `runtime/durable_store.py` and
`experiments/durable_stores.py` are kept, annotated as having no production
caller. Owner:
[`durable_backend_decision_20260923.md`](../stages/stage-1/phase_c2_behavioural/analyses/durable_backend_decision_20260923.md).

**A session may no longer be authorized to spend past what its campaign has
left.** The continuation gate charged the campaign for the work a session
PLANS; nothing charged it for what that session could cost if the work went
wrong, because `all_in_hard_usd` came from the frozen full-session
decomposition and was the same figure for the first attempt and the fifth. With
`$2.5425` settled that was safe by one cent. With `$22.2466` settled a full
session would have been authorized to reach `$55.4565` against a `$42.0000`
campaign ceiling, and every gate it passed would have said yes. The ceiling is
now shortened to the campaign's remaining money — `$19.7533` over 1070.96
minutes — with the runtime shortened alongside it and the limit floored rather
than rounded. The work owes 491.51 minutes, so nothing scientific is shortened.
Owner: `behavioural_governance.session_ceiling_under_campaign`.

**Settled campaign spend has one owner now.** It was derived in the pricer and
again inside the launcher's gate, and the issuer derives a session ceiling from
it — so two readers disagreeing would cap a session against one figure and
charge it against another. `behavioural_governance.settled_campaign_all_in`
reads the closeouts; the gate keeps its stricter per-resource R9 reconciliation
and refuses when the two disagree in the dangerous direction.

**attempt5 measured ten of twelve probes and then ran out of disk.** The
trainer hit `No space left on device` writing probe 11; all ten completed
probes are trained, scored and durable off-pod with their per-prompt rows.
**There is no verdict**, and that is R5 working: the estimand is a paired
difference over three seeds and four confirmation probes are not that quantity.
`$19.7041`. The remaining two probes were measured by attempt12 (which trained
one and preserved it unscored) and attempt13 (which restored it, scored it
without retraining, trained B, and returned the verdict), and attempt14
reproduced probe 11 for `$2.0670`. The authoritative totals are derived, not
restated here — see the `project cap` row of the summary table and
`budget` in `current.json`, both written by `derive_budget.py`. This sentence
used to restate them anyway, and went stale the moment the C3 engineering
campaign booked `$0.0822`.

**GO was arithmetically excluded before the third seed ran** — both completed
confirmation seeds put the candidate behind B (−0.0036, −0.0035) and the frozen
rule needs 2 of 3 positive. The third seed, measured by attempt12 and attempt13,
did not change the direction, and the rule returned **`NO_GO`** with
`delta −0.008235` and `lcb −0.016863`. B scored `0.0412` on the confirmation battery against C1's
treatment pooling `105/2550 = 0.0412` on the same battery, so B reproduces its
own lineage and the `0.0247` screening figure was a disjoint, harder prompt
set rather than a regression.

**The root cause was a storage derivation that contradicted its own config,
and I had found it fourteen hours earlier and bounded the wrong quantity.**
`storage_requirement` charged a trained probe at the size of the bf16 leaf it
started from while the recipe declares `dtype: float32`. I recorded that,
then computed headroom from the 2.22 GiB durable artifact size instead of the
~5.6 GiB a probe actually occupies locally, and called it comfortable.

Repaired, and the repairs are the point rather than the number:

* the byte model is generic and lives in `aadistill.runtime.cost` — every
  dtype is an argument, an unknown dtype raises instead of defaulting, and no
  parameter count, model family or experiment name appears in it;
* **container residency and durable capacity are two resources.**
  `destination_gate` charges the derived durable requirement (26.654 GiB, was
  a hardcoded 13.3) and a new `container_gate` charges peak local residency
  (72.174 GiB) against the provisioned disk. The derived provision fell from a
  double-counted 140 GB to 100 GB; the authorization stays at 120;
* **the probe-local lifecycle has a real acknowledgement boundary.** The
  launcher writes a release ack only after a probe's bytes arrived off-pod AND
  re-identified there; the driver releases acked probes before training the
  next one and refuses to continue if a release fails. `announce_durable`
  could never have authorized this — it runs before the transfer;
* **and the driver measures.** `require_probe_headroom` reads the filesystem
  before each probe and refuses if it cannot hold the next one, so a wrong
  derivation costs a clean stop with every finished probe durable.

**Nothing is prepared for launch.** The next chain is **attempt9** and it has
no grant, readiness record, authorization or bundle; attempts 1–8 hold theirs
as consumed evidence. Four of those were consumed at `$0`: attempt4 by a dry
run that recorded a run (which is why that flag now writes to a separate run
id), attempt6 and attempt7 by two further defects in the dry-run mechanism
itself, and attempt8 by a provider that had no L40S in the one datacenter the
session had pinned itself to. Each was found for free, which is what the
rehearsal is for — but a launcher repair moves the executable closure, so each
costs a fresh chain. Owner: [`current.json`](current.json) `:: prepared_launch`.

**attempt3 FAILED in stage P and there is no verdict.** Not `NO_GO`, not
`INCONCLUSIVE` — those are complete results of a run that measured something.
This one stopped in initialization, having trained no probe. `$2.5425` all-in.

**All five frozen Top-5 candidates rebuilt to their exact recorded digests** on
fresh hardware, each identity-gated before it was announced — a real second
reproduction of the replay, and it survives the failure. `1d284448` among them,
the leaf that failed at step 0 in replay attempt 8: the evidence-bound root pin
holds on new hardware.

The sixth arm, incumbent B, completed all seven greedy rounds of
`depth.causal_kl_greedy_v1` in 32.8 min and then died at `Writing model shards`
with **`No space left on device`**. No digest mismatched. The cause:
`materialize_fixed_path` writes every step of a four-step path and nothing
deleted the intermediates, so all six arms' full paths stayed resident — while
the storage derivation charged that transient exactly ONCE, "released when it
is verified". Nothing released it. Repaired: intermediates are now freed after
the identity gate and after the durable announcement. 120 GB remains correct;
peak residency with the repair is ~59 GiB.

**The campaign ceiling has been raised twice.**
attempt3 spent `$2.5425` and produced no probe, so a fresh full attempt no
longer fitted under the original `$33.2099`. The maintainer raised the
**cumulative campaign** ceiling to **`$35.7600`** (+`$2.5501`) on 2026-09-20,
and to **`$42.0000`** on 2026-09-23 alongside the execute-to-completion
authorization. No science changed: the plan hash has never moved and is
still `31088b98…`.

**The two ceilings are now separate numbers.** They were one figure doing two
jobs, which is why a gate reading either passed every test:

```text
session (fresh)    1800.53 min · GPU 32.7097 · disk 0.5002 · all-in 33.2099
session (continued) 1070.96 min · GPU 19.4558 · disk 0.2975 · all-in 19.7533
campaign                                                    all-in 42.0000
```

The continued session line is the fresh derivation SHORTENED to what the
campaign has left after `$22.2466` settled, and it is re-derived per attempt —
unchanged across attempts 6 to 9 because no attempt since has settled a cent. The campaign figure is the
maintainer's; the two session lines are derived.

`all_in_hard_usd` bounds ONE attempt and is what the window, the watchdog and
every in-pod spend check are built from. `campaign_all_in_hard_usd` bounds the
campaign cumulatively and is the only figure prior spend is charged against. The
larger campaign ceiling buys another attempt and nothing else — no runtime, no
disk, no probes, no seeds, no scientific scope. Owners:
`behavioural_governance.CAMPAIGN_ALL_IN_CEILING_USD` and
`authorization_terms`; the separation is asserted by driving the two apart in
`tests/c2_behavioural_preflight/test_behavioural_continuation.py`.

**Cleanup failure now fails closed at the caller.** `release_intermediates`
stays non-raising — a cleanup error must not destroy a verified, announced arm
— but if anything failed to delete, stage P stops there and no next arm is
built. The 120 GB provision is derived on the assumption that a verified arm's
intermediates are freed; a failed release falsifies it, and attempt3 is what
discovering that five arms later costs.

**The project cumulative is `$331.4509`** of the `$370.0000` cap, leaving
`$38.5491`. It was corrected from `$309.2043` to `$311.7468` before attempt5
and attempt5's `$19.7041` took it to `$331.4509`. The `$2.5425` was invisible for two independent reasons, both
closed: no behavioural run was in `logs/index.json`, and the extractor read only
`budget.this_attempt` and `cost.actual_usd` while the behavioural closeout
states `money.all_in_usd` — the only one of the three that is all-in. An
affirmative `provider_resource_created: false` is now read as a stated `$0.0000`;
a genuinely unknown cost still stays UNKNOWN, leaving **`$58.2532`**. Owner:
[`budget/ledger.md`](../budget/ledger.md), derived by `derive_budget.py`.

**attempt3's grant carries a wrong date.** `granted_utc = 2026-09-21` while the
session ran on 2026-09-20 UTC — a local-timezone date in a UTC field. The grant
is consumed evidence and is not rewritten; the anomaly is recorded in
[`attempt3/closeout/README.md`](../stages/stage-1/phase_c2_behavioural/runs/attempt3/closeout/README.md)
and it distorts no money. Issuance now refuses a grant dated after the current
UTC date.

Earlier: the REPLAY's attempt 3 pod
`ulit767od813i8` was deleted behind its teardown gate after 386.2 min; the
provider confirms it is gone and an account-wide list returns `[]`.

**The C2 behavioural launch is AUTHORIZED and the chain is being built.** An
independent final review returned **GO** on `197088e` and the maintainer
granted the spend: campaign `c2-behavioural-12probe-v1`, a
**cumulative** all-in ceiling of `$33.2099` — **since raised to `$35.7600`
and then to `$42.0000`**, see above — across every run attempt and
provider resource, at a quoted L40S securePrice of `$1.09/h`. (The grant's
`granted_utc = 2026-09-21` is the timezone anomaly noted above; the real date
was 2026-09-20 UTC.) The live
securePrice was re-queried at issuance and is `$1.09/h` — the reviewed basis
unchanged, so no dollar authorization was materially altered and no return to
the maintainer was owed. Owner:
[`attempt1/governance/grant.json`](../stages/stage-1/phase_c2_behavioural/runs/attempt1/governance/grant.json).

Chain order, and it is binding: grant → launch-bound readiness → commit only
the record → authorization → commit only the artifact → exact-session bundle →
final live quote and all pre-provider gates → provider resource → formal
execution → evidence, closeout, provider-confirmed teardown.

**attempts 1 and 2 are RETIRED at `$0`, before any provider contact;
attempt3 is the live chain.** Two launcher repairs, each found by running the
eight pre-provider gates before creating a resource rather than discovering
them on a meter, each moving the executable closure and so each forcing a fresh
chain. Neither spent anything, so the cumulative `$33.2099` ceiling was entirely
intact when attempt3 launched. Closure `f08d2aaf…` → `d99ef79c…` → `ec2c894c…`;
**the plan hash never moved**, so no science changed. Both amendments are
recorded in attempt3's grant. Running the eight pre-provider gates
before creating a resource — rather than discovering them on a meter — found
that `readiness_gate` compared `record.get("kind")` when no readiness record
this repository writes carries that field. It is `record_kind`. The gate could
not have passed for any record, and it never checked the sweep's verdict
either, so a FAILED sweep would have satisfied it. Both halves are repaired and
regression-tested. The repair is inside the executable closure, so the closure
moved `f08d2aaf…` → `d99ef79c…` and attempt1's authorization — which binds the
old one — is superseded rather than edited. Nothing billed; the whole
`$33.2099` campaign ceiling is intact. The amendment is recorded in attempt2's
grant with the before/after closure and the reason. **The plan hash is
byte-identical: no science moved.**

**All five Top-5 checkpoints are reconstructed, exact, and durable off-pod.**
Attempt 9 (pod `i0uku41wc6ph3e`, 113.8 min, `$2.07`) reproduced every one of the
20 intermediate artifact digests and every leaf identity attempt 3 recorded.
Verified again independently after teardown, against the frozen selection rather
than the run's own claims: **5/5 exact**, each 1,192,135,096 bytes, at
`/home/ecs-user/aad-artifacts/phase_c2_full_search/attempt3_replay/`.

| leaf | first operator | time | result |
| --- | --- | --- | --- |
| `d005dfb2` | DEPTH causal-KL | 22.2 min | exact |
| `7da1e4e2` | ATTENTION | 32.1 min | exact |
| `1d284448` | FFN | 24.8 min | exact — **failed at step 0 in attempt 8** |
| `88086555` | DEPTH positional | 2.5 min | exact — never attempted before |
| `1a2b5b03` | WIDTH | 20.6 min | exact — never attempted before |

The repair was the **evidence-bound root pin**. Path 3 failed in attempt 8 with
expected `449c71cf…` and actual `5c479cd3…`; under the derived root state
`use_cache=False` it produced `449c71cf…` exactly. The two paths the `$0`
forensic predicted would fail at step 0 both reproduced completely. One bit of
unpinned mutable state was the entire divergence, and `artifact_digest` was
never weakened to find that out.

Each leaf was fetched, re-identified from the delivered bytes and given a
durable ACK **while the next path computed** — the first at 11:32, all five
before teardown. The closeout reported *"5 leaf/leaves already secured during
the run"* and transferred nothing again. Attempt 8's loss cannot recur on this
path.

One defect remains recorded rather than hidden: attempt 9's session record says
`INCOMPLETE` because the runner reads the driver's markers from the **status
file** and this driver wrote them only to stdout, so `C2_REPLAY_ALL_DONE` was
never seen and the session was classified by exit code. The driver exited 0. The
label is wrong; the result is not, and the driver now appends its markers to the
file the launcher tails.

**The C2 behavioural selection is FULLY IMPLEMENTED and PROPOSED, not
authorized.** Twelve probes exactly — six screening over the five reconstructed
candidates plus incumbent B on one preregistered seed, then six confirmation on
the one advanced candidate plus B over three paired seeds. Only confirmation may
name an incumbent; `NO_GO` and `INCONCLUSIVE` are results.

**The six arms are BUILT on the pod, not shipped to it.** This changed after
measuring, not after guessing. Each arm is a 1.19 GB checkpoint and neither
transport can carry six: `local_assets` are scp'd *after* the pod exists, with a
hardcoded 600 s per-asset timeout against a dev-box uplink needing ~1650 s for
one of them — the arithmetic that killed recovery-continuation attempt 2 at
exactly this size — and the hub relay, which repaired that attempt, refuses
5.95 GB for private-storage quota. The LFS batch endpoint was asked directly at
`$0` on 2026-09-19 and again on 2026-09-20: one 1.19 GB object ACCEPTED, 5.95 GB
REFUSED. So every arm is materialized from the teacher along a path pinned at
every step to the digest the frozen record holds — the mechanism the replay just
proved by reproducing all five byte-for-byte — and gated on its exact identity
before any probe starts. It is not a search: no beam, no expansion, no ranking.
*A maintainer freeing or buying HF storage would remove these minutes; that is a
maintainer decision, never an autonomous repair.*

B is built the same way for a different reason: baseline-completion attempt 8
preserved its evidence and not its bytes. Its construction is bound from C1's
own constructor, spec hash `3a233a9017b3…` matching what C1's preregistration
froze. Preparing the six arms is initialization, bounded at **155.83 min**
(125.68 for the candidates + 30.15 for B) from attempt 3's telemetry bounded per
operator *implementation*. The protocol stays at twelve probes.

Derived, not inherited: storage **120 GB**, converted GiB→GB through the
repository's recorded basis. At a live `$1.09/h` L40S quote the ceiling is
**`$33.2099` all-in** (`$32.7097` GPU + `$0.5002` disk) with `$23.8832`
expected — up from `$30.8918` because the arms are now built rather than
staged. Project headroom if the shortened session ceiling were spent in full:
**`$18.7958`**,
derived by `write_c2_behavioural_proposal.py` from the `$311.7468` cumulative
**as it stood when the proposal was written**. The live project figure is
`$333.3808` (above); this paragraph records the proposal's own derivation and
is not a second opinion about today's balance.

What is built: the launch governance and its one-use authorization type, the
launcher with eight `$0` prechecks, a **standalone** driver (it does *not*
subclass `C1Driver`, which would inherit C1's authorization, plan identity,
seeds and audit roots), the C2 decision module, a screening scorer pinned to its
own battery identity record, per-poll off-pod durability with destination
re-identification against all six identity fields, and a registered resume
policy. One `$0` production-path rehearsal drives the real driver P→D and
reaches **all three terminal states** from separate deterministic fixtures, and
a second rehearsal completes the same campaign from a *replacement* run attempt
without retraining a probe.

### The final launch review's five corrections are implemented

A `$0` independent review of `147b2c6` accepted the behavioural science and
refused the launch for four execution/governance blockers and one
authorization-scope error. All five are closed, and none needed a GPU.

**One budget model.** The launcher built a second budget on top of the
proposal's final one: it subtracted the materialization term out of the already
final 1800.53-minute window, fed the remainder into a fresh `BudgetSpec` beside
its own setup/transfer/materialize phases, and applied the frozen probe model's
10% contingency and artifact-recovery reserve a *second* time. `plan_session`
answered **2036.62 hard minutes, ≈`$36.9987`** of GPU — larger than the whole
proposed `$33.2099` all-in ceiling, so a correct authorization would have
refused the launch at the gate for reserves nobody granted twice. There is now
ONE canonical decomposition, `behavioural.session_decomposition`, which both the
proposal and the launcher's `BudgetSpec` consume; it reconciles against the
frozen pricing record's own expected and hard figures and refuses if either has
moved. The ceiling was **not** raised to pay for the duplication: expected
`$23.8832` and hard `$33.2099` are unchanged.

**The closure names what the executable reads.** The prepare stage calls
`build_replay_leaves`, whose output is decided by attempt 3's frozen selection,
compact state journal, telemetry and architecture-spec lineage — which is to say
by *which six checkpoints get built* — and none of the four was declared. They
are now named through `replay_specs`' own constants. The closure went 128 → 133
files: those four plus `selection_pricing.py`, which the budget decomposition
reads. All five are git-tracked and travel in the bundle.

**Campaign identity ≠ run identity.** `--campaign` was `ctx.args.run_id`, so a
replacement resource became a new *campaign* and R1 forced it to refuse every
probe its predecessor had trained and verified off-pod — the rule against
cross-experiment pooling was preventing continuation of one experiment. There is
now a stable `BG.CAMPAIGN_ID`, carried by the authorization and in the plan
hash; the run attempt stays unique per invocation and resource. The durable
store is keyed campaign-then-attempt, restored probes are checked against the
descriptor their rung derives (not only against their own record), screening
**commits once** per campaign, and a `$0` `campaign_continuation_gate` requires
every prior resource to be provider-confirmed non-billing and bounds cumulative
campaign spend.

### Continuation is now REACHABLE, not merely expressible

The second review accepted A/B/D/E and refused C on production grounds: the
campaign id made the policy expressible while three things kept it unreachable.
All three are closed.

**A replacement pod could not get the probes.** `load_campaign_journal` reads
`audit/probes/*.json` and each entry's `model_dir` — both of which die with the
producing pod. The old continuation test handed attempt 2 attempt 1's own
`--audit-dir`/`--eval-dir`/`--b-workdir`, so it proved only that a second
*process* can read a first process's files. There is now a real handoff:
`behavioural_continuation.py` reads the campaign's verified state from the
durable destination, and the launcher's `materialize_inputs` step — after setup,
before the driver starts, pod torn down on failure — pushes each eligible
probe's bytes and science evidence to a **new** pod path with a manifest naming
the identity to reproduce there. Verified three times: at the destination when
the probe landed, on the host before it is sent, and **on the replacement pod
from the bytes that arrive**. Only probes whose `durable_ack.json` records a
matched destination re-identification are eligible. *The bytes are not
ceremony:* a probe trained but not validly scored resumes at scoring, and
scoring reads the weights.

A gap inside that: the probe records, scores and per-sample rows were collected
only by the **success** artifact spec, so on the one path where continuation is
needed — a failed session — the science evidence never came home and the
campaign's screening commitment did not either. Both are now secured beside the
bytes during the run, on every poll, and again at closeout.

**A paid attempt that finished no probe was invisible.** `campaign_attempts`
derived predecessors from durable probes alone, so the most ordinary failure
there is — a pod that bills and dies in setup or arm materialization — returned
`[]`, and the next attempt called itself the campaign's *first resource*,
summing no spend and checking no release. It is now the union of the campaign's
**run records** and the durable store: the run record is the authority for *a
resource existed*, the store only for *it left science*. A run naming another
campaign is excluded; a run whose campaign cannot be read is included and
refuses as UNKNOWN.

**A continuation reserved a whole fresh session.** `planned = gpu_hard + disk_hard`
meant `settled + 33.2099 > 33.2099` for any prior spend above `$0`, so no
continuation could ever pass. `session_decomposition` now prices **remaining
work** — and a fresh campaign's remaining work is all of it, so the full session
and every continuation come from one derivation, with the authorized `$23.8832`
/ `$33.2099` unchanged at the defaults. A completed probe is never priced again;
the arms its *remaining* probes need are, because a replacement filesystem must
rebuild them; the restore is a phase, bounded at the slowest recorded uplink
(0.23 MB/s, ~80 min per 1.11 GiB probe) because it is billed pod time. A
complete campaign owes nothing and the gate refuses to continue it at all —
GO, NO_GO and INCONCLUSIVE are terminal.

*Still fail-closed where it matters:* if the remainder does not fit the approved
campaign ceiling the gate refuses and returns to the maintainer. The experiment
is never shortened to fit and the ceiling is never raised. **A maintainer
freeing Hugging Face private storage would move the restore to the `$0` pre-pod
relay and delete those billed minutes entirely** — that is the single largest
lever on continuation cost, and it is a maintainer decision.

### Four places the production path disagreed with all that

The third review accepted the handoff and refused C again on four mismatches
between what was *priced* or *written down* and what the driver actually does.

**Stage P now consumes `arms_needed`.** `remaining_work` charged a continuation
for only the arms its remaining probes need — `{advanced, B}` after a committed
screening — while `stage_p` rebuilt all six unconditionally. The budget could
therefore sit *below* the GPU work, which is the one direction a budget must
never be wrong in. Stage P reads `arms_needed` from the continuation manifest
(the set the launcher priced before a pod existed), builds only those, and
records both sets in its stage evidence. Every candidate's frozen **metadata**
is assembled either way, so the schedule, the ranking and the frozen tie-break
are unchanged by building two arms instead of six; a non-materialized
candidate's `durable_path` is a sentinel that names its own reason, and
training from one is a refusal.

**`run_rung` has three states, not two.** Its only test was `name in
self.scores`, so a probe that trained, became destination-verified, and then
failed scoring was **retrained** on the replacement — against R3 and against
this repository's own continuation text. A restored trained-but-unscored probe
now resumes at scoring through `score_existing`, which never calls the trainer
and charges the battery only.

**The training descriptor is durable before scoring.** The per-probe record was
written only after a *successful* score, so the real failure sequence — train,
become durable, scoring dies, pod dies — left the destination holding verified
bytes and an ack carrying only the checkpoint's identity, with nothing to say
which probe it was. It is now written when training finishes and again after
the durability announcement, both before scoring is attempted. Bytes without
that descriptor are preserved and explicitly **not** consumable. The old
rehearsal fixture hid this by pre-writing a completed record, which is stronger
than anything production produces; the new test drives the real order and
pre-writes nothing.

### R10's last two edges: a cost proxy is not an identity, and three states price as three

**A cost proxy was deciding which candidate's bytes exist.** While screening is
uncommitted the confirmation rung's candidate is unknown, so the dearest
admissible candidate was substituted as a materialization bound — sound as a
*cost* figure, and then handed to Stage P as an *arm identity*. Screening
scores decide who wins and have nothing to do with build cost, so a ranking
that advanced any other leaf would meet `NOT_MATERIALIZED` at stage C and the
campaign would fail for a perfectly legitimate winner. The old
partial-screening test stopped at "a partial field cannot rank" and never ran
`P → S → R → C`, so it could not see it.

**Plan A**, chosen over deferring the winner's build to after the ranking:
while screening is uncommitted, every candidate that can still be advanced is
materialized. The bound is more conservative than one worst case and it is
*exactly what executes*, which is the property a proxy cannot have. The
alternative saves a couple of arms and buys a new conditional materialization
between stages R and C — new machinery on the paid path for about `$0.8`.
Once screening has committed, only the advanced candidate and the anchor are
owed.

**Remaining work now prices the three states the driver executes.** A
trained-but-unscored probe resumes at scoring, so it owes the battery and not
the trainer, and owes no arm rebuild at all — it already holds its trained
checkpoint. `session_decomposition` takes `train_and_score_probes` and
`score_only_probes`; `arms_needed` is derived from the *untrained* probes only.
The split is not invented: the frozen record carries per-probe `train_minutes`
and `eval_minutes` as means and observed maxima, twelve times each
reconstructs its own `bounding_basis` totals, and that reconstruction is
checked — a record whose parts stop summing refuses rather than being split on
an assumption. Defaults still reproduce `1294.87` / `1800.53` exactly.

**Cumulative campaign spend is all-in.** `SessionRunner` records
`cost.actual_usd` from `self.usd()`, which is GPU only — the provider bills the
provisioned container disk separately and the runner never sees it. Summing
that against `all_in_hard_usd` checked `prior GPU + future GPU + future disk`
and dropped every predecessor's disk. `prior_attempt_actual` now derives it
from the authorization's own disk rate times that resource's elapsed minutes,
ceiled to the 4-decimal quantum because a spend accumulating against a ceiling
rounds up. A created resource whose cost or minutes cannot be read is
**UNKNOWN and refuses** — never `$0`. Generic core is unchanged: the arithmetic
belongs to whoever holds the all-in ceiling.

**Live-rate authorization.** `authorized_gpu_usd()` derived from the `$1.09/h`
constant and `main()` called `window_minutes(args.max_price)` without the
authorization's own GPU amount, so a valid authorization re-quoted at another
rate still inherited the old dollar window. The authorization now carries
`rate_usd_per_hour`, `hard_runtime_minutes`, `gpu_hard_usd`, `disk_hard_usd` and
`all_in_hard_usd` distinctly; its loader refuses a missing amount and reconciles
the dollars against the runtime and the all-in against the sum; and the deadline
is the **shorter** of what the authorized dollars buy at the live rate and the
authorized runtime. `--max-price` above the authorized rate is a `$0` refusal.

**Scope.** The authorization said "Plus ONE materialization of B"; six execute.
It now states six exact-digest-gated fixed-path materializations followed by
exactly twelve probes, and still forbids every beam, re-ranking, B
state-eval remeasurement, fourth seed, C3 and C4.

Three defects were found adjacent to this work and fixed, all `$0`:

* the launcher's `record_run` call passed `present=` and `stage_id=`, neither of
  which exists in that signature, so every invocation raised `TypeError` into a
  `finally`'s `except` and printed a warning — **the run manifest was never
  written on any path**, including the `$0` refusals whose only evidence it is.
  The continuation gate reads those records, which is how it surfaced.
* **a pod-side test asserted a dev-box path.**
  `test_the_durable_store_can_hold_twelve_probes` lived in the pod selection and
  asserted `/home/ecs-user/aad-artifacts` is a directory. True on the dev box,
  true under `simulate_pod_env.sh` — which isolates `$HOME` as an *environment
  variable* and does not hide absolute paths outside the repository — and false
  on a container that has no `/home/ecs-user`. So the launch-bound readiness
  sweep would have been green about a gate that fails at TESTS_OK a minute or
  two into a billing pod. Reproduced at `$0` with `unshare -r -m` and a tmpfs
  over the store: **at `147b2c6` the selection fails; on this tree all 121 pass.**
  The check moved to `tests/autoinit/test_c2_behavioural_launch_governance.py`
  with the other three dev-box-only cases, and its real production caller,
  `destination_gate`, now has tests — it had none, and had drifted to reading
  the `DURABLE_STORE` constant while the fetcher honoured `--ckpt-store`.
* the three new `tests/**/test_*.py` files staled the committed skip-predicate
  audit digest, as they always do; regenerated.
* **a one-in-eight flake inside the paid pod's blocking gate.** The rehearsal's
  per-sample fixture seeded itself from Python's built-in `hash()`, which is
  randomized per process, so the decision the real rule reached on that data
  moved between runs: at `PYTHONHASHSEED=7`,
  `test_a_null_effect_does_not_manufacture_a_winner` comes out GO and fails.
  Confirmed identical at `147b2c6` in a detached worktree. That test runs in
  `tests/c2_behavioural_preflight/`, which IS the pod's TESTS_OK gate — so
  roughly one launch in eight would have died at setup on a billing machine for
  a reason nobody could reproduce. The seed is now a stable sha256 of the
  probe's identity.

**The grant now exists; the rest of the chain is being built in order.** What
this section said before — that no grant existed and none could be created
without a maintainer decision — was true until 2026-09-21, when that decision
was made. Owners:
[`c2_behavioural_grant_proposal.json`](../stages/stage-1/phase_c2_behavioural/plans/c2_behavioural_grant_proposal.json)
(regenerate with `scripts/autoinit/write_c2_behavioural_proposal.py`) and
[`c2_behavioural_resume_preregistration.json`](../stages/stage-1/phase_c2_behavioural/plans/c2_behavioural_resume_preregistration.json).

**No scientific run is in flight.** Replay campaign: `$4.77` authorized,
`$3.27` spent across nine attempts, `$1.50` left and no further replay owed.
Project: `$309.2043` of `$370.0000` — owner
`scripts/consolidate/derive_budget.py --json :: project`.

## The suite is 38 red, and a reader deserves the attribution

A permanently red suite is a hazard — it is what let 14 failures sit unnoticed
at a remote HEAD once — so the count is named here rather than left as folklore.
Every one of the 38 fails in the direction that REFUSES rather than permits, and
none blocks development.

**35 were already red at `c87f876`**, verified by running each failing module in
a detached worktree at that commit. They are two long-standing families, both
needing a maintainer because re-cutting a frozen digest is a change to a frozen
record:

| family | count | what it says |
| --- | --- | --- |
| Phase-B / continuation-B frozen executable drift | 21 | the declared set no longer describes the tree; launch gates refuse, correctly |
| C1 session contract and readiness | 13 | C1's committed readiness record no longer binds the live harness, for an experiment closed by a verdict |
| C2 full-search proposal + architecture declarations | 4 | recorded proposals and core-change declarations predate later commits |

**3 are new, and they are mine.** All three are the same fact: this round edited
two files that belong to *other phases'* declared harness sets —
`src/aadistill/runtime/leaf_durability.py` (one identity construction shared by
sender and receiver) and `scripts/pod/autoinit_preflight_setup.sh` (the
`SESSION_KIND=c2_behavioural` branch, without which the session cannot
authenticate at all). Both edits are required and neither is revertible without
breaking the work they enable.

* `test_phase_b_historical_amendments.py::test_the_writer_refuses_to_reaccount_for_the_same_tree`
* `test_continuation_b_executes.py::test_the_SHARED_commit_gate_accepts_the_continuation_source_identity`
* `test_continuation_b_executes.py::test_the_gate_probe_itself_can_fail`

The repository HAS the mechanism for this — the Phase-B historical amendment
ledger, 19 entries, which records why a shared-owner file moved and why it does
not invalidate the frozen experiment. **It was deliberately not used here.** Its
writer takes a `--maintainer` argument, and the family it would touch is the one
a previous session explicitly reserved: *"re-freezing is a change to a frozen
record."* Recording an amendment autonomously would move a frozen record's
status without the decision that owns it. It is offered as the obvious repair,
not taken.

One core change WAS declared, because that mechanism is agent-usable and has no
maintainer field: `tests/architecture/test_cuda_surface_preserved.py` gained a
round declaring the `leaf_durability.py` extraction, which closed 10 failures.

## The full joint re-search RAN, produced a Top-5, and then lost it

**Attempt 3, 386.2 min, `$7.02`, RETURNED TO REVIEW — not retried.** Formal
measurement had begun, so the instruction is preserve, tear down, reconcile and
return, and that is what happened: no fourth chain, no grant, no sweep, no
provider resource.

What the beam produced, before anything went wrong:

| | |
| --- | --- |
| expansions | **108** — width 34, FFN 26, ATTENTION 22, DEPTH-causal 16, DEPTH-positional 8, composite 2 |
| states journalled | **202** — 20 at path length 1, 120 at 2, 50 at 3, 12 at 4 |
| complete leaves ranked | **14** |
| **selected** | **5**, with the selection carrying its own `sha256` and the journal's |
| operator + load time | 354.1 min of the 386.2 |

That is the shape beam width 6 with one warmup level produces over a
four-operator space, and the selection records the policy hash, the config
hash, the seed `20260815`, the suite and both profiles. **Whether it is an
admissible scientific result is a review judgment**: the driver's own terminus,
`commit_top_k`, never executed.

### What failed, and that it was written down

After the line `stage-1 selection committed: … (5 leaves)`:

```
OSError: Repo id must be in the form 'repo_name' or 'namespace/repo_name':
'/workspace/aad/artifacts/stage1/qwen3_0p6b_init_v0/checkpoint'
```

`run_phase_a_search` injects the canonical 0.6B control as its measured control
once the beam finishes. This session **deliberately does not stage it** — the
driver passes `conditional_candidates=None` and its own comment says *"B is now
measured and frozen, and this session compares nothing"*. Absent on the pod,
the path was read as a HuggingFace repo id.

**Search-1's preflight predicted this in as many words.**
`test_the_canonical_control_checkpoint_is_readable` says: *"`run_phase_a_search`
injects it as the measured control and verifies its frozen single-file sha256;
**a missing config aborts after the search**."* I read that test while building
this session's preflight, used it to conclude the control was Search-1's
concern, and asserted positively that this session does not stage it. I checked
the **driver** and the **launcher** for `CANONICAL_INIT` and found nothing. I
did not check the shared search entry point they call, which is where the
reference lives.

No gate could have caught it: every gate and all 18 preflight tests run
*before* the beam. This line is reached only after a complete beam finishes.

### The five checkpoints are lost

`commit_top_k` never ran, so the failed-run artifact policy collected evidence
rather than weights, and the pod was deleted. The five are **identified
exactly** — state ids, paths, artifact digests, `checkpoint_sha256` each, and
596,049,920 parameters — and their bytes are gone. This is the failure mode
AGENTS.md names outright: C1 attempt 17 trained six probes over ten hours and
lost every one the same way.

Re-materializing them is a **deterministic replay** in principle, not a new
search: the seed, config hash, space, policy, operator paths and calibration
profiles are all recorded. Whether that replay is scientifically equivalent,
and whether it may stand in for the originals, is a decision for review — and
it costs GPU time no authorization covers.

### Evidence preserved

`evidence/` holds the selection, 108 telemetry rows, a 202-state compact
journal with every digest and checkpoint hash, the driver's record, and a
pointer to the 61.4 MiB full journal — which lives out of tree per §2.5, with
its `sha256` matching the one the selection itself recorded. That journal is on
**one machine**; if it matters beyond the compact form, a durable-storage
decision is owed.

`$7.16` of the `$34.8742` ceiling is spent across three attempts. Project
cumulative `$305.8841` of `$370.0000`.

**TWO DECISIONS ARE OWED, and the second one arrived today.** The launch review
is the first. The second is that adopting the measured `state_eval`
optimization moved an evaluator hash that C2's **frozen** baseline-completion
protocol binds by content, so that protocol's gate now refuses a future B
re-measurement — correctly. No completed result is affected and the full search
is not gated on it, but amending a frozen protocol, re-measuring B, and
reverting a measured optimization are all maintainer calls. The options are
laid out under *[The full suite is not green](#the-full-suite-is-not-green-20-failures-two-families-one-of-them-new)*
and nothing was done in any of those directions.

**B WAS MEASURED, THE B→C COMPARISON EXISTS, AND IT HAS BEEN REVIEWED AND
ACCEPTED** as valid **search-stage** evidence. Baseline-completion attempt 8
(`$0.5872`) rebuilt the frozen C1 treatment baseline to its expected digest
`53e30566…`, measured it once on the frozen `state_eval@v1` suite, and computed
the preregistered comparison:
[`c2_baseline_comparison.json`](../stages/stage-1/phase_c2_baseline_completion/runs/attempt8/evidence/c2_baseline_comparison.json).
Verdict `CANDIDATE_IN_A_BETTER_FRONT_THAN_BASELINE` — front 0 holds four
candidates, B sits in front 1 with one, and B is dominated on all three ranked
objectives by two candidates while dominating none. All of that evidence is now
**frozen**.

**The next C2 step changed.** The maintainer decision of 2026-09-17 **withdrew**
the preregistered local Search-2 refinement. Search-1 is kept as a
**restricted-space validation experiment**: its value is that after promoting
`attention.activation_importance_v1`, changing order and composition *alone*
produced a real structural signal on the cheap metric. That is evidence about
the **search procedure**, so the informative next step is to widen the search
rather than polish locally inside a restriction.

```text
C2 Search-1 restricted search [DONE / FROZEN]
  → C2 full joint re-search
  → Top-K / Top-5 candidate selection
  → bounded 0.86M behavioural recovery selection
  → C2 incumbent
```

**The full joint space is derived, and `576` is not the space.** Enumerating the
live registry gives **578** reachable leaves — **576** four-operator leaves plus
**2** single-step `COMPOSITE_STAGE1` leaves that reach the target directly.
Phase B's comparable space was **290**, and the growth is exactly one extra
branching factor: the promoted ATTENTION operator consumes calibration where
`attention.weight_proxy_v0` declared `CalibrationNeed.NONE` and was therefore
offered once however many mixtures were active. Nothing is pinned — every
applicable implementation, every applicable profile and every order compete, so
calibration choices can affect pruning. Owner:
[`full_search_space.py`](../../scripts/experiments/phase_c2/full_search_space.py),
with a test that refuses those integers as literals.

**One exclusion, and it is scientific, not economic.**
`attention.weight_proxy_v0` is out because **C1 is** the isolation experiment
between it and the promoted operator, and it has a completed `GO` verdict.
Re-admitting the loser would cost ~50% more search (866 leaves) to re-decide a
closed question. The cheap alternatives `depth.positional_v0` and
`composite.stage1_sandwich_v0` are **in**.

**Cost is better measured than it was.** The table pools both committed searches
and takes the per-cell maximum. `attention.activation_importance_v1` is **no
longer an unmeasured input** — Search-1 priced it at `1.5×
width.global_pca_v0`, C2 attempt 4 then ran it 14 times *below* that proxy, so
the margin was conservative in the safe direction and is retired. One cell moved
the other way and it is the one the price turns on:
`depth.causal_kl_greedy_v1` deeper, `31.10 → 36.07` min.

**C2's BEHAVIOURAL QUESTION IS NOW STATED SIMPLY.** Two rounds of review
corrected it. C1 already established the behavioural incumbent **B**: after the
frozen `0.86M` recovery, pooled `correct_overall` was `105/2550 = 4.12%` against
the old incumbent's `70/2550 = 2.75%`, a paired `+0.01372549` with a `GO`
verdict. Neither Search-1 nor the full joint search performs any recovery
training, so the only behavioural question after the search is:

> Does the selected full-search initialization **C**, after the same frozen
> `0.86M` recovery, outperform **B**?

**The original control is no longer a C2 arm.** For this comparison it answers
nothing extra: a candidate that beats the old control but loses to B must not
promote, and one that beats B gains no promotion information from it. The
guardrails use the incumbent-relative semantics C1 already used, with B in the
comparator position. That also keeps the cycle scalable — C4 and beyond challenge
whatever incumbent the previous turn left, instead of repeatedly retraining the
project's original initialization.

**Screening is now disjoint in BOTH dimensions, and it had to be.** Disjoint
recovery seeds alone are insufficient: C0's inferential unit is the **prompt**,
and it measured substantial same-prompt cross-seed dependence — ICC `0.25 ±
0.095`, `P(correct | correct on another seed) = 0.257` against a `0.022`
marginal, an `11.7×` lift. Selecting and confirming on the same prompts would
leak the selection into the confirmation however fresh the seeds were. So
[`c2_screening_v1`](../stages/stage-1/phase_c2/plans/c2_screening_battery.json)
was built and frozen: 950 prompts / 850 scorable, C1's mixture preserved exactly,
content `0ad76fc7…`, and **measured** disjoint from
`c1_confirmation_v1` by stable id *and* normalized prompt content — zero shared
on both — as well as from calibration, `state_eval`, the recovery corpus and the
reserved final-promotion battery. It produces no verdict and may promote nothing.

> **It was rebuilt once, for a real bug.** `rank_take` accepted a `domain`
> argument and did not forward it to `rank_key`, so the first build silently used
> C1's rank domain: disjoint and deterministic, but drawn under an ordering its
> own manifest did not claim. The repaired sample differs in **every** stratum
> (`c04d9d64…` → `0ad76fc7…`). Three regressions now cover it,
> each confirmed to fail with the bug reinstated — including one that re-derives
> a stratum under the declared domain and requires the frozen sample to be that
> one and *not* the default domain's, which is the provenance claim itself.

| rung | seeds | arms | battery | probes | decides |
| --- | --- | --- | --- | --- | --- |
| screening | 1 | Top-5 + B | `c2_screening_v1` | 6 | which **one** candidate advances |
| confirmation | 3 | C + B | `c1_confirmation_v1` | 6 | the C2 incumbent |

**12 probes, exact** — down from 15, and no conditional rung. The screening rule
is frozen: maximize the paired single-seed Δ`correct_overall`(candidate − B), B
as anchor, `usable_rollout` never positive credit, ties broken by the frozen
full-search ordering then the deterministic state id.

**The interpretation boundary is recorded.** The search stages are
initialization-search only and train nothing; their KL/`state_eval` output is
hypothesis-generation and candidate-selection evidence that may never promote;
C1's `4.12%` is a *post-recovery* measurement, not raw initialization accuracy;
and C2 promotion depends only on the fresh recovery comparison of C against B.

**The search execution path exists and was run for real.** The
[full-search driver](../../scripts/pod/autoinit_phase_c2_full_search_driver.py)
does `bind_identities` → `full_joint_search` → `commit_top_k` and **stops**, with
no code path into a behavioural stage. It was executed end to end at toy scale —
real operators, real checkpoints, real reloads, real measurement — which found
and closed a real defect (a `relative_to` that raises when the workdir sits
outside the repository). The behavioural session and the launcher/governance
chain are **owed at authorization time** and deliberately unbuilt: two
authorizations, never one.

**THE BLOCKER IS BUDGET, NOT DESIGN.** A funding decision is required, at the
**standing** beam width 6.

| | expected | ceiling |
| --- | --- | --- |
| full search, **beam 6 — standing design** | `$16.0998` | `$33.1829` |
| behavioural selection (12 probes) | `$20.6926` | `$29.8788` |
| full search, container disk (400 GB) | | `$1.6913` |
| behavioural selection, disk upper bound | | `$1.5229` |
| **complete standing chain, TOTAL** | **`$36.7924`** | **`$66.2759`** |
| remaining headroom | | `$72.1205` |
| **headroom after the chain's total** | | **`$5.8446`** |
| minimum cumulative cap that contains both | | `$364.1554` |
| *(estimate only)* optimized window, not the ceiling | | *`$26.2607` GPU over `1445.54` min* |

**The window above is the CONSERVATIVE one, deliberately.** The measured
component speedups imply `1445.54` bounding minutes and `$27.5992`, and that
figure is recorded — in the pricing record's `optimized_planning_estimate`
block — as an engineering **planning estimate**. Review kept `1826.57` as the
authorization basis for the first optimized formal search: a hard ceiling
derived by component-level extrapolation can under-authorize a run, and an
optimized implementation that finishes early simply spends less than its
ceiling. After that search completes, **its own** per-expansion telemetry
becomes the measured basis and the adjustment retires.

The two ceiling rows above are **GPU runtime only**. The provider bills
Container Disk separately at `$0.10/GB/month`, and this session
provisions 400 GB of it, so a GPU-only ceiling did not cover the
session — and no figure in the record disagreed with any other, which is
why review found it rather than a gate. The GPU rate is re-quoted live;
the storage price is a dated stated basis and
[`provider_storage_pricing.json`](../../configs/infrastructure/provider_storage_pricing.json)
says so in a field a machine reads. The behavioural session's disk term
is **bounded, not derived**: its launcher and provision do not exist yet,
so it is bounded above by the search's own 400 GB and should fall when
that session is bound.

The chain **fits the accounting envelope and is still NOT AUTHORIZED**: the
maintainer raised the cumulative cap to `$370.0000` on 2026-09-17 as an
accounting envelope, explicitly not a spend authorization and not transferable
to C3 or C4. Fitting is not permission.

Stating that minimum is **not** requesting it, and these are **not yet**
funding-decision numbers: the behavioural protocol they price has just been
repaired and is awaiting review.

**Beam 2/3/4 are priced as scientific alternatives, not as cost options.**
Narrowing the beam merely to fit the existing cap is not permitted; the scope was
not shrunk to fit.

**A mean is not doing a bound's job.** The per-probe ceiling rests on the
observed **maxima** from C1 attempt 18's per-probe marker stream — training
spread `1.003×` and is effectively deterministic, scoring spread `1.241×` and is
not, so a named `generation_length_risk` reserve funds a doubling of its observed
maximum for unseen checkpoints.

**A >1-session search is not currently available.** Search state ids are
content-derived, which gives a state an identity — not its bytes. The frozen
Search-1 plan records that the multi-gigabyte search workdir *cannot be relayed
for resume*, so a fresh provider resource must re-derive lost state. No durable
cross-session mechanism was implemented or validated this round, and none was
built: it is a possible future design option and no plan here assumes it.

**Every probe is trained fresh.** The historical-probe-reuse ruling records
`reuse_verified: false` — all eleven examined probes fail
`scoring_contract_matches_live` — so there is no admissible reuse to net off.

**The roadmap is now C1–C4**, and the repeated shape is written down once as a
family-neutral pattern in
[`OPERATOR_PROMOTION_CYCLE.md`](../../docs/OPERATOR_PROMOTION_CYCLE.md):
operator R&D → isolation → promotion → full joint re-search → behavioural
selection → new incumbent. C3 is causal-KL ATTENTION isolation on the C2
incumbent and cannot start before C2 names one; C4 is conditional on C3
promoting.

## The state-eval certification — `$0.8446`, and what it did and did not settle

**Closed 2026-09-19.** The full-suite certification review made a launch
precondition. Evidence:
[`validations/state-eval-certification/v1/closeout.json`](../stages/stage-1/phase_c2/validations/state-eval-certification/v1/closeout.json)
· the collected report is in
[`c2_state_eval_cert/runs/c2_state_eval_cert_20260919_s2/artifacts/`](../stages/stage-1/c2_state_eval_cert/runs/c2_state_eval_cert_20260919_s2/artifacts/).

Run on the **complete** frozen suite — 80 items, `74,022` prediction positions,
5 domains, 7 sub-types, 4 declared critical-token classes — with the full
`StateEvaluation` reconstructed under both implementations on **provably
identical** logits (a sha256 per id sequence, checked on every later forward in
any pass).

| predeclared requirement | result |
| --- | --- |
| ranked-objective **absolute** drift `< 1e-5` | **`9.032e-06`** on `worst_domain` — **MET**, `11.1×` below the `1e-4` epsilon |
| identical objective ordering | **MET** |
| identical Pareto front membership and selected ids | **MET** — `[['m0_05'], ['m0_35'], ['m1_0']]` under both |
| identical decisions at the epsilon boundary | **NOT MET** — 2 of 5 cases changed |
| every emitted metric present on both sides | MET |
| identical critical-token position counts | MET, all 6 emitted tags |
| top-1 agreement | drift **exactly `0`** |

**The drift is proportional to the metric, not a fixed offset.** Relative
disagreement is ~`1.7e-05` at every magnitude, so the absolute figure tracks
the value: `9.032e-06` at `m=1.0` where `worst_domain` KL is `0.316`, and
`1.4e-09` at `m=0.05` where it is `0.0008`. What this certifies is the
coefficient; a real candidate's absolute drift depends on how far that
candidate sits from the teacher.

**The two boundary cases that changed are the two closer to the boundary than
the drift.** `at_epsilon` sits at `0` from it and `just_outside_epsilon` at
`1.0e-07`, against a drift of `9.032e-06`. The three cases placed at or beyond
the drift all held — including both placed at exactly `ε ± drift`. Reproduced
at `$0`.

That is **not a finding about the optimization**: any nonzero disagreement
flips a decision for a pair whose gap lies within it of `ε`, including the
float32 noise of one implementation against itself, which this project has
never measured. The requirement as I implemented it was unsatisfiable —
placing a pair `1e-07` outside a boundary and applying a `9e-06` disagreement
must cross it, and at *exactly* `ε` the rule turns on `>` versus `≥` of a
difference floating point does not represent exactly. **I did not re-engineer
the construction or re-run.** The instruction was that an exceeded target stops
the run and the judgment returns to review.

> **What review is being asked:** whether a `9.032e-06` absolute drift is
> acceptable against a `1e-4` epsilon, knowing it can only move a decision for
> candidates whose gap on a ranked objective lies within ~9% of `ε` of the
> boundary, that the real-candidate decisions were identical, and that the
> drift scales with the metric value.

### The 76× does not survive the complete suite

| | old | new | ratio |
| --- | --- | --- | --- |
| whole state-eval pass | `228.7 s` | `86.8 s` | **`2.63×`** |
| the reduction alone | `133.3 s` | `13.3 s` | **`10.06×`** |
| the forward (unchanged) | | `73.6 s` | |

The four-item benchmark measured the new reduction at `0.0246` ms/position and
reported `76.0×`. On the complete suite it is `13.256 s / 74,022 = 0.179`
ms/position — **seven times slower per position** — while the old path
reproduces almost exactly (`1.80` against `1.8699` ms/position). So the honest
full-suite figures are `10.06×` on the reduction and `2.63×` on the pass.

**This is precisely the extrapolation review forbade, and it is why keeping the
conservative `1826.58`-minute window was right**: the planning estimate applied
the `76×` figure to the whole `state_evaluation` phase. Nothing here re-prices
anything — and this pass is not an expansion either, since an expansion
forwards a 596M student where this perturbs the teacher's own logits.

Two subruns, `$0.8446` of a `$1.50` ceiling, both pods provider-confirmed gone,
one billing resource at a time. **s1 failed on my instrumentation and cost
`$0.4925` of evidence**: `--out` defaulted to `None` while the launcher
collects `artifacts/validation` and passes no `--out`, so the run measured all
three candidates and wrote nothing. Its repairs — report on every exit path, a
diagnostic bound that does not divide by a near-zero value, the forward timed
apart from the reduction, and the candidate reusing the reference forward — are
each held by a `$0` test.

## The performance round — `$0.2252`, and the price it moved

**Closed 2026-09-18.** Four candidates, two adopted, one refused on evidence
and one instrumented to a negative finding. **No scientific semantics changed:**
the 578/576 space, beam 6 / warmup 1, both calibration profiles, every
calibration item, the 260-evaluation causal-KL greedy rule, the frozen
`state_eval` suite, the Pareto and ranking policy, Top-5 semantics and the C2c
behavioural protocol are all as they were. Evidence:
[`validations/full-search-performance/v1/closeout.json`](../stages/stage-1/phase_c2/validations/full-search-performance/v1/closeout.json)
· [`analyses/full_search_performance_round.md`](../stages/stage-1/phase_c2/analyses/full_search_performance_round.md).

| candidate | outcome | measured |
| --- | --- | --- |
| **1.** keep the `state_eval` reduction on the card | **ADOPTED** | **`76.0×`** (`1.8699` → `0.0246` ms/position) |
| **2.** DEPTH forward-KL-only hot path | **ADOPTED** | `1.10×` on the scoring loop |
| **3.** reuse reference-side normalization | **NOT ADOPTED** | needs `33.83 GiB` against a `16.91 GiB` constraint |
| **4.** diagnose reference-cache variability | **instrumented only** | `0.013 GiB` reclaimable; cache admits `67/67` |

**What that round measured, and two corrections review made to how it was
read.** Worst **relative** drift `3.03e-05`, on pooled per-item KL over four
calibration items; item ordering identical, top-1 agreement exact, DEPTH's
removal order `[17, 18]` in three independent measurements. That is a valid
**kernel-level** result. It was reported as a decision-level one, twice over:

* **`0.007782` is not the search's decision threshold.** It is C2's pre-B
  numerical-**sensitivity disclosure** trigger — the tightest gap observed
  *between the frozen C candidates* on `worst_domain` — and its own record says
  it is "NOT an estimated noise bound, NOT a measurement of cross-session
  variance, and NOT evidence of numerical determinism". The Pareto decision
  epsilon is **`1e-4` absolute**, per objective. Dividing an absolute gap by a
  *relative* drift also gives a number in no units, so the `257×` was not a
  safety factor.
* **`sqrt(V)·ε` is an error-scale heuristic, not a hard floor.** It is fine for
  setting a tolerance and proves nothing about what agreement is achievable.

The decision-level claim is the
[state-eval certification](../stages/stage-1/phase_c2/validations/state-eval-certification/v1/)'s:
**absolute** drift on the ranked objectives over the complete frozen suite,
against the `1e-4` epsilon, with the Pareto decisions checked directly.

**Candidate 4 refuted its own hypothesis.** It was instrumented to test whether
allocator hoarding explains the historical `2.6 GiB`-free observations. On a
card holding only the teacher there is nothing for `empty_cache()` to return
and the whole cache is admitted, so the answer is **live tensors elsewhere in
the search** — and the measurement has to be taken mid-search, not on a clean
card. **Nothing was flushed and no saving is claimed.** The reference-cache
recompute waste — `182,780` recomputes against `323,180` ablated forwards,
`36.1%` of every forward pass — remains the largest known saving in the search
and is deliberately **not** priced in.

**The cost model was refreshed from the measurement, not from the speedup.**
Each cell is adjusted by the component saving it actually contains, capped at
the phase that saving belongs to — never a ratio applied to a whole cell —
and every input is named in
[`phase_c2_measured_optimization.json`](../stages/stage-1/phase_c2/plans/phase_c2_measured_optimization.json).
DEPTH `36.07` → `30.79` min, the others `3.93` → `1.65` and below. Beam 6 is
unchanged and was never a lever.

**One thing for a reviewer to notice:** the protocol document *embeds its own
cost model*, so re-pricing moved its hash from `26de0bb6` to `d7678d7d`. Exactly
**two** top-level keys differ — `cost_model` and the document's own
`protocol_sha256` — which is `29` changed leaves plus the self-hash. The
science subtree hashes **identically** before and after at
`4897d470eed2b5a3…` — the document with those two keys removed and the rest
canonicalized with sorted keys, so `git show HEAD:<protocol>` against the
working tree reproduces both figures. A document
declared frozen as science should probably not move when a price does; that is
a structural remark, not a change made here.

Three subruns, all on the formal target card, all torn down
provider-confirmed, `$0.2252` of a `$1.50` ceiling — the `$0.90` soft stop was
never reached. Each failure was in the instrumentation rather than in the
optimizations, and each is written down with its root cause.

## Getting here cost three aborted sessions and `$0.1033`

The maintainer decision of 2026-09-17 authorized two formal sessions to establish
B, measure it once on the frozen suite, and compute the preregistered B→C
comparison. **Both were consumed without reaching a rebuild.**

**The retry rule then changed shape.** The decision of 2026-09-16 continued the
work, raised **no** envelope, and prospectively replaced the two-session limit
with a **money** boundary: a fresh formal chain requires
`cumulative completion spend + $1.1950 <= $2.3900`. There is no fixed maximum
attempt number, and an incrementing attempt number is not a scope expansion — a
cheap pre-measurement abort consumes its actual cost and its one-use chain,
nothing more. The ceiling, the `$1.09/h` L40S boundary and every frozen
scientific identity stayed unchanged throughout; the project cap was
`$320.0000` for those attempts and is now `$370.0000`, owned by
`configs/experiments/phase_c1/authorization.json ::
accepted_pricing.cumulative_cap_usd`. Attempts 7
and 8 both ran under that rule, and it is what let the work finish without
another approval round.

| attempt | where it stopped | cost |
| --- | --- | --- |
| [5](../stages/stage-1/phase_c2_baseline_completion/runs/attempt5/closeout/outcome.json) | the launcher's **first statement**: `claim_output_root` took the stage id positionally where the signature takes `outputs` by keyword. No gate ran, no price was queried, **no provider resource existed**. The chain was consumed anyway — its one-use rule counts the invocation | `$0.0000` |
| [6](../stages/stage-1/phase_c2_baseline_completion/runs/attempt6/closeout/outcome.json) | **10/10 `$0` gates passed** and setup refused at **`ROPE_OK`**, which globs `artifacts/stage1/*/checkpoint/config.json`. This session stages no checkpoint — it rebuilds B on the pod — so the step had nothing to look at. `SETUP_RC=1`, no driver stage | `$0.0412` |
| [7](../stages/stage-1/phase_c2_baseline_completion/runs/attempt7/closeout/outcome.json) | **10/10 `$0` gates passed twice**, the pod came up and SSH answered — and the **launcher process was killed two minutes in**, by the agent's own blocking tool call. Setup never ran; `stages` is `{}`. The pod outlived its orchestrator and an explicit provider query removed it | `$0.0621` |
| [8](../stages/stage-1/phase_c2_baseline_completion/runs/attempt8/closeout/outcome.json) | **COMPLETE.** `SETUP_RC=0`, driver detached and confirmed by descriptor probe, both stages passed, B rebuilt to its expected digest, measured once, comparison computed. Pod deleted behind its teardown gate | `$0.5872` |

**Attempt 7 was not a repository failure, and recording it as one would hide the
real defect.** Every gate passed, the live price sat exactly on the `$1.09/h`
boundary, the bundle round-tripped to the authorized commit. The launcher was
started from a single tool call that also contained a foreground `sleep`, which
that harness blocks; the call ran to its two-minute timeout and was killed, and
the kill took the whole process tree — the `setsid`-detached launcher included.
**`setsid` defeats a process-*group* signal, not a supervisor that walks
descendants.** Worse, the watchdog was inside the tree it was meant to outlive:
its journal has exactly two polls, `13:10:28Z` and `13:11:29Z`, so the recorded
65.78-minute hard terminate could never fire, and this project has never once
seen the provider's own `--terminate-after` fire either. Attempt 8 starts the
launcher inside a **`tmux`** server — not a descendant of the starting call —
and that call returns immediately. No repository code changed, no gate was added,
and the closure and all six frozen identities are byte-identical.

Attempts 5 and 6 share a different class, and it is the one this repository keeps
paying for: **an inherited declaration rather than an inherited need.** C1
attempt 2 died on that same `ROPE_OK` line for `$0.1013`.

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

**Attempt 7 runs under the money rule, and it stops at the measurement.** A
failure *before* the durable `baseline_measurement` exists is handled
autonomously — teardown with provider-confirmed zero billing, preserve,
reconcile, diagnose, minimal repair, minimal regression, fresh identity, fresh
chain — for as long as `spend + $1.1950 <= $2.3900` holds, and an identical
unchanged failure is never retried. The instant that measurement exists the
authority inverts: **no GPU remeasurement of B**, no second `state_eval`, and a
downstream comparison failure is repaired at `$0` from the durable measurement
plus the frozen five. `$2.3488` of the `$2.3900` envelope is unspent and that is
still not permission for anything outside this scope.

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
| phase | C0 **COMPLETE**. C1 — fixed-path ATTENTION isolation, **CLOSED by a `GO` verdict**. C2 — **CLOSED WITHOUT PROMOTION** (maintainer decision 2026-09-24): Search-1 DONE and FROZEN, the local Search-2 refinement WITHDRAWN, the full joint re-search COMPLETE with an accepted frozen Top-5, screening complete, and all twelve behavioural probes trained and scored — but the confirmation field mixed evaluation-protocol identities, so **no canonical promotion verdict is claimed and no new incumbent is named**. **B, the frozen C1 treatment, remains the accepted incumbent.** C3 (causal-KL isolation) is **NOT STARTED** and is the next scientific stage | [`phase_c2/plans/phase_c2_full_search_protocol.json`](../stages/stage-1/phase_c2/plans/phase_c2_full_search_protocol.json) · [`phase_c1/plans/phase_c_roadmap.md`](../stages/stage-1/phase_c1/plans/phase_c_roadmap.md) |
| replay | **MEASURED — 2/2 PASS**, for the third time (attempts 9, 17, 18). Passing replay is not a result: 9 and 17 are **NO DECISION**, pre-treatment aborts that measured no endpoint. Attempt 18 is the only attempt that decided anything | [`attempt18/closeout/outcome.json`](../stages/stage-1/phase_c1/runs/attempt18/closeout/outcome.json) |
| treatment, endpoint | **MEASURED** — six probes trained and six evaluated on the frozen battery; the frozen Stage-I rule returned **`GO`**. Figures in the block below | [`attempt18/evidence/c1_decision.json`](../stages/stage-1/phase_c1/runs/attempt18/evidence/c1_decision.json) |
| launch chain | **every C2 chain is consumed and nothing is prepared** — Search-1 attempts 1–4, baseline completion 5–8, and behavioural 1–14. **No C2 chain may be built:** the stage is closed and no further C2 scientific spend is authorized. No further C1 attempt is authorized or prepared either; a complete verdict ended that round. C3 has no chain of any kind | [`phase_c2_baseline_completion/runs/attempt8/governance/`](../stages/stage-1/phase_c2_baseline_completion/runs/attempt8/governance/) |
| last attempt | **behavioural attempt14 — COMPLETE, `$2.0670`.** It reconstructed probe 11's evaluation for the P4 repair and reproduced the terminal state with a recorded two-prompt discrepancy, which is preserved rather than resolved. It also left a pod billing ~116 min behind a blocked artifact gate; that defect is repaired and accounted as PHB-HA-026/027 | [`attempt8/closeout/outcome.json`](../stages/stage-1/phase_c2_baseline_completion/runs/attempt8/closeout/outcome.json) |
| blocker | **NOTHING IS BLOCKED, AND NO SPEND IS AUTHORIZED.** C2 is **CLOSED WITHOUT PROMOTION**: no probes owed, no canonical verdict claimed, no new incumbent, **B stands by absence of a valid challenger**, no C2 launch prepared, and **no further C2 scientific spend authorized** — remaining allowance under the `$25.00` stage envelope or the `$42.0000` campaign ceiling is not permission. C3 is **NOT STARTED** and is the next scientific stage; it will be designed and authorized independently from the final C2 remote HEAD, and its accounting envelope is not transferable from C2 | [`phase_c2_full_search_pricing.json`](../stages/stage-1/phase_c2/plans/phase_c2_full_search_pricing.json) · [`budget/decisions.md`](../budget/decisions.md) |
| spend | owned by the budget block below | [`budget/ledger.md`](../budget/ledger.md) |

## C3 — the refactor is on review, and C3 has not started

**Branch `review/c3-operator-batching`. NOT merged into `main`.** `main` is
still `ab53ba14`. Everything in this section is engineering; no C3 science has
run and the C3 `$25.00` stage envelope is untouched.

**What the branch contains.** The operator-topology migration
(`operators/{attention/gqa,ffn/dense,width/residual,depth,composite}`) and
calibration micro-batching, delivered as one refactor. Batch size is a
**runtime** input — `ExecutionConfig`, threaded through `OperatorContext` and
never hashed — so it cannot enter a state identity. `batch_size=1` is
bit-identical to the pre-refactor path for all six operators. Every
initialization-side KL over variable-length items goes through one masked
batched per-item reduction, `forward_kl_mean_batch`; a pooled
`sum(KL*mask)/sum(mask)` is a token-weighted batch mean, is a different
objective, and is refused by test.

**The a4 CUDA finding was returned to review as insufficiently attributed.**
Four subruns totalling `$0.0822` produced a headline — a batched forward
differing from a solo one by `4.88e-02` relative on the logits in bf16, with
FFN top-k selection moving in 26 of 28 layers — and the maintainer rejected the
attribution on 2026-09-26. Two reasons, both accepted: the decisive numbers
came from ad-hoc scripts rather than a committed executable, and the
environment was unpinned. The record is **kept unchanged** as history:
[`finding.json`](../stages/stage-1/phase_c3/validations/batching-refactor-cuda/v1/finding.json).

**Nothing was applied in response to it.** `DEFAULT_MICRO_BATCH_SIZE` is still
`4`; state identity semantics are unchanged; no operator definition, seed or
recipe moved.

**This repository contains three runtimes, and which one a number came from is
the point.** The formal operator search executes under `/opt/train` — python
3.12, **torch 2.11.0+cu128, transformers 5.13.1**, installed offline from the
relay wheelhouse (`POD_IMAGE['remote_python']`, and C1 attempts 17/18 evidence).
The engineering CUDA validations run under the image's own python, **torch
2.9.1+cu130**. The rejected a4 run used a third: image `1.0.3-cu1281`, torch
2.9.1+cu128, and `pip install transformers` with no version pin.

**The root-cause investigation is the current work.** Ceiling **`$3.00`
cumulative, inheriting the `$0.0822`** already spent; `$2.9178` remains and no
part of it is C3's envelope. Its executable is
`scripts/validation/batch_invariance_diagnostic.py`, and the point of it is
that every number a conclusion rests on is emitted by that file: the verdict is
COMPUTED by `derive_conclusion()` from the stage outputs rather than written
beside them, and that function is tabled and mutation-checked in
`tests/validation/test_batch_invariance_conclusion.py`. Records:
[`scope.json`](../stages/stage-1/phase_c3/investigations/batch-invariance-root-cause/v1/scope.json),
[`authorization.json`](../stages/stage-1/phase_c3/investigations/batch-invariance-root-cause/v1/authorization.json),
[`campaign.json`](../stages/stage-1/phase_c3/investigations/batch-invariance-root-cause/v1/campaign.json).

**Five `$0` CPU rehearsals ran the real executable before any pod existed**, and
found five defects in it — two of them verdict defects that would have been
paid for. The largest: `derive_conclusion` read "not every backend diverges" as
"some backend is exact", and returned locus `attention_backend_kernel` for a run
in which nothing diverged at all.

**The measurement was made: four reports, one L40S, `$0.1005`, 5.45 minutes.**
Attempt `bi_20260926_d2`, pod `c6btbh69ql85ng`, teardown provider-confirmed.
Cumulative diagnostic spend `$0.2312` of `$3.00`. Owner:
[`finding.json`](../stages/stage-1/phase_c3/investigations/batch-invariance-root-cause/v1/finding.json),
**derived** by `scripts/validation/batch_invariance_finding.py` from the four
raw reports in
[`evidence/`](../stages/stage-1/phase_c3/investigations/batch-invariance-root-cause/v1/evidence/),
not typed.

**Verdict: the a4 finding is CONFIRMED in substance and its cause is
relocated.** On the a4 checkpoint under the pinned science runtime, FFN top-k
selection moves in **25–26 of 28 layers at every keep ratio and every micro
batch size** — a4 reported 22–27 of 28 across the same four ratios. On the
**parent**, which is what C3's operators actually calibrate on, it is worse:
**32–36 of 36**.

**The cause is a shape-dependent GEMM, not attention, not masking, not the
operator code, and not the dtype alone.** Only some projections move, and what
separates them is the reduction depth over the output width, `K/N` — the
split-K signature:

| | exact, `K/N` | shape-dependent, `K/N` |
| --- | --- | --- |
| parent | q `0.63`, gate `0.26`, up `0.26`, lm_head `0.017` | k `2.5`, v `2.5`, attn_out `1.6`, ffn_out `3.8` |
| 596M | q `0.5`, k `1.0`, v `1.0`, gate `0.33`, up `0.33`, lm_head `0.0067` | attn_out `2.0`, ffn_out `3.0` |

Every bit-identical projection has `K/N ≤ 1.0`; every divergent one `≥ 1.6`.
The separation is clean in all four reports.

**Four things it is NOT**, each measured rather than argued:

* **not nondeterminism** — each shape repeats itself bit-identically, 3×;
* **not cross-row contamination** — the focal row is *exactly* independent of
  neighbour token content (`C_vs_D` bitwise identical), and an all-ones mask
  changes nothing (`A_vs_B` identical). Batching changes the schedule, not the
  arithmetic's inputs;
* **not attention** — every backend diverges and none is exact (eager, sdpa,
  sdpa:MATH, sdpa:CUDNN). On Qwen3's GQA with a mask, SDPA falls back to math
  anyway: flash, memory-efficient and cuDNN all report themselves disabled;
* **not the collector's accumulation order** — re-summing the *same* captured
  activations in both groupings drifts by exactly `0.000e+00`. The forward
  moved; the float64 accumulator did not.

**The dtype is the scale, not the cause.** float32 has the *same* shape-dependent
GEMM — `fp32_gemm_also_shape_dependent` is `true` in all four reports — but the
logit-level magnitude is `876×` to `28,044×` smaller. So "that is the dtype
through 28 layers" was the wrong reading of the right observation.
`torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False`
does **not** remove it either, and on the parent it made the logit `max_abs`
*worse* (`1.88 → 2.61`).

**The runtime was not the explanation.** torch `2.11.0+cu128` (science) and
`2.9.1+cu130` (engineering) give the same answer on the same checkpoint: 26
layers moved either way, statistic drift `3.374e-02` vs `3.349e-02`. Pinning the
environment was the right thing to demand and it changed nothing.

**`batch_size=1` with no padding is exactly bit-identical, in every sweep.**
That is why the pre-refactor code was reproducible: it always put one item in a
forward. The refactor did not introduce this — it made the execution shape a
variable, and this property was already there.

**What it means for C3, stated but NOT acted on.** Operators whose output is a
dense top-k over activation statistics are not reproducible across
`micro_batch_size` on this hardware in bf16, and `DEFAULT_MICRO_BATCH_SIZE` is
`4`. The causal-KL scorer is affected more mildly — the mean moves `0.14%`–`0.43%`
on the parent because both sides share a batch composition and the perturbation
largely cancels — but the per-item **ranking is not preserved**. Nothing here
has been changed in response: no default, no identity semantics, no operator
definition. That is a maintainer decision.

**Two corrections to the previous round's report.** It claimed `0` new failing
nodeids against `ab53ba14`; there were **three**, all found here and all now
fixed: `test_every_pod_script_is_classified` (two new `scripts/pod` entries
never catalogued) and two budget-snapshot tests left stale by the `$0.0822`
booking. It also stated the formal environment as torch 2.9.1+cu130, which is
the engineering runtime, not the one the science runs in.

## Readiness

<!-- readiness:begin -->

| readiness | | owner |
| --- | --- | --- |
| latest POINTED-TO sweep — C1 attempt18 | **launch_bound — PASS**, swept at `e80eb60b`; **does not describe the current tree** | [`c1_pod_environment_verification.json`](../stages/stage-1/phase_c1/analyses/c1_pod_environment_verification.json) |
| every other experiment's readiness | **run-owned and not pointed at from here** — one record per attempt, under that attempt's `governance/readiness.json`, so a later sweep cannot overwrite what an earlier one launched under | [`stages/stage-1/`](../stages/stage-1/) |
| launch-bound for the next session | **not prepared** — no launch-bound sweep describes the current tree. Whether one is owed depends on whether a launch is authorized, which this file's launch-chain section owns | this file's launch-chain section |
| last launch-bound failure | swept at `82745981` on 2026-09-12 — kept as history, not a current state | [`readiness_history.json`](../stages/stage-1/phase_c1/history/readiness_history.json) |

*Generated from the record by `scripts/consolidate/render_log_navigation.py`; do not edit by hand — it went stale within hours when it was prose.*

<!-- readiness:end -->

## The full suite is not green: 14 failures, one family

Measured on the settled tree, 2026-09-19: **14 failed, 3207 passed, 15
skipped** over `tests/{docs,pod,architecture,autoinit,validation}`, with
`tests/{initialization,runtime,init}` clean in the same round. Every one fails
in the direction that **refuses** rather than permits, and all fourteen are one
family.

| family | tests | why it stays red |
| --- | --- | --- |
| **A committed digest no longer describes the live tree.** Phase B's amendments ledger accounts to `c20e3a80b6c0` while the tree digests to `c9121aadff77`; C1's preregistration, readiness record, skip-predicate audit and preflight selection are in the same position | 14 | The gates return `ok = False`, so a paid Phase-B or C1 launch is **refused** — correct. The remedy each message names is *"re-freeze it"*, which edits a frozen scientific record, and no C1 or Phase-B launch exists to justify a `launch_bound` sweep (AGENTS.md P8.3). Review confirmed these stay red as fail-closed guards and are not a blocker to the C2 full search |

<details><summary>the 14, by nodeid</summary>

```text
autoinit/test_phase_b_historical_amendments.py::test_the_ledger_verifies_against_the_live_tree
autoinit/test_phase_b_historical_amendments.py::test_an_incorrect_source_commit_is_refused
autoinit/test_phase_b_historical_amendments.py::test_a_missing_changed_file_is_refused
autoinit/test_phase_b_historical_amendments.py::test_a_false_numstat_is_refused
autoinit/test_phase_b_historical_amendments.py::test_a_false_after_file_hash_is_refused
autoinit/test_phase_b_historical_amendments.py::test_a_false_patch_hash_is_refused
autoinit/test_phase_b_plan.py::test_completed_phase_b_drift_is_historically_accounted_for
autoinit/test_skip_predicate_audit.py::test_the_committed_audit_record_matches_the_live_one
autoinit/test_staging_contract.py::test_c1_runs_only_its_own_preflight_on_a_paid_pod
autoinit/test_c1_readiness_gates.py::test_the_committed_record_still_binds_the_live_executable
autoinit/test_c1_readiness_gates.py::test_the_pod_selection_is_exactly_the_preflight_directory
pod/test_c1_session_contract.py::test_the_writer_refuses_to_rewrite_the_frozen_preregistration
pod/test_continuation_b_one_probe_contract.py::test_the_preregistration_binds_the_live_executable_digest
pod/test_phase_b_driver_and_launcher.py::test_the_preregistration_gate_refuses_a_tree_the_freeze_does_not_describe
```

</details>

**The evaluator-drift family is closed.** Six tests were red because adopting
the device-resident reduction moved `planning/metrics.py`, which C2's frozen
baseline-completion contract binds by content. Review's decision was to keep
that protocol and its historical hash **frozen exactly as they are** — the
contract is doing the correct thing by refusing a future measurement joining
the old B↔C series under a different evaluator. So the tests now assert the
**refusal**, and the position is recorded prospectively in
[`phase_c2_evaluator_lineage.json`](../stages/stage-1/phase_c2/plans/phase_c2_evaluator_lineage.json):
baseline completion is COMPLETE/CLOSED; its B and frozen C measurements remain
valid because both sides used the historical evaluator; the optimized evaluator
is the current Full Search implementation and is **not** eligible to append to
that series; reopening it needs a new explicit scientific decision.

**Attribute against the commit the session started from, not `HEAD`.** Eleven
architecture-guard cases once read as pre-existing in both the working tree and
a detached worktree at `HEAD` — they were mine, from a commit two back in the
same session. And a worktree has no untracked files, so four failures looked
pre-existing there for an unrelated reason (`artifacts/stage1/state_eval_v1` is
simply absent).

## Budget — four limits that do not transfer

Derived by [`scripts/consolidate/derive_budget.py`](../../scripts/consolidate/derive_budget.py)
from the approved package and each session's own closeout. **Do not restate
these by hand; run the deriver.**

<!-- budget:begin -->

| limit | remaining |
| --- | --- |
| formal sessions | `$22.8249` of `$45.4425` |
| GPU engineering | `$5.7688` of `$6.0000` |
| package | `$28.5937` of `$51.4425` |
| project cap | `$342.2014` spent of `$370.0000`, leaving `$27.7986` |

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

## The launch chain — nothing is prepared; the next one waits on funding

**No chain is owed and none is prepared.** C1's round ended with a verdict and
C2's baseline completion ended with a measurement. The seven steps below have
now been executed four times end to end (completion attempts 5–8), so the
ordering below is not theoretical — and the next chain cannot be built until the
full joint re-search is funded. The readiness
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

This changed no experiment result, no authorization and no frozen evidence, and
the launch chain is unaffected by it.

## Engineering, not results

The initialization migration and the two CUDA validations are engineering
records. The stage-F device repair is **CONFIRMED ON REAL CUDA** at execution
SHA `7027a8f4`. Neither is a C1 result and neither authorizes anything:
[`maintenance/source-relocations/initialization-core/v1/`](../maintenance/source-relocations/initialization-core/v1/) ·
[`phase_c1/validations/cuda-stage-f/v1/`](../stages/stage-1/phase_c1/validations/cuda-stage-f/v1/)

The **C2 full-search driver** is likewise **CONFIRMED ON REAL CUDA**: one
L40S (cc 8.9, bf16, torch 2.9.1+cu130) drove all three driver stages to
`ALL_DONE` over the real 578-leaf joint space, and every one of the 35
materialized states reloaded on `cuda` in `bfloat16`; the real 1024x28
student (596,049,920 parameters) built, saved, reloaded canonically and kept
its rope base of `5000000.0` and its tied head, peaking at 1.118 GiB. Three
subruns, `$0.1453` of a `$0.9000` ceiling, every teardown provider-confirmed.
Two of the three failed first, each on a different producer of non-source
input, which is now derived from code rather than listed. It measures no
behaviour and authorizes nothing, least of all the formal search:
[`phase_c2/validations/full-search-cuda/v1/`](../stages/stage-1/phase_c2/validations/full-search-cuda/v1/)

The **performance round** is a third engineering campaign on the same card:
`$0.2252` of a `$1.50` ceiling, three subruns, all torn down
provider-confirmed. It changed no science, adopted two optimizations on
measured equivalence, refused one on memory evidence and refuted candidate 4's
hypothesis. It is what the current price rests on — see the section above:
[`phase_c2/validations/full-search-performance/v1/`](../stages/stage-1/phase_c2/validations/full-search-performance/v1/)

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
