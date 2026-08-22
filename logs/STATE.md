**Updated:** 2026-08-22 · branch `main` · attempt 3 proved the transport and died in the test gate

# Current state

The **human view of [`current_state.json`](current_state.json)**. That file owns
the live facts; this one says the same things in prose and adds nothing it does
not carry. If the two disagree, a structural test fails.

**Nothing is running. Nothing is billing. Nothing is authorized. Nothing is
prepared for launch.**

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

**The five leaves currently have no route to a pod.** scp is infeasible above;
the relay alternative is off because it reported **1.60 GiB** of headroom against
**5.55 GiB** of leaves. This is a transport decision for the maintainer, not
something to route around. Full record:
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
cumulative spend   $213.9214
approved cap       $234.00    RAISED AND APPROVED 2026-08-21
remaining          $20.0786   $2.9698 SHORT of one more full Phase-A attempt

```

The cap went **$219.00 → $231.00** (2026-08-20, to fund exactly one Phase-A
attempt) **→ $234.00** (2026-08-21). Since then attempt 12 **ran** for $3.7872 —
far under its $23.0484 ceiling, because it stopped at Stage 2 rather than
training nine probes — and the three recovery continuations spent $0.0100,
$0.2389 and $0.2011 without executing a stage. A continuation's derived
**$16.7456** ceiling still fits inside the remaining $20.0786, with $3.3330 to
spare; a full Phase-A attempt does not. **Every grant issued so far is spent.**

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

After retiring 8.4752 GiB of remote copies and verifying the five-leaf
transport mirror. CPU only — no checkpoint loaded, no metric measured, no GPU
used:

* full suite **2160 passed, 12 skipped, 0 errors** in 20:03 — one more test
  runs and one fewer skips, because the transport is now verified rather than
  absent.
* pod simulator **2119 passed, 23 skipped**; artifact tree restored **exactly** —
  listing hash `c1726a62…` before and after.
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
