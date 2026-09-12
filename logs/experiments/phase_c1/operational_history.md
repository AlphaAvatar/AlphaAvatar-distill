# phase_c1 — operational history

The session-by-session record of Phase C1: what each attempt did, what it
cost, what failed and what was repaired. **Moved here from `logs/STATE.md`
on 2026-09-12**, unedited, because a current-state document that also carries
every incident becomes a place where the current state is hard to find.

It is history. For what is true **now** read [`../../STATE.md`](../../STATE.md)
and [`../../current_state.json`](../../current_state.json); for money read
[`../../BUDGET_LEDGER.md`](../../BUDGET_LEDGER.md); for the runs themselves
read [`../../runs/index.json`](../../runs/index.json).

Nothing here authorizes anything.

---


## APPROVED execution package — maintainer decision, 2026-09-11

Approved after independent engineering review of `55bb324a..199711af`. **An
approval is not a launch.** A formal attempt still needs, in this order: this
run's grant, a `launch_bound` readiness sweep on the clean pre-authorization
tree, a new one-use authorization, an exact-session bundle, a live quote at or
below the accepted rate, and every pre-provider gate. Full terms:
[`BUDGET_LEDGER.md`](../../BUDGET_LEDGER.md), [`decisions.md`](../../decisions.md) and
`execution_package` in
[`configs/experiments/phase_c1/authorization.json`](../../../configs/experiments/phase_c1/authorization.json),
which is also where the issuer reads the cap it refuses a mis-stated grant
against. **Booked against the package so far: `$0.3960`** (attempts 10 and 11;
attempt 12 cost `$0`).

> **AMENDED the same day.** After attempt 11 the maintainer **withdrew the
> three-attempt cap and the one-provider-resource-per-invocation rule.** The
> table and terms below are reproduced with that amendment applied; the money is
> untouched. Source: `execution_package._attempts_amendment_2026_09_11b`.

| | approved |
| --- | --- |
| formal C1 sessions, **including the first** | **no cap** — withdrawn 2026-09-11. Money bounds them: at `$15.1475` worst case, the `$51.0465` balance funds three full-ceiling sessions and no more |
| per-attempt hard ceiling | **`$15.1475`** — unchanged, derived from [`phase_c1_pricing.json`](../../phase_c1_pricing.json) at secure L40S `$1.09/h`. It already contains a 10% contingency and a 20-minute artifact-recovery reserve; nothing re-buys them |
| formal sub-total | **`$45.4425`** — approved as `3 × 15.1475` and unchanged by the amendment. It is now a *pool*, not three slots |
| GPU **engineering** allowance, cumulative across every subrun | **`$6.0000`** |
| package total | **`$51.4425`** |
| spend at approval | `$267.8998` |
| booked since | `$0.3960` (attempts 10 and 11), cumulative `$268.2958` |
| new cumulative cap | **`$320.0000`** (`267.8998 + 51.4425 = 319.3423`, plus `$0.6577` of reconciliation margin — the ledger has already needed a `$0.0073` correction once, and a cap with no margin turns an arithmetic fix into a breach) |
| cap **increase** | **`+$36.2400`**, not `+$51.4425`: the old cap already covered `$15.2025` of the package |

**The money is the ceiling, not a target**, and the engineering allowance is
used only when a question genuinely needs an accelerator — it is not a
requirement to add a GPU test before a formal session. **The four remaining
limits bind separately and do not transfer**: an unspent engineering allowance
does not raise the per-session ceiling, the package total does not draw on the
project cap, and the margin is for real accounting corrections, not spending.
What a cheap failure *does* now buy is another session — that is exactly what
the amendment changed — but only while the balance still funds a **complete**
one.

**Counting a formal session.** Invoking the formal launcher under a **new
one-use authorization is one formal session, even if it then refuses at `$0`**
before any provider resource exists — attempts 10, 11 and 12 are each one.
Read-only checks, code repair and engineering validation before issuance do not
count. **The counting survived the amendment; only the cap on the count was
withdrawn**, so sessions are still numbered, recorded and ledgered one by one.

**Inside a session:** one provider-create call per **draw**, **at most three
draws**, **at most one billing resource at any instant**, every draw recorded
with its own pod id, cost, raw provider response and watchdog journal, and all
of them sharing the session's single `$15.1475` ceiling. A redraw is not a
hidden retry; an *unrecorded* one would be. A fresh `run_id` or a re-issuance is
a new session and is counted as one.

> **This is prospective and rewrites nothing.** The 2026-09-04 ruling — that
> attempt 3 was *not* consumed because no provider resource was created — stands
> verbatim below as the rule attempt 3 actually ran under. Two rules, each with
> its scope written down, because a superseded rule that reads as current is how
> a spent attempt gets recovered by citation.

**The formal budget funds COMPLETE sessions, not cheap failures.** All eleven
paid C1 attempts so far aborted early and cost `$0.0786`, `$0.1013`, `$0.3482`,
`$0.6986`, `$0.3150`, `$0.3665`, `$0.4231`, `$0.6248`, `$1.0440`, `$0.1177` and
`$0.2783` — `$4.3961` for eleven. Budgeting the remaining balance at that rate
would be budgeting for failure modes we have already fixed. Every session is
funded at the full ceiling, so a run that reaches the 6 × 61.55-minute training
block and fails there — the expensive failure E8b actually hit twice, at step
110 and near step 900 — is affordable rather than a surprise.

**Why `$6.0000` of engineering.** The closed CUDA campaign cost `$0.0400` for
three subruns on an RTX 2000 Ada at `$0.24/h`. That is the cheap case and not
the planning case: setup time on this project has varied 30× for the same script
and image, and a single cold L40S setup at 150 minutes is `$2.72` before any
work happens. `$6.0000` funds roughly two worst-case L40S engineering sessions
plus many minutes-long cheap-card device tests, and is bounded well below one
formal attempt.

**Resource bounds.** At most **one active or potentially billing resource at a
time**, across engineering and formal work alike, and one create call per
**draw**. A draw that replaces an unusable host is permitted only after the
previous resource is **provider-confirmed not billing**; when that cannot be
confirmed the session **aborts** rather than create a second one. Formal attempts: secure L40S priced at `$1.09/h`, with
`--max-price` derived from the pricing record — a live quote above it refuses at
`$0` before any resource exists, and a different GPU class is a **repricing**
that this package does not authorize. Engineering: the cheapest card that can
actually observe the property, rate `≤ $1.10/h`, hard-terminate 120 min per
subrun, campaign record updated before the next paid action. Before every paid
action: settled + outstanding + active must leave a complete attempt plus its
teardown reserve inside the package ceiling.

**Autonomous formal retry inside the package requires ALL SIX** — not any one of
them:

1. it can be **confirmed** that no formal probe training has started;
2. the failure is an ordinary infrastructure failure whose cause is identified
   and has been addressed;
3. frozen science, the input contracts and the decision rule are unchanged;
4. the previous resource is **confirmed** no longer billing;
5. the package balance still funds a **complete** session plus its teardown —
   this is now the binding limit, since the attempt count was withdrawn;
6. the new session uses a new run identity and a complete, valid one-use
   authorization chain — never a consumed one.

In practice that covers a pre-provider gate refusal, provider acquisition
failure, any setup failure before `SETUP_DONE` including the CPU test gate, a
driver failure in stages **B–F** (attempt 9's class), and a launcher, watchdog,
relay or collection failure before training begins. A rerun never resets the
cumulative total, and a replacement resource does not get a fresh allocation.

**Stop, preserve everything, and report:**

* **the first probe has STARTED training — finished or not.** This is the line,
  and it is *not* "one or more probes trained": a probe that began and died
  mid-training is outside pre-authorized retry just as much as one that
  completed. This is a boundary on pre-approved retry, not a claim that starting
  training exposes an endpoint;
* **it cannot be confirmed whether training started.** Absence of confirmation
  is not confirmation of absence, and the unconfirmable case is never inferred
  to be retryable;
* **a real replay mismatch at stage D or E.** The frozen rule is: preserve the
  evidence and stop. It is a *result*; a larger budget does not license a
  re-roll;
* **an input-identity conflict that cannot be restored to its frozen binding**,
  or anything that would require changing the scientific protocol;
* **a completed run with a verdict — GO, NO-GO or INCONCLUSIVE.** All three end
  the package's formal sessions — the amendment removed a cap, not this line. `INCONCLUSIVE` is a result, not an engineering
  failure, and must never be re-run in pursuit of a GO: at Δ = 0 the design
  returns it 26% of the time and at Δ = SESOI 45% of the time, which is a
  property of the experiment accepted in advance;
* the package total, the per-session ceiling or the project cap is reached —
  the attempt count is no longer one of these; or a resource's billing state is
  unknown.

**Never**, under any budget: splicing probes across attempts, substituting a
seed, selectively retaining outputs, or changing arms, seeds, recipe, battery,
statistics or the behavioural vetoes.

**`NO DECISION` is recorded separately from `INCONCLUSIVE`, and is not bounded
by stage G.** How far a run got, whether its evidence was collected, and whether
teardown was confirmed are three facts recorded independently of whether a valid
scientific conclusion exists. An engineering failure is never written up as a
statistical `INCONCLUSIVE`.

> **THE ISSUER DECLARED PATHS THE MIGRATION HAD DELETED, 2026-09-11 —
> `$0.0000`, caught by a read-only pre-flight BEFORE the launcher was invoked,
> so no formal attempt was consumed.**
>
> `session_commit_gate` re-digests the authorization's `harness_source_files` at
> the session commit — literally `git show <commit>:<path>` for each one — and
> compares the result to `harness_source_digest`. Those two fields have to
> describe the same files.
>
> They stopped doing so at the initialization cutover. `harness_source_digest`
> became the **derived post-migration closure** (97 current paths) while
> `harness_source_files` stayed frozen at `C1_HARNESS_SOURCE_FILES_V1` — the
> pre-migration declaration, deliberately still pointing at
> `src/aadistill/autoinit/…`, which no longer exists. So the first authorization
> issued under the approved package declared 73 paths that are not in the tree,
> and **two gates refused**: `session_commit_and_lineage` with *"does not
> contain \['src/aadistill/autoinit/\_\_init\_\_.py', …]"*, and
> `bundle_staged_gate` with the same list after a real round-trip of the
> uploaded bundle.
>
> **Why no `$0` test caught it.** `session_commit_gate` is the one gate that
> reads this field, and it is in `ALWAYS_STRUCTURALLY_UNAVAILABLE` for the
> candidate sweep — it binds a *real issued* commit, which a scratch candidate
> does not have. So it had never run against a real issuance on the migrated
> tree. This is exactly the risk the exclusion list's own docstring names: an
> exclusion list is where a real gate goes to die quietly.
>
> **What found it instead.** A read-only pre-flight that calls the same thirteen
> gate functions with the same real authorization and the same real argument
> namespace, without invoking the launcher and without calling `open_c1_run`.
> Under this package an attempt is consumed at *invocation*, so the cheap check
> has to happen outside it. It cost nothing and saved one of three attempts.
>
> **The repair.** The issuer now declares the set its digest was computed over —
> `tuple(f["path"] for f in harness["files"])` — and `c1_harness_gate` compares
> the declaration against the live derived set rather than against the frozen
> constant, naming the pre-migration case explicitly so the message does not
> send a reader looking for another phase's grant. `C1_HARNESS_SOURCE_FILES_V1`
> and `c1_historical_harness_digest` are untouched: they are the record of what
> the completed attempts ran, and the guard that stops an old authorization
> being revalidated on this tree.
>
> **Five `$0` regressions close the gap**, and the pre-fix code fails all five:
> the declared set must equal the digested set; every declared path must exist
> at `HEAD` (which is what the gate actually does); the historical declaration
> must be refused *by name*; and the set the issuer really writes must be
> accepted, so the refusal is not refusing everything.
>
> **The first authorization is VOID and was never used.** It is preserved in git
> at `52d6b68`, was never passed to the launcher, created no resource and cost
> `$0.0000`. `0` of 3 formal attempts were used *as of that day* — a figure
> from before attempts 10–12 and from before the cap was withdrawn. The current
> count and the amendment are at the top of this file.

> **THE GOVERNANCE INPUT HAD NOWHERE LEGAL TO GO, 2026-09-11 — `$0.0000`, no
> pod, no GPU, no provider resource, no grant, no authorization.**
>
> A grant is an **input**, and it is the one governance artifact that must exist
> *before* the launch: the launch-bound sweep is taken on the final clean
> pre-authorization tree, and the authorization is issued from the grant. So it
> is committed first, and the run directory is where it belongs.
>
> `open_run` refused exactly that. Its occupancy rule — right for what it was
> written for — treats any file in an unrecorded run as the residue of a
> launcher that died before writing its manifest, which is precisely when those
> files are the only evidence left. It could not tell that case from a declared,
> expected input. Nine attempts had avoided the question by putting the grant in
> `logs/autoinit_c1_attempt<N>_grant.json`: nine flat files in the log root, each
> a per-attempt fact with no run to belong to.
>
> **What changed, and how little.** `open_run` takes `prepared`: role names
> written before the run opens. They are exempt from the occupancy rule **and
> from nothing else** — a recorded run is still refused, an undeclared file is
> still refused, a half-written evidence tree beside the grant is still refused
> and still named in the refusal. A prepared name that is not a declared role is
> itself an error, because an exemption for a path the run records no owner for
> is the ownership problem again under another name. No prepared/running state
> machine, no second run manager, no new directory: `grant` is one more role, in
> the area it was always going to be in, and `record_run` gives it an owner.
>
> **And the grant now has to be the right one.** The authorization has always
> recorded the grant it was issued from — path and content hash — and nothing
> ever looked at that reference again. `grant_provenance_gate` is the thirteenth
> pre-provider gate: the reference must resolve to **this run's**
> `governance/grant.json`, the file must exist, and it must still hash to the
> recorded value. An edited grant, a deleted one and *another attempt's* grant
> were previously indistinguishable from the right one, and there are nine
> structurally valid grants sitting in the log root to pick up by mistake.
> Because there is nowhere else a grant can now be and still pass, the flat file
> is unnecessary rather than merely discouraged.
>
> **Verified at `$0`, on CPU, which is the right environment for it.** This is
> file layout and lifecycle: no CUDA, no device, no model. Eight mutations were
> applied to the real implementation and each was caught by the case written for
> it — ignore `prepared`; exempt everything; drop the unknown-name check; stop
> declaring the grant prepared; and four separate defeats of the gate. The
> convention layer is covered against three disjoint stage vocabularies
> (Stage-0 collection, Stage-3 recovery, Stage-4 rollout) plus a
> whole-directory role, so the mechanism is not C1-shaped.
>
> **Frozen science is unmoved, and the diff says so rather than asserting it.**
> The C1 harness moves `f2789673` → `1d9c71ab`: 97 files before and after, none
> added, none removed, exactly two edited — `scripts/experiments/run_layout.py`
> and `scripts/pod/autoinit_c1_launch.py`. `executable_source` is **unchanged**
> at `ca1f1df5`. Of **395** non-harness preregistration leaf fields, 15 move:
> the document's own self-hash, `head_commit`, the gate count `12 → 13`, and the
> twelve entries of the derived gate-order list shifting by one position.
> **Scientific fields moved: 0** — seeds, both replay digests, the plan hash,
> both path hashes, the battery, the teacher, the scoring contract and the
> decision rule are identical. `src/aadistill` was not touched at all: the
> exemption belongs to the convention layer that names `logs/runs`, and the core
> still names no directory, no role and no experiment.
>
> **The readiness record is superseded by this change, as designed.** It binds
> the executable, the harness digest moved, so `pod_environment_gate` refuses
> until a new sweep is recorded. That is the gate working.

> **TWO ENGINEERING BLOCKERS, FOUND ON REVIEW AND REPAIRED, 2026-09-11 —
> `$0.0000`, no pod, no GPU, no provider resource, no grant, no authorization.**
>
> The previous entry ended by saying no engineering blocker remained. **That was
> wrong**, and the correction matters more than either repair: a tenth attempt
> launched on 2026-09-10 would have created a pod, completed setup, materialized
> its inputs and then died on a `NameError`.
>
> **1. `SessionRunner.run()` read an undefined `REPO`.** It built the driver's
> `JobSpec` with `workdir=REPO` — a module constant deleted when the image layout
> moved into `ExecutionCommands`, three lines above an `env` that had been
> converted correctly. It raises only after the provider resource exists, so the
> cheapest possible defect would have been found at the most expensive possible
> moment. Now `self.repo`, i.e. `spec.commands.checkout_root`: config-derived, no
> constant restored, no global injected by a launcher.
>
> *Why nothing caught it.* The removal was verified against the nineteen
> f-strings that build remote commands; this is a keyword argument. And every
> test that drives the real acquisition loop stubs `setup_on_draw` to a
> **failure** outcome, so `run()` returned before reaching the line. The success
> path had no execution coverage at all — it does now, under two deployment
> layouts that share no path component.
>
> The same CLASS is covered statically: `inventory.py` gains one rule using
> stdlib `symtable` — names read as module globals the module never binds. It
> reports **0** here and exactly `('run', 'REPO')` on the pre-fix bytes, which is
> asserted, so the guard is not merely quiet. Not a lint sweep.
>
> **2. A run could collect outputs it did not produce.** `open_c1_run` refused a
> colliding `run_id` under `logs/runs/` and said nothing about `--scr`, where the
> outputs actually accumulate. A fresh run id aimed at attempt 9's scratch,
> failing *before its driver started*, came home with attempt 9's driver
> evidence, status stream and artifact manifest recorded as its own — every role
> present, every file real, `verify_run_manifest` green.
>
> Output ownership is now explicit: claimed at open, required at closeout,
> decided by the run's **declared outputs** so a shared read-only cache neither
> claims nor blocks. A foreign claim is refused by name; an unclaimed directory
> holding this run's outputs is refused as *unknown* ownership. Nothing is
> deleted, nothing overwritten, and mtime is never consulted. The CUDA
> engineering launcher uses the same mechanism.
>
> **Frozen science is unmoved.** `session_runner.py` belongs to Phase B's frozen
> executable set, so the change is declared additively as **PHB-HA-010** with a
> git-derived numstat (`+9 −1`, `lines_removed: 1` — the real number, not a
> convenient zero). The C1 harness moves `bef1e52b` → `f2789673`: no file added,
> none removed, exactly the three edited; of **394** non-harness preregistration
> leaf fields, **two** move — `head_commit` and the document's own self-hash.
> **Scientific fields moved: 0.**
>
> **A RUN NOW HAS AN IDENTITY BEFORE IT RUNS, 2026-09-10 — `$0.0000`, no pod, no
> GPU, no provider resource, no grant, no authorization, no bundle.**
>
> `RunLayout`, `ArtifactSpec` and `build_run_manifest` had been in the tree since
> the Milestone-A migration with **no production caller at all**: every reference
> was a test. Meanwhile every C1 attempt wrote its session record to the flat
> `logs/experiments/phase_c1/autoinit_c1_session.json`, which the next attempt overwrote, and
> `logs/autoinit_c1_attempt9/` was assembled by hand afterwards. Three
> consequences, each of them observed rather than imagined:
>
> * the live record and the preserved copy are byte-identical duplicates of one
>   fact, and only the copy survives the next launch;
> * `CATALOG.md` described the live file as **attempt 5's** while it held attempt
>   9's — stale for four attempts, because a fact written twice goes stale on one
>   side;
> * `logs/runs/index.json` reported `runs_current: 0` while **three** real CUDA
>   stage-F subruns sat under `logs/runs/`, invisible to both discovery rules.
>
> **What changed.** `scripts/experiments/run_layout.py` is the application-layer
> convention — the only place `logs/runs` and the five areas `governance/`,
> `runtime/`, `evidence/`, `artifacts/`, `closeout/` are written down. The core
> still names no directory and no role. The C1 launcher now **requires**
> `--run-id`, has no `--out` to point elsewhere, snapshots the one-use governance
> artifacts at open, collects the small evidence at close and records
> `manifest.json`; the CUDA engineering launcher does the same through the same
> functions with a **disjoint** role vocabulary, which is what distinguishes a
> mechanism from C1's habits.
>
> **What did not change.** Attempts 1–9, the frozen Phase-A/B evidence and the
> three CUDA subrun directories are untouched — no file moved, renamed or
> deleted, and no manifest was back-filled for a run that never declared its own
> roles. Those three are now listed under `unrecorded` with their digests
> instead of being dropped. `logs/experiments/phase_c1/autoinit_c1_session.json` stays where it is,
> holding attempt 9, marked HISTORICAL.
>
> **Cost of the change.** The launcher is inside the C1 harness closure, so
> `c1_harness_digest` moves `64664a5f…` → `bef1e52b…` and the execution
> preregistration is re-emitted. Diffed rather than asserted: the harness gains
> exactly the **two** `run_layout` modules and edits **one** file, the launcher;
> nothing is removed. Of the **394** non-harness leaf fields, **two** move —
> `head_commit` and the document's own `preregistration_sha256`. **Scientific
> fields moved: 0.** The
> 2026-09-10 pod-environment record is a **diagnostic** bound to
> `d8e6896a`; it is not promoted, not rewritten, and a fresh launch-bound record
> is owed before any tenth attempt regardless.
>
> **MILESTONE-A MERGE-REVIEW CLOSURE, 2026-09-09 — `$0.0000`, no pod, no GPU, no
> provider resource, no grant, no authorization, no bundle.** The four merge
> blockers, closed on the same branch; the existing 43 commits are untouched.
>
> **`path_literals` reaches 0** and the ratchet's allow list is now empty, so
> `core_boundary_baseline.json` is a plain refusal rather than a ratchet. Every
> other rule was already at zero. What the list held was the core naming which
> log a phase's accounting lives in and which scripts the session runner
> executes; both come from the caller now — `ExecutionCommands` has no defaults,
> so **the runner cannot know a repository script name**.
>
> **Semantic policy is detected by shape, not keyword.** A new AST gate for the
> class the literal detectors cannot see: `SEED_SA` is an int, a capability list
> is a tuple of ordinary words, `CATASTROPHIC_V1` arrived as a dataclass
> default. Seven rules, each narrowed against the real false positives the first
> pass produced — 43 findings, most of them the mechanism's own vocabulary.
> Three were genuine and are fixed rather than baselined. **Semantic hardcode:
> 0.** `SuccessiveHalvingPlan` now requires its policy by keyword, proven with
> **two** callers, because one caller cannot distinguish "used what I passed"
> from "used the default that happens to match".
>
> **The snapshot stopped contradicting itself.** `phase_c.c1` said `TEN LABELS,
> NINE PAID / REPLAY MEASURED`; `phases.phase_c1`, four keys away, still said
> `NINE LABELS, EIGHT PAID / NEVER MEASURED`. The repair is mostly deletion —
> two copies of a fact are two chances to be wrong — and the new gate scans
> every string rather than pinning a key, since the key I would have trusted was
> right the whole time.
>
> **The CUDA check now runs the stage that failed.** The per-operator matrix
> could not have caught attempt 9 twice over: the failure was in the
> composition, and the matrix does not execute the treatment operator at all. It
> now builds the two-arm world, gates a real parent, runs the tail through
> `materialize_fixed_path_suffix`, and observes five placements rather than
> assuming them. Restore the old `stats_to` across a device split and it goes
> red where the L40S did. **Still NOT RUN on CUDA** — see the request above.
> *(Superseded 2026-09-10: it has since RUN and PASSED on real CUDA at execution
> SHA `7027a8f4`. See the status table at the top of this file.)*
>
> **Historical reuse stays REFUSED**, and the four conclusions that look
> contradictory are now derived in one place rather than reconciled from four
> documents: valid bytes, reconstruction under the historical contract,
> byte-identical relocation outputs across 570 samples, and a refused live
> reuse. Nothing was relaxed and no equivalence bypass exists.
>
> **The readiness contract left the runtime too.** Nine tuples of concrete test
> node ids — one session's expectation about which of its own tests skip on a
> pod — were the last thing `pod_environment.py` owned that was C1's. Neither
> `path_literals` nor the semantic gate looks at a node id, and the module was
> still refusing in C1's vocabulary. `evaluate_sweep` now takes a caller-declared
> `ReadinessGroups`, and the record's keys are DERIVED from the caller's group
> names, so C1 reproduces the existing schema exactly.
>
> **The shell dispatch table was a consumer nobody rewrote — and it was
> pod-fatal.** `autoinit_preflight_setup.sh`'s default `SESSION_KIND` branch
> loaded the bare `SpendAuthorization` and read `allows_phase_a`. §1 removed
> that property and made the bare primitive refuse without an `ActionPolicy`, so
> the branch would have **raised on a paid pod, after setup**. Every import
> rewriter here walks Python; this is embedded Python inside a heredoc. Four
> launchers were half-repaired the same way — three importing
> `PreflightAuthorization as SpendAuthorization`, correct behaviour under a name
> that says the opposite. The test that should have caught it had the literal
> string `SpendAuthorization` in it, so it went stale in the same commit; it now
> derives the narrow type from the launchers themselves.
>
> **Accounting:** Phase-B ledger amendments **PHB-HA-003**, **004** and **005**
> — three, because the closure continued after each was written, and one entry
> rewritten to cover later work would be a declaration whose numstat no longer
> matches the diff it names. Continuation B re-declared twice. The migration
> record runs to the branch tip over 52 commits. **Scientific fields moved: 0**
> of 94 examined in the C1 preregistration — seeds, battery, both path hashes,
> isolation plan, recipe, geometry and pricing all byte-identical; and the
> 570-sample scoring comparison, regenerated at this tree, is still identical
> per sample and in aggregate.
>

> **C1 attempt 9, 2026-09-07 — `$1.0440`, pod `8gtnsbigpgaz76`, 57.47 min,
> provider confirms gone.** Cumulative **`$267.8598`** of `$283.7600`, leaving
> **`$15.9002`**. Against the `$15.1475` per-attempt ceiling that is a reserve of
> **`$0.7527`**, so **exactly one** ceiling-sized attempt still fits (worst case
> `$283.0073` of `$283.7600`) — and no second one. Headroom is not permission.
> *(Those were the figures immediately after attempt 9. `$0.0400` of engineering
> spend has been booked since; the current cumulative is `$267.8998`, leaving
> `$15.8602` and a `$0.7127` reserve. The status table at the top of this file
> and `logs/BUDGET_LEDGER.md` are authoritative.)*
> *(Corrected 2026-09-08: this line previously said `$15.9002` "no longer covers
> a full `$15.1475` attempt", which contradicts its own subtraction. See the
> attempt-9 entry in `logs/BUDGET_LEDGER.md` and
> `tests/docs/test_budget_arithmetic.py`.)*
>
> **THE FIRST C1 SCIENTIFIC OBSERVATION, after every prior C1 label had failed
> before any scientific stage.** (The attempt counts and per-attempt accounting
> as of this entry are owned by `logs/BUDGET_LEDGER.md` and summarised once, at
> the top of this file — restating them in a dated entry is how a historical
> paragraph came to read as a current claim, and how two documents came to
> disagree about the same number.) Setup completed, the
> driver ran, and **stages B, C, D
> and E all PASSED**. Both fail-stop replay gates matched exactly:
>
> | step | operator | realized | expected | |
> | --- | --- | --- | --- | --- |
> | 2 | `width.global_pca_v0` | `eea90c91346a0745…` | `eea90c91346a0745…` | **MATCHED** |
> | 3 | `attention.weight_proxy_v0` | `c313d1b4081b9a3b…` | `c313d1b4081b9a3b…` | **MATCHED** |
>
> `all_pinned_digests_matched: true`, `n_pinned: 2`, path hash
> `8b0bb455bfba069f…`. **The frozen `fe9683` path reproduces its recorded
> identity** on an NVIDIA L40S under torch 2.11.0+cu128, transformers 5.13.1 and
> CUDA 12.8. That is exactly what C1's two gates exist to test, and it is now
> measured rather than assumed.
>
> **It says nothing about ATTENTION.** No probe was trained, none evaluated, no
> endpoint computed. `training_started: false`, `probes_trained: 0`,
> `probes_evaluated: 0`, `decision_ran: false`.
>
> **Stage F failed** — `RuntimeError: Expected all tensors to be on the same
> device, but found at least two devices, cuda:0 and cpu!` at
> `init/attention_stats.py:146`, inside the treatment operator
> `attention.activation_importance_v1`. The persistent `StatsCache` is
> host-resident **by design**; the weights are on `cuda:0`; `head_write_energy`
> multiplies the two without co-locating them. **That operator had never executed
> on a GPU** — every `$0` regression runs it on CPU, where host stats and host
> weights are trivially co-located, and no earlier attempt reached stage F.
> **REPAIRED at `$0` on 2026-09-08** — see the entry below. The repair has not
> been observed on a GPU, and nothing about it is a treatment result.
>
> **Three repairs were observed working on real hardware.** The launcher's
> neutral failure note refused to call this a replay mismatch. `watchdog
> detached` was logged **before** `created 8gtnsbigpgaz76`, so the backstop owned
> the resource from its first instant. And `provider_resource_created` and
> `one_use_grant_consumed` are both recorded — the grant is honestly marked
> consumed. Exactly one provider-create call, one watchdog, zero redraws.
>
> **Attempt 9's pod is gone.** Deleted, `provider_confirms_gone: true`, final
> state `TERMINATED`/not billing, watchdog ended `pod_gone` never over the hard
> limit, and an independent read-only poller recorded an empty inventory at
> `15:20:22Z`. *(This paragraph is about attempt 9. It read "Nothing is
> running" until 2026-09-11, when attempt 10 went live and that sentence --
> true of its own attempt, sitting in the current section -- became a
> present-tense claim that a pod was not billing while one was.)* The grant and authorization are **CONSUMED** and permit no retry
> and no replacement pod.

> **MILESTONE A — THE INITIALIZATION MIGRATION IS GREEN AND AWAITING MERGE
> REVIEW, 2026-09-09 — `$0.0000`, no pod, no GPU, no provider resource, no
> grant, no authorization, no bundle.** Branch
> **`migration/initialization-milestone-a`**, pushed. **`main` is untouched.**
>
> **Full suite: 3375 passed, 16 skipped, 0 failed, 0 errors** — from 213 failed
> / 64 errors when the cutover landed. **Pod-environment sweep: PASS** (3223
> passed, 0 failed, 0 error) with 1020 repo artifacts hidden and only the staged
> set visible.
>
> **The package cutover.** 59 modules into `src/aadistill/initialization/`
> (specs · adapters · calibration/statistics/device · transforms · operators ·
> planning); `aadistill.init` and `aadistill.autoinit` deleted; production
> imports of either **0**; no forwarding wrappers. **Dependency cycles 3 → 0**
> (it was 1 *before* the migration).
>
> **Core boundary: seven of eight rules at zero.** experiment-named modules 0,
> sha256 literals 0, repo-id literals 0, family access outside adapters 0,
> import-time instance registration 0, package cycles 0, core-imports-scripts 0.
> **path_literals 23** is the only remaining debt: the core naming which log a
> phase's accounting lives in, and which scripts the session runner executes.
> Both should be caller-injected; both are on the paid-pod path, so that change
> wants its own review rather than the tail of a migration.
>
> **Registration is explicit everywhere.** Adapters, operators, calibration
> profiles and dataset assets no longer register by import side effect —
> `get_adapter("qwen3")` used to resolve only if something had already imported
> the right module, which under `pytest-randomly` made whole test files fail at
> collection in some orders and not others. Order-independence is checked in
> **fresh subprocesses**, the only place it can be.
>
> **Two behaviour-preservation proofs, both mutation-checked:**
> * the recovery scorer relocation (v2 → v3) moved **no number**: 570 frozen
>   Phase-A samples re-scored through the pre- and post-migration trees are
>   byte-identical. `logs/architecture_scoring_equivalence.json`.
> * routing 28 model-family accesses out of `sandwich.py` changed **no
>   parameter**: every weight `init_student` produces is bit-identical across
>   three geometries. `logs/experiments/architecture/architecture_sandwich_equivalence.json`.
>
> **The C1 executable identity is derived, not listed** — 89 files, up from a
> hand-written 73 that was wrong in both directions. It had omitted
> `provider.py`, `remote.py` and `log_relay.py`, the modules that create, reach
> and tear down billed pods. Ten mutation tests require an edit to provider
> creation, remote execution, the log relay, the runner, the watchdog module or
> script, the setup shell, the execution config, the artifact contract or the
> preregistration to move the identity.
>
> **Historical evidence is unchanged and old launch paths fail closed.** No file
> under `logs/` was moved, renamed or rewritten. `C1_HARNESS_SOURCE_FILES_V1`
> still names pre-migration paths and refuses outright; the repointed
> declarations fail by digest mismatch. Both are exercised, not asserted.
> Accounting: `logs/migrations/initialization-core/v1/source-relocation.json`
> (eleven declarations) and Phase-B ledger amendment **PHB-HA-002** under the
> maintainer's P12 decision of 2026-09-08. **Scientific fields moved: 0** — the
> C1 preregistration's path hashes, seeds, battery and pricing are byte-identical.
>
> **⚠ ONE MAINTAINER DECISION IS OWED.** All eleven historical probes still
> reconstruct — bytes, artifact digests, seeds, battery, protocol hash — but
> **reuse is now refused**, for one reason: `scoring_contract_matches_live`. The
> live contract moved v2 → v3 with the scorer's path. The *numbers* are provably
> unchanged (570 samples), so this is the conservative identity rule, not a
> defect. Relaxing it — admitting a superseded contract because equivalence was
> demonstrated — would change what counts as a reusable scientific observation,
> which the P12 decision explicitly does not authorize. The refusal is recorded
> at both the verifier and the `$0` pre-provider gate.
>
> **Three latent defects the work surfaced**, each of which would have cost a
> paid session: a test that appended junk to a committed governance ledger
> whenever its "this will be refused" premise stopped holding; six Python
> imports embedded in the setup **shell** script, invisible to every AST
> rewriter; and an unanchored `datasets/` gitignore rule that silently refused
> to track a config the pod must read — caught by the sweep, at `$0`.
>
> **CUDA validation is ready and NOT RUN.**
> `scripts/validation/cuda_engineering_check.py` with
> `configs/validation/cuda_engineering.json`. No CUDA here → NOT RUN, exit 3,
> **no artifact written**. Its whole body is executed at `$0` by
> `tests/validation/`, which substitutes only the device gate. **No paid
> resource is requested or authorized.**
> *(Superseded 2026-09-10: authorized, run and PASSED. See the status table.)*

> **Architecture inventory, all-stage runtime layout, and the first experiment
> extraction, 2026-09-08 — `$0.0000`, no pod, no GPU, no provider resource, no
> grant, no authorization, no bundle.** Four of the maintainer's ten sections;
> the rest is measured and reported, not attempted.
>
> **The core is now measured, not argued about.**
> `scripts/architecture/inventory.py` reads `src/aadistill` from the SYNTAX
> TREE — literals carry the module/class/function that owns them, docstrings are
> excluded, comments are absent from the AST by construction. Two detector bugs
> were found and fixed before any gate was built on it: `logs/runs` and
> `governance/grant.json` were scoring as model repo ids (65 findings, all
> false), and a chain like `self_attn.q_proj.weight` was counted once per
> matching probe rather than once per access (roughly 4x inflation). The honest
> debt is in `configs/architecture/core_boundary_baseline.json`.
>
> **The gates are a ratchet, not a refusal.** A flat refusal would be red for the
> whole migration and stop carrying information. Instead: observed <= baseline,
> and every observed site already listed. A NEW violation fails at once; a FIXED
> violation ALSO fails, demanding the baseline drop in the same commit — which it
> did three times today, unprompted, as files were removed.
>
> **run-layout-v2 was compression-shaped.** It hard-coded `logs/runs`, fixed the
> roles to replay/treatment/training/evaluation/decision, and required
> `adapter`/`source_spec_hash`/`compression`/`operator_path` in every manifest. A
> dataset build has no operator path. `src/aadistill/runtime/run_layout.py`
> knows only that a run is `(experiment_id, run_id)` under a CALLER-supplied
> root with a caller-declared role -> path map; an `ArtifactSpec` supplied by the
> experiment says which roles are required. Ten stages — including one
> deliberately unnamed — build and verify through it with no source change.
>
> **The index was counting artifact roots.** It said 77 runs; `…attempt9` and
> `…attempt9_grant.json` are ONE attempt with two components. It is now **41
> logical runs over 77 components**, `phase_c1` = 10 (attempts 1-9 plus 3r).
> Nothing was moved, renamed or copied, and every component digest is checked
> against the tree.
>
> **No frozen set was disturbed.** All eight declarations were tested against the
> 20 changed files: no member changed, and the C1 harness digest is still
> `6ac5120f…`. No historical amendment is owed and no preregistration was
> regenerated. **Scientific fields moved = 0** — nothing scientific was touched.
>
> **What was NOT done, and why.** The initialization consolidation
> (`init` + `autoinit` -> `initialization`) and the production-writer integration
> are measured, not attempted: **164 files import those two packages across 54
> core modules**, and every frozen source-set declaration lists paths that would
> all move. A half-migrated tree with three packages and broken frozen sets is
> worse than today's, and forwarding wrappers are prohibited. Remaining core debt:
> 13 experiment-named modules, 23 path literals, 10 import-time registrations, 9
> repo ids, 8 digests, 4 adapter-boundary escapes, 1 package cycle
> (`calibration <-> operators`).

> **Stage-F device repair and closeout normalization, 2026-09-08 — `$0.0000`, no
> pod, no GPU, no provider resource, no grant, no authorization.**
>
> **Two device defects, one behind the other.** The first is the one attempt 9
> hit: `AttentionHeadStatsCollector.state()` returns a **host-resident** snapshot
> *by design* — that is the evidence form — and `apply()` handed it straight to
> `head_write_energy`, where it met `o_proj.weight` on `cuda:0`. The persistent
> cache POLICY was never wrong; what was missing was the **per-invocation working
> copy**. `apply()` now snapshots to the host, releases the collector's device
> accumulator, and builds exactly one working copy on `model_device(parent)` with
> the existing `aadistill.autoinit.device.stats_to`. No new cache abstraction, no
> model weights moved to CPU, and the cache is not made CUDA-resident.
>
> The second was latent behind the first and would have failed the very next
> line: `scores = torch.empty(num_heads, dtype=torch.float64)` defaults to CPU,
> so on a GPU the first `scores[h] = …` is a second cross-device store. It now
> derives placement from the tensor it meets (`device=w.device`), and
> `head_write_energy` **fails closed** on a mismatch rather than transferring —
> silently repairing it there would hide a caller that forgot its working copy.
> The whole score vector moves to the host once, after it is built.
>
> **The science did not move.** Same formula
> `score_h = <W_o,h^T W_o,h, M_h>_F / n_tokens`, same float64 accumulation, same
> per-GQA top-k, same ascending-index tie-break, same child slicing, same local
> metrics, same `impl_id`, `version` and `ATTENTION_STATS_SPEC`.
> `DEVICE_CONTRACT_VERSION` is unchanged.
> `tests/autoinit/test_attention_treatment_science_invariance.py` proves it by
> recomputing the treatment from its definition — every `a_h(t)` retained,
> brute force — rather than against a stored number from before the repair.
>
> **Why no `$0` gate caught it, and what now would.** Every prior regression ran
> the treatment operator on **one device**, where host stats and host weights are
> trivially co-located, and no attempt had ever reached stage F. Three new
> modules close that: the operator's device boundary
> (`test_attention_operator_device_split.py`), the science
> (`…_science_invariance.py`), and **stage F end to end**
> (`test_stage_f_treatment_integration.py`) — the real four-step path, the real
> `materialize_fixed_path_suffix`, the real registered operator, with the prefix
> proven unexecuted **from the checkpoints on disk** rather than from patched
> `execute` methods. Restoring the old operator makes that integration raise the
> attempt-9 error from inside the materializer. Five source mutations were
> applied and each turned the suite red.
>
> **Not observed on a GPU.** The repair is verified logically, at `$0`. Nothing
> here is ATTENTION evidence, and attempt 9 remains **NO DECISION after a
> pre-treatment infrastructure abort**.
> *(Superseded 2026-09-10 on the first clause only: the repair is now confirmed
> on real CUDA at execution SHA `7027a8f4`. It is still not ATTENTION evidence,
> and attempt 9 is still NO DECISION.)*
>
> **One item left open, deliberately.**
> `test_every_gate_but_the_commit_binding_passes_against_the_candidate` FAILS:
> `c1_harness_gate` compares the harness digest inside the scratch **candidate**
> authorization (`~/aad-scratch/sessions/c1-candidate/`, pinned at
> `4437074249d5…`) against the live tree, now `a3566eec79b3…`. The gate is right
> and the fixture is stale. Refreshing it means running the authorization
> **issuer**, which this session is prohibited from doing — so it is reported
> rather than fixed. It confers nothing either way: the launcher reads
> `logs/autoinit_c1_authorization.json`, gate 1 binds a real issued commit and
> gate 10 needs an uploaded bundle for it. The pod-like sweep is unaffected,
> because the simulated `$HOME` has no candidate and the test skips there.

> **Pre-Attempt-10 platform closure, 2026-09-08 — `$0.0000`, no pod, no GPU, no
> provider resource, no grant, no authorization, no bundle.** Three engineering
> closures the review required before grant review.
>
> **1. The candidate authorization is in-repo.** `test_c1_session_contract.py`
> drove the real pre-provider gates from `~/aad-scratch/sessions/c1-candidate/`,
> a hand-issued file outside the repository: it skipped under the empty `$HOME`
> the pod contract uses, and on the dev box it pinned a harness digest that went
> stale the moment the harness moved — so a CORRECT gate reported a false alarm.
> The payload derivation now lives in
> `src/aadistill/initialization/c1_authorization_payload.py`, pure (no git, no clock,
> no writing); the CLI issuer keeps the effects and calls the same builder, so a
> test candidate and a live authorization cannot diverge. `CANDIDATE` and the
> whole-test `skipif` are gone. **113 skips in the sweep, down from 114** — that
> single number is the closure.
>
> **2. The treatment path resolves through the adapter.** The statistics
> collector took `model.model.layers` and `layer.self_attn.o_proj`; the operator
> reached through `adapter.attention(block).q_proj`. Both now use role maps —
> `stream_out_projections()["attn_out"]`, `stream_in_projections()["q"]` — and
> the collector receives its hook modules rather than finding them.
> **`arch.py` and `adapters/qwen3.py` are untouched**, which matters because the
> mechanical inventory shows both sit inside two COMPLETED frozen sets; they
> already exposed everything needed. Geometry matrix: 4Q/2KV, 8Q/2KV, 12Q/3KV,
> MQA 8Q/1KV and 6Q/1KV, head_dim 4/8/16, 1–6 layers. An MHA reduction now
> refuses by NAME — each query head owns its KV head, so preserving KV heads is
> impossible — instead of being rejected by a modulo test that hid the reason.
>
> **3. run-layout-v2, future-only.** `logs/runs/<experiment_id>/<attempt_id>/`
> with manifest, `governance/`, `runtime/`, `evidence/`, `closeout/`. Generic:
> C1, Stage 4 and Stage 5 use the same `RunLayout`, and nothing branches on
> model, geometry, ratio, stage or attempt. **Nothing was moved or copied** —
> `logs/runs/index.json` registers **77 legacy-v1 references** with digests, so
> "byte-for-byte unchanged" is checkable. It also exposed a latent bug:
> `.gitignore` had a bare `runs/` that silently swallowed the whole evidence
> root, so every future manifest would have been written and never committed.
>
> **The science did not move.** 274 preregistration leaf fields, 6 moved,
> **scientific fields moved = 0**; file sets identical at 8/8 and 73/73, with
> only the two refactored files' hashes differing.
>
> **Two process failures of mine, both instructive.** I edited a test module and
> did not re-run it, so a text-scanning defect reached the sweep — the sweep
> caught it, which is what it is for. Then I killed a running sweep to reclaim
> disk; that outran its EXIT trap and left **1020 repo artifacts moved aside**
> with `git status` still clean. Restored by hand and confirmed with
> `verify_frozen_assets.py`. The rules: reclaim disk before a sweep, never kill
> one, and regenerate the skip audit and sweep after the last test edit.
>
> **GPU engineering validation was NOT run.** This box has no CUDA device
> (`nvidia-smi` absent, `torch 2.13.0+cpu`), so it needs a paid provider, and
> this session was told to stop and propose a contract rather than spend. The
> Stage-F device repair therefore remains verified logically, at `$0`, and has
> still never executed on an accelerator.
> *(Superseded 2026-09-10: the contract was authorized and the validation PASSED
> on real CUDA at execution SHA `7027a8f4`, for `$0.0400` across three subruns.)*

> **Post-provider ownership repair and the P12 record-rule split, 2026-09-07 —
> `$0.0000`, no pod, no GPU, no provider resource.** The second authorization
> review found one more material defect and, because repairing it collided with
> a frozen-set contract, approved a narrow amendment to how that contract is
> accounted for.
>
> **The defect.** `SessionRunner.create()` judged the returned `costPerHr`
> **before** registering the resource: a one-shot unconfirmed
> `remove pod <pid>` and `return False`, all above `self.pod_id = pid`.
> Everything downstream keys on `pod_id`, so that path had **no watchdog, no
> `teardown_now`, no `provider_confirms_gone`, no cost** — and nothing for
> `run_session`'s handler to tear down — while a pod billed. It recorded itself
> exactly like a `$0` pre-provider refusal, and the one-use grant would have
> been reported unconsumed after being consumed.
>
> **The boundary is the first non-empty pod id.** Registration is now
> unconditional and saved first — id, start epoch, actual price,
> `provider_resource_created`, `one_use_grant_consumed` — and the detached
> watchdog starts *there*, not after `create()` returns success, so a rejected
> pod is still under an independent hard-cap backstop. `launch_watchdog` is
> idempotent per pod id, so the accepted path gets exactly one; `run()` no
> longer starts it. Over-price is then a **post-provider consumed abort**: no
> setup, no driver, no retry, no redraw, canonical `teardown_now()` with
> confirmation polling, and `is_zero_dollar_pre_provider_refusal: false` written
> as a field. If confirmation fails, `provider_confirms_gone` stays false, the id
> and cost stay, and the watchdog remains the backstop.
>
> **Why this needed a maintainer.** `session_runner.py` sits in **five**
> hash-bound sets, and the repair removes lines Phase B ran — measured, `29/0`
> became `73/5` against the Phase-B baseline. It is genuinely **not additive**,
> and the sealed declaration says a non-additive change *"must fail the gate
> rather than be appended to this list"*. Declaring it additive would have been
> false; leaving it undeclared would have left an unexplained digest. The session
> stopped there and asked.
>
> **The approved split.** Historical accounting and launch compatibility are now
> different questions with different mechanisms:
>
> | question | mechanism | answer here |
> | --- | --- | --- |
> | is the drift explained? | `post_freeze.historical_accounted_for` over the new append-only ledger | **TRUE** |
> | may Phase B launch against this tree? | `post_freeze.accounted_for`, unchanged | **FALSE** |
>
> That pair is the required invariant, and it is asserted as a test.
> [`autoinit_phase_b_historical_amendments.json`](../../autoinit_phase_b_historical_amendments.json)
> is append-only, anchored by hash to the immutable preregistration and to the
> sealed v1 note, chained by entry hash, and it **confers nothing** — every entry
> asserts `launch_compatible_with_frozen_preregistration: false`, and the paid
> gate is asserted never to read it. Nothing in it is believed: the numstat, the
> patch hash, the per-file after-hashes and the sealed note's hash are all
> re-derived from git and the tree.
>
> **Both immutable Phase-B records are byte-identical to `bd4e5880`**, asserted
> by test. `record_phase_b_post_freeze.py` is **sealed read-only** — run once on
> 2026-09-07 it silently dropped four accumulated `history` entries and moved
> `c1` out of `pre_existing_unchanged`, making the gate it feeds refuse its own
> output. The continuation-B declaration, whose rule has no additive
> requirement, was **appended** to honestly.
>
> **Verification.** 19 ownership cases and 23 ledger cases. Six ownership
> mutations red — raw remove before registration, `pod_id` below the price check,
> confirmation suppressed, watchdog delayed, a second watchdog, a second create.
> Fourteen ledger falsifications refused, including the additive lie, a false
> patch hash, a broken chain and `launch_compatible: true`. Frozen science: 43
> rows, **MOVED = 0**.

> **Authorization review, 2026-09-07 — issuance NO-GO, one narrow `$0` repair.**
> The Attempt-9 grant and its first launch-bound record were reviewed at
> `origin/main 414a3fa` and accepted as genuine `$0` evidence. **Issuance was
> withheld**, on two material pre-authorization defects.
>
> **1. The launcher did not enforce the grant it was launched under.** The grant
> permits exactly one provider resource and says in as many words that there is
> no replacement pod — and `autoinit_c1_launch.py` defaulted to
> `--create-attempts 8` and `--host-draws 3`. A cold or endpoint-less first pod
> would have been deleted and a **second drawn**; a create failure would have
> slept 300 s and retried, seven more times, against the stock the grant says not
> to chase. Neither reads as a violation in a transcript — both look like
> ordinary resilience.
>
> Both values are now fixed at **1**, by argparse *type* and again at spec
> construction, so neither a command line nor a hand-built namespace can raise
> them. That makes `draw < host_draws` false by construction, so the
> cold/`no_endpoint` **redraw branch is unreachable** and every abort falls
> through to `teardown_now`; and `attempt < create_attempts` false, so there is
> exactly **one** provider-create invocation and **no sleep**. The same field
> bounds the zero-provider price READ, which becomes a single `$0` read that
> returns for review — it cannot create a resource or wait for stock.
>
> **`SessionRunner` is untouched.** Phase A, Phase B, both continuations and the
> preflight keep multi-draw acquisition, which is correct for them: they have no
> one-resource grant. 20 cases cover it, and three mutations were confirmed red —
> restoring `host_draws` to 3, restoring `create_attempts` to 8, and letting the
> cold branch redraw regardless of the budget.
>
> **2. The snapshot declared a grant and denied one.** `current_state.json` said
> the Attempt-9 grant was present and one-use, while `blocker`,
> `phase_c.c1.not_built` and `next_starting_point.the_ask` still read *"No C1
> grant exists"* / *"no grant"* / *"No pod, grant or authorization was created"*.
> Three fields had simply not been updated. All are corrected, and
> `tests/autoinit/test_state.py` now refuses a snapshot that declares a grant and
> denies one in any owned live field — verified by putting each of the three
> denials back and watching it fail.
>
> Those guards were first written with `pytest.skip`, and the skip-predicate
> audit refused to resolve them: a skip keyed on repository content is one more
> thing the pod/sweep comparison must account for. They are conditionals now, not
> skips, and the audit is back to **0 unaccounted, 0 stale, PASS**.
>
> **Issuance remains ABSENT** and is conditional on a **fresh** launch-bound
> review: this repair moves the executable, so the readiness record swept before
> it no longer describes this tree. Unchanged by the review: one issuance, one
> launch attempt, one provider resource, the consumption semantics, `$1.09/h`,
> `$15.1475`, `$283.7600`, and the frozen scientific scope in full. The label
> stays **Attempt 9** — nothing was issued and no resource was created, so there
> is nothing to retire and no Attempt 10.

> **Attempt-9 grant, 2026-09-07 — `$0.0000`, no pod, no GPU, no provider
> resource.** The four post-attempt-8 `$0` repairs were reviewed at `origin/main
> 5956a5fd` and **ACCEPTED**: the raw-calibration → operator-ready boundary;
> declared-device root loading with exhaustive parameter/buffer placement
> verification; Stage-F verified treatment suffix execution with its treatment
> record and operator budget fail-stop; and correct Stage-D/Stage-E
> ordinary-failure attribution with atomic Stage-E replay evidence. No further
> source repair, canary or rehearsal is authorized.
>
> **[`autoinit_c1_attempt9_grant.json`](../../autoinit_c1_attempt9_grant.json) is a
> GRANT, not an authorization.** It permits ONE issuance. It creates no pod,
> reserves no provider resource, stages no bundle and spends nothing. The
> canonical authorization artifact is untouched and still holds the CONSUMED
> attempt-8 issuance.
>
> | | |
> | --- | --- |
> | spend at approval | **`$266.8158`** of the unchanged **`$283.7600`** cap |
> | per-attempt hard ceiling | **`$15.1475`** |
> | worst case after one full attempt | **`$281.9633`**, leaving **`$1.7967`** |
> | accepted secure L40S price | **≤ `$1.09/h`**, `securePrice`, no chasing |
>
> That `$1.7967` reserve is **about a ninth of one attempt and is not
> authorization for another**. A pre-provider price, stock or gate refusal spends
> `$0.0000`, leaves the grant unconsumed, and permits no repricing and no
> different GPU product. Consumption is the creation of a **provider resource**,
> not the invocation of the launcher — and after it there is no retry and no
> replacement pod.
>
> **What is still absent:** the authorization, an Attempt-9 relay bundle (the one
> on record is attempt 8's, stale and for a superseded commit), the 12 `$0`
> pre-provider gates, the live price re-check, and the launch itself. **C1 remains
> SCIENTIFICALLY UNMEASURED** — no completed replay, no training, no evaluation,
> no `correct_overall`, no `usable_rollout`, no decision.

> **Stage-F and D/E repairs, 2026-09-07 — `$0.0000`, no pod, no grant, no
> authorization.** Two more defects closed after the stage-D work below. **C1 is
> still SCIENTIFICALLY UNMEASURED**: no completed replay, no treatment execution
> on real weights, no training, no evaluation, no `correct_overall`, no
> `usable_rollout`, no decision. Nothing here is ATTENTION evidence.
>
> *The stage-F repair was previously recorded only in `current_state.json`,
> `BUDGET_LEDGER.md` and `CATALOG.md`. This file is the human view and it did not
> carry it at all; that gap is closed here.*
>
> **3. Stage F could not run.** `self.parent` is the incumbent's **step-2**
> output — after DEPTH, FFN and WIDTH — and `self.arms["treatment"]` is the full
> **four**-step path, and `materialize_fixed_path` starts at step 0. Reproduced
> at `$0` before it was repaired:
>
> ```
> step 0 (depth.causal_kl_greedy_v1): not applicable to qwen3(…) —
> num_hidden_layers already at target (4)
> ```
>
> So it was not a wasted-prefix inefficiency; the stage raised. The repair is
> `materialize_fixed_path_suffix`, which takes the **unchanged** full spec, checks
> its hash against the frozen one and narrows only the executed index range —
> `StepResult.index` stays 3 and the checkpoint stays `03_attention`. A
> synthesized one-step spec was rejected as the fix: it would hash differently,
> and an arm's identity **is** its full frozen path. Eight premises are refused
> before any operator runs, plus checkpoint re-identification from disk, the
> loaded model's ArchSpec, and root placement. `TREATMENT_SUFFIX_START_INDEX` is
> `len(PREFIX_STEPS)`, derived, never written as `3`.
>
> The treatment arm now writes its own record, `c1_treatment_record.json`, and
> the success artifact spec **requires both arm records**. It is deliberately not
> called a replay: the treatment output was never pinned, so
> `output_digest_was_pre_pinned` is an explicit `false` rather than an absent
> field. Device verification became exhaustive at the same time — every parameter
> and buffer, refusing meta tensors, mixed types, mixed CUDA ordinals and a wrong
> exact ordinal, without moving the model. And the fixed path finally passes a
> deadline: `OperatorContext.deadline` had existed for a long time and the search
> passed one while the fixed path did not, so `depth.causal_kl_greedy_v1`'s
> per-candidate check was a no-op on the C1 path. `C1OperatorDeadline` invents no
> timeout and adds no second clock — it reads `usd()` and `soft_stop_usd`.
>
> **4. A stage-E failure was recorded against stage D.** `run()` walks
> `("DE", self.stage_de)` because one method owns **two** observable gates, and
> reported `self.fail(letter[0], …)`. So any ordinary exception in that method
> was written as `STAGE_FAILED:D` — including one after the parent digest had
> matched and `complete("D")` had recorded a PASS. It **overwrote** the passing
> entry while `stages_completed` still carried `replay_parent`, so the session's
> own evidence contradicted itself about which gate held. Reproduced before
> repair; the unrepaired driver printed `STAGE_START:E` then `STAGE_FAILED:D`.
>
> Attribution now follows explicit driver state (`active_gate`), advanced to `E`
> at the same point `STAGE_START:E` is emitted — never parsed from the status
> log. And stage E's evidence is atomic: `complete("E")` used to fire from
> `on_step` **before** the replay record was written, so a write failure could
> leave `STAGE_PASSED:E` with no record to show. E now passes only after the
> record is written **and reads back** — schema, path hash, both pinned steps'
> expected and realized digests, `all_pinned_digests_matched`. If that fails, D
> keeps its pass, E fails as ordinary infrastructure, and nothing claims a
> mismatch, because both digests did match in memory.
>
> | | ordinary failure in the incumbent step | complete successful replay |
> | --- | --- | --- |
> | markers | `START:D → PASSED:D → START:E → FAILED:E → C1_FAILED` | `START:D → PASSED:D → START:E → PASSED:E → ALL_DONE` |
>
> **Verification.** Full suite **3067 passed / 16 skipped**, the only failures
> being provenance records regenerated in the following commits. Fourteen
> mutations of the production lines were each confirmed to turn the tests red —
> eleven for the suffix and device work, and for this one: restoring
> `self.fail(letter[0])`, moving `complete("E")` before the record, and neutering
> the readback. Frozen science did not move: a 43-row table comparing
> live-computed values against `git show eedd8ee:` reported **MOVED = 0**, and
> the preregistration diff reports **scientific fields moved = 0** for both
> rewrites. Neither expected replay digest was touched.

> **Attempt-8 stage-D repair, 2026-09-07 — `$0.0000`, no pod, no grant, no
> authorization.** Three defects closed, none scientific. **C1 is still
> SCIENTIFICALLY UNMEASURED**: no completed parent replay, no incumbent replay,
> no recovery training, no confirmation generation, no `correct_overall`, no
> `usable_rollout`, no Stage-I decision. Nothing here is ATTENTION evidence.
>
> **1. The calibration preparation boundary was missing from the fixed path.**
> Confirmed from source, not assumed. A materialized mixture stores each item's
> tokens under **`ids`** — that is what `mixture_content_sha256` hashes and what
> the pinned `d65c1f40…` identity is defined over — while `depth.py:176`,
> `ffn.py:85`, `width.py:96` and `attention_activation.py:177` all read
> `item["input_ids"]`. The conversion existed only in
> `scripts/autoinit/phase_a_search.as_operator_items`, a **script**, so the
> search had it and `fixed_path` did not.
>
> [`calibration_items.py`](../../../src/aadistill/initialization/calibration/items.py) is now
> the single boundary, and `materialize_fixed_path` runs **both**
> profile-resolved and caller-supplied items through it — so no operator gets an
> `ids` fallback of its own, and a caller cannot route around the contract by
> passing items itself. Raw metadata and raw `ids` are preserved untouched; only
> `input_ids` is added. An item carrying both must agree token for token.
>
> **No frozen calibration asset was modified.** `resolve()` still returns raw
> evidence, asserted directly against both real mixtures, and neither items file,
> manifest, profile hash, content hash nor token sequence moved.
> `as_operator_items` is inside three CLOSED preregistrations and is therefore
> **not** edited to remove the duplication; a test requires the two to agree
> token for token on the real mixture instead, and fails at `$0` if they drift.
>
> **2. Stage D's root was loaded on the CPU under a `cuda` declaration.** Found
> by auditing beside the first defect. `build_arm_specs(workdir_device="cuda")`
> declares the device, and the loader was a bare
> `AutoModelForCausalLM.from_pretrained(...).eval()` with no transfer — while
> `depth.apply` reads `model_device(model)`, the fact rather than the intent. The
> entire parent replay would have executed on the host CPU inside a paid GPU
> hour, and nothing would have said so. Both roots now load through the adapter
> that already owns path/dtype/device, on their own arm's declared device (stage
> F's second literal `"cuda"` is derived too), and `materialize_fixed_path`
> refuses a root that is not on `spec.device` **before the first operator runs**,
> reading weights rather than `ctx.device`.
>
> **3. The launcher narrated a replay mismatch it had not observed.**
> `SessionRunner` prints `failure_note` for **any** marker in `markers.failure`;
> it is one constant string chosen before the run. On attempt 8 it announced that
> the frozen path *"did not reproduce its recorded digest"* against a `KeyError`,
> when nothing had been compared — a false claim about exactly the property C1
> exists to test. The note is now neutral. The explicit path is untouched:
> `C1Driver.replay_mismatch` writes the evidence, reads it back, and only then
> emits `MARKER:C1_REPLAY_MISMATCH`, and it remains the **only** emitter.
>
> **Why no `$0` gate caught any of it.** Every `test_fixed_path` case passes
> `calibration_items=`; `conftest.make_items()` builds items already carrying
> `input_ids`; `test_c1_driver_execution` monkeypatches `stage_de` whole. The
> resolve-a-real-profile branch and both root loaders had **never executed**. The
> new tests refuse the override, build a REAL materialized profile from temporary
> JSONL through the production hash rules, and run DEPTH → FFN → WIDTH →
> ATTENTION on a tiny Qwen model — recording from inside `execute` what the
> operators actually received. **43 new cases; nine mutations of the production
> lines were each confirmed to turn them red**, including restoring attempt 8's
> exact launcher sentence and re-inserting the raw-items line, which reproduces
> `KeyError: 'input_ids'` verbatim.
>
> **Both expected replay digests are unchanged and deliberately not
> "repaired".** The working runtime has never materialized this path. If it now
> produces a different digest, that is a REAL mismatch and must stop before
> training; pre-adjusting an expected value to match new code is the one edit
> that would make the gate meaningless.
>
> **Frozen science did not move.** Only the C1 harness: `85329e7e701b…` →
> `d342b499238d…`, 72 → 73 files. By set, the delta is exactly the five files
> touched — one added, four changed, none removed. Phase A, Phase B and
> continuation-B source sets contain none of them, so **no post-freeze
> declaration is owed**. The preregistration was rewritten for executable
> identity only (`1cdc49f3e847` → `a7ba9bbcc764`): a field-level diff shows
> **scientific fields moved = 0**, with the C0 protocol, session contract, both
> path hashes, arms, seeds, ATTENTION definition, recovery recipe, both replay
> digests, battery, generation protocol, scoring and decision rule byte-identical.

> **CPU-test parity, 2026-09-06 — `$0.0000`, no pod, no grant.** The strict
> skip-set comparison added after attempt 5 was correct machinery pointed at two
> DIFFERENT machines: a CPU diagnostic with an empty HF cache against an L40S
> with the pinned teacher downloaded. It would have refused a HEALTHY pod. The
> one GPU predicate in the selected suite RUNS without CUDA and SKIPS with it —
> the opposite decision there — and the tokenizer cases skip here on an empty
> cache and would have run there.
>
> The comparison is kept. What changed is that both sides now run pytest under
> ONE declared, **command-scoped** environment,
> `aadistill.autoinit.cpu_test_env`: `CUDA_VISIBLE_DEVICES=""`, a fresh empty
> `HOME`, an isolated empty `HF_HOME`/`HF_HUB_CACHE`, and `HUGGINGFACE_HUB_CACHE`
> / `HF_DATASETS_CACHE` / `TRANSFORMERS_CACHE` / `XDG_CACHE_HOME` cleared. The
> real `HF_TOKEN` is kept — the cache is what differs, never the credential, and
> `AAD_SYNTHETIC_HF_TOKEN` is never set on a pod. Nothing is exported over setup:
> the pinned install, the teacher download and the CUDA proof still run for real,
> and CUDA plus the teacher cache are re-asserted immediately after the gate.
>
> **The fresh `HOME` had to be made to matter.** Host-local stores were
> hardcoded as `/home/ecs-user/aad-artifacts`, which is immune to it — so those
> cases ran in the diagnostic and skipped on the pod. They now resolve through
> `Path.home()`, the pattern `verify_c1_scoring_equivalence.EVIDENCE_ROOTS`
> already used. On the real dev box this resolves identically.
>
> `strict_cpu_test_parity_ready = **PASS**`: 97 predicates across 144 selected
> modules — 50 same-on-pod, 29 normalized by the contract, 9 hidden in both, 9
> guaranteed dependencies, **0 unresolved**. Each of the nine optional
> dependencies names what guarantees it on both machines; "the pod's image is not
> the dev box's venv" is explicitly rejected as evidence.
>
> **One claim cannot be executed at `$0`:** that `CUDA_VISIBLE_DEVICES=""` hides
> an L40S. A CPU dev box cannot prove it. The contract is applied and the
> decision is proven stable here; the GPU half is first exercised on the next pod.

> **Attempt 8, 2026-09-06 — the driver RAN; stage D crashed.** Pod
> `fbuggw0x9efqsz`, created at `$1.09/h`, deleted at 34.39 min for **`$0.6248`**,
> provider confirms gone. Cumulative **`$266.8158`** of `$283.7600`, leaving
> `$16.9442`. **No scientific stage completed — no finished replay, no training,
> no evaluation, no decision.**
>
> **The first C1 attempt whose driver executed a stage.** The CLI-seam repair
> held: `MARKER:DRIVER_START` fired, **stage B passed** (teacher
> `768f209d9ea8`, 3 shards) and **stage C passed** (registered
> `attention.activation_importance_v1` `1171f3b791e2`). The CPU gate passed again
> at 1352 s with the strict skip-set comparison silent — a second consecutive
> exact sweep/pod agreement.
>
> **Then stage D crashed**, with a full traceback preserved:
> `KeyError: 'input_ids'` at `operators/depth.py:176`, reached through
> `stage_de` → `materialize_fixed_path` → `depth.apply`, after 398 weight shards
> had loaded.
>
> **The launcher's own summary line is WRONG and is corrected in the record.** It
> printed *"C1_REPLAY_MISMATCH … the frozen path did not reproduce its recorded
> digest"* — a canned sentence it emits for any blocking-stage failure. **No
> digest was compared.** The stage raised before any replay digest could be
> computed, so the frozen parent path has neither reproduced nor failed to
> reproduce its recorded value. Recording a mismatch would put a false claim
> about reproducibility into the record for exactly the property C1 exists to
> test. The driver's own `c1_evidence.json` is the artifact; the launcher line is
> not.
>
> **Open, and not answered here:** whether the calibration items are built
> without `input_ids` on this path, or `depth.apply` reads a key its supplier
> never promised.

> **Attempt-8 grant CONSUMED, 2026-09-06 — a provider resource was created.**
> A NEW one-use grant ([`autoinit_c1_attempt8_grant.json`](../../autoinit_c1_attempt8_grant.json)).
> Attempt 7's grant and authorization remain CONSUMED and are not reused.
> Spend stays `$266.1910` of `$283.7600` with `$17.5690` uncommitted; one
> full `$15.1475` ceiling fits, leaving about `$2.4215` — **which is not
> authorization for another attempt**, and an early termination still
> requires a completely new review. This commit is the frozen
> pre-authorization tree: after it, only the readiness record and the
> authorization artifact may move.
>
> **Attempt-7 postmortem repair, 2026-09-06 — `$0.0000`, no pod, no grant.**
> One production line: `--stage all` is REMOVED from
> `autoinit_c1_launch.driver_command`. The driver's parser is untouched and
> still has no `--stage` — `run()` already executes the whole fixed sequence
> B → C → DE → F → G → H → I, and adding the flag even as a no-op would
> create a stage-selection surface that partial C1 execution is not
> authorized to have.
>
> **The seam now has the regression it never had.**
> `tests/pod/test_launcher_driver_cli_seam.py` calls the real
> `driver_command`, tokenizes with `shlex.split`, and hands every remaining
> token to the real `autoinit_c1_driver.build_parser()`. It never restates
> the driver's option list — the parser is the authority — and an AST scan
> enforces that. Re-injecting attempt 7's exact `--stage all` argv is
> required to raise `SystemExit(2)`. Restoring the flag in production makes
> three of the five cases fail, so this would have caught attempt 7 at `$0`.
>
> **No gate for the CLI seam.** `pod_environment_gate` binds the complete
> launch-bound sweep to the final executable tree, so a passing sweep that
> contains this module IS the `$0` pre-provider evidence for the seam.
>
> *(Written 2026-09-06 as "no thirteenth gate — the count stays at 12". The
> claim about the seam is unchanged and still true; the count is not. A
> thirteenth gate was added on 2026-09-11 for an unrelated reason —
> `grant_provenance_gate` — and the test enforcing this paragraph asserted the
> total rather than its actual subject, so it went red while nothing it was
> about had changed. It now asserts that no gate claims the driver CLI seam.)*
>
> **Attempt 7 itself is unchanged:** CONFIRMED launcher→driver CLI mismatch,
> replay NOT REACHED, training 0, evaluation 0, score 0, decision 0.

> **Attempt 7, 2026-09-06 — the CPU gate FELL, then argparse.** Pod
> `gfd8buh5tr51qb`, created at `$1.09/h`, deleted at 23.29 min for **`$0.4231`**,
> provider confirms gone. Cumulative **`$266.1910`** of `$283.7600`, leaving
> `$17.5690`. **No scientific stage ran — no replay, no training, no evaluation,
> no decision.**
>
> **The first C1 attempt ever to clear the pod CPU test gate.**
> `MARKER:TESTS_OK:861s`, then `AUTHORIZATION_OK`, then `SETUP_DONE`. Attempts
> 3R, 4, 5 and 6 all died there. The complete marker sequence survived too,
> relayed off the pod while it ran rather than glimpsed through a `tail -40`
> window: `ENV_READY → REPO_READY → ASSETS_STAGED → TRAIN_ENV → ASSETS_READY →
> VLLM_READY → TEACHER_READY → ROPE_OK → TESTS_OK → AUTHORIZATION_OK →
> SETUP_DONE`.
>
> **All four attempt-6 repairs are validated on real hardware.** The strict
> skip-set comparison is fail-closed on any difference and **did not fire**, so
> the pod's complete skip set equalled the launch-bound sweep's 114 nodeids at
> `ee74f9e3dd305c5d` — the first exact sweep/pod agreement in this project. The
> `AAD_C1_CPU_TEST_SCOPE` skip behaved identically on a pod, the explicit
> `PODSIM_PYTHON` held where there is no repo venv, and CUDA and the teacher
> cache were restored after the isolated scope.
>
> **Then the driver never ran.** Root cause **CONFIRMED**, quoted from its own
> relayed stderr rather than attributed:
> `autoinit_c1_driver.py: error: unrecognized arguments: --stage all`.
> `autoinit_c1_launch.py:701` builds that flag; the C1 driver's parser defines
> seven options and no `--stage`. argparse exits 2 before a line of it runs.
>
> **No `$0` gate could have caught it, and that is the finding.** Every
> pre-provider gate checks the LAUNCHER's inputs — its own argument namespace,
> the harness digest, the authorization, the bundle, the staged view, the
> readiness record. Nothing parses the string the launcher hands the DRIVER with
> the driver's own parser. This is the device-canary failure one level out: that
> one produced `missing_arguments(args)` for the runner's namespace, and the
> launcher→driver seam was left without the equivalent. **Unrepaired.**

> **Attempt-7 grant CONSUMED, 2026-09-06 — a provider resource was created.**
> A NEW one-use grant ([`autoinit_c1_attempt7_grant.json`](../../autoinit_c1_attempt7_grant.json)).
> Attempt 6's grant and authorization remain CONSUMED and are not reused. Spend
> stays `$265.7679` of `$283.7600` with `$17.9921` uncommitted; one full
> `$15.1475` ceiling fits, leaving about `$2.8446` — **which is not
> authorization for another run**, and an early termination still requires a
> completely new review. This commit is the frozen pre-authorization tree:
> after it, only the readiness record and the authorization artifact may move.
>
> **Attempt-6 postmortem repair, 2026-09-06 — `$0.0000`, no pod, no grant.**
> Four repairs, none scientific. The strict complete skip-set comparison is
> KEPT: it worked, and it refused three real divergences.
>
> 1. **Measurement circularity, resolved by scope.** `record_pod_environment.py`
>    moves the readiness record aside while producing its replacement, so the two
>    tests that CONSUME that record cannot run inside the sweep that makes it.
>    The shared CPU-test contract now carries `AAD_C1_CPU_TEST_SCOPE`, set
>    identically by the simulator and the pod, and only those two nodeids consult
>    it. **It is not a simulator marker** — a simulator flag is set by one
>    machine, this by both, and the audit checks it strictly ahead of
>    `simulator_marker` so the two can never be confused. Gate 12 still runs
>    `verify_record`, issued-session lineage still checks the committed record,
>    and an ordinary `pytest tests/` run still executes both.
> 2. **`REPO_LAYOUT` asks only portable questions now.** Its trailing
>    host-absence `pytest.skip` is gone: a pod is not required to hold
>    `/home/ecs-user/aad-artifacts`, and that legitimate absence must not turn a
>    portable contract test from PASS into SKIP. The declaration and
>    relative-path assertions are untouched, and host-side existence stays owned
>    by `test_storage_inventory.py`. It passes with the roots present AND absent.
> 3. **The simulator's interpreter is an input.** `PODSIM_PYTHON` is passed by
>    the recorder and by the tests as their own `sys.executable`; the repo venv
>    is used only if it exists; otherwise it REFUSES. The ambient
>    `command -v python3` fallback is gone from executable code — a pod has no
>    repo venv, so the pod always took a fallback the dev box never exercised.
>    The marker is unset before `PODSIM_CMD` so it cannot leak into a nested run.
> 4. **Failure detail survives.** Every failure and error now carries its
>    message, type and traceback into `pytest_outcomes.json`, one `WHY` line
>    reaches the launcher's `tail -40` window, and the raw `pytest.log` and
>    `pytest_junit.xml` are pulled off the pod too, so a parser bug can never
>    again be the only surviving evidence.
>
> **Attempt-6 root cause remains ATTRIBUTED to interpreter resolution; the
> original mechanism is UNPROVEN because failure tracebacks were not preserved.**
> All 18 nodeids pass under the explicit-interpreter path, which is evidence the
> repair is right — not evidence the diagnosis was.

> **Attempt 6, 2026-09-06 — LAUNCHED, ABORTED at the pod CPU test gate.** Pod
> `n71opk7lv4fhzf`, created at `$1.09/h`, deleted at 20.17 min for **`$0.3665`**,
> provider confirms gone. Cumulative **`$265.7679`** of `$283.7600`, leaving
> `$17.9921`. **No
> scientific stage ran — no replay, no training, no evaluation, no decision.**
>
> **The strict skip-set comparison did exactly its job, on its first real L40S.**
> Pod `2791 passed / 113 skipped / 18 failed` against the sweep's
> `2808 / 114 / 0` on the same 2922 selected tests. It named all three
> divergences and refused at `$0.37` rather than after six trainings:
>
> 1. two tests the sweep skips **only because the recorder moves its own record
>    aside** so a sweep cannot certify itself — so they are absent here and
>    present on any pod. Structural: no classification can make the two machines
>    agree.
> 2. one test that skips on a pod because `REPO_LAYOUT.md` names
>    `/home/ecs-user/aad-artifacts/` as an **absolute literal in Markdown**,
>    which a fresh `HOME` cannot neutralize. **My error**: the registry gave the
>    class `devbox_only_artifact` ONE parity claim and asserted it for every
>    member. That is the excuse list the registry exists to prevent.
> 3. 18 failures, all in the single module the parity repair made shell out to
>    `cpu_test_env_args.py` through a `.venv` path a pod does not have.
>    **Attributed, not proven** — the outcomes file keeps skip reasons but only
>    failure nodeids.
>
> **What the money bought.** `setup_failure_files` pulled `pytest_outcomes.json`
> off the pod before teardown — 36,728 characters naming all 113 skips with
> reasons and all 18 failures, where attempt 5's list died with the pod. The
> CPU-test isolation did not leak: after the gate, on the real L40S, CUDA was
> available again and the teacher cache was intact. And the GPU predicate is in
> neither divergence list, so `CUDA_VISIBLE_DEVICES=""` **did** hide an L40S —
> the one claim that could not be tested on a CPU box.

> **Attempt-6 grant CONSUMED, 2026-09-06 — a provider resource was created.**
> A NEW one-use grant ([`autoinit_c1_attempt6_grant.json`](../../autoinit_c1_attempt6_grant.json)).
> Attempt 5's grant and authorization remain CONSUMED and are not reused. Spend
> stays `$265.4014` of `$283.7600`; one full `$15.1475` ceiling still fits,
> leaving about `$3.2111` — **which is not authorization for another attempt**,
> and an early termination still requires a completely new review. This commit
> is the frozen pre-authorization tree: after it, only the readiness record and
> the authorization artifact may move.
>
> **Attempt 5 remains an INFRASTRUCTURE ABORT with ZERO C1 measurement**, and is
> not reinterpreted as ATTENTION evidence by anything here.

> **Postmortem repair, 2026-09-06 — `$0.0000`, no pod, no grant.** Three things
> closed, none of them scientific:
>
> 1. **The two guards are repaired by their real premise.** They no longer read
>    `AAD_SYNTHETIC_HF_TOKEN`. `sources_on_disk` and `undeclared_in_destination`
>    ask the filesystem what it holds, so the dev box runs them, and the pod and
>    the simulation skip them for the same stated reason. Three-environment
>    regressions execute the actual test functions against pod-shaped and
>    simulation-shaped roots; four mutations were confirmed caught.
> 2. **The CLASS is closed, not just the incident.**
>    [`c1_skip_predicate_audit.json`](skip_predicate_audit.json) walks all
>    **95** skip predicates in the C1-selected suite and resolves each path
>    premise against the git index and the SetupManifest — the two mechanisms
>    that actually put a file on a pod — rather than guessing from keywords. **45**
>    that cannot be resolved carry an explicit classification; **0** unaccounted,
>    **0** stale. A simulator-keyed premise anywhere is now a standing failure.
> 3. **The skip set is evidence.** The pod gate writes JUnit, and
>    `summarize_pytest_outcomes.py` names every FAILED, ERROR and SKIPPED nodeid
>    with reasons, digests the skip set, and prints the exact set difference
>    against the launch-bound sweep. It is **fail-closed** on `expected-skip-but-ran`
>    and `unexpected pod-only skip`. `SessionSpec.setup_failure_files` pulls the
>    summary off the pod before teardown, because a setup abort never reaches
>    artifact collection and the launcher's window is `tail -40` — which is why
>    attempt 5's 99 skip identities died with the pod.
>
> **The strict comparison has never run on a pod.** Its stability is unproven,
> and it could itself abort attempt 6 at setup cost on a benign difference. That
> is deliberate — a named `$0.30` abort beats six probes trained under an
> unnamed environment difference — but it is a reviewer's call, not mine.
>
> **Verification after the repair.** The full dev-box suite is **2942 passed /
> 16 skipped / 0 failed** with nothing excluded, including
> `test_phase_b_stage1_executes.py`, which had been failing on `ENOSPC` rather
> than on any defect. One complete C1 manifest-derived diagnostic sweep is
> **PASS — 2802 passed / 101 skipped / 0 failed / 0 error**, with all 101 skip
> nodeids and reasons recorded, skip-set digest `bf53c0697a8a387c…`, the 15
> readiness-owned groups exact, zero unexpected environment skips, leaf
> transport 5/5, the staged battery role PASSED rather than vacuously skipped,
> and the staged view unchanged at `9ef2356ee807…`. Renderer parity is 7/7 and
> battery isolation passes all five roles with zero collisions.
>
> Repairing the harness moved its digest (`8b4dd4c0…` → `e3fb53b7…`, 70 files),
> so the preregistration was rewritten for executable identity only and the
> Phase-B and continuation-B post-freeze declarations were updated additively
> (65 lines added, 0 removed). A **launch-bound** sweep is still owed before any
> launch; this one is diagnostic.

> **Attempt 5, 2026-09-05.** The one-use grant was approved and consumed. All
> **12** pre-provider gates passed on the launcher's own fresh run, the bundle
> round-tripped, pod `e2ghwfuerdrvni` was created at `$1.09/h`, and setup emitted
> `VLLM_READY → TEACHER_READY → ROPE_OK` before dying at the CPU test gate with
> **2 failed / 2769 passed / 99 skipped**. Those three are the ONLY markers the
> evidence carries: the launcher echoes back a failed setup's tail, so earlier
> markers are neither confirmed nor denied by this record. `$0.3150`, 17.34 min, pod deleted, provider confirms gone.
> **No scientific stage ran — no replay, no training, no evaluation, no
> decision.** Cumulative `$265.4014` of `$283.7600`.
>
> **Attempt 4's cause did not recur.** The manifest-derived positive staged view
> held: the pod's staging matched contract `9ef2356ee807`, and the 49 extra skips
> and 6 failures the generic `HIDDEN_PATHS` complement produced are gone.
>
> **The new cause is a skip guard, not the contract.** Both failures are
> `tests/autoinit/test_staging_contract.py` cases that assert facts about files
> which exist only on the dev box — that `corpus_v2`/`battery_v2` appear in the
> HIDDEN set, and that the checkpoint directory holds an undeclared file. A pod
> has neither, because they were never transferred. They are *correctly*
> classified as dev-box-only — `pod_environment.DEVBOX_ONLY_NODEIDS` names these
> two exact nodeids and the sweep recorded `devbox_only_skipped_as_expected:
> true` — but the skip is produced by
> `skipif(os.environ.get("AAD_SYNTHETIC_HF_TOKEN"))`, a flag **the simulator sets
> and the pod does not**. The predicate encodes *"am I inside the simulator?"*
> when the property it needed is *"does this machine hold the unstaged
> artifacts?"*.
>
> **The pod is the environment that breaks that equivalence.** It is
> behaviourally identical to the simulation — no unstaged artifacts — while
> carrying none of the simulation's markers, so a simulator-keyed guard is
> exactly inverted there: it skips where the assertion would hold and runs where
> it cannot. Gate 12 certified the skip as expected because the skip did occur;
> what it could not ask is *why*.
>
> **One divergence is UNEXPLAINED.** If only those two moved skip→fail the pod's
> skip count would be 98; it is 99, and the pod passed one fewer test than the
> sweep. One test that PASSED in the sweep SKIPPED on the pod. The pod
> diagnostics grep `^FAILED|^ERROR` only, so a skip divergence is invisible
> unless it also fails — that list died with the pod. It did not cause this abort
> and must not be assumed benign.
>
> Full enumeration and root cause:
> [`autoinit_c1_attempt5/test_gate_failures.json`](../../autoinit_c1_attempt5/test_gate_failures.json).
> **Nothing is repaired** — this closeout is metadata only.

