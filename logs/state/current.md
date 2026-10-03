# Current state

**Updated:** 2026-10-03. The human view. Every number here has an owner named
beside it, and this file restates none of them from memory — a second
hand-maintained copy of a cost or a status is how two documents come to
disagree.

Start at [`README.md`](../README.md) if you do not know which document you want.

## Right now

**Nothing is running and nothing is billing.** `a3_attempt38`'s pod was deleted
after 78.9 min and is provider-confirmed gone; an account-wide re-query returns
**0 pods and 0 network volumes**.

**A3 is TERMINAL. D1 is DESIGNED, IMPLEMENTED at `$0`, PRICED and BLOCKED
THREE TIMES.** The maintainer's order of **2026-10-03** superseded the
2026-10-01 stop: skip the FFN-specific F1 experiment, carry
`ffn.activation_importance_v0` forward as the current best FFN, and take D1
through design, implementation, validation, pricing and preparation for
independent review — without launching. A **second `$0` round on 2026-10-03**,
also on maintainer order, then closed the materialization-ownership gaps,
corrected the behavioural-selection rationale and the pricing claims, and
designed the D-series battery family. Both rounds spent `$0` and created no
provider resource.

The next action is a **maintainer decision on three blockers**, any one of
which alone prevents D1 from executing:

```text
EVIDENCE   0 of 2 fresh disjoint batteries for D1, and 0 of 6 for the
           D-series family. THREE strata cannot fund six roles from their
           pinned sources: math_verified short 830 items, code short 321,
           gsm8k short 11. Extending math alone unblocks D1's two and
           leaves the family short.
FUNDING    chain hard ceiling $60.7509 -- a PROVISIONAL planning ceiling,
           not a price -- against $13.1777 of project headroom, short by
           $47.5732, and D1 is not in the C1 package's
           funds_formal_sessions_of list, so no existing allowance covers it.
CEILING    the SEARCH session alone prices at $31.1577 against the package's
           $30.00 per_attempt_hard_ceiling_usd. This binds SEPARATELY from
           the cumulative cap: a grant that moved only the cap still could
           not authorize the search session.
```

**A short GPU qualification is OWED and has not run.** The per-expansion
minutes were measured unbatched; D1 runs batched, in an unmeasured direction.
Until that qualification measures real CUDA/bf16 execution, the real state-eval
memory peak, the target-aware batched path's correctness and the actual timing
of a representative expansion, the dollar figures size a grant request rather
than price one.

Start at [`d1_design.json`](../stages/stage-1/phase_d1/plans/d1_design.json),
which owns every D1 figure and its claim boundary;
[`d1_evidence_capacity.json`](../stages/stage-1/phase_d1/analyses/d1_evidence_capacity.json)
owns the battery arithmetic. The phase `README.md` beside them is generated from
the index and carries no narrative.

## The test suite has a boundary now, and the trees line up

**`pytest` means the CORE suite: 3,030 tests, ~4m50s.** Reusable framework
behaviour and generic integration contracts, nothing else. It aims to be
**green** — a red test in it means a current core problem, not a closed
experiment's record. AGENTS.md **§2.8a** names the three suites and
`testpaths = ["tests"]` makes the first the default.

```text
pytest                                              core full suite
pytest scripts/experiments/stage-1/phase_d1/tests   a current experiment
pytest scripts/experiments/stage-1/phase_c2/tests   historical verification
```

**The eleven "expected" failures are gone from the default run, and none was
repaired to get there.** Every one was a closed experiment's historical-state
assertion — C1's readiness gates and session contract, C2's behavioural proposal
and three full-search chain proposals, C2's full-search budget/determinism/
headroom, C1's provider-resource agreement, continuation-B's preregistration
digest. They live with their experiments and still run on request.

**The experiment tree mirrors the evidence tree, row for row.** One owner per
`logs/stages/index.json` experiment id — the E-series ladder has `e3`…`e8b`, and
the C2 family's four separate ids (`phase_c2_full_search`, `phase_c2_replay`,
`phase_c2_behavioural`, `phase_c2_baseline_completion`) plus `continuation_b` own
their own tests rather than sharing a prefix.

```text
scripts/experiments/stage-1/phase_d1/   <->  logs/stages/stage-1/phase_d1/
scripts/experiments/stage-1/phase_c3/   <->  logs/stages/stage-1/phase_c3/
scripts/experiments/stage-3/tests/      <->  logs/stages/stage-3/
```

**Specific experiment imports from the core suite: ZERO**, enforced by
`tests/architecture/test_core_suite_boundary.py` — no core file imports an
experiment package, every `experiments.*` import from core is a shared
application module, no core file loads a named experiment launcher, and no core
file reads a concrete historical run. The allowed set is derived from the tree,
so adding an experiment cannot widen it.

**And the core result no longer depends on historical evidence existing.**
Archiving attempt 12's records, its preserved leaves or C1's cuda-stage-f
directory changes nothing — not even a skip. The Stage-1 importer's nine
refusals are proved against a two-leaf search built under `tmp_path`; the CUDA
launcher's budget accounting against an authorization and ledger the test
writes. All 14 core skips are live declarations, not absent records.

**NEXT SESSION, BEFORE RESUMING D1.** `review/d1-target-aware` (`161215c9`)
diverged from this refactor at `4dc579ba` — three commits on its side, four
behind. Once this refactor is on `main`, **rebase the D1 work onto the new main
first**; continuing on the old tree would reintroduce `tests/autoinit/` and the
flat `scripts/experiments/phase_d1/` paths this round removed.

Stage ownership comes from [`index.json`](../stages/index.json), not
from names: Stage 1 holds `phase_a`, `phase_a3`, `phase_b`, `phase_c1`,
`phase_c2`, `phase_c3`, `phase_d1`, `measurement` and `recovery_continuation`;
Stage 3 holds the E-series ladder. **A3 is its own package now** — the index has
always called it its own experiment, while its six modules sat inside
`phase_c3`.

`stage-1` is not a Python identifier, so `scripts/experiments/__init__.py`
extends `__path__` over the stage directories and `experiments.phase_d1` keeps
resolving. No import in the repository grew a stage, and core knows nothing about
stages — the closure deriver follows a grouped package by globbing one level,
reading no directory name.

**What a session's pod gate runs is now declared POSITIVELY.**
`SetupManifest.test_paths` names the suite; the ignore-complement is gone.
`autoinit_c1_launch` recorded that complement going stale six times, once per
experiment preflight directory created after C1 closed — each one a directory a
C1 pod would have collected on its own meter. And `scripts/pod/setup.sh` ran the
whole suite **twice**, once for a `tail -3` and once for an exit status; it is one
invocation with `tee` and `PIPESTATUS[0]`.

## What `main` carries

**The test-suite boundary refactor, integrated 2026-10-03 by SQUASH AND MERGE
per AGENTS.md P12.2.** One commit on `main` collapsing six:

```text
squash commit : 06cab9c8f106ed6f35db4c5cd6bf9d47288da139   on `main`
source branch : refactor/test-suite-boundary   (NOT deleted -- P12.2)
branch tip    : c7a39f3b070dc0b8fba16e7dff3b0b6c26c0a778
merge base    : 4dc579ba4938e1a2bfb05565d0b3594077bb531e
tested tree   : a2dab91ceb773699079cab4d15b03dfb67691ed2
```

**What the `2945 passed` measurement does and does not cover — corrected
2026-10-04 on maintainer review.** The squash was correct and is not being
rewritten; the claim made about it was too strong. Precisely:

```text
a2dab91c  the measured implementation/test tree, and the approved commit
c7a39f3b  adds pre-integration state/decision records on top
          (logs/state/current.json, logs/state/current.md,
           logs/budget/decisions.md -- nothing executable)
06cab9c8  tree == c7a39f3b tree, byte for byte, verified before pushing
          tree != a2dab91c tree
```

The earlier wording said the squash preserved *the measured tree*. It preserved
the **branch tip's** tree, which is a different object: `c7a39f3b` edits three
records, and **6, 4 and 1 core test files read them respectively** — the suite
asserts *on* those documents, so "touches nothing the suite executes" was wrong.
No executable implementation changed after `a2dab91c`, which is why a rerun was
not thought necessary; but those final state-only changes were **not separately
rerun at that point**, and that is the honest limit of the evidence.

The later D1 reconciliation's green core suite is new validation covering those
records — on a different tree again, with its own content. Where a record needs
the commit that produced the `2945 passed` measurement, it is `a2dab91c`.

**The branch is kept permanently, and it is load-bearing.** 623 distinct commit
hashes across the records resolve through branches, not through squash commits —
`session_commit`, `authorized_session_commit`, `head_commit`,
`swept_base_commit`, `declared_at_commit`, `git_commit` and the commit a
comparison names as having computed it. Checked before the squash and again
after: zero unreachable. The usual post-squash branch cleanup would invalidate
every one of them.

What changed: `pytest` now means the **core suite** and nothing else;
experiment-owned tests live with their experiments under
`scripts/experiments/stage-<n>/<experiment_id>/tests/`, mirroring `logs/stages/`
row for row; the core suite is **green** rather than carrying eleven standing
failures; and a static guard refuses a core test that imports a specific
experiment package, loads a named experiment launcher, or reads a concrete
historical run. **43m20s → 3m18s.**

**A merge records work. It is not a release, not a promotion, not a grant and not
a public claim.** `README.md` is unchanged by the whole range. No GPU or paid
resource is authorized by it, and no scientific result moved.

The validation is the core suite measured on this exact tree —
`2959 collected, 2945 passed / 14 skipped, 0 failed` — and a squash preserves the
tree byte for byte, so it stands for `main` without a second run.

**Before `main` carried this, it stood at `4dc579ba`**: the C3, A3 and D1-design
rounds, fast-forwarded on 2026-10-03 in a 597-commit integration that is the
reason P12.2 exists. `git log main` owns the hashes.

**The operational consequence, and it bites immediately.** A squash commit does
not have the branch's commits as ancestors, so `refactor/test-suite-boundary` is
no longer an ancestor of `main` and `main..refactor/test-suite-boundary` still
lists all six. **Continuing on it would re-apply the whole range** — cut a fresh
branch from `main` instead, and keep the old one: P12.2 forbids deleting it, and
it is where this round's commit hashes live.

## This branch: D1 replayed onto the squashed `main`

**`review/d1-identity-correction`, cut fresh from `06cab9c8`.** The D1 identity
round was **replayed**, not merged: `review/d1-target-aware` is no longer an
ancestor of `main`, so merging it would have re-applied its whole range and
recreated `tests/autoinit/` and the flat `scripts/experiments/phase_d1/` the
refactor removed. The old branch is kept — P12.2 — and its commits
`e93b627c 38c46a32 161215c9` are where the round's own hashes live.

Where the files landed, by §2.8a's question — what would change when this
experiment closes?

```text
tests/initialization/test_scoring_content_identity.py        CORE
tests/initialization/test_target_aware_scoring_end_to_end.py CORE
tests/initialization/test_materialization_identity.py        CORE
scripts/experiments/stage-1/phase_d1/tests/test_d1_design.py          D1's
scripts/experiments/stage-1/phase_d_series/{__init__,battery_family}.py
scripts/experiments/stage-1/phase_d_series/tests/                     the series'
```

**The battery family's placement was the real judgement.** Flat under
`scripts/experiments/` would have made the boundary guard read it as SHARED
APPLICATION LAYER — the guard derives that from tree position — and let any core
test import one series' arms. It is Stage-1 program material spanning D1, D2 and
D3, so it sits under `stage-1/`, where `experiments/__init__.py`'s existing
`stage-*` path extension already resolves it: no new mechanism, and
`from experiments.phase_d_series import battery_family` is unchanged. Its record
stays in `logs/shared/analyses/`, covered by that directory's existing
declaration as Stage-1 material belonging to no single experiment.

**Two defects the reconciliation itself found.** The reachability gate P12.2
names was blind to `git_commit` — the field AGENTS.md §3.6 names for an
experiment log, used by 20 records — found by comparing a hand sweep's 622
hashes against the tool's 606; now 623, zero unreachable. And
`phase_b_result` was reclaimed a second time on the claim that nothing reads it,
when eight assertions in a continuation_b test do; one grep had the wrong
subscript spelling and the other was truncated by `| head`. Restored, and
because those readers are an *experiment* test the core suite could not have
caught it, so `test_every_snapshot_key_a_test_reads_still_exists` now scans both
trees.

**Nothing scientific moved.** No GPU, no pod, no provider resource, `$0`, no
grant or authorization change, no frozen evidence rewritten, and D1 remains
blocked on the same three blockers.

## D1 — target-aware search. DESIGNED, NOT AUTHORIZED, BLOCKED.

**The A3 precondition is MET.** The A3 closeout refused to let A-bsz3 enter
D1/D2/D3 until the repository bound the numerical execution fingerprint to
materialization/resume identity: `compute_state_id` binds neither the
`ExecutionConfig` nor the artifact digest, so `53e30566c5f7` and `7dd2f6f6980b`
— two different artifacts from one path, one `result_spec_hash` — would have
collided on one resumable, deduplicable state id.
`aadistill.initialization.specs.materialization` adds a second coordinate
rather than forking the hypothesis:

```text
semantic_state_id                 scientific/path identity, still blind to execution
numerical_execution_fingerprint   batch size, packing, device class, dtypes
materialization_id                semantic + fingerprint -- what resume may key on
artifact_digest                   the bytes, observed and bound once
```

Resume, deduplication, the statistics-cache key and the operator/measurer policy
check all key on the right one, and every refusal is mutation-tested. No fake
operator identity was created: the registry is asserted clean of ids naming a
batch size.

**OWNERSHIP, not only checking — the 2026-10-03 correction round.** Detecting a
mismatch is not owning a destination, and four gaps were closed:

* **Checkpoint paths are materialization-keyed.** `_materialize_and_measure`
  wrote `workdir/states/<state_id>`, so two materializations of one path owned
  one directory. `BeamSearch.checkpoint_dir()` is now the single owner of
  `states/<semantic_state_id>/<materialization_id>/`.
* **Resume finds ITS materialization**, via `latest_by_materialization_id()`
  beside an unchanged `latest_by_state_id()` — the semantic view collapses
  materializations and cannot find an earlier matching one once another
  protocol has written newer records. The frozen C2 canonical-record rule reads
  the semantic view, so it was left alone rather than migrated.
* **Parent lineage is bound:**
  `materialization_id = H(semantic_state_id, fingerprint, parent_materialization_id)`,
  because a child's bytes depend on the parent bytes actually consumed. The root
  derives its identity from the pinned teacher revision, not from an experiment
  special case. All three terms were shown to move the id independently.
* **The scoring MASK is bound to scientific identity.** A mixture's
  `content_sha256` hashes item ids and token ids only, so two assets could share
  every hash in the operator path and carry different supervised masks — a
  scientific identity collision. `scoring/content.py` binds item id, token
  content, the aggregation labels and **the position metadata the policy
  declares it reads**, asked of the policy rather than named here, so a D2/D3
  policy reading `final_answer` binds the right thing with no edit. The word
  `assistant` appears nowhere in it.

**One measurement identity replaced a growing list of comparisons.**
`scoring/protocol_identity.py` combines the suite's **structural** identity,
the suite's **content** identity, the scoring content, the policy, the
reduction semantics (chunk, reference strategy, aggregation rule) and the
execution fingerprint into one `measurement_protocol_id`, which `StateEvaluator`
stamps into every evaluation; `_restore` asks one question instead of the two it
had and the three it would have grown. A record predating the identity is judged
by its own historical fields and is **never** reinterpreted — its id is not
reconstructed, because that would mean asserting a reduction nobody recorded.

**And the suite's content is bound THERE, not in `suite_hash` — the first
version of this repair got that wrong.** `load_state_eval.py` was constructing
its suite without the manifest's `content_sha256`, so the obvious fix was to
pass it in. That moved `suite_hash` for an unchanged asset, from `6421fa4c…` to
`a39df3c0…`, and `suite_hash` is the **structural** identity by this project's
own design: `autoinit_phase_a_driver` pins it as "the STRUCTURAL suite hash —
suite_id/version/domains/subtypes/critical_tags" with
`STATE_EVAL_CONTENT_SHA256` pinned beside it as a separate field, **45
committed records bind the structural value**, and the C2 baseline driver
refuses a measurement whose staged suite hash differs from the one its frozen
candidates were measured under — which is exactly what it did, in five tests.
The repair now: the loader still **requires** the manifest to carry a content
hash, so a caller cannot bind `None`, and `StateEvaluator` takes it as
`suite_content_sha256` and binds it in the new protocol id, which pins nothing
historical. An unbound content identity is recorded as the literal `unbound`
rather than refused, so the omission stays visible.

**And it is `unbound` in production today, which is the one piece of wiring
this round deliberately did not do.** `load_state_eval.load()` returns the
manifest, so every caller already holds the content hash; passing it is one
keyword argument. The caller that would pass it is `phase_a_search.py`, which
four declared source sets name by digest — three of them belonging to closed
experiments — so editing it to improve a metadata field would spend frozen-set
drift on nothing. **D1's driver is the first production caller that must pass
`suite_content_sha256`**, and until one does, every stamped protocol id records
`unbound` for the suite content: honest, visible, and not yet the full
binding.

**The scoring policy is generic and hash-bound.** One
`ScoringPositionPolicy`, consumed by all four operators, by the global state
evaluator and therefore by the beam — a candidate selected on supervised
positions cannot be pruned on all of them, and a search whose operators and
measurer disagree is refused with a named error.
`positions.supervised_target_v1` reads each frozen mixture's **own**
`assistant` tag rather than re-deriving a chat-template rule, and admits
`74.9%` of `calib.domain_balanced@v1`'s prediction positions, `78.4%` of
`calib.reasoning_heavy@v2`'s and `73.0%` of `state_eval_v1`'s. Untemplated
raw-LM items keep every prediction position, so D1's **data** is identical to
the incumbent search's.

**The incumbent policy is numerically inert, and that was measured rather than
argued.** A four-operator toy chain rebuilt against this branch's merge base
produces the same artifact digest, the same kept layers, the same kept neurons
and the same kept heads. The treatment moves all four.

**Batched execution is global from D1 onward**, at `batch_size=3` /
`length_sorted_v1`: DEPTH, FFN, RESIDUAL_WIDTH, ATTENTION, the calibration
statistics, the causal scoring, the global state evaluation and the beam
candidate evaluation. A maintainer **uniformity** decision, not a performance
claim — A3 measured it 8-10% *slower* on the ATTENTION scorer. The value is
configuration, never a core constant. Every reduction obeys
`score = sum_t(w_t·v_t)/sum_t(w_t)` with `w_t = 0` at padding.

**The behavioural design is a judgement, with the arithmetic beside it rather
than behind it.** Top-K **2**, **2** screening seeds, **3** confirmation seeds
on a fresh disjoint battery — retained as a pragmatic balance of behavioural
breadth, two-seed screening stability (C2's specific weakness was a single-draw
ordering), fresh independent confirmation and cost. **Not** a demonstrated
optimum.

The 2026-10-03 review corrected the argument that first justified those
numbers. The max-of-K inflation is the winner's curse **on the screening
estimate**; a confirmation rung on genuinely fresh disjoint prompts and fresh
seeds is unbiased under the global null however inflated the screening number
was. So `screening bias < SESOI` is **not** a validity condition, and nothing
admits or rejects a design by that comparison — the filter that did was
removed, together with the test that had encoded it. What governs detection is
`P(a good candidate is in the Top-K) x P(advance | in Top-K)`: the second is
`0.7808` at the point estimate but `0.549`–`0.932` across the per-seed spread's
own 95% interval, because that spread comes from **three** A3 deltas; the first
is **UNKNOWN**, and C2's negative result argues against assuming it is near 1.
Owner:
[`selection_noise.py`](../../scripts/experiments/stage-1/phase_d1/selection_noise.py),
whose quadrature is self-checked against two closed forms and whose
`CLAIM_BOUNDARY` travels into the design record. No C2 figure is re-analysed.

**The search stage:** 384 reachable leaves (24 orderings × 16 calibration
assignments), 92 expansions at the standing beam width 6 / warmup 1, planning
ceiling `$31.1577`. The per-expansion minutes come from unbatched telemetry and
D1 runs batched — in an **unmeasured direction**, since A3 measured the
ATTENTION scorer 8–10% slower at batch 3 while causal-KL's length-sorted
packing won `1.1884x`. The earlier claim that batching "can only reduce the
time per expansion" was removed; A3 refutes it.

**One engineering gap is identified and deferred to D2 on purpose.** The
activation collectors implement only the binary form of a position policy —
their divisor is an `int64` token count. D2's continuous `c_ref(t)` needs a
weighted denominator in `StatsSpec` and three divisor call sites.
`require_binary_token_weights` refuses by name rather than rounding a confidence
weight to a mask.

## The D-series behavioural battery family. DESIGNED, NOT MATERIALIZED.

**Six roles, one rule, frozen before any D1 outcome exists.** `d1_screening`,
`d1_confirmation`, `d2_screening`, `d2_confirmation`, `d3_screening`,
`d3_confirmation` — each disjoint from the others and from every historical role
by stable id **and** normalized prompt content. Owner:
[`battery_family.py`](../../scripts/experiments/stage-1/phase_d_series/battery_family.py);
record:
[`autoinit_d_series_battery_family.json`](../shared/analyses/autoinit_d_series_battery_family.json).

It adds **no selection code**: `battery_render.rank_take` already orders a pool
by `SHA256(base_digest : rank_domain : stratum : stable_id)`, which is how
`c2_screening_v1` was drawn beside `c1_confirmation_v1`. Distinct rank domains
give *independent* samples, not disjoint ones, so disjointness comes from a
fixed build order in which each role excludes every role before it. There is no
seed and no date in the rule — nothing an agent could choose after seeing a
result. `allocation_rule_id` `ced017a1f3f155ba5aaf383e61156c12` is computable
with no pool and no source decision, which is what makes "frozen prospectively"
checkable; `family_content_id` is **null**, because binding source pins and
realized items today would be invented provenance.

**It is a NEW behavioural distribution and says so.** Stratum names, domains and
counts are inherited unchanged from the C1 mixture (imported, never restated),
so the balance and the 950/850 denominators hold. The population changes, so
`d_series_behavioural_v1` is **not** the `c1_confirmation` distribution: B's
historical C1 number is not imported, and C0's SESOI is carried forward as a
**recorded assumption** about this population. That is sound because every
D-series comparison re-measures challenger and incoming incumbent together, on
the same fresh battery, under one protocol — no D-series decision reads a
historical score.

**What the derivation found.** Six roles need `6x` the mixture at once, and at
that scale **three** strata are short rather than one:

```text
stratum         per role    x6   eligible   short   roles fundable
math_verified        150   900         70     830                0
code                 100   600        279     321                2
gsm8k                150   900        889      11                5
knowledge/multihop/rag/tool                      0             36+
```

`gsm8k` and `code` are answerable from files of repositories **already pinned at
frozen revisions**; MBPP's remaining splits may not cover 321, and if they do
not, a second code source or a smaller behaviour-only code component is a
maintainer decision — the second moves the `usable_rollout_rate` denominator.
`math_verified` needs the full Hendrycks MATH release: a new dataset, so
licence, revision, digests, renderer parity and contamination all have to be
recorded first. **Nothing was fetched and no source is pinned.**

## A3 IS TERMINAL. A-bsz3 IS A DISTINCT NUMERICAL MATERIALIZATION PROTOCOL.

**Finished 2026-10-03 on `a3_attempt38`.** One chain — initialization →
recovery training → evaluation → aggregation → closeout — answered the
practical question once. Owner:
[`a3_closeout.md`](../stages/stage-1/phase_a3/analyses/a3_closeout.md), every
figure derived by `scripts/autoinit/aggregate_a3.py` into
[`a3_comparison.json`](../stages/stage-1/phase_a3/analyses/a3_comparison.json)
(`75f2641041a66882…`), computed **off pod at `$0`**.

```text
A_bsz1   53e30566c5f7   == the frozen C3 incumbent, rebuilt on fresh hardware
A_bsz3   7dd2f6f6980b   DIFFERENT
         -> DISTINCT_NUMERICAL_MATERIALIZATION_PROTOCOL
result_spec_hash         IDENTICAL
```

**The registered prediction held.** `attn_out` reduces shape-dependently on an
L40S and only `bsz=1` unpadded is exactly reproducible, so the digest was
predicted *before* the GPU ran not to match. Each protocol reproduces its
**own** digest across four interleaved rounds, so this is determinism, not
noise. One kept-head slot of 448 differs — layer 7, head 13 vs 12, margin
`0.00047`, a near-tie. Rank correlation `0.9999997`, max score drift `0.0034`,
peak VRAM `3.1737 GiB`. The masking invariant holds and the `$0` position
prediction matched observation exactly: 67 forwards / 0 padded for bsz1, 23
forwards / 2,398 padded for bsz3.

**bsz=3 is SLOWER, on every pod that measured it.** Scorer means, 3 timed
rounds each with the warm-up excluded: `2.8073 / 3.0807` on attempt34 (9.7%),
`2.8548 / 3.0892` on attempt35 (8.2%), `2.7868 / 3.0647` on attempt36 (10.0%)
— **mean 9.3%, and the sign never flips**. Peak VRAM `2.4480 → 3.1737 GiB`
(+29.6%), identical on all three. It is also a modest slowdown in a small part
of the work: the whole ATTENTION suffix differs by `0.4%`, inside each arm's
own round-to-round spread.

**No runtime threshold was inherited** — the `1.25x` bar belongs to the
causal-KL packing pilot, which is a different workload. Fitting
`T = N·F + P·c` to each operator's two batch points explains the sign:
causal-KL pays `+12.91 ms` per invocation across `60,099` forwards and has
something to amortize; activation-importance fits a physically impossible
`−3.45 / −2.57 / −3.56 ms` across **67** forwards and has nothing. The
per-position term meanwhile clusters at `50.57–50.78 µs` across the three
fits — the real term is stable and the fixed term is the model being wrong.
Two batch points per session is an exactly-determined fit, so `F` and `c` are
not independently identified; drawing the curve needs a third batch size, not
a third session.

**No detectable correctness effect.** Paired over the three frozen C3 seeds,
`n_scorable = 850` each, controls **reused from attempt75 and not retrained**:

| seed | Δ correct | McNemar b/d | Δ usable |
| --- | --- | --- | --- |
| 217230555 | **+0.012941** | 29 / 18 | +0.0811 |
| 1151307191 | **−0.011765** | 16 / 26 | −0.0094 |
| 2045359208 | **+0.007059** | 29 / 23 | +0.0674 |

Pooled correctness delta `+0.002745`, descriptive 95% CI
`[−0.006275, +0.012157]`, one-sided LCB `−0.005098`, pooled usable delta
`+0.046367`, no guardrail fired. Mixed signs with a pooled delta an order of
magnitude below the `±0.022353` seed-level spread C3 measured is the signature
of no effect — which one near-tie slot flip is what would predict.

**The usable-rollout axis moved more, and it is the axis the session caveat
bites hardest.** Every component is reported in the comparison artifact rather
than averaged. The three *treatment* non-empty rates cluster at
`0.8526 / 0.7726 / 0.8463` while the *controls* spread
`0.7863 / 0.8116 / 0.6821`, and the largest positive delta is the one whose
control is the outlier — more consistent with between-session variation than
with a batching-protocol effect, and A3 cannot separate the two. No sample hit
the context limit on either side.

**Three things this evidence may NOT be used to claim.** It is **not** a
non-inferiority result: three seeds support no population claim, the interval
is descriptive (prompt-level, seeds as fixed blocks), the SESOI `0.01` is
reported as the *scale* of a material loss and decided nothing, and the
withdrawn `−0.030` is the measured seed-level noise band, never an equivalence
margin. **Session is an unquantified alternative explanation** — treatment and
control were measured on different physical hardware, bounded only in part by
A-bsz1 rebuilding `53e30566c5f7` there. And **A-bsz3 may not enter D1/D2/D3**
until the byte-identity gap below is bound.

**Adoption: nothing to adopt.** The protocol is slower, builds a different
artifact, and shows no correctness effect. It is not rejected on quality
grounds; it simply is not an optimization on this operator. The cheap test for
a future operator is the intercept of per-invocation time against sequence
length at batch 1.

**The probes were restored, not retrained.** `a3_attempt35` trained all three
treatment probes and was then refused by the protocol admission; AGENTS.md
P8.4 state 2 says a trained, durable, unscored checkpoint is resumed at
scoring. `a3_attempt38` ran the ladder `B, C, F(restore), G, H` and cited D and
E from the preserving attempt. The three probes stay durable at
`a3_preserved_probes/a3_attempt35/` with per-file digests, now P8.4 **state 1**
— archival evidence, not an execution dependency, eligible for the standing
retirement policy once no declared consumer reads the bytes. The comparison
consumed per-sample rows and scored records, never weights.

**Cost: `$13.4600` across 34 pods**, every one provider-confirmed deleted.
Terminal run 78.9 min, `$1.43`, clean `DRIVER_EXITED:0`. Four limits after
execution: formal `$7.2431` of `$76.6523` · GPU engineering `$1.9395` of
`$10.0000` · package `$9.1826` of `$86.6523` · project `$23.8977` of
`$410.0000`. Owner: [`ledger.md`](../budget/ledger.md).

## The 21 paid rediscoveries of one missing dictionary entry

**The maintainer's round-4 judgement, and the repair it required.** A3's
launcher did not put `SESSION_FROZEN_EXPECT` in the pod environment, the shared
setup script requires it with `${VAR:?}`, and the session therefore died during
setup — **21 times, for `$2.7400`**, each time on a freshly created provider
resource. The failure was fully deterministic and knowable at `$0`. The
maintainer accepted the root-cause fix and the direct contract regression and
**forbade** building a canary or rehearsal subsystem for it.

**Two mechanisms exist now, and both are small.**
`tests/integration/test_setup_env_requirements.py` parses the shell's own `${VAR:?}`
requirements and asserts every session launcher supplies them — the contract,
checked from the side that breaks.
[`failure_signature.py`](../../src/aadistill/infrastructure/failure_signature.py)
normalizes a failure to `CLASS:token` and `same_failure_gate` refuses a paid
retry of an identical deterministic signature **without a corrective change**.
Transients — provider capacity, cold host, unreachable — return `None` by
design, so the existing backoff and reacquisition policy still applies to them
untouched.

**And one host question moved before the money.** `a3_attempt35` trained three
probes across a `$4.46` session and was then refused, because the host NVIDIA
driver *branch* had moved `580 → 595` — which `generation_compat@v2` calls a
real runtime event
rather than provenance. The property is knowable one ssh round trip after the
pod answers, so `SessionSpec.host_admission` now asks it between the confirmed
image identity and setup, and a refusal is **redrawable** exactly like a cold
host. Default `None`, so no prior session changes behaviour.

## The A3 chain — what it was, and what each gate caught

**Every failure in A3 was found by a gate rather than by reading**, and each
one produced a durable repair plus a `$0` regression. The ones worth carrying
forward:

* **The aggregation never ran on the meter.** attempt75 trained, preserved and
  scored nine probes over 919 min and `$16.71`, then lost its decision artifact
  to a crash in on-pod stage I. A3's driver **ends at preservation** and the
  ladder has no stage I. That removes the failure class instead of guarding it.
* **Three wrong historical comparability sides, in sequence** — A3's own
  attestation under v1, then the controls' *train* runtime, then the
  aggregator demanding exact fingerprint equality. All replaced by v2 against
  the controls' **engine probe**, verified from each pod's own admission
  record (`comparable is true`, the v2 rule id, `identities_equal`,
  `driver_branch_equal`). Four runtimes live in this repo and picking the
  wrong one looks like a science failure.
* **The wrong ATTENTION calibration profile** (`reasoning_heavy@v2` where the
  frozen arm says `domain_balanced@v1`). Derived from the frozen arm now; the
  frozen design document came out byte-identical, which is what proves it was
  an engineering repair and not a science change.
* **The pod's blocking test gate was undeclared**, so the shared setup would
  have run the whole repository suite on a billing L40S and exited non-zero on
  the documented development-only failures. It is `tests/a3_preflight` by
  derived complement now.
* **The disk floor was estimated at 18.61 GiB and measured 50.3 GiB** — the
  third disk exhaustion in this programme. Container disk is provisioned from
  the measured floor with a `1.5×` margin: 110 GB.
* **A driver writing markers to a path its launcher does not poll** made every
  marker invisible on attempt75, `ALL_DONE` included. The status path is named
  once.

**The production-path rehearsal is the part attempt75 never had.** It drives
the **real** `A3Driver.run` with only hardware-bound calls faked, emitting the
real filesystem layout, and it found a pod-fatal defect before a pod existed:
stage H handed `C1ProbeRecord` empty `counts` and `rates`, which that type
refuses by construction — three trainings and three evaluations into a paid
session. The resume ladder got the same treatment, and a `$0` admission dry run
proved the host rule satisfiable before anything was created.

**The comparison is tested on a replica of a finished run.** An identical field
reports exactly `0.0` with zero discordant McNemar pairs; a planted `+17/850`
improvement comes back at `+0.02` with the right sign; four integrity refusals
fire (generation fingerprint, scoring contract, battery, wrong estimand). The
bootstrap moves with its seed — the C3 defect where every record asserted
`654678655` while the resampler used C1's `816109261`.

## The D-series directive is recorded and NOT started

**Received 2026-10-01.** D1 (target-aware search), D2 (target-aware +
reference-confidence weighting) and D3 (target-aware + KL + pure student
confidence) are **global** scoring/search experiments — not attention-only —
applied consistently to DEPTH, FFN, RESIDUAL_WIDTH, ATTENTION, every other
calibration-consuming structural operator, global state evaluation and beam
ranking/pruning. Operator set, beam width, beam schedule, target geometry,
calibration data and search breadth stay identical across the three; the
scoring semantics are the variable. A search-stage metric does not name the
winner: the best candidates from each, plus the incumbent anchor, enter ONE
common behavioural-selection design.

Then **Stage-0 teacher-native v2**, with data scale as a first-class variable
on a deterministic *nested* ladder of ~60k → ~240k → ~960k → ~3.84M target
positions.

**Order, and it is the maintainer's:** A3 is done; **do not start the FFN
experiment and do not start D1/D2/D3.** When the maintainer resumes the
programme the recorded sequence is FFN, then freeze the current-best
implementation for every structural kind, then design and price D1/D2/D3 and
push the protocol for independent review *before* any paid execution. Owner:
[`phase_c_roadmap.md`](../stages/stage-1/phase_c1/plans/phase_c_roadmap.md).

## A-bsz3 — what the implementation is

**A-bsz3 is canonical A under B3's batching protocol.** Same operator
(`attention.activation_importance_v1`), same composition, same scientific
identities, same frozen shared parent — executed at
`calibration_forward_batch_size = 3` and
`calibration_batch_packing = length_sorted_v1`. It is **not** a new operator,
imports nothing from causal-KL, and has **no implementation id of its own**:
both knobs are `ExecutionConfig` fields and neither enters a hash, so A-bsz1
and A-bsz3 are **the same scientific state**. That is the point — giving the
knob an impl id would have answered the equivalence question by definition.

**The shared parent is untouched.** `micro_batch_size: 1` on the prefix stays
pinned; the comparison drives `materialize_fixed_path_suffix`, which starts
from the already-verified parent and applies the override to the ATTENTION tail
alone.

**The old negative result did not transfer, which is why this was measured.**
`bsz=4` at `0.946×` for the statistics collector was measured at
*original-order* packing, costing **36.73%** padding; length-sorting at bsz3
costs **4.01%**. Neither did causal-KL's `1.1884×`. A3 measured the right
thing and the answer was still no.

**The operator emits the evidence the comparison reads, and none of it is
identity.** `OperatorStep.identity()` does not look at `trace`, and the
artifact digest comes from the written bytes. Per-head score vectors, the
forward / executed / valid / padded counters **counted by the loop as it ran**,
and a CUDA-synchronized `scorer_seconds` around the statistics pass alone. The
counters matter because `padding_profile` already predicts the same three from
item lengths — a trace that restated that prediction could never contradict it,
and these can. The loop's `valid_positions` and the collector's independent
`calibration_tokens` give a masking invariant a consumer checks rather than
trusts.

## Scientific identity is not materialization identity

**The branch below that fired is the second one.** A-bsz3's digest differs, so
it is a **distinct numerical materialization protocol**, and the obligation is
now live rather than hypothetical: **A-bsz3 may not enter D1/D2/D3 execution**
until the repository binds the numerical execution fingerprint to
materialization/resume identity. It did not stop A3 — the differing digest was
recorded as a finding and the chain continued to a behavioural result — and no
materialization-identity framework was built during the experiment, which is
why this remains an open precondition and not a completed one.

**The gap is stated and tested.** `compute_state_id` binds the root
teacher, the target spec, each step's implementation id and signature, the
calibration profile hash, the operator config hash and the seed. It binds
**neither the `ExecutionConfig` nor the artifact digest** — correctly, because
otherwise two runs of the same science at different batch sizes would be
different *scientific* states, a resume would not find its own journal, and
whether A-bsz1 and A-bsz3 agree would be settled by definition instead of
measured.

But a semantic id cannot own bytes, and today it is the only id the beam has.
So:

* **digests identical** → A-bsz3 would be a **transparent execution
  optimization**, keeping one scientific *and* one materialization identity,
  with adoption turning on runtime alone. **This branch did not fire.**
* **digests differ** ← **MEASURED.** A-bsz3 is a **distinct numerical
  materialization protocol**, and **A-bsz3 may not enter D1/D2/D3 execution**
  until the repository binds the numerical execution fingerprint to
  materialization/resume identity. Two artifacts that differ in bytes must
  never share a resumable, deduplicable state id. A3's behavioural result
  passed and does **not** clear this — it is an engineering correctness
  property, not a behavioural one.

**A second implementation id is forbidden as the fix.**
`attention.activation_importance_bsz3` would fork the *scientific* identity to
repair a *materialization* problem, and would answer the equivalence question
by definition — the exact error this study exists to avoid.

**The hazard is where the resume reads.** `BeamSearch._restore` looks a state
up by `state_id`, then re-identifies the checkpoint on disk and refuses if the
bytes disagree with the record. That catches a stale or tampered checkpoint; it
cannot catch a record that is internally consistent and was written by a
different numerical protocol. The guard that *does* hold is now locked by
`test_resume_refuses_a_record_from_a_different_numerical_protocol`, which
appends such a record to the real journal — same state id, another protocol's
digest — and requires the refusal. It was mutation-checked: disabling the
digest comparison in `_restore` turns it red.

**The `−0.030` seed-1 stop is an engineering catastrophic-regression stop.** It
is **not** an equivalence margin and must never be reported as one: clearing it
means the protocol is not visibly broken at one seed, and bounds nothing.

**FORMAL C3 MEASURED ALL NINE PROBES AND THEN FAILED TO AGGREGATE THEM.**
`attempt75`, secure L40S at `$1.09/h`, **`$16.7083`** of a `$22.1452` derived
ceiling. Both frozen digest gates passed. All three arms rebuilt to the exact
identities attempt66 recorded. **9 of 9 trained, 9 of 9 preserved, 9 of 9
scored**, zero argparse errors. Then stage I raised before writing the decision:

```text
C1ResultsError: autoinit.v1.phase_c3.A_incumbent.217230555:
                unknown arm 'A_incumbent';
                this record allows ['incumbent', 'treatment']
```

**The measurement is complete and durable; the on-pod decision artifact is
not.** 235 files across 19 classes came home — nine per-sample row files, nine
scored aggregates, nine generation-admission records, 63 generation files —
and live at `/home/ecs-user/aad-artifacts/phase_c3/attempt75`. Owner:
[`attempt75/closeout/outcome.json`](../stages/stage-1/phase_c3/runs/attempt75/closeout/outcome.json).

**The field is protocol-uniform, which is the property C2's was not.** One
generation fingerprint `c318d1c62197…` across all nine, one scoring contract,
one battery, `comparable=true` on every admission record.

## C3 IS COMPLETE. THE VERDICT IS `NO_GO`.

**Authorized by the maintainer decision of 2026-10-01**, which accepted that
attempt75 completed the measurement the preregistered design needs and that
stage I is deterministic post-measurement analysis whose failure justifies no
retraining. The aggregation ran **off pod, at `$0`**, from attempt75's
immutable evidence alone.

```text
Delta_primary   -0.001961      B_causal_b1 - A_incumbent
LCB one-sided   -0.009412
UCB one-sided   +0.005490
SESOI           +0.010         -> UCB < SESOI, so NO-GO
```

**Nothing failed to produce this verdict.** No behavioural veto fired, and seed
robustness *passed* 2 of 3. The result rests on the effect being bounded below
the smallest effect worth having — an informative NO_GO, not an unresolved one.

* **B remains the incumbent.** causal-KL does not promote on the primary C3
  operator-isolation claim.
* **C4 remains NOT AUTHORIZED and must not start.**
* The B1-vs-B3 protocol choice stays a post-C3 maintainer decision.

**Both secondary contrasts and the usable-rollout result are reported
completely, and do not redefine the primary verdict.** `Delta_batch`
(B3 − B1) `+0.001961`; `Delta_practical` (B3 − incumbent) `+0.000000`. On the
secondary axis causal-KL was **better**: pooled usable-rollout delta
`+0.025263`. It produced more usable rollouts, just not more correct answers.

**It is reproducible from the committed record.** Byte-identical across two
runs except the commit field, which necessarily moves when the artifact is
committed. The artifact binds all 27 consumed input files by `sha256`, the
preregistration hash `ac44662c…`, the isolation plan hash, and the
implementation's commit and module hashes — and it refuses to run on a dirty
tree, because a commit recorded beside uncommitted edits names bytes that did
not execute. Owner:
[`attempt75_stage_i/c3_decision.json`](../stages/stage-1/phase_c3/analyses/attempt75_stage_i/c3_decision.json),
produced by `scripts/autoinit/aggregate_c3_stage_i.py`.

**attempt75's stage-I failure remains historical fact** and is not rewritten as
though the live session had reached stage I successfully.

## The `$0.9492` formal overspend — HISTORICAL, against the old `$55.00`

**It is a fact about attempt75, not the current balance.** The 2026-10-01
amendment raised the formal allowance to `$65.6523`, so the live remaining is
`$9.7031` — read the summary table, not this heading. What follows is why
attempt75 was issued against a book it exceeded, and it is **not** rewritten
into compliance: the authorization-gate defect, the seven earlier sessions that
also spend the allowance, and the `$55.9492` already spent all stand exactly as
recorded. A forward-looking raise is not retroactive permission.

**`derive_budget` attributed the formal allowance to one hardcoded
experiment**, `FORMAL_EXPERIMENT = "phase_c1"`, so every formal C3 session
reached the project cumulative and spent nothing from the `$55.00` allowance
the 2026-09-28 amendment funds it from — the same amendment that makes
*remaining formal allowance ≥ derived session ceiling* a C3 issuance condition.
A session that must pass a formal-allowance gate is a session that spends it.

Repaired: the funding scope is package **configuration** now, and a run stating
its own `package_id` is attributed by that instead. The exact derivation is
worse than the estimate that prompted the repair, because **seven earlier C3
formal sessions also spend the allowance** — every one created a provider
resource under a formal launcher invocation, which the package's own
`attempt_counting` rule makes a formal session:

```text
phase_c1   $22.6176
phase_c3   $33.3316   = $2.9533 pre-science + $13.6700 attempt66 + $16.7083 attempt75
           --------
total      $55.9492   of $55.0000   ->   OVER by $0.9492
```

**One stated ground for accepting attempt75 does not hold.** The 2026-10-01
decision accepted it partly because *"total actual formal spend remains below
`$55.00`"*; under the exact derivation it does not. The project cap is still
respected — `$383.3623` of `$400.0000`.

**The authorization-gate defect is larger than estimated, and it is
attempt75's alone.** Reconstructed chronologically by
`scripts/consolidate/audit_formal_allowance.py`:

| session | formal remaining before | derived ceiling | gate |
| --- | --- | --- | --- |
| attempt66 | `$29.4291` | `$22.1452` | **would have PASSED** |
| attempt67–74 | `$15.7591` | `$22.1452` | would have been refused — all `$0` |
| **attempt75** | **`$15.7591`** | **`$22.1452`** | **refused, short `$6.3861`** |

attempt66 was correctly authorized. attempt75 should not have been issued. The
defect affected the risk ceiling and the accounting gate, not arms, seeds,
training, evaluation, scoring or the estimand — which is the ground on which
the maintainer accepted its evidence retrospectively. That acceptance is **not
a budget increase and not permission to spend further.** Owner:
[`formal_allowance_audit.json`](../stages/stage-1/phase_c3/analyses/formal_allowance_audit.json).

**No further C3 session is issuable, on either book.** `formal_pricing.assess`
returns `FUNDABLE=False` on both approved devices and now fails **two**
conditions rather than one — project cap short `$5.5075` (L40S) / `$0.1046`
(L40), and formal allowance short `$23.0944` / `$17.6915`.

**Four chains were consumed at `$0` before it.** attempts **71–74** each passed
every pre-provider gate — including the new `durable_capacity_gate` — and were
then refused at the create call with *"no longer any instances available"*.
That is the ordinary behaviour of `Low` stock on this account, already seen at
attempts 67–69. Each is a consumed one-use chain under P12.1, not a retried
experiment: no provider resource existed and nothing billed.

**Three more defects were found, and the third IS scientific.**

**0. The bootstrap seed was C1's.** `stratified_cluster_bootstrap` defaults to
`isolation.bootstrap_seed()`, domain-separated as `phase-c1:bootstrap` =
**816109261**. C3's preregistration declares **654678655** under
`phase-c3:bootstrap`, the authorization certifies that figure and the session
contract reports it — and nothing passed it to the resampler. So every C3
record asserted a seed the computation did not use. It moves the *interval*,
not the point estimate, and a verdict reads the LCB, so it is a scientific
defect rather than a cosmetic one. Repaired: the driver now passes
`CS.bootstrap_seed()` explicitly. **On this data it changed nothing** — the
replay reports both seeds side by side and the LCB and the verdict are
identical, which is a measured fact rather than an assumption.

**Two operational defects. Neither is scientific.**

**1. The driver's markers went to a file the launcher does not read.** The C3
driver writes `mark()` to `/workspace/autoinit_c3.status`; the C3 launcher
declares `status_path = /workspace/autoinit_c1.status` and polls it with
`tail -1`. So **every** driver marker — `STAGE_PASSED:D`, `STAGE_START:G`,
and `ALL_DONE` itself — was invisible to the launcher, which saw only the
setup script's markers and stopped at `SETUP_DONE`. This is not new: it is why
attempt66's session record says `terminal = DRIVER_EXITED:40` rather than
naming a marker, and it is the same defect the C2 replay driver hit on its
attempt 9 — *"the label is wrong; the result is not"*.

Two consequences, one harmless and one not:

* **harmless** — a successful run would be classified by exit code, so
  `collect_and_teardown` would use the FAILED artifact spec. The two C3 specs
  carry **identical patterns** and differ only in `min_matches`/`required`, so
  everything still comes home; what is lost is the completeness *assertion*,
  not the evidence.
* **not harmless** — `c3_acquire.sh` decides whether to launch ANOTHER paid
  attempt by grepping the launcher log for `ALL_DONE` or `STAGE_START:G`. With
  the markers invisible, a completed formal run reads as *"no measurement
  began"* and the loop builds the next chain and launches a **second formal
  attempt**. That is a scientific and budget violation, not a labelling one.

Repaired mid-run, at the supervision channel and nowhere near the science: a
detached `tail -F /workspace/autoinit_c3.status >> /workspace/autoinit_c1.status`
on the pod. The launcher resumed echoing markers immediately
(`MARKER:STAGE_START:F` at `$1.44`). **The source fix — one status path, named
once — is owed after this run, because editing the tree a live session is
bound to is not a repair.**

**2. The `$0` capacity watch does not back off.** `c3_acquire.sh` sleeps only
when the approved tier is *dry*, and `Low` stock that never converts reads as
usable. The loop rebuilt a chain roughly every 66 seconds and would have spent
all forty of its rounds in about 44 minutes rather than pacing them over hours.
attempt75 acquired on round 5, so it cost nothing this time. A backoff after a
capacity refusal is owed before the next acquisition run.

**attempt66 stands closed with no result, and ONE FRESH FORMAL C3 ATTEMPT IS
AUTHORIZED.** attempt66 trained all **nine** formal probes on L40S over 12.5
hours for `$13.67` and passed **both** frozen digest gates. Then two things
happened: stage H exited 2 on a scorer CLI argument, and every one of the nine
2.22 GiB checkpoints was refused by Hugging Face for private-storage quota. So
there is a trained nine-probe matrix with **zero** evaluations and **zero**
surviving weights, and **no C3 verdict**. It is not resumed, not pooled and not
reinterpreted as a partial measurement. See
[C3 — nine probes trained, no result](#c3--nine-probes-trained-no-result).

**The maintainer decision of 2026-09-30 authorizes a NEW measurement**, not a
retry: the same three arms, the same three preregistered seeds, the same digest
gates, recipe, battery, scoring contract, estimands, bootstrap and decision
rule. No fourth seed, no protocol change, no reuse of attempt66. **C4 remains
NOT AUTHORIZED** and no budget increase was granted.

**Storage was the blocker and it is cleared — measured, not estimated.**
Fourteen E1 scaling arms were permanently retired from the relay under the
consumer rule (AGENTS.md P8.4): `31.0869 GiB`, every one a completed and validly
scored arm of an experiment that closed on 2026-08-02, with its config, run
manifest, holdout, behaviour and GSM8K evaluations all git-tracked and its
content hash recorded in a tombstone. The five arms any record names as a
checkpoint source or behavioural anchor were **kept**. Headroom went `1.4374
GiB` → `32.5244 GiB`, and the LFS batch endpoint was then asked with **fresh
random oids** — an oid that already exists dedups and returns a false PASS:

```text
 2.22 GiB  (one C3 probe)    -> 200, 1/1 granted an upload action
19.98 GiB  (nine C3 probes)  -> 200, 9/9 granted an upload action
```

Owner:
[`archival_retirement_20260930.json`](../maintenance/inventories/archival_retirement_20260930.json).

**That check is now a pre-provider gate, not a fact someone remembers.**
`durable_capacity_gate` asks the same question at launch, at the measured
per-probe size (`2,384,236,592` bytes, from attempt66's own preservation
payload) and at the true nine-probe total, and refuses at `$0`. AGENTS.md names
this failure twice — C1 attempt 18 lost six probes to it, C3 attempt66 lost
nine — and nothing had ever asked.

**Padded tensor batching is not invariant; parallel B=1 item forwards ARE.**
Bitwise identical to sequential B=1 on both objects, repeatable, independent of
stream identity — but no useful speedup (best `1.034×`). The investigation is
MEASURED and its findings are derived, on branch `review/c3-operator-batching`, which is **not merged
into `main`**. In one line: a batched forward is not the same computation as a
solo one because a GEMM whose reduction is deep relative to its output width
reduces in a shape-dependent order, and in bf16 that moves an FFN top-k in most
layers. See
[C3 — the operator, the pilots, and the formal run](#c3--the-operator-the-pilots-and-the-formal-run)
below. **Nothing was changed in response** — no default, no identity semantics,
no operator definition.

*This paragraph used to end "C3 itself remains NOT STARTED and its `$25.00`
stage envelope is untouched", which contradicted this file's own stage ladder
two screens down. Both halves were stale: C3 executed on 2026-09-28, and the
`$25.00` envelope was superseded by the 2026-09-28 amendment — the live bound
is a `$30.00` per-session envelope with the ceiling derived from the selected
device's live price. C3's state is the ladder's row and
[Nine probes trained, no result](#nine-probes-trained-no-result); its money is
the summary table's `project cap` row. Neither is restated here.*

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
C3  COMPLETE / NO_GO                    <- canonical verdict, 2026-10-01
      three-arm preregistration         corrected and re-frozen
      attempt66 (2026-09-28)            9 trained, 0 scored, 0 preserved
      attempt75 (2026-09-29/30)         the authorized fresh measurement
        hardware qualification          L40S PASSED both digest gates
        three arm identities            all reproduce attempt66's
        nine formal probes              ALL TRAINED
        nine checkpoints                ALL PRESERVED   19.9844 GiB
        nine evaluations                ALL SCORED      protocol-uniform
        stage I aggregation             FAILED on the pod (C1 arm vocabulary)
      stage-I aggregation, off pod      AUTHORIZED and RUN at $0
      PRIMARY verdict                   NO_GO   UCB +0.005490 < SESOI +0.010
      incumbent after C3                B, unchanged; causal-KL does not promote
      B1 vs B3 protocol choice          a post-C3 maintainer decision, open
A-bsz3  DESIGNED, NOT FUNDED           <- the shortened design, 2026-10-01
      $0 padding/forward analysis       MEASURED  67 forwards -> 23
      implementation + driver           complete, toy-scale executed
      16-probe non-inferiority design   WITHDRAWN for scope
      step 1, structural/runtime        unrun; fits the engineering allowance
      step 2, <=3 treatment probes      conditional on step 1; formal book, overspent
FFN     NOT STARTED                     <- next after A-bsz3
D1/D2/D3  DIRECTIVE RECEIVED            <- global scoring/search; not designed
Stage-0 v2  DIRECTIVE RECEIVED          <- after the best D-method is established
C4  NOT AUTHORIZED
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
must not be spent. The project envelope did not move during that window; it has been `$400.0000` since the 2026-09-28 amendment.

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
`scripts/experiments/stage-1/phase_c2/tests/test_behavioural_continuation.py`.

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
  The check moved to `scripts/experiments/stage-1/phase_c2/tests/test_c2_behavioural_launch_governance.py`
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
[`full_search_space.py`](../../scripts/experiments/stage-1/phase_c2/full_search_space.py),
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
| phase | C0 **COMPLETE**. C1 — fixed-path ATTENTION isolation, **CLOSED by a `GO` verdict**. C2 — **CLOSED WITHOUT PROMOTION** (maintainer decision 2026-09-24): Search-1 DONE and FROZEN, the local Search-2 refinement WITHDRAWN, the full joint re-search COMPLETE with an accepted frozen Top-5, screening complete, and all twelve behavioural probes trained and scored — but the confirmation field mixed evaluation-protocol identities, so **no canonical promotion verdict is claimed and no new incumbent is named**. **B, the frozen C1 treatment, remains the accepted incumbent.** C3 (causal-KL isolation) **EXECUTED and returned no result** — nine probes trained, none evaluated, none preserved; the row in the stage ladder owns its state | [`phase_c2/plans/phase_c2_full_search_protocol.json`](../stages/stage-1/phase_c2/plans/phase_c2_full_search_protocol.json) · [`phase_c1/plans/phase_c_roadmap.md`](../stages/stage-1/phase_c1/plans/phase_c_roadmap.md) |
| replay | **MEASURED — 2/2 PASS**, for the third time (attempts 9, 17, 18). Passing replay is not a result: 9 and 17 are **NO DECISION**, pre-treatment aborts that measured no endpoint. Attempt 18 is the only attempt that decided anything | [`attempt18/closeout/outcome.json`](../stages/stage-1/phase_c1/runs/attempt18/closeout/outcome.json) |
| treatment, endpoint | **MEASURED** — six probes trained and six evaluated on the frozen battery; the frozen Stage-I rule returned **`GO`**. Figures in the block below | [`attempt18/evidence/c1_decision.json`](../stages/stage-1/phase_c1/runs/attempt18/evidence/c1_decision.json) |
| launch chain | **every C2 chain is consumed and nothing is prepared** — Search-1 attempts 1–4, baseline completion 5–8, and behavioural 1–14. **No C2 chain may be built:** the stage is closed and no further C2 scientific spend is authorized. No further C1 attempt is authorized or prepared either; a complete verdict ended that round. The **C3 batching-adoption pilot is authorized** (`$5.00` all-in) but has no readiness record, authorization document or bundle yet; **formal C3 has no chain of any kind** | [`phase_c2_baseline_completion/runs/attempt8/governance/`](../stages/stage-1/phase_c2_baseline_completion/runs/attempt8/governance/) |
| last attempt | **behavioural attempt14 — COMPLETE, `$2.0670`.** It reconstructed probe 11's evaluation for the P4 repair and reproduced the terminal state with a recorded two-prompt discrepancy, which is preserved rather than resolved. It also left a pod billing ~116 min behind a blocked artifact gate; that defect is repaired and accounted as PHB-HA-026/027 | [`attempt8/closeout/outcome.json`](../stages/stage-1/phase_c2_baseline_completion/runs/attempt8/closeout/outcome.json) |
| blocker | **NOTHING IS BLOCKED.** C2 is **CLOSED WITHOUT PROMOTION**: no probes owed, no canonical verdict claimed, no new incumbent, **B stands by absence of a valid challenger**, no C2 launch prepared, and **no further C2 scientific spend authorized** — remaining allowance under the `$25.00` stage envelope or the `$42.0000` campaign ceiling is not permission. **Formal C3 is FUNDED** by the 2026-09-28 amendment (project cap `$400.0000`, per-session envelope `$30.0000`, formal allowance `$55.0000`) and its 3-arm/9-probe design is preregistered and hash-bound; the envelope is not the grant, and C3's one-use authorization takes the ceiling **derived from live `securePrice` at issuance**. All three C3 engineering pilots are closed and are not formal C3 | [`budget/decisions.md`](../budget/decisions.md) · [`c3_preregistration.json`](../stages/stage-1/phase_c3/plans/c3_preregistration.json) |
| spend | owned by the budget block below | [`budget/ledger.md`](../budget/ledger.md) |

## C3 — the operator, the pilots, and the formal run

**Formal C3 has now executed and is stopped without a result** — see
[Nine probes trained, no result](#nine-probes-trained-no-result) at the end
of this section, which is the part a reader wants first. Everything before
it is the engineering that got there: the operator-topology refactor, the
batch-invariance investigation and the three adoption pilots. Those were
written while C3 had not yet started and are kept as they were measured.

**Branch `review/c3-operator-batching`.** `main` was `ab53ba14` when this
section was written.

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

**This repository contains FOUR runtimes, and which one a number came from is
the point.** The formal operator search and every recovery probe execute under
`/opt/train` — python 3.12, **torch 2.11.0+cu128, transformers 5.13.1**,
installed offline from the relay wheelhouse
(`POD_IMAGE['remote_python']`, and C1 attempts 17/18 evidence). Rollout runs in
a deliberately separate `/opt/vllm` at torch 2.13.0+cu130 / transformers 5.15.0
— `continuation_b/runs/attempt5/continuation_evidence.json` shows both inside
one formal run. The engineering CUDA validations run the image's own python at
**torch 2.9.1+cu130 / transformers 5.17.0** (`c2_full_search_cuda`,
`c2_full_search_perf`). The rejected a4 run used a fourth: image `1.0.3-cu1281`,
torch 2.9.1+cu128, and `pip install transformers` with no version pin.

**The root-cause investigation is the current work.** Ceiling **`$3.00`
cumulative, inheriting the `$0.0822`** already spent; `$2.9178` remains and no
part of it is C3's envelope. Its executable is
`scripts/validation/batch_invariance_diagnostic.py`, and the point of it is
that every number a conclusion rests on is emitted by that file: the verdict is
COMPUTED by `derive_conclusion()` from the stage outputs rather than written
beside them, and that function is tabled and mutation-checked in
`scripts/experiments/stage-1/phase_c3/tests/test_batch_invariance_conclusion.py`. Records:
[`scope.json`](../stages/stage-1/phase_c3/investigations/batch-invariance-root-cause/v1/scope.json),
[`authorization.json`](../stages/stage-1/phase_c3/investigations/batch-invariance-root-cause/v1/authorization.json),
[`campaign.json`](../stages/stage-1/phase_c3/investigations/batch-invariance-root-cause/v1/campaign.json).

**Five `$0` CPU rehearsals ran the real executable before any pod existed**, and
found five defects in it — two of them verdict defects that would have been
paid for. The largest: `derive_conclusion` read "not every backend diverges" as
"some backend is exact", and returned locus `attention_backend_kernel` for a run
in which nothing diverged at all.

**The measurement was made: six reports, one L40S, `$0.1694`, 9.2 minutes.**
Attempt `bi_20260926_d3`, pod `hdwsp4btdq5t1n`, teardown provider-confirmed, all
six from ONE executable (`6f096fc5…`). It follows `d2`, which answered the root
cause on four reports for `$0.1005` and whose numbers `d3` reproduces — a
cross-session check that cost `$0.10` and was worth it. Cumulative diagnostic
spend `$0.4006` of `$3.00`. Owner:
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

**It is not a subsample artefact, and the prediction that said so was half
wrong.** Those numbers read 8 of the mixture's 67 items, so a prediction was
recorded and committed BEFORE the check
([`fullmix_prediction.json`](../stages/stage-1/phase_c3/investigations/batch-invariance-root-cause/v1/fullmix_prediction.json)):
the relative drift should fall by roughly `sqrt(8) ≈ 2.8×`, and the selection
should still move. At all 67 items (59,830 tokens):

| | drift at bs=4 | fell by | layers moved, keep 0.50 |
| --- | --- | --- | --- |
| parent | `1.094e-03` → `4.183e-04` | **2.62×** | 34/36 → **34/36** |
| a4 596M | `3.932e-02` → `2.824e-03` | **13.92×** | 26/28 → **21/28** |

The `2–4×` prediction holds for the parent and is **wrong by 3.5× for the
596M** — recorded as a miss rather than reinterpreted. The consequence claim
survives on both: the minimum relative cutoff margin also grew (`1.9e-07` →
`2.6e-06` on the parent), and the drift still exceeds it by two orders of
magnitude.

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

**It reproduces exactly across pods.** The four labels `d2` and `d3` share
agree to every digit on two different pods hours apart — `A_vs_E` relative
`1.218718e-02` both times on the parent, the same 34 of 36 layers, the same
statistic drift, the same locus. Within-run repeatability was already
bit-identical; this is the stronger claim, and it is what makes "deterministic
divergence" a fair description rather than a contradiction.

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

### The split-K causal control — the intervention that had never been run

**Review found a material defect in the diagnostic, and it is mine.**
`torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False` does
**not** disable split-K. torch's parser is explicit — `if isinstance(value, bool):
return value, True` — so the boolean form turns off reduced-precision
accumulation and leaves `allow_splitk = True`. Only the tuple form reaches the
second flag. My stage called itself "turn off bf16 split-k reduction" and its
`_reading` said the knob did not remove the divergence, a conclusion drawn at
the LOGIT level while the per-projection table underneath already showed
`attn_out` and `ffn_out` going **bit-exact** under the boolean and k/v halving.
I reported the summary and missed my own data.

**The three states are now named** so the ambiguity cannot recur: `default`,
`reduced_precision_off_splitk_on` (what actually ran), and
`reduced_precision_off_splitk_off`. Historical reports are unchanged.

**Two facts the runtime supplied, not my reading of it:**

* torch **2.9.1+cu130 cannot express the tuple at all** —
  `set_allow_bf16_reduction_cublas expects a bool`. Recorded UNSUPPORTED.
* torch **2.11.0+cu128 accepts `(False, False)` at the setter and refuses at
  the first GEMM** — `allow_splitk=False requires the cuBLASLt backend`, raised
  inside `F.linear`. Attempt `bi_20260926_d4` (`$0.0934`) lost four stages to
  this because `supported` meant "setattr returned".

**Three repairs, and the third is the one that matters scientifically.**
`supported` now means a real bf16 GEMM executed under the policy. The BLAS
library is selected explicitly and its readback checked, because
`preferred_blas_library` can accept a name and leave the backend where it was.
And a **fourth control**, `cublaslt_defaults`, isolates the backend switch
alone — `allow_splitk=False` requires cuBLASLt, so the intervention changes two
things, and without that control an improvement could not be attributed to
either. `improvement_attributable_to` is a derived field for exactly that.

**The control ran, and the answer is SPLIT_K_PARTIAL.** Derived by
`batch_invariance_finding.py` into
[`splitk_finding.json`](../stages/stage-1/phase_c3/investigations/batch-invariance-root-cause/v1/splitk_finding.json)
from five reports across three pods (`d4` `$0.0934`, `d5` `$0.1543`, `d6`
`$0.0424`).

**Fixed — every bare GEMM becomes bit-exact**, and the fourth control was
needed to say why. On the parent the two changes repair **disjoint** sets:

| control | BLAS | split-K | still shape-dependent |
| --- | --- | --- | --- |
| default | cuBLAS | on | attn_out, ffn_out, k, v |
| reduced-precision off | cuBLAS | on | k, v |
| cuBLASLt alone | cuBLASLt | on | attn_out, ffn_out |
| **both off** | **cuBLASLt** | **off** | **none** |

On the 596M the backend alone suffices. So neither "split-K was the cause" nor
"cuBLASLt was the cause" is right on its own, and the derived field reads
`both_changes_are_needed_and_repair_disjoint_sets`.

**Not fixed, and one of these is disqualifying.** The **historical solo output
changes** under the new policy — `max_abs 1.812` on the parent, argmax
`0.9873`. That is the condition the review named: batch and solo agreeing at a
*third* value is not invariance. FFN selection still moves **31 of 36** layers
(from 34). And batching buys nothing on this workload anyway: `bs=4` runs at
`0.946×` the speed of `bs=1` for the statistics collector, while split-K-off
itself costs only `1.005×`.

**ATTENTION was never the problem** — 0 of 28 layers on the 596M, 1–2 of 36 on
the parent, at every control including the default.

**The residual is the PADDING path, and it enters inside attention.** Under
split-K off on the parent: mask presence exact, neighbour content exact,
**equal-length batch now bit-identical to solo**, ragged padded batch still
`2.312`. The first divergent tap of 362 is **`L00.out.attn_out`** — the q, k
and v projections are exact, so the difference is introduced by the attention
computation over a padded width, not by any linear. On the 596M the
equal-length batch is not yet exact either, so its residual is
`batch_dimension_and_padding`.

**API facts, from the runtimes rather than from reading them.** torch
`2.11.0+cu128` accepts the tuple and exposes a readable `..._split_k`, but
`allow_splitk=False` **requires the cuBLASLt backend** — it accepts at the
setter and raises inside `F.linear`, which is why every policy here is proved
by running a real bf16 GEMM under it. torch `2.9.1+cu130` cannot express the
tuple at all: `set_allow_bf16_reduction_cublas expects a bool`. Recorded
UNSUPPORTED.

### Parallel independent B=1 item forwards — EXACT, and no faster

**A different architecture from padded tensor batching**, and the distinction
is the point: each item is presented to the model as `[1, T_i]` exactly as it
was historically — no padding, no batch dimension, the same attention path, the
**historical** numerical policy with no split-K intervention. Only the schedule
changes: N forwards submitted to N CUDA streams, one synchronize.

**Verdict `PARALLEL_B1_EXACT_BUT_NO_SPEEDUP`, on both objects**
(`bi_20260926_d7`, pod `y1rg8rsl4qy4lp`, `$0.0412`, 26 s of compute).

The scientific requirement passes completely:

| check | parent 4B | 596M |
| --- | --- | --- |
| logits vs sequential B=1 | **bitwise identical**, `max_abs 0.000e+00` | **bitwise identical** |
| 3 parallel waves repeat | bitwise | bitwise |
| identity / reversed / rotated stream assignment | all bitwise | all bitwise |
| causal KL vs scalar oracle | `1.101e-07` rel, inside the `1e-6` bound | `1.084e-08` |
| causal-KL item ranking | identical | identical |

The causal-KL bar is the reducer's own validated contract rather than
bitwise — that comparison is the scalar oracle against the batched reducer, two
reductions of the same logits, and the bound was declared before the run. The
**forwards** are what must be bitwise, and they are.

**But concurrency buys nothing here.** Best speedup `1.034×` on the parent and
`1.053×` on the 596M, against a `1.10×` bar declared before the measurement:

| concurrency | parent seq → par | 596M seq → par |
| --- | --- | --- |
| 1 | `0.0412 → 0.0435` (`0.947×`) | `0.0299 → 0.0399` (`0.750×`) |
| 2 | `0.0792 → 0.0779` (`1.018×`) | `0.0619 → 0.0587` (`1.053×`) |
| 4 | `0.1615 → 0.1562` (`1.034×`) | `0.1198 → 0.1190` (`1.007×`) |
| 8 | — | `0.2356 → 0.2471` (`0.953×`) |

Peak VRAM rises modestly: parent `8.13 → 8.96 GiB` at c=4, 596M `2.59 → 3.54`
at c=8. The likely mechanism is simply that a 4B forward already saturates the
L40S, so concurrent streams find no idle SMs — there is nothing to overlap.

**Scope, so this is not over-read:** the throughput figures are for **plain
forwards only**. The real calibration workload also carries the float64
`residual_sqsum` accumulation, which is not in this measurement. Phase 3 (the
statistics path with per-item buffers merged in item order) was **not built** —
the review's own outcome for this result is to return with the throughput
evidence and decide whether the concurrency machinery is worth retaining.

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
booking.

**And a correction to my own correction.** I wrote that the `torch 2.9.1+cu130 /
transformers 5.17.0` pair named in the investigation brief did not describe the
formal environment. That is true, but I first reported finding no record of
transformers 5.17.0 at all, and there is one: it is exactly what the
`c2_full_search_cuda` and `c2_full_search_perf` engineering validations ran
under. An a4-style CUDA validation is that kind of session, so naming that
runtime was reasonable — it is simply not the one the operator search executes
in. Both were measured here, and they agree.

### The C3 batching-adoption pilot: implemented, priced, NOT launched

The acceptance question moved, by maintainer decision: not *does padded
batching reproduce every top-k choice* but *is the B4 causal scorer materially
faster while producing a practically equivalent downstream model after frozen
recovery*. Two arms differing in one config value.

**`attention.causal_kl_v1` exists** at
[`operators/attention/gqa/causal_kl.py`](../../src/aadistill/initialization/operators/attention/gqa/causal_kl.py).
Each query head is scored by one-shot causal ablation — forward KL of the
parent against the parent with that head's `o_proj` column block zeroed,
aggregated domain-balanced exactly as DEPTH aggregates — then the shared
per-GQA-group top-k selects. One shot, no rescoring, no joint search.

Three properties are load-bearing and each is tested by execution, not
assertion:

* **zeroing the columns equals deleting the head, bit-for-bit.** Checked
  against an independent deletion mechanism (zeroing the head's *input* to
  `o_proj`), because the same mechanism twice would be a tautology. The
  columns are restored in a `finally`, and a full scoring run leaves every
  parameter of the model it was handed unchanged.
* **the batch size is hashed, not runtime.** Alone among the operators it
  reads `calibration_forward_batch_size` from its own step config and never
  `ctx.execution` — an AST check enforces that the source cannot mention
  `micro_batch_size`. An undeclared batch size is `1`, never the
  process-wide default of `4`.
* **it joins nothing by being imported.** Absent from `BUILTIN_OPERATORS`,
  registered only by an explicit `register()`, and the regenerated C1
  executable closure still names 105 files without it.

**The prefix had two real defects, both found by the maintainer and both
fixed.** `prefix_steps()` gave all three pre-ATTENTION operators
`calib.domain_balanced@v1`; WIDTH actually ran against
`calib.reasoning_heavy@v2`, so the replay would have reconstructed a
*different* parent — refused by the digest gate, but only after paying for the
replay. And `step_operator_config` let a step override `n_calibration_items`,
which would have PLANNED an operator against 8 items while EXECUTING it
against 67; it now fails closed, and the test that had blessed the override is
reversed.

**The B=1 pin is now proved by running the pilot's own path.** `replay_prefix`
is the single function that materializes anything for this pilot, it takes no
`execution` parameter, and the evidence reads the batch size each prefix
operator *recorded* after a real toy execution. The earlier version asserted
that a call it had just written passed a `1`, which a pilot that forgot the
argument entirely would also have passed.

**Pricing is conservative and arm-identical**, derived by
[`phase_c3/pricing.py`](../../scripts/experiments/stage-1/phase_c3/pricing.py) from the
frozen mixture through the real loader and the real grouper:

| | groups | item-forward equiv. | physical invocations | padded positions | pad/valid |
| --- | --- | --- | --- | --- | --- |
| B1 | 67 | 60,099 | 60,099 | 0 | 0.0000 |
| B4 | 17 | 60,099 | 15,249 | 21,978 | 0.3673 |

67 items, 59,830 valid tokens, 28 layers x 32 heads = 896 ablations plus one
reference. `operator_cost` cannot price the two arms differently — it has no
config parameter — so ~1130 GPU-seconds per arm, about `$0.68` for both at an
*estimated* `$1.09/h`. **That is not a quote**; the live `securePrice` is
re-queried immediately before acquisition, and
`L40S_MEASURED.price_per_hour_usd = $0.99` is historical profile metadata that
must never be used as one.

**The padding overhead is the pilot's most useful pre-run number.** B4
computes 36.7% more token-positions than B1. A 1.25x speedup gate is therefore
asking the packing to win back that overhead *and* a quarter again — which is
exactly why the gate was predeclared rather than chosen afterwards.

**It is visible and stoppable.** 897 corpus passes is tens of minutes to
hours, and the first version had neither a progress line nor a deadline check
— the pair whose absence let `depth.causal_kl_greedy_v1` run 10.78 h against a
3.0 h budget, silent for 10 h 47 m. One line and one `ctx.deadline.check()`
per (group, layer): 28 per group rather than 896. A stop names the group, the
layer and the forwards completed, and — tested by stopping it — leaves every
parameter of the model it was handed unchanged.

### The pilot RAN, and B4 is slower: `B4_NOT_WORTH_ADOPTION_PILOT`

**Measured 2026-09-27, `c3pilot_20260927_a5`, `$2.1113`, pod `m8t9tf8ftqlmh3`.**
Owner: [`result.json`](../stages/stage-1/phase_c3/pilots/batching-adoption/v1/result.json).

|  | scorer | physical invocations | item-forward equiv. | padded positions | peak VRAM |
| --- | --- | --- | --- | --- | --- |
| causal-B1 | **2190.2 s** | 60,099 | 60,099 | 0 | 3.94 GiB |
| causal-B4 | **3187.2 s** | 15,249 | 60,099 | 21,978 | 11.76 GiB |

**Speedup 0.6872× against a predeclared 1.25× gate**, so the gate is not met
and no recovery was triggered — the speed gate is first, and that ordering was
frozen before either clock was read. Issuing a quarter of the kernel launches
did not make it faster: the pre-run derivation had already shown this protocol
computes **36.7% more token-positions** than B1 on this mixture, and the
measured penalty exceeds even that.

**WHAT THIS DOES AND DOES NOT ESTABLISH.** The measured protocol was
`batch_size=4` **and** original frozen item order **and** consecutive grouping
**and** right padding to each group's longest item — four things varied at
once. The verdict belongs to that protocol. It does **not** establish that
padded model batching is unhelpful in general: the generic batcher states
plainly that it never sorts by length, so a smaller batch with deterministic
length-aware packing is an obvious degree of freedom that has not been
measured. That is what the packing-optimization pilot below tests.

**The frozen parent reproduced exactly, twice.** One replay per pilot, at B=1:
DEPTH 1225.9 s, FFN 45.2 s, RESIDUAL_WIDTH 34.0 s → `eea90c91346a0745…` in
22.1 min, and again on attempt a4 in 21.5 min. Both arms re-identified that
one checkpoint from disk before scoring (`premise: verified`). This is the
first time this tree has been shown to reconstruct the C1 pre-ATTENTION parent
bit-for-bit.

**The head maps do differ, and far less than the layer count suggests:**

```text
layers          5 / 28        <- the number that overstates
GQA groups     16 / 224
retained slots 16 / 448       = 3.57%
```

All 16 changed selections are "wide" crossings, but the margins are small in
absolute terms (1.94e-06 … 6.76e-03) and 15 of the 16 sit in layers 1–4.
Rank correlation is 0.981 overall (per-layer min 0.793); score drift has
abs_max 0.284 with a median relative drift of 0.24% and a p95 of 27.5%. **This
is descriptive only** — the pilot may not and does not produce a GO/NO_GO, a
confidence interval or an incumbent.

**Four attempts preceded it, all engineering, all repaired, each caught
earlier than the last:** a1 `$0.7209` (staged one of the two calibration
mixtures; found 24 min in at WIDTH), a2 `$0.0000` (provider capacity — and my
pod-id regex matched the word `specifications` out of the refusal), a3
`$0.0391` (`ssh` ate the push list; **a1's repair caught a1's failure in 2
minutes instead of 24**), a4 `$0.4765` (the prefix reproduced `eea90c91…`,
then the parent loaded on cpu against a `cuda:0` path — the CPU-rehearsal
blind spot). Every pod torn down and confirmed.

**Pilot spend `$3.3478` of the `$5.00` ceiling. Project `$346.0499` of
`$370.00`.**

**Formal C3 was NOT STARTED and had no frozen seed set when this pilot ran.** Both changed afterwards: the 3-arm design was preregistered with three mechanical seeds, and the 2026-09-28 amendment funded it. The old `$25.00` envelope was never current permission and still is not. The pilot seed
`1139220455` is pilot-only.

### The packing-optimization pilot: the screen is measured, the full scorer is not funded

**Measured 2026-09-27, `c3pack_20260927_b2`.** Owner:
[`screen_result.json`](../stages/stage-1/phase_c3/pilots/packing-optimization/v1/screen_result.json).

The previous pilot's verdict was about ONE protocol. `micro_batches` states
plainly that it never sorts by length, so consecutive B4 paid 36.73% padding.
The `$0` table said sorting would fix most of that, and the screen measured
what it buys — 67 items, 32 heads, layers `[0, 13, 27]`:

|    | B | packing | wall | invocations | valid pos/s | peak VRAM | vs P0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P0 | 1 | original | 292.5 s | 6,499 | 19,633 | 3.94 GiB | 1.000 |
| P1 | 2 | original | 289.3 s | 3,298 | 19,852 | 6.55 GiB | **1.011** |
| P2 | 2 | sorted | 252.6 s | 3,298 | 22,737 | 6.53 GiB | **1.158** |
| P3 | 4 | sorted | 250.7 s | 1,649 | 22,912 | 11.30 GiB | **1.167** |

**Packing buys the speed; batch size alone does not.** P1 and P2 issue the
same 3,298 invocations over the same items and differ only in which items
share a forward: 1.011× against 1.158×. And **B4 failed on its packing, not
its batch size** — P3 is 1.167× faster where consecutive B4 measured 0.687×.

P3 advances the 1.10× screen gate, by wall time alone. **That is not the
adoption gate**: 1.25× on the full 28-layer scorer remains unmeasured.

**The ±5% comparability rule fired.** This session's P0 is 1.2467× the scaled
prior full B1, so the prior may not serve as the reference and the chain needs
**two** full scorers, a fresh B1 and the candidate.

**STOPPED under §18.** Ceiling `$3.00`, spent `$1.2760`, `$1.6240` after the
teardown reserve = 89 min at `$1.09/h`. The chain needs 100–109 min. Running
only the candidate would be dropping the fresh B1 the comparability rule
requires, so the pilot stops rather than silently dropping it.

Three subruns: b1 `$0.0000` (the launcher asked its own driver for a mode
that lived in the other driver; fail-closed, no pod), b2 `$0.7897` (parent
verified a third time, screen complete, then the full scorer hit the *same*
unprepared-items line the screen had hit earlier in the session — one fix,
one call site, and the toy path supplies its own items so the real branch
never ran), b3 `$0.4863` (the toy preflight hung for 23+ minutes against 16 s
on the batching pilot's pod, and was unbounded; terminated by removing the
pod). All torn down, provider confirms **pods 0, volumes 0**.

**To resume**, a new ceiling of about `$2.00` covers the parent replay and
both full scorers. The measured screen is reusable — pushable with
`SCREEN_FROM`, sha256 recorded — so it is not re-paid for. Three repairs are
required first: bound the preflight, run it on `cuda:0` or pin its thread
count from the cgroup quota, and stream its output.

### Packing v2: `PACKED_BATCH_NOT_WORTH_ADOPTION` at 1.1884×

**Measured 2026-09-27, `c3pack2_20260927_v6`, `$2.6378`.** Owner:
[`v2/result.json`](../stages/stage-1/phase_c3/pilots/packing-optimization/v2/result.json).

The counterbalanced screen — round A `R0→R2→R3→R4`, round B its exact
reverse, selection on the pooled sum:

|    | round A | round B | pooled | var | vs R0 | peak VRAM | inv |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R0 B1 orig | 293.33 s | 293.18 s | 586.509 s | 0.05% | 1.000 | 3.94 GiB | 6,499 |
| R2 B2 sort | 252.92 s | 252.61 s | 505.523 s | 0.12% | 1.160 | 6.53 GiB | 3,298 |
| **R3 B3 sort** | 246.33 s | 246.23 s | **492.553 s** | 0.04% | **1.191** | 9.12 GiB | 2,231 |
| R4 B4 sort | 250.41 s | 250.61 s | 501.018 s | 0.08% | 1.171 | 11.30 GiB | 1,649 |

**B3 wins, and it had never been measured.** v1's single pass put R4
marginally ahead; pooling reorders them. R3 is fastest *and* lower-memory
than R4, so the near-tie rule confirms the choice rather than overriding it.

**Position was not the confound.** Variation is 0.04–0.12% everywhere, and
R4 ran position 4 then position 1 for 250.41 s and 250.61 s. On this card
position is worth ~0.1%, so v1's 0.76% gap was real but small —
counterbalancing did not correct a bias, it produced a clean enough number
to reorder two genuinely close candidates.

**The full pair:** fresh full B1 **2720.66 s**, R3 **2289.37 s** →
**1.1884×** against the 1.25× gate. **Not adopted.** No structural
comparison (the gate did not clear) and no recovery (not authorized in v2);
the pilot seed `1139220455` remains unconsumed.

**Two things the protocol earned:**

* **The ±5% comparability rule changed the sign of the answer.** The fresh
  B1 measured 2720.66 s against the prior session's 2190.17 s — 1.242×,
  matching the 1.2467× v1 predicted. The stale reference would have given
  `2190.17/2289.37 = 0.957×`, reporting the candidate as *slower*. The rule
  did not tighten a number; it prevented a wrong conclusion in the opposite
  direction.
* **The three-layer screen predicted the full scorer to 0.2%** — 1.1908×
  against 1.1884×. A cheap screen over all items and all heads is a very
  good predictor of the full ratio on this workload.

**So: length-aware packing is worth about 19% of a full causal-KL scorer.**
Real, reproducible across two sessions, and below the threshold set for
adopting it.

Campaign `$2.7048` of `$4.00`; project `$350.0307` of `$370.00`. Six
acquisition attempts: four L40S capacity refusals at `$0.00`, one `$0.067`
where the repaired preflight caught a toy root on the wrong device in five
seconds, then this one. All torn down, provider confirms **pods 0,
volumes 0**.

### Nine probes trained, no result

**The headline.** `attempt66`, on secure **L40S**, 12.5 hours, **`$13.67`**.
Stages B, C, D, E, F and G all passed — including **both** frozen digest
gates — and **all nine formal recovery probes trained**. Then stage H exited
2 before evaluating any of them, and the nine trained checkpoints were lost.
So C3 has a complete trained matrix, **zero** evaluations and **zero**
surviving weights, and **no C3 verdict is claimed**.

```text
B  teacher fetch + verify      PASS   3 shards
C  register operator           PASS   both experimental implementations
D  replay parent               PASS   reproduced eea90c91…
E  replay incumbent B          PASS   reproduced 53e30566…
F  materialize arms            PASS   three initializations, built once each
G  recovery probes             PASS   9 of 9 trained
H  evaluate                    FAIL   argparse exit 2, nothing evaluated
```

**Stage H, in full:**

```text
score_c1_confirmation.py: error: argument --arm: invalid choice:
'A_incumbent' (choose from 'incumbent', 'treatment')
```

The same C1-constant class as every earlier defect, in the one place the
static checks could not reach: an argparse `choices` constraint inside a
**separate script** the driver shells out to. The driver passed C3's arm id;
the scorer accepted C1's two role names. Nothing in the driver's imports,
names or digests can see another process's parser.

**Repaired — on the caller's side, after I first repaired the wrong file.**
My first fix widened the scorer's `--arm` to accept any label. That was
wrong: `score_c1_confirmation.py` is named by **C1's frozen executable
closure and its execution preregistration**, and by C3's closure and
C2-behavioural's grant proposal. Editing it moved a digest a **completed GO
experiment** binds, and the closeout suite went from 11 failures to 23 —
thirteen new ones, the whole candidate/closure/preregistration family,
every one caused by that edit. The scorer is restored byte for byte.

The repair now lives where C3 owns it: **the driver no longer passes
`--arm` at all.** That flag's vocabulary is C1's two *roles*; C3 has three
*arms*. C3 needs nothing from it — `--label` is the probe id, which carries
the arm, and the driver keys `(arm, seed)` off the **training** record,
never off the scorer's output.

**The gate that should have caught it already existed.** The driver has a
`scorer_preflight` stage. It passed on attempt66. It sent **four** flags
where the real stage-H call sent **eight** — and a flag the preflight never
sends is a flag it cannot prove the scorer accepts. That is the actual
defect: not a missing gate, a gate supplying its own inputs. It now sends
every flag the real call sends and fails loudly on argparse exit 2
specifically. Verified by running the repaired command for real against the
frozen battery: exit 1, reaches `no generations for [...]`, zero argparse
errors.

Two `$0` checks added: no call site may pass `--arm`, and the preflight's
flag set must be a **superset** of the real call's. Both by AST over every
call site — the driver invokes the scorer twice, and an index-based slice
grabbed the preflight one. This does **not** authorize a rerun; it means the
next authorized attempt does not buy this defect again.

**The checkpoints.** All nine durability uploads were refused:

```text
BadRequestError: Private repository storage limit reached
```

The mechanism behaved exactly as designed — it never raised, it disturbed no
stage, and it recorded each unit's identity, its inputs' hashes and a content
hash together with the exact reason. It preserved nothing, because the
backend is full. AGENTS.md documents this failure twice and says to confirm a
durable backend has **room** before a long experiment that will produce large
artifacts. That check was not run before launching. Nine trained probes at
2.22 GiB each are the cost.

**One thing of scientific value survived.** Stage F built all three
initializations from the same verified frozen pre-ATTENTION parent, once
each, and they are distinct:

```text
A_incumbent    attention.activation_importance_v1   53e30566…  (= frozen B)
B_causal_b1    attention.causal_kl_v1   b=1, original_order_v1  6eb231ca…
C_causal_b3    attention.causal_kl_v1   b=3, length_sorted_v1   dec88683…
```

`B_causal_b1 ≠ C_causal_b3` at full scale on the real operator: the
calibration batch shape and packing order really do change which heads
`causal_kl_v1` selects, so the secondary contrast is **not** degenerate and
B3 cannot be assumed to stand in for B1. This is a **structural** fact about
the initializations — it is not a behavioural result and decides nothing
about the operator, which is what the nine lost evaluations were for.

**The blocker is live, not historical.** Re-asked at `$0` on 2026-09-29 via
the LFS batch endpoint against `AlphaAvatar/aadistill-artifacts`, with **fresh
random oids** — an oid that already exists dedups and returns success
regardless of quota, which is a false pass:

```text
2.22 GiB  (one probe)    -> 403 Private repository storage limit reached
19.98 GiB (nine probes)  -> 403 Private repository storage limit reached
```

A fresh nine-probe attempt launched today would lose its checkpoints the same
way. This is the call that should have run *before* attempt66; it costs
seconds and moves no bytes.

**Why this stops here rather than retrying.** Formal measurement has begun:
nine probes trained. §13 freezes the session at that point and forbids
restarting the matrix, splicing probes, substituting seeds or pooling
attempts unless an existing resume contract supports it. No resume contract
covers *re-evaluate checkpoints that no longer exist*, and inventing one is
explicitly forbidden. Buying storage or permanently deleting historical LFS
objects is a maintainer decision, never an autonomous repair.

**The acquisition history.** Seventy attempt identities; **three** ever
created a pod.

| attempt | reached | outcome |
| --- | --- | --- |
| 1–6 | pre-provider gates | six distinct C1 constants in the ported chain, each fixed and covered by a `$0` preflight check |
| 7 | **a pod ran** | driver refused C1's authorization by type. `$0.15`, torn down |
| 10 | **a pod ran** | never produced an SSH endpoint in 15 min. `$0.26` |
| 49 | *orphaned* | I killed the loop's tmux session mid-launch; its pod billed 46 min unattended before a routine poll found it. **`$0.8617`** |
| 61, 65 | stage D gate | parent came back `d6d8d7ee` / `4cf33ed0`, not `eea90c91` — the missing execution config |
| **66** | **stage H** | **nine probes trained**, evaluation failed, checkpoints lost. **`$13.67`** |
| 67–69 | provider | capacity refusal at `$0`; chains consumed and closed |
| 70 | grant only | the loop was stopped here; PREPARED, never authorized |

**The defect that cost attempts 61 and 65** was the material one:
`materialize_fixed_path` was called without `execution=`, so DEPTH ran at the
repository default micro-batch instead of the plan's pinned `micro_batch_size:
1`. L40S GEMMs reduce shape-dependently, so round 7 chose layer 21 over 17 by
a margin of `1.403e-03` and the parent digest came out wrong. Runtime knobs
are deliberately not hashed into any state id — which is exactly why the plan
must pin this one and the driver must apply it. No digest catches it until
the digest itself is wrong.

**Money.** `$13.67` for attempt66; **phase_c3 total `$16.6233`**. Cumulative
project spend **`$366.654`**, remaining **`$33.346`** of the amended
**`$400.00`** cap; formal allowance **`$32.3824`** of `$55.00` remaining.
Provider state verified clean: **pods 0, network volumes 0**. Owners:
[`derive_budget.py`](../../scripts/consolidate/derive_budget.py) and
[attempt66's closeout](../stages/stage-1/phase_c3/runs/attempt66/closeout/outcome.json).

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

## The full suite is not green: 11 failures, and the count is trustworthy

**Closing measurement, D1 design round: 11 failed, 5683 passed, 229 skipped,
ZERO errors** in 42m54s, on the clean tree at `fc73c09a`. Exactly the eleven
documented failures, as an **identical node-id set** to the previous run — no
new ones, and none of the eleven fixed. `5555 → 5683` passed is the 128 tests
the round added. The set was diffed against the `<details>` list below
programmatically rather than read off, which is how the unlisted eleventh
below was found.

*Earlier closeouts measured the same eleven at 5555 and 5392 passed. Not
restated beyond that: a count belongs to the tree it was taken on.*

**THE FIRST FULL SUITE OF THIS ROUND READ 28, AND 17 WERE MINE.** Five causes,
not seventeen — and three were records the round owed rather than code defects:

* **the executable closure went stale a third time** (12 of the 17).
  `derive_closure.py --write` ran, then three more core and script files
  changed, and every grant-issuing gate correctly refused a closure that no
  longer described the tree. Running the convergence tool *before* the last
  edit is not running it.
* **the C2 evaluator lineage owed a second entry for `planning/metrics.py`.**
  The frozen baseline-completion protocol binds four evaluator files BY
  CONTENT, and the test asserts the drift SET equals the DOCUMENTED set — the
  mechanism working. Making the global state metric target-aware *is* a change
  to that evaluator, so the obligation was to record it, with the superseded
  prefix kept rather than overwritten. **The protocol stays frozen exactly as
  it is, and that is the point:** a `B` measurement taken under a target-aware
  policy measures a different quantity and must not join the old series.
* **`phase_d1` had no stage-index row.** A directory under a stage with no row
  is the drift the index exists to prevent, so the row was owed before the
  directory existed.
* **the snapshot's size contract, and a word a gate reads.** `current.json`
  stood at 13,739 of 14,000, so 261 bytes of headroom had to absorb a fourth
  live subject and could not. `a_bsz3` was consolidated to its live facts
  first — A3 is terminal and its closeout owns the figures — and the ceiling
  then moved to 15,500 with the arithmetic in the guard's own docstring. The
  squeeze also dropped "Top-5 **ACCEPTED** and FROZEN" and the attempt-4 gate
  refused, which is the **third** time that squeeze has broken that gate.
* **one phrasing lock of the round's own making.** A test written this round
  matched the sentence "collision A3 found"; a later pass removing a
  campaign's experiment label from reusable core rewrote it. The refusal was
  still correct and the test had locked the wording. It asserts the content
  now.

**The eleventh documented failure is named below for the first time.** The
`<details>` block enumerated ten while the prose said eleven, so attributing
`test_the_headroom_verdict_matches_what_plan_session_actually_does` cost a
HEAD-worktree run to establish it was not this round's. A list that claims to
enumerate the set should be the set.

**An earlier full suite in this round read 15, and four were mine.** Each was
one mistake: a producer changed and its consumers not enumerated. Adding
`host_not_admitted` to the runner's redrawable set broke a test that pinned
the set as the exact literal `('cold', 'no_endpoint')`; pointing `latest_run`
at `phase_a3` broke two tests under `tests/pod` and one under `tests/runtime`.
All four are repaired by asserting the property rather than the coincidence,
and the redrawable one is mutation-checked in both directions.

**Repairing them found the same blind spot a THIRD time, in a second
producer.** `record_run_index` also read only `classification`, so all 38
`phase_a3` and 17 `phase_c3` runs were filed under "predates the run-manifest
convention" while every one of them states a `status` — the index said nothing
about why a run that *finished* had no manifest. The tables now live in
`scripts/consolidate/closeout_reader.py` and both producers import them, so a
fourth family needs one edit in one place.

**Two derived records went stale twice each and were regenerated, not
waived** — `skip_predicate_audit.json` after each batch of test edits, and
`logs/stages/index.json` after the new analyses appeared. A third failure was
a real omission, and the same one as last round: trimming the snapshot under
its 14 KB contract dropped the word the attempt-4 consistency gate requires,
that C2's Top-5 ruling is recorded as **accepted** rather than owed.

**A partial selection's total is not comparable to the full suite's.** A
convergence run over five test directories read 22 failed with 15 errors, of
which the bulk were `phase_c2_full_search_chain` collection errors — the
documented order-dependent inflation from process-global operator registry
pollution, which appears and disappears by collection order. Per-file isolated
counts are the trustworthy reading mid-round; only a full run is comparable to
a previous full run.

**The order-dependent inflation is gone from the full suite.** It used to
report 22–29 failed with 15–20 errors, of which only eleven were real. The
cause was process-global operator registry pollution: the C3 launcher's
`spec()` registers the experimental ATTENTION implementations, the
launcher-loading tests never unregistered, and C2's joint-space enumeration
then met `attention.causal_kl_v1` — which its cost model has never measured —
and raised `CostModelError`. `BeamSearch._allowed_impl_ids` falls back to
*every* registered implementation when `allowed_impls` is None, so a leaked
registration silently adds a branch to an unrelated search, exactly as
`register.py` warns.

A module-scoped `conftest` fixture now removes whatever a test module added.
**The first version of it made things worse** — it snapshotted the whole
mapping and restored it, so a module that legitimately *unregistered* a leaked
implementation had it put back, carrying the leak forward instead of clearing
it, and the suite went 23 → 29 failures. It removes additions and nothing else
now, then calls the builtin registrar, which is documented idempotent.

**Eleven real failures, and they are the same eleven.** Five are committed
records that no longer bind the live tree — C1/C2 executable-closure and
preregistration gates — and six are a closed phase's consumed proposal
tracking a balance it can never spend. None is repaired deliberately: C2 is
CLOSED WITHOUT PROMOTION and authorizes no further spend, so nothing will ever
launch from those records, and rewriting a closed phase's history to make a
test green would be editing the past. Each failure points in the refusing
direction.

<details><summary>all eleven, by nodeid</summary>

```text
autoinit/test_c1_readiness_gates.py::test_the_committed_record_still_binds_the_live_executable
autoinit/test_c2_behavioural_proposal.py::test_the_ceiling_fits_the_project_cap_with_headroom
autoinit/test_phase_c2_full_search.py::test_the_budget_position_is_derived_not_restated
autoinit/test_phase_c2_full_search.py::test_the_documents_are_deterministic_and_regenerating_verifies_them
autoinit/test_phase_c2_full_search.py::test_the_headroom_verdict_matches_what_plan_session_actually_does
pod/test_c1_one_provider_resource.py::test_M_the_live_grant_and_the_launcher_agree_on_acquisition
pod/test_c1_session_contract.py::test_the_writer_refuses_to_rewrite_the_frozen_preregistration
pod/test_continuation_b_one_probe_contract.py::test_the_preregistration_binds_the_live_executable_digest
pod/test_phase_c2_full_search_chain.py::test_the_proposal_regenerates_byte_identically
pod/test_phase_c2_full_search_chain.py::test_the_proposal_reproduces_the_live_identities
pod/test_phase_c2_full_search_chain.py::test_the_proposal_states_the_figures_a_launch_review_needs
```

*The fifth entry is the one the list used to omit. It refuses because the C2
full-search pricing record says the beam-6 search session fits while
`plan_session` says it does not — a closed phase's proposal tracking a balance
it can never spend, which is the same reason as its three `_chain` siblings. It
is red at `4ae52e2e` too, measured in a detached worktree, so a reader
attributing a suite result does not have to re-establish that.*

*Reading it off a `<details>` block is also not the way to attribute a result.
Diff the observed node-ids against this list programmatically; eleven entries
and a summary that said ten is exactly the kind of near-miss an eye slides
over.*

</details>

**The C2 cap test was repaired, not the C2 records** (2026-09-28 round, kept
as history). A funding amendment made it fail because it pinned C2's
full-search config to the project's canonical cap, while C2's config, grant
and proposals all name `$370.00` and are internally consistent with the
decision they ran under. A closed phase's mirror cannot match both its own
grant and a later amendment — and this has now happened twice, because the cap
moved again to `$410.00` on 2026-10-02. The repair was to assert the invariant
that actually matters: the historical config matches the C2 authorization it
ran under and is **not** the current canonical project-cap owner, whose figure
is read from the authorization rather than pinned here. That is why a second
cap amendment did not break it a second time.

**An earlier run of this same suite read 23 failures.** Thirteen were mine,
from repairing the stage-H defect inside a file named by C1's frozen
executable closure and preregistration. The scorer was restored and the
repair moved to the caller; all thirteen went with it. That measurement is
recorded here because a suite result that was wrong for a knowable reason is
worth more than one quietly replaced.

## Budget — four limits that do not transfer

Derived by [`scripts/consolidate/derive_budget.py`](../../scripts/consolidate/derive_budget.py)
from the approved package and each session's own closeout. **Do not restate
these by hand; run the deriver.**

<!-- budget:begin -->

| limit | remaining |
| --- | --- |
| formal sessions | `$7.2431` of `$76.6523` |
| GPU engineering | `$1.9395` of `$10.0000` |
| package | `$9.1826` of `$86.6523` |
| project cap | `$396.8223` spent of `$410.0000`, leaving `$13.1777` |

**Full-ceiling sessions the FORMAL allowance funds: 0.** 1 ceilings cost `$30.0000` and the formal allowance has `$7.2431`. Dividing the PACKAGE balance instead gives 0, which is the error: the engineering allowance cannot pay for a formal probe.

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
