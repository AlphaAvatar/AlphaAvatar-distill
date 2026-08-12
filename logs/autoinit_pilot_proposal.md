# Proposed first paid AutoInitializer pilot — 4B → 596M

**Status: PROPOSAL, revised 2026-08-13 after the pre-GPU correction pass. Nothing
here is authorized and no compute has been launched.** Numbers come from
[`autoinit_v1_search_space.json`](autoinit_v1_search_space.json), regenerable with
`PYTHONPATH=src python scripts/autoinit/plan_search.py`.

---

## 1. Question

Does **conditional operator order + calibration choice + beam selection** produce a
better 596M initialization than the incumbent fixed recipe?

Not "which FFN algorithm is best" and not "is the contribution depth map good". The
v1 operator library is deliberately the five algorithms that already exist. The
variable is *composition*.

E8a is why. Its depth map preserved the full-width teacher **3.11×** better
(KL 0.620586 vs 1.932531) and, once composed with width/FFN/attention compression,
initialized **2.8 nats worse**. An operator's own objective did not survive
composition. The AutoInitializer measures the composed state instead of assuming it.

## 2. Search space — corrected

An operator declaring `CalibrationNeed.NONE` consumes no mixture, so branching it
over calibration profiles manufactures byte-identical states. `depth.positional_v0`
is a fixed positional heuristic and `attention.weight_proxy_v0` scores weights;
both are invoked once, against a canonical `calib.none@v1` sentinel. The decomposed
space is therefore

```
24 orderings × (1 + P) DEPTH × P WIDTH × P FFN × 1 ATTENTION
```

| P | complete decomposed paths |
| --- | ---: |
| 1 | **48** |
| 2 | **288** |
| 3 | **864** |

not `48 × P⁴`. Because state identity is derived from each step's profile hash and
a no-calibration step always carries the sentinel's hash, this is structural rather
than a deduplication pass.

## 3. Prerequisites — all zero cost, all blocking

1. **Build and freeze the initializer-state evaluation suite.** It does not exist,
   and the beam cannot rank without it. Role `STATE_EVALUATION`, five domains,
   critical-token tags, leakage-checked against the promotion battery, the recovery
   rung and the validation slice. `check_role_isolation` must report
   `complete: true` — it now fails closed when two roles share no comparable
   identity kind, which a prompt-hash-only check could never detect.
2. **Build the recovery search battery.** Role `RECOVERY_BATTERY`. It must **not**
   be the 150-prompt promotion battery, and must not reuse the 0.86M rung's
   prompts: that battery's inclusion mask was itself sampled using an 0.86M
   checkpoint, so an 0.86M probe scored on it is not out-of-sample.
3. **Freeze the ranking policy, the beam schedule and the halving plan** into a
   preregistration, before the run they judge (`SuccessiveHalvingPlan.freeze` +
   `assert_preregistered`).
4. **Measure the statistics-pass GPU/CPU split.** Every cost below is a range solely
   because of this one unmeasured quantity. One CPU-side profile collapses a ~3.8×
   spread. Do it before booking a pod.
5. **Decide whether to move the float64 accumulation to the accelerator.** Not
   required; it is the single largest lever on search wall-clock.

Also settled by the correction pass and no longer a risk to carry: the reference
teacher logits are **recomputed per candidate**, not cached. Caching them for the
intended suite is 33.8 GiB (59,763 positions × 151,936 vocab × 4 B); recomputing
costs one teacher forward per candidate — 5.6 s on an L40S, ~3.9 minutes across the
whole search. `CACHE_IN_MEMORY` remains available and refuses to allocate past an
explicit budget rather than discovering the limit at runtime.

## 4. Recommended shape

### Delayed pruning, then beam width 6

Level 0 offers exactly five children at one profile — the five distinct structural
hypotheses the search exists to compare. **None of them is pruned.** `SCHEDULE_V1`
(`warmup_levels=1, width=6`) retains every child of the root and begins quality
pruning only after the next expansion. Discarding a first-step hypothesis on a
single step-0 measurement is precisely the mistake E8a documented.

Width 6 afterwards is wider than the five lineages it carries, so the first pruning
level can keep every surviving lineage and still admit a second variant of one.

### ε-Pareto, per-domain fidelity, NLL demoted

Objectives are original-teacher fidelity seen three ways: equal-domain mean KL,
**worst-domain** KL (so a path cannot buy an average by sacrificing tool or code
fidelity outright), and critical-token KL. **NLL is not an objective and appears
nowhere in the tie-break** — E7 measured a −5.22 nat swing moving behaviour by
exactly +0.0000. It is recorded per domain and displayed beside every ranking
decision as a diagnostic.

`state.nll.general` is now computed from the general domain **alone**. The
pooled-over-all-domains quantity still exists, under
`state.nll.pooled_all_domains`, which is what it always was.

Dominance is ε-tolerant (1e-4 nats per objective): a state dies only when another
is *meaningfully* better. Selection then rotates over lineages before the
deterministic state-id tie-break, so a beam cannot fill with variants of one
hypothesis.

### Phase A, and Phase B on an independent trigger

**Phase A — order, one calibration profile.** `calib.domain_balanced@v1`: already
built, leakage-checked, token-level content hash `d65c1f40…` re-derivable from its
items.

**Phase B — operator-specific calibration.** Deferred **for budget, not for
evidence**. Operator order and operator-specific calibration are separate
hypotheses, and "no detectable order effect under one profile" does not imply
"calibration profile is irrelevant". Phase B's trigger is funding plus a built
`calib.reasoning_heavy@v1`; it is *not* conditional on Phase A showing an order
effect.

### Recovery: 5 searched leaves + 1 canonical control

```
rung 1   5 searched leaves + the retained canonical control, seed sa   (6 probes)
rung 2   the control + the best 2 searched leaves,          seed sb   (3 probes)
rung 3   only candidates inside the equivalence interval,   seed sc   (conditional)
```

**The control is the retained `qwen3_0p6b_init_v0` checkpoint, injected by frozen
artifact hash — not a re-executed composite.** A composite run inside the search is
built from *this run's* calibration statistics; the canonical checkpoint was built
from the original Stage-0 statistics over the 949,859-token warm-up mixture. Same
algorithm, different input, therefore different weights — and every existing
behaviour number in this project belongs to the latter. `make_control_state`
verifies the frozen single-file sha256 before admitting it.

**The control advances to rung 2 regardless of its rung-1 result**, and is never
excluded by the feasibility gate. A baseline eliminated at rung 1 leaves the
two-seed comparison with nothing to compare against.

Selection is a **constraint, then an objective, and never a weighted sum**:

1. `usable_rollout_rate` ≥ a preregistered floor — a *feasibility* gate. It is
   blind to correctness by construction, so a terse contentless reply scores
   perfectly on it; it may exclude, never rank.
2. `correct_overall` — the capability objective, among feasible candidates.
3. `correct_given_usable` — reported to explain a ranking, never to change one.

## 5. Cost

L40S at $0.99/h. Ranges are the unmeasured statistics-pass split (§3.4).

| item | low | high |
| --- | ---: | ---: |
| Phase A search (beam 6, warmup 1, 1 profile, 39–56 states) | $0.93 | $3.57 |
| Recovery rungs 1+2 (9 probes) | $14.73 | $14.73 |
| Conditional third seed (3 probes) | $0.00 | $4.91 |
| Setup / redraw reserve | $0.00 | $3.00 |

**Expected $17.00. Hard backstop $26.21.**

Probe pricing is 1,023 optimizer steps (frozen E1 0.86M config) × 4.15 s/step (E6b,
measured) × 1.20 overhead + $0.236 per battery evaluation (E6: $2.36 over 10 arms).
The 1.20 overhead factor is **stated, not measured**. Pod setup has varied 30×
across this project's sessions and is covered only by the reserve.

Phase B, when funded, adds a beam-6 two-profile search at **$1.90–7.43** and
**245 GiB** of working storage.

### Budget position

```
actual cumulative spend            $180.7033
authorized cumulative cap          $211.07
unused, uncommitted                 $30.3667   (released by E8b's termination)
this proposal, hard backstop        $26.21
margin against the release           $4.16
resulting projected spend          $206.91     if the hard backstop is fully consumed
```

No cap increment is required, but the margin is thin: **$4.16**. If the third seed
fires *and* setup goes badly, the reserve is what absorbs it. Plan from the
$180.7033 actual spend, not from the cap.

## 6. Storage — still the binding constraint

| quantity | beam 6 / 1 profile | beam 6 / 2 profiles |
| --- | ---: | ---: |
| peak concurrent working set | **106 GiB** | **245 GiB** |
| total bytes written | 135 GiB | 322 GiB |
| retained after pruning | 36 GiB | 36 GiB |
| peak GPU resident | 14.3 GiB | 14.3 GiB |

Delayed pruning raises the working peak: the level-1 expansion has five parents
instead of a pruned beam. **Provision ≥150 GiB of container disk for Phase A and
≥300 GiB for Phase B.** GPU memory is not the constraint at this scale — 14.3 GiB
fits an L40S comfortably, which is why the pilot does not need an A100.

Checkpoints must be assumed **sharded**. A depth-only 4B intermediate is 5.99 GiB
and already sits near the default shard threshold in some supported Transformers
versions. Identity is an artifact digest over every shard, the shard index, the
config, the architecture signature and the tokenizer; metrics bind to that digest.
A frozen single-file hash such as `86fbba78…` remains checkable through
`single_shard_sha256`, and a sharded rebuild reports as a *different layout* rather
than as corruption.

## 7. Risks

**The beam may prune the leaf that would have recovered best.** Still the central
unresolved risk, now better mitigated: no pruning at level 0, ε-tolerant dominance,
lineage diversity, NLL removed as a selector, and 5 of 7–13 leaves admitted to
recovery. None of that *demonstrates* that composed step-0 fidelity predicts
post-recovery behaviour. **That is itself a result the pilot produces**: nine probes
carrying both their step-0 metrics and their behaviour give the first direct
measurement of whether the beam's ranking correlates with what it is meant to
select for. Report it either way.

**Order effects may be small relative to measurement noise.** Step-0 KL is
deterministic given a checkpoint, so the search's own ranking is noiseless. Noise
enters at the probe stage, where the behaviour metric's seed-only spread is 0.1290
— hence two seeds, a preregistered equivalence interval, and a third seed for ties.

**The composite may simply win.** A decomposed path rounds to the working dtype
three extra times that the one-shot float64 recipe never incurs. That is a real
finding, not a failed pilot.

**Cost is dominated by an unmeasured quantity.** See §3.4.

## 8. Scaling risks for 30B → 4.xB

1. **Activation statistics scale as `d²`.** 1.81 GiB at the 4B teacher; ~19.6 GiB of
   float64 host state at a 6144-wide, 64-layer teacher, per pass. The statistics
   cache now shares one pass between WIDTH and FFN on the same parent — keyed on
   the parent artifact digest, so cross-parent reuse is impossible — which halves
   the passes without weakening the re-measurement invariant.
2. **Intermediate checkpoints are near-teacher-sized.** A depth-only intermediate of
   a 30B teacher is ~26B parameters, ~52 GiB in bf16, certainly sharded. A beam-6
   level's working set crosses a terabyte; pruned-weight release stops being an
   optimization and becomes a requirement.
3. **The Top-N leaves are themselves ~4.xB models**, so the probes inherit E8b's
   unresolved memory problem: fixed cost at 3.2B was 54.82 GB and `masked_ce`'s
   unchunked `sel.float()` alone was 4,978 MB of transient.
4. **A 30B teacher is likely MoE.** The kind registry accepts `MOE_EXPERT_SET`,
   `MOE_ROUTER` and `MOE_SHARED_EXPERT` without a core edit — demonstrated by test
   against a non-transformers fixture — but no MoE adapter or operator exists.
5. **Attention family.** An MLA or linear-attention teacher needs its own
   `ATTENTION` implementation; the dispatcher already refuses to offer a GQA
   operator to a non-GQA adapter.
6. **Full-vocabulary KL in state evaluation** costs `2·V·d` per token. Recompute is
   affordable at 4B; at 30B the per-candidate reference pass is ~7.5× more
   expensive and the strategy should be revisited then. The measurer is pluggable,
   and the long-term direction is sparse Top-K over
   `TopK_teacher ∪ TopK_student`.

## 9. What this proposal does not include

* full recovery of the Top-1 winner — separate authorization;
* Phase B — separate authorization, on an independent trigger;
* any new operator algorithm — v1 is the five that exist;
* Top-K KD — explicitly future work;
* reopening E8b in any form.
