# Proposed first paid AutoInitializer pilot — 4B → 596M

**Status: PROPOSAL. Nothing here is authorized and no compute has been launched.**
Written 2026-08-12 alongside the zero-cost implementation. Numbers come from
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

## 2. What is already built and verified at $0

| piece | state |
| --- | --- |
| operator kind / implementation registry, immutable ids, frozen ledger | done, 112 tests |
| Qwen3 adapter; param arithmetic exact on both frozen counts | done |
| five v1 operators wrapping the existing algorithms + the incumbent composite | done |
| versioned search state, hash-bound metrics, recovery admission gate | done |
| Pareto beam ranking policy, versioned and hashed | done |
| deterministic resumable beam search, family-agnostic (proven on a fake MoE family) | done |
| immutable search manifest | done |
| end-to-end dry run on real tiny checkpoints | done, [`autoinit_dryrun_summary.json`](autoinit_dryrun_summary.json) |
| E8a's frozen depth map replayed from its frozen rounds | done |
| composite operator bitwise-identical to `init_student` | done |

## 3. Prerequisites — all zero cost, all blocking

1. **Build and freeze the initializer-state evaluation suite.** It does not exist.
   Role `STATE_EVALUATION`, domain-balanced with the five E8a domains, critical-token
   tags (`think_close`, `eos`, `final_answer`, `tool_close`, `reasoning`), leakage-checked
   against the promotion battery, the recovery rung and the validation slice.
   `check_role_isolation` must pass with `complete: true`.
2. **Build the recovery search battery.** Role `RECOVERY_BATTERY`. It must **not** be
   the 150-prompt promotion battery, and it must not reuse the 0.86M rung's prompts:
   that battery's inclusion mask was itself sampled using an 0.86M checkpoint, so an
   0.86M probe scored on it is not out-of-sample.
3. **Freeze the ranking policy and the halving plan** into a preregistration, before
   the run they judge (`SuccessiveHalvingPlan.freeze` + `assert_preregistered`).
4. **Measure the statistics-pass GPU/CPU split.** Every cost below is a range solely
   because of this one unmeasured quantity — the float64 `X^T X` accumulation runs on
   the CPU while the model runs on the GPU, and only the CPU-only end-to-end rate
   (5.24 ms/token) has ever been measured. One CPU-side profile collapses a 3.6×
   spread. Do this before booking a pod, not on one.
5. **Decide whether to move the accumulation to the accelerator.** Not required for
   the pilot; it is the single largest lever on search wall-clock.

## 4. Recommended shape

### Beam width **4**

Level 0 offers exactly 5 (kind, implementation) children per profile. Beam 4 keeps
four of five, so nearly every first-operator choice survives one level — which is the
minimum for an order search to be an order search. Beam 3 discards two of five on a
single measurement each; beam 6 exceeds the level-0 child count, so the extra width
buys nothing at the top and costs 127 GiB instead of 85 GiB.

### Two phases, gated

**Phase A — order only, one calibration profile.** Use `calib.domain_balanced@v1`:
already built, already leakage-checked, content hash `d65c1f40…` re-derivable from its
tokens. This isolates operator order.

**Phase B — calibration choice, conditional on Phase A.** Only if Phase A shows that
order matters. Adds `calib.reasoning_heavy@v1` (a zero-cost reweighted draw from the
same pool; not yet built) on the surviving orders.

Running both at once doubles the states and the storage to answer two questions at
once, at a point where the first is unanswered.

### Top-N **4**, survivors **2**

Beam 4 yields 5–9 complete leaves. Top-4 → 2 → 1 is a clean halving that uses most of
them, at 6 probes. Top-6 costs 50% more for candidates the beam already ranked below
the top four.

**The incumbent composite leaf is admitted to Top-N unconditionally as a control**,
whatever the beam ranked it. Its recipe is the one behind every existing behaviour
number in this project, so it is the only candidate whose probe result is comparable
to something.

## 5. Cost

L40S at $0.99/h. Ranges are the unmeasured statistics-pass split (§3.4).

| item | low | high |
| --- | ---: | ---: |
| Phase A search (beam 4, 1 profile, 30–42 states) | $0.75 | $2.72 |
| Phase B search (beam 4, +1 profile, 60–84 states total) | $1.51 | $5.43 |
| Recovery probes, Top-4 → 2 survivors, 6 × 0.86M | $9.82 | $9.82 |

**Phase A + probes: expected $10.57, hard backstop $12.54.**
**Phase A + Phase B + probes: expected $11.33, hard backstop $15.25.**

Probe pricing is 1,023 optimizer steps (frozen E1 0.86M config) × 4.15 s/step (E6b,
measured) × 1.20 overhead + $0.236 per battery evaluation (E6: $2.36 over 10 arms).
The 1.20 overhead factor is **stated, not measured**.

Add a **$3.00 setup/redraw reserve**: this project's pod setup has varied 30× across
sessions (5 to 150+ minutes) and no figure above includes it.

**Proposed authorization: $16.00 expected, $19.00 hard backstop**, covering Phase A,
Phase B and the six probes. Full recovery of the Top-1 winner is **not** in this
proposal and needs its own authorization.

### Budget position

```
actual cumulative spend            $180.7033
authorized cumulative cap          $211.07
unused, uncommitted                 $30.3667   (released by E8b's termination)
this proposal, hard backstop        $19.00
resulting cumulative cap            $211.07    unchanged — fits inside the release
resulting projected spend           $199.70    if the hard backstop is fully consumed
```

No cap increment is required. Plan from the $180.7033 actual spend, not from the cap.

## 6. Storage — the actual binding constraint

| quantity | beam 4 / 1 profile | beam 4 / 2 profiles |
| --- | ---: | ---: |
| peak concurrent working set | 85 GiB | 135 GiB |
| total bytes written | 107 GiB | 214 GiB |
| retained after pruning | 27 GiB | 27 GiB |
| peak GPU resident (parent + child + stats) | 14.3 GiB | 14.3 GiB |

The working peak is what a pod's container disk has to hold: a whole level's children
must be on disk simultaneously, because ranking cannot start until every child has
been measured. **Provision ≥150 GiB for Phase A and ≥200 GiB for Phase B.** GPU memory
is not a constraint at this scale — 14.3 GiB fits an L40S with room to spare, which is
why the pilot does not need an A100.

Individual checkpoint sizes: teacher 7.49 GiB, depth-only intermediate 5.99 GiB,
target leaf 1.11 GiB, activation statistics 1.81 GiB per pass.

## 7. Risks

**The beam may prune the leaf that would have recovered best.** This is the central
unresolved risk and it is not fully mitigated. The beam ranks on step-0 state metrics;
E7 established that general NLL does not predict autonomous behaviour, and E8a
established that a proxy metric can reverse under composition. The Pareto policy means
no single scalar prunes alone, and Top-4 out of 5–9 leaves is generous — but nothing
here *demonstrates* that composed step-0 fidelity predicts post-recovery behaviour.
**That is itself a result the pilot produces**: six probes with both their step-0
metrics and their behaviour give the first direct measurement of whether the beam's
ranking correlates with the thing it is supposed to select for. Report it either way.

**Order effects may be small relative to measurement noise.** Unlike behaviour, step-0
KL/NLL are deterministic given a checkpoint, so the search's own ranking is noiseless.
The noise enters at the probe stage, where the behaviour metric's seed-only spread is
0.1290 — which is why survivors get a second seed and why Top-1 is not taken from a
single sa run.

**The composite may simply win.** A plausible outcome is that the incumbent
one-shot float64 recipe beats every decomposed path, because decomposition rounds to
the working dtype three extra times. That is a real finding, not a failed pilot, and
the manifest will support it: every leaf carries its own hash-bound metrics.

**Cost is dominated by an unmeasured quantity.** See §3.4.

## 8. Scaling risks for 30B → 4.xB

Recorded now because the architecture was built to survive them, not to be rewritten.

1. **Activation statistics scale as `d²`.** 1.81 GiB at the 4B teacher; the same
   formula at a 6144-wide, 64-layer teacher is ~19.6 GiB of float64 state per pass,
   held in host memory while the model occupies the accelerator. This is the first
   thing that breaks.
2. **Intermediate checkpoints are near-teacher-sized.** A depth-only intermediate of a
   30B teacher is ~26B parameters, ~52 GiB in bf16. A beam-4 level's working set
   crosses a terabyte. Pruned-weight release stops being an optimization and becomes
   a requirement.
3. **The Top-N leaves are themselves ~4.xB models**, so the recovery probes inherit
   E8b's unresolved memory problem: fixed cost at 3.2B was 54.82 GB, and `masked_ce`'s
   unchunked `sel.float()` alone was 4,978 MB of transient. Six probes at 4.xB is a
   different hardware conversation from six at 596M.
4. **A 30B teacher is likely MoE.** The kind registry accepts `MOE_EXPERT_SET`,
   `MOE_ROUTER` and `MOE_SHARED_EXPERT` without a core edit — demonstrated by test —
   but no MoE *adapter* or *operator* exists, and expert-set reduction interacts with
   routing in ways none of the v1 operators model.
5. **Attention family.** The Qwen3 adapter declares GQA. An MLA or linear-attention
   teacher needs its own `ATTENTION` implementation; the dispatcher already refuses to
   offer a GQA operator to a non-GQA adapter.
6. **Full-vocabulary KL in state evaluation** costs `2·V·d` per token and grows with
   the teacher. The long-term direction is sparse Top-K KD over
   `TopK_teacher ∪ TopK_student`; the measurer is pluggable so that change does not
   touch the search.

## 9. What this proposal does not include

* full recovery of the Top-1 winner — separate authorization;
* any new operator algorithm — v1 is the five that exist;
* Top-K KD — explicitly future work;
* reopening E8b in any form.
