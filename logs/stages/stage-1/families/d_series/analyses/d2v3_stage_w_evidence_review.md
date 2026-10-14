# D2 v3 Stage W GPU evidence review (2026-10-14)

**Status:** the Stage W engineering validation ran once and is COMPLETE and
CLOSED. Its verdict is **validation FAIL**, so its evidence is **not
adopted**. Nothing is billing. This review covers the failure, the
descriptive science and the price. It authorizes nothing. Preregistration v3
stays DRAFT. The D2 search, recovery, screening, confirmation and D3 are not
granted.

| | |
| --- | --- |
| run | `d2v3val_20261014_170824` (acquisition attempt 3; attempts 1 and 2 were $0 capacity refusals, `d2v3val_20261014_170220` and `d2v3val_20261014_170522`) |
| pod | `20ji79950u9arw`, NVIDIA L40S (46 GiB, driver 595.91.07), $1.09/h live secure |
| runtime | torch 2.11.0+cu128, transformers 5.13.1, CUDA 12.8; matmul TF32 off |
| executed commit | `4248d7dbce4b2e97de32700fb06542a19f4e3339`. It equals the grant's `executable_commit`, and the tracked tree was clean. |
| wall | 13.9 min billed. W1–W6 took 574.7 s against a registered 1166 s step bound. |
| cost | **$0.2538** all-in, of the $1.22 grant |
| teardown | provider-confirmed on the first attempt and hashed into the campaign. 0 pods and 0 volumes were listed at 2026-10-14 17:29:40 UTC. |
| evidence | `../validations/d2v3-gpu-validation/v1/runs/d2v3val_20261014_170824/out/adoption_all_v1/adoption.json`, sha256 `66a80d17…` |
| adoption | **REFUSED.** `d2v3_validation_evidence` reports *validation is FAIL, not PASS*. The identities match: objective `90e851fd…`, contract `1321017a…`, prereg `b5281092…`, plan `931c7502…`. The executed commit matches. The verdicts recomputed from the measurements equal the recorded ones. |

## Verdicts, recomputed from the measurements

| verdict | result |
| --- | --- |
| arithmetic | **FAIL**: `kl_bounds` only |
| integrity | **PASS** |
| row equivalence (scored_v1) | **PASS** |
| **validation** | **FAIL** |

**Arithmetic checks against the registered tolerances:**

| check | measured | tolerance | |
| --- | --- | --- | --- |
| union KL float32 vs float64, per position (46,056) | max 1.15e-4 nats; 0 positions outside abs + rel | 1e-4 + 1e-4·KL | pass |
| retrieval vs full head | 1.11e-4 nats | 2e-4 | pass |
| K+1 confidence vs float64 (both profiles) | ≤ 4.4e-6 (entropy ≤ 2.3e-5) | 1e-5 | pass |
| item objective reduction vs float64 | 7.2e-9 rel | 1e-5 | pass |
| raw-KL control (7 metrics) | 0.0 rel | 1e-6 | pass |
| float64 oracle bounds (topk ≤ union ≤ full) | **0** violations | slack 1e-12 | pass |
| **float32 bounds beyond the 4·eps·\|log Z_teacher\| slack** | **175** of 46,056 | **0** | **FAIL** |

**Integrity checks, all pass.**

- The teacher identity matches.
- The stores match their declaration.
- The FFN statistic is full-scope and weighted, and its denominator is the
  weight sum (40,892.43).
- DEPTH reads the root store with retrieval.
- The DEPTH table is consistent.
- State evaluation reads its own union store.

**Row equivalence passes.**

- DEPTH all_v1 vs scored_v1: 36 candidates, max relative difference 3.4e-9,
  Spearman 1.0, same choice.
- State evaluation: 28 metrics, max absolute difference 5.4e-8, against a
  tolerance of 1e-5.

## The failure: diagnosis and classification

**What the gate does.** W3 checks the two orderings
`KL_ref32 ≤ KL_U32 ≤ KL_full64` in float32. It allows a slack of
`4·eps·|log Z_teacher|`, which is about 4.8e-7 × |log Z|: roughly 5e-6 at
|log Z| = 10 and 1.4e-5 at 30. The registered maximum number of violations
is 0. That slack was calibrated at $0 on a CPU, on a small same-vocabulary
stand-in checkpoint with hidden size 1024 (`d2v3_real_logits_check.json`),
where the union-KL float32 error peaked at 1.6e-5.

**What the GPU measured.**

- On the real teacher, on the L40S, the same float32 error reached 1.15e-4
  nats. That is still inside the registered CLASS-1 tolerance of 1e-4 + 1e-4·KL
  at every position.
- The exact float64 bounds hold with no violations.
- The union KL is extremely close to the full KL: the union/full ratio has
  median 0.99999994. Any float32 rounding larger than the slack therefore
  crosses a bound.

**The mechanism is inconsistent with the plan's own tolerance.**

- The plan accepts float32 error in `KL_U` up to abs 1e-4 + rel 1e-4·KL.
- The bound check gives the same quantity a purely absolute slack, about ten
  times tighter, with no relative term.
- The slack models only the teacher's log-normalizer. It omits the student's
  log-normalizer, the product with p, the sum over about 313 union terms and
  the tail term.
- The plan's own comment explains why a purely absolute tolerance is wrong
  for this quantity. The bound check uses one anyway.

**What can be proven from the recorded facts:**

- **Upper bound.** Every violation of the upper bound is within the
  registered float32 accuracy:
  `KL_U32 − KL_full64 ≤ |KL_U32 − KL_U64| + (KL_U64 − KL_full64) ≤ (1e-4 + 1e-4·KL) + 1e-12·(1 + KL)`.
- **Lower bound.** This side also carries the float32 error of `KL_ref32`,
  which no gate measures.
- **What the record cannot say.** It stores only the combined count (175).
  It does not store the split between the two sides or the per-position
  magnitudes, so neither can be recovered without a rerun.

**Classification.** This is an implementation defect in my own gate, not
evidence that the union KL is wrong. Under P12.1 that would make it an
ordinary engineering failure. **It is not repaired or rerun here, for this
reason:**

- The repair would change a registered tolerance after its FAIL was seen.
  The grant forbids relaxing a tolerance post hoc.
- The FAIL record is preserved unchanged.
- A repair is a maintainer decision that registers the corrected gate
  **before** any new session. The corrected gate would keep the float64
  oracle bounds exact and judge float32 deviations by the CLASS-1 abs + rel
  tolerance, recording each violation's side and magnitude.
- A rerun would need a new grant. The same shape is $0.28 expected and
  $1.22 all-in; this session cost $0.2538.
- The $0.9662 left in this grant is not reused: the campaign is closed.

Lesson: a float32 gate calibrated on a CPU stand-in does not transfer to the
GPU and the real teacher. The 1.6e-5 → 1.15e-4 gap is the evidence.

## Engineering evidence (gated or descriptive; none decides science)

- **Devices.** The root sketch, confidence, retrieval and every forward ran on
  `cuda:0`. The FFN child was handed CPU → `cuda:0` in 1.46 s.
- **Memory.** The step peak was 22.7 GB (W4) of 47.7 GB.
  - The pool was hoarded (30.5 GiB).
  - The parent reference stayed **cached**: 0 fills and 0 recomputes. W4 had
    2,558 reference hits and W5 had 4,970.
- **Teacher forwards during state evaluation.** v3 made 0; the historical
  Top-K KL path made 27.
  - v3 evaluation took 5.14 s against 10.43 s (−5.3 s).
  - v3 peaked at 17.1 GB against 19.9 GB.
- **Union stores (bsz = 3).** domain_balanced took 7.34 s, 59,763 positions,
  145 MB sketch + 306 MB retrieval. reasoning_heavy took 6.04 s, 44,324
  positions, 107 MB + 227 MB.
- **Union reduction cost.** 7 µs per row above the K+1 reduction. A DEPTH
  objective costs 0.078 s per candidate (all_v1) and 0.069 s (scored_v1).
- **bf16 ties.**
  - 80% of rows have a tie at the Top-200 boundary.
  - The deterministic tie-break never spills.
  - Its set equals `torch.topk`'s on every GPU row; on the CPU stand-in, 60%
    of rows differed.
- **Execution drift, bsz 1 vs 3 (reported, never gated).**
  - Store confidence: max |Δ| 0.110 / 0.050.
  - Top-K support: identical in 8.7% / 5.7% of rows, mean Jaccard 0.977.
  - State objective: max |Δ| 0.033.

## Science diagnostics: the five questions

All of these are descriptive. Each rests on **one DEPTH round** (round 0, 36
candidates) of **one non-root parent** (W2's full-scope FFN child), on
**domain_balanced**. None of it is evidence that student reasoning improves.
The evidence was not adopted, but these numbers are not what failed.

**c_T (K+1 = Top-200 + tail) over scored positions:**

| profile | positions | median | IQR | ESS fraction | q05 | min |
| --- | --- | --- | --- | --- | --- | --- |
| domain_balanced@v1 | 44,746 | 0.9971 | **0.123** | **0.975** | 0.566 | 0.200 |
| reasoning_heavy@v2 | 46,825 | 0.9990 | **0.095** | **0.981** | 0.642 | 0.200 |
| v2 (Stage V), for reference | | 0.999 | 0.042–0.055 | ≥ 0.994 | | |

**DEPTH round 0, the same 36 candidates (W4, all_v1):**

| pair | Spearman | same choice | top-3 overlap |
| --- | --- | --- | --- |
| parent KL vs root teacher-Top-K KL | 0.804 | **no** (26 vs 17) | 1/3 |
| root teacher-Top-K KL vs root union KL (unweighted) | **1.000**, identical order | yes | 3/3 |
| root union KL unweighted vs weighted | **0.997** | yes (17) | 3/3 |
| parent KL vs root union KL weighted | 0.803 | no | 1/3 |

The top five are:

- parent KL: 26, 29, 30, 20, 21;
- root teacher-Top-K and unweighted union KL: 17, 18, 30, 10, 16;
- weighted union KL: 17, 18, 30, 16, 31.

The gap between the first and second choice is 1.07–1.14% relative under every
root variant.

**1. Does the weighting give a meaningful selection signal?** It is
meaningful as a reweighting of the objective. It is not meaningful as a
DEPTH selection signal.

- **Against v2:**
  - The c_T IQR is 2.2–2.9× v2's.
  - ESS falls from ≥ 0.994 to 0.975–0.981.
  - The weighted objective is **84.9%** of the raw KL mass, against v2's
    96–97%.
- **Correlation with KL.** On this child, Spearman(c_T, KL) = **−0.72**: the
  weighting discounts the high-KL positions. On B under v2 it was about 0, but
  that was a different student, so the two are not a like-for-like contrast.
- **Shares that move:**
  - `general` falls from 37.6% to 32.4% of the objective (its mean c_T is 0.71).
  - `other_scored` spans fall from 38.6% to 33.6%.
  - reasoning spans rise from 50.1% to 53.8%; math from 22.4% to 24.6%.
- **c_T tracks the teacher's own target probability** (Spearman 0.97). Its
  mean is 0.955 where the target is the teacher's top-1 and 0.657 elsewhere.
- **The DEPTH choice does not move.** Unweighted vs weighted gives ρ 0.997,
  the same choice and the same top 3; only ranks 4–5 change.
- **FFN statistics move a little.** The overlap between full-scope weighted
  and D1's narrow unweighted selection is 0.964 mean (0.935 min; 0/36 layers
  identical; lowest in layers 24–31). v2's was 0.984 / 0.973.
  - This contrast mixes scope with weighting.
  - Its behavioural effect is unknown.

**2. Does the union beat teacher-only Top-K?** It improves the KL
approximation slightly and the ranking not at all.

- **Ratio to the full KL** (45,294 rows), union against teacher Top-K:
  - mean 0.99643 against 0.99571, closing about 17% of the gap;
  - p01 0.931 against 0.910, about 24% of the gap;
  - min 0.078 against 0.053.
- **Union size.** The mean is 313 entries (p01 259, max 400). The overlap
  with the teacher's Top-200 averages 87.
- **Student mass outside the union.** The mean is 0.049 and p99 is 0.54, on
  heavily damaged bypass candidates.
- **On state evaluation,** the union raw KL exceeds the teacher-Top-K KL by
  8.4e-4 nats (0.054%).
- **DEPTH ranking:** identical order.

**3. How much of the change from D1 comes from the root reference?** All of
it in this round.

- Parent → root moves the choice from 26 to 17 (ρ 0.804, top-3 overlap 1/3).
- The union leaves the order identical.
- The weighting moves ρ by 0.003.
- This reproduces v2's Stage V result: ρ 0.816, a different choice, layer 17.
- The scope is one round on one parent and one profile, so it cannot be
  extrapolated to the deeper rounds of a full search, where candidates sit
  closer together.

**4. Does the FP32 head definition add differences?** Small ones on average,
not zero at the tails. Its ranking effect was not measured.

- **The comparison is the float32 head on the formal bsz = 3 store's hidden
  state against a native bf16 bsz = 1 forward.** It therefore mixes the head
  definition with batch-shape drift. W1's drift is of the same order, so the
  numbers below **bound** the head's contribution rather than isolate it.
- KL(head ‖ native) per position:
  - mean 4.3e-4 nats, median 2.1e-5, p99 5.5e-3, max 0.064;
  - about 0.03% of a candidate's mean KL;
  - about 2–3% of the 0.015–0.018-nat first/second DEPTH gap.
- |Δc_T|: mean 1.7e-3, p99 0.0125, max 0.052.
- Top-200 sets: identical in 4.0% of rows, mean Jaccard 0.975 (mostly bf16
  boundary ties).
- The head is the same for every candidate, so any ranking effect is second
  order. A DEPTH round on a native-bf16 store would be needed to measure it.

**5. The IQR widened but the DEPTH ordering did not change. Is a full search
worth running?** Not as a test of what v3 adds.

- Neither of v3's two distinctive changes moved the measured DEPTH decision.
  The union's effect on the ordering is exactly zero, and the weighting
  changes only ranks 4–5.
- A full v3 search would therefore mainly test the root reference again,
  which v2 was already defined to test.
- The weighting's real effects are the 15% objective-mass shift, the
  domain/span reweighting and a 3.6% change in FFN selection. Whether they
  matter would show only in deeper rounds or after recovery, in behaviour.
  This round gives no estimate of that.
- **What this evidence does not support** (per the grant's instruction):
  - an IQR ≥ 0.11 on one profile;
  - an unchanged DEPTH choice;
  - a GO label.
  None of these is evidence that student reasoning will improve.

## Re-pricing

**The price in force is unchanged**, because the evidence is not adopted.
Version 2's Stage V terms apply, plus PROVISIONAL union terms
(`d2_budget_v1.json`):

| D2 v3 search | expected | planning bound | vs $30 cap |
| --- | --- | --- | --- |
| in force (cache_fix_validated, provisional union terms) | $12.36 | **$30.70** | over by $0.70 |
| scored_v1 lever | — | $29.34 | **not available**: the scored_v1 PASS is inside a FAIL record, and the gate reads only adopted evidence |

**Counterfactual — Stage W's measured terms, NOT ADOPTED, NOT IN FORCE.**
These numbers show what an adopted rerun would likely price at. They are
computed with `measured_union_overhead` and the ×1.5 margin for a single
measurement:

| | expected | planning bound |
| --- | --- | --- |
| all_v1, cache held | $11.45 | $24.90 |
| scored_v1, cache held | $11.25 | $24.21 |
| all_v1, cache regressed (records_max) | $11.45 | $34.53 (over the cap; the driver's cache guard stops the search) |

The measured terms are:

- per candidate: 0.51 s, against the provisional 1.22 s;
- per state evaluation: 0 s (v3 is faster than the historical path);
- statistics operators: ×1.099 (W2), against ×1.027;
- union stores: 0.36 min.

**Programme against the live limits** (`derive_budget.py`):

- project remaining: $119.2341;
- engineering remaining: $3.4297;
- per-session cap: $30.

At hard ceilings with teardown reserves, on the cache_fix_validated pricing:

| scenario | hard | outcome |
| --- | --- | --- |
| D2 | $60.98 | fits; $58.25 left |
| D2 + D3 | $113.64 | fits; $5.59 left |
| D2 + D3 + fresh confirmation | $129.14 | **$9.91 short** (fits at expected, $82.55) |

- The shortfall fell from $10.87 to $9.91 only because the closed campaign
  now counts at its bill ($0.2538), not its grant ($1.22).
- K = 4 and both caps are unchanged.
- No split-session resume is proposed. The in-force bound misses the cap by
  $0.70 on provisional terms, and the measured terms fit once adopted. A
  two-session run is therefore not the next lever.

## Recommendation for preregistration v3

**Engineering: not adoptable as executed.**

- The FAIL is root-caused to a gate-definition defect and is clearable.
  Clearing it requires a maintainer-approved, prospectively registered bounds
  gate and one new Stage W session (about $0.28 expected).
- Every other gate passed:
  - integrity;
  - scored_v1;
  - raw-KL control;
  - retrieval;
  - confidence;
  - union KL against float64.

**Registered rubric:**

- GO needs a PASS. Even on the measured numbers, no GO condition holds:
  - the weighted and unweighted choices are the same;
  - ρ is 0.997, not below 0.95;
  - the IQR reaches 0.11 on only one profile.
- The strict HOLD thresholds (IQR ≤ 0.06, ESS ≥ 0.99) are not met either.
- On a PASS this would read "between". Under the grant's "if weak, report
  HOLD" rule, the reading is **HOLD**.

**Science recommendation: HOLD.** v3 adds no measured DEPTH selection signal
over the root reference. The maintainer's choice:

- **Stop D2 v3 here (recommended).**
  - The root-reference effect was already measured in v2 and again here.
  - Neither the weighting nor the union changes what DEPTH selects.
  - The money stays with D3 and the fresh confirmation, where the programme
    is $9.91 short at hard.
- **Run v3 with an explicit claim boundary** — "a root-reference search;
  weighting and union expected inert at DEPTH". It needs, in order:
  1. the prospective bounds-gate re-registration and a new Stage W grant to
     obtain an adoptable PASS;
  2. re-pricing on the adopted terms;
  3. preregistration v3 approval;
  4. a decision on the $9.91 programme shortfall before D3's or a fresh
     confirmation's formal sessions.

**Not on the table** (per the rubric):

- tuning K, the mapping, a temperature, a mask or the tail formula;
- redefining the confidence on these numbers.
