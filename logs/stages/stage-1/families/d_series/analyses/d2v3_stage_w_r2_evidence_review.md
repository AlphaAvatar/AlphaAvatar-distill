# D2 v3 Stage W revision 2: GPU evidence review (2026-10-14)

**Status:**

- Stage W revision 2 ran once, is COMPLETE and CLOSED, and its validation
  is **PASS**. The evidence is **adopted**. Nothing is billing.
- Revision 1 (`d2v3val_20261014_170824`, FAIL) is unchanged and adopts
  nothing. Its review is
  [`d2v3_stage_w_evidence_review.md`](d2v3_stage_w_evidence_review.md).
- This record authorizes nothing. Preregistration v3 (`b5281092…`) stays
  DRAFT. The D2 search, recovery, behavioural screening and confirmation,
  and D3 are not granted.

| | |
| --- | --- |
| run | `d2v3r2val_20261014_185259`, the first acquisition attempt |
| pod | `ehienq52s3pjzp`, NVIDIA L40S, $1.09/h live secure |
| runtime | torch 2.11.0+cu128, transformers 5.13.1, CUDA 12.8 |
| executed commit | `7ef64f3b92d0a243474563dbd151954d316a16fd`, the grant's `executable_commit`, with a clean tree |
| plan | `validations/d2v3-gpu-validation/v2/plan.json`, sha256 `31377a77…` (revision 2) |
| wall | 13.6 min billed; W1–W6 took 574 s against a 1169 s registered bound |
| cost | **$0.2493** all-in, of the new $1.22 grant. Stage W over both revisions: **$0.5031** |
| teardown | Provider-confirmed and hashed. 0 pods and 0 volumes were listed at 2026-10-14 19:07:31 UTC. The acquisition loop exited, and no observer is left. |
| evidence | `../validations/d2v3-gpu-validation/v2/runs/d2v3r2val_20261014_185259/out/adoption_all_v1/adoption.json`, sha256 `5edc0b92…` |
| adoption | `d2v3_validation_evidence` reports **usable**. Every identity matches (objective `90e851fd…`, contract `1321017a…`, preregistration `b5281092…`, plan `31377a77…`). The executed commit equals the granted one. The recomputed verdicts equal the recorded ones. |

The run is deterministic where it should be. Every measured number below
equals revision 1's on the same items, candidates and tensors; only timings
differ. Revision 1's slack, applied to revision 2's numbers, flags the same
**175** positions.

## 1. Engineering validation result

| verdict | result |
| --- | --- |
| arithmetic | **PASS** (9 of 9 checks) |
| integrity | **PASS** (6 of 6) |
| row equivalence (scored_v1) | **PASS**: D2 may run scored_v1 |
| **validation** | **PASS** |

**Revision 2's KL family** covers 46,056 positions: 16 items, bypass layers
5, 18 and 31 of W2's FFN child.

| check | measured | registered | |
| --- | --- | --- | --- |
| teacher Top-K + tail KL, float32 vs float64 | max 1.146e-4; p99 1.84e-5; 0 violations; worst excess −9.08e-5 | 1e-4 + 1e-4·KL64 | pass |
| union KL, float32 vs float64 | max 1.146e-4; p99 1.84e-5; 0 violations; worst excess −9.08e-5 | 1e-4 + 1e-4·KL64 | pass |
| float64 order `KL_T ≤ KL_U ≤ KL_full` | 0 lower and 0 upper violations | slack 1e-12·(1 + KL) | pass |
| float32 crossings beyond the combined bound | **0** (lower and upper) | 0 | pass |
| anomalies (float64 violations + crossings beyond the bound + finiteness mismatches) | **0** | — | — |

**Error ÷ tolerance across every position** (identical for both
partitions, to within 10 positions):

| bin | count |
| --- | --- |
| [0, 0.01) | 34,880 |
| [0.01, 0.1) | 11,064 |
| [0.1, 0.25) | 102 |
| [0.25, 0.5) | 10 |
| ≥ 0.5 | 0 |

**Crossings, recorded per side:**

| side | count | largest | combined bound there | float64 at the largest (topk / union / full) |
| --- | --- | --- | --- | --- |
| lower `KL32_T − KL32_U > 0` | 89 | 4.8e-7 | 1.21e-3 | 5.042638075 / 5.042638098 / 5.042638288 (`rag_evidence/squad_v2-000000#t1`, layer 5, position 160) |
| upper `KL32_U − KL64_full > 0` | 15,266 | 1.01e-4 | 2.93e-4 | 1.9255773523 / 1.9255773525 / 1.9255773529 (`tool_calling/glaive-000219#t5`, layer 5, position 304) |

- The upper crossings are float32 noise of either sign around a union KL
  that almost equals the full KL: the union/full ratio has median
  0.99999994. All of them are inside the bound.
- Per-position records: 15,311 positions qualified (every crossing, plus
  every position above half its tolerance). The 2,000 worst are kept in
  full, as the cap states. The histograms above cover every position.

**Attribution** is diagnostic and not gated. The same reduction was run on
the head logits the oracle reads:

| | float32 reduction alone vs float64 | store input path vs same-input |
| --- | --- | --- |
| teacher Top-K + tail | max 3.37e-5, p99 1.16e-5 | max 1.15e-4, p99 1.80e-5 |
| union | max 3.37e-5, p99 1.16e-5 | max 1.15e-4, p99 1.81e-5 |

- The largest error comes from the store's teacher input path: block-shaped
  head GEMMs, plus row-gathered retrieval for the candidate-only entries.
  It does not come from the reduction.
- The Top-200 support re-formed from the recomputed head equals the stored
  support at 99.993% of positions.
- **The answer to "is there any true mathematical anomaly beyond the
  numerical error bound": no.**

**Other arithmetic checks:**

| check | measured | tolerance |
| --- | --- | --- |
| retrieval vs full head | 1.11e-4 | 2e-4 |
| K+1 confidence vs float64 | ≤ 4.4e-6 | 1e-5 |
| K+1 entropy vs float64 | ≤ 2.3e-5 | — |
| item objective reduction | 7.2e-9 relative | 1e-5 |
| raw-KL control (7 metrics) | 0.0 | 1e-6 |

**Integrity:**

- The teacher identity is the bound one.
- Both union stores match their declaration, including the executed build.
- The full-scope FFN arm is weighted by this session's store, and its
  denominator equals the weight sum.
- DEPTH reads the root union store with retrieval.
- The DEPTH table is consistent.
- Every v3 evaluation reads its union store with **0 teacher forwards**.

**Row equivalence:**

- DEPTH all_v1 vs scored_v1: 36 candidates, max relative difference 3.4e-9,
  ρ 1.0, same choice.
- State evaluation: 28 metrics, max absolute difference 5.4e-8, against a
  tolerance of 1e-5.

**The validity report** (schema v2) now states the store's own definition:
`c_T = 1 − H_T^{K+1}/log(K+1)`, 201 categories, normalized by log 201 =
5.303 nats.

## 2. The registered rubric, applied literally

The plan registered the rubric before any v3 GPU number existed.

| condition | measured | holds? |
| --- | --- | --- |
| validation PASS | PASS | yes |
| **GO**, any one of the following: | | |
| · the weighted and unweighted DEPTH choices differ | both 17 | no |
| · ρ(weighted, unweighted) < 0.95 | 0.997 | no |
| · K+1 c_T IQR ≥ 0.11 on both profiles | 0.123 and 0.095 | no |
| **HOLD**, all of the following: | | |
| · IQR ≤ 0.06 on both profiles | 0.123 and 0.095 | no |
| · ESS ≥ 0.99 | 0.975 and 0.981 | no |
| · ρ ≥ 0.99 with the same choice | 0.997, same choice | yes |

**The literal outcome is "between"**: a PASS that is neither GO nor HOLD.
The rubric then says "reported as measured with both readings; the
maintainer decides".

## 3. Descriptive science

None of this is gated, and none of it is evidence that student reasoning
improves. It rests on one DEPTH round (round 0) of one non-root parent (W2's
FFN child) on domain_balanced. Beam search, later states, the cumulative
WIDTH/ATTENTION path and recovered correctness are not measured by Stage W.

**Teacher confidence (K+1) over scored positions:**

| profile | positions | median | IQR | ESS fraction | q05 |
| --- | --- | --- | --- | --- | --- |
| domain_balanced@v1 | 44,746 | 0.9971 | **0.123** | **0.975** | 0.566 |
| reasoning_heavy@v2 | 46,825 | 0.9990 | **0.095** | **0.981** | 0.642 |
| v2 (Stage V), for reference | | 0.999 | 0.042–0.055 | ≥ 0.994 | |

**How the weighting moves the objective.**

- The weighted objective is **84.9%** of the raw KL mass; v2's was 96–97%.
- Spearman(c_T, KL) = −0.72 on this child: high-KL positions are discounted.
- Shares of the unweighted KL against shares of the weighted objective:

| group | unweighted KL | weighted objective |
| --- | --- | --- |
| general | 37.6% | 32.4% |
| math | 22.4% | 24.6% |
| code | 14.8% | 16.2% |
| tool | 15.1% | 16.1% |
| RAG / multi-hop | 10.2% | 10.8% |
| reasoning span | 50.1% | 53.8% |
| final-answer span | 11.3% | 12.6% |
| other scored positions | 38.6% | 33.6% |

**Full-scope FFN selection.**

- Full-scope weighted against D1's narrow unweighted: mean overlap 0.964,
  min 0.935. No layer of 36 is identical, and layers 24–31 change most.
- v2's overlap was 0.984 / 0.973.
- This contrast mixes scope and weighting.

**DEPTH, the four metrics over the same 36 candidates:**

| pair | ρ | same choice | top-3 |
| --- | --- | --- | --- |
| parent KL vs root teacher-Top-K KL | 0.804 | no (26 → 17) | 1/3 |
| root teacher-Top-K vs root union, unweighted | 1.000, identical order | yes | 3/3 |
| root union, unweighted vs weighted | 0.997 | yes (17) | 3/3 |

The top five under each metric:

- parent KL: 26, 29, 30, 20, 21;
- root teacher-Top-K and unweighted union: 17, 18, 30, 10, 16;
- weighted union: 17, 18, 30, 16, 31.

The gap between the first and second choice is 1.07–1.14% under every root
metric.

**Union against teacher-only Top-K, as a fraction of the full KL:**

- mean 0.99643 against 0.99571: about 17% of the gap closed;
- p01 0.931 against 0.910: about 24% closed.
- The union holds 313 entries on average (p01 259, max 400). On average 87
  of them are also in the teacher's Top-200.
- In state evaluation, the union raw KL exceeds the teacher-Top-K KL by
  8.4e-4 nats (0.05%).

**State evaluation and scored_v1:**

- The raw-KL control is exact.
- v3 evaluation takes 5.10 s against 10.29 s on the historical path, with 0
  teacher forwards against 27.
- Peak memory is 17.1 GB against 19.9 GB.
- scored_v1 is equivalent (section 1).
- Batch-shape drift between bsz 1 and bsz 3, reported only: state objective
  max |Δ| 0.033; store confidence max |Δ| 0.110 / 0.050.

**Float32 head against native bf16.** This bounds the head's effect, mixed
with bsz 1-vs-3 drift:

- KL: mean 4.3e-4 nats, p99 5.5e-3;
- |Δc_T|: mean 1.7e-3;
- the Top-200 set is identical in 4.0% of rows (bf16 boundary ties: 80% of
  rows).

**GPU cost and memory:**

- $0.2493 for this session.
- Step peaks: W4 22.7 GB, W5 20.1 GB, W6 19.9 GB, W3 14.7 GB, W2 14.1 GB,
  W1 10.2 GB, on a 47.7 GB L40S.
- The DEPTH reference stayed cached under a hoarded pool, with 0 recomputes.

## 4. Re-pricing from the adopted evidence

The measured union terms replace the provisional ones, with a ×1.5 margin
for a single measurement:

| term | measured | bound |
| --- | --- | --- |
| per DEPTH candidate | 0.450 s | 0.674 s |
| per state evaluation | 0 s (faster than the historical path) | 0 s |
| per DEPTH expansion | 27.5 s | 41.3 s |
| statistics operators | ×1.040 | ×1.061 |
| union stores | 0.36 min | 0.54 min |

W2's full/narrow ratio was 1.099 in revision 1 and 1.040 here, so the
operator factor carries run-to-run variance of about 6%.

| D2 v3 search | expected | planning bound | vs the $30 cap |
| --- | --- | --- | --- |
| **in force** (all_v1, cache held) | **$11.32** | **$24.32** | fits |
| scored_v1 (now eligible: v3's own equivalence passed) | — | $23.69 | fits |
| cache regressed (records_max) | — | $33.94 | over the cap; the driver's cache guard stops the search |

**Live limits** (`derive_budget.py`):

- project remaining: $118.9848 of $590;
- engineering remaining: $3.1804;
- package remaining: $131.1428;
- formal remaining: $127.9624.

**Programme at hard ceilings with teardown reserves** (cache_fix_validated):

| scenario | hard | outcome |
| --- | --- | --- |
| D2 | $54.60 | fits; $64.38 left |
| D2 + D3 | $107.26 | fits; $11.72 left |
| D2 + D3 + fresh confirmation | $122.76 | **$3.78 short** (fits at expected, $81.51) |

- **The reservation rule is not satisfied.** It requires that D2 at its
  hard ceilings still leave D3 and a fresh confirmation covered. The
  shortfall is $3.78.
- **Levers listed by the programme record** (none is taken here):
  - D2 at scored_v1: −$0.63, leaving $3.15 short;
  - re-deriving after D2's actual spend: D1 spent $10.03 of a $21.49 search
    ceiling;
  - D3 at K = 2: −$9.66, but the maintainer has excluded this in advance;
  - a cap change: a maintainer decision.
- The rule may not be sidestepped by splitting stages.

## 5. Recommendation

| | |
| --- | --- |
| engineering | **PASS**: every gate holds, the evidence is adopted, scored_v1 is eligible, and the search fits its session cap at $24.32 |
| registered rubric (literal) | **between**: neither GO nor HOLD; the maintainer decides |
| descriptive science | The weighting materially reweights the objective (84.9% of mass; general −5.2 pts, reasoning +3.7 pts) and changes 3.6% of FFN neurons. It does not move this round's DEPTH choice. The union improves the KL approximation slightly and the DEPTH order not at all. The root reference alone moves the DEPTH choice. |
| budget | The search fits its cap; the programme reservation is **$3.78 short** |

**Recommendation: proceed to the maintainer's review of preregistration v3
and the formal full search, as the maintainer's stated inclination
requires.**

- Both conditions now hold: Stage W PASS, and a measured price ($24.32)
  under the hard per-session cap.
- **The claim boundary should be stated with it.** A D1-vs-D2 difference is
  attributable to the combined v3 objective. On this evidence its DEPTH part
  is mostly the root reference. Nothing here supports attributing it to the
  weighting alone, or predicting better recovered correctness.
- **Before any formal D2 session:**
  1. the $3.78 reservation shortfall needs a maintainer decision;
  2. preregistration v3 approval;
  3. `phase_d2` in `funds_formal_sessions_of`;
  4. the D2 search grant.
- No change to the confidence formula, K, the mapping, the support or the
  beam is proposed.
