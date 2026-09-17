# Phase-C2 full search — performance round

Analysis for the narrowly scoped performance-engineering round of 2026-09-18.
**Changes no scientific semantics.** The space, beam, operators, exclusions,
calibration, the 260-evaluation rule, the state-eval suite, the ranking policy
and Top-5 semantics are all untouched.

Every figure below is measured from the two committed telemetry files
(`phase_b/runs/attempt5/search_telemetry.jsonl`,
`phase_c2/runs/attempt4/evidence/telemetry.jsonl`) or from the code, not
estimated from a speedup claim.

---

## What the telemetry actually says

Pooled over 126 expansions, with `operator_seconds` and `parent_load_seconds`
included — the two fields the cost model prices on and which a first pass at
this analysis omitted:

| implementation | n | mean min | **max min** | where the max goes |
| --- | --- | --- | --- | --- |
| `depth.causal_kl_greedy_v1` | 19 | 27.84 | **36.07** | `operator` 33.34 (92%) |
| `width.global_pca_v0` | 38 | 2.24 | 3.93 | `state_evaluation` 2.92 (74%) |
| `composite.stage1_sandwich_v0` | 2 | 3.84 | 3.88 | `state_evaluation` 3.09 (80%) |
| `attention.weight_proxy_v0` | 18 | 1.57 | 3.86 | `state_evaluation` 2.59 (67%) |
| `depth.positional_v0` | 6 | 1.81 | 3.73 | `state_evaluation` 2.96 (79%) |
| `attention.activation_importance_v1` | 14 | 2.58 | 3.59 | `state_evaluation` 2.73 (76%) |
| `ffn.activation_importance_v0` | 29 | 2.05 | 3.00 | `state_evaluation` 2.32 (77%) |

Two distinct cost structures, and they need different optimizations:

* **DEPTH** is 92% its own operator work — the 260 candidate subsets over 67
  calibration items — and is an order of magnitude more expensive per expansion
  than anything else. It dominates the beam ceiling.
* **Every other operator** is 67–80% `state_evaluation_seconds`.

A first reading of this analysis summed only the five phase fields and
concluded that state-eval was 96.3% of everything and DEPTH was 1.73 min. That
was wrong because `operator_seconds` sits outside those five fields, and it
would have sent the whole round at the wrong target. The corrected split is
above.

---

## The reference cache is the largest single waste in the search

Across all nineteen measured DEPTH expansions:

| items | cached | recomputed/candidate | recomputes | `mem_get_info` free | mode |
| --- | --- | --- | --- | --- | --- |
| 67 | 67 | 0 | 0 | 26.3 GiB | cached |
| 67 | 52 | 15 | 3,900 | 20.3 GiB | partial |
| 51 | 23 | 28 | 8,320 | 11.5 GiB | partial |
| 67 | 35 | 32 | 8,320 | 11.5 GiB | partial |
| 67 | **11** | **56** | **14,560** | **2.6 GiB** | partial |

Pooled: **323,180 ablated forwards and 182,780 reference recomputes — 36.1% of
every forward pass the operator performed.** With a fully resident cache they
would be 0.4%.

The cache needs 16.91 GiB (59,763 positions × 151,936 vocab × 2 B, bf16). On a
48 GB L40S it fitted once and then saw 2.6 GiB free for six consecutive
expansions. **That is a memory-availability question, not a compute one**, and
`mem_get_info` alone cannot answer it — which is what candidate 4 instruments.

---

## Candidate 1 — state-eval reduction on the device: ADOPTED

`StateEvaluator` moved both `[T, ~152k]` float32 logit tensors to the host with
`.float().cpu()` and reduced them there. At the frozen suite that is ~0.5 GiB
per model per item across the bus, and then a 152k-wide log-softmax on the
host. This is where 67–80% of six of the seven operators' time went.

`distortion` is now device-resident when its inputs are: chunk-wise float32
reductions accumulated in **float64 device scalars**, with one host transfer of
five numbers at the end instead of six synchronising `float()` calls per chunk.
The tag masks move to the logits' device — not an optimization but a
requirement, since a host boolean mask indexing a CUDA tensor raises.

Equivalence, measured on CPU:

* at fixed chunk size the restructured accumulators are **exactly** equal to
  the previous path — relative drift `0.000e+00` over six cases;
* the device branch is driven at `$0` by overriding a named predicate, so it is
  not a branch that first executes on a paid pod;
* a tag matching no position is still omitted rather than reported with zero
  positions.

**Chunk size is not neutral: it moves the result by `8.7e-08` relative.** That
is five orders below the ~`7.8e-3` threshold C2's own comparisons use, and it
is the reason "preserve chunking" is a real constraint on the other candidates
rather than a formality.

What CPU cannot establish is whether CUDA's kernels agree with the host's to
the same tolerance. That is owed to the L40S validation.

## Candidate 2 — forward-KL-only hot path: ADOPTED

`depth.causal_kl_greedy_v1` read `sums["kl"]` from a six-quantity reduction
17,420 times per expansion and discarded the rest. `forward_kl_mean` computes
that one quantity and skips the reverse-KL term, both cross-entropy gathers and
both argmaxes. With no cross-entropy there are no targets to gather, so the
operator no longer builds or transfers a target list at all.

Preserved exactly: float32 log-softmax, the chunk boundaries, position counts,
domain/subtype aggregation, `greedy_removal`'s tie-breaking, and candidate
order.

Equivalence, against a predeclared `1e-9` tolerance:

* the scalar equals the full reduction's `kl` at five shapes and five chunk
  sizes;
* driven through the **real `greedy_removal`**, the removal order, every chosen
  layer, every round's `chosen_score`, and the complete per-round candidate
  table in order are identical;
* and the smallest candidate gap in that table is `1.7e-07` — **170× the
  tolerance** — so "identical choices" is evidence rather than a test that
  could not have failed.

## Candidate 3 — reuse reference-side normalization: NOT ADOPTED

Correct in principle: the intact-parent reference is identical across a round's
candidates, and its `log_softmax` is recomputed for every one of them.

It is not adoptable in the form that keeps the loop structure. The reference
cache holds **bf16 logits** at 16.91 GiB. Caching the *normalized* reference
means float32 log-probs:

| what is cached | bytes/value | size |
| --- | --- | --- |
| bf16 logits (today) | 2 | 16.91 GiB |
| float32 log-probs (candidate 3) | 4 | **33.83 GiB** |

33.83 GiB is double the binding constraint and **exactly the figure that
OOM-killed the first real rehearsal of this operator**. Caching log-probs in
bf16 instead is not an option: log-probabilities lose the precision the KL
depends on. So the in-place form trades the 36.1% recompute problem for a
guaranteed OOM.

**There is a form that works, and it is bigger than this round.** Invert the
scoring loop to item-outer: forward each item's reference once, normalize it
once, and reuse it across all 260 candidates while holding one reference and
one ablated tensor — about 1 GiB instead of 16.91 GiB. The per-candidate
arithmetic is unchanged, because each candidate still sums over items in item
order. That would eliminate every reference recompute *and* the cache itself.

It requires `greedy_removal` to hand the operator a whole round's candidate
list rather than calling `score_fn(skip)` once per candidate — a change to the
frozen selection driver's interface. Recorded as the next candidate; not taken
here.

## Candidate 4 — diagnose cache variability: INSTRUMENTED, no behaviour change

`memory_snapshot()` records, at the operator boundary and before the admission
decision reads anything:

* `mem_get_info` free and total — the driver's view;
* `memory_allocated` — what the allocator has given to live tensors;
* `memory_reserved` — what it has taken from the driver, including freed blocks
  it is holding for reuse;
* `reserved − allocated`, **the number the diagnosis turns on**;
* `total − free − reserved`, what this process's allocator cannot account for.

The two hypotheses have opposite remedies, which is why nothing is changed yet:

* if `reserved − allocated` is large, the cache is being sized against memory
  PyTorch is merely hoarding, and `empty_cache()` before sizing would recover
  it — cheap and safe;
* if `allocated` is itself large, live tensors hold the card and releasing the
  allocator's cache achieves nothing.

No unconditional flushing has been added. The snapshot rides into
`reference_cache.memory_at_admission` in the telemetry, is diagnostic only, and
never reaches a metric, a score or a hash.

---

## What is still owed

A bounded real-L40S engineering/performance validation, because these changes
touch the CUDA hot path and the existing validation certified a different
executable. It must establish numerical equivalence on real kernels, identical
DEPTH greedy decisions, identical state-eval/Pareto decisions, real wall-clock
improvement, device/dtype correctness, and the memory behaviour above — and it
must run with `AADISTILL_DEPTH_SYNC_TELEMETRY=1`, because without it the
forward/reduction split is unattributed and the share candidates 2 and 3 act on
is unmeasured.

**No new formal cost may be claimed from any of this until that run measures
it.** The pricing refresh follows the telemetry, not the other way round.

AUTHORIZES NOTHING.
