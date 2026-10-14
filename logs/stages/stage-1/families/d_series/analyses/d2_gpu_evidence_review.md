# D2 GPU evidence review — R0 and the reference calibration (2026-10-13)

> **Read with the 2026-10-14 D2 redesign (contract revision 6).** In this
> review, "the D2 objective" means the student-uncertainty D2 (`063bafd4…`).
> That objective is now SUPERSEDED and was never executed. Its measured
> overhead and its scored_v1 PASS therefore apply to it alone. The overhead
> prices **D3**, which does the same student Top-K entropy work, and never
> the redesigned D2. The redesigned D2 is priced provisionally until its own
> GPU validation runs
> ([`../../../phase_d2/plans/d2_gpu_validation_plan.json`](../../../phase_d2/plans/d2_gpu_validation_plan.json)).
> Objective-independent findings still hold: R0's PASS, the validated cache
> fix, λ, the thresholds and the rescoring cost.

**Status:** both engineering sessions COMPLETE and CLOSED; no resource billing.
This record reviews their evidence before the D2 search is priced or
authorized. It authorizes nothing.

| session | run | pod | GPU | wall | cost | grant |
| --- | --- | --- | --- | --- | --- | --- |
| R0 (`scored_v1`, KL path) | `r0_20261013_154325` | `2xmyzes6r0vcge` | L40S, $1.09/h | 82.35 min | **$1.5074** | $4.00 / $3.60 |
| reference calibration (stage L) | `refcal_20261013_170714` | `7cr5l46wa7ptfy` | L40S, $1.09/h | 44.08 min | **$0.8069** | $1.76 / $1.42 |
| **total** | | | | | **$2.3143** | $5.76 |

- **Pod lifecycle.** Each pod was created by its launcher, watched by an
  independent watchdog, and deleted by the launcher once its evidence was
  fetched. The provider confirmed deletion on the first attempt each time
  (`teardown_confirmed.txt` in each run directory). The provider lists 0 pods.
- **Failed attempts.** Two R0 create calls were refused for lack of L40S
  capacity before any pod existed. Both are booked at $0.
- **Where the records live.** Owners are each run's `cost.json`, `launch.log`
  and `watchdog.jsonl` under `validations/<name>/v1/runs/<run>/`, and each
  `campaign.json`, which books the subruns.

## R0 — `ExecutionConfig.prediction_rows="scored_v1"` on the KL path: **PASS**

All tolerances were registered in R0's `plan.json` before the session. The
verdict was computed off-pod by `prediction_rows_r0_compare.py`, in
`r0_comparison.json`. History is a8, the oracle is `all_v1`, and the
treatment is `scored_v1`, all on one pod.

| check | domain_balanced@v1 | reasoning_heavy@v2 | tolerance |
| --- | --- | --- | --- |
| DEPTH removal order, history = oracle = treatment | [2, 3, 32, 16, 26, 15, 17, 27] ×3 | [2, 3, 32, 16, 15, 26, 17, 27] ×3 | exact |
| max relative candidate-score difference, 260 candidates | 1.45e-7 | 1.65e-7 | 1e-5 |
| DEPTH invocation, oracle → treatment | 1166.3 → 1108.0 s (0.950) | 1154.0 → 1111.2 s (0.963) | reported |
| DEPTH peak memory | 16.28 → 13.91 GiB | 16.25 → 14.24 GiB | reported |

State evaluation, ranked metrics:

- Maximum absolute drift is **5.7e-8** under full_vocab and **1.28e-7**
  under reference Top-K + tail, against a tolerance of 1e-5.
- The worst domain agrees under both supports.
- Peak memory falls from 20.49 to 18.65 GiB.

The oracle reproduces history exactly: it has the same protocol ids as a8
(`d463a1dc…` and `9c9b0b43…`) and the same removal orders. `scored_v1` is a
new execution fingerprint (`52fa6135…` and `79e43b8c…`) and is now
**eligible** for the KL path.

## Reference calibration — stage L

### λ and the diagnostic thresholds (the deliverable)

The pair is teacher `768f209d` against B `53e30566…`. B was rebuilt on the pod
in 1270.9 s at A_bsz1, and all four identities were verified.

| quantity | value | rule |
| --- | --- | --- |
| D3 λ | **7.34** (unrounded 7.3358; IQR KL 1.208, IQR confidence 0.1647) | λ rule v2, 147 unique items |
| sensitivity (not used) | SD ratio 7.22; with duplicate draws 7.52; domain_balanced alone 6.53; reasoning_heavy alone 7.01 | — |
| high_confidence (p90) | 0.7758 | `diagnostic_threshold_rule` |
| high_kl (p90) | 13.854 nats | over 98,760 scored positions |
| low_topk_mass (p10) | 0.4323 | |

The pool is the registered one: 209 draws, 147 unique items. The record is
committed as `analyses/d2_d3_reference_calibration.json` (`record_sha256`
`ea5d3ff9…`). Contract revision 5 freezes D3 from it (`a01b1be7…`).

The KL scale is B's. As an untrained initialization, B's state-evaluation
teacher KL is 8.57 nats (equal-domain mean), and its per-item KL quartiles
are 7.87 / 8.49 / 9.07.

### The reference-cache fix, on the GPU: **validated**

The setup was one DEPTH round at the root (36 candidates), with the allocator
deliberately hoarded before the operator ran. That reproduces D1's search,
which entered 15 of its 16 DEPTH expansions with 0.12 GiB driver-free.

| arm | driver-free before probe | reserved before | reclaimable | driver-free after probe | mode | fills / hits / recomputes | wall | peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KL-only | 0.249 GiB | 43.60 GiB | 34.94 GiB | 35.17 GiB | **cached** | 67 / 2345 / 0 | 209.2 s | 17.4 GiB |
| D2 objective | 0.249 GiB | 43.60 GiB | 34.94 GiB | 35.17 GiB | **cached** | 67 / 2345 / 0 | 215.9 s | 17.4 GiB |

- The sketch estimate is 0.134 GiB.
- `headroom_source` reads "cuda.mem_get_info after empty_cache".
- `cache_fix_validated = true` on both arms.

**Cached against recomputed references.** A KL-only arm was forced to
recompute and compared with the cached table: the maximum relative
difference is **0.0** over 36 candidates, with the same choice and Spearman
1.0. That is EXACT, as registered. Recomputing took 383.8 s, **1.83×** the
cached arm.

**Limit.** This is one root profile. It proves the mechanism; it does not
prove that every expansion of a search stays cached. The D2 search therefore
keeps recording cache modes, and its driver stops the search safely on the
first recomputing DEPTH expansion while the cached price is in force.

### The D2 objective's overhead: **measured**

| path | KL-only | D2 | overhead |
| --- | --- | --- | --- |
| DEPTH round, 36 candidates | 209.2 s | 215.9 s | **+0.184 s per candidate** (+3.2%) |
| state evaluation, 80 items | 8.20 / 8.38 s | 8.66 s | **+0.466 s per evaluation** (+5.7%) |

- **Raw-KL control on the GPU.** Of `state.objective.raw_kl.*`, 7 metrics
  were compared with D1's KL metrics on the same B evaluation: the maximum
  absolute difference is **0.0**.
- **The price uses one measurement:** the expectation is the measurement,
  and the planning bound is 1.5× it.

### D2 objective under `scored_v1`: **PASS**

The tolerances are R0's, registered in the driver before the session.

| comparison | result | tolerance |
| --- | --- | --- |
| state evaluation, 24 `state.objective.*` metrics | max \|Δ\| **8.0e-7** | 1e-5 |
| DEPTH round, 36 candidate objective values | max relative **2.66e-7**; same choice; Spearman 1.0 | 1e-5, exact choice |
| DEPTH wall time | 215.9 → 203.7 s (−5.6%) | reported |

D2's objective tables and choices were compared, never recorded, so nothing
resembling a D2 search result exists. Under the review's rule, `scored_v1` is
eligible for D2 because this objective-specific comparison passed, not merely
because R0 did. Choosing it for the search is a matter for the D2 grant.

### Rescoring cost: **measured**

- On the real teacher and B, the rescoring costs 0.1155 s per 1k tokens.
  Projected to D1's 951,268 tokens per probe, that is **1.83 min/probe**,
  against the 5.0 min estimate.
- It now sits inside the behavioural ceilings, with the 1.5× margin for a
  single measurement.

## The updated budget model

The budget is `plans/d2_budget_v1.json` plus the programme snapshot
`plans/d_series_programme_budget.json`. Live limits are a project remaining
of $120.14, $30 per session, and an engineering remaining of $4.34.

- **D2 search price in force: `cache_fix_validated`.** The switch was
  automatic: the stage-L session is booked in its campaign with sha256
  `6806eb84…`, and the identity checks (objective, both rules, incumbent,
  completion) all pass.
  - Expected: **$10.80**.
  - Planning bound: **$22.38**. At the records price it would be $32.01,
    above the cap.
  - The enforceable stop is the session cap, plus the soft stop and the
    cache-regression guard.
- **D2 total:**
  - search $22.38, screening $20.83, confirmation $8.55, each at hard with
    the measured rescoring included;
  - **$52.66** with teardown reserves, **$34.38** expected.
- **Programme:**
  - D2 + D3 is **$105.32** at hard. It fits, with $14.82 left.
  - Adding one fresh confirmation makes **$120.82**: **$0.68 short** at hard,
    $80.99 expected.
  - The proposed reservation rule is short by the same $0.68.
- **Engineering reserves are released against the actual bills.** Both
  campaigns are CLOSED, and their recorded costs are already inside the live
  limits.

Levers, none applied:

- run D2 under `scored_v1`, now eligible; about 5% of DEPTH time, not yet
  priced in;
- re-derive after the D2 search from actual spend; D1 spent $10.03 of
  $21.49;
- D3 at K = 2, which saves $9.66 at hard;
- a cap decision.
