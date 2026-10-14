# D2 Stage V GPU evidence review (2026-10-14)

**Status:** the Stage V engineering validation is COMPLETE and CLOSED, and
its evidence is **adopted**. Nothing is billing. This record reviews the
evidence before any formal D2 decision. It authorizes nothing: the D2 search,
recovery, screening and confirmation are not granted.

| | |
| --- | --- |
| run | `d2val_20261014_103301` (acquisition attempt 2; attempt 1 was a $0 capacity refusal, `d2val_20261014_103104`) |
| pod | `pln3gsuw6lvlzb`, NVIDIA L40S, $1.09/h live secure |
| runtime | torch 2.11.0+cu128, transformers 5.13.1, CUDA 12.8 |
| executed commit | `d09fc8f754dd38f87c3b1250b559ada1847609f5`. The pod's `source` and the stage-V premises both record it; it equals the grant's `executable_commit` and the tree was clean. |
| wall | 35.7 min (V1–V5 in 1919 s) |
| cost | **$0.6532** all-in, of the $1.70 grant |
| teardown | provider-confirmed on the first attempt; the confirmation is hashed into the campaign |
| evidence | `../validations/d2-gpu-validation/v1/runs/d2val_20261014_103301/out/adoption_all_v1/adoption.json`, sha256 `a75e4104…` |
| adoption | `d2_validation_evidence` reports usable. The identities (objective, contract, preregistration, plan) and the executed commit match, and the verdicts recomputed from the measurements equal the recorded ones. |

## Verdicts, recomputed from the measurements

| verdict | result | what it rests on |
| --- | --- | --- |
| arithmetic | **PASS** | see below |
| integrity | **PASS** | see below |
| row equivalence (scored_v1) | **PASS** | see below |
| **validation** | **PASS** | arithmetic and integrity |

**Arithmetic, both profiles, every check:**

- Entropy, float32 chunked against float64 on identical teacher logits:
  - entropy at most **2.3e-6 nats** against a 1e-4 tolerance;
  - chunk 512 against 64 at most **1.4e-6**;
  - confidence at most **2.1e-7** against 1e-5.
- The c×KL item mean against float64 is at most **4.8e-9 relative** (1e-5).
- The raw-KL control is **0.0** over 7 metrics.

**Integrity, all checks:**

- B's four identities and the teacher binding verified.
- Both root stores match their declaration at bsz 1 and 3.
- Full scope:
  - FFN, WIDTH and ATTENTION executed both arms;
  - each full-scope denominator is the weight sum exactly (42920.87 = 42920.87);
  - the narrow arms are unweighted;
  - the children reach the target, finite.
- DEPTH reads the root, and every decision score equals the recorded
  objective.

**Row equivalence (scored_v1):**

- DEPTH: 36/36 candidates, max relative **0.0**, same choice, Spearman 1.0.
- State evaluation: 23 metrics, max |Δ| **8.0e-7** (1e-5).
- Consequence: **D2 may run scored_v1**.

## Engineering evidence

**Root stores (V1).**

- Build time: **5.8 s and 5.2 s** at bsz 3 (4.9 s and 4.8 s at bsz 1).
  - `calib.domain_balanced@v1`: 59,763 positions, sketches 0.135 GiB.
  - `calib.reasoning_heavy@v2`: 44,324 positions, 0.100 GiB.
- Mode is `root_reference_store`, cached.

**Device handoff (V4/V5).**

- V3's FFN child moved cpu → cuda:0 in 1.6 s.
- The parent forwards, the root sketch and the confidence were all on cuda:0
  before DEPTH ran.
- Peak VRAM: **20.6 GiB** (V4) and 18.2 GiB (V5), under a deliberately
  hoarded 30.4 GiB pool. Both references stayed cached: the root store, and
  the parent sketches with 0 recomputes.

**Timings:**

| step | measurement |
| --- | --- |
| V4, one DEPTH round on a non-root parent, 36 candidates | 172.0 s |
| — of which the objective (c×KL), unsynchronized | 27.9 s; this includes the forwards' async tail, since the scored_v1 round measured only 1.5 s |
| — of which the round-0 parent-KL diagnostic | 27.2 s |
| V5, the same round at scored_v1 | 155.2 s, −9.8% |
| V6, state evaluation | D2 8.64 s against KL-only 7.95 s: **+0.70 s** |
| V3, full over narrow scope | ATTENTION +3.5%, FFN +2.2%, WIDTH +2.4% |

**BF16 execution drift, bsz 1 against bsz 3** (reported, never gated):

- Root-store entropy:
  - p50 **6e-4 nats**, p99 **0.084**, max **1.07** on domain_balanced;
  - max 0.55 on reasoning_heavy.
- Top-200 support: Jaccard mean **0.976**, but only 7.7% / 4.7% of positions
  have identical sets.
- State-evaluation objective metrics: max |Δ| **0.235**.

This confirms why the review separated drift from the gates. bsz 1 and bsz 3
are different computations, and D2 is frozen at bsz 3.

## Science diagnostics (descriptive; nothing here changes a definition)

**Teacher confidence c_T over scored positions:**

| profile | positions | mean | median | IQR | ESS fraction | min |
| --- | --- | --- | --- | --- | --- | --- |
| domain_balanced@v1 | 44,746 | 0.959 | 0.9987 | **0.055** | **0.994** | 0.128 |
| reasoning_heavy@v2 | 46,825 | 0.967 | 0.9995 | **0.042** | **0.995** | 0.144 |

- By span: reasoning 0.979 / 0.981 against answer 0.992 / 0.992.
- By domain: `general` is the lowest at 0.857 / 0.855; math, code, RAG and
  tool are 0.97–0.99.
- By token type: word tokens 0.944 / 0.954 against digits 0.994 / 0.996.
- c_T tracks the teacher's own target probability (Spearman **0.975 /
  0.976**). Mean c_T is 0.98 where the target is the teacher's top-1 (86–89%
  of positions) and 0.83–0.84 elsewhere.

**Against B's root KL (V2):**

- Spearman(c_T, KL) = **0.012 / −0.041**: essentially uncorrelated.
- The weighted objective is **96.3% / 96.9%** of the unweighted KL mass.
- The span shares barely move (reasoning 61.5% → 62.6%).
- On B's state evaluation: objective 8.252 against raw KL 8.568.

**DEPTH on a non-root parent (V4),** comparing the same 36 candidates:

| pair | Spearman | same choice | top-3 overlap |
| --- | --- | --- | --- |
| unweighted vs weighted root KL | **0.999** | yes (layer 17) | 3/3 |
| parent KL vs unweighted root KL | 0.816 | **no** (26 vs 17) | 1/3 |
| parent KL vs weighted root KL | 0.806 | **no** | 1/3 |

**Full vs narrow scope (V3), selection overlap:**

- ATTENTION: 35/36 layers identical (mean Jaccard 0.997, min 0.882).
- FFN: mean Jaccard 0.984, min 0.973, no layer identical.
- WIDTH: subspace overlap 0.989 (random 0.40). This is approximate, recovered
  from the bf16 children.

**What this means, stated as measured.**

- **The teacher-confidence weighting has a narrow dynamic range on these
  mixtures.** It moves the objective about 3–4%. On this parent it leaves the
  DEPTH ranking essentially unchanged, and it shifts the statistics operators'
  selections only slightly.
- **The larger D1 → D2 difference measured here is the reference change**
  (parent → root), which reorders DEPTH (ρ = 0.81) and changes its choice.
- The frozen definition and normalization are not revised on this evidence
  (contract and preregistration rule). Under the frozen claim boundary, a
  D1-vs-D2 difference would be attributable mostly to the root reference,
  with the weighting a small perturbation. This applies one DEPTH round on
  one parent, and B.

## Re-pricing (Stage V measured, one-measurement margin ×1.5 on added work)

| | expected | planning bound |
| --- | --- | --- |
| D2 search (cache_fix_validated DEPTH cell) | **$11.75** | **$25.84** (cap $30) |
| D2 search if the cache regresses (records_max) | — | $35.47 (above the cap: the driver's cache guard stops the search) |

The measured terms replace the provisional ones:

- per candidate: 0.775 s, an upper estimate;
- per DEPTH expansion: 27.2 s;
- per state evaluation: 0.70 s;
- statistics operators: ×1.027;
- root stores: 0.18 min.

**Programme against the live limits** (project remaining $119.49, $30 per
session), at hard with reserves:

| scenario | hard | outcome |
| --- | --- | --- |
| D2 | $56.12 | fits; $63.37 left |
| D2 + D3 | $108.78 | fits; $10.71 left |
| D2 + D3 + fresh confirmation | $124.28 | **$4.79 short** (fits at expected, $81.94) |

The closed validation now counts at its bill ($0.65), not its grant ($1.70).
K = 4 and both caps are unchanged.

## Recommendation for the formal D2 search

- **Engineering: GO.**
  - Every gate passes.
  - `scored_v1` is eligible, which saves about 10% of DEPTH time.
  - The caches held under a hoarded pool.
  - The device handoff is verified.
  - The search fits its session cap at $25.84.
- **Science: a maintainer decision before approving preregistration v2.** As
  frozen, D2 is valid and executable. The measured weighting contrast is
  small, so its search would mainly test the root reference with a mild
  confidence perturbation.
  - *Approve v2 as frozen:* the claim boundary already forbids attributing a
    D1/D2 difference to the weighting alone.
  - *Hold for a new scientific version:* a redesign is a maintainer decision
    with its own preregistration, never a post-hoc change to this one.
- **Budget:**
  - The D2 search fits.
  - The $4.79 programme shortfall is unchanged in kind: decide it before D3's
    or a fresh confirmation's formal sessions.
  - Re-derive after D2's actual spend.
