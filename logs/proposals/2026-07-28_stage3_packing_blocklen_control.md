# 2026-07-28 — Stage 3 packing / `block_len` control run (+ the project's first variance measurement)

**Status:** pre-registered, approved by the maintainer 2026-07-28. Two arms on
one L40S pod. This document is written *before* the run; the decision rules in
§6 are binding.

## 1. Why this run exists

Every Stage 3 conclusion so far compares runs that share one data path:
concatenate-then-cut packing (`pack_blocks`) at `block_len` 1024. That path
tears samples at every block boundary — with 64,484 samples packed into 21,610
blocks, roughly one sample in three is cut, and the second half of a cut sample
trains as a sequence whose premises are not in context.

`best_fit_blocks` (length-aware bin packing, after Ding et al., *Fewer
Truncations Improve Language Modeling*, ICML 2024, arXiv:2404.10830) was
implemented and unit-tested on 2026-07-28 but **was never reachable from a
config** — `build_blocks` still called `pack_blocks`. So no run has ever used
it. This session wires it up and measures it.

The reason it must happen *before* any teacher-trace experiment: a trace run
compared against the current baseline would confound "reasoning traces helped"
with "samples are no longer torn". Those are the same number today.

## 2. Corpus measurement (CPU, dev box, 2026-07-28)

Measured on `data/stage2_v1` train split with the pinned teacher tokenizer;
full output at `artifacts/stage3/packing_survey_v1.json` (gitignored).

- 64,484 samples / 22,133,631 tokens.
- Token length: p50 **201**, p90 **765**, p95 1,177, p99 **2,082**, max **6,658**.
- Samples exceeding a block: **4,433** over 1024 · **665** over 2048 · 194 over 4096.
- Per-group p90: long_context 5,195 · multihop_qa 1,915 · instruction 1,239 ·
  code_math 632 · tool_calling 597 · rag_evidence 298 · refusal 285 ·
  short_realtime 182.

| data path | blocks | draws @2700 steps | epochs | supervised tokens | efficiency | truncated |
|---|---:|---:|---:|---:|---:|---:|
| concat@1024 (**baseline**) | 21,610 | 43,200 | 2.00 | 11,681,472 | 1.0000 | — |
| best_fit@1024 | 19,560 | 43,200 | 2.21 | 9,520,678 | 0.9728 | 4,433 |
| best_fit@2048 (**control**) | 10,769 | 21,600 | 2.01 | 10,787,265 | 0.9624 | 665 |

**Correction to a logged number.** `logs/STATE.md` cited a corpus length p90 of
**1,508**. That figure is from `logs/decisions.md` (2026-07-28), where it was
measured on **four slices only** — the slices relevant to the teacher-trace
proposal — not the whole mixture. Corpus-wide p90 is **765**; 1,508 sits between
p95 and p99. The control is still indicated (tearing is caused by block
boundaries, not by sample length), but it is a weaker motivation than the
STATE.md phrasing implied. STATE.md is corrected in the same commit.

## 3. What changes, and what does not

Only the data path. `configs/stage3_s2v1_bl2048.json` vs
`configs/stage3_s2v1_from_init.json` differ in exactly:

| field | baseline | control |
|---|---|---|
| `packing` | absent (= `concat`) | `best_fit` |
| `block_len` | 1024 | 2048 |
| `batch.blocks_per_step` | 16 | 8 |
| `batch.micro_blocks` | 4 | 2 |

plus `run_name` / `out_dir`. Everything else is byte-identical: same Stage 1
init start point, same teacher and revision, same mixture v1, same freeze set,
CE 0.25 + full-vocab KD 1.0 at τ=1 scope `all`, lr 2e-4 / warmup 60 / cosine to
0.1×, fp32 master + bf16 autocast, 2700 steps, **seed 20260726**, eval every 150.

**Fixed budget (P6).** Tokens/step **16,384** in both. Total token budget
**44,236,800** in both. Optimizer steps 2,700 in both. Epochs 2.00 vs 2.01.
Same GPU class (L40S), same seed.

## 4. A padding bug this run would have hit, fixed first

`kd_scope: "all"` returned `ones_like(...)` — *every* position. Under concat
packing there is no padding, so that was every real position and the four logged
runs are unaffected. Under best-fit, 3.8% of positions are padding, so KD would
have trained the student to match the teacher on a degenerate pad run and
inflated the KD normalizer.

Fixed on 2026-07-28: `best_fit_blocks(return_content_mask=True)` returns a
real-token mask, threaded through `build_blocks` → `Trainer` → `prediction_mask`.
`"all"` now means every *real* position; with no content mask (concat) the
behavior is unchanged, so **the baseline needs no re-run**. Verified on the CPU
smoke run: `kd_positions` 2,028–2,038 of a possible 2,047, and `val_kd` at step 0
moved 0.9826 → 0.7935.

## 5. Declared asymmetries (things this design cannot hold constant)

1. **Supervised tokens are 7.7% lower in the control** (10,787,265 vs
   11,681,472). Best-fit truncates the tails of the 665 oversized samples and
   pads 3.8% of slots; concat keeps those tails as (torn) blocks. This is
   inherent to the treatment — token budget and supervised-token count cannot
   both be held fixed. The direction is **conservative**: the control trains on
   less supervision, so a win is not explained by more signal.
2. **`long_context` pays for this disproportionately** (group p90 5,195, far
   above `block_len` 2048). Best-fit truncates those samples rather than
   splitting them across blocks. If the control loses ground on long-context
   behavior specifically, sample-splitting (rather than truncation) is the
   follow-up, not a bigger block.
3. **The in-training primary val is not comparable across arms** — different
   `block_len` means different val blocks, and the seed replicate draws a
   different 64-block subset (`seed + 777`). The comparable metrics are
   `holdout_v1` NLL (per-sample, `--max-seq-len` 1024, independent of training
   `block_len`) and `eval_behavior_v0`.

## 6. Pre-registered decision rules

Arms: **A** = `s2v1_bl2048` (seed 20260726) · **B** = `s2v1_bl2048_seedB`
(seed 20260728, identical otherwise). B is the variance measurement — the seed
changes data order, packing order and the val subset, which is the run-to-run
variation two independently-designed runs would differ by.

The baseline `s2v1_from_init@2700` is re-scored **on this pod, same device,
same cap 512** before training starts (`score_refs.sh`), per the 2026-07-27
same-device comparability rule. Dev-box scorecards are not used for the delta.

- **R0 — noise floor.** `noise := |behavior_score(A) − behavior_score(B)|`.
  Report it with the number of prompts. This is the project's first noise
  estimate and it is reported whatever else happens.
- **R1 — adoption.** Adopt best-fit@2048 as the default Stage 3 data path only
  if `behavior_score(A) − behavior_score(baseline) > noise` **and** the delta is
  positive. Otherwise the baseline data path stands and the control is recorded
  as neutral-or-negative.
- **R2 — guard rail.** If `holdout_v1` NLL regresses more than **1%** against
  the baseline's 3.8285, do not adopt on behavior alone; report it as a
  tradeoff and escalate the decision.
- **R3 — the stated mechanism (directional, secondary).** The hypothesis on
  record is that *grounding and multi-hop improve, because those are the long
  slices being cut*. Note the corpus measurement partly contradicts it:
  `rag_evidence` is the second-**shortest** group (p90 298), so its samples are
  torn by block boundaries rather than by their own length; `multihop_qa`
  (p90 1,915) does fit the stated mechanism. If the overall delta is positive
  but these two axes are flat, the result is recorded as "the effect is real,
  the proposed mechanism is not confirmed".
- **R4 — abort.** For either arm: if primary-val CE at step 300 exceeds its
  step-0 value, or a non-finite loss appears, stop that arm and report a
  negative result rather than retuning mid-session.
- **R5 — re-read the ablation.** If `noise` exceeds 0.05 on `behavior_score_v0`,
  the 2026-07-27 start-point ablation's behavior ranking (0.2015 / 0.1290 /
  0.0947 / 0.0891) must be re-read with that band attached, and the "single-stage
  is best-behaved" conclusion re-stated with the appropriate confidence.

## 7. Budget and hardware (P8.2)

- **Operation:** two 2,700-step Stage 3 recovery runs with on-the-fly full-vocab
  KD from the 4B teacher, plus one reference scorecard and two post-run gate
  eval sets.
- **Why not CPU:** the CPU smoke run took ~12 s/step at `blocks_per_step` 1;
  2,700 steps × 8 blocks would be ~30 days per arm. CPU is a correctness path
  only, and it has been used as one.
- **Hardware:** 1× **L40S** (46 GB), pod-local disk 150 GB, `--volume-in-gb 0`.
  L40S keeps throughput/memory comparable to s1, the A/B, `s2_blocks_v1` and the
  ablation. Measured headroom at 4×1024 microbatches was ~37 GB of 46 GB; the
  control uses 2×2048 — the same tokens per forward, with SDPA (memory-efficient,
  not quadratic in memory). **OOM fallback:** `micro_blocks` 1, which changes
  only grad-accumulation granularity, not the step budget.
- **Estimated runtime:** setup ~0.7 h (paid once, two arms) · reference scorecard
  ~0.25 h · training ~2.7 h × 2 · post-run gate evals ~0.4 h × 2 ≈ **7.1 h**.
- **Estimated cost: ~$7.0** at $0.99/h. **Hard cap:** `--terminate-after` set to
  +10 h ≈ $9.9 worst case. Approved band was $5–7 (control) + $1–2 (variance).
- **Prior spend:** ~$17.5 across the project; balance $233.17.

## 8. Validation gate

- Both arms reproducible from logged command + config sha256 + git commit.
- Checkpoints resume (verified on CPU for the new data path).
- `packing`, per-group packing efficiency and truncation counts recorded in
  `dataset_loaded` and the run manifest.
- Gate evals per arm: bf16 `holdout_v1`, INT8 both scopes, `eval_behavior_v0`
  (cap 512), generation smoke.
- sha256 verified at every transfer hop; pod deleted only after upload
  verification passes.
- Result written to `logs/experiments/2026-07-28_stage3_packing_control.md`
  before any STATE.md conclusion changes.

## 9. Links

- `logs/decisions.md` 2026-07-28 (the queue this run came from)
- `logs/experiments/2026-07-27_stage3_start_point_ablation.md` (the baseline)
- `logs/experiments/2026-07-28_teacher_behavior_v0.md` (the ceiling, and the
  `block_len` evidence in its finding 4)
- `scripts/pod/AGENTS.md` (session playbook)
