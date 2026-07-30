# 2026-07-30 — Stage 3 teacher-target SFT warm-up: 2x2 result

- **Pre-registration:** [`proposals/stage3/2026-07-30_stage3_teacher_target_2x2.md`](../../proposals/stage3/2026-07-30_stage3_teacher_target_2x2.md)
- **Generated mechanically** by `scripts/pod/report_tt2x2.py` from the runs' own artifacts. Numbers are measured; the rules below are applied mechanically; the stage verdict is left to review.

## 1. Corpus

- accepted prompts: **540** (dropped 212 public fallbacks)
- corpus sha256: `18028f0cb8009132…`
- capability scope: {"allowed_groups": ["rag_evidence", "multihop_qa", "code_math"], "dropped_out_of_scope": {}}
- packing: {"mode": "best_fit", "block_len": 8192, "seed": 20260726}

- **control**: supervised 27526 tokens (fraction 0.0929), lossless=True
- **treatment**: supervised 519478 tokens (fraction 0.6591), lossless=True

## 2. Arm means (two seeds, mean ±half-spread)

| readout | control (2 seeds) | treatment (2 seeds) |
|---|---|---|
| holdout NLL (bf16) | 7.7260 ±1.1033 (n=2) | 6.2255 ±0.4844 (n=2) |
| holdout NLL (INT8) | 7.6878 ±0.9717 (n=2) | 6.2447 ±0.4824 (n=2) |
| p(`</think>`) | 0.7925 ±0.0515 (n=2) | 0.0000 ±0.0000 (n=2) |
| p(`<|im_end|>`) | 0.3678 ±0.0379 (n=2) | 0.0052 ±0.0011 (n=2) |
| `behavior_score_v0` | 0.2175 ±0.0025 (n=2) | 0.0008 ±0.0004 (n=2) |
| `format_ok` | 0.6250 ±0.0461 (n=2) | 0.0000 ±0.0000 (n=2) |
| `think_closed` | 0.9276 ±0.0461 (n=2) | 0.0066 ±0.0066 (n=2) |
| `terminated` | 0.6579 ±0.0395 (n=2) | 0.0066 ±0.0066 (n=2) |
| `empty_answer` | 0.0724 ±0.0461 (n=2) | 0.9803 ±0.0066 (n=2) |

## 3. Per run

| run | last step | holdout NLL | p(</think>) | p(<\|im_end\|>) | behavior | format_ok | terminated |
|---|---|---|---|---|---|---|---|
| `ttb_ctrl_a` | 135 | 6.6227 | 0.8440 | 0.4057 | 0.2200 | 0.6711 | 0.6974 |
| `ttb_ctrl_b` | 135 | 8.8293 | 0.7410 | 0.3299 | 0.2150 | 0.5789 | 0.6184 |
| `ttb_treat_a` | 135 | 5.7411 | 0.0000 | 0.0062 | 0.0004 | 0.0000 | 0.0000 |
| `ttb_treat_b` | 135 | 6.7098 | 0.0000 | 0.0041 | 0.0011 | 0.0000 | 0.0132 |

## 4. Pre-registered decision rules

Rules registered in `logs/proposals/stage3/2026-07-30_stage3_teacher_target_2x2.md` §7, applied mechanically here. Effect sizes are read against the **seed spread**, not a p-value.

- **R1 — the treatment wins**: p(`</think>`) 0.0000 vs 0.7925 (control spread 0.1030) → inside spread; p(`<|im_end|>`) 0.0052 vs 0.3678 (control spread 0.0758) → inside spread; holdout NLL 6.2255 vs 7.7260 = -19.42% (band ±1%) → OUTSIDE.
  → **R1 does not fire.**
- **R2 — rejection**: holdout NLL -19.42% (reject if > +1%); `terminated` 0.0066 vs 0.6579 (control spread 0.0789) → REGRESSED.
  → **R2 FIRES: reject the treatment.** `terminated` regressed beyond the control's seed spread, which is the metric the exit gate is blocked on.
- R3 does not fire: the arms are separated by more than the seed spread on at least one probe, so the comparison is conclusive — in whichever direction R1/R2 report.
- **R4 — abort** is applied per-arm during the run by the orchestrator (val CE at the first eval above step 0, or non-finite loss). Arms with training logs: ttb_ctrl_a, ttb_ctrl_b, ttb_treat_a, ttb_treat_b.

**Context, not a decision input:** `behavior_score_v0` 0.0008 ±0.0004 vs 0.2175 ±0.0025, Δ=-0.2167 against the measured seed-only noise floor of 0.1290. This delta exceeds the noise floor, but the composite is still not a pre-registered readout for this experiment.

## 5. Declared asymmetry (P6)

Total training tokens are **identical** across arms (steps x blocks_per_step x block_len). Passes over the prompt set and supervised token counts are **not** equal, because teacher targets are several times longer than public ones on the same prompts. That is inherent to the comparison and is reported rather than engineered away (maintainer decision 2026-07-30). See the corpus section for the measured supervised-token counts per arm.

