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
| holdout NLL (bf16) | 4.0622 ±0.0026 (n=2) | 3.9653 ±0.0019 (n=2) |
| holdout NLL (INT8) | 4.0776 ±0.0016 (n=2) | 3.9788 ±0.0020 (n=2) |
| p(`</think>`) | 0.7723 ±0.0193 (n=2) | 0.0648 ±0.0059 (n=2) |
| p(`<|im_end|>`) | 0.6391 ±0.0111 (n=2) | 0.5016 ±0.0120 (n=2) |
| `behavior_score_v0` | 0.3760 ±0.0458 (n=2) | 0.0576 ±0.0041 (n=2) |
| `format_ok` | 0.6250 ±0.0461 (n=2) | 0.1250 ±0.0197 (n=2) |
| `think_closed` | 0.9013 ±0.0329 (n=2) | 0.2697 ±0.0197 (n=2) |
| `terminated` | 0.6711 ±0.0921 (n=2) | 0.1579 ±0.0000 (n=2) |
| `empty_answer` | 0.0197 ±0.0197 (n=2) | 0.7039 ±0.0066 (n=2) |

## 3. Per run

| run | last step | holdout NLL | p(</think>) | p(<\|im_end\|>) | behavior | format_ok | terminated |
|---|---|---|---|---|---|---|---|
| `tt2x2_ctrl_a` | 135 | 4.0648 | 0.7916 | 0.6280 | 0.3302 | 0.5789 | 0.5789 |
| `tt2x2_ctrl_b` | 135 | 4.0596 | 0.7530 | 0.6502 | 0.4217 | 0.6711 | 0.7632 |
| `tt2x2_treat_a` | 135 | 3.9673 | 0.0707 | 0.5136 | 0.0535 | 0.1053 | 0.1579 |
| `tt2x2_treat_b` | 135 | 3.9634 | 0.0590 | 0.4896 | 0.0617 | 0.1447 | 0.1579 |

## 4. Pre-registered decision rules

Rules registered in `logs/proposals/stage3/2026-07-30_stage3_teacher_target_2x2.md` §7, applied mechanically here. Effect sizes are read against the **seed spread**, not a p-value.

- **R1 — the treatment wins**: p(`</think>`) 0.0648 vs 0.7723 (control spread 0.0386) → inside spread; p(`<|im_end|>`) 0.5016 vs 0.6391 (control spread 0.0222) → inside spread; holdout NLL 3.9653 vs 4.0622 = -2.38% (band ±1%) → OUTSIDE.
  → **R1 does not fire.**
- **R2 — rejection**: holdout NLL -2.38% (reject if > +1%); `terminated` 0.1579 vs 0.6711 (control spread 0.1842) → REGRESSED.
  → **R2 FIRES: reject the treatment.** `terminated` regressed beyond the control's seed spread, which is the metric the exit gate is blocked on.
- R3 does not fire: the arms are separated by more than the seed spread on at least one probe, so the comparison is conclusive — in whichever direction R1/R2 report.
- **R4 — abort** is applied per-arm during the run by the orchestrator (val CE at the first eval above step 0, or non-finite loss). Arms with training logs: tt2x2_ctrl_a, tt2x2_ctrl_b, tt2x2_treat_a, tt2x2_treat_b.

**Context, not a decision input:** `behavior_score_v0` 0.0576 ±0.0041 vs 0.3760 ±0.0458, Δ=-0.3184 against the measured seed-only noise floor of 0.1290. This delta exceeds the noise floor, but the composite is still not a pre-registered readout for this experiment.

## 5. Declared asymmetry (P6)

Total training tokens are **identical** across arms (steps x blocks_per_step x block_len). Passes over the prompt set and supervised token counts are **not** equal, because teacher targets are several times longer than public ones on the same prompts. That is inherent to the comparison and is reported rather than engineered away (maintainer decision 2026-07-30). See the corpus section for the measured supervised-token counts per arm.


---

# 6. Post-hoc: both pre-registered probes are invalid instruments here

§4 applies the rules as written, and its verdict — **R2 fires, reject** — is the
pre-registered outcome. It should not be read as "teacher-native targets are
worse", because two of the three inputs to those rules cannot measure what this
experiment changed. Both problems were found after the run, from its own
artifacts, and neither was anticipated in the pre-registration.

## 6.1 `p(</think>)` is inverted by construction

It is measured at the position the **public** render demands the token — right
after the empty `<think>\n\n</think>` block the Qwen3-Thinking template injects.
It therefore scores *"would you skip reasoning entirely?"*.

That is exactly the behaviour teacher-native training is meant to remove. The
control scoring **0.772** and the treatment **0.065** is the probe working
backwards, not the treatment failing. The probe was correct for the 2026-07-28
CE/KD experiment, where skipping reasoning *was* the question; it was carried
into this pre-registration without rechecking what it measures.

## 6.2 The scorecard's 512-token cap was saturated for one arm only

Teacher targets have a rendered p50 of 1,149 tokens. At `max_new_tokens` 512 the
treatment arm hit the cap on **84.2%** of prompts against 42%/24% for the
controls, so `terminated`, `format_ok` and `empty_answer` were largely measuring
truncation. Re-scored at **2048** on the same 24 prompts, 2 seeds per arm:

| metric | start ckpt | control | treatment |
|---|---:|---:|---:|
| `truncated_at_cap` | 0.708 | **0.375** ±0.042 | 0.604 ±0.062 |
| `format_ok` | 0.250 | **0.625** ±0.042 | 0.354 ±0.021 |
| `terminated` | 0.292 | **0.625** ±0.042 | 0.396 ±0.062 |
| `think_closed` | 0.583 | **0.979** ±0.021 | 0.375 ±0.042 |
| `empty_answer` | 0.125 | **0.021** ±0.021 | 0.542 ±0.083 |

Raising the cap recovered much of the treatment arm (`format_ok` 0.105 → 0.354,
`terminated` 0.158 → 0.396) but **did not close the gap**. Seed spreads are small
(≤0.083) against inter-arm differences of 0.23–0.60, so the ordering is real.

## 6.3 What the gap actually is — and what the control is buying

Conditioned on generations that did **not** hit the 2048 cap:

| | n finished | `format_ok` | `terminated` | `think_closed` | `empty_answer` |
|---|---:|---:|---:|---:|---:|
| control | 30 | 1.000 | 1.000 | 1.000 | 0.000 |
| treatment | 19 | 0.909 | **1.000** | 0.909 | 0.045 |

**When the treatment arm finishes, it terminates every time and is format-valid
91% of the time.** The entire headline gap is that it finishes *less often*. It
is not protocol-incompetent; it has not learned to *bound* its reasoning within
137 steps.

And the control's advantage is partly degenerate. Median answer length among
finished generations: **control 2 words, treatment 34**. The control was trained
on public rag/multihop targets whose answers are a few tokens after an empty
think block, so it maximises `format_ok` and `terminated` by answering
trivially. AGENTS.md P10 forbids exactly this reading — protocol metrics that a
maximally terse policy wins are not a measure of realtime usefulness, and a
latency or format win obtained by degrading the answer does not count.

## 6.4 What both arms cost on the guard rail

Holdout NLL, bf16: start checkpoint **3.8285** → control **4.0622** ±0.0026,
treatment **3.9653** ±0.0019. **Both arms regressed** against the start
checkpoint (+6.1% and +3.6%); the treatment regressed *less*, and is 2.38%
better than the control. INT8 fake-quant tracks bf16 within +0.35% on every arm,
so nothing here is a quantization artifact.

A 137-step warm-up at lr 5e-5 on 487 prompts damages the language model in both
arms. That is a cost neither arm's protocol gain has been shown to be worth.

## 7. Verdict

* **Pre-registered verdict: R2 fires — the treatment is rejected.** Recorded as
  the outcome, per the rules agreed before the run.
* **That verdict is not trustworthy evidence about teacher-native targets**,
  because `p(</think>)` is inverted here (§6.1) and the scorecard was saturated
  (§6.2), and because the surviving gap is dominated by non-termination rather
  than malformed output (§6.3), measured against a control whose median answer
  is 2 words.
* **What is solidly established:** a short SFT warm-up at `block_len` 8192
  substantially improves protocol competence over the Stage 1→3 checkpoint on
  *either* target set (`format_ok` 0.250 → 0.625 control / 0.354 treatment;
  `truncated_at_cap` 0.708 → 0.375 / 0.604), and costs holdout NLL in both.
* **Not established:** whether teacher-native targets beat public ones. This
  design cannot separate "teacher-native" from "18.9× more supervised tokens"
  (corpus log §3), and its readouts favour terseness.

## 8. Next actions

1. **Replace the protocol readouts before re-running.** `terminated` and
   `format_ok` must be paired with an answer-quality axis, or a 2-word answer
   wins. `p(</think>)` must be measured on teacher-native renders, or dropped.
2. **Give the treatment arm a budget it can finish in** — either more steps, or
   a length-aware curriculum. It terminates reliably when it finishes.
3. **Re-examine the lr/step budget**: both arms lost 3.6–6.1% holdout NLL. A
   warm-up that damages the LM to buy format is a bad trade at this size.
4. **Do not carry this into Stage 4/5 as a settled result.** The rejection is
   pre-registered and honest, but §6 shows the instrument, not the intervention,
   is what was mainly measured.
