
---

# Interpretation (hand-written after the mechanical section above)

## Verdict: the mechanism is confirmed, and by a wide margin

| | control (2 seeds) | treatment (2 seeds) | Δ |
|---|---:|---:|---:|
| **p(`</think>`)** | 0.2995 ±0.0176 | **0.9989 ±0.0004** | **+0.6994** |
| `think_closed` | 0.2763 ±0.0790 | 0.6776 ±0.1447 | +0.4013 |
| `format_ok` | 0.0066 ±0.0132 | 0.2894 ±0.0789 | +0.2828 |
| `empty_answer` | 0.4342 | 0.0263 | −0.4079 |
| `terminated` | 0.3158 ±0.0000 | 0.3750 ±0.0658 | +0.0592 |
| `holdout_v1` NLL | 4.0136 ±0.0022 | 4.0261 ±0.0018 | +0.31% |

- **R1 fires.** Δp = +0.6994 against a max within-condition spread of 0.0176 —
  a factor of **40**. The CE/KD conflict is *causal* for the contested token,
  not merely present.
- **R2 fires.** `think_closed` +0.4013 against its own spread of 0.1447 (2.8×).
- **R3 clear.** +0.31% holdout, inside the ±1% guard rail. Removing ~1% of KD
  positions costs essentially nothing in language modeling.
- **R5 not triggered.** The pre-committed refutation clause did not apply.

## R4 did **not** fire, reported as written

R4 was registered on *absolute* seed spread, and the treatment's spread is
**larger** on both behavior metrics (`think_closed` 0.1447 vs 0.0790;
`format_ok` 0.0789 vs 0.0132). On a *relative* basis the reverse holds —
`format_ok`'s spread falls from 200% of its mean to 27% — and the control's tiny
absolute spread is largely a **floor artifact**: at `format_ok` 0.0066 there is
nothing to vary. Switching to the relative reading now, after seeing the
numbers, is exactly what pre-registration exists to prevent. **R4 did not fire.**

What *did* collapse is the mechanism variable: **p(`</think>`) spread fell
0.0176 → 0.0004, a 44× reduction.** That is the knife-edge story confirmed —
remove the opposing force and the equilibrium stops being seed-sensitive. The
behavior metrics did not stabilise because they moved *off the floor* into a
range where they can vary at all.

**So the maintainer's variance hypothesis is confirmed at the mechanism and
unproven at the behavior level.** The instability at the contested token was a
force balance and is gone; whether behavioral variance falls once every such
conflict is removed is a separate question this run cannot answer.

## The treatment at 1000 steps beats the baseline at 2700

| | baseline `s2v1_from_init@2700` | treatment @1000 |
|---|---:|---:|
| `format_ok` | 0.2237 | **0.2894** |
| `think_closed` | 0.6053 | **0.6776** |
| `empty_answer` | 0.1711 | **0.0263** |

At **37% of the step budget**. This is a mechanism test, not a recipe proposal
(the proposal said so in advance), but the size of the effect makes a
full-length run worth costing.

## `terminated` barely moved — as predicted

0.3158 → 0.3750, against `think_closed`'s +0.40. The addendum to
`2026-07-28_kd_ce_protocol_conflict.md` predicted this before the arms finished:
`all_no_think` masks the think span only, and the *second* conflict — at
`<|im_end|>`, where the teacher puts p≈0.00003–0.004 on stopping for the terse
slices — is untouched. `format_ok` is therefore still ceilinged near
`terminated` ≈ 0.375, and the treatment's 0.2894 is 77% of that ceiling.

**Stage 3's exit gate remains blocked on termination, not on think-closing.**

## What to do with this

1. **Do not adopt `all_no_think` as the recipe yet.** It is a confirmed
   diagnostic. Adopting it permanently treats one disagreement while the corpus
   contains an unknown number — the terminator is the second already found.
2. **The general statement now has two confirmed instances:** the teacher
   disagrees with any target it did not write, and where it disagrees KD at 2×
   per-position weight wins. That is an argument for **teacher-generated
   targets**, which remove every such disagreement at once rather than one mask
   at a time.
3. **The cheap next diagnostic** is a terminator-span mask, testing whether the
   same fix transfers to `<|im_end|>`. If it does, it confirms the general
   statement and bounds how much of the exit gate is reachable by masking alone.
4. **A full-length (2700-step) treatment arm** would say where `format_ok`
   actually lands, and whether the 1000-step advantage persists or washes out.

## Cost

**$4.48** ($226.00 → $221.52), under the $5.0 estimate. 1× L40S, ~4.4 h,
4 arms verified 17/17 files each, pod deleted after verification, 0 pods
remaining.
