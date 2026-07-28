# 2026-07-28 — CE/KD protocol conflict: the intervention

**Status:** pre-registered. Written before the run; §4's rules are binding.
**Approved by the maintainer** ("please do the experiment", 2026-07-28).

## 1. What is being tested

`logs/experiments/2026-07-28_kd_ce_protocol_conflict.md` measured, on CPU for
$0, that at the token immediately after `<think>\n\n`:

- **CE** (loss mask confirmed `True`) demands `</think>`, one-hot, at weight 0.25;
- **KD** (`scope: all`, τ=1) puts **p(`</think>`) = 0.000000** and ~1.0 on
  `Okay`/`Hmm`, at weight 1.0 — **2.02× CE's per-position pull**;
- the student equilibrates at **p(`</think>`) = 0.334**, uniform to ±0.015
  across seven groups.

The *conflict* is measured. That it **causes** the protocol failure and the
0.1290 behavior noise floor is inference. This run is the intervention that
decides it.

**Intervention:** `kd_scope: "all_no_think"` — KD keeps every position except
the template-inserted span `<think> … </think>` inclusive. CE is untouched, the
data is untouched, and ~1% of KD positions are removed (measured on the CPU
smoke: 972–1017 KD positions per 1024-token block, against 1023 for `all`).

## 2. Design

A 2×2, all four arms on one pod, everything identical but the two factors:

| | seed 20260726 | seed 20260728 |
|---|---|---|
| `kd_scope: all` (control) | `kdconf_ctrl_a` | `kdconf_ctrl_b` |
| `kd_scope: all_no_think` (treatment) | `kdconf_nothink_a` | `kdconf_nothink_b` |

Config diff against the current Stage 3 baseline (`stage3_s2v1_from_init.json`)
is `kd_scope`, `seed`, `total_steps` 2700→**1000**, `save_every` 300→250, plus
`run_name`/`out_dir`. Verified mechanically: `ctrl_a` vs `nothink_a` differ in
`loss`, `run_name`, `out_dir` and nothing else.

**Two seeds per condition is not optional.** The measured seed-only noise floor
on `behavior_score_v0` is **0.1290**, wider than any inter-arm difference this
project has ever reported. A one-run-per-condition version of this experiment
would be unreadable, and the standing rule from the same day requires ≥2.

**Why 1000 steps and not 2700.** The readout is a *format* property, and it
develops early: the Stage 1 init sits at p(`</think>`) = 0.0000 and 660 steps of
recovery already move it to 0.334. Removing the opposing force should show up
fast, and matched-budget arms are what the comparison needs — not a full-length
run. This is explicitly a mechanism test, **not** a recipe candidate: no
checkpoint from it is proposed as a branch point.

## 3. Readouts

**Primary — `p(</think>)` at the contested position** (`scripts/probe_think_close.py`,
28 samples over 7 groups, run per arm in `post_run.sh`). Chosen over
`think_closed` because it is **continuous**, so it tracks the force balance
smoothly instead of flipping at an argmax boundary, and is therefore far less
seed-noisy — the per-group spread on the s1 checkpoint was 0.309–0.353 against
a 0.1290 floor on the generation composite. It is also the number the mechanism
claim is literally about.

**Secondary — behavior:** `think_closed`, `format_ok`, `empty_answer` from
`eval_behavior_v0`. Reported individually, **not** as the composite: the
composite's noise floor is the reason this project cannot read it.

**Guard rail — `holdout_v1` NLL.** Removing ~1% of KD positions must not cost
language modeling.

**Variance — seed spread within condition**, on both the primary and the
behavior metrics. This is the readout that speaks to the maintainer's
hypothesis that the instability is a training-variance problem.

## 4. Pre-registered rules

Let `P_ctrl`, `P_treat` be `p_close_mean` averaged over each condition's two
seeds, and `S_ctrl`, `S_treat` the within-condition seed spreads.

- **R1 — the mechanism.** Fires if `P_treat − P_ctrl > max(S_ctrl, S_treat)`
  **and** the sign is positive. Then the conflict is confirmed as *causal* for
  the contested token, not merely present.
- **R2 — the behavior consequence.** Fires if `think_closed` rises in the
  treatment by more than its own seed spread. **R1 without R2 is a real and
  reportable outcome**: it would mean the conflict is genuine but that closing
  the think block is not what gates the behavior score, and the search moves on.
- **R3 — guard rail.** If `holdout_v1` regresses more than **1%** against the
  control's mean, the treatment is not adopted regardless of R1/R2; report the
  tradeoff and escalate.
- **R4 — variance.** If `S_treat < S_ctrl` on the behavior metrics, that is
  direct support for the maintainer's hypothesis that the instability is a
  training-variance problem with a data/objective fix. If the spreads are
  comparable, the noise has another source and the ≥2-seed rule stands
  independently of this experiment's outcome.
- **R5 — falsification.** If `P_treat ≈ P_ctrl`, the mechanism claim is
  **dead**: removing the contested positions from KD did not move the token
  they are contested over. Record it as a refutation, not as an inconclusive
  result, and stop pursuing this line. Two mechanism claims were already
  retracted on 2026-07-28; this one is pre-committed to the same standard.
- **A — abort.** Per arm: non-finite loss, or primary-val CE at step 300 above
  its step-0 value → stop that arm and report it.

## 5. What this cannot decide

- It does not test the maintainer's **teacher-generated warm-up**, which
  removes the *cause* (teacher forced off its own manifold) rather than the
  contested positions. If R1 and R2 fire, that is indirect support for it; if
  R5 fires, the warm-up's rationale is unaffected, because its case never
  rested on this mechanism.
- 1000 steps is not 2700. A treatment that helps at 1000 steps might wash out,
  and the run cannot say otherwise.
- `all_no_think` is a *diagnostic*, not automatically the recipe. Masking a
  span the teacher disagrees with treats a symptom of teacher-forcing the
  teacher off-manifold; adopting it permanently is a separate decision that
  needs a full-length run.

## 6. Budget and hardware (P8.2)

- **Operation:** 4 × 1000-step Stage 3 recovery runs with on-the-fly full-vocab
  KD from the 4B teacher, plus per-arm gate evals.
- **Why not CPU:** the CPU smoke ran ~6 s/step at `blocks_per_step` 1; a single
  arm would take ~11 days. CPU is the correctness path and has been used as one
  (smoke + resume + 158 tests).
- **Hardware:** 1× **L40S** (46 GB), container disk 150 GB, `--volume-in-gb 0`.
  Same class as every prior Stage 3 run. At 16×1024 blocks the measured peak was
  ~37 GB, comfortably inside.
- **Estimated runtime:** setup ~0.7 h (paid once for four arms) · training
  4 × ~0.8 h · gate evals 4 × ~0.3 h ≈ **5.0 h**.
- **Estimated cost ≈ $5.0** at $0.99/h. **Hard cap:** `--terminate-after` at
  +8 h ≈ $7.9. Balance $226.00; project GPU spend to date ~$24.7.
- No reference scoring this session — the comparison is entirely within the
  2×2, so there is no cross-device gap to close.

## 7. Validation gate

- All four arms reproducible from logged command + config sha256 + git commit.
- `kd_scope` and the resolved `think_ids` recorded in each run's
  `dataset_loaded` event, so a run's own log proves which loss it used.
- Trainer **fails loudly** if `all_no_think` is requested without `think_ids`:
  a silent fallback to `all` would make a control arm masquerade as treatment.
- Gate evals per arm: bf16 holdout, INT8 both scopes, `eval_behavior_v0`,
  **`probe_think_close.json`**, generation smoke.
- sha256 verified at every transfer hop; pod deleted only after upload
  verification.
- Result written to `logs/experiments/2026-07-28_kd_conflict_intervention.md`
  before any STATE conclusion changes.

## 8. Links

- `logs/experiments/2026-07-28_kd_ce_protocol_conflict.md` (the measurement)
- `logs/experiments/2026-07-28_stage3_packing_control.md` (the noise floor, §2–3)
- `logs/decisions.md` 2026-07-21 (empty-think targets), 2026-07-22 (KD design)
