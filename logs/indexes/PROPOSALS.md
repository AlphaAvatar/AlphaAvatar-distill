# Proposal index

Pre-registrations, by area and date. Written **before** spend so decision rules
cannot be chosen after seeing the numbers (P6). **Status** distinguishes what was
run, what was retired unrun, and what is awaiting approval.

## rollout — engine selection

| date | proposal | status | outcome |
|---|---|---|---|
| 07-30 | [current_engine_benchmark](../proposals/rollout/2026-07-30_current_engine_benchmark.md) | **run** | vLLM 0.26.0 vs SGLang 0.5.12; $0.93 of a $3.00 ceiling. Neither adopted — criterion 6 outstanding |
| 07-30 | [rollout_engine_comparison](../proposals/rollout/2026-07-30_rollout_engine_comparison.md) | **retired unrun** | Built around vLLM 0.11.0, an obsolete compatibility build. Its correction experiment was carried forward |
| 07-30 | [isolated_engine_and_cap](../proposals/rollout/2026-07-30_isolated_engine_and_cap.md) | **run** | Isolated-venv vLLM + openmath cap arm; $1.85 of $2.50 |
| 07-29 | [engine_benchmark](../proposals/rollout/2026-07-29_engine_benchmark.md) | **run** | First engine benchmark; its rule R1 (exact token agreement) is now retired |
| 07-29 | [inference_engine_survey](../proposals/rollout/2026-07-29_inference_engine_survey.md) | reference | Landscape survey; selected no engine and measured nothing by design |

## stage3 — recovery and targets

| date | proposal | status | outcome |
|---|---|---|---|
| 07-30 | [stage3_teacher_target_2x2](../proposals/stage3/2026-07-30_stage3_teacher_target_2x2.md) | **awaiting approval — tomorrow's Gate 1** | 750-prompt corpus + 2×2 public vs teacher targets, 2 seeds/arm, `best_fit`@8192. $3–6, ceiling $7 |
| 07-28 | [kd_ce_conflict_intervention](../proposals/stage3/2026-07-28_kd_ce_conflict_intervention.md) | **run** | Confirmed the CE/KD conflict causal |
| 07-28 | [stage3_packing_blocklen_control](../proposals/stage3/2026-07-28_stage3_packing_blocklen_control.md) | **run** | Packing change rejected; noise floor found |
| 07-27 | [stage3_start_point_ablation](../proposals/stage3/2026-07-27_stage3_start_point_ablation.md) | **run** | Warm-up ladder retired |

## stage2 — data

| date | proposal | status | outcome |
|---|---|---|---|
| 07-27 | [stage2_teacher_generated_answers](../proposals/stage2/2026-07-27_stage2_teacher_generated_answers.md) | superseded | Teacher-target direction; superseded by the 2026-07-30 2×2 and the scope decisions |
| 07-26 | [stage2_mixture_v1_scaleup](../proposals/stage2/2026-07-26_stage2_mixture_v1_scaleup.md) | **run** | Mixture v1 |
