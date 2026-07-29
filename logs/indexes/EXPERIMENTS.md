# Experiment index

Every experiment log, by area and date. **Status** says whether the log's
*conclusion* still stands — measurements always stand; conclusions sometimes do
not. Newest first within each area.

## rollout — engines, generation, correction

| date | log | status | what it established |
|---|---|---|---|
| 07-30 | [current_engine_benchmark](../experiments/rollout/2026-07-30_current_engine_benchmark.md) | **current** | vLLM 0.26.0 vs SGLang 0.5.12, both native images on CUDA-13 hosts: throughput tied (247.5 vs 241.0 tok/s; wall 0.3% apart), both do token-in/token-out with aligned log-probs, **off-policy rate 0.000** and KL ~1e-4. Deterministic mode costs 55%. Neither adopted |
| 07-30 | [isolated_engine_and_cap](../experiments/rollout/2026-07-30_isolated_engine_and_cap.md) | **conclusion retired** | Measured an isolated-venv vLLM **0.11.0** at 5.29× with 0/8 token agreement, and wrongly turned that into an adoption gate. Retains the openmath cap result (§4), which stands |
| 07-29 | [engine_benchmark_gpu](../experiments/rollout/2026-07-29_engine_benchmark_gpu.md) | **partly retired** | In-stack HF is flat in batch size (37.5/43.9/39.3 tok/s) — stands. "Both serving engines are incompatible" — retired, it was host selection |
| 07-29 | [engine_adapter_and_bf16_invariance](../experiments/rollout/2026-07-29_engine_adapter_and_bf16_invariance.md) | **current** | bf16 batched greedy decoding is **not batch-invariant** (1/6 vs fp32 6/6, padding eliminated). Built the engine adapter layer |

## stage3 — recovery, targets, protocol

| date | log | status | what it established |
|---|---|---|---|
| 07-30 | [stage3_target_preflight](../experiments/stage3/2026-07-30_stage3_target_preflight.md) | **current** | Teacher targets are 4.2× longer; `concat`@1024 splits 48.5% of them; `best_fit`@1024 loses **56%** of supervision; **`best_fit`@8192 is lossless**, bounded by construction |
| 07-29 | [pilot_slice_analysis](../experiments/stage3/2026-07-29_pilot_slice_analysis.md) | **reframed** | openmath is cap-bound; teacher answers 10/40 unanswerable questions. Its refusal argument (length-based) is withdrawn — scope is the reason |
| 07-28 | [kd_conflict_intervention](../experiments/stage3/2026-07-28_kd_conflict_intervention.md) | **current** | CE/KD conflict confirmed causal: p(`</think>`) 0.2995 → 0.9989 |
| 07-28 | [kd_ce_protocol_conflict](../experiments/stage3/2026-07-28_kd_ce_protocol_conflict.md) | **current** | Found the conflict: CE and KD want opposite things at `<think>\n\n`, KD at 2× per-position weight |
| 07-28 | [stage3_packing_control](../experiments/stage3/2026-07-28_stage3_packing_control.md) | **current** | Packing/`block_len` change rejected for public targets; **first noise-floor measurement, 0.1290** |
| 07-27 | [stage3_start_point_ablation](../experiments/stage3/2026-07-27_stage3_start_point_ablation.md) | **half retired** | NLL half stands (single-stage reaches quality at 33% fewer steps); the behavior ranking is inside the noise floor |
| 07-26 | [stage3_s2_blocks_v1_gpu_run](../experiments/stage3/2026-07-26_stage3_s2_blocks_v1_gpu_run.md) | current | Best held-out NLL, 3.8003 |
| 07-25 | [stage3_s2_ab_gpu_run](../experiments/stage3/2026-07-25_stage3_s2_ab_gpu_run.md) | current | Attention-unfrozen freeze set adopted; mixture v0 exhausted |
| 07-22 | [stage3_s1_gpu_run](../experiments/stage3/2026-07-22_stage3_s1_gpu_run.md) | current | Sub-stage 1 gate passed, holdout 4.21 |
| 07-22 | [stage3_trainer_toy](../experiments/stage3/2026-07-22_stage3_trainer_toy.md) | current | Trainer toy validation |

## stage2 — offline data

| date | log | status | what it established |
|---|---|---|---|
| 07-26 | [stage2_offline_v1](../experiments/stage2/2026-07-26_stage2_offline_v1.md) | current | Mixture v1: 64,484 samples / 22.13M train tokens |
| 07-21 | [stage2_offline_v0](../experiments/stage2/2026-07-21_stage2_offline_v0.md) | current | Mixture v0, 5.39M tokens |

## stage1 / stage0 — initialization

| date | log | status | what it established |
|---|---|---|---|
| 07-14 | [stage1_qwen3_0p6b_init_v0](../experiments/stage1/2026-07-14_stage1_qwen3_0p6b_init_v0.md) | current | Init gate passed; holdout 11.75 vs random 12.13 vs teacher 2.63 |
| 07-13 | [stage0_v1](../experiments/stage0/2026-07-13_stage0_qwen3_4b_thinking_v1.md) | current | 949,859 tokens of teacher statistics |
| 07-12 | [stage0_v0](../experiments/stage0/2026-07-12_stage0_qwen3_4b_thinking_v0.md) | current | First collection run |

## evaluation — harness and ceilings

| date | log | status | what it established |
|---|---|---|---|
| 07-28 | [teacher_behavior_v0](../experiments/evaluation/2026-07-28_teacher_behavior_v0.md) | current | Teacher ceiling **0.7443** vs student 0.2015; grounding ceiling only 0.562 |
| 07-27 | [eval_behavior_v0](../experiments/evaluation/2026-07-27_eval_behavior_v0.md) | current | Built the 76-prompt behavior eval; echo-credit and truncation findings |
| 07-26 | [int8_fakequant_eval](../experiments/evaluation/2026-07-26_int8_fakequant_eval.md) | current | INT8 fake-quant evaluation path (P9) |

## infrastructure

| date | log | status | what it established |
|---|---|---|---|
| 07-26 | [runpod_pod_readiness_misdiagnosis](../experiments/infrastructure/2026-07-26_runpod_pod_readiness_misdiagnosis.md) | current | `uptimeSeconds` is always 0 in runpodctl 2.7.1; use GraphQL. Cost ~$0.95 in healthy pods deleted as stuck |
