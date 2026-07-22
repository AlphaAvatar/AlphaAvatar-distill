# Supported / in-progress models

Status values follow AGENTS.md 3.4. A model is listed here only once real work exists for it; nothing below is a released or validated result.

| Model | Teacher | Student target | Status | Stages passed | Best checkpoint | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| qwen3-4b-thinking-distill (working name) | Qwen/Qwen3-4B-Thinking-2507 @ `768f209d` | 0.6B-class (hidden 1024, 28L, FFN 3072, 16Q/8KV, tied emb; user-chosen 2026-07-13) | stage3-running | Stage 0, Stage 1, Stage 2, Stage 3 s1 | `artifacts/stage3/s1_ffn_norm_v0/checkpoints/step_000660/model` (local, gitignored; fp32) | Stage 0 v1: 949,859 tokens ([log](experiments/2026-07-13_stage0_qwen3_4b_thinking_v1.md)). Stage 1 init gate passed; holdout NLL 11.75 vs random 12.13 vs teacher 2.63 ([log](experiments/2026-07-14_stage1_qwen3_0p6b_init_v0.md)). Stage 2 offline mixture v0: 8 groups, 5.39M train tokens ([log](experiments/2026-07-21_stage2_offline_v0.md)). Stage 3 recovery sub-stage 1 (FFN+norm, 660 steps, L40S) gate passed 2026-07-22: holdout NLL **4.21** (init 11.75, teacher 2.63), stage2-val ce 12.01→2.18, generation smoke OK ([log](experiments/2026-07-22_stage3_s1_gpu_run.md)). Sub-stages 2+ pending sizing decision. Deployment target: INT8. |
