# Supported / in-progress models

Status values follow AGENTS.md 3.4. A model is listed here only once real work exists for it; nothing below is a released or validated result.

| Model | Teacher | Student target | Status | Stages passed | Best checkpoint | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| qwen3-4b-thinking-distill (working name) | Qwen/Qwen3-4B-Thinking-2507 @ `768f209d` | 0.6B-class (hidden 1024, 28L, FFN 3072, 16Q/8KV, tied emb; user-chosen 2026-07-13) | stage2-passed | Stage 0, Stage 1, Stage 2 | `artifacts/stage1/qwen3_0p6b_init_v0/checkpoint` (local, gitignored; 596.0M params) | Stage 0 v1: 949,859 tokens ([log](experiments/2026-07-13_stage0_qwen3_4b_thinking_v1.md)). Stage 1 init gate passed; holdout NLL 11.75 vs random 12.13 vs teacher 2.63 ([log](experiments/2026-07-14_stage1_qwen3_0p6b_init_v0.md)). Stage 2 offline mixture v0: 8 groups, 5.39M train tokens, loader dry run passed ([log](experiments/2026-07-21_stage2_offline_v0.md)). Deployment target: INT8. Next: Stage 3 recovery. |
