# Supported / in-progress models

Status values follow AGENTS.md 3.4. A model is listed here only once real work exists for it; nothing below is a released or validated result.

| Model | Teacher | Student target | Status | Stages passed | Best checkpoint | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| qwen3-4b-thinking-distill (working name) | Qwen/Qwen3-4B-Thinking-2507 @ `768f209d` | TBD (not chosen) | stage0-passed | Stage 0 | none | Stage 0 activation-stats collected on CPU, 47-sample warm-up v0, 4068 tokens, gate passed. See [experiment log](experiments/2026-07-12_stage0_qwen3_4b_thinking_v0.md). Student architecture undecided (blocks Stage 1). |
