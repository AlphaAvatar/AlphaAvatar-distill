# Reasoning-inheritance research (Stage 1): index

How much of the teacher's fluid reasoning (AGENTS.md P10.2) a compressed student inherits from its initialization,
and how much it keeps after the same Stage-3 recovery. The teacher throughout is the hybrid `Qwen/Qwen3-4B@1cfa9a72`
(one lineage, P17.1). Measurement is eval-set-1 in both modes; it is development-exposed, so no result here is a
never-seen blind one.

**Working rule.** The research stays on `research/reasoning-inheritance`. Each round registers its own study,
configs, logs, tests and report. Merging to `main` is deferred, and it is the maintainer's decision.

## Default baseline (maintainer decision, 2026-11-18 UTC)

**TPI1's A (`A_teacher_parent`, teacher-parent initialization) is the research's default baseline. Research A's B
(`B_packed_top1`, the packed search's Top-1) is kept as a historical reference.**

What "baseline" means for a new round:

* *The method.* Every selection is made on the original teacher, with the operator, calibration profile and
  selection rule A used. Every width transform comes from the teacher's own statistics. A selection already recorded
  on the teacher is reused when its target matches, and every rebuild must equal its recorded digest.
* *The weights.* A's 596M weights are a baseline only on their own structure (28 layers, width 1024, FFN 3072,
  16 / 8 heads). A round at another structure builds its own initialization with A's method.
* *The evidence.* At the same 596M structure and recovery recipe, A against B: thinking-mode static fluid +1.40 pp
  (95% interval [−0.53, +3.45], p 0.195), non-thinking +2.37 pp ([+0.61, +4.23]).
* *TPI1's verdict is unchanged:* COMPLETE / NOT_DISTINGUISHED. The non-thinking result stays descriptive under the
  registered fixed sequence, and the choice of baseline does not turn either difference into a finding.

## Rounds

| round | question | state | report |
| --- | --- | --- | --- |
| RI-S1 | untrained: depth pruning alone (D), all non-depth compression alone (W), both (DW) | COMPLETE, closed 2026-11-14, $19.96 | [`REPORT.md`](REPORT.md) |
| RI-LM1 | layer merging at D's structure instead of deletion | COMPLETE, closed 2026-11-15, $4.10: both merges worse than deletion | [`LAYER_MERGE_REPORT.md`](LAYER_MERGE_REPORT.md) |
| TPI1 | teacher-parent initialization (A) against the packed Top-1 (B) after the same recovery, 596M | COMPLETE, closed 2026-11-18, $23.13: NOT_DISTINGUISHED, +1.40 pp | [`../teacher_parent_initialization/REPORT.md`](../teacher_parent_initialization/REPORT.md) |
| **SOB1** (next) | at N ≈ 2 (about 2.0B parameters): depth only vs FFN + attention only vs residual width only, A's method, before and after the same recovery | **proposed, $0**: structures and budget submitted for review | [`../single_operator_budget/README.md`](../single_operator_budget/README.md) |

Literature read for these rounds: [`LITERATURE.md`](LITERATURE.md). Evidence lives under
`logs/stages/stage-1/{reasoning_inheritance,teacher_parent_initialization,single_operator_budget}/`.
