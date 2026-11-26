# depth_recurrence (Stage 6): DMR1 variants 2 and 3

`depth_loop` (SOB1's 16 kept blocks on the 36-logical-layer schedule) and `depth_loop_hidden` (the same with the
per-iteration hidden-state term). Design: `configs/stages/stage-6/depth_recurrence/design.json`; evidence:
`logs/stages/stage-6/depth_recurrence/`. Status: PREPARED at $0, READY_FOR_REVIEW (2026-11-26); no paid run is
authorized. The training, takeover, completion and analysis code is the round's shared chain in
`scripts/stages/stage-1/depth_merge/`.

| file | role |
| --- | --- |
| `schedule.py` | the schedule from the mapping, the recurrent config, the initialization selection (`select`), the recurrent init record over the selected base (`init-record`), `init_record_problems` (the two legal paths) |
| `launch.py` | the variants' launcher (kinds `dmr1_loop_train`, `dmr1_loop_hidden_train`); refuses until a selection, a plan, a proposal and a grant exist |
| `tests/` | `pytest scripts/stages/stage-6/depth_recurrence/tests`, including the real training driver end to end for all three DMR1 arms |

The model itself is core: `aadistill.initialization.adapters.recurrent_depth` (HF) and `recurrent_depth_vllm` (vLLM),
and the trainer's `loss.hidden_state` (`aadistill.training.hidden_state`).
