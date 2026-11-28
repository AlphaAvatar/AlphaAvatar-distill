# depth_recurrence (Stage 1 research, Stage 3 recovery): DMR1 variant 2 (revised 2026-11-28) and variant 3 (not started)

Variant 2 runs SOB1's 16 kept blocks on the 36-logical-layer schedule from TWO training-pre initializations, one
training seed (20260726) each: `depth_loop_kept` (V2-D: the kept teacher blocks, SOB1's depth init) and
`depth_loop_merged` (V2-M: variant 1's verified alpha-merged init). Only the initialization differs. Variant 3
(`depth_loop_hidden`) is not started: it has no base, plan or kind. Design: `configs/stages/stage-1/depth_recurrence/
design.json` (the 2026-11-26 selection rule is kept there under `superseded_2026_11_28`); evidence:
`logs/stages/stage-1/depth_recurrence/`. Status: REVISED at $0, READY_FOR_REVIEW (2026-11-28). No paid run is
authorized. The training, takeover, completion and analysis code is the round's shared chain in
`scripts/stages/stage-1/depth_merge/`.

| file | role |
| --- | --- |
| `schedule.py` | the schedule from the mapping, the recurrent config, each loop arm's recurrent init record over the base the design fixes for it (`init-record --arm`), `init_record_problems` |
| `engine_check.py` | the GPU engine validation the first session runs before training: HF cached vs no-cache, vLLM's greedy generation re-scored by HF, a shared-cache fault that must fail; each against the same weights as a stock 16-layer Qwen3 |
| `launch.py` | the launcher: kind `dmr1_loop_train`, one session per arm (`--arm`); refuses until a plan, a proposal and a grant exist |
| `tests/` | `pytest scripts/stages/stage-1/depth_recurrence/tests`, including the real training driver end to end for depth_merge and both loop arms |

The model itself is core: `aadistill.initialization.adapters.recurrent_depth` (HF) and `recurrent_depth_vllm` (vLLM),
the trainer's `loss.hidden_state` and its eval-only `state_probe` (`aadistill.training.hidden_state`).
