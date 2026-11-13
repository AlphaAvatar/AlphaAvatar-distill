# reasoning_inheritance (Stage 1) -- structural ablation and reasoning-damage analysis

Four untrained models from one frozen mapping of the packed full search's Top-1 -- T (teacher), D (depth pruning
only), W (all non-depth compression only), DW (both) -- on `state_eval_hybrid@v3` and every unit of eval-set-1 in both
modes. Experiment: [`configs/stages/stage-1/reasoning_inheritance/design.json`](../../../../configs/stages/stage-1/reasoning_inheritance/design.json);
code and commands: [`scripts/stages/stage-1/reasoning_inheritance/`](../../../../scripts/stages/stage-1/reasoning_inheritance/README.md).

| area | what it holds |
| --- | --- |
| [`plans/`](plans/) | the study's proposal (the grant is `logs/budget/approvals/stage1_reasoning_inheritance_grant.json`) |
| [`analyses/`](analyses/) | teacher reuse audit, the structural-ablation results, the round's report |
| [`runs/`](runs/) | one directory per session attempt: governance, runtime, evidence, closeout |

Study RI-S1: ceiling $35 over every attempt, $24 per session, project cap $1,000 unchanged; L40S only.
Heavy products (the mapping tensors, D/W/DW checkpoints) live on the private relay under
`durable/stage1_reasoning_inheritance/<mapping digest[:16]>/`, by sha256; the dev box holds each attempt's secured
evidence under `artifacts/stages/stage-1/reasoning_inheritance/runs/<run>/scr/store/`.

Nothing here authorizes anything.
