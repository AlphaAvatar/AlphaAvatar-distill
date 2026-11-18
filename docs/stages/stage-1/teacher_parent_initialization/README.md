# Teacher-parent initialization (TPI1): design and how to run it

Stage 1 owns this round, and the training execution stage is `stage3_recovery`. The experiment definition is
[`design.json`](../../../../configs/stages/stage-1/teacher_parent_initialization/design.json), the code is in
[`scripts/stages/stage-1/teacher_parent_initialization/`](../../../../scripts/stages/stage-1/teacher_parent_initialization/README.md),
and the evidence is in [`logs/stages/stage-1/teacher_parent_initialization/`](../../../../logs/stages/stage-1/teacher_parent_initialization/).
The final report is [`REPORT.md`](REPORT.md) (COMPLETE 2026-11-18: the primary comparison NOT_DISTINGUISHED).

The project's target is fluid reasoning and adaptive problem-solving. That is the ability, faced with a problem never
seen before, to use the information the task itself supplies to find the rules, abstract, deduce and generalize, and
so to reach a correct answer or complete the task (AGENTS.md P10.2). Low KL, fluent output or correct format is not
evidence of it.

## 1. The comparison

Both arms have the same 596M structure: 28 layers, residual width 1024, FFN 3072, 16 query / 8 KV heads, head
dimension 128, tied embedding, 596,049,920 parameters. They also share the same operators, the same per-operator
calibration profiles, the same teacher (`Qwen/Qwen3-4B@1cfa9a72`, the hybrid, not Thinking-2507) and the same
Stage-3 recipe.

| | `B_packed_top1` (control) | `A_teacher_parent` (new) |
| --- | --- | --- |
| FFN `ffn.activation_importance_v0`, base | on the teacher | the same state (B's first step) |
| DEPTH `depth.causal_kl_greedy_v1`, base | on the FFN-compressed parent | on the teacher: greedy 36 → 28, the reference is the unbypassed teacher (`parent_state`) |
| RESIDUAL_WIDTH `width.global_pca_v0`, rh | statistics of the FFN → DEPTH parent, 29 points | the teacher's statistics at `sorted({0, *K}) + [36]` (29 points), weights 9 / 1 / 8 |
| ATTENTION `attention.activation_importance_v1`, rh | on the width child (1024-dim) | on the teacher (2560-dim head outputs) |
| build order | FFN → DEPTH → WIDTH → ATTENTION | the same, from one frozen mapping |
| training | A-MIX-6cat-sr, two seeds (done, $23.41) | the same recipe, the same two seeds |

A difference between the arms belongs to the whole A recipe, never to one change.

## 2. What is reused and what is computed ($0 audit, 2026-11-17)

The packed search (`a_packed_search_20261105_195011`) expanded the teacher itself with every operator under both
profiles before anything else. Those first-level states are A's selections, and
[`selection.json`](../../../../configs/stages/stage-1/teacher_parent_initialization/selection.json) records them:

* **FFN**: state `3f743039…`, which is B's first step (same state id, same 36 × 3,072 neurons).
* **DEPTH**: state `fede45db…`.
  * Kept layers K: 0 1 4 5 6 7 9 10 11 12 13 14 15 16 17 18 19 21 22 23 24 25 27 29 31 33 34 35.
  * Removed: 2 3 8 20 26 28 30 32, in the order 2 32 3 26 30 8 20 28.
  * 8 rounds and 260 candidate subsets, every one scored against the unbypassed teacher.
  * `selection.py` checks the record round by round: each round starts from the earlier choices, scores every
    remaining layer, and chooses the lowest.
* **ATTENTION**: state `87ee6205…`, the kept query heads of all 36 teacher blocks.
* **RESIDUAL_WIDTH (rh)**: state `02b33196…`. This is not a selection but the reproduction gate for the statistics
  pass.

All four have the root as parent, the root's materialization, the session's numerical fingerprint and the journal's
operator config hashes. `init_driver.py --check-only` re-derives all four hashes under the packed session at $0, and
they are equal.

Computed on the pod, because the per-point residual statistics of the teacher were never stored (the search volume
is gone, and the journal holds only diagnostics):

* **The teacher statistics pass.** It replays the root RESIDUAL_WIDTH(rh) state with its statistics retained, about
  3 minutes on an L40S. It is valid only if the replayed child equals the journal's digest bit for bit, and P over
  all 37 points reproduces the recorded diagnostics. RI-S1 reproduced its width child this way.
* **The gates.** The FFN, DEPTH and ATTENTION children rebuilt on the teacher must each equal their journal digest,
  which proves each selection was made on the unmodified teacher.
* **A itself.** It is built from one frozen mapping, with geometry and parameter count checked.

B's evidence is audited in [`audit/b_evidence.json`](../../../../logs/stages/stage-1/teacher_parent_initialization/audit/b_evidence.json):

* both seeds × both modes × every eval-set-1 unit are present and complete;
* every protocol field and identity is bound;
* rescored, B's thinking-mode static fluid accuracy is 10.35% and 10.18%, which reproduces A-MIX's 10.26%.

## 3. Training and evaluation

**Training.** [`plan.py`](../../../../scripts/stages/stage-1/teacher_parent_initialization/plan.py) derives the two
units from the frozen `mixed_training_plan.json`. Only the unit and arm names, the run name, the student path, the
output directory and the initialization lineage change, and any other difference is refused. The recipe:

* 9,701 optimizer steps (3 epochs of 6,467 packed sequences of 8,192 tokens);
* batch size 2, micro-batch size 1, 2 gradient accumulation steps;
* fp32 parameters with bf16 autocast and gradient checkpointing;
* CE 0.25 + KD 1.0 at the assistant positions, Top-200 + tail;
* AdamW 5e-5, 485 warmup steps, cosine to 0.1;
* the embedding and tied head frozen;
* sequential packing with truncation and standard causal attention.

The data is 11,862,564 unique supervised tokens before packing (thinking 5,935,809 / non-thinking 5,926,755), and
no Stage-2 data is regenerated.

**Evaluation.** It uses B's protocol exactly:

* eval-set-1 (development-exposed), with `nucleus_v1`;
* the context resolved by the runners' trained-context rule to 8,192, as B's was. A sample generates until EOS or
  that context minus its prompt;
* action protocol v2 and the hybrid's tokenizer files;
* thinking and non-thinking mode, every item of every unit, for both seeds.

Generation streams and seals each finished sample, so a later attempt fills only the missing ones.

## 4. The registered analysis

It is registered before any A result exists, in `design.json :: analysis`. `analysis.py` implements it.

* **Primary comparison.** Thinking-mode static fluid formal core (570 items in 558 independence groups), A − B,
  pooled over the two seeds. An item and its other-seed copy fall in one group.
* **Test.** A group-level paired sign-flip test (100,000 draws), with a group-bootstrap 95% interval beside it.
* **Multiplicity.** Fixed sequence: the key secondary comparison (non-thinking mode) is tested only if the primary
  rejects at 0.05.
* **Verdicts.** A_BETTER, A_WORSE or NOT_DISTINGUISHED (which is not equivalence). A verdict is given only when A's
  records of both seeds cover all 570 items in that mode.
* **Descriptive only.** Per-seed differences and whether the two seeds agree in sign, the interactive episodes, each
  reasoning axis, the supporting units, terminations apart from correctness, state evaluation, training curves and
  cost.

Two training seeds bound what can be said about training randomness.

## 5. Sessions, money, storage

| session | does | expected | hard cap |
| --- | --- | --- | --- |
| `teacher_parent_init` | build A, the statistics, state evaluation; relay copies; `init_record.json` | ~$0.90, ~45 min | $2.00 |
| `teacher_parent_train` | two seeds × (training ~9.0 h + evaluation ~1.5 h) | ~$24.30, ~22 h | $28.00 |

* **Proposal** ([`plans/proposal.json`](../../../../logs/stages/stage-1/teacher_parent_initialization/plans/proposal.json)):
  $31.50 for sessions plus a $1.00 teardown reserve, $32.50 in all, inside the $32.67 the $1,000 project cap leaves.
  It is not granted, and no earlier study's allowance funds it.
* **Hardware:** one L40S, secure cloud, at most $1.11/h.
* **Storage:** no network volume. The relay holds the round's weights while it runs: the init, the statistics,
  rolling save points (weights + AdamW state + RNG + data position, about 5.9 GB, the previous one permanently
  deleted after the next verifies), and the finals. The dev box keeps the init, its statistics and both finals at the
  end (`launch.py retain`).

The sequence, once granted (`launch.py` = `PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python
scripts/stages/stage-1/teacher_parent_initialization/launch.py`):

```
launch.py grant --decision "<the maintainer's words>" --granted-by maintainer     # then commit
launch.py issue --kind teacher_parent_init --run-id tpi_init_<utc> --expected-usd 0.9 --hard-cap-usd 2.0
#   commit the artifact (the session commit), stage the bundle (stages.phase_c1.bundle.stage_bundle), push
launch.py launch --kind teacher_parent_init --minutes 45 --run-id ... --scr <scr> --session-commit <c> \
    --bundle aad_autoinit_<c8>.bundle --max-price 1.11 --disk-gb 150 --create-attempts 72 --create-retry-seconds 300
launch.py --kind teacher_parent_init retain --run-id ... --from <scr>       # A, its mapping and statistics on the dev box
#   copy <scr>/store/out/init_record.json into the run's evidence and commit it
plan.py --init-record logs/stages/stage-1/teacher_parent_initialization/runs/<init run>/evidence/init_record.json --write
launch.py issue --kind teacher_parent_train --run-id tpi_train_<utc> --expected-usd 24.3 --hard-cap-usd 28.0
launch.py launch --kind teacher_parent_train --minutes 1380 ...                    # as above
launch.py --kind teacher_parent_train retain --run-id ... --from <scr>      # both finals on the dev box
analysis.py --a-root <scr>/store/out                                       # analyses/teacher_parent.json
```

After a pod loss, `launch.py stage-resume --run-id <new> --from <old scr>` packs the secured evidence, and its
`<path>@<sha256>` goes to `--resume-from` at issue and launch. The new attempt restores the latest save point or
the final weights from the relay, and fills only the missing samples.
