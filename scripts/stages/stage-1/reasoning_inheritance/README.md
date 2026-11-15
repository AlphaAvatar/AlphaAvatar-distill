# reasoning_inheritance (Stage 1)

Stage 1 structural ablation: four **untrained** models built from one frozen structural mapping of the packed full
search's Top-1, measured on the state suite and on every unit of eval-set-1 in both modes. No recovery training, no new
search. The experiment is `configs/stages/stage-1/reasoning_inheritance/design.json`; evidence lives in
`logs/stages/stage-1/reasoning_inheritance/`.

**Status: COMPLETE, study CLOSED (2026-11-14).** `launch.py issue` / `launch` refuse (`study_engineering.require_open`);
the $0 scripts below still run. D, W and DW are on the dev box only,
`~/aad-artifacts/stage1_reasoning_inheritance/913433664fce0fed/` (`storage/local_checkpoints.json`); their relay weight
copies were deleted, so `driver.relay_restore` (the resume path) can no longer fetch them. A new study that needs them
reads the dev-box copy or rebuilds them from the mapping, which stays on the relay and the dev box.

| group | depth | hidden | FFN | Q / KV heads | parameters |
| --- | ---: | ---: | ---: | ---: | ---: |
| T: full teacher (Qwen/Qwen3-4B @ `1cfa9a72`) | 36 | 2560 | 9728 | 32 / 8 | 4,022,468,096 |
| D: depth pruning only | 28 | 2560 | 9728 | 32 / 8 | 3,215,021,568 |
| W: all non-depth compression only | 36 | 1024 | 3072 | 16 / 8 | 721,897,472 |
| DW: both | 28 | 1024 | 3072 | 16 / 8 | 596,049,920 |

## Files

| file | runs on | what it does |
| --- | --- | --- |
| `selection.py` | dev box, $0 | reads the anchor journal (`617d1512...`) into `frozen_selection.json`: 36 blocks' FFN neurons, the kept blocks, the 28 kept blocks' query heads in root numbering, the width step's recorded diagnostics and every ancestor's identity; `--check` re-reads it byte for byte |
| `driver.py` | pod | contract (config hashes vs the journal) -> mapping -> relay copy -> state evaluation -> eval-set-1; `--check-only` runs the contract at $0. Its runners `fluid` and `capability` serve RI-LM1 too: a capability unit is sealed the moment the runner has written it, each finished sample is streamed and sealed, and a later attempt generates only the samples with no sealed record (by id and seed) |
| `launch.py` | dev box | the RI-S1 study on `shared.study_engineering`: grant, one-use artifact, launch, closeout, `stage-resume`, `archive-evidence` |
| `teacher_reuse.py` | dev box, $0 | T's evaluation from `a_teacher_20261030_012648`: archive and file hashes, every unit's identity against this round's protocol, then item by item (coverage of the registered unit, every seed against the decoding rule, every prompt's fields against this round's groups, `--store`, and the non-fluid units' fields against the registered items rendered with the teacher's verified tokenizer, `--tokenizer`); a field absent or invalid on either side is `not recorded`, never equal; prompt content identity reported apart (fluid: verified from the text; the rest: derived from the construction path at both session commits, never from equal lengths); rescored with the current scorer; regression `tests/test_teacher_reuse.py` |
| `analysis.py` | dev box, $0 | every group rescored; per-unit correct and usable, the five structural contrasts with family-cluster bootstrap intervals (and the registered independence grouping beside them), retention of T-correct items, what every unscored row ended as (answer read and wrong, no answer the scorer can read, protocol error, degeneration stop, context limit), the sealed degeneration stops checked for continuation, state KL/NLL; descriptive extras (correct among usable rollouts, T-correct items kept by T's own length), each scored directory bound by content digest |
| `render.py` | dev box, $0 | the report's tables from `analyses/structural_ablation.json` |
| `layer_merge.py` | pod (and dev box, $0) | the next round's instance of `aadistill.initialization.planning.layer_merge`: spans from the frozen selection, the statistics forward at the FFN operator's positions, the alignments, M-avg / M-act, the fold and permutation equivalence checks, the bit checks against D's digests, the update diagnostics; `digests` (D's per-tensor digests, $0) and `spans` |
| `layer_merge_driver.py` | pod | RI-LM1: contract (and every product's identity) -> per candidate, one after the other: build (D rebuilt and compared; M-act: statistics, alignments, checks), save, its span-block delta, the reconstruction D + delta -> its artifact digest, relay copy, `durable` -> restore as needed -> diagnostics -> state evaluation -> eval-set-1 through RI-S1's runners; a product is reused only under this attempt's identity, else superseded; `--check-only` runs the contract at $0 |
| `layer_merge_launch.py` | dev box | the RI-LM1 study on `shared.study_engineering` (proposal only, no grant): `issue` (with the teardown-reserve and budget-terminal checks), `launch` (with the relay-room check for the two deltas), `closeout`, `retain` (each pulled delta checked against D's dev-box copy and kept in `~/aad-artifacts/stage1_layer_merge/`), `position`, `grant`, `stage-resume`, `archive-evidence` |
| `layer_merge_analysis.py` | dev box, $0 | the registered decision rule over the 570 registered items: the group sign-flip test, Holm over both registered comparisons, verdicts only when both primary units are complete (else `descriptive_only`), the group-bootstrap effect interval, the rule-family supplement |

The reusable mechanism is `src/aadistill/initialization/planning/frozen_mapping.py` (core, no instance facts):
the mapping contract and builders that repeat the four operators' surgery bit for bit
(`tests/initialization/test_frozen_mapping.py`).

## How the mapping is built (driver `mapping` phase)

1. The teacher, every shard hashed (`full_search.load_root`).
2. The FFN child (36 blocks) from the frozen FFN selection, and the FFN -> DEPTH parent from the kept blocks; each
   compared with the journal's artifact digest.
3. `width.global_pca_v0` and `attention.activation_importance_v1` replayed on that parent by the fixed-path executor
   under the search's own session (`packed_full_search.build_session`, inputs `v3`), each step's config hash pinned to
   the journal's (`58c86cde...`, `958e14e6...`); the width operator's statistics kept (`RetainingStatsCache`) and `P`,
   its scale and the final-norm vector recomputed from them, then the width child rebuilt from `P` and compared with
   the operator's bit for bit.
4. The attention operator once more, on the 36-block parent with the frozen FFN selection and `P`: only the eight
   removed blocks' heads are taken; the 28 kept blocks keep the journal's.
5. The mapping saved (`mapping.json` + `tensors.safetensors`, content digest), D / W / DW materialized from the saved
   copy and checked (parameters, D's blocks are T's, W's kept blocks and shared tensors are DW's), then uploaded to the
   private relay under `durable/stage1_reasoning_inheritance/<digest[:16]>/` and downloaded again and re-hashed.

If the rebuilt parent, width child and DW all have the journal's digests, the historical Top-1 is reproduced exactly;
otherwise the snapshot has its own identity, W and DW both come from it, and the record says the reproduction was not
established.

## Commands

```bash
export PYTHONPATH=src:scripts:scripts/stages/stage-1
# $0 checks
.venv/bin/python scripts/stages/stage-1/reasoning_inheritance/selection.py --check
.venv/bin/python scripts/stages/stage-1/reasoning_inheritance/driver.py --check-only --out /tmp/ri/out --work /tmp/ri/work --status /tmp/ri/st
.venv/bin/python scripts/stages/stage-1/reasoning_inheritance/teacher_reuse.py \
    --store artifacts/stages/stage-1/reasoning_inheritance/runs/ri_diagnosis_20261113_193508/scr/store

# a session (one-use artifact, then the launch; the bundle is staged with stages.phase_c1.bundle.stage_bundle)
RUN=ri_diagnosis_$(date -u +%Y%m%d_%H%M%S); C=$(git rev-parse HEAD)
.venv/bin/python scripts/stages/stage-1/reasoning_inheritance/launch.py issue --run-id $RUN --expected-usd 16 --hard-cap-usd 24
.venv/bin/python scripts/stages/stage-1/reasoning_inheritance/launch.py launch --minutes 1120 --run-id $RUN \
    --scr artifacts/stages/stage-1/reasoning_inheritance/runs/$RUN/scr --session-commit $C \
    --bundle aad_autoinit_${C:0:8}.bundle --max-price 1.09 --disk-gb 150 --create-attempts 72 --create-retry-seconds 300 \
    --poll-limit-min 1300   # the runner stops polling (and tears down) after this; its default is 600

# resume a later attempt from an earlier one's secured evidence (checkpoints come back from the relay, re-hashed)
R=$(.venv/bin/python scripts/stages/stage-1/reasoning_inheritance/launch.py stage-resume --run-id $RUN2 \
      --from artifacts/stages/stage-1/reasoning_inheritance/runs/$RUN/scr)
.venv/bin/python scripts/stages/stage-1/reasoning_inheritance/launch.py issue --resume-from $R --run-id $RUN2 ...
.venv/bin/python scripts/stages/stage-1/reasoning_inheritance/launch.py launch --resume-from $R --run-id $RUN2 ...

# analysis ($0): every attempt's store, later ones winning per unit; tables for the report
.venv/bin/python scripts/stages/stage-1/reasoning_inheritance/analysis.py \
    --store artifacts/stages/stage-1/reasoning_inheritance/runs/$RUN/scr/store
.venv/bin/python scripts/stages/stage-1/reasoning_inheritance/render.py \
    logs/stages/stage-1/reasoning_inheritance/analyses/structural_ablation.json

# the final store as a durable archive on the relay ($0), downloaded again and re-hashed
.venv/bin/python scripts/stages/stage-1/reasoning_inheritance/launch.py archive-evidence --run-id $RUN \
    --from artifacts/stages/stage-1/reasoning_inheritance/runs/$RUN/scr
```

The run that measured every unit is `ri_diagnosis_20261113_193508`; its archive is
`durable/stage1_reasoning_inheritance/evidence/ri_diagnosis_20261113_193508/c69706526086bdf3/evidence.tgz`. The
session commands above are historical (the study is closed).

## Resume

Every finished file leaves the pod on each poll (`launch.secure_incremental`); the fluid unit streams finished
episodes and seals them into immutable parts every two minutes; a capability unit is sealed when the runner has moved
past it. A new attempt takes over the mapping, records, state evaluation and sealed parts (`driver.take_over`),
restores the checkpoints from the relay by sha256, and generates only the fluid tasks with no sealed record, each
under its fixed per-item seed. No item is generated twice for a model and mode, and none is chosen by its outcome.

## Next round: layer merging at D's structure (RI-LM1) -- prepared, NOT authorized

Design `configs/stages/stage-1/reasoning_inheritance/layer_merge/design.json`; the plan and the decision rule in
`docs/stages/stage-1/reasoning_inheritance/REPORT.md` section 11; the budget draft in
`logs/stages/stage-1/reasoning_inheritance/plans/layer_merge_proposal.json` (`status: proposed`). Regression:
`tests/initialization/test_layer_merge.py` (core) and `tests/test_layer_merge.py` here (the whole pipeline on the tiny
Qwen3).

```bash
export PYTHONPATH=src:scripts:scripts/stages/stage-1
# $0 checks
.venv/bin/python scripts/stages/stage-1/reasoning_inheritance/layer_merge.py spans
.venv/bin/python scripts/stages/stage-1/reasoning_inheritance/layer_merge_driver.py --check-only --out /tmp/lm/out --work /tmp/lm/work --status /tmp/lm/st
.venv/bin/python scripts/stages/stage-1/reasoning_inheritance/layer_merge_launch.py position --hard-cap-usd 11

# only after the maintainer's grant (it is written from the proposal, quoting the decision)
.venv/bin/python scripts/stages/stage-1/reasoning_inheritance/layer_merge_launch.py grant --decision '<quoted>' --granted-by maintainer
RUN=ri_merge_$(date -u +%Y%m%d_%H%M%S)
.venv/bin/python scripts/stages/stage-1/reasoning_inheritance/layer_merge_launch.py issue --run-id $RUN --expected-usd 7.5 --hard-cap-usd 11
git add -A && git commit -m "RI-LM1 session commit: one-use authorization for $RUN"; C=$(git rev-parse HEAD)
#   stage the bundle for $C (stages.phase_c1.bundle.stage_bundle), then:
.venv/bin/python scripts/stages/stage-1/reasoning_inheritance/layer_merge_launch.py launch --minutes 480 --run-id $RUN \
    --scr artifacts/stages/stage-1/reasoning_inheritance/runs/$RUN/scr --session-commit $C \
    --bundle aad_autoinit_${C:0:8}.bundle --max-price 1.09 --disk-gb 150 --create-attempts 72 --create-retry-seconds 300 \
    --poll-limit-min 620 --driver-extra '--max-num-seqs 7'
.venv/bin/python scripts/stages/stage-1/reasoning_inheritance/layer_merge_launch.py closeout --run-id $RUN \
    --scr artifacts/stages/stage-1/reasoning_inheritance/runs/$RUN/scr
# each durable candidate's pulled delta, checked against D's dev-box copy and kept beside it ($0)
.venv/bin/python scripts/stages/stage-1/reasoning_inheritance/layer_merge_launch.py retain --run-id $RUN \
    --from artifacts/stages/stage-1/reasoning_inheritance/runs/$RUN/scr

# a later attempt: the earlier attempt's records and sealed parts (reused only under this attempt's identity), each
# durable candidate restored from D and its delta, never rebuilt
R=$(.venv/bin/python scripts/stages/stage-1/reasoning_inheritance/layer_merge_launch.py stage-resume --run-id $RUN2 \
      --from artifacts/stages/stage-1/reasoning_inheritance/runs/$RUN/scr)
#   issue and launch $RUN2 with --resume-from $R

# the decision ($0)
.venv/bin/python scripts/stages/stage-1/reasoning_inheritance/layer_merge_analysis.py \
    --d-store artifacts/stages/stage-1/reasoning_inheritance/runs/ri_diagnosis_20261113_193508/scr/store \
    --store artifacts/stages/stage-1/reasoning_inheritance/runs/$RUN/scr/store
```

