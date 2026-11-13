# reasoning_inheritance (Stage 1)

Stage 1 structural ablation: four **untrained** models built from one frozen structural mapping of the packed full
search's Top-1, measured on the state suite and on every unit of eval-set-1 in both modes. No recovery training, no new
search. The experiment is `configs/stages/stage-1/reasoning_inheritance/design.json`; evidence lives in
`logs/stages/stage-1/reasoning_inheritance/`.

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
| `driver.py` | pod | contract (config hashes vs the journal) -> mapping -> relay copy -> state evaluation -> eval-set-1; `--check-only` runs the contract at $0 |
| `launch.py` | dev box | the RI-S1 study on `shared.study_engineering`: grant, one-use artifact, launch, closeout, `stage-resume` |
| `teacher_reuse.py` | dev box, $0 | T's evaluation from `a_teacher_20261030_012648`: archive and file hashes, every unit's identity against this round's protocol, rescored with the current scorer |
| `analysis.py` | dev box, $0 | every group rescored; per-unit correct and usable, the five structural contrasts with family-cluster bootstrap intervals, retention of T-correct items, failure types, state KL/NLL |

The reusable mechanism is `src/aadistill/initialization/transforms/frozen_mapping.py` (core, no instance facts):
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
.venv/bin/python scripts/stages/stage-1/reasoning_inheritance/teacher_reuse.py

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

# analysis ($0): every attempt's store, later ones winning per unit
.venv/bin/python scripts/stages/stage-1/reasoning_inheritance/analysis.py \
    --store artifacts/stages/stage-1/reasoning_inheritance/runs/$RUN/scr/store
```

## Resume

Every finished file leaves the pod on each poll (`launch.secure_incremental`); the fluid unit streams finished
episodes and seals them into immutable parts every two minutes; a capability unit is sealed when the runner has moved
past it. A new attempt takes over the mapping, records, state evaluation and sealed parts (`driver.take_over`),
restores the checkpoints from the relay by sha256, and generates only the fluid tasks with no sealed record, each
under its fixed per-item seed. No item is generated twice for a model and mode, and none is chosen by its outcome.
