# teacher_parent_initialization configs (Stage 1)

| file | what it is | identity |
| --- | --- | --- |
| `design.json` | the experiment: the question, the teacher lineage (the hybrid Qwen3-4B, not Thinking-2507), the fixed 596M target, the two arms (`A_teacher_parent`, `B_packed_top1`), where A's selections come from (the packed search's teacher-level states), the PCA point rule and weights, the build order, the training derivation from A-MIX-6cat-sr, B's evaluation protocol, the registered analysis, storage | bound by the TPI1 grant (study identity) |
| `selection.json` | A's frozen selections and their journal identities, written by `scripts/stages/stage-1/teacher_parent_initialization/selection.py`; never edited by hand | bound by the grant; `selection.py --check` reproduces it from the journal |
| `artifacts.json` | the teardown collector's spec for both session kinds | -- |

Instance facts (geometry, parameter counts, seeds, point weights, paths) live here and in the scripts, never in
`src/aadistill`. Changing `design.json` or `selection.json` changes the study identity, and the grant then refuses
to fund an artifact. The training plan, which exists only after the init session, is bound per artifact instead
(`launch.kind_conditions`). Nothing here authorizes anything.
