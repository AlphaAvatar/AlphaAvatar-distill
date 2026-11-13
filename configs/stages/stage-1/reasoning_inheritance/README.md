# reasoning_inheritance configs (Stage 1)

| file | what it is | identity |
| --- | --- | --- |
| `design.json` | the experiment: questions, teacher lineage, anchor (packed search `a_packed_search_20261105_195011`, Top-1 `1f5e4b42...`), the four groups and their geometry, how the mapping is built, inputs, evaluation protocol (eval-set-1, `nucleus_v1`, 40,960 tokens, both modes), teacher reuse, analysis | bound by the RI-S1 grant (study identity) |
| `frozen_selection.json` | the anchor journal's selections and identities, written by `scripts/stages/stage-1/reasoning_inheritance/selection.py`; never edited by hand | bound by the grant; `selection.py --check` reproduces it from the journal |
| `artifacts.json` | the teardown collector's spec for this session kind | -- |

Instance facts (geometry, parameter counts, seeds, paths) live here and in the scripts, never in `src/aadistill`.
Changing `design.json` or `frozen_selection.json` changes the study identity: the grant then refuses to fund an
artifact, which is the point. Nothing here authorizes anything.
