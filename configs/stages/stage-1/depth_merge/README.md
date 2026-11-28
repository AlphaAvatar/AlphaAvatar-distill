# depth_merge configs (Stage 1, DMR1 variant 1)

| file | what it is | identity |
| --- | --- | --- |
| `design.json` | DMR1-M: the base (SOB1's depth-only training-pre init, its mapping, plan, control sessions), the spans, the alpha (importance-normalized) merge, the statistics, what is reused and recomputed, the checks, SOB1's recovery recipe and evaluation protocol (verbatim), storage, the cost basis | study identity (`launch.py` binds its sha256) |
| `depth_init_tensor_digests.json` | the base's 178 tensor digests, from the dev box's copy (`merge.py digests`); the build compares every unmerged tensor with them bit for bit | study identity |
| `artifacts.json` | the collector's spec for `dmr1_depth_merge` sessions (every entry may match nothing) | |

This variant's comparison C1 is registered in `scripts/stages/stage-1/depth_merge/analysis.py :: C1` (first
registered 2026-11-26 in the recurrence design, `configs/stages/stage-1/depth_recurrence/design.json`, kept there
under `superseded_2026_11_28`). Instance facts
(kept blocks, spans, seeds, prefixes, prices) live here and in `scripts/stages/stage-1/depth_merge/`, never in
`src/aadistill`. Nothing here authorizes anything.
