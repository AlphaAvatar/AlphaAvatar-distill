# depth_recurrence configs (Stage 1 research, Stage 3 recovery: DMR1 variant 2 revised 2026-11-28; variant 3 not started)

| file | what it is | identity |
| --- | --- | --- |
| `design.json` | DMR1-R / DMR1-RH: the recurrent-execution identity (what is shared, how often it runs, the initialization, the data and packing, the objective, the compute accounting), the 36-logical-layer schedule, each loop arm's fixed base (V2-D KEPT, V2-M MERGED; the 2026-11-26 selection kept under `superseded_2026_11_28`), the engine validation, the eval-only state probe, the hidden-state term (variant 3, not started), the registered analysis (R1-R4), storage | study identity |
| `base_config.json` | a byte copy of the base's `config.json` (sha256 `191c51bb…`: SOB1's depth init and variant 1's build save the same file) | study identity |
| `recurrent_config.json` | the recurrent model's `config.json` (`schedule.py config`: `recurrent_depth.recurrent_config_from(base_config, layer_to_block)`) | study identity |
| `artifacts.json` | the collector's spec for `dmr1_depth_recurrence` sessions (incl. the engine check's files) | |

The schedule and every count are derived from SOB1's committed mapping (`schedule.py schedule`); nothing in
`src/aadistill` names them. Nothing here authorizes anything; no paid run of variant 2 or 3 is authorized (variant 2's proposal awaits review).
