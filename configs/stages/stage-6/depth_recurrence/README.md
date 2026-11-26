# depth_recurrence configs (Stage 6, DMR1 variants 2 and 3)

| file | what it is | identity |
| --- | --- | --- |
| `design.json` | DMR1-R / DMR1-RH: the Stage-6 identity (what is shared, how often it runs, the initialization, the data and packing, the objective, the compute accounting), the 36-logical-layer schedule, the initialization selection and its two legal paths, the hidden-state term, the round's registered analysis (C1-C3), storage | study identity |
| `base_config.json` | a byte copy of the base's `config.json` (sha256 `191c51bb…`: SOB1's depth init and variant 1's build save the same file) | study identity |
| `recurrent_config.json` | the recurrent model's `config.json` (`schedule.py config`: `recurrent_depth.recurrent_config_from(base_config, layer_to_block)`) | study identity |
| `artifacts.json` | the collector's spec for `dmr1_depth_recurrence` sessions | |

The schedule and every count are derived from SOB1's committed mapping (`schedule.py schedule`); nothing in
`src/aadistill` names them. Nothing here authorizes anything; no paid run of variant 2 or 3 is authorized this round.
