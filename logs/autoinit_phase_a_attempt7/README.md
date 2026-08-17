# Phase A, attempt 7 — $0.3955

Further into stage 1 than attempt 6, and a **second** device bug:
`ActivationStatsCollector`'s accumulators were unplaced. This is the run that
came home with a traceback — [`stage1_traceback.log`](stage1_traceback.log) —
because attempt 6 had taught the driver to send one. Diagnosis in
[`../decisions.md`](../decisions.md) (2026-08-17).

Evidence as the run produced it. Nothing here is rewritten. The launcher's own
session record is [`../autoinit_phase_a_session.json`](../autoinit_phase_a_session.json).

## Not stored here

| file | sha256 | the copy that is kept |
| --- | --- | --- |
| `engine_probe.json` | `49593edb8bdf3047ee4feccd4bb59d0f954124b9ca3c894460b6be0dece8d431` | [`../autoinit_stage3_complete/engine_probe.json`](../autoinit_stage3_complete/engine_probe.json) |

Byte-identical to attempt 6's and to the Stage-3 run's, which is what says all
three ran the same preregistered engine. The kept copy is the one executable
code and the protocol-compatibility record name by path.
