# Phase A, attempt 6 — $0.3552

Stage 0 **passed**. Stage 1 died on device placement: `_validate` probed on
`config.device` while the child was on the host. Diagnosis in
[`../decisions.md`](../decisions.md) (2026-08-16); the fix became
`autoinit.stage1_device_contract@v1`.

Evidence as the run produced it. Nothing here is rewritten. The launcher's own
session record is [`../autoinit_phase_a_session.json`](../autoinit_phase_a_session.json).

## Not stored here

| file | sha256 | the copy that is kept |
| --- | --- | --- |
| `engine_probe.json` | `49593edb8bdf3047ee4feccd4bb59d0f954124b9ca3c894460b6be0dece8d431` | [`../autoinit_stage3_complete/engine_probe.json`](../autoinit_stage3_complete/engine_probe.json) |

This attempt's probe was **byte-identical** to that one, which is the evidence
that it ran the preregistered engine — and that evidence is the hash above, not a
third copy of 1,779 bytes. The kept copy is the one
`scripts/pod/autoinit_phase_a_driver.py`,
`logs/autoinit_phase_a_protocol_compat_v2.json` and two tests name by path.
