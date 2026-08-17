# Stage 3 COMPLETE — continuation attempt 8, $0.6816

The run that finished the Stage-3 control characterization: both permanent
controls re-measured on the complete path, and the thresholds materialized from
them. Attempt 7's `sa` result is valid partial evidence and is **not** combined
with this run's `sb` — see
[`../autoinit_continuation_attempts/README.md`](../autoinit_continuation_attempts/README.md)
for the seven attempts that did not complete.

Evidence as the run produced it. Nothing here is rewritten.

## Products this directory owns

`materialized_thresholds.json`, the two controls' `*_recovery_search.json` and
`*_per_sample.jsonl`, `manifest.json`, `attested_evaluation_protocol.json`,
`generation_smoke.json`, `watchdog.jsonl`, and `engine_probe.json` — which the
Phase-A driver, `logs/autoinit_phase_a_protocol_compat_v2.json` and two tests
name by path, so it is the canonical copy of that probe.

## Not stored here

| file | sha256 | the copy that is kept |
| --- | --- | --- |
| `session.json` | `9d7b5caa60c6…` (full hash in [`../log_inventory.json`](../log_inventory.json)) | [`../autoinit_continuation_session.json`](../autoinit_continuation_session.json) |

The launcher writes its session record to `logs/autoinit_continuation_session.json`
— that is its `--out` default, and it is what this run's `launcher.log` and
`monitor.log` refer to. The copy that used to sit here was that same file under
a second name.
