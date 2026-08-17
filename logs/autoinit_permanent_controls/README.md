# The two permanent controls — their own evidence

`preflight_ctl_r0860k_sa` and `_sb`: probe identity, run manifest and run
completion for each. Produced by the micro-preflight's **fourth** attempt
(L40S, 217.9 min, $3.6000), whose full evidence is in
[`../autoinit_preflight_run4/`](../autoinit_preflight_run4/).

The controls' weights live in the out-of-tree store and on the relay under
`permanent_controls/`; both are inventoried in
[`../checkpoint_registry.json`](../checkpoint_registry.json) and are protected
from deletion by the retention rule.

Evidence as the run produced it. Nothing here is rewritten.

## Not stored here

| file | sha256 | the copy that is kept |
| --- | --- | --- |
| `session_evidence.json` | `b39bc39e5908…` (full hash in [`../log_inventory.json`](../log_inventory.json)) | [`../autoinit_preflight_run4/preflight_evidence.json`](../autoinit_preflight_run4/preflight_evidence.json) |

One session wrote one evidence file. The kept path is the one
`scripts/autoinit/write_preregistration.py` and
`logs/autoinit_phase_a_preregistration_materialized.json` cite; the copy that
used to sit here was the same 45,362 bytes under a second name.
