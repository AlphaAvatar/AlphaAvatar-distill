# Device canary, attempt 2 (the retry) — $0.0637, zero canary runs

Died **inside setup**: the wrapper declared `LOCAL_ASSETS = ()` because it needed
neither asset, and the shared setup script copies both unconditionally. Attempt 1
had died before setup on three inherited `self.a` attributes the wrapper never
declared. Neither reached the canary script, so neither says anything about
device placement on CUDA.

**The paid device-canary path is TERMINATED** (2026-08-18). Two authorized
sessions, $0.1240, no measurement. The generic lesson — a session inheriting a
requirement it never declared — is in [`../decisions.md`](../decisions.md) and is
what [`../../docs/SESSION_ARCHITECTURE.md`](../../docs/SESSION_ARCHITECTURE.md)
specifies away. No further canary is prepared.

Evidence as the run produced it. Nothing here is rewritten. Attempt 1's evidence
is in [`../autoinit_device_canary_attempt1/`](../autoinit_device_canary_attempt1/),
including its own `session.json`, which the top-level record no longer holds.

## Not stored here

| file | sha256 | the copy that is kept |
| --- | --- | --- |
| `session.json` | `e75b2c5dcce6…` (full hash in [`../log_inventory.json`](../log_inventory.json)) | [`../autoinit_device_canary_session.json`](../autoinit_device_canary_session.json) |

The launcher writes its session record to
`logs/autoinit_device_canary_session.json` — its `--out` default, and what both
attempts' `launcher.out` refer to. This attempt was the last to write it, so the
top-level file **is** attempt 2's record; the copy that used to sit here was the
same 9,982 bytes under a second name.
