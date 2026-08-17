# Archived planning documents

Retained for provenance. **None of these is an active plan.** Each carries an
ARCHIVED header naming what superseded it. The current state is
[`../STATE.md`](../STATE.md) and the current handoff is
[`../../docs/archive/HANDOFF_AUTOINITIALIZER_20260812.md`](../../docs/archive/HANDOFF_AUTOINITIALIZER_20260812.md)
— itself archived, and bannered as provenance rather than a plan.

* **PROPOSAL.md** — titled 'Active proposal — Experiment 2'; phases 2-3 were never authorized and phase 3 was built around a since-retired metric
* **e8b_preregistration.md** — preregisters the E8b 2x2 recovery design, which is strategically terminated with no valid comparison
* **e8_preregistration.md** — preregisters the original E8 2.96M recovery, cancelled before execution and replaced by E8a/E8b

## Removed, and where it went

**`current_state_20260817_full.json`** — the pre-normalization
`logs/current_state.json`, 37,871 bytes, kept here as a snapshot of a living-state
file. Git already holds those exact bytes:

```
git show 3261f6b67e513a9c7c4260e3a7ccc91c847dc127:logs/current_state.json
```

That is verified rather than asserted — `sha256 4a4ed901b088f987…` on both sides,
checked on every run of `scripts/consolidate/build_log_inventory.py` and asserted
by `tests/docs/test_storage_inventory.py`. A file in the working tree can drift
from what it claims to be a copy of; a git object cannot, so the reference is the
stabler citation and the second copy is gone.
