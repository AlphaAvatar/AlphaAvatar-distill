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

## What is in here

| group | what it holds |
| --- | --- |
| [`handoffs/`](handoffs/) | superseded handoff documents, kept verbatim: [`HANDOFF_next_session.md`](handoffs/HANDOFF_next_session.md) — the current one is [`../state/current.md`](../state/current.md) |
| [`indexes/`](indexes/) | superseded experiment indexes; the live index is [`../runs/index.json`](../runs/index.json) |
| [`superseded/`](superseded/) | documents retired by a later decision, each with the record that replaced it |
| [`STATE_superseded_through_2026-09-11.md`](STATE_superseded_through_2026-09-11.md) | the repository state as it stood, spanning Phase A, Phase B and the continuations |
| [`CATALOG_detail_through_2026-09-11.md`](CATALOG_detail_through_2026-09-11.md) | the per-entry log detail that `state/ownership.md` used to carry |

Every group has a stated origin. `archive/` is **not** where unclassified files
go: an object with no owner gets one, not a shelf here.

Links inside these documents are **not** repointed when the repository moves.
They record where a file was when the document was written; follow them through
[`../migrations/log-layout-v1/manifest.json`](../migrations/log-layout-v1/manifest.json).
