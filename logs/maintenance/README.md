# logs/maintenance

Records of looking after the repository itself. None of it is experiment
evidence and none of it authorizes anything.

Four owners, by what the record is *about* — not one drawer of dated files.

| directory | what it owns |
| --- | --- |
| [`storage/`](storage/) | how much space is where, and what a closeout freed |
| [`inventories/`](inventories/) | what exists: logs, checkpoints, architecture declarations, scratch |
| [`cleanup/`](cleanup/) | what was deleted, and the proof it was safe to delete |
| [`source-relocations/`](source-relocations/) | code that moved, checked against every frozen declaration it touched |

## storage/

* `storage_measurements.json` — sizes for every root, the standing measurement.
* `storage_closeout_20260831.json` — the Phase-B storage closeout: all four
  roots, the promotions performed **before** any deletion, and the two
  checkpoint duplicate pairs (2.92 GiB) found and deliberately kept, with the
  reason.
* `c1_storage_recovery.json` — the C1 storage recovery.

## inventories/

`log_inventory.json`, `checkpoint_registry.json`, `checkpoint_tombstones.json`,
`scratch_inventory_20260829.json`, and the `architecture_*.json` set. An
inventory owns pointers and identities, never the facts they point at.

**The registry and the tombstones are read by executables at these paths.** A
tombstone names where a deleted checkpoint's result is frozen, which is what
made deleting the weights defensible; do not move either without following the
naming sites.

## cleanup/

* `derived_cache_cleanup.json`, `derived_cache_cleanup_20260827.json` —
  rebuildable caches, removed with the rebuild command recorded.
* `scratch_closeout_20260831.json` — the proof behind the scratch deletion: all
  200 files hashed and matched against canonical copies, every member of the
  three remaining tarballs hashed individually, and what was promoted to
  `aad-artifacts` and to the per-attempt log directories first.

## source-relocations/

`<name>/<version>/source-relocation.json` — one code migration against **every**
frozen declaration it touched: old and new path per file, before/after hashes,
rename versus rename-and-modification versus in-place edit versus removal, the
identity each declaration produced at the base and produces now, and —
exercised rather than asserted — whether an old launch document still fails.
Hierarchical because a relocation is a subject with its own versions.

This is about **source**, not about logs. Where a log file used to live is
git's to answer, plus the minimal old → new table in
[`../index.json`](../index.json) for the paths current tooling must resolve.
