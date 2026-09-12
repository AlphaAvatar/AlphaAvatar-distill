# logs/maintenance

Records of looking after the repository itself: storage measurements, scratch
inventories and closeouts, derived-cache cleanups, and the relocation record.

None of it is experiment evidence and none of it authorizes anything.

| record | what it owns |
| --- | --- |
| [`log_relocation.json`](log_relocation.json) | the 2026-09-12 relocation: every move old → new, and every exception with the reason it could not move |
| [`storage_closeout_20260831.json`](storage_closeout_20260831.json) | the Phase-B storage closeout |
| [`scratch_closeout_20260831.json`](scratch_closeout_20260831.json) | the proof behind that deletion |
| [`scratch_inventory_20260829.json`](scratch_inventory_20260829.json) | the scratch inventory it rests on |
| [`derived_cache_cleanup.json`](derived_cache_cleanup.json), [`derived_cache_cleanup_20260827.json`](derived_cache_cleanup_20260827.json) | rebuildable-cache cleanups |

Storage totals and the checkpoint registry are **not** here: they are still read
by executables at their existing paths, and are listed as exceptions in the
relocation record with the naming sites.
