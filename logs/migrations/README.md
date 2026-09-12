# logs/migrations

How the repository's own structure changed, and how to follow an old path
forward. One directory per migration.

| migration | what it did |
| --- | --- |
| [`log-layout-v1/`](log-layout-v1/) | gave `logs/` a single canonical layout: one run hierarchy instead of three, and no loose files at the root |
| [`initialization-core/`](initialization-core/) | relocated initialization source, against every frozen declaration it touched |

## Reading a migration manifest

Each entry maps `old_path` → `new_path` with the identity the object had before
the move, so "nothing changed but the location" is checkable rather than
asserted. That is what makes an old path in a frozen payload readable instead of
broken: a consumed authorization naming `logs/autoinit_c1_attempt5/…` is stating
where that evidence was when it was issued, which is still true and is what git
history holds. The manifest says where it is now.

Those strings are **not** rewritten. Editing a payload to match a new directory
layout would be changing evidence to suit housekeeping.
