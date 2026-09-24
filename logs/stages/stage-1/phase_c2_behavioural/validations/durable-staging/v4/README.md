# v4 — the P4 reproducibility repair's one probe

`confirmation.1a2b5b030e7e4e202fda3a810ed53c5f.s1523147638` again, and this
time for a different reason. v3 staged it so attempt13 could SCORE it. It was
scored, the verdict consumed that score, and its 950 rows were never collected
— so the archive could not recompute the verdict and the reviewer withheld C2's
final closure on P4 alone.

This round stages the same preserved, identity-verified checkpoint so its
evaluation can be **reconstructed** under the frozen confirmation protocol. The
bytes are the same bytes: artifact `d9af797a565291b3…`, weights
`c5a5c25a954cc180…`, single shard `d84414b35807d723…`, 596,049,920 parameters,
and `durable_ack.json` records `re_identified_from_delivered_bytes: true`.

`ALL_STAGED`, 70.3 min of transfer at 0.57 MB/s, 72.5 min of pod, `$0.0725`,
teardown provider-confirmed.

## A 10 GB volume, not 40

`59qt99zeg5` was 40 GB because it held nine probes. Exactly one probe's weights
now have a consumer — the scorer reads this checkpoint and nothing else — so
`a0zqgxsm7p` is 10 GB. Sizing the backend to the working set is the same rule
as not moving the bytes (AGENTS.md P8.4), applied to the backend instead of the
transfer.

The plan equalled the transfer this time: **"1 probes, 2.22 GiB to stage"**. v3
had to be stopped and restarted because the tool plans from campaign membership
and announced eleven.
