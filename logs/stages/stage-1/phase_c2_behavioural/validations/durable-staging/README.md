# durable-staging

Pre-staging this campaign's completed probes onto a provider network volume, so
that every later pod **attaches** them instead of paying an accelerator to
receive them.

This is engineering infrastructure. It measures nothing, decides nothing and
advances no candidate, and it is deliberately filed here rather than under
[`runs/`](../../runs/) — a directory under `runs/` is summed into this
campaign's settled spend by `campaign_attempts` and `settled_campaign_all_in`,
and this cost belongs to the stage envelope, not to the campaign's scientific
ceiling.

| | |
| --- | --- |
| tool | [`stage_c2_probes_to_volume.py`](../../../../../../scripts/autoinit/stage_c2_probes_to_volume.py) |
| decision | [`durable_backend_decision_20260923.md`](../../analyses/durable_backend_decision_20260923.md) |
| consumed by | `autoinit_c2_behavioural_launch.py :: volume_gate`, at `$0`, before a pod is drawn |

## Rounds

* [`v1`](v1/) — 2026-09-23, ten probes, 22.21 GiB, volume `59qt99zeg5`
  (40 GB, `EU-NL-1`) mounted at `/durable`.

## What the record is, and is not

`staging_record.json` says which probes were copied and that every file
re-hashed to the launcher host's value **from the bytes that landed on the
volume**. That is a transport claim.

It is not an admission decision. A continuation still re-identifies every probe
on the pod that consumes it, against the identity its own campaign announced —
R2 and R6 are unchanged, and so is the check. What changed is only when the
copy was paid for.
