# a2 has a journal and no parseable record

`out/adoption/adoption.json` arrived TRUNCATED at 211 MB of an approximately
300 MB file — the driver recorded 4,096 per-position mass floats per
reduction across 17,420 reductions, roughly 71 million floats, and the fetch
did not complete. The launcher warned (`could not fetch the evidence
archive`) and tore the pod down; the file on disk will not parse.

`journal.jsonl` is the evidence. It is append-only and fsynced per event, so
it survived intact and records every stage, its duration and its outcome.

What a2 established: stage C ran 1967.7 s and `analyse_ABC` PASSED in-run,
which means the corrected domain-balanced aggregation reproduced the
operator's winner in every round — the driver raises otherwise. So a2 is an
independent confirmation of a1's Top-K result, from a second pod.

What it did not establish: stage D, which failed 1987 s in on a BARE policy
id where the registry keys on qualified ones.

The driver now reduces the mass on the pod, so this record shape cannot
recur.
