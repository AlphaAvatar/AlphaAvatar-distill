# logs/shared

Records owned by no single experiment. **Not** a shelf: every
entry below states whether it is stage-neutral, and if it is not,
which stage owns it and why it is still here.

## Genuinely stage-neutral

The subject is the machine, the provider or the transport — not
any pipeline stage.

| what | where | status |
| --- | --- | --- |
| Provider / device canary | [`validations/device-canary/`](validations/device-canary/) | terminated — two authorized sessions, zero canary runs |
| Artifact relay mirror verification | [`validations/relay-mirror/`](validations/relay-mirror/) | complete |

## Here, but not stage-neutral

**D-series family and source records, spanning D1/D2/D3** — stage **1**, status `here by design, not by inheritance`.

Stage 1, and genuinely owned by no single experiment: the allocation rule and the source evidence span D1, D2 and D3, so freezing either inside `phase_d1/` would make the next round inherit a record named for its predecessor. Unlike the AutoInit program material below, nothing pins these in place — they are here because the subject is a family, and if the D-series ever collapses to one experiment they move. Both are generated, both authorize nothing, and neither materializes anything.

**AutoInit program analyses and harness validations, spanning the Stage-1 experiments** — stage **1**, status `pinned in place`.

Stage-1 AutoInit program material that belongs to no single experiment, and it is NOT stage-neutral. It stays under `logs/shared/` because roughly twenty-five scripts read these exact paths, including `autoinit_phase_b_driver.py` and `autoinit_continuation_b_driver.py`, which consumed Phase-B and continuation-B authorizations name by digest. Moving the files means editing frozen-set members to tidy a directory, which is not a trade this project makes. Declared here as a known exception with its blocker rather than left looking stage-neutral.

Recorded in [`../stages/index.json`](../stages/index.json),
which carries the evidence for every row.
