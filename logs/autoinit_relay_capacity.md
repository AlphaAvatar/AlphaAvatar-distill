# Relay capacity accounting, before any upload. $0.

Produced before staging anything, because **deleting from this relay reclaims
nothing**: Hugging Face bills LFS storage including history, and the 2026-08-01
attempt to free space by deleting 19.07 GB of superseded weights dropped the
working tree to 80.31 GB and reclaimed **zero** ([`decisions.md`](decisions.md)
2026-08-02, STATE §6). Every gigabyte staged here is permanent.

## Current position

| quantity | value | how it is known |
| --- | --- | --- |
| relay tracked size | **82.36 GiB** (88.44 GB), 831 files | queried live via `HfApi().repo_info(files_metadata=True)` |
| hard quota | **not exposed by the API** | inferred below, not measured |
| sa checkpoint (`model/`) | 2.23 GiB | measured on disk |
| sb checkpoint (`model/`) | 2.22 GiB | measured on disk |
| **sa + sb together** | **4.45 GiB** (4.78 GB) | |

**The quota is the weak number in this table and it is the one the decision turns
on.** The Hugging Face API exposes repository size but not the account's private
LFS allowance, so it cannot be read. What is known: on 2026-08-01 this relay
*did* hit its private-storage limit while the tree held roughly 99 GB, which puts
the ceiling near **100 GB (93.1 GiB)**. That is an inference from one event, not
a measurement, and it should be confirmed from the account's billing page before
anything large is staged.

## The accounting, against the inferred 93.1 GiB ceiling

```
inferred hard quota                              93.13 GiB   (100 GB, INFERRED)
current irreclaimable usage                      82.36 GiB
                                                 ---------
headroom today                                   10.77 GiB

+ sa and sb (this session's staging)               4.45 GiB
                                                 ---------
headroom after staging the controls                6.32 GiB

minimum expected Phase-A staging requirement       NOT ESTABLISHED
safety reserve (one re-upload of both controls)    4.45 GiB
                                                 ---------
uncommitted headroom after reserve                 1.87 GiB
```

**Phase-A's relay requirement is not established, and I decline to invent it.**

**`k` is a storage decision only.** The scientific design is frozen and is not
in question: **5 searched leaves + the exact canonical control**, two seeds, the
preregistered halving schedule. `k` is *how many leaf checkpoints need
simultaneous relay retention or long-term physical storage* — nothing else. The
number of searched or recovery candidates must never be reduced to solve a
storage problem; if storage is short, the answer is to retain fewer *checkpoints*
after the fact, or to store them elsewhere, not to search less.

The search and its recovery probes run on one pod, so intermediates never need
the relay. What needs it is retention after the session ends: *k* leaves at
~2.15 GiB each. At k=2 that is 4.3 GiB, which does not fit the 1.87 GiB left
above; at k=0 (retain metrics and hashes only, rebuild a leaf from its recorded
recipe if it is ever needed again) it is free. **This has to be decided before
Phase A, not discovered during it** — and it is a decision about what is kept,
not about what is run.

## Recommendation

**STAGING IS ON HOLD** at the maintainer's direction until the real quota
is confirmed. **Do not stage yet.** The accounting is dominated by an inferred quota, and the
margin under it is ~1.87 GiB after a single re-upload reserve — thin enough that
one mistake is unrecoverable, because deletion does not reclaim.

Two cheap things resolve it, in order:

1. **Confirm the actual quota** from the account's storage/billing page. One
   number turns this table from inference into accounting.
2. **Decide Phase-A leaf retention** (*k*). Until *k* is chosen, "minimum
   expected Phase-A staging requirement" cannot be filled in and the reserve
   above is guesswork.

If the confirmed quota leaves comfortable headroom, stage both checkpoints
**completely, before a paid pod exists** — the uplink is 0.72 MB/s, so 4.45 GiB
is ~1.8 hours per checkpoint of *unpaid* dev-box time, against roughly $0.05 of
paid pod time to pull them back down. Doing it while a pod bills would be the
expensive mistake, and doing it blind would be the irreversible one.

## The alternative, if headroom is short

The continuation does not strictly need the relay. The controls could be
transferred **dev-box → pod directly over scp** while the pod runs: 4.45 GiB at
the dev box's uplink is ~1.8 h of *paid* pod time, about **$1.80** — which alone
exceeds the whole continuation's expected cost. That is the trade: ~4.45 GiB of
permanent, irreclaimable relay quota, or ~$1.80 of pod time per attempt, every
attempt.


## Transport decision, 2026-08-14 — BLOCKED on one input

Both routes are quantified. **Neither can be selected without the maintainer**,
and the authorization is therefore not issued.

| route | transfer time for 4.45 GiB | cost of the transfer | continuation total |
| --- | --- | --- | --- |
| **relay**, pre-staged before a pod exists | 1.8–5.4 h of **unpaid** dev-box uplink | **$0.00** | **$0.90 / $1.75** unchanged |
| **scp**, while the pod bills | same wall time, on the clock | **$1.82–$5.61** | **$2.72–$6.51 expected** |

The uplink is the second uncertain number: **0.72 MB/s** is this project's
recorded figure, while today's 3.7 MB bundle upload moved at ~**0.23 MB/s** —
small enough that overhead may dominate, so the true sustained rate is somewhere
in that range. It does not change the ranking, only the size of the penalty.

**scp is not viable under the current pricing**, and repricing it has a
consequence beyond this session: a $3.60 continuation would leave **$20.03** of
the project cap against a **$21.01** provisional Phase-A hard bound — Phase A
would no longer fit at all. Paying for transport twice (once now, once for any
future session that needs these controls) is how a $4.45 GiB storage decision
turns into a Phase-A cancellation.

### The three options, none taken here

1. **Confirm the quota → relay.** One number from the account's storage/billing
   page. If headroom ≥ ~4.5 GiB plus a reserve, pre-stage before any pod exists,
   keep $0.90 / $1.75, and every future session that needs these controls pulls
   them at ~$0.05 instead of paying transport again.
2. **Reprice for scp.** ~$2.72 expected / ~$3.60 hard at the optimistic rate,
   worse at the pessimistic one, and it costs Phase A its remaining headroom.
3. **Publish the two checkpoints to a public HF repo.** Public storage is not
   charged against the private quota, which dissolves the constraint entirely —
   but publishing model weights is a **release decision** (AGENTS.md P12/P15),
   not an operational one, and it is not mine to make.

### One further item, whichever route is chosen

The pod does not clone from GitHub: `autoinit_preflight_setup.sh` fetches a **git
bundle** from the relay and checks out `SESSION_COMMIT`. So "the remote the pod
checks out" is the bundle, and a bundle at `6428c39` does not exist yet. It is
3.7 MB — negligible against the quota — but it must be built and uploaded from
the *final* commit, which is why it is deliberately not done before the transport
and pricing are fixed.
