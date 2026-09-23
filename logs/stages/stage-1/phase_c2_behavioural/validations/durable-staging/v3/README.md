# v3 — staging the one probe whose weights a remaining operation reads

`confirmation.1a2b5b030e7e4e202fda3a810ed53c5f.s1523147638` is the advanced
candidate at the third confirmation seed. attempt12 trained it, announced it
durable and then failed in `attest` before scoring it, so the campaign holds a
probe that is **trained and not validly scored** — AGENTS.md P8.4 state 2, the
one state where a checkpoint's bytes genuinely have a consumer: `score_existing`
reads the model. The resume preregistration's R2 supplies the mechanism and
`what_a_resume_may_never_do` forbids the alternative, so it is scored from these
bytes and never retrained.

`ALL_STAGED`, 63.7 min, `$0.0637`, teardown provider-confirmed.

## Why `--campaign-root` is not the campaign root

It is a **hardlink view** at `aad-scratch/c2b-stage-root` holding nine probes:
this one, plus the eight already verified on the volume. Hardlinks, so no bytes
were copied — `links=2`, same inode.

The first run of this staging (`c2-durable-staging-3`) was stopped after it
announced *"11 probes, 24.43 GiB to stage"*. The tool plans from campaign
membership, and two of those eleven —
`screening.B.s616738081` and `screening.d005dfb27e7bb5adc46eb988ae79e2a9.s616738081`
— are completed, validly scored, and read by nothing. Staging them would have
moved ~4.8 GiB instead of 2.22 GiB and cost about 2.6 more hours of wall clock
during which the launch-bound sweep cannot run, because the simulator moves the
repository's gitignored artifacts aside and a live staging job writes into them.

That is P8.4 one layer further down than the continuation: the rule was applied
to what a *pod* restores, and the *staging tool* still planned from membership.
Filtering the root rather than patching the tool keeps the index truthful — it
lists the nine probes that are actually on the volume and omits the two that
are not.

Stopping mid-transfer was safe because the tool's teardown is in a `finally`,
not an `except Exception`: `KeyboardInterrupt` is not an `Exception`, so an
`except Exception` teardown would have leaked a billing pod. That was checked
before the signal was sent, and the provider then reported zero pods.

`$0.0006` of the run before it was spent on a draw the load backstop refused —
a host at 218.8 per vCPU against its 100.0 limit.
