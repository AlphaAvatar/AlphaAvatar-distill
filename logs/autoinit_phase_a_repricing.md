# Phase A repricing from measured Stage-1/Stage-2 values

Supersedes the cost table in
[`autoinit_pilot_proposal.md`](autoinit_pilot_proposal.md) §5, whose ranges came
from an unmeasured statistics-pass split and a **stated, not measured** 1.20
overhead factor. Both are now measured. L40S at $0.99/h throughout.

## The two `gpu_fraction` measurements: both valid, both kept

They are **not** a correction of one another. Same script, same frozen mixture
(59,763 tokens, 892 seq_len, 66 sequences, 3 repeats), same device class, same
image tag and driver (`@580.126.09`), identical peak memory (13.98 GiB) — two
different physical hosts.

| | attempt 3 (`6xrudht3i7wc62`) | attempt 4 (`hwyay7b9df9e77`) | pooled |
| --- | ---: | ---: | ---: |
| GPU forward, mean | 4.657 s | 4.153 s | 4.405 s |
| CPU float64 accumulate, mean | 3.645 s | 3.868 s | 3.757 s |
| total per pass | 8.303 s | 8.021 s | 8.162 s |
| **`gpu_fraction`** | **0.5609** | **0.5177** | **0.5397** |
| s / 1k tokens | 0.1389 | 0.1342 | 0.1366 |

The *split* varies by 4.3 points across hosts while the *total* varies by 3.5%.
The faster-GPU host spent proportionally more time on the CPU term, which is the
behaviour the measurement was commissioned to expose: **the CPU term does not
shrink with a faster GPU.**

**For repricing, use the total (8.02–8.30 s per pass, pooled 8.16 s), not the
fraction.** The fraction is reported as a **range, 0.5177–0.5609**, and is what
answers the design question "would a faster accelerator help" — answer: at most
about half of this pass, and the observed spread already spans much of the
plausible gain. Neither figure replaces the other and neither is an average of
record; both are in
[`autoinit_preflight_run4/`](autoinit_preflight_run4/preflight_evidence.json) and
attempt 3's session store.

## Measured inputs now available

| quantity | old basis | measured | source |
| --- | --- | --- | --- |
| 0.86M recovery probe, end to end | 1023 × 4.15 s × 1.20 = 84.9 min | **61.55 min** (61.6 / 61.5, two arms) | attempt 4 Stage 2 |
| training step time | 4.15 s (E6b) | **3.15 s** in-loop | attempt 4 train logs |
| activation-statistics pass | unmeasured, drove a 3.6× range | **8.02–8.30 s** per 59,763-token mixture | attempts 3 and 4 |
| state evaluation on `state_eval_v1` | unmeasured | **≈2.6 min** (10 evaluations inside a 30.2 min Stage 1) | attempts 3 and 4 |
| peak GPU resident, widest operator | 14.3 GiB derived | **13.98 GiB** | attempts 3 and 4 |
| battery evaluation of one checkpoint | $0.236 (E6, different battery) | **still unmeasured** — Stage 3 has never completed | — |
| operator build (SVD/PCA/selection) compute | inside the old range | **still unmeasured** | — |

The 1.20 overhead factor is retired: the 61.55 min figure is wall clock including
periodic evaluation and checkpointing, measured twice on the arms this prices.

## Repriced

| item | expected | hard |
| --- | ---: | ---: |
| Phase A search (beam 6, warmup 1, 1 profile, 39–56 states) | $1.76 | $3.00 |
| Recovery rungs 1+2 (9 probes) | $11.26 | $11.26 |
| Conditional third seed (3 probes) | $0.00 | $3.75 |
| Setup / redraw reserve | $0.00 | $3.00 |
| **total** | **$13.02** | **$21.01** |

Probe price is now 61.55 min × $0.99/h = $1.0156, plus $0.236 per battery
evaluation carried unchanged from E6 **and still unmeasured for this battery**.
Search cost is 39–56 states × (≈2.6 min evaluation + ≈8 s statistics) with the
operator build unmeasured and covered by the gap to the hard column. The reserve
stays at $3.00: three of this week's four sessions hit an infrastructure event
(a cold draw cost $0.41 by itself), and setup has varied 30× historically.

Against the old **$17.00 / $26.21**, this is **$13.02 / $21.01** — the reduction
is almost entirely the probe, which is 27% cheaper than priced because it runs at
3.15 s/step rather than 4.15.

## Budget position

```
authorized cumulative cap                       $211.07
actual cumulative spend                         $187.4402
unused                                          $ 23.6298
Phase A hard backstop, repriced                 $ 21.01     fits, margin $2.62
Stage-3 completion of the existing controls     not included below; see the plan
```

**The margin is thin and Phase A is not authorized.** A Phase-A authorization
should be decided against this table, not against the superseded one, and should
account separately for finishing Stage 3 on the two existing controls.

## Finishing Stage 3 on the existing controls: what it costs

The controls are on the dev box (`/home/ecs-user/aad-artifacts/autoinit/`,
2.3 GB of weights each). They must reach a pod, and the dev-box uplink is
0.72 MB/s, so the transfer is ~53 min **per checkpoint** — free if it happens to
the relay before a pod exists, paid pod time if it happens while one is running.

    stage the two `model/` directories to the relay   $0.00, ~1.8 h of uplink
    pod: setup                                        ~$0.11 warm, $0.52 observed
    pod: pull both checkpoints from the relay         ~$0.05
    pod: Stage-0 re-attestation of generation only    ~$0.05
    pod: two generation waves + scoring               ~$0.35, UNMEASURED
    reserve for one redraw                            ~$0.50
    ------------------------------------------------------------------
    expected ~$0.6, hard ~$1.6

This is the only spend that should be considered until the tool-rendering
migration question in
[`autoinit_tool_rendering_migration.md`](autoinit_tool_rendering_migration.md) is
decided, because generation cannot run before it is.


## Continuation session: repriced for the WHOLE session

The `$0.6 expected / $1.6 hard` sketch covered the characterization calls and
little else. Priced item by item, from this project's own measurements:

| item | expected | hard | basis |
| --- | ---: | ---: | --- |
| pod setup | 6.2 min | 6.2 min | measured twice (5.6, 6.2) |
| cold-draw redraw reserve | 0.0 | 25.0 min | observed once; abandoned and deleted correctly |
| stage 0, import verification | 1.0 | 2.0 min | CPU, local files |
| stage 1, attestation + engine probe | 2.0 | 3.0 min | measured (Stage 0 took 2.0 min) |
| stage 2, v2 tool + RAG smoke | 4.0 | 6.0 min | measured (~4 min for two sets) |
| stage 3, characterize both controls | 36.0 | 50.0 min | **UNMEASURED**, priced at the old 18 min/control |
| collection + verification | 3.5 | 6.0 min | measured (3.5 min in run 4) |
| teardown + provider check | 0.5 | 1.0 min | measured (0.4 min) |
| **total** | **53.2 min → $0.88** | **99.2 min → $1.64** | at $0.99/h |

**So `$0.6` was too low — the whole session is `$0.88` expected — while `$1.60`
happens to be about right at `$1.64`.** Proposed authorization: **$0.90 expected
/ $1.75 hard**, which keeps a little room above the priced hard threshold without
inviting drift.

The dominant term is the one thing nobody has measured: characterization of one
control on this battery. Everything else in the table is a measurement from the
last four sessions. That is exactly what the session is for, and it is why Phase A
must be repriced again afterwards rather than now.

### Effect on the project cap

```
project cap                                      $211.07
spent                                            $187.4402
remaining                                        $ 23.6298
continuation at its hard bound                   $  1.75
                                                 ---------
remaining after a worst-case continuation        $ 21.8798
Phase A hard bound, provisional                  $ 21.01
                                                 ---------
cushion                                          $  0.87
```

**That cushion is too thin to authorize Phase A**, which is the same conclusion
as before and is unchanged by this repricing. Phase A stays unauthorized until
the continuation returns a measured battery-evaluation cost and the bound is
recomputed from it.

---

## 2026-08-15 — repriced again, from the completed Stage 3

The one line that said **"still unmeasured — Stage 3 has never completed"** is now
measured. Stage 3 completed on 2026-08-15 (`ALL_DONE`, $0.6816, 41.3 min):

| item | previously | measured 2026-08-15 |
| --- | --- | --- |
| battery evaluation of one checkpoint | $0.236, carried from E6 on a different battery | **8.43 min marginal / 9.82 min conservative = $0.1391 / $0.1621** |
| whole Stage 3, two controls | 48 min allowed (24/control) | **19.65 min**, both batteries + both scorings + threshold materialization |
| stage 0 strict import, two controls | — | 0.06 min |
| stage 1 attestation | — | 2.10 min |
| stage 2 real v2 tool+RAG smoke | — | 1.48 min |

The conservative figure — half of the whole stage, so it carries engine load,
scoring and materialization — is used below. The marginal 8.43 min is what one
additional battery actually costs once the engine is warm.

| item | expected | hard |
| --- | ---: | ---: |
| Phase A search (unchanged) | $1.76 | $3.00 |
| Recovery rungs 1+2 (9 probes @ $1.1777) | $10.60 | $10.60 |
| Conditional third seed (3 probes) | $0.00 | $3.53 |
| Setup / redraw reserve (unchanged) | $0.00 | $3.00 |
| **total** | **$12.36** | **$20.13** |

Probe price is `61.55 min × $0.99/h = $1.0156` plus `$0.1621` per battery, so
`$1.1777` per probe against `$1.2516` before. Every input is now measured except
the operator build, which remains covered by the gap to the hard column.

## It still does not fit

```
authorized cumulative cap                       $211.07
actual cumulative spend                         $191.5462
remaining                                       $ 19.5238
Phase A expected, repriced                      $ 12.36     fits, margin $7.16
Phase A hard, repriced                          $ 20.13     DOES NOT FIT, short $0.61
```

Repricing from measurement moved the hard bound $21.02 → $20.13 and did not close
the gap. **Phase A cannot be authorized against the current cap** without one of:

* raising the project cap by at least $0.61 (a cap change is a maintainer
  decision, and $1.00 would leave a real margin);
* dropping the conditional third seed from the hard bound, which is $3.53 and is
  conditional by construction — that would give hard $16.60, fitting with $2.92
  margin, but it removes the seed-disagreement escape hatch;
* reducing the $3.00 setup/redraw reserve, which this week's history argues
  against: four of eight continuation attempts hit an infrastructure event.

Recommendation: the second option is the cheapest honest one, but it changes what
Phase A can do when its two seeds disagree, so it is a design decision rather than
an accounting one. Nothing is decided here.

---

## 2026-08-15 — SUPERSEDED by the launcher's own plan

**Do not price a Phase-A authorization from the table above.** Everything in this
document is search-and-probes only; it never priced the session around them. The
Phase-A launcher now exists (`scripts/pod/autoinit_phase_a_launch.py`), and its
`make_plan` is what `plan_session` computes before a pod can be created — so that
is the figure an authorization must be granted against.

| | this document | `PhaseA.make_plan` |
| --- | ---: | ---: |
| search | $1.76 | 180 min allowance |
| 9 probes (rungs 1 + 2) | $10.60 | priced through the step-time model |
| conditional seed-sc rung | $3.53 (hard only) | priced as headroom |
| setup, Stage-0 attestation, selection, manifest, sync, transfer | — | 48 min |
| 10% contingency + 20 min artifact-recovery reserve | — | included |
| **expected** | **$12.36** | **$17.8933** |
| expected if it resolves after two seeds | — | **$14.3604** |
| soft | — | $19.6826 |
| **hard** | **$20.13** | **$20.0126** |

The hard bounds nearly agree by coincidence: this document added a flat $3.00
setup/redraw reserve, while the plan derives session overhead and contingency
from measured phase times. The *expected* figures differ by $2.00 because the
session overhead here was simply absent.

Two things this document got right and the plan keeps: the probe price comes from
the measured 61.55 min end-to-end (not the 3.15 s/step in-loop figure, which
would under-book each probe by ~8 min), and the battery uses the **conservative**
9.82 min rather than the 8.43 min marginal.

Against the cap raised to `$213.00` on 2026-08-15, remaining is `$21.4538` and
the hard bound fits with `$1.4412` of margin. See
[`BUDGET_LEDGER.md`](BUDGET_LEDGER.md).

**Correction, later on 2026-08-15.** The `$21.4538 remaining / $1.4412 margin`
above was true when written. Two attempts have since failed closed without
training anything — $0.1075 (setup test gate) and $0.4665 (driver stage 0) — so
cumulative spend is **$192.1202** and remaining is **$20.8798**, leaving
**$0.8672** against the unchanged $20.0126 hard bound. That margin is not a retry
reserve.
