# 2026-09-23 — The frozen decision rule, exercised on the real confirmation rows

**`$0`. Nothing here is a result.** The third confirmation seed does not exist;
it is fabricated four ways below to reach the verdict code path and to show the
instrument moves. No figure in this document is a measurement of anything.

## Why

`behavioural_decision.confirm` has never run on this campaign's data. Stage D is
the last stage of a ~9-hour session, so a defect there costs the whole run —
and this project has three times found a real defect by running a scorer
against known-bad inputs before spending on it.

## What it needs, and what it refuses

Six probes, exactly the three frozen seeds, arms `{incumbent, treatment}`.
Alignment comes from `decision_inputs`, which refuses a duplicate prompt id, a
prompt set that differs between probes, a scorable count that is not 850 and a
total that is not 950.

```text
frozen confirmation seeds   1936324010, 1916380711, 1523147638   (registered order)
present on the dev box      1936324010, 1916380711               both arms
MISSING                     1523147638                           both arms  ← attempt6's work
bootstrap seed              834816710   (C1's default 816109261 is asserted different)
SESOI                       0.01        seed robustness: 2 of 3 positive required
n_prompts 950 · n_scorable 850 · 6 strata
```

## It runs, and it discriminates

| third seed, fabricated as | terminal_state | per-seed Δ correct | one-sided LCB |
| --- | --- | --- | --- |
| a copy of a real seed | `NO_GO` | −0.00353, −0.00353, −0.00353 | −0.01490 |
| treatment all correct | `INCONCLUSIVE` | +0.95882, −0.00353, −0.00353 | +0.30745 |
| treatment all wrong | `NO_GO` | −0.04118, −0.00353, −0.00353 | −0.02588 |
| treatment correct **and** incumbent wrong | `INCONCLUSIVE` | +1.0, −0.00353, −0.00353 | +0.32392 |

Two things follow, and they are different in kind.

**The instrument is live.** The verdict moves, the LCB swings from −0.0149 to
+0.324, and the code path completes. A rule that answered `NO_GO` to everything
would be indistinguishable from a broken one on the data that exists.

**GO is unreachable from here, and that is arithmetic rather than a prediction.**
Seed robustness needs two of three seeds positive. Both measured seeds are
negative at −0.00353, so no value of the third can reach two. Even a treatment
that answers every scorable prompt correctly while the incumbent answers none
returns `INCONCLUSIVE`, not `GO`.

**So what attempt6 decides is `NO_GO` versus `INCONCLUSIVE`**, and that turns on
the third seed's own delta and on where the bootstrap interval falls against the
SESOI. That is a real question with a real answer, and the frozen protocol is
what answers it — which is why the pair is still run rather than assumed.

## What this does not say

It says nothing about what the third seed will measure. The `−0.00353` repeated
in every row above is one real seed's delta reused as a stand-in, not a forecast;
the two measured seeds happen to agree to five decimals, which is a fact about
them and not a law. And a dry exercise on saved rows validates the verdict
stage only — it exercises no trainer, no battery and no scorer.
