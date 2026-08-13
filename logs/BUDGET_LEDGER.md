# Budget ledger — actual spend, reconciled 2026-08-12

**Do not infer available authorization from an old cap.** Caps were raised several times
and one was exceeded. What follows separates what was *spent* from what was *authorized*.

## Reconciliation method, and its limit

Per-session evidence files (`logs/*_evidence.json`) are written **per session name and
overwritten on reuse** — `e8b_s2_session_evidence.json` holds only the sixth S2 attempt,
and no evidence file survives for E1–E6. They are therefore *not* a complete provider
record and cannot reconstruct history on their own.

The authoritative figure is the cumulative total carried forward in the budget planners
and cross-checked against each experiment's recorded cost in
[`EXPERIMENTS.md`](EXPERIMENTS.md). Surviving evidence files agree with those
per-session figures where both exist (E7 $10.49, E8a $0.53 for pod A, E8b-S1 $4.07 at
polling end, E8b-S2 $7.21).

Where a launcher figure is a lower bound because a pod was terminated manually
afterwards, the higher manual figure is the one carried.

## Actual cumulative spend

```
E1  data-scaling matrix, 24 arms                            $ 47.6000
E2  0.86M diagnostics, phase 1                               (see §12)
E3  attention-restriction at 0.86M                           (see §20)
E4  P2 CE-heavy 0.86M -> 1.60M                               (see §21)
E5  teacher- vs student-prefix, 5 attempts                   $  9.7800
E6  E1 scale curve on the frozen battery                     $  2.3600
E6b P2 CE-heavy at 2.96M                                     $  7.6800
    diagnostics/hardening/canaries (§14-19, §30, §32-33)      $  9.0000 approx
E7  FineWeb-Edu KD                                           $ 10.4900
E8a contribution-guided depth search                         $  3.7253
--- carried-forward pre-E8b baseline ------------------------ $163.8833
E8b step-0 (S1)                                              $  5.2100
E8b S2 attempts 1-3, setup failures, nothing trained         $  3.1000
E8b S2 attempt 4, 20-step gate then OOM                      $  0.5500
E8b S2 attempt 5, OOM at step 110                            $  0.7500
E8b S2 attempt 6, DP-sa trained, DC-sa OOM at ~step 900      $  7.2100
--- E8b total ---------------------------------------------- $ 16.8200
AutoInit micro-preflight attempt 1 (L40S, 1.5 min)           $  0.0300
    setup aborted at the frozen-asset gate: it invoked
    /opt/train/bin/python before `uv sync` created it. Pod
    deleted by the launcher, provider confirmed gone. No
    stage ran; nothing was trained or measured.
AutoInit micro-preflight attempt 2 (L40S, 17.4 min)          $  0.2869
    setup passed; Stage 0 attested both protocol identities;
    Stage 1's evaluator-repeatability gate raised and the
    session stopped there. The permanent controls were NOT
    trained, which is the staging working as designed. Pod
    deleted, provider confirmed gone.

ACTUAL CUMULATIVE SPEND                                      $181.0202
```

The pre-E8b baseline `$163.8833` is the figure every E8/E8b planner was built on
(`scripts/training/plan_e8b_budget.py`, `ACTUAL_BASELINE_USD`). E8b's $16.82 is itemised
above from §38–§42.

## Authorized caps, in order

```
$149.03   EXCEEDED by $0.56 during E6b, and CLOSED. The overrun is recorded,
          not rewritten.
$150.41   temporary canary cap, spent and closed
$162.49   E8 baseline cap
$172.57   +$10.08 for E8 (2026-08-11)
$173.40   +$0.83 increment
$211.07   +$47.18 E8b hard backstop (2026-08-11), rounded up from $211.0633
```

## Protocol deviations and overruns

| what | figure | recorded |
| --- | --- | --- |
| E6b exceeded the $149.03 cap | +$0.56 | `EXPERIMENTS.md` §29 |
| E8b-S1 exceeded its $3.25 session plan | $5.21 actual | §38 — the DP step-0 probe ran 200 min against a 20 min estimate |
| E8b-S2 spent $11.06 on six attempts, five of them infrastructure | — | §39–§41 |
| DC's step-0 probe deferred, never run | — | §38, declared |
| the 20-step memory gate was falsified by the real run | — | §41–§42 |

## Current position

```
authorized cumulative cap                                    $211.07
actual cumulative spend                                      $181.0202
unused authorization remaining                               $ 30.0498
of which was earmarked for E8b completion                    $ 30.36
paid compute currently running                                NONE
```

**E8b is strategically terminated, so its earmark is released.** The $30.37 of unused
authorization is not committed to anything. No completion budget for E8b was requested
and none should be.

**No paid compute is running.** Verified by provider query: zero live pods.

## Standing rules

* Plan from **actual spend**, never from unused room under a previous authorization.
* Never silently shrink an experiment to fit a shortfall — report the exact figure and
  ask. `budget.plan_session` enforces this by raising with the number rather than
  trimming the run.
* `--authorized-usd` needs **4 decimal places**: `plan_session` compares unrounded, so
  `18.76` fails against a plan of `$18.760145`.
* Budget every session for **1–2 abandoned host draws**; the observed rate across E8 was
  nine abandoned draws in six launches.
