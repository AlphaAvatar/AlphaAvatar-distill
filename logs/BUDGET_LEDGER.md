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

AutoInit micro-preflight attempt 3 (L40S, 170.9 min)         $  2.8200
    Stages 0-2 passed; both permanent controls trained and
    strictly verified, then DELETED unfetched by a launcher
    condition gated on total success. Stage 3 generation
    failed and its cause was lost with the pod, because no
    preflight artifact spec had ever been loadable.

AutoInit micro-preflight attempt 4 (L40S, 217.9 min)         $  3.6000
    Stages 0-2 passed; BOTH permanent controls trained,
    strictly verified and RETRIEVED (5.51 + 5.50 GiB, hashes
    re-verified locally). Stage 3 blocked on tool rendering
    under transformers 5.15. ~$0.41 of it was a cold first
    draw, abandoned and deleted by the launcher.

AutoInit characterization continuation, attempt 1 (L40S,      $  0.6312
    38.25 min). PRODUCED NOTHING. 29 min of it was a cold
    host, abandoned and deleted by the launcher; the redraw
    then failed the pod's blocking test gate on seven tests
    that read `recovery_search_v1`, which the v2 migration
    stopped staging. Pod deleted, provider-confirmed gone.
    Both causes fixed and locked by tests before relaunch.

AutoInit characterization continuation, attempt 2 (L40S,      $  0.6367
    38.59 min). PRODUCED NOTHING. All THREE host draws tripped
    HOST_COLD in the uv-sync window; the launcher abandoned and
    deleted each, then aborted. Pod deleted, provider-confirmed
    gone. The test-gate defect that killed attempt 1 was fixed
    and verified under the pod simulator before this launch, so
    this failure is a different, infrastructure one.

AutoInit characterization continuation, attempt 3 (L40S,      $  0.0700
    4.3 min). PRODUCED NOTHING, but failed FAST and loudly:
    `uv sync --frozen` installs from the source recorded in the
    lock, and torch's is the pytorch registry, which
    `--find-links` does not override. The offline gate caught it
    in 4 minutes instead of a 28-minute burn.

AutoInit Stage-3 continuation, attempt 8 COMPLETE (L40S, 41.3 min) $  0.6816
AutoInit characterization continuation, attempt 7 (L40S, 27.0 min) $  0.4500
AutoInit characterization continuation, attempt 6 (L40S, 8.0 min)  $  0.1324
AutoInit characterization continuation, attempt 5 (L40S, 8.3 min)  $  0.1369
AutoInit characterization continuation, attempt 4 (L40S,      $  1.3672
    82.9 min). The train env installed OFFLINE IN 11 SECONDS —
    the fix works on hardware. Then `pip install vllm`, still
    unpinned and still going to PyPI, hung 76 min on the same
    host that had failed three cold draws. Torn down manually
    once the outcome was determined; provider-confirmed gone.

ACTUAL CUMULATIVE SPEND                                      $191.5462
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

Within that project cap, the characterization continuation carries its own bound:

```
$1.75   continuation cap as handed off (expected $0.90)
$2.30   RAISED 2026-08-14 with maintainer approval, after attempt 1 spent
        $0.6312 and produced nothing. Covers that sunk cost plus one full
        attempt at its $1.6352 hard threshold. Expected $1.97.
        Raising the cap does NOT loosen a session: `make_plan` still prices
        one run at soft $1.4702 / hard $1.6352.
```

## Protocol deviations and overruns

| what | figure | recorded |
| --- | --- | --- |
| E6b exceeded the $149.03 cap | +$0.56 | `EXPERIMENTS.md` §29 |
| E8b-S1 exceeded its $3.25 session plan | $5.21 actual | §38 — the DP step-0 probe ran 200 min against a 20 min estimate |
| E8b-S2 spent $11.06 on six attempts, five of them infrastructure | — | §39–§41 |
| DC's step-0 probe deferred, never run | — | §38, declared |
| the 20-step memory gate was falsified by the real run | — | §41–§42 |
| continuation attempt 1 bought nothing: cold host + a test gate reading an unstaged battery | $0.6312 | `decisions.md` 2026-08-14, `STATE.md` |
| continuation attempt 2 bought nothing: three consecutive cold hosts, all in the uv-sync window | $0.6367 | `decisions.md` 2026-08-14 |
| continuation attempt 3 bought nothing, but failed in 4 min | $0.0700 | `uv sync` cannot install a registry-pinned wheel offline |
| continuation attempt 4: train env offline in 11 s, then `pip install vllm` hung 76 min | $1.3672 | the exposure named in the offline commit |
| continuation attempt 5: every offline fix worked on hardware; died on the LAST line of setup, a stale binding to the micro-preflight authorization | $0.1369 | `autoinit_continuation_attempt5/` |
| continuation attempt 6: the session-scoped authorization gate PASSED on the pod; setup finished with SETUP_RC=0 and the launcher misread it as setup_failed, because the shared setup wrote markers to the preflight's status filename | $0.1324 | `autoinit_continuation_attempt6/` |
| continuation attempt 7: **the driver ran.** Stages 0/1/2 passed; Stage 3 characterized sa and failed on sb, whose checkpoint has no tokenizer files | $0.4500 | `autoinit_continuation_attempt7/` |
| continuation attempt 8: **COMPLETE.** ALL_DONE, both controls characterized, thresholds materialized | $0.6816 | `autoinit_stage3_complete/` |
| **continuation total: $4.1060 spent across eight attempts; Stage 3 COMPLETE** | $4.1060 | arithmetic corrected: the earlier $1.2712 line double-counted a rounded print; the per-attempt entries were always right |

## Current position

```
authorized cumulative cap                                    $211.07
actual cumulative spend                                      $191.5462
unused authorization remaining                               $ 19.5238
continuation spent across eight attempts                     $  4.1060
    0.6312 + 0.6367 + 0.0700 + 1.3672 + 0.1369 + 0.1324 + 0.4500
    + 0.6816.  STAGE 3 IS COMPLETE: attempt 8 returned ALL_DONE,
    characterized both controls and materialized the thresholds.
GRANTED cumulative continuation authorization                $  4.54
    expected $4.23. BOTH issued artifacts are now CONSUMED, each
    having been granted for ONE launcher invocation:
      e4854818…  attempt 5, INCOMPLETE, $0.1369
      f21b4038…  attempt 6, INCOMPLETE, $0.1324

    expected $4.10, granted by the maintainer 2026-08-15.
    = $2.7051 spent + one newly-priced hard attempt ($1.6896).
    PER-LAUNCH HARD LIMIT                                    $  1.6896
    Named by the maintainer and now ENFORCED in code
    (`SpendAuthorization.per_launch_hard_usd`, checked in
    `make_plan` before a pod can be created). Previously only
    the plan's own arithmetic self-limited, so the cumulative
    figure was reachable by a single run.
    The re-priced plan is expected 84.0 min / $1.3860, soft
    $1.5246, hard $1.6896 — the limit binds exactly.
    The earlier artifact `759eaf8c…` asserted these amounts
    BEFORE the approval existed. It is void and must not be
    reused; no pod was created under it.
micro-preflight spent of its $8.60 authorization             $  6.7369
micro-preflight remaining under that authorization           $  1.8631
    LESS THAN A SESSION COSTS (~$3.2). Nothing further is
    launched without a new increment.
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
