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
Phase A attempt 1 (L40S, 6.5 min) SETUP GATE FAILED, NOTHING RAN $  0.1075
Phase A attempt 2 (L40S, 28.3 min) STAGE-0 GATE FAILED, NOTHING RAN $  0.4665
Phase A attempt 3 (L40S, 12.8 min) STAGE-0 GATE FAILED, NOTHING RAN $  0.2103
Phase A attempt 4 (L40S, 12.4 min) STAGE-3 BINDING REFUSED       $  0.2052
Phase A attempt 5 (L40S, 38.9 min) STAGE 0 PASSED, STAGE 1 FAILED $  0.6426
    FIRST TIME STAGE 0 PASSED. The v2 comparability migration held on
    hardware: attestation written, protocol 250f72ef, comparable
    identity 70a26e0b. Caveat: this pod drew driver 580.159.03, the
    SAME as Stage 3, so v1 would also have passed here -- v2 was not
    the deciding factor on this particular host.
    Stage 1 then failed on its first real execution:
      CalibrationError: calib.domain_balanced@v1:
      artifacts/stage1/e8_calibration_v1/items.jsonl is missing
    phase_a_search.py:124 calls DOMAIN_BALANCED_V1.resolve(), which
    reads that file. It is NOT in the launcher's LOCAL_ASSETS, NOT in
    the relay precheck's `need` list, and is INCOMPLETE on the dev box
    (only docs.jsonl + general_disjointness.json). It IS on the relay
    at e8_inputs_20260810/calibration_v1/. A required Phase-A input
    that nothing stages and no $0 gate checks for.
    38.9 min includes a redrawn host (vvsohv60cuuokx -> s797g6xphdibms).
    Pod deleted, provider confirms gone; nothing trained.
    NOT a code defect. Stage 0's real body ran end to end -- the fix
    verified at $0 held -- and the NEW Stage-3 protocol binding then
    refused, as designed: the pod's evaluation protocol hashed to
    17218f7c, not the pinned 250f72ef under which the equivalence
    interval and feasibility floor were materialized.
    The cause is ONE field. Every observation governing generation
    semantics matched exactly (vLLM 0.27.1, transformers 5.15.0, torch
    2.13.0+cu130, bfloat16, gpu_mem 0.9, max_num_seqs 256,
    max_num_batched_tokens 8192, enforce_eager False, tokenizer sha,
    chat-template sha, resolved_context 8192, context_source,
    stop_token_ids). Only runtime_digest differed, and within the
    runtime only image_digest, whose suffix is the HOST NVIDIA DRIVER
    version appended by read_image_digest:
        stage 3   ...ubuntu2404@580.159.03
        attempt 4 ...ubuntu2404@580.126.09
    Pod deleted, provider confirms gone, no stage passed, nothing
    trained.
    Setup passed in 5.2 min (warm image). The driver detached and died
    ~2 min into stage 0, AFTER the engine probe ran: the driver called
    declared_generation_protocol(engine_probe_json) but that function
    takes no positional arguments. The continuation's working form is
    declared_generation_protocol().materialized(...). A second defect
    in the same never-executed stage 0, of a class the attempt-2
    regression did not cover: that one checked argv for external
    SCRIPTS, this is an in-process CALL. Failed closed, pod deleted,
    provider confirms gone, no stage passed, nothing trained.
    Setup PASSED this time: the SESSION_KIND test fix held, so the
    pod's blocking CPU suite cleared and the driver detached. It then
    died ONE SECOND later in stage 0. The driver's engine-probe call
    omitted `--model`, which autoinit_engine_probe.py marks
    required=True, so argparse exited rc=2 before vLLM ever loaded.
    A defect in the Phase-A driver, not in any gate: the stage-0 gate
    refused correctly and the session failed closed. The rehearsal
    SCRIPTED stage 0 rather than building its argv, so that line had
    never executed anywhere. Pod deleted, provider confirms gone;
    no stage passed, nothing was trained, no permanent artifact touched.
    The pod's blocking CPU test suite failed one test:
    test_setup_verifies_THIS_sessions_authorization_and_fails_closed.
    The setup gate itself behaved CORRECTLY. The launcher exports
    SESSION_KIND=phase_a into setup.sh, which exports it to the test
    gate; that test extracts setup's authorization block and ran its
    phase_a branch against the CONTINUATION's artifact, which is
    correctly refused with exit 98. The test controlled two session
    variables and not the third. It passed on the dev box and under the
    pod simulator, and could only fail on a real Phase-A pod.
    Failed CLOSED: aborted after draw 1, pod deleted, provider confirms
    gone, driver never started, no stage ran, nothing trained.
AutoInit characterization continuation, attempt 7 (L40S, 27.0 min) $  0.4500
AutoInit characterization continuation, attempt 6 (L40S, 8.0 min)  $  0.1324
AutoInit characterization continuation, attempt 5 (L40S, 8.3 min)  $  0.1369
AutoInit characterization continuation, attempt 4 (L40S,      $  1.3672
    82.9 min). The train env installed OFFLINE IN 11 SECONDS —
    the fix works on hardware. Then `pip install vllm`, still
    unpinned and still going to PyPI, hung 76 min on the same
    host that had failed three cold draws. Torn down manually
    once the outcome was determined; provider-confirmed gone.

ACTUAL CUMULATIVE SPEND                                      $193.1783
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
$213.00   +$1.93 for Phase A (2026-08-15). The maintainer approved raising the
          cap "e.g. $213-214" after being shown that Phase A did not fit;
          $213.00 was selected from that range as the conservative end and is
          the figure recorded here. Nothing was spent against it.
$219.00   +$2.00 (2026-08-17). APPROVED. Funds AT MOST two things: one
          infrastructure-only Stage-1 GPU device canary with hard cap
          $1.0197, and -- ONLY if that canary passes and tears down
          cleanly -- one complete Phase-A Attempt 8 under the existing
          $23.0484 per-launch hard bound. It does NOT authorize a canary
          retry, Attempt 9, or any change to the frozen science.
$217.00   +$4.00 for Phase A attempt 6 (2026-08-17). APPROVED. The maintainer's
          words: "My recommended new cumulative project cap is $217.00",
          followed by "Record this approval in the ledger and machine-readable
          state ... and issue one fresh PhaseAAuthorization." The raise covers
          the repriced hard bound after the reference-cache fallback audit and
          the beam-6 pricing correction: $23.048325 against $19.8217 remaining
          under $213.00. Against $193.1783 spent it leaves $23.8217, a margin of
          $0.7734 over one complete attempt. FUNDS ONE ATTEMPT: "This funds one
          attempt only and does not imply Attempt 7."
```

### Why the cap moved, and by how much

Phase A did not fit under `$211.07`. Repriced from the measured Stage-3 battery
it was `$12.36 expected / $20.13 hard` against `$19.5238` remaining — short by
`$0.61`. The alternatives were dropping the conditional third seed (a design
change, and Stage 3 showed real seed disagreement) or cutting the `$3.00`
infrastructure reserve (four of eight continuation attempts hit an
infrastructure event). Raising the cap preserves the frozen design.

**The launcher's own model prices it higher than the repricing document did**,
and the launcher's figure is the one that binds, because it is what
`plan_session` computes before a pod can exist:

```
                                    repricing doc     launcher make_plan
search                                    $1.76        180 min allowance
9 probes (rung 1 + rung 2)               $10.60        priced per step
conditional seed-sc rung (3 probes)       $3.53        priced as headroom
setup, attestation, selection,
  manifest, sync, transfer                  --         48 min
contingency 10% + 20 min reserve            --         included
-------------------------------------------------------------------------
expected                                 $12.36        $17.8933
  of which conditional tie-break              --        $3.5328
  expected without the tie-break              --       $14.3604
soft                                        --         $19.6826
hard                                     $20.13        $20.0126
```

The document priced search and probes; it did not price the session around
them. The `$14.3604` figure is the honest expected cost of a Phase A that
resolves after two seeds.

The conditional rung is **priced as headroom, not as an expectation**: if it were
left out, a legitimately triggered seed-sc rung would be killed by the watchdog
mid-probe. An unused leash costs nothing, because the pod is torn down on
completion rather than at the threshold.

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
| continuation attempt 5: every offline fix worked on hardware; died on the LAST line of setup, a stale binding to the micro-preflight authorization | $0.1369 | `autoinit_continuation_attempts/attempt5/` |
| continuation attempt 6: the session-scoped authorization gate PASSED on the pod; setup finished with SETUP_RC=0 and the launcher misread it as setup_failed, because the shared setup wrote markers to the preflight's status filename | $0.1324 | `autoinit_continuation_attempts/attempt6/` |
| continuation attempt 7: **the driver ran.** Stages 0/1/2 passed; Stage 3 characterized sa and failed on sb, whose checkpoint has no tokenizer files | $0.4500 | `autoinit_continuation_attempts/attempt7/` |
| continuation attempt 8: **COMPLETE.** ALL_DONE, both controls characterized, thresholds materialized | $0.6816 | `autoinit_stage3_complete/` |
| **continuation total: $4.1060 spent across eight attempts; Stage 3 COMPLETE** | $4.1060 | arithmetic corrected: the earlier $1.2712 line double-counted a rounded print; the per-attempt entries were always right |

## Current position

```
authorized cumulative cap                                    $219.00
    RAISED AND APPROVED 2026-08-17. See the caps list above.
    Funds the $1.0197 canary and, conditionally, one $23.0484
    Attempt 8. Not a retry reserve; no Attempt 9.
actual cumulative spend                                      $193.9893
    = $193.1783 + $0.3552 attempt 6 + $0.3955 attempt 7
      + $0.0603 device canary attempt 1 (launcher error; the
      canary itself never ran)
unused authorization remaining                               $ 25.0107
    = $219.00 - $193.9893. Committed against it:
      device canary hard                                       $  1.0197
        SPENT $0.0603 on attempt 1, which died in the one-use
        wrapper before the canary ran. NOT a canary result and
        NOT a retry authorization: the grant covered one
        launcher invocation and is consumed.
      Attempt 8 hard, CONDITIONAL on the canary passing        $ 23.0484
        The canary has not run, so Attempt 8 is not authorized.
      remaining after a canary retry and Attempt 8             $  0.9426
    Attempt 7's authorization
    `autoinit.phase_a.2026-08-17T0850Z` covered exactly ONE
    launcher invocation, is SPENT, and its lineage gate refuses
    every later commit by construction. No attempt 8 is
    authorized, funded, prepared or implied.
--- the attempt-7 grant, now spent -------------------------------------------
Phase A attempt-7 per-launch hard authorization              $ 23.0484
cumulative-cap margin at the full hard bound                 $  0.4181
    ATTEMPT 7 WAS AUTHORIZED (2026-08-17), under the EXISTING
    $217.00 cap. The cap is NOT raised. The authorization covers
    exactly one Phase-A launcher invocation and does not authorize
    attempt 8, any increase above $217.00, or any change to the
    frozen search, recovery, seeds, thresholds,
    runtime-comparability rules, or the science/session plans.
    The attempt-6 authorization `autoinit.phase_a.2026-08-16T1912Z`
    is SPENT; its lineage gate refuses every later commit by
    construction, so attempt 7 runs under a freshly issued one.
paid compute currently running                                NONE
    attempt 6 pod wgm2tamw8nu9f5 and attempt 7 pod n2kfqhyoya4zzj
    both deleted; provider confirms gone for both

    SUPERSEDED Phase-A pricing, kept because a threshold that
    moved silently is how E6b overran:
      $20.0126 hard  no fallback reserve, no beam-6 correction
      $22.4508 hard  fallback reserve placed AFTER the soft stop.
                     Wrong: the fallback is consumed inside
                     stage 1, and `afford()` gates on the SOFT
                     stop, so it would have truncated the
                     conditional seed-sc rung to pay for an
                     infrastructure risk.
      $23.0483 hard  CURRENT. Both reserves before the soft stop:
                     +147.7683 min reference-cache fallback and
                     +36.2158 min beam-6 search correction, on a
                     $17.8933 expected and a $22.7183 soft stop,
                     plus the 20-minute artifact-recovery reserve.
                     Authorization cap is the 4-dp ceiling,
                     $23.0484. Derivation in
                     `autoinit_phase_a_fallback_audit.json`.
GRANTED Phase-A authorization, 2026-08-15T12:32:08Z          $ 20.0126
    `autoinit.phase_a.2026-08-15T1232Z`, sha256 14360ef4…
    expected $17.8933 ($14.3604 if it resolves after two
    seeds); per-launch hard equals the cap because this is
    ONE session. Bound to session plan 9377a2dc…, science
    plan 02be33b9…, harness digest ea2f360b… and commit
    9b05a058…. Stages 0-5. Nothing spent against it.

    **ISSUED IS NOT LAUNCHED.** The artifact authorizes the
    spend and the stages; the maintainer's instruction that
    requested it also says "Do not launch Phase A yet", so
    the run waits for a separate explicit go.
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

## Provenance note on the $213.00 cap (recorded post-run, 2026-08-15)

The Phase-A authorization's `granted_by` field says the cap was raised on "the
maintainer's own words 'Raise it further, e.g. $213-214'". **That attribution is
imprecise and is corrected here rather than in the artifact.**

What actually happened: the assistant presented four budget options; that
sentence was the *option label the assistant authored*, and the maintainer
**selected** it. The approval is explicit and substantively valid — the
maintainer chose to raise the cap into $213–214 over three alternatives, and
$213.00 is the conservative end of that range. What the maintainer did not do is
type that sentence.

The artifact is **not** edited: `granted_by` is inside `authorization_sha256`,
and `phase_a.py` is inside the harness digest, so correcting the wording would
move both and force a rebuild and reissue for an attribution nuance that is not
an authorization defect. Recorded in prose, at the maintainer's direction.
