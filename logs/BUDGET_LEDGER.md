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
$234.00   +$3.00 (2026-08-21). APPROVED, in the maintainer's own words: "I
          approve the cumulative project cap increase from $231.00 to $234.00,
          but this is only the project ceiling. The Phase-A per-launch ceiling
          remains $23.0484, and no Attempt 12 launch is authorized yet."
          Arithmetic: $209.6842 spent, so one full hard-ceiling attempt reaches
          $232.7326, leaving $1.2674. It does NOT authorize an Attempt 13.
$231.00   +$12.00 (2026-08-20). APPROVED, in the maintainer's own words: "I
          approve raising the cumulative AlphaAvatar-distill project cap from
          $219.00 to $231.00." Explicitly scoped: "This is a cumulative project
          ceiling only. It does not change the existing Phase-A per-launch hard
          ceiling of $23.0484, does not authorize any subsequent attempt, and
          does not authorize spending outside the next explicitly approved
          session."
          The arithmetic it was granted against: $206.4741 spent, so one full
          hard-ceiling Phase-A attempt reaches $206.4741 + $23.0484 =
          $229.5225, leaving $1.4775 of margin under the new cap. It funds
          ONE Phase-A Attempt 11 and nothing else. Attempt 12 is NOT funded,
          authorized, prepared or implied by it.
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
authorized cumulative cap                                    $234.00
    RAISED AND APPROVED 2026-08-21, from $231.00 (itself raised from
    $219.00 on 2026-08-20). A CUMULATIVE PROJECT
    CEILING ONLY: it does not change the $23.0484 Phase-A per-launch
    hard ceiling, does not authorize any subsequent attempt, and does
    not authorize spend outside the next explicitly approved session.
    See the caps list above.
actual cumulative spend                                      $230.0350
    The header below drifted from its own itemization between attempts
    4 and 7 -- the per-attempt terms were always added, the total was
    not re-summed. Re-derived 2026-08-24; the terms sum to $230.0350
    exactly, and $234.00 - $230.0350 = $3.9650.
    = $194.0530 + $0.1900 attempt 8 + $0.3400 attempt 9
      + $11.4300 attempt 10 + $0.0700 measurement attempt 1
      + $0.1834 measurement attempt 2 + $0.2077 measurement attempt 3
      + $3.2101 Phase-A attempt 11 + $3.7872 Phase-A attempt 12
      + $0.0100 recovery continuation attempt 1
      + $0.2389 recovery continuation attempt 2
      + $0.2011 recovery continuation attempt 3
      + $0.4112 recovery continuation attempt 4
      + $1.3511 recovery continuation attempt 5
      + $1.4926 recovery continuation attempt 6
      + $12.8587 recovery continuation attempt 7.
remaining under the $234.00 cap                              $  3.9650
    FUNDS NO FURTHER PAID SESSION OF ANY KIND: a continuation ceiling
    is $16.7456 and a Phase-A attempt $23.0484.
    REMAINING BALANCE IS NOT AUTHORIZATION. Every grant issued, through
    attempt 7's, is spent.
recovery continuation attempt 7 (L40S, 779.3 min) PHASE A COMPLETE $ 12.8587
    AUTHORIZED 2026-08-23 as autoinit.recovery_continuation.2026-08-23T1314Z
    (sha256 d354557e) against
    logs/autoinit_recovery_continuation_attempt7_grant.json.
    Base 7e1d429, session commit d968b20, harness b824441c over 22 files.
    Ceiling $16.7456, of which $12.8587 was spent -- UNDER the $14.9233
    expected. Pod 3c1g6e01kdu1ya; provider confirms gone; watchdog ended
    pod_gone.
    ALL_DONE. Six stages, ELEVEN probes trained and scored: 6 in rung 1
    on seed sa, 3 in rung 2 on sb, 2 in the conditional rung 3 on sc.
    RESULT: unresolved_equivalence, winner None, tie_break_ran true.
    Pooled correct_overall: cca699c93f34 0.029412, 85bde4ded2c3 0.019608,
    control-qwen 0.008824, against a 0.011695 interval. The two searched
    leaves are tied with each other; cca699c9 IS separated from the
    canonical control and 85bde4de is not. No fourth seed follows.
    Comparability held: comparable_identity 70a26e0b, live == historical.
    Probe training was 61.0-61.1 min against 61.55 priced, so attempt 6's
    71.9 min was host variance.
    THE LAUNCHER THEN CRASHED AFTER COLLECTION on a fetch_products
    contract mismatch (AttributeError: 'str' object has no attribute
    'get'), mislabelling the session INCOMPLETE despite DRIVER_EXITED:0.
    Nothing was lost: 9 reports fetched, local_hash_problems [], archive
    extracted with all 11 probe trees, and the two retained finalists are
    initializations already preserved at 1.2 GiB each.
recovery continuation attempt 6 (L40S, 90.5 min) BATTERY REACHED $  1.4926
    AUTHORIZED 2026-08-23 as autoinit.recovery_continuation.2026-08-23T0944Z
    (sha256 b15aab3c) against
    logs/autoinit_recovery_continuation_attempt6_grant.json.
    Base 948b1e8, session commit 08670e5, harness 0dbf1272 over 22 files.
    Ceiling $16.7456, of which $1.4926 was spent. Pod ifp8feyil1gp7v;
    provider confirms gone; watchdog ended pod_gone after 91 ticks.
    THE ATTEMPT-5 REPAIR IS CONFIRMED: trained_model_dir() resolved the
    probe checkpoint with no FileNotFoundError and execution entered
    battery(), past the line that ended attempt 5. Setup $0.13 and TCP 22
    in 0.2 min, both the best recorded.
    FAILED in generation, 50 s after PROBE_TRAINED: uncapped_eval raised
    ValueError, tokenizer.chat_template is not set. Trainer.save_checkpoint
    writes weights and config only; battery() passes --model <that dir>
    with no --tokenizer, and that flag defaults to the checkpoint's own
    tokenizer. Every proven caller had pointed --model at CANONICAL_INIT,
    a full checkpoint.
    PROTOCOL WARNING: passing --tokenizer would fix the crash and break
    comparability. tokenizer_source is material under
    generation_runtime_comparability@v2 and Stage 0 attested it as "the
    evaluated checkpoint" with tokenizer_sha256 c1db93c8.
    The probe trained in 71.9 min against 61.55 priced (+16.6%) and was
    again lost with the pod -- the second time paid training was discarded.
recovery continuation attempt 5 (L40S, 81.9 min) PROBE TRAINED $  1.3511
    AUTHORIZED 2026-08-22 as autoinit.recovery_continuation.2026-08-22T1925Z
    (sha256 7f575d5f) against
    logs/autoinit_recovery_continuation_attempt5_grant.json.
    Base 4794193, session commit 63625e1, harness 95cf336d over 22 files.
    Ceiling $16.7456, of which $1.3511 was spent. Pod 9jxov5bjtiy2xu;
    provider confirms gone; watchdog ended pod_gone after 82 ticks.
    BOTH MEMORY REPAIRS VERIFIED ON HARDWARE. Against attempt 4's
    identical `before`: freed_allocated_bytes 8,101,709,824 vs 0,
    allocated after 0.008 GiB vs 7.55, free 43.87 GiB vs 36.32,
    live_retention false vs true. require_headroom passed on 43.87 against
    the 43.65 requirement -- the figure the repair predicted, to two
    decimals -- so the 41.65 GiB basis is very nearly exact rather than
    merely conservative.
    AND THE FIRST RECOVERY PROBE TRAINED: MARKER:PROBE_TRAINED for
    rung1.cca699c93f34.sa after 61.7 min against the 61.55 min the budget
    is priced from, confirming the pricing basis on hardware.
    FAILED at stage 2 AFTER that training, reading the probe's checkpoint:
    the driver reads out_dir/latest.txt and out_dir/<tag>/model, while the
    trainer writes out_dir/checkpoints/latest.txt and
    checkpoints/<tag>/model. train_stage3.py's resume path reads it
    correctly -- one writer, two consumers, one wrong, on a line only a
    completed 62-minute probe can reach. The probe's own artifacts were
    lost with the pod: the fetch spec collects finalists, and a
    trained-but-unscored probe is not one.
recovery continuation attempt 4 (L40S, 24.9 min) STAGES 0-1   $  0.4112
    AUTHORIZED 2026-08-22 as autoinit.recovery_continuation.2026-08-22T1454Z
    (sha256 874d54f6) against
    logs/autoinit_recovery_continuation_attempt4_grant.json.
    Base 38db4f2, session commit ef4353c, harness 162c09ed over 22 files --
    unchanged from attempt 3, since the portability repair touched files
    outside the set. Ceiling $16.7456, of which $0.4112 was spent. Pod
    k1mgu38q0y6sei; provider confirms gone; watchdog ended pod_gone after
    26 ticks.
    THE PORTABILITY REPAIR WORKED: setup passed for the first time in any
    continuation, and the driver started confirmed by descriptor probe.
    STAGE 0 AND STAGE 1 PASSED ON HARDWARE. Stage 0 attested interval
    0.011695, floor 0.3000, plan 02be33b9. Stage 1 imported the five
    Attempt-12 leaves in frozen selected order, RE-IDENTIFIED FROM THE POD
    BYTES against the Stage-1 artifact and shard digests (config
    567d32789ba6), and measured the canonical control once on state_eval@v1
    over 74022 positions (artifact_digest dc9500d3). No search ran.
    FAILED at stage 2, first rung-1 probe, on CUDA OOM in kd_forward_kl.
    Two compounding defects: release_to_subprocess reported
    freed_allocated_bytes=0 with live_retention=true (7.55 GiB still held
    after drop+del), and require_headroom demands RECOVERY_TRAINER_BYTES
    22 GiB + 2 margin = 24.00 against 36.32 free, so it passed -- while the
    probe actually used 36.30 GiB and OOM'd asking for 298 MiB more.
    Attempt 12's class: the gate written to stop it, whose message names
    attempt 12, was calibrated ~14 GiB below the trainer that ran.
recovery continuation attempt 3 (L40S, 12.2 min) NO STAGE RAN $  0.2011
    AUTHORIZED 2026-08-22 as autoinit.recovery_continuation.2026-08-22T1311Z
    (sha256 021d8830) against
    logs/autoinit_recovery_continuation_attempt3_grant.json.
    Base 7368568, session commit ad73e05, harness 162c09ed over 22 files.
    Ceiling $16.7456, of which $0.2011 was spent. Pod ku8vcn5mu8hp9i;
    provider confirms gone; watchdog ended pod_gone after 13 ticks.
    THE TRANSPORT CLOSURE WORKED: the pre-provider gate read 25 relay
    inputs (10 main + 15 transport) and 2 local assets, and the pod
    reached ASSETS_STAGED, ASSETS_READY and VLLM_READY -- so 5.5513 GiB
    of Stage-1 leaves were pulled from the transport repo and every
    declared digest verified. Attempt 2's failure class is closed.
    FAILED at the setup CPU test gate: publish_selected_leaves.verify()
    calls tempfile.mkdtemp(dir="/home/ecs-user/aad-scratch"), a dev-box
    path absent on a pod, so 5 tests raise FileNotFoundError. Reproduced
    at $0 in a mount namespace -- 5 failed, matching the pod exactly.
    Attempt 8's class one step out: a $0 test EXECUTING dev-box-only code
    rather than merely asserting dev-box state.
recovery continuation attempt 2 (L40S, 14.5 min) NO STAGE RAN $  0.2389
    AUTHORIZED 2026-08-21 as autoinit.recovery_continuation.2026-08-21T2004Z
    against logs/autoinit_recovery_continuation_attempt2_grant.json
    (sha256 a29dac6fd120). Ceiling $16.7456, of which $0.2389 was spent.
    Pod 7hthdteyc25xgx; provider confirms gone; watchdog ended pod_gone.
    THE RESILIENCE CLOSURE WORKED: the readiness poll that killed attempt 1
    succeeded, reaching TCP 22 at 3.7 min, and the image identity was
    confirmed. Every $0 gate passed and the bundle round-tripped.
    FAILED at LOCAL_ASSET staging: SessionRunner scps each declared local
    asset with subprocess.run(..., timeout=600), which RAISES. One stage-1
    leaf is 1.110 GiB = 1192 MB, so fitting the 600 s timeout needs
    1.99 MB/s sustained. This session's own bundle upload minutes earlier
    ran at 0.44 MB/s; the recorded dev-box uplink is 0.72 MB/s. One leaf
    therefore needs 28-45 min against a 10-minute timeout -- 3-4.5x over.
    It could not have succeeded, and would have repeated four more times.
    NO $0 GATE COULD SEE IT: selected_leaves_present_gate asks whether the
    leaves exist and verify LOCALLY, the structural staging test covers
    SESSION_RELAY_INPUTS (pulled) not LOCAL_ASSETS (pushed), and the pod
    simulator never scps. Declared, verified, and undeliverable.
    BOTH TRANSPORTS ARE CLOSED: scp needs 1.99 MB/s against <=0.72 MB/s;
    the relay has 1.60 GiB headroom against 5.55 GiB of leaves, which is
    why --stage-leaves-to-relay is off. The five leaves currently have no
    route to a pod. Not relaunched: the grant is spent and the arithmetic
    says a rerun fails identically.
recovery continuation attempt 1 (L40S, 0.7 min) NO STAGE RAN  $  0.0100
    AUTHORIZED 2026-08-21 as autoinit.recovery_continuation.2026-08-21T1642Z
    against logs/autoinit_recovery_continuation_grant.json. Ceiling
    $16.7456, of which $0.0100 was spent. EVERY pre-provider gate
    passed; pod dckc72mtoe9ijw was created and then deleted 27 s later
    when the launcher's readiness poll raised URLError (SSL
    UNEXPECTED_EOF). Provider confirms gone: True. Nothing billing.
    NOT a gate failure and NOT a scientific result: no stage ran, no
    leaf was touched, no science changed. The five preserved stage-1
    leaves are untouched inputs on the dev box.
    ROOT CAUSE, measured afterwards at $0: the RunPod GraphQL endpoint
    was returning transport errors at 5/20 = 25% (SSL EOF, ECONNRESET,
    RemoteDisconnected). `session_runner.wait_endpoint` calls
    `provider._gql` DIRECTLY, bypassing `provider.get`, which is the
    one documented as "Never raises. A watchdog that dies on a
    transient 502 is not a backstop." That poll makes up to 90 calls;
    at 25% loss it cannot survive. The 15-hour main poll DOES use
    `get()` and is not exposed. Relaunching unchanged would repeat.
    The single-issuance grant is spent, so the fix needs a new grant.
Phase A attempt 12 (L40S, 229.5 min) STAGE 1 PASSED, KEPT     $  3.7872
    AUTHORIZED 2026-08-20 as autoinit.phase_a.2026-08-20T1856Z
    against logs/autoinit_phase_a_attempt12_grant.json (sha256
    69e8150ea9d4). Ceiling $23.0484, of which $3.7872 was spent.
      stage 0   PASSED, attested, 1.9 min.
      stage 1   PASSED, 203.8 min, SEARCH_DONE:5.
      stage 2   FAILED on CUDA OOM -- NOT the tokenizer, which is
                closed: training reached a loss computation.
      stages 3-5  not reached.
    THE DURABILITY CLOSURE WORKED. All five stage-1 selected leaves
    transferred to /home/ecs-user/aad-artifacts/autoinit/phase_a/ with
    digest=MATCHED, AFTER the stage-2 failure -- the exact case that
    returned early before -- and required_products_secured recorded
    {ok: true, "all 5 stage-1 selected leaves verified off-pod"}. Each
    was re-verified independently afterwards from local bytes: 5/5
    artifact_digest and single_shard_sha256 match the stage-1 record,
    1.110 GiB each, tokenizer_sha256 None (still weight-only).
    Attempt 11 produced the SAME five leaves and destroyed all of them.
    THE SEARCH IS DETERMINISTIC, and this is the second independent
    paid confirmation: identical config_hash 567d32789ba6dcef,
    identical 43 states / 7 complete leaves, identical selected state
    ids in order, and the first depth invocation chose layer 21 with
    score 0.625600, runner-up 17, margin 1.529e-03 -- byte-identical to
    attempt 11 on a different host three days earlier. Only wall time
    differs (203.8 vs 180.3 min), which is host speed.
    THE OOM, in its own numbers: the driver runs the beam search
    IN-PROCESS and still holds ~24.05 GiB at the end of stage 1; stage
    2 spawns train_stage3.py as a SUBPROCESS needing ~17.97 GiB; the
    L40S has 44.39 GiB and 2.36 GiB was free when cross_entropy asked
    for 3.58. Structural, not a race: it will recur at the same point
    on every attempt on this hardware. Same shape as the
    reference-cache finding -- the search's residency is larger than a
    standalone measurement of a stage suggests.
    Pod deleted, provider confirms gone at 229.5 min against a 1397-min
    bound. manifest rc=0, 14 files, 7 classes. No launcher error.
    AUTHORIZATION AND GRANT CONSUMED. NO ATTEMPT 13 IS PREPARED,
    GRANTED, FUNDED OR IMPLIED.
    Evidence: logs/autoinit_phase_a_attempt12/
Phase A attempt 11 (L40S, 194.6 min) STAGE 1 PASSED             $  3.2101
    AUTHORIZED 2026-08-20 as autoinit.phase_a.2026-08-20T0940Z
    against logs/autoinit_phase_a_attempt11_grant.json (sha256
    6a6251f7a534). Ceiling $23.0484, of which $3.2101 was spent.
    THE BEAM SEARCH RAN TO COMPLETION FOR THE FIRST TIME.
      stage 0   PASSED, attested, 2.0 min.
      stage 1   PASSED, 180.3 min. 43 states, 4 levels, 7 complete
                leaves, 18 pruned. Five leaves selected, each
                596,049,920 parameters, each a distinct four-operator
                composition. Control injected and frozen-hash verified.
                THE EXISTING COMPOSITE_STAGE1 RECIPE LANDS ON FRONT 4
                AND IS NOT SELECTED -- four search-discovered orderings
                dominate it. First evidence that operator ORDERING
                carries signal, which is the question Phase A exists
                to ask.
      stage 2   FAILED CLOSED on the first rung-1 probe:
                "teacher and student tokenizers differ; refusing to
                train". Diagnosed and REPRODUCED at $0:
                Qwen3Adapter.save() calls save_pretrained(), which
                writes weights and config and NO tokenizer files;
                AutoTokenizer.from_pretrained() on such a directory
                does NOT raise -- it returns a ONE-TOKEN vocabulary.
                The guard caught it. Without the guard the probe would
                have trained against a 1-token tokenizer and produced
                numbers. The canonical init's own tokenizer is fine and
                hashes identically to the teacher's; what is missing is
                any step carrying those files into a SEARCHED leaf.
                The control_sb class again (2026-08-16): identity gates
                pass while the checkpoint cannot be used, because the
                gates check what the PRODUCER needs.
      stages 3-5  not reached.
    THE REFERENCE CACHE FELL BACK ALL FOUR TIMES. 16.9 GiB does not fit
    in 66% of the 20.3 GiB free INSIDE the search; the measurement saw
    36.42 GiB free standalone. Four causal-depth invocations, each a
    full 260 evals: 37.3 / 27.0 / 33.7 / 24.1 min = 122.1 min, 68% of
    the search, at 6.96-10.79 eval/min against the standalone 12.07.
    The measurement was not wrong; it measured a different memory
    regime, and the cached path may never be reachable inside the
    search at this teacher size.
    THE DEADLINE FIX WAS LOAD-BEARING BY 17 SECONDS. Stage 1 took
    180.283 min against the old 180.0000 base bound it would have been
    killed at, and 363.9841 as derived in 16e382f. That commit is the
    only reason this search produced a result.
    Pod deleted by the launcher, provider confirms gone at 194.5 min
    against a 1397-min bound. manifest rc=0, 13 files, 7 classes.
    LEAF WEIGHTS ARE GONE: finalists are fetched after stage-5
    selection, which never ran, so the five checkpoints died with the
    pod. The RECORD survives; regenerating the weights costs another
    ~180 min of search.
    AUTHORIZATION AND GRANT CONSUMED: one launch, spent, not reusable.
    NO ATTEMPT 12 IS PREPARED, GRANTED, FUNDED OR IMPLIED.
    Evidence: logs/autoinit_phase_a_attempt11/
Bounded measurement attempt 3 (L40S, 12.6 min) **COMPLETE**    $  0.2077
    AUTHORIZED 2026-08-20 as autoinit.measurement.2026-08-20T0512Z
    against logs/autoinit_measurement_grant3.json (sha256 2124eaef0fc5).
    A SpendAuthorization: phase_a_authorized FALSE by type. NOT a
    Phase-A attempt; hard ceiling $1.6294, plan's own hard stop
    $0.8910, of which $0.2077 was spent. ALL_DONE, passed=true,
    manifest rc=0, teardown on the NORMAL gate, pod fsk7tz1rnx43xr
    deleted with provider confirmation. No launcher error.
    THE MEASUREMENT RAN AND ANSWERED EVERY QUESTION IT WAS SET.
      rate            12.07 weighted evaluations/min against E8a's
                      frozen 12.0/min anchor -- +0.6%. The 260-eval
                      schedule prices at 21.53 min vs E8a's 21.7.
                      Attempt 10's host path needed >= 647 min for ONE
                      expansion and never finished: >= 30x slower.
      backend         max AND mean per-item KL delta EXACTLY 0.0 at
                      both |skip|=1 and |skip|=8. Not below a
                      threshold -- identically zero.
      aggregation     0.0228 and 0.0340 apart, which is the DECLARED
                      position-weighted vs unweighted-mean difference
                      (predicted ~0.027), not drift. It is ~300x the
                      8.195e-05 decision margin, which is exactly why
                      the contract compares per item.
      VRAM            production peak 26.82 GiB (the Phase-A number),
                      comparison peak 10.45 GiB, on a 44.39 GiB L40S:
                      17.57 GiB headroom. The dual-cache repair holds.
      cache           CACHED at the frozen mixture. 16.913 GiB estimate
                      against 36.42 GiB available at fraction 0.66,
                      headroom read from cuda.mem_get_info. No
                      fallback; the priced basis stands.
      GPU             mean 98.3%, median 98%, min 94, max 100 over 221
                      nvidia-smi samples. ZERO samples below 10%.
                      Attempt 10 sat at 0-1% for 11 hours. The .cpu()
                      diagnosis is confirmed on hardware.
      cgroup          visible_cpus 128, torch threads 128 -> 13 from
                      cgroup.v2. The container reported CPUs it did
                      not have and the driver corrected for it.
    THESE VALUES AUTHORIZE NOTHING. They are inputs to a repricing and
    a SEPARATE cumulative-budget decision. No Phase-A attempt 11 is
    prepared, granted, funded or implied.
    AUTHORIZATION AND GRANT CONSUMED: one launch, spent, not reusable.
    Evidence: logs/autoinit_measurement_attempt3/
Bounded measurement attempt 2 (L40S, 11.1 min) ENTRYPOINT      $  0.1834
    AUTHORIZED 2026-08-19 as autoinit.measurement.2026-08-19T1738Z
    against logs/autoinit_measurement_grant2.json (sha256 82f5104d49e4).
    A SpendAuthorization: phase_a_authorized FALSE by type. NOT a
    Phase-A attempt; hard ceiling $1.6294, of which $0.1834 was spent.
    NO MEASUREMENT RAN. SETUP PASSED END TO END -- SETUP_RC=0, all
    eleven markers, both frozen assets staged and verified, in 6.5 min.
    The attempt-1 setup-contract repair HELD on hardware.
    The driver then exited 1 in the first statement of main() that
    touches the repository:
      ImportError: cannot import name 'as_operator_items' from
                   'aadistill.autoinit.datasets'
    It lives in scripts/autoinit/phase_a_search.py and always has.
    WHY $0 COULD NOT SEE IT: 22 tests covered this job and every one
    called run_measurement / skip_set / GpuSampler / the stop conditions
    DIRECTLY. None called main(), so the entire production entrypoint --
    argument defaults, pinned revision, model loading, calibration
    resolution, identity assembly, report, stop conditions, artifact
    write -- was reachable only from a paid pod. The tested surface and
    the executed surface were different surfaces.
    REPAIRED at $0: run_entrypoint(args, *, hardware, teacher_loader,
    calibration, ...) holds all of it and main() is three lines, so the
    CPU test drives the production path with a toy loader rather than a
    parallel imitation of it. No second main(). Mutation-verified:
    restoring the bad import fails 1; bypassing the seam fails 1;
    moving the unpinned-revision guard after loading fails 2.
    A FIFTH MUTATION PASSED and exposed a real hole: load_teacher is the
    one function the seam injects past, so dropping revision= from
    from_pretrained -- measuring against whatever the Hub published that
    morning -- was invisible. Now covered by a stubbed from_pretrained.
    SECOND DEFECT, $0.0034 of the above: the launcher then raised
    ArtifactError because the emergency teardown demanded this session
    name the event streams it was truncating. It declares NONE by
    design; quiescence failed because a final_required artifact was
    MISSING, not because a producer was mid-write.
    evaluate_teardown now takes streams_at_risk (the manifest's own
    completion_marker_failures + still_being_written) and requires
    naming only when there is something to name. streams_at_risk=None
    keeps the strict rule, so FAIL-CLOSED BEHAVIOUR FOR TRAINING
    SESSIONS WITH INCOMPLETE EVENT STREAMS IS UNCHANGED.
    Pod deleted by the launcher, provider confirms gone at 11.1 min
    against a 54-min hard bound.
    AUTHORIZATION AND GRANT CONSUMED: one launch, spent, not reusable.
    NO ATTEMPT 3 IS PREPARED OR AUTHORIZED.
    Evidence: logs/autoinit_measurement_attempt2/
Bounded measurement attempt 1 (L40S, 4.0 min) SETUP CONTRACT   $  0.0700
    AUTHORIZED 2026-08-19 as autoinit.measurement.2026-08-19T1142Z
    against logs/autoinit_measurement_grant.json (sha256 ec73be8c1962).
    A SpendAuthorization: phase_a_authorized FALSE by type. NOT a
    Phase-A attempt; hard ceiling $1.6294, of which $0.07 was spent.
    NO MEASUREMENT RAN. Setup refused at the frozen-asset gate
    (SETUP_RC=91, MARKER:FROZEN_ASSETS_FAILED) before the teacher
    download and before any evaluation:
      "state_eval_v1: artifacts/stage1/state_eval_v1 is absent"
    The session declared LOCAL_ASSETS = () because it reads only the
    calibration and the teacher, both from the relay. True, and beside
    the point: autoinit_preflight_setup.sh runs verify_frozen_assets.py
    UNCONDITIONALLY, and that verifier checks both frozen roots whatever
    the session is doing. What binds is what the SETUP REQUIRES, not
    what the session reads.
    THE DEVICE-CANARY RETRY AGAIN, sixteen days later: that session also
    declared LOCAL_ASSETS = (), also for a true reason, and also died in
    setup ($0.0637). The 2026-08-18 fix stopped the setup COPYING
    undeclared assets -- correct, and it held here -- but nothing told a
    session which assets it MUST declare.
    REPAIRED at $0: tests/pod/test_session_setup_contract.py asserts
    verifier_required_local_roots is a subset of every session's
    installed local roots, comparing DECLARATIONS not filesystem
    presence, with the requirement DERIVED from verify_frozen_assets
    .FROZEN rather than transcribed -- so a third frozen root added to
    the verifier and to no session fails at $0. Mutation-verified.
    The device canary's declaration was ALSO still wrong and is
    corrected; that is not reviving it.
    Pod deleted by the launcher, provider confirms gone. Failed closed.
    AUTHORIZATION AND GRANT CONSUMED: one launch, spent, not reusable.
    Evidence: logs/autoinit_measurement_attempt1/
Phase A attempt 10 (L40S, 692.5 min) INCOMPLETE, RUNTIME COST  $ 11.4300
    AUTHORIZED 2026-08-18 as autoinit.phase_a.2026-08-18T1746Z against
    logs/autoinit_phase_a_attempt10_grant.json (sha256 3ef080d91d58).
    NOT A SCIENTIFIC RESULT AND NOT A STAGE-1 SELECTION RESULT.
    Setup and Stage 0 passed; Stage 0 attested interval 0.011695,
    floor 0.3000, plan 02be33b9. THE THREE STAGE-1 DEVICE FIXES HELD --
    stream_projection ran on CUDA and the composite expansion below it
    completed, so the attempt-9 defect did not recur.
    Stage 1 then entered its THIRD operator expansion,
    depth.causal_kl_greedy_v1 (third in deterministic registry order;
    the two written states are exactly the first two), and was still
    inside it 10 h 47 m later with the L40S at 0-1% utilisation and 192
    driver threads saturating a 13-vCPU cgroup.
    THE COST, IN THE OPERATOR'S OWN NUMBERS: greedy_removal(36, 8) is
    260 evaluations x 67 calibration items = 17,420 forward+distortion
    pairs. Each copies the logits device->host (_forward_logits returns
    .cpu(), targets are .cpu()) and runs a full 151,936-vocabulary
    softmax/KL on the CPU: ~0.33 TiB of CPU traffic per evaluation,
    ~86 TiB over the expansion, ~8.6 TiB copied off the device. The
    transfer is deliberate and documented; the CPU cost of the reduction
    was never priced.
    MEASURED VS PRICED: --search-minutes 180.0 priced the WHOLE beam
    search at 3.0 h. This ONE expansion ran 10.78 h without finishing --
    at least 3.6x the entire search budget, and a lower bound.
    NOTHING ENFORCED IT: --search-minutes feeds only an affordability
    check before the search starts (driver:433); search.py records
    elapsed but never checks a deadline, and _expand_one has no clock.
    The only backstop was the watchdog at the full $23.05 ceiling.
    Stopped on maintainer instruction at 05:18 UTC via the supported
    path -- the driver was stopped, the launcher's poll broke on
    DRIVER_EXITED:143 and ran its normal collect_and_teardown. Manifest
    rc=0, 9 files, gate allowed, pod deleted, provider confirms gone.
    The pod was NOT repaired.
    How far through the 260 evaluations it got is UNKNOWN: greedy_removal
    journals only on completion, so the run emitted no external progress
    signal. The two states' weights were not in the failed-artifact spec
    and are gone; their specs and hashes survive.
    Evidence: logs/autoinit_phase_a_attempt10/
Phase A attempt 9 (L40S, 20.4 min) STAGE 0 PASSED, STAGE 1 FAILED $  0.3400
    AUTHORIZED 2026-08-18 by an explicit maintainer GO, issued against
    logs/autoinit_phase_a_attempt9_grant.json (sha256 7b62b5c516be) as
    autoinit.phase_a.2026-08-18T1512Z.
    THE ATTEMPT-8 FIX HELD: setup reached SETUP_DONE in 7.4 min and the
    blocking test gate PASSED -- the same tests/docs suite that failed
    attempt 8. That question is closed.
    STAGE 0 PASSED and attested, the second time on hardware:
      evaluation_protocol 250f72ef  comparable_identity 70a26e0b
      science_plan 02be33b9         source_digest a1b51736
    all three frozen identities matched under comparability v2. This host
    drew driver 580.159.03 -- the same as Stage 3 and attempt 5 -- so as
    on attempt 5 it does not by itself discriminate v2 from v1.
    STAGE 1 FAILED, and the traceback CAME HOME:
      RuntimeError: Expected all tensors to be on the same device, but
      found at least two devices, cuda:0 and cpu!
      src/aadistill/init/project.py:60  avg += w * (m / m.trace())
    project.py:57 allocates `avg` with a dtype and NO device, so it lands
    on CPU, while uncentered_moment() follows `state` -- CPU on the dev
    box, cuda:0 on a pod. A CPU rehearsal cannot see it: both operands
    agree there and the arithmetic is correct.
    THIRD Stage-1 device-placement defect: attempt 6 ($0.3552) the
    _validate probe, attempt 7 ($0.3955) the ActivationStatsCollector
    accumulators, now this. autoinit.stage1_device_contract@v1 closed the
    first two and did not reach a freshly-allocated accumulator two call
    levels below the operator. Three for three, every one is a tensor
    allocated without a device in a path only a GPU executes.
    Nothing trained; no checkpoint, probe or search leaf produced. The
    permanent controls are inputs here and were untouched. Manifest rc=0,
    10 files, teardown gate allowed; pod deleted, provider confirms gone.
    Watchdog's last tick read $0.3498 at 21.2 min, which counts until it
    noticed the pod was already gone; the pod's own 20.4 min lifetime at
    $0.99/h is $0.3366 -> $0.34 carried.
    Failed closed; STOPPED FOR REVIEW. Grant spent; no attempt 10.
    Evidence: logs/autoinit_phase_a_attempt9/
Phase A attempt 8 (L40S, 11.28 min) SETUP TEST GATE FAILED   $  0.1900
    AUTHORIZED 2026-08-18 by an explicit maintainer GO, issued
    against the one-use grant logs/autoinit_phase_a_attempt8_grant.json
    (sha256 09541ef547c6) as authorization
    autoinit.phase_a.2026-08-18T1244Z. The device-canary condition
    that had gated attempt 8 was superseded by that decision: the
    canary path is TERMINATED and was not revived.
    NO STAGE RAN. Nothing trained, measured, or written to the relay.
    Setup reached ROPE_OK -- eight markers in -- so the new
    manifest-driven relay staging, the pod-side frozen-asset gate and
    the staged checkpoint all worked on hardware for the first time.
    It then failed the blocking CPU test gate: 2 failed, 1789 passed,
    63 skipped. Both failures are dev-box-environment tests that
    cannot pass in a container, and both were added by the 2026-08-18
    inventory/cleanup work, not by the relay-staging fix:
      test_every_path_named_in_the_repo_layout_exists -- REPO_LAYOUT.md
        names /home/ecs-user/aad-artifacts/ and /home/ecs-user/aad-scratch/,
        and `REPO / ref` discards the base for an absolute operand, so
        the test reads the host filesystem. A pod has neither (69b2e74).
      test_no_tombstoned_path_is_still_on_disk -- the tombstone
        stage3_ladder_uniform_local_cache names artifacts/stage3/
        ladder_uniform, which the pod's setup STAGES as the recovery
        pack's mirror. Same shape as the podsim tombstone dded03e
        itself withdrew (dded03e).
    The pod simulator could not catch either: it cannot unmake an
    out-of-tree path, and for the tombstone it produces the OPPOSITE
    of the pod's state -- it HIDES that gitignored directory, so the
    assertion passes for the wrong reason.
    Pod 2maapdxqg566r5 deleted by the launcher, provider confirms gone.
    Watchdog accrued $0.1833 at its last tick; the launcher's rounded
    $0.19 is carried, per the higher-figure rule above.
    Failed closed as instructed; STOPPED FOR REVIEW. The grant covered
    one launch and is SPENT. No retry was attempted and no attempt 9
    is authorized, funded, prepared or implied.
    Evidence: logs/autoinit_phase_a_attempt8/
unused authorization remaining                               $ 20.5286
    = $234.00 - $213.4714. NOT enough for another full Phase-A attempt
    at the $23.0484 per-launch ceiling -- it is $2.5198 short. The cap
    funded one attempt, that attempt has run, and the approval says it
    does not authorize Attempt 13.
    (superseded: $24.3158 before attempt 12) RAISED AND APPROVED 2026-08-21 from
    $231.00. Enough for one full-ceiling Phase-A attempt ($23.0484)
    with $1.2674 of margin -- and for nothing after it. The approval
    says in terms that it is "only the project ceiling", that the
    per-launch ceiling is unchanged, and that no Attempt 12 launch is
    authorized yet. Attempt 12 still requires its own one-use grant and
    authorization; a $234.00 cap does not authorize an Attempt 13.
    (superseded arithmetic, kept for the audit trail: This is NOT enough for another full Phase-A
    attempt at the $23.0484 per-launch ceiling -- it is $1.7326 short.
    RECOMMENDED AND NOT APPROVED (2026-08-20): raise the cumulative
    project cap to $234.00. One further full-ceiling launch would reach
    $209.6842 + $23.0484 = $232.7326, so $234.00 leaves $1.2674 of
    margin. That is a PROJECT-LEVEL CEILING recommendation only: the
    Phase-A per-launch hard ceiling stays $23.0484, a future Attempt 12
    still requires a new explicit one-use grant and authorization, and
    a $234.00 cap does NOT authorize an Attempt 13. Until it is
    approved in the imperative, no Phase-A attempt may be prepared.
    The 2026-08-20 raise funded ONE attempt, that attempt has run, and
    the maintainer's approval says in terms that it "does not authorize
    any subsequent attempt".
    UNUSED BALANCE IS STILL NOT AUTHORIZATION. The cap makes Attempt 11
    affordable; the GRANT and the AUTHORIZATION are what make it
    permitted, and both are one-use. After Attempt 11 the remaining
    figure MUST NOT be read as funding an Attempt 12: the maintainer's
    approval says the raise "does not authorize any subsequent attempt".
    Nothing below is a standing permission to spend; each line is a
    consumed or lapsed commitment kept for arithmetic.
    Previously committed against it:
      device canary hard                                       $  1.0197
        SPENT $0.1240 across two sessions, NEITHER of which ran
        the canary script: attempt 1 died on the launcher's
        argument contract, the retry on the shared setup's asset
        contract. Both grants are consumed and the retry was
        explicitly the only one authorized. NOT canary results.
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

## 2026-08-27 — cumulative cap raised to $256.84 for ONE Phase-B execution

```
previous cumulative actual spend                             $230.0350
previous authorized cumulative cap                           $234.00
NEW authorized cumulative cap                                $256.8400
newly approved headroom                                      $ 26.8050
Phase-B SINGLE-SESSION hard ceiling                          $ 26.8049
Phase-B planning floor (NOT an expected spend)               $ 13.0800
```

**APPROVED by the maintainer**, in their own words: "Increase the
AlphaAvatar-distill cumulative budget cap from **$234.00 to $256.84**." The
reviewer's verdict on the same message: Phase-B science GO, Phase-B
implementation/readiness GO, cumulative increase APPROVED, and one Phase-B paid
execution authorized to prepare and launch subject to the frozen fail-closed
gates.

**The $0.0001 difference is deliberate.** `$256.8400 - $230.0350 = $26.8050`,
while the frozen single-session ceiling is `$26.8049`. The rounding margin
belongs to the **cumulative** cap; it does not enlarge the session ceiling, and
`PhaseBAuthorization.require_within_cap` is issued against `$26.8049`.

**`$13.0800` is a planning floor, not an expected spend.** No expected-value
assumption over survivor identity or tie-break probability is defined anywhere.

**Unused headroom is not authorization.** Whatever Phase B does not spend funds
nothing else; a later experiment needs its own maintainer decision.

## 2026-08-26 — Phase-B attempt 1: $0.1500, aborted at setup

```
cumulative spend before                                      $230.0350
Phase-B attempt 1 (pod 4dbqycjrivhq17, 8.8 min, L40S)        $  0.1500
cumulative spend after                                       $230.1850
authorized cumulative cap                                    $256.8400
remaining headroom                                           $ 26.6550
```

Aborted at the pod-side setup test gate; **no scientific stage ran**. The pod was
deleted and the provider confirms it is gone. The one-use authorization
`autoinit.phase_b.20260826T053826Z` is **consumed**; a retry needs a new grant.

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
