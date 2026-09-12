# Recovery continuation attempt 6 — the checkpoint read is FIXED; generation has no tokenizer, $1.4926

**Verdict: the attempt-5 repair worked. The probe trained, its checkpoint
resolved, and the run reached the battery — one step further than any attempt.
Generation then failed because a trainer-written checkpoint carries no
tokenizer.** Fail-closed, torn down with provider confirmation, no relaunch.

| | |
| --- | --- |
| authorization | `autoinit.recovery_continuation.2026-08-23T0944Z`, sha256 `b15aab3c…` |
| grant | `logs/autoinit_recovery_continuation_attempt6_grant.json` |
| base commit | `948b1e8` |
| session commit | `08670e5e7c499c6623bd7505c3efe0a509cf348c` |
| harness digest | `0dbf1272…`, 22 files, search excluded |
| bundle | `aad_autoinit_08670e5e.bundle`, sha256 `6e2ac713…` |
| pod | `ifp8feyil1gp7v`, L40S $0.99/h, **90.46 min, $1.4926**, provider confirms gone |
| terminal | `DRIVER_EXITED:22`, `PHASE_A_FAILED` at stage 2 |

## What advanced

```
10:15:49 MARKER:STAGE_PASSED:0      attested 0.011695 / 0.3000 / 02be33b9
10:15:55 MARKER:STAGE1_IMPORTED:5   re-identified from bytes, config 567d32789ba6
10:18:36 control measured on state_eval@v1 (74022 positions)
10:18:37 device handoff: 0.01 GiB allocated … not a model leak     live_retention=false
10:18:37 MARKER:STAGE_PASSED:1
10:18:37 MARKER:STAGE_START:2
11:30:28 MARKER:PROBE_TRAINED:autoinit.v1.phase_a.rung1.cca699c93f34.sa
```

* setup at **$0.13**, the cheapest of any attempt; TCP 22 in **0.2 min**;
* Stages 0 and 1 reproduced attempt 5 exactly, including a clean handoff;
* **the probe trained in 71.9 min** (attempt 5: 61.7; priced 61.55) — see below;
* **`trained_model_dir()` resolved the checkpoint.** No `FileNotFoundError`. The
  attempt-5 defect is closed on hardware: execution passed the exact line that
  ended that run and entered `battery()`.

## What failed

`uncapped_eval.py`, 50 s after `PROBE_TRAINED`:

```
File "/workspace/aad/scripts/evaluation/uncapped_eval.py", line 321, in main
    prompt = tok.apply_chat_template(turns, tools=s.get("tools"), …)
ValueError: Cannot use chat template functions because tokenizer.chat_template
is not set and no template argument was passed!
```

`Trainer.save_checkpoint` writes `self.student.save_pretrained(ckpt_dir/"model")`
— **weights and config, no tokenizer**. `battery()` passes
`--model <that dir>` and no `--tokenizer`, and `uncapped_eval`'s `--tokenizer`
"defaults to the checkpoint's own tokenizer". The probe checkpoint has none.

**Every previously proven caller passed `--model CANONICAL_INIT`** — the
preflight driver, the continuation driver, the generation smoke tests — and that
directory *is* a full checkpoint with `tokenizer*.json` and
`chat_template.jinja`. The Phase-A battery is the first caller ever to point
`--model` at a **trainer-written** checkpoint, so the default rule had never been
exercised against one. This is the third appearance of the
checkpoint-without-a-tokenizer class, after the control-sb packaging repair and
attempt 11's one-token tokenizer.

### The obvious one-line fix is NOT protocol-neutral — do not apply it

Passing `--tokenizer <canonical init>` would fix the crash and **silently break
comparability**:

* Stage 0 of this very run attested `tokenizer_source: "the evaluated
  checkpoint"`, `tokenizer_sha256: c1db93c8…`, under evaluation protocol
  `250f72ef…` — the frozen Stage-3 hash;
* `RecoveryEvaluationProtocol.identity()` returns **every** declared field,
  including `tokenizer_source`;
* frozen rule `generation_runtime_comparability@v2` declares material:
  *"every field the protocol already declares except `runtime_digest`,
  `evaluation_protocol_hash`, `generation_protocol_fingerprint`"*.

`tokenizer_source` is therefore **material**. Passing `--tokenizer` changes it to
`"external: …"`, which makes every probe incomparable to the Stage-3 controls
that materialized the equivalence interval and feasibility floor.

The protocol-neutral shape is instead to make *"the evaluated checkpoint"* true —
give the probe's `model` directory the frozen tokenizer whose files hash to the
attested `c1db93c8…` — so both `tokenizer_source` and `tokenizer_sha256` are
reproduced rather than redefined. Whether that is done by the driver after
training, or by the trainer at save time (which would change the trainer and the
checkpoint bytes, currently out of bounds), is a maintainer decision and is
**not** taken here.

## A second observation, recorded not diagnosed

The probe took **71.9 min** against attempt 5's **61.7** and the **61.55** the
budget is priced from — **+16.6%** on a different host, same recipe and card
model. One observation on one host is not a trend, but the pricing basis assumes
61.55 and nine probes; if 71.9 were typical the expected envelope would move.
Recorded for the next pricing review, not acted on.

## Not attempted, and why

No repair on the live pod and no relaunch. The candidate repair touches
`autoinit_phase_a_driver.py`, inside the 22-file harness, so it moves the digest
as well as the session commit — and the protocol question above must be settled
first.

**The trained probe was again lost with the pod** (`checkpoints_fetched` empty):
71.9 paid minutes, for the second time.

## State

* pod deleted, **provider confirms gone**; watchdog ended `pod_gone` after 91
  ticks; poller stopped; provider returns zero pods; nothing billing;
* the five Attempt-12 leaves are **untouched**; permanent controls not retrained;
  frozen science untouched;
* `$217.1763` cumulative against the `$234.00` cap — **`$16.8237` remains against
  a `$16.7456` ceiling, a margin of `$0.0781`**. One more full-ceiling
  continuation fits by less than eight cents.
