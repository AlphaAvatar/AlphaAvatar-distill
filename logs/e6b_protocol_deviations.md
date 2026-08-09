# Experiment 6b — protocol deviation record

**Classification: scientific endpoint valid · operational protocol noncompliant.**

Both statements hold at once and neither cancels the other. The result stands
because both arms completed the frozen schedule, the final checkpoints were
retrieved and hash-verified, and the frozen evaluation artifacts are complete.
The session was nonetheless run outside its authorized cost and lost a class of
artifact the project requires (AGENTS.md P4, 3.7). This file is the permanent,
non-superseded record of both. It is not a retraction and not an excuse.

Maintainer disposition, 2026-08-09: **result accepted, neither arm to be rerun.**

---

## 1. Cost deviation

| | |
| --- | --- |
| authorized hard backstop | **$7.12** |
| actual E6b cost | **$7.68** |
| overrun | **$0.56** |
| project cumulative | $149.59 against a $149.03 cap — **over by $0.56** |

Two paid events: the $0.12 pod that died at `INIT_READY` on a missing
`TEACHER_REVISION`, and the $7.56 session that produced the result.

**Proximate cause — a 14% step-time miss.** The session was priced at
**3.625 s/step**, derived from E4's comparable arms. It sustained **4.15 s/step**.
Over 2 × 2,916 steps that is ~49 minutes, ~$0.81 of unbudgeted time, and it put
the session past its authorization before any teardown decision existed.

**Structural cause — two independent stop layers were inert at once.**

* **RunPod's `--terminate-after` did not fire.** The absolute deadline was set at
  creation for 00:28:47. The pod was still `RUNNING` at 00:34 and was deleted by
  the launcher at 00:56. This flag has been the documented last-resort budget
  layer since E4 and **has never once been observed to fire in this project** —
  every prior session was torn down by its launcher first. It was trusted and
  never tested.
* **The launcher's driver-start ssh did not detach.** The invocation
  (`setsid nohup … > log 2>&1 < /dev/null & disown`) is byte-identical to E6's,
  which returned in 74 s. Here it blocked for the entire 434-minute run, so the
  launcher never reached its polling loop and could not tear down at completion.

**Monitoring failure.** The session watcher tailed the orchestrator **log**. A
blocked launcher writes no lines, so seven hours of a billing pod was
indistinguishable from seven hours of nothing happening.

**Structural cause behind the cause.** The session had *one* number: the
authorization was also the kill point. There was no threshold at which the run
was supposed to stop starting new work, and no time reserved behind it in which
to collect artifacts.

---

## 2. Artifact deviation

| | |
| --- | --- |
| original machine-readable training event streams | **LOST** |
| driver console log | survives (`e6b_run.log`, 133,084 bytes) |
| final checkpoints | survive, hash-verified |
| frozen evaluation artifacts | survive, complete |

**What was lost.** `artifacts/stage3/e6b_p2_r2960k_{sa,sb}/train_log.jsonl` and
the matching `run_manifest.json`. These existed on the pod, were written
correctly for nine hours, and were destroyed with it.

**Correction to the earlier account.** The first write-up (EXPERIMENTS.md §29.7,
STATE.md, 2026-08-09) attributed the loss to "a bundling glob that did not expand
inside ssh quoting". **That is wrong, and the corrected cause changes the fix.**
The E6b bundling command at the run commit `6375e29` contains no glob:

```bash
tar czf /workspace/e6b_artifacts.tar.gz \
  artifacts/audit/three_mode artifacts/audit/e6_checkpoint_manifest.json \
  artifacts/audit/e6_notrain_proof.json 2>/dev/null
```

The list was inherited verbatim from E6, a session that did **not train**, so
`train_log.jsonl` and `run_manifest.json` were never named at all. Two of the
three paths that *were* named do not exist in an E6b session either; `2>/dev/null`
swallowed the error and the `;`-chained `sha256sum` ran regardless. The retrieved
tarball contains `artifacts/audit/three_mode/**` and nothing else — verified by
`tar tzf` on 2026-08-09.

Every downstream check then passed on the incomplete bundle: tar produced a file,
the pod-side digest matched the local digest, and the transfer verified. **No
check asked whether everything that had to survive was present.**

The `$(ls -d …)`-inside-ssh construct is a real and separate fragility — a
pattern matching nothing yields an empty substitution and a silently shorter
archive — and it is still present in `e3_launch.sh`, `e4_launch.sh` and
`e5_launch.sh`. It is banned for new launchers and lint-enforced
(`tests/pod/test_operational_hardening.py`); those three are exempted by name as
frozen records of completed runs.

**Derived replacement, and what it is not.**
[`e6b_reconstructed_training_events.json`](e6b_reconstructed_training_events.json)
is parsed from the surviving console log by
`scripts/pod/reconstruct_training_events.py`. It carries

```json
{"provenance": "reconstructed_from_driver_console",
 "original_event_stream_available": false}
```

and a per-field `field_provenance` block. It is **not** the original event
stream and must never be described as one.

| class | fields | note |
| --- | --- | --- |
| exact | `step`, `seconds`, `val_blocks`, `val_ce`, `val_ppl`, `val_kd` | printed in full |
| truncated | `loss`, `ce`, `kd` (4 dp vs 6 stored), `lr` (`%.2e` vs full float) | precision lost to the print format |
| derived from config | `tokens_seen`, `run_name` | recomputed, not read |
| bounded only | `time` | no per-event timestamp was printed; per-arm bounds recorded, none invented |
| unrecoverable | `grad_norm`, `ce_targets`, `kd_positions`, `logical_block_tokens`, `executed_positions`, `executed_nonpad_tokens`, `supervised_tokens`, `truncate_padding`, `gpu_mem_gb`, and the `run_start` / `dataset_loaded` / `teacher_loaded` / `student_loaded` / `checkpoint_saved` / `run_end` events entire | never printed |

Recovered content: 291 `train_step` and 10 `eval_result` events per arm, both
arms, plus the final evaluation (`val_ce` 1.169355 / 1.174017).

**A step-time correction that follows from it.** The reconstruction separates two
quantities the single figure "4.15 s/step" conflated:

| measure | sa | sb |
| --- | ---: | ---: |
| printed per-step timing, mean of 291 samples | 4.1485 s | 4.1099 s |
| wall clock per step, driver command → `TRAIN_DONE` | 4.211 s | 4.215 s |

The wall-clock figure is larger because it also carries the ten periodic
evaluations and the checkpoint writes. Future budgets price the **step** at
4.15 s and add evaluation and checkpointing as explicit, separately named phases
(`aadistill.infrastructure.budget`), rather than folding them into one number.

---

## 3. What is unaffected

* Both arms completed the frozen 2,916-step schedule.
* Both final checkpoints were retrieved and hash-verified against pod-side
  digests: sa `89b14b83…`, sb `3c4709b5…`.
* All four generation sets and the full frozen-battery evaluation survive.
* The registration was prospective and unmodified
  ([`e6b_registration.json`](e6b_registration.json)).

The scientific endpoints in [`e6b_report.md`](e6b_report.md) and
[`e6b_results.json`](e6b_results.json) rest on those artifacts alone.

---

## 4. Remediation

Completed at zero GPU cost before any further billable run, 2026-08-09 —
see EXPERIMENTS.md §30 and the decision record of the same date. Summary:

| deviation | fix |
| --- | --- |
| launcher blocked on the driver ssh | `infrastructure/remote.start_detached` — bounded start, durable job descriptor, out-of-band confirmation; `scripts/pod/start_job.py` |
| `--terminate-after` inert | `infrastructure/watchdog.Watchdog` — independent provider polling, terminate, **verify the pod is gone**, retry, journal; `scripts/pod/watchdog.py` |
| log silence read as idleness | `SessionWatcher.assess` requires provider state; no verdict is reachable from markers alone, and none of the verdicts is `IDLE` |
| one number for authorization and kill point | `infrastructure/budget` — expected / soft stop / artifact-recovery reserve / hard terminate, with the reserve held back inside the authorization |
| priced from a superseded step time | the 4.15 s/step floor is enforced; a lower estimate needs a recorded reason, and 3.625 s/step is refused by name |
| event stream lived only on the pod | `infrastructure/log_relay` — continuous incremental mirroring; already-synced events survive the pod |
| bundle list inherited from a different session | `infrastructure/artifact_gate` — declared spec expanded in Python, archive built from the manifest, ordered teardown gate with an emergency-override path that records what was lost |

Regression coverage: 110 new tests — `tests/infrastructure/` (82, including a
15-test end-to-end replay of this failure), `tests/pod/test_operational_hardening.py`
(23) and `tests/pod/test_reconstruct_training_events.py` (9).
