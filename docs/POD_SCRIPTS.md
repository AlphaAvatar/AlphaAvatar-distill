# Pod script catalog

Every executable under `scripts/pod/`, classified. Nothing here is deleted for
tidiness: reproducing a recorded result means reproducing the implementation
that produced it (AGENTS.md P4), so a retired experiment's launcher stays and is
labelled instead of removed.

`tests/docs/test_repository_structure.py` requires every file in `scripts/pod/`
to appear below.

| class | meaning |
| --- | --- |
| **ACTIVE** | part of the current session architecture; may be edited |
| **DIAGNOSTIC** | a $0 or near-$0 tool that is still useful |
| **HISTORICAL** | produced a recorded result. Reproducibility-critical, frozen |
| **SUPERSEDED** | replaced by something in ACTIVE; kept for provenance |
| **TERMINATED** | a path deliberately stopped; not to be resumed |

## ACTIVE — the current session architecture

| file | role |
| --- | --- |
| `autoinit_preflight_launch.py` | micro-preflight **session specification** — `spec(args) -> SessionSpec`, no flow |
| `autoinit_phase_a_launch.py` | Phase-A **session specification** |
| `autoinit_continuation_launch.py` | the Stage-3 continuation's **session specification** |
| `autoinit_measurement_launch.py` | the bounded causal-depth runtime/backend measurement's **session specification**. Names `SpendAuthorization`, so it cannot start Phase A; runs no search, selects no depth map, writes no checkpoint |
| `autoinit_recovery_continuation_launch.py` | the recovery continuation's **session specification**. Priced by `continuation_budget` ($16.7456 hard, no search), declares attempt 12's five preserved leaves as staged session inputs, and names a driver that cannot search |
| `autoinit_recovery_continuation_driver.py` | its pod-side driver. Stage 1 IMPORTS the verified attempt-12 result — it never imports `phase_a_search`, never delegates to the searching `stage1`, and has no `--stage` value that searches |
| `autoinit_preflight_driver.py` | the micro-preflight's pod-side driver |
| `autoinit_phase_a_driver.py` | the Phase-A pod-side driver, six stages |
| `autoinit_preflight_setup.sh` | the shared pod setup. **Manifest-driven** since 2026-08-18: it reads `SESSION_ASSETS`, `SESSION_RELAY_INPUTS` and `SESSION_TEST_IGNORES`, and names no session's assets, relay paths, destinations or digests itself |
| `autoinit_science_inputs.py` | the frozen relay science inputs — source, destination, digest — that sessions compose their `relay_inputs` from. Lifted out of the shared setup on 2026-08-18; here rather than in `src/` because `docs/REPO_LAYOUT.md` rule 1 keeps frozen hashes in the scripts that own them |
| `autoinit_continuation_driver.py` | the continuation's pod-side driver |
| `autoinit_engine_probe.py` | vLLM engine identity probe, run at stage 0 |
| `watchdog.py` | the independent provider-side kill switch |
| `collect_artifacts.py` | artifact manifest, gate and collection |
| `simulate_pod_env.sh` | runs the pod's exact test command locally with pod-absent artifacts hidden |
| `retain_checkpoints.py` | per-run checkpoint retention: derives the keep set from a run's own log |
| `start_job.py`, `run_env.sh` | detached start with a durable descriptor |

## DIAGNOSTIC

| file | role |
| --- | --- |
| `canary.py` | **control-plane** canary: verifies detached start, watchdog, GraphQL termination fallback and provider-confirmed disappearance on a disposable pod. Unrelated to the device canary below |
| `throughput_gate.py` | step-time gate |
| `probe_peak_memory.py` (in `scripts/autoinit/`) | peak-VRAM probe |
| `benchmark_padding_truncation.py` | padding/truncation cost |
| `reconstruct_training_events.py` | rebuilds a training event stream from artifacts |
| `verify_and_report.py`, `post_run.sh`, `score_refs.sh` | post-run verification |
| `build_wheelhouse.py` | offline wheelhouse construction |
| `test_cold_host_tripwire.sh` (in `tests/pod/`) | cold-host detection |

## TERMINATED — the paid device canary

| file | status |
| --- | --- |
| `autoinit_device_canary.py` | the workload: one invocation of each frozen operator on CUDA, through the production lifecycle |
| `autoinit_device_canary_launch.py` | its one-use session specification |

**Strategically terminated 2026-08-18.** Two authorized sessions, **$0.1240**,
**zero canary runs**: attempt 1 died before setup on an inherited argument the
wrapper did not declare, the retry died inside setup on assets the wrapper had
declared it did not want. Neither reached the canary script, so neither says
anything about device placement on CUDA.

Kept, not deleted, for three reasons: the evidence in
`logs/autoinit_device_canary_attempt{1,2}/` is accountable spend; the workload is
a correct description of what a device canary would do; and the *generic* lesson
— that reusing shared machinery means satisfying its whole contract, not the
part your session happens to need — is now enforced structurally by
`SessionSpec`, which is where the value ended up.

**No further canary is prepared or authorized.** If one is ever wanted, the
starting point is `logs/STATE.md`, not this directory.

## HISTORICAL — produced recorded results, frozen

Retired experiment machinery. Each triple is a launcher, a driver and a setup
script for one experiment; the results they produced are in `logs/`.

| experiment | files |
| --- | --- |
| D0 diagnostics | `d0diag_launch.sh`, `d0diag_driver.py`, `d0diag_setup.sh` |
| E2 diagnostics | `e2diag_launch.sh`, `e2diag_driver.py`, `e2diag_setup.sh` |
| E2 P1 | `e2p1_launch.sh`, `e2p1_driver.py`, `e2p1_setup.sh` |
| E3 | `e3_launch.sh`, `e3_driver.py`, `e3_setup.sh` |
| E4 | `e4_launch.sh`, `e4_driver.py`, `e4_setup.sh` |
| E5 | `e5_launch.sh`, `e5_driver.py`, `e5_setup.sh`, `e5_pilot.py` |
| E6 | `e6_launch.sh`, `e6_driver.py`, `e6_setup.sh`, `e6_stage_checkpoints.py` |
| E6b | `e6b_launch.sh`, `e6b_driver.py`, `e6b_setup.sh` |
| E7 | `e7_launch.py`, `e7_driver.py`, `e7_setup.sh` |
| E8a | `e8a_launch.py`, `e8a_driver.py`, `e8a_setup.sh` |
| E8b | `e8b_launch.py`, `e8b_driver.py`, `e8b_setup.sh` |
| P0 assistant | `p0asst_launch.sh`, `p0asst_driver.py`, `p0asst_setup.sh` |
| P2 | `p2_launch.sh`, `p2_driver.py`, `p2_setup.sh` |
| E5/E6 registration | `register_p0_real.py` |

Transfer manifests recorded by those sessions: `hashes_ckpt.txt`,
`hashes_ckpt_pca.txt`, `hashes_ckpt_rand.txt`, `hashes_ladder.txt`,
`hashes_transfer.txt`.

## SUPERSEDED

| file | superseded by |
| --- | --- |
| `setup.sh` | the per-experiment setup scripts, then `autoinit_preflight_setup.sh` |
| `orchestrate.sh` | the Python launchers |
| `train.sh` | `scripts/training/train_stage3.py` |
| `checkpoint_inventory.py` | `scripts/consolidate/build_checkpoint_registry.py`. Both inventory both stores; this one's `REQUIRED` set is written around Experiment 2 and has not moved since, and two inventories with different stale opinions is worse than one. Its LFS insight — that removing a file from a Hugging Face repo's current revision reclaims no quota — is preserved in the replacement's docstring and in `logs/relay_mirror_verification.json` |

## Local notes

`AGENTS.md` (that is, `scripts/pod/AGENTS.md`) carries directory-local rules for this area and is the
authority on how a pod session must be started, watched and torn down.
