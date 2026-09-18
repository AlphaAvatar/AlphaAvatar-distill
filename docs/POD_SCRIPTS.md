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
| `autoinit_c1_launch.py` | **Phase C1** session specification: fixed-path ATTENTION isolation. `SESSION_KIND=c1`, `C1Authorization`, a `BudgetSpec` derived from `logs/stages/stage-1/phase_c1/plans/phase_c1_pricing.json`, and **twelve** pre-provider gates — the ninth proves a pod can obtain the exact authorized commit from the relay, the tenth that the shared setup's `ROPE_OK` step has the 1,418-byte config it globs for, and the last two are renderer parity and the pod-environment readiness record. It also **requires `--run-id`** and has no `--out`: the session record, the governance snapshots, the collected evidence and the run manifest all live under `logs/stages/stage-1/phase_c1/runs/<run_id>/`. Runs no search |
| `autoinit_c1_driver.py` | the C1 driver: replays the frozen path under the `eea90c91`/`c313d1b4` digest gates, then runs 2 arms x 3 fresh seeds. `stage1`, `run_rung` and `selection_row` all raise — no search, no rungs, no ranking |
| `autoinit_phase_c2_launch.py` | **Phase C2 Search-1** session specification: one beam search over four operator kinds, order free, ATTENTION fixed to `activation_importance_v1` and branching over two calibration mixtures. `C2Authorization`, a `BudgetSpec` derived from `logs/stages/stage-1/phase_c2/plans/phase_c2_pricing.json` with the beam-composition and conditional baseline-rebuild reserves named separately, and three pre-provider gates: disk against a derived 87.4 GiB peak working set, the grant's ceiling against the pricing record, and the grant's plan hash against the live configured space. Trains nothing and fetches no product. **NOT AUTHORIZED**: the grant it names does not exist, so it can only refuse |
| `autoinit_phase_c2_driver.py` | the C2 driver, standalone rather than a `PhaseADriver` subclass because it uses none of the probe/rung/battery machinery. Two stages: `bind_identities` (executable digest, teacher, both mixtures by spec AND content hash, and the frozen C1 baseline construction — the `$0` gate) and `search_and_baseline` (the beam through `run_phase_a_search`, then the baseline resolved ONCE: searched when the beam re-derived it, rebuilt through the frozen C1 fixed path when it did not, never both). It then writes `c2_baseline_comparison.json`, the required record that makes B's cheap-metric result durable in either branch, and the beam and the rebuild run on separate deadlines so the beam cannot spend the rebuild's reserve |
| `autoinit_phase_c2_full_search_launch.py` | **Phase C2 full joint re-search** session specification. A beam, so it declares what a beam needs: a re-quoted rate, a long poll limit and the same staged assets as Search-1. Two things are derived rather than inherited. The STORAGE: Search-1 provisions for a 87.4 GiB peak (five retained level-0 states into eighteen children), but in the joint space every operator kind may go first, so level 0 generates eleven children of which nine continue and level 1 expands those nine into SIXTY — `peak_resident_gib` derives 243.4 GiB and the provision is 350, because the search generates a whole level before pruning and a volume that fills at level 1 loses every state measured. The INPUTS: `staged_assets` and `tracked_non_source_inputs` ask the code what the search reads, after two paid CUDA-validation subruns died one per producer on a hand-maintained list. `SESSION_KIND=c2_full_search`. Eleven `$0` gates, including a `frozen_space_gate` with no counterpart in the other C2 sessions — it refuses an authorization issued against a different number of leaves, a pinned profile mapping or a narrower beam. `FullSearchAuthorization` reports `authorizes_c2_search1`, `authorizes_c2_baseline_completion` and `authorizes_behavioural_selection` all False, and the loaders refuse each other's schemas. It ENDS at `commit_top_k`; the products are fetched on that stage completing rather than on session success |
| `autoinit_phase_c2_baseline_launch.py` | **Phase C2 baseline completion** session specification. Deliberately THIN: Search-1's launcher defends a ten-hour beam over 87 GiB of intermediates, and this session rebuilds one checkpoint, measures it once and writes one record. So it declares no canonical control, no vLLM environment, no beam envelope and no conditional reserve, and asks for 60 GiB rather than 200. `SESSION_KIND=c2_baseline_completion`, which the shared setup script has its own branch for — a missing branch falls through to `spend`, loads the wrong authorization type and exits 98 after setup has run on a billing pod, which Phase-B attempt 2 paid `$0.2300` to establish. Seven `$0` gates: the commit/lineage gate, an independently re-derived completion closure, an authorization-scope gate that refuses any artifact claiming it can authorize the beam, the frozen comparison inputs and the ranking they were extracted from, the frozen-asset expectation, the pricing/protocol identities, and the volume. `BaselineCompletionAuthorization` is a distinct type reporting `authorizes_c2_search1 = False`, and the two loaders refuse each other's schemas, so a completion grant cannot buy a search |
| `autoinit_phase_c2_baseline_driver.py` | **Phase C2 baseline completion**: the half attempt 4 did not reach. Its beam completed and committed a ranking of five measured candidates; its conditional rebuild then hit a reserve that could not fund the work, so B carries no measurement and the B->C comparison was never computed. Two stages: `bind_identities` checks every identity the two halves must share — suite hash, policy hash and epsilon, teacher revision, the frozen B spec, and the evaluator's own source hashes — and loads nothing; `rebuild_measure_compare` rebuilds B through the complete frozen fixed path, measures it ONCE on the same suite through the same canonical-reload/state_eval path every searched candidate took, and computes the comparison against the frozen candidate record. **No beam search is reachable from it**: it names no search entry point, nothing it imports reaches `phase_a_search`, and it calls `BaselineFallback.rebuild` rather than the `conditional_candidates` hook a beam would call. No candidate is remeasured — the five arrive already measured, on an object with no method that could measure them again |
| `autoinit_phase_c2_full_search_driver.py` | **Phase C2 full joint re-search**, the successor to Search-1 and not a variant of it: `autoinit_phase_c2_driver.py` executes the frozen restricted space and couples the search to a conditional baseline rebuild, both of which are Search-1's and neither of which this session needs. Three stages and no fourth: `bind_identities` derives the joint space from the live registry and refuses an unmaterialized branch profile, loading no model; `full_joint_search` runs the beam through `run_phase_a_search` with `impl_profiles=None` — every calibration-consuming operator branching over every active mixture, which is the restriction Search-1 imposed and this session reopens — and no `conditional_candidates`, because B is measured and frozen and this session compares nothing; `commit_top_k` freezes the candidate set and **STOPS**. It trains nothing, measures no behaviour, and **has no code path into a behavioural stage**: a test parses its AST and refuses any call or import naming recovery, probes, `correct_overall`, screening, confirmation or `usable_rollout`, so one authorization cannot buy both a search and a promotion decision. Calibration profiles are registered at module import, not in the stage that needs them, because `get_profile` raises on an empty registry and this project has already lost a paid pod at stage D to exactly that |
| `autoinit_c2_replay_launch.py` | **Phase C2 replay-only artifact reconstruction** session specification. The full joint re-search committed its Top-5 and lost the weights, so this session rebuilds the five checkpoints behind a selection that is already frozen and accepted, and decides nothing. `SESSION_KIND=c2_replay`. Six `$0` gates; `source_binding_gate` is the one with no counterpart elsewhere — it re-checks with `git ls-tree` that no operator-bearing file has moved since attempt 3's session commit `2421f630`, because a digest-pinned replay whose operators moved fails on a paid pod to learn what costs nothing here. `ReplayAuthorization` reports `authorizes_c2_full_search`, `authorizes_c2_search1`, `authorizes_c2_baseline_completion` and `authorizes_behavioural_selection` all False: the search is COMPLETE and an artifact able to buy a beam would buy the one thing the review forbade, at a tenth of the price. Unlike the full-search launcher it declares `fetch_products` — the product here IS weights, and each leaf is re-identified from the bytes that land on the destination before teardown is permitted |
| `autoinit_c2_replay_driver.py` | the replay driver: five `FixedPathSpec`s pinned at EVERY step to the artifact digest attempt 3 recorded, not merely at the leaf, so a compensating pair of errors cannot pass as a correct replay. A `FixedPathDigestMismatch` is terminal and is reported as a scientific finding rather than retried — the path is deterministic and would diverge identically. Each finished leaf is copied clear of its path, given a sidecar and announced before the next path starts, so a failure at leaf N leaves 1..N-1 safe. Admission control, not a trip-wire: a path begins only when the remaining soft-stop budget can fund that path's full bound, derived from attempt 3's own telemetry |
| `autoinit_recovery_continuation_driver.py` | its pod-side driver. Stage 1 IMPORTS the verified attempt-12 result — it never imports `phase_a_search`, never delegates to the searching `stage1`, and has no `--stage` value that searches |
| `autoinit_preflight_driver.py` | the micro-preflight's pod-side driver |
| `autoinit_phase_a_driver.py` | the Phase-A pod-side driver, six stages. Also the **base class Phase B inherits**, so it is inside the Phase-B executable-source identity too |
| `autoinit_phase_b_launch.py` | Phase-B **session specification**: joint P=2 search, `PhaseBAuthorization`, and **ten** priced probes rather than twelve because three candidates are cited from verified Phase-A evidence. Prechecks refuse at `$0` on an unbound executable, a stale preregistration or an unverified reuse record |
| `autoinit_phase_b_driver.py` | the Phase-B pod-side driver. Subclasses the Phase-A driver and overrides four things: the governing artifacts, stage 0's additional bindings, the joint two-profile search, and `restore_probe`'s comparability rule for imported evidence. It redirects the inherited `mark()` to the status file the Phase-B launcher polls |
| `autoinit_continuation_b_launch.py` | the **behavioural continuation** session specification. Phase-B Stage 1 is complete and retained, so this session buys only the missing `sb` and at most two conditional `sc`: ceiling `$8.0691` against the full session's `$35.6660`. `ContinuationAuthorization` is a distinct type whose `runs_search` is `False` **by type**, `SESSION_KIND=continuation_b` has its own dispatch branch, the budget strips the search phase and both soft-stop reserves, and the poll lifetime is derived from THIS session's plan rather than Phase B's 32 h one |
| `autoinit_continuation_b_driver.py` | the continuation pod-side driver. Subclasses the Phase-A driver; its stage map is `0,1,3,4,5` with **no stage 2**, because Phase A's stage 2 is rung 1 on seed `sa` and this session imports that result. `stage1()` and `run_search()` raise, `enter()` orders against the continuation plan, `stage_bind` runs the inherited stage 0 before binding six cited identities, and `restore_probe` cites imported evidence by comparability. Six evidence candidates narrow to three active finalists before any probe |
| `autoinit_preflight_setup.sh` | the shared pod setup. **Manifest-driven** since 2026-08-18: it reads `SESSION_ASSETS`, `SESSION_RELAY_INPUTS` and `SESSION_TEST_IGNORES`, and names no session's assets, relay paths, destinations or digests itself |
| `autoinit_science_inputs.py` | the frozen relay science inputs — source, destination, digest — that sessions compose their `relay_inputs` from. Lifted out of the shared setup on 2026-08-18; here rather than in `src/` because `docs/REPO_LAYOUT.md` rule 1 keeps frozen hashes in the scripts that own them |
| `autoinit_continuation_driver.py` | the continuation's pod-side driver |
| `autoinit_engine_probe.py` | vLLM engine identity probe, run at stage 0 |
| `watchdog.py` | the independent provider-side kill switch |
| `collect_artifacts.py` | artifact manifest, gate and collection |
| `cpu_test_env_args.py` | emits the C1 CPU-test environment as `env(1)` arguments, from the one declaration in `aadistill.autoinit.cpu_test_env`. The pod's gate and the dev-box simulator both consume it, so the diagnostic and the paid pod run pytest under the same hardware- and cache-neutral scope |
| `summarize_pytest_outcomes.py` | the CPU gate's complete outcome — every FAILED, ERROR and SKIPPED nodeid with reasons, a skip-set digest, and the exact set difference against the launch-bound sweep. `--strict` refuses a pod whose suite passed but whose skip set is not the one the sweep certified |
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
starting point is `logs/state/current.md`, not this directory.

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
| `checkpoint_inventory.py` | `scripts/consolidate/build_checkpoint_registry.py`. Both inventory both stores; this one's `REQUIRED` set is written around Experiment 2 and has not moved since, and two inventories with different stale opinions is worse than one. Its LFS insight — that removing a file from a Hugging Face repo's current revision reclaims no quota — is preserved in the replacement's docstring and in `logs/shared/validations/relay-mirror/relay_mirror_verification.json` |

## Local notes

`AGENTS.md` (that is, `scripts/pod/AGENTS.md`) carries directory-local rules for this area and is the
authority on how a pod session must be started, watched and torn down.
