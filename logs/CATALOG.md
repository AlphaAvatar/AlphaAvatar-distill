# Log catalog and fact ownership

Which file owns which fact, and what "current", "historical", "superseded" and
"terminated" mean here. A repository this size fails by having the same fact in
four places and three of them stale; this page exists to make that a rule
violation rather than a discovery.

## Ownership rules

**One owner per fact.** A file that does not own a fact links to the file that
does. It does not restate it.

| class | meaning | may be edited? |
| --- | --- | --- |
| **CURRENT** | describes the repository *now*. Exactly one owner per fact. | yes — that is the point |
| **HISTORICAL** | evidence of something that happened. | **no.** Append a superseding note elsewhere; never rewrite |
| **SUPERSEDED** | was current, has been replaced, kept for provenance | no — mark, do not edit |
| **TERMINATED** | a path deliberately stopped. Evidence kept, work not resumed | no |
| **REFERENCE** | durable design/protocol material. Changes when the design changes, not when a run finishes | yes, with a decision record |

**The three live-fact owners, and nobody else:**

| fact | owner |
| --- | --- |
| everything current, machine-readable | [`current_state.json`](current_state.json) |
| the same, in prose | [`STATE.md`](STATE.md) — must agree with the above |
| spend, caps, authorizations | [`BUDGET_LEDGER.md`](BUDGET_LEDGER.md) |

`README.md` owns **no** live facts: no spend, no cap, no attempt count, no
authorization status, no "current state" table. It is a timeless public
overview and links here instead. A structural test enforces this.

## CURRENT

| file | owns |
| --- | --- |
| `current_state.json` | the minimal machine snapshot: budget, running, authorized, frozen identities, next starting point |
| `STATE.md` | the human view of the above |
| `BUDGET_LEDGER.md` | actual spend, the cap history and every authorization's status |
| `decisions.md` | why things are the way they are — append-only in practice |
| `EXPERIMENT_INDEX.md` | what each experiment proved, and what it does not support |
| `EXPERIMENTS.md` | per-session chronology |
| `checkpoint_registry.json` | every checkpoint and weight artifact in all three storage areas — repo `artifacts/`, the out-of-tree store `/home/ecs-user/aad-artifacts`, and the relay — with hash, references, relay correspondence and proposed disposition |
| `checkpoint_tombstones.json` | what was deleted and how to rebuild it |
| `log_inventory.json` | every documentary file with its hash, class and disposition, the byte-identical duplicate groups and which copy is canonical, and the record of copies already removed |
| `derived_cache_cleanup_20260827.json` | the 2026-08-27 reclamation of 1.5000 GiB of rebuildable derived caches (HF datasets/xet, pre-commit, pip) that cleared `ckpt_store_capacity_gate` for Phase B. Deliberately **not** tombstones — every path is recreated by a routine command — and it records what was explicitly *not* touched |
| `derived_cache_cleanup.json` | derived-cache reclamation events (uv build cache, upstream teacher weight blobs) — deliberately **not** tombstones, because both paths are recreated by routine commands. Carries verified blob hashes, the pinned reconstruction revision and the resulting free space |
| `storage_measurements.json` | storage in the four areas, before and after each cleanup, measured the same way |
| `autoinit_depth_backend_equivalence.json` | the causal-depth runtime repair's equivalence proof: the frozen greedy rule by known answer, the refactor bit-identical, the bf16 cache exact, and the per-round decision margins |
| `autoinit_measurement_authorization.json` | the bounded measurement's one-use authorization — a `SpendAuthorization`, so `allows_phase_a` is False by type |
| `autoinit_measurement_session.json` | that session's record |
| `autoinit_causal_depth_pricing_bound.json` | the DERIVED Stage-1 wall-time and VRAM bound for the repaired path, and the measurement protocol that would confirm it |
| `relay_mirror_verification.json` | file-by-file proof that a local tree is byte-identical to its relay copy, and by which method each file was checked. The evidence a "stale local cache" deletion stands on |
| `autoinit_historical_probe_reuse.json` | the strict reconstruction check Phase B's probe reuse is conditional on, regenerable with `scripts/autoinit/verify_historical_probe_reuse.py`: per probe, completeness, the frozen seed, the artifact digest **re-derived from the retained checkpoint bytes**, the battery and scoring-contract identities, and the attested protocol hash. Also records the one leg that cannot be closed at `$0` — Phase B's own runtime comparability, a Stage-0 precondition |
| `autoinit_phase_b_attempt1.json` | Phase-B attempt 1: ABORTED at the pod-side test gate after 8.8 min and **$0.15**. No scientific stage ran. Records what passed on hardware, the two defects that stopped it, and what must change before a retry |
| `autoinit_phase_b_attempt2.json` | Phase-B attempt 2: the pod **test gate passed** (2207 passed), then ABORTED at the authorization-binding step after 13.9 min and **$0.23**. No scientific stage ran. Records the SESSION_KIND dispatch gap and the two ways to close it |
| `autoinit_phase_b_attempt3.json` | Phase-B attempt 3: **Stage 0 PASSED**, Stage 1 exhausted its 544 min allowance and stopped fail-closed. 575.9 min, **$9.50**, no Top-5 and no rungs — a **null result**, not evidence about either calibration distribution. Carries the Stage-0 bindings, the reconstructed DEPTH cost, the search-journal evidence gap and the repricing finding |
| `autoinit_phase_b_attempt3/` | that attempt's retained evidence: Stage-0 binding, evidence file, Stage-1 traceback, gate logs, filtered driver log, and the hashes of the 8 imported historical citations |
| `autoinit_phase_b_session.json` | the launcher's own session record for the most recent attempt |
| `autoinit_phase_b_grant.json` | the ONE-USE maintainer decision authorizing a single Phase-B execution: who permitted what, the cumulative spend at approval, the new cap, and an explicit list of what it does **not** authorize. An input to the issuer, never a constant in executable source. **Consumed by attempt 1** |
| `autoinit_phase_b_grant_attempt3.json` | the ONE-USE maintainer decision for Phase-B **attempt 3**, after attempt 2 aborted at the authorization gate: Option A (an explicit `phase_b` dispatch branch), cap `$256.99` → `$257.22` replacing attempt 2's lost setup cost, session ceiling and floor unchanged |
| `autoinit_phase_b_grant_retry.json` | the ONE-USE maintainer decision for the Phase-B **retry** after attempt 1 aborted in setup: cap `$256.84` → `$256.99`, restoring the retry envelope attempt 1 spent, with the `$26.8049` session ceiling and `$13.0800` floor unchanged. Does not revive attempt 1's consumed authorization |
| `autoinit_phase_b_authorization.json` | the issued one-use `PhaseBAuthorization` for the **current** launch, binding the session commit, executable-source digest, preregistration, both plan hashes, both calibration spec **and** content identities, the historical-reuse verdict, and the `$26.8049` hard ceiling. Written by `scripts/autoinit/issue_phase_b_authorization.py`, which refuses to overwrite an existing one. Attempt 1's is consumed and retired to `superseded/` |
| `autoinit_phase_b_pricing.json` | the paid work Phase B still owes under the 2026-08-25 terminal procedure, regenerable with `scripts/autoinit/price_phase_b.py`: the historical probe inventory read off disk, which candidate/seed pairs are reusable, the probes still owed at both ends of a coherent best/worst case, and the search's P=2 cost and storage. **Not an authorization** — the binding per-launch ceiling is issued by the authorization code against a frozen plan |
| `autoinit_phase_b_preregistration.json` | **the frozen Phase-B preregistration**, emitted before any Phase-B result exists by `scripts/autoinit/write_phase_b_preregistration.py`: both calibration identities (spec *and* materialized content), the joint P=2 search, beam/schedule/policy, the operator branching rule, the executable-source digest, the Stage-0 fail-closed comparability condition, the closed cross-phase candidate set with its exclusions, the rung procedure and what may **not** break a tie. Self-verifying via `preregistration_sha256`. Not an authorization |
| `autoinit_phase_b_reconstruction.md` | what Phase B **is** according to this repository, what blocks it, and what it would cost — the reconstruction of the committed design, its three open scientific choices, the arithmetic showing `calib.reasoning_heavy@v1` is unbuildable as specified, and the readiness checklist. Owns no live spend or authorization fact; it links to `BUDGET_LEDGER.md` for those |
| `supported_models.md` | the supported-model table (AGENTS.md §3.4) |
| `artifact_manifests.md` | external artifact manifests |
| `CATALOG.md` | this file |

## REFERENCE — frozen science and protocol

Byte-for-byte preserved. These are cited by hash from executable code and from
consumed authorizations; changing one silently invalidates recorded results.

| file | what it pins |
| --- | --- |
| `autoinit_phase_a_recovery_plan_frozen.json` | the frozen **science** plan, `02be33b9…` |
| `autoinit_phase_a_preregistration.json` / `.md` | the preregistration (draft) |
| `autoinit_phase_a_preregistration_materialized.json` | the materialized preregistration |
| `autoinit_v1_search_space.json` | the search space, its cost model and the operator ledger |
| `autoinit_state_eval_v1_manifest.json` | the `state_eval@v1` asset identity |
| `autoinit_recovery_search_v2_build.json` / `_audit.json` | the battery's build and its $0 audit |
| `autoinit_threshold_characterization.json` | how the thresholds were derived |
| `autoinit_phase_a_protocol_compat_v2.json` | `generation_runtime_comparability@v2` |
| `autoinit_stage1_device_audit.json` | `autoinit.stage1_device_contract@v1` |
| `autoinit_phase_a_fallback_audit.json` | the reference-cache fallback derivation and pricing |
| `autoinit_phase_a_full_mixture_depth.json` | the full-mixture depth probe |
| `autoinit_phase_a_preregistration.md` | the preregistration in prose |
| `autoinit_recovery_search_v2_audit.json` | the battery's $0 item-by-item audit |

## HISTORICAL — evidence, never rewritten

`superseded/` holds artifacts that were superseded before use and are kept
as evidence rather than deleted — currently the Phase-B authorization that was
invalidated by the naming defect **before provider creation, at $0 spend**. Files
here are never revived or reused; the `_UNUSABLE` suffix says so in the name.


Per-run directories. Each holds what a session actually produced.

| directory | session |
| --- | --- |
| `autoinit_stage3_complete/` | **Stage 3 COMPLETE** — both permanent controls, thresholds materialized |
| `autoinit_permanent_controls/` | the controls' own evidence |
| `autoinit_preflight_run4/` | the micro-preflight's fourth and last attempt |
| `autoinit_continuation_attempts/` | the eight characterization-continuation attempts |
| `autoinit_phase_a_attempts/` | Phase-A attempts 1–5 |
| `autoinit_phase_a_attempt6/`, `autoinit_phase_a_attempt7/` | attempts 6 and 7, with attempt 7's stage-1 traceback |
| `autoinit_phase_a_attempt8/` | attempt 8 — failed closed at the setup test gate, $0.19, no stage ran. Records what DID pass on hardware: the manifest-driven relay staging, the pod-side frozen-asset gate, `ROPE_OK` |
| `autoinit_phase_a_attempt9/` | attempt 9 — **Stage 0 passed and attested**, Stage 1 failed on device placement, $0.34. Carries the recovered `stage1_traceback.log` and the Stage-0 attestation |
| `autoinit_measurement_attempt1/` | the bounded measurement's first launch — **failed closed at the frozen-asset gate**, $0.07, no measurement ran. Its grant and authorization are consumed |
| `autoinit_phase_a_attempt10/` | attempt 10 — **incomplete, operator runtime-cost failure**, $11.43. Stage 0 passed; `depth.causal_kl_greedy_v1` ran 10 h 47 m without finishing, GPU idle. Carries `search_states.jsonl` (2 states, specs only — weights gone with the pod) |
| `autoinit_measurement_attempt3/` | the bounded measurement's third launch — **COMPLETE**, $0.2077, `ALL_DONE`. Carries `result.json` (the reviewed artifact), the driver log with all 24 timings, and the artifact manifest. Its grant and authorization are consumed |
| `autoinit_measurement_attempt2/` | the bounded measurement's second launch — **setup passed end to end (`SETUP_RC=0`); the driver died in its entrypoint's first repository import**, $0.1834, no measurement ran. Carries the driver traceback. Its grant and authorization are consumed |
| `autoinit_recovery_continuation_attempt2/` | continuation attempt 2 — **no stage ran**, $0.2389. Every gate passed and the resilience closure held (TCP 22 at 3.7 min, image identity confirmed); it then failed staging the first 1.110 GiB leaf against a hard-coded 600 s scp timeout that needs 1.99 MB/s. Carries the launcher log, session record and watchdog journal |
| `autoinit_recovery_continuation_attempt7/` | continuation attempt 7 — **PHASE A COMPLETE as a SELECTION experiment**, `ALL_DONE`, $12.8587. All six stages, 11 matched 0.86M *selection* probes across seeds sa/sb/sc — not recovery training. Result `unresolved_equivalence`, winner `None`: `cca699c93f34` is the provisional leader, separated from the canonical **initialization** control `qwen3_0p6b_init_v0`, and not separated from `85bde4ded2c3`. Carries `phase_a_result.json`, both rung selections, leaf retention, all 11 probe records, the attested protocol, the driver run log and the launcher/watchdog/poller journals. The launcher crashed AFTER collection on a `fetch_products` contract mismatch, mislabelling the session INCOMPLETE |
| `autoinit_recovery_continuation_attempt6/` | continuation attempt 6 — **the attempt-5 checkpoint repair CONFIRMED on hardware and the battery reached**, $1.4926. `trained_model_dir()` resolved the probe checkpoint with no error; generation then failed because a trainer-written checkpoint carries no tokenizer, so `apply_chat_template` raised. Records the protocol finding that passing `--tokenizer` is NOT neutral: `tokenizer_source` is material under `generation_runtime_comparability@v2`. Carries the driver run log, the attested protocol, the handoff and import records, the probe train tail, and the launcher, watchdog and poller journals |
| `autoinit_recovery_continuation_attempt5/` | continuation attempt 5 — **both memory repairs VERIFIED on hardware and the first recovery probe TRAINED**, $1.3511. `freed_allocated_bytes` 8,101,709,824 against attempt 4's 0, `live_retention` false, 43.87 GiB free against the 43.65 GiB gate; `PROBE_TRAINED` after 61.7 min against 61.55 priced. Stage 2 then failed reading `latest.txt` without the `checkpoints/` component the trainer writes. Carries the driver run log, the stage-2 traceback, the handoff and stage-1 import records, the probe train tail, and the launcher, watchdog and poller journals |
| `autoinit_recovery_continuation_attempt4/` | continuation attempt 4 — **Stages 0 and 1 PASSED on hardware**, Stage 2 OOM, $0.4112. The five leaves were imported and re-identified from pod bytes and the canonical control was measured on `state_eval@v1`; the first rung-1 probe then OOM'd because `release_to_subprocess` freed 0 allocated bytes and `require_headroom` demands 24 GiB for a probe needing ≳36.6. Carries the driver run log, the stage-1 import, control measurement and device-handoff records, the probe train tail, and the launcher, watchdog and poller journals |
| `autoinit_recovery_continuation_attempt3/` | continuation attempt 3 — **no stage ran**, $0.2011. **The 25-input multi-repo transport staging PASSED on the pod**, proving the five-leaf mirror end to end; setup then failed its CPU test gate because `publish_selected_leaves.verify()` mkdtemps into the dev-box-only `/home/ecs-user/aad-scratch`. Root cause reproduced at $0 in a mount namespace. Carries the launcher log, session record, watchdog journal and the independent poller journal |
| `autoinit_recovery_continuation_attempt1/` | continuation attempt 1 — **no stage ran**, $0.0100. Every gate passed and the pod died 27 s later when `wait_endpoint` hit an uncaught `URLError` against an endpoint measured at 25% transport failure. Carries the launcher log, the session record and the 20-request endpoint probe |
| `autoinit_phase_a_attempt12/` | attempt 12 — **Stage 1 PASSED and its five leaves were preserved off-pod**, $3.7872. Stage 2 failed on CUDA OOM (driver holds ~24 GiB from the in-process search). Carries the durability report, the OOM traceback and the search result, which is byte-identical to attempt 11's |
| `autoinit_phase_a_attempt11/` | attempt 11 — **Stage 1 PASSED**, $3.2101. The first completed AutoInit beam search: 43 states, 5 leaves selected, the composite baseline beaten. Stage 2 failed closed on a tokenizer guard. Carries `search_result.json`, `search_states_reduced.jsonl` and the Stage-2 traceback; the 25 MB full state journal is out of tree (see `search_states_FULL_RECORD.md`) |
| `autoinit_device_canary_attempt1/`, `autoinit_device_canary_attempt2/` | the two terminated canary sessions |
| `e7_canary/`, `e7_canary_rerun/` | the control-plane canary runs |
| `e8b_s2_dp_sa/`, `e8b_step0_records/` | E8b evidence |

Grant documents — the one-use maintainer decision an authorization is issued
*against*, kept because a spent grant records what was permitted and by whom:
`autoinit_phase_a_attempt8_grant.json`,
`autoinit_phase_a_attempt9_grant.json`,
`autoinit_phase_a_attempt10_grant.json`,
`autoinit_phase_a_attempt11_grant.json` (consumed),
`autoinit_phase_a_attempt12_grant.json` (consumed),
`autoinit_measurement_grant.json` (consumed),
`autoinit_measurement_grant2.json` (consumed),
`autoinit_measurement_grant3.json` (consumed),
`autoinit_recovery_continuation_grant.json` (consumed — single-issuance),
`autoinit_recovery_continuation_attempt2_grant.json` (consumed — single-issuance),
`autoinit_recovery_continuation_attempt3_grant.json` (consumed — single-issuance),
`autoinit_recovery_continuation_attempt4_grant.json` (consumed — single-issuance),
`autoinit_recovery_continuation_attempt5_grant.json` (consumed — single-issuance),
`autoinit_recovery_continuation_attempt6_grant.json` (consumed — single-issuance),
`autoinit_recovery_continuation_attempt7_grant.json`. The issuer requires one, refuses a grant
that asserts an identity it did not compute, and hashes it into the artifact.

Consumed authorizations — kept because a spent grant is the record of what was
permitted: `autoinit_phase_a_authorization.json`,
`autoinit_device_canary_authorization.json`,
`autoinit_micro_preflight_authorization.json`,
`autoinit_continuation_authorization.json`,
`autoinit_recovery_continuation_authorization.json`. Each one's lineage gate refuses the
current HEAD by construction, which is what "consumed" means operationally.

Session records: `autoinit_phase_a_session.json`,
`autoinit_device_canary_session.json`, `autoinit_preflight_session.json`,
`autoinit_continuation_session.json`, `autoinit_recovery_continuation_session.json`,
and the `e*_session_evidence.json` family.

Experiment records: everything named `e1_*` through `e8b_*`.

$0 audits and measurements, each the evidence behind a decision that cites it:
`autoinit_leaf_transport_quota_finding.json`, `autoinit_relay_retention_20260822.json`,
`autoinit_selected_leaf_transport_manifest.json` (the verified five-leaf transport mirror),
`autoinit_recovery_trainer_memory_basis.json` (the measured basis for RECOVERY_TRAINER_BYTES),
`autoinit_control_availability.json`,
`autoinit_control_sb_packaging_repair.json`,
`autoinit_dryrun_fresh.json`, `autoinit_dryrun_resume.json`,
`autoinit_recovery_fingerprint_audit.json`,
`autoinit_recovery_scoring_validation.json`,
`autoinit_repeatability_cpu_smoke.json`, `autoinit_role_isolation.json`,
`autoinit_tool_rendering_audit_tf5.json`, `autoinit_tool_scoring_audit.json`,
`autoinit_relay_capacity.md`, `autoinit_phase_a_storage.md`,
`autoinit_tool_rendering_migration.md`.

## SUPERSEDED — kept for provenance, not for use

| file | superseded by |
| --- | --- |
| `archive/` | earlier proposals and preregistrations; see its own README |
| `archive/current_state_20260817_full.json` | **removed 2026-08-18.** It was a snapshot of a living-state file, byte-identical to `git show 3261f6b6…:logs/current_state.json`; that reference replaces it, and the equivalence is verified on every inventory run |
| `autoinit_phase_a_repricing.md` | superseded **twice** — $20.0126, then $22.4508, now $23.0483. Kept because the `gpu_fraction` measurements in it are not recorded anywhere else |
| `autoinit_recovery_search_v1_manifest.json` | `recovery_search_v2`. v1 was INVALID before first use — 0/20 tool prompts rendered |
| `autoinit_pilot_proposal.md`, `autoinit_micro_preflight_plan.md`, `autoinit_preflight_remaining_gates.md` | executed and closed; the sessions' own evidence is authoritative |
| `e5_proposal.md`, `e7_canary_proposal.md`, `e7_preregistration.md` | their runs |

## TERMINATED — deliberately stopped

| path | why |
| --- | --- |
| **the paid device-canary session** | two authorized sessions, $0.1240, **zero canary runs** — both died in the wrapper's inherited contracts, neither reached the canary script. Evidence in `autoinit_device_canary_attempt{1,2}/`; the generic lesson is in `decisions.md` and is now enforced by the session specification. **No further canary is prepared.** |
| **E8b** | strategically terminated; no valid recovered-behaviour comparison |
| **student-prefix recovery (E5)** | prefix-conditioned targets teach continuation, not closure |
| **`recovery_search_v1`** | invalid before first use; preserved unmodified with a sibling `SUPERSEDED.md` |

## One copy of every raw artifact

A run's evidence directory holds what that run produced. When two runs produced
**byte-identical** bytes — the same engine probe, the same control import, the
same session record written to a second name — one copy is kept and the others
are replaced by a `README.md` in the directory that lost the file, naming the
survivor and carrying the sha256. The evidence chain stays complete by reference,
and the hash is the part that was ever load-bearing.

Six such copies were removed on 2026-08-18 (76.7 KiB). Which file, which hash and
which survivor is in [`log_inventory.json`](log_inventory.json) under `removed`,
which survives regeneration by design.

## Adding a log

1. Decide its class from the table above.
2. If it is CURRENT, name its owner here and make sure nothing else claims the
   same fact.
3. If it supersedes something, mark the old entry — do not delete or edit it.
4. `tests/docs/test_repository_structure.py` requires every top-level entry in
   `logs/` to be classified here.
