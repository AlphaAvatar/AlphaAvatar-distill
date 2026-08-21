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
| `derived_cache_cleanup.json` | derived-cache reclamation events (uv build cache, upstream teacher weight blobs) — deliberately **not** tombstones, because both paths are recreated by routine commands. Carries verified blob hashes, the pinned reconstruction revision and the resulting free space |
| `storage_measurements.json` | storage in the four areas, before and after each cleanup, measured the same way |
| `autoinit_depth_backend_equivalence.json` | the causal-depth runtime repair's equivalence proof: the frozen greedy rule by known answer, the refactor bit-identical, the bf16 cache exact, and the per-round decision margins |
| `autoinit_measurement_authorization.json` | the bounded measurement's one-use authorization — a `SpendAuthorization`, so `allows_phase_a` is False by type |
| `autoinit_measurement_session.json` | that session's record |
| `autoinit_causal_depth_pricing_bound.json` | the DERIVED Stage-1 wall-time and VRAM bound for the repaired path, and the measurement protocol that would confirm it |
| `relay_mirror_verification.json` | file-by-file proof that a local tree is byte-identical to its relay copy, and by which method each file was checked. The evidence a "stale local cache" deletion stands on |
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
`autoinit_recovery_continuation_attempt2_grant.json` (consumed — single-issuance). The issuer requires one, refuses a grant
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
`autoinit_control_availability.json`, `autoinit_control_sb_packaging_repair.json`,
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
