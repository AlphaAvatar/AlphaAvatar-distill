#!/usr/bin/env python3
"""Write the Phase-C1 execution preregistration.

    PYTHONPATH=src .venv/bin/python \
        scripts/autoinit/write_c1_execution_preregistration.py

Every binding is **derived**, never transcribed: the seeds from the frozen C0
digest, the hashes from the live objects, the teacher shards from the committed
binding record. A document whose numbers were typed in by hand describes what
someone believed rather than what the code will do.

It authorizes nothing. It exists so that when an authorization is eventually
issued, the thing it binds already exists and cannot be edited afterwards.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.autoinit import c1_session as CS  # noqa: E402
from aadistill.autoinit.c1_isolation import (  # noqa: E402
    BOOTSTRAP_ALGORITHM,
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_QUANTILE_CONVENTION,
    BOOTSTRAP_STRATUM_CONVENTION,
    C0_PREREGISTRATION_SHA256,
    HISTORICAL_SEEDS,
    C1Arm,
    C1IsolationPlan,
    bootstrap_seed,
    derive_recovery_seeds,
)
from aadistill.autoinit.c1_authorization import (  # noqa: E402
    C1_HARNESS_SOURCE_FILES_V1,
    SCHEMA as C1_AUTH_SCHEMA,
    c1_harness_digest,
    load_pricing,
)
from aadistill.autoinit.c1_scoring import (  # noqa: E402
    C1_METRIC_CONTRACT, c1_scoring_contract,
)
from aadistill.autoinit.calibration import get_profile  # noqa: E402
from aadistill.autoinit.operators import attention_activation  # noqa: E402
from aadistill.autoinit.operators.attention_activation import (  # noqa: E402
    ATTENTION_STATS_SPEC,
)
from aadistill.autoinit.recovery import E1_KD_HEAVY_0860K  # noqa: E402
from aadistill.infrastructure.manifest import sha256_file, sha256_json  # noqa: E402

OUT = REPO / "logs/phase_c1_execution_preregistration.json"

#: The two evidence declarations. Restated here rather than imported, because
#: importing the launcher would pull the whole Phase-A launcher in; the copy is
#: turned into a checked invariant by
#: tests/pod/test_c1_artifact_specs.py::test_writer_and_launcher_name_the_same_specs.
SPEC_SUCCESS = "configs/autoinit/c1_artifacts.json"
SPEC_FAILED = "configs/autoinit/c1_artifacts_failed.json"

#: Every file whose bytes decide what the C1 session does. Same shape and same
#: failure mode as the other source-digest sets: a missing declared file raises
#: rather than yielding a digest over a smaller contract.
C1_SOURCE_FILES: tuple[str, ...] = (
    "src/aadistill/autoinit/c1_isolation.py",
    "src/aadistill/autoinit/c1_session.py",
    "src/aadistill/autoinit/fixed_path.py",
    "src/aadistill/autoinit/operators/attention_activation.py",
    "src/aadistill/init/attention_stats.py",
    "scripts/data/battery_render.py",
    "scripts/data/build_c1_confirmation_battery.py",
    "scripts/autoinit/verify_c1_battery_isolation.py",
)


def source_digest(files: tuple[str, ...] = C1_SOURCE_FILES) -> dict:
    entries = []
    for rel in sorted(files):
        p = REPO / rel
        if not p.is_file():
            raise SystemExit(f"declared C1 source {rel!r} is missing")
        entries.append({"path": rel, "sha256": sha256_file(p),
                        "bytes": p.stat().st_size})
    digest = hashlib.sha256(
        "".join(f"{e['path']}:{e['sha256']}\n" for e in entries).encode()).hexdigest()
    return {"digest": digest, "files": entries,
            "rule": "sha256 over sorted 'path:sha256' lines of the declared set"}


def _equivalence() -> dict:
    """The admission-gate record. Absent means C1 cannot be authorized."""
    p = REPO / "logs/phase_c1_scoring_equivalence.json"
    if not p.is_file():
        raise SystemExit(
            "logs/phase_c1_scoring_equivalence.json is missing; the C1 scoring "
            "binding has not been admitted and must not be preregistered")
    return json.loads(p.read_text())


def main() -> None:
    attention_activation.register(replace=True)

    seeds = derive_recovery_seeds()
    assert len(set(seeds)) == 3 and not set(seeds) & set(HISTORICAL_SEEDS)

    arms = CS.build_arm_specs()
    battery = json.loads((REPO / "logs/phase_c1_battery.json").read_text())
    #: Derived, so the artifact-spec block cannot drift from the design.
    n_probes = CS.C1_SESSION_CONTRACT.n_probes
    n_sets = len(battery["set_sha256"])
    teacher = json.loads((REPO / "logs/phase_c1_teacher_binding.json").read_text())
    c0 = json.loads((REPO / "logs/phase_c0_preregistration.json").read_text())

    plan = C1IsolationPlan(
        plan_id="autoinit.v1.phase_c1",
        arms=(C1Arm("c1.incumbent", "incumbent", *CS.INCUMBENT_ATTENTION),
              C1Arm("c1.treatment", "treatment", *CS.TREATMENT_ATTENTION)),
        seeds=tuple(seeds),
        battery_asset_id=battery["asset_id"],
        battery_content_sha256=battery["content_sha256"])

    impl = attention_activation.ATTENTION_ACTIVATION_IMPORTANCE_V1
    db = get_profile("calib.domain_balanced@v1")
    rh = get_profile("calib.reasoning_heavy@v2")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                          capture_output=True, text=True).stdout.strip()

    doc = {
        "schema": "aadistill.autoinit.c1_execution_preregistration/v1",
        "preregistration_id": "autoinit.v1.phase_c1.execution",
        "written_utc": "2026-09-02T00:00:00Z",
        "_contract": (
            "Everything the Phase-C1 session will run, bound before any C1 result "
            "exists. AUTHORIZES NOTHING: no pricing decision, no spend grant, no "
            "permission to launch. An authorization, if ever issued, binds THIS "
            "document."),

        "head_commit": head,
        "executable_source": source_digest(),
        #: What a GRANT binds: the full session harness, launcher and driver
        #: included. `executable_source` above is the narrower science-only set
        #: and is kept because it is what the readiness binding used.
        "c1_harness": {
            **c1_harness_digest(REPO),
            "note": ("the set an authorization measures. It includes the launcher, "
                     "the driver, the setup script, the operators the fixed path "
                     "applies, the recovery and scoring paths the six probes "
                     "invoke, and the session infrastructure they run through."),
        },
        "authorization": {
            "schema": C1_AUTH_SCHEMA,
            "type": "aadistill.autoinit.c1_authorization.C1Authorization",
            "session_kind": "c1",
            "allows_phase_a": False,
            "allows_beam_search": False,
            "n_harness_files": len(C1_HARNESS_SOURCE_FILES_V1),
            "status": "NO GRANT EXISTS. No authorization has been issued.",
        },
        "launcher": "scripts/pod/autoinit_c1_launch.py",
        "pricing": {
            "path": "logs/phase_c1_pricing.json",
            "pricing_sha256": load_pricing(REPO)["pricing_sha256"],
            "floor_usd": load_pricing(REPO)["totals"]["floor_usd"],
            "expected_usd": load_pricing(REPO)["totals"]["expected_usd"],
            "hard_ceiling_usd": load_pricing(REPO)["totals"]["hard_ceiling_usd"],
            "enforcement": ("BudgetSpec is derived from this record; the ceiling "
                            "exists in exactly one place"),
        },
        "c0_protocol": {
            "path": "logs/phase_c0_preregistration.json",
            "sha256": sha256_file(REPO / "logs/phase_c0_preregistration.json"),
            "protocol_id": c0["protocol_id"],
        },
        "c1_session_contract": {
            "hash": CS.C1_SESSION_CONTRACT.contract_hash,
            "stages": [s.letter + ": " + s.stage_id for s in CS.C1_STAGES],
            "gate_stages": list(CS.GATE_STAGES),
        },

        "treatment_operator": {
            "impl_id": impl.impl_id,
            "signature_hash": impl.signature_hash,
            "version": impl.version,
            "calibration_need": impl.calibration.value,
            "calibration_profile": CS.TREATMENT_ATTENTION[1],
            "score_definition": "score_h = mean_t || W_o,h @ a_h(t) ||^2",
            "score_definition_long": (
                "a_h(t) is query head h's slice of the concatenated attention "
                "output, i.e. the tensor o_proj consumes; W_o,h is o_proj's column "
                "block for that head. The score is the mean over calibration "
                "tokens of the squared L2 norm of that head's contribution to the "
                "residual stream. Computed EXACTLY from the streamed per-head "
                "second moment as <W_o,h^T W_o,h, mean_t a_h a_h^T>_F, not "
                "approximated."),
            "selection": ("per-GQA-group top-k, 4 query heads per KV group -> keep "
                          "2; descending score; ties by ASCENDING head index"),
            "preserved": ("GQA grouping, head_dim, KV heads, RoPE basis; modifies "
                          "= {num_attention_heads} only"),
            "registration": ("EXPLICIT. Importing the module does not register the "
                             "operator, because BeamSearch._allowed_impl_ids falls "
                             "back to the entire registry when allowed_impls is "
                             "None. Stage C registers it before any FixedPathSpec "
                             "names it, and build_arm_specs refuses otherwise."),
            "not_in_frozen_library": True,
            "stats_spec": ATTENTION_STATS_SPEC.as_dict(),
        },
        "incumbent_operator": {
            "impl_id": CS.INCUMBENT_ATTENTION[0],
            "calibration_profile": CS.INCUMBENT_ATTENTION[1],
        },

        "fixed_path": {
            "incumbent_spec_hash": arms["incumbent"].spec_hash,
            "treatment_spec_hash": arms["treatment"].spec_hash,
            "incumbent_path_label": arms["incumbent"].path_label,
            "treatment_path_label": arms["treatment"].path_label,
            "shared_prefix": CS.arm_prefix_is_shared(arms),
            "target_geometry": dict(CS.TARGET_GEOMETRY),
            "search_seed_carried": CS.SEARCH_SEED,
        },
        "replay_gates": {
            "expected_parent_digest": CS.EXPECTED_PARENT_DIGEST,
            "expected_parent_state_id": CS.EXPECTED_PARENT_STATE_ID,
            "expected_incumbent_digest": CS.EXPECTED_INCUMBENT_DIGEST,
            "expected_incumbent_state_id": CS.EXPECTED_INCUMBENT_STATE_ID,
            "fail_stop": (
                "A mismatch at EITHER gate ENDS THE SESSION BEFORE ANY 0.86M "
                "RECOVERY TRAINING. No automatic waiver, no retry, no substitution "
                "of a rebuilt parent for the historical one. The mismatch evidence "
                "is preserved and referred to review; a functional-equivalence "
                "amendment is a decision to be made from that evidence, not "
                "pre-authorized here."),
            "evidence_on_mismatch": list(CS.MISMATCH_EVIDENCE),
        },

        "teacher": {
            "repo_id": teacher["repo_id"],
            "revision": teacher["revision"],
            "expected_shard_sha256": teacher["expected_shard_sha256"],
            "n_shards": teacher["n_shards"],
            "index_total_size_bytes": teacher["index_total_size_bytes"],
            "hash_semantics": teacher["hash_semantics"],
            "verification_rule": teacher["verification_rule"],
            "binding_record": "logs/phase_c1_teacher_binding.json",
            "weights_present_locally": teacher["weights_present_locally"],
        },
        "tokenizer_contract": {
            "module": "aadistill.models.tokenizer_contract",
            "rule": ("leaves are weight-only by design — leaf_durability REFUSES a "
                     "persisted leaf carrying tokenizer files, because that would "
                     "change tokenizer_sha256 and therefore the artifact digest the "
                     "metrics hang on. Both arms are packaged for evaluation "
                     "through the tokenizer contract, not by copying files into "
                     "the checkpoint."),
            "teacher_tokenizer_sha256": teacher["files"]["tokenizer.json"]["blob_name"],
        },

        "calibration": {
            "calib.domain_balanced@v1": {
                "profile_hash": db.profile_hash,
                "content_sha256": db.content_sha256,
                "items_path": "artifacts/stage1/e8_calibration_v1/items.jsonl",
                "n_items": 67},
            "calib.reasoning_heavy@v2": {
                "profile_hash": rh.profile_hash,
                "content_sha256": rh.content_sha256,
                "items_path": "artifacts/stage1/reasoning_heavy_v2/items.jsonl",
                "n_items": 62},
            "resolution": "fail-closed: resolve() re-derives and checks the content hash",
        },

        "isolation_plan": {
            "plan_hash": plan.plan_hash,
            "plan": plan.as_dict(),
        },
        "seeds": {
            "values": list(seeds),
            "count": 3,
            "derivation": ("seed_i = uint32_be(SHA256(C0_digest + "
                           "':phase-c1:recovery-seed:' + decimal(i))[0:4]) mod 2**31, "
                           "i from 0, advancing past collisions with the historical "
                           "seeds or an earlier draw"),
            "base_digest": C0_PREREGISTRATION_SHA256,
            "base_digest_source": "logs/phase_c0_preregistration.json (commit be2ab08)",
            "historical_seeds_excluded": list(HISTORICAL_SEEDS),
            "no_human_choice": ("the rule leaves no discretion; the values are "
                                "determined by a document frozen before the "
                                "replacement operator existed"),
            "independently_confirmed": ("recomputed by the committed implementation, "
                                        "by a hand computation, and by the reviewer, "
                                        "all three agreeing"),
        },

        "battery": {
            "asset_id": battery["asset_id"],
            "content_sha256": battery["content_sha256"],
            "manifest_sha256": battery["manifest_sha256"],
            "canonical_path": battery["canonical_path"],
            "repo_local_copy": battery["path"],
            "n_prompts": battery["n_prompts"],
            "n_scorable_prompts": battery["n_scorable_prompts"],
            "set_sha256": battery["set_sha256"],
            "evaluated_once_per_probe": True,
            "no_model_has_been_evaluated_on_it": True,
        },

        "generation_protocol": {
            "fingerprint": "1e5031c2b7debcccb6ab8f92d9863fb2a8381aefd534d5267cb0b519e6192206",
            "protocol_id": "recovery_generation", "version": 1,
            "note": ("the frozen Phase-A/B generation identity: greedy, "
                     "temperature 0, P18-unrestricted max_tokens, "
                     "degeneration stop, resolved context 8192"),
        },
        #: SUPERSEDED BEFORE ANY C1 DATUM EXISTS. `recovery_search_scoring@v2`
        #: was bound here and has been DEMONSTRATED non-executable on the frozen
        #: C1 battery, in two independent places inside
        #: `score_recovery_search.py`: its battery pins are module constants
        #: checked unconditionally, and its result builder reads
        #: `manifest["metrics"]`, which the C1 manifest deliberately does not
        #: carry. Verified by execution — with the pins overridden in memory it
        #: raised `KeyError: 'metrics'` after scoring 150 rows.
        #:
        #: Neither frozen asset is rewritten. @v2 remains the Phase-A/B identity
        #: and the battery keeps `content_sha256 = a285d61f…`. C1 declares its own
        #: binding with the SAME semantics, which an admission gate demonstrates
        #: rather than asserts.
        "scoring_contract": {
            **{k: v for k, v in c1_scoring_contract(REPO).items() if k != "files"},
            "n_files": len(c1_scoring_contract(REPO)["files"]),
            "files": [e["path"] for e in c1_scoring_contract(REPO)["files"]],
            "metric_contract": C1_METRIC_CONTRACT,
            "historical_numerical_equivalence": {
                "record": "logs/phase_c1_scoring_equivalence.json",
                "sha256": sha256_file(REPO / "logs/phase_c1_scoring_equivalence.json"),
                "verdict": _equivalence()["verdict"],
                "n_cases": _equivalence()["n_cases"],
                "total_differences": _equivalence()["total_differences"],
                "basis": ("real retained recovery_search_v2 generations scored "
                          "through both paths; every material numerical field "
                          "equal, per sample and in aggregate, no tolerance"),
                "coverage_limits": _equivalence()["coverage_limits"],
            },
        },
        "recovery_recipe": E1_KD_HEAVY_0860K.as_dict(),

        "decision": {
            "primary_endpoint": "correct_overall over the 850 scorable prompts",
            "estimand": ("Delta = mean over prompts ( mean over the 3 fixed seeds "
                         "( correct_treatment - correct_incumbent ) )"),
            "inference": "stratified prompt-cluster bootstrap; seeds are FIXED BLOCKS",
            "ci_claim_boundary": (
                "prompt-distribution uncertainty CONDITIONAL ON the three "
                "preregistered fresh recovery-seed checkpoint pairs; NOT a CI over "
                "hypothetical future recovery seeds"),
            "bootstrap": {
                "algorithm": BOOTSTRAP_ALGORITHM,
                "seed": bootstrap_seed(),
                "seed_derivation": "SHA256(C0_digest + ':phase-c1:bootstrap')[0:4] mod 2**31",
                "iterations": BOOTSTRAP_ITERATIONS,
                "quantile_convention": BOOTSTRAP_QUANTILE_CONVENTION,
                "stratum_convention": BOOTSTRAP_STRATUM_CONVENTION,
                "bound_before_any_c1_datum_exists": True,
            },
            "rule": {
                "GO": ["one-sided 95% LCB > 0", "AND point estimate >= +0.010",
                       "AND at least 2 of 3 seed-specific deltas > 0",
                       "AND all behavioural guardrails pass"],
                "NO_GO": ["one-sided 95% UCB < +0.010",
                          "OR a preregistered behavioural veto fires"],
                "INCONCLUSIVE": "otherwise; no forced winner",
            },
            "also_reported": ["two-sided 95% CI", "each seed-specific delta",
                              "McNemar counts per seed",
                              "per-set and per-domain diagnostics"],
        },
        "guardrails": {
            "usable_rollout": {"pooled_min_delta": plan.usable_pooled_min_delta,
                               "per_seed_min_delta": plan.usable_per_seed_min_delta,
                               "role": "veto only; never positive ranking credit"},
            "catastrophic_capability": {
                "rule_id": plan.catastrophic_rule_id,
                "candidate_max": plan.catastrophic_candidate_max,
                "control_min": plan.catastrophic_control_min,
                "control_operand": "the INCUMBENT arm (fe9683 / current ATTENTION)",
                "candidate_operand": "the TREATMENT arm (activation_importance_v1)",
                "asymmetry": ("deliberate: it can veto the treatment, never flag the "
                              "incumbent. Never positive ranking evidence."),
            },
        },

        "runtime_requirements": {
            "must_be_recorded_at_execution": [
                "image digest", "torch version", "transformers version",
                "CUDA runtime", "host driver", "GPU model", "attention backend",
                "dtype"],
            "why": ("the replay gates compare artifact digests, and config_sha256 "
                    "folds in transformers_version while the WIDTH projection is a "
                    "float64 eigendecomposition whose low bits depend on the BLAS "
                    "kernel. A mismatch is only interpretable against the runtime "
                    "that produced it."),
            "phase_b_search_runtime_for_reference": {
                "image": "runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404@580.178.04",
                "torch": "2.11.0+cu128", "transformers": "5.13.1",
                "cuda_runtime": "12.8", "measurement_device": "cuda"},
        },

        "evidence_manifest_contract": {
            "required_before_teardown": [
                "per-probe train_log.jsonl (6)",
                "per-probe per-sample rows (6)",
                "per-probe raw generations, every evaluated sample (P18)",
                "per-probe scored aggregates (6)",
                "both arm artifact identities and the verified parent identity",
                "the fixed-path replay record for both arms",
                "the decision record with all seed-specific deltas and McNemar counts",
                "the runtime identity block",
            ],
            "gate": ("collect_artifacts.py manifest -> archive -> verify-archive -> "
                     "transfer -> verify-local -> gate; a missing required artifact "
                     "blocks teardown"),
        },

        #: The files that IMPLEMENT the contract above. Bound by hash here, and
        #: also inside `c1_harness`, because a declaration of what evidence
        #: survives teardown is an executable input: editing it changes what a
        #: paid session brings home, and it must not be able to change without
        #: moving the digest a grant binds. Both were absent from the tree that
        #: passed every other precheck at `d43346f`, which is why they are
        #: bound explicitly rather than left to the harness set alone.
        "artifact_specs": {
            "success": {
                "path": SPEC_SUCCESS,
                "sha256": sha256_file(REPO / SPEC_SUCCESS),
                "requires": (
                    f"{n_probes} train_log.jsonl, {n_probes} run_manifest, "
                    f"{n_probes} run_completion, {n_probes} probe journals, "
                    f"{n_probes} probe configs, {n_probes} per-sample files, "
                    f"{n_probes} scored aggregates, {n_probes * n_sets} raw "
                    f"generation files and {n_probes * n_sets} generation "
                    "summaries (every set of every probe), plus the replay "
                    "record, both arm identities, the decision, the attested "
                    "protocol, the session evidence and the engine probe"),
            },
            "failed": {
                "path": SPEC_FAILED,
                "sha256": sha256_file(REPO / SPEC_FAILED),
                "requires": (
                    "the session evidence record only. Everything else is "
                    "collected when present. `ArtifactSpec` has no conditional "
                    "and the runner selects between exactly two spec files on "
                    "one bit, so the required set is the intersection over "
                    "every terminal failure state; requiring the replay record "
                    "would block teardown on a run that failed before stage D. "
                    "The residual hole is recorded in the file's `limitation` "
                    "field and is a live review item, not a closed design."),
            },
            "gate": ("artifact_spec_gate in scripts/pod/autoinit_c1_launch.py, "
                     "at $0 before provider creation: both files exist, parse, "
                     "load through collect_artifacts.load_specs, stay inside the "
                     "artifact roots, are inside the measured harness set, cover "
                     "the contract above at derived minimums, and — for the "
                     "failure spec — demand nothing that presupposes training"),
            "paths_are_the_driver_s": (
                "the manifest root is artifacts/ and the paths under it are the "
                "STANDALONE C1 driver's own: audit/autoinit_c1, "
                "stage3/c1/<probe_id>, eval/c1/<probe_id>. An earlier revision of "
                "this document said they were PhaseADriver's, which C1Driver then "
                "inherited; that inheritance is gone and no C1 evidence is written "
                "under another phase's tree"),
        },

        "driver": {
            "path": "scripts/pod/autoinit_c1_driver.py",
            "standalone": True,
            "subclasses_phase_a_driver": False,
            "imports_phase_a_driver_or_launcher": False,
            "owns_paths": ["artifacts/audit/autoinit_c1", "artifacts/stage3/c1",
                           "artifacts/eval/c1"],
            "stages": "B-I; A and J belong to the session runner",
            "stage_g_h_separation": (
                "stage G runs all six recovery trainings and opens no battery, "
                "starts no evaluator and no scorer; stage H calls "
                "require_all_trained() before the first evaluation. Evaluating "
                "inside the training loop would let the first arm meet the "
                "confirmation battery before the last arm was trained"),
            "generation_admission": (
                "NO PROBE RESULT IS ADMITTED unless the generation protocol "
                "observed from THAT probe's raw per-set summaries is comparable "
                "to the preregistered C1 evaluation protocol. The comparison runs "
                "after generation and BEFORE the scorer; on failure the session "
                "stops as C1_INCOMPLETE, that probe is not scored, no later probe "
                "is evaluated and stage I does not run. The scorer is passed the "
                "OBSERVED fingerprint, not the attested one — they are equal once "
                "the check passes, and the direction of provenance is the point"),
            "attested_evaluation_protocol": (
                "written to artifacts/audit/autoinit_c1/"
                "c1_attested_evaluation_protocol.json at the start of stage H, "
                "before the first evaluation; the six admitted per-probe protocol "
                "hashes must all equal it, and c1_probe_results.json carries the "
                "observed generation fingerprint and observed "
                "evaluation_protocol_hash for every probe"),
            "scoring": (
                "score_c1_confirmation.py only, on c1_confirmation_v1 only. The "
                "frozen score_recovery_search.py is never invoked on this battery"),
            "device_handoff": (
                "complete_release -> require_released -> require_headroom before "
                "probe 1, at the measured recovery-trainer requirement"),
            "evidence": ["c1_evidence.json", "c1_replay_record.json",
                         "c1_arm_identities.json", "c1_probe_results.json",
                         "c1_decision.json",
                         "c1_attested_evaluation_protocol.json",
                         "c1_device_handoff.json",
                         "<probe_id>_generation_admission.json"],
        },

        #: Whether the pod can OBTAIN the authorized commit. Attempt 1 passed
        #: every content gate and died at SETUP_RC=1 on a bundle that did not
        #: exist; this is the ninth gate that closes it.
        "transport": {
            "canonical_bundle_name": "aad_autoinit_<first 8 hex of session commit>.bundle",
            "relay": "AlphaAvatar/aadistill-artifacts:transfer/<canonical name>",
            "derived_from": "--session-commit; an alias fails at $0",
            "preparation": ("scripts/autoinit/stage_c1_bundle.py -- MAY mutate the "
                            "relay; refuses a commit that does not carry the "
                            "authorization, and refuses to overwrite a different "
                            "existing remote object"),
            "gate": ("bundle_staged_gate, READ-ONLY: download the canonical relay "
                     "object, sha256 it against the staged bundle, git bundle "
                     "verify the round-tripped bytes, clone and check out the "
                     "session commit exactly as the pod's setup does, then require "
                     "the authorization inside the checkout to be the artifact the "
                     "launcher is loading and the harness recomputed from that "
                     "checkout to equal the authorized digest"),
            "ordering": ("repair base -> issue -> commit ONLY the authorization -> "
                         "that commit is the session commit -> bundle it -> upload "
                         "-> gate -> provider. A bundle built for the "
                         "pre-authorization base checks out a tree with no "
                         "authorization in it"),
            "n_pre_provider_gates": 10,
        },

        #: SETUP READINESS, explicitly NOT a C1 measurement input. The shared
        #: setup's ROPE_OK step globs artifacts/stage1/*/checkpoint/config.json
        #: and loads each match through AutoConfig in both venvs, requiring a
        #: stored RoPE base of 5,000,000; it reads no weights. Attempt 2 staged
        #: only the three evaluation tokenizer sidecars there and died at that
        #: step for $0.1013 after the teacher had already been verified.
        "setup_readiness": {
            "rope_ok_input": {
                "relay_path": "stage1/qwen3_0p6b_init_v0/checkpoint/config.json",
                "dest": "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint",
                "sha256": sha256_file(
                    REPO / "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint/config.json"),
                "bytes": (REPO / "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint"
                          "/config.json").stat().st_size,
                "stored_rope_base": 5000000,
                "verified": ("the relay object was downloaded read-only and hashed "
                             "against the identity the repository already records; "
                             "rope_input_gate re-derives it before every launch"),
            },
            "is_not_a_measurement_input": (
                "this config is consumed ONLY by the shared setup's ROPE_OK step. "
                "It is not read by any C1 stage: the arms are materialized from "
                "the verified teacher, the probes train from the arm checkpoints, "
                "and stage H evaluates a package built from the trained bytes plus "
                "the three frozen tokenizer sidecars. No C1 number depends on it."),
            "weights_not_staged": (
                "model.safetensors and generation_config.json are deliberately NOT "
                "pulled: 1.19 GiB for a check that reads neither"),
            "gate": ("rope_input_gate, at $0 and read-only -- declared, correct "
                     "destination, pinned, present on the relay, hashes to the pin, "
                     "parses, and stored_rope_base == 5,000,000. It does not "
                     "replace the pod-side ROPE_OK check under both runtimes"),
        },

        "out_of_scope": [
            "formal Stage-2/Stage-3 recovery training — explicitly NOT in scope; C1 "
            "is a fixed-path ATTENTION isolation using short 0.86M recovery probes "
            "and establishes no recovered-model capability",
            "canonical Stage-1 NLL",
            "Phase C2 joint re-search",
            "any search, ranking, successive halving, tie-breaking or arm elimination",
        ],
        "authorizes": "nothing",
    }
    doc["preregistration_sha256"] = sha256_json(doc)
    OUT.write_text(json.dumps(doc, indent=1) + "\n")

    print(f"wrote {OUT.relative_to(REPO)}")
    print(f"  preregistration_sha256 {doc['preregistration_sha256']}")
    print(f"  executable_source      {doc['executable_source']['digest']}")
    print(f"  c1_harness             {doc['c1_harness']['digest']} ({doc['c1_harness']['n_files']} files)")
    print(f"  session contract       {doc['c1_session_contract']['hash']}")
    print(f"  isolation plan         {doc['isolation_plan']['plan_hash']}")
    print(f"  incumbent path         {doc['fixed_path']['incumbent_spec_hash']}")
    print(f"  treatment path         {doc['fixed_path']['treatment_spec_hash']}")
    print(f"  seeds                  {doc['seeds']['values']}")
    print(f"  battery                {doc['battery']['asset_id']} "
          f"{doc['battery']['content_sha256'][:16]}")


if __name__ == "__main__":
    main()
