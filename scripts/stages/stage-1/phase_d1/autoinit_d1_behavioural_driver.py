#!/usr/bin/env python3
"""Train and score one D1 behavioural rung's probes, then advance exactly one.

    PYTHONPATH=src:scripts python scripts/stages/stage-1/phase_d1/autoinit_d1_behavioural_driver.py \
        --out artifacts/audit/autoinit_d1_behavioural --run-id <id> --rung screening \
        --authorization <governance/authorization.json> --arm-root /workspace/aad_arms

Screening is ten probes: four candidates and the incumbent, at two
preregistered seeds each. Confirmation is six: the advancing candidate and B at
three fresh seeds. The seed is the REPLICATE and the initialization is the
TREATMENT, so two probes at one seed differ only in which checkpoint they
started from, and that is what makes their difference attributable.

**THE ARMS ARE MATERIALIZED ON THE POD, each gated on its exact recorded
identity.** A 1.1-1.2 GiB checkpoint fits neither transport, so the candidates
replay their digest-pinned search paths (the machinery `d1_replay_002`
validated byte-exactly) and B is built from its one construction owner,
`phase_c2.baseline.frozen_baseline_spec`. A digest mismatch is TERMINAL: the
paths are deterministic, and a probe trained from a near-equivalent checkpoint
measures something nobody asked about.

**THE CONTRACT IS ASSERTED BEFORE ANY PROBE TRAINS, and the authorization
binds it.** The arms, the battery role, the recipe, the seeds and the probe
schedule are derived from frozen records and hashed into one LOCATION-FREE
value; the one-use artifact binds that hash at issuance on the dev host and
this driver recomputes and compares it here, so a session measuring a
different field is refused before the trainer runs. The cheap checks -- the
battery bytes, the role disjointness, B's construction pins, the recovery
pack's digest -- run even earlier, before the materialization stage loads any
weights.

**EVERY ASSET IS IMPORTED FROM ITS OWNER.** The trainer, the uncapped
evaluator, the frozen recipe, the pack, the evaluation-tokenizer pins and the
allowed override set are `autoinit_c1_driver`'s; the evaluation package is
built by `phase_c1.packaging.build_evaluation_package`, exactly as C1's own
probes were evaluated. A second copy of any of them is a second thing that can
disagree about what recovery or evaluation means.

**DURABILITY RUNS DURING THE SESSION, NOT AT CLOSEOUT.** Each probe's result is
written and announced the moment it finishes, so the launcher's poll hook can
pull it off-pod. C1 attempt 17 trained six probes over ten hours and lost every
one because nothing left the pod until a closeout that never came.

**THE RANKING IS MECHANICAL AND THE SELECTION IS ONE CANDIDATE** (screening
only). It reads the frozen decision rule's endpoint, pools each arm's seeds by
the declared aggregation, applies the usable-rollout guardrail as a VETO ONLY,
and advances the best surviving candidate -- or none, and says so. The
confirmation rung records its probes' scores and STOPS: the three-way verdict
is computed off-pod at $0, because A3 proved that an on-pod aggregation stage
is where a completed session loses its decision artifact.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
for _extra in ("src", "scripts", "scripts/pod", "scripts/autoinit",
               "scripts/stages/stage-1"):
    _path = str(REPO_ROOT / _extra)
    if _path not in sys.path:
        sys.path.insert(0, _path)

SUCCESS_MARKER = "ALL_DONE"
FAILURE_MARKER = "RUN_FAILED"
CHECK_ONLY_MARKER = "CHECK_ONLY_OK"
MISMATCH_MARKER = "DIGEST_MISMATCH"

#: The pinned scorer PER RUNG. Each takes no battery argument -- the role is
#: pinned inside the scorer, which is the whole point: a flag is exactly how a
#: rung comes to be scored on the wrong prompts.
SCORER = REPO_ROOT / "scripts/stages/stage-1/phase_d1/score_d1_screening.py"
CONFIRMATION_SCORER = (REPO_ROOT /
                       "scripts/stages/stage-1/phase_d1/score_d1_confirmation.py")


class D1BehaviouralDriverError(RuntimeError):
    """This session cannot proceed on the evidence it has."""


def mark(status_path: str, name: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(status_path, "a") as handle:
        handle.write(f"{stamp} MARKER:{name}\n")
        handle.flush()


def terminal_marker(*, check_only: bool, status: str,
                    evidence_written: bool, mismatch: bool = False) -> str:
    """The marker the launcher polls for. ONE predicate, ONE caller.

    `evidence_written` gates success on both paths: a session that trained ten
    probes and could not write its record has not produced what it owes.
    `mismatch` is an arm that completed its pinned path and does not carry the
    recorded identity -- a scientific stop condition, distinguishable from an
    engineering failure because it must go to review rather than be retried.
    """
    if mismatch:
        return MISMATCH_MARKER
    if not evidence_written:
        return FAILURE_MARKER
    if check_only:
        return CHECK_ONLY_MARKER if status == "CHECK_ONLY_OK" else FAILURE_MARKER
    return SUCCESS_MARKER if status == "COMPLETE" else FAILURE_MARKER


def build_parser() -> argparse.ArgumentParser:
    """At module scope so the dispatch probe can build it. Every required
    argument is a STRING: the probe fills them with one."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True,
                    help="evidence directory, repository-relative on the pod")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--rung", default="screening",
                    choices=("screening", "confirmation"))
    ap.add_argument("--authorization", default=None,
                    help="this run's one-use behavioural authorization. "
                         "Required for a formal run; omitted only with "
                         "--check-only")
    ap.add_argument("--arm-root", default=None,
                    help="where the arms' bytes live or are materialized. "
                         "None on the dev host (the secured durable-store "
                         "copies); a pod passes its own root and the missing "
                         "arms are rebuilt there along digest-pinned paths")
    ap.add_argument("--replay-plan", default=None,
                    help="the committed digest-pinned plan the candidate "
                         "materialization replays; defaults to "
                         "behavioural_materialize.PLAN_REL")
    ap.add_argument("--advancing-candidate", default=None,
                    help="confirmation only: the candidate screening advanced. "
                         "Must equal the one the authorization names.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--image-digest", default=None,
                    help="the pod image digest, from the launcher; the engine "
                         "probe binds it into the generation-runtime identity")
    ap.add_argument("--status", default=None,
                    help="where the terminal marker is appended")
    ap.add_argument("--probe-train-minutes", type=float, default=60.0,
                    help="per-probe training timeout, from the priced cell")
    ap.add_argument("--probe-eval-minutes", type=float, default=40.0)
    ap.add_argument("--check-only", action="store_true",
                    help="assert the contract and resolve every symbol; "
                         "train nothing, materialize nothing, write no probe")
    return ap


def probe_config(probe: Any, *, audit: Path, frozen_recipe: Path,
                 pack_dir: str, overrides: frozenset[str]) -> Path:
    """This probe's training config: the frozen recipe plus the allowed set.

    The override set is C1's, unchanged, and a derived config that differs
    anywhere else is REFUSED. That is the mechanical form of "identical
    recovery": the only intended difference between two probes at one seed is
    the initialization, and a config free to differ elsewhere would make a
    paired difference uninterpretable.
    """
    frozen = json.loads(frozen_recipe.read_text())
    derived = {
        **frozen,
        "run_name": probe.probe_id,
        "out_dir": f"artifacts/stages/stage-3/d1_behavioural/{probe.probe_id}",
        "data_dir": pack_dir,
        "seed": probe.seed,
        "student_path": probe.checkpoint_dir,
        "_purpose": (
            f"D1 {probe.rung} probe, arm {probe.arm_id}, seed {probe.seed}. "
            "Identical recovery; the only intended difference between probes "
            f"at one seed is the initialization. Derived from "
            f"{frozen_recipe.name} by overriding run identity, pack path, "
            "seed and student_path."),
    }
    differ = sorted(key for key in set(frozen) | set(derived)
                    if frozen.get(key) != derived.get(key))
    outside = sorted(set(differ) - overrides)
    if outside:
        raise D1BehaviouralDriverError(
            f"{probe.probe_id}: the derived probe config differs from the "
            f"frozen recipe in {outside}, outside the allowed override set "
            f"{sorted(overrides)}. Recovery must be identical across arms or "
            "a paired difference is not a difference between initializations.")
    path = audit / "configs" / f"{probe.probe_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(derived, indent=2) + "\n")
    return path


def cheap_preflight(rung: str, repo_root: Path) -> dict[str, Any]:
    """Everything refusable WITHOUT loading weights, refused first.

    The battery bytes against the manifest, the two roles' disjointness, B's
    construction pins against the design, and the recovery pack's content
    against the digest the frozen recipe declares. Each is the pin its asset
    SHIPS -- this session declares no ASSETS_READY sweep, so these checks are
    where a truncated or substituted staging dies, in seconds.
    """
    import hashlib

    from shared.recipes import E1_KD_HEAVY_0860K as recipe
    from stages.phase_c1.autoinit_c1_driver import PACK_DIR
    from stages.phase_d1 import behavioural as D1B

    role = f"d1_{rung}"
    battery = D1B.battery_role(role, repo_root)
    disjoint = D1B.roles_are_disjoint(repo_root)
    blocks = repo_root / PACK_DIR / "blocks.npz"
    if not blocks.is_file():
        raise D1BehaviouralDriverError(
            f"the recovery pack has no {blocks}; a probe cannot train on a "
            "pack that is not staged")
    got = hashlib.sha256(blocks.read_bytes()).hexdigest()
    if got != recipe.pack_sha256:
        raise D1BehaviouralDriverError(
            f"the staged pack hashes to {got[:12]} and the frozen recipe "
            f"declares {recipe.pack_sha256[:12]}. Identical recovery starts "
            "with identical data; refusing before anything trains.")
    return {"battery": {k: battery[k] for k in
                        ("role", "family_content_id", "allocation_rule_id",
                         "item_ids_sha256", "n_prompts", "n_scorable")},
            "roles_disjoint": disjoint["disjoint"],
            "pack_sha256": got}


def d1_scoring_contract(rung: str, repo_root: Path) -> dict[str, Any]:
    """The scoring digest a D1 behavioural result binds to. C1's rule.

    The metric IS C1's (`score_battery`/`build_result`, imported unchanged),
    so the contract id is C1's; the digested FILE SET is C1's declared closure
    plus the rung's own pinned entry point, because that file is part of what
    actually scores and a digest over a smaller scorer than the one that runs
    describes nothing.
    """
    from stages.phase_c1.scoring import c1_scoring_contract

    scorer = SCORER if rung == "screening" else CONFIRMATION_SCORER
    from stages.phase_c1.scoring import C1_SCORING_FILES_V1

    files = tuple(sorted({*C1_SCORING_FILES_V1,
                          str(scorer.relative_to(REPO_ROOT))}))
    return c1_scoring_contract(repo_root, files=files)


def battery_protocol_fields(contract: dict[str, Any]) -> dict[str, str]:
    """The battery identity triple `RecoveryEvaluationProtocol` binds.

    The D-series roles have no per-role `manifest.json`; their identity lives
    in the realized family manifest, already byte-verified by the session
    contract. The mapping is stated once, here: the artifact names the family
    and the role, the manifest identity is the role's frozen membership
    (`item_ids_sha256`), and the content identity is the family's bound bytes
    (`family_content_id`).
    """
    battery = contract["battery"]
    return {
        "battery_artifact": f"{battery['family_id']}:{battery['role']}",
        "battery_manifest_sha256": battery["item_ids_sha256"],
        "battery_content_sha256": battery["family_content_id"],
    }


def build_attested_protocol(package: Path, *, image_digest: str | None,
                            scoring: dict[str, Any],
                            battery_fields: dict[str, str],
                            out: Path) -> Any:
    """This session's evaluation-protocol attestation, from a REAL package.

    C1's mechanism, reused: the shared engine probe observes the actual vLLM
    runtime over the packaged checkpoint, the declared generation protocol is
    materialized with those observations, and the result is the ONE hash
    every probe's generations must be comparable to. Run once, before the
    first probe generates; each probe is then admitted against it from its
    own raw summaries.
    """
    from aadistill.infrastructure.manifest import sha256_file, sha256_json
    from aadistill.initialization.planning.generation import (
        RecoveryEvaluationProtocol, declared_generation_protocol,
    )
    from shared.source_sets import generation_source_digest

    probe_out = out / "engine_probe.json"
    engine = subprocess.run(
        ["/opt/vllm/bin/python",
         str(REPO_ROOT / "scripts/shared/pod/autoinit_engine_probe.py"),
         "--model", str(package), "--out", str(probe_out)]
        + (["--image-digest", image_digest] if image_digest else []),
        capture_output=True, text=True, timeout=1800)
    (out / "engine_probe_tail.log").write_text(
        (engine.stdout + engine.stderr)[-2000:])
    if engine.returncode != 0 or not probe_out.is_file():
        raise D1BehaviouralDriverError(
            f"the engine probe exited {engine.returncode}; without it no "
            "generation protocol can be attested and no probe may be scored")
    observed = json.loads(probe_out.read_text())

    gen = declared_generation_protocol().materialized(
        generation_source_digest=generation_source_digest(REPO_ROOT)["digest"],
        degeneration_source_digest=sha256_file(
            REPO_ROOT / "src/aadistill/evaluation/degeneration.py"))
    gen = gen.materialized(
        vllm_version=observed["vllm_version"],
        transformers_version=observed["transformers_version"],
        torch_version=observed["torch_version"],
        runtime_digest=observed["runtime_digest"], dtype=observed["dtype"],
        gpu_memory_utilization=observed["gpu_memory_utilization"],
        max_num_seqs=observed["max_num_seqs"],
        max_num_batched_tokens=observed["max_num_batched_tokens"],
        enforce_eager=observed["enforce_eager"],
        tokenizer_sha256=observed["tokenizer_sha256"],
        chat_template_sha256=observed["chat_template_sha256"],
        resolved_context=observed["resolved_context"],
        context_source=observed["context_source"],
        stop_token_ids=tuple(observed["stop_token_ids"]))
    gen.require_materialized(context="phase D1 behavioural attestation")

    attested = RecoveryEvaluationProtocol(
        generation=gen,
        scoring_contract=scoring["contract"],
        scoring_digest=scoring["digest"],
        **battery_fields)
    report = {
        "schema": "aadistill.phase_d1.behavioural_attested_protocol/v1",
        "generation_protocol_fingerprint": gen.fingerprint,
        "evaluation_protocol_hash": attested.evaluation_protocol_hash,
        "scoring_contract": scoring,
        "evaluation_protocol": attested.as_dict(),
        "engine_probe": "engine_probe.json",
    }
    report["report_sha256"] = sha256_json(report)
    (out / "d1_behavioural_attested_protocol.json").write_text(
        json.dumps(report, indent=2) + "\n")
    return attested


def admit_generation(name: str, gen_dir: Path, *, attested: Any,
                     scoring: dict[str, Any],
                     battery_fields: dict[str, str], out: Path) -> dict:
    """Refuse a probe whose generations were not produced under the protocol.

    C1's rule, unchanged: the attestation says what the runtime SHOULD do;
    this reconstructs what it actually DID, from this probe's own raw per-set
    summaries, and requires the two comparable under the project's versioned
    relation. Fail-closed and BEFORE the scorer runs -- a probe generated
    under a drifted protocol must not be scored, and scoring first then
    recording the fingerprint lets a result claim an identity whose evidence
    was never admitted.
    """
    from aadistill.initialization.planning.generation import (
        RecoveryEvaluationProtocol, observe_generation_protocol,
    )

    summaries = [json.loads(p.read_text())
                 for p in sorted(gen_dir.glob("*.json"))
                 if not p.name.endswith(".generations.jsonl")]
    observed_gen = observe_generation_protocol(summaries).protocol
    observed = RecoveryEvaluationProtocol(
        generation=observed_gen,
        scoring_contract=scoring["contract"],
        scoring_digest=scoring["digest"],
        **battery_fields)
    admission = {
        "probe_id": name,
        "generation_fingerprint": observed_gen.fingerprint,
        "evaluation_protocol_hash": observed.evaluation_protocol_hash,
        "attested_evaluation_protocol_hash":
            attested.evaluation_protocol_hash,
        "n_summaries": len(summaries),
    }
    try:
        observed.require_comparable(attested, context=name)
    except Exception as exc:                                      # noqa: BLE001
        admission["comparable"] = False
        admission["reason"] = str(exc)[-1500:]
        (out / f"{name}_generation_admission.json").write_text(
            json.dumps(admission, indent=2) + "\n")
        raise D1BehaviouralDriverError(
            f"{name}: the generations were not produced under the attested "
            f"evaluation protocol, so this probe cannot be scored. {exc}"
        ) from exc
    admission["comparable"] = True
    (out / f"{name}_generation_admission.json").write_text(
        json.dumps(admission, indent=2) + "\n")
    return admission


def trained_checkpoint_state(model_root: Path) -> dict[str, Any] | None:
    """Is a COMPLETED trained checkpoint already on this host for a probe?

    The resume half of the trained-but-unscored contract (P8.4 state 2: a
    trained, durable, not-validly-scored probe is RESUMED at evaluation, never
    retrained). True in two situations: a driver restart on the same pod, and
    a replacement pod whose launcher restored the preserved checkpoint. The
    training-completion record is the gate -- a crash mid-training leaves no
    `run_completion.json`, so a partial checkpoint retrains.

    When the restorer shipped a `preserved_identity.json` sidecar, the shard
    is re-hashed against it: restored bytes that are not the preserved bytes
    must refuse here, before forty minutes of generation measure the wrong
    model.
    """
    import hashlib

    completion = model_root / "run_completion.json"
    if not completion.is_file():
        return None
    from stages.phase_c1.autoinit_c1_driver import trained_model_dir

    model_dir = trained_model_dir(model_root)
    if not (Path(model_dir) / "config.json").is_file():
        return None
    shard = Path(model_dir) / "model.safetensors"
    state: dict[str, Any] = {
        "model_dir": str(model_dir),
        "run_completion": str(completion),
    }
    sidecar = model_root / "preserved_identity.json"
    if sidecar.is_file():
        want = json.loads(sidecar.read_text()).get("trained_sha256")
        got = hashlib.sha256(shard.read_bytes()).hexdigest() \
            if shard.is_file() else None
        if not want or got != want:
            raise D1BehaviouralDriverError(
                f"{model_root.name}: restored trained checkpoint hashes to "
                f"{str(got)[:12]} and the preservation record says "
                f"{str(want)[:12]}; resuming on it would score bytes nobody "
                "preserved")
        state["restored_sha256_verified"] = True
    return state


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    status = args.status or "/workspace/autoinit_d1_behavioural.status"
    advancing = args.advancing_candidate or None

    record: dict[str, Any] = {
        "schema": "aadistill.phase_d1.behavioural_session/v1",
        "run_id": args.run_id,
        "rung": args.rung,
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "RUNNING",
        "probes": [],
    }
    evidence = out / "d1_behavioural.json"

    def save() -> None:
        evidence.write_text(json.dumps(record, indent=2, default=str) + "\n")

    mark(status, "DRIVER_START")
    mismatch = False
    try:
        from aadistill.infrastructure.manifest import sha256_json
        from stages.phase_d1 import behavioural as D1B
        from stages.phase_d1 import behavioural_materialize as M

        #: THE CHEAP REFUSALS FIRST: the battery bytes, the disjointness, the
        #: pack digest -- everything answerable before a weight loads, so a
        #: staging defect costs seconds rather than an arm materialization.
        record["cheap_preflight"] = cheap_preflight(args.rung, REPO_ROOT)
        save()

        #: THE ARMS, materialized where they are missing and digest-gated
        #: either way. Only under an explicit --arm-root: the dev host reads
        #: the secured durable-store copies and builds nothing.
        student_paths: dict[str, str] = {}
        if args.arm_root and not args.check_only:
            plan_path = REPO_ROOT / (args.replay_plan or M.PLAN_REL)
            mark(status, "MATERIALIZE_START")
            try:
                materialization = M.materialize_arms(
                    args.rung, plan_path=plan_path, arm_root=args.arm_root,
                    device=args.device, advancing_candidate=advancing,
                    repo_root=REPO_ROOT,
                    say=lambda s: print(s, flush=True))
            except M.D1ArmIdentityMismatch:
                #: ONLY a completed pinned path carrying the wrong identity
                #: after its inputs were verified is a SCIENTIFIC stop. A
                #: missing plan leaf, a wrong path, an unreadable journal or
                #: any other harness error stays an ordinary engineering
                #: failure (RUN_FAILED): diagnose, repair, retry -- P12.1.
                mismatch = True
                raise
            record["arm_materialization"] = materialization
            save()
            mark(status, "MATERIALIZE_DONE")
            student_paths = M.resolve_arm_checkpoints(
                args.rung, args.arm_root, materialization,
                advancing_candidate=advancing, repo_root=REPO_ROOT)
        else:
            for arm in D1B.arms(REPO_ROOT, arm_root=args.arm_root):
                if arm.checkpoint_dir:
                    student_paths[arm.arm_id] = arm.checkpoint_dir

        #: THE CONTRACT, hashed LOCATION-FREE so the value the authorization
        #: bound on the dev host is the value recomputed here -- and asserted
        #: before any probe trains, because a session measuring a field nobody
        #: authorized is the one failure no amount of later evidence repairs.
        contract = D1B.session_contract(
            args.rung, REPO_ROOT, advancing_candidate=advancing,
            arm_root=args.arm_root,
            #: THE PRODUCER'S OWN FINAL-CHECKPOINT PATHS.
            #: `materialize_fixed_path` writes `workdir/steps/{i:02d}_{kind}`,
            #: so the bytes the identity gate must hash are the ones the
            #: materialization rows name -- verifying `arm_root/<state_id>`
            #: directly would refuse the session's own output after paying
            #: for it.
            resolved_paths=student_paths or None)
        record["contract"] = contract
        record["contract_hash"] = sha256_json(contract)
        #: The machine-local verification evidence -- which directories were
        #: hashed, to what -- beside the contract, never inside it. Scoped
        #: exactly as the contract's own verification was.
        record["arm_verification"] = D1B.require_arms_present(
            REPO_ROOT, arm_root=args.arm_root,
            resolved_paths=student_paths or None,
            restrict_to=({advancing, "B"} if args.rung == "confirmation"
                         else None))
        save()
        print(f"contract {record['contract_hash'][:16]} — "
              f"{contract['n_probes']} probes, {contract['arms']['n_arms']} "
              f"arms, battery {contract['battery']['role']} "
              f"({contract['battery']['n_prompts']} prompts, "
              f"{contract['battery']['n_scorable']} scorable)", flush=True)

        #: THE ONE-USE AUTHORIZATION, loaded through its own type. Binds the
        #: rung, the design revision and the contract hash; a confirmation
        #: artifact must name the same candidate this invocation was given.
        if args.authorization:
            from stages.phase_d1.behavioural_governance import (
                D1BehaviouralAuthorization,
            )

            auth = D1BehaviouralAuthorization.load(args.authorization)
            auth.require_rung(args.rung)
            auth.require_plan(contract["design_hash"])
            auth.require_contract(record["contract_hash"])
            if args.rung == "confirmation":
                named = auth.require_advancing_candidate()
                if named != advancing:
                    raise D1BehaviouralDriverError(
                        f"the authorization names advancing candidate "
                        f"{named!r} and this invocation was given "
                        f"{advancing!r}; the confirmation field is not this "
                        "session's to choose")
            record["authorization"] = {
                "path": str(args.authorization),
                "authorization_id": auth.authorization_id,
                "rung": auth.rung,
                "contract_hash": auth.contract_hash,
            }
            save()
        elif not args.check_only:
            raise D1BehaviouralDriverError(
                "a formal behavioural session runs under its one-use "
                "authorization; --authorization may be omitted only with "
                "--check-only")

        #: Frozen assets from their ONE owner. Importing the C1 driver module
        #: is inert -- it defines constants and registers the builtin profiles
        #: and adapters, which this session needs too -- and `C1Driver` itself
        #: is deliberately NOT imported.
        from stages.phase_c1.autoinit_c1_driver import (
            C1_PROBE_OVERRIDES, FROZEN_RECIPE, PACK_DIR, TOKENIZER_SIDECAR_SHA256,
            TOKENIZER_SOURCE, TRAINER, UNCAPPED_EVAL, _trainer_bytes,
            trained_model_dir,
        )
        from stages.phase_c1.packaging import build_evaluation_package
        from aadistill.runtime.device_handoff import (
            complete_release, cuda_memory, require_headroom, require_released,
        )

        probes = D1B.probes(args.rung, REPO_ROOT,
                            advancing_candidate=advancing,
                            arm_root=args.arm_root)
        record["schedule"] = [p.probe_id for p in probes]
        save()

        scorer = SCORER if args.rung == "screening" else CONFIRMATION_SCORER
        if not scorer.is_file():
            raise D1BehaviouralDriverError(
                f"the pinned {args.rung} scorer is missing at {scorer}")

        if args.check_only:
            record["status"] = "CHECK_ONLY_OK"
            record["_check_only"] = (
                "the cheap preflight passed, every arm's bytes verified at "
                "its recorded identity (or its construction pins checked), "
                "the battery re-hashed against the manifest, the two roles "
                "shown disjoint, the schedule agreed with the design's "
                "declared count, the location-free contract hashed, and every "
                "symbol the training and scoring path names imported. No "
                "probe was trained, nothing was materialized and no model "
                "was loaded.")
            save()
            return 0

        #: Each probe's student path is the directory the materialization
        #: VERIFIED, which for B may sit under a step subdirectory of its
        #: workdir. A probe whose arm has no resolved bytes is refused here,
        #: by name, rather than discovered by the trainer. After the
        #: check-only return: a $0 contract check on a host that holds no
        #: incumbent bytes is legitimate, training on one is not.
        resolved = []
        for probe in probes:
            path = student_paths.get(probe.arm_id)
            if not path:
                raise D1BehaviouralDriverError(
                    f"{probe.probe_id}: arm {probe.arm_id} has no resolved "
                    "checkpoint on this host; a probe cannot train from a "
                    "checkpoint nobody can name")
            resolved.append(dataclasses.replace(probe, checkpoint_dir=path))
        probes = resolved
        record["schedule"] = [p.probe_id for p in probes]
        save()

        #: THE EVALUATION-PROTOCOL IDENTITIES this rung's results bind to,
        #: derived once; the attestation itself is built from the FIRST
        #: probe's real package, below, because it needs a checkpoint to
        #: probe the engine over.
        scoring = d1_scoring_contract(args.rung, REPO_ROOT)
        battery_fields = battery_protocol_fields(contract)
        record["scoring_contract"] = {k: scoring[k]
                                      for k in ("contract", "digest")}
        attested = None

        results: list[dict[str, Any]] = []
        for index, probe in enumerate(probes):
            entry: dict[str, Any] = {
                "probe_id": probe.probe_id, "arm": probe.arm_id,
                "seed": probe.seed, "index": index,
                "trained": False, "scored": False,
            }
            record["probes"].append(entry)
            save()
            print(f"[{index + 1}/{len(probes)}] {probe.probe_id}", flush=True)

            model_root = (REPO_ROOT
                          / f"artifacts/stages/stage-3/d1_behavioural/{probe.probe_id}")
            #: RESUME AT EVALUATION when a completed trained checkpoint is
            #: already here -- a driver restart on this pod, or a replacement
            #: pod whose launcher restored the preserved probe. P8.4 state 2:
            #: trained + durable + not validly scored is resumed at scoring,
            #: never retrained for a different outcome.
            resumed = trained_checkpoint_state(model_root)
            if resumed is not None:
                entry["resumed_from_trained"] = resumed
                entry["trained"] = True
                model_dir = Path(resumed["model_dir"])
                entry["model_dir"] = str(model_dir)
                save()
                print(f"  resuming at evaluation from {model_dir}", flush=True)
            else:
                config = probe_config(probe, audit=out,
                                      frozen_recipe=FROZEN_RECIPE,
                                      pack_dir=PACK_DIR,
                                      overrides=C1_PROBE_OVERRIDES)
                entry["config"] = str(config.relative_to(REPO_ROOT))

                #: HAND THE CARD OVER, AND PROVE IT. A C1 attempt read a
                #: verdict saying 7.55 GiB was still allocated, started the
                #: trainer anyway and lost the probe. Both conditions are
                #: enforced: the release worked, and the card has room for
                #: the measured peak.
                import gc

                before = cuda_memory()
                gc.collect()
                handoff = complete_release(before)
                require_released(handoff, what="the D1 recovery trainer")
                require_headroom(handoff["after"],
                                 need_bytes=_trainer_bytes(),
                                 what="the D1 recovery trainer")
                entry["device_handoff"] = handoff.get("verdict")
                save()

                started = time.time()
                train = subprocess.run(
                    ["/opt/train/bin/python", str(TRAINER),
                     "--config", str(config)],
                    capture_output=True, text=True,
                    timeout=int(args.probe_train_minutes * 60 * 2))
                (out / f"{probe.probe_id}_train_tail.log").write_text(
                    (train.stdout + train.stderr)[-2000:])
                entry["train_seconds"] = round(time.time() - started, 1)
                entry["train_rc"] = train.returncode
                if train.returncode != 0:
                    entry["failed"] = "training"
                    save()
                    raise D1BehaviouralDriverError(
                        f"{probe.probe_id}: the trainer exited "
                        f"{train.returncode}; see "
                        f"{probe.probe_id}_train_tail.log")
                entry["trained"] = True
                model_dir = trained_model_dir(model_root)
                entry["model_dir"] = str(model_dir)

            #: THE TRAINED CHECKPOINT'S OWN IDENTITY, hashed the moment the
            #: unit of work completes (P8.2.1). This is what the launcher's
            #: preservation verifies a fetched trained-but-unscored probe
            #: against, and what a replacement pod's resume re-checks -- a
            #: preserved probe without an identity could never be shown to be
            #: the probe that was trained.
            import hashlib as _hashlib

            shard = Path(model_dir) / "model.safetensors"
            if shard.is_file():
                entry["trained_sha256"] = _hashlib.sha256(
                    shard.read_bytes()).hexdigest()
                entry["trained_bytes"] = shard.stat().st_size
            #: ANNOUNCED BEFORE EVALUATION: a probe that trained is thirty-plus
            #: paid minutes whether or not its generation succeeds, and the
            #: launcher's teardown gate reads this flag to refuse deleting a
            #: pod holding the only copy.
            save()

            #: THE EVALUATION PACKAGE, built the way C1 built every probe's:
            #: the trained weights beside the frozen evaluation tokenizer,
            #: each sidecar verified against the pinned hash BEFORE anything
            #: is linked. The evaluator is then aimed at the package, with the
            #: battery's own per-stratum files -- its real CLI. The first
            #: version of this driver passed a `--battery` flag the evaluator
            #: does not have, which argparse would have refused with exit 2 on
            #: a billing pod, after the probe had already trained.
            package_dir = out / "packages" / probe.probe_id
            build_evaluation_package(
                model_dir, tokenizer_source=TOKENIZER_SOURCE,
                dest=package_dir,
                expected_sidecar_sha256=TOKENIZER_SIDECAR_SHA256)

            #: THE ATTESTATION, once, from the first REAL package: the shared
            #: engine probe observes the actual vLLM runtime and the declared
            #: generation protocol is materialized with it. Every probe's
            #: generations are then ADMITTED against this before scoring.
            if attested is None:
                attested = build_attested_protocol(
                    package_dir, image_digest=args.image_digest,
                    scoring=scoring, battery_fields=battery_fields, out=out)
                record["attested_evaluation_protocol_hash"] = \
                    attested.evaluation_protocol_hash
                save()
                print(f"  attested protocol "
                      f"{attested.evaluation_protocol_hash[:12]}", flush=True)

            battery_root = REPO_ROOT / contract["battery"]["root"]
            prompt_files = [str(battery_root / f"{name}.jsonl")
                            for name in sorted(D1B.FROZEN_STRATA)]
            gen_dir = out / "generations" / probe.probe_id
            started = time.time()
            generate = subprocess.run(
                ["/opt/vllm/bin/python", str(UNCAPPED_EVAL),
                 "--model", str(package_dir), "--label", probe.probe_id,
                 "--prompts", *prompt_files,
                 "--out-dir", str(gen_dir), "--diagnostics"],
                capture_output=True, text=True,
                timeout=int(args.probe_eval_minutes * 60 * 2))
            (out / f"{probe.probe_id}_eval_tail.log").write_text(
                (generate.stdout + generate.stderr)[-2000:])
            entry["generate_seconds"] = round(time.time() - started, 1)
            entry["generate_rc"] = generate.returncode
            if generate.returncode != 0:
                entry["failed"] = "generation"
                save()
                raise D1BehaviouralDriverError(
                    f"{probe.probe_id}: generation exited "
                    f"{generate.returncode}")

            #: ADMISSION, before scoring: what the runtime actually DID,
            #: reconstructed from this probe's own raw summaries, comparable
            #: to the attestation under the project's versioned relation.
            admission = admit_generation(
                probe.probe_id, gen_dir, attested=attested,
                scoring=scoring, battery_fields=battery_fields, out=out)
            entry["generation_admission"] = {
                "comparable": True,
                "generation_fingerprint": admission["generation_fingerprint"],
                "evaluation_protocol_hash":
                    admission["evaluation_protocol_hash"],
            }
            save()

            result_path = out / "results" / f"{probe.probe_id}.json"
            per_sample = out / "per_sample" / f"{probe.probe_id}.jsonl"
            arm = next(a for a in D1B.arms(REPO_ROOT, arm_root=args.arm_root)
                       if a.arm_id == probe.arm_id)
            score = subprocess.run(
                ["/opt/train/bin/python", str(scorer),
                 "--generations", str(gen_dir), "--label", probe.probe_id,
                 "--seed", str(probe.seed), "--out", str(result_path),
                 "--per-sample", str(per_sample),
                 "--screening-arm", probe.arm_id,
                 "--init-digest", arm.artifact_digest,
                 #: The TRAINING COMPLETION identity and the OBSERVED (not
                 #: attested) fingerprint, bound into the result -- the
                 #: direction of provenance is the point.
                 "--trained-run", str(model_root / "run_completion.json"),
                 "--generation-fingerprint",
                 admission["generation_fingerprint"]],
                capture_output=True, text=True, timeout=1800)
            (out / f"{probe.probe_id}_score_tail.log").write_text(
                (score.stdout + score.stderr)[-2000:])
            if score.returncode != 0 or not result_path.is_file():
                entry["failed"] = "scoring"
                save()
                raise D1BehaviouralDriverError(
                    f"{probe.probe_id}: scoring exited {score.returncode}")
            scored = json.loads(result_path.read_text())
            entry.update({
                "scored": True,
                "result": str(result_path.relative_to(REPO_ROOT)),
                "correct_overall": scored["correct_overall"],
                "usable_rollout_rate": scored["usable_rollout_rate"],
                "n": scored["n"], "n_scorable": scored["n_scorable"],
                "result_sha256": scored["result_sha256"],
            })
            results.append({**entry, "scored_record": scored})
            #: ANNOUNCED THE MOMENT IT FINISHES, so the launcher's poll hook
            #: can pull it off-pod. A probe that finished is evidence whether
            #: or not the probes after it do.
            save()
            print(f"  correct_overall {scored['correct_overall']:.4f} "
                  f"usable {scored['usable_rollout_rate']:.4f}", flush=True)

        if args.rung == "screening":
            #: THE RANKING, from the scored probes this session produced.
            #: Pooled by the seed mean because the confirmation estimand is
            #: the "prompt-mean of the seed-mean paired difference", and a
            #: screening statistic computed another way would rank on a
            #: quantity the confirmation does not estimate.
            ranking = D1B.rank_screening(
                [{"arm": e["arm"], "seed": e["seed"],
                  "correct_overall": e["correct_overall"],
                  "usable_rollout_rate": e["usable_rollout_rate"]}
                 for e in record["probes"] if e.get("scored")],
                REPO_ROOT)
            record["ranking"] = ranking
            save()
            print("\nranking (pooled over "
                  f"{len(ranking['seeds'])} seeds, delta vs B):", flush=True)
            for row in ranking["ranked"]:
                flag = " VETOED" if row["vetoed"] else ""
                print(f"  {row['arm']}  delta {row['delta_vs_b']:+.4f}  "
                      f"usable {row['delta_usable_pooled']:+.4f}{flag}",
                      flush=True)

            selection = D1B.advance_one(ranking)
            record["selection"] = selection
            if selection.get("advanced"):
                print(f"\nADVANCING: {selection['arm']} "
                      f"({selection['state_id']}) delta "
                      f"{selection['delta_vs_b']:+.4f}, margin "
                      f"{selection['margin_over_runner_up']}", flush=True)
            else:
                print(f"\n{selection['outcome']}: {selection['why']}",
                      flush=True)
        else:
            #: THE CONFIRMATION DRIVER ENDS AT PRESERVATION. The three-way
            #: verdict (stratified prompt-cluster bootstrap, seeds as fixed
            #: blocks) is computed OFF-POD at $0 from the per-sample rows this
            #: session secured -- A3 lost its decision artifact to an on-pod
            #: aggregation stage, and the ladder has no such stage since.
            record["_verdict"] = (
                "NOT COMPUTED HERE, deliberately. The confirmation verdict is "
                "derived off-pod at $0 from the secured per-sample rows under "
                "the frozen decision rule; this session's product is the six "
                "probes' evidence.")
        record["status"] = "COMPLETE"
        save()
    except BaseException as exc:                                  # noqa: BLE001
        import traceback

        record["status"] = "FAILED"
        record["failure"] = {"type": type(exc).__name__, "message": str(exc),
                             "traceback": traceback.format_exc()}
        raise
    finally:
        record["ended_utc"] = datetime.now(timezone.utc).isoformat(
            timespec="seconds")
        written = False
        try:
            save()
            written = True
        except Exception as exc:                                  # noqa: BLE001
            print(f"could not write {evidence}: {exc}", flush=True)
        print(f"\nwrote {evidence}  status={record['status']}", flush=True)
        mark(status, terminal_marker(
            check_only=bool(args.check_only), status=record["status"],
            evidence_written=written, mismatch=mismatch))
    return 0 if record["status"] in ("COMPLETE", "CHECK_ONLY_OK") else 11


if __name__ == "__main__":
    raise SystemExit(main())
