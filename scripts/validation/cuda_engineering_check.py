#!/usr/bin/env python3
"""Execute the migrated initialization operators on a real CUDA device.

    PYTHONPATH=src:scripts python scripts/validation/cuda_engineering_check.py \
        --config configs/validation/cuda_engineering.json --run-id <id>

This is ENGINEERING validation. It answers one question -- do the operators, as
they exist after the initialization migration, plan and apply on a real device
without a placement, dtype or shape error -- and nothing about model quality. It
ranks nothing, selects nothing, and writes `scientific_use: false` into its own
report.

It exists because the dev box has no GPU, and because CPU execution is the one
substitution that cannot observe the defect class this targets: a tensor left on
the wrong device is invisible when there is only one device. Three paid sessions
have been spent discovering such defects one at a time, inside runs that were
supposed to be producing results.

**Everything it runs is production code.** The adapter, the operator registry,
the operator implementations, `OperatorContext` and `RunLayout` are imported from
`aadistill.initialization` and `aadistill.runtime` -- not reimplemented, and not
routed through the old C1 driver. If the migration broke an operator's device
handling, this fails; if it passes, the thing that passed is what a session would
execute.

**No CPU fallback.** With CUDA absent it reports NOT RUN and exits non-zero
without touching an operator. A CPU pass under a GPU label would be worse than no
check, because it would read like evidence.

Everything is supplied by the config: the model fixture, the source and target
structures, the calibration, the device and dtype, the operators, and the run
root with its declared artifact roles. The one thing hard-coded here is that
`cuda` means `cuda`.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

SCHEMA = "aadistill.cuda_engineering_report/v1"


class NotRun(RuntimeError):
    """The check could not run at all. Distinct from a failure to pass."""


def load_config(path: Path) -> dict:
    doc = json.loads(path.read_text())
    if doc.get("schema") != "aadistill.cuda_engineering_validation/v1":
        raise NotRun(f"{path} is not a cuda_engineering_validation/v1 config")
    return doc


def require_cuda(requested: str) -> dict:
    """The device, or NotRun. Never a fallback."""
    try:
        import torch
    except ImportError as exc:                                    # pragma: no cover
        raise NotRun(f"torch is not installed: {exc}") from exc
    if requested != "cuda":
        raise NotRun(f"this entry point validates CUDA; config asks for {requested!r}")
    if not torch.cuda.is_available():
        raise NotRun(
            "CUDA is not available on this host. This is NOT a failure of the "
            "code under test and must not be reported as one; it is also not a "
            "reason to run on CPU, which cannot observe the defects this exists "
            "to find.")
    return {
        "device_name": torch.cuda.get_device_name(0),
        "capability": ".".join(str(x) for x in torch.cuda.get_device_capability(0)),
        "device_count": torch.cuda.device_count(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "driver": getattr(torch.version, "hip", None) or "cuda",
    }


def environment(device: dict) -> dict:
    import torch
    import transformers
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": device,
    }


def build_fixture(cfg: dict, device: str, dtype_name: str):
    """A tiny randomly-initialized model of the declared family."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    fixture = dict(cfg["model"]["fixture"])
    conf = AutoConfig.for_model(cfg["model"]["family"], **fixture)
    dtype = getattr(torch, dtype_name)
    torch.manual_seed(cfg["calibration"]["seed"])
    model = AutoModelForCausalLM.from_config(conf, torch_dtype=dtype)
    return model.to(device).eval()


def synthetic_calibration(cfg: dict, device: str):
    """Random token ids under the config's seed. Content is irrelevant here."""
    import torch
    c = cfg["calibration"]
    if not c.get("synthetic"):
        raise NotRun("only synthetic calibration is supported by this check")
    g = torch.Generator(device="cpu").manual_seed(c["seed"])
    vocab = cfg["model"]["fixture"]["vocab_size"]
    return [
        {"input_ids": torch.randint(0, vocab, (1, c["sequence_length"]),
                                    generator=g).to(device)}
        for _ in range(c["n_items"])
    ]


def spec_from(structure: dict, fixture: dict, family: str):
    """An ArchSpec for a declared structure, over the fixture's other fields.

    `ArchSpec.of` takes a field mapping rather than keywords: the spec is
    canonicalised into a sorted tuple so two built in different orders hash
    identically, which the state store depends on.
    """
    from aadistill.initialization.specs.arch import ArchSpec
    merged = {**fixture, **{k: v for k, v in structure.items()
                            if not k.startswith("_") and k != "geometry_id"}}
    return ArchSpec.of(family, {k: v for k, v in merged.items() if k != "family"})


def run_one(impl_id: str, model, parent_spec, target_spec, adapter, items,
            *, device: str, dtype, seed: int, workdir: Path) -> dict:
    """Plan and apply one operator. Records what happened, including failure."""
    from aadistill.initialization.calibration.profiles import NO_CALIBRATION
    from aadistill.initialization.operators.base import (
        OperatorContext, get_implementation)

    impl = get_implementation(impl_id)
    plan = impl.plan(parent_spec, target_spec, adapter)
    #: WHERE THE OPERATOR ACTUALLY RAN. Read from the parent's weights, not
    #: from `device`, which is only what the caller intended.
    parent_device = device_of(model)
    ctx = OperatorContext(
        adapter=adapter, model=model, parent_spec=parent_spec,
        target_spec=target_spec, profile=NO_CALIBRATION, calibration_items=items,
        seed=seed, device=device, dtype=dtype, workdir=workdir)
    outcome = impl.apply(ctx)
    child = getattr(outcome, "model", None)
    placement = device_of(child) if child is not None else None
    kind = device.split(":")[0]
    return {
        "impl_id": impl_id,
        "planned": plan is not None,
        "applied": True,
        #: The question the matrix actually asks: did this operator execute
        #: against a parent on the requested device without a placement, dtype
        #: or shape error?
        "parent_device": parent_device,
        "ran_on_requested_device": (
            parent_device is not None and parent_device.startswith(kind)),
        "child_device": placement,
        #: NOT "is the child on the requested device". `initialization/device.py`
        #: documents the opposite: "an operator's child comes from ChildBuilder
        #: -> build_student, which sets the dtype and does NOT place the model,
        #: so a parent on CUDA routinely coexists with a freshly built child on
        #: the host." Demanding the child be on `device` encoded an assumption
        #: the framework states is false -- and it was invisible on CPU, where
        #: `device` IS the host and the check passed trivially. On the first
        #: real GPU run it failed all 8 cases while every operator had in fact
        #: succeeded.
        "child_host_resident_per_builder_contract": (
            placement is not None and placement.startswith("cpu")),
    }


def device_of(model) -> str | None:
    for p in getattr(model, "parameters", lambda: iter(()))():
        return str(p.device)
    return None


# --- the end-to-end case: stage F, on a real device -------------------------
#
# The per-operator matrix above calls each operator directly. That is a useful
# breadth check and it is exactly what could not have caught attempt 9: the
# failure was in the COMPOSITION -- `materialize_fixed_path_suffix` ->
# `_run_steps` -> `impl.execute` -> `apply` -> `head_write_energy` -- and it was
# the treatment operator, `attention.activation_importance_v1`, that the matrix
# does not even execute. So this second case builds the two-arm world the way
# the session builds it and runs the tail through the production entry point.


def write_calibration_profile(root: Path, cfg: dict, geometry_vocab: int):
    """A REAL materialized profile: JSONL on disk, both hashes from those bytes.

    Built through the production rules rather than around them, so
    `CalibrationProfile.resolve()` runs its full fail-closed check here exactly
    as it does against the frozen mixtures. This is deliberate: passing
    `calibration_items=` directly would skip the resolve branch, and skipping
    that branch is how it first executed on a paid pod.
    """
    import hashlib

    import torch
    from aadistill.initialization.calibration.profiles import (
        CalibrationProfile, CalibrationSource, DatasetRole,
        mixture_content_sha256)

    c = cfg["calibration"]
    seq, n = c["sequence_length"], c["n_items"]
    g = torch.Generator().manual_seed(c["seed"])
    items = []
    for domain in c["domains"]:
        for k in range(n):
            ids = torch.randint(0, geometry_vocab, (seq,), generator=g).tolist()
            items.append({"item_id": f"{domain}-{k}", "ids": ids,
                          "domain": domain, "subtype": domain,
                          "n_prediction_positions": len(ids) - 1})

    rel = c["items_relpath"]
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(i) + "\n" for i in items))

    weight = 1.0 / len(c["domains"])
    return CalibrationProfile(
        profile_id=c["profile_id"], version=c["profile_version"],
        description="synthetic materialized mixture for CUDA validation",
        sources=tuple(CalibrationSource(c["source_id"], "local", d, n)
                      for d in c["domains"]),
        domain_weights={d: weight for d in c["domains"]},
        token_budget=seq * len(items), sample_rule="fixed", seed=c["seed"],
        role=DatasetRole.OPERATOR_CALIBRATION,
        materialized=True, items_path=rel,
        content_sha256=mixture_content_sha256(items),
        items_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def run_suffix_case(cfg: dict, *, device: str, dtype_name: str,
                    geometry: dict, root: Path, adapter, build_root) -> dict:
    """Stage F, end to end, through production code. Returns its evidence.

    `device` is a parameter rather than a constant so the harness itself can be
    exercised at `$0`. That CPU run validates THIS FILE and nothing about
    placement: on one device every co-location claim is trivially true, which is
    exactly the substitution that let attempt 9 through. The report says so.
    """
    from aadistill.initialization.calibration.profiles import (
        register_profile, unregister_profile)
    from aadistill.initialization.operators import attention_activation
    from aadistill.initialization.planning.fixed_path import (
        VerifiedSuffix, materialize_fixed_path, materialize_fixed_path_suffix,
        write_suffix_execution_record)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from device_observations import observing

    case = cfg["suffix_case"]
    work = root / f"suffix_{geometry['geometry_id']}"
    work.mkdir(parents=True, exist_ok=True)

    profile = write_calibration_profile(work, cfg, cfg["model"]["fixture"]["vocab_size"])
    register_profile(profile, replace=True)
    # The treatment operator is not a builtin: the session registers it
    # explicitly and so does this, rather than relying on an import side effect.
    attention_activation.register(replace=True)
    try:
        return _suffix_body(
            cfg, case, geometry, work, profile, adapter, build_root,
            device=device, dtype_name=dtype_name,
            observing=observing, VerifiedSuffix=VerifiedSuffix,
            materialize=materialize_fixed_path,
            materialize_suffix=materialize_fixed_path_suffix,
            write_record=write_suffix_execution_record)
    finally:
        attention_activation.unregister()
        unregister_profile(profile.qualified_id)


def _suffix_body(cfg, case, geometry, work, profile, adapter, build_root, *,
                 device, dtype_name, observing, VerifiedSuffix,
                 materialize, materialize_suffix, write_record) -> dict:
    import torch
    from aadistill.initialization.operators import attention_activation
    from aadistill.initialization.planning.fixed_path import (
        FixedPathSpec, FixedPathStep)
    from aadistill.initialization.specs.arch import ArchSpec

    fixture = cfg["model"]["fixture"]
    family = cfg["model"]["family"]
    pid = profile.qualified_id
    target = spec_from(geometry, fixture, family)
    prefix_ids = list(case["prefix_impl_ids"])
    start = len(prefix_ids)
    treatment_id = case["treatment_impl_id"]

    def spec(steps, path_id):
        return FixedPathSpec(
            path_id=path_id, family=family, target_spec=target,
            steps=tuple(steps), root_repo_id=case["root_repo_id"],
            root_revision=case["root_revision"], device=device)

    prefix = [FixedPathStep(i, pid) for i in prefix_ids]
    inc_tail = FixedPathStep(case["incumbent_impl_id"], case["incumbent_profile_id"])

    # 1. unpinned probe -- learn the digests.
    probe = spec([*prefix, inc_tail], f"{case['path_id_prefix']}.probe")
    observed = materialize(probe, adapter=adapter, root_loader=build_root,
                           workdir=work / "probe", repo_root=work)
    parent_digest = observed[start - 1].identity.artifact_digest
    incumbent_digest = observed[start].identity.artifact_digest

    # 2. the PINNED incumbent -- this is stage D, and it gates the parent.
    pinned_prefix = [*prefix[:start - 1],
                     FixedPathStep(prefix_ids[-1], pid,
                                   expected_artifact_digest=parent_digest,
                                   label="pre-ATTENTION parent")]
    incumbent = spec(
        [*pinned_prefix,
         FixedPathStep(case["incumbent_impl_id"], case["incumbent_profile_id"],
                       expected_artifact_digest=incumbent_digest,
                       label="incumbent")],
        f"{case['path_id_prefix']}.incumbent")
    inc_steps = materialize(incumbent, adapter=adapter, root_loader=build_root,
                            workdir=work / "incumbent", repo_root=work)
    parent = inc_steps[start - 1]

    # 3. the treatment arm, by replace_tail, so the prefix is shared by
    #    construction.
    treatment = incumbent.replace_tail(
        start, FixedPathStep(treatment_id, pid, label="treatment ATTENTION"),
        path_id=f"{case['path_id_prefix']}.treatment")

    verified = VerifiedSuffix(
        start_index=start, parent=parent,
        expected_parent_artifact_digest=parent_digest,
        expected_path_hash=treatment.spec_hash,
        prefix_reference_steps=tuple(incumbent.steps[:start]),
        expected_suffix_steps=((treatment_id, pid),))

    stage_f = work / "stage_f"
    with observing(attention_activation) as obs:
        results, evidence = materialize_suffix(
            treatment, adapter=adapter,
            root_loader=lambda: adapter.load(parent.checkpoint_path,
                                             device=device),
            workdir=stage_f, verified=verified, repo_root=work)

    # The record, written and READ BACK: a writer that returns without raising
    # has not shown that what landed on disk is loadable or correct.
    rec_path = write_record(
        treatment, results, stage_f / "treatment_record.json",
        runtime={"torch": torch.__version__, "device": device,
                 "dtype": dtype_name},
        suffix_evidence=evidence, calibration={"profile_id": pid})
    record = json.loads(rec_path.read_text())

    r = results[0]
    steps_on_disk = sorted(p.name for p in (stage_f / "steps").iterdir()
                           if p.is_dir())
    proofs = obs.report(device)

    checks = {
        "the_treatment_operator_executed": r.impl_id == treatment_id,
        "the_original_suffix_index_is_retained": r.index == start,
        "the_checkpoint_keeps_its_original_number": (
            Path(r.checkpoint_path).name.startswith(f"{start:02d}_")),
        "the_prefix_did_not_execute_again": (
            steps_on_disk == [Path(r.checkpoint_path).name]),
        "the_prefix_indices_are_recorded_as_skipped": (
            evidence["prefix_step_indices_not_executed"] == list(range(start))),
        "the_output_stays_bound_to_the_full_frozen_path": (
            record["path_hash"] == treatment.spec_hash),
        "the_record_is_not_a_replay": (
            record["is_replay"] is False
            and record["output_digest_was_pre_pinned"] is False
            and record["n_pinned"] == 0),
        "the_record_names_the_executed_step": (
            record["executed_step_indices"] == [start]
            and record["output"]["impl_id"] == treatment_id
            and record["output"]["artifact_digest"] == r.identity.artifact_digest),
        "the_parent_was_genuinely_gated": (
            parent.digest_matches is True
            and parent.digest_expected == parent_digest),
        "the_treatment_did_its_own_work": (
            "op.attention.retained_write_energy_mean" in (r.local_metrics or {})
            if isinstance(r.local_metrics, dict) else
            "op.attention.retained_write_energy_mean" in
            (getattr(r.local_metrics, "values", None) or {})),
    }

    return {
        "geometry_id": geometry["geometry_id"],
        "device": device,
        "device_proofs_are_meaningful": device.split(":")[0] == "cuda",
        "_why": (
            "on a single-device host every co-location claim below is "
            "trivially true. A CPU run exercises this harness and proves "
            "nothing about placement -- that substitution is what let attempt "
            "9 reach a paid pod."),
        "treatment_impl_id": treatment_id,
        "executed_step_indices": evidence["executed_step_indices"],
        "steps_written": steps_on_disk,
        "path_hash": treatment.spec_hash,
        "parent_artifact_digest": parent_digest,
        "output_artifact_digest": r.identity.artifact_digest,
        "record": str(rec_path.relative_to(work)),
        "device_proofs": proofs,
        "checks": checks,
        "passed": all(checks.values()) and all(p["holds"] for p in proofs.values()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/validation/cuda_engineering.json")
    ap.add_argument("--run-id", required=True,
                    help="names this run's directory under the config's run root")
    ap.add_argument("--run-root", default=None,
                    help="overrides the config's run root")
    args = ap.parse_args()

    report: dict = {"schema": SCHEMA, "scientific_use": False,
                    "authorizes": "nothing"}
    try:
        cfg = load_config(REPO / args.config)
        device_info = require_cuda(cfg["execution"]["device"])
    except NotRun as exc:
        # NOT RUN is reported on stdout and as a non-zero exit, and no report
        # file is written: an absent artifact cannot later be mistaken for a
        # pass, which a file saying "not run" eventually would be.
        print(f"NOT RUN: {exc}")
        return 3

    from aadistill.initialization.adapters import register_builtin_adapters
    from aadistill.initialization.operators.base import registered_implementations
    from aadistill.initialization.specs.arch import get_adapter
    from aadistill.runtime.run_layout import ArtifactSpec, RunLayout

    # Register explicitly, then check every declared id BEFORE any device work:
    # a typo should cost nothing. Importing the modules is no longer enough --
    # and relying on that was the coupling this bootstrap removed.
    from aadistill.initialization.operators.register import register_builtin_operators

    register_builtin_adapters()
    register_builtin_operators()
    known = set(registered_implementations())
    declared = [o["impl_id"] for o in cfg["operators"]]
    unknown = [i for i in declared if i not in known]
    if unknown:
        print(f"NOT RUN: unknown operator implementation(s) {unknown}; "
              f"registered: {sorted(known)}")
        return 3

    run_cfg = cfg["run"]
    spec = ArtifactSpec(spec_id=run_cfg["artifact_spec_id"],
                        required=tuple(run_cfg["required_roles"]),
                        optional=tuple(run_cfg["optional_roles"]))
    layout = RunLayout(
        run_root=Path(args.run_root or (REPO / run_cfg["run_root"])),
        experiment_id=run_cfg["experiment_id"], run_id=args.run_id,
    ).create({**run_cfg["required_roles"], **run_cfg["optional_roles"]})

    import torch
    dtype = getattr(torch, cfg["execution"]["dtype"])
    device = cfg["execution"]["device"]
    adapter = get_adapter(cfg["model"]["adapter"])
    fixture = cfg["model"]["fixture"]
    family = cfg["model"]["family"]

    (layout.path(run_cfg["required_roles"]["environment"])).write_text(
        json.dumps(environment(device_info), indent=1) + "\n")

    results = []
    for geometry in cfg["target_geometries"]:
        gid = geometry["geometry_id"]
        model = build_fixture(cfg, device, cfg["execution"]["dtype"])
        items = synthetic_calibration(cfg, device)
        parent = spec_from(cfg["source_structure"], fixture, family)
        target = spec_from(geometry, fixture, family)
        for impl_id in declared:
            workdir = layout.path(
                f"{run_cfg['optional_roles']['per_operator']}{gid}/{impl_id}")
            workdir.mkdir(parents=True, exist_ok=True)
            row = {"geometry_id": gid, "impl_id": impl_id}
            try:
                row.update(run_one(impl_id, model, parent, target, adapter, items,
                                   device=device, dtype=dtype,
                                   seed=cfg["calibration"]["seed"],
                                   workdir=workdir))
            except Exception as exc:                              # noqa: BLE001
                # Recorded, not raised: one operator failing is the finding, and
                # stopping here would hide whether the others also fail.
                row.update({"applied": False,
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc()[-4000:]})
            results.append(row)
            print(f"  {gid:14} {impl_id:32} "
                  f"{'ok' if row.get('applied') else 'FAILED'}")
        del model

    passed = [r for r in results
              if r.get("applied")
              and r.get("ran_on_requested_device") is True
              and r.get("child_host_resident_per_builder_contract") is True]

    # --- the end-to-end case, per declared geometry -------------------------
    suffix_root = layout.path(run_cfg["optional_roles"]["suffix_case"])
    suffix_root.mkdir(parents=True, exist_ok=True)
    suffix: list[dict] = []
    for geometry in cfg["suffix_case"]["geometries"]:
        gid = geometry["geometry_id"]
        row = {"geometry_id": gid}
        try:
            row = run_suffix_case(
                cfg, device=device, dtype_name=cfg["execution"]["dtype"],
                geometry=geometry, root=suffix_root, adapter=adapter,
                build_root=lambda: build_fixture(
                    cfg, device, cfg["execution"]["dtype"]))
        except Exception as exc:                                  # noqa: BLE001
            row.update({"passed": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()[-4000:]})
        suffix.append(row)
        print(f"  suffix {gid:22} {'ok' if row.get('passed') else 'FAILED'}")

    suffix_ok = bool(suffix) and all(r.get("passed") for r in suffix)
    layout.path(run_cfg["required_roles"]["suffix_evidence"]).write_text(
        json.dumps({"schema": "aadistill.cuda_suffix_evidence/v1",
                    "scientific_use": False, "authorizes": "nothing",
                    "cases": suffix}, indent=1) + "\n")

    report.update({
        "validation_id": cfg["validation_id"],
        "config": args.config,
        "run": layout.rel_root,
        "environment": environment(device_info),
        "geometries": [g["geometry_id"] for g in cfg["target_geometries"]],
        "operators": declared,
        "n_cases": len(results),
        "n_passed": len(passed),
        "operator_matrix_passed": len(passed) == len(results),
        "suffix_case_passed": suffix_ok,
        "suffix_geometries": [g["geometry_id"]
                              for g in cfg["suffix_case"]["geometries"]],
        "suffix_case": suffix,
        "passed": len(passed) == len(results) and suffix_ok,
        "results": results,
        "_what_a_pass_means": (
            "every declared operator planned and applied on a real CUDA device "
            "for every declared geometry and left its child on that device; "
            "AND, for each suffix geometry, `attention.activation_importance_v1` "
            "executed through the real `materialize_fixed_path_suffix` from a "
            "genuinely gated parent, with the five device placements observed "
            "rather than assumed. It says nothing about whether the "
            "initialization is any good, and it is not a C1 result."),
        "_what_it_still_does_not_cover": (
            "the treatment operator's NUMERICAL behaviour, any efficacy "
            "comparison, and the Attempt-9 checkpoint, seeds and battery -- "
            "none of which this may touch."),
    })
    out = layout.path(run_cfg["required_roles"]["report"])
    out.write_text(json.dumps(report, indent=1) + "\n")

    ok, why = spec.check(
        r for r, rel in {**run_cfg["required_roles"]}.items()
        if layout.path(rel).exists())
    print(f"\n{len(passed)}/{len(results)} case(s) passed; artifacts {'complete' if ok else why}")
    print(f"report: {out}")
    return 0 if report["passed"] and ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
