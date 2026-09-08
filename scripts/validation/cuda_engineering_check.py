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
    ctx = OperatorContext(
        adapter=adapter, model=model, parent_spec=parent_spec,
        target_spec=target_spec, profile=NO_CALIBRATION, calibration_items=items,
        seed=seed, device=device, dtype=dtype, workdir=workdir)
    outcome = impl.apply(ctx)
    child = getattr(outcome, "model", None)
    placement = device_of(child) if child is not None else None
    return {
        "impl_id": impl_id,
        "planned": plan is not None,
        "applied": True,
        "child_device": placement,
        "child_on_requested_device": (
            placement is not None and placement.startswith(device.split(":")[0])),
    }


def device_of(model) -> str | None:
    for p in getattr(model, "parameters", lambda: iter(()))():
        return str(p.device)
    return None


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

    passed = [r for r in results if r.get("applied")
              and r.get("child_on_requested_device") is not False]
    report.update({
        "validation_id": cfg["validation_id"],
        "config": args.config,
        "run": layout.rel_root,
        "environment": environment(device_info),
        "geometries": [g["geometry_id"] for g in cfg["target_geometries"]],
        "operators": declared,
        "n_cases": len(results),
        "n_passed": len(passed),
        "passed": len(passed) == len(results),
        "results": results,
        "_what_a_pass_means": (
            "every declared operator planned and applied on a real CUDA device "
            "for every declared geometry, and left its child on that device. It "
            "says nothing about whether the initialization is any good."),
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
