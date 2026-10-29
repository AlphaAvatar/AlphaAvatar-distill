#!/usr/bin/env python3
"""On-pod: Research A's ONE common initialization -- D4 Top-1's operator path, on the hybrid teacher.

    cd /workspace/aad && HF_HUB_OFFLINE=1 PYTHONPATH=src:scripts:scripts/stages/stage-1 /opt/train/bin/python \\
        scripts/stages/stage-2/teacher_modes/initialization.py --records <durable>/generation \\
        --out <durable>/initialization --work <scratch> --device cuda:0 --identity <study identity> \
        [--micro-batch-size 1]

d4_scale_design.json :: initialization. The path is D4 Top-1's, step for step
(its search record's ``impl_ids`` and profiles): ATTENTION
(``attention.activation_importance_v1``) on the rh_d4 prompts, then DEPTH
(``depth.causal_kl_greedy_v1``), RESIDUAL_WIDTH (``width.global_pca_v0``) and
FFN (``ffn.activation_importance_v0``) on the base_d4 prompts, seed 20260815,
the Stage-0 search's execution (the forward baseline's, DEPTH prefix reuse on)
and target geometry -- materialized ONCE by `materialize_fixed_path` on
Qwen/Qwen3-4B@1cfa9a72 with the hybrid's own mode-balanced calibration items
(`calibration.items_from_records`) registered as the two hybrid profiles.
Each operator's internal measurement (which heads, layers, neurons; the width
PCA) runs on the hybrid: nothing is ranked or selected beyond what the
operators themselves do, and no expected digest is pinned (the bytes are new).

The final checkpoint is persisted the moment it exists (`persist_unit`, under
``--out``) with its identity, its inputs' hashes, every step's selection and
seconds, and ``initialization.json`` last: its presence is what "the
initialization exists" means, and a later attempt never re-materializes it.

MEMORY (d4_scale_design.json :: initialization.execution). The execution is the
forward baseline's (micro-batch 3). DEPTH widens a whole micro-batch's scored
rows to float32 at once: on 8k-token items that is about 12.4 GiB at
micro-batch 3, the regime that ran an L40S out of memory before
(pilot_20261018_122808). A CUDA out-of-memory error exits ``OOM_EXIT`` after
writing ``oom_mb<k>.json``; the caller then runs the declared fallback,
``--micro-batch-size 1`` (`execution.py`'s one-item-per-forward reference
path): the same operators, profiles, seed and estimand, a different grouping of
the same forwards (an execution knob, recorded in the trace, never a search).
Each step's peak allocated and reserved device memory is recorded.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.teacher_modes import hybrid_calibration as CAL  # noqa: E402

PROTOCOL = REPO / "configs/stages/stage-2/teacher_modes/generation_protocol.json"
DESIGN = REPO / "configs/stages/stage-2/teacher_modes/d4_scale_design.json"
STAGE0_PROFILES = REPO / "configs/stages/stage-0/calibration_volume/profiles_formal.json"
STAGE0_PREREG = REPO / "logs/stages/stage-0/calibration_volume/plans/volume_preregistration.json"
#: D4 Top-1's path (its search record :: commit.selected_rows[0]); the profile is each step's Stage-0 one
PATH = (("attention.activation_importance_v1", "calib.s0v_rh_d4@v1"),
        ("depth.causal_kl_greedy_v1", "calib.s0v_base_d4@v1"),
        ("width.global_pca_v0", "calib.s0v_base_d4@v1"),
        ("ffn.activation_importance_v0", "calib.s0v_base_d4@v1"))
SEED = 20260815
UNIT = "common_initialization"
SCHEMA = "aadistill.stage2.teacher_modes.initialization/v1"
#: the exit status of a CUDA out-of-memory error, which the caller answers with the declared fallback
OOM_EXIT = 75


def is_out_of_memory(exc: BaseException | None) -> bool:
    """A CUDA out-of-memory error, raised as itself or wrapped by an operator (its cause or context chain)."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if type(exc).__name__ == "OutOfMemoryError" or "CUDA out of memory" in str(exc):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def records(shards: Path, identity: str) -> dict[str, list[dict]]:
    """Each mode's sealed calibration records (the teacher session's ``cal`` stage)."""
    from stages.reasoning_data import generation_driver as GD
    out = {}
    for mode in ("H-T", "H-N"):
        got = {}
        for done in sorted(shards.glob(f"cal_{mode}_pass0_*.complete")):
            got.update(GD.sealed(shards, done.name[:-len(".complete")], identity))
        out[mode] = list(got.values())
    return out


def profiles_doc(built: dict, items_dir: Path) -> dict:
    """The two hybrid profiles: each Stage-0 profile's declaration with the hybrid items in place."""
    from aadistill.initialization.calibration.profiles import PROFILE_DOCUMENT_SCHEMA, mixture_content_sha256
    stage0 = {f"{p['profile_id']}@v{p['version']}": p for p in json.loads(STAGE0_PROFILES.read_text())["profiles"]}
    profiles = []
    for hybrid, b in sorted(built.items()):
        src = copy.deepcopy(stage0[b["source_profile"]])
        name, version = hybrid.rsplit("@v", 1)
        path = items_dir / f"{name}.items.jsonl"
        path.write_text("".join(json.dumps(i, sort_keys=True) + "\n" for i in b["items"]))
        domains = sorted({i["domain"] for i in b["items"]})
        weights = {d: src["domain_weights"].get(d, 0.0) for d in domains}
        total = sum(weights.values()) or 1.0
        src.update(profile_id=name, version=int(version),
                   description=(f"the hybrid teacher's own H-T and H-N trajectories on {b['source_profile']}'s "
                                f"prompts, mode-balanced ({b['balance']})"),
                   sources=[{"dataset_id": "aadistill/stage2_teacher_modes_calibration", "revision": "teacher_session",
                             "domain": d, "n_items": sum(i["domain"] == d for i in b["items"]),
                             "content_sha256": None, "license": "the teacher's outputs (Apache-2.0 teacher)",
                             "note": "generated on the pod, sealed with the session"} for d in domains],
                   domain_weights={d: w / total for d, w in weights.items()},
                   token_budget=sum(i["n_tokens"] for i in b["items"]), materialized=True,
                   items_path=str(path), content_sha256=mixture_content_sha256(b["items"]),
                   items_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rendered_manifest_sha256=None)
        profiles.append(src)
    return {"schema": PROFILE_DOCUMENT_SCHEMA, "profiles": profiles}


def register_operators() -> None:
    """The frozen operators and the forward baseline's, registered (idempotent). A spec validates every step's
    implementation when it is built, so this comes before `fixed_path_spec`, in the process that builds it:
    nothing registers on import (aadistill.initialization.operators.register)."""
    from shared import forward_baseline as FB
    from stages.phase_d1 import d1_session as D1S
    D1S._register_frozen_operators()
    FB.register_operators(FB.selected(REPO))


def fixed_path_spec(device: str, profile_of=None):
    """D4 Top-1's four steps on the hybrid profiles, the Stage-0 target geometry, the hybrid root.
    ``profile_of`` maps each step's Stage-0 profile to the profile it runs on (default: its hybrid one;
    the L40S check runs the Stage-0 profiles themselves, as diagnostic inputs)."""
    profile_of = profile_of or CAL.HYBRID
    from aadistill.initialization.planning.fixed_path import FixedPathSpec, FixedPathStep
    from shared import forward_baseline as FB
    from stages.phase_a3 import a3_session as A3S
    register_operators()
    teacher = json.loads(PROTOCOL.read_text())["teacher"]
    terms = FB.session_terms(FB.selected(REPO))
    target = A3S.path_spec(workdir_device=device).target_spec
    steps = tuple(FixedPathStep(impl_id=impl, profile_id=profile_of[prof], expected_artifact_digest=None,
                                label=f"{impl.split('.')[0].upper()}({profile_of[prof]})") for impl, prof in PATH)
    return FixedPathSpec(path_id="teacher_modes.common_initialization.d4_path", family=target.family,
                         target_spec=target, steps=steps, root_repo_id=teacher["model"],
                         root_revision=teacher["revision"], device=device, seed=SEED,
                         position_policy=terms["position_policy"], distribution_support=terms["distribution_support"])


def execution(micro_batch_size: int | None = None):
    """The Stage-0 search's execution: the forward baseline's, with DEPTH prefix reuse as its prereg ran it;
    ``micro_batch_size`` only as the design's out-of-memory fallback."""
    from shared import forward_baseline as FB
    ex = FB.session_terms(FB.selected(REPO))["execution"]
    if json.loads(STAGE0_PREREG.read_text())["execution"]["depth_prefix_reuse"]:
        ex = dataclasses.replace(ex, depth_prefix_reuse=True)
    if micro_batch_size is not None:
        ex = dataclasses.replace(ex, micro_batch_size=int(micro_batch_size))
    return ex


def device_peaks(device: str) -> dict:
    """Peak allocated and reserved memory since the last reset, GiB, and the card's total; then reset."""
    import torch
    if not torch.cuda.is_available():
        return {}
    dev = torch.device(device)
    out = {"allocated_gib": round(torch.cuda.max_memory_allocated(dev) / 2**30, 2),
           "reserved_gib": round(torch.cuda.max_memory_reserved(dev) / 2**30, 2),
           "total_gib": round(torch.cuda.get_device_properties(dev).total_memory / 2**30, 2)}
    torch.cuda.reset_peak_memory_stats(dev)
    return out


def materialize(spec, ex, work: Path, device: str, steps: list) -> list:
    """`materialize_fixed_path` on the registered profiles, each step's identity, seconds and memory into `steps`."""
    from aadistill.initialization.planning.fixed_path import materialize_fixed_path
    from aadistill.initialization.specs.arch import get_adapter
    from stages.phase_d1 import replay_specs as R
    register_operators()
    device_peaks(device)
    return materialize_fixed_path(
        spec, adapter=get_adapter(spec.family), workdir=work / "path", repo_root=REPO, execution=ex,
        root_loader=lambda: R.load_root_model(spec, config_overrides={"use_cache": False}, device=device),
        on_step=lambda r: steps.append({"index": r.index, "impl_id": r.impl_id,
                                        "artifact_digest": r.identity.artifact_digest,
                                        "num_parameters": r.identity.num_parameters,
                                        "seconds": round(float(r.seconds), 1), "selection": r.selection,
                                        "device_memory": device_peaks(device)}))


def run(args) -> dict:
    from aadistill.initialization.calibration.profiles import load_profiles
    from aadistill.initialization.specs.arch import get_adapter
    from stages.reasoning_data import persist_unit as PU

    out, work = Path(args.out), Path(args.work)
    proto = json.loads(PROTOCOL.read_text())
    recs = records(Path(args.records), args.identity)
    cal_tasks = {t["task_id"]: t for t in (json.loads(l) for l in (REPO / proto["inputs"]["calibration"]["path"])
                                                .read_text().splitlines() if l.strip())}
    built = CAL.items_from_records(recs, cal_tasks, proto["teacher"]["stop_ids"])
    work.mkdir(parents=True, exist_ok=True)
    items_dir = out / "calibration"
    items_dir.mkdir(parents=True, exist_ok=True)
    doc = profiles_doc(built, items_dir)
    (items_dir / "profiles.json").write_text(json.dumps(doc, indent=1) + "\n")
    load_profiles(doc)
    spec = fixed_path_spec(args.device)
    adapter = get_adapter(spec.family)
    ex = execution(args.micro_batch_size)
    steps: list = []
    t0 = time.time()
    try:
        results = materialize(spec, ex, work, args.device, steps)
    except Exception as exc:
        if is_out_of_memory(exc):
            (out / f"oom_mb{ex.micro_batch_size}.json").write_text(json.dumps(
                {"execution": ex.as_trace(), "steps_done": steps, "seconds": round(time.time() - t0, 1),
                 "error": str(exc)[:2000]}, indent=1, default=str) + "\n")
        raise
    final = results[-1]
    inputs = {"protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
              "design_sha256": hashlib.sha256(DESIGN.read_bytes()).hexdigest(),
              "profiles": {p["profile_id"]: {"content_sha256": p["content_sha256"],
                                             "items_file_sha256": p["items_file_sha256"]} for p in doc["profiles"]},
              "spec_hash": spec.spec_hash, "study_identity": args.identity}
    persisted = PU.persist_checkpoint(UNIT, Path(final.checkpoint_path), out, adapter=adapter,
                                      arch_signature=final.identity.arch_signature,
                                      num_parameters=final.identity.num_parameters, inputs=inputs)
    rec = {"schema": SCHEMA, "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "path": [{"impl_id": i, "stage0_profile": p, "profile": CAL.HYBRID[p]} for i, p in PATH], "seed": SEED,
           "teacher": proto["teacher"], "spec_hash": spec.spec_hash,
           "materialization_identity": {
               "artifact_digest": final.identity.artifact_digest, "weights_digest": final.identity.weights_digest,
               "config_sha256": final.identity.config_sha256, "arch_signature": final.identity.arch_signature,
               "single_shard_sha256": final.identity.single_shard_sha256,
               "num_parameters": final.identity.num_parameters},
           "checkpoint": {"unit": UNIT, "descriptor": str(out / UNIT / "checkpoint.descriptor.json"),
                          "status": persisted["status"]},
           "calibration": {h: {k: v for k, v in b.items() if k != "items"} for h, b in built.items()},
           "steps": steps, "seconds": round(time.time() - t0, 1), "device": args.device,
           "execution": ex.as_trace(),
           "out_of_memory_attempts": sorted(p.name for p in out.glob("oom_mb*.json")),
           "scientific_use": "the study's common initialization for every arm (d4_scale_design.json :: initialization)",
           "never": "a teacher-forced or downstream result; an approximation of the hybrid, never described as it"}
    (out / "initialization.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    shutil.rmtree(work / "path", ignore_errors=True)
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--identity", required=True, help="the study identity the calibration shards were sealed under")
    ap.add_argument("--micro-batch-size", type=int, default=None,
                    help="only the design's out-of-memory fallback (initialization.execution.on_out_of_memory)")
    args = ap.parse_args(argv)
    if (Path(args.out) / "initialization.json").is_file():
        print("initialization.json exists: the common initialization is never re-materialized")
        return 0
    try:
        rec = run(args)
    except BaseException as exc:  # noqa: BLE001 -- the parent records the failure from the exit code and this log
        traceback.print_exc()
        return OOM_EXIT if is_out_of_memory(exc) else 1
    print(json.dumps({k: rec[k] for k in ("materialization_identity", "seconds")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
