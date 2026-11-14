#!/usr/bin/env python3
"""On-pod driver of the layer-merge round (RI-LM1): the statistics, both candidates, their checks and diagnostics,
state evaluation and eval-set-1 in both modes. Every finished unit is written the moment it exists.

    cd /workspace/aad && HF_HUB_OFFLINE=1 PYTHONPATH=src:scripts:scripts/stages/stage-1 /opt/train/bin/python \\
        scripts/stages/stage-1/reasoning_inheritance/layer_merge_driver.py --run-id <run> --out <evidence dir> \\
        --work <scratch> --status <file> --device cuda --vllm-python /opt/vllm/bin/python [--resume <archive>]

    # $0 on the dev box: the design, the spans, D's digests, the protocol pin and the anchor session's config hashes
    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python \\
        scripts/stages/stage-1/reasoning_inheritance/layer_merge_driver.py --check-only --out <s>/out --work <s>/work \\
        --status <s>/st

Phases (``configs/stages/stage-1/reasoning_inheritance/layer_merge/design.json`` is the experiment), each GPU phase its
own child process so the memory is released between them:

* ``contract`` ($0, host): the design and the frozen selection by sha256; the spans re-derived from the selection; D's
  digests file is D's; RI-S1's evaluation protocol unchanged (its sha256); the anchor session's operator config hashes
  against the journal (`driver.contract`).
* ``build``: the teacher (every shard hashed); D rebuilt from it with D's own builder and compared tensor by tensor with
  D's digests (it is the base, so a mismatch stops here); the statistics forward (`layer_merge.collect_statistics`) and
  the alignments, saved at once (``records/merge_statistics.*``) -- skipped when a previous attempt's are taken over;
  the fold and permutation equivalence checks; M-avg and M-act built, checked and saved under ``work/groups``.
* ``relay``: both candidates and the statistics to the private relay, downloaded again and re-hashed.
* ``diagnostics``: each span's residual updates, D's / M-avg's / M-act's block against the teacher span's.
* ``state_eval``: KL and NLL against the root teacher on ``state_eval_hybrid@v3`` (diagnostic only).
* ``evaluate``: eval-set-1, every unit, in the design's order, through RI-S1's runners (`driver.fluid`,
  `driver.capability`, `driver.package`) under RI-S1's protocol.

``--resume <archive>`` takes over a previous attempt's records and sealed parts (`driver.take_over`); the candidates
come back from the relay by sha256 (`driver.relay_restore`). Nothing a record or a sealed part holds is redone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.reasoning_inheritance import driver as RI  # noqa: E402
from stages.reasoning_inheritance import layer_merge as LMX  # noqa: E402

SCHEMA = "aadistill.stage1.reasoning_inheritance.layer_merge.driver_report/v1"
GROUPS = LMX.CANDIDATES
STATS = "merge_statistics"


class DriverError(RuntimeError):
    pass


def sha(p: Path) -> str:
    return RI.sha256_file(p)


def protocol_sha256(ri_design: dict) -> str:
    return hashlib.sha256(json.dumps(ri_design["evaluation"], sort_keys=True).encode()).hexdigest()


# ---- the contract ($0) ----------------------------------------------------------------------------------------------

def contract(work: Path) -> dict:
    d = LMX.design()
    base = d["base"]
    bad = []
    for k in ("ri_s1_design", "frozen_selection"):
        got = sha(REPO / base[k]["path"])
        if got != base[k]["sha256"]:
            bad.append(f"{base[k]['path']} hashes to {got[:12]}, the design pins {base[k]['sha256'][:12]}")
    spans = LMX.merge_spans(d)
    dig = RI.read_json(LMX.DIGESTS)
    if dig["checkpoint"]["weights_sha256"] != base["d_checkpoint"]["weights_sha256"]:
        bad.append("d_tensor_digests.json is not D's")
    ri = RI.read_json(REPO / base["ri_s1_design"]["path"])
    if protocol_sha256(ri) != d["evaluation"]["protocol_sha256"]:
        bad.append("RI-S1's evaluation protocol is not the one the design pins")
    if bad:
        raise DriverError("; ".join(bad))
    ri_contract = RI.contract(work, "cpu")
    return {"design_sha256": sha(LMX.DESIGN), "digests_sha256": sha(LMX.DIGESTS), "spans": spans,
            "protocol_sha256": d["evaluation"]["protocol_sha256"], "anchor_session": ri_contract}


# ---- build (GPU) ----------------------------------------------------------------------------------------------------

def load_statistics(out: Path) -> dict | None:
    """A taken-over statistics file, verified against its record; None when there is none."""
    rec_p, ten_p = out / "records" / f"{STATS}.json", out / "records" / f"{STATS}.safetensors"
    if not (rec_p.is_file() and ten_p.is_file()):
        return None
    rec = RI.read_json(rec_p)
    if sha(ten_p) != rec["tensors_sha256"]:
        raise DriverError(f"{ten_p.name} does not hash to its record's {rec['tensors_sha256'][:12]}")
    from safetensors.torch import load_file
    return load_file(str(ten_p))


def build(out: Path, work: Path, device: str, status) -> dict:
    import torch
    from safetensors.torch import save_file
    from aadistill.initialization.planning import frozen_mapping as FM
    from aadistill.initialization.specs.arch import get_adapter
    from stages.teacher_modes import full_search as FS
    d, sel = LMX.design(), RI.selection()
    s = RI.session(work, device)
    adapter = get_adapter(s.config.target_spec.family)
    spans = LMX.merge_spans(d, sel)
    seed = int(sel["source"]["seed"])
    digests = RI.read_json(LMX.DIGESTS)["tensors"]
    rec: dict = {"started_utc": RI.utc(), "spans": spans}
    t0 = time.time()
    teacher, rec["root"] = FS.load_root(device)
    teacher.config.use_cache = False
    rec["seconds_teacher"] = round(time.time() - t0, 1)

    #: the base: D rebuilt with D's own builder, compared tensor by tensor with D (a mismatch means the base is not D)
    base = FM.keep_blocks(adapter, teacher, [int(b) for b in sel["kept_blocks"]], seed=seed)
    sd = base.state_dict()
    tied = bool(getattr(base.config, "tie_word_embeddings", False))
    names = [n for n in sd if not (tied and n == "lm_head.weight")]
    diff = [n for n in names if FM.tensor_sha256(sd[n]) != digests.get(n)]
    rec["d_rebuild"] = {"tensors": len(names), "equal_to_D": len(names) - len(diff), "differ": diff[:8],
                        "note": "every tensor of D rebuilt from the teacher compared with D's digests (a tensor "
                                "identity; the safetensors file is not written, so this is not a file-hash check)"}
    del base, sd
    if diff or set(names) != set(digests):
        raise DriverError(f"D rebuilt from the teacher is not D: {diff[:6]} / names {sorted(set(names) ^ set(digests))[:6]}")
    status.mark("D_REBUILT_EQUAL")

    n_q, n_kv, hd = adapter.head_groups(adapter.spec_of(teacher))
    _items, active, batches = LMX.calibration(s, teacher, device)
    stats = load_statistics(out)
    if stats is None:
        t1 = time.time()
        st = LMX.collect_statistics(adapter, teacher, spans, batches, active)
        rec["statistics_seconds"] = round(time.time() - t1, 1)
        exp = d["statistics"]["expected"]
        if st.sequences != exp["packed_sequences"] or st.positions != exp["statistics_positions"]:
            raise DriverError(f"{st.sequences} sequences / {st.positions} positions, the design expects "
                              f"{exp['packed_sequences']} / {exp['statistics_positions']}")
        t2 = time.time()
        al = LMX.alignments(st, n_q, n_kv)
        rec["alignment_seconds"] = round(time.time() - t2, 1)
        stats = LMX.statistics_tensors(st, al)
        del st
        torch.cuda.empty_cache()
        ten_p = out / "records" / f"{STATS}.safetensors"
        ten_p.parent.mkdir(parents=True, exist_ok=True)
        tmp = ten_p.with_name(f".{ten_p.name}.partial")
        save_file(stats, str(tmp))
        os.replace(tmp, ten_p)
        RI.write_json(out / "records" / f"{STATS}.json", {
            "tensors_sha256": sha(ten_p), "sequences": exp["packed_sequences"], "positions": exp["statistics_positions"],
            "profile_id": d["statistics"]["profile_id"], "position_policy": active.report(),
            "forward_context": s.config.forward_context.as_dict(),
            "pairs": {f"{a}->{b}": x["summary"] for (a, b), x in al.items()},
            "permutation_sha256": {k: hashlib.sha256(v.numpy().tobytes()).hexdigest()
                                   for k, v in stats.items() if "perm" in k},
            "producer_run": out.name, "utc": RI.utc()})
        status.mark("STATISTICS_SAVED")
    else:
        rec["statistics"] = "taken over from a previous attempt (records/merge_statistics.*, re-hashed)"

    tol = float(d["checks"]["fold_equivalence"]["tolerance"])
    first = next(iter(batches()))
    eq = LMX.equivalence_checks(adapter, teacher, spans, stats, first, tol)
    RI.write_json(out / "records" / "merge_checks.json", eq)
    if not eq["passed"]:
        raise DriverError(f"folding or a permutation changes a block's forward by {eq['worst']:.2e} > {tol}")
    status.mark("EQUIVALENCE_OK")

    rec["candidates"] = {}
    for name in GROUPS:
        model, crec = LMX.build_candidate(adapter, teacher, sel, spans, name, stats, digests, seed=seed)
        if crec["checks"]["num_parameters"] != int(d["base"]["d_checkpoint"]["num_parameters"]):
            raise DriverError(f"{name}: {crec['checks']['num_parameters']} parameters, D has "
                              f"{d['base']['d_checkpoint']['num_parameters']}")
        if adapter.spec_of(model) != adapter.spec_of(teacher).replace(num_hidden_layers=len(sel["kept_blocks"])):
            raise DriverError(f"{name}: the structure is not D's")
        ident = RI._save(adapter, model, work / "groups" / name)
        crec.update(RI._ident_record(ident))
        crec["config_equals_D"] = ident.config_sha256 == d["base"]["d_checkpoint"]["config_sha256"]
        rec["candidates"][name] = crec
        del model
        torch.cuda.empty_cache()
        RI.write_json(out / "records" / "merge_build_progress.json", rec)
        status.mark(f"BUILT:{name}")
    rec["finished_utc"] = RI.utc()
    rec["gpu_peak"] = RI._gpu_peak(device)
    RI.write_json(out / "records" / "merge_build.json", rec)
    status.mark("CANDIDATES_BUILT")
    return rec


# ---- relay ----------------------------------------------------------------------------------------------------------

def relay_upload(work: Path, out: Path, run_id: str) -> dict:
    """Both candidates and the statistics to the private relay, then downloaded again and re-hashed."""
    from huggingface_hub import HfApi, hf_hub_download
    import tempfile
    d = LMX.design()
    repo, prefix = d["products"]["relay_repo"], d["products"]["relay_prefix"]
    srec = RI.read_json(out / "records" / f"{STATS}.json")
    token = RI.POD_TOKEN.read_text().strip() if RI.POD_TOKEN.is_file() else None
    api = HfApi(token=token)
    files = {f"records/{STATS}.safetensors": out / "records" / f"{STATS}.safetensors"}
    for g in GROUPS:
        for p in sorted((work / "groups" / g).iterdir()):
            if p.is_file():
                files[f"groups/{g}/{p.name}"] = p
    base = f"{prefix}/{srec['tensors_sha256'][:16]}"
    want = {k: sha(p) for k, p in files.items()}
    t0 = time.time()
    for k, p in sorted(files.items()):
        api.upload_file(path_or_fileobj=str(p), path_in_repo=f"{base}/{k}", repo_id=repo, repo_type="model",
                        commit_message=f"stage1 layer merge {run_id}: {k}")
    t1, got = time.time(), {}
    with tempfile.TemporaryDirectory(dir=str(work)) as tmp:
        for k in sorted(files):
            got[k] = sha(Path(hf_hub_download(repo, f"{base}/{k}", repo_type="model", token=token,
                                              cache_dir=f"{tmp}/hf", force_download=True)))
    return {"relay": repo, "prefix": base, "files_sha256": want, "verified": got == want,
            "bytes": {k: p.stat().st_size for k, p in files.items()},
            "upload_seconds": round(t1 - t0, 1), "verify_seconds": round(time.time() - t1, 1),
            "producer_run": run_id, "utc": RI.utc(),
            "rule": "uploaded from the pod's copy, downloaded again into a fresh cache and re-hashed"}


# ---- diagnostics and state evaluation (GPU) -------------------------------------------------------------------------

def diagnostics(out: Path, work: Path, device: str, status) -> dict:
    import copy
    import torch
    from aadistill.initialization.specs.arch import get_adapter
    from stages.teacher_modes import full_search as FS
    d, sel = LMX.design(), RI.selection()
    s = RI.session(work, device)
    adapter = get_adapter(s.config.target_spec.family)
    spans = LMX.merge_spans(d, sel)
    teacher, _root = FS.load_root(device)
    teacher.config.use_cache = False
    brec = RI.read_json(out / "records" / "merge_build.json")
    blocks = adapter.blocks(teacher)
    cands = {"D": {sp[0]: copy.deepcopy(blocks[sp[0]]) for sp in spans}}
    for g in GROUPS:
        ident = RI._identify(adapter, work / "groups" / g)
        if ident.artifact_digest != brec["candidates"][g]["artifact_digest"]:
            raise DriverError(f"{g}'s checkpoint is not the recorded one")
        model = adapter.load(str(work / "groups" / g), device=device)
        cands[g] = {sp[0]: copy.deepcopy(adapter.blocks(model)[LMX.child_index(sel, sp[0])]) for sp in spans}
        del model
        torch.cuda.empty_cache()
    _items, active, batches = LMX.calibration(s, teacher, device)
    t0 = time.time()
    rec = LMX.diagnostics(adapter, teacher, spans, cands, batches, active)
    rec.update(seconds=round(time.time() - t0, 1), utc=RI.utc())
    RI.write_json(out / "records" / "merge_diagnostics.json", rec)
    status.mark("DIAGNOSTICS_OK")
    return rec


def state_eval(out: Path, work: Path, device: str, status) -> dict:
    from aadistill.initialization.specs.arch import get_adapter
    from stages.teacher_modes import full_search as FS
    s = RI.session(work, device)
    adapter = get_adapter(s.config.target_spec.family)
    teacher, _root = FS.load_root(device)
    t0 = time.time()
    s.evaluator.prime_reference(teacher)
    rec = {"started_utc": RI.utc(), "measurement_protocol_id": s.evaluator.measurement_protocol_id,
           "suite": s.suite_manifest.get("suite_id"), "suite_content_sha256": s.suite_manifest.get("content_sha256"),
           "forward_context": s.config.forward_context.as_dict(), "prime_seconds": round(time.time() - t0, 1),
           "role": "diagnostic only", "groups": {}}
    brec = RI.read_json(out / "records" / "merge_build.json")
    for g in GROUPS:
        ident = RI._identify(adapter, work / "groups" / g)
        if ident.artifact_digest != brec["candidates"][g]["artifact_digest"]:
            raise DriverError(f"{g}'s checkpoint is not the recorded one")
        model = adapter.load(str(work / "groups" / g), device=device)
        model.config.use_cache = False
        t1 = time.time()
        ev = s.evaluator.evaluate(model, ident.artifact_digest)
        rec["groups"][g] = {"artifact_digest": ident.artifact_digest, "seconds": round(time.time() - t1, 1),
                            "values": dict(getattr(ev, "values", {}) or {}), "positions": getattr(ev, "positions", None),
                            "detail": json.loads(json.dumps(getattr(ev, "detail", {}) or {}, default=str))}
        del model
        RI.write_json(out / "records" / "merge_state_eval.json", rec)
    rec["finished_utc"] = RI.utc()
    RI.write_json(out / "records" / "merge_state_eval.json", rec)
    status.mark("STATE_EVAL_OK")
    return rec


# ---- evaluation (vLLM, RI-S1's runners) -----------------------------------------------------------------------------

def evaluation_plan(kinds=None, modes=None, groups=None) -> list[tuple[str, str, str]]:
    plan = [tuple(x) for x in LMX.design()["evaluation"]["order"]]
    return [x for x in plan if (not groups or x[0] in groups) and (not modes or x[1] in modes)
            and (not kinds or x[2] in kinds)]


def evaluate(out: Path, work: Path, vllm_py: str, status, attempt: str, opts: dict | None = None,
             kinds=None, modes=None, groups=None) -> dict:
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    brec = RI.read_json(out / "records" / "merge_build.json")
    summary = RI.read_json(out / "state" / "evaluation.json") if (out / "state" / "evaluation.json").is_file() else {}
    summary["_execution"] = {**(opts or {}), "attempt": attempt, "kinds": kinds, "modes": modes, "groups": groups}
    for g, mode, kind in evaluation_plan(kinds, modes, groups):
        name = f"{g}_{mode}"
        unit_dir = out / "eval" / name
        pkg = work / "packages" / g
        if not (pkg / "config.json").exists():
            pk = RI.package(work / "groups" / g, pkg)
            pk["artifact_digest"] = brec["candidates"][g]["artifact_digest"]
            RI.write_json(out / "eval" / f"package_{g}.json", pk)
        status.mark(f"EVAL_START:{name}:{kind}")
        t0 = time.time()
        run = RI.fluid if kind == "fluid" else RI.capability
        mns = (opts or {}).get("max_num_seqs")
        unit_opts = {**(opts or {}), "max_num_seqs": (mns.get(g) if isinstance(mns, dict) else mns)}
        res = run(pkg, unit_dir, name, mode, vllm_py, env, attempt, unit_opts)
        summary.setdefault(name, {})[kind] = {**res, "seconds": round(time.time() - t0, 1), "utc": RI.utc()}
        RI.write_json(out / "state" / "evaluation.json", summary)
        status.mark(f"EVAL_OK:{name}:{kind}")
    return summary


# ---- main -----------------------------------------------------------------------------------------------------------

def child(phase: str, a) -> int:
    out, work, status = Path(a.out), Path(a.work), RI.Status(a.status)
    if phase == "build":
        build(out, work, a.device, status)
    elif phase == "relay":
        rec = relay_upload(work, out, a.run_id)
        RI.write_json(out / "records" / "products.json", rec)
        if not rec["verified"]:
            raise DriverError("the relay copy does not re-hash to the pod's")
    elif phase == "restore":
        rec = RI.relay_restore(work, RI.read_json(out / "records" / "products.json"))
        RI.write_json(out / "records" / f"restored.{a.run_id}.json", rec)
    elif phase == "diagnostics":
        diagnostics(out, work, a.device, status)
    elif phase == "state_eval":
        state_eval(out, work, a.device, status)
    else:
        raise DriverError(f"unknown phase {phase!r}")
    return 0


def run_child(phase: str, a, *, online: bool = False) -> None:
    env = {**os.environ, **({"HF_HUB_OFFLINE": "0"} if online else {})}
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", phase, "--run-id", a.run_id, "--out", a.out,
           "--work", a.work, "--status", a.status, "--device", a.device]
    log = Path(a.out) / "logs" / f"{phase}.{a.run_id}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as lf:
        rc = subprocess.run(cmd, cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise DriverError(f"phase {phase} exited {rc}; see logs/{log.name}")


def _finished(out: Path, name: str) -> bool:
    p = out / "records" / name
    return p.is_file() and bool(RI.read_json(p).get("finished_utc") or RI.read_json(p).get("positions"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", default="check")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--status", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--vllm-python", default="/opt/vllm/bin/python")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--phases", default="contract,build,relay,diagnostics,state_eval,evaluate")
    ap.add_argument("--kinds", default="", help="only these unit kinds this run: fluid,capability")
    ap.add_argument("--modes", default="", help="only these modes this run: H-T,H-N")
    ap.add_argument("--groups", default="", help="only these candidates this run: M-avg,M-act")
    ap.add_argument("--max-num-seqs", default=None,
                    help="the engines' concurrent-sequence limit: one integer, or per candidate, e.g. M-avg=7,M-act=7")
    ap.add_argument("--time-box-minutes", type=float, default=None,
                    help="stop each unit's runner after this; sealed work is kept and the unit stays incomplete")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--child", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.child:
        return child(a.child, a)
    out, work, status = Path(a.out), Path(a.work), RI.Status(a.status)
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    report = {"schema": SCHEMA, "run_id": a.run_id, "started_utc": RI.utc(), "phases": {}, "status": "STARTED"}
    status.mark("DRIVER_START")
    phases = [p for p in a.phases.split(",") if p]
    try:
        if a.resume:
            report["resume"] = RI.take_over(Path(a.resume), out, a.run_id)
        t0 = time.time()
        report["phases"]["contract"] = {**contract(work), "seconds": round(time.time() - t0, 1)}
        RI.write_json(out / "state" / "report.json", report)
        status.mark("CONTRACT_OK")
        if a.check_only:
            report["status"] = "CHECK_ONLY_OK"
            return 0
        have_build = _finished(out, "merge_build.json")
        have_products = (out / "records" / "products.json").is_file() and \
            RI.read_json(out / "records" / "products.json").get("verified")
        if "build" in phases:
            if have_build and have_products:
                run_child("restore", a, online=True)
                report["phases"]["build"] = {"status": "restored from the relay"}
            else:
                t0 = time.time()
                run_child("build", a)
                report["phases"]["build"] = {"status": "built", "seconds": round(time.time() - t0, 1)}
                have_products = False
        if "relay" in phases and not have_products:
            t0 = time.time()
            run_child("relay", a, online=True)
            report["phases"]["relay"] = {"status": "verified", "seconds": round(time.time() - t0, 1)}
            status.mark("PRODUCTS_OFF_POD_OK")
        RI.write_json(out / "state" / "report.json", report)
        for phase, rec in (("diagnostics", "merge_diagnostics.json"), ("state_eval", "merge_state_eval.json")):
            if phase not in phases:
                continue
            if _finished(out, rec):
                report["phases"][phase] = {"status": "taken over"}
            else:
                t0 = time.time()
                run_child(phase, a)
                report["phases"][phase] = {"status": "measured", "seconds": round(time.time() - t0, 1)}
            RI.write_json(out / "state" / "report.json", report)
        if "evaluate" in phases:
            mns = a.max_num_seqs
            if mns and "=" in mns:
                mns = {k: int(v) for k, v in (x.split("=") for x in mns.split(",") if x)}
            elif mns:
                mns = int(mns)
            opts = {"max_num_seqs": mns, "time_box_s": (a.time_box_minutes * 60 if a.time_box_minutes else None)}
            report["phases"]["evaluate"] = evaluate(
                out, work, a.vllm_python, status, a.run_id, opts,
                kinds=[k for k in a.kinds.split(",") if k] or None, modes=[m for m in a.modes.split(",") if m] or None,
                groups=[g for g in a.groups.split(",") if g] or None)
        ev_state = (report["phases"].get("evaluate") or {})
        report["incomplete_units"] = sorted(f"{name}:{kind}" for name, v in ev_state.items() if not name.startswith("_")
                                            for kind, r in v.items() if (r or {}).get("status") != "complete")
        report["status"] = "COMPLETE"
    except BaseException as exc:  # noqa: BLE001
        report["status"] = "FAILED"
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)[:3000],
                             "traceback": traceback.format_exc()[-6000:]}
    finally:
        report["ended_utc"] = RI.utc()
        RI.write_json(out / "state" / "report.json", report)
        if not a.check_only:
            RI.write_json(out / "report.json", report)
            status.mark("ALL_DONE" if report["status"] == "COMPLETE" else "RUN_FAILED")
    return 0 if report["status"] in ("COMPLETE", "CHECK_ONLY_OK") else 1


if __name__ == "__main__":
    raise SystemExit(main())
