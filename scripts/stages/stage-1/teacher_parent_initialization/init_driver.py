#!/usr/bin/env python3
"""On-pod driver of A_teacher_parent's initialization (session 1 of the round): the teacher-parent selections built
into one 596M checkpoint, its residual PCA from the teacher's own statistics at the kept layers, every expensive
product made durable the moment it exists.

    cd /workspace/aad && HF_HUB_OFFLINE=1 PYTHONPATH=src:scripts:scripts/stages/stage-1 /opt/train/bin/python \\
        scripts/stages/stage-1/teacher_parent_initialization/init_driver.py --run-id <run> --out <evidence dir> \\
        --work <scratch> --status <file> --device cuda [--resume <secured evidence dir or archive>]

    # $0 on the dev box: the design, the selection reproduced from the journal, and the session's operator config
    # hashes against the journal's (nothing is loaded)
    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python \\
        scripts/stages/stage-1/teacher_parent_initialization/init_driver.py --check-only --out <s>/out --work <s>/w \\
        --status <s>/st

What runs (`design.json :: arms.A_teacher_parent`), in one child process so the GPU is released after it:

1. the teacher, every shard hashed (`full_search.load_root`);
2. three identity gates, each a pure selection on the teacher compared bit for bit with its journal state's artifact
   digest: the FFN child (``kept_neurons``, all 36 blocks), the DEPTH child (``kept_layers`` K) and the ATTENTION child
   (``kept_heads``, all 36 blocks) -- equal digests prove each selection was made on, and is applied to, the
   unmodified teacher;
3. the teacher statistics pass: ``width.global_pca_v0`` replayed on the teacher under the packed search's own session
   (its config hash pinned to the journal's root RESIDUAL_WIDTH(rh) state) with the statistics retained
   (`RetainingStatsCache`); the replayed child must equal that state's digest bit for bit, and P over all 37 points
   recomputed from the retained statistics must reproduce its recorded diagnostics -- the statistics are then the
   search's own. They are saved (safetensors, every point) and kept;
4. A's width transform from those statistics at ``selection.json :: width.points`` (sorted({0, *K}) + [36]) with its
   weights, the teacher's post-norm point and the teacher's final norm (`width_transform_from_statistics`);
5. one `FrozenMapping` keyed by ROOT indices (FFN neurons, K, the transform, query heads), saved with every tensor's
   sha256, loaded back, materialized onto the teacher with ``blocks="kept"`` in the fixed order FFN -> DEPTH ->
   RESIDUAL_WIDTH -> ATTENTION; geometry and parameter count against the target; reloaded, a forward checked finite;
6. state evaluation (diagnostic only): ``state_eval_hybrid@v3`` against the teacher, the search's evaluator;
7. the durable copies: the checkpoint, the mapping and the statistics to the relay (`relay.put`, re-downloaded and
   re-hashed), then ``init_record.json`` -- the provenance record the training plan binds (the shape
   `shared.training.initialization_provenance.read` reads: ``root_teacher`` and ``commit.selected_rows``).

A gate that fails stops the driver before anything is built on it (a mismatch is first a question about this harness,
AGENTS.md P12.1). ``--resume``: a previous attempt's secured evidence; an init record whose relay copies verified
means the initialization exists and nothing is recomputed.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.reasoning_inheritance.driver import (  # noqa: E402
    Status, _identify, _ident_record, _gpu_peak, read_json, sha256_file, utc, write_json)

DESIGN = REPO / "configs/stages/stage-1/teacher_parent_initialization/design.json"
SELECTION = REPO / "configs/stages/stage-1/teacher_parent_initialization/selection.json"
SCHEMA = "aadistill.stage1.teacher_parent_initialization.init_record/v1"
ARM = "A_teacher_parent"
STATS_FILE = "teacher_residual_statistics.safetensors"
FILES = ("model.safetensors", "config.json")


class DriverError(RuntimeError):
    pass


def design() -> dict:
    return read_json(DESIGN)


def selection() -> dict:
    return read_json(SELECTION)


def session(work: Path, device: str):
    """The packed search's own session (inputs v3), refused unless its inputs are their frozen identity."""
    from stages.teacher_modes import packed_full_search as PFS
    which = design()["session"]["inputs_key"]
    s = PFS.build_session(workdir=work / "search_session", device=device, which=which, repo_root=REPO)
    bad = PFS.input_problems(s, which)
    if bad:
        raise DriverError("the inputs are not their frozen identity: " + "; ".join(bad))
    return s


def replay_spec(s, sel: dict, device: str, keys: tuple[str, ...] = ("RESIDUAL_WIDTH",)):
    """The journal's root-level steps ``keys`` as a fixed path on the teacher, each config hash pinned."""
    from aadistill.initialization.planning.fixed_path import FixedPathSpec, FixedPathStep
    t = design()["teacher"]
    steps = tuple(FixedPathStep(impl_id=sel["steps"][k]["impl_id"], profile_id=sel["steps"][k]["profile_id"],
                                label=sel["steps"][k]["kind"], expected_config_hash=sel["steps"][k]["config_hash"])
                  for k in keys)
    return FixedPathSpec(path_id=f"stage1.teacher_parent_initialization.{'_'.join(keys).lower()}",
                         family=s.config.target_spec.family, target_spec=s.config.target_spec, steps=steps,
                         root_repo_id=t["model"], root_revision=t["revision"], device=device,
                         seed=int(sel["source"]["seed"]), position_policy=s.config.position_policy,
                         distribution_support=s.config.distribution_support,
                         forward_context=s.config.forward_context)


def contract(work: Path, device: str) -> dict:
    """$0: the selection reproduced from the journal, every journal step's operator config hash derived again."""
    from stages.reasoning_inheritance.driver import derived_config_hashes
    from stages.teacher_parent_initialization import selection as SEL
    sel = selection()
    s = session(work, device)
    keys = ("FFN", "DEPTH", "ATTENTION", "RESIDUAL_WIDTH")
    hashes = derived_config_hashes(replay_spec(s, sel, device, keys))
    bad = [h for h in hashes if not h["matches"]]
    if bad:
        raise DriverError(f"the session derives other operator configs than the journal recorded: {bad}")
    want = (SEL.render(SEL.extract()) if (REPO / design()["source_search"]["journal"]["path"]).is_file() else None)
    return {"design_sha256": sha256_file(DESIGN), "selection_sha256": sha256_file(SELECTION),
            "selection_reproduced_from_journal": None if want is None else want == SELECTION.read_text(),
            "session": {"config_hash": s.config.config_hash,
                        "measurement_protocol_id": s.evaluator.measurement_protocol_id,
                        "forward_context": s.config.forward_context.as_dict(),
                        "profiles": {p.qualified_id: p.profile_hash for p in s.config.profiles}},
            "config_hashes": hashes}


def _save(adapter, model, ckpt: Path):
    shutil.rmtree(ckpt, ignore_errors=True)
    ckpt.mkdir(parents=True)
    adapter.save(model, str(ckpt))
    return _identify(adapter, ckpt)


def gate(rec: dict, name: str, ident, state: dict) -> None:
    rec["gates"][name] = {"artifact_digest": ident.artifact_digest, "journal_state": state["state_id"],
                          "journal_artifact_digest": state["artifact_digest"],
                          "equal": ident.artifact_digest == state["artifact_digest"]}
    if not rec["gates"][name]["equal"]:
        raise DriverError(f"{name}: the rebuilt child {ident.artifact_digest[:12]} is not the journal state's "
                          f"{state['artifact_digest'][:12]}")


def save_statistics(state: dict, path: Path) -> dict:
    """The retained statistics state, every tensor, as safetensors (float64 kept), with its sha256."""
    import torch
    from safetensors.torch import save_file
    tensors = {k: v.detach().to("cpu").contiguous() for k, v in state.items() if torch.is_tensor(v)}
    tmp = path.with_name(f".{path.name}.partial")
    save_file(tensors, str(tmp))
    os.replace(tmp, path)
    return {"file": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size,
            "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in sorted(tensors.items())}}


def build(out: Path, work: Path, device: str, status: Status, run_id: str) -> dict:
    """Steps 1-6 of the module docstring. Writes ``out/records/build.json`` as it goes."""
    import torch
    from aadistill.initialization.planning import frozen_mapping as FM
    from aadistill.initialization.planning.fixed_path import _run_steps
    from aadistill.initialization.specs.arch import get_adapter
    from stages.teacher_modes import full_search as FS
    d, sel = design(), selection()
    arm = d["arms"][ARM]
    s = session(work, device)
    adapter = get_adapter(s.config.target_spec.family)
    rec: dict = {"started_utc": utc(), "run_id": run_id, "selection_sha256": sha256_file(SELECTION),
                 "design_sha256": sha256_file(DESIGN), "gates": {}}
    progress = out / "records" / "build.json"
    t0 = time.time()
    teacher, rec["root"] = FS.load_root(device)
    rec["seconds_teacher"] = round(time.time() - t0, 1)
    if rec["root"]["root_teacher_sha256"] != d["teacher"]["root_teacher_sha256"]:
        raise DriverError("the loaded teacher is not the design's root")
    n_root, kept = int(sel["n_root_blocks"]), [int(b) for b in sel["kept_blocks"]]
    seed = int(sel["source"]["seed"])
    ffn_sel = [sel["ffn_neurons"][str(b)] for b in range(n_root)]
    heads = [sel["query_heads"][str(b)] for b in range(n_root)]
    gw = work / "gates"

    # 2. the three selection gates on the teacher
    for name, fn, arg in (("FFN", FM.select_ffn_neurons, ffn_sel), ("DEPTH", FM.keep_blocks, kept),
                          ("ATTENTION", FM.select_query_heads, heads)):
        child = fn(adapter, teacher, arg, seed=seed)
        ident = _save(adapter, child, gw / name)
        del child
        gate(rec, name, ident, sel["states"][name])
        shutil.rmtree(gw / name, ignore_errors=True)
        torch.cuda.empty_cache()
        write_json(progress, rec)
    status.mark("SELECTION_GATES_OK")

    # 3. the teacher statistics pass, gated on the journal's root width state
    cache = FM.RetainingStatsCache()
    t1 = time.time()
    (w_res,) = _run_steps(replay_spec(s, sel, device), adapter=adapter, model=teacher, indices=[0],
                          parent_spec=adapter.spec_of(teacher), parent_digest=rec["root"]["root_teacher_sha256"],
                          workdir=work / "width_replay", repo_root=str(REPO), stats_cache=cache,
                          calibration_items=None, on_step=None, deadline=None, execution=s.execution)
    rec["statistics_seconds"] = round(time.time() - t1, 1)
    gate(rec, "RESIDUAL_WIDTH", w_res.identity, sel["states"]["RESIDUAL_WIDTH"])
    rec["width_replay"] = {"trace": dict(w_res.trace), "local_metrics": dict(w_res.local_metrics)}
    shutil.rmtree(work / "width_replay", ignore_errors=True)
    if len(cache.retained) != 1:
        raise DriverError(f"the statistics pass left {len(cache.retained)} states, not one")
    state = next(iter(cache.retained.values()))
    w = sel["width"]
    gate_diag = w["reproduction_gate"]["all_points_diagnostics"]
    all37 = FM.width_transform_from_statistics(
        state, d_child=int(w["d_child"]), points=list(range(n_root + 1)), point_weights=gate_diag["point_weights"],
        final_norm_weight=adapter.final_norm(teacher).weight, post_norm_point=n_root, device=device)
    keys = ("energy_captured_frac", "top_eigenvalue", "min_kept_eigenvalue")
    rec["statistics_reproduction"] = {"recomputed": {k: all37.diagnostics.get(k) for k in keys},
                                      "recorded": {k: gate_diag[k] for k in keys},
                                      "equal": all(all37.diagnostics.get(k) == gate_diag[k] for k in keys)}
    if not rec["statistics_reproduction"]["equal"]:
        raise DriverError(f"the retained statistics do not reproduce the journal's 37-point diagnostics: "
                          f"{rec['statistics_reproduction']}")
    del all37
    prod = work / "products"
    prod.mkdir(parents=True, exist_ok=True)
    rec["statistics"] = {**save_statistics(state, prod / STATS_FILE),
                         "parent": "the root teacher", "profile_id": sel["steps"]["RESIDUAL_WIDTH"]["profile_id"],
                         "operator": sel["steps"]["RESIDUAL_WIDTH"]["impl_id"],
                         "config_hash": sel["steps"]["RESIDUAL_WIDTH"]["config_hash"],
                         "stats_tokens": int(state["residual_count"][0]) if "residual_count" in state else None,
                         "points": f"0..{n_root}: inputs of root layers 0..{n_root - 1}, then the post-final-norm output"}
    write_json(progress, rec)
    status.mark("STATISTICS_OK")

    # 4. A's width transform at the kept layers
    transform = FM.width_transform_from_statistics(
        state, d_child=int(w["d_child"]), points=[int(p) for p in w["points"]],
        point_weights=[float(x) for x in w["point_weights"]], final_norm_weight=adapter.final_norm(teacher).weight,
        post_norm_point=int(w["post_norm_point"]), device=device)
    rec["width_transform"] = {"points": [int(p) for p in w["points"]], "point_weights": w["point_weights"],
                              "post_norm_point": int(w["post_norm_point"]), "scale": transform.scale,
                              "diagnostics": dict(transform.diagnostics),
                              "rule": arm["width_points"]["rule"]}
    del state
    cache.clear()
    cache.retained.clear()

    # 5. the mapping, saved and loaded back; A materialized from the saved copy
    mapping = FM.FrozenMapping(
        root={"model": d["teacher"]["model"], "revision": d["teacher"]["revision"],
              "root_teacher_sha256": rec["root"]["root_teacher_sha256"]},
        n_root_blocks=n_root, kept_blocks=tuple(kept),
        ffn_neurons={b: tuple(ffn_sel[b]) for b in range(n_root)},
        query_heads={b: tuple(heads[b]) for b in range(n_root)}, width=transform,
        provenance={k: f"journal {d['source_search']['journal']['sha256'][:12]} state {v['state_id']}"
                    for k, v in arm["selection"].items() if k != "RESIDUAL_WIDTH"}
        | {"width": f"width.global_pca_v0 statistics on the teacher, gate {sel['states']['RESIDUAL_WIDTH']['state_id']}",
           "selection_sha256": rec["selection_sha256"]})
    bad = FM.problems(mapping, root_spec=adapter.spec_of(teacher))
    if bad:
        raise DriverError(f"the mapping is malformed: {bad}")
    saved = FM.save(mapping, prod / "mapping")
    mapping = FM.load(prod / "mapping", expected_content_digest=saved["content_digest"])
    rec["mapping"] = {"content_digest": saved["content_digest"],
                      "files_sha256": {n: sha256_file(prod / "mapping" / n) for n in (FM.MAPPING_FILE, FM.TENSOR_FILE)}}
    model, roots = FM.materialize(adapter, teacher, mapping, blocks="kept", non_depth=True, seed=seed)
    ckpt = prod / "init"
    ident = _save(adapter, model, ckpt)
    geo = {k: (bool(adapter.spec_of(model)[k]) if k == "tie_word_embeddings" else int(adapter.spec_of(model)[k]))
           for k in d["target"]["geometry"]}
    n_params = int(sum(p.numel() for p in model.parameters()))
    rec["checkpoint"] = {**_ident_record(ident), "root_blocks": roots, "geometry": geo,
                         "geometry_matches_target": geo == d["target"]["geometry"], "num_parameters_counted": n_params,
                         "num_parameters_match_target": n_params == int(d["target"]["num_parameters"]),
                         "order": arm["build"]["order"]}
    if not (rec["checkpoint"]["geometry_matches_target"] and rec["checkpoint"]["num_parameters_match_target"]
            and roots == kept and len(ident.shards) == 1):
        raise DriverError(f"A is {geo}, {n_params} parameters, blocks {roots}: not the target")
    del model
    torch.cuda.empty_cache()
    a = adapter.load(str(ckpt), device=device)
    a.config.use_cache = False
    ids = torch.arange(1, 65, device=next(a.parameters()).device).unsqueeze(0)
    logits = a(input_ids=ids).logits
    rec["checkpoint"]["reload_forward_finite"] = bool(torch.isfinite(logits).all().item())
    if not rec["checkpoint"]["reload_forward_finite"]:
        raise DriverError("the reloaded A gives non-finite logits")
    write_json(progress, rec)
    status.mark("INIT_BUILT")

    # 6. state evaluation, diagnostic only (a failure is recorded, never a gate)
    try:
        t2 = time.time()
        s.evaluator.prime_reference(teacher)
        ev = s.evaluator.evaluate(a, ident.artifact_digest)
        rec["state_evaluation"] = {"measurement_protocol_id": s.evaluator.measurement_protocol_id,
                                   "suite": s.suite_manifest.get("suite_id"),
                                   "values": dict(getattr(ev, "values", {}) or {}),
                                   "seconds": round(time.time() - t2, 1), "role": "diagnostic only"}
        status.mark("STATE_EVAL_OK")
    except Exception as exc:  # noqa: BLE001
        rec["state_evaluation"] = {"error": f"{type(exc).__name__}: {str(exc)[:1500]}", "role": "diagnostic only"}
        status.mark("STATE_EVAL_FAILED")
    rec["gpu_peak"] = _gpu_peak(device)
    rec["finished_utc"] = utc()
    write_json(progress, rec)
    return rec


def init_record(rec: dict, relay: dict) -> dict:
    """The provenance record the training plan binds: ``root_teacher`` and ``commit.selected_rows`` as
    `initialization_provenance.read` reads them, the gates, and where every durable byte is."""
    ck = rec["checkpoint"]
    return {"schema": SCHEMA, "status": "COMPLETE", "arm": ARM, "run_id": rec["run_id"],
            "root_teacher": rec["root"],
            "commit": {"selected_rows": [{"state_id": f"{ARM}/{rec['mapping']['content_digest'][:32]}",
                                          "path": "FFN(base) -> DEPTH(base) -> RESIDUAL_WIDTH(rh, teacher statistics "
                                                  "at K) -> ATTENTION(rh), every selection on the teacher",
                                          "artifact_digest": ck["artifact_digest"],
                                          "single_shard_sha256": ck["shards"][0]["sha256"]}]},
            "checkpoint": ck, "gates": rec["gates"], "statistics_reproduction": rec["statistics_reproduction"],
            "statistics": rec["statistics"], "width_transform": rec["width_transform"], "mapping": rec["mapping"],
            "state_evaluation": rec.get("state_evaluation"), "relay": relay,
            "design_sha256": rec["design_sha256"], "selection_sha256": rec["selection_sha256"],
            "finished_utc": utc()}


def persist(out: Path, work: Path, run_id: str, status: Status) -> dict:
    """Step 7: the products to the relay (online child processes), then the init record."""
    from stages.teacher_parent_initialization import relay as R
    rec = read_json(out / "records" / "build.json")
    prod = work / "products"
    digest = rec["mapping"]["content_digest"][:16]
    groups = {"init": {f"init/{n}": prod / "init" / n for n in FILES if (prod / "init" / n).is_file()},
              "mapping": {f"mapping/{n}": prod / "mapping" / n for n in rec["mapping"]["files_sha256"]},
              "statistics": {STATS_FILE: prod / STATS_FILE}}
    relay = {}
    for name, files in groups.items():
        res = R.run_online("put", {"files": {k: str(v) for k, v in files.items()},
                                   "prefix": f"{R.ROOT_PREFIX}/{ARM}/init/{digest}",
                                   "message": f"{ARM} {run_id}: {name}", "scratch": str(work)})
        relay[name] = res
        if not res.get("verified"):
            write_json(out / "records" / "relay.json", relay)
            raise DriverError(f"the relay copy of {name} did not verify: {str(res)[:600]}")
        status.mark(f"RELAY_OK:{name}")
    write_json(out / "records" / "relay.json", relay)
    doc = init_record(rec, relay)
    write_json(out / "init_record.json", doc)
    status.mark("INIT_RECORD_WRITTEN")
    return doc


def take_over(resume: Path, out: Path) -> dict | None:
    """A previous attempt's init record, if its relay copies verified: the initialization exists."""
    import tarfile
    if resume.is_file():
        unpacked = out.parent / ".resume"
        shutil.rmtree(unpacked, ignore_errors=True)
        with tarfile.open(resume) as t:
            t.extractall(unpacked, filter="data")
        resume = unpacked
    hits = sorted(resume.rglob("init_record.json"))
    for p in hits:
        doc = read_json(p)
        if doc.get("status") == "COMPLETE" and all(v.get("verified") for v in (doc.get("relay") or {}).values()):
            write_json(out / "init_record.json", doc)
            return doc
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", default="check")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--status", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--phase", choices=("build",), default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    out, work, status = Path(a.out), Path(a.work), Status(a.status)
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    if a.phase == "build":                      #: the GPU child
        build(out, work, a.device, status, a.run_id)
        return 0
    report = {"schema": SCHEMA.replace("init_record", "driver_report"), "run_id": a.run_id, "started_utc": utc()}
    ok = True
    try:
        report["contract"] = contract(work, "cpu" if a.check_only else a.device)
        write_json(out / "report.json", report)
        status.mark("CONTRACT_OK")
        if a.check_only:
            return 0
        if a.resume and take_over(Path(a.resume), out):
            report["resumed"] = "the previous attempt's init record verified: nothing recomputed"
        else:
            import subprocess
            env = {**os.environ, "HF_HUB_OFFLINE": "1"}
            rc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--phase", "build", "--run-id",
                                 a.run_id, "--out", str(out), "--work", str(work), "--status", str(status.path),
                                 "--device", a.device], env=env, cwd=REPO).returncode
            if rc != 0:
                raise DriverError(f"the build phase exited {rc}; see records/build.json and the session log")
            report["init_record"] = persist(out, work, a.run_id, status)["commit"]
    except BaseException as exc:  # noqa: BLE001
        ok = False
        report["error"] = {"error": repr(exc)[:2000], "traceback": traceback.format_exc()[-6000:]}
    report.update(finished_utc=utc(), terminal="ALL_DONE" if ok else "RUN_FAILED")
    write_json(out / "report.json", report)
    status.mark("ALL_DONE" if ok else "RUN_FAILED")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
