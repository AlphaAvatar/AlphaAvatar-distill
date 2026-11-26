#!/usr/bin/env python3
"""On-pod driver of DMR1 variant 1's build (session kind ``dmr1_merge_build``): the alpha-merged initialization at
SOB1's depth-only structure, built from the original teacher, durable the moment it exists.

    cd /workspace/aad && HF_HUB_OFFLINE=1 PYTHONPATH=src:scripts:scripts/stages/stage-1:scripts/stages/stage-0/calibration_volume \\
        /opt/train/bin/python scripts/stages/stage-1/depth_merge/build_driver.py --run-id <run> --out <evidence> \\
        --work <scratch> --status <file> --device cuda [--resume <secured evidence>]

    # $0 on the dev box: the design's pins, the spans, the base digests and the session's operator config hashes
    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python scripts/stages/stage-1/depth_merge/build_driver.py \\
        --check-only --out <s>/out --work <s>/w --status <s>/st

In order, each step's product written as soon as it exists (``out/depth_merge/records/``):

1. **the contract** -- the packed search's session (inputs v3) derives the journal's operator config hashes
   (`teacher_parent_initialization.init_driver.contract`); the design's pinned records hash to their sha256;
2. **the base** -- SOB1's depth child rebuilt from the teacher (`frozen_mapping.keep_blocks` on the committed kept
   blocks) and compared tensor by tensor with the base's digests (`depth_init_tensor_digests.json`);
3. **the statistics** -- one teacher forward over the calibration at the statistics positions for the 29 span blocks
   (RI-LM1's `collect_statistics`), the counts checked against the design (79 packed sequences, 231,242 positions);
4. **the alignments** -- the 20 pairs one at a time (`merge.pair_alignments`): each written to
   ``records/alignments/<kept>-<absorbed>.json`` with the sha256 of the correlation it came from the moment it exists;
   a later attempt recomputes the one-pass statistics and reuses a pair only when its correlation hashes the same;
5. **the checks** -- folding alone and each permutation alone reproduce the block (RI-LM1's `equivalence_checks`, the
   design's tolerance); a failure stops the build before anything is saved;
6. **the candidate** -- RI-LM1's `build_candidate` with method ``M-act`` (the alpha merge): the base with the 9 merged
   blocks rewritten, every other tensor equal to the base's digests and every merged block different; saved as one
   bf16 shard whose config FILE must be the base's; reloaded with a finite forward;
7. **diagnostics** (recorded, never a gate) -- the state evaluation against the teacher and each merged span's residual
   update against the teacher span's, for the base's kept block and the merged one;
8. **durable** -- the checkpoint and the statistics to the relay (re-downloaded and re-hashed), then
   ``depth_merge/init_record.json``, COMPLETE only when every relay copy verified.

``--resume``: a previous attempt's secured evidence: an init record whose products verified is not rebuilt; persisted
alignments are reused under the rule of step 4; everything else is recomputed.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1", "scripts/stages/stage-0/calibration_volume"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.depth_merge import common as DC  # noqa: E402
from stages.depth_merge import merge as M  # noqa: E402
from stages.reasoning_inheritance.driver import Status, _gpu_peak, _ident_record, read_json, sha256_file, utc, write_json  # noqa: E402,E501

SCHEMA = "aadistill.stage1.depth_merge.init_record/v1"
ARM = "depth_merge"
INIT_FILES = ("model.safetensors", "config.json")


class DriverError(RuntimeError):
    pass


def records(out: Path) -> Path:
    return Path(out) / ARM / "records"


def contract(work: Path, device: str) -> dict:
    """$0: the design's pins hash to their records, the spans are the mapping's, the base digests are the base's,
    and the session derives the journal's operator config hashes."""
    from stages.teacher_parent_initialization import init_driver as TI
    d = DC.design()
    b = d["baseline"]
    bad = [f"{v['path']} hashes to {sha256_file(REPO / v['path'])[:12]}, the design pins {v['sha256'][:12]}"
           for v in (b["sob1_design"], b["init_record"], b["training_plan"], b["study_completion"],
                     b["control_sessions"]["completion"]) if sha256_file(REPO / v["path"]) != v["sha256"]]
    if bad:
        raise DriverError("; ".join(bad))
    spans = M.merge_spans(d)
    M.base_digests(d)
    base = TI.contract(work, device)
    return {"design_sha256": sha256_file(DC.DESIGN), "digests_sha256": sha256_file(M.DIGESTS),
            "spans": [list(s) for s in spans], "pairs": [list(p) for p in M.pairs(spans)],
            "session_contract": base, "method_identity": M.method_identity()}


def check_base(adapter, teacher, sel: dict, digests: dict) -> dict:
    """The base rebuilt from the teacher, every tensor against the base's digests (RI-LM1's `check_base` rule)."""
    from aadistill.initialization.planning import frozen_mapping as FM
    base = FM.keep_blocks(adapter, teacher, [int(b) for b in sel["kept_blocks"]], seed=int(sel["source"]["seed"]))
    sd = base.state_dict()
    tied = bool(getattr(base.config, "tie_word_embeddings", False))
    names = [n for n in sd if not (tied and n == "lm_head.weight")]
    diff = [n for n in names if FM.tensor_sha256(sd[n]) != digests.get(n)]
    if diff or set(names) != set(digests):
        raise DriverError(f"the base rebuilt from the teacher is not SOB1's depth init: {diff[:6]} / names "
                          f"{sorted(set(names) ^ set(digests))[:6]}")
    return base, {"tensors": len(names), "equal_to_base": len(names)}


def persisted_pairs(out: Path) -> dict:
    d = records(out) / "alignments"
    return {p.stem.replace("-", "->"): read_json(p) for p in sorted(d.glob("*.json"))} if d.is_dir() else {}


def build(out: Path, work: Path, device: str, status: Status, run_id: str) -> dict:
    """Steps 2-7 of the module docstring. Writes ``records/build.json`` as it goes; returns it."""
    import torch
    from safetensors.torch import save_file
    from aadistill.initialization.specs.arch import get_adapter
    from stages.teacher_modes import full_search as FS
    from stages.teacher_parent_initialization import init_driver as TI
    LMX = M.lmx()
    torch.backends.cuda.matmul.allow_tf32 = False             #: PearsonMoments refuses TF32
    torch.set_float32_matmul_precision("highest")
    d = DC.design()
    rec: dict = {"started_utc": utc(), "run_id": run_id, "arm": ARM, "design_sha256": sha256_file(DC.DESIGN),
                 "method": M.METHOD, "method_identity": M.method_identity(), "checks": {}}
    rec_p = records(out) / "build.json"
    s = TI.session(work / "session", device)
    adapter = get_adapter(s.config.target_spec.family)
    teacher, rec["root"] = FS.load_root(device)
    if rec["root"]["root_teacher_sha256"] != d["teacher"]["root_teacher_sha256"]:
        raise DriverError("the loaded teacher is not the design's root")
    n_q, n_kv, _hd = adapter.head_groups(adapter.spec_of(teacher))
    sel, spans, digests = M.selection_view(d), M.merge_spans(d), M.base_digests(d)
    base, rec["checks"]["base"] = check_base(adapter, teacher, sel, digests)
    write_json(rec_p, rec)
    status.mark("BASE_IS_SOB1_DEPTH")
    _items, active, batches = M.calibration(s, teacher, device, d)
    t0 = time.time()
    st = LMX.collect_statistics(adapter, teacher, spans, batches, active)
    exp = d["statistics"]["expected"]
    rec["statistics"] = {"sequences": st.sequences, "positions": st.positions, "seconds": round(time.time() - t0, 1),
                         "position_policy": active.report(), "forward_context": s.config.forward_context.as_dict()}
    if (st.sequences, st.positions) != (exp["packed_sequences"], exp["statistics_positions"]):
        raise DriverError(f"{st.sequences} sequences / {st.positions} positions, the design expects "
                          f"{exp['packed_sequences']} / {exp['statistics_positions']}")
    write_json(rec_p, rec)
    status.mark("STATISTICS_COLLECTED")

    def on_pair(tag: str, x: dict) -> None:
        write_json(records(out) / "alignments" / f"{tag.replace('->', '-')}.json", {**x, "run_id": run_id, "utc": utc()})
        status.mark(f"PAIR_ALIGNED:{tag}")
    t1 = time.time()
    al, rec["statistics"]["alignment_reuse"] = M.pair_alignments(st, n_q, n_kv, persisted=persisted_pairs(out),
                                                                 on_pair=on_pair)
    rec["statistics"]["alignment_seconds"] = round(time.time() - t1, 1)
    stats = LMX.statistics_tensors(st, al)
    del st
    torch.cuda.empty_cache()
    ten_p = records(out) / f"{M.STATS}.safetensors"
    tmp = ten_p.with_name(f".{ten_p.name}.partial")
    save_file(stats, str(tmp))
    os.replace(tmp, ten_p)
    rec["statistics"].update(tensors_sha256=sha256_file(ten_p),
                             pairs={f"{a}->{b}": x["summary"] for (a, b), x in al.items()},
                             permutation_sha256={k: hashlib.sha256(v.numpy().tobytes()).hexdigest()
                                                 for k, v in stats.items() if "perm" in k})
    write_json(rec_p, rec)
    status.mark("STATISTICS_SAVED")
    tol = float(d["checks"]["fold_equivalence"]["tolerance"])
    eq = LMX.equivalence_checks(adapter, teacher, spans, stats, next(iter(batches())), tol)
    rec["checks"]["equivalence"] = eq
    write_json(rec_p, rec)
    if not eq["passed"]:
        raise DriverError(f"folding / permutation does not reproduce the block: worst {eq['worst']}, non-finite "
                          f"{eq['non_finite']} (tolerance {tol})")
    model, rec["candidate"] = LMX.build_candidate(adapter, teacher, sel, spans, M.METHOD, stats, digests,
                                                  seed=int(sel["source"]["seed"]))
    ckpt = work / "products" / "init"
    ident = TI._save(adapter, model, ckpt)
    want = d["baseline"]["init_record"]
    files = {"model.safetensors": ident.shards[0].sha256, "config.json": sha256_file(ckpt / "config.json")}
    n_params = int(sum(p.numel() for p in model.parameters()))
    rec["checkpoint"] = {**_ident_record(ident), "files_sha256": files, "num_parameters_counted": n_params,
                         "config_file_is_the_base_s": files["config.json"] == want["config_json_sha256"],
                         "num_parameters_match_base": n_params == int(want["num_parameters"])}
    if not (len(ident.shards) == 1 and rec["checkpoint"]["config_file_is_the_base_s"]
            and rec["checkpoint"]["num_parameters_match_base"]):
        raise DriverError(f"the candidate is not the base's structure: {rec['checkpoint']}")
    del model
    torch.cuda.empty_cache()
    m = adapter.load(str(ckpt), device=device)
    m.config.use_cache = False
    ids = torch.arange(1, 65, device=next(m.parameters()).device).unsqueeze(0)
    rec["checkpoint"]["reload_forward_finite"] = bool(torch.isfinite(m(input_ids=ids).logits).all().item())
    if not rec["checkpoint"]["reload_forward_finite"]:
        raise DriverError("the reloaded candidate gives non-finite logits")
    write_json(rec_p, rec)
    status.mark("INIT_BUILT")
    try:                                     #: diagnostics only: a failure is recorded, never a gate
        t2 = time.time()
        s.evaluator.prime_reference(teacher)
        ev = s.evaluator.evaluate(m, ident.artifact_digest)
        rec["state_evaluation"] = {"measurement_protocol_id": s.evaluator.measurement_protocol_id,
                                   "values": dict(getattr(ev, "values", {}) or {}),
                                   "seconds": round(time.time() - t2, 1), "role": "diagnostic only"}
    except Exception as exc:  # noqa: BLE001
        rec["state_evaluation"] = {"error": f"{type(exc).__name__}: {str(exc)[:1500]}", "role": "diagnostic only"}
    try:
        blocks_m, blocks_b = adapter.blocks(m), adapter.blocks(base)
        idx = {s0: [int(x) for x in sel["kept_blocks"]].index(s0) for s0 in (sp[0] for sp in spans)}
        cands = {"kept": {s0: blocks_b[i] for s0, i in idx.items()}, "merged": {s0: blocks_m[i] for s0, i in idx.items()}}
        rec["diagnostics"] = LMX.diagnostics(adapter, teacher, spans, cands, batches, active)
    except Exception as exc:  # noqa: BLE001
        rec["diagnostics"] = {"error": f"{type(exc).__name__}: {str(exc)[:1500]}", "role": "diagnostic only"}
    rec["gpu_peak"] = _gpu_peak(device)
    rec["finished_utc"] = utc()
    write_json(rec_p, rec)
    return rec


def init_record(rec: dict, relay: dict) -> dict:
    """The provenance record a training plan binds (`initialization_provenance.read`: ``root_teacher``,
    ``commit.selected_rows``) and where every durable byte is."""
    ck = rec["checkpoint"]
    return {"schema": SCHEMA, "status": "COMPLETE", "arm": ARM, "run_id": rec["run_id"], "root_teacher": rec["root"],
            "commit": {"selected_rows": [{"state_id": f"{ARM}/{ck['artifact_digest'][:32]}",
                                          "path": "SOB1's depth selection, alpha-merged (M-act) at its spans",
                                          "artifact_digest": ck["artifact_digest"],
                                          "single_shard_sha256": ck["shards"][0]["sha256"]}]},
            "checkpoint": ck, "base": DC.design()["baseline"]["init_record"], "method": rec["method"],
            "method_identity": rec["method_identity"], "statistics": rec["statistics"], "checks": rec["checks"],
            "candidate": rec["candidate"], "state_evaluation": rec.get("state_evaluation"),
            "diagnostics": rec.get("diagnostics"), "relay": relay, "design_sha256": rec["design_sha256"],
            "finished_utc": utc()}


def products(rec: dict) -> dict[str, dict[str, str]]:
    """What goes to the relay, by group: the checkpoint's two files and the statistics tensors."""
    return {"init": {f"init/{n}": s for n, s in rec["checkpoint"]["files_sha256"].items()},
            "statistics": {f"statistics/{M.STATS}.safetensors": rec["statistics"]["tensors_sha256"]}}


def product_problems(doc: dict | None) -> list[str]:
    if doc is None:
        return ["no init record"]
    if doc.get("status") != "COMPLETE" or doc.get("arm") != ARM:
        return [f"the init record is {doc.get('status')!r} for {doc.get('arm')!r}"]
    bad = []
    for g, files in products(doc).items():
        r = (doc.get("relay") or {}).get(g) or {}
        if not files or r.get("verified") is not True or r.get("files_sha256") != files:
            bad.append(f"{g}: the relay copy is not exactly {sorted(files)} verified")
    shard = doc["checkpoint"]["shards"][0]["sha256"]
    if doc["checkpoint"]["files_sha256"].get("model.safetensors") != shard:
        bad.append("the manifest's weights are not the checkpoint's shard")
    return bad


def durable_init(out: Path) -> dict | None:
    p = Path(out) / ARM / "init_record.json"
    doc = read_json(p) if p.is_file() else None
    return doc if doc is not None and not product_problems(doc) else None


def persist(out: Path, work: Path, run_id: str, status: Status, put=None) -> dict:
    """The products to the relay (online child processes; ``put`` for a test), each group verified only when the
    relay's re-downloaded hashes are its manifest; then the init record, COMPLETE only when `product_problems` is
    empty."""
    from stages.teacher_parent_initialization import relay as R
    put = put or (lambda payload, timeout_s: R.run_online("put", payload, timeout_s=timeout_s))
    rec = read_json(records(out) / "build.json")
    local = {"init": work / "products", "statistics": records(out).parent}
    prev = read_json(records(out) / "relay.json") if (records(out) / "relay.json").is_file() else {}
    relay = {}
    for name, files in products(rec).items():
        if (prev.get(name) or {}).get("verified") is True and prev[name].get("files_sha256") == files:
            relay[name] = prev[name]                 #: verified by an earlier attempt: never sent again
            continue
        paths = {k: (local["init"] / k if name == "init" else records(out) / Path(k).name) for k in files}
        nbytes = sum(p.stat().st_size for p in paths.values())
        prefix = f"{DC.design()['storage']['relay_prefix_root']}/{ARM}/{name}/{rec['checkpoint']['artifact_digest'][:16]}"
        res = put({"files": {k: str(v) for k, v in paths.items()}, "prefix": prefix,
                   "message": f"DMR1-M {run_id}: {name}", "scratch": str(work)},
                  max(3600.0, nbytes / 2.6e6 * 1.25 + 600))
        relay[name] = res
        write_json(records(out) / "relay.json", {**prev, **relay})
        if not (res.get("verified") is True and res.get("files_sha256") == files):
            raise DriverError(f"the relay copy of {name} is not its manifest verified: {str(res)[:600]}")
        status.mark(f"RELAY_OK:{name}")
    doc = init_record(rec, relay)
    bad = product_problems(doc)
    if bad:
        raise DriverError(f"the init record would not show its products durable: {bad}")
    write_json(Path(out) / ARM / "init_record.json", doc)
    status.mark("INIT_RECORD_WRITTEN")
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", default="check")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--status", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args(argv)
    out, work, status = Path(a.out), Path(a.work), Status(a.status)
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    report = {"schema": SCHEMA.replace("init_record", "build_report"), "run_id": a.run_id, "started_utc": utc()}
    ok = True
    try:
        if a.resume:
            from stages.teacher_parent_initialization import train_driver as TD
            report["resumed_from"] = TD.take_over(Path(a.resume), out)
        report["contract"] = contract(work, "cpu" if a.check_only else a.device)
        write_json(out / "report.json", report)
        status.mark("CONTRACT_OK")
        if a.check_only:
            return 0
        if durable_init(out):
            report["init"] = {"resumed": "an earlier attempt's init record verified: nothing rebuilt"}
        else:
            build_rec = read_json(records(out) / "build.json") if (records(out) / "build.json").is_file() else {}
            on_pod = (work / "products" / "init" / "model.safetensors").is_file() and build_rec.get("finished_utc")
            if not (on_pod and sha256_file(work / "products" / "init" / "model.safetensors")
                    == build_rec["checkpoint"]["files_sha256"]["model.safetensors"]):
                build(out, work, a.device, status, a.run_id)
            report["init"] = persist(out, work, a.run_id, status)["commit"]
        write_json(out / "report.json", report)
        status.mark("INIT_DURABLE")
    except BaseException as exc:  # noqa: BLE001
        ok = False
        report["error"] = {"error": repr(exc)[:2000], "traceback": traceback.format_exc()[-6000:]}
    report.update(finished_utc=utc(), terminal="ALL_DONE" if ok else "RUN_FAILED")
    write_json(out / "report.json", report)
    status.mark("ALL_DONE" if ok else "RUN_FAILED")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
