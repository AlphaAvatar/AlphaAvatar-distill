#!/usr/bin/env python3
"""On-pod driver of the layer-merge round (RI-LM1): the statistics, each candidate built and made durable in turn,
their diagnostics, state evaluation and eval-set-1 in both modes. Every finished product is written, and every
candidate made durable, the moment it exists.

    cd /workspace/aad && HF_HUB_OFFLINE=1 PYTHONPATH=src:scripts:scripts/stages/stage-1 /opt/train/bin/python \\
        scripts/stages/stage-1/reasoning_inheritance/layer_merge_driver.py --run-id <run> --out <evidence dir> \\
        --work <scratch> --status <file> --device cuda --vllm-python /opt/vllm/bin/python [--resume <archive>]

    # $0 on the dev box: the design, the spans, D's digests, the protocol pin, the anchor session's config hashes and
    # every product's identity
    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python \\
        scripts/stages/stage-1/reasoning_inheritance/layer_merge_driver.py --check-only --out <s>/out --work <s>/work \\
        --status <s>/st

Phases (``configs/stages/stage-1/reasoning_inheritance/layer_merge/design.json`` is the experiment), each GPU step its
own child process so the memory is released between them:

* ``contract`` ($0, host): the design and the frozen selection by sha256; the spans re-derived; D's digests file is
  D's; RI-S1's evaluation protocol unchanged; the anchor session's operator config hashes against the journal; and
  every product's IDENTITY (`identities`), written to ``state/identities.json`` for the children.
* per candidate, in the design's order, one after the other -- ``build`` then ``relay``: the teacher (every shard
  verified) and D rebuilt from it, compared tensor by tensor with D's digests; for M-act the statistics forward and the
  alignments (``records/merge_statistics.*``) and the fold and permutation checks (``records/merge_checks.json``),
  each reused when an earlier attempt's has this attempt's identity; the candidate built, checked and saved; its span
  blocks written as ``products/<g>/delta.safetensors`` (D plus these tensors IS the candidate); D rebuilt again, the
  delta FILE read back into it, saved and required to give the candidate's artifact digest (the reconstruction
  identity); then the delta (and the statistics) to the private relay, downloaded again into a fresh cache and
  re-hashed: ``records/candidates/<g>.json`` says ``durable``. Only then the next candidate. The delta is in the
  evidence directory, so every poll also pulls it to the dev box, where D is kept (`layer_merge_launch.py retain`).
* ``restore`` (as needed): a durable candidate's checkpoint rebuilt from D and its delta (this attempt's file, or the
  relay's), its artifact digest required to be the recorded one.
* ``diagnostics``: each span's residual updates, D's / M-avg's / M-act's block against the teacher span's.
* ``state_eval``: KL and NLL against the root teacher on ``state_eval_hybrid@v3`` (diagnostic only), per candidate.
* ``evaluate``: eval-set-1, every unit, in the design's order, through RI-S1's runners (`driver.fluid`,
  `driver.capability`, `driver.package`) under RI-S1's protocol.

``--resume <archive>`` takes over a previous attempt's records and sealed parts (`driver.take_over`). Nothing taken
over is used for being there: each product is reused only when its recorded identity is this attempt's (teacher, data,
packing and positions, statistics and merge method and the code that computes them, D, the checkpoint, the evaluation
protocol); one that is not is moved to ``superseded/<attempt>/`` with the reason and redone. An engineering change that
touches none of a product's identity (a launcher, the relay code, a log line, a comment) keeps it.
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

SCHEMA = "aadistill.stage1.reasoning_inheritance.layer_merge.driver_report/v2"
GROUPS = LMX.CANDIDATES
STATS = "merge_statistics"
#: the candidates whose merge reads the statistics (M-avg reads none, so its identity does not depend on them)
NEEDS_STATISTICS = ("M-act",)


class DriverError(RuntimeError):
    pass


def sha(p: Path) -> str:
    return RI.sha256_file(p)


def jsha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def protocol_sha256(ri_design: dict) -> str:
    return hashlib.sha256(json.dumps(ri_design["evaluation"], sort_keys=True).encode()).hexdigest()


def identity(kind: str, **parts) -> dict:
    """A product's identity: what it was computed from, and by what. Two attempts share a product only when equal."""
    return {"kind": kind, "parts": parts, "sha256": jsha({"kind": kind, **parts})}


def with_artifacts(static: dict, **digests) -> dict:
    """A measurement's identity: its static part and the artifact digests it was measured on."""
    return identity(static["kind"], static=static["sha256"], artifacts=digests)


# ---- identities ($0, host) ------------------------------------------------------------------------------------------

def identities(work: Path, device: str = "cpu") -> dict:
    """Every product's identity, from the design, the session, the evaluation protocol and the code that computes it
    (`layer_merge.code_identity`) -- never the whole commit, so an engineering fix elsewhere keeps a valid product."""
    from shared import eval_sets as ES
    d, sel = LMX.design(), RI.selection()
    s = RI.session(work, device)
    spans = [list(x) for x in LMX.merge_spans(d, sel)]
    teacher = {"model": d["teacher"]["model"], "revision": d["teacher"]["revision"],
               "identity_sha256": sha(REPO / d["teacher"]["identity"])}
    prof = [p for p in s.config.profiles if p.qualified_id == d["statistics"]["profile_id"]]
    if len(prof) != 1:
        raise DriverError(f"the session holds no profile {d['statistics']['profile_id']}")
    pol = s.config.position_policy
    data = {"profile": prof[0].qualified_id, "profile_hash": prof[0].profile_hash,
            "positions": {"policy": pol.qualified_id, "policy_hash": pol.policy_hash,
                          "of_operator": d["statistics"]["positions_of_operator"]},
            "packing": {"batch_packing": s.execution.calibration_batch_packing,
                        "forward_context": s.config.forward_context.as_dict()},
            "expected": d["statistics"]["expected"]}
    base = {"d_weights_sha256": d["base"]["d_checkpoint"]["weights_sha256"],
            "d_artifact_digest": d["base"]["d_checkpoint"]["artifact_digest"], "d_digests_sha256": sha(LMX.DIGESTS),
            "selection_sha256": sha(RI.SELECTION), "seed": int(sel["source"]["seed"])}
    st = identity("statistics", teacher=teacher, data=data, spans=spans, method=jsha(d["statistics"]),
                  code=LMX.code_identity(*LMX.statistics_code())["sha256"])
    checks = identity("checks", statistics=st["sha256"], checks=jsha(d["checks"]),
                      code=LMX.code_identity(LMX.equivalence_checks, LMX.capture_block_inputs, LMX.LM.fold_norm_gains,
                                             LMX.LM.permute_block, LMX.LM.block_like, LMX.LM.block_tensors)["sha256"])
    cand = {}
    for g in GROUPS:
        parts = {"teacher": teacher, "base": base, "spans": spans, "method": jsha(d["candidates"][g]),
                 "code": LMX.code_identity(*LMX.candidate_code())["sha256"]}
        if g in NEEDS_STATISTICS:
            parts["statistics"] = st["sha256"]
        cand[g] = identity("candidate", **parts)
    ri = RI.read_json(REPO / d["base"]["ri_s1_design"]["path"])
    ev = ri["evaluation"]
    return {
        "family": s.config.target_spec.family, "statistics": st, "checks": checks, "candidates": cand,
        #: measurements: the static parts; each record's identity adds the artifact digests it was measured on
        "diagnostics": identity("diagnostics", teacher=teacher, data=data, spans=spans, method=jsha(d["diagnostics"]),
                                code=LMX.code_identity(LMX.diagnostics, LMX.LM.UpdateMoments)["sha256"]),
        "state_eval": identity("state_eval", teacher=teacher, measurement_protocol_id=s.evaluator.measurement_protocol_id,
                               suite=s.suite_manifest.get("suite_id"),
                               suite_content_sha256=s.suite_manifest.get("content_sha256"),
                               forward_context=s.config.forward_context.as_dict()),
        "evaluation": identity("evaluation", protocol_sha256=protocol_sha256(ri),
                               decoding_sha256=sha(REPO / ev["decoding"]), eval_set=ev["eval_set"],
                               eval_set_manifest_sha256=jsha(ES.manifest(ev["eval_set"])),
                               teacher_identity_sha256=sha(REPO / ri["teacher"]["identity"])),
        "rule": "a product is reused only when its recorded identity sha256 equals this attempt's (driver docstring)"}


def ids_of(out: Path) -> dict:
    return RI.read_json(out / "state" / "identities.json")


def supersede(out: Path, rels: list[str], attempt: str, why: str) -> list[str]:
    """Move earlier products that are not this attempt's identity out of the way (never deleted), with the reason."""
    moved = []
    dest = out / "superseded" / attempt
    for rel in rels:
        src = out / rel
        if src.exists():
            tgt = dest / rel
            if tgt.exists():
                tgt = tgt.with_name(f"{tgt.name}.{len(list(tgt.parent.glob(tgt.name + '*')))}")
            tgt.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(tgt))
            moved.append(rel)
    if moved:
        notes = RI.read_json(dest / "why.json") if (dest / "why.json").is_file() else []
        notes.append({"moved": moved, "why": why, "utc": RI.utc()})
        RI.write_json(dest / "why.json", notes)
    return moved


def admitted(out: Path, rel: str, want: dict, attempt: str, also: tuple[str, ...] = ()) -> dict | None:
    """The record at ``rel`` when its identity is ``want``; otherwise it (with ``also``) is superseded and None."""
    p = out / rel
    if not p.is_file():
        return None
    rec = RI.read_json(p)
    have = (rec.get("identity") or {}).get("sha256")
    if have == want["sha256"]:
        return rec
    supersede(out, [rel, *also], attempt, f"identity {(have or 'absent')[:16]} is not this attempt's "
                                          f"{want['sha256'][:16]}")
    return None


# ---- the contract ($0) ----------------------------------------------------------------------------------------------

def contract(out: Path, work: Path) -> dict:
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
    ids = identities(work, "cpu")
    RI.write_json(out / "state" / "identities.json", ids)
    shas = {k: v["sha256"] for k, v in ids.items() if isinstance(v, dict) and "sha256" in v}
    shas.update({f"candidate:{g}": v["sha256"] for g, v in ids["candidates"].items()})
    return {"design_sha256": sha(LMX.DESIGN), "digests_sha256": sha(LMX.DIGESTS), "spans": spans,
            "protocol_sha256": d["evaluation"]["protocol_sha256"], "anchor_session": ri_contract, "identities": shas}


# ---- build (GPU): one candidate -------------------------------------------------------------------------------------

def check_base(adapter, teacher, sel: dict, digests: dict) -> dict:
    """D rebuilt from the teacher with D's own builder, compared tensor by tensor with D's digests: the base of every
    candidate (a mismatch stops here)."""
    from aadistill.initialization.planning import frozen_mapping as FM
    base = FM.keep_blocks(adapter, teacher, [int(b) for b in sel["kept_blocks"]], seed=int(sel["source"]["seed"]))
    sd = base.state_dict()
    tied = bool(getattr(base.config, "tie_word_embeddings", False))
    names = [n for n in sd if not (tied and n == "lm_head.weight")]
    diff = [n for n in names if FM.tensor_sha256(sd[n]) != digests.get(n)]
    del base, sd
    if diff or set(names) != set(digests):
        raise DriverError(f"D rebuilt from the teacher is not D: {diff[:6]} / names {sorted(set(names) ^ set(digests))[:6]}")
    return {"tensors": len(names), "equal_to_D": len(names),
            "note": "every tensor of D rebuilt from the teacher compared with D's digests (a tensor identity)"}


def load_statistics(out: Path, want: dict, attempt: str) -> dict | None:
    """An earlier attempt's statistics, when their identity is this attempt's and the tensors re-hash to the record."""
    rec = admitted(out, f"records/{STATS}.json", want, attempt, also=(f"records/{STATS}.safetensors",))
    ten_p = out / "records" / f"{STATS}.safetensors"
    if rec is None or not ten_p.is_file():
        return None
    if sha(ten_p) != rec["tensors_sha256"]:
        raise DriverError(f"{ten_p.name} does not hash to its record's {rec['tensors_sha256'][:12]}")
    from safetensors.torch import load_file
    return load_file(str(ten_p))


def statistics_and_checks(out: Path, s, adapter, teacher, spans, d: dict, ids: dict, device: str, attempt: str,
                          status) -> dict:
    """M-act's inputs: the statistics and alignments, and the fold and permutation checks -- each reused when an
    earlier attempt's has this attempt's identity, else computed and written at once."""
    import torch
    from safetensors.torch import save_file
    n_q, n_kv, _ = adapter.head_groups(adapter.spec_of(teacher))
    _items, active, batches = LMX.calibration(s, teacher, device)
    stats = load_statistics(out, ids["statistics"], attempt)
    if stats is None:
        t1 = time.time()
        st = LMX.collect_statistics(adapter, teacher, spans, batches, active)
        secs = round(time.time() - t1, 1)
        exp = d["statistics"]["expected"]
        if st.sequences != exp["packed_sequences"] or st.positions != exp["statistics_positions"]:
            raise DriverError(f"{st.sequences} sequences / {st.positions} positions, the design expects "
                              f"{exp['packed_sequences']} / {exp['statistics_positions']}")
        t2 = time.time()
        al = LMX.alignments(st, n_q, n_kv)
        stats = LMX.statistics_tensors(st, al)
        del st
        torch.cuda.empty_cache()
        ten_p = out / "records" / f"{STATS}.safetensors"
        ten_p.parent.mkdir(parents=True, exist_ok=True)
        tmp = ten_p.with_name(f".{ten_p.name}.partial")
        save_file(stats, str(tmp))
        os.replace(tmp, ten_p)
        RI.write_json(out / "records" / f"{STATS}.json", {
            "identity": ids["statistics"], "tensors_sha256": sha(ten_p), "sequences": exp["packed_sequences"],
            "positions": exp["statistics_positions"], "profile_id": d["statistics"]["profile_id"],
            "position_policy": active.report(), "forward_context": s.config.forward_context.as_dict(),
            "pairs": {f"{a}->{b}": x["summary"] for (a, b), x in al.items()},
            "permutation_sha256": {k: hashlib.sha256(v.numpy().tobytes()).hexdigest()
                                   for k, v in stats.items() if "perm" in k},
            "statistics_seconds": secs, "alignment_seconds": round(time.time() - t2, 1),
            "producer_run": attempt, "utc": RI.utc()})
        status.mark("STATISTICS_SAVED")
    if admitted(out, "records/merge_checks.json", ids["checks"], attempt) is None:
        tol = float(d["checks"]["fold_equivalence"]["tolerance"])
        eq = LMX.equivalence_checks(adapter, teacher, spans, stats, next(iter(batches())), tol)
        RI.write_json(out / "records" / "merge_checks.json", {**eq, "identity": ids["checks"], "producer_run": attempt})
        if not eq["passed"]:
            raise DriverError(f"folding or a permutation changes a block's forward: worst {eq['worst']:.2e} > {tol}, "
                              f"non-finite {eq['non_finite']}")
        status.mark("EQUIVALENCE_OK")
    return stats


def build(out: Path, work: Path, device: str, status, g: str, attempt: str) -> dict:
    """One candidate: built, checked, saved; its delta written; the reconstruction from D and the delta FILE required
    to give its artifact digest; ``records/candidates/<g>.json`` (status ``built``)."""
    import torch
    from safetensors.torch import load_file, save_file
    from aadistill.initialization.planning import frozen_mapping as FM
    from aadistill.initialization.specs.arch import get_adapter
    from stages.teacher_modes import full_search as FS
    d, sel, ids = LMX.design(), RI.selection(), ids_of(out)
    s = RI.session(work, device)
    adapter = get_adapter(s.config.target_spec.family)
    spans = LMX.merge_spans(d, sel)
    seed = int(sel["source"]["seed"])
    digests = RI.read_json(LMX.DIGESTS)["tensors"]
    rec: dict = {"identity": ids["candidates"][g], "status": "building", "started_utc": RI.utc(), "spans": spans,
                 "producer_run": attempt}
    t0 = time.time()
    teacher, rec["root"] = FS.load_root(device)
    teacher.config.use_cache = False
    rec["seconds_teacher"] = round(time.time() - t0, 1)
    rec["d_rebuild"] = check_base(adapter, teacher, sel, digests)
    status.mark(f"D_REBUILT_EQUAL:{g}")
    stats = (statistics_and_checks(out, s, adapter, teacher, spans, d, ids, device, attempt, status)
             if g in NEEDS_STATISTICS else None)
    model, crec = LMX.build_candidate(adapter, teacher, sel, spans, g, stats, digests, seed=seed)
    if crec["checks"]["num_parameters"] != int(d["base"]["d_checkpoint"]["num_parameters"]):
        raise DriverError(f"{g}: {crec['checks']['num_parameters']} parameters, D has "
                          f"{d['base']['d_checkpoint']['num_parameters']}")
    if adapter.spec_of(model) != adapter.spec_of(teacher).replace(num_hidden_layers=len(sel["kept_blocks"])):
        raise DriverError(f"{g}: the structure is not D's")
    ident = RI._save(adapter, model, work / "groups" / g)
    rec["build"] = crec
    rec["checkpoint"] = RI._ident_record(ident)
    rec["config_equals_D"] = ident.config_sha256 == d["base"]["d_checkpoint"]["config_sha256"]
    if not rec["config_equals_D"]:
        raise DriverError(f"{g}: the configuration is not D's")
    delta = LMX.candidate_delta(sel, spans, model)
    tdig = LMX.candidate_tensor_digests(model, sorted(delta))
    del model
    torch.cuda.empty_cache()
    dp = out / "products" / g / "delta.safetensors"
    dp.parent.mkdir(parents=True, exist_ok=True)
    tmp = dp.with_name(f".{dp.name}.partial")
    save_file(delta, str(tmp), metadata={"candidate": g, "identity_sha256": ids["candidates"][g]["sha256"],
                                         "base_weights_sha256": d["base"]["d_checkpoint"]["weights_sha256"],
                                         "artifact_digest": ident.artifact_digest,
                                         "rule": "D's tensors with these replacing D's give the candidate"})
    os.replace(tmp, dp)
    del delta
    rec["delta"] = {"path": f"products/{g}/delta.safetensors", "sha256": sha(dp), "bytes": dp.stat().st_size,
                    "tensors": len(tdig), "tensor_digests": tdig,
                    "base": {"weights_sha256": d["base"]["d_checkpoint"]["weights_sha256"],
                             "tensor_digests": str(LMX.DIGESTS.relative_to(REPO))}}
    #: the reconstruction identity: D rebuilt, the delta FILE read back into it, saved by the same writer
    t1 = time.time()
    rebuilt = FM.keep_blocks(adapter, teacher, [int(b) for b in sel["kept_blocks"]], seed=seed)
    LMX.apply_delta(rebuilt, load_file(str(dp)))
    got = LMX.candidate_tensor_digests(rebuilt, sorted(tdig))
    rident = RI._save(adapter, rebuilt, work / "verify" / g)
    del rebuilt
    shutil.rmtree(work / "verify" / g, ignore_errors=True)
    rec["reconstruction"] = {"from": "D rebuilt from the teacher + the delta file read back, saved by the same writer",
                             "artifact_digest": rident.artifact_digest,
                             "equal": rident.artifact_digest == ident.artifact_digest and got == tdig,
                             "seconds": round(time.time() - t1, 1)}
    if not rec["reconstruction"]["equal"]:
        raise DriverError(f"{g}: D + the delta gives {rident.artifact_digest[:12]}, the candidate is "
                          f"{ident.artifact_digest[:12]}")
    rec.update(status="built", finished_utc=RI.utc(), gpu_peak=RI._gpu_peak(device))
    RI.write_json(out / "records" / "candidates" / f"{g}.json", rec)
    status.mark(f"BUILT:{g}")
    return rec


# ---- relay: one candidate -------------------------------------------------------------------------------------------

def relay_files(out: Path, g: str, rec: dict) -> dict[str, tuple[Path, str]]:
    """relay path -> (local file, sha256): the candidate's delta, and the statistics for a candidate that reads them."""
    d = LMX.design()
    prefix = d["products"]["relay_prefix"]
    files = {f"{prefix}/{g}/{rec['identity']['sha256'][:16]}/delta.safetensors": (out / rec["delta"]["path"],
                                                                                    rec["delta"]["sha256"])}
    if g in NEEDS_STATISTICS:
        srec = RI.read_json(out / "records" / f"{STATS}.json")
        files[f"{prefix}/statistics/{srec['identity']['sha256'][:16]}/{STATS}.safetensors"] = (
            out / "records" / f"{STATS}.safetensors", srec["tensors_sha256"])
    return files


def relay_upload(out: Path, run_id: str, g: str) -> dict:
    """The candidate's delta (and the statistics) to the private relay, downloaded again into a fresh cache and
    re-hashed; then the record says ``durable``."""
    from huggingface_hub import HfApi, hf_hub_download
    import tempfile
    d, ids = LMX.design(), ids_of(out)
    repo = d["products"]["relay_repo"]
    rp = out / "records" / "candidates" / f"{g}.json"
    rec = RI.read_json(rp)
    if rec.get("status") not in ("built", "durable") or rec["identity"]["sha256"] != ids["candidates"][g]["sha256"]:
        raise DriverError(f"{g}: no built candidate of this attempt's identity to make durable")
    files = relay_files(out, g, rec)
    for _path, (p, want) in files.items():
        if sha(p) != want:
            raise DriverError(f"{p.name} does not hash to its record")
    token = RI.POD_TOKEN.read_text().strip() if RI.POD_TOKEN.is_file() else None
    api = HfApi(token=token)
    t0 = time.time()
    for path, (p, _w) in sorted(files.items()):
        api.upload_file(path_or_fileobj=str(p), path_in_repo=path, repo_id=repo, repo_type="model",
                        commit_message=f"stage1 layer merge {run_id}: {g} {Path(path).name}")
    t1, got = time.time(), {}
    with tempfile.TemporaryDirectory(dir=str(out.parent)) as tmp:
        for path in sorted(files):
            got[path] = sha(Path(hf_hub_download(repo, path, repo_type="model", token=token, cache_dir=f"{tmp}/hf",
                                                 force_download=True)))
    want = {path: w for path, (_p, w) in files.items()}
    rel = {"relay": repo, "files_sha256": want, "verified": got == want,
           "delta_path": next(k for k in files if k.endswith("/delta.safetensors")),
           "upload_seconds": round(t1 - t0, 1), "verify_seconds": round(time.time() - t1, 1), "producer_run": run_id,
           "utc": RI.utc(), "rule": "uploaded from the pod's copy, downloaded again into a fresh cache and re-hashed"}
    rec["relay"] = rel
    if rel["verified"]:
        rec["status"] = "durable"
    RI.write_json(rp, rec)
    if not rel["verified"]:
        raise DriverError(f"{g}: the relay copy does not re-hash to the pod's")
    return rel


# ---- restore: a durable candidate's checkpoint ----------------------------------------------------------------------

def checkpoint_ready(work: Path, g: str, rec: dict, family: str) -> bool:
    """The scratch checkpoint exists and identifies as the recorded candidate."""
    from aadistill.initialization.specs.arch import get_adapter
    if not (work / "groups" / g / "config.json").is_file():
        return False
    try:
        ident = RI._identify(get_adapter(family), work / "groups" / g)
    except Exception:  # noqa: BLE001
        return False
    return ident.artifact_digest == rec["checkpoint"]["artifact_digest"]


def restore(out: Path, work: Path, device: str, g: str, attempt: str) -> dict:
    """D rebuilt from the teacher (checked against D's digests) with the delta written into it -- this attempt's file
    when it re-hashes, else the relay's; the saved checkpoint must give the recorded artifact digest."""
    from safetensors.torch import load_file
    from aadistill.initialization.planning import frozen_mapping as FM
    from aadistill.initialization.specs.arch import get_adapter
    from stages.teacher_modes import full_search as FS
    rec = RI.read_json(out / "records" / "candidates" / f"{g}.json")
    sel, ids = RI.selection(), ids_of(out)
    dp = out / rec["delta"]["path"]
    source = "this attempt's delta file"
    if not (dp.is_file() and sha(dp) == rec["delta"]["sha256"]):
        from huggingface_hub import hf_hub_download
        token = RI.POD_TOKEN.read_text().strip() if RI.POD_TOKEN.is_file() else None
        rel = rec["relay"]
        got = Path(hf_hub_download(rel["relay"], rel["delta_path"], repo_type="model", token=token,
                                   cache_dir=str(work / "hf_relay")))
        dp = work / "deltas" / g / "delta.safetensors"
        dp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(got, dp)
        shutil.rmtree(work / "hf_relay", ignore_errors=True)
        source = f"{rel['relay']}:{rel['delta_path']}"
        if sha(dp) != rec["delta"]["sha256"]:
            raise DriverError(f"{g}: the relay's delta does not hash to the record's {rec['delta']['sha256'][:12]}")
    teacher, _root = FS.load_root(device)
    adapter = get_adapter(ids["family"])
    base_check = check_base(adapter, teacher, sel, RI.read_json(LMX.DIGESTS)["tensors"])
    base = FM.keep_blocks(adapter, teacher, [int(b) for b in sel["kept_blocks"]], seed=int(sel["source"]["seed"]))
    LMX.apply_delta(base, load_file(str(dp)))
    ident = RI._save(adapter, base, work / "groups" / g)
    if ident.artifact_digest != rec["checkpoint"]["artifact_digest"]:
        shutil.rmtree(work / "groups" / g, ignore_errors=True)
        raise DriverError(f"{g}: D + the delta gives {ident.artifact_digest[:12]}, the record says "
                          f"{rec['checkpoint']['artifact_digest'][:12]}")
    res = {"candidate": g, "from": source, "artifact_digest": ident.artifact_digest, "base": base_check,
           "attempt": attempt, "utc": RI.utc()}
    RI.write_json(out / "records" / f"restored.{attempt}.{g}.json", res)
    return res


# ---- diagnostics and state evaluation (GPU) -------------------------------------------------------------------------

def candidate_digests(out: Path) -> dict:
    digs = {"D": LMX.design()["base"]["d_checkpoint"]["artifact_digest"]}
    for g in GROUPS:
        digs[g] = RI.read_json(out / "records" / "candidates" / f"{g}.json")["checkpoint"]["artifact_digest"]
    return digs


def diagnostics(out: Path, work: Path, device: str, status) -> dict:
    import copy
    import torch
    from aadistill.initialization.specs.arch import get_adapter
    from stages.teacher_modes import full_search as FS
    d, sel, ids = LMX.design(), RI.selection(), ids_of(out)
    s = RI.session(work, device)
    adapter = get_adapter(s.config.target_spec.family)
    spans = LMX.merge_spans(d, sel)
    teacher, _root = FS.load_root(device)
    teacher.config.use_cache = False
    blocks = adapter.blocks(teacher)
    digs = candidate_digests(out)
    cands = {"D": {sp[0]: copy.deepcopy(blocks[sp[0]]) for sp in spans}}
    for g in GROUPS:
        ident = RI._identify(adapter, work / "groups" / g)
        if ident.artifact_digest != digs[g]:
            raise DriverError(f"{g}'s checkpoint is not the recorded one")
        model = adapter.load(str(work / "groups" / g), device=device)
        cands[g] = {sp[0]: copy.deepcopy(adapter.blocks(model)[LMX.child_index(sel, sp[0])]) for sp in spans}
        del model
        torch.cuda.empty_cache()
    _items, active, batches = LMX.calibration(s, teacher, device)
    t0 = time.time()
    rec = LMX.diagnostics(adapter, teacher, spans, cands, batches, active)
    rec.update(identity=with_artifacts(ids["diagnostics"], **digs), seconds=round(time.time() - t0, 1),
               finished_utc=RI.utc())
    RI.write_json(out / "records" / "merge_diagnostics.json", rec)
    status.mark("DIAGNOSTICS_OK")
    return rec


def state_eval(out: Path, work: Path, device: str, status, attempt: str) -> dict:
    """Per candidate; a candidate an earlier attempt measured under this identity is kept, never measured again."""
    from aadistill.initialization.specs.arch import get_adapter
    from stages.teacher_modes import full_search as FS
    ids = ids_of(out)
    digs = candidate_digests(out)
    p = out / "records" / "merge_state_eval.json"
    old = RI.read_json(p) if p.is_file() else {}
    want = {g: with_artifacts(ids["state_eval"], candidate=digs[g]) for g in GROUPS}
    kept = {g: v for g, v in (old.get("groups") or {}).items()
            if g in want and (v.get("identity") or {}).get("sha256") == want[g]["sha256"]}
    dropped = sorted(set(old.get("groups") or {}) - set(kept))
    if dropped:
        supersede(out, ["records/merge_state_eval.json"], attempt,
                  f"state evaluation of {dropped} is not this attempt's identity")
    todo = [g for g in GROUPS if g not in kept]
    s = RI.session(work, device)
    rec = {"started_utc": old.get("started_utc") or RI.utc(),
           "measurement_protocol_id": s.evaluator.measurement_protocol_id, "suite": s.suite_manifest.get("suite_id"),
           "suite_content_sha256": s.suite_manifest.get("content_sha256"),
           "forward_context": s.config.forward_context.as_dict(), "role": "diagnostic only", "groups": kept,
           "kept_from_earlier": sorted(kept), "superseded": dropped}
    if todo:
        adapter = get_adapter(s.config.target_spec.family)
        teacher, _root = FS.load_root(device)
        t0 = time.time()
        s.evaluator.prime_reference(teacher)
        rec["prime_seconds"] = round(time.time() - t0, 1)
        for g in todo:
            ident = RI._identify(adapter, work / "groups" / g)
            if ident.artifact_digest != digs[g]:
                raise DriverError(f"{g}'s checkpoint is not the recorded one")
            model = adapter.load(str(work / "groups" / g), device=device)
            model.config.use_cache = False
            t1 = time.time()
            ev = s.evaluator.evaluate(model, ident.artifact_digest)
            rec["groups"][g] = {"identity": want[g], "artifact_digest": ident.artifact_digest,
                                "seconds": round(time.time() - t1, 1), "values": dict(getattr(ev, "values", {}) or {}),
                                "positions": getattr(ev, "positions", None),
                                "detail": json.loads(json.dumps(getattr(ev, "detail", {}) or {}, default=str)),
                                "producer_run": attempt}
            del model
            RI.write_json(p, rec)
    rec["finished_utc"] = RI.utc()
    RI.write_json(p, rec)
    status.mark("STATE_EVAL_OK")
    return rec


# ---- evaluation (vLLM, RI-S1's runners) -----------------------------------------------------------------------------

def evaluation_plan(kinds=None, modes=None, groups=None) -> list[tuple[str, str, str]]:
    plan = [tuple(x) for x in LMX.design()["evaluation"]["order"]]
    return [x for x in plan if (not groups or x[0] in groups) and (not modes or x[1] in modes)
            and (not kinds or x[2] in kinds)]


def unit_identity(ids: dict, artifact_digest: str, mode: str) -> dict:
    ri = RI.read_json(REPO / LMX.design()["base"]["ri_s1_design"]["path"])
    return identity("evaluation_unit", evaluation=ids["evaluation"]["sha256"], candidate=artifact_digest, mode=mode,
                    template_kwargs=ri["evaluation"]["modes"][mode]["template_kwargs"])


def admit_unit(out: Path, unit_dir: Path, want: dict, attempt: str) -> str:
    """A (candidate, mode) directory's sealed work is taken only under this attempt's identity -- written before the
    first sample, compared on every later attempt. Sealed work with another identity, or with none, is superseded."""
    p = unit_dir / "identity.json"
    have = RI.read_json(p) if p.is_file() else None
    if have is not None and have.get("sha256") == want["sha256"]:
        return "taken"
    held = unit_dir.is_dir() and any(x.is_file() for x in unit_dir.rglob("*"))
    if held:
        why = (f"identity {str(have.get('sha256'))[:16]} is not this attempt's {want['sha256'][:16]}"
               if have is not None else "sealed work with no recorded identity")
        supersede(out, [str(unit_dir.relative_to(out))], attempt, why)
    RI.write_json(p, want)
    return "superseded" if held else "new"


def evaluate(out: Path, work: Path, vllm_py: str, status, attempt: str, opts: dict | None = None,
             kinds=None, modes=None, groups=None) -> dict:
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    ids = ids_of(out)
    digs = candidate_digests(out)
    summary = RI.read_json(out / "state" / "evaluation.json") if (out / "state" / "evaluation.json").is_file() else {}
    summary["_execution"] = {**(opts or {}), "attempt": attempt, "kinds": kinds, "modes": modes, "groups": groups}
    seen: dict[str, str] = {}
    for g, mode, kind in evaluation_plan(kinds, modes, groups):
        name = f"{g}_{mode}"
        unit_dir = out / "eval" / name
        if name not in seen:
            seen[name] = admit_unit(out, unit_dir, unit_identity(ids, digs[g], mode), attempt)
        pkg = work / "packages" / g
        if not (pkg / "config.json").exists():
            pk = RI.package(work / "groups" / g, pkg)
            pk["artifact_digest"] = digs[g]
            RI.write_json(out / "eval" / f"package_{g}.json", pk)
        status.mark(f"EVAL_START:{name}:{kind}")
        t0 = time.time()
        run = RI.fluid if kind == "fluid" else RI.capability
        mns = (opts or {}).get("max_num_seqs")
        unit_opts = {**(opts or {}), "max_num_seqs": (mns.get(g) if isinstance(mns, dict) else mns)}
        res = run(pkg, unit_dir, name, mode, vllm_py, env, attempt, unit_opts)
        summary.setdefault(name, {})[kind] = {**res, "seconds": round(time.time() - t0, 1), "utc": RI.utc(),
                                              "sealed_work": seen[name]}
        RI.write_json(out / "state" / "evaluation.json", summary)
        status.mark(f"EVAL_OK:{name}:{kind}")
    return summary


# ---- main -----------------------------------------------------------------------------------------------------------

def child(phase: str, a) -> int:
    out, work, status = Path(a.out), Path(a.work), RI.Status(a.status)
    if phase == "build":
        build(out, work, a.device, status, a.group, a.run_id)
    elif phase == "relay":
        relay_upload(out, a.run_id, a.group)
    elif phase == "restore":
        restore(out, work, a.device, a.group, a.run_id)
    elif phase == "diagnostics":
        diagnostics(out, work, a.device, status)
    elif phase == "state_eval":
        state_eval(out, work, a.device, status, a.run_id)
    else:
        raise DriverError(f"unknown phase {phase!r}")
    return 0


def run_child(phase: str, a, *, online: bool = False, group: str | None = None) -> None:
    env = {**os.environ, **({"HF_HUB_OFFLINE": "0"} if online else {})}
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", phase, "--run-id", a.run_id, "--out", a.out,
           "--work", a.work, "--status", a.status, "--device", a.device, *(["--group", group] if group else [])]
    log = Path(a.out) / "logs" / f"{phase}{'.' + group if group else ''}.{a.run_id}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as lf:
        rc = subprocess.run(cmd, cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise DriverError(f"phase {phase}{' ' + group if group else ''} exited {rc}; see logs/{log.name}")


def candidate_state(out: Path, g: str, attempt: str) -> str:
    """``durable`` (this identity, verified on the relay), ``built`` (this identity, its delta here and re-hashing,
    not yet on the relay) or ``absent``. An earlier attempt's record of another identity, or an unfinished one, is
    superseded."""
    rec = admitted(out, f"records/candidates/{g}.json", ids_of(out)["candidates"][g], attempt, also=(f"products/{g}",))
    if rec is None:
        return "absent"
    if rec.get("status") == "durable" and (rec.get("relay") or {}).get("verified"):
        return "durable"
    dp = out / ((rec.get("delta") or {}).get("path") or f"products/{g}/delta.safetensors")
    if rec.get("status") == "built" and dp.is_file() and sha(dp) == rec["delta"]["sha256"]:
        return "built"
    supersede(out, [f"records/candidates/{g}.json", f"products/{g}"], attempt, "an unfinished build (no verified delta)")
    return "absent"


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
    ap.add_argument("--group", default=None, help=argparse.SUPPRESS)
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
        report["phases"]["contract"] = {**contract(out, work), "seconds": round(time.time() - t0, 1)}
        RI.write_json(out / "state" / "report.json", report)
        status.mark("CONTRACT_OK")
        if a.check_only:
            report["status"] = "CHECK_ONLY_OK"
            return 0
        #: one candidate at a time: built, its reconstruction verified, durable on the relay -- then the next
        cands = report["phases"].setdefault("candidates", {})
        for g in GROUPS:
            st = candidate_state(out, g, a.run_id)
            cands[g] = {"found": st}
            if st == "absent" and "build" in phases:
                t0 = time.time()
                run_child("build", a, group=g)
                cands[g]["build_seconds"] = round(time.time() - t0, 1)
                st = "built"
            if st == "built" and "relay" in phases:
                t0 = time.time()
                run_child("relay", a, online=True, group=g)
                cands[g]["relay_seconds"] = round(time.time() - t0, 1)
                st = "durable"
                status.mark(f"DURABLE:{g}")
            cands[g]["now"] = st
            RI.write_json(out / "state" / "report.json", report)
        ids = ids_of(out)
        if any(p in phases for p in ("diagnostics", "state_eval", "evaluate")):
            for g in GROUPS:
                if cands[g]["now"] != "durable":
                    raise DriverError(f"{g} is {cands[g]['now']}: nothing measures a candidate that is not durable")
                rec = RI.read_json(out / "records" / "candidates" / f"{g}.json")
                if not checkpoint_ready(work, g, rec, ids["family"]):
                    t0 = time.time()
                    run_child("restore", a, online=True, group=g)
                    cands[g]["restore_seconds"] = round(time.time() - t0, 1)
                    status.mark(f"RESTORED:{g}")
        if "diagnostics" in phases:
            want = with_artifacts(ids["diagnostics"], **candidate_digests(out))
            if admitted(out, "records/merge_diagnostics.json", want, a.run_id):
                report["phases"]["diagnostics"] = {"status": "taken over"}
            else:
                t0 = time.time()
                run_child("diagnostics", a)
                report["phases"]["diagnostics"] = {"status": "measured", "seconds": round(time.time() - t0, 1)}
            RI.write_json(out / "state" / "report.json", report)
        if "state_eval" in phases:
            t0 = time.time()
            run_child("state_eval", a)
            report["phases"]["state_eval"] = {"status": "done", "seconds": round(time.time() - t0, 1)}
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
