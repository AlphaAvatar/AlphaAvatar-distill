#!/usr/bin/env python3
"""On-pod driver of SOB1's phase 1 (one session kind, ``sob1_phase1``): the three ~2B initializations built from the
original teacher with A's operators, each durable the moment it exists, then every untrained arm and the teacher
evaluated on all of eval-set-1 in both modes at 8,192 tokens, each finished sample sealed.

    cd /workspace/aad && HF_HUB_OFFLINE=1 PYTHONPATH=src:scripts:scripts/stages/stage-1:scripts/stages/stage-0/calibration_volume \\
        /opt/train/bin/python scripts/stages/stage-1/single_operator_budget/phase1_driver.py --run-id <run> \\
        --out <evidence dir> --work <scratch> --status <file> --device cuda --vllm-python /opt/vllm/bin/python \\
        [--resume <secured evidence dir or archive>]

    # $0 on the dev box: the design, the reused selections and the session's operator config hashes (nothing loaded)
    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python \\
        scripts/stages/stage-1/single_operator_budget/phase1_driver.py --check-only --out <s>/out --work <s>/w --status <s>/st

Building (`design.json :: phase_1.build_order`; one GPU child process per arm, the teacher loaded in each, every shard
hashed), always on the unmodified teacher under the packed search's own session (inputs v3; `selection.json`):

* **residual width only** -- no teacher statistics pass: TPI1's sealed teacher statistics
  (``teacher_residual_statistics.safetensors``, staged by setup from the relay) are loaded and checked against
  `selection.json :: residual_width.statistics` (sha256, every tensor's layout, the statistics positions); the pass
  that produced them passed TPI1's digest gate. P at 1,024 over all 37 points recomputed from them must reproduce the
  recorded diagnostics; then P at 1,280 from the same statistics over all 37 points, weights 9 / 1 / 8;
* **FFN + attention only** -- the FFN operator at 3,072 (must build the journal's FFN state bit for bit) and at 3,840
  sharing one statistics pass, both under the journal's config hash; the journal's 16 / 8 heads (their child must
  equal the ATTENTION state's digest);
* **depth only** -- the DEPTH operator 36 -> 16 under the journal's config hash, resuming its eight committed rounds
  (`selection.json :: depth_only.rounds`) and every round an earlier attempt of this study persisted: the operator
  re-checks them, re-scores round 8's chosen subset and the latest persisted round's exactly, and scores the rest of
  rounds 9-20 (270 candidate subsets in all) against the unbypassed teacher; rounds 1-8 must equal the journal's order.
  Each new round (its choice and every candidate's score) is written to ``depth_only/records/depth_rounds.json`` the
  moment it commits, so an interrupted session resumes from its latest verified round.

Each arm: one `FrozenMapping` keyed by root indices, saved and loaded back (`save_mapping`: its files from save's own
record -- ``mapping.json``, plus the tensor file only for the declared width projection), materialized onto the teacher
(FFN -> blocks -> width -> heads), geometry and parameter count against `structures.json`, reloaded with a finite
forward, state evaluation against the teacher (diagnostic only), then exactly the build record's manifest to the relay
(`persist`; re-downloaded and re-hashed) and the arm's ``<arm>/init_record.json``, COMPLETE only when it shows every
product durable (`common.product_problems`). A gate that fails stops the session before anything is built on it.

Evaluating (`phase_1.evaluation_order`): eight sets, each untrained arm (its checkpoint on this pod, else fetched from
its init record's relay copy and re-hashed) and the teacher (the pinned snapshot) in both modes, through one package
rule (`common.package`, which records the weights' sha256) and the runners with the registered protocol
(`common.protocol`, 8,192 tokens). A set is complete only on its files (`completion.check`: item coverage, repeats,
generation records, protocol summary, the formal report, each sealed part's provenance); one an earlier attempt
completed is taken as it is, otherwise the smallest repair before anything is merged -- a ``{}`` or stale report
rescored, every part whose generation's weights, mode or protocol cannot be shown isolated and the proven ones reused,
only the items no proven part holds generated -- and checked again. Phase 1
ends ALL_DONE only when the three inits are durable, the eight sets are complete and each mode's four sets share one
protocol (`completion.phase`); then the comparison (`analysis.phase1_table`) is written to ``comparison/``.

``--resume``: a previous attempt's secured evidence: an arm whose init record shows every product durable is not
rebuilt; its persisted DEPTH rounds and sealed parts are kept (each part then checked before it is reused).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1", "scripts/stages/stage-0/calibration_volume"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.reasoning_inheritance.driver import (  # noqa: E402
    Status, _identify, _ident_record, _gpu_peak, read_json, sha256_file, utc, write_json)
from stages.single_operator_budget import common as C  # noqa: E402

SCHEMA = "aadistill.stage1.single_operator_budget.init_record/v1"
TEACHER = "teacher"


class DriverError(RuntimeError):
    pass


def selection() -> dict:
    return read_json(C.SELECTION)


def tpi1_selection() -> dict:
    from stages.teacher_parent_initialization import init_driver as TI
    return TI.selection()


def phase() -> dict:
    return C.design()["phase_1"]


def fixed_spec(s, step: dict, target, device: str, label: str, expected_digest: str | None = None):
    """One root-level operator step on the teacher as a fixed path, its config hash pinned to the journal's."""
    from aadistill.initialization.planning.fixed_path import FixedPathSpec, FixedPathStep
    t = C.design()["teacher"]
    return FixedPathSpec(
        path_id=f"stage1.single_operator_budget.{label}", family=s.config.target_spec.family, target_spec=target,
        steps=(FixedPathStep(impl_id=step["impl_id"], profile_id=step["profile_id"], label=label,
                             expected_config_hash=step["config_hash"], expected_artifact_digest=expected_digest),),
        root_repo_id=t["model"], root_revision=t["revision"], device=device,
        seed=int(selection()["source"]["seed"]), position_policy=s.config.position_policy,
        distribution_support=s.config.distribution_support, forward_context=s.config.forward_context)


def run_step(spec, adapter, teacher, root_sha: str, work: Path, s, cache, resume=None, observer=None):
    from aadistill.initialization.planning.fixed_path import _run_steps
    (res,) = _run_steps(spec, adapter=adapter, model=teacher, indices=[0], parent_spec=adapter.spec_of(teacher),
                        parent_digest=root_sha, workdir=work, repo_root=str(REPO), stats_cache=cache,
                        calibration_items=None, on_step=None, deadline=None, execution=s.execution,
                        resume_states=None if resume is None else {0: resume},
                        progress_observers=None if observer is None else {0: observer})
    return res


def gate(rec: dict, name: str, got: str, want: str) -> None:
    rec["gates"][name] = {"artifact_digest": got, "journal_artifact_digest": want, "equal": got == want}
    if got != want:
        raise DriverError(f"{name}: the rebuilt child {got[:12]} is not the journal state's {want[:12]}")


# ---- the three builds --------------------------------------------------------------------------------------------------

def load_statistics(path: Path, want: dict) -> tuple[dict, dict]:
    """The sealed teacher statistics (`selection.json :: residual_width.statistics`): refused unless the file's sha256,
    every tensor's name, shape and dtype, and its statistics positions (``residual_count``) are the recorded ones.
    Returns the state and the checks."""
    from safetensors.torch import load_file
    if not path.is_file():
        raise DriverError(f"{path} is not on this pod: setup stages it from {want['relay']['path']}")
    got = sha256_file(path)
    if got != want["sha256"]:
        raise DriverError(f"{path.name} hashes to {got[:12]}, the sealed statistics are {want['sha256'][:12]}")
    state = load_file(str(path), device="cpu")
    layout = {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in sorted(state.items())}
    count = int(state["residual_count"].reshape(-1)[0].item()) if "residual_count" in state else None
    checks = {"sha256": got, "sha256_equal": True, "layout_equal": layout == want["tensors"],
              "residual_count": count, "positions_equal": count == int(want["stats_tokens"]),
              "points": int(state["residual_sum"].shape[0]) if "residual_sum" in state else None}
    if not (checks["layout_equal"] and checks["positions_equal"]):
        raise DriverError(f"{path.name} is not the sealed statistics' layout and positions: {checks}")
    return state, checks


def select_residual_width(rec, s, adapter, teacher, root_sha, work, device):
    """No teacher statistics pass: TPI1's sealed statistics (the pass that passed TPI1's digest gate) are loaded,
    checked (`load_statistics`) and must reproduce the recorded 37-point diagnostics at 1,024 before P at 1,280."""
    import torch
    from aadistill.initialization.planning import frozen_mapping as FM
    sel = selection()["residual_width"]
    n_root = selection()["n_root_blocks"]
    t0 = time.time()
    state, checks = load_statistics(C.statistics_input()["local"], sel["statistics"])
    rec["statistics"] = {"source": "loaded: TPI1's sealed teacher statistics (no teacher forward here)",
                         "relay": sel["statistics"]["relay"], "tpi1_init_record": sel["statistics"]["tpi1_init_record"],
                         **checks, "load_seconds": round(time.time() - t0, 1)}
    if checks["points"] != n_root + 1 or max(sel["points"]) >= checks["points"] or sel["post_norm_point"] != n_root:
        raise DriverError(f"the statistics hold {checks['points']} points; the selection reads {sel['points']} "
                          f"and the post-norm point {sel['post_norm_point']}")
    diag = sel["gate"]["all_points_diagnostics"]
    w1024 = FM.width_transform_from_statistics(
        state, d_child=1024, points=sel["points"], point_weights=diag["point_weights"],
        final_norm_weight=adapter.final_norm(teacher).weight, post_norm_point=sel["post_norm_point"], device=device)
    keys = ("energy_captured_frac", "top_eigenvalue", "min_kept_eigenvalue")
    rec["statistics_reproduction"] = {"recomputed": {k: w1024.diagnostics.get(k) for k in keys},
                                      "recorded": {k: diag[k] for k in keys},
                                      "equal": all(w1024.diagnostics.get(k) == diag[k] for k in keys)}
    rec["gates"]["STATISTICS_DIAGNOSTICS_1024"] = dict(rec["statistics_reproduction"])
    if not rec["statistics_reproduction"]["equal"]:
        raise DriverError(f"the sealed statistics do not reproduce the 37-point diagnostics: "
                          f"{rec['statistics_reproduction']}")
    del w1024
    transform = FM.width_transform_from_statistics(
        state, d_child=int(sel["hidden_size"]), points=sel["points"], point_weights=sel["point_weights"],
        final_norm_weight=adapter.final_norm(teacher).weight, post_norm_point=sel["post_norm_point"], device=device)
    rec["width_transform"] = {"d_child": int(sel["hidden_size"]), "points": sel["points"],
                              "point_weights": sel["point_weights"], "post_norm_point": sel["post_norm_point"],
                              "scale": transform.scale, "diagnostics": dict(transform.diagnostics)}
    del state
    torch.cuda.empty_cache()
    return {"kept_blocks": tuple(range(n_root)), "width": transform}, "all"


def select_ffn_attention(rec, s, adapter, teacher, root_sha, work, device):
    import torch
    from aadistill.initialization.planning import frozen_mapping as FM
    from aadistill.initialization.statistics.spec import StatsCache
    sel = selection()["ffn_attention"]
    n_root = selection()["n_root_blocks"]
    base = adapter.spec_of(teacher)
    cache = StatsCache()
    step = {"impl_id": sel["ffn"]["impl_id"], "profile_id": sel["ffn"]["profile_id"],
            "config_hash": sel["ffn"]["config_hash"]}
    t0 = time.time()
    g = run_step(fixed_spec(s, step, base.replace(intermediate_size=sel["ffn"]["gate"]["intermediate_size"]), device,
                            "ffn_gate_3072", sel["ffn"]["gate"]["artifact_digest"]),
                 adapter, teacher, root_sha, work / "ffn_gate", s, cache)
    gate(rec, "FFN@3072", g.identity.artifact_digest, sel["ffn"]["gate"]["artifact_digest"])
    shutil.rmtree(work / "ffn_gate", ignore_errors=True)
    r = run_step(fixed_spec(s, step, base.replace(intermediate_size=sel["ffn"]["intermediate_size"]), device,
                            "ffn_3840"), adapter, teacher, root_sha, work / "ffn_3840", s, cache)
    rec["ffn_seconds"] = round(time.time() - t0, 1)
    shutil.rmtree(work / "ffn_3840", ignore_errors=True)
    kept, small = r.selection["kept_neurons"], g.selection["kept_neurons"]
    rec["ffn"] = {"intermediate_size": sel["ffn"]["intermediate_size"], "config_hash": step["config_hash"],
                  "result_spec_hash": r.result_spec_hash, "child_artifact_digest": r.identity.artifact_digest,
                  "gate_selection_nested": all(set(a) <= set(b) for a, b in zip(small, kept)),
                  "trace": {k: v for k, v in r.trace.items() if k not in ("kept_neurons",)}}
    heads = [sel["attention"]["query_heads"][str(b)] for b in range(n_root)]
    child = FM.select_query_heads(adapter, teacher, heads, seed=int(selection()["source"]["seed"]))
    from stages.teacher_parent_initialization.init_driver import _save
    ident = _save(adapter, child, work / "gates" / "ATTENTION")
    del child
    gate(rec, "ATTENTION", ident.artifact_digest, sel["attention"]["artifact_digest"])
    shutil.rmtree(work / "gates", ignore_errors=True)
    torch.cuda.empty_cache()
    return {"kept_blocks": tuple(range(n_root)), "ffn_neurons": {b: tuple(kept[b]) for b in range(n_root)},
            "query_heads": {b: tuple(heads[b]) for b in range(n_root)}}, "all"


DEPTH_ROUNDS = "depth_rounds.json"


def depth_rounds_path(out: Path) -> Path:
    return out / "depth_only" / "records" / DEPTH_ROUNDS


def persisted_rounds(out: Path, step: dict, journal: list[dict]) -> list[dict]:
    """The rounds after the journal's that an earlier attempt of this study committed (`persist_round`), refused unless
    they were written for this step (operator, profile, config hash) and continue the journal's rounds without a gap."""
    p = depth_rounds_path(out)
    if not p.is_file():
        return []
    doc = read_json(p)
    if doc.get("step") != {k: step[k] for k in ("impl_id", "profile_id", "config_hash")}:
        raise DriverError(f"{p.name} was written for another step: {doc.get('step')}")
    rounds = [r["record"] for r in doc.get("rounds") or []]
    want = list(range(len(journal), len(journal) + len(rounds)))
    if [int(r["round"]) for r in rounds] != want:
        raise DriverError(f"{p.name} holds rounds {[r.get('round') for r in rounds]}, not {want} after the journal's")
    return rounds


def persist_round(out: Path, step: dict, run_id: str, record: dict) -> None:
    """One committed DEPTH round (its choice and every candidate's score), appended atomically to the evidence
    directory the moment the operator commits it: the dev box pulls it on its next poll and a later attempt resumes
    from it (`persisted_rounds`)."""
    p = depth_rounds_path(out)
    doc = read_json(p) if p.is_file() else {
        "schema": "aadistill.stage1.single_operator_budget.depth_rounds/v1",
        "step": {k: step[k] for k in ("impl_id", "profile_id", "config_hash")}, "rounds": []}
    doc["rounds"].append({"attempt": run_id, "utc": utc(), "record": record})
    write_json(p, doc)


def select_depth(rec, s, adapter, teacher, root_sha, work, device, out: Path | None = None, run_id: str = "check"):
    """DEPTH 36 -> 16, resuming the journal's eight rounds (round 8's chosen subset re-scored exactly) and every round
    an earlier attempt of this study persisted (the latest of them re-scored exactly too); each new round is persisted
    as it commits (`persist_round`)."""
    sel = selection()["depth_only"]
    keep = int(sel["resume"]["target_layers"])
    base = adapter.spec_of(teacher)
    journal = sel["rounds"]
    own = persisted_rounds(out, sel["step"], journal) if out is not None else []
    rec["depth_resume"] = {"journal_rounds": len(journal), "persisted_rounds": len(own),
                           "anchors": sorted({len(journal) - 1, len(journal) + len(own) - 1})}
    t0 = time.time()
    r = run_step(fixed_spec(s, sel["step"], base.replace(num_hidden_layers=keep), device, "depth_36_to_16"),
                 adapter, teacher, root_sha, work / "depth", s, None,
                 resume={"rounds": journal + own, "anchors": [len(journal) - 1],
                         "source": f"journal state {sel['state_id']}" + (f" + {len(own)} rounds this study persisted"
                                                                       if own else "")},
                 observer=None if out is None else (lambda r_: persist_round(out, sel["step"], run_id, r_)))
    rec["depth_seconds"] = round(time.time() - t0, 1)
    order = [int(x) for x in r.trace["removal_order"]]
    replayed = [int(x["chosen"]) for x in journal]
    rec["depth"] = {"removal_order": order, "kept_layers": [int(x) for x in r.trace["kept_layers"]],
                    "first_rounds_are_the_journal": order[:len(replayed)] == replayed,
                    "persisted_rounds_resumed": len(own),
                    "local_metrics": r.local_metrics, "result_spec_hash": r.result_spec_hash,
                    "child_artifact_digest": r.identity.artifact_digest,
                    "search_rounds": r.selection.get("search_rounds"), "packing": r.trace.get("packing")}
    if not rec["depth"]["first_rounds_are_the_journal"] or len(order) != selection()["n_root_blocks"] - keep:
        raise DriverError(f"DEPTH removed {order}; its first rounds must be the journal's {replayed}")
    shutil.rmtree(work / "depth", ignore_errors=True)
    return {"kept_blocks": tuple(sorted(rec["depth"]["kept_layers"]))}, "kept"


SELECT = {"residual_width": select_residual_width, "ffn_attention": select_ffn_attention, "depth_only": select_depth}


def save_mapping(mapping, directory: Path, arm: str):
    """`frozen_mapping.save`, the mapping loaded back under the content digest save recorded, and its file manifest
    from save's own record (`common.mapping_manifest`: ``mapping.json``, plus the tensor file only for a declared
    width projection). Returns (the reloaded mapping, the manifest)."""
    from aadistill.initialization.planning import frozen_mapping as FM
    saved = FM.save(mapping, directory)
    back = FM.load(directory, expected_content_digest=saved["content_digest"])
    return back, C.mapping_manifest(saved, directory, arm)


def checkpoint_files(ident, ckpt: Path) -> dict[str, str]:
    """The checkpoint's manifest (`common.INIT_FILES`): the single shard's sha256 as identified, the config hashed."""
    if [s.filename for s in ident.shards] != ["model.safetensors"]:
        raise DriverError(f"{ckpt} is {[s.filename for s in ident.shards]}, not one model.safetensors")
    missing = [n for n in C.INIT_FILES if not (ckpt / n).is_file()]
    if missing:
        raise DriverError(f"{ckpt} lacks {missing}")
    return {"model.safetensors": ident.shards[0].sha256, "config.json": sha256_file(ckpt / "config.json")}


def build(arm: str, out: Path, work: Path, device: str, status: Status, run_id: str) -> dict:
    """One arm (module docstring), in a GPU child process. Writes ``out/<arm>/records/build.json`` as it goes."""
    import torch
    from aadistill.initialization.planning import frozen_mapping as FM
    from aadistill.initialization.specs.arch import get_adapter
    from stages.teacher_modes import full_search as FS
    from stages.teacher_parent_initialization import init_driver as TI
    d = C.design()
    st = read_json(C.STRUCTURES)["arms"][arm]
    s = TI.session(work / "session", device)
    adapter = get_adapter(s.config.target_spec.family)
    rec: dict = {"started_utc": utc(), "run_id": run_id, "arm": arm, "design_sha256": sha256_file(C.DESIGN),
                 "selection_sha256": sha256_file(C.SELECTION), "gates": {}}
    progress = out / arm / "records" / "build.json"
    t0 = time.time()
    teacher, rec["root"] = FS.load_root(device)
    rec["seconds_teacher"] = round(time.time() - t0, 1)
    if rec["root"]["root_teacher_sha256"] != d["teacher"]["root_teacher_sha256"]:
        raise DriverError("the loaded teacher is not the design's root")
    root_sha = rec["root"]["root_teacher_sha256"]
    aw = work / arm
    extra = {"out": out, "run_id": run_id} if arm == "depth_only" else {}
    parts, blocks = SELECT[arm](rec, s, adapter, teacher, root_sha, aw, device, **extra)
    write_json(progress, rec)
    status.mark(f"SELECTED:{arm}")
    n_root = selection()["n_root_blocks"]
    mapping = FM.FrozenMapping(
        root={"model": d["teacher"]["model"], "revision": d["teacher"]["revision"], "root_teacher_sha256": root_sha},
        n_root_blocks=n_root, kept_blocks=tuple(parts["kept_blocks"]), ffn_neurons=parts.get("ffn_neurons"),
        query_heads=parts.get("query_heads"), width=parts.get("width"),
        provenance={"arm": arm, "selection_sha256": rec["selection_sha256"], "design_sha256": rec["design_sha256"],
                    "rule": d["arms"][arm]["selection"]})
    bad = FM.problems(mapping, root_spec=adapter.spec_of(teacher))
    if bad:
        raise DriverError(f"the mapping is malformed: {bad}")
    prod = aw / "products"
    mapping, rec["mapping"] = save_mapping(mapping, prod / "mapping", arm)
    model, roots = FM.materialize(adapter, teacher, mapping, blocks=blocks, non_depth=True,
                                  seed=int(selection()["source"]["seed"]))
    ckpt = prod / "init"
    ident = TI._save(adapter, model, ckpt)
    spec = adapter.spec_of(model)
    geo = {k: (bool(spec[k]) if k == "tie_word_embeddings" else int(spec[k])) for k in st["geometry"]}
    n_params = int(sum(p.numel() for p in model.parameters()))
    rec["checkpoint"] = {**_ident_record(ident), "root_blocks": roots, "geometry": geo,
                         "geometry_matches_structures": geo == st["geometry"], "num_parameters_counted": n_params,
                         "num_parameters_match_structures": n_params == int(st["num_parameters"])}
    if arm == "depth_only":
        rec["checkpoint"]["equals_the_operators_child"] = ident.artifact_digest == rec["depth"]["child_artifact_digest"]
    if not (rec["checkpoint"]["geometry_matches_structures"] and rec["checkpoint"]["num_parameters_match_structures"]
            and len(ident.shards) == 1 and rec["checkpoint"].get("equals_the_operators_child", True)):
        raise DriverError(f"{arm} is {geo}, {n_params} parameters: not structures.json's arm")
    rec["checkpoint"]["files_sha256"] = checkpoint_files(ident, ckpt)
    del model
    torch.cuda.empty_cache()
    m = adapter.load(str(ckpt), device=device)
    m.config.use_cache = False
    ids = torch.arange(1, 65, device=next(m.parameters()).device).unsqueeze(0)
    rec["checkpoint"]["reload_forward_finite"] = bool(torch.isfinite(m(input_ids=ids).logits).all().item())
    if not rec["checkpoint"]["reload_forward_finite"]:
        raise DriverError(f"the reloaded {arm} gives non-finite logits")
    write_json(progress, rec)
    status.mark(f"INIT_BUILT:{arm}")
    try:                                     #: diagnostic only: a failure is recorded, never a gate
        t2 = time.time()
        s.evaluator.prime_reference(teacher)
        ev = s.evaluator.evaluate(m, ident.artifact_digest)
        rec["state_evaluation"] = {"measurement_protocol_id": s.evaluator.measurement_protocol_id,
                                   "suite": s.suite_manifest.get("suite_id"),
                                   "values": dict(getattr(ev, "values", {}) or {}),
                                   "seconds": round(time.time() - t2, 1), "role": "diagnostic only"}
    except Exception as exc:  # noqa: BLE001
        rec["state_evaluation"] = {"error": f"{type(exc).__name__}: {str(exc)[:1500]}", "role": "diagnostic only"}
    rec["gpu_peak"] = _gpu_peak(device)
    rec["finished_utc"] = utc()
    write_json(progress, rec)
    return rec


def init_record(arm: str, rec: dict, relay: dict) -> dict:
    """The provenance record a training plan binds (`initialization_provenance.read`: ``root_teacher``,
    ``commit.selected_rows``) and where every durable byte of the arm is."""
    ck = rec["checkpoint"]
    return {"schema": SCHEMA, "status": "COMPLETE", "arm": arm, "run_id": rec["run_id"], "root_teacher": rec["root"],
            "commit": {"selected_rows": [{"state_id": f"{arm}/{rec['mapping']['content_digest'][:32]}",
                                          "path": C.design()["arms"][arm]["label"] + ": every selection on the teacher",
                                          "artifact_digest": ck["artifact_digest"],
                                          "single_shard_sha256": ck["shards"][0]["sha256"]}]},
            "checkpoint": ck, "gates": rec["gates"], "mapping": rec["mapping"],
            **{k: rec[k] for k in ("depth", "ffn", "width_transform", "statistics_reproduction", "statistics")
               if k in rec},
            "state_evaluation": rec.get("state_evaluation"), "relay": relay,
            "design_sha256": rec["design_sha256"], "selection_sha256": rec["selection_sha256"], "finished_utc": utc()}


def persist(arm: str, out: Path, work: Path, run_id: str, status: Status, put=None) -> dict:
    """The arm's products to the relay (online child processes; ``put`` for a test), exactly the build record's
    manifest (`common.products`), each group verified only when the relay's re-downloaded hashes are that manifest;
    then its init record, written COMPLETE only when `common.product_problems` finds nothing."""
    from stages.teacher_parent_initialization import relay as R
    put = put or (lambda payload, timeout_s: R.run_online("put", payload, timeout_s=timeout_s))
    rec = read_json(out / arm / "records" / "build.json")
    prod = work / arm / "products"
    digest = rec["mapping"]["content_digest"][:16]
    relay = {}
    for name, files in C.products(rec).items():
        local = {k: prod / k for k in files}
        nbytes = sum(p.stat().st_size for p in local.values())
        res = put({"files": {k: str(v) for k, v in local.items()}, "prefix": f"{C.relay_root()}/{arm}/init/{digest}",
                   "message": f"SOB1 {arm} {run_id}: {name}", "scratch": str(work)},
                  max(3600.0, nbytes / 2.6e6 * 1.25 + 600))
        relay[name] = res
        if not (res.get("verified") is True and res.get("files_sha256") == files and files):
            write_json(out / arm / "records" / "relay.json", relay)
            raise DriverError(f"{arm}: the relay copy of {name} is not its manifest {sorted(files)} verified: "
                              f"{str(res)[:600]}")
        status.mark(f"RELAY_OK:{arm}:{name}")
    write_json(out / arm / "records" / "relay.json", relay)
    doc = init_record(arm, rec, relay)
    bad = C.product_problems(doc, arm)
    if bad:
        raise DriverError(f"{arm}: the init record would not show its products durable: {bad}")
    write_json(out / arm / "init_record.json", doc)
    status.mark(f"INIT_RECORD_WRITTEN:{arm}")
    return doc


def durable_init(out: Path, arm: str) -> dict | None:
    """The arm's init record when it shows every product durable (`common.product_problems`), else None."""
    p = out / arm / "init_record.json"
    doc = read_json(p) if p.is_file() else None
    return doc if doc is not None and not C.product_problems(doc, arm) else None


# ---- evaluation -----------------------------------------------------------------------------------------------------

def model_for(target: str, out: Path, work: Path) -> Path:
    """The weights a target is evaluated from: the teacher's pinned snapshot, or the arm's checkpoint (on this pod,
    else fetched from its init record's relay copy and re-hashed)."""
    from shared.pod.teacher_cache import local_snapshot
    from stages.teacher_parent_initialization import relay as R
    d = C.design()
    if target == TEACHER:
        return Path(local_snapshot(d["teacher"]["model"], d["teacher"]["revision"]))
    doc = durable_init(out, target)
    if doc is None:
        raise DriverError(f"{target} has no durable initialization to evaluate")
    dest = work / target / "products" / "init"
    man = {k.removeprefix("init/"): v for k, v in C.products(doc)["init"].items()}
    here = lambda: all((dest / n).is_file() and sha256_file(dest / n) == h for n, h in man.items())  # noqa: E731
    if here():
        return dest
    rec = doc["relay"]["init"]
    flat = {**rec, "files_sha256": man, "prefix": f"{rec['prefix']}/init"}
    got = R.run_online("get", {"record": flat, "dest": str(dest)}, timeout_s=7200)
    if not got.get("verified") or not here():
        raise DriverError(f"{target}'s initialization did not restore as its manifest: {str(got)[:400]}")
    return dest


def evaluate_set(target: str, mode: str, out: Path, work: Path, vllm_py: str, attempt: str, status: Status) -> dict:
    """One set (one model x one mode) to completion, on its files (`completion.check`): a set an earlier attempt
    completed is taken as it is; otherwise the smallest repair (`completion.repair`: every sealed part whose
    generation's weights, mode or protocol cannot be shown isolated, a generations file holding a record no proven
    part holds superseded), the runners over only what no proven part holds (`reasoning_inheritance.driver.fluid` /
    `capability` with the registered protocol), the formal report rescored from the generations, and the check
    again. Each generation round has its own id (`completion.generation_id`: attempt, target, mode, round): its package
    record (weights, mode, protocol and the parts already sealed) is written before anything is generated, every part
    it seals carries that id and every record the runners' label (`completion.label_of`). Two
    rounds at most; a set still incomplete stops the session with its problems (its sealed parts stay for the next
    attempt)."""
    from stages.reasoning_inheritance import driver as RID
    from stages.single_operator_budget import completion as CP
    d = CP.ev_dir(out, target, mode)
    label = CP.label_of(target)
    st = CP.check(d, target, mode, out, label=label)
    row = {"at_start": {"complete": st["complete"], "problems": st["problems"]}}
    if st["complete"]:
        CP.write(d, st)
        return {**row, "status": "complete", "generated": False}
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    ev = C.protocol()
    for rnd in (1, 2):
        gen = CP.generation_id(attempt, target, mode, rnd)
        row.setdefault("repairs", []).append(CP.repair(d, st, gen))
        if CP.needs_generation(d):
            model_dir = model_for(target, out, work)
            pkg = work / "packages" / f"{label}_{mode}"
            entry = {"generation": gen, "attempt": attempt, "utc": utc(), "target": target, "mode": mode,
                     "protocol_sha256": CP.protocol_sha256(mode), "parts_at_start": CP.part_inventory(d),
                     **C.package(model_dir, pkg)}
            write_json(d / CP.PACKAGES, CP.packages(d) + [entry])   #: before anything is generated
            status.mark(f"EVAL_GENERATE:{target}:{mode}:{gen}")
            row.setdefault("runs", []).append({
                "generation": gen,
                "fluid": RID.fluid(pkg, d, label, mode, vllm_py, env, gen, None, ev),
                "capability": RID.capability(pkg, d, label, mode, vllm_py, env, gen, None, ev)})
        st = CP.check(d, target, mode, out, label=label)
        CP.rescore(d, st)                       #: a stale or {} report is replaced by these generations' own
        st = CP.check(d, target, mode, out, label=label)
        if st["complete"]:
            break
    CP.write(d, st)
    if not st["complete"]:
        raise DriverError(f"{target}/{mode} is not complete after {rnd} repair rounds: {st['problems']}")
    return {**row, "status": "complete", "generated": bool(row.get("runs"))}


def evaluate_targets(out: Path, work: Path, vllm_py: str, attempt: str, status: Status, report: dict) -> None:
    for target in phase()["evaluation_order"]:
        row = report["evaluations"].setdefault(target, {})
        for mode in C.mode_dirs():
            status.mark(f"EVAL_START:{target}:{mode}")
            t0 = time.time()
            row[mode] = {**evaluate_set(target, mode, out, work, vllm_py, attempt, status),
                         "eval_seconds": round(time.time() - t0, 1)}
            write_json(out / "report.json", report)
            status.mark(f"EVAL_DONE:{target}:{mode}")


# ---- the session ------------------------------------------------------------------------------------------------------

def contract(work: Path, device: str) -> dict:
    """$0: the session derives the journal's operator config hashes (TPI1's contract), and this round's selection
    record is what the journal and TPI1's selection give."""
    from stages.single_operator_budget import selection as SSEL
    from stages.teacher_parent_initialization import init_driver as TI
    base = TI.contract(work, device)
    fresh = SSEL.render(SSEL.extract()) if (REPO / read_json(C.SELECTION)["source"]["journal"]["path"]).is_file() else None
    return {**base, "sob1_design_sha256": sha256_file(C.DESIGN), "sob1_selection_sha256": sha256_file(C.SELECTION),
            "sob1_selection_reproduced": None if fresh is None else fresh == C.SELECTION.read_text(),
            "build_order": phase()["build_order"], "evaluation_order": phase()["evaluation_order"]}


def take_over(resume: Path, out: Path) -> dict:
    """A previous attempt's secured evidence into ``out`` (never a live stream); returns what was taken."""
    from stages.teacher_parent_initialization import train_driver as TD
    return TD.take_over(resume, out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", default="check")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--status", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--vllm-python", default="/opt/vllm/bin/python")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--phase", choices=("build",), default=None, help=argparse.SUPPRESS)
    ap.add_argument("--arm", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    out, work, status = Path(a.out), Path(a.work), Status(a.status)
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    if a.phase == "build":                      #: the GPU child, one arm
        build(a.arm, out, work, a.device, status, a.run_id)
        return 0
    report = {"schema": SCHEMA.replace("init_record", "phase1_report"), "run_id": a.run_id, "started_utc": utc(),
              "inits": {}, "evaluations": {}}
    ok = True
    try:
        if a.resume:
            report["resumed_from"] = take_over(Path(a.resume), out)
        report["contract"] = contract(work, "cpu" if a.check_only else a.device)
        if report["contract"].get("sob1_selection_reproduced") is False:
            raise DriverError("selection.json is not what the journal and TPI1's selection give")
        write_json(out / "report.json", report)
        status.mark("CONTRACT_OK")
        if a.check_only:
            return 0
        env = {**os.environ, "HF_HUB_OFFLINE": "1"}
        for arm in phase()["build_order"]:
            if durable_init(out, arm):
                report["inits"][arm] = {"resumed": "an earlier attempt's init record verified: nothing rebuilt"}
                continue
            rc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--phase", "build", "--arm", arm,
                                 "--run-id", a.run_id, "--out", str(out), "--work", str(work), "--status",
                                 str(status.path), "--device", a.device], env=env, cwd=REPO).returncode
            if rc != 0:
                raise DriverError(f"{arm}: the build phase exited {rc}; see {arm}/records/build.json")
            report["inits"][arm] = persist(arm, out, work, a.run_id, status)["commit"]
            write_json(out / "report.json", report)
        status.mark("INITS_DURABLE")
        evaluate_targets(out, work, a.vllm_python, a.run_id, status, report)
        from stages.single_operator_budget import completion as CP
        report["completion"] = CP.phase(out, write_records=True)
        write_json(out / "report.json", report)
        if not report["completion"]["complete"]:
            raise DriverError(f"phase 1 is not complete on its files: {report['completion']['n_complete']} of "
                              f"{report['completion']['n_sets']} sets, inits {report['completion']['inits']}, "
                              f"modes {report['completion']['modes']}")
        status.mark("PHASE1_COMPLETE")
        try:                                     #: a derived product: a failure here is recorded, re-derivable at $0
            from stages.single_operator_budget import analysis as SA
            table = SA.phase1_table(out, work / "comparison")
            write_json(out / "comparison" / "comparison.json", table)
            (out / "comparison" / "comparison.md").write_text(SA.render_table(table))
            report["comparison"] = {"json": "comparison/comparison.json", "markdown": "comparison/comparison.md"}
            status.mark("COMPARISON_WRITTEN")
        except Exception as exc:  # noqa: BLE001
            report["comparison"] = {"error": f"{type(exc).__name__}: {str(exc)[:1500]}",
                                    "rederive": "analysis.py phase1-table --evidence <the secured copy>"}
    except BaseException as exc:  # noqa: BLE001
        ok = False
        report["error"] = {"error": repr(exc)[:2000], "traceback": traceback.format_exc()[-6000:]}
    report.update(finished_utc=utc(), terminal="ALL_DONE" if ok else "RUN_FAILED")
    write_json(out / "report.json", report)
    status.mark("ALL_DONE" if ok else "RUN_FAILED")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
