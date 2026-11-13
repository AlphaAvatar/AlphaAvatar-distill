#!/usr/bin/env python3
"""On-pod driver of the Stage-1 reasoning-inheritance diagnosis: one frozen mapping, four untrained groups, state
evaluation and eval-set-1 in both modes. Every finished unit is written the moment it exists.

    cd /workspace/aad && HF_HUB_OFFLINE=1 PYTHONPATH=src:scripts:scripts/stages/stage-1 /opt/train/bin/python \\
        scripts/stages/stage-1/reasoning_inheritance/driver.py --run-id <run> --out <evidence dir> \\
        --work <scratch> --status <file> --device cuda:0 --vllm-python /opt/vllm/bin/python [--resume <dir>]

    # $0 on the dev box: the design, the frozen selection, and the packed session's operator config hashes against
    # the journal's (nothing is loaded)
    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python \\
        scripts/stages/stage-1/reasoning_inheritance/driver.py --check-only --out <s>/out --work <s>/work --status <s>/st

`design.json` is the experiment; `frozen_selection.json` is the anchor search's selections (`selection.py`). Phases,
each its own child process so the GPU is released between them:

* ``mapping`` -- the teacher (every shard hashed); the FFN child rebuilt from the frozen FFN selection and the FFN ->
  DEPTH parent from the kept blocks, each compared with the journal's artifact digest; ``width.global_pca_v0`` and
  ``attention.activation_importance_v1`` replayed on that parent through the fixed-path executor under the search's
  own session (calibration, packed forward context, position policy, Top-200 support; each step's config hash pinned
  to the journal's), the width operator's statistics kept (`RetainingStatsCache`) and ``P`` recomputed from them; the
  width child rebuilt from ``P`` and compared with the operator's bit for bit; the attention operator run once more on
  the 36-block parent with the frozen FFN selection and ``P`` to select the eight removed blocks' heads. The mapping is
  saved, D, W and DW materialized from the saved copy, checked (parameters, D's blocks are T's, W's kept blocks and
  shared tensors are DW's) and copied to the private relay, downloaded again and re-hashed.
* ``state_eval`` -- KL and NLL against the root teacher on ``state_eval_hybrid@v3`` (the search's evaluator).
* ``evaluate`` -- eval-set-1, every unit, for each (group, mode) of ``design.evaluation.order`` at 40,960 tokens of
  context with ``nucleus_v1``; the fluid unit streams each finished episode into ``live/`` and seals complete lines into
  immutable parts, so a later attempt generates only what is missing; a capability unit is sealed when the runner
  moves past it. T is never generated: its evaluation is reused from ``a_teacher_20261030_012648`` on the dev box.

``--resume <dir>`` is a previous attempt's secured evidence: its mapping, checkpoint records, state evaluation and
sealed evaluation parts are taken over (the checkpoints themselves are downloaded from the relay and re-hashed),
recording which attempt produced each. Nothing is regenerated that a sealed part holds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

DESIGN = REPO / "configs/stages/stage-1/reasoning_inheritance/design.json"
SELECTION = REPO / "configs/stages/stage-1/reasoning_inheritance/frozen_selection.json"
SCHEMA = "aadistill.stage1.reasoning_inheritance.driver_report/v1"
GROUPS = ("D", "W", "DW")
TOKENIZER_SIDECARS = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json")
POD_TOKEN = Path("/workspace/hf/token")
SEAL_EVERY_S = 120


class DriverError(RuntimeError):
    pass


# ---- small utilities ----------------------------------------------------------------------------------------------

def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    """Atomic: a reader (the poll) never sees a half-written file; '.'-prefixed temporaries are never pulled."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.partial")
    tmp.write_text(json.dumps(obj, indent=1, default=str) + "\n")
    os.replace(tmp, path)


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def design() -> dict:
    return read_json(DESIGN)


def selection() -> dict:
    return read_json(SELECTION)


class Status:
    def __init__(self, path: Path):
        self.path = Path(path)

    def mark(self, m: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(f"{utc()} MARKER:{m}\n")
        print(f"MARKER:{m}", flush=True)


# ---- the contract ($0) ----------------------------------------------------------------------------------------------

def session(work: Path, device: str):
    """The anchor search's own session: profiles, suite, forward context, position policy, support, execution."""
    from stages.teacher_modes import packed_full_search as PFS
    d = design()
    which = d["inputs"]["inputs_key"]
    s = PFS.build_session(workdir=work / "search_session", device=device, which=which, repo_root=REPO)
    bad = PFS.input_problems(s, which)
    if bad:
        raise DriverError("the inputs are not their frozen identity: " + "; ".join(bad))
    return s


def anchor_spec(s, sel: dict, device: str, *, n_blocks: int | None = None, path_id: str | None = None):
    """The anchor path as a fixed path, every step's config hash pinned to the journal's (no digest pins: the driver
    compares digests itself and records a mismatch instead of stopping on it). ``n_blocks`` gives a target of another
    depth (W's attention step on 36 blocks)."""
    from aadistill.initialization.planning.fixed_path import FixedPathSpec, FixedPathStep
    t = s.config.target_spec
    target = t if n_blocks is None else t.replace(num_hidden_layers=int(n_blocks))
    steps = tuple(FixedPathStep(impl_id=st["impl_id"], profile_id=st["profile_id"], label=st["kind"],
                                expected_config_hash=st["config_hash"]) for st in sel["steps"])
    if n_blocks is not None:
        steps = tuple(x for x in steps if x.label != "DEPTH")
    root = design()["teacher"]
    return FixedPathSpec(path_id=path_id or "stage1.reasoning_inheritance.anchor", family=target.family,
                         target_spec=target, steps=steps, root_repo_id=root["model"], root_revision=root["revision"],
                         device=device, seed=int(sel["source"]["seed"]), position_policy=s.config.position_policy,
                         distribution_support=s.config.distribution_support, forward_context=s.config.forward_context)


def derived_config_hashes(spec) -> list[dict]:
    """Each step's operator config hash as the fixed-path executor derives it, beside the journal's ($0)."""
    from aadistill.initialization.calibration.items import prepare_calibration_items
    from aadistill.initialization.calibration.profiles import get_profile, profile_for
    from aadistill.initialization.operators.base import get_implementation
    from aadistill.initialization.planning.operator_config import hashed_operator_config, operator_config_hash
    out, items_of = [], {}
    for st in spec.steps:
        impl = get_implementation(st.impl_id)
        prof = profile_for(impl, get_profile(st.profile_id))
        if prof.qualified_id not in items_of:
            items_of[prof.qualified_id] = prepare_calibration_items(prof.resolve(str(REPO)),
                                                                    profile_id=prof.qualified_id)
        cfg = hashed_operator_config(implementation=impl, policy=spec.position_policy,
                                     support=spec.distribution_support, items=items_of[prof.qualified_id],
                                     declared=st.config, objective=spec.scoring_objective,
                                     forward_context=spec.forward_context)
        got = operator_config_hash(cfg)
        out.append({"step": st.label, "impl_id": st.impl_id, "profile_id": st.profile_id, "derived": got,
                    "journal": st.expected_config_hash, "matches": got == st.expected_config_hash})
    return out


def contract(work: Path, device: str) -> dict:
    d, sel = design(), selection()
    from stages.reasoning_inheritance import selection as SEL
    s = session(work, device)
    spec = anchor_spec(s, sel, device)
    hashes = derived_config_hashes(spec)
    bad = [h for h in hashes if not h["matches"]]
    if bad:
        raise DriverError(f"the session derives other operator configs than the search recorded: {bad}")
    want = SEL.render(SEL.extract()) if (REPO / d["anchor"]["journal"]["path"]).is_file() else None
    return {"design_sha256": sha256_file(DESIGN), "selection_sha256": sha256_file(SELECTION),
            "selection_reproduced_from_journal": (None if want is None else want == SELECTION.read_text()),
            "session": {"config_hash": s.config.config_hash, "measurement_protocol_id": s.evaluator.measurement_protocol_id,
                        "forward_context": s.config.forward_context.as_dict(),
                        "profiles": {p.qualified_id: p.profile_hash for p in s.config.profiles}},
            "fixed_path_spec_hash": spec.spec_hash, "config_hashes": hashes}


# ---- the mapping (GPU) ----------------------------------------------------------------------------------------------

def _identify(adapter, ckpt: Path):
    from aadistill.initialization.specs.artifact import identify_checkpoint
    from transformers import AutoConfig
    spec = adapter.spec_from_config(AutoConfig.from_pretrained(str(ckpt)))
    return identify_checkpoint(ckpt, adapter=adapter, spec=spec, num_parameters=adapter.param_count(spec))


def _save(adapter, model, ckpt: Path):
    shutil.rmtree(ckpt, ignore_errors=True)
    ckpt.mkdir(parents=True)
    adapter.save(model, str(ckpt))
    return _identify(adapter, ckpt)


def _ident_record(ident, journal: dict | None = None) -> dict:
    rec = {"artifact_digest": ident.artifact_digest, "config_sha256": ident.config_sha256,
           "shards": [{"filename": s.filename, "sha256": s.sha256, "size_bytes": s.size_bytes} for s in ident.shards],
           "num_parameters": ident.num_parameters, "arch_signature": ident.arch_signature}
    if journal is not None:
        rec["journal"] = {k: journal[k] for k in ("state_id", "artifact_digest", "single_shard_sha256", "config_sha256")}
        rec["artifact_digest_matches"] = ident.artifact_digest == journal["artifact_digest"]
        rec["weights_match"] = (len(ident.shards) == 1 and ident.shards[0].sha256 == journal["single_shard_sha256"])
        rec["config_matches"] = ident.config_sha256 == journal["config_sha256"]
    return rec


def _gpu_peak(device: str) -> dict:
    import torch
    if not torch.cuda.is_available():
        return {}
    return {"max_allocated_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 2),
            "max_reserved_gib": round(torch.cuda.max_memory_reserved(device) / 2**30, 2)}


def build_mapping(out: Path, work: Path, device: str, status: Status) -> dict:
    """The mapping phase (module docstring). Returns the mapping record; writes ``out/records/mapping.json``."""
    import torch
    from aadistill.initialization.planning.fixed_path import _run_steps
    from aadistill.initialization.specs.arch import get_adapter
    from aadistill.initialization.planning import frozen_mapping as FM
    from stages.teacher_modes import full_search as FS
    d, sel = design(), selection()
    s = session(work, device)
    adapter = get_adapter(s.config.target_spec.family)
    spec = anchor_spec(s, sel, device)
    states = sel["states"]
    rec: dict = {"started_utc": utc(), "anchor": d["anchor"], "selection_sha256": sha256_file(SELECTION)}
    t0 = time.time()
    teacher, rec["root"] = FS.load_root(device)
    rec["seconds_teacher"] = round(time.time() - t0, 1)
    n_root, kept = int(sel["n_root_blocks"]), [int(b) for b in sel["kept_blocks"]]
    removed = [int(b) for b in sel["removed_blocks"]]
    seed = int(sel["source"]["seed"])
    rp = work / "replay"

    # 1. the FFN child and the FFN -> DEPTH parent, from the frozen selections, against the journal
    ffn = FM.select_ffn_neurons(adapter, teacher, [sel["ffn_neurons"][str(b)] for b in range(n_root)], seed=seed)
    rec["ffn_child"] = _ident_record(_save(adapter, ffn, rp / "00_ffn"), states[0])
    del ffn
    ffn = adapter.load(str(rp / "00_ffn"), device=device)       #: the search expanded the checkpoint on disk
    parent = FM.keep_blocks(adapter, ffn, kept, seed=seed)
    parent_ident = _save(adapter, parent, rp / "01_depth")
    rec["depth_parent"] = _ident_record(parent_ident, states[1])
    del parent
    parent = adapter.load(str(rp / "01_depth"), device=device)
    write_json(out / "records" / "mapping_progress.json", rec)
    status.mark("PARENT_REBUILT")

    # 2. the width and attention operators replayed on that parent, the width statistics kept
    cache = FM.RetainingStatsCache()
    t1 = time.time()
    results = _run_steps(spec, adapter=adapter, model=parent, indices=range(2, 4), parent_spec=adapter.spec_of(parent),
                         parent_digest=parent_ident.artifact_digest, workdir=rp / "fixed_path", repo_root=str(REPO),
                         stats_cache=cache, calibration_items=None, on_step=None, deadline=None,
                         execution=s.execution)
    rec["replay_seconds"] = round(time.time() - t1, 1)
    w_res, a_res = results
    rec["width_child"] = {**_ident_record(w_res.identity, states[2]), "trace": dict(w_res.trace),
                          "local_metrics": dict(w_res.local_metrics)}
    rec["attention_child"] = {**_ident_record(a_res.identity, states[3]),
                              "kept_heads_equal_journal": (a_res.selection.get("kept_heads") ==
                                                           [sel["query_heads_kept_blocks"][str(b)] for b in kept])}
    if len(cache.retained) != 1:
        raise DriverError(f"the width replay left {len(cache.retained)} statistics states, not one")
    state = next(iter(cache.retained.values()))
    wr = sel["width_recorded"]
    n_parent = len(kept)
    transform = FM.width_transform_from_statistics(
        state, d_child=int(wr["d_child"]), points=list(range(n_parent + 1)), point_weights=wr["point_weights"],
        final_norm_weight=adapter.final_norm(parent).weight, post_norm_point=n_parent, device=device)
    rec["width_transform"] = {"recomputed": dict(transform.diagnostics), "scale": transform.scale,
                              "recorded": {k: wr[k] for k in ("energy_captured_frac", "top_eigenvalue",
                                                              "min_kept_eigenvalue", "scale_compensation")}}
    rec["width_transform"]["diagnostics_equal_recorded"] = all(
        transform.diagnostics.get(k) == wr[k] for k in ("energy_captured_frac", "top_eigenvalue", "min_kept_eigenvalue"))
    width_child = FM.project_residual_width(adapter, parent, transform, seed=seed)
    replayed = adapter.load(w_res.checkpoint_path, device=device)
    diff = FM.block_differences(adapter, width_child, replayed, [(i, i) for i in range(n_parent)])
    diff += FM.shared_parameter_differences(adapter, width_child, replayed)
    rec["width_builder_equals_operator"] = not diff
    if diff:
        raise DriverError(f"the width builder does not write the width operator's child: {diff[:6]}")
    del replayed, parent, state
    cache.clear()
    cache.retained.clear()

    # 3. W before attention (36 blocks: frozen FFN selection and P), and the attention operator on it
    w_pre = FM.project_residual_width(adapter, ffn, transform, seed=seed)
    w_pre_ident = _save(adapter, w_pre, rp / "w_pre_attention")
    rec["w_pre_attention"] = _ident_record(w_pre_ident)
    del w_pre
    w_pre = adapter.load(str(rp / "w_pre_attention"), device=device)
    w_spec = anchor_spec(s, sel, device, n_blocks=n_root, path_id="stage1.reasoning_inheritance.w_attention")
    t2 = time.time()
    (w_attn,) = _run_steps(w_spec, adapter=adapter, model=w_pre, indices=[2], parent_spec=adapter.spec_of(w_pre),
                           parent_digest=w_pre_ident.artifact_digest, workdir=rp / "w_fixed_path", repo_root=str(REPO),
                           stats_cache=None, calibration_items=None, on_step=None, deadline=None,
                           execution=s.execution)
    rec["w_attention_seconds"] = round(time.time() - t2, 1)
    heads36 = w_attn.selection.get("kept_heads")
    if not heads36 or len(heads36) != n_root:
        raise DriverError("the attention operator on the 36-block parent returned no full head selection")
    recorded = {int(b): tuple(v) for b, v in sel["query_heads_kept_blocks"].items()}
    rec["removed_blocks_heads"] = {
        "rule": d["mapping"]["query_heads_removed_blocks"],
        "selected": {str(b): heads36[b] for b in removed},
        "scores": {str(b): w_attn.trace.get("head_scores_per_layer", [None] * n_root)[b] for b in removed},
        "margins": {str(b): w_attn.trace.get("selection_margin_per_layer", [None] * n_root)[b] for b in removed},
        "kept_blocks_agreement_on_36_block_parent": {
            "blocks_identical": sum(tuple(heads36[b]) == recorded[b] for b in kept),
            "slots_shared": sum(len(set(heads36[b]) & set(recorded[b])) for b in kept),
            "slots": sum(len(recorded[b]) for b in kept),
            "note": "diagnostic only: the kept blocks keep the journal's selection"}}
    del w_pre

    # 4. the mapping, saved; the groups materialized from the SAVED copy
    root_rec = {"model": d["teacher"]["model"], "revision": d["teacher"]["revision"],
                "root_teacher_sha256": rec["root"]["root_teacher_sha256"]}
    reproduced = bool(rec["depth_parent"]["artifact_digest_matches"] and rec["width_child"]["artifact_digest_matches"])
    mapping = FM.FrozenMapping(
        root=root_rec, n_root_blocks=n_root, kept_blocks=tuple(kept),
        ffn_neurons={b: tuple(sel["ffn_neurons"][str(b)]) for b in range(n_root)},
        query_heads={**{b: tuple(recorded[b]) for b in kept}, **{b: tuple(heads36[b]) for b in removed}},
        width=transform,
        provenance={"ffn_neurons": f"journal {d['anchor']['journal']['sha256'][:12]} FFN step kept_neurons",
                    "kept_blocks": "journal DEPTH step kept_layers",
                    "query_heads_kept_blocks": "journal ATTENTION step kept_heads, child -> root numbering",
                    "query_heads_removed_blocks": d["mapping"]["query_heads_removed_blocks"],
                    "width": ("width.global_pca_v0's own statistics on the rebuilt FFN -> DEPTH parent, "
                              f"parent digest {parent_ident.artifact_digest[:12]}"),
                    "historical_parent_reproduced": rec["depth_parent"]["artifact_digest_matches"],
                    "historical_width_child_reproduced": rec["width_child"]["artifact_digest_matches"],
                    "selection_sha256": sha256_file(SELECTION)})
    bad = FM.problems(mapping, root_spec=adapter.spec_of(teacher))
    if bad:
        raise DriverError(f"the mapping is malformed: {bad}")
    saved = FM.save(mapping, out / "mapping")
    mapping = FM.load(out / "mapping", expected_content_digest=saved["content_digest"])
    rec["mapping"] = {"content_digest": saved["content_digest"], "tensor_file": saved.get("tensor_file"),
                      "historical_top1_exact_reproduction": None}
    write_json(out / "records" / "mapping_progress.json", rec)
    status.mark("MAPPING_SAVED")
    del ffn
    torch.cuda.empty_cache()

    groups = {}
    built = {}
    for g in GROUPS:
        spec_g = d["groups"][g]
        model, roots = FM.materialize(adapter, teacher, mapping, blocks=spec_g["blocks"], non_depth=spec_g["non_depth"],
                                      seed=seed)
        ident = _save(adapter, model, work / "groups" / g)
        geo = {k: int(adapter.spec_of(model)[k]) for k in spec_g["geometry"]}
        n_params = sum(p.numel() for p in model.parameters())
        groups[g] = {**_ident_record(ident, states[3] if g == "DW" else None), "root_blocks": roots,
                     "geometry": geo, "geometry_matches_design": geo == spec_g["geometry"],
                     "num_parameters_counted": int(n_params),
                     "num_parameters_match_design": int(n_params) == int(spec_g["num_parameters"])}
        if not (groups[g]["geometry_matches_design"] and groups[g]["num_parameters_match_design"]):
            raise DriverError(f"group {g}: {geo}, {n_params} parameters, not the design's")
        built[g] = model
        if g == "D":
            groups[g]["blocks_equal_teacher"] = not FM.block_differences(
                adapter, teacher, model, [(r, i) for i, r in enumerate(roots)])
            groups[g]["shared_equal_teacher"] = not FM.shared_parameter_differences(adapter, teacher, model)
            del built["D"], model
            torch.cuda.empty_cache()
    w, dw = built["W"], built["DW"]
    pairs = [(groups["W"]["root_blocks"].index(r), i) for i, r in enumerate(groups["DW"]["root_blocks"])]
    groups["W"]["kept_blocks_equal_DW"] = not FM.block_differences(adapter, w, dw, pairs)
    groups["W"]["shared_equal_DW"] = not FM.shared_parameter_differences(adapter, w, dw)
    dw_attn = adapter.load(a_res.checkpoint_path, device=device)
    groups["DW"]["equals_attention_replay"] = not (
        FM.block_differences(adapter, dw, dw_attn, [(i, i) for i in range(len(kept))])
        + FM.shared_parameter_differences(adapter, dw, dw_attn))
    del w, dw, built, dw_attn
    checks = {"D_blocks_are_T": groups["D"]["blocks_equal_teacher"], "D_shared_are_T": groups["D"]["shared_equal_teacher"],
              "W_kept_blocks_are_DW": groups["W"]["kept_blocks_equal_DW"], "W_shared_are_DW": groups["W"]["shared_equal_DW"]}
    if not all(checks.values()):
        raise DriverError(f"the groups do not share the mapping as designed: {checks}")
    rec["groups"] = groups
    rec["pairing_checks"] = checks
    exact = bool(reproduced and groups["DW"].get("artifact_digest_matches"))
    rec["mapping"]["historical_top1_exact_reproduction"] = exact
    rec["mapping"]["statement"] = (
        "historical Top-1 exact reproduction established: the rebuilt parent, width child and DW have the journal's "
        "artifact digests" if exact else
        "historical Top-1 exact reproduction NOT established; the snapshot has its own identity and W and DW both "
        "come from it")
    rec["gpu_peak"] = _gpu_peak(device)
    rec["finished_utc"] = utc()
    write_json(out / "records" / "mapping.json", rec)
    status.mark("GROUPS_BUILT")
    return rec


# ---- persistence of the heavy products (relay) ----------------------------------------------------------------------

def relay_upload(work: Path, out: Path, run_id: str) -> dict:
    """The mapping and the three checkpoints to the private relay, then downloaded again into a fresh cache and
    re-hashed. Runs with the hub online (its own process: ``HF_HUB_OFFLINE`` is read at import)."""
    from huggingface_hub import HfApi, hf_hub_download
    import tempfile
    d = design()
    repo, prefix = d["products"]["relay_repo"], d["products"]["relay_prefix"]
    mrec = read_json(out / "records" / "mapping.json")
    digest = mrec["mapping"]["content_digest"]
    token = POD_TOKEN.read_text().strip() if POD_TOKEN.is_file() else None
    api = HfApi(token=token)
    files = {f"mapping/{n}": out / "mapping" / n for n in ("mapping.json", "tensors.safetensors")}
    for g in GROUPS:
        for p in sorted((work / "groups" / g).iterdir()):
            if p.is_file():
                files[f"groups/{g}/{p.name}"] = p
    base = f"{prefix}/{digest[:16]}"
    want = {k: sha256_file(p) for k, p in files.items()}
    t0 = time.time()
    for k, p in sorted(files.items()):
        api.upload_file(path_or_fileobj=str(p), path_in_repo=f"{base}/{k}", repo_id=repo, repo_type="model",
                        commit_message=f"stage1 reasoning inheritance {run_id}: {k}")
    t1, got = time.time(), {}
    with tempfile.TemporaryDirectory(dir=str(work)) as tmp:
        for k in sorted(files):
            got[k] = sha256_file(Path(hf_hub_download(repo, f"{base}/{k}", repo_type="model", token=token,
                                                      cache_dir=f"{tmp}/hf", force_download=True)))
    return {"relay": repo, "prefix": base, "files_sha256": want, "verified": got == want,
            "bytes": {k: p.stat().st_size for k, p in files.items()},
            "upload_seconds": round(t1 - t0, 1), "verify_seconds": round(time.time() - t1, 1),
            "producer_run": run_id, "utc": utc(),
            "rule": "uploaded from the pod's copy, downloaded again into a fresh cache and re-hashed"}


def relay_restore(work: Path, products: dict) -> dict:
    """A previous attempt's checkpoints from the relay, every file re-hashed against its record."""
    from huggingface_hub import hf_hub_download
    repo, base, out = products["relay"], products["prefix"], {}
    token = POD_TOKEN.read_text().strip() if POD_TOKEN.is_file() else None
    for k, want in sorted(products["files_sha256"].items()):
        if not k.startswith("groups/"):
            continue
        dest = work / k
        dest.parent.mkdir(parents=True, exist_ok=True)
        src = Path(hf_hub_download(repo, f"{base}/{k}", repo_type="model", token=token, cache_dir=str(work / "hf_relay")))
        shutil.copyfile(src, dest)
        got = sha256_file(dest)
        if got != want:
            raise DriverError(f"{k} from the relay hashes to {got[:12]}, the record says {want[:12]}")
        out[k] = got
    shutil.rmtree(work / "hf_relay", ignore_errors=True)
    return {"restored": sorted(out), "from": f"{repo}:{base}", "utc": utc()}


# ---- state evaluation (GPU) ----------------------------------------------------------------------------------------

def state_eval(out: Path, work: Path, device: str, status: Status) -> dict:
    from aadistill.initialization.specs.arch import get_adapter
    from stages.teacher_modes import full_search as FS
    s = session(work, device)
    adapter = get_adapter(s.config.target_spec.family)
    teacher, root = FS.load_root(device)
    t0 = time.time()
    s.evaluator.prime_reference(teacher)
    rec = {"started_utc": utc(), "measurement_protocol_id": s.evaluator.measurement_protocol_id,
           "suite": s.suite_manifest.get("suite_id"), "suite_content_sha256": s.suite_manifest.get("content_sha256"),
           "forward_context": s.config.forward_context.as_dict(), "prime_seconds": round(time.time() - t0, 1),
           "groups": {}}
    mrec = read_json(out / "records" / "mapping.json")
    for g in GROUPS:
        ident = _identify(adapter, work / "groups" / g)
        if ident.artifact_digest != mrec["groups"][g]["artifact_digest"]:
            raise DriverError(f"group {g}'s checkpoint is not the recorded one")
        model = adapter.load(str(work / "groups" / g), device=device)
        model.config.use_cache = False
        t1 = time.time()
        ev = s.evaluator.evaluate(model, ident.artifact_digest)
        rec["groups"][g] = {"artifact_digest": ident.artifact_digest, "seconds": round(time.time() - t1, 1),
                            "values": dict(getattr(ev, "values", {}) or {}), "positions": getattr(ev, "positions", None),
                            "detail": json.loads(json.dumps(getattr(ev, "detail", {}) or {}, default=str))}
        del model
        write_json(out / "records" / "state_eval.json", rec)
    rec["finished_utc"] = utc()
    write_json(out / "records" / "state_eval.json", rec)
    status.mark("STATE_EVAL_OK")
    return rec


# ---- eval-set-1 (vLLM) ---------------------------------------------------------------------------------------------

def package(model_dir: Path, dest: Path) -> dict:
    """The group's weights and config with the TEACHER's tokenizer files and generation config (the lineage's
    tokenizer, chat template and stop ids), every sidecar hashed against the teacher identity."""
    from shared.pod.teacher_cache import local_snapshot
    d = design()
    ident = read_json(REPO / d["teacher"]["identity"])
    snap = Path(local_snapshot(d["teacher"]["model"], d["teacher"]["revision"]))
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    for p in sorted(model_dir.iterdir()):
        if p.is_file() and p.name not in TOKENIZER_SIDECARS:
            os.symlink(p.resolve(), dest / p.name)
    side = {}
    for n in TOKENIZER_SIDECARS:
        got = sha256_file(snap / n)
        if got != ident["files_sha256"][n]:
            raise DriverError(f"the teacher's {n} hashes to {got[:12]}, the identity pins {ident['files_sha256'][n][:12]}")
        shutil.copyfile(snap / n, dest / n)
        side[n] = got
    return {"model_dir": str(model_dir), "sidecars": side, "teacher_snapshot": str(snap)}


class Sealer(threading.Thread):
    """Seals the complete lines a stream has gained into immutable numbered parts (``parts/part_NNNN.jsonl``)."""

    def __init__(self, stream: Path, parts: Path, every: float = SEAL_EVERY_S):
        super().__init__(daemon=True)
        self.stream, self.parts, self.every = stream, parts, every
        self.stop = threading.Event()
        self.offset = 0

    def seal(self) -> int:
        if not self.stream.is_file():
            return 0
        data = self.stream.read_bytes()[self.offset:]
        cut = data.rfind(b"\n")
        if cut < 0:
            return 0
        chunk = data[:cut + 1]
        self.parts.mkdir(parents=True, exist_ok=True)
        n = len(list(self.parts.glob("part_*.jsonl")))
        tmp = self.parts / f".part_{n:04d}.partial"
        tmp.write_bytes(chunk)
        os.replace(tmp, self.parts / f"part_{n:04d}.jsonl")
        self.offset += len(chunk)
        return chunk.count(b"\n")

    def run(self):
        while not self.stop.wait(self.every):
            try:
                self.seal()
            except OSError:
                pass

    def close(self):
        self.stop.set()
        self.join(timeout=30)
        self.seal()


def sealed_records(unit_dir: Path) -> dict[str, dict]:
    """Every finished episode a sealed part holds, by task id (a task sealed twice is a defect, refused)."""
    out: dict[str, dict] = {}
    for p in sorted((unit_dir / "parts").glob("part_*.jsonl")):
        for line in p.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                r = json.loads(line)
                if r["task_id"] in out:
                    raise DriverError(f"{r['task_id']} is sealed twice in {unit_dir}")
                out[r["task_id"]] = r
    return out


def _run_bounded(cmd: list[str], log: Path, env: dict, time_box_s: float | None, poll=None) -> tuple[int, bool]:
    """``cmd`` in its own process group, its output in ``log``; past ``time_box_s`` the whole group is stopped.
    Returns (exit code, stopped by the time box). ``poll()`` runs every 30 s while it lives."""
    import signal
    t0 = time.time()
    with log.open("w") as lf:
        proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        boxed = False
        while proc.poll() is None:
            time.sleep(30)
            if poll is not None:
                poll()
            if time_box_s and time.time() - t0 > time_box_s and proc.poll() is None:
                boxed = True
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
    return proc.returncode, boxed


def exec_args(opts: dict) -> list[str]:
    """The runner flags of the execution knobs (never the protocol's)."""
    return ((["--max-num-seqs", str(opts["max_num_seqs"])] if opts.get("max_num_seqs") else []) +
            (["--engine-metrics"] if opts.get("engine_metrics") else []))


def fluid(pkg: Path, unit_dir: Path, label: str, mode: str, vllm_py: str, env: dict, attempt: str,
          opts: dict | None = None) -> dict:
    """fluid_reasoning_720 with streaming and resume (module docstring). Returns the unit's record. ``opts``: the
    execution knobs (``max_num_seqs``, ``engine_metrics``) and ``time_box_s``: a run stopped by its time box keeps
    every sealed episode and leaves the unit incomplete, to be resumed."""
    opts = opts or {}
    from shared import eval_sets as ES
    d = design()
    ev = d["evaluation"]
    tasks_path = ES.unit_path(ev["eval_set"], "fluid_reasoning_720", "dev_v1")
    #: split on "\n" only, as run_battery.read_tasks does (an item may carry a raw U+2028)
    tasks = [json.loads(l) for l in tasks_path.read_text(encoding="utf-8").split("\n") if l.strip()]
    key = "fluid_reasoning_720__dev_v1"
    final = unit_dir / "generations" / f"{key}.jsonl"
    if final.is_file():
        return {"status": "complete", "restored": True}
    tid = lambda t: t["task_id"]  # noqa: E731
    done = sealed_records(unit_dir)
    todo = [t for t in tasks if tid(t) not in done]
    if opts.get("shuffle_submission"):
        #: the ORDER requests reach the engine (which it starts first under a concurrency limit), never an item's
        #: seed: each item's seed is derived from its own id. A fixed key, so the order is reproducible.
        todo.sort(key=lambda t: hashlib.sha256(f"{opts['shuffle_submission']}|{tid(t)}".encode()).hexdigest())
    runs = read_json(unit_dir / "fluid_runs.json") if (unit_dir / "fluid_runs.json").is_file() else []
    if todo:
        live = unit_dir.parent.parent / "live" / f"{unit_dir.name}.{attempt}.stream.jsonl"
        live.parent.mkdir(parents=True, exist_ok=True)
        live.unlink(missing_ok=True)
        sub = unit_dir.parent.parent / "live" / f"{unit_dir.name}.{attempt}.tasks.jsonl"
        sub.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in todo))
        tmp_out = unit_dir.parent.parent / "live" / f"{unit_dir.name}.{attempt}.out.jsonl"
        sealer = Sealer(live, unit_dir / "parts")
        sealer.start()
        t0 = time.time()
        log = unit_dir / "logs" / f"eval_fluid.{attempt}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        rc, boxed = _run_bounded(
            [vllm_py, "scripts/stages/stage-0/calibration_volume/run_battery.py", "--model", str(pkg),
             "--label", label, "--tasks", str(sub), "--mode", "eval", "--out", str(tmp_out),
             "--trained-context", str(ev["context"]["tokens"]), "--action-protocol", ev["action_protocol"],
             "--decoding", str(REPO / ev["decoding"]), "--seed-namespace", key,
             "--check-every", str(ev["degeneration_check_every"]), "--system", ev["system_message"],
             "--chat-template-kwargs", json.dumps(ev["modes"][mode]["template_kwargs"], sort_keys=True),
             "--stream-completed", str(live), *exec_args(opts)], log, env, opts.get("time_box_s"))
        sealer.close()
        summary = tmp_out.with_suffix(".summary.json")
        done = sealed_records(unit_dir)
        runs.append({"attempt": attempt, "rc": rc, "time_boxed": boxed, "tasks_requested": len(todo),
                     "sealed_after": len(done), "seconds": round(time.time() - t0, 1),
                     "execution": {k: opts.get(k) for k in ("max_num_seqs", "engine_metrics", "time_box_s",
                                                             "shuffle_submission")},
                     "submitted_order": [tid(t) for t in todo],
                     "summary": read_json(summary) if summary.is_file() else None,
                     "log": str(log.relative_to(unit_dir))})
        write_json(unit_dir / "fluid_runs.json", runs)
        if boxed:
            return {"status": "incomplete", "why": "time box", "sealed": len(done), "missing": len(tasks) - len(done)}
        if rc != 0:
            raise DriverError(f"run_battery exited {rc} for {label}; see {log.name}")
    missing = [tid(t) for t in tasks if tid(t) not in done]
    if missing:
        raise DriverError(f"{label}: {len(missing)} fluid tasks have no sealed record after the run")
    gens = unit_dir / "generations"
    gens.mkdir(parents=True, exist_ok=True)
    tmp = gens / f".{key}.partial"
    tmp.write_text("".join(json.dumps(done[tid(t)], ensure_ascii=False) + "\n" for t in tasks))
    os.replace(tmp, final)
    write_json(gens / f"{key}.summary.json", {
        "composition": "the union of the sealed parts in the unit's task order (fluid_resume)",
        "runs": runs, "n_tasks": len(tasks), "tasks_sha256": sha256_file(tasks_path),
        "ends": {k: sum(1 for r in done.values() if r["end"] == k) for k in sorted({r["end"] for r in done.values()})}})
    return {"status": "complete", "runs": len(runs)}


def capability(pkg: Path, unit_dir: Path, label: str, mode: str, vllm_py: str, env: dict, attempt: str,
               opts: dict | None = None) -> dict:
    """Every stage3_capability_846 unit (TriviaQA from its verified local copy) and stage3_legacy_150, one runner call;
    a unit's files are sealed into ``generations/`` when the runner has moved past it."""
    from shared import eval_sets as ES
    from stages.calibration_volume.dose_driver import legacy_prompts
    d = design()
    ev = d["evaluation"]
    m = ES.manifest(ev["eval_set"])
    gens = unit_dir / "generations"
    gens.mkdir(parents=True, exist_ok=True)
    todo, withheld = {}, []
    for u in m["components"]["stage3_capability_846"]["units"]:
        p = ES.unit_path(ev["eval_set"], "stage3_capability_846", u)
        if p is None:
            withheld.append(f"stage3_capability_846/{u}")
            continue
        todo[f"stage3_capability_846__{u}"] = p
    todo["stage3_legacy_150__items"] = legacy_prompts(unit_dir / "legacy_150.prompts.jsonl")
    missing = {k: p for k, p in todo.items() if not (gens / f"{k}.jsonl").is_file()}
    rec = {"withheld": withheld, "units": sorted(todo), "missing_at_start": sorted(missing)}
    if not missing:
        return {**rec, "status": "complete"}
    staged = unit_dir.parent.parent / "live" / f"{unit_dir.name}.{attempt}.cap_in"
    raw = unit_dir.parent.parent / "live" / f"{unit_dir.name}.{attempt}.cap_out"
    shutil.rmtree(staged, ignore_errors=True)
    shutil.rmtree(raw, ignore_errors=True)
    staged.mkdir(parents=True)
    order = sorted(missing)
    for k in order:
        shutil.copyfile(missing[k], staged / f"{k}.jsonl")
    log = unit_dir / "logs" / f"eval_capability.{attempt}.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    def seal(k: str) -> None:
        if (gens / f"{k}.jsonl").is_file():
            return
        shutil.copyfile(raw / f"{k}.json", gens / f".{k}.summary.partial")
        os.replace(gens / f".{k}.summary.partial", gens / f"{k}.summary.json")
        shutil.copyfile(raw / f"{k}.generations.jsonl", gens / f".{k}.partial")
        os.replace(gens / f".{k}.partial", gens / f"{k}.jsonl")

    opts = opts or {}

    def seal_finished() -> None:
        #: unit i is finished once unit i+1's summary exists (the runner writes units in order)
        for i, k in enumerate(order[:-1]):
            if (raw / f"{order[i + 1]}.json").is_file() and (raw / f"{k}.generations.jsonl").is_file():
                seal(k)

    t0 = time.time()
    rc, boxed = _run_bounded(
        [vllm_py, "scripts/shared/evaluation/uncapped_eval.py", "--model", str(pkg), "--label", label,
         "--prompts", *[str(staged / f"{k}.jsonl") for k in order], "--out-dir", str(raw),
         "--trained-context", str(ev["context"]["tokens"]), "--decoding", str(REPO / ev["decoding"]),
         "--check-every", str(ev["degeneration_check_every"]), "--system", ev["system_message"],
         "--chat-template-kwargs", json.dumps(ev["modes"][mode]["template_kwargs"], sort_keys=True),
         *(["--max-num-seqs", str(opts["max_num_seqs"])] if opts.get("max_num_seqs") else [])],
        log, env, opts.get("time_box_s"), poll=seal_finished)
    seal_finished()
    if boxed:
        rec.update(status="incomplete", why="time box", seconds=round(time.time() - t0, 1), attempt=attempt,
                   sealed=sorted(k for k in order if (gens / f"{k}.jsonl").is_file()))
        write_json(unit_dir / f"capability_run.{attempt}.json", rec)
        return rec
    if rc != 0:
        raise DriverError(f"uncapped_eval exited {rc} for {label}; see {log.name}")
    for k in order:
        seal(k)
    rec.update(status="complete", seconds=round(time.time() - t0, 1), attempt=attempt)
    write_json(unit_dir / f"capability_run.{attempt}.json", rec)
    return rec


#: the order units are generated in. ``pair``: each (group, mode) of the design's order, its fluid unit then its
#: capability units. ``mode_then_unit``: for each mode in the design's order, every group's fluid unit, then every
#: group's capability units -- the same pairs and units, so a budget that ends early leaves the primary axis (fluid,
#: H-T) measured for every group rather than everything for one group. The order decides nothing about a result:
#: every item has its fixed seed, and a unit is complete or absent.
UNIT_ORDERS = ("pair", "mode_then_unit")


def evaluation_plan(order: str, groups: list[str] | None = None, kinds: list[str] | None = None,
                    modes: list[str] | None = None, only_groups: list[str] | None = None) -> list[tuple[str, str, str]]:
    """(group, mode, unit kind) in execution order (`UNIT_ORDERS`); ``groups`` reorders the groups within a mode
    (an execution order, never a selection: an omitted group keeps its design place after them), ``kinds`` keeps
    only those unit kinds for this run."""
    pairs = [tuple(x) for x in design()["evaluation"]["order"]]
    if groups:
        rank = {g: i for i, g in enumerate(groups)}
        pairs = sorted(pairs, key=lambda p: (rank.get(p[0], len(rank)), pairs.index(p)))
        mode_rank = list(dict.fromkeys(m for _, m in [tuple(x) for x in design()["evaluation"]["order"]]))
        pairs = sorted(pairs, key=lambda p: mode_rank.index(p[1]))
    if order == "pair":
        plan = [(g, m, k) for g, m in pairs for k in ("fluid", "capability")]
    elif order == "mode_then_unit":
        in_order = list(dict.fromkeys(m for _, m in pairs))
        plan = [(g, m, k) for m in in_order for k in ("fluid", "capability") for g, mm in pairs if mm == m]
    else:
        raise DriverError(f"unit order {order!r} is not one of {UNIT_ORDERS}")
    return [x for x in plan if (not kinds or x[2] in kinds) and (not modes or x[1] in modes)
            and (not only_groups or x[0] in only_groups)]


def evaluate(out: Path, work: Path, vllm_py: str, status: Status, attempt: str, order: str = "pair",
             groups: list[str] | None = None, kinds: list[str] | None = None, opts: dict | None = None,
             modes: list[str] | None = None, only_groups: list[str] | None = None) -> dict:
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    mrec = read_json(out / "records" / "mapping.json")
    summary = read_json(out / "state" / "evaluation.json") if (out / "state" / "evaluation.json").is_file() else {}
    summary["_execution"] = {"unit_order": order, "groups": groups, "kinds": kinds, "modes": modes,
                             "only_groups": only_groups, **(opts or {}), "attempt": attempt}
    for g, mode, kind in evaluation_plan(order, groups, kinds, modes, only_groups):
        name = f"{g}_{mode}"
        unit_dir = out / "eval" / name
        pkg = work / "packages" / g
        if not (pkg / "config.json").exists():
            pk = package(work / "groups" / g, pkg)
            pk["artifact_digest"] = mrec["groups"][g]["artifact_digest"]
            write_json(out / "eval" / f"package_{g}.json", pk)
        status.mark(f"EVAL_START:{name}:{kind}")
        t0 = time.time()
        run = fluid if kind == "fluid" else capability
        mns = (opts or {}).get("max_num_seqs")
        unit_opts = {**(opts or {}), "max_num_seqs": (mns.get(g) if isinstance(mns, dict) else mns)}
        res = run(pkg, unit_dir, name, mode, vllm_py, env, attempt, unit_opts)
        summary.setdefault(name, {})[kind] = {**res, "seconds": round(time.time() - t0, 1), "utc": utc()}
        write_json(out / "state" / "evaluation.json", summary)
        status.mark(f"EVAL_OK:{name}:{kind}")
    return summary


# ---- resume ---------------------------------------------------------------------------------------------------------

def take_over(resume: Path, out: Path, attempt: str) -> dict:
    """A previous attempt's secured evidence into this attempt's output: the mapping, its records, the state
    evaluation and every sealed evaluation file (never a live stream); returns what was taken."""
    taken = []
    if resume.is_file():                   #: `launch.py stage-resume`'s archive, staged by setup from the relay
        import tarfile
        unpacked = out.parent / f".resume_{attempt}"
        shutil.rmtree(unpacked, ignore_errors=True)
        with tarfile.open(resume) as t:
            t.extractall(unpacked, filter="data")
        resume = unpacked
    for sub in ("mapping", "records", "eval"):
        src = resume / sub
        if not src.is_dir():
            continue
        for p in sorted(src.rglob("*")):
            if p.is_file() and not p.name.startswith("."):
                dest = out / sub / p.relative_to(src)
                if dest.exists():
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, dest)
                taken.append(str(dest.relative_to(out)))
    return {"from": str(resume), "files": len(taken), "consumer_attempt": attempt, "utc": utc()}


# ---- main -----------------------------------------------------------------------------------------------------------

def child(phase: str, a) -> int:
    """One GPU phase in its own process: everything it allocated is released when it exits."""
    out, work, status = Path(a.out), Path(a.work), Status(a.status)
    if phase == "mapping":
        build_mapping(out, work, a.device, status)
    elif phase == "relay":
        rec = relay_upload(work, out, a.run_id)
        write_json(out / "records" / "products.json", rec)
        if not rec["verified"]:
            raise DriverError("the relay copy does not re-hash to the pod's")
    elif phase == "restore":
        rec = relay_restore(work, read_json(out / "records" / "products.json"))
        write_json(out / "records" / f"restored.{a.run_id}.json", rec)
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", default="check")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--status", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--vllm-python", default="/opt/vllm/bin/python")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--phases", default="contract,mapping,relay,state_eval,evaluate")
    ap.add_argument("--unit-order", default="pair", choices=UNIT_ORDERS)
    ap.add_argument("--group-order", default="", help="execution order of the groups within a mode, e.g. W,DW,D")
    ap.add_argument("--kinds", default="", help="only these unit kinds this run: fluid,capability")
    ap.add_argument("--modes", default="", help="only these modes this run: H-T,H-N")
    ap.add_argument("--groups", default="", help="only these groups this run: D,W,DW")
    ap.add_argument("--max-num-seqs", default=None,
                    help="the engines' concurrent-sequence limit: one integer, or per group, e.g. D=7,W=6,DW=8")
    ap.add_argument("--engine-metrics", action="store_true", help="the fluid runner records engine metrics")
    ap.add_argument("--shuffle-submission", default=None,
                    help="a key: the fluid tasks reach the engine in sha256(key|task id) order (seeds unchanged)")
    ap.add_argument("--time-box-minutes", type=float, default=None,
                    help="stop each unit's runner after this; sealed work is kept and the unit stays incomplete")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--child", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.child:
        return child(a.child, a)
    out, work, status = Path(a.out), Path(a.work), Status(a.status)
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    report = {"schema": SCHEMA, "run_id": a.run_id, "started_utc": utc(), "phases": {}, "status": "STARTED"}
    status.mark("DRIVER_START")
    phases = [p for p in a.phases.split(",") if p]
    try:
        if a.resume:
            report["resume"] = take_over(Path(a.resume), out, a.run_id)
        t0 = time.time()
        #: on the host always: it derives config hashes and loads no model, and a CUDA context in this process would
        #: hold memory the evaluation's vLLM engines need
        report["phases"]["contract"] = {**contract(work, "cpu"),
                                        "seconds": round(time.time() - t0, 1)}
        write_json(out / "state" / "report.json", report)
        status.mark("CONTRACT_OK")
        if a.check_only:
            report["status"] = "CHECK_ONLY_OK"
            return 0
        have_mapping = (out / "records" / "mapping.json").is_file()
        have_products = (out / "records" / "products.json").is_file() and \
            read_json(out / "records" / "products.json").get("verified")
        if "mapping" in phases:
            if have_mapping and have_products:
                run_child("restore", a, online=True)
                report["phases"]["mapping"] = {"status": "restored from the relay"}
            else:
                t0 = time.time()
                run_child("mapping", a)
                report["phases"]["mapping"] = {"status": "built", "seconds": round(time.time() - t0, 1)}
                have_products = False
        if "relay" in phases and not have_products:
            t0 = time.time()
            run_child("relay", a, online=True)
            report["phases"]["relay"] = {"status": "verified", "seconds": round(time.time() - t0, 1)}
            status.mark("PRODUCTS_OFF_POD_OK")
        write_json(out / "state" / "report.json", report)
        if "state_eval" in phases:
            if (out / "records" / "state_eval.json").is_file() and \
                    read_json(out / "records" / "state_eval.json").get("finished_utc"):
                report["phases"]["state_eval"] = {"status": "restored"}
            else:
                t0 = time.time()
                run_child("state_eval", a)
                report["phases"]["state_eval"] = {"status": "measured", "seconds": round(time.time() - t0, 1)}
        write_json(out / "state" / "report.json", report)
        if "evaluate" in phases:
            mns = a.max_num_seqs
            if mns and "=" in mns:      #: per group, e.g. D=7,W=6,DW=8 (floor of each engine's KV cache / context)
                mns = {k: int(v) for k, v in (x.split("=") for x in mns.split(",") if x)}
            elif mns:
                mns = int(mns)
            opts = {"max_num_seqs": mns, "engine_metrics": a.engine_metrics,
                    "shuffle_submission": a.shuffle_submission,
                    "time_box_s": (a.time_box_minutes * 60 if a.time_box_minutes else None)}
            report["phases"]["evaluate"] = evaluate(
                out, work, a.vllm_python, status, a.run_id, a.unit_order,
                groups=[g for g in a.group_order.split(",") if g] if a.group_order else None,
                kinds=[k for k in a.kinds.split(",") if k] if a.kinds else None, opts=opts,
                modes=[m for m in a.modes.split(",") if m] if a.modes else None,
                only_groups=[g for g in a.groups.split(",") if g] if a.groups else None)
        ev_state = (report["phases"].get("evaluate") or {})
        report["incomplete_units"] = sorted(f"{name}:{kind}" for name, v in ev_state.items() if not name.startswith("_")
                                            for kind, r in v.items() if (r or {}).get("status") != "complete")
        report["status"] = "COMPLETE"     #: the driver ran its plan; a time-boxed unit is listed, never hidden
    except BaseException as exc:  # noqa: BLE001
        report["status"] = "FAILED"
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)[:3000],
                             "traceback": traceback.format_exc()[-6000:]}
    finally:
        report["ended_utc"] = utc()
        write_json(out / "state" / "report.json", report)
        if not a.check_only:
            write_json(out / "report.json", report)
            status.mark("ALL_DONE" if report["status"] == "COMPLETE" else "RUN_FAILED")
    return 0 if report["status"] in ("COMPLETE", "CHECK_ONLY_OK") else 1


if __name__ == "__main__":
    raise SystemExit(main())
