#!/usr/bin/env python3
"""On-pod driver of DMR1 variant 1's build (session kind ``dmr1_merge_build``): the alpha-merged initialization at
SOB1's depth-only structure, built from the original teacher, durable the moment each product exists.

    cd /workspace/aad && HF_HUB_OFFLINE=1 PYTHONPATH=src:scripts:scripts/stages/stage-1:scripts/stages/stage-0/calibration_volume \\
        /opt/train/bin/python scripts/stages/stage-1/depth_merge/build_driver.py --run-id <run> --out <evidence> \\
        --work <scratch> --status <file> --device cuda [--resume <secured evidence>]

    # $0 on the dev box: the design's pins, the spans, the base digests, the session's operator config hashes and
    # the products' scientific identity
    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python scripts/stages/stage-1/depth_merge/build_driver.py \\
        --check-only --out <s>/out --work <s>/w --status <s>/st

**Identity** (`identity`). Every product is bound to what it is a function of, never to prose: the teacher (root id,
weights sha256, config), the base (SOB1's depth init record, weights, config, per-tensor digests), the mapping (kept
blocks, merged spans, pairs), the calibration (profile and its data hash, the statistics positions' operator and
policy, the packed forward context, the session's config hash and measurement protocol, the execution's packing and
micro-batch) and the code (`merge.method_identity`: ast digests, so a docstring or comment changes nothing). The
statistics' identity is the sha256 of the first four and the statistics code; the checkpoint's adds the base, the
method, its seed and the candidate code. Each product carries its OWN identity (`stamp`), written when the product is
made or verified for reuse -- never before, and never copied onto a product made under another one.

**Products, in order**, each recorded in ``out/depth_merge/records/build.json`` the moment it exists and copied to
the relay at once:

1. **the contract** -- the packed search's session derives the journal's operator config hashes; the design's pinned
   records hash to their sha256 (`contract`);
2. **the base check** -- SOB1's depth child rebuilt from the teacher and compared tensor by tensor with the base's
   digests (`check_base`), whenever the teacher is loaded;
3. **the statistics** -- one teacher forward over the calibration at the statistics positions (RI-LM1's
   `collect_statistics`), the counts checked against the design; the 20 alignments pair by pair
   (`merge.pair_alignments`, each persisted with its inputs' identity); ``records/merge_statistics.safetensors``
   (per block E|a| and head energies, per pair the three permutations) -> relay group ``statistics``;
4. **the weights** -- the equivalence checks (a failure stops the build), RI-LM1's `build_candidate` (method
   ``M-act``) from the SAVED statistics file, one bf16 shard whose config FILE is the base's, reloaded with a finite
   forward -> relay group ``init``;
5. **diagnostics** (recorded, never a gate) -- the state evaluation and each merged span's residual update against
   the teacher span's, for the base's kept block and the merged one (each candidate block on the teacher's device,
   `span_candidates`);
6. ``depth_merge/init_record.json``, COMPLETE only when both relay groups are verified AND handed off, under this
   identity.

**A relay group's two states** (`secure`, recorded in ``records/relay.json`` at each): the files uploaded, downloaded
again and re-hashed (``verified``) -- recorded at once; then the handoff (``handoff.verified``): the group's record put
beside its files (`relay.POINTER`) and the relay listing read back naming it with its files present. The record carries
the product's identity and description (`product_view`), so a later attempt finds and restores it even when the pod
was lost before the dev box pulled ``build.json`` or ``relay.json`` (`recover`). A handoff that does not verify leaves
the files where they are: only the handoff is retried (later in the attempt, and by the next attempt), the files are
never sent again, nothing is recomputed, and no init record is written until it is done.

**Takeover** (`run`; ``--resume`` copies a previous attempt's secured evidence into ``out`` first). A product is
taken over when ITS identity is this attempt's and its bytes are where its record says -- locally with their sha256,
or on the relay (relay.json's verified record, or the relay's own listing). Only what is missing is made:

* the init record durable under this identity -> nothing runs;
* weights durable, diagnostics recorded -> only the pending handoffs and the init record (no teacher, no GPU);
* weights durable, diagnostics missing -> the weights are restored from their verified relay copy (re-hashed) for
  the diagnostics; nothing is rebuilt;
* statistics durable, weights missing or made under another checkpoint identity (the base, the method, its seed or
  the candidate code changed) -> the weights are built from the statistics file (fetched from the relay only then; no
  statistics forward); the old weights' records are set aside with their own identity (``superseded``), never
  relabelled;
* statistics missing -> the statistics forward runs; each persisted pair is reused only under the same inputs.

A build record whose statistics were made under another identity is moved aside (``build.superseded_<utc>.json``),
kept as evidence, never reused. `held` is the same lookup without side effects, for the launch gate's room arithmetic
(`session.relay_room`: a takeover asks the relay only for what the chain still adds).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
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
#: the arms this driver builds (`bind`): V1 (the alpha merge at the spans) and v1-new (long runs split off, `split`)
SCHEMAS = {"depth_merge": "aadistill.stage1.depth_merge.init_record/v1",
           "depth_merge_split": "aadistill.stage1.depth_merge_split.init_record/v1"}
SPLIT_ARM = "depth_merge_split"
GROUPS = ("statistics", "init")
#: what a product's identity is made of (`identity`): the statistics', then the checkpoint's on top
STATISTICS_PARTS = ("teacher", "mapping", "calibration", "statistics_code")
CHECKPOINT_PARTS = ("base", "merge")
#: what a weights rebuild or a new statistics pass makes stale in the build record (and ``checks.equivalence``)
DOWNSTREAM_OF = {"statistics": ("checkpoint", "candidate", "state_evaluation", "diagnostics", "gpu_peak",
                                "finished_utc"),
                 "checkpoint": ("checkpoint", "candidate", "state_evaluation", "diagnostics", "gpu_peak",
                                "finished_utc")}
#: a relay group's product, by the identity it carries
WHAT = {"statistics": "statistics", "init": "checkpoint"}


class DriverError(RuntimeError):
    pass


def bind(arm: str) -> str:
    """Which arm this process builds (``--arm``): its design, schema, records and relay prefixes. Returns the previous
    arm (a caller that binds temporarily restores it: `bound`)."""
    global ARM, SCHEMA
    if arm not in SCHEMAS:
        raise DriverError(f"{arm!r} is not an arm this driver builds ({sorted(SCHEMAS)})")
    prev, ARM, SCHEMA = ARM, arm, SCHEMAS[arm]
    return prev


class bound:
    """``with bound(arm):`` -- `bind` for the block, the previous arm restored after it."""

    def __init__(self, arm: str):
        self.arm = arm

    def __enter__(self):
        self.prev = bind(self.arm)
        return self

    def __exit__(self, *exc):
        bind(self.prev)
        return False


def design() -> dict:
    """The bound arm's design: V1's through `common.design` exactly as before the binding, another arm's by its file."""
    return DC.design() if ARM == "depth_merge" else DC.design_of(ARM)


def is_split() -> bool:
    return ARM == SPLIT_ARM


def stat_spans(d: dict | None = None) -> list[tuple[int, ...]]:
    """The spans this build merges (its statistics pass, its alignments): V1's merged spans; v1-new's separate runs."""
    if is_split():
        from stages.depth_merge import split as SPLIT
        return SPLIT.new_spans()
    return M.merge_spans(d or design())


def checkpoint_parts() -> tuple[str, ...]:
    return CHECKPOINT_PARTS + (("reuse",) if is_split() else ())


def v1_init_ref() -> dict:
    """v1-new's reuse source: V1's training-pre initialization as V1's committed training plan binds it."""
    from stages.depth_merge import plan as PL
    with bound("depth_merge"):
        plan = read_json(PL.out_path("depth_merge"))
    u = plan["units"][0]
    return {"init_record": plan["init_record"], "relay": u["initialization"]["relay"],
            "model_safetensors_sha256": plan["init_record"]["model_safetensors_sha256"]}


def records(out: Path) -> Path:
    return Path(out) / ARM / "records"


def canonical_sha256(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


# ---- the contract and the identity ($0) --------------------------------------------------------------------------

def contract(work: Path, device: str) -> dict:
    """$0: the design's pins hash to their records, the spans are the mapping's, the base digests are the base's,
    the session derives the journal's operator config hashes; and what of the session the statistics read (its
    execution and its position policy), for `identity`."""
    from stages.teacher_parent_initialization import init_driver as TI
    d = design()
    b = d["baseline"]
    bad = [f"{v['path']} hashes to {sha256_file(REPO / v['path'])[:12]}, the design pins {v['sha256'][:12]}"
           for v in (b["sob1_design"], b["init_record"], b["training_plan"], b["study_completion"],
                     b["control_sessions"]["completion"]) if sha256_file(REPO / v["path"]) != v["sha256"]]
    if bad:
        raise DriverError("; ".join(bad))
    spans = stat_spans(d)
    M.base_digests(d)
    base = TI.contract(work, device)
    s = TI.session(work / "session", device)
    pp = s.config.position_policy
    split = {}
    if is_split():
        from stages.depth_merge import split as SPLIT
        bad = SPLIT.groups_problems()
        if bad:
            raise DriverError("; ".join(bad))
        split = {"groups": [{"layers": list(x["layers"]), "anchor": int(x["anchor"]), "kind": x["kind"]}
                            for x in SPLIT.groups()],
                 "config_sha256": sha256_file(SPLIT.CONFIG),
                 "split_code": M.lmx().code_identity(SPLIT.groups, SPLIT.student_layers, SPLIT.v1_layer_tensors,
                                                     SPLIT.build_candidate)["sha256"],
                 "v1_init": {k: v for k, v in v1_init_ref().items() if k != "relay"}}
    return {"design_sha256": sha256_file(DC.files_of(ARM)[0]), "digests_sha256": sha256_file(M.DIGESTS),
            "spans": [list(x) for x in spans], "pairs": [list(p) for p in M.pairs(spans)], **split,
            "session_contract": base, "method_identity": M.method_identity(),
            #: the byte-affecting execution fields only (micro-batch, packing, prediction rows), not its trace
            "execution": s.execution.as_fingerprint(),
            "position_policy": {"qualified_id": pp.qualified_id, "policy_hash": pp.policy_hash}}


def identity(c: dict, d: dict | None = None) -> dict:
    """Each product's scientific identity (module docstring) from the contract and the design's facts -- never from
    the design file's bytes, so a changed description recomputes nothing."""
    d = d or design()
    t, b, st = d["teacher"], d["baseline"], d["statistics"]
    sel = M.selection_view(d)
    ses = c["session_contract"]
    parts = {
        "teacher": {k: t[k] for k in ("root_teacher_id", "root_teacher_sha256", "config_sha256", "num_parameters")},
        "base": {**{k: b["init_record"][k] for k in ("sha256", "artifact_digest", "model_safetensors_sha256",
                                                      "config_json_sha256", "num_parameters")},
                 "tensor_digests_sha256": canonical_sha256(M.base_digests(d))},
        "mapping": {"n_root_blocks": int(b["mapping"]["n_root_blocks"]), "kept_blocks": sel["kept_blocks"],
                    "merged_spans": c["spans"], "pairs": c["pairs"], **({"groups": c["groups"]} if "groups" in c else {})},
        "calibration": {"profile_id": st["profile_id"], "positions_of_operator": st["positions_of_operator"],
                        "expected": st["expected"], "profiles": ses["session"]["profiles"],
                        "session_config_hash": ses["session"]["config_hash"],
                        "measurement_protocol_id": ses["session"]["measurement_protocol_id"],
                        "forward_context": ses["session"]["forward_context"],
                        "operator_config_hashes": {h["step"]: h["derived"] for h in ses["config_hashes"]},
                        "execution": c["execution"], "position_policy": c["position_policy"]},
        "statistics_code": c["method_identity"]["statistics"]["sha256"],
        "merge": {"method": M.METHOD, "seed": int(sel["source"]["seed"]),
                  "candidate_code": c["method_identity"]["candidate"]["sha256"],
                  **({"config_sha256": c["config_sha256"], "split_code": c["split_code"]} if "config_sha256" in c
                     else {})},
    }
    if "v1_init" in c:
        parts["reuse"] = {"v1_init": c["v1_init"]}
    stats = canonical_sha256({k: parts[k] for k in STATISTICS_PARTS})
    return {"parts": parts, "statistics": stats,
            "checkpoint": canonical_sha256({"statistics": stats, **{k: parts[k] for k in checkpoint_parts()}}),
            "rule": "sha256 of canonical JSON: statistics = teacher, mapping, calibration, statistics code; "
                    "checkpoint = the statistics' identity, the base, the merge (method, seed, candidate code)"}


def _parts_of(what: str) -> tuple[str, ...]:
    return STATISTICS_PARTS + (checkpoint_parts() if what == "checkpoint" else ())


def stamp(ident: dict, what: str) -> dict:
    """The identity a product (``statistics`` / ``checkpoint``) is recorded under, once it is made or verified for
    reuse: its digest and the parts it is a function of."""
    return {"sha256": ident[what], "parts": {k: ident["parts"][k] for k in _parts_of(what)}}


def identity_problems(recorded: dict | None, ident: dict, what: str) -> list[str]:
    """Why a product recorded under ``recorded`` (its `stamp`, or a whole `identity`) is not this attempt's ``what``
    (``statistics`` / ``checkpoint``), naming the parts that differ."""
    rec = recorded or {}
    sha = rec.get("sha256", rec.get(what))
    if not sha:
        return [f"{what}: no identity recorded"]
    if sha == ident[what]:
        return []
    have = rec.get("parts") or {}
    differ = [k for k in _parts_of(what) if have.get(k) != ident["parts"][k]]
    return [f"{what}: made under another identity ({', '.join(differ) or 'the digest only'})"]


# ---- the teacher-side context (GPU), opened only when a product has to be made -----------------------------------

class Gpu:
    """The session, the adapter, the teacher (checked to be the design's root), the base rebuilt from it and checked
    against the base's digests (`check_base`), the calibration. Opened only when a statistics pass, a weights build
    or the diagnostics have to run (`open_gpu`)."""

    def __init__(self, work: Path, device: str, rec: dict):
        import torch
        from aadistill.initialization.specs.arch import get_adapter
        from stages.teacher_modes import full_search as FS
        from stages.teacher_parent_initialization import init_driver as TI
        torch.backends.cuda.matmul.allow_tf32 = False             #: PearsonMoments refuses TF32
        torch.set_float32_matmul_precision("highest")
        d = design()
        self.device = device
        self.s = TI.session(work / "session", device)
        self.adapter = get_adapter(self.s.config.target_spec.family)
        self.teacher, root = FS.load_root(device)
        rec["root"] = root
        if root["root_teacher_sha256"] != d["teacher"]["root_teacher_sha256"]:
            raise DriverError("the loaded teacher is not the design's root")
        self.n_q, self.n_kv, _hd = self.adapter.head_groups(self.adapter.spec_of(self.teacher))
        self.sel, self.spans, self.digests = M.selection_view(d), stat_spans(d), M.base_digests(d)
        self.base, rec.setdefault("checks", {})["base"] = check_base(self.adapter, self.teacher, self.sel, self.digests)
        self.items, self.active, self.batches = M.calibration(self.s, self.teacher, device, d)


def open_gpu(work: Path, device: str, rec: dict) -> Gpu:
    return Gpu(work, device, rec)


def check_base(adapter, teacher, sel: dict, digests: dict) -> tuple:
    """The base rebuilt from the teacher, every tensor against the base's digests (RI-LM1's `check_base` rule). The
    base stays on the host."""
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


def span_candidates(adapter, base, merged, sel: dict, spans, device, teacher=None) -> dict:
    """The diagnostics' candidate blocks per merged span start, each on the teacher's ``device``: the base's kept block
    (only those blocks are moved; the rest of the base stays on the host) and the merged model's (already there).
    The caller drops the result after the diagnostics, which frees the moved blocks with it."""
    bm = adapter.blocks(merged)
    if is_split():
        #: a separate run's references: its first teacher block alone (the run otherwise deleted), and the new layer
        from stages.depth_merge import split as SPLIT
        at = {a: j for j, a in enumerate(SPLIT.student_layers())}
        return {"anchor_alone": {s[0]: adapter.blocks(teacher)[s[0]] for s in spans},
                "merged": {s[0]: bm[at[s[0]]] for s in spans}}
    kept = [int(x) for x in sel["kept_blocks"]]
    idx = {s[0]: kept.index(s[0]) for s in spans}
    bb = adapter.blocks(base)
    return {"kept": {s0: bb[i].to(device) for s0, i in idx.items()},
            "merged": {s0: bm[i] for s0, i in idx.items()}}


def persisted_pairs(out: Path) -> dict:
    d = records(out) / "alignments"
    return {p.stem.replace("-", "->"): read_json(p) for p in sorted(d.glob("*.json"))} if d.is_dir() else {}


# ---- the products ------------------------------------------------------------------------------------------------

def statistics_step(g: Gpu, rec: dict, out: Path, status: Status, run_id: str) -> dict:
    """Step 3 of the module docstring: the statistics forward, the alignments, the statistics file. Returns the
    statistics' record; `run` stamps it with its identity and records it (a pass interrupted before then is not a
    product: ``saved_utc`` is what makes one)."""
    import torch
    from safetensors.torch import save_file
    LMX = M.lmx()
    d = design()
    t0 = time.time()
    st = LMX.collect_statistics(g.adapter, g.teacher, g.spans, g.batches, g.active)
    exp = d["statistics"]["expected"]
    stats_rec = {"sequences": st.sequences, "positions": st.positions, "seconds": round(time.time() - t0, 1),
                 "position_policy": g.active.report(), "forward_context": g.s.config.forward_context.as_dict()}
    if (st.sequences, st.positions) != (exp["packed_sequences"], exp["statistics_positions"]):
        raise DriverError(f"{st.sequences} sequences / {st.positions} positions, the design expects "
                          f"{exp['packed_sequences']} / {exp['statistics_positions']}")
    rec["statistics"] = stats_rec
    write_json(records(out) / "build.json", rec)
    status.mark("STATISTICS_COLLECTED")

    def on_pair(tag: str, x: dict) -> None:
        write_json(records(out) / "alignments" / f"{tag.replace('->', '-')}.json", {**x, "run_id": run_id, "utc": utc()})
        status.mark(f"PAIR_ALIGNED:{tag}")
    t1 = time.time()
    al, stats_rec["alignment_reuse"] = M.pair_alignments(st, g.n_q, g.n_kv, persisted=persisted_pairs(out),
                                                         on_pair=on_pair)
    stats_rec["alignment_seconds"] = round(time.time() - t1, 1)
    stats = LMX.statistics_tensors(st, al)
    del st
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    ten_p = records(out) / f"{M.STATS}.safetensors"
    tmp = ten_p.with_name(f".{ten_p.name}.partial")
    save_file(stats, str(tmp))
    os.replace(tmp, ten_p)
    stats_rec.update(tensors_sha256=sha256_file(ten_p), saved_utc=utc(),
                     pairs={f"{a}->{b}": x["summary"] for (a, b), x in al.items()},
                     permutation_sha256={k: hashlib.sha256(v.numpy().tobytes()).hexdigest()
                                         for k, v in stats.items() if "perm" in k})
    return stats_rec


def weights_step(g: Gpu, rec: dict, out: Path, work: Path, status: Status):
    """Step 4: the checks and the candidate from the saved statistics file; returns the reloaded model and the
    checkpoint's record, which `run` stamps with its identity and records."""
    import torch
    from safetensors.torch import load_file
    from stages.teacher_parent_initialization import init_driver as TI
    LMX = M.lmx()
    d = design()
    ten_p = records(out) / f"{M.STATS}.safetensors"
    if sha256_file(ten_p) != rec["statistics"]["tensors_sha256"]:
        raise DriverError("the statistics file is not the one the build record names")
    stats = load_file(str(ten_p))
    tol = float(d["checks"]["fold_equivalence"]["tolerance"])
    eq = LMX.equivalence_checks(g.adapter, g.teacher, g.spans, stats, next(iter(g.batches())), tol)
    rec["checks"]["equivalence"] = eq
    write_json(records(out) / "build.json", rec)
    if not eq["passed"]:
        raise DriverError(f"folding / permutation does not reproduce the block: worst {eq['worst']}, non-finite "
                          f"{eq['non_finite']} (tolerance {tol})")
    if is_split():
        from stages.depth_merge import split as SPLIT
        v1_dir = fetch_v1_init(work)
        rec["v1_init"] = {"path": str(v1_dir), "model_safetensors_sha256": v1_init_ref()["model_safetensors_sha256"]}
        model, rec["candidate"] = SPLIT.build_candidate(g.adapter, g.teacher, stats, g.digests, v1_dir,
                                                        seed=int(g.sel["source"]["seed"]))
    else:
        model, rec["candidate"] = LMX.build_candidate(g.adapter, g.teacher, g.sel, g.spans, M.METHOD, stats,
                                                      g.digests, seed=int(g.sel["source"]["seed"]))
    ckpt = work / "products" / "init"
    ident = TI._save(g.adapter, model, ckpt)
    want = d["baseline"]["init_record"]
    files = {"model.safetensors": ident.shards[0].sha256, "config.json": sha256_file(ckpt / "config.json")}
    n_params = int(sum(p.numel() for p in model.parameters()))
    ck = {**_ident_record(ident), "files_sha256": files, "num_parameters_counted": n_params,
          "statistics_sha256": rec["statistics"]["tensors_sha256"]}
    if is_split():
        from stages.depth_merge import split as SPLIT
        ck.update(config_file_is_the_derived_one=files["config.json"] == sha256_file(SPLIT.CONFIG),
                  num_parameters_match_design=n_params == int(d["arms"][ARM]["num_parameters"]))
        ok = ck["config_file_is_the_derived_one"] and ck["num_parameters_match_design"]
    else:
        ck.update(config_file_is_the_base_s=files["config.json"] == want["config_json_sha256"],
                  num_parameters_match_base=n_params == int(want["num_parameters"]))
        ok = ck["config_file_is_the_base_s"] and ck["num_parameters_match_base"]
    if not (len(ident.shards) == 1 and ok):
        raise DriverError(f"the candidate is not the design's structure: {ck}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    m = g.adapter.load(str(ckpt), device=g.device)
    m.config.use_cache = False
    ids = torch.arange(1, 65, device=next(m.parameters()).device).unsqueeze(0)
    ck["reload_forward_finite"] = bool(torch.isfinite(m(input_ids=ids).logits).all().item())
    if not ck["reload_forward_finite"]:
        raise DriverError("the reloaded candidate gives non-finite logits")
    return m, ck


def fetch_v1_init(work: Path) -> Path:
    """v1-new's reuse source on the pod: V1's init from its verified relay copy, every file re-hashed
    (`train_driver.fetch_init`'s rule), under ``work/v1_init``."""
    from stages.teacher_parent_initialization import relay as R
    from stages.teacher_parent_initialization import train_driver as TD
    ref = v1_init_ref()
    dest = Path(work) / "v1_init"
    w = dest / "model.safetensors"
    if not (w.is_file() and sha256_file(w) == ref["model_safetensors_sha256"]):
        rec = ref["relay"]
        sub = {k: v for k, v in rec["files_sha256"].items() if k.startswith("init/")}
        flat = {**rec, "files_sha256": {k.removeprefix("init/"): v for k, v in sub.items()},
                "prefix": f"{rec['prefix']}/init"}
        nbytes = sum(v for k, v in (rec.get("bytes") or {}).items() if k.startswith("init/"))
        got = R.run_online("get", {"record": flat, "dest": str(dest)},
                           timeout_s=max(3600.0, TD.transfer_seconds(nbytes, durability(), up=False) + 600))
        if not got.get("verified"):
            raise DriverError(f"V1's init did not restore from the relay: {str(got)[:400]}")
    if sha256_file(w) != ref["model_safetensors_sha256"]:
        raise DriverError("V1's init on the pod is not the one V1's plan binds")
    return dest


def diagnostics_step(g: Gpu, rec: dict, out: Path, work: Path, m=None) -> None:
    """Step 5: recorded, never a gate; ``m`` the merged model on the device (loaded from ``work`` when not given)."""
    import torch
    if m is None:
        m = g.adapter.load(str(work / "products" / "init"), device=g.device)
        m.config.use_cache = False
    try:
        t2 = time.time()
        g.s.evaluator.prime_reference(g.teacher)
        ev = g.s.evaluator.evaluate(m, rec["checkpoint"]["artifact_digest"])
        rec["state_evaluation"] = {"measurement_protocol_id": g.s.evaluator.measurement_protocol_id,
                                   "values": dict(getattr(ev, "values", {}) or {}),
                                   "seconds": round(time.time() - t2, 1), "role": "diagnostic only"}
    except Exception as exc:  # noqa: BLE001
        rec["state_evaluation"] = {"error": f"{type(exc).__name__}: {str(exc)[:1500]}", "role": "diagnostic only"}
    cands = None
    try:
        dev = next(g.teacher.parameters()).device
        cands = span_candidates(g.adapter, g.base, m, g.sel, g.spans, dev, teacher=g.teacher)
        rec["diagnostics"] = M.lmx().diagnostics(g.adapter, g.teacher, g.spans, cands, g.batches, g.active)
    except Exception as exc:  # noqa: BLE001
        rec["diagnostics"] = {"error": f"{type(exc).__name__}: {str(exc)[:1500]}", "role": "diagnostic only"}
    finally:
        del cands
        g.base = None                       #: the base's moved blocks go with it: nothing else reads the base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    rec["gpu_peak"] = _gpu_peak(g.device)
    rec["finished_utc"] = utc()
    write_json(records(out) / "build.json", rec)


# ---- durability: the relay groups ----------------------------------------------------------------------------------

def durability() -> dict:
    from stages.teacher_parent_initialization import train_driver as TD
    return {**TD.DURABILITY, **design()["training"]["durability"]}


def group_root(name: str) -> str:
    return f"{design()['storage']['relay_prefix_root']}/{ARM}/{name}"


def group_files(name: str, rec: dict) -> dict[str, str]:
    """A relay group's files (relative name -> sha256) from the build record."""
    if name == "init":
        return {f"init/{n}": s for n, s in rec["checkpoint"]["files_sha256"].items()}
    return {f"statistics/{M.STATS}.safetensors": rec["statistics"]["tensors_sha256"]}


def group_prefix(name: str, rec: dict) -> str:
    key = rec["checkpoint"]["artifact_digest"] if name == "init" else rec["statistics"]["tensors_sha256"]
    return f"{group_root(name)}/{key[:16]}"


def local_files(name: str, out: Path, work: Path) -> dict[str, Path]:
    if name == "init":
        return {f"init/{n}": work / "products" / "init" / n for n in ("model.safetensors", "config.json")}
    return {f"statistics/{M.STATS}.safetensors": records(out) / f"{M.STATS}.safetensors"}


def products(rec: dict) -> dict[str, dict[str, str]]:
    """What goes to the relay, by group: the checkpoint's two files and the statistics tensors."""
    return {name: group_files(name, rec) for name in GROUPS}


def _put(payload: dict, timeout_s: float) -> dict:
    from stages.teacher_parent_initialization import relay as R
    return R.run_online("put", payload, timeout_s=timeout_s)


def _get(record: dict, dest: Path, timeout_s: float) -> dict:
    from stages.teacher_parent_initialization import relay as R
    return R.run_online("get", {"record": record, "dest": str(dest)}, timeout_s=timeout_s)


def _listing(prefix: str) -> dict:
    from stages.teacher_parent_initialization import train_driver as TD
    try:
        return TD.list_records(prefix, durability())
    except TD.ListingUnreadable as exc:
        raise DriverError(f"the relay listing is unreadable, never read as empty: {exc}") from exc


def _handoff(rec: dict, scratch: Path, listing: str, message: str) -> dict:
    from stages.teacher_parent_initialization import train_driver as TD
    return TD.hand_off_with_retries(rec, scratch, listing, message, newest=False, d=durability())


def _bytes_ok(files: dict[str, str], local: dict[str, Path]) -> bool:
    return all(local[k].is_file() and sha256_file(local[k]) == s for k, s in files.items())


def product_view(name: str, rec: dict) -> dict:
    """What a group's relay record carries of the build record (it is also the record put beside the files): the
    product's own record with its identity, so an attempt whose build record never came back restores it (`recover`)."""
    if name == "statistics":
        return {"statistics": rec["statistics"]}
    return {"checkpoint": rec["checkpoint"], "candidate": rec.get("candidate"),
            "equivalence": (rec.get("checks") or {}).get("equivalence")}


def handed_off(r: dict | None) -> bool:
    """The group's record is beside its files on the relay and the listing names it with its files present."""
    return ((r or {}).get("handoff") or {}).get("verified") is True


def _listed(name: str) -> list[dict]:
    """The relay's records under the group's root that have their files present, as a takeover adopts them: found by
    the listing, which is the handoff's own check."""
    out = []
    for x in _listing(group_root(name)).get("records", []):
        if x.get("present"):
            out.append({**(x.get("record") or {}), "found_by": f"the relay listing under {group_root(name)}",
                        "handoff": {"verified": True, "by": "the relay listing names it with its files present",
                                    "utc": utc()}})
    return out


def _is(r: dict, files: dict, want: str) -> bool:
    return r.get("verified") is True and r.get("files_sha256") == files and r.get("identity_sha256") == want


def relay_copy(name: str, rec: dict, prev: dict, ident: dict) -> dict | None:
    """The group's verified relay copy of exactly the build record's files, recorded under this attempt's identity of
    the product: relay.json's record (this or an earlier attempt's, handed off or not), else a record the relay's
    own listing names; None when there is none."""
    files, want = group_files(name, rec), ident[WHAT[name]]
    r = prev.get(name) or {}
    if _is(r, files, want):
        return r
    return next((x for x in _listed(name) if _is(x, files, want)), None)


def recover(name: str, rec: dict, prev: dict, ident: dict) -> dict | None:
    """A product the build record does not name but the relay holds under this identity -- relay.json's record, or a
    record the relay's listing names (the pod lost before the dev box pulled its latest ``build.json``) -- restored
    into ``rec`` from the description its record carries (`product_view`), once that description is under this
    identity and names exactly the record's files (the weights: made from the build record's statistics). Returns
    the relay record (also put in ``prev``), or None."""
    want, what = ident[WHAT[name]], WHAT[name]
    cands = ([prev[name]] if (prev.get(name) or {}).get("verified") is True else []) + _listed(name)
    for r in cands:
        view = r.get("product") or {}
        if r.get("verified") is not True or r.get("identity_sha256") != want or what not in view:
            continue
        trial = copy.deepcopy(rec)
        trial[what] = copy.deepcopy(view[what])
        if name == "init":
            if view[what].get("statistics_sha256") != (rec.get("statistics") or {}).get("tensors_sha256"):
                continue
            trial["candidate"] = view.get("candidate")
            trial.setdefault("checks", {})["equivalence"] = view.get("equivalence")
        if identity_problems(trial[what].get("identity"), ident, what):
            continue
        try:
            files = group_files(name, trial)
        except (KeyError, TypeError, AttributeError):
            continue
        if files != r.get("files_sha256"):
            continue
        rec.clear()
        rec.update(trial)
        prev[name] = r
        return r
    return None


def secure(name: str, rec: dict, out: Path, work: Path, run_id: str, status: Status, prev: dict,
           ident: dict) -> dict:
    """The group durable (the module docstring's two states): an existing verified copy under this identity is taken
    (never sent again); otherwise the local files, which must be the record's, are put and verified, and that record
    is written to relay.json AT ONCE; then, unless it is already, the group is handed off. A handoff that does not
    verify is recorded and left pending (`run` retries it before the init record; the next attempt takes the verified
    copy over and retries only the handoff)."""
    files, local = group_files(name, rec), local_files(name, out, work)
    relay_p = records(out) / "relay.json"
    here = _bytes_ok(files, local)
    try:
        have = relay_copy(name, rec, prev, ident)
    except DriverError:
        if not here:
            raise
        have = None                          #: the files are here: `relay.put` sends only what the relay lacks
    if have is None:
        if not here:
            raise DriverError(f"{name}: no verified relay copy, and the local files are not the record's")
        nbytes = sum(p.stat().st_size for p in local.values())
        res = _put({"files": {k: str(v) for k, v in local.items()}, "prefix": group_prefix(name, rec),
                    "message": f"DMR1-M {run_id}: {name}", "scratch": str(work)},
                   max(3600.0, nbytes / 2.6e6 * 1.25 + 600))
        if not (res.get("verified") is True and res.get("files_sha256") == files):
            prev[name] = res
            write_json(relay_p, prev)
            raise DriverError(f"the relay copy of {name} is not its manifest verified: {str(res)[:600]}")
        have = {**res, "group": name, "identity_sha256": ident[WHAT[name]], "run_id": run_id,
                "product": product_view(name, rec), "handoff": {"verified": False, "why": "not attempted yet"}}
        prev[name] = have
        write_json(relay_p, prev)            #: the verified bytes recorded before anything else can fail
        status.mark(f"RELAY_VERIFIED:{name}")
    else:
        prev[name] = have
        write_json(relay_p, prev)
        status.mark(f"RELAY_TAKEN_OVER:{name}")
    if not handed_off(have):
        hand_off(name, have, work, run_id, prev, relay_p)
    status.mark(f"RELAY_OK:{name}" if handed_off(have) else f"RELAY_HANDOFF_PENDING:{name}")
    return have


def hand_off(name: str, have: dict, work: Path, run_id: str, prev: dict, relay_p: Path) -> bool:
    """One handoff of the group's verified record (`train_driver.hand_off_with_retries`: only its pointer is
    written and the listing read; the files are never sent again), its answer kept in relay.json; True when it
    verified."""
    tries = (have.get("handoff") or {}).get("attempts_by_run") or []
    h = _handoff(have, work, group_root(name), f"DMR1-M {run_id}: {name}")
    have["handoff"] = {**h, "attempts_by_run": [*tries, {"run_id": run_id, "verified": h.get("verified") is True,
                                                         "why": h.get("why"), "utc": utc()}]}
    prev[name] = have
    write_json(relay_p, prev)
    return handed_off(have)


def restore(name: str, rec: dict, relay_rec: dict, out: Path, work: Path) -> None:
    """The group's files back from its verified relay copy, each re-hashed (`relay.get`), to where the build reads
    them."""
    dest = work / "products" if name == "init" else work / "restore"
    got = _get(relay_rec, dest, max(3600.0, sum((relay_rec.get("bytes") or {}).values()) / 2.6e6 * 1.25 + 600))
    if not got.get("verified"):
        raise DriverError(f"{name}: the relay copy did not come back: {str(got)[:600]}")
    if name == "statistics":
        src = dest / f"statistics/{M.STATS}.safetensors"
        records(out).mkdir(parents=True, exist_ok=True)
        os.replace(src, records(out) / f"{M.STATS}.safetensors")
    if not _bytes_ok(group_files(name, rec), local_files(name, out, work)):
        raise DriverError(f"{name}: the restored files are not the record's")


# ---- the build as a sequence of products, each taken over or made -----------------------------------------------

def _checkpoint_problems(rec: dict, ident: dict) -> list[str]:
    """Why the build record's weights are not this attempt's (empty when they are)."""
    ck = rec.get("checkpoint") or {}
    why = identity_problems(ck.get("identity"), ident, "checkpoint")
    if not why and ck.get("statistics_sha256") != (rec.get("statistics") or {}).get("tensors_sha256"):
        why = ["checkpoint: made from other statistics than the build record's"]
    if not why and ck.get("reload_forward_finite") is not True:
        why = ["checkpoint: its reload was never checked finite"]
    return why


def run(out: Path, work: Path, device: str, status: Status, run_id: str, ident: dict) -> dict:
    """The module docstring's takeover rule. Returns ``{"init_record", "actions"}``."""
    doc = durable_init(out, ident)
    if doc is not None:
        status.mark("INIT_TAKEN_OVER")
        return {"init_record": doc, "actions": ["init record: taken over (durable under this identity)"]}
    rec_p, relay_p = records(out) / "build.json", records(out) / "relay.json"
    rec = read_json(rec_p) if rec_p.is_file() else {}
    prev = read_json(relay_p) if relay_p.is_file() else {}
    actions: list[str] = []
    if rec.get("statistics") and not rec["statistics"].get("saved_utc"):
        rec.pop("statistics")                #: an interrupted pass: not a product
    why = identity_problems(rec["statistics"].get("identity"), ident, "statistics") if rec.get("statistics") else []
    if why:
        aside = rec_p.with_name(f"build.superseded_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
        os.replace(rec_p, aside)
        actions.append(f"build record set aside ({'; '.join(why)}): {aside.name}")
        rec = {}
    if not rec:
        rec = {"started_utc": utc(), "run_id": run_id, "arm": ARM, "method": M.METHOD, "checks": {}}
    rec["method_identity"] = M.method_identity()
    rec.setdefault("attempts", []).append({"run_id": run_id, "utc": utc(), "actions": actions,
                                           "identity": {k: ident[k] for k in ("statistics", "checkpoint")}})
    records(out).mkdir(parents=True, exist_ok=True)
    write_json(rec_p, rec)
    g: Gpu | None = None

    def gpu() -> Gpu:
        nonlocal g
        if g is None:
            g = open_gpu(work, device, rec)
            write_json(rec_p, rec)
            status.mark("BASE_IS_SOB1_DEPTH")
        return g

    def stale(what: str) -> None:
        for k in DOWNSTREAM_OF[what]:
            rec.pop(k, None)
        rec.setdefault("checks", {}).pop("equivalence", None)

    # the statistics
    have_stats = bool(rec.get("statistics"))
    if not have_stats and recover("statistics", rec, prev, ident) is not None:
        have_stats = True
        actions.append("statistics: not in the build record; restored from the record beside their verified relay "
                       "copy (this identity)")
    s_local = have_stats and _bytes_ok(group_files("statistics", rec), local_files("statistics", out, work))
    s_copy = None
    if have_stats and not s_local:
        s_copy = relay_copy("statistics", rec, prev, ident)
        if s_copy is None:
            have_stats = False
            actions.append("statistics: recorded but no copy is left anywhere: recomputed")
        else:
            actions.append("statistics: taken over (their verified relay copy; fetched only for a weights build)")
    elif have_stats:
        actions.append("statistics: taken over (the secured file, sha256 the record's)")
    if not have_stats:
        stale("statistics")
        made = statistics_step(gpu(), rec, out, status, run_id)
        rec["statistics"] = {**made, "identity": stamp(ident, "statistics")}
        write_json(rec_p, rec)
        status.mark("STATISTICS_SAVED")
        s_local = True
        actions.append("statistics: computed")
    write_json(rec_p, rec)
    secure("statistics", rec, out, work, run_id, status, prev, ident)

    # the weights
    m = None
    if rec.get("checkpoint"):
        why = _checkpoint_problems(rec, ident)
        if why:
            old = {k: rec[k] for k in DOWNSTREAM_OF["checkpoint"] if k in rec}
            old["equivalence"] = (rec.get("checks") or {}).get("equivalence")
            old_relay = prev.pop("init", None)
            rec.setdefault("superseded", []).append({"product": "checkpoint", "why": why, "utc": utc(),
                                                     "record": old, "relay": old_relay})
            if old_relay is not None:
                prev.setdefault("superseded", []).append(old_relay)
                write_json(relay_p, prev)
            stale("checkpoint")
            write_json(rec_p, rec)
            actions.append(f"weights: set aside ({'; '.join(why)}), kept as evidence under their own identity")
    have_w = bool(rec.get("checkpoint"))
    if not have_w and recover("init", rec, prev, ident) is not None:
        have_w = True
        actions.append("weights: not in the build record; restored from the record beside their verified relay copy "
                       "(this identity)")
    w_local = have_w and _bytes_ok(group_files("init", rec), local_files("init", out, work))
    if have_w and not w_local and relay_copy("init", rec, prev, ident) is None:
        have_w = False
        actions.append("weights: recorded but no copy is left anywhere: rebuilt from the statistics file")
    if not have_w:
        stale("checkpoint")
        if not s_local:
            restore("statistics", rec, s_copy or relay_copy("statistics", rec, prev, ident), out, work)
            actions.append("statistics: fetched from their verified relay copy (re-hashed) for the weights build")
        m, made = weights_step(gpu(), rec, out, work, status)
        rec["checkpoint"] = {**made, "identity": stamp(ident, "checkpoint")}
        rec["method"] = ident["parts"]["merge"]["method"]       #: the method these weights were built with
        write_json(rec_p, rec)
        status.mark("INIT_BUILT")
        w_local = True
        actions.append("weights: built from the statistics file")
    else:
        actions.append("weights: taken over (" + ("the pod's files" if w_local else "the verified relay copy") + ")")
    secure("init", rec, out, work, run_id, status, prev, ident)

    # the diagnostics
    if not rec.get("finished_utc"):
        if not w_local:
            restore("init", rec, relay_copy("init", rec, prev, ident), out, work)
            actions.append("weights: restored from the verified relay copy for the diagnostics")
        diagnostics_step(gpu(), rec, out, work, m)
        actions.append("diagnostics: computed")
    else:
        actions.append("diagnostics: taken over")

    # the handoffs still pending, then the init record
    for name in GROUPS:
        if not handed_off(prev[name]):
            ok = hand_off(name, prev[name], work, run_id, prev, relay_p)
            actions.append(f"{name}: handoff " + ("completed" if ok else "still pending"))
    rec["attempts"][-1]["actions"] = actions
    write_json(rec_p, rec)
    pending = [name for name in GROUPS if not handed_off(prev[name])]
    if pending:
        status.mark("RELAY_HANDOFF_PENDING")
        raise DriverError(f"the relay copies of {pending} are verified but not handed off (their records beside them, "
                          "the listing naming them): kept as they are; the next attempt retries only the handoff")
    doc = init_record(rec, {name: prev[name] for name in GROUPS}, ident)
    bad = product_problems(doc, ident)
    if bad:
        raise DriverError(f"the init record would not show its products durable: {bad}")
    write_json(Path(out) / ARM / "init_record.json", doc)
    status.mark("INIT_RECORD_WRITTEN")
    return {"init_record": doc, "actions": actions}


def held(ident: dict, rec: dict | None = None, prev: dict | None = None, present=None) -> dict[str, dict]:
    """What a build attempt starting from ``rec`` and ``prev`` (a previous attempt's ``build.json`` and
    ``relay.json`` as its ``--resume`` carries them; both empty for a first run) takes over from the relay instead of
    putting it again, by group: `run`'s own lookups (a product of this identity in the build record: `relay_copy`;
    else one the relay records under this identity: `recover`), each counted only while ``present(record)`` (the
    caller's read of the relay's paths) finds its files there. No side effect; the weights are never held without
    the statistics (`run` rebuilds weights whose statistics are made again). Raises `DriverError` when the relay
    listing is unreadable: what it holds is then unknown, never empty."""
    rec, prev = copy.deepcopy(rec or {}), copy.deepcopy(prev or {})
    present = present or (lambda r: True)
    out: dict[str, dict] = {}
    st = rec.get("statistics") or {}
    if st and not st.get("saved_utc"):
        rec.pop("statistics")
    elif st and identity_problems(st.get("identity"), ident, "statistics"):
        rec = {}
    r = relay_copy("statistics", rec, prev, ident) if rec.get("statistics") else recover("statistics", rec, prev, ident)
    if r is None or not present(r):
        return out
    out["statistics"] = r
    if rec.get("checkpoint") and _checkpoint_problems(rec, ident):
        rec.pop("checkpoint")
        prev.pop("init", None)
    r = relay_copy("init", rec, prev, ident) if rec.get("checkpoint") else recover("init", rec, prev, ident)
    if r is not None and present(r):
        out["init"] = r
    return out


def init_record(rec: dict, relay: dict, ident: dict) -> dict:
    """The provenance record a training plan binds (`initialization_provenance.read`: ``root_teacher``,
    ``commit.selected_rows``), under which identity it was made (``ident``: written only once every product carries
    it, `product_problems`) and where every durable byte is."""
    ck = rec["checkpoint"]
    return {"schema": SCHEMA, "status": "COMPLETE", "arm": ARM, "run_id": rec["run_id"], "root_teacher": rec["root"],
            "commit": {"selected_rows": [{"state_id": f"{ARM}/{ck['artifact_digest'][:32]}",
                                          "path": ("SOB1's depth selection, long removed runs merged into new layers "
                                                   "(M-act over each run), short runs V1's" if is_split() else
                                                   "SOB1's depth selection, alpha-merged (M-act) at its spans"),
                                          "artifact_digest": ck["artifact_digest"],
                                          "single_shard_sha256": ck["shards"][0]["sha256"]}]},
            "identity": ident, "checkpoint": ck, "base": design()["baseline"]["init_record"],
            "method": _checkpoint_method(ck), "method_identity": rec["method_identity"],
            "statistics": rec["statistics"],
            "checks": rec["checks"], "candidate": rec["candidate"], "state_evaluation": rec.get("state_evaluation"),
            "diagnostics": rec.get("diagnostics"), "relay": relay, "attempts": rec.get("attempts", []),
            "superseded": rec.get("superseded", []), "finished_utc": utc()}


def _checkpoint_method(ck: dict) -> str | None:
    """The merge method the checkpoint's own identity names: what the init record's ``method`` must say."""
    return (((ck.get("identity") or {}).get("parts") or {}).get("merge") or {}).get("method")


def product_problems(doc: dict | None, ident: dict | None = None) -> list[str]:
    """Why ``doc`` is not a durable init record of THIS build (empty when it is): COMPLETE for this arm, each relay
    group verified with exactly the record's files AND handed off, the weights the checkpoint's shard, made from the
    recorded statistics, the equivalence checks within the design's tolerance and, given ``ident`` (`identity` of
    this contract), each product, its relay record and the record itself under it."""
    if doc is None:
        return ["no init record"]
    if doc.get("status") != "COMPLETE" or doc.get("arm") != ARM:
        return [f"the init record is {doc.get('status')!r} for {doc.get('arm')!r}"]
    bad = []
    relay = doc.get("relay") or {}
    for g, files in products(doc).items():
        r = relay.get(g) or {}
        if not files or r.get("verified") is not True or r.get("files_sha256") != files:
            bad.append(f"{g}: the relay copy is not exactly {sorted(files)} verified")
        elif not handed_off(r):
            bad.append(f"{g}: the relay copy is verified but not handed off (its record beside it, the listing "
                       "naming it)")
    ck = doc["checkpoint"]
    if ck["files_sha256"].get("model.safetensors") != ck["shards"][0]["sha256"]:
        bad.append("the manifest's weights are not the checkpoint's shard")
    if ck.get("statistics_sha256") != (doc.get("statistics") or {}).get("tensors_sha256"):
        bad.append("the weights were not made from the recorded statistics")
    if doc.get("method") != _checkpoint_method(ck):
        bad.append(f"the init record says method {doc.get('method')!r}, its checkpoint was built with "
                   f"{_checkpoint_method(ck)!r}")
    eq = (doc.get("checks") or {}).get("equivalence") or {}
    tol = float(design()["checks"]["fold_equivalence"]["tolerance"])
    if not (eq.get("passed") is True and float(eq.get("worst", float("inf"))) <= tol):
        bad.append(f"the equivalence checks are not passed within the design's tolerance {tol}")
    if ident is not None:
        bad += identity_problems((doc.get("statistics") or {}).get("identity"), ident, "statistics")
        bad += identity_problems(ck.get("identity"), ident, "checkpoint")
        whole = doc.get("identity") or {}
        if (whole.get("statistics"), whole.get("checkpoint")) != (ident["statistics"], ident["checkpoint"]):
            bad.append("the init record is not written under this identity")
        for g in GROUPS:
            if (relay.get(g) or {}).get("identity_sha256") != ident[WHAT[g]]:
                bad.append(f"{g}: its relay copy is recorded under another identity")
    elif not doc.get("identity"):
        bad.append("the init record carries no identity")
    return bad


def durable_init(out: Path, ident: dict | None = None) -> dict | None:
    p = Path(out) / ARM / "init_record.json"
    doc = read_json(p) if p.is_file() else None
    return doc if doc is not None and not product_problems(doc, ident) else None


#: the device a build runs on (`session.command`: ``--device cuda``); the session's config hash, part of the
#: calibration's identity, includes it
BUILD_DEVICE = "cuda"


def current_identity(work: Path, device: str = BUILD_DEVICE) -> dict:
    """This tree's identity of the build's products as the pod derives it ($0, no GPU: `contract` then `identity`,
    for the build's device -- only its name enters the session config), for a dev-box check of a record (the session's
    closeout, the training plan's derivation, a takeover's room)."""
    return identity(contract(Path(work), device))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", default="check")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--status", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--arm", default="depth_merge", choices=sorted(SCHEMAS))
    a = ap.parse_args(argv)
    bind(a.arm)
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
        report["identity"] = identity(report["contract"])
        write_json(out / "report.json", report)
        status.mark("CONTRACT_OK")
        if a.check_only:
            return 0
        res = run(out, work, a.device, status, a.run_id, report["identity"])
        report["init"], report["actions"] = res["init_record"]["commit"], res["actions"]
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
