#!/usr/bin/env python3
"""DMR1 v1-new (arm ``depth_merge_split``): long removed runs merged into a NEW layer of their own -- the instance.

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python scripts/stages/stage-1/depth_merge/split.py \\
        {groups | params | design [--write] | config [--write]}

The rule (the maintainer's brief of 2026-11-28), from SOB1's committed depth selection (16 of 36 kept), never searched
again. For each kept block and the run of removed blocks after it (the threshold counts removed blocks only):
- **no removed block:** the kept block, as it is;
- **a run of 1-2:** absorbed into the kept block, exactly as variant 1 (V1) merged it;
- **a run of 3 or more:** the kept block stays its original weights, and the run becomes ONE NEW layer right after
  it, merged from the run's own teacher layers only.

The core derives the groups (`layer_merge.groups_from_kept_blocks`, the threshold the design's); nothing names a
layer here. At this selection, two runs are split off (1-based: keep 13 + new M(14-18), keep 25 + new M(26-29)):
18 layers, 2,205,713,408 parameters.

How each layer is initialized (`build_candidate`), and what is reused by identity:
- a **kept** group: the base's block (SOB1's depth init, rebuilt from the teacher and checked against the base's
  per-tensor digests);
- an **absorbed** group: V1's built block, byte for byte, from V1's verified init (its tensors checked against V1's
  file). These spans, anchors, statistics and alignments are V1's, so the merge is V1's. It is never recomputed;
- a **separate** group: M-act (RI-LM1's alpha, unchanged) over the run's blocks only -- importance normalized within
  the run, the run's first block the alignment anchor, its own statistics pass and alignments (`new_spans`). It is
  never V1's span block with the kept block put back in front.

Every layer runs once; no recurrence, no hidden-state loss. "The kept layer unchanged" describes the initialization
only. Recovery is SOB1's depth recipe (Stage 3) with SOB1's trainable patterns (`TRAINABLE`): one seed, one final, both
evaluation modes.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.depth_merge import common as DC  # noqa: E402
from stages.depth_merge import merge as M  # noqa: E402

ARM = "depth_merge_split"
SPEC = REPO / "configs/stages/stage-1/depth_merge/split.json"
DESIGN = REPO / "configs/stages/stage-1/depth_merge/split_design.json"
CONFIG = REPO / "configs/stages/stage-1/depth_merge/split_config.json"
BASE_CONFIG = REPO / "configs/stages/stage-1/depth_recurrence/base_config.json"


class SplitError(RuntimeError):
    pass


def read_json(p: Path):
    return json.loads(Path(p).read_text())


def spec() -> dict:
    """The v1-new facts the maintainer fixed (`split.json`): the threshold, the seed, the label, the question."""
    return read_json(SPEC)


# ---- the groups --------------------------------------------------------------------------------------------------

def groups(v1: dict | None = None) -> list[dict]:
    """The 18 groups in student order, from SOB1's committed selection (V1's design pins it) and the threshold."""
    from aadistill.initialization.planning.layer_merge import groups_from_kept_blocks
    v1 = v1 or DC.design()
    return groups_from_kept_blocks(int(v1["baseline"]["mapping"]["n_root_blocks"]), M.kept_blocks(v1),
                                   separate_run_min=int(spec()["separate_run_min"]))


def new_spans(g: list[dict] | None = None) -> list[tuple[int, ...]]:
    """The separate groups: what this build merges (its statistics pass, its alignments)."""
    return [tuple(x["layers"]) for x in (g or groups()) if x["kind"] == "separate"]


def absorbed_spans(g: list[dict] | None = None) -> list[tuple[int, ...]]:
    return [tuple(x["layers"]) for x in (g or groups()) if x["kind"] == "absorbed"]


def student_layers(g: list[dict] | None = None) -> list[int]:
    """Per student layer, the teacher block its weights start from (the group's anchor)."""
    return [int(x["anchor"]) for x in (g or groups())]


def groups_problems(v1: dict | None = None) -> list[str]:
    """The absorbed groups must be V1's merged spans exactly (their blocks are reused from V1's init), and every root
    block in exactly one group."""
    v1 = v1 or DC.design()
    g = groups(v1)
    bad = []
    v1_spans = [tuple(s) for s in M.merge_spans(v1)]
    for s in absorbed_spans(g):
        if s not in v1_spans:
            bad.append(f"the absorbed group {list(s)} is not one of V1's merged spans")
    if sorted(b for x in g for b in x["layers"]) != list(range(int(v1["baseline"]["mapping"]["n_root_blocks"]))):
        bad.append("the groups do not cover every root block exactly once")
    for x in g:
        if x["kind"] == "separate" and x["anchor"] in M.kept_blocks(v1):
            bad.append(f"the separate group {list(x['layers'])} is anchored at a kept block")
    return bad


# ---- the config and the parameter count --------------------------------------------------------------------------

def config_text(base_text: str | None = None) -> str:
    """The 18-layer config: the base's config file (SOB1's depth init and V1's, sha256 the design's) with
    ``num_hidden_layers`` and ``layer_types`` for the student's layers -- nothing else changes."""
    base_text = BASE_CONFIG.read_text() if base_text is None else base_text
    want = DC.design()["baseline"]["init_record"]["config_json_sha256"]
    if hashlib.sha256(base_text.encode()).hexdigest() != want:
        raise SplitError("base_config.json is not the base's config file")
    cfg = json.loads(base_text)
    n = len(student_layers())
    kinds = set(cfg["layer_types"])
    if len(kinds) != 1:
        raise SplitError(f"the base's layer types {sorted(kinds)} are not one kind: a new layer's type is not defined")
    cfg["num_hidden_layers"] = n
    cfg["layer_types"] = [cfg["layer_types"][0]] * n
    return json.dumps(cfg, indent=2, sort_keys=True) + "\n"


def parameter_count(text: str | None = None) -> int:
    """Counted by instantiating the config on the meta device (transformers' own model class): not arithmetic."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "config.json").write_text(text or config_text())
        cfg = AutoConfig.from_pretrained(tmp)
    with torch.device("meta"):
        m = AutoModelForCausalLM.from_config(cfg)
    return int(sum(p.numel() for p in m.parameters()))


def checkpoint_bytes(text: str | None = None) -> dict:
    """SOB1's size rule (`single_operator_budget.structures`: init 2P, final 4P, save point 4P + 8T bytes, P the
    parameters, T the trainable ones -- the embedding, tied with the head, is frozen by the recipe) for this config."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "config.json").write_text(text or config_text())
        cfg = AutoConfig.from_pretrained(tmp)
    with torch.device("meta"):
        m = AutoModelForCausalLM.from_config(cfg)
    n = int(sum(q.numel() for q in m.parameters()))
    trainable = n - int(m.get_input_embeddings().weight.numel())
    return {"init_bf16": 2 * n, "final_fp32": 4 * n, "save_point": 4 * n + 8 * trainable,
            "num_parameters": n, "trainable_parameters": trainable,
            "rule": "derived: 2P / 4P / 4P + 8T (single_operator_budget.structures' rule), P and T from the config"}


# ---- the candidate -------------------------------------------------------------------------------------------------

def v1_layer_tensors(v1_init: Path, v1_layer: int) -> dict:
    """One layer's tensors of V1's init file by its own index (``model.layers.<i>.<name>`` -> tensor)."""
    from safetensors import safe_open
    out = {}
    with safe_open(str(Path(v1_init) / "model.safetensors"), framework="pt") as f:
        for n in f.keys():
            if n.startswith(f"model.layers.{v1_layer}."):
                out[n.split(".", 3)[3]] = f.get_tensor(n)
    if not out:
        raise SplitError(f"V1's init has no layer {v1_layer}")
    return out


def build_candidate(adapter, teacher, stats, d_digests: dict, v1_init: Path, *, seed: int,
                    g: list[dict] | None = None, base_kept: list[int] | None = None):
    """The candidate (module docstring) and its record; every tensor's provenance checked:
    - kept groups: equal the base's tensor digests (by the base's own layer index);
    - absorbed groups: equal V1's init tensors of that span's layer;
    - separate groups: rewritten by M-act over the run, and different from the anchor's original teacher block;
    - the embedding and the final norm (the tied head with them): the base's.
    ``g`` and ``base_kept`` (the groups, the base's kept blocks) default to this round's; a test passes a toy's."""
    import torch
    from aadistill.initialization.planning import frozen_mapping as FM
    from aadistill.initialization.planning import layer_merge as LM
    LMX = M.lmx()
    if g is None:
        v1 = DC.design()
        g, base_kept = groups(v1), M.kept_blocks(v1)
    base_kept = [int(b) for b in base_kept]
    layers = student_layers(g)
    tblocks = adapter.blocks(teacher)

    def names(model, j):
        return [n for n in model.state_dict() if n.startswith(f"model.layers.{j}.")]

    def same(a, b):
        return torch.equal(a, b.to(device=a.device, dtype=a.dtype))

    with torch.no_grad():
        model = FM.keep_blocks(adapter, teacher, layers, seed=seed)
        blocks = adapter.blocks(model)
        rec = {"method": M.METHOD, "groups": [], "checks": {}}
        v1_tensors = {}
        for j, x in enumerate(g):
            row = {"layer": j, "layers": list(x["layers"]), "anchor": int(x["anchor"]), "kind": x["kind"]}
            if x["kind"] == "absorbed":
                i = base_kept.index(x["anchor"])
                v1_tensors[j] = v1_layer_tensors(v1_init, i)
                sd = model.state_dict()
                for name, v in v1_tensors[j].items():
                    dst = sd[f"model.layers.{j}.{name}"]
                    dst.copy_(v.to(device=dst.device, dtype=dst.dtype))
                row["from"] = f"V1's init layer {i}"
            elif x["kind"] == "separate":
                tensors, how = LMX.merged_tensors(adapter, teacher, tuple(x["layers"]), M.METHOD, stats)
                LM.write_block(adapter, blocks[j], tensors)
                row["from"] = "M-act over the run's blocks only"
                row["weighting"] = how
            else:
                row["from"] = f"the base's layer {base_kept.index(x['anchor'])}"
            rec["groups"].append(row)
        sd = model.state_dict()
        kept_ok, absorbed_ok, separate_differ, outside_ok = [], [], [], []
        for j, x in enumerate(g):
            if x["kind"] == "kept":
                i = base_kept.index(x["anchor"])
                kept_ok.append(all(FM.tensor_sha256(sd[n]) == d_digests[n.replace(f"model.layers.{j}.",
                                                                                   f"model.layers.{i}.", 1)]
                                   for n in names(model, j)))
            elif x["kind"] == "absorbed":
                absorbed_ok.append(all(same(sd[f"model.layers.{j}.{k}"], v) for k, v in v1_tensors[j].items())
                                   and len(v1_tensors[j]) == len(names(model, j)))
            else:
                orig = tblocks[x["anchor"]].state_dict()
                separate_differ.append(any(not same(sd[f"model.layers.{j}.{k}"], v) for k, v in orig.items()))
        tied = bool(getattr(model.config, "tie_word_embeddings", False))
        for n, tt in sd.items():
            if n.startswith("model.layers.") or (tied and n == "lm_head.weight"):
                continue
            outside_ok.append(FM.tensor_sha256(tt) == d_digests.get(n))
        rec["checks"] = {"kept_layers_equal_base": all(kept_ok), "n_kept": len(kept_ok),
                         "absorbed_layers_equal_v1": all(absorbed_ok), "n_absorbed": len(absorbed_ok),
                         "separate_layers_differ_from_anchor": all(separate_differ), "n_separate": len(separate_differ),
                         "embedding_norm_head_equal_base": bool(outside_ok) and all(outside_ok),
                         "num_parameters": int(sum(q.numel() for q in model.parameters())),
                         "num_hidden_layers": len(layers)}
        c = rec["checks"]
        if not (c["kept_layers_equal_base"] and c["absorbed_layers_equal_v1"] and c["separate_layers_differ_from_anchor"]
                and c["embedding_norm_head_equal_base"]):
            raise SplitError(f"the candidate's provenance does not hold: {c}")
    return model, rec


# ---- the design, derived from V1's --------------------------------------------------------------------------------

#: the only top-level keys of V1's design that v1-new's design changes (`design_doc`); every other key is V1's
CHANGED = ("schema", "status", "recorded_utc", "question", "decided_by", "baseline", "merge", "statistics", "reuse",
           "arms", "training", "evaluation", "analysis", "storage", "checks", "variant")

#: what recovery trains: SOB1's depth unit's trainable patterns, inherited field by field (`plan.py`), nothing added
TRAINABLE = ("SOB1's trainable patterns, unchanged: every block's attention (q/k/v/o projections, q/k norms), MLP "
             "(gate/up/down projections) and both layer norms, and the final norm, train; the embedding, tied with the "
             "head, stays frozen; no further freezing")
#: V1's wording this design replaces: two seeds and "the same trainable patterns" (`design_doc` refuses to run if
#: V1's text no longer holds it, so a changed V1 is never silently half-rewritten)
_V1_SEEDS = "seeds 20260726 and 20260801 in that order"
_V1_TRAINABLE = "the same trainable patterns (embedding / tied head frozen)"


def design_doc() -> dict:
    """v1-new's design: V1's design with exactly `CHANGED` replaced (a test compares every other key)."""
    v1 = DC.design()
    s = spec()
    g = groups(v1)
    d = copy.deepcopy(v1)
    d["schema"] = "aadistill.stage1.depth_merge_split.design/v1"
    d["variant"] = {"id": s["label"], "arm": ARM, "derived_from": {"path": str(DC.DESIGN.relative_to(REPO)),
                                                                   "sha256": M.sha256_file(DC.DESIGN)},
                    "spec": {"path": str(SPEC.relative_to(REPO)), "sha256": M.sha256_file(SPEC)},
                    "rule": "V1's design with only the keys `split.CHANGED` names replaced"}
    d["status"] = "PREPARED at $0 (2026-11-28), READY_FOR_REVIEW: nothing granted, issued, built or run"
    d["recorded_utc"] = s["recorded_utc"]
    d["question"] = s["question"]
    d["decided_by"] = s["decided_by"]
    d["merge"] = {
        "method": "alpha_importance_merge: RI-LM1's M-act, unchanged (the brief keeps alpha; no beta, no weight search)",
        "groups": {"rule": ("layer_merge.groups_from_kept_blocks(36, kept_blocks, separate_run_min="
                            f"{s['separate_run_min']}): each kept block keeps its run of 1-2 removed blocks (V1's "
                            "span); a run of >= 3 removed blocks (the kept block not counted) becomes a NEW layer "
                            "after the unchanged kept block, merged from the run's own blocks"),
                   "separate_run_min": int(s["separate_run_min"]),
                   "table": [{"layers": list(x["layers"]), "anchor": int(x["anchor"]), "kind": x["kind"]} for x in g],
                   "n_layers": len(g), "absorbed": [list(x) for x in absorbed_spans(g)],
                   "separate": [list(x) for x in new_spans(g)],
                   "one_based": {"separate": [[b + 1 for b in x] for x in new_spans(g)],
                                 "kept_before_separate": [x["anchor"] + 1 for i, x in enumerate(g)
                                                          if i + 1 < len(g) and g[i + 1]["kind"] == "separate"]}},
        "formula": ("for a separate run R: theta~ = sum_{i in R} alpha_i A_{r0<-i}(theta_i), alpha_i = I_i / sum_{k in R} "
                    "I_k per parameter slot, r0 the run's first block (the alignment anchor); the kept block before R "
                    "is not part of it. An absorbed span is V1's merge, unchanged"),
        "slots": v1["merge"]["slots"],
        "never": "V1's merged block of a long span with the kept block put back in front of it",
        "execution": ("every layer once: no recurrence, no hidden-state loss. The kept layer before a separate run is "
                      "unchanged at INITIALIZATION only; recovery trains " + TRAINABLE),
    }
    d["statistics"] = {**v1["statistics"],
                       "spans": [list(x) for x in new_spans(g)],
                       "pairs": [list(p) for p in M.pairs(new_spans(g))],
                       "blocks": sorted({b for x in new_spans(g) for b in x}),
                       "rule": "one teacher forward over V1's calibration (the same profile, positions and session) for "
                               "the separate runs' blocks and pairs only; the absorbed spans' statistics are V1's and are "
                               "not recomputed (their blocks are reused whole)"}
    d["reuse"] = {**v1["reuse"],
                  "reused": [*v1["reuse"]["reused"][:3],
                             "SOB1's depth recovery recipe and plan; the same-seed (20260726) controls: SOB1's depth_only "
                             "and V1's depth_merge, their registered finals and sealed sets (never retrained or "
                             "regenerated)",
                             "V1's calibration statistics session's profile and positions; V1's built blocks for the "
                             "absorbed groups (v1_init)"],
                  "recomputed": [f"the statistics of the {len({b for x in new_spans(g) for b in x})} blocks and the "
                                 f"{len(M.pairs(new_spans(g)))} pairs of the separate runs only",
                                 "their permutations", "the separate layers' merged tensors"],
                  "v1_init": {"what": "the absorbed groups' blocks, byte for byte", "init_record": v1["reuse"].get(
                      "init_record") or "logs/stages/stage-1/depth_merge/plans/training_plan_depth_merge.json :: "
                                        "init_record", "checked": "each reused tensor equal to V1's init file's"},
                  "base": "SOB1's depth init (rebuilt from the teacher, checked against its per-tensor digests) for the "
                          "kept groups, the embedding, the final norm and the tied head"}
    d["arms"] = {ARM: {"label": s["label"] + ": depth only, long removed runs merged into a new layer",
                       "geometry": {"num_hidden_layers": len(g)}, "num_parameters": int(s["num_parameters"]),
                       "unchanged": v1["arms"]["depth_merge"]["unchanged"], "build": ["MERGE_SPLIT"],
                       "execution": "each layer once", "seeds": [int(x) for x in s["seeds"]],
                       "config": str(CONFIG.relative_to(REPO)), "checkpoint_bytes": checkpoint_bytes()}}
    t = copy.deepcopy(v1["training"])
    if _V1_SEEDS not in t["recipe"] or _V1_TRAINABLE not in t["recipe"]:
        raise SplitError("V1's training.recipe no longer reads as split.py expects; derive v1-new's wording anew")
    t["recipe"] = t["recipe"].replace(_V1_SEEDS, "seed 20260726 only (one seed: training.single_seed)").replace(
        _V1_TRAINABLE, TRAINABLE)
    t["seeds"] = [int(x) for x in s["seeds"]]
    t["unit_order"] = {"arms": [ARM], "seeds": [int(x) for x in s["seeds"]]}
    t["plans"] = f"logs/stages/stage-1/depth_merge/plans/training_plan_{ARM}.json (plan.py, from the build's init record)"
    t["single_seed"] = s["single_seed_rule"]
    t["memory"] = {**t["memory"], "expected": s["memory_estimate"]}
    d["training"] = t
    d["analysis"] = s["analysis"]
    d["baseline"] = {**v1["baseline"],
                     "what": ("SOB1's depth-only arm: its TRAINING-PRE initialization is this variant's base (the kept "
                              "layers, the embedding, the final norm, the tied head); its seed-20260726 final and sealed "
                              "eval-set-1 sets are control D (reused, never retrained or regenerated)")}
    d["evaluation"] = {**v1["evaluation"],
                       "models": {"post_recovery": f"{ARM}'s one seed (20260726) after recovery: one final, both "
                                                   "modes (H-T, H-N), every item",
                                  "control": "D = SOB1 depth_only s20260726 and M = V1 depth_merge s20260726: their "
                                             "registered finals and sealed sets, reused"}}
    st = copy.deepcopy(v1["storage"])
    st["phase_2_finals"] = {
        "placement": {ARM: "relay"},
        "_placement": (f"{ARM}'s one fp32 final ({checkpoint_bytes()['final_fp32']:,} B) stays at its relay prefix, "
                       "checked there by LFS sha256 at closeout (`launch.retain`); nothing is downloaded: the dev box is "
                       "shared and has about 7 GiB free. It is kept with the init and the evaluation evidence for the "
                       "maintainer's review (2026-11-29) and is not retired ahead of any later round's storage")}
    st["relay_rotation"] = "keep (the final is not rotated off; a later round's room is planned against the live headroom)"
    st["storage_plan"] = ("logs/stages/stage-1/depth_merge/plans/programme_storage_plan.json (storage.py --variant "
                          "programme --write: v1-new's sessions' needs against the relay's live headroom)")
    d["storage"] = st
    d["checks"] = {**v1["checks"],
                   "base": "the base rebuilt from the teacher is compared tensor by tensor with V1's base digests before "
                           "any merge; the kept layers, the embedding, the final norm and the tied head must equal them",
                   "provenance": "the absorbed layers equal V1's init tensors; the separate layers differ from their "
                                 "anchor's original block (split.build_candidate)",
                   "config": f"the candidate's config.json equals {CONFIG.relative_to(REPO)} (the base's file with 18 "
                             "layers), compared file with file",
                   "parameters": f"{int(s['num_parameters']):,} (split.parameter_count, the config on the meta device)"}
    return d


def render(doc) -> str:
    return json.dumps(doc, indent=1) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("groups")
    sub.add_parser("params")
    for name in ("design", "config"):
        c = sub.add_parser(name)
        c.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "groups":
        g = groups()
        print(json.dumps({"groups": [[list(x["layers"]), x["kind"]] for x in g], "n_layers": len(g),
                          "problems": groups_problems()}))
        return 0 if not groups_problems() else 1
    if a.cmd == "params":
        n = parameter_count()
        print(json.dumps({"num_parameters": n, "registered": int(spec()["num_parameters"]),
                          "equal": n == int(spec()["num_parameters"])}))
        return 0 if n == int(spec()["num_parameters"]) else 1
    path, text = (DESIGN, render(design_doc())) if a.cmd == "design" else (CONFIG, config_text())
    if a.write:
        path.write_text(text)
        print(path.relative_to(REPO), hashlib.sha256(text.encode()).hexdigest())
        return 0
    same = path.is_file() and path.read_text() == text
    print(f"{path.relative_to(REPO)} is the derivation" if same else f"REFUSED: {path.relative_to(REPO)} is not")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
