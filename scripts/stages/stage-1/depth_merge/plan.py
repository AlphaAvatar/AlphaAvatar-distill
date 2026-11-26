#!/usr/bin/env python3
"""One DMR1 arm's two training units, derived from SOB1's committed depth plan ($0, dev box).

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python scripts/stages/stage-1/depth_merge/plan.py \\
        --arm depth_merge --init-record <its committed init_record.json> --write
    ... plan.py --arm depth_loop --init-record logs/stages/stage-6/depth_recurrence/inits/<path>/init_record.json --write
    ... plan.py --arm <arm> --check          # the committed plan is what a fresh derivation writes

Each unit is SOB1's depth unit of the same seed (``logs/stages/stage-1/single_operator_budget/plans/
training_plan_depth_only.json``, its sha256 pinned by the variant-1 design) with exactly the variant's
``training.changed_fields`` replaced; every other field is compared and must be equal, so the recipe is SOB1's depth
recipe by construction (the pack and its proportions, random order, sequential packing with truncation, 8,192 tokens,
causal attention, micro-batch 1 x 2 accumulation steps, 9,701 steps, the online KD teacher, CE / KD, AdamW and its
schedule, seeds, save points).

* **depth_merge** (variant 1): the unit and arm names, the run name, the student path, the output directory and
  ``lineage.initialization`` (the alpha-merged init's weights and its committed init record).
* **depth_loop / depth_loop_hidden** (variants 2, 3): the same, plus ``config.stage`` = the Stage-6 execution stage,
  and for depth_loop_hidden ``config.loss.hidden_state`` (the Stage-6 design's registered configuration). Their
  initialization is the recurrent init record over the SELECTED base (`depth_recurrence.schedule`): the base's
  weights byte for byte and the recurrent config, whose exact text the plan carries (``initialization.
  recurrent_config``) so the pod writes it over the base's config and checks its sha256. Refused without a COMPLETE
  selection, or for a base that is not the selection's legal path.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.depth_merge import common as DC  # noqa: E402

SCHEMA = "aadistill.dmr1.training_plan/v1"
UNITS_DIR = "units"


class PlanError(RuntimeError):
    pass


def sha256_file(p: Path) -> str:
    from stages.teacher_parent_initialization.relay import sha256_file as _s
    return _s(p)


def out_path(arm: str) -> Path:
    return DC.files_of(arm)[1] / "plans" / f"training_plan_{arm}.json"


def unit_name(arm: str, seed: int) -> str:
    return f"{arm}-s{int(seed)}"


def allowed_fields(d: dict, arm: str) -> set[str]:
    f = {x.removeprefix("config.") for x in d["training"]["changed_fields"] if x.startswith("config.")}
    if arm != "depth_loop_hidden":
        f.discard("loss.hidden_state")
    return f


def outside(diff: list[str], allowed: set[str]) -> list[str]:
    return [x for x in diff if not any(x == f or x.startswith(f + ".") for f in allowed)]


def source_plan(root: Path = REPO) -> dict:
    """SOB1's committed depth plan, refused unless it is the one the variant-1 design pins."""
    b = DC.design()["baseline"]["training_plan"]
    p = root / b["path"]
    if sha256_file(p) != b["sha256"]:
        raise PlanError(f"{p} is not SOB1's depth plan the design pins")
    return json.loads(p.read_text())


def init_problems(arm: str, rec: dict, rec_path: str, root: Path = REPO) -> list[str]:
    """Why ``rec`` is not a legal training-pre initialization of ``arm`` (empty when it is)."""
    d = DC.design_of(arm)
    if rec.get("status") != "COMPLETE":
        return [f"{rec_path} is {rec.get('status')!r}, not COMPLETE"]
    if not DC.is_recurrent(arm):
        bad = [] if rec.get("arm") == arm else [f"{rec_path} is {rec.get('arm')!r}'s init record, not {arm}'s"]
        if not all((v or {}).get("verified") for v in (rec.get("relay") or {}).values()) or not rec.get("relay"):
            bad.append(f"{rec_path}: the initialization's relay copies are not all verified")
        return bad
    from stages.depth_recurrence import schedule as SCH
    return SCH.init_record_problems(rec, arm=arm, root=root)


def derive(arm: str, init_record: Path, root: Path = REPO) -> dict:
    from shared.training.initialization_provenance import read as read_provenance
    from stages.single_operator_budget import common as SC
    from stages.teacher_parent_initialization.plan import differences
    before = DC.bind_for(arm)
    try:
        d = DC.design_of(arm)
        if arm not in d["arms"]:
            raise PlanError(f"{arm!r} is not an arm of {DC.files_of(arm)[0].name}")
        src = source_plan(root)
        rel = str(Path(init_record).resolve().relative_to(Path(root).resolve()))
        rec_sha = sha256_file(root / rel)
        rec = json.loads((root / rel).read_text())
        bad = init_problems(arm, rec, rel, root)
        if bad:
            raise PlanError("; ".join(bad))
        sha = rec["checkpoint"]["files_sha256"]["model.safetensors"]
        prov = read_provenance(root / rel, rec_sha, sha)
        if (prov["root_teacher_id"], prov["root_teacher_sha256"]) != (d["teacher"]["root_teacher_id"],
                                                                      d["teacher"]["root_teacher_sha256"]):
            raise PlanError("the init record names another root teacher")
        allowed = allowed_fields(d, arm)
        pod = d["storage"]["pod_root"]
        units, checks = [], {}
        for u in sorted(src["units"], key=lambda x: src["seed_order"].index(x["seed"])):
            a = copy.deepcopy(u)
            name = unit_name(arm, u["seed"])
            a["unit"], a["arm"] = name, arm
            c = a["config"]
            c["run_name"] = f"dmr1_{name}"
            c["student_path"] = f"{pod}/init/{arm}"
            c["out_dir"] = f"{pod}/{UNITS_DIR}/{name}/train"
            c["lineage"]["initialization"] = {
                "root_teacher": {"model_id": d["teacher"]["model"], "revision": d["teacher"]["revision"]},
                "model_safetensors_sha256": sha, "provenance": {"record": rel, "record_sha256": rec_sha}}
            a["initialization"] = {"artifact_digest": rec["checkpoint"]["artifact_digest"],
                                   "relay": rec["relay"]["init"], "init_record": rel}
            if DC.is_recurrent(arm):
                c["stage"] = d["training_execution_stage"]
                cfg_p = root / rec["execution"]["config"]["path"]
                text = cfg_p.read_text()
                if sha256_file(cfg_p) != rec["execution"]["config"]["sha256"]:
                    raise PlanError(f"{cfg_p} is not the recurrent config the init record binds")
                a["initialization"]["recurrent_config"] = {"path": rec["execution"]["config"]["path"],
                                                           "sha256": rec["execution"]["config"]["sha256"],
                                                           "text": text}
                a["initialization"]["execution"] = {k: rec["execution"][k] for k in
                                                    ("layer_to_block", "root_layer_of_layer", "n_logical_layers",
                                                     "n_parameter_blocks")}
            if arm == "depth_loop_hidden":
                c["loss"]["hidden_state"] = copy.deepcopy(d["hidden_state"]["config"])
            elif "hidden_state" in c["loss"]:
                raise PlanError(f"{name}: a hidden-state term in a unit that registers none")
            diff = differences(u["config"], c)
            out = outside(diff, allowed)
            if out:
                raise PlanError(f"{name}: fields outside the declared changes differ from SOB1's depth unit: {out[:6]}")
            b = c["batch"]
            if (b["blocks_per_step"], b["micro_blocks"], b.get("context_strategy")) != (2, 1, "cross_context"):
                raise PlanError(f"{name}: batch {b}")
            checks[name] = {"differs_from_sob1_depth_unit": diff, "sob1_unit": u["unit"], "outside_declared": out}
            units.append(a)
        if [u["seed"] for u in units] != d["training"]["seeds"]:
            raise PlanError(f"seeds {[u['seed'] for u in units]}, not the design's")
        return {"schema": SCHEMA, "study_id": d["study_id"], "arm": arm,
                "derived_from": {"path": d["training"]["derived_from"]["path"],
                                 "sha256": d["training"]["derived_from"]["sha256"]},
                "design_sha256": sha256_file(DC.files_of(arm)[0]),
                "init_record": {"path": rel, "sha256": rec_sha, "artifact_digest": rec["checkpoint"]["artifact_digest"],
                                "model_safetensors_sha256": sha},
                "seed_order": src["seed_order"], "unit_order": [unit_name(arm, s) for s in d["training"]["seeds"]],
                "units_dir": UNITS_DIR, "pod_root": pod, "relay_prefix_root": d["storage"]["relay_prefix_root"],
                "durability": d["training"]["durability"], "evaluation_protocol": SC.protocol(),
                "checkpoint_policy": src["checkpoint_policy"], "units": units, "derivation_checks": checks,
                "authorizes": "nothing"}
    finally:
        SC.restore(before)


def render(doc: dict) -> str:
    return json.dumps(doc, indent=1) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", required=True, choices=sorted(DC.ARMS))
    ap.add_argument("--init-record", default=None)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    out = out_path(a.arm)
    if a.check:
        cur = json.loads(out.read_text())
        same = out.read_text() == render(derive(a.arm, REPO / cur["init_record"]["path"]))
        print("the training plan is the derivation's" if same else "REFUSED: the committed plan is not the derivation's")
        return 0 if same else 1
    if not a.init_record:
        ap.error("--write needs --init-record")
    doc = derive(a.arm, Path(a.init_record))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(doc))
    print(f"wrote {out.relative_to(REPO)}: {[u['unit'] for u in doc['units']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
