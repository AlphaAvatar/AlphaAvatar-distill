#!/usr/bin/env python3
"""One arm's two Stage-3 units, derived from the frozen A-MIX-6cat-sr plan ($0, after phase 1 commits the arm's init).

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python \\
        scripts/stages/stage-1/single_operator_budget/plan.py --arm <arm> --init-record <committed init_record.json> --write
    ... plan.py --arm <arm> --check       # the committed plan is what a fresh derivation writes

Each unit is the A-MIX-6cat-sr unit of the same seed with exactly `design.json :: training.changed_fields` replaced:
the unit and arm names, the run name, the student path (where the pod places the arm's initialization), the output
directory, and ``lineage.initialization`` (the arm's weights sha256 and its committed init record, the same root
teacher). Every other field is compared with the A-MIX unit and must be equal, and with TPI1's A unit of that seed
(A's recovery recipe, the baseline) outside the same fields: the pack and its data proportions, random order,
sequential packing with truncation, 8,192 tokens, causal attention, batch size 2 as micro-batch 1 x 2 gradient
accumulation steps, the online KD teacher and the lineage, CE / KD weights and support, AdamW and its schedule, the
steps and save points, tokenizer and seeds -- so the recipe is Research A's by construction. The training execution
stage stays ``stage3_recovery``.

Beside the units, what TPI1's train driver reads from this plan: ``unit_order`` (`design.json :: training.unit_order`),
``durability`` (the transfer floors, retries and the memory probe), ``evaluation_protocol`` (8,192 tokens, the same
protocol as the untrained arms and the teacher), ``relay_prefix_root``, and each unit's ``bytes`` (save point and
final, from `structures.json`) for the session deadline.
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

from stages.single_operator_budget import common as C  # noqa: E402

SCHEMA = "aadistill.stage1.single_operator_budget.training_plan/v1"
SOURCE_ARM = "A-MIX-6cat-sr"
UNITS_DIR = "units"


class PlanError(RuntimeError):
    pass


def sha256_file(p: Path) -> str:
    from stages.teacher_parent_initialization.relay import sha256_file as _s
    return _s(p)


def out_path(arm: str) -> Path:
    return C.LOGS / "plans" / f"training_plan_{arm}.json"


def unit_name(arm: str, seed: int) -> str:
    return f"{arm}-s{int(seed)}"


def allowed_fields(d: dict) -> set[str]:
    return {f.removeprefix("config.") for f in d["training"]["changed_fields"] if f.startswith("config.")}


def outside(diff: list[str], allowed: set[str]) -> list[str]:
    return [x for x in diff if not any(x == f or x.startswith(f + ".") for f in allowed)]


def derive(init_record: Path, arm: str, root: Path = REPO) -> dict:
    from stages.teacher_parent_initialization.plan import differences
    d = C.design()
    if arm not in d["arms"]:
        raise PlanError(f"{arm!r} is not an arm of the design")
    tr = d["training"]
    src_path = root / tr["derived_from"]["path"]
    if sha256_file(src_path) != tr["derived_from"]["sha256"]:
        raise PlanError(f"{src_path} is not the A-MIX plan the design binds")
    src = json.loads(src_path.read_text())
    tpi1 = json.loads((root / tr["derived_from"]["baseline_plan"]).read_text())
    rel = str(Path(init_record).resolve().relative_to(Path(root).resolve()))
    rec_sha = sha256_file(root / rel)
    rec = json.loads((root / rel).read_text())
    if rec.get("status") != "COMPLETE" or rec.get("arm") != arm:
        raise PlanError(f"{rel} is not a COMPLETE {arm} init record")
    if not all(v.get("verified") for v in (rec.get("relay") or {}).values()):
        raise PlanError(f"{rel}: the initialization's relay copies are not all verified")
    from shared.training.initialization_provenance import read as read_provenance
    sha = rec["checkpoint"]["shards"][0]["sha256"]
    prov = read_provenance(root / rel, rec_sha, sha)
    if (prov["root_teacher_id"], prov["root_teacher_sha256"]) != (d["teacher"]["root_teacher_id"],
                                                                  d["teacher"]["root_teacher_sha256"]):
        raise PlanError("the init record names another root teacher")
    st = json.loads(C.STRUCTURES.read_text())["arms"][arm]
    if rec["checkpoint"].get("num_parameters_counted") != st["num_parameters"]:
        raise PlanError(f"{arm}'s init has {rec['checkpoint'].get('num_parameters_counted')} parameters, "
                        f"structures.json {st['num_parameters']}")
    allowed = allowed_fields(d)
    pod = C.pod_root()
    units, checks = [], {}
    for u in sorted((x for x in src["units"] if x["arm"] == SOURCE_ARM), key=lambda x: src["seed_order"].index(x["seed"])):
        a = copy.deepcopy(u)
        name = unit_name(arm, u["seed"])
        a["unit"], a["arm"] = name, arm
        c = a["config"]
        c["run_name"] = f"stage1_single_operator_{name}"
        c["student_path"] = f"{pod}/init/{arm}"
        c["out_dir"] = f"{pod}/{UNITS_DIR}/{name}/train"
        c["lineage"]["initialization"] = {
            "root_teacher": {"model_id": d["teacher"]["model"], "revision": d["teacher"]["revision"]},
            "model_safetensors_sha256": sha,
            "provenance": {"record": rel, "record_sha256": rec_sha}}
        a["initialization"] = {"artifact_digest": rec["checkpoint"]["artifact_digest"],
                               "relay": rec["relay"]["init"], "init_record": rel}
        a["bytes"] = {"save_point": st["checkpoint_bytes"]["save_point"], "final": st["checkpoint_bytes"]["final_fp32"],
                      "init": st["checkpoint_bytes"]["init_bf16"]}
        a["evaluations"] = [{**e, "context_tokens": d["evaluation"]["protocol"]["context"]["tokens"]}
                            for e in u["evaluations"]]
        diff = differences(u["config"], c)
        bad = outside(diff, allowed)
        if bad:
            raise PlanError(f"{name}: fields outside the declared changes differ from A-MIX's: {bad[:6]}")
        base = next(x for x in tpi1["units"] if x["seed"] == u["seed"])
        bdiff = differences(base["config"], c)
        bbad = outside(bdiff, allowed)
        if bbad:
            raise PlanError(f"{name}: fields outside the declared changes differ from TPI1's A unit: {bbad[:6]}")
        if c["stage"] != d["training_execution_stage"]:
            raise PlanError(f"{name}: execution stage {c['stage']!r}")
        b = c["batch"]
        if (b["blocks_per_step"], b["micro_blocks"], b.get("context_strategy")) != (2, 1, "cross_context"):
            raise PlanError(f"{name}: batch {b}, not batch size 2 as micro-batch 1 x 2 accumulation steps, cross_context")
        checks[name] = {"differs_from_a_mix_unit": diff, "a_mix_unit": u["unit"], "outside_declared": bad,
                        "differs_from_tpi1_unit": bdiff, "tpi1_unit": base["unit"], "outside_declared_tpi1": bbad}
        units.append(a)
    if [u["seed"] for u in units] != tr["seeds"]:
        raise PlanError(f"seeds {[u['seed'] for u in units]}, not the design's")
    order = tr["unit_order"]
    return {"schema": SCHEMA, "study_id": d["study_id"], "arm": arm,
            "derived_from": {"path": tr["derived_from"]["path"], "sha256": tr["derived_from"]["sha256"]},
            "baseline_plan": {"path": tr["derived_from"]["baseline_plan"],
                              "sha256": sha256_file(root / tr["derived_from"]["baseline_plan"])},
            "design_sha256": sha256_file(C.DESIGN),
            "init_record": {"path": rel, "sha256": rec_sha, "artifact_digest": rec["checkpoint"]["artifact_digest"],
                            "model_safetensors_sha256": sha},
            "seed_order": src["seed_order"], "unit_order": [unit_name(arm, s) for s in order["seeds"]],
            "units_dir": UNITS_DIR, "pod_root": pod, "relay_prefix_root": C.relay_root(),
            "durability": tr["durability"], "evaluation_protocol": C.protocol(),
            "checkpoint_policy": src["checkpoint_policy"], "units": units, "derivation_checks": checks,
            "authorizes": "nothing"}


def render(doc: dict) -> str:
    return json.dumps(doc, indent=1) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--init-record", default=None)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    out = out_path(a.arm)
    if a.check:
        cur = json.loads(out.read_text())
        same = out.read_text() == render(derive(REPO / cur["init_record"]["path"], a.arm))
        print("the training plan is the derivation's" if same else "REFUSED: the committed plan is not the derivation's")
        return 0 if same else 1
    if not a.init_record:
        ap.error("--write needs --init-record")
    doc = derive(Path(a.init_record), a.arm)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(doc))
    print(f"wrote {out.relative_to(REPO)}: {[u['unit'] for u in doc['units']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
