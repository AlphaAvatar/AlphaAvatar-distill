#!/usr/bin/env python3
"""A_teacher_parent's two Stage-3 units, derived from the frozen A-MIX-6cat-sr plan ($0, after the init session).

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python \\
        scripts/stages/stage-1/teacher_parent_initialization/plan.py --init-record <committed init_record.json> --write
    ... plan.py --check          # the committed plan is what a fresh derivation writes

Each unit is the A-MIX-6cat-sr unit of the same seed with exactly `design.json :: training.changed_fields` replaced:
the unit and arm names, the run name, the student path (where the pod places A's initialization), the output
directory, and ``lineage.initialization`` (A's weights sha256 and its committed init record as provenance, the same
root teacher). Every other field is compared with the A-MIX unit and must be equal (`differences`), so the recipe,
data, pack, tokenizer, schedule, save points, evaluations and seeds are A-MIX's by construction. The training
execution stage stays ``stage3_recovery``.
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

DESIGN = REPO / "configs/stages/stage-1/teacher_parent_initialization/design.json"
OUT = REPO / "logs/stages/stage-1/teacher_parent_initialization/plans/training_plan.json"
SCHEMA = "aadistill.stage1.teacher_parent_initialization.training_plan/v1"
ARM = "A_teacher_parent"
B_ARM = "A-MIX-6cat-sr"
#: where the pod keeps the round's bytes (container disk; no network volume)
POD_ROOT = "/workspace/teacher_parent"
INIT_ON_POD = f"{POD_ROOT}/init"
UNITS_DIR = "units"


class PlanError(RuntimeError):
    pass


def sha256_file(p: Path) -> str:
    from stages.teacher_parent_initialization.relay import sha256_file as _s
    return _s(p)


def unit_name(seed: int) -> str:
    return f"{ARM}-s{int(seed)}"


def differences(a, b, path: str = "") -> list[str]:
    """Every leaf path at which ``a`` and ``b`` differ (pure, tested)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}.{k}" if path else str(k))
            else:
                out += differences(a[k], b[k], f"{path}.{k}" if path else str(k))
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [x for i, (u, v) in enumerate(zip(a, b)) for x in differences(u, v, f"{path}[{i}]")]
    return [] if a == b else [path]


def derive(init_record: Path, root: Path = REPO) -> dict:
    d = json.loads(DESIGN.read_text())
    tr = d["training"]
    src_path = root / tr["derived_from"]["path"]
    if sha256_file(src_path) != tr["derived_from"]["sha256"]:
        raise PlanError(f"{src_path} is not the A-MIX plan the design binds")
    src = json.loads(src_path.read_text())
    rel = str(Path(init_record).resolve().relative_to(Path(root).resolve()))
    rec_sha = sha256_file(root / rel)
    rec = json.loads((root / rel).read_text())
    if rec.get("status") != "COMPLETE" or rec.get("arm") != ARM:
        raise PlanError(f"{rel} is not a COMPLETE {ARM} init record")
    if not all(v.get("verified") for v in (rec.get("relay") or {}).values()):
        raise PlanError(f"{rel}: the initialization's relay copies are not all verified")
    from shared.training.initialization_provenance import read as read_provenance
    sha = rec["checkpoint"]["shards"][0]["sha256"]
    prov = read_provenance(root / rel, rec_sha, sha)
    if (prov["root_teacher_id"], prov["root_teacher_sha256"]) != (d["teacher"]["root_teacher_id"],
                                                                  d["teacher"]["root_teacher_sha256"]):
        raise PlanError("the init record names another root teacher")
    units, checks = [], {}
    for u in sorted((x for x in src["units"] if x["arm"] == B_ARM), key=lambda x: src["seed_order"].index(x["seed"])):
        a = copy.deepcopy(u)
        name = unit_name(u["seed"])
        a["unit"], a["arm"] = name, ARM
        c = a["config"]
        c["run_name"] = f"stage1_teacher_parent_{name}"
        c["student_path"] = INIT_ON_POD
        c["out_dir"] = f"{POD_ROOT}/{UNITS_DIR}/{name}/train"
        c["lineage"]["initialization"] = {
            "root_teacher": {"model_id": d["teacher"]["model"], "revision": d["teacher"]["revision"]},
            "model_safetensors_sha256": sha,
            "provenance": {"record": rel, "record_sha256": rec_sha}}
        a["initialization"] = {"artifact_digest": rec["checkpoint"]["artifact_digest"],
                               "relay": rec["relay"]["init"], "init_record": rel}
        diff = differences(u["config"], c)
        allowed = {f.removeprefix("config.") for f in tr["changed_fields"] if f.startswith("config.")}
        outside = [x for x in diff if not any(x == f or x.startswith(f + ".") for f in allowed)]
        if outside:
            raise PlanError(f"{name}: fields outside the declared changes differ: {outside[:6]}")
        if c["stage"] != d["training_execution_stage"]:
            raise PlanError(f"{name}: execution stage {c['stage']!r}")
        checks[name] = {"differs_from_b_unit": diff, "b_unit": u["unit"], "outside_declared": outside}
        units.append(a)
    if [u["seed"] for u in units] != [20260726, 20260801]:
        raise PlanError(f"seeds {[u['seed'] for u in units]}, not the design's")
    return {"schema": SCHEMA, "study_id": d["study_id"], "arm": ARM,
            "derived_from": tr["derived_from"], "design_sha256": sha256_file(DESIGN),
            "init_record": {"path": rel, "sha256": rec_sha, "artifact_digest": rec["checkpoint"]["artifact_digest"],
                            "model_safetensors_sha256": sha},
            "seed_order": src["seed_order"], "units_dir": UNITS_DIR, "pod_root": POD_ROOT,
            "checkpoint_policy": src["checkpoint_policy"], "units": units, "derivation_checks": checks,
            "authorizes": "nothing"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--init-record", default=None)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if a.check:
        cur = json.loads(OUT.read_text())
        text = json.dumps(derive(REPO / cur["init_record"]["path"]), indent=1) + "\n"
        same = OUT.read_text() == text
        print("the training plan is the derivation's" if same else "REFUSED: the committed plan is not the derivation's")
        return 0 if same else 1
    if not a.init_record:
        ap.error("--write needs --init-record")
    doc = derive(Path(a.init_record))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}: {[u['unit'] for u in doc['units']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
