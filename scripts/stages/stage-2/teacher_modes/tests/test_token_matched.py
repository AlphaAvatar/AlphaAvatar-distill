"""Research A v2 (token-matched) at $0: the quotas, the dev box's selection rule and the supplement's round sizing."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
for _p in ("src", "scripts", "scripts/stages/stage-1", "scripts/stages/stage-2"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.teacher_modes import build_supplement_tasks as BST  # noqa: E402
from stages.teacher_modes import supplement_session as SS  # noqa: E402
from stages.teacher_modes import token_admission as TA  # noqa: E402


def test_the_quotas_add_up_to_each_category_and_the_total():
    d = BST.design()
    q = BST.quotas(d, mbpp_cap=38_431)
    fluid = sum(v for c, v in q.items() if BST.category_of(c) == "fluid")
    base = sum(v for c, v in q.items() if BST.category_of(c) == "base_counterpart")
    assert fluid == pytest.approx(d["scale"]["categories"]["fluid"], abs=1.0)
    assert base == pytest.approx(d["scale"]["categories"]["base_counterpart"], abs=1.0)
    assert fluid + base == pytest.approx(d["scale"]["per_mode_total"], abs=2.0)
    assert q[("supporting.code_algorithmic", "mbpp_train")] == 38_431
    assert ("primary.task_contained_deduction", "prontoqa") not in q        # dropped from both modes


def _supply(cells: dict[str, dict[str, list[int]]]) -> dict:
    return {c: {f: [(f"{c}|{f}|{i}", n) for i, n in enumerate(ns)] for f, ns in fams.items()} for c, fams in cells.items()}


def test_the_selection_stops_on_the_first_task_reaching_the_target_and_tracks_each_cell_share():
    sup = _supply({"a": {"x": [100] * 50, "y": [300] * 50}, "b": {"z": [50] * 200}})
    quotas = {"a": 6000.0, "b": 4000.0}
    targets = {"a": {"x": 0.5, "y": 0.5}, "b": {"z": 1.0}}
    got = TA.select(sup, quotas, targets, 10_000.0)
    assert not got["shortfall"]
    assert 10_000 <= got["tokens"] < 10_000 + 300                          # one task's natural excess at most
    assert abs(got["by_cell"]["a"] - 6000) <= 300 and abs(got["by_cell"]["b"] - 4000) <= 300
    fa = got["by_family"]["a"]
    assert abs(fa["x"] - fa["y"]) <= 300                                   # families toward their target shares


def test_a_cell_that_runs_dry_is_a_shortfall_never_filled_from_another():
    sup = _supply({"a": {"x": [100] * 5}, "b": {"z": [100] * 500}})
    got = TA.select(sup, {"a": 5000.0, "b": 5000.0}, {"a": {"x": 1.0}, "b": {"z": 1.0}}, 10_000.0)
    assert got["shortfall"] == ["a"]
    assert got["tokens"] < 10_000
    empty = TA.select(_supply({"a": {}, "b": {"z": [10]}}), {"a": 1.0, "b": 1.0}, {"a": {}, "b": {"z": 1.0}}, 2.0)
    assert empty["shortfall"] == ["a"] and not empty["tasks"]


def _contract():
    order = [{"task_id": f"t{i}", "cell": "s/src", "cell_position": i, "slice": "S", "stratum": "s", "source": "src",
              "family_id": "dead" if i % 2 else "live"} for i in range(400)]
    order += [{"task_id": "v0", "cell": "r/gsm8k_train", "cell_position": 0, "slice": "V2", "stratum": "r",
               "source": "gsm8k_train", "family_id": "gsm8k_train"}]
    plan = {"cells": {"s/src": {"modes": {m: {"need": 1000.0, "tokens_per_generated_task": 10.0} for m in SS.MODES}}},
            "skip": {"H-T": [], "H-N": ["s/src/dead"]}}
    sup = {"pod_margin": 0.05, "sizing": 1.25, "min_batch": 8}
    return {"order": order, "plan": plan, "supplement": sup, "tasks": {t["task_id"]: t for t in order}}


def test_a_round_takes_the_next_eligible_tasks_sized_from_the_yield_and_v2_only_in_round_one():
    st = SS.State(_contract())
    ids, why = st.select("H-N", 1)
    assert ids[0] == "v0"
    assert why["s/src"]["tasks"] == len(ids) - 1 == 132                      # ceil(1000 x 1.05 / 10 x 1.25)
    assert all(int(t[1:]) % 2 == 0 for t in ids[1:])                         # the skipped family is never generated
    st.got[("H-N", "s/src")] = 2000.0
    assert st.select("H-N", 2)[0] == []                                      # the need is met: nothing more
    ids_t, _ = st.select("H-T", 2)
    assert "v0" not in ids_t and ids_t[0] == "t0" and "t1" in ids_t


# ---- the supplement session end to end: real order_v2 and calibration_v2 rows, a fake teacher -----------------------

def _tiny_supplement(tmp: Path) -> tuple[Path, dict]:
    import gzip
    import hashlib
    import json
    from stages.teacher_modes import teacher_session as TS
    root = tmp / "repo"
    order = []
    for t in BST.read_jsonl(BST.ORDER):
        if t["cell"] in ("primary.compositional_generalization/s0_generators_v2_calsplit",
                         "supporting.reading_verified/squad_v2_train_answerable") and t["cell_position"] < 40:
            order.append(t)
    cal = [json.loads(l) for l in (REPO / "data/stages/stage-2/teacher_modes/tasks/calibration_v2.jsonl").read_text()
           .splitlines()[:6]]
    cal[0]["modes"] = ["H-N", "H-T"]
    plan = {"cells": {c: {"modes": {m: {"need": 3000.0, "tokens_per_generated_task": 500.0} for m in SS.MODES}}
                      for c in {t["cell"] for t in order}}, "skip": {"H-T": [], "H-N": []}}
    files = {}
    for name, rows in (("order.jsonl.gz", order), ("cal.jsonl", cal)):
        p = root / "inputs" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        body = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode()
        p.write_bytes(gzip.compress(body, mtime=0) if name.endswith(".gz") else body)
        files[name] = {"path": f"inputs/{name}", "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    (root / "inputs/plan.json").write_text(json.dumps(plan))
    files["plan"] = {"path": "inputs/plan.json", "sha256": hashlib.sha256((root / "inputs/plan.json").read_bytes()).hexdigest()}
    base = root / TS.PROTOCOL
    base.parent.mkdir(parents=True, exist_ok=True)
    base.write_text((REPO / TS.PROTOCOL).read_text())
    sp = {"base_protocol": {"path": str(TS.PROTOCOL), "sha256": hashlib.sha256(base.read_bytes()).hexdigest()},
          "seed_base": 20261031, "pod_margin": 0.05, "sizing": 1.25, "min_batch": 8, "max_rounds": 6,
          "inputs": {"order": files["order.jsonl.gz"], "plan": files["plan"], "calibration": files["cal.jsonl"]}}
    (root / SS.SUPPLEMENT_PROTOCOL).write_text(json.dumps(sp))
    return root, {"order": order, "cal": cal}


def test_the_supplement_session_generates_calibration_then_rounds_to_the_need_and_resumes_without_regenerating(
        tmp_path, monkeypatch):
    import json
    from stages.reasoning_data import generation_adapter as GA
    from stages.teacher_modes.tests.test_d4_scale_chain import FakeTeacher, _verdict
    root, got = _tiny_supplement(tmp_path)
    monkeypatch.setattr(GA, "candidate", _verdict)
    vol, fake = tmp_path / "vol", FakeTeacher()
    auth = lambda path, repo: {"authorization_id": "a", "identity": "e" * 64, "harness_digest": "h"}  # noqa: E731
    args = ["--authorization", "a.json", "--durable-root", str(vol), "--status", str(vol / "status")]
    assert SS.main(args, run=fake, authorize=auth, repo=root) == 0
    rec = json.loads((vol / "supplement_session.json").read_text())
    assert rec["terminal"] == "ALL_DONE" and not rec["short_after"]
    assert rec["rounds"][0]["modes"]["H-N"]["tasks"] == 6 and rec["rounds"][0]["modes"]["H-T"]["tasks"] == 1
    assert all(v["pod_admitted_tokens"] >= 3000 * 1.05 for v in rec["final"].values())
    assert (vol / "inputs/order.jsonl.gz").is_file()
    assert {c[2] for c in fake.calls} >= {"20261031", "20261032"}            # the supplement's own seed base (+ pass)
    n = len(fake.calls)
    assert SS.main(args, run=fake, authorize=auth, repo=root) == 0
    assert len(fake.calls) == n                                               # every sealed record is read back
