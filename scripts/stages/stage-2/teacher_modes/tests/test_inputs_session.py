"""The A1 phase's inputs session: the calibration top-up in each listed mode, then the state evaluation's queues in
order until every (type, mode) holds its scored positions and its minimum of terminated tasks; a resume generates
nothing already sealed; every real input row renders."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.teacher_modes import inputs_session as IS  # noqa: E402
from stages.teacher_modes import teacher_session as TS  # noqa: E402

TYPES = ("gsm8k", "scan_language@v1", "code")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _real(rel: str) -> list[dict]:
    return [json.loads(l) for l in (REPO / rel).read_text().splitlines() if l.strip()]


def _tiny(tmp: Path, positions: int = 6000, min_tasks: int = 4) -> Path:
    root = tmp / "repo"
    ip = json.loads((REPO / IS.INPUTS_PROTOCOL).read_text())
    se = [t for t in _real(ip["inputs"]["state_evaluation"]["path"]) if t["subtype"] in TYPES and t["queue_position"] < 12]
    top = _real(ip["inputs"]["calibration_topup"]["path"])
    cal = [t for t in top if "calibration" in t][:3] + [t for t in top if "calibration" not in t][:1]
    files = {}
    for name, rows in (("se.jsonl", se), ("topup.jsonl", cal)):
        p = root / "inputs" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
        files[name] = {"path": f"inputs/{name}", "sha256": _sha(p)}
    base = root / TS.PROTOCOL
    base.parent.mkdir(parents=True, exist_ok=True)
    base.write_text((REPO / TS.PROTOCOL).read_text())
    rules = {**ip["state_evaluation"], "positions": positions, "min_tasks": min_tasks}
    doc = {**ip, "base_protocol": {"path": str(TS.PROTOCOL), "sha256": _sha(base)},
           "inputs": {"calibration_topup": files["topup.jsonl"], "state_evaluation": files["se.jsonl"]},
           "state_evaluation": rules}
    (root / IS.INPUTS_PROTOCOL).parent.mkdir(parents=True, exist_ok=True)
    (root / IS.INPUTS_PROTOCOL).write_text(json.dumps(doc))
    return root




def test_the_inputs_session_tops_up_calibration_then_fills_every_type_and_mode_and_resumes(tmp_path):
    from stages.teacher_modes.tests.test_d4_scale_chain import FakeTeacher
    root, vol, fake = _tiny(tmp_path), tmp_path / "vol", FakeTeacher()
    auth = lambda path, repo: {"authorization_id": "a", "identity": "e" * 64, "harness_digest": "h"}  # noqa: E731
    args = ["--authorization", "a.json", "--durable-root", str(vol), "--status", str(vol / "status")]
    assert IS.main(args, run=fake, authorize=auth, repo=root) == 0
    rec = json.loads((vol / "inputs_session.json").read_text())
    assert rec["terminal"] == "ALL_DONE" and not rec["short_after"], rec.get("short_after")
    cal = [json.loads(l) for l in (root / "inputs/topup.jsonl").read_text().splitlines()]
    want = {m: sum(m in t["modes"] for t in cal) for m in IS.MODES}
    assert {m: v["tasks"] for m, v in rec["rounds"][0]["modes"].items()} == want
    for k, v in rec["final"].items():
        assert v["scored_positions"] >= 6000 and v["terminated_tasks"] >= 4, (k, v)
    # thinking-mode episodes are longer: fewer of its tasks are needed for the same positions
    assert rec["final"]["gsm8k|H-T"]["generated_tasks"] <= rec["final"]["gsm8k|H-N"]["generated_tasks"]
    assert {c[2] for c in fake.calls} == {"20261102"}                       # the session's own seed base, one pass
    n = len(fake.calls)
    assert IS.main(args, run=fake, authorize=auth, repo=root) == 0
    assert len(fake.calls) == n                                               # every sealed record is read back


def test_a_queue_that_cannot_reach_its_quota_is_exhausted_and_reported(tmp_path):
    from stages.teacher_modes.tests.test_d4_scale_chain import FakeTeacher
    root, vol = _tiny(tmp_path, positions=10 ** 6), tmp_path / "vol"
    auth = lambda path, repo: {"authorization_id": "a", "identity": "e" * 64, "harness_digest": "h"}  # noqa: E731
    args = ["--authorization", "a.json", "--durable-root", str(vol), "--status", str(vol / "status")]
    assert IS.main(args, run=FakeTeacher(), authorize=auth, repo=root) == 0
    rec = json.loads((vol / "inputs_session.json").read_text())
    assert set(rec["exhausted"]) == {f"{t}|{m}" for t in TYPES for m in IS.MODES}
    assert set(rec["short_after"]) == set(rec["exhausted"])


def test_terminated_positions_is_the_generated_tokens_of_a_naturally_terminated_turn():
    stop = [151645, 151643]
    ok = {"turns": [{"stop_reason": "eos", "output_token_ids": [5, 6, 151645]}]}
    assert IS.terminated_positions(ok, stop) == 3
    assert IS.terminated_positions({"turns": [{"stop_reason": "length_limited", "output_token_ids": [5, 6]}]}, stop) is None
    assert IS.terminated_positions({"turns": [{"stop_reason": "eos", "output_token_ids": [5, 6]}]}, stop) is None


def test_every_real_input_row_renders_into_run_battery_inputs(tmp_path):
    """The session renders each stage's inputs on the pod; a row that cannot render would end it after setup."""
    ip = json.loads((REPO / IS.INPUTS_PROTOCOL).read_text())
    se = _real(ip["inputs"]["state_evaluation"]["path"])
    top = _real(ip["inputs"]["calibration_topup"]["path"])
    c = {"protocol": json.loads((REPO / ip["base_protocol"]["path"]).read_text()),
         "tasks": {**{t["task_id"]: t for t in se},
                   **{t["task_id"]: ({k: v for k, v in t.items() if k not in ("modes", "system", "source")}
                                     if "calibration" in t else t) for t in top}}}
    files = TS.write_inputs(c, tmp_path, "render", [t["task_id"] for t in se + top])
    rows = [json.loads(l) for f in files for l in (tmp_path / "inputs" / f["file"]).read_text().splitlines()]
    assert len(rows) == len(se) + len(top) and all(r.get("messages") or r.get("calibration") for r in rows)
