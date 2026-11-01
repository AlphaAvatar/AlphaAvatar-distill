"""The A1 phase's search and training drivers, at $0: the root verified by its bytes, the first evaluation's critical
group and repeatability checks, the Top-1 copied by its commit's sha256, the units' state from the volume, and A1's
plan bound to a completed search record's Top-1 and root."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
for _p in ("src", "scripts", "scripts/stages/stage-1", "scripts/stages/stage-0/calibration_volume"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.teacher_modes import a1_units as AU  # noqa: E402
from stages.teacher_modes import full_search as FS  # noqa: E402


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def test_the_root_is_verified_by_its_bytes(tmp_path):
    ident = {"model": "org/m", "revision": "r" * 40,
             "weight_shards": [{"file": "a.safetensors", "lfs_sha256": _sha(b"A")},
                               {"file": "b.safetensors", "lfs_sha256": _sha(b"B")}]}
    (tmp_path / "a.safetensors").write_bytes(b"A")
    (tmp_path / "b.safetensors").write_bytes(b"B")
    got = FS.verify_root(tmp_path, ident)
    assert got["root_teacher_id"] == "org/m@" + "r" * 40 and got["verified"] == ["a.safetensors", "b.safetensors"]
    (tmp_path / "b.safetensors").write_bytes(b"X")
    with pytest.raises(FS.FullSearchError, match="b.safetensors hashes to"):
        FS.verify_root(tmp_path, ident)


def test_the_hybrid_root_identity_is_the_lineage_rule_on_the_pinned_shards():
    from aadistill.training.lineage import shard_set_sha256
    ident = json.loads(FS.TEACHER.read_text())
    rid = FS.root_identity(ident)
    assert rid["root_teacher_id"] == "Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c"
    assert rid["root_teacher_sha256"] == shard_set_sha256({s["file"]: s["lfs_sha256"] for s in ident["weight_shards"]})


def test_the_first_evaluation_must_carry_both_modes_groups_and_their_mean():
    ok = {"state.critical_token_kl.group.H-T": 0.3, "state.critical_token_kl.group.H-N": 0.1, "state.critical_token_kl": 0.2}
    assert FS.critical_problems(ok) == []
    assert "no critical group" in FS.critical_problems({"state.critical_token_kl": 0.2})[0]
    assert "is not the groups' mean" in FS.critical_problems({**ok, "state.critical_token_kl": 0.22})[0]


class _Eval:
    def __init__(self, values, noise=0.0):
        self.values, self.noise, self.n = values, noise, 0

    def evaluate(self, model, digest):
        self.n += 1
        v = {k: x + (self.noise * self.n) for k, x in self.values.items()}
        return type("E", (), {"values": v})()


def _session(evaluator):
    from aadistill.initialization.planning.ranking import PARETO_V1
    cfg = type("C", (), {"policy": PARETO_V1})()
    return type("S", (), {"evaluator": evaluator, "config": cfg})()


VALUES = {"state.teacher_kl.equal_domain_mean": 1.0, "state.teacher_kl.worst_domain": 2.0,
          "state.critical_token_kl.group.H-T": 0.3, "state.critical_token_kl.group.H-N": 0.1,
          "state.critical_token_kl": 0.2}


def test_the_first_evaluation_runs_the_frozen_repeatability_rule_and_stops_on_noise(tmp_path):
    torch = pytest.importorskip("torch")
    from support.toy import TEACHER_GEOMETRY, build_tiny_model
    from shared.epsilon_response import EPSILON_RESPONSE_V1
    model = build_tiny_model(dict(TEACHER_GEOMETRY))
    ev = _Eval(VALUES)
    m = FS.FirstEvaluationCheck(_session(ev), None, scratch=tmp_path)
    m(model, "d" * 64)
    assert ev.n == EPSILON_RESPONSE_V1.repeats and m.record["passes"] and m.record["measured_range"] == 0.0
    m(model, "d" * 64)
    assert ev.n == EPSILON_RESPONSE_V1.repeats + 1              # checked once, at the first evaluation only
    assert json.loads((tmp_path / "first_evaluation_check.json").read_text())["decision"]["passes"]
    noisy = FS.FirstEvaluationCheck(_session(_Eval(VALUES, noise=1e-3)), None, scratch=tmp_path)
    with pytest.raises(FS.StopForReview, match="epsilon repeatability"):
        noisy(model, "d" * 64)
    assert torch is not None


def test_the_top1_copy_must_be_the_commits_bytes(tmp_path):
    src = tmp_path / "state"
    src.mkdir()
    (src / "model.safetensors").write_bytes(b"W")
    (src / "config.json").write_text("{}")
    row = {"checkpoint_path": str(src), "single_shard_sha256": _sha(b"W"), "state_id": "s"}
    got = FS.persist_top1(row, tmp_path / "vol")
    assert got["files_sha256"]["model.safetensors"] == _sha(b"W") and Path(got["path"]).name == "top1"
    with pytest.raises(FS.FullSearchError, match="the commit names"):
        FS.persist_top1({**row, "single_shard_sha256": _sha(b"other")}, tmp_path / "vol2")


def test_a_units_state_comes_from_the_volume(tmp_path):
    out, ev = tmp_path / "train", tmp_path / "evaluation"
    out.mkdir()
    assert AU.state(out, ev) == "fresh"
    (out / "checkpoints").mkdir()
    (out / "checkpoints/latest.txt").write_text("step_000826\n")
    assert AU.state(out, ev) == "resume"
    (out / "run_completion_resume_step001653.json").write_text("{}")
    assert AU.state(out, ev) == "trained"
    ev.mkdir()
    (ev / "formal_report.json").write_text("{}")
    assert AU.state(out, ev) == "done"


def test_a1s_plan_binds_the_searchs_top1_and_root_and_refuses_an_incomplete_search(tmp_path):
    from aadistill.training import lineage as LIN
    from stages.teacher_modes import training_plan as TP
    rid = FS.root_identity()
    rec = {"status": "COMPLETE", "root_teacher": {"root_teacher_id": rid["root_teacher_id"],
                                                  "root_teacher_sha256": rid["root_teacher_sha256"]},
           "commit": {"selected_rows": [{"state_id": "s1", "single_shard_sha256": "7" * 64}]},
           "top1": {"files_sha256": {"model.safetensors": "7" * 64}, "state_id": "s1", "artifact_digest": "a" * 64}}
    p = tmp_path / "logs/search_record.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(rec))
    doc = TP.a1_plan(p, root=tmp_path)
    assert [u["unit"] for u in doc["units"]] == ["A1-s20260726", "A1-s20260801"] and doc["seed_order"][0] == 20260726
    for u in doc["units"]:
        init = u["config"]["lineage"]["initialization"]
        assert u["config"]["student_path"] == TP.TOP1_ON_VOLUME and init["model_safetensors_sha256"] == "7" * 64
        assert init["provenance"]["record"] == "logs/search_record.json"
        assert u["config"]["lineage"]["mode"] == "thinking" and u["config"]["schedule"]["total_steps"] == 1653
        assert not LIN.declared(u["config"]["lineage"])
    p.write_text(json.dumps({**rec, "status": "INCOMPLETE"}))
    with pytest.raises(SystemExit, match="no Top-1 before the search completes"):
        TP.a1_plan(p, root=tmp_path)
    p.write_text(json.dumps({**rec, "root_teacher": {"root_teacher_id": "Qwen/Qwen3-4B-Thinking-2507@x",
                                                     "root_teacher_sha256": "0" * 64}}))
    with pytest.raises(SystemExit, match="recorded root"):
        TP.a1_plan(p, root=tmp_path)
