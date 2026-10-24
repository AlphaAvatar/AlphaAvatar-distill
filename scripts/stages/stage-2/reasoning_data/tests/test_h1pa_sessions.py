"""H1-PA's session governance and launcher: nothing is funded without the grant, kinds never cross, setup binds.

The grant does not exist until the maintainer decides; these tests write a
synthetic one under ``tmp_path`` (never in the repository) to exercise the
envelope, the kinds and setup's branch against a real artifact.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import pytest

from aadistill.governance.authorization import AuthorizationError
from stages.reasoning_data import assets as A
from stages.reasoning_data import h1pa_governance as HG
from stages.reasoning_data import h1pa_launch as L


def write_grant(tmp_path, **over):
    g = {"granted_by": "test", "ceiling_usd": 45.0, "project_cumulative_cap_usd": 730.0, "occupancy_basis_usd": 680.67,
         "phases": {"PA2v": {"scope": "test validation"}, "PA2": {"scope": "test formal"}}, **over}
    p = tmp_path / "grant.json"
    p.write_text(json.dumps(g))
    return p


@pytest.fixture
def grant(tmp_path, monkeypatch):
    monkeypatch.setattr(HG, "GRANT", write_grant(tmp_path))


def test_without_the_grant_nothing_is_issued(tmp_path, monkeypatch):
    monkeypatch.setattr(HG, "GRANT", tmp_path / "absent.json")
    with pytest.raises(AuthorizationError, match="not authorized"):
        HG.issue("validation", "t_v", expected_usd=1.0, hard_cap_usd=2.0, out=tmp_path / "a.json")


def test_each_kind_loads_only_under_its_own_policy(tmp_path, grant):
    v = HG.issue("validation", "t_v", expected_usd=1.0, hard_cap_usd=2.0, out=tmp_path / "v.json")
    assert HG.load_for("validation", v).plan_hash == HG.study_hash()
    with pytest.raises(AuthorizationError):
        HG.load_for("formal", v)


def test_a_cap_beyond_the_project_remainder_refuses(tmp_path, grant):
    with pytest.raises(AuthorizationError, match="does not fit"):
        HG.issue("formal", "t_f", expected_usd=40.0, hard_cap_usd=60.0, out=tmp_path / "f.json")


def test_a_grant_naming_another_study_identity_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(HG, "GRANT", write_grant(tmp_path, study_identity_sha256="0" * 64))
    with pytest.raises(AuthorizationError, match="names study identity"):
        HG.issue("formal", "t_f", expected_usd=1.0, hard_cap_usd=2.0, out=tmp_path / "f.json")


def test_the_studys_own_storage_is_the_only_storage_booked():
    # H1's retired volume record is H1's; H1-PA's envelope books only h1pa_* storage records
    assert all(h["volume"] != "1980hnh75i" for h in HG.spent()["storage"])


def test_setups_branch_binds_the_artifact_on_the_pods_own_files(tmp_path, grant):
    text = (A.REPO / "scripts/shared/pod/autoinit_preflight_setup.sh").read_text()
    body = text[text.index('elif [ "$SESSION_KIND" = "stage2_h1pa" ]'):]
    code = re.search(r'/opt/train/bin/python -c "\n(.*?)\n" \|\|', body, re.S).group(1)
    code = code.replace('\\"', '"').replace("\\$", "$")
    v = HG.issue("validation", "t_s", expected_usd=1.0, hard_cap_usd=2.0, out=tmp_path / "a.json")
    env = {**os.environ, "PYTHONPATH": "src:scripts", "SESSION_AUTH_PATH": str(v), "SESSION_PLAN_HASH": HG.study_hash()}
    r = subprocess.run([sys.executable, "-c", code], cwd=A.REPO, env=env, capture_output=True, text=True)
    assert r.returncode == 0 and "validation" in r.stdout, r.stderr[-800:]
    env["SESSION_PLAN_HASH"] = "0" * 64
    assert subprocess.run([sys.executable, "-c", code], cwd=A.REPO, env=env, capture_output=True).returncode != 0


def args(*extra):
    return L.build_parser().parse_args(["--run-id", "t", "--scr", "/tmp/x", "--session-commit", "d" * 40,
                                        "--bundle", "b.bundle", "--network-volume-id", "v",
                                        "--volume-mount-path", "/runpod-volume", *extra])


def test_a_session_stages_only_what_its_units_read():
    s = L.spec(args("--phase", "PA2", "--unit", "PCA-Original-s20260801", "--unit", "D4-Original-s20260801"))
    dests = {r.dest for r in s.setup.relay_inputs}
    assert f"{L.CHECKPOINTS}/20260801" in dests and f"{L.PACKS}/e1_2p96M" in dests
    assert f"{L.CHECKPOINTS}/20260726" not in dests and f"{L.PACKS}/e1_2p96M_plus_fluid" not in dests
    st = L.study()
    pins = {r.path.rsplit("/", 1)[1]: r.sha256 for r in s.setup.relay_inputs if r.dest == f"{L.CHECKPOINTS}/20260801"}
    assert pins["model.safetensors"] == st["arms"]["PCA-Original"]["checkpoints"]["20260801"]["model_safetensors_sha256"]
    assert s.setup.env["SESSION_KIND"] == "stage2_h1pa"


def test_training_is_priced_from_the_slower_of_the_committed_rate_and_pa2v():
    base = L.step_seconds(args("--phase", "PA2"))
    assert base == pytest.approx(1.7836 * 2)
    assert L.step_seconds(args("--phase", "PA2", "--step-seconds", "5.0")) == 5.0
    assert L.step_seconds(args("--phase", "PA2", "--step-seconds", "1.0")) == base
    s = L.spec(args("--phase", "PA2", "--unit", "D4-Augmented-s20260726"))
    t = {p.name: p.minutes for p in s.budget.other_phases}
    assert t["D4-Augmented-s20260726_train"] == round(3702 * base / 60, 1)


def test_the_driver_command_runs_the_h1pa_driver_and_marks_its_terminal():
    a = args("--phase", "PA2v")
    cmd = L.spec(a).driver_command(type("C", (), {"args": a})(), None)
    assert "h1pa_driver.py" in cmd and "--validation --validation-steps" in cmd and "MARKER:$m" in cmd
    assert "--unit PCA-Original-s20260726 --unit D4-Augmented-s20260726" in cmd
