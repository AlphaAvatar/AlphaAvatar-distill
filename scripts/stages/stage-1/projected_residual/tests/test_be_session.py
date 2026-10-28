"""B-E's session as its launcher builds it: the real parser and spec, the plan inside the cap, the inputs, the binding."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

from aadistill.governance.authorization import harness_source_digest
from shared import study_engineering as SE
from support.session_specs import load_session_launcher, session_args

REPO = Path(__file__).resolve().parents[5]
L = load_session_launcher("be_launch")


def _spec(**over):
    args = session_args(L, ("--run-id", "be_spec_check"), max_price=1.09, **over)
    return args, L.spec(args)


def test_the_plan_fits_the_proposed_cap_at_the_quoted_rate_and_the_granted_ceiling():
    _args, s = _spec()
    prop = json.loads((REPO / L.STUDY.proposal).read_text())
    at_quote = s.budget.plan(price_per_hour=1.09, authorized_usd=prop["per_session_hard_cap_usd"])
    bound = SE.all_in(at_quote, 1.09, L.launch_spec().disk_gb, prop["per_session_hard_cap_usd"])
    assert bound["all_in_usd"] <= prop["per_session_hard_cap_usd"]
    #: the proposal's rate ceiling is derived from the plan, not typed beside it
    assert prop["max_price_usd_per_hour"] == bound["highest_price_inside_cap"]
    #: the runner's funds gate calls this on the real plan (a property mistaken for a method failed the
    #: first $0 dry run, 2026-10-28)
    assert s.budget.account_balance_required_usd(at_quote) == at_quote.hard_terminate_usd
    assert set(s.budget.accounted_provider_resources()) <= {"am5m373kuq"}
    top = s.budget.plan(price_per_hour=prop["max_price_usd_per_hour"], authorized_usd=prop["per_session_hard_cap_usd"])
    assert SE.all_in(top, prop["max_price_usd_per_hour"], 120)["all_in_usd"] <= prop["per_session_hard_cap_usd"]


def test_setup_gets_the_thinking_teacher_both_environments_and_no_suite():
    _args, s = _spec()
    design = json.loads((REPO / L.DESIGN).read_text())
    env = s.setup_environment(session_commit="0" * 40, bundle="b")
    assert env["SESSION_KIND"] == "stage1_projected_residual" and env["TEACHER_MODEL"] == design["teacher"]["model"]
    assert env["TEACHER_REVISION"] == design["teacher"]["revision"]
    m = s.setup.setup_markers
    assert {"TRAIN_ENV", "VLLM_READY", "TEACHER_READY", "AUTHORIZATION_OK", "SETUP_DONE"} <= set(m)
    assert "TESTS_OK" not in m and "ROPE_OK" not in m


def test_the_calibration_items_are_the_designs_and_nothing_comes_from_the_relay():
    _args, s = _spec()
    items = json.loads((REPO / L.DESIGN).read_text())["calibration"]["items_path"]
    assert [(a.repo_path, f"{a.install_to}/{a.dest_name}") for a in s.setup.local_assets] == [(items, items)]
    assert not s.setup.relay_inputs


def test_the_driver_command_runs_the_driver_offline_and_marks_a_death():
    args, s = _spec()
    cmd = s.driver_command(type("C", (), {"args": args})(), None)
    for needle in ("HF_HUB_OFFLINE=1", "/opt/train/bin/python", L.launch_spec().driver, "--vllm-python /opt/vllm/bin/python",
                   f"--status {s.status_path}", "MARKER:RUN_FAILED"):
        assert needle in cmd, needle


def test_the_binding_is_complete_and_the_setup_branch_reads_this_study():
    for f in L.STUDY.identity_files:
        assert (REPO / f).is_file(), f
    harness_source_digest(REPO, files=SE.harness_files(L.STUDY, REPO))      # every harness file exists
    assert importlib.import_module("stages.projected_residual.be_launch").STUDY == L.STUDY
    assert (REPO / L.launch_spec().artifact_spec).is_file()


def test_until_the_maintainer_grants_it_only_a_proposal_exists():
    g = REPO / L.STUDY.grant
    if g.is_file():
        doc = json.loads(g.read_text())
        assert doc["status"] == "granted" and doc["study"] == "B-E" and doc["decision"]
    else:
        prop = json.loads((REPO / L.STUDY.proposal).read_text())
        assert prop["status"] == "proposed" and prop["study"] == "B-E" and prop["gpu"] == L.GPU
        assert prop["project_cumulative_cap_usd"] == 740.0 and prop["ceiling_usd"] == 3.0
