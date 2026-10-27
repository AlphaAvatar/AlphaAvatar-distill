"""P1t at $0: the frozen 59-task list, its own launch identity, its budget and its request.

No pod, no provider call, no grant: artifacts are issued in tmp_path from test
grant documents.
"""
from __future__ import annotations

import json
import subprocess

import pytest

from stages.reasoning_data import assets as A
from stages.reasoning_data import generation_driver as D
from stages.reasoning_data import governance as GOV
from stages.reasoning_data import topup_plan as T

from aadistill.governance.authorization import AuthorizationError, harness_source_digest
from support.session_specs import load_session_launcher, session_args

L = load_session_launcher("h1_generation_launch")
PLAN = A.read_json(T.PLAN)
P1_LID = A.read_json(T.P1_GRANT)["launch_identity_sha256"]


def _grant(tmp_path, phase="P1t", **over):
    g = {"granted_by": "test", "decision": "test only", "phase": phase,
         "launch_identity_sha256": GOV.launch_hash(A.REPO, GOV.PHASE_PLANS[phase]),
         "harness_digest": harness_source_digest(A.REPO, files=GOV.HARNESS)["digest"],
         "expected_usd": 1.0, "hard_cap_usd": 2.6, **over}
    p = tmp_path / f"grant_{phase}.json"
    p.write_text(json.dumps(g))
    return p


def test_the_committed_plan_is_what_the_builder_derives_byte_for_byte(tmp_path):
    """The list is the application's (its id hashes), re-drawn by P1's rule from
    P1's bound pools; the plan and its inputs rebuild identically."""
    doc = T.build(tmp_path / "inputs")
    assert json.dumps(doc, indent=1) + "\n" == T.PLAN.read_text()
    assert [A.sha256_file(tmp_path / "inputs" / f["file"]) for f in doc["inputs"]] == \
        [A.sha256_file(T.INPUTS / f["file"]) for f in PLAN["inputs"]]
    assert PLAN["totals"]["tasks"] == 59 and PLAN["totals"]["episodes"] == 0
    assert {c: v["tasks"] for c, v in PLAN["cells"].items()} == {
        "primary.abstract_analogical/s0_generators_v2_calsplit": 25,
        "primary.compositional_generalization/reasoning_gym": 34}
    assert PLAN["excludes"] == {"phase": "P1", "run_id": "h1gen_20261022_190653", "launch_identity_sha256": P1_LID,
                                "tasks": 1587, "tasks_shared": 0}
    assert PLAN["checks"]["isolation_R0_R9"]["pass"] and not PLAN["checks"]["prompts_also_in_P1"]


def test_a_draw_that_is_not_the_applications_list_refuses(tmp_path, monkeypatch):
    app = A.read_json(T.APPLICATION)
    app["cells"]["primary.compositional_generalization/reasoning_gym"]["tasks"]["tasks"] = 33
    edited = tmp_path / "application.json"
    edited.write_text(json.dumps(app))
    monkeypatch.setattr(T, "APPLICATION", edited)
    with pytest.raises(SystemExit, match="not the application's list"):
        T.build(tmp_path / "inputs")


def test_p1t_has_its_own_launch_identity_bound_to_the_frozen_corpus_and_p1s_protocol():
    ident = GOV.launch_identity(A.REPO, GOV.TOPUP_PLAN)
    p1 = GOV.launch_identity(A.REPO, GOV.PLAN)
    assert GOV.launch_hash(A.REPO, GOV.TOPUP_PLAN) != GOV.launch_hash() == P1_LID   # P1's is unchanged
    assert ident["generation_protocol"] == p1["generation_protocol"]                  # the same protocol bytes
    assert ident["pools"] == A.read_json(A.REPO / GOV.PLAN)["pools"] != p1["pools"]   # P1's pools, not its reuse
    assert ident["inputs"] != p1["inputs"]
    assert all(f["file"].startswith(str(GOV.TOPUP_PLAN.parent)) for f in ident["inputs"])
    assert PLAN["corpus"] == {"path": "configs/stages/stage-2/reasoning_data/corpus_v3.json",
                              "sha256": A.sha256_file(T.CORPUS)}


def test_the_corpus_is_the_proposal_the_application_priced_with_the_maintainers_choice():
    corpus = A.read_json(T.CORPUS)
    prop = json.loads(subprocess.run(["git", "show", f"{corpus['proposal']['commit']}:{corpus['proposal']['path']}"],
                                     cwd=A.REPO, capture_output=True, check=True).stdout)
    for k in corpus["proposal"]["design_unchanged"]:
        assert corpus[k] == prop[k], k
    assert corpus["chosen"]["corpus_id"] == corpus["corpus_id"] == "stage2.reasoning_master@v3"
    assert corpus["chosen"]["by"] == "maintainer" and corpus["shares"]["primary.abstract_analogical"] == 0.09
    assert [s["phase"] for s in corpus["data"]["generation_sessions"]] == ["P1", "P1t"]


def test_a_p1t_grant_issues_a_p1t_artifact_that_never_funds_p1(tmp_path):
    path = GOV.issue("h1p1t_test", grant_path=_grant(tmp_path), out=tmp_path / "a.json")
    auth = GOV.load(path)
    assert GOV.plan_of(auth) == GOV.TOPUP_PLAN and auth.stage_conditions["phase"] == "P1t"
    auth.require_plan(GOV.launch_hash(A.REPO, GOV.TOPUP_PLAN))
    with pytest.raises(AuthorizationError):
        auth.require_plan(P1_LID)
    bound = D.authorize(path, A.REPO)                     # the driver's own contract, on this checkout
    assert bound["plan_path"] == str(GOV.TOPUP_PLAN)
    c = D.contract(A.REPO, path)
    assert set(c["tasks"]) >= {r["task_id"] for r in A.jsonl(T.INPUTS / PLAN["inputs"][0]["file"])}
    with pytest.raises(AuthorizationError, match="does not derive"):   # the P1t identity under P1's phase
        GOV.issue("h1p1t_test2", grant_path=_grant(tmp_path, phase="P1",
                                                   launch_identity_sha256=GOV.launch_hash(A.REPO, GOV.TOPUP_PLAN)),
                  out=tmp_path / "b.json")


def test_the_launcher_binds_the_p1t_plan_and_inputs_and_prices_the_application():
    args = session_args(L, ("--phase", "P1t"))
    spec = L.spec(args)
    spec.validate()
    assert spec.plan_hash == GOV.launch_hash(A.REPO, GOV.TOPUP_PLAN)
    assert [a.repo_path for a in spec.setup.local_assets] == [PLAN["inputs_dir"]]
    app = A.read_json(T.APPLICATION)["cost"]
    m = app["measured_basis"]
    bp = spec.budget.plan(price_per_hour=m["price_per_hour"], authorized_usd=app["hard_cap_usd"])
    plan = bp.as_dict()
    assert plan["expected_minutes"] == pytest.approx(app["session_minutes_expected"], abs=0.05)
    # billing stops where the application's rule says; the watchdog fires its time-to-act earlier
    f = A.read_json(A.REPO / L.FORECAST)["contingency_factors_not_measurements"]["hard_cap"]
    rule = f * (plan["expected_minutes"] + m["pod_to_driver_minutes"] + m["all_done_to_pod_deleted_minutes"]
                + L.ARTIFACT_RECOVERY_MINUTES)
    bound = L.all_in(bp, m["price_per_hour"], 150, app["hard_cap_usd"])
    assert bound["billable_minutes"] == pytest.approx(rule, abs=0.01)
    assert plan["hard_terminate_minutes"] == pytest.approx(rule - L.TERMINATION_MINUTES, abs=0.01)
    assert bound["all_in_usd"] <= app["hard_cap_usd"] and bound["highest_price_inside_cap"] == m["price_per_hour"]
    assert L.budget("P1").setup_minutes == 25.0 and L.budget("P1").soft_stop_reserves[0].minutes == 20.0  # P1's as run
    assert L.budget("P1").artifact_recovery_reserve_minutes == L.ARTIFACT_RECOVERY_MINUTES


@pytest.mark.parametrize("price,disk_gb,ok", [(1.79, 150, True), (1.80, 150, False), (1.81, 150, False),
                                              (1.79, 300, False)])
def test_gpu_disk_and_termination_together_stay_inside_the_cap_or_refuse_at_0(price, disk_gb, ok):
    """$1.81 fitted the GPU alone ($2.597) and not the bill ($2.626): the
    precheck prices GPU, the container disk and the watchdog's time to act at
    --max-price, before any pod, and never raises the cap or shrinks the run."""
    cap = A.read_json(T.APPLICATION)["cost"]["hard_cap_usd"]
    b = L.budget("P1t")
    ctx = type("Ctx", (), {"plan": b.plan(price_per_hour=price, authorized_usd=cap), "evidence": {},
                           "args": type("A", (), {"max_price": price, "disk_gb": disk_gb})(),
                           "auth": type("Au", (), {"hard_cap_usd": cap})()})()
    got, why = L.all_in_within_cap(ctx)
    assert got is ok, why
    assert (ctx.evidence["all_in_bound"]["all_in_usd"] <= cap) is ok
    assert ok or "The cap is not raised" in why
    assert L.all_in_within_cap(type("Ctx", (), {"plan": None})())[0] is False
    spec = L.spec(session_args(L, ("--phase", "P1t")))
    assert spec.precheck[0] is L.all_in_within_cap


def test_the_request_names_what_is_on_disk_now():
    """Stale the moment the plan or the code moves: a grant must name these."""
    if not T.REQUEST.exists():
        pytest.skip("no request written yet")
    req = A.read_json(T.REQUEST)
    assert req["launch_identity_sha256"] == GOV.launch_hash(A.REPO, GOV.TOPUP_PLAN)
    issued = A.REPO / GOV.authorization_path(req["run_id"])
    if issued.is_file():
        # Granted and issued: the request is history. The authorization binds
        # exactly what it named; a later harness change binds a later grant.
        auth = GOV.load(issued)
        assert (auth.plan_hash, auth.harness_source_digest) == (req["launch_identity_sha256"], req["harness_digest"])
    else:
        assert req["harness_digest"] == harness_source_digest(A.REPO, files=GOV.HARNESS)["digest"]
    assert req["generation_plan"]["sha256"] == A.sha256_file(T.PLAN) and req["corpus"] == PLAN["corpus"]
    corpus_run = [s["run_id"] for s in A.read_json(T.CORPUS)["data"]["generation_sessions"] if s["phase"] == "P1t"]
    assert [req["run_id"]] == corpus_run
    assert req["budget"]["fits_under_the_existing_cap"] and not req["budget"]["cap_raised"]
    bound = req["cost"]["all_in_bound_at_p1_price"]
    assert bound["all_in_usd"] <= req["cost"]["hard_cap_usd"] and bound["container_disk_gb"] == 150
    assert req["cost"]["fits_a_live_secure_price_up_to"] == bound["highest_price_inside_cap"] == 1.79
