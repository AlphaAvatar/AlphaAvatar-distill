"""The search-completion contract: a reservation is released by evidence only.

Every case starts from a COPY of the real D2 search run (which satisfies the
contract) and breaks one record, so each clause is shown to bite on the field
it claims to read -- a contract that passes the real run proves nothing until
a mutation of each premise fails it.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[6]
RUN = "d2_search_20261015_051722"
RUN_DIR = REPO / "logs/stages/stage-1/phase_d2/runs" / RUN

from stages.d_series import search_completion as SC  # noqa: E402


def _spec(booked=(5.5364, "money.all_in_usd")):
    from stages.phase_d2 import d2_session as S
    from stages.phase_d2.d2_authorization import D2Authorization

    prereg = S.preregistration(REPO)
    return SC.SearchCompletionSpec(
        experiment_id="phase_d2",
        closeout_schema="aadistill.phase_d2.search_closeout/v1",
        k=int(prereg["retention"]["k"]),
        preregistration_hash=prereg["preregistration_hash"],
        load_authorization=D2Authorization.load,
        evidence_filename="d2_search.json",
        booked_cost=lambda e, r: booked if r == RUN else None)


@pytest.fixture
def run(tmp_path):
    """A copy of the real run's records (no out-of-tree bytes are needed)."""
    dest = tmp_path / "runs" / RUN
    shutil.copytree(RUN_DIR, dest)
    return dest


def _edit(path: Path, fn) -> None:
    doc = json.loads(path.read_text())
    fn(doc)
    path.write_text(json.dumps(doc, indent=1) + "\n")


def test_the_real_run_satisfies_every_clause(run):
    got = SC.assess(run, _spec())
    assert got["complete"] is True, got["problems"]
    assert got["failed_clauses"] == []
    assert got["booked_all_in_usd"] == 5.5364


def test_the_d2_budget_releases_the_real_search_and_books_its_actual():
    from stages.d_series import d2_budget as BUD

    done = BUD.completed_d2_search()
    assert done["run_id"] == RUN
    assert done["booked_all_in_usd"] == 5.5364
    assert done["contract"]["all_satisfied"] is True


CLOSEOUT = "closeout/outcome.json"
RUNTIME = "runtime/session.json"
EVIDENCE = "evidence/d2_search.json"
MANIFEST = "evidence/manifest.json"


def _set(*keys, value):
    def fn(doc):
        node = doc
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = value
    return fn


MUTATIONS = [
    # authorized_identity
    ("authorized_identity", CLOSEOUT, _set("run_id", value="d2_search_other")),
    ("authorized_identity", CLOSEOUT,
     _set("authorization_sha256", value="0" * 64)),
    ("authorized_identity", CLOSEOUT,
     _set("preregistration_hash", value="1" * 64)),
    ("authorized_identity", CLOSEOUT,
     _set("authorized_base_commit", value="2" * 40)),
    ("authorized_identity", RUNTIME,
     _set("session_commit_check", "session_commit", value="3" * 40)),
    ("authorized_identity", RUNTIME,
     _set("one_use_grant_consumed", value=False)),
    ("authorized_identity", "governance/authorization.json",
     _set("hard_cap_usd", value=99.0)),
    # terminal_complete
    ("terminal_complete", CLOSEOUT,
     _set("execution", "terminal", value="RUN_FAILED")),
    ("terminal_complete", RUNTIME, _set("passed", value=False)),
    ("terminal_complete", EVIDENCE, _set("status", value="FAILED")),
    # full_search
    ("full_search", EVIDENCE, _set("deadline", "expired", value=True)),
    ("full_search", CLOSEOUT,
     _set("search_trajectory", "expansions", value=91)),
    ("full_search", CLOSEOUT,
     _set("search_trajectory", "levels", -1, "expanded", value=1)),
    ("full_search", EVIDENCE, _set("stages", 4, "status", value="failed")),
    # top_k_committed
    ("top_k_committed", CLOSEOUT,
     lambda d: d["committed_top_4"]["members"].pop()),
    ("top_k_committed", CLOSEOUT,
     lambda d: d["committed_top_4"]["members"].reverse()),
    ("top_k_committed", "evidence/stage1_selection.json",
     _set("n_selected", value=3)),
    ("top_k_committed", CLOSEOUT,
     _set("committed_top_4", "selection_sha256", value="4" * 64)),
    # checkpoints_secured
    ("checkpoints_secured", CLOSEOUT,
     _set("products_secured", "secured", value=3)),
    ("checkpoints_secured", CLOSEOUT,
     _set("products_secured", "all_identity_verified", value=False)),
    ("checkpoints_secured", CLOSEOUT,
     _set("execution", "teardown_gate", "checks",
          "checkpoint_hashes_matched", value=False)),
    # manifest_and_archive
    ("manifest_and_archive", MANIFEST, _set("missing", value=["x"])),
    ("manifest_and_archive", EVIDENCE, _set("run_id", value=RUN + "_edited")),
    ("manifest_and_archive", CLOSEOUT,
     lambda d: d["evidence_in_tree"]["out_of_tree"].pop("archive_sha256")),
    ("manifest_and_archive", CLOSEOUT,
     _set("evidence_in_tree", "out_of_tree", "states_jsonl_sha256",
          value="5" * 64)),
    # provider_teardown
    ("provider_teardown", CLOSEOUT,
     _set("execution", "provider_confirms_gone", value=False)),
    ("provider_teardown", CLOSEOUT,
     _set("execution", "final_pod_state", "billing", value=True)),
    ("provider_teardown", CLOSEOUT,
     _set("execution", "teardown_gate", "allowed", value=False)),
    # cost_closed
    ("cost_closed", CLOSEOUT, lambda d: d["money"].pop("all_in_usd")),
    ("cost_closed", CLOSEOUT, _set("money", "all_in_usd", value=0.0)),
    ("cost_closed", CLOSEOUT, _set("money", "disk_usd", value=0.5)),
    ("cost_closed", CLOSEOUT,
     _set("money", "authorized_hard_usd", value=5.0)),
    ("cost_closed", RUNTIME, _set("cost", "actual_usd", value=4.0)),
    ("cost_closed", CLOSEOUT,
     _set("money", "provider_balance_cross_check", "delta_usd", value=9.0)),
    # no_unsettled_resources
    ("no_unsettled_resources", CLOSEOUT,
     _set("money", "provider_resources_created", value=2)),
    ("no_unsettled_resources", RUNTIME,
     _set("blocked_teardown_ownership", value={"pod": "unknown"})),
    ("no_unsettled_resources", CLOSEOUT,
     _set("money", "pod_id", value="someotherpod")),
]


@pytest.mark.parametrize("clause,rel,mutate", MUTATIONS,
                         ids=[f"{c}:{i}" for i, (c, _, _) in
                              enumerate(MUTATIONS)])
def test_each_clause_bites_on_its_own_field(run, clause, rel, mutate):
    _edit(run / rel, mutate)
    got = SC.assess(run, _spec())
    assert got["complete"] is False
    assert clause in got["failed_clauses"], got["problems"]


@pytest.mark.parametrize("booked", [None, (5.5364, "cost.actual_usd"),
                                    (5.0, "money.all_in_usd")])
def test_cost_must_be_booked_by_the_budget_authority_as_stated(run, booked):
    got = SC.assess(run, _spec(booked=booked))
    assert got["failed_clauses"] == ["cost_closed"]


@pytest.mark.parametrize("missing", [CLOSEOUT, RUNTIME, EVIDENCE, MANIFEST,
                                     "governance/authorization.json",
                                     "evidence/stage1_selection.json"])
def test_a_missing_record_fails_closed_and_never_raises(run, missing):
    (run / missing).unlink()
    got = SC.assess(run, _spec())
    assert got["complete"] is False


def test_a_sentence_is_not_completion(run):
    """The old rule: a minimal closeout whose classification says COMPLETE."""
    for p in run.rglob("*"):
        if p.is_file():
            p.unlink()
    (run / "closeout").mkdir(exist_ok=True)
    (run / CLOSEOUT).write_text(json.dumps({
        "run_id": RUN, "classification": "COMPLETE FORMAL SEARCH: 92/92",
        "money": {"all_in_usd": 5.5, "authorized_hard_usd": 23.7}}))
    status = SC.completion(run.parent, _spec(), pattern="d2_search_*")
    assert status["complete"] is None
    assert status["contradictions"] and "claims completion" in \
        status["contradictions"][0]


def test_a_copied_run_cannot_pass_under_another_name(run):
    """The grant binds its run id, so a duplicated directory fails identity."""
    twin = run.parent / (RUN + "_twin")
    shutil.copytree(run, twin)
    _edit(twin / CLOSEOUT, _set("run_id", value=twin.name))
    status = SC.completion(run.parent, _spec(), pattern="d2_search_*")
    assert status["complete"]["run_id"] == RUN
    twin_row = next(r for r in status["runs"] if r["run_id"] == twin.name)
    assert "authorized_identity" in twin_row["failed_clauses"]


def test_two_passing_runs_are_a_contradiction_and_release_nothing(
        run, monkeypatch):
    twin = run.parent / (RUN + "_twin")
    shutil.copytree(run, twin)
    monkeypatch.setattr(SC, "assess", lambda d, s: {
        "run_id": d.name, "complete": True, "failed_clauses": [],
        "claims_complete": True})
    status = SC.completion(run.parent, _spec(), pattern="d2_search_*")
    assert status["complete"] is None
    assert any("one preregistration has one formal search" in c
               for c in status["contradictions"])


def test_a_failed_search_keeps_the_reservation_without_a_contradiction(run):
    _edit(run / CLOSEOUT, _set("classification", value="FAILED FORMAL SEARCH"))
    _edit(run / CLOSEOUT, _set("execution", "terminal", value="RUN_FAILED"))
    status = SC.completion(run.parent, _spec(), pattern="d2_search_*")
    assert status["complete"] is None
    assert status["contradictions"] == []
