"""The D-series is CLOSED: the record reads its owners, and no issuer issues.

Maintainer final decision 2026-10-17. What must hold for the closure to mean
anything:

* every D-series issuer refuses every closed authority -- by path and by
  content -- before it does anything else;
* the closeout restates nothing: each outcome is the terminal closeout's own
  field, each authority is bound by the hash of its untouched file, and the
  money is the deriver's;
* no historical verdict changed, and the funding scope that attributes money
  already spent is still in place.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[6]

from stages.d_series import programme_closeout as PC  # noqa: E402


@pytest.fixture(scope="module")
def committed() -> dict:
    return json.loads((REPO / PC.CLOSEOUT).read_text())


def _issuer(rel: str):
    spec = importlib.util.spec_from_file_location(
        "issuer_" + Path(rel).stem, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: Each grant, and the arguments its issuer needs to reach the gate. The gate
#: runs first, so nothing past the arguments is exercised -- and nothing is
#: written, priced or committed.
ISSUER_ARGS = {
    "logs/budget/approvals/autoinit_d1_grant.json":
        ["--run-id", "closed_check"],
    "logs/budget/approvals/autoinit_d1_behavioural_phase_grant.json":
        ["--run-id", "closed_check", "--rung", "screening"],
    "logs/budget/approvals/autoinit_d1_behavioural_screening_grant.json":
        ["--run-id", "closed_check", "--rung", "screening"],
    "logs/budget/approvals/autoinit_d1_behavioural_screening_grant_amendment.json":
        ["--run-id", "closed_check", "--rung", "screening"],
    "logs/budget/approvals/autoinit_d1_behavioural_confirmation_grant.json":
        ["--run-id", "closed_check", "--rung", "confirmation"],
    "logs/budget/approvals/autoinit_d2_grant.json":
        ["--run-id", "closed_check", "--prediction-rows", "scored_v1"],
    "logs/budget/approvals/autoinit_d2_screening_grant.json":
        ["--run-id", "closed_check"],
    "logs/budget/approvals/autoinit_d2_confirmation_grant.json":
        ["--run-id", "closed_check", "--rung", "confirmation"],
    "logs/budget/approvals/autoinit_d3_structural_grant.json":
        ["--run-id", "closed_check", "--rung", "structural"],
    "logs/budget/approvals/autoinit_d3_probes_grant.json":
        ["--run-id", "closed_check", "--rung", "probes"],
}


class TestNoIssuerIssuesUnderAClosedAuthority:
    @pytest.mark.parametrize("grant", sorted(ISSUER_ARGS))
    def test_the_issuer_refuses_its_grant(self, grant):
        issuer = _issuer(PC.AUTHORITIES[grant]["issuer"])
        with pytest.raises(PC.AuthorityClosed, match="CLOSED D-series"):
            issuer.main(["--grant", str(REPO / grant), *ISSUER_ARGS[grant]])

    def test_the_replay_issuer_refuses_even_a_dry_run(self):
        issuer = _issuer("scripts/stages/stage-1/phase_d1/"
                         "issue_d1_replay_authorization.py")
        with pytest.raises(PC.AuthorityClosed, match="CLOSED D-series"):
            issuer.main(["--rate", "1.09", "--dry-run"])

    def test_every_authority_has_a_refusal_test(self):
        replay = "logs/budget/approvals/autoinit_d1_replay_authorization.json"
        assert set(ISSUER_ARGS) | {replay} == set(PC.AUTHORITIES)

    def test_a_copy_of_a_closed_grant_is_refused_by_content(self, tmp_path):
        copy = tmp_path / "renamed.json"
        shutil.copy(REPO / "logs/budget/approvals/autoinit_d3_probes_grant.json",
                    copy)
        with pytest.raises(PC.AuthorityClosed):
            PC.refuse_closed_authority(copy, repo_root=REPO)

    def test_an_unrelated_grant_is_not_refused(self, tmp_path):
        other = tmp_path / "new_programme_grant.json"
        other.write_text(json.dumps({"rung": "something_new"}))
        PC.refuse_closed_authority(other, repo_root=REPO)

    def test_a_repository_without_the_record_closes_nothing(self, tmp_path):
        assert PC.closed_authorities(tmp_path) == {}


class TestTheRecordReadsItsOwners:
    def test_it_regenerates_identically(self, committed):
        assert PC.build(REPO, provider=committed["provider"]) == committed

    def test_the_grants_are_untouched_since_closure(self, committed):
        for a in committed["closed_authorities"]:
            got = hashlib.sha256((REPO / a["path"]).read_bytes()).hexdigest()
            assert got == a["sha256_at_closure"], a["path"]
            assert a["status"] == "CLOSED"

    def test_each_terminal_closeout_is_bound_by_its_hash(self, committed):
        for phase in committed["terminal_runs"].values():
            for run in phase.values():
                got = hashlib.sha256(
                    (REPO / run["closeout"]).read_bytes()).hexdigest()
                assert got == run["closeout_sha256"], run["closeout"]

    def test_no_verdict_is_rewritten(self, committed):
        t = committed["terminal_runs"]
        assert t["phase_d1"]["confirmation"]["verdict"] == "NO-GO"
        assert t["phase_d2"]["confirmation"]["verdict"] == "NO-GO"
        assert t["phase_d3"]["structural"]["structural_outcome"] == \
            "STRUCTURAL_CHANGE"
        assert t["phase_d3"]["probes"]["labels"] == {
            "vs_q2": "INCONCLUSIVE", "vs_B": "INCONCLUSIVE"}

    def test_the_unspent_d3_envelopes_are_the_grant_minus_the_spend(
            self, committed):
        from stages.phase_d3 import d3_budget

        spent = d3_budget.state(REPO)["spent_by_rung_usd"]
        rows = {a["rung"]: a for a in committed["closed_authorities"]
                if "unspent_usd" in a}
        assert set(rows) == {"structural", "probes"}
        for rung, row in rows.items():
            grant = json.loads((REPO / row["path"]).read_text())
            assert row["unspent_usd"] == round(
                grant["cumulative_envelope_usd"] - spent[rung], 4)

    def test_the_limits_are_the_derivers(self, committed):
        from maintenance.consolidation.derive_budget import derive

        d = derive(REPO)
        after = committed["money"]["limits_after"]
        assert after["project_remaining_usd"] == d["project"]["remaining_usd"]
        assert after["formal_remaining_usd"] == d["formal"]["remaining_usd"]
        assert after["package_remaining_usd"] == d["package"]["remaining_usd"]
        assert after["engineering_remaining_usd"] == \
            d["engineering"]["remaining_usd"]

    def test_the_funding_scope_still_attributes_the_spent_money(self):
        pkg = json.loads((REPO / "configs/stages/stage-1/phase_c1/"
                          "authorization.json").read_text())
        scope = json.dumps(pkg)
        for phase in ("phase_d1", "phase_d2", "phase_d3"):
            assert f'"{phase}"' in scope

    def test_the_provider_was_empty_and_nothing_was_created(self, committed):
        p = committed["provider"]
        assert p["pods"] == [] and p["network_volumes"] == []
        assert p["current_spend_per_hr_usd"] == 0
        assert p["created_during_closure"] == "nothing"

    def test_it_points_at_the_selected_baseline(self, committed):
        sel = REPO / committed["maintainer_selection"]["selection"]
        assert hashlib.sha256(sel.read_bytes()).hexdigest() == \
            committed["maintainer_selection"]["selection_sha256"]
