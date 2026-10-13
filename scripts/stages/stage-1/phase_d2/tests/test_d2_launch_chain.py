"""D2's launch chain at $0: the typed grant, the launcher, the driver guard.

Nothing here contacts a provider. What is tested is every decision the chain
makes from data: a grant that cannot be edited or mistaken for another
experiment's, the issuer's refusals, the driver's soft stop derived from the
plan's real field, the command the pod will run parsed by the driver's own
parser, and the reference-cache regression guard.
"""
from __future__ import annotations

import json
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[5]

from stages.phase_d2 import autoinit_d2_driver as DRV  # noqa: E402
from stages.phase_d2 import autoinit_d2_launch as L  # noqa: E402
from stages.phase_d2 import d2_authorization as A  # noqa: E402
from stages.phase_d2 import d2_session as S  # noqa: E402

from test_d2_search_chain import _grant  # noqa: E402


class TestTheTypedGrant:

    def test_it_round_trips_through_its_own_loader(self, tmp_path):
        path, _ = _grant(tmp_path)
        auth = A.D2Authorization.load(path)
        assert auth.allows_beam_search and auth.authorizes_d2_search
        assert not auth.allows_recovery and not auth.allows_behavioural
        auth.require_plan("p" * 64)
        with pytest.raises(A.D2AuthorizationRefused):
            auth.require_plan("q" * 64)

    def test_an_edited_grant_is_refused(self, tmp_path):
        path, _ = _grant(tmp_path)
        doc = json.loads(Path(path).read_text())
        doc["hard_cap_usd"] = 99.0
        Path(path).write_text(json.dumps(doc))
        with pytest.raises(A.D2AuthorizationRefused, match="sha256"):
            A.D2Authorization.load(path)

    def test_a_grant_claiming_recovery_is_refused(self, tmp_path):
        path, _ = _grant(tmp_path, allows_recovery=True)
        with pytest.raises(Exception, match="recovery|refus|claim"):
            A.D2Authorization.load(path)

    def test_another_experiments_schema_is_refused(self, tmp_path):
        from aadistill.infrastructure.manifest import sha256_json
        path, _ = _grant(tmp_path)
        doc = json.loads(Path(path).read_text())
        doc["schema"] = "aadistill.phase_d1.authorization/v1"
        doc.pop("authorization_sha256")
        doc["authorization_sha256"] = sha256_json(doc)
        Path(path).write_text(json.dumps(doc))
        with pytest.raises(A.D2AuthorizationRefused, match="schema"):
            A.D2Authorization.load(path)


class TestTheIssuerRefuses:

    PREREG = None

    def _approved(self):
        return {"approves_preregistration_hash": S.preregistration_hash(),
                "granted_by": "test"}

    def test_a_grant_may_not_state_a_derived_figure(self):
        with pytest.raises(A.D2AuthorizationRefused, match="DERIVES"):
            A.build_payload(grant={**self._approved(), "hard_cap_usd": 1.0},
                            session_commit="a" * 40, granted_utc="t",
                            run_id="r1", live_rate=1.09)

    def test_an_unapproved_revision_is_refused(self):
        with pytest.raises(A.D2AuthorizationRefused, match="approve"):
            A.build_payload(grant={"approves_preregistration_hash": "0" * 64},
                            session_commit="a" * 40, granted_utc="t",
                            run_id="r1", live_rate=1.09)

    def test_scored_rows_need_d2s_own_equivalence(self):
        with pytest.raises(A.D2AuthorizationRefused, match="scored-rows"):
            A.build_payload(grant=self._approved(), session_commit="a" * 40,
                            granted_utc="t", run_id="r1",
                            prediction_rows="scored_v1", live_rate=1.09)

    def test_the_funding_scope_is_the_maintainers(self):
        from stages.phase_d1.d1_authorization import live_money
        if S.EXPERIMENT_ID in live_money(REPO)["funds_formal_sessions_of"]:
            pytest.skip("phase_d2 has been added to the formal scope")
        with pytest.raises(A.D2AuthorizationRefused,
                           match="funds_formal_sessions_of"):
            A.build_payload(grant=self._approved(), session_commit="a" * 40,
                            granted_utc="t", run_id="r1", live_rate=1.09)

    def test_the_ceiling_is_the_budget_price_in_force(self):
        from stages.d_series import d2_budget as BUD
        record = json.loads((REPO / BUD.RECORD).read_text())
        ceiling = A.session_ceiling(REPO)
        assert ceiling["pricing_in_force"] == \
            record["steps"]["d2_search"]["pricing_in_force"]
        assert ceiling["hard_ceiling_usd"] == \
            record["steps"]["d2_search"]["hard"]["all_in_usd"]
        assert A.reprice_at(ceiling["price_per_hour"], REPO)[
            "hard_ceiling_usd"] == pytest.approx(ceiling["hard_ceiling_usd"],
                                                 abs=0.01)


class TestTheLauncher:

    def test_the_soft_stop_is_the_plans_real_field_minus_elapsed(self):
        plan = SimpleNamespace(soft_stop_minutes=600.0)
        assert L.driver_deadline_seconds(SimpleNamespace(elapsed_minutes=20.0),
                                         plan) == 580 * 60
        #: Past the soft stop: a positive floor, never a falsy "no deadline".
        assert L.driver_deadline_seconds(SimpleNamespace(elapsed_minutes=700.0),
                                         plan) == 60
        #: D1's launcher read a field BudgetPlan does not have. This one
        #: reads the one it has, on the real class.
        from aadistill.infrastructure.budget import BudgetPlan
        assert "soft_stop_minutes" in BudgetPlan.__dataclass_fields__
        assert "soft_stop_seconds" not in BudgetPlan.__dataclass_fields__

    def test_the_pod_command_parses_with_the_drivers_parser(self, tmp_path):
        ctx = SimpleNamespace(args=SimpleNamespace(run_id="d2_search_x",
                                                   prediction_rows="all_v1"),
                              elapsed_minutes=30.0)
        cmd = L.driver_command(ctx, SimpleNamespace(soft_stop_minutes=900.0))
        argv = shlex.split(cmd)[2:]
        seen = {}
        real = DRV.main

        import argparse
        parsed = {}

        def capture(self, args=None, namespace=None):
            ns = original(self, args, namespace)
            parsed.update(vars(ns))
            raise SystemExit(0)

        original = argparse.ArgumentParser.parse_args
        argparse.ArgumentParser.parse_args = capture
        try:
            with pytest.raises(SystemExit):
                real(argv)
        finally:
            argparse.ArgumentParser.parse_args = original
        assert parsed["run_id"] == "d2_search_x"
        assert parsed["deadline_s"] == 870 * 60
        assert parsed["status"] == S.STATUS_PATH
        assert parsed["authorization"].endswith(
            L.auth_path_for("d2_search_x"))
        del seen

    def test_the_failure_spec_keeps_journal_telemetry_and_state(self):
        doc = json.loads((REPO / "configs/stages/stage-1/phase_d2/"
                                 "d2_search_artifacts_failed.json").read_text())
        patterns = {e["pattern"] for e in doc["entries"]}
        for need in ("audit/autoinit_d2/d2_search.json",
                     "audit/autoinit_d2/journal.jsonl",
                     "audit/autoinit_d2/search/telemetry.jsonl",
                     "audit/autoinit_d2/search/states.jsonl"):
            assert need in patterns
        assert not any(e["required"] for e in doc["entries"])

    def test_the_driver_record_is_the_evidence_file_the_runner_fetches(self):
        assert L.AUDIT_DIRNAME in L.EVIDENCE_DIR
        assert "d2_search.json" in open(DRV.__file__).read()


class TestTheCacheRegressionGuard:

    @staticmethod
    def _telemetry(path, *modes):
        rows = [{"impl_id": "ffn.activation_importance_v0", "state_id": "f"}]
        rows += [{"impl_id": "depth.causal_kl_greedy_v1", "state_id": f"s{i}",
                  "reference_cache": {"mode": m}, "operator_seconds": 10.0}
                 for i, m in enumerate(modes)]
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    def test_a_recompute_expires_a_cache_priced_search(self, tmp_path):
        import time
        tel = tmp_path / "telemetry.jsonl"
        self._telemetry(tel, "cached", "recomputed")
        d = DRV.guarded_deadline(3600, started=time.monotonic(),
                                 telemetry_path=tel, guard_cache=True)
        assert d.remaining() == 0.0 and d.expired()
        assert d.regression["state_id"] == "s1"
        assert "regression" in d.fired_at

    def test_cached_expansions_leave_the_soft_stop_alone(self, tmp_path):
        import time
        tel = tmp_path / "telemetry.jsonl"
        self._telemetry(tel, "cached", "cached")
        d = DRV.guarded_deadline(3600, started=time.monotonic(),
                                 telemetry_path=tel, guard_cache=True)
        assert d.remaining() > 3500 and d.regression is None

    def test_a_records_priced_search_is_not_guarded(self, tmp_path):
        import time
        tel = tmp_path / "telemetry.jsonl"
        self._telemetry(tel, "recomputed")
        d = DRV.guarded_deadline(3600, started=time.monotonic(),
                                 telemetry_path=tel, guard_cache=False)
        assert d.remaining() > 3500

    def test_the_guard_raises_through_the_search_deadline_check(self, tmp_path):
        import time
        from aadistill.initialization.planning.search import (
            SearchDeadlineExceeded,
        )
        tel = tmp_path / "telemetry.jsonl"
        self._telemetry(tel, "partial")
        d = DRV.guarded_deadline(3600, started=time.monotonic(),
                                 telemetry_path=tel, guard_cache=True)
        with pytest.raises(SearchDeadlineExceeded):
            d.check("candidate 3")
        assert d.regression["mode"] == "partial"
