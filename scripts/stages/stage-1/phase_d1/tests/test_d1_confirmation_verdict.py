"""D1's confirmation verdict: the frozen rule, reused and not reimplemented.

Written and committed BEFORE any confirmation probe exists, which is the only
moment at which "the rule was not chosen to fit the numbers" is checkable
rather than asserted.

What these pin is D1's COMPOSITION over C0's rule -- the arm vocabulary, the
plan's frozen values, the domain-separated bootstrap seed, the evidence
ingestion and the binding of consumed inputs. The statistics themselves are
`phase_c1.isolation`'s and have C1's own tests; re-testing them here would be
a second opinion about a frozen rule.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
for _extra in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _extra) not in sys.path:
        sys.path.insert(0, str(REPO / _extra))

from stages.phase_d1 import behavioural as B  # noqa: E402
from stages.phase_d1 import behavioural_verdict as V  # noqa: E402

#: The six scorable strata C0 names for the bootstrap; `code` is
#: behaviour-only, which is what makes 950 prompts 850 scorable.
SCORABLE = {"gsm8k", "math_verified", "multihop", "rag", "knowledge", "tool"}
BATTERY = (REPO / "artifacts/stages/stage-1/families/d_series/batteries"
                  "/d_series_behavioural_v1/d1_confirmation")


def _battery_ids() -> list[tuple[str, str]]:
    return [(json.loads(line)["id"], name)
            for name in sorted(B.FROZEN_STRATA)
            for line in (BATTERY / f"{name}.jsonl").read_text().splitlines()
            if line.strip()]


def _write_evidence(store: Path, *, advancing: str, seeds, lift: float,
                    usable_drop: float = 0.0, rs: int = 7,
                    drop_probe: str | None = None) -> None:
    """Synthetic per-sample rows in the PRODUCTION contract.

    The real battery's ids and strata, and the field names the real scorer
    writes (`id`, `set`, `scorable`, `correct`, `usable`) -- a fixture in a
    different shape would test the fixture.
    """
    ids = _battery_ids()
    rng = random.Random(rs)
    base = {pid: rng.random() < 0.42 for pid, _ in ids}
    for arm in ("B", advancing):
        for seed in seeds:
            probe_id = f"d1_confirmation_{arm}_s{seed}"
            if probe_id == drop_probe:
                continue
            rows = []
            for pid, stratum in ids:
                scorable = stratum in SCORABLE
                correct = base[pid]
                if arm == advancing and scorable and rng.random() < lift:
                    correct = True
                usable = not (arm == advancing
                              and rng.random() < usable_drop)
                rows.append({"id": pid, "set": stratum, "scorable": scorable,
                             "correct": bool(correct and scorable and usable),
                             "usable": bool(usable),
                             "label": probe_id, "seed": seed})
            path = store / probe_id / "per_sample.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


class TestTheBootstrapSeedIsDomainSeparatedPerPhase:
    """C3's recorded defect: `bootstrap_seed()` defaults to
    `phase-c1:bootstrap` = 816109261, C3's plan froze 654678655, and the
    driver did not pass it -- four of five parameters coincided exactly, so
    only the seed exposed the mismatch."""

    def test_it_is_not_c1s_default(self):
        from stages.phase_c1.isolation import bootstrap_seed as c1_seed

        assert V.bootstrap_seed(REPO) != c1_seed()

    def test_it_derives_from_d1s_own_frozen_design_hash(self):
        import hashlib

        base = B.design(REPO)["design_hash"]
        want = int.from_bytes(hashlib.sha256(
            f"{base}:phase-d1:confirmation-bootstrap".encode()).digest()[:4],
            "big") % (2 ** 31)
        assert V.bootstrap_seed(REPO) == want

    def test_the_verdict_passes_it_explicitly(self, tmp_path):
        plan = V.confirmation_plan("q2", REPO)
        _write_evidence(tmp_path, advancing="q2", seeds=plan.seeds, lift=0.10)
        out = V.compute(tmp_path, "q2", repo_root=REPO, iterations=400)
        assert out["inference"]["bootstrap_seed"] == V.bootstrap_seed(REPO)
        assert "domain-separated per phase" in \
            out["inference"]["bootstrap_seed_derivation"]


class TestThePlanCarriesTheFrozenValuesAndNotC1s:

    def test_every_threshold_comes_from_the_frozen_records(self):
        plan = V.confirmation_plan("q3", REPO)
        bd = B.behavioural_design(REPO)
        assert plan.seeds == tuple(B.confirmation_seeds(REPO))
        assert plan.sesoi == float(bd["sesoi"]) == 0.01
        assert plan.seed_robustness_min_positive == 2
        assert plan.usable_pooled_min_delta == B.GUARDRAIL_POOLED_MIN_DELTA
        assert plan.usable_per_seed_min_delta == B.GUARDRAIL_PER_SEED_MIN_DELTA

    def test_decide_reads_exactly_what_this_plan_offers(self):
        """Duck-typed on purpose, so the check is that nothing `decide` reads
        is missing -- the failure C3 hit was a plan silently carrying C1's
        value for an attribute nobody noticed."""
        import inspect

        from stages.phase_c1.isolation import decide

        source = inspect.getsource(decide)
        plan = V.confirmation_plan("q1", REPO)
        for attribute in ("seeds", "sesoi", "seed_robustness_min_positive",
                          "usable_pooled_min_delta",
                          "usable_per_seed_min_delta"):
            assert f"plan.{attribute}" in source, attribute
            assert hasattr(plan, attribute), attribute

    def test_the_incumbent_may_not_be_the_candidate(self):
        with pytest.raises(V.D1VerdictError, match="may not be the incumbent"):
            V.confirmation_plan("B", REPO)

    def test_it_is_not_a_c1_isolation_plan(self):
        """C1's type refuses two arms sharing an ATTENTION implementation --
        C1's own hypothesis. Every D1 candidate AND B are built by
        `attention.activation_importance_v1`; D1 isolates the SCORING method,
        so reusing that type would refuse every correct D1 verdict."""
        from stages.phase_c1.isolation import C1IsolationPlan

        assert not isinstance(V.confirmation_plan("q2", REPO),
                              C1IsolationPlan)
        design = B.design(REPO)["incumbent"]
        assert design["impl_id"] == "attention.activation_importance_v1"


class TestTheEvidenceIngestionRefusesAnIncompleteDesign:

    def test_all_six_probes_are_required(self, tmp_path):
        plan = V.confirmation_plan("q2", REPO)
        _write_evidence(tmp_path, advancing="q2", seeds=plan.seeds, lift=0.05,
                        drop_probe=f"d1_confirmation_q2_s{plan.seeds[-1]}")
        with pytest.raises(V.D1VerdictError, match="is missing"):
            V.compute(tmp_path, "q2", repo_root=REPO, iterations=200)

    def test_it_binds_every_consumed_file_by_sha256(self, tmp_path):
        import hashlib

        plan = V.confirmation_plan("q2", REPO)
        _write_evidence(tmp_path, advancing="q2", seeds=plan.seeds, lift=0.05)
        out = V.compute(tmp_path, "q2", repo_root=REPO, iterations=200)
        digests = out["inputs_consumed"]["per_sample_sha256"]
        assert len(digests) == 6
        for probe_id, want in digests.items():
            path = tmp_path / probe_id / "per_sample.jsonl"
            assert hashlib.sha256(path.read_bytes()).hexdigest() == want

    def test_it_uses_d1s_arm_vocabulary(self, tmp_path):
        """`decision_inputs` defaults to C1's ('incumbent', 'treatment') and
        would refuse every D1 probe by name -- the same class of defect as
        C3's `allowed_arms`."""
        plan = V.confirmation_plan("q4", REPO)
        _write_evidence(tmp_path, advancing="q4", seeds=plan.seeds, lift=0.05)
        out = V.compute(tmp_path, "q4", repo_root=REPO, iterations=200)
        assert out["advancing_candidate"] == "q4"
        assert out["incumbent"] == "B"
        assert out["inputs_consumed"]["n_prompts_paired"] == 850

    def test_the_confirmation_battery_is_the_one_consumed(self, tmp_path):
        plan = V.confirmation_plan("q2", REPO)
        _write_evidence(tmp_path, advancing="q2", seeds=plan.seeds, lift=0.05)
        out = V.compute(tmp_path, "q2", repo_root=REPO, iterations=200)
        identity = B.battery_role("d1_confirmation", REPO)
        assert out["battery"]["role"] == "d1_confirmation"
        assert out["battery"]["item_ids_sha256"] == \
            identity["item_ids_sha256"]


class TestTheThreeWayRuleIsAppliedUnchanged:

    def _verdict(self, tmp_path, **kw):
        plan = V.confirmation_plan("q2", REPO)
        _write_evidence(tmp_path, advancing="q2", seeds=plan.seeds, **kw)
        return V.compute(tmp_path, "q2", repo_root=REPO, iterations=2000)

    def test_a_clear_positive_effect_is_a_go(self, tmp_path):
        out = self._verdict(tmp_path, lift=0.10)
        d = out["decision"]
        assert d["verdict"] == "GO"
        assert d["criteria"]["lcb_above_zero"]
        assert d["criteria"]["point_at_or_above_sesoi"]
        assert d["criteria"]["seed_robustness"]["passed"]
        assert d["criteria"]["guardrails_passed"]
        assert len(out["per_seed_delta"]) == 3

    def test_a_conclusive_null_is_a_no_go(self, tmp_path):
        """UCB below the SESOI is a NO-GO by the frozen rule -- a
        conclusively-null result is a complete answer, not an inconclusive
        one."""
        out = self._verdict(tmp_path, lift=0.0)
        assert out["decision"]["verdict"] == "NO-GO"
        assert out["decision"]["criteria"]["ucb_below_sesoi"]

    def test_a_usable_rollout_collapse_vetoes_despite_correctness(self,
                                                                  tmp_path):
        out = self._verdict(tmp_path, lift=0.10, usable_drop=0.20)
        d = out["decision"]
        assert d["verdict"] == "NO-GO"
        assert d["vetoes"], "the guardrail must fire"
        assert not d["criteria"]["guardrails_passed"]

    def test_the_verdict_is_deterministic_and_self_hashed(self, tmp_path):
        first = self._verdict(tmp_path, lift=0.08)
        second = V.compute(tmp_path, "q2", repo_root=REPO, iterations=2000)
        assert first["verdict_sha256"] == second["verdict_sha256"]
        assert first["decision"] == second["decision"]

    def test_the_claim_boundary_travels_with_the_verdict(self, tmp_path):
        out = self._verdict(tmp_path, lift=0.08)
        assert "CONDITIONAL" in out["claim_boundary"].upper()
        assert "seed" in out["claim_boundary"].lower()
        assert "not a statement about the other three" in \
            out["_what_this_verdict_is_about"]


class TestNothingStatisticalIsReimplemented:

    def test_every_rule_is_imported_from_its_owner(self):
        src = (REPO / "scripts/stages/stage-1/phase_d1/"
                      "behavioural_verdict.py").read_text()
        for owner in ("probe_results import decision_inputs",
                      "paired_differences",
                      "stratified_cluster_bootstrap",
                      "decide"):
            assert owner in src, owner
        #: and no local arithmetic that would constitute a second rule
        for forbidden in ("def paired_differences", "def decide(",
                          "def stratified_cluster_bootstrap",
                          "percentile", "import statistics"):
            assert forbidden not in src, forbidden

    def test_it_binds_the_preregistration_and_the_implementation(self,
                                                                 tmp_path):
        plan = V.confirmation_plan("q2", REPO)
        _write_evidence(tmp_path, advancing="q2", seeds=plan.seeds, lift=0.05)
        out = V.compute(tmp_path, "q2", repo_root=REPO, iterations=200)
        prereg = json.loads(
            (REPO / "logs/stages/stage-1/phase_d1/plans/"
                    "d1_behavioural_preregistration.json").read_text())
        assert out["preregistration_sha256"] == \
            prereg["preregistration_sha256"]
        assert out["design_hash"] == B.design(REPO)["design_hash"]
        assert len(out["implementation"]["commit"]) == 40
        assert out["implementation"]["rule_owners"]
