"""What the diagnostic may spend, and what it must refuse.

This arithmetic used to live inline in the launcher's bash, where the only way
to learn what it computed was to create a pod. Here it can be wrong for free.

The fixtures deliberately give the AUTHORIZATION a different ceiling from any
number in the campaign record, so a reader that took the ceiling from the wrong
file would produce a visibly different answer. While two copies of a number
agree, no test can tell you which one a gate read.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/pod/batch_invariance_budget.py"


def _module():
    spec = importlib.util.spec_from_file_location("batch_invariance_budget", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _module()


@pytest.fixture
def records(tmp_path):
    """A campaign and its authorization, written under a fake repo root.

    The module resolves `authorization` relative to the REPOSITORY root, so the
    fake root is patched onto the module rather than faked with a relative path
    that would happen to work from the test's cwd.
    """
    def build(*, ceiling=7.77, reserve=0.25, inherited=(), subruns=(),
              authorization="auth.json"):
        (tmp_path / "auth.json").write_text(json.dumps({
            "schema": "aadistill.validation_authorization/v1",
            "all_in_ceiling_usd": ceiling}))
        campaign = tmp_path / "campaign.json"
        campaign.write_text(json.dumps({
            "schema": "aadistill.engineering_campaign/v1",
            "authorization": authorization,
            "teardown_reserve_usd": reserve,
            "inherited_spend": [{"subrun_id": f"i{n}", "cost_usd": c}
                                for n, c in enumerate(inherited)],
            "subruns": [{"subrun_id": f"s{n}", "cost_usd": c}
                        for n, c in enumerate(subruns)]}))
        return campaign
    return build, tmp_path


def test_the_ceiling_comes_from_the_authorization(mod, records, monkeypatch):
    build, root = records
    campaign = build(ceiling=7.77, reserve=0.25, inherited=(0.10, 0.20))
    monkeypatch.setattr(mod, "__file__", str(root / "a/b/c.py"))
    d = mod.derive(campaign, session_cap_usd=99.0, rate_usd_per_hour=1.0)
    assert d["campaign_ceiling"] == 7.77
    assert d["spent"] == pytest.approx(0.30)
    assert d["remaining"] == pytest.approx(7.47)
    assert d["session_ceiling"] == pytest.approx(7.22)


def test_a_ceiling_written_into_the_campaign_record_is_IGNORED(mod, records,
                                                               monkeypatch):
    """The test above cannot tell the two files apart, and said it could.

    With no competing number in the campaign record, a reader that preferred
    the campaign and fell back to the authorization produced an identical
    answer and passed everything — a mutation demonstrated exactly that. So
    this one writes a DIFFERENT, larger ceiling into the campaign and requires
    the derivation to ignore it. The live record deliberately carries no such
    field; this test is what keeps re-adding one from silently taking effect.
    """
    build, root = records
    campaign = build(ceiling=3.0, reserve=0.10)
    doc = json.loads(campaign.read_text())
    doc["all_in_ceiling_usd"] = 99.0
    campaign.write_text(json.dumps(doc))
    monkeypatch.setattr(mod, "__file__", str(root / "a/b/c.py"))
    d = mod.derive(campaign, session_cap_usd=50.0, rate_usd_per_hour=1.0)
    assert d["campaign_ceiling"] == 3.0
    assert d["session_ceiling"] == pytest.approx(2.90)


def test_the_live_campaign_record_carries_no_ceiling_of_its_own(mod):
    """One owner per number, checked on the record a launch actually reads."""
    doc = json.loads((
        REPO / "logs/stages/stage-1/phase_c3/investigations"
               "/batch-invariance-root-cause/v1/campaign.json").read_text())
    assert "all_in_ceiling_usd" not in doc


def test_spend_is_recomputed_from_both_lists(mod, records, monkeypatch):
    build, root = records
    campaign = build(ceiling=10.0, reserve=0.0,
                     inherited=(0.0205, 0.0214), subruns=(1.5, 0.25))
    monkeypatch.setattr(mod, "__file__", str(root / "a/b/c.py"))
    d = mod.derive(campaign, session_cap_usd=99.0, rate_usd_per_hour=1.0)
    assert d["spent"] == pytest.approx(1.7919)


def test_the_session_limit_is_floored_not_rounded(mod, records, monkeypatch):
    """A limit handed downward rounds DOWN, or it is not a limit."""
    build, root = records
    campaign = build(ceiling=1.0, reserve=0.0, inherited=(0.001,))
    monkeypatch.setattr(mod, "__file__", str(root / "a/b/c.py"))
    d = mod.derive(campaign, session_cap_usd=99.0, rate_usd_per_hour=1.0)
    assert d["remaining"] == pytest.approx(0.999)
    assert d["session_ceiling"] == 0.99          # not 1.00


def test_the_session_cap_bounds_a_large_remainder(mod, records, monkeypatch):
    build, root = records
    campaign = build(ceiling=100.0, reserve=0.10)
    monkeypatch.setattr(mod, "__file__", str(root / "a/b/c.py"))
    d = mod.derive(campaign, session_cap_usd=2.50, rate_usd_per_hour=1.09)
    assert d["session_ceiling"] == 2.50
    assert d["max_seconds"] == int(2.50 / 1.09 * 3600)


def test_an_exhausted_campaign_is_refused(mod, records, monkeypatch):
    build, root = records
    campaign = build(ceiling=3.0, reserve=0.10, subruns=(2.95,))
    monkeypatch.setattr(mod, "__file__", str(root / "a/b/c.py"))
    d = mod.derive(campaign, session_cap_usd=2.50, rate_usd_per_hour=1.09)
    assert d["ok"] is False


def test_just_enough_is_allowed(mod, records, monkeypatch):
    build, root = records
    campaign = build(ceiling=0.70, reserve=0.10)
    monkeypatch.setattr(mod, "__file__", str(root / "a/b/c.py"))
    d = mod.derive(campaign, session_cap_usd=2.50, rate_usd_per_hour=1.09)
    assert d["session_ceiling"] == pytest.approx(mod.MIN_USEFUL_USD)
    assert d["ok"] is True


def test_an_unpriced_subrun_is_not_a_free_one(mod, records, monkeypatch):
    """A resource that billed and was never priced must not read as $0."""
    build, root = records
    campaign = build(ceiling=3.0)
    doc = json.loads(campaign.read_text())
    doc["subruns"] = [{"subrun_id": "s0"}]
    campaign.write_text(json.dumps(doc))
    monkeypatch.setattr(mod, "__file__", str(root / "a/b/c.py"))
    with pytest.raises(mod.BudgetError, match="no cost_usd"):
        mod.derive(campaign, session_cap_usd=2.5, rate_usd_per_hour=1.09)


def test_an_inline_authorization_object_is_refused(mod, records, monkeypatch):
    """The path form is what derive_budget.py resolves `granted_utc` through."""
    build, root = records
    campaign = build(ceiling=3.0)
    doc = json.loads(campaign.read_text())
    doc["authorization"] = {"all_in_ceiling_usd": 3.0}
    campaign.write_text(json.dumps(doc))
    monkeypatch.setattr(mod, "__file__", str(root / "a/b/c.py"))
    with pytest.raises(mod.BudgetError, match="must be a PATH"):
        mod.derive(campaign, session_cap_usd=2.5, rate_usd_per_hour=1.09)


def test_a_missing_authorization_is_refused_not_defaulted(mod, records,
                                                          monkeypatch):
    build, root = records
    campaign = build(ceiling=3.0, authorization="nope.json")
    monkeypatch.setattr(mod, "__file__", str(root / "a/b/c.py"))
    with pytest.raises(mod.BudgetError, match="cannot read"):
        mod.derive(campaign, session_cap_usd=2.5, rate_usd_per_hour=1.09)


def test_the_real_records_derive_a_usable_session():
    """The live records, through the real module. No fixtures.

    This pinned `spent == 0.0822` and went red the first time the campaign
    booked a subrun -- which is every paid action, i.e. exactly when the check
    matters least and costs most. A live figure is not an assertion; the
    PROPERTIES are.
    """
    mod = _module()
    campaign = (REPO / "logs/stages/stage-1/phase_c3/investigations"
                       "/batch-invariance-root-cause/v1/campaign.json")
    d = mod.derive(campaign, session_cap_usd=2.50, rate_usd_per_hour=1.09)
    assert d["campaign_ceiling"] == 3.0

    #: Recomputed from the file, so the test cannot drift from the record.
    doc = json.loads(campaign.read_text())
    expected = sum(float(e["cost_usd"])
                   for key in ("inherited_spend", "subruns")
                   for e in doc[key])
    assert d["spent"] == pytest.approx(expected)
    #: The four carried-forward batching-refactor-cuda subruns are always in it.
    assert d["spent"] >= 0.0822
    assert d["remaining"] == pytest.approx(3.0 - expected)
    assert 0 < d["session_ceiling"] <= 2.50
    assert d["ok"] is (d["session_ceiling"] >= mod.MIN_USEFUL_USD)


def test_the_cli_prints_seven_shell_readable_fields(capsys):
    """The launcher `read -r`s this line. Its shape is the contract."""
    mod = _module()
    rc = mod.main([str(REPO / "logs/stages/stage-1/phase_c3/investigations"
                              "/batch-invariance-root-cause/v1/campaign.json")])
    assert rc == 0
    fields = capsys.readouterr().out.split()
    assert len(fields) == 7, fields
    assert fields[-1] in {"yes", "no"}
    float(fields[0]); float(fields[4]); int(fields[5])
