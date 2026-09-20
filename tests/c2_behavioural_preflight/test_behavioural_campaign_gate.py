"""The campaign continuation gate: the two conditions the driver cannot check.

The preregistered continuation policy has four conditions. Two are about bytes
and belong to the driver — the probe must match its descriptor and re-identify
from the delivered bytes. The other two are about resources and money, they
have to be answered before a pod exists, and they are what this gate is:

* at most one resource of a campaign may bill at a time, so a previous resource
  must be PROVIDER-CONFIRMED released before another is created;
* the approved all-in ceiling bounds the campaign cumulatively, so settled
  spend plus this session's planned all-in must fit inside it.

Every case here runs at `$0` against a synthetic durable store and synthetic
session records, and every one of them refuses in the direction that costs
nothing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for _p in ("src", "scripts", "scripts/autoinit", "scripts/pod"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from experiments.phase_c2 import behavioural_governance as BG  # noqa: E402

import autoinit_c2_behavioural_launch as L  # noqa: E402

#: The amounts a campaign of one full session would be authorized for.
GPU_HARD, DISK_HARD = 32.7097, 0.5002
ALL_IN = round(GPU_HARD + DISK_HARD, 4)


class _Ctx:
    """Only what the gate touches."""

    def __init__(self, store: Path, run_id: str = "attempt2",
                 *, all_in: float = ALL_IN):
        self.evidence: dict = {}
        self.args = type("A", (), {"run_id": run_id,
                                   "ckpt_store": str(store)})()
        self.auth = type("Auth", (), {"campaign_id": BG.CAMPAIGN_ID,
                                      "gpu_hard_usd": GPU_HARD,
                                      "disk_hard_usd": DISK_HARD,
                                      "all_in_hard_usd": all_in})()

    def say(self, _msg: str) -> None:
        pass


def _durable_probe(store: Path, run_id: str, unit_id: str = "screening.B.s1"):
    """A prior attempt of this campaign that left a probe off-pod."""
    dest = L.campaign_store(BG.CAMPAIGN_ID, store) / run_id / unit_id
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "durable_ack.json").write_text(json.dumps(
        {"unit_id": unit_id, "campaign": BG.CAMPAIGN_ID,
         "run_attempt": run_id}))
    return dest


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A tmp repo root the gate reads run records from.

    The gate looks under `REPO_ROOT / session_record_path(attempt)`. Writing
    real session records into the actual repository would leave stray evidence
    of runs that never happened — indistinguishable, to a later reader, from
    the real thing — so the root is redirected instead. The PATH SHAPE is still
    the launcher's own `session_record_path`, so this does not invent a
    convention of its own.
    """
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(L, "REPO_ROOT", root)
    return root


def _session_record(repo_root: Path, run_id: str, *, created=True,
                    confirmed_gone=True, actual_usd=0.0, broken=False) -> Path:
    """A prior attempt's session record, where the gate looks for it."""
    path = repo_root / L.session_record_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if broken:
        path.write_text("{ this is not json")
    else:
        path.write_text(json.dumps({
            "provider_resource_created": created,
            "provider_confirms_gone": confirmed_gone,
            "cost": {"actual_usd": actual_usd}}))
    return path


def test_a_first_resource_passes(tmp_path, repo):
    ok, why = L.campaign_continuation_gate(_Ctx(tmp_path, "attempt1"))
    assert ok, why
    assert "first resource" in why


def test_a_continuation_passes_when_the_predecessor_is_released_and_it_fits(
        tmp_path, repo):
    """The permitted case, so the refusals below are known to be selective."""
    _durable_probe(tmp_path, "attempt1")
    _session_record(repo, "attempt1", actual_usd=2.0)
    ctx = _Ctx(tmp_path, "attempt2", all_in=ALL_IN + 5.0)
    ok, why = L.campaign_continuation_gate(ctx)
    assert ok, why
    assert ctx.evidence["campaign"]["prior_attempts_with_durable_probes"] == [
        "attempt1"]
    assert ctx.evidence["campaign"]["settled_campaign_spend_usd"] == 2.0


def test_an_unconfirmed_predecessor_refuses(tmp_path, repo):
    """Two resources on the meter is the failure this prevents."""
    _durable_probe(tmp_path, "attempt1")
    _session_record(repo, "attempt1", confirmed_gone=False, actual_usd=2.0)
    ok, why = L.campaign_continuation_gate(
        _Ctx(tmp_path, "attempt2", all_in=ALL_IN + 5.0))
    assert not ok
    assert "never confirmed released" in why


def test_an_unreadable_predecessor_record_refuses(tmp_path, repo):
    """UNKNOWN is a stop condition, not a clear one."""
    _durable_probe(tmp_path, "attempt1")
    _session_record(repo, "attempt1", broken=True)
    ok, why = L.campaign_continuation_gate(
        _Ctx(tmp_path, "attempt2", all_in=ALL_IN + 5.0))
    assert not ok
    assert "UNKNOWN" in why


def test_a_missing_predecessor_record_refuses(tmp_path, repo):
    """Durable probes with no session record: whose resource was that?"""
    _durable_probe(tmp_path, "attempt1")
    ok, why = L.campaign_continuation_gate(
        _Ctx(tmp_path, "attempt2", all_in=ALL_IN + 5.0))
    assert not ok
    assert "UNKNOWN" in why


def test_a_continuation_that_cannot_fit_the_campaign_ceiling_refuses(
        tmp_path, repo):
    """Cumulative means cumulative. A rerun does not reset the ceiling.

    This is the case a maintainer needs to see: with a ceiling sized for ONE
    full session, a continuation after a resource that already spent money is
    refused here rather than permitted to overspend.
    """
    _durable_probe(tmp_path, "attempt1")
    _session_record(repo, "attempt1", actual_usd=7.02)
    ctx = _Ctx(tmp_path, "attempt2")
    ok, why = L.campaign_continuation_gate(ctx)
    assert not ok
    assert "cumulative across every resource" in why
    assert ctx.evidence["campaign"]["settled_campaign_spend_usd"] == 7.02
    assert ctx.evidence["campaign"][
        "this_session_planned_all_in_usd"] == pytest.approx(ALL_IN, abs=1e-9)


def test_a_predecessor_that_never_created_a_resource_owes_no_confirmation(
        tmp_path, repo):
    """A $0 pre-provider refusal has nothing to confirm released."""
    _durable_probe(tmp_path, "attempt1")
    _session_record(repo, "attempt1", created=False, confirmed_gone=False,
                    actual_usd=0.0)
    ok, why = L.campaign_continuation_gate(
        _Ctx(tmp_path, "attempt2", all_in=ALL_IN + 1.0))
    assert ok, why


def test_this_run_is_never_its_own_predecessor(tmp_path, repo):
    """A re-entered launcher must not refuse on its own durable probes."""
    _durable_probe(tmp_path, "attempt2")
    ok, why = L.campaign_continuation_gate(_Ctx(tmp_path, "attempt2"))
    assert ok, why
    assert "first resource" in why


def test_another_campaigns_attempts_are_not_this_campaigns_predecessors(
        tmp_path, repo):
    """The store is keyed by campaign; a foreign campaign is invisible here."""
    dest = tmp_path / "some-other-campaign" / "attempt1" / "probe"
    dest.mkdir(parents=True)
    (dest / "durable_ack.json").write_text("{}")
    ok, why = L.campaign_continuation_gate(_Ctx(tmp_path, "attempt1"))
    assert ok, why
    assert "first resource" in why


def test_an_empty_attempt_directory_is_not_a_predecessor(tmp_path, repo):
    """An empty directory is not durable work, and refusing on one would strand
    every campaign whose store was merely created."""
    (L.campaign_store(BG.CAMPAIGN_ID, tmp_path) / "attempt1").mkdir(
        parents=True)
    ok, why = L.campaign_continuation_gate(_Ctx(tmp_path, "attempt2"))
    assert ok, why
    assert "first resource" in why


def test_the_destination_gate_checks_the_volume_the_fetcher_writes_to(tmp_path):
    """One flag decides where probes are secured, and every consumer reads it.

    The gate checked the `DURABLE_STORE` constant while `probe_destination`
    honoured `--ckpt-store`, so an operator passing a different store would have
    had capacity verified on a volume nothing writes to — a durability check
    that is green about the wrong disk. No test called this gate at all, which
    is how the two came apart.
    """
    ctx = _Ctx(tmp_path / "store")
    ok, why = L.destination_gate(ctx)
    assert ok, why
    #: The path it reported is the one a probe actually lands on.
    dest = L.probe_destination(ctx, "screening.B.s1")
    assert str(L.campaign_store(BG.CAMPAIGN_ID, ctx.args.ckpt_store)) in why
    assert dest.is_relative_to(
        L.campaign_store(BG.CAMPAIGN_ID, ctx.args.ckpt_store))


def test_the_destination_gate_refuses_an_unusable_store(tmp_path):
    """A run that trains work it cannot preserve is a run that will lose it."""
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("this is a file, so the store cannot be created under it")
    ctx = _Ctx(blocked / "store")
    ok, why = L.destination_gate(ctx)
    assert not ok
    assert "unusable" in why
