"""Strict historical-probe reconstruction. **Dev box only — never on a pod.**

Every test here calls `verify_historical_probe_reuse.verify()`, which re-derives
each Phase-A probe's `student_artifact_digest` from the **retained checkpoint
bytes** under `/home/ecs-user/aad-artifacts/autoinit/phase_a`. That store is a
dev-box artifact store; it is deliberately not transported to a pod, and Phase B
does not need it there.

Phase-B attempt 1 died here. These tests lived in `test_phase_b_pricing.py`, the
pod's setup gate runs the whole suite, and on the pod all 11 probes failed
`artifact_digest_re_derives_from_bytes` because the store does not exist — a
`$0.15` setup abort caused by a test asking the wrong machine a question only
this one can answer. Splitting the module puts the question where the evidence
is, **without** weakening the verifier: it still fails closed when a citation
does not reconstruct.

The responsibility split, which both halves must keep:

* **dev box, before a pod exists** — historical probe ↔ canonical retained
  checkpoint bytes, and the pricing/reuse citation integrity built on it. Also
  enforced as a pre-provider gate in `autoinit_phase_b_launch.py`, so this is
  proven again at launch time and not merely at commit time;
* **pod** — staged imported-finalist bytes ↔ canonical artifact digests, runtime
  comparability, and restored probe ↔ imported candidate identity.

Excluded from the pod run by `PHASE_B_TEST_IGNORES`, which is Phase-B-specific
and leaves the historical Phase-A ignore contract untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/autoinit"))

from verify_historical_probe_reuse import verify  # noqa: E402


def test_every_historical_probe_reconstructs_and_reuse_is_verified():
    r = verify()
    assert r["n_probes"] == 11
    assert r["reuse_verified"] is True, r["failures"]
    assert not r["failures"]
    # The three unadmitted leaves verify too; they are simply not in the
    # candidate set, which is a procedure fact and not an identity failure.
    assert len(r["verifiable_but_not_admitted"]) == 3
    assert len(r["admitted_reusable_probes"]) == 8


def test_the_load_bearing_check_is_the_digest_re_derived_from_BYTES():
    """A probe belongs to a checkpoint only if the bytes still say so."""
    r = verify()
    for probe in r["probes"]:
        assert probe["checks"]["artifact_digest_re_derives_from_bytes"]
        assert probe["recomputed_artifact_digest"] == probe["recorded_artifact_digest"]
        assert probe["recomputed_artifact_digest"], "a digest was never computed"


def test_the_unclosable_leg_is_reported_rather_than_assumed():
    """Phase B's runtime does not exist yet, so comparability cannot be checked."""
    pre = verify()["open_precondition"]
    assert "runtime" in pre["what"] and "comparab" in pre["what"]
    assert "does not exist yet" in pre["why_not_checkable_now"]
    assert "ALL historical reuse is lost" in pre["if_it_fails"]


def test_a_checkpoint_whose_BYTES_disagree_is_not_reusable(monkeypatch):
    """M5: the load-bearing check must actually compare against the bytes.

    Host-local for a second reason beyond cost: without the store *every* probe
    already fails this check, so on a pod the assertions below would pass
    vacuously — a green test proving nothing.
    """
    import verify_historical_probe_reuse as vhr

    swapped = dict(vhr.CHECKPOINTS)
    # Point one finalist at the OTHER finalist's retained checkpoint. Same shape,
    # same parameter count, different weights — so only a real byte comparison
    # can tell, and the probe must stop being reusable.
    swapped["cca699c93f34"] = vhr.CHECKPOINTS["85bde4ded2c3"]
    monkeypatch.setattr(vhr, "CHECKPOINTS", swapped)

    r = vhr.verify()
    assert r["reuse_verified"] is False
    bad = [p for p in r["probes"] if p["candidate"] == "cca699c93f34"]
    assert bad and all(
        "artifact_digest_re_derives_from_bytes" in p["failed"] for p in bad)
    assert all(p["recomputed_artifact_digest"] != p["recorded_artifact_digest"]
               for p in bad)


def test_a_changed_scoring_contract_invalidates_reuse(monkeypatch):
    """M6: old numbers may not be silently re-interpreted under a new scorer."""
    import verify_historical_probe_reuse as vhr

    monkeypatch.setattr(vhr, "recovery_scoring_contract",
                        lambda: {"digest": "f" * 64})
    r = vhr.verify()
    assert r["reuse_verified"] is False
    assert all("scoring_contract_matches_live" in p["failed"] for p in r["probes"])


# --- the pre-provider gate, exercised against the REAL store ----------------
#
# The gate is where this check now happens for a paid launch, so it is proven
# here against the actual retained bytes. Its refusal paths that need no store
# live in `tests/pod/test_phase_b_driver_and_launcher.py` and do run on a pod.


def _launcher():
    sys.path.insert(0, str(REPO / "scripts/pod"))
    import autoinit_phase_b_launch as pbl
    return pbl


def test_the_pre_provider_gate_passes_against_the_retained_store():
    import types

    pbl = _launcher()
    ok, why = pbl.historical_reuse_reconstruction_gate(types.SimpleNamespace())
    assert ok, why
    assert "re-derived from retained bytes" in why


def test_the_gate_refuses_when_the_retained_store_is_ABSENT(monkeypatch):
    """The condition that aborted attempt 1, now caught at `$0` before a pod.

    This is the whole repair in one assertion: the same missing-store situation
    that cost `$0.15` inside a pod's setup gate is a refusal on the dev box, with
    no provider involved.
    """
    import types

    import verify_historical_probe_reuse as vhr

    pbl = _launcher()
    monkeypatch.setattr(vhr, "CHECKPOINTS",
                        {k: "/nonexistent/store/" + k for k in vhr.CHECKPOINTS})
    ok, why = pbl.historical_reuse_reconstruction_gate(types.SimpleNamespace())
    assert not ok
    assert "no longer re-derive" in why


def test_the_gate_refuses_a_SWAPPED_checkpoint(monkeypatch):
    """Not merely absence: bytes that exist and belong to something else."""
    import types

    import verify_historical_probe_reuse as vhr

    pbl = _launcher()
    swapped = dict(vhr.CHECKPOINTS)
    swapped["cca699c93f34"] = vhr.CHECKPOINTS["85bde4ded2c3"]
    monkeypatch.setattr(vhr, "CHECKPOINTS", swapped)
    ok, why = pbl.historical_reuse_reconstruction_gate(types.SimpleNamespace())
    assert not ok and "no longer re-derive" in why
