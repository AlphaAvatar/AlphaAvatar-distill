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


def test_every_historical_probe_reconstructs_but_reuse_is_now_REFUSED():
    """A material consequence of the initialization migration.

    Every probe still reconstructs: the bytes are there, the artifact digests
    re-derive, the seeds are the frozen ones, the battery matches and the
    protocol hash matches what was attested. Exactly ONE check fails, for all
    eleven -- `scoring_contract_matches_live` -- because the live scoring
    contract moved v2 -> v3 when the scorer was relocated.

    The numbers those probes carry are NOT in question: 570 frozen samples
    re-scored through the pre- and post-migration trees are byte-identical
    (logs/maintenance/inventories/architecture_scoring_equivalence.json). What has changed is the
    IDENTITY rule, which is deliberately conservative and refuses a probe whose
    recorded scorer digest is not the live one.

    Relaxing that rule -- accepting a superseded contract because equivalence
    was demonstrated -- would be a change to what counts as a reusable
    scientific observation. That is a maintainer decision and not a migration
    one, so this test records the refusal instead of removing it.
    """
    r = verify()
    assert r["n_probes"] == 11
    assert r["reuse_verified"] is False, (
        "reuse is expected to be refused after the scorer relocation")
    assert len(r["failures"]) == 11
    for f in r["failures"]:
        assert f["failed"] == ["scoring_contract_matches_live"], (
            "only the live-contract identity may fail; anything else means the "
            "probes themselves stopped reconstructing")


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


def test_the_pre_provider_gate_now_REFUSES_against_the_retained_store():
    """The $0 pre-provider gate catches the same thing, before a pod exists.

    It used to pass. It refuses now for one reason -- the scorer relocation
    moved the live contract -- and refusing before any provider resource is
    created is the behaviour that matters: a session that would reuse probes
    the identity rule no longer admits stops for free.
    """
    import types

    pbl = _launcher()
    ok, why = pbl.historical_reuse_reconstruction_gate(types.SimpleNamespace())
    assert not ok, (
        "the gate must refuse while the recorded scorer identity differs from "
        f"the live one; it passed with: {why}")
    assert "scoring_contract_matches_live" in why or "reuse" in why, why


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


# --- the four conclusions, held together ------------------------------------
#
# Separately they read as though one must be wrong: the probes are valid, they
# reconstruct, the numbers are byte-identical -- and reuse is refused anyway.
# `historical_reuse_position.py` derives all four from the same `verify()` call
# the pre-provider gate uses, so the position is computed rather than narrated.
#
# These tests live in THIS module deliberately. It is the one the pod launchers
# exclude by exact filename, and every test here needs the dev-box artifact
# store. A new file would be a `$0.15` setup abort on the next pod, which is
# precisely how these tests came to be split out in the first place.

def _position(**kw):
    import sys

    sys.path.insert(0, str(REPO / "scripts/autoinit"))
    from historical_reuse_position import position

    return position(**kw)


def test_all_four_conclusions_hold_at_once():
    doc = _position()
    v = {k: c["verdict"] for k, c in doc["conclusions"].items()}
    assert v == {
        "1_historical_probe_bytes_are_valid": "PASS",
        "2_probes_reconstruct_under_their_historical_contract": "PASS",
        "3_behavior_v0_relocation_outputs_are_byte_identical": "PASS",
        "4_live_reuse_under_scoring_contract_v3": "REFUSED",
    }, v
    assert doc["all_four_hold_simultaneously"] is True


def test_the_refusal_is_derived_from_the_live_verifier_not_asserted():
    """If `verify()` ever stopped refusing, this must go red rather than keep
    printing REFUSED from a constant."""
    def not_refusing():
        return {"probes": [], "failures": [], "reuse_verified": True,
                "n_probes": 0, "live_scoring_contract_digest": "x",
                "probes_dir_digest": "y"}

    doc = _position(verify_fn=not_refusing)
    assert doc["conclusions"]["4_live_reuse_under_scoring_contract_v3"][
        "verdict"] == "NOT REFUSED"
    assert doc["all_four_hold_simultaneously"] is False


def test_a_probe_failing_anything_else_breaks_the_position():
    """Conclusion 4 is 'ONLY the live contract fails'. A second failing check is
    a different and much worse finding, and must not be absorbed."""
    def two_failures():
        return {
            "probes": [{"checks": {"complete": True,
                                   "artifact_digest_re_derives_from_bytes": False,
                                   "scoring_contract_matches_live": False},
                        "failed": ["artifact_digest_re_derives_from_bytes",
                                   "scoring_contract_matches_live"],
                        "recomputed_artifact_digest": "a",
                        "recorded_artifact_digest": "b"}],
            "failures": [{"failed": ["artifact_digest_re_derives_from_bytes",
                                     "scoring_contract_matches_live"]}],
            "reuse_verified": False, "n_probes": 1,
            "live_scoring_contract_digest": "x", "probes_dir_digest": "y"}

    doc = _position(verify_fn=two_failures)
    c = doc["conclusions"]
    assert c["1_historical_probe_bytes_are_valid"]["verdict"] == "FAIL"
    assert c["4_live_reuse_under_scoring_contract_v3"][
        "only_the_live_contract_check_fails"] is False
    assert doc["all_four_hold_simultaneously"] is False


def test_conclusion_three_is_read_from_the_equivalence_record(tmp_path):
    """Not from a sentence in a docstring: weaken the evidence and the verdict
    moves."""
    import json

    weakened = tmp_path / "eq.json"
    weakened.write_text(json.dumps({
        "all_scores_identical": False, "total_samples": 570,
        "paths_compared": 3, "coverage": {"limitation": "x"}}))
    doc = _position(equivalence_path=weakened)
    assert doc["conclusions"][
        "3_behavior_v0_relocation_outputs_are_byte_identical"]["verdict"] == "FAIL"
    assert doc["all_four_hold_simultaneously"] is False


def test_the_committed_record_agrees_with_a_live_derivation():
    import json

    path = REPO / "logs/shared/analyses/autoinit_historical_reuse_position.json"
    assert path.is_file(), "run scripts/autoinit/historical_reuse_position.py"
    recorded = json.loads(path.read_text())
    live = _position()
    assert {k: c["verdict"] for k, c in recorded["conclusions"].items()} == \
        {k: c["verdict"] for k, c in live["conclusions"].items()}
    assert recorded["live_scoring_contract_digest"] == \
        live["live_scoring_contract_digest"]


def test_no_old_to_new_equivalence_bypass_was_added():
    """The refusal must still be a plain identity comparison. A branch that
    admitted a superseded digest 'because equivalence was demonstrated' is the
    thing the maintainer decision forbids."""
    src = (REPO / "scripts/autoinit/verify_historical_probe_reuse.py").read_text()
    assert '"scoring_contract_matches_live":' in src
    body = src.split('"scoring_contract_matches_live":')[1].split(",\n")[0]
    assert "==" in body and "or" not in body, (
        f"the live-contract check gained an alternative branch: {body!r}")
