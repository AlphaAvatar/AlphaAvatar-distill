"""The behavioural proposal binds real inputs, and authorizes nothing.

It is a proposal: no grant, no readiness record, no authorization, no bundle,
no provider resource. What these check is that the things it DOES claim are
derived from committed evidence rather than transcribed — the protocol from the
frozen document by its own hash, the candidates by joining the frozen selection
to destination-verified products, the storage from measured artifact sizes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for extra in ("src", "scripts", "scripts/autoinit"):
    if str(ROOT / extra) not in sys.path:
        sys.path.insert(0, str(ROOT / extra))

from experiments.phase_c2 import behavioural as B  # noqa: E402

PROPOSAL = ("logs/stages/stage-1/phase_c2_behavioural/plans/"
            "c2_behavioural_grant_proposal.json")


def proposal() -> dict:
    return json.loads((ROOT / PROPOSAL).read_text())


def test_the_proposal_authorizes_nothing():
    """The whole point. A proposal that could be mistaken for a grant is worse
    than no proposal."""
    doc = proposal()
    assert doc["authorizes"] == "nothing"
    assert doc["project_position"]["_no_cap_increase_requested"] is True
    #: None of the artifacts that DO authorize may exist yet.
    run_dir = ROOT / "logs/stages/stage-1/phase_c2_behavioural/runs"
    for role in ("grant.json", "readiness.json", "authorization.json",
                 "bundle.json"):
        assert not list(run_dir.glob(f"*/governance/{role}")), (
            f"a {role} exists; this stage is a proposal only")


def test_the_schedule_is_the_frozen_one_and_totals_twelve():
    sched = B.schedule(ROOT)
    assert sched["total_probes"] == 12
    assert sched["screening"]["probes"] == 6
    assert sched["confirmation"]["probes"] == 6
    assert sched["screening"]["arms"] == 6      # five candidates + B
    assert sched["confirmation"]["arms"] == 2   # one candidate + B
    assert sched["advanced_candidates"] == 1


def test_the_protocol_is_read_from_its_own_verified_hash():
    """A frozen document that no longer matches its hash is not frozen."""
    doc = B.protocol(ROOT)
    assert doc["protocol_sha256"] == proposal()["protocol_binding"]["protocol_sha256"]


def test_every_candidate_joins_the_frozen_selection_to_a_verified_product():
    cands = B.candidate_manifest(ROOT)
    assert len(cands) == 5
    from experiments.phase_c2 import replay_specs as RS

    selection = {e["state_id"]: e for e in RS.load_selection(ROOT)["selected"]}
    for c in cands:
        entry = selection[c["state_id"]]
        for field in ("artifact_digest", "weights_digest",
                      "single_shard_sha256", "arch_signature",
                      "num_parameters"):
            assert c[field] == entry[field], field
        assert c["bytes"] == 1_192_135_096
        assert Path(c["durable_path"]).is_dir()


def test_a_product_that_disagrees_with_the_selection_is_refused(tmp_path):
    """The join IS the check. A store holding a different artifact under the
    right name must not become a behavioural input."""
    import shutil

    cands = B.candidate_manifest(ROOT)
    victim = cands[0]
    fake = tmp_path / "store"
    (fake / victim["state_id"]).mkdir(parents=True)
    for name in ("durable_ack.json", "replay_leaf.json"):
        shutil.copy(Path(victim["durable_path"]) / name,
                    fake / victim["state_id"] / name)
    sidecar = fake / victim["state_id"] / "replay_leaf.json"
    doc = json.loads(sidecar.read_text())
    doc["identity"]["artifact_digest"] = "0" * 64
    sidecar.write_text(json.dumps(doc))

    with pytest.raises(B.BehaviouralProposalError, match="different artifacts"):
        B.candidate_manifest(ROOT, store=fake)


def test_a_missing_product_is_refused_rather_than_reconstructed(tmp_path):
    """This session consumes reconstructed checkpoints; it does not make them."""
    with pytest.raises(B.BehaviouralProposalError,
                       match="no destination-verified product"):
        B.candidate_manifest(ROOT, store=tmp_path)


def test_the_storage_provision_is_derived_and_is_not_the_full_searchs():
    cands = B.candidate_manifest(ROOT)
    st = B.storage_requirement(cands, B.schedule(ROOT), ROOT)
    assert st["provision_gb"] < 400, (
        "the full search's provision was carried over; it was sized for a beam "
        "holding sixty states at a level boundary")
    #: Every component is present and the subtotal is their sum.
    parts = {k: v for k, v in st["components_gib"].items()
             if not k.startswith("_")}
    assert abs(sum(parts.values()) - st["subtotal_gib"]) < 1e-6
    assert st["provision_gb"] >= st["subtotal_gib"], (
        "the provision is below the derived requirement")
    #: And it must cover the largest single thing the session holds.
    assert parts["one_probe_training_working_set"] > parts["staged_initializations"] / 2


def test_gpu_and_storage_money_are_derived_apart():
    """A live GPU quote is not evidence that separately priced storage is free."""
    cands = B.candidate_manifest(ROOT)
    st = B.storage_requirement(cands, B.schedule(ROOT), ROOT)
    m = B.money(ROOT, gpu_rate_usd_per_hour=1.09, provision_gb=st["provision_gb"],
                b_preparation_minutes=B.b_preparation_minutes(ROOT)["bounded_minutes"])
    for label in ("expected", "hard_ceiling"):
        x = m[label]
        assert x["disk_usd"] > 0, "storage costed at zero again"
        assert abs(x["all_in_usd"] - round(x["gpu_usd"] + x["disk_usd"], 4)) < 1e-9
    assert m["hard_ceiling"]["all_in_usd"] > m["expected"]["all_in_usd"]


def test_the_ceiling_fits_the_project_cap_with_headroom():
    doc = proposal()
    pos = doc["project_position"]
    assert pos["headroom_after_usd"] > 0, (
        "the proposed ceiling does not fit the project cap; that is a "
        "maintainer decision, not a test failure")
    assert (round(pos["cumulative_spend_usd"] + pos["this_ceiling_all_in_usd"], 4)
            == pos["cumulative_if_fully_spent_usd"])


def test_every_built_claim_names_a_file_that_exists():
    """A proposal claiming more exists than does is the expensive kind of wrong.

    Derived rather than phrased. The first version of this test asserted the
    literal string "NOT BUILT" in the launcher's entry, which was true when the
    launcher did not exist and became a false failure the moment it did. What
    actually matters is that a BUILT claim can be checked against the tree, so
    every claim naming a path is checked against that path.
    """
    import re

    state = proposal()["implementation_state"]
    checked = 0
    for key, claim in state.items():
        for path in re.findall(r"(?:scripts|tests|src|configs)/[\w./-]+", claim):
            target = ROOT / path.rstrip(".")
            if "BUILT" in claim and "NOT BUILT" not in claim:
                assert target.exists(), (
                    f"implementation_state[{key!r}] claims {path} is BUILT and "
                    "it does not exist")
                checked += 1
            else:
                assert not target.exists(), (
                    f"implementation_state[{key!r}] claims {path} is not built "
                    "and it does exist")
    assert checked >= 6, f"only {checked} BUILT claims were checkable"


def test_no_authorizing_artifact_is_claimed_or_present():
    """The proposal proposes. Nothing here may permit a paid run.

    This is the invariant the old launcher-phrasing assertion was really
    reaching for: not that a particular file is absent, but that no grant,
    readiness record, authorization or bundle exists for this session.
    """
    doc = proposal()
    assert doc["authorizes"] == "nothing"
    assert "NOT REQUESTED" in doc["implementation_state"]["grant"]

    runs = ROOT / "logs/stages/stage-1/phase_c2_behavioural/runs"
    found = sorted(str(p.relative_to(ROOT)) for p in runs.rglob("*.json")
                   if p.parent.name == "governance") if runs.is_dir() else []
    assert found == [], f"authorizing artifacts already exist: {found}"


def test_the_units_are_converted_through_the_recorded_basis():
    """The residency is GiB and the provider's flag is GB. Rounding one into the
    other under-provisions by 7%, which is the bug the full search was already
    repaired for and which this file repeated."""
    import json as _json

    st = B.storage_requirement(B.candidate_manifest(ROOT), B.schedule(ROOT), ROOT)
    basis = _json.loads((ROOT / B.STORAGE_PRICING).read_text())["gb_versus_gib"]
    assert st["gb_per_gib"] == basis["gb_per_gib"]
    assert st["with_margin_gb"] > st["with_margin_gib"], (
        "the conversion went the wrong way; GB are smaller than GiB so the "
        "number must grow")
    assert st["provision_gb"] >= st["with_margin_gb"]


def test_incumbent_b_is_bound_to_c1s_frozen_construction():
    """B is the sixth arm and is NOT a staged durable input. Its construction
    comes from C1's own constructor and its spec hash must be what C1's
    preregistration froze."""
    from experiments.phase_c2 import baseline as BL

    b = B.b_binding(ROOT)
    assert b["construction"]["spec_hash"] == BL.B_SPEC_HASH
    assert b["required_identity"]["artifact_digest"] == BL.B_ARTIFACT_DIGEST
    assert b["construction"]["built_by"].endswith("build_arm_specs")
    #: The bytes really are gone; the proposal must not pretend otherwise.
    assert b["availability"]["must_materialize"] is True
    assert "NO SCREENING PROBE MAY START" in b["gate"]


def test_b_preparation_is_bounded_from_evidence_and_is_not_a_probe():
    prep = B.b_preparation_minutes(ROOT)
    assert len(prep["steps"]) == 4
    assert prep["bounded_minutes"] > 25, (
        "B's path includes causal-KL DEPTH; a bound under 25 minutes is not "
        "bounding that operator")
    #: It lengthens the session, it does not add a thirteenth probe.
    assert B.schedule(ROOT)["total_probes"] == 12
    m = B.money(ROOT, gpu_rate_usd_per_hour=1.09, provision_gb=120,
                b_preparation_minutes=prep["bounded_minutes"])
    assert (m["expected"]["minutes"]
            == m["probe_minutes"]["expected"] + prep["bounded_minutes"])
