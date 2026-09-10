"""The interpretation amendment corrects two claims and touches nothing else.

An amendment that can silently reach the observed device fields, the costs, the
provider identities or the PASS/FAIL outcomes is not an amendment, it is an
edit. So this checks both halves: that the two corrections are stated, and that
everything the review named as untouchable is byte-identical to what the run
wrote.

The binding is the mechanism. Each corrected artifact is pinned by content hash
at the moment the amendment was written, so a later edit to any of them breaks
the binding rather than letting the amendment attach itself to different
evidence. `test_the_binding_still_matches` is that check, and it is the one that
fails if someone rewrites a raw subrun report.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
V = REPO / "logs/validations/cuda-stage-f/v1"
AMENDMENT = V / "interpretation_amendment_1.json"


@pytest.fixture(scope="module")
def doc():
    return json.loads(AMENDMENT.read_text())


# --- it is what it says it is ----------------------------------------------

def test_it_exists_and_authorizes_nothing(doc):
    assert doc["schema"].startswith("aadistill.engineering_validation_interpretation")
    assert doc["authorizes"] == "nothing"
    assert doc["execution_sha_under_review"].startswith("7027a8f4")


def test_it_is_one_amendment_not_several_prose_corrections(doc):
    """The review asked for a single self-hashed artifact. Scattered prose
    corrections are how two readers end up with two different accounts."""
    assert len(doc["corrections"]) == 2
    assert {c["id"] for c in doc["corrections"]} == {"A", "B"}
    others = list(V.glob("interpretation_amendment_*.json"))
    assert others == [AMENDMENT], f"more than one interpretation amendment: {others}"


def test_its_self_hash_verifies(doc):
    """Recomputed the way it was produced: over the body without the hash."""
    body = {k: v for k, v in doc.items() if k != "amendment_sha256"}
    assert doc["amendment_sha256"] == hashlib.sha256(
        json.dumps(body, indent=1, sort_keys=True).encode()).hexdigest()


# --- correction A: subrun 2's failure class ---------------------------------

def test_A_reclassifies_the_second_subrun(doc):
    a = next(c for c in doc["corrections"] if c["id"] == "A")
    assert a["subject"] == "cuda_stage_f_20260910_s2"
    assert a["was"] == "operator"
    assert a["now"] == "harness_acceptance_criterion"
    assert "completed" in a["why"].lower()
    assert "ChildBuilder" in a["why"]


def test_A_leaves_the_outcome_and_the_money_alone(doc):
    a = next(c for c in doc["corrections"] if c["id"] == "A")
    blob = " ".join(a["does_not_change"])
    assert "FAIL" in blob and "0.0145" in blob and "zoz95844krv2ze" in blob


# --- correction B: subrun 3's PASS wording ----------------------------------

def test_B_states_the_observed_contract(doc):
    b = next(c for c in doc["corrections"] if c["id"] == "B")
    now = b["now"]
    assert now["parent_or_operator_computation_device"] == "cuda:0"
    assert now["final_child_device"] == "cpu"
    assert now["child_host_resident_per_builder_contract"] is True


def test_B_does_not_downgrade_the_pass(doc):
    b = next(c for c in doc["corrections"] if c["id"] == "B")
    assert any("CUDA ENGINEERING VALIDATION PASS" in s
               for s in b["does_not_change"])
    assert doc["unchanged"]["verdict"] == (
        "subrun 3 remains CUDA ENGINEERING VALIDATION PASS")


def test_formal_c1_is_explicitly_unchanged(doc):
    c1 = doc["unchanged"]["formal_c1"]
    assert "UNCHANGED" in c1
    assert "MEASURED 2/2 PASS" in c1
    assert "UNMEASURED" in c1
    assert "NO DECISION" in c1


# --- append-only: the evidence it corrects is untouched ---------------------

def test_the_binding_still_matches(doc):
    """The whole mechanism. If a raw subrun report, the campaign record or the
    outcome has been rewritten, this fails rather than the amendment quietly
    describing bytes that no longer exist."""
    for entry in doc["binds"]:
        p = REPO / entry["path"]
        assert p.is_file(), entry["path"]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == entry["sha256"], (
            f"{entry['path']} has changed since the amendment bound it; "
            "re-derive the amendment rather than applying it to new bytes")
        assert p.stat().st_size == entry["bytes"]


def test_it_binds_the_artifacts_it_claims_to_correct(doc):
    bound = {Path(e["path"]).name for e in doc["binds"]}
    assert "campaign.json" in bound
    assert "outcome.md" in bound
    assert sum(1 for e in doc["binds"] if "cuda_stage_f_20260910_s" in e["path"]) == 2


def test_the_raw_records_still_carry_their_original_values():
    """Read from the run records themselves, not from the amendment. An
    amendment that had edited them would make its own `does_not_change` true by
    construction."""
    camp = json.loads((V / "campaign.json").read_text())
    by_id = {s["subrun_id"]: s for s in camp["subruns"]}
    s2 = by_id["cuda_stage_f_20260910_s2"]
    s3 = by_id["cuda_stage_f_20260910_s3"]
    assert s2["cost_usd"] == 0.0145 and s2["pod_id"] == "zoz95844krv2ze"
    assert s2["verdict"] == "FAIL"
    assert s3["cost_usd"] == 0.0182 and s3["pod_id"] == "8tbsixglzz64ox"
    assert s3["verdict"] == "CUDA ENGINEERING VALIDATION PASS"
    assert camp["booked_usd"] == 0.04
    # And the raw record still says `operator`, because it was not rewritten.
    assert s2["failure_class"] == "operator"


def test_the_observed_device_fields_are_untouched():
    """The five placements the PASS rests on, read from the subrun's own
    report."""
    rep = json.loads(
        (REPO / "logs/runs/cuda_stage_f/cuda_stage_f_20260910_s3/artifacts"
                "/cuda_engineering/cuda_stage_f_20260910_s3/suffix_evidence.json"
         ).read_text())
    proofs = [(case["geometry_id"], name, p)
              for case in rep["cases"]
              for name, p in (case.get("device_proofs") or {}).items()]
    assert proofs, "no device proofs in the subrun's evidence"
    for geometry, name, p in proofs:
        assert p["observed"] is True, f"{geometry}/{name} was never observed"
        assert p["holds"] is True, f"{geometry}/{name} does not hold"
    # Five placements per geometry, both geometries: the count the PASS rests on.
    assert len(proofs) == 10, len(proofs)
    assert {g for g, _, _ in proofs} == {"suffix_narrow", "suffix_mid"}
    assert all(case["device"].startswith("cuda") for case in rep["cases"])
    assert all(case["device_proofs_are_meaningful"] is True for case in rep["cases"])
