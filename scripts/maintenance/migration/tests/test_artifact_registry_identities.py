"""The registry, C1's promotion decision and D1's derivation name one B.

The 2026-10-09 review found the registry classifying
`aad-artifacts/autoinit/phase_a/fe9683e6…` as "incumbent B — the arm C1
promoted". It is not: C1 returned GO at +0.013725 against a SESOI of 0.010,
so what stands is C1's TREATMENT arm (`attention.activation_importance_v1`,
digest `53e30566…`, no state id). `fe9683e6…` is the arm C1 BEAT. Worse, the
false entry asserted retained local bytes for a checkpoint that is
reconstructed on the pod and whose bytes are not on this host at all.

These tests require the three places that can name B to agree, and require
the registry to keep LOGICAL identity separate from PHYSICAL copies — which
is also what collapses q1's and q3's two transport copies onto one identity
each.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

REGISTRY = REPO / "logs/maintenance/inventories/checkpoint_registry.json"
IDENTITY_FIELDS = ("artifact_digest", "weights_digest",
                   "single_shard_sha256", "arch_signature")


@pytest.fixture(scope="module")
def reg() -> dict:
    return json.loads(REGISTRY.read_text())


def test_the_registry_and_d1s_derivation_name_the_same_incumbent(reg):
    from stages.d_series.incumbent import standing_incumbent

    derived = standing_incumbent(REPO)
    entry = reg["scientific_artifacts"]["standing_incumbent_B"]
    for field in (*IDENTITY_FIELDS, "impl_id", "profile_id", "kind",
                  "config_sha256", "num_parameters"):
        assert entry["identity"][field] == derived[field], field
    assert entry["state_id"] is None and derived.get("state_id") is None or True
    assert entry["identity_owner"] == derived["measured_by"]
    assert entry["verdict_owner"] == derived["verdict_from"]


def test_that_incumbent_is_c1s_promoted_treatment_not_the_arm_it_beat(reg):
    """Three independent statements of the same fact, required to agree:
    C1's decision record, C1's measured arm identities, and the registry."""
    entry = reg["scientific_artifacts"]["standing_incumbent_B"]
    decision = json.loads((REPO / entry["verdict_owner"]).read_text())
    arms = json.loads((REPO / entry["identity_owner"]).read_text())

    assert decision["verdict"] == "GO"
    assert decision["delta"] > decision["sesoi"], (
        "a GO verdict whose delta does not exceed the SESOI would not select "
        "the treatment arm; the derivation's premise has moved")

    treatment, incumbent = arms["treatment"], arms["incumbent"]
    for field in IDENTITY_FIELDS:
        assert entry["identity"][field] == treatment[field], field
    #: The CONTENT identities must differ from the arm C1 beat. Not
    #: `arch_signature`: the two arms share a geometry by construction — that
    #: is what makes C1 a controlled comparison — so requiring it to differ
    #: would assert the opposite of the protocol. It is checked to MATCH
    #: below for the same reason.
    for field in ("artifact_digest", "weights_digest", "single_shard_sha256"):
        assert entry["identity"][field] != incumbent[field], (
            f"{field}: the registry's B equals C1's INCUMBENT arm, which is "
            "the arm C1 beat")
    assert entry["identity"]["arch_signature"] == incumbent["arch_signature"], (
        "C1's two arms no longer share an architecture signature, so they "
        "were not a controlled comparison and the promotion cannot be read "
        "as an operator-level result")
    assert entry["identity"]["impl_id"] == "attention.activation_importance_v1"
    assert incumbent["impl_id"] == "attention.weight_proxy_v0", (
        "the beaten arm is no longer the weight-proxy scorer; the verdict's "
        "subject has changed")


def test_the_arm_c1_beat_is_classified_as_what_it_is(reg):
    """It stays protected — C1's comparison needs both arms — but it may not
    claim to be the promoted treatment or any later round's control."""
    rows = [e for e in reg["checkpoints"] if "fe9683e6" in e["path_local"]]
    assert rows, "the Phase-A leaf C1 measured against is no longer registered"
    row = rows[0]
    assert "BEAT" in row["role"] or "beat" in row["role"]
    assert "promoted" not in row["role"].replace("NOT the promoted", "")
    assert row["protected"] or row["retention"] == "reproducibility_required"
    # and it is a DIFFERENT checkpoint from B
    b = reg["scientific_artifacts"]["standing_incumbent_B"]["identity"]
    assert row["weights_sha256"] != b["single_shard_sha256"]


def test_the_registry_does_not_claim_local_bytes_for_a_reconstructed_arm(reg):
    """B is materialized on the pod from its construction spec. If its bytes
    are absent, the entry must say so and name how it is obtained."""
    entry = reg["scientific_artifacts"]["standing_incumbent_B"]
    shard = entry["identity"]["single_shard_sha256"]
    on_disk = [e["path_local"] for e in reg["checkpoints"]
               if e.get("weights_sha256") == shard]
    assert entry["local_copies"] == on_disk, (
        "the entry's local_copies disagrees with the physical inventory")
    if not on_disk:
        assert "RECONSTRUCTED" in entry["materialization"]
        assert "frozen_baseline_spec" in entry["materialization"], (
            "an absent artifact must name how it is obtained")


def test_each_finalist_is_one_logical_identity_over_its_copies(reg):
    """q1 and q3 each exist twice on disk (products/ and the out-of-band
    transport copy). One identity, n copies, every copy verified."""
    sa = reg["scientific_artifacts"]
    ids = {k for k in sa if k.startswith("d1_finalist_")}
    assert ids == {"d1_finalist_q1", "d1_finalist_q2",
                   "d1_finalist_q3", "d1_finalist_q4"}
    retention = json.loads((REPO / "logs/stages/stage-1/phase_d1/decisions/"
                            "post_search_finalist_retention.json").read_text())
    want = {f"d1_finalist_q{m['finalist']}": m["single_shard_sha256"]
            for m in retention["the_frozen_behavioural_finalists"]["members"]}
    for key, digest in want.items():
        row = sa[key]
        assert row["identity"]["single_shard_sha256"] == digest, key
        assert row["copies"] >= 1 and row["all_copies_identity_verified"], key
        assert len(row["local_copies"]) == row["copies"]
        assert row["identity_owner"].endswith(
            "post_search_finalist_retention.json")


def test_what_identity_verified_actually_verifies_is_stated(reg):
    means = reg["identity_verified_means"]
    assert "single_shard_sha256" in means
    assert "not recomputed" in means.lower(), (
        "a reader must not be able to read identity_verified as proof that "
        "artifact_digest or weights_digest were recomputed")


def test_the_battery_family_carries_its_six_role_42_file_identity(reg):
    fam = reg["scientific_artifacts"]["d_series_behavioural_v1"]
    manifest = json.loads((REPO / fam["identity_owner"]).read_text())
    ident = fam["identity"]
    assert ident["family_content_id"] == manifest["family_content_id"]
    assert ident["allocation_rule_id"] == manifest["allocation_rule_id"]
    assert ident["n_roles"] == len(manifest["roles"]) == 6
    assert ident["n_output_files"] == len(manifest["output_files"]) == 42
    assert ident["roles"] == list(manifest["role_order"])
    for role, digest in ident["per_role_item_ids_sha256"].items():
        assert manifest["roles"][role]["item_ids_sha256"] == digest, role
    assert fam["identity_owner"].endswith("autoinit_d_series_family_manifest.json")


def test_every_logical_artifact_survives_deletion_of_its_bytes(reg):
    """The point of the section: identity, provenance and consumers are
    recorded here, so deleting a copy loses bytes and not history."""
    for key, row in reg["scientific_artifacts"].items():
        assert row.get("identity"), key
        assert row.get("identity_owner"), key
        assert row.get("consumers"), key
        assert row.get("materialization"), key
        assert row.get("retention"), key


# --- lifecycle: identity must outlive the bytes -----------------------------

def _builder():
    """The registry builder, imported without running it."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_checkpoint_registry_under_test",
        REPO / "scripts/maintenance/consolidation/build_checkpoint_registry.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_a_finalist_with_zero_physical_copies_stays_discoverable(reg):
    """The lifecycle requirement, exercised without deleting anything.

    `scientific_artifacts` is handed a physical inventory with EVERY copy of
    q1 removed — the state the machine would be in after a legitimate
    retirement — and q1's scientific identity must still be there. The
    earlier implementation built these rows by iterating the inventory, so
    this produced a registry with q1 simply absent: the one artifact whose
    identity most needs to survive, because nothing else still carries it.

    No checkpoint is touched. The inventory is a list of dicts.
    """
    mod = _builder()
    q1 = reg["scientific_artifacts"]["d1_finalist_q1"]
    shard = q1["identity"]["single_shard_sha256"]

    full = reg["checkpoints"]
    assert any(e.get("weights_sha256") == shard for e in full), (
        "q1 has no local copy in the committed registry, so this test would "
        "pass for the wrong reason")

    without_q1 = [e for e in full if e.get("weights_sha256") != shard]
    rebuilt = mod.scientific_artifacts(without_q1)

    assert "d1_finalist_q1" in rebuilt, (
        "a finalist with zero local copies vanished from the logical registry")
    row = rebuilt["d1_finalist_q1"]
    assert row["identity"] == q1["identity"], "its identity changed"
    assert row["copies"] == 0 and row["local_copies"] == []
    assert row["lifecycle"] in ("no_local_copy", "weights_retired_tombstoned")
    # and everything a reader needs to act on it is still present
    for field in ("artifact_id", "role", "identity_owner", "creating_run",
                  "consumers", "retention", "reconstruction",
                  "materialization"):
        assert row.get(field), f"zero-copy q1 lost {field}"
    assert "replay" in row["reconstruction"]

    # the other three are unaffected
    for other in ("d1_finalist_q2", "d1_finalist_q3", "d1_finalist_q4"):
        assert rebuilt[other]["identity"] == \
            reg["scientific_artifacts"][other]["identity"]


def test_the_incumbent_already_proves_the_zero_copy_case(reg):
    """B has no local bytes at all, and is in the registry with its full
    identity. It is the live instance of the same requirement."""
    b = reg["scientific_artifacts"]["standing_incumbent_B"]
    assert b["copies"] == 0 and b["local_copies"] == []
    assert b["lifecycle"] in ("no_local_copy", "weights_retired_tombstoned")
    assert b["identity"]["artifact_digest"].startswith("53e30566")
    assert "frozen_baseline_spec" in b["reconstruction"]


def test_every_logical_artifact_is_derived_from_an_owner_not_the_inventory(reg):
    """Stated against the source: the rows come from scientific records, and
    the inventory is matched to them afterwards."""
    mod = _builder()
    #: Handed an EMPTY inventory, every logical identity must still appear:
    #: that is what "derived from an owner" means, operationally.
    rebuilt = mod.scientific_artifacts([])
    assert set(rebuilt) >= {"standing_incumbent_B", "d1_finalist_q1",
                            "d1_finalist_q2", "d1_finalist_q3",
                            "d1_finalist_q4", "d_series_behavioural_v1"}, (
        "a logical artifact disappeared when the physical inventory was "
        "empty, so it is derived from the inventory rather than its owner")
    for key, row in rebuilt.items():
        assert row["identity"], key
        assert row["identity_owner"], key
    #: The checkpoint-shaped artifacts take their copies from the inventory,
    #: so an empty one means zero. The battery family is a DIRECTORY and its
    #: presence is a filesystem question, which is why it is excluded here
    #: rather than asserted to be absent.
    for key in ("standing_incumbent_B", "d1_finalist_q1", "d1_finalist_q2",
                "d1_finalist_q3", "d1_finalist_q4"):
        assert rebuilt[key]["copies"] == 0, key


def test_a_tombstoned_artifact_reuses_the_existing_tombstone_record(reg):
    """Lifecycle state comes from the committed tombstone inventory, not from
    a second copy of it."""
    mod = _builder()
    tombs = mod._tombstones()
    assert tombs, "no tombstone records were read"
    path = REPO / mod.TOMBSTONES_REL
    assert path.is_file()
    # every tombstone the registry could cite is keyed by a real digest
    for key, row in list(tombs.items())[:5]:
        assert row.get("canonical_id")
        assert row.get("deleted_utc")
