"""A builder may not re-spell the frozen record it regenerates.

The 2026-10-09 review found `configs/stage3/e6b/provenance.json` rewritten in
place: a path sweep had updated the E6b builder's `objective_parent` /
`rung_parent` EMISSION templates along with its read locations, and then the
builder's own regenerate-identically test executed it and sealed the drift
into a historical record. The same drift sat latent in five sibling builders,
one of which would also have dropped a later tokenizer-contract repair.

The invariant: **what a builder records is frozen; where files live is not.**
A value written into a frozen config keeps its freeze-time spelling so
regeneration stays byte-identical, and reads resolve through
`logs/index.json :: historical_paths.map`.

These tests hold that invariant two ways — the frozen records carry no
post-migration spelling, and the builders' declared emission constants are
exactly what their records contain.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
S3 = REPO / "configs/stages/stage-3"

#: Spellings that only exist AFTER the information-architecture migration. A
#: frozen stage-3 record must not contain one: it was written before.
POST_MIGRATION = (
    "configs/stages/stage-", "artifacts/stages/stage-", "data/stages/stage-",
    "docs/stages/stage-", "docs/shared/", "docs/maintenance/",
)

#: builder -> the records it writes, and the emission constants that must
#: appear verbatim in them. Explicit, so adding a builder is a decision.
BUILDERS = {
    "e1/build_experiment1_configs.py": {
        "records": ["e1/e1_r0250k_sa_pca.json", "e1/e1_r2960k_sb_rand.json"],
        "constants": ['"artifacts/stage1/qwen3_0p6b_init_v0/checkpoint"',
                      '"artifacts/stage1/qwen3_0p6b_init_v0/random_baseline"',
                      '"artifacts/stage3/ladder_uniform"'],
    },
    "e2/build_experiment2_configs.py": {
        "records": ["e2/e2_d1_sa_pca.json"],
        "constants": ['"artifacts/stage1/qwen3_0p6b_init_v0/checkpoint"',
                      '"artifacts/stage3/rung_0860k_clean_median"'],
    },
    "e7/build_e7_configs.py": {
        "records": ["e7/e7_configs.json", "e7/e7_fineweb_r1600k_sa.json"],
        "constants": ['"configs/stage3/e1/e1_r1600k_{seed}_pca.json"',
                      '"artifacts/stage3/e7_fineweb_kd"'],
    },
    "e8/build_e8_configs.py": {
        "records": ["e8/arms.json", "e8/e8_contrib_r2960k_sa.json"],
        "constants": ['"configs/stage3/e1"', '"artifacts/stage3"',
                      '"artifacts/stage1/e8_contribution_init_v1/checkpoint"'],
    },
    "e8b/build_e8b_configs.py": {
        "records": ["e8b/arms.json", "e8b/e8b_dc_r1600k_sa.json"],
        "constants": ['"configs/stage3/e1"'],
    },
    "e6b/build_e6b_configs.py": {
        "records": ["e6b/provenance.json"],
        "constants": ['"configs/stage3/e4/e4_p2_r1600k_{seed}.json"',
                      '"configs/stage3/e1/e1_r2960k_{seed}_pca.json"'],
    },
}

#: NO frozen stage-3 record may carry a post-migration spelling. The earlier
#: version of this file granted E6b's two arm configs an exception, on the
#: reasoning that round 1 had regenerated them together with their provenance
#: and the pair was internally consistent. That reasoning was wrong, and the
#: 2026-10-09 lineage review found why: consistency with the provenance is
#: not the binding that matters. `e6b_registration.json` PROSPECTIVELY
#: registered each arm's `config_sha256` before the experiment ran, and round
#: 1 moved both arms away from those registered hashes. An exception here
#: hid a broken prospective registration behind an internally consistent
#: pair. The arms are restored to their registered identity and this table
#: is empty — a frozen record that needs an entry has a defect, not an
#: exception.
EXPECTED_POST_MIGRATION: dict[str, str] = {}

#: The identities `e6b_registration.json` registered PROSPECTIVELY. Pinned
#: here as literals, from the registration at the pre-migration base commit
#: 35259b64, so a future regeneration cannot quietly redefine what was
#: registered by regenerating the registration too.
E6B_REGISTERED_SHA256 = {
    "P2-2.96M-sa": "963aa00ead1676820abd66c1fced12a90424bb144c19b4dede8ea08b0e985002",
    "P2-2.96M-sb": "da71974841a39b96aceb05d6c39a2723a18624afc6c8c78630937d49ecfa1aef",
}


@pytest.mark.parametrize("builder", sorted(BUILDERS))
def test_the_emission_constants_are_what_the_frozen_records_contain(builder):
    src = (REPO / "scripts/stages/stage-3" / builder).read_text()
    bodies = " ".join((S3 / r).read_text()
                      for r in BUILDERS[builder]["records"])
    for literal in BUILDERS[builder]["constants"]:
        assert literal in src, (
            f"{builder} no longer declares {literal}; if the emission moved, "
            "the frozen record it writes has been re-spelled")
        # the literal's value (minus quotes and any {seed} template) must be
        # the spelling the record actually carries
        value = literal.strip('"').split("{")[0].rstrip("_/")
        assert value in bodies, (
            f"{builder} emits {value!r}, which its own frozen records do not "
            "contain: regeneration would rewrite sealed evidence")


def test_no_frozen_stage_3_record_carries_a_post_migration_spelling():
    offenders = []
    for path in sorted(S3.rglob("*.json")):
        rel = path.relative_to(S3).as_posix()
        text = path.read_text()
        hits = [s for s in POST_MIGRATION if s in text]
        if not hits:
            continue
        if rel in EXPECTED_POST_MIGRATION:
            continue
        offenders.append(f"{rel}: {hits}")
    assert not offenders, (
        "frozen stage-3 records contain spellings that post-date them, which "
        "means something regenerated them after the migration:\n  "
        + "\n  ".join(offenders))


def test_there_is_no_exception_table_to_grow():
    """The table is empty and must stay empty. An entry means some frozen
    record was re-spelled and somebody decided to live with it."""
    assert EXPECTED_POST_MIGRATION == {}, EXPECTED_POST_MIGRATION


def test_the_e6b_arms_still_hash_to_what_was_prospectively_registered():
    """THE load-bearing check. A prospective registration fixes an identity
    before the experiment runs; if the config it registered no longer hashes
    to the registered value, the registration has stopped describing what
    ran, and no amount of internal consistency elsewhere repairs that."""
    from aadistill.infrastructure.manifest import sha256_json

    reg = json.loads((REPO / "logs/stages/stage-3/e6b/analyses/"
                      "e6b_registration.json").read_text())
    for alias, want in E6B_REGISTERED_SHA256.items():
        arm = reg["arms"][alias]
        assert arm["config_sha256"] == want, (
            f"{alias}: the registration's own recorded hash moved; a "
            "prospective registration is not a regenerable record")
        #: Read through the resolver: the registration spells the config at
        #: its pre-migration path, which is the registered evidence.
        import sys
        sys.path.insert(0, str(REPO / "scripts"))
        from shared.run_layout import resolve_historical
        cfg = json.loads(
            (REPO / resolve_historical(arm["config"], REPO)).read_text())
        assert sha256_json(cfg) == want, (
            f"{alias}: the config at {arm['config']} hashes to "
            f"{sha256_json(cfg)[:16]}…, not the registered {want[:16]}…")


def test_the_e6b_provenance_agrees_with_the_registered_identity():
    """provenance.json records the same two hashes. It is a derived record,
    so it follows the arms rather than defining them — but a disagreement
    means one of the two was regenerated alone."""
    from aadistill.infrastructure.manifest import sha256_json

    prov = json.loads((S3 / "e6b/provenance.json").read_text())
    by_run = {"e6b_p2_r2960k_sa": "P2-2.96M-sa",
              "e6b_p2_r2960k_sb": "P2-2.96M-sb"}
    for run, block in prov.items():
        cfg = json.loads((S3 / f"e6b/{run}.json").read_text())
        assert block["config_sha256"] == sha256_json(cfg), run
        assert block["config_sha256"] == E6B_REGISTERED_SHA256[by_run[run]], (
            f"{run}: provenance binds a hash the registration did not "
            "register")


def test_the_e6b_arms_are_byte_identical_to_the_pre_migration_base():
    """The strongest form of the same statement, checked against git rather
    than against another file in the tree."""
    import subprocess
    for name in ("e6b_p2_r2960k_sa.json", "e6b_p2_r2960k_sb.json",
                 "provenance.json"):
        blob = subprocess.run(
            ["git", "-C", str(REPO), "show", f"35259b64:configs/stage3/e6b/{name}"],
            capture_output=True)
        if blob.returncode != 0:
            pytest.skip("the pre-migration base commit is not in this checkout")
        assert blob.stdout == (S3 / "e6b" / name).read_bytes(), name


def test_the_restored_provenance_keeps_its_freeze_time_parents():
    prov = json.loads((S3 / "e6b/provenance.json").read_text())
    for arm, block in prov.items():
        for key in ("objective_parent", "rung_parent"):
            assert block[key].startswith("configs/stage3/"), (
                f"{arm}.{key} = {block[key]!r}: a sealed provenance value was "
                "re-spelled; it must keep the path it recorded")


def test_those_freeze_time_parents_still_resolve_to_real_files():
    """Preserving the spelling only works if access resolves."""
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    from shared.run_layout import resolve_historical

    prov = json.loads((S3 / "e6b/provenance.json").read_text())
    for arm, block in prov.items():
        for key in ("objective_parent", "rung_parent"):
            target = REPO / resolve_historical(block[key], REPO)
            assert target.is_file(), f"{arm}.{key} -> {target}"


def test_the_parent_hashes_the_provenance_records_still_match():
    """The bytes behind those resolved paths are what E6b was derived from."""
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    from shared.run_layout import resolve_historical
    from aadistill.infrastructure.manifest import sha256_json

    prov = json.loads((S3 / "e6b/provenance.json").read_text())
    for arm, block in prov.items():
        for key in ("objective_parent", "rung_parent"):
            doc = json.loads(
                (REPO / resolve_historical(block[key], REPO)).read_text())
            assert sha256_json(doc) == block[f"{key}_sha256"], f"{arm}.{key}"
