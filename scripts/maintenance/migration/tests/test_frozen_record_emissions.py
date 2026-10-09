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

#: The one frozen record a post-migration spelling is EXPECTED in, with the
#: reason. E6b's arm configs were regenerated during round 1 and their
#: `config_sha256` in provenance.json was regenerated with them, so the pair
#: is internally consistent; re-spelling them back now would break that
#: binding to fix a cosmetic difference.
EXPECTED_POST_MIGRATION = {
    "e6b/e6b_p2_r2960k_sa.json": "out_dir, regenerated with its provenance in round 1",
    "e6b/e6b_p2_r2960k_sb.json": "out_dir, regenerated with its provenance in round 1",
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


def test_the_documented_exception_is_still_exactly_two_files():
    """If this list needs to grow, a record was rewritten and the reason has
    to be stated rather than added silently."""
    for rel in EXPECTED_POST_MIGRATION:
        assert (S3 / rel).is_file(), rel
    prov = json.loads((S3 / "e6b/provenance.json").read_text())
    for arm, block in prov.items():
        cfg = json.loads((S3 / f"e6b/{arm}.json").read_text())
        from aadistill.infrastructure.manifest import sha256_json
        assert block["config_sha256"] == sha256_json(cfg), (
            f"{arm}: provenance no longer binds its arm config, so the two "
            "are not the consistent round-1 pair this exception rests on")


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
