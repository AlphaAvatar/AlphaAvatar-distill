"""The three fixture categories the initialization migration made necessary.

One question became three, and a test that does not say which one it asks gets a
confusing answer:

  A. CURRENT IMPLEMENTATION BEHAVIOUR — does the code work, on this tree,
     against a binding built from this tree? Uses an ephemeral document under
     `tmp_path`. Writes nothing to `logs/`, issues no grant or authorization and
     stages no bundle: an artifact left behind eventually gets read as a record
     of a run.
  B. HISTORICAL EVIDENCE VALIDITY — are the completed run's recorded bytes
     intact and self-consistent? Reads `logs/` and asserts nothing about whether
     this tree could run it.
  C. HISTORICAL LAUNCH COMPATIBILITY — fed the immutable historical document,
     does the migrated live gate REFUSE? It must, and that refusal is the point:
     an old authorization must not be revalidatable against code it never ran on.

Category A is not a weakening of the gates. The gate still runs, in full, over
real assets; what changes is only WHICH identity it is asked to match. Category
C keeps the production default and requires the refusal, so nothing is lost.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _root in (REPO / "src", REPO / "scripts", REPO / "scripts/autoinit"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

#: A fixed stamp, so a category-A document is reproducible: two runs of one test
#: must produce the same bytes.
TEST_TIMESTAMP = "2026-09-08T00:00:00+00:00"


def current_tree_expectation(tmp_path: Path) -> Path:
    """CATEGORY A: this tree's own frozen-assets identity, written to tmp_path.

    The scientific configuration is untouched -- the same assets, the same
    roots, the same hash conventions, checked just as strictly. What differs
    from the historical document is only the IMPLEMENTATION identity: the
    scoring contract this tree computes, which the migration moved from v2 to v3
    without moving any number it produces.
    """
    import verify_frozen_assets as vfa
    from experiments.source_sets import recovery_scoring_contract

    contract = recovery_scoring_contract(REPO)
    doc = {
        "schema": "aadistill.autoinit.frozen_asset_expectation/v1",
        "_category": ("A -- current implementation behaviour. NOT a "
                      "preregistration, describes no run, authorizes nothing."),
        "generated_utc": TEST_TIMESTAMP,
        "assets": vfa.FROZEN,
        "scoring_contract": {"contract": contract["contract"],
                             "digest": contract["digest"]},
        "authorizes": "nothing",
    }
    out = tmp_path / "current_expectation.json"
    out.write_text(json.dumps(doc, indent=1))
    return out


def use_current_tree_binding(driver_module, tmp_path: Path,
                             monkeypatch=None) -> Path:
    """Point a loaded Phase-A driver module at a category-A binding.

    Pass `monkeypatch` when the module is SHARED -- a subclassing driver imports
    the parent once, so a bare assignment would leak into every later test in
    the session and be invisible when it did.
    """
    path = current_tree_expectation(tmp_path)
    _set(driver_module, "FROZEN_ASSETS_EXPECTATION", str(path), monkeypatch)
    return path


def _set(module, name: str, value, monkeypatch=None) -> None:
    if monkeypatch is not None:
        monkeypatch.setattr(module, name, value, raising=False)
    else:
        setattr(module, name, value)


def historical_document(relative: str) -> dict:
    """CATEGORY B/C: an immutable record, read from `logs/`. Never written."""
    return json.loads((REPO / relative).read_text())


def current_tree_stage3_binding(driver_module, tmp_path: Path,
                                monkeypatch=None) -> dict:
    """CATEGORY A: a Stage-3 reference this tree is comparable to.

    The stage-0 body binds the selection thresholds to the protocol the Stage-3
    controls were measured under, and refuses when the live protocol is not
    comparable. After the migration it genuinely is not: `scoring_digest` moved
    v2 -> v3, and `generation_source_digest` moved because `uncapped_eval.py`'s
    import line was rewritten. Both are relocations, and the comparability rule
    is deliberately conservative about any byte change in a declared source set
    -- so the refusal is correct and must NOT be weakened.

    What this builds is therefore not a weaker gate but a different question:
    the same science, measured under THIS implementation. Every scientific field
    -- battery identity, tokenizer, chat template, system message, dtype, engine
    settings, thresholds, the equivalence interval and feasibility floor -- is
    copied from the historical artifacts byte for byte. Only the three
    implementation-identity fields are refreshed, and the protocol hash is
    recomputed from them.

    The historical artifacts under `logs/` are read and never written. Category
    C keeps them and requires the refusal.
    """
    from aadistill.infrastructure.manifest import sha256_json

    att = historical_document(
        "logs/cross-stage/phase_a/results/autoinit_stage3_complete/attested_evaluation_protocol.json")
    thresholds = historical_document(
        "logs/cross-stage/phase_a/results/autoinit_stage3_complete/materialized_thresholds.json")
    compat = json.loads(driver_module.COMPAT_V2.read_text())

    protocol = json.loads(json.dumps(att["evaluation_protocol"]))  # deep copy
    live_scoring = _live_scoring()
    protocol["generation"]["generation_source_digest"] = _live_generation_digest()
    protocol["scoring_contract"] = live_scoring["contract"]
    protocol["scoring_digest"] = live_scoring["digest"]

    #: The documented rule, applied to the refreshed protocol. Recomputing it
    #: rather than editing a recorded hash is the difference between building a
    #: new binding and rewriting evidence.
    identity = {k: v for k, v in protocol["generation"].items()
                if k not in ("generation_protocol_fingerprint", "is_materialized",
                             "unmaterialized_fields", "excluded_by_design",
                             "why_excluded")}
    fingerprint = sha256_json(identity)
    protocol["generation"]["generation_protocol_fingerprint"] = fingerprint
    new_hash = sha256_json({
        "generation_protocol_fingerprint": fingerprint,
        "scoring_contract": protocol["scoring_contract"],
        "scoring_digest": protocol["scoring_digest"],
        "battery_artifact": protocol["battery"]["artifact"],
        "battery_manifest_sha256": protocol["battery"]["manifest_sha256"],
        "battery_content_sha256": protocol["battery"]["content_sha256"]})
    protocol["evaluation_protocol_hash"] = new_hash

    att = {**att, "evaluation_protocol": protocol,
           "evaluation_protocol_hash": new_hash,
           "generated_utc": TEST_TIMESTAMP,
           "_category": "A -- current implementation behaviour. Authorizes nothing."}
    thresholds = {**thresholds, "evaluation_protocol_hash": new_hash,
                  "_category": "A -- current implementation behaviour."}
    compat = {**compat,
              "bound_to_historical_protocol": {
                  **compat["bound_to_historical_protocol"],
                  "evaluation_protocol_hash": new_hash},
              "_category": "A -- current implementation behaviour."}

    paths = {}
    for name, doc in (("attestation", att), ("thresholds", thresholds),
                      ("compat", compat)):
        out = tmp_path / f"stage3_{name}.json"
        out.write_text(json.dumps(doc, indent=1))
        paths[name] = out

    _set(driver_module, "STAGE3_ATTESTATION", paths["attestation"], monkeypatch)
    _set(driver_module, "STAGE3_THRESHOLDS", paths["thresholds"], monkeypatch)
    _set(driver_module, "COMPAT_V2", paths["compat"], monkeypatch)
    _set(driver_module, "STAGE3_EVALUATION_PROTOCOL_HASH", new_hash, monkeypatch)
    return {"paths": paths, "evaluation_protocol_hash": new_hash}


def _live_scoring() -> dict:
    from experiments.source_sets import recovery_scoring_contract
    return recovery_scoring_contract(REPO)


def _live_generation_digest() -> str:
    from experiments.source_sets import generation_source_digest
    return generation_source_digest(REPO)["digest"]
