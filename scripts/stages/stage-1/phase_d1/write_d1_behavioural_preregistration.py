#!/usr/bin/env python3
"""Write D1's behavioural EXECUTION PREREGISTRATION. Derived. AUTHORIZES NOTHING.

    PYTHONPATH=src:scripts .venv/bin/python \
        scripts/stages/stage-1/phase_d1/write_d1_behavioural_preregistration.py --write

C0 fixes the seed COUNT and requires freshness, and is explicit that the exact
values "MUST be materialized and hash-bound in the execution preregistration
BEFORE any candidate behavioural result exists". `derive_seeds` is the
materialization; THIS DOCUMENT is the hash-binding, written while no probe has
trained, no authorization exists and no resource has been created -- which is
checkable, because every one of those is a committed record.

**EVERY VALUE IS READ FROM THE RECORD THAT OWNS IT** -- the frozen design, the
maintainer's retention decision, the realized family manifest, the committed
replay plan, the frozen recipe, C0's decision rule -- and never restated from
memory. The writer is a GENERATOR: run it twice and the bytes agree, which is
what lets a test keep it honest (and what a hand-written record cannot offer).

**WHAT THIS BINDS** (and the one-use authorization later re-binds per run):

* the design revision (`design_hash`) and the five arms' full content
  identities;
* the materialized seeds of BOTH rungs and the exclusion set they avoided;
* both battery roles' realized identities, file digests and disjointness;
* the frozen recovery recipe and the allowed override set;
* the probe schedules -- screening's ten by id, confirmation's SHAPE (the
  advancing candidate is screening's output and is bound at confirmation
  issuance by `require_advancing_candidate`, never pre-named here);
* the committed replay plan's byte hash -- the digest-pinned paths the pod
  materializes the candidates along;
* C0's endpoint, estimand, inference, guardrail vetoes, three-way rule and
  SESOI, and the mechanical screening ranking's tie-break.

**WHAT IT DOES NOT BIND:** money (the design's `budget.chain` owns the priced
cells; the one-use artifact derives the live ceiling), run identities, and the
confirmation verdict machinery's implementation -- which is owed before the
confirmation launch and is named here as owed.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
for _extra in ("src", "scripts", "scripts/autoinit", "scripts/stages/stage-1"):
    _path = str(REPO_ROOT / _extra)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from aadistill.infrastructure.manifest import sha256_json  # noqa: E402

OUT = "logs/stages/stage-1/phase_d1/plans/d1_behavioural_preregistration.json"
SCHEMA = "aadistill.phase_d1.behavioural_execution_preregistration/v1"


def preregistration_identity(doc: dict[str, Any]) -> str:
    """The document's canonical hash: everything except the hash itself and
    the wall clock. A wall clock inside the identity makes the hash move on
    every regeneration of an unchanged tree, which is how a proposal's hash
    once came to be unverifiable."""
    body = {k: v for k, v in doc.items()
            if k not in ("preregistration_sha256", "generated_utc")}
    return sha256_json(body)


def build(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    from shared.recipes import E1_KD_HEAVY_0860K as recipe
    from stages.phase_c1.autoinit_c1_driver import C1_PROBE_OVERRIDES
    from stages.phase_d1 import behavioural as B
    from stages.phase_d1 import behavioural_materialize as M

    design = B.design(repo_root)
    if design["open_blockers"]:
        raise SystemExit(
            f"the design reports open blockers {design['open_blockers']}; a "
            "preregistration over an open blocker preregisters a session that "
            "cannot run")
    bd = design["behavioural_design"]
    retention = json.loads((repo_root / B.RETENTION_REL).read_text())
    frozen = retention["the_frozen_behavioural_finalists"]

    screening_seeds = list(B.screening_seeds(repo_root))
    confirmation_seeds = list(B.confirmation_seeds(repo_root))
    excluded = list(B.excluded_seeds(repo_root))

    #: Both roles, verified against their realized bytes as a side effect of
    #: reading their identities -- `battery_role` refuses a byte mismatch.
    screening_battery = B.battery_role("d1_screening", repo_root)
    confirmation_battery = B.battery_role("d1_confirmation", repo_root)
    disjoint = B.roles_are_disjoint(repo_root)

    #: The five arms, with ALL FOUR content identities each, and the dev-host
    #: verification run as a precondition of writing this document at all.
    verified = B.require_arms_present(repo_root)
    field = B.arms(repo_root)
    arms = [{
        "arm": a.arm_id, "role": a.role,
        "quality_position": a.quality_position,
        "state_id": a.state_id,
        "identities": dict(a.identities),
        "construction": a.construction,
    } for a in field]

    incumbent = design["incumbent"]
    screening_probes = B.probes("screening", repo_root)
    screening_contract = B.session_contract("screening", repo_root)

    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "_contract": (
            "The EXECUTION preregistration for D1's behavioural rungs: the "
            "frozen design revision, the five arms' content identities, the "
            "materialized seeds, the realized batteries, the frozen recovery "
            "recipe, the probe schedules and the decision rules, hash-bound "
            "BEFORE any candidate behavioural result exists. It AUTHORIZES "
            "NOTHING: funding is the maintainer's, the one-use authorization "
            "binds money per run, and this document is what both are checked "
            "against."),
        "_generated_by":
            "scripts/stages/stage-1/phase_d1/write_d1_behavioural_preregistration.py",
        "authorizes": "nothing",
        "generated_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "experiment_id": "phase_d1",
        "stage_id": "1",
        "design": {
            "path": B.DESIGN_REL,
            "design_hash": design["design_hash"],
            "open_blockers": [],
        },
        "c0_materialization_requirement": {
            "quote": (
                "the seed values MUST be materialized and hash-bound in the "
                "execution preregistration BEFORE any candidate behavioural "
                "result exists"),
            "satisfied_because": (
                "this document binds the derived values below, and at its "
                "writing no behavioural probe has trained, no behavioural "
                "authorization has been issued (the schema has no instance) "
                "and no provider resource exists -- each checkable from "
                "committed records rather than asserted."),
        },
        "seeds": {
            "derivation_rule": (
                "H_i = SHA256(design_hash + ':phase-d1:<rung>-seed:' + "
                "decimal(i)); seed_i = uint32_be(H_i[0:4]) mod 2**31, "
                "advancing past collisions with the exclusion set, earlier "
                "draws of the rung, and (for confirmation) every screening "
                "seed. The base is the design_hash bound above, so the values "
                "are fixed by a document that predates every D1 behavioural "
                "result."),
            "owner": "stages.phase_d1.behavioural.derive_seeds",
            "screening": screening_seeds,
            "confirmation": confirmation_seeds,
            "excluded": excluded,
            "_excluded_because": (
                "sa/sb/sc selected the incumbent B (C0), and C1's three "
                "materialized confirmation seeds measured B's standing "
                "result; a rung that reused either would compare a fresh "
                "candidate against the incumbent on the incumbent's own "
                "seeds."),
        },
        "arms": {
            "n_arms": verified["n_arms"],
            "members": arms,
            "candidates_owner": B.RETENTION_REL,
            "retention_rule": frozen["rule_applied"],
            "incumbent_owner": f"{B.DESIGN_REL} :: incumbent",
            "incumbent_construction": B.INCUMBENT_CONSTRUCTION,
            "incumbent_impl_id": incumbent["impl_id"],
            "incumbent_profile_id": incumbent["profile_id"],
            "incumbent_c1_arm": incumbent["c1_arm"],
            "_verified_at_writing": (
                "all four candidates' secured bytes hashed against all four "
                "recorded identities on this host, and the incumbent's "
                "construction pins compared against the design, as a "
                "precondition of writing this document"),
        },
        "arm_materialization": {
            "how": (
                "ON THE POD, each arm along a digest-pinned path: the four "
                "candidates replay the completed search's recorded operator "
                "sequences (the machinery d1_replay_002 validated "
                "byte-exactly), and B is built from "
                "phase_c2.baseline.frozen_baseline_spec, its one construction "
                "owner. A digest mismatch after a completed path is a "
                "scientific stop condition, never a substitution."),
            "plan_path": M.PLAN_REL,
            "plan_sha256": M.plan_sha256(repo_root),
            "adoption_requires": ["artifact_digest", "weights_digest",
                                  "single_shard_sha256", "arch_signature"],
        },
        "recovery_recipe": {
            "recipe_id": recipe.recipe_id,
            "tokens": recipe.tokens,
            "pack": recipe.pack,
            "pack_sha256": recipe.pack_sha256,
            "ce_weight": recipe.ce_weight,
            "kd_weight": recipe.kd_weight,
            "temperature": recipe.temperature,
            "kd_scope": recipe.kd_scope,
            "block_len": recipe.block_len,
            "allowed_override_set": sorted(C1_PROBE_OVERRIDES),
            "_identical_recovery": (
                "a derived probe config may differ from the frozen recipe "
                "only inside the allowed override set -- run identity, pack "
                "path, seed and student_path -- so the only intended "
                "difference between two probes at one seed is the "
                "initialization."),
        },
        "batteries": {
            "family_id": screening_battery["family_id"],
            "family_content_id": screening_battery["family_content_id"],
            "allocation_rule_id": screening_battery["allocation_rule_id"],
            "d1_screening": {
                k: screening_battery[k] for k in (
                    "role", "item_ids_sha256", "n_prompts", "n_scorable",
                    "per_stratum", "files")},
            "d1_confirmation": {
                k: confirmation_battery[k] for k in (
                    "role", "item_ids_sha256", "n_prompts", "n_scorable",
                    "per_stratum", "files")},
            "roles_disjoint": disjoint["disjoint"],
            "_sesoi_population_note": (
                "D-series absolute scores are not directly interchangeable "
                "with historical C1 absolute scores (three strata draw from "
                "prospectively widened sources); the SESOI is carried forward "
                "as an explicit assumption about the within-family PAIRED "
                "difference, which both rungs estimate on one role under one "
                "protocol."),
        },
        "screening": {
            "n_probes": len(screening_probes),
            "probe_ids": [p.probe_id for p in screening_probes],
            "contract_hash_at_preregistration": sha256_json(
                screening_contract),
            "_contract_hash_note": (
                "informational: the location-free contract as this tree "
                "derives it today. The one-use authorization binds the live "
                "value at issuance, and the driver recomputes and compares it "
                "on the pod."),
            "ranking": {
                "pooling": "seed mean over the preregistered screening seeds",
                "delta": "arm pooled correct_overall minus B's, paired on the "
                         "same role and seeds",
                "tie_break": "frozen quality position, then state id -- both "
                             "fixed before any behavioural datum existed",
                "guardrail_vetoes": {
                    "pooled_usable_delta_min": B.GUARDRAIL_POOLED_MIN_DELTA,
                    "per_seed_usable_delta_min":
                        B.GUARDRAIL_PER_SEED_MIN_DELTA,
                    "_veto_only": (
                        "usable_rollout never earns positive ranking credit; "
                        "a candidate can only be removed for being materially "
                        "less usable than the incumbent"),
                },
                "selection": (
                    "exactly one surviving candidate advances "
                    "(advance_one); if every candidate is vetoed, NONE "
                    "advances and the rung says so -- no forced winner"),
            },
        },
        "confirmation": {
            "n_probes": int(bd["confirmation_probes"]),
            "n_arms": int(bd["confirmation_arms"]),
            "arms": "the advancing candidate and B",
            "seeds": confirmation_seeds,
            "_candidate_binding": (
                "the advancing candidate is screening's mechanical output and "
                "CANNOT be named here without pre-selecting it. It is bound "
                "at confirmation issuance: the authorization type requires "
                "`advancing_candidate` on a confirmation artifact and forbids "
                "it on a screening one, and the probe schedule refuses a "
                "confirmation field that is not {that candidate, B}."),
            "verdict": {
                "endpoint": bd["decision_rule"]["primary_endpoint"],
                "estimand": bd["decision_rule"]["estimand"],
                "inference": bd["decision_rule"]["inference"],
                "guardrails": bd["decision_rule"]["guardrails"],
                "rule": bd["decision_rule"]["rule"],
                "sesoi": bd["sesoi"],
                "computed": (
                    "OFF-POD at $0 from the secured per-sample rows. The "
                    "confirmation driver ends at preservation -- A3 lost its "
                    "decision artifact to an on-pod aggregation stage."),
                "_implementation_owed": (
                    "the off-pod verdict computation (stratified "
                    "prompt-cluster bootstrap, seeds as fixed blocks) is OWED "
                    "BEFORE THE CONFIRMATION LAUNCH and does not block "
                    "screening; it consumes evidence screening does not "
                    "produce."),
            },
        },
        "endpoint_chain": {
            "screening_outcomes": ["ONE_CANDIDATE_ADVANCES",
                                   "NO_CANDIDATE_ADVANCES"],
            "confirmation_outcomes": ["GO", "NO_GO", "INCONCLUSIVE"],
            "_all_are_complete_results": (
                "a NO_CANDIDATE_ADVANCES screening and an INCONCLUSIVE or "
                "NO_GO confirmation are complete, reportable endpoints; "
                "nothing reruns a valid measurement looking for another "
                "result."),
        },
        "pricing_reference": {
            "owner": f"{B.DESIGN_REL} :: budget.chain.sessions",
            "_not_money_authorization": (
                "the priced cells live in the design and the live ceiling is "
                "derived at issuance against the live secure rate and the "
                "four budget conditions; this document authorizes no spend."),
        },
        "out_of_scope": [
            "another beam search or any re-opening of the completed D1 search",
            "re-selection of the candidate field",
            "re-measurement of B's standing state evaluation",
            "a promotion decision outside the frozen three-way rule",
            "automatic follow-on starts (confirmation is separately "
            "authorized and its artifact names the advancing candidate)",
            "D2 or D3 work of any kind",
        ],
    }
    doc["preregistration_sha256"] = preregistration_identity(doc)
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true",
                    help="write the record; without it, derive and print only")
    ap.add_argument("--check", action="store_true",
                    help="regenerate and compare against the committed record")
    args = ap.parse_args(argv)

    doc = build(REPO_ROOT)
    out = REPO_ROOT / OUT

    if args.check:
        if not out.is_file():
            raise SystemExit(f"{OUT} does not exist; nothing to check")
        committed = json.loads(out.read_text())
        mine = preregistration_identity(doc)
        theirs_stated = committed.get("preregistration_sha256")
        theirs_recomputed = preregistration_identity(committed)
        problems = []
        if theirs_stated != theirs_recomputed:
            problems.append(
                f"the committed record states {theirs_stated} and recomputes "
                f"to {theirs_recomputed}; it is not the document it claims to "
                "be")
        if mine != theirs_recomputed:
            problems.append(
                f"this tree derives {mine} and the committed record carries "
                f"{theirs_recomputed}; an owner record moved and the "
                "preregistration was not regenerated")
        if problems:
            raise SystemExit("\n".join(problems))
        print(f"OK {OUT} preregistration_sha256={mine}")
        return 0

    if args.write:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        print(f"wrote {OUT}")
    print(f"  preregistration_sha256  {doc['preregistration_sha256']}")
    print(f"  design_hash             {doc['design']['design_hash'][:16]}")
    print(f"  screening seeds         {doc['seeds']['screening']}")
    print(f"  confirmation seeds      {doc['seeds']['confirmation']}")
    print(f"  replay plan sha256      "
          f"{doc['arm_materialization']['plan_sha256'][:16]}")
    print(f"  screening probes        {doc['screening']['n_probes']}")
    print("  AUTHORIZES NOTHING.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
