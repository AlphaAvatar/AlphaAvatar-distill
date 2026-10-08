#!/usr/bin/env python3
"""The canonical C3 stage-I aggregation, run off pod. $0, CPU, deterministic.

    PYTHONPATH=src:scripts python scripts/autoinit/aggregate_c3_stage_i.py \
        --evidence /home/ecs-user/aad-artifacts/phase_c3/attempt75 \
        --out logs/stages/stage-1/phase_c3/analyses/attempt75_stage_i

**AUTHORIZED by the maintainer decision of 2026-10-01**, which accepted that
attempt75 completed the measurement the preregistered design needs -- nine
probes trained, nine checkpoints preserved, nine evaluations scored under one
admitted protocol, scoring contract and battery -- and that stage I is
deterministic post-measurement analysis whose failure does not justify
retraining or re-measuring anything.

**It re-measures NOTHING.** Every input is an immutable artifact attempt75
already produced and shipped off pod. The permitted set is exactly: read the
nine scored results and their per-sample evidence, verify their hashes,
identities, seeds, protocol fingerprint, scoring contract and battery, build
the complete 3-arm x 3-seed field, run the three preregistered contrasts and
the preregistered bootstrap, and apply the frozen decision rule. It does not
retrain, regenerate, rescore, substitute a seed, exclude an observation, pool
attempt66, or change the bootstrap or the thresholds.

**The two stage-I defects it repairs, both caller-side.**

1. `C1ProbeRecord.allowed_arms` defaults to C1's two ROLES
   (`incumbent`, `treatment`); the C3 caller constructed records carrying
   `A_incumbent`, `B_causal_b1`, `C_causal_b3` without passing C3's arm set,
   so every record raised `unknown arm 'A_incumbent'`. C3 supplies its own
   vocabulary now. C1's default is untouched.
2. `stratified_cluster_bootstrap` was called without a seed, so it fell back
   to `isolation.bootstrap_seed()` -- `phase-c1:bootstrap` = 816109261 --
   while C3's plan freezes 654678655, 20,000 iterations, the stratified
   prompt-cluster method and SESOI 0.010. All four are derived from the C3
   preregistration and PASSED now, and so are the usable-rollout guardrails
   that `decide` reads off the plan.

Neither repair changes the scientific protocol; both restore the
implementation to the protocol that was already frozen.

**The artifact binds what it consumed.** Every one of the twenty-seven input
files by sha256, the preregistration hash, the plan hash, and the commit plus
module hashes of the implementation that ran -- and it refuses a dirty tree,
because a commit recorded beside uncommitted edits names bytes that did not
execute.

**attempt75's stage-I failure stays historical fact.** Nothing here rewrites
the live session as though it had reached stage I successfully.
"""
from __future__ import annotations

import argparse
import json
import re
import hashlib
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from experiments.phase_c1.isolation import (  # noqa: E402
    C1Arm, C1IsolationPlan, decide,
)
from experiments.phase_c1.isolation import (  # noqa: E402
    bootstrap_seed as c1_bootstrap_seed,
    paired_differences, stratified_cluster_bootstrap,
)
from experiments.phase_c1.probe_results import (  # noqa: E402
    C1ProbeRecord, build_probe_results, decision_inputs,
)

#: The battery identity the driver reads, at the same path.
BATTERY_IDENTITY = REPO_ROOT / "logs/stages/stage-1/phase_c1/plans/battery.json"
from experiments.phase_c3 import session as CS  # noqa: E402
from aadistill.infrastructure.manifest import sha256_file  # noqa: E402

LABEL = re.compile(r"^autoinit\.v1\.phase_c3\.(?P<arm>[A-Za-z0-9_]+?)\.(?P<seed>\d+)$")


def implementation_identity() -> dict:
    """What performed this aggregation, so the result can be re-derived.

    The commit alone is not enough: a dirty tree would let the artifact name a
    commit whose content did not run. So the tree state is recorded too, and a
    dirty tree is refused at the call site below.
    """
    def git(*a: str) -> str:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *a],
                              capture_output=True, text=True,
                              check=True).stdout.strip()

    dirty = bool(git("status", "--porcelain"))
    return {
        "commit": git("rev-parse", "HEAD"),
        "tree_is_dirty": dirty,
        "aggregator_path": "scripts/autoinit/aggregate_c3_stage_i.py",
        "aggregator_sha256": sha256_file(Path(__file__).resolve()),
        "session_module_sha256": sha256_file(
            REPO_ROOT / "scripts/experiments/stage-1/phase_c3/session.py"),
        "isolation_module_sha256": sha256_file(
            REPO_ROOT / "scripts/experiments/stage-1/phase_c1/isolation.py"),
        "probe_results_module_sha256": sha256_file(
            REPO_ROOT / "scripts/experiments/stage-1/phase_c1/probe_results.py"),
    }


def load(evidence: Path):
    audit = evidence / "audit" / "autoinit_c3"
    scored, per_sample = {}, {}
    for path in sorted(audit.glob("*_c1_confirmation.json")):
        result = json.loads(path.read_text())
        m = LABEL.match(result["label"])
        if not m:
            raise SystemExit(f"unparseable probe label {result['label']!r}")
        arm, seed = m["arm"], int(m["seed"])
        rows_path = audit / f"{result['label']}_per_sample.jsonl"
        if not rows_path.is_file():
            raise SystemExit(f"{rows_path} is missing; stage I cannot be replayed")
        rows = [json.loads(x) for x in rows_path.open() if x.strip()]
        per_sample[(arm, seed)] = rows
        training = json.loads(
            (audit / "probes" / f"{result['label']}.training.json").read_text())
        #: The OBSERVED protocol is the probe's own admission record, not the
        #: aggregate: `admit_generation` reconstructs it from the raw summaries
        #: and checks it against the attestation BEFORE the scorer runs, and
        #: that evidence is what `build_probe_results` re-checks. The aggregate
        #: never carried it, so reading it from there yields "" and the
        #: uniformity check passes vacuously on nine empty strings.
        admission = json.loads(
            (audit / f"{result['label']}_generation_admission.json").read_text())
        scored[(arm, seed)] = {
            "result": result, "training": training, "admission": admission,
            "result_path": path, "per_sample_path": rows_path,
            "admission_path":
                audit / f"{result['label']}_generation_admission.json",
        }
    return scored, per_sample


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--run-id", default="attempt75")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="produce a NON-canonical draft from an uncommitted tree")
    a = ap.parse_args()

    #: A DIRTY TREE CANNOT PRODUCE A CANONICAL ARTIFACT. The maintainer
    #: requires this result to bind the implementation that computed it; a
    #: commit hash recorded beside uncommitted edits names bytes that did not
    #: run. `--allow-dirty` exists for iterating, and stamps the artifact
    #: `canonical: false` so a draft can never be mistaken for the record.
    impl = implementation_identity()
    if impl["tree_is_dirty"] and not a.allow_dirty:
        raise SystemExit(
            "the working tree is dirty, so the commit this artifact would "
            "name does not describe the code that ran. Commit first, or pass "
            "--allow-dirty to produce a NON-canonical draft.")

    arms = list(CS.arm_ids())
    seeds = list(CS.recovery_seeds())
    scored, per_sample = load(a.evidence)

    expected = {(arm, seed) for arm in arms for seed in seeds}
    if set(scored) != expected:
        raise SystemExit(
            f"the evidence holds {sorted(scored)}, the design needs "
            f"{sorted(expected)}; a partial field may not reach the rule")

    #: UNIFORMITY FIRST. C2's confirmation field carried THREE generation
    #: protocol fingerprints and that is why it produced no canonical verdict.
    #: A field that mixes them cannot support a paired contrast, so it is
    #: checked before anything is computed rather than noticed afterwards.
    fingerprints = {s["admission"]["generation_fingerprint"]
                    for s in scored.values()}
    observed_protocols = {s["admission"]["evaluation_protocol_hash"]
                          for s in scored.values()}
    contracts = {json.dumps(s["result"]["scoring_contract"], sort_keys=True)
                 for s in scored.values()}
    batteries = {json.dumps(s["result"]["battery"], sort_keys=True)
                 for s in scored.values()}
    if (len(fingerprints) != 1 or len(contracts) != 1 or len(batteries) != 1
            or len(observed_protocols) != 1):
        raise SystemExit(
            f"the field is not protocol-uniform: {len(fingerprints)} generation "
            f"fingerprint(s), {len(contracts)} scoring contract(s), "
            f"{len(batteries)} batter(ies). A paired contrast over a mixed "
            "field is not the estimand.")

    records = [
        C1ProbeRecord(
            probe_id=s["result"]["label"], arm=arm, seed=seed,
            #: THE REPAIR UNDER TEST. Without it every record raises.
            allowed_arms=tuple(arms),
            initialization_artifact_digest=s["training"][
                "initialization_artifact_digest"],
            trained_run=s["result"].get("trained_run") or {},
            result_path=str(s["result_path"]),
            result_sha256=sha256_file(s["result_path"]),
            per_sample_path=str(s["per_sample_path"]),
            per_sample_sha256=sha256_file(s["per_sample_path"]),
            generations=s["result"]["generations"],
            counts={k: s["result"][k] for k in
                    ("n", "usable", "correct", "n_scorable", "usable_scorable")},
            rates={k: s["result"][k] for k in
                   ("usable_rollout_rate", "correct_overall",
                    "correct_given_usable")},
            per_capability=s["result"]["per_capability"],
            scoring_contract=s["result"]["scoring_contract"],
            battery=s["result"]["battery"],
            observed_generation_fingerprint=s["admission"][
                "generation_fingerprint"],
            observed_evaluation_protocol_hash=s["admission"][
                "evaluation_protocol_hash"])
        for (arm, seed), s in sorted(scored.items())
    ]

    control, candidate = CS.primary_operands()
    #: THE ONE OWNER. This file used to rebuild `C1IsolationPlan` itself, as
    #: the driver did -- two constructions of the object `decide` reads its
    #: thresholds from. Both omitted C3's SESOI and guardrails and inherited
    #: C1's, which happen to be identical; the seed, which is not, is what
    #: exposed the pattern.
    plan = CS.frozen_isolation_plan()

    inputs = decision_inputs(per_sample, seeds=seeds, arms=arms,
                             candidate_operand=candidate,
                             control_operand=control)
    attested = {s["admission"]["attested_evaluation_protocol_hash"]
                for s in scored.values()}
    if len(attested) != 1:
        raise SystemExit(f"nine probes, {len(attested)} attested protocols")
    protocol_hash = next(iter(attested))
    if any(not s["admission"]["comparable"] for s in scored.values()):
        raise SystemExit("a probe was admitted as NOT comparable")
    results = build_probe_results(
        records, plan_hash=plan.plan_hash, seeds=seeds,
        inputs=inputs, arms=arms,
        attested_evaluation_protocol_hash=protocol_hash)

    prereg = CS.preregistration()
    est = prereg["estimand"]
    b3 = next(x for x in arms if x not in (control, candidate))
    contrasts = (("primary", est["primary"]["symbol"], candidate, control),
                 ("secondary", est["secondary"]["symbol"], b3, candidate),
                 ("practical", est["practical"]["symbol"], b3, control))

    def paired(cand: str, ctl: str, *, seed: int) -> dict:
        d = paired_differences(inputs.arm(ctl), inputs.arm(cand))
        boot = stratified_cluster_bootstrap(
            d, inputs.strata, seed=seed, iterations=iterations)
        return {
            "candidate": cand, "control": ctl,
            "delta": boot["delta"], "lcb_one_sided": boot["lcb_one_sided"],
            "ucb_one_sided": boot.get("ucb_one_sided"),
            "per_seed_delta": [
                sum(bool(inputs.correct[cand][s][j])
                    - bool(inputs.correct[ctl][s][j]) for j in d) / len(d)
                for s in seeds],
            "bootstrap": boot,
        }

    #: BOTH SEEDS, reported side by side. C3's plan declares
    #: `seeds.bootstrap` and the driver did not pass it until 2026-09-30, so
    #: the resampler fell back to C1's `phase-c1:bootstrap` value. The
    #: preregistered seed is the one the plan binds and is what `reported`
    #: carries; the fallback is computed too, so a reader can see whether the
    #: defect would have changed the verdict rather than being asked to trust
    #: that it did not.
    inf = CS.inference()
    prereg_seed = int(inf["seed"])
    iterations = int(inf["iterations"])
    fallback_seed = c1_bootstrap_seed()

    reported, as_run = {}, {}
    for role, symbol, cand, ctl in contrasts:
        reported[role] = {"symbol": symbol, **paired(cand, ctl, seed=prereg_seed)}
        as_run[role] = {"symbol": symbol,
                        **paired(cand, ctl, seed=fallback_seed)}
        r, f = reported[role], as_run[role]
        print(f"  {role:9} {symbol:16} {cand} - {ctl}: "
              f"delta {r['delta']:+.6f}  LCB {r['lcb_one_sided']:+.6f}"
              f"   [C1-seed fallback LCB {f['lcb_one_sided']:+.6f}]")

    primary = reported["primary"]
    decision_as_run = decide(
        plan, boot=as_run["primary"]["bootstrap"],
        per_seed_delta=as_run["primary"]["per_seed_delta"],
        usable_pooled_delta=inputs.usable_pooled_delta,
        usable_per_seed_delta=list(inputs.usable_per_seed_delta),
        catastrophic_violations=inputs.catastrophic_violations)
    decision = decide(
        plan, boot=primary["bootstrap"],
        per_seed_delta=primary["per_seed_delta"],
        usable_pooled_delta=inputs.usable_pooled_delta,
        usable_per_seed_delta=list(inputs.usable_per_seed_delta),
        catastrophic_violations=inputs.catastrophic_violations)

    decision.update({
        "_schema": "aadistill.phase_c3.decision/v1",
        "_what_this_is": (
            "The CANONICAL C3 stage-I aggregation, computed off pod from "
            "attempt75's immutable evidence under the maintainer decision of "
            "2026-10-01. attempt75's own stage I raised C1ResultsError before "
            "writing a decision; that failure is historical fact and is NOT "
            "rewritten as though the live session had reached stage I "
            "successfully. This artifact re-measures nothing: it reads the "
            "nine scored results and their per-sample evidence, verifies "
            "their identities, and applies the frozen contrasts, bootstrap "
            "and decision rule. It binds every input it consumed by sha256, "
            "the preregistration hash, and the implementation that ran."),
        "run_id": a.run_id,
        "plan_hash": plan.plan_hash,
        "probe_results_sha256": results["results_sha256"],
        "preregistration_sha256": prereg["preregistration_sha256"],
        "canonical": not impl["tree_is_dirty"],
        "primary_contrast": prereg["claim_boundary"]["primary_contrast"],
        "_verdict_owner": ("the PRIMARY contrast alone. The secondary and "
                           "practical contrasts are reported completely and "
                           "do not redefine it."),
        "contrasts": reported,
        "terminal_semantics": prereg["terminal_outcomes"],
        "generation_protocol_fingerprint": next(iter(fingerprints)),
        "inference": {
            "method": inf["method"],
            "iterations": iterations,
            "seed": prereg_seed,
            "sesoi": float(inf["sesoi"]),
            "_all_four_are_PASSED_not_inherited": (
                "Every parameter here is read from the C3 preregistration and "
                "passed explicitly. `isolation.py` carries C1's values as "
                "defaults and four of the five coincide exactly, which is why "
                "the one that does not -- the bootstrap seed, domain-separated "
                "per phase -- was invisible until it was looked for."),
        },
        "_parameters_the_preregistration_does_NOT_freeze": {
            "alpha": {
                "value_used": 0.05,
                "source": "stratified_cluster_bootstrap's own default",
                "note": ("C3's preregistration names one-sided LCB and UCB and "
                         "never states a confidence level; `decide`'s contract "
                         "reads 'one-sided 95%'. 0.05 is therefore the only "
                         "available value and is consistent with the rule, but "
                         "it is NOT frozen by the C3 plan and is recorded here "
                         "rather than left implicit."),
            },
            "design_alternative": {
                "value_used": 0.015,
                "source": "C1IsolationPlan's default",
                "note": ("Not declared by C3 and not read by `decide`. It "
                         "constrains plan construction only -- the plan "
                         "refuses a SESOI above it -- so it cannot move a "
                         "verdict; it is disclosed for completeness."),
            },
        },
        "bootstrap_seed_used": prereg_seed,
        "_bootstrap_seed": (
            "C3's PREREGISTERED seed, from `phase-c3:bootstrap`. The driver "
            "omitted it until 2026-09-30, so the resampler fell back to "
            f"`phase-c1:bootstrap` = {fallback_seed}. `contrasts_as_the_pod_"
            "would_have_run_them` below is that fallback, for comparison; it "
            "is NOT the plan's seed."),
        "contrasts_as_the_pod_would_have_run_them": as_run,
        "field_is_protocol_uniform": True,
        "b1_vs_b3_is_a_post_c3_maintainer_decision": True,
        "c4_is_not_started": True,
        "verdict_under_the_c1_seed_fallback": decision_as_run["verdict"],
        "_verdict_is_seed_robust": decision["verdict"] == decision_as_run["verdict"],
    })

    #: EVERY CONSUMED INPUT, BY HASH. The artifact must be re-derivable from
    #: the committed record, which means naming exactly which bytes it read --
    #: not "attempt75's evidence" but these nine scored aggregates, these nine
    #: per-sample row files and these nine admission records, each by sha256.
    consumed = {}
    for (arm_id, seed), sc in sorted(scored.items()):
        label = sc["result"]["label"]
        consumed[label] = {
            "arm": arm_id, "seed": seed,
            "scored_result": {
                "path": str(sc["result_path"].relative_to(a.evidence)),
                "sha256": sha256_file(sc["result_path"])},
            "per_sample": {
                "path": str(sc["per_sample_path"].relative_to(a.evidence)),
                "sha256": sha256_file(sc["per_sample_path"])},
            "generation_admission": {
                "path": str(sc["admission_path"].relative_to(a.evidence)),
                "sha256": sha256_file(sc["admission_path"])},
            "initialization_artifact_digest":
                sc["training"]["initialization_artifact_digest"],
            "counts": {k: sc["result"][k] for k in
                       ("n", "usable", "correct", "n_scorable",
                        "usable_scorable")},
        }
    decision["consumed_inputs"] = consumed
    decision["consumed_inputs_sha256"] = hashlib.sha256(
        json.dumps(consumed, sort_keys=True).encode()).hexdigest()
    decision["evidence_root"] = str(a.evidence)
    decision["implementation"] = impl

    #: THE PROBE INVENTORY, as a by-product of the same walk. It is what
    #: establishes 9/9 trained, 9/9 preserved and 9/9 scored FROM THE COMMITTED
    #: RECORD: the heavy bytes live outside git by policy, so without this the
    #: only evidence that nine checkpoints exist is prose in a closeout and a
    #: line in a driver log. Small enough to commit, and each entry names the
    #: durable location and the content hash that identifies what is there.
    inventory = {
        "schema": "aadistill.phase_c3.probe_inventory/v1",
        "_what_this_is": (
            "One row per formal probe: what it was built from, that it "
            "trained, where its checkpoint was preserved and under which "
            "content hash, and what it scored. Derived from the run's own "
            "training records and scored aggregates; nothing here is typed."),
        "run_id": a.run_id,
        "n_probes": len(scored),
        "probes": {},
    }
    for (arm_id, seed), sc in sorted(scored.items()):
        tr, res = sc["training"], sc["result"]
        pres = tr.get("preserved") or {}
        inventory["probes"][tr["probe_id"]] = {
            "arm": arm_id, "seed": seed,
            "initialization_artifact_digest":
                tr["initialization_artifact_digest"],
            "config_sha256": tr["config_sha256"],
            "trained": bool(tr.get("complete")),
            "train_minutes": tr.get("train_minutes"),
            "preserved": bool(pres.get("preserved")),
            "preserved_bytes": pres.get("bytes"),
            "preserved_content_sha256": pres.get("content_sha256"),
            "preserved_relay_repo": pres.get("relay_repo"),
            "preserved_relay_prefix": pres.get("relay_prefix"),
            "scored": True,
            "counts": {k: res[k] for k in ("n", "usable", "correct",
                                           "n_scorable", "usable_scorable")},
            "generation_fingerprint": sc["admission"]["generation_fingerprint"],
        }
    trained = sum(1 for v in inventory["probes"].values() if v["trained"])
    preserved = sum(1 for v in inventory["probes"].values() if v["preserved"])
    inventory["totals"] = {
        "trained": trained, "preserved": preserved,
        "scored": len(inventory["probes"]),
        "preserved_bytes_total": sum(
            v["preserved_bytes"] or 0 for v in inventory["probes"].values()),
    }

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "probe_inventory.json").write_text(
        json.dumps(inventory, indent=1) + "\n")
    (a.out / "c3_probe_results.json").write_text(
        json.dumps(results, indent=2) + "\n")
    (a.out / "c3_decision.json").write_text(
        json.dumps(decision, indent=2) + "\n")
    print(f"\nPRIMARY ({decision['primary_contrast']}): {decision['verdict']}"
          f"  delta {decision['delta']:+.6f}  "
          f"LCB {decision['lcb_one_sided']:+.6f}")
    print(f"written to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
