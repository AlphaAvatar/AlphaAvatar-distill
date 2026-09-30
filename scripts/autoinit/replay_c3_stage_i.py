#!/usr/bin/env python3
"""Recompute C3 stage I from a completed run's collected evidence. $0, CPU.

    PYTHONPATH=src:scripts python scripts/autoinit/replay_c3_stage_i.py \
        --evidence /home/ecs-user/aad-artifacts/phase_c3/attempt75 \
        --out /home/ecs-user/aad-artifacts/phase_c3/attempt75/stage_i_replay

**This re-measures NOTHING.** Every input is a frozen artifact the run already
produced and shipped off-pod: nine `*_c1_confirmation.json` aggregates, nine
`*_per_sample.jsonl` row files, the attested evaluation protocol, and the
preregistration. Stage I is a deterministic function of those plus the frozen
bootstrap seed, so running it here reproduces what the pod would have written
had it not raised.

**Why it is needed.** attempt75 trained, preserved and scored all nine probes
and then failed in stage I with

    C1ResultsError: autoinit.v1.phase_c3.A_incumbent.217230555:
                    unknown arm 'A_incumbent';
                    this record allows ['incumbent', 'treatment']

`C1ProbeRecord.allowed_arms` defaults to C1's two ROLES and the C3 driver never
passed C3's three ARM IDS, although the field exists for exactly that and says
so. The driver is fixed; this script proves the fix against the real inputs
that broke it, which no synthetic fixture can do.

**What it does NOT do.** It does not claim a C3 verdict. It writes
`c3_decision_replay.json`, which is a *recomputation*, explicitly marked as
such. Whether an off-pod recomputation may stand as the canonical C3 result is
a post-measurement scientific decision and belongs to the maintainer.
"""
from __future__ import annotations

import argparse
import json
import re
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
        }
    return scored, per_sample


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--run-id", default="attempt75")
    a = ap.parse_args()

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
    #: EXACTLY the primary contrast's two arms, rebuilt as the driver builds
    #: it: `C1IsolationPlan` holds two and `C1Arm.role` must be
    #: incumbent/treatment, so the third arm is reported in both secondary
    #: contrasts and is simply not what the decision rule reads.
    battery = json.loads(BATTERY_IDENTITY.read_text())
    plan = C1IsolationPlan(
        plan_id="autoinit.v1.phase_c3",
        arms=(C1Arm(f"c3.{control}", "incumbent", *CS.arm(control)["attention"]),
              C1Arm(f"c3.{candidate}", "treatment",
                    *CS.arm(candidate)["attention"])),
        seeds=tuple(seeds),
        battery_asset_id=battery["asset_id"],
        battery_content_sha256=battery["content_sha256"])
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
        boot = stratified_cluster_bootstrap(d, inputs.strata, seed=seed)
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
    prereg_seed = CS.bootstrap_seed()
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
        "_schema": "aadistill.phase_c3.decision_replay/v1",
        "_this_is_a_RECOMPUTATION_not_a_claimed_verdict": (
            "Stage I of " + a.run_id + " raised C1ResultsError before writing "
            "c3_decision.json. This recomputes it off-pod from that run's "
            "frozen, collected evidence -- the same nine per-sample row files, "
            "the same nine aggregates, the same preregistration and the same "
            "bootstrap seed -- so it is deterministic and re-measures nothing. "
            "Whether it may stand as the CANONICAL C3 verdict is a "
            "post-measurement scientific decision for the maintainer, and is "
            "NOT asserted here."),
        "run_id": a.run_id,
        "plan_hash": plan.plan_hash,
        "probe_results_sha256": results["results_sha256"],
        "preregistration_sha256": prereg["preregistration_sha256"],
        "primary_contrast": prereg["claim_boundary"]["primary_contrast"],
        "_verdict_owner": ("the PRIMARY contrast alone. The secondary and "
                           "practical contrasts are reported completely and "
                           "do not redefine it."),
        "contrasts": reported,
        "terminal_semantics": prereg["terminal_outcomes"],
        "generation_protocol_fingerprint": next(iter(fingerprints)),
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

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "c3_probe_results_replay.json").write_text(
        json.dumps(results, indent=2) + "\n")
    (a.out / "c3_decision_replay.json").write_text(
        json.dumps(decision, indent=2) + "\n")
    print(f"\nPRIMARY ({decision['primary_contrast']}): {decision['verdict']}"
          f"  delta {decision['delta']:+.6f}  "
          f"LCB {decision['lcb_one_sided']:+.6f}")
    print(f"written to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
