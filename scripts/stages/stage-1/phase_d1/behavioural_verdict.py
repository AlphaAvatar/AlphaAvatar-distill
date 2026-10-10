"""D1's CONFIRMATION verdict, off-pod, under the already frozen rule.

    PYTHONPATH=src:scripts .venv/bin/python \
        scripts/stages/stage-1/phase_d1/behavioural_verdict.py \
        --store <scr>/store/probes --advancing q2 --run-id <confirmation run> \
        --out logs/.../runs/<run>/evidence/d1_confirmation_verdict.json

**THIS MODULE IMPLEMENTS NOTHING STATISTICAL.** Every rule it applies is
C0's, already written and already used by C1 (which returned GO) and C3
(which returned NO_GO), imported from the modules that own them:

    probe_results.decision_inputs        align 6 probes into paired vectors,
                                         refusing a duplicate id, a short
                                         prompt set, a non-frozen stratum or
                                         a scorable set that differs by probe
    isolation.paired_differences         d_j = mean_s[cand(j,s) - B(j,s)]
    isolation.stratified_cluster_bootstrap  resample PROMPTS within strata,
                                         each prompt's whole arm x seed vector
                                         kept together, seeds NOT resampled
    isolation.decide                     the three-way GO / NO-GO /
                                         INCONCLUSIVE rule, no forced winner

A second implementation of any of them would be a second thing that can
disagree about what D1 measured, and the comparability with C1's GO is the
whole reason the endpoint was inherited rather than re-derived.

**WHY IT RUNS OFF-POD.** A3 trained nine probes over 919 minutes and lost its
decision artifact to a crash in an on-pod aggregation stage. The confirmation
driver therefore ends at preservation, and this reads the secured per-sample
rows afterwards at `$0` -- recomputable, as many times as a reviewer likes,
from evidence that is already durable.

**THE BOOTSTRAP SEED IS DOMAIN-SEPARATED PER PHASE, and C3 paid to learn it.**
`isolation.bootstrap_seed()` defaults to `phase-c1:bootstrap` = 816109261;
C3's own plan froze 654678655 and the driver did not pass it, so the resampler
silently used C1's. Four of the five parameters coincided exactly, which is why
only the seed exposed the mismatch. D1 derives its own below, from its own
frozen design hash, and passes it explicitly -- never the default.

**WHAT THIS MODULE MAY NOT DO:** change the rule, the SESOI, the guardrails or
the estimand in response to what the numbers turn out to be. It is written
and committed BEFORE any confirmation probe exists, which is the only time
that promise is checkable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
for _extra in ("src", "scripts", "scripts/stages/stage-1"):
    _path = str(REPO_ROOT / _extra)
    if _path not in sys.path:
        sys.path.insert(0, _path)

SCHEMA = "aadistill.phase_d1.confirmation_verdict/v1"


class D1VerdictError(RuntimeError):
    """The confirmation evidence cannot support the frozen rule."""


def bootstrap_seed(repo_root: Path = REPO_ROOT) -> int:
    """D1's resampling seed, derived from D1's own frozen design identity.

    Same shape and same domain-separation discipline as C1's, with D1's
    separator, so the two phases cannot coincide and neither can be reached by
    the other's default. Derived rather than chosen: the base is the
    `design_hash` that predates every D1 behavioural result, so there is no
    discretion left to exercise after seeing a number.
    """
    from stages.phase_d1 import behavioural as B

    base = B.design(repo_root)["design_hash"]
    digest = hashlib.sha256(
        f"{base}:phase-d1:confirmation-bootstrap".encode()).digest()
    return int.from_bytes(digest[:4], "big") % (2 ** 31)


@dataclass(frozen=True)
class D1ConfirmationPlan:
    """Exactly what `decide()` reads, carrying D1's frozen values.

    NOT a `C1IsolationPlan`, and the reason is scientific rather than
    mechanical: that type refuses two arms sharing an ATTENTION
    implementation, because C1's hypothesis WAS the attention operator. D1
    isolates the SCORING METHOD -- all four candidates and B are built by
    `attention.activation_importance_v1`, and what differs is the supervised
    target-aware scoring the search selected under. Reusing C1's plan type
    would either refuse every correct D1 verdict or require weakening a guard
    that is right for C1.

    `decide` reads five attributes and this carries exactly those, so the
    frozen rule is reused unchanged over D1's frozen numbers.
    """

    plan_id: str
    seeds: tuple[int, ...]
    candidate: str
    sesoi: float
    seed_robustness_min_positive: int
    usable_pooled_min_delta: float
    usable_per_seed_min_delta: float

    def __post_init__(self) -> None:
        if len(self.seeds) != 3:
            raise D1VerdictError(
                f"the confirmation rung has three frozen seeds; got "
                f"{list(self.seeds)}")
        if self.candidate == "B":
            raise D1VerdictError(
                "the candidate may not be the incumbent; a confirmation "
                "compares the advancing candidate AGAINST B")


def confirmation_plan(advancing: str,
                      repo_root: Path = REPO_ROOT) -> D1ConfirmationPlan:
    """The plan, every value READ from the record that froze it.

    And asked back afterwards, which is the check C3 added after discovering
    that a constructor silently carrying C1's defaults is indistinguishable
    from one carrying the frozen values -- until one of them differs.
    """
    from stages.phase_d1 import behavioural as B

    bd = B.behavioural_design(repo_root)
    plan = D1ConfirmationPlan(
        plan_id="autoinit.v1.phase_d1_behavioural.confirmation",
        seeds=tuple(B.confirmation_seeds(repo_root)),
        candidate=advancing,
        sesoi=float(bd["sesoi"]),
        #: C0's robustness condition: at least 2 of the 3 seed-specific
        #: deltas positive. A ROBUSTNESS condition, not a seed-level
        #: significance test -- the preregistration says so in as many words.
        seed_robustness_min_positive=2,
        usable_pooled_min_delta=B.GUARDRAIL_POOLED_MIN_DELTA,
        usable_per_seed_min_delta=B.GUARDRAIL_PER_SEED_MIN_DELTA,
    )
    if plan.sesoi != float(bd["sesoi"]):
        raise D1VerdictError("the plan does not carry the frozen SESOI")
    if (plan.usable_pooled_min_delta != B.GUARDRAIL_POOLED_MIN_DELTA
            or plan.usable_per_seed_min_delta
            != B.GUARDRAIL_PER_SEED_MIN_DELTA):
        raise D1VerdictError(
            "the plan's usable-rollout guardrails are not the frozen ones")
    return plan


def load_per_sample(store: Path, advancing: str, seeds: tuple[int, ...],
                    ) -> tuple[dict[tuple[str, int], list[dict]], dict[str, str]]:
    """The six confirmation probes' secured rows, keyed (arm, seed).

    Reads the destination the launcher's poll hook writes --
    `<store>/<probe_id>/per_sample.jsonl` -- so the verdict consumes the
    evidence that actually came home, not a path on a pod that no longer
    exists. Every file's sha256 is returned beside the rows, because an
    artifact that cannot name the bytes it consumed is not reproducible (P4).
    """
    rows: dict[tuple[str, int], list[dict]] = {}
    digests: dict[str, str] = {}
    for arm in ("B", advancing):
        for seed in seeds:
            probe_id = f"d1_confirmation_{arm}_s{seed}"
            path = store / probe_id / "per_sample.jsonl"
            if not path.is_file():
                raise D1VerdictError(
                    f"{path} is missing; the confirmation verdict consumes all "
                    f"six probes' secured per-sample rows and will not average "
                    "an incomplete design")
            data = path.read_bytes()
            digests[probe_id] = hashlib.sha256(data).hexdigest()
            rows[(arm, seed)] = [json.loads(line)
                                 for line in data.decode().splitlines()
                                 if line.strip()]
    return rows, digests


def compute(store: Path, advancing: str, *, repo_root: Path = REPO_ROOT,
            iterations: int | None = None) -> dict[str, Any]:
    """The verdict artifact. Every rule imported, every input bound."""
    from stages.phase_c1.isolation import (
        BOOTSTRAP_ITERATIONS, decide, paired_differences,
        stratified_cluster_bootstrap,
    )
    from stages.phase_c1.probe_results import decision_inputs
    from stages.phase_d1 import behavioural as B

    plan = confirmation_plan(advancing, repo_root)
    per_sample, digests = load_per_sample(store, advancing, plan.seeds)
    #: D1's OWN arm vocabulary. `decision_inputs` defaults to C1's
    #: ("incumbent", "treatment") and would refuse every D1 probe by name.
    inputs = decision_inputs(
        per_sample, seeds=plan.seeds,
        arms=("B", advancing),
        control_operand="B", candidate_operand=advancing)

    d = paired_differences(inputs.arm("B"), inputs.arm(advancing))
    seed = bootstrap_seed(repo_root)
    n_iter = int(iterations or BOOTSTRAP_ITERATIONS)
    boot = stratified_cluster_bootstrap(
        d, inputs.strata, seed=seed, iterations=n_iter)
    per_seed_delta = [
        sum(bool(inputs.correct[advancing][s][j])
            - bool(inputs.correct["B"][s][j]) for j in d) / len(d)
        for s in plan.seeds]
    decision = decide(
        plan, boot=boot, per_seed_delta=per_seed_delta,
        usable_pooled_delta=inputs.usable_pooled_delta,
        usable_per_seed_delta=list(inputs.usable_per_seed_delta),
        catastrophic_violations=inputs.catastrophic_violations)

    battery = B.battery_role("d1_confirmation", repo_root)
    prereg = json.loads(
        (repo_root / "logs/stages/stage-1/phase_d1/plans/"
                     "d1_behavioural_preregistration.json").read_text())
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, cwd=repo_root).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"],
                                capture_output=True, text=True,
                                cwd=repo_root).stdout.strip())
    out = {
        "schema": SCHEMA,
        "_contract": (
            "D1's confirmation verdict under the rule C0 froze and C1 and C3 "
            "already ran. Computed OFF-POD at $0 from secured per-sample "
            "rows, recomputable by anyone holding them."),
        "rung": "confirmation",
        "advancing_candidate": advancing,
        "incumbent": "B",
        "design_hash": B.design(repo_root)["design_hash"],
        "preregistration_sha256": prereg["preregistration_sha256"],
        "plan": {
            "plan_id": plan.plan_id,
            "seeds": list(plan.seeds),
            "sesoi": plan.sesoi,
            "seed_robustness_min_positive":
                plan.seed_robustness_min_positive,
            "usable_pooled_min_delta": plan.usable_pooled_min_delta,
            "usable_per_seed_min_delta": plan.usable_per_seed_min_delta,
            "_every_value_read_from": "the frozen design and C0",
        },
        "inference": {
            "method": "stratified prompt-cluster bootstrap",
            "owner": "stages.phase_c1.isolation.stratified_cluster_bootstrap",
            "iterations": n_iter,
            "bootstrap_seed": seed,
            "bootstrap_seed_derivation": (
                "SHA256(design_hash + ':phase-d1:confirmation-bootstrap')"
                "[0:4] as uint32 mod 2**31 -- domain-separated per phase, "
                "because C3 discovered that C1's default silently applies "
                "when a phase does not pass its own"),
            "seeds_are_fixed_blocks": True,
        },
        "decision": decision,
        "per_seed_delta": per_seed_delta,
        "battery": {k: battery[k] for k in (
            "role", "family_content_id", "item_ids_sha256", "n_prompts",
            "n_scorable")},
        "inputs_consumed": {
            "per_sample_sha256": digests,
            "n_probes": len(per_sample),
            "n_prompts_paired": len(d),
        },
        "implementation": {
            "commit": head,
            "dirty": dirty,
            "rule_owners": [
                "stages.phase_c1.probe_results.decision_inputs",
                "stages.phase_c1.isolation.paired_differences",
                "stages.phase_c1.isolation.stratified_cluster_bootstrap",
                "stages.phase_c1.isolation.decide",
            ],
            "_nothing_statistical_is_implemented_here": (
                "the endpoint, estimand, inference and SESOI are C0's and are "
                "reused unchanged so D1 stays comparable with C1's GO and "
                "C3's NO_GO"),
        },
        "claim_boundary": decision["claim_boundary"],
        "_what_this_verdict_is_about": (
            f"it confirms {advancing} against B, conditional on the three "
            "confirmation seeds, having been selected on disjoint screening "
            "prompts and disjoint screening seeds. It is not a statement "
            "about the other three finalists, and the interval is prompt "
            "uncertainty at these fixed seed pairs -- never a seed-population "
            "claim."),
    }
    from aadistill.infrastructure.manifest import sha256_json

    out["verdict_sha256"] = sha256_json(out)
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True, type=Path,
                    help="the secured probe store: <scr>/store/probes")
    ap.add_argument("--advancing", required=True,
                    help="the candidate screening advanced (q1..q4)")
    ap.add_argument("--out", default=None, type=Path)
    ap.add_argument("--iterations", type=int, default=None,
                    help="override only for a $0 self-check; the frozen "
                         "default is the one a verdict reports")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out = compute(args.store, args.advancing, iterations=args.iterations)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1) + "\n")
        print(f"wrote {args.out}")
    d = out["decision"]
    print(f"  verdict           {d['verdict']}")
    print(f"  delta             {d['delta']:+.6f}  (SESOI {d['sesoi']})")
    print(f"  one-sided 95% LCB {d['lcb_one_sided']:+.6f}")
    print(f"  one-sided 95% UCB {d['ucb_one_sided']:+.6f}")
    print(f"  per-seed deltas   "
          f"{[round(x, 6) for x in out['per_seed_delta']]}")
    print(f"  criteria          {d['criteria']}")
    if d["vetoes"]:
        print(f"  VETOES            {d['vetoes']}")
    print(f"  bootstrap seed    {out['inference']['bootstrap_seed']} "
          f"({out['inference']['iterations']} iterations)")
    print(f"  verdict_sha256    {out['verdict_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA", "D1ConfirmationPlan", "D1VerdictError", "bootstrap_seed",
           "compute", "confirmation_plan", "load_per_sample"]
