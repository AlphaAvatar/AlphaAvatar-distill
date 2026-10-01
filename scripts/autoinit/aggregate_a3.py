#!/usr/bin/env python3
"""The A3 comparison. OFF POD, at `$0`, from preserved evidence.

    PYTHONPATH=src:scripts python scripts/autoinit/aggregate_a3.py \
        --evidence /home/ecs-user/aad-artifacts/phase_a3/<attempt> --write

**Why this is not a pod stage.** attempt75 trained, preserved and scored all
nine of its probes and then lost its decision artifact to a crash in the
on-pod aggregation — a C1 arm-vocabulary default meeting a C3 arm id, after
919 minutes and `$16.71`. The verdict was recovered by running the aggregation
off pod from the immutable evidence, which is what this module does by
construction. A3's driver ends at preservation; nothing scientific depends on
the pod surviving one more stage.

**What it compares.** Three newly trained A-bsz3 probes against attempt75's
three matched-seed `A_incumbent` controls, paired at prompt level within each
seed. The controls are READ from the committed stage-I record; their weights
were retired on 2026-10-01 and are not needed, because a control contributes
its per-sample evidence to this comparison and never its bytes.

**What it refuses.** A field that is not one field. The battery, the scoring
contract, the generation protocol fingerprint, the prompt set and the strata
must match attempt75's exactly; C2's confirmation set carried three generation
fingerprints and produced no canonical verdict for precisely this reason. A
mismatch here is an integrity failure, not a result, and this module raises
rather than reporting a number.

**What it is not.** A formal population-level non-inferiority proof. The
interval is descriptive, the SESOI is reported as the scale of a material loss,
and the cross-session limitation is carried in the output rather than left for
a reader to remember.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from experiments.phase_c1.probe_results import (  # noqa: E402
    N_PROMPTS, N_SCORABLE, PRIMARY_STRATA,
)

STAGE_I = REPO / "logs/stages/stage-1/phase_c3/analyses/attempt75_stage_i"
DESIGN = REPO / "logs/stages/stage-1/phase_c3/plans/a3_design.json"
OUT = "logs/stages/stage-1/phase_a3/analyses"

CONTROL_ARM = "A_incumbent"
TREATMENT_ARM = "A_bsz3"


class A3AggregationError(RuntimeError):
    """The A3 comparison cannot be computed from the evidence as it stands."""


# --- inputs ---------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_controls() -> dict[int, dict[str, Any]]:
    """attempt75's three A_incumbent probes, from the committed record."""
    doc = json.loads((STAGE_I / "c3_probe_results.json").read_text())
    out = {p["seed"]: p for p in doc["probes"] if p["arm"] == CONTROL_ARM}
    if len(out) != 3:
        raise A3AggregationError(
            f"expected three {CONTROL_ARM} controls, found {sorted(out)}")
    return out


def control_field() -> dict[str, Any]:
    """The protocol identity the treatment must match, from the same record."""
    doc = json.loads((STAGE_I / "c3_probe_results.json").read_text())
    return {
        "battery": doc["battery"]["artifact"],
        "battery_manifest_sha256": doc["battery"]["manifest_sha256"],
        "battery_content_sha256": doc["battery"]["content_sha256"],
        "scoring_contract": doc["scoring_contract"]["contract"],
        "scoring_contract_digest": doc["scoring_contract"]["digest"],
        "generation_protocol_fingerprint": doc["observed_generation_fingerprint"],
        "evaluation_protocol_hash": doc["observed_evaluation_protocol_hash"],
        "n_prompts": doc["decision_inputs_audit"]["n_prompts"],
        "n_scorable": doc["decision_inputs_audit"]["n_scorable"],
        "strata_sizes": doc["decision_inputs_audit"]["strata_sizes"],
    }


def _rows(path: Path) -> dict[str, dict[str, Any]]:
    """Per-sample rows, keyed by prompt id, with the duplicates refused."""
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        pid = row["id"]
        if pid in out:
            raise A3AggregationError(f"{path.name}: duplicate prompt id {pid!r}")
        out[pid] = row
    if len(out) != N_PROMPTS:
        raise A3AggregationError(
            f"{path.name}: {len(out)} prompts, not {N_PROMPTS}")
    return out


def assert_one_field(treatment_results: dict[int, dict[str, Any]],
                     expected: dict[str, Any]) -> dict[str, Any]:
    """The treatment field must be the control field. Raises, never reports.

    Checked per probe rather than once: a session that drifted mid-run would
    otherwise pass on its first probe's metadata.
    """
    observed: dict[str, set[str]] = {}
    for seed, res in sorted(treatment_results.items()):
        got = {
            "battery": res["battery"]["artifact"],
            "battery_manifest_sha256": res["battery"]["manifest_sha256"],
            "battery_content_sha256": res["battery"]["content_sha256"],
            "scoring_contract": res["scoring_contract"]["contract"],
            "scoring_contract_digest": res["scoring_contract"]["digest"],
            "generation_protocol_fingerprint":
                res["generation_protocol_fingerprint"],
        }
        for key, value in got.items():
            observed.setdefault(key, set()).add(str(value))
            if key in expected and str(expected[key]) != str(value):
                raise A3AggregationError(
                    f"probe at seed {seed} does not share attempt75's field: "
                    f"{key} is {value!r} and the controls' is "
                    f"{expected[key]!r}. The comparison would pool two fields "
                    "as one. This is an integrity failure, not a result.")
    plural = {k: sorted(v) for k, v in observed.items() if len(v) > 1}
    if plural:
        raise A3AggregationError(
            f"the treatment probes do not share one field among themselves: "
            f"{plural}")
    return {k: sorted(v)[0] for k, v in observed.items()}


# --- statistics -----------------------------------------------------------


def paired_deltas(control_rows: dict[int, dict[str, dict[str, Any]]],
                  treatment_rows: dict[int, dict[str, dict[str, Any]]],
                  ) -> dict[str, Any]:
    """Per-seed and pooled correctness, paired at prompt level.

    The scorable prompt set must be identical across every probe; a set that
    differs would still produce a number, and that number would be an average
    over different questions.
    """
    seeds = sorted(control_rows)
    if sorted(treatment_rows) != seeds:
        raise A3AggregationError(
            f"seeds differ: controls {seeds}, treatment {sorted(treatment_rows)}")

    reference: set[str] | None = None
    strata: dict[str, str] = {}
    for by_seed in (control_rows, treatment_rows):
        for seed, rows in by_seed.items():
            scorable = {pid for pid, r in rows.items() if r["scorable"]}
            if len(scorable) != N_SCORABLE:
                raise A3AggregationError(
                    f"seed {seed}: {len(scorable)} scorable prompts, not "
                    f"{N_SCORABLE}")
            if reference is None:
                reference = scorable
            elif scorable != reference:
                raise A3AggregationError(
                    f"seed {seed}: the scorable prompt set differs from the "
                    "reference probe; an incomplete field must not be "
                    "silently averaged")
            for pid in scorable:
                s = rows[pid]["set"]
                if s not in PRIMARY_STRATA:
                    raise A3AggregationError(
                        f"scorable prompt {pid!r} is in set {s!r}, not one of "
                        f"the frozen strata {list(PRIMARY_STRATA)}")
                if strata.setdefault(pid, s) != s:
                    raise A3AggregationError(
                        f"prompt {pid!r} changes stratum between probes")
    assert reference is not None
    ids = sorted(reference)

    per_seed, mcnemar = [], []
    for seed in seeds:
        c, t = control_rows[seed], treatment_rows[seed]
        cc = sum(1 for pid in ids if c[pid]["correct"])
        tc = sum(1 for pid in ids if t[pid]["correct"])
        b = sum(1 for pid in ids
                if t[pid]["correct"] and not c[pid]["correct"])
        d = sum(1 for pid in ids
                if c[pid]["correct"] and not t[pid]["correct"])
        per_seed.append({
            "seed": seed,
            "control_correct": cc, "treatment_correct": tc,
            "control_correct_overall": round(cc / len(ids), 6),
            "treatment_correct_overall": round(tc / len(ids), 6),
            "delta": round((tc - cc) / len(ids), 6),
        })
        mcnemar.append({"seed": seed, "treatment_only_correct": b,
                        "control_only_correct": d,
                        "discordant": b + d,
                        "_convention": "b = gained by treatment, d = lost"})

    pooled_delta = sum(r["delta"] for r in per_seed) / len(per_seed)
    return {
        "n_scorable": len(ids), "seeds": seeds,
        "per_seed": per_seed, "mcnemar": mcnemar,
        "pooled_delta": round(pooled_delta, 6),
        "_aggregation": ("the mean over the three fixed paired seeds of the "
                         "prompt-mean difference, which is C3's own "
                         "aggregation"),
        "_strata": strata,
        "_ids": ids,
    }


def stratified_bootstrap(control_rows, treatment_rows, deltas, *, seed: int,
                         iterations: int = 20000,
                         level: float = 0.95) -> dict[str, Any]:
    """A DESCRIPTIVE interval, by the same machinery C3 used.

    Prompts are resampled with replacement WITHIN each stratum, independently
    across strata; seeds are FIXED BLOCKS and are not resampled. That is the
    claim boundary: uncertainty over the prompt distribution CONDITIONAL on
    these three checkpoint pairs, and not an interval over hypothetical future
    seeds.
    """
    import random

    ids, strata = deltas["_ids"], deltas["_strata"]
    by_stratum: dict[str, list[str]] = {}
    for pid in ids:
        by_stratum.setdefault(strata[pid], []).append(pid)
    for v in by_stratum.values():
        v.sort()

    #: Per prompt, the mean over seeds of the paired correctness difference.
    diff = {}
    for pid in ids:
        diff[pid] = sum(
            (1 if treatment_rows[s][pid]["correct"] else 0)
            - (1 if control_rows[s][pid]["correct"] else 0)
            for s in deltas["seeds"]) / len(deltas["seeds"])

    rng = random.Random(seed)
    total = len(ids)
    stats = []
    for _ in range(iterations):
        #: n_h draws WITHIN each stratum, summed, then divided by the total.
        #: That is the size-weighted mean of the stratum means, which equals
        #: the unstratified mean while the stratum sizes are held at their
        #: observed values -- C3's stated convention, kept so the two
        #: intervals are commensurable.
        acc = 0.0
        for members in by_stratum.values():
            n = len(members)
            for _ in range(n):
                acc += diff[members[rng.randrange(n)]]
        stats.append(acc / total)
    stats.sort()
    alpha = 1.0 - level
    lo = stats[int(alpha * iterations)]
    hi = stats[min(iterations - 1, int((1 - alpha) * iterations))]
    two_lo = stats[int(alpha / 2 * iterations)]
    two_hi = stats[min(iterations - 1, int((1 - alpha / 2) * iterations))]
    return {
        "method": "stratified prompt-cluster bootstrap",
        "iterations": iterations, "seed": seed, "level": level,
        "n_prompts": total, "n_strata": len(by_stratum),
        "lcb_one_sided": round(lo, 6), "ucb_one_sided": round(hi, 6),
        "ci_two_sided_low": round(two_lo, 6),
        "ci_two_sided_high": round(two_hi, 6),
        "algorithm": "python.random.Random(seed).randrange, Mersenne Twister",
        "resamples": "prompts within strata; seeds are FIXED BLOCKS",
        "_descriptive_only": (
            "this is NOT a non-inferiority test. It describes prompt-level "
            "uncertainty conditional on these three checkpoint pairs."),
        "claim_boundary": (
            "prompt-distribution uncertainty CONDITIONAL ON the three "
            "preregistered recovery seeds; NOT an interval over hypothetical "
            "future seeds, and NOT a population-level statement"),
    }


def behaviour(control_results, treatment_results) -> dict[str, Any]:
    """usable_rollout and EVERY component rate, per seed and pooled.

    Reported together and never combined. `protocol_valid` subsumes two of the
    components by construction, so the conjunction must not be presented as
    five agreeing checks.
    """
    components = ("non_empty", "natural_termination", "no_severe_repetition",
                  "no_context_limit", "protocol_valid")
    rows = []
    for seed in sorted(control_results):
        c, t = control_results[seed], treatment_results[seed]
        row = {"seed": seed,
               "control_usable_rollout_rate": c["usable_rollout_rate"],
               "treatment_usable_rollout_rate": t["usable_rollout_rate"],
               "usable_delta": round(t["usable_rollout_rate"]
                                     - c["usable_rollout_rate"], 6)}
        for comp in components:
            if comp in c and comp in t:
                row[f"control_{comp}"] = c[comp]
                row[f"treatment_{comp}"] = t[comp]
                row[f"{comp}_delta"] = round(t[comp] - c[comp], 6)
        rows.append(row)
    pooled = sum(r["usable_delta"] for r in rows) / len(rows)
    return {
        "per_seed": rows, "pooled_usable_delta": round(pooled, 6),
        "components_reported": list(components),
        "_no_weighted_scalar": (
            "usable_rollout_rate and correct_overall are reported separately "
            "and never combined; usable_rollout is blind to correctness by "
            "construction"),
        "_components_are_not_independent": (
            "protocol_valid subsumes two of them, so the conjunction is not "
            "five agreeing checks"),
    }


def breakdowns(control_results, treatment_results) -> dict[str, Any]:
    """Per-capability, per-domain and per-set deltas, from the scored records."""
    out: dict[str, Any] = {}
    for axis in ("per_capability", "per_domain", "per_set"):
        rows: dict[str, Any] = {}
        keys: set[str] = set()
        for res in list(control_results.values()) + list(treatment_results.values()):
            keys |= set(res.get(axis) or {})
        for key in sorted(keys):
            def mean(results, field):
                vals = [(r.get(axis) or {}).get(key, {}).get(field)
                        for r in results.values()]
                vals = [v for v in vals if isinstance(v, (int, float))]
                return round(sum(vals) / len(vals), 6) if vals else None
            c_corr = mean(control_results, "correct_overall")
            t_corr = mean(treatment_results, "correct_overall")
            c_use = mean(control_results, "usable_rollout_rate")
            t_use = mean(treatment_results, "usable_rollout_rate")
            rows[key] = {
                "control_correct_overall": c_corr,
                "treatment_correct_overall": t_corr,
                "correct_delta": (round(t_corr - c_corr, 6)
                                  if None not in (c_corr, t_corr) else None),
                "control_usable_rollout_rate": c_use,
                "treatment_usable_rollout_rate": t_use,
                "usable_delta": (round(t_use - c_use, 6)
                                 if None not in (c_use, t_use) else None),
            }
        out[axis] = rows
    return out


def guardrails(behaviour_block: dict[str, Any],
               design_doc: dict[str, Any]) -> dict[str, Any]:
    """C3's behavioural guardrails, REPORTED as safety observations.

    They do not gate A3's completion: the 2026-10-01 decision wants the
    complete three-seed result, and a guardrail that ended the chain would be
    the fail-fast under another name. A firing guardrail is a finding the
    closeout must state prominently.
    """
    g = design_doc["aggregation_and_reporting"]["safety_checks_reused_from_c3"]
    pooled_min = float(g["pooled_usable_delta_min"])
    per_seed_min = float(g["per_seed_usable_delta_min"])
    pooled = behaviour_block["pooled_usable_delta"]
    per_seed = [r["usable_delta"] for r in behaviour_block["per_seed"]]
    fired = []
    if pooled < pooled_min:
        fired.append(f"pooled usable delta {pooled} < {pooled_min}")
    for row in behaviour_block["per_seed"]:
        if row["usable_delta"] < per_seed_min:
            fired.append(f"seed {row['seed']} usable delta "
                         f"{row['usable_delta']} < {per_seed_min}")
    return {
        "pooled_usable_delta_min": pooled_min,
        "per_seed_usable_delta_min": per_seed_min,
        "observed_pooled": pooled, "observed_per_seed": per_seed,
        "fired": fired, "any_fired": bool(fired),
        "_role": ("reported, not gating. A3 completes all three seeds and the "
                  "closeout states a firing guardrail prominently."),
    }


# --- the comparison -------------------------------------------------------


def _read_treatment(evidence: Path) -> tuple[dict[int, dict[str, Any]],
                                             dict[int, Path]]:
    """The treatment probes' scored records and per-sample row files."""
    audit = evidence / "audit" / "autoinit_a3"
    if not audit.is_dir():
        raise A3AggregationError(
            f"no treatment evidence at {audit}; A3's driver writes its scored "
            "records there and the comparison reads them rather than "
            "recomputing anything")
    results: dict[int, dict[str, Any]] = {}
    rows: dict[int, Path] = {}
    for result in sorted(audit.glob("*_c1_confirmation.json")):
        doc = json.loads(result.read_text())
        seed = int(doc["seed"])
        if seed in results:
            raise A3AggregationError(f"two scored records for seed {seed}")
        results[seed] = doc
        per_sample = audit / f"{doc['label']}_per_sample.jsonl"
        if not per_sample.is_file():
            raise A3AggregationError(
                f"seed {seed} has a scored aggregate and no per-sample rows at "
                f"{per_sample}; the paired comparison reads the rows")
        rows[seed] = per_sample
    if not results:
        raise A3AggregationError(f"{audit} holds no scored A3 records")
    return results, rows


def load_control_results() -> dict[int, dict[str, Any]]:
    """The controls' FULL scored records, hash-verified against the summary.

    The committed `c3_probe_results.json` carries counts, rates and
    per-capability figures. It does NOT carry the usable-rollout component
    rates or the per-domain and per-set axes, and A3 is required to report all
    of them -- so the comparison reads each control's own
    `*_c1_confirmation.json` from the durable evidence and checks it against
    the `result_sha256` the committed record names. Reading the summary and
    reporting components it never held is how a report comes to state numbers
    nothing produced.
    """
    out: dict[int, dict[str, Any]] = {}
    for seed, probe in load_controls().items():
        p = Path(probe["result_path"])
        if not p.is_file():
            raise A3AggregationError(
                f"control seed {seed} names its scored record at {p}, which "
                "is not present. A3 reports the usable-rollout components and "
                "the per-domain axes, and the committed summary does not "
                "carry them.")
        digest = _sha256_file(p)
        if digest != probe["result_sha256"]:
            raise A3AggregationError(
                f"control seed {seed} scored record hashes to {digest[:12]} "
                f"and the committed record says {probe['result_sha256'][:12]}; "
                "the evidence has changed since it was scored")
        out[seed] = json.loads(p.read_text())
    return out


def _control_rows() -> dict[int, Path]:
    """attempt75's per-sample row files, from the paths its record names."""
    controls = load_controls()
    out: dict[int, Path] = {}
    for seed, probe in controls.items():
        p = Path(probe["per_sample_path"])
        if not p.is_file():
            raise A3AggregationError(
                f"control seed {seed} names per-sample rows at {p}, which is "
                "not present. The controls' EVIDENCE is what A3 consumes; "
                "their weights were retired and are not the problem here.")
        digest = _sha256_file(p)
        if digest != probe["per_sample_sha256"]:
            raise A3AggregationError(
                f"control seed {seed} per-sample rows hash to {digest[:12]} "
                f"and the record says {probe['per_sample_sha256'][:12]}; the "
                "evidence has changed since it was scored")
        out[seed] = p
    return out


def build(evidence: Path, *, bootstrap_seed: int,
          iterations: int = 20000) -> dict[str, Any]:
    """The whole comparison, from evidence to one artifact."""
    design_doc = json.loads(DESIGN.read_text())
    controls = load_controls()
    expected = control_field()
    treatment_results, treatment_row_paths = _read_treatment(evidence)

    missing = sorted(set(controls) - set(treatment_results))
    if missing:
        raise A3AggregationError(
            f"the treatment is missing seeds {missing}; A3's estimand is a "
            "paired difference over three seeds and fewer than three is not "
            "that quantity")

    observed_field = assert_one_field(treatment_results, expected)

    control_row_paths = _control_rows()
    c_rows = {s: _rows(p) for s, p in control_row_paths.items()}
    t_rows = {s: _rows(p) for s, p in treatment_row_paths.items()}

    #: The controls' FULL scored records, so the components and the extra axes
    #: come from what was actually scored rather than from a summary.
    control_results = load_control_results()

    deltas = paired_deltas(c_rows, t_rows)
    interval = stratified_bootstrap(c_rows, t_rows, deltas,
                                    seed=bootstrap_seed, iterations=iterations)
    beh = behaviour(control_results, treatment_results)
    brk = breakdowns(control_results, treatment_results)
    rails = guardrails(beh, design_doc)

    consumed = {
        "a3_design.json": design_doc["design_sha256"],
        "c3_probe_results.json": _sha256_file(STAGE_I / "c3_probe_results.json"),
    }
    for seed, probe in sorted(controls.items()):
        consumed[f"control_scored_{seed}"] = probe["result_sha256"]
    for seed, p in sorted(control_row_paths.items()):
        consumed[f"control_per_sample_{seed}"] = _sha256_file(p)
    for seed, p in sorted(treatment_row_paths.items()):
        consumed[f"treatment_per_sample_{seed}"] = _sha256_file(p)

    doc = {
        "schema": "aadistill.phase_a3.comparison/v1",
        "_what_this_is": (
            "The complete A3 engineering/scientific comparison: three newly "
            "trained A-bsz3 probes against attempt75's three matched-seed "
            "A_incumbent controls, computed OFF POD at $0 from preserved "
            "evidence."),
        "_what_it_is_not": (
            "a formal population-level non-inferiority proof. The interval is "
            "descriptive and the SESOI is the scale of a material loss, not a "
            "decision rule."),
        "experiment_id": "phase_a3",
        "treatment_arm": TREATMENT_ARM, "control_arm": CONTROL_ARM,
        "control_source_run": "attempt75",
        "controls_retrained": False,
        "field_is_one_field": observed_field,
        "protocol_identity_matched": expected,
        "correctness": deltas | {"_strata": "omitted", "_ids": "omitted"},
        "interval": interval,
        "behaviour": beh,
        "breakdowns": brk,
        "guardrails": rails,
        "sesoi": 0.010,
        "_sesoi_role": (
            "reported as the SCALE of a material loss. No threshold here "
            "decides anything."),
        "claim_boundary": design_doc["claim_boundary"],
        "consumed_inputs_sha256": consumed,
        "_aggregation_ran_off_pod": (
            "attempt75 lost its on-pod decision artifact after a complete "
            "measurement. A3's comparison never runs on the meter."),
    }
    doc["comparison_sha256"] = hashlib.sha256(
        json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--evidence", required=True,
                    help="the A3 attempt's durable evidence root")
    ap.add_argument("--bootstrap-seed", type=int, default=None,
                    help="omit to use the C3 preregistration's phase-c3 seed")
    ap.add_argument("--iterations", type=int, default=20000)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    seed = args.bootstrap_seed
    if seed is None:
        #: PASSED EXPLICITLY, never defaulted. C3's records asserted a
        #: bootstrap seed that the resampler never received, because
        #: `isolation.bootstrap_seed()` carries C1's value as a default and
        #: four of the five parameters coincided.
        seed = int(json.loads(
            (REPO / "logs/stages/stage-1/phase_c3/plans/"
                    "c3_preregistration.json").read_text())["seeds"]["bootstrap"])

    doc = build(Path(args.evidence), bootstrap_seed=seed,
                iterations=args.iterations)
    c = doc["correctness"]
    print("A3 comparison  (A_bsz3 - A_incumbent, paired)")
    print(f"  bootstrap seed      {seed}")
    for row in c["per_seed"]:
        print(f"  seed {row['seed']:>11d}   control "
              f"{row['control_correct_overall']:.4f}  treatment "
              f"{row['treatment_correct_overall']:.4f}  delta "
              f"{row['delta']:+.6f}")
    print(f"  pooled delta        {c['pooled_delta']:+.6f}")
    i = doc["interval"]
    print(f"  descriptive 95% CI  [{i['ci_two_sided_low']:+.6f}, "
          f"{i['ci_two_sided_high']:+.6f}]  (one-sided LCB "
          f"{i['lcb_one_sided']:+.6f})")
    print(f"  pooled usable delta {doc['behaviour']['pooled_usable_delta']:+.6f}")
    print(f"  guardrails fired    {doc['guardrails']['fired'] or 'none'}")
    print(f"  comparison_sha256   {doc['comparison_sha256']}")

    if args.write:
        out = Path(args.out) if args.out else (REPO / OUT / "a3_comparison.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
