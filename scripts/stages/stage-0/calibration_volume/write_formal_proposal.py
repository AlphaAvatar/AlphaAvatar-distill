#!/usr/bin/env python3
"""The formal volume study as amended by the maintainer, designed and priced.

    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-0/calibration_volume/write_formal_proposal.py \
        --telemetry <a root-level search telemetry.jsonl on these profiles> \
        --telemetry-gpu "NVIDIA A100-SXM4-80GB" --price-per-hour 1.79 --gpu "NVIDIA A100-SXM4-80GB"

Writes `logs/stages/stage-0/calibration_volume/plans/formal_design_proposal.json`.
It is a PROPOSAL, never a preregistration: the grant opens the formal phase only
on `plans/volume_preregistration.json`, which needs the money below.

**Maintainer amendment 2026-10-18.** No independent resamples: one D1 full
search and one fixed recovery per dose, 1x / 2x / 4x, from one frozen base
mixture and one nested draw, with the same teacher, geometry, forward
baseline, recovery recipe, recovery seed and evaluation contract
(`data/eval-set/eval-set-1`). The study can claim only comparisons between
these three treatments; selection variance and statistical saturation are
out of reach without resamples and are never claimed. Composition (realized
UESP, sources, rule families) is recorded per dose as covariates, and a
family-diversity change is never reported as a pure volume effect. The pilot's
exploratory resampled measurements stand and are not repeated.

Pricing is measured, never assumed:

* a full search's trajectory is the forward baseline's measured one -- D2's
  92 expansions per implementation (`search_analysis.json`); each
  implementation's minutes are scaled by its ROOT minutes on this programme's
  profiles at the dose (the pilot telemetry) over its D2 root minutes;
* per-expansion overhead is the worst measured (1.25 min here, 0.14 in D2);
* recovery is the fixed recipe's measured training time (D1 confirmation
  probes, 1,023 steps) and evaluation is eval-set-1 at the measured per-model
  rates; session overheads, contingency and reserves are the launcher's.

The 4x search does not fit one $30 session. It is split into sessions over
ONE search identity on a persistent network volume (the engine's journal
resume is exact and tested: `tests/initialization/test_search.py ::
test_resume_is_deterministic_and_recomputes_nothing`), never by changing the
selected D1 baseline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.calibration_volume import governance as G  # noqa: E402

LOGS = REPO / "logs/stages/stage-0/calibration_volume"
CONF = REPO / "configs/stages/stage-0/calibration_volume"
D2_ANALYSIS = REPO / "logs/stages/stage-1/phase_d2/runs/d2_search_20261015_051722/evidence/search_analysis.json"
D2_TELEMETRY = Path("/home/ecs-user/aad-artifacts/phase_d2/d2_search_20261015_051722/store/"
                    "extracted/audit/autoinit_d2/search/telemetry.jsonl")
D2_OVERHEAD_MIN = 0.14
#: The fixed recipe's measured training time (1,023 steps, 0.86M rung) on an
#: L40S; the A100 is not slower, so this bounds it there.
PROBE_RECORD = REPO / "logs/stages/stage-1/phase_d1/runs/d1_behavioural_20261012_064341/closeout/outcome.json"
#: eval-set-1 per recovered student at its trained context (8,192), A100:
#: fluid_reasoning_720 measured 13.0-18.8 min per student (battery_20261018_092345);
#: the 996 static prompts of the other two components are bounded by the D1
#: confirmation battery's 950 prompts at the same context (20.1-29.6 min, L40S).
EVAL_MINUTES = {"fluid_reasoning_720": 19.0, "stage3_capability_846+stage3_legacy_150": 30.0}
VOLUME_GB = 400                  # every materialized state of one 4x search (5-7 GB each)
VOLUME_USD_PER_GB_MONTH = 0.07
SESOI = {"pooled_primary": 0.05, "per_axis": 0.08}
#: The resolution gate's floor: every fixed-recipe student measured so far sits
#: inside it (0/720, 0/720, 10/720), and a student inside it cannot register a
#: treatment difference the size of the SESOI on the static items.
FLOOR_STATIC_RATE = 0.02


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rel(p: Path) -> str:
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)


def d2_root_minutes() -> dict[str, float]:
    """D2's ROOT operator minutes per implementation (max over its two profiles)."""
    depth = {}
    for line in D2_TELEMETRY.with_name("states.jsonl").read_text().splitlines():
        s = json.loads(line)
        depth[s["state_id"]] = s.get("depth")
    out: dict[str, float] = defaultdict(float)
    for line in D2_TELEMETRY.read_text().splitlines():
        r = json.loads(line)
        if r.get("event") == "expansion" and depth.get(r["state_id"]) == 1:
            out[r["impl_id"]] = max(out[r["impl_id"]], r["operator_seconds"] / 60)
    return dict(out)


def measured(telemetry: Path, tokens: dict[str, int]) -> dict:
    rows = [json.loads(x) for x in telemetry.read_text().splitlines()]
    rows = [r for r in rows if r.get("event") == "expansion"]
    keys = ("canonical_reload_seconds", "identify_seconds", "materialize_seconds",
            "state_evaluation_seconds", "validation_seconds")
    per_impl: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for r in rows:
        per_impl[r["impl_id"]].append((tokens[r["profile"]], r["operator_seconds"] / 60))
    overhead = max(sum((r.get(k) or {}).get("seconds", 0.0) for k in keys)
                   + r.get("parent_load_seconds", 0.0) for r in rows) / 60
    return {"per_impl": {k: sorted(v) for k, v in per_impl.items()}, "overhead": overhead,
            "expansions": len(rows)}


def root_minutes_at(impl: str, t: int, m: dict, d2_root: dict) -> tuple[float, str]:
    pts = m["per_impl"].get(impl)
    if impl.startswith("depth."):
        rate = max(mins / tok for tok, mins in pts)
        return rate * t, "measured per-token rate"
    if pts:
        near = min((tok for tok, _ in pts), key=lambda x: abs(x - t))
        return (max(mins for tok, mins in pts if abs(tok - near) <= 0.15 * near),
                "measured at this dose (max over its resamples)")
    ratio = max(max(v for _, v in pts2) / d2_root[k]
                for k, pts2 in m["per_impl"].items() if not k.startswith("depth."))
    return d2_root[impl] * ratio, f"D2 root x {ratio:.2f} (not measured here)"


def session_cost(minutes: float, price: float, *, setup: float = 20.0) -> dict:
    """The launcher's own budget arithmetic for one session of `minutes` work."""
    from stages.calibration_volume import launch
    b = launch.search_budget({"treatments": [{"treatment": "w", "bound_minutes": minutes}],
                              "contingency_fraction": 0.2, "reserve_minutes": 30})
    plan = b.plan(price_per_hour=price, authorized_usd=1e6)
    return {"work_minutes": round(minutes, 1), "expected_usd": round(plan.expected_usd, 2),
            "hard_usd": round(plan.hard_terminate_usd, 2)}


def split(minutes: float, price: float, cap: float) -> list[float]:
    """Fewest equal parts whose sessions each fit the per-session hard cap."""
    for n in range(1, 8):
        if session_cost(minutes / n, price)["hard_usd"] <= cap:
            return [minutes / n] * n
    raise SystemExit("cannot split under the per-session cap")


def sequence(sessions: list[dict], envelope: float) -> dict:
    """Launch in order; each launch needs its HARD cap inside what is left after
    the expected spend of everything before it (P12.1). The first that does not
    fit is where the money runs out."""
    left, rows = envelope, []
    for s in sessions:
        fits = s["hard_usd"] <= left + 1e-9
        rows.append({"session": s["session"], "hard_usd": s["hard_usd"],
                     "left_before_usd": round(left, 2), "fits": fits})
        if not fits:
            return {"envelope_usd": envelope, "completes": False,
                    "first_unfundable": s["session"], "steps": rows}
        left -= s["expected_usd"]
    return {"envelope_usd": envelope, "completes": True, "left_after_usd": round(left, 2),
            "steps": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--telemetry", required=True)
    ap.add_argument("--telemetry-gpu", required=True)
    ap.add_argument("--price-per-hour", type=float, required=True)
    ap.add_argument("--gpu", required=True)
    args = ap.parse_args(argv)
    price = args.price_per_hour

    prof_doc = json.loads((CONF / "profiles.json").read_text())
    tokens = {f"{p['profile_id']}@v{p['version']}": p["metadata"]["tokens"] for p in prof_doc["profiles"]}
    by_dose = defaultdict(list)
    for p in prof_doc["profiles"]:
        by_dose[p["metadata"]["dose"]].append(p)
    m = measured(Path(args.telemetry), tokens)
    d2 = json.loads(D2_ANALYSIS.read_text())["operator_timing"]["per_impl"]
    d2_root = d2_root_minutes()
    n_exp = sum(v["expansions"] for v in d2.values())

    probes = json.loads(PROBE_RECORD.read_text())["probes"]
    train_min = max(p["train_minutes"] for p in probes)
    recovery_eval_min = train_min + sum(EVAL_MINUTES.values())
    cap = G.grant()["per_session_cap_usd"]

    searches = {}
    for dose, profs in sorted(by_dose.items()):
        t = max(p["metadata"]["tokens"] for p in profs)
        op_min, per_impl = 0.0, {}
        for impl, v in d2.items():
            root, basis = root_minutes_at(impl, t, m, d2_root)
            scale = root / d2_root[impl]
            per_impl[impl] = {"minutes": round(v["operator_minutes_total"] * scale, 1),
                              "scale_vs_d2": round(scale, 3), "basis": basis}
            op_min += v["operator_minutes_total"] * scale
        searches[f"{dose}x"] = {
            "profile_tokens_max": t, "per_impl": per_impl, "expansions": n_exp,
            "search_minutes": round(op_min + n_exp * m["overhead"], 1),
            "search_minutes_with_overhead_fix": round(op_min + n_exp * D2_OVERHEAD_MIN, 1),
            "composition_at_this_dose_in_the_pilot_draws": [
                {k: p["metadata"][k] for k in ("uesp", "tokens", "n_items", "n_units",
                                               "n_rule_families")} for p in profs]}

    def units(minutes_key: str) -> list[dict]:
        """The launch order. The 1x unit carries its own recovery and evaluation
        (the resolution gate reads it before anything else is spent); every
        later search is its own session, split where its hard cap exceeds the
        per-session cap; the remaining recoveries share one closing session.
        All sessions work on one network volume, which carries the journal,
        the Top-1 initializations and the recovered students between them."""
        out = []
        w1 = searches["1x"][minutes_key] + recovery_eval_min
        if session_cost(w1, price)["hard_usd"] <= cap:
            out.append({"session": "1x: search + recovery + eval-set-1 (gate)",
                        **session_cost(w1, price)})
            later = ("2x", "4x")
        else:
            later = ("1x", "2x", "4x")
        for dose in later:
            parts = split(searches[dose][minutes_key], price, cap)
            for i, w in enumerate(parts, 1):
                tag = f" part {i}/{len(parts)} (one search identity, journal on the volume)" \
                    if len(parts) > 1 else ""
                out.append({"session": f"{dose}: search{tag}", **session_cost(w, price)})
        rec = [d for d in ("1x", "2x", "4x") if d in later]
        out.append({"session": f"{'+'.join(rec)}: recovery + eval-set-1",
                    **session_cost(len(rec) * recovery_eval_min, price)})
        return out

    volume_days = 3
    volume_usd = round(VOLUME_GB * VOLUME_USD_PER_GB_MONTH * volume_days / 30, 2)
    variants = {}
    for name, key in (("as_measured", "search_minutes"),
                      ("with_overhead_fix", "search_minutes_with_overhead_fix")):
        ss = units(key)
        variants[name] = {"sessions": ss,
                          "network_volume_usd": volume_usd,
                          "expected_usd": round(sum(s["expected_usd"] for s in ss) + volume_usd, 2),
                          "largest_session_hard_usd": max(s["hard_usd"] for s in ss),
                          "sum_of_hard_caps_usd": round(sum(s["hard_usd"] for s in ss), 2)}

    g = G.grant()
    closeouts = [json.loads(o.read_text()) for o in (LOGS / "runs").glob("*/closeout/outcome.json")]
    spent_by_phase = defaultdict(float)
    for c in closeouts:
        spent_by_phase[c.get("phase")] += (c.get("money") or {}).get("all_in_usd") or 0.0
    phase_left = {p: round(v["ceiling_usd"] - spent_by_phase[p], 4) for p, v in g["phases"].items()}
    programme_left = round(g["programme_ceiling_usd"] - sum(spent_by_phase.values()), 4)
    envelopes = {"formal_phase_usd": phase_left["formal_volume"],
                 "all_programme_phases_usd": programme_left}
    for v in variants.values():
        v["sequenced"] = {k: sequence(v["sessions"], e) for k, e in envelopes.items()}
        v["gap_to_formal_phase_expected_usd"] = round(max(0.0, v["expected_usd"]
                                                          - envelopes["formal_phase_usd"]), 2)
        v["gap_to_programme_expected_usd"] = round(max(0.0, v["expected_usd"]
                                                       - envelopes["all_programme_phases_usd"]), 2)

    resolution = LOGS / "design/resolution_check.json"
    doc = {
        "schema": "aadistill.stage0.calibration_volume.formal_design_proposal/v2",
        "status": ("PROPOSAL -- NOT FROZEN AND NOT A PREREGISTRATION. Amended by the maintainer "
                   "2026-10-18 (single draw per dose). The formal phase opens only on "
                   "plans/volume_preregistration.json; writing it needs the funding decision "
                   "below."),
        "supersedes": ("formal_design_proposal v1 (three independent resamples per dose), "
                       "withdrawn by the maintainer's amendment; its record is in git history"),
        "fixed": {"method": "the forward baseline, resolved through shared.forward_baseline "
                            "(stage1_init.d1_target_aware.scored_v1@v1) -- never edited to save cost",
                  "teacher_and_geometry": "as in configs/stages/stage-0/calibration_volume/pilot_search_session.json",
                  "recovery_recipe": "e1_p1_kd_heavy@0.86M (the D-series fixed recovery)",
                  "recovery_seed": "one, identical for all three doses",
                  "state_evaluation": "state_eval@v1",
                  "evaluation_contract": ("data/eval-set/eval-set-1 at its registered identity, every "
                                          "unit and capability reported separately (AGENTS.md P10.3)")},
        "treatments": {
            "doses": {f"{d}x": {"target_uesp": 45000 * d} for d in sorted(by_dose)},
            "draws": ("ONE nested draw: the 2x set contains the 1x set and the 4x set the 2x "
                      "set, from one frozen base mixture (the pilot's weights: 0.18 per static "
                      "primary axis, 0.28 supporting) and its frozen reasoning-heavy variant; "
                      "no independent resamples"),
            "per_treatment": ("one D1 full search (beam width 6, delayed_prune warm-up 1, pareto@v2) "
                              "with operator order and the profile per operator FREE over the two "
                              "mixtures at that dose; the committed Top-1 recovered once under the "
                              "fixed recipe and seed; eval-set-1 measured once"),
            "covariates_recorded_per_dose": ["realized UESP", "tokens", "items and units",
                                             "sources and subtypes", "distinct rule families",
                                             "distinct generators"],
            "covariate_rule": ("rule-family diversity rises with dose in the pilot draws (10-12, 15-17, "
                               "18-22): any H1/H2 difference is reported as a volume+diversity "
                               "contrast, never as a pure volume effect"),
            "hardware": f"{args.gpu}: 8,192-token pool items OOM DEPTH on a 44 GiB card",
        },
        "claims": {
            "allowed": "comparisons between these three specific treatments",
            "not_allowed": ["selection variance or stability (needs resamples; the pilot's "
                            "root-level resampled measurement remains the only one)",
                            "statistical saturation (H3 is described, never tested)",
                            "a pure volume effect where composition changed with dose"],
        },
        "endpoints": {
            "H1": ("descriptive: the committed Top-1 of each dose -- operator order, per-step profile, "
                   "and per operator FFN/ATTENTION per-layer Jaccard, DEPTH removed-set Jaccard, WIDTH "
                   "subspace overlap between doses (2x vs 1x, 4x vs 2x, 4x vs 1x), each beside chance; "
                   "no stability claim"),
            "H2": ("eval-set-1, every unit and capability separately; primary = fluid_reasoning_720 "
                   "static items pooled over the four static axes and per axis, interactive "
                   "separately; supporting and guard-rail units reported beside it; stage3_legacy_150 "
                   "only as in-sample regression"),
            "H3": "the 1x/2x/4x shape of H1 and H2, described",
        },
        "statistics": {
            "sesoi": SESOI,
            "inference": ("task-paired contrasts 2x-1x, 4x-2x, 4x-1x with a rule-family cluster "
                          "bootstrap (10,000 resamples, stratified by axis); the interval covers "
                          "EVALUATION sampling only -- training-seed and selection variance are "
                          "unmeasured with one draw and one seed, and every verdict says so"),
            "decision": {
                "GO": "a contrast's interval lies wholly above +SESOI",
                "NO-GO": "the interval lies wholly inside (-SESOI, +SESOI)",
                "INCONCLUSIVE": "neither",
                "EVALUATION NOT RESOLVABLE": "the resolution gate below fails",
                "SATURATION": ("never declared; a 2x-1x GO with a 4x-2x NO-GO is reported as "
                               "'consistent with saturation for these treatments' only"),
            },
        },
        "resolution_gate": {
            "before_any_search": ("the reference check (configs/.../resolution_session.json): does "
                                  "fluid_reasoning_720 resolve models in the student's capacity range? "
                                  f"result: {rel(resolution) if resolution.is_file() else 'PENDING'}"),
            "after_the_1x_unit": (f"the 1x recovered D1-initialized student must answer at least "
                                  f"{FLOOR_STATIC_RATE:.0%} of fluid_reasoning_720's static items; "
                                  "below it H2 is EVALUATION NOT RESOLVABLE (measured on a D1 student, "
                                  "not assumed from PCA students) and the 2x/4x sessions are not "
                                  "launched without a new maintainer decision"),
            "why": ("every fixed-recipe student measured so far is PCA-initialized and inside the floor "
                    "(0/720, 0/720, 10/720); whether a D1-initialized one is has not been measured"),
        },
        "split_execution_of_4x": {
            "why": "its hard cap exceeds the $30 per-session cap",
            "mechanism": [
                "one network volume (the provider's persistent disk, ONE datacenter) holds the "
                "search workdir: the state journal and every materialized state",
                "part 1 runs the search with the treatment's plan; the watchdog's hard stop or a "
                "planned stop ends it; nothing is collected but the journal tail and telemetry",
                "part 2 attaches the same volume (same datacenter), runs the IDENTICAL plan, "
                "config, seed, profiles and baseline resolution; the engine restores every "
                "journaled state by content-derived id and hash-checks it from disk, and "
                "continues the first unfinished expansion",
                "identity check: part 2's search config hash, materialization protocol and "
                "measurement protocol must equal part 1's; any declined restore "
                "(telemetry `restore_declined`) is a stop condition, not a silent re-measure",
                "the final record carries both sessions' run ids, n_resumed, and the volume id",
            ],
            "engineering_not_yet_built": [
                "search_driver: --workdir on the volume and a resume mode that asserts the "
                "identity check above (the engine already resumes)",
                "launch: attach --network-volume-id / --volume-mount-path (SessionRunner "
                "already supports both) and a 4x two-part plan",
                "a $0 CPU test: stop a tiny-model search mid-tree, resume, and compare the final "
                "journal with an uninterrupted run",
                "creating the volume is a paid provider action inside the formal budget",
            ],
            "volume_usd_estimate": volume_usd,
        },
        "durable_artifacts": ("each dose's committed Top-1 init and its recovered student (~2.4 GB "
                              "each) must survive a later failure (P8.2.1): the network volume is "
                              "the durable store during the study; the HF relay's private quota "
                              "is full, so archival retention after the study is a maintainer "
                              "decision"),
        "pricing": {
            "gpu": args.gpu, "price_per_hour": price,
            "telemetry": {"path": rel(Path(args.telemetry)), "sha256": sha(Path(args.telemetry)),
                          "measured_on": args.telemetry_gpu, "expansions": m["expansions"],
                          "overhead_minutes": round(m["overhead"], 3)},
            "trajectory": {"path": rel(D2_ANALYSIS), "sha256": sha(D2_ANALYSIS)},
            "recovery_minutes": {"train": train_min, "source": rel(PROBE_RECORD),
                                 "eval_set_1": EVAL_MINUTES},
            "full_search_by_dose": searches,
        },
        "design": {"name": "F_single_draw", **variants},
        "budget": {
            "phase_remaining_usd": phase_left,
            "programme_remaining_usd": programme_left,
            "per_session_cap_usd": cap,
            "reading": ("the formal phase alone cannot fund the study in either variant; see "
                        "design.<variant>.sequenced for where the money runs out when every "
                        "launch must fit its hard cap"),
        },
        "authorizes": "nothing",
    }
    out = LOGS / "plans/formal_design_proposal.json"
    out.write_text(json.dumps(doc, indent=1) + "\n")
    for k, v in variants.items():
        print(k, v["expected_usd"], "hard sum", v["sum_of_hard_caps_usd"],
              {e: (s["completes"], s.get("first_unfundable")) for e, s in v["sequenced"].items()})
    print({k: (v["search_minutes"], v["search_minutes_with_overhead_fix"]) for k, v in searches.items()})
    print("envelopes", envelopes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
