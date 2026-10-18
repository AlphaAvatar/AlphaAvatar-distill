#!/usr/bin/env python3
"""The formal volume study, designed and priced -- a PROPOSAL, never a preregistration.

    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-0/calibration_volume/write_formal_proposal.py \
        --telemetry <a root-level search telemetry.jsonl on these profiles> --price-per-hour 1.79 \
        --gpu "NVIDIA A100-SXM4-80GB"

Writes `logs/stages/stage-0/calibration_volume/plans/formal_design_proposal.json`.
The grant opens the formal phase only on `plans/volume_preregistration.json`;
this file is not that, and it states why it cannot become it in this
programme: its precondition (a fixed-recipe control that the capability
battery can resolve) is computed here from the dev analysis and fails.

Pricing is measured, never assumed:

* a full search's trajectory is the forward baseline's measured one -- D2's
  92 expansions, per implementation (`search_analysis.json`), whose two
  profiles mirror a treatment's two mixtures;
* each implementation's minutes are scaled by the ratio of its ROOT minutes on
  this programme's profiles at the treatment's dose (the telemetry) to its
  root minutes in D2 -- DEPTH per token, the activation operators at the
  measured dose or the nearest measured one;
* per-expansion overhead is the worst measured; session overheads, contingency
  and reserves are `launch.search_budget`'s;
* a recovery probe is the D3 probe session's measured cost per probe.
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
D2_TOKENS = 59830
D3_PROBES = REPO / "logs/stages/stage-1/phase_d3/runs/d3_probes_20261017_110022/closeout/outcome.json"
D3_PROBES_USD = 5.0395      # that run's booked all-in cost (decisions.md 2026-10-17)
SESOI = {"pooled_primary": 0.05, "per_axis": 0.08}
#: Three independent resamples per dose is the minimum the pilot's frozen
#: reading rule can read: it calibrates an effect against the spread of the
#: between-resample pairs at a dose, and two resamples give one pair and no
#: spread -- exactly why the pilot's DEPTH carries no label.
RESAMPLES = 3


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rel(p: Path) -> str:
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)


def d2_root_minutes() -> dict[str, float]:
    """D2's ROOT operator minutes per implementation (max over its two profiles)."""
    depth = {}
    states = D2_TELEMETRY.with_name("states.jsonl")
    for line in states.read_text().splitlines():
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
        # the dose's own resamples sit within a few percent of each other in tokens
        near = min((tok for tok, _ in pts), key=lambda x: abs(x - t))
        return (max(mins for tok, mins in pts if abs(tok - near) <= 0.15 * near),
                "measured at this dose (max over its resamples)")
    measured_ratio = max(max(v for _, v in pts2) / d2_root[k]
                         for k, pts2 in m["per_impl"].items() if not k.startswith("depth."))
    return d2_root[impl] * measured_ratio, f"D2 root x {measured_ratio:.2f} (not measured here)"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--telemetry", required=True)
    ap.add_argument("--telemetry-gpu", required=True)
    ap.add_argument("--price-per-hour", type=float, required=True)
    ap.add_argument("--gpu", required=True)
    args = ap.parse_args(argv)

    from stages.calibration_volume import launch

    prof_doc = json.loads((CONF / "profiles.json").read_text())
    tokens = {f"{p['profile_id']}@v{p['version']}": p["metadata"]["tokens"] for p in prof_doc["profiles"]}
    dose_tokens = defaultdict(list)
    for p in prof_doc["profiles"]:
        dose_tokens[p["metadata"]["dose"]].append(p["metadata"]["tokens"])
    tel = Path(args.telemetry)
    m = measured(tel, tokens)
    d2 = json.loads(D2_ANALYSIS.read_text())["operator_timing"]["per_impl"]
    d2_root = d2_root_minutes()

    searches = {}
    for dose, toks in sorted(dose_tokens.items()):
        t = max(toks)
        impls, minutes = {}, 0.0
        for impl, v in d2.items():
            root, basis = root_minutes_at(impl, t, m, d2_root)
            scale = root / d2_root[impl]
            impls[impl] = {"d2_expansions": v["expansions"], "d2_operator_minutes": v["operator_minutes_total"],
                           "root_minutes_here": round(root, 2), "basis": basis, "scale": round(scale, 3),
                           "minutes": round(v["operator_minutes_total"] * scale, 1)}
            minutes += v["operator_minutes_total"] * scale
        n_exp = sum(v["expansions"] for v in d2.values())
        search_minutes = minutes + n_exp * m["overhead"]
        budget = launch.search_budget({"treatments": [{"treatment": f"d{dose}", "bound_minutes": search_minutes}],
                                       "contingency_fraction": 0.2, "reserve_minutes": 30})
        b = budget.plan(price_per_hour=args.price_per_hour, authorized_usd=1e6)
        searches[f"{dose}x"] = {"profile_tokens_max": t, "per_impl": impls, "expansions": n_exp,
                                "search_minutes": round(search_minutes, 1),
                                "expected_usd": round(b.expected_usd, 2),
                                "hard_usd": round(b.hard_terminate_usd, 2),
                                "within_per_session_cap": b.hard_terminate_usd <= G.grant()["per_session_cap_usd"]}

    probe_usd = D3_PROBES_USD / len(json.loads(D3_PROBES.read_text())["seeds"])
    dev = json.loads((LOGS / "design/dev_analysis.json").read_text())
    pair = dev["labels"]["pair"]
    res = dev["seed_noise_and_resolution"]["all"]
    fixed = {p: dev["rates"][p]["overall"] for p in pair}
    resolvable = not (res["discordance_rate"] == 0 and all(r["correct"] == 0 for r in fixed.values()))

    def design(doses, n_resamples, probes_per_treatment):
        s = sum(searches[f"{d}x"]["expected_usd"] for d in doses) * n_resamples
        h = max(searches[f"{d}x"]["hard_usd"] for d in doses)
        p = len(doses) * n_resamples * probes_per_treatment * probe_usd
        return {"doses": [f"{d}x" for d in doses], "resamples_per_dose": n_resamples,
                "full_searches": len(doses) * n_resamples,
                "recovery_probes": len(doses) * n_resamples * probes_per_treatment,
                "searches_expected_usd": round(s, 2), "probes_expected_usd": round(p, 2),
                "expected_usd": round(s + p, 2), "largest_session_hard_usd": h}

    # the formal phase's own figures; a live session elsewhere does not move them
    g = G.grant()
    formal_spent = sum(json.loads(o.read_text())["money"]["all_in_usd"]
                       for o in (LOGS / "runs").glob("*/closeout/outcome.json")
                       if json.loads(o.read_text()).get("phase") == "formal_volume")
    programme_left = round(g["programme_ceiling_usd"] - sum(
        json.loads(o.read_text())["money"]["all_in_usd"]
        for o in (LOGS / "runs").glob("*/closeout/outcome.json")), 4)
    formal_left = {"phase": "formal_volume",
                   "phase_ceiling_usd": g["phases"]["formal_volume"]["ceiling_usd"],
                   "phase_spent_usd": round(formal_spent, 4),
                   "phase_remaining_usd": round(g["phases"]["formal_volume"]["ceiling_usd"]
                                                - formal_spent, 4),
                   "per_session_cap_usd": g["per_session_cap_usd"],
                   "programme_remaining_usd": programme_left}
    designs = {
        "F_full": {"what": "the brief's study: 1x/2x/4x, three independent resamples each, one D1 "
                           "full search per treatment, the fixed recovery on each Top-1 at two "
                           "recovery seeds, capability battery final split",
                   **design((1, 2, 4), RESAMPLES, 2)},
        "MVP_A_h1_only": {"what": "H1 (and H3 at two doses) only: 1x/2x x three resamples, six "
                                  "full searches, no recovery. Answers whether full-search "
                                  "selection stabilizes with volume; says nothing about capability",
                          **design((1, 2), RESAMPLES, 0)},
        "MVP_B_h1_h2": {"what": "the smallest study that can address H2: 1x/2x x three resamples with "
                                "two recovery seeds per Top-1 -- under a recovery recipe that first "
                                "passes the precondition below, which the fixed 0.86M rung fails",
                        **design((1, 2), RESAMPLES, 2)},
    }
    for d in designs.values():
        d["fits_formal_phase"] = d["expected_usd"] <= formal_left["phase_remaining_usd"]
        d["fits_programme_remainder"] = d["expected_usd"] <= programme_left
        d["fits_per_session_cap"] = d["largest_session_hard_usd"] <= formal_left["per_session_cap_usd"]
        d["gap_to_formal_phase_usd"] = round(max(0.0, d["expected_usd"]
                                                 - formal_left["phase_remaining_usd"]), 2)

    doc = {
        "schema": "aadistill.stage0.calibration_volume.formal_design_proposal/v1",
        "status": ("PROPOSAL -- NOT FROZEN AND NOT A PREREGISTRATION. The formal phase opens only "
                   "on plans/volume_preregistration.json, which this programme does not write: "
                   "the precondition below fails under the fixed recovery recipe."),
        "fixed": {"method": "the forward baseline, resolved through shared.forward_baseline "
                            "(stage1_init.d1_target_aware.scored_v1@v1)",
                  "teacher_and_geometry": "as in configs/stages/stage-0/calibration_volume/pilot_search_session.json",
                  "recovery_recipe": "e1_p1_kd_heavy@0.86M (the D-series fixed recovery)",
                  "state_evaluation": "state_eval@v1"},
        "treatments": {
            "dose_unit": "unique effective scored positions (UESP) under positions.supervised_target_v1",
            "doses": {f"{d}x": {"target_uesp": 45000 * d, "realized_tokens_max": max(t)}
                      for d, t in sorted(dose_tokens.items())},
            "resamples": "independent sampling seeds per dose; doses nested within a resample",
            "per_treatment": ("one D1 full search (beam width 6, delayed_prune warm-up 1, "
                              "pareto@v2) with operator order and the profile per operator FREE, "
                              "over two mixtures at the treatment's dose and resample: base "
                              "(0.72 static primary / 0.28 supporting, the pilot's) and a "
                              "reasoning-heavy variant -- the D1 search's own two-profile shape"),
            "hardware": f"{args.gpu}: the base pool's 8,192-token items put up to 22,141 scored rows "
                        "into one micro-batch of 3; a 44 GiB card OOMs in DEPTH (pilot_20261018_122808)",
        },
        "endpoints": {
            "H1": ("between-resample agreement at each dose of the committed Top-1: its operator "
                   "order and per-step profile, and per operator FFN/ATTENTION per-layer Jaccard, "
                   "DEPTH removed-set Jaccard, WIDTH subspace overlap; each beside chance"),
            "H2": ("correctness of each treatment's recovered Top-1 on the capability battery's "
                   "final split (evaluation families; the five evaluation-only generators "
                   "reported separately), per P10.2 axis and pooled over the four static "
                   "primary axes; interactive reported, not pooled"),
            "H3": "the dose-response shape across 1x/2x/4x for H1 and H2",
        },
        "statistics": {
            "sesoi": SESOI,
            "inference": ("rule-family cluster bootstrap (10,000 resamples, stratified by axis), "
                          "tasks paired across treatments, recovery seeds and resamples averaged "
                          "within treatment; 95% intervals on each dose contrast"),
            "decision": {
                "GO": "the 2x-1x (or 4x-1x) interval lies wholly above +SESOI",
                "NO-GO": "the interval lies wholly inside (-SESOI, +SESOI)",
                "INCONCLUSIVE": "neither",
                "SATURATION SUPPORTED": "2x-1x is GO and 4x-2x is NO-GO",
                "EVALUATION NOT RESOLVABLE": "the precondition fails",
            },
            "h1_reading": "the pilot's noise-calibrated rule (plans/pilot_analysis_plan.json), on full-search Top-1s",
        },
        "precondition": {
            "rule": ("before any formal spend, a control recovered under the fixed recipe must score "
                     "above floor on dev_v1, with a seed-discordance-based minimal detectable effect "
                     "at or below the pooled SESOI"),
            "evidence": {"dev_analysis": rel(LOGS / "design/dev_analysis.json"),
                         "fixed_recipe_controls": fixed,
                         "seed_discordance_rate": res["discordance_rate"]},
            "met": resolvable,
            "why_not": (None if resolvable else
                        "both 0.86M controls score 0/720 and never disagree, so no minimal "
                        "detectable effect exists: every recovered treatment would score zero by "
                        "construction"),
        },
        "pricing": {
            "gpu": args.gpu, "price_per_hour": args.price_per_hour,
            "telemetry": {"path": rel(tel), "sha256": sha(tel), "measured_on": args.telemetry_gpu,
                          "expansions": m["expansions"], "overhead_minutes": round(m["overhead"], 3)},
            "trajectory": {"path": rel(D2_ANALYSIS), "sha256": sha(D2_ANALYSIS)},
            "probe_usd": round(probe_usd, 3),
            "probe_source": f"{rel(D3_PROBES)}: ${D3_PROBES_USD} for its probes",
            "full_search_by_dose": searches,
        },
        "designs": designs,
        "cost_levers": {
            "dominant_term": ("DEPTH: token-linear, and the pool's long teacher-native items make "
                              "it the bulk of every search (see full_search_by_dose.per_impl)"),
            "per_expansion_overhead": ("measured ~1.25 min per expansion here against D2's 0.14; the "
                                       "reload validation runs on the host, where torch sizes its "
                                       "pools from the host's CPU count rather than the pod's cgroup "
                                       "grant (suspected, not measured). An execution knob worth "
                                       "~115 min per full search"),
            "item_length_bound": ("a data-design choice, not an execution knob: D1's [256, 2,048] "
                                  "window keeps 66 of 572 verified teacher units and no analogy or "
                                  "interactive unit (audit section C)"),
            "faster_card": "an H100-class card is unmeasured on this workload; its price is ~2.2x the A100's",
        },
        "formal_phase": formal_left,
        "authorizes": "nothing",
    }
    out = LOGS / "plans/formal_design_proposal.json"
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: {x: v[x] for x in ("expected_usd", "largest_session_hard_usd",
                                            "fits_formal_phase", "fits_per_session_cap")}
                      for k, v in designs.items()}, indent=1))
    print({k: (v["search_minutes"], v["expected_usd"], v["hard_usd"]) for k, v in searches.items()},
          "precondition met:", resolvable)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
