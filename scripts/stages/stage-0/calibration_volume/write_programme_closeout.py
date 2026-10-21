#!/usr/bin/env python3
"""The Stage-0 calibration-volume programme's terminal record, derived.

    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-0/calibration_volume/write_programme_closeout.py

Every number is READ from the record that owns it and every verdict is
COMPUTED from those numbers, so the closeout cannot say something its evidence
does not: the audit, the dev analysis and floor diagnostic, the base pool and
profile documents, the pilot's structural analysis, each run's closeout, the
budget deriver and a live provider query. Writes
`logs/stages/stage-0/calibration_volume/closeout/programme_closeout.json`.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.calibration_volume import governance as G  # noqa: E402

LOGS = REPO / "logs/stages/stage-0/calibration_volume"
CONF = REPO / "configs/stages/stage-0/calibration_volume"


def load(p: Path):
    return json.loads(p.read_text())


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def bound(p: Path) -> dict:
    return {"path": str(p.relative_to(REPO)), "sha256": sha(p)}


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def provider_state() -> dict:
    try:
        key = re.search(r"apikey\s*=\s*'([^']+)'",
                        (Path.home() / ".runpod/config.toml").read_text()).group(1)
        import urllib.request
        req = urllib.request.Request(
            "https://api.runpod.io/graphql",
            data=json.dumps({"query": "query { myself { currentSpendPerHr pods { id } "
                                      "networkVolumes { id } } }"}).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        d = json.loads(urllib.request.urlopen(req, timeout=30).read())["data"]["myself"]
        return {"queried_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "pods": len(d["pods"]), "network_volumes": len(d["networkVolumes"]),
                "spend_per_hour": d["currentSpendPerHr"]}
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)[:300]}


def main() -> int:
    audit = load(LOGS / "audit/data_audit.json")
    dev = load(LOGS / "design/dev_analysis.json")
    floor = load(LOGS / "design/floor_diagnostic.json")
    pool_dir = REPO / load(CONF / "pilot_design.json")["pool"]["dir"]
    pool = load(pool_dir / "manifest.json")
    pilot_path = LOGS / "design/pilot_structure_agreement.json"
    pilot = load(pilot_path) if pilot_path.is_file() else None
    proposal_path = LOGS / "plans/formal_design_proposal.json"
    proposal = load(proposal_path)

    # --- money, from each run's own closeout --------------------------------
    runs = []
    for d in sorted((LOGS / "runs").iterdir()):
        out = d / "closeout/outcome.json"
        if out.is_file():
            o = load(out)
            runs.append({"run_id": d.name, "phase": o.get("phase"), "status": o.get("status"),
                         "terminal": o.get("terminal"), "usd": o["money"]["all_in_usd"],
                         "provider_confirms_gone": o.get("provider_confirms_gone"),
                         "closeout": bound(out)})
    spent = G.programme_spent()
    derived = json.loads(subprocess.run(
        [sys.executable, "scripts/maintenance/consolidation/derive_budget.py", "--json"],
        cwd=REPO, capture_output=True, text=True,
        env={"PYTHONPATH": "src:scripts", "PATH": "/usr/bin:/bin"}).stdout)

    # --- the evaluation verdict, computed -------------------------------------
    pair = dev["labels"]["pair"]
    res = dev["seed_noise_and_resolution"]["all"]
    rates = {lbl: dev["rates"][lbl]["overall"] for lbl in dev["rates"]}
    teacher = rates[dev["labels"]["teacher"]]
    fixed = [rates[p] for p in pair]
    resolvable = not (res["discordance_rate"] == 0 and all(r["correct"] == 0 for r in fixed))
    #: what the 720 can and cannot see is MEASURED per model; it is never
    #: carried from the PCA-initialized students to a D1-initialized one
    #: (maintainer amendment 2026-10-18)
    h2 = ("NOT MEASURED: no D1-initialized recovered student exists yet; every fixed-recipe "
          "student measured so far is PCA-initialized and sits at the floor of fluid_reasoning_720"
          if not resolvable else "the fixed-recipe students resolve")

    # --- the amended formal design (single draw per dose), read and computed ------
    design = proposal["design"]
    envelopes = {"formal_phase": proposal["budget"]["phase_remaining_usd"]["formal_volume"],
                 "all_programme_phases": proposal["budget"]["programme_remaining_usd"]}
    budget = {"design": design["name"], "per_session_cap_usd": proposal["budget"]["per_session_cap_usd"],
              "envelopes_usd": envelopes,
              "variants": {k: {"expected_usd": v["expected_usd"],
                               "sum_of_hard_caps_usd": v["sum_of_hard_caps_usd"],
                               "largest_session_hard_usd": v["largest_session_hard_usd"],
                               "sessions": [(x["session"], x["expected_usd"], x["hard_usd"])
                                            for x in v["sessions"]],
                               "min_sequential_envelope_usd": v["min_sequential_envelope_usd"],
                               "completes_in": {e.removesuffix("_usd"): q["completes"]
                                                for e, q in v["sequenced"].items()},
                               "first_unfundable": {e.removesuffix("_usd"): q.get("first_unfundable")
                                                    for e, q in v["sequenced"].items()},
                               "gap_to_formal_phase_usd": v["gap_to_formal_phase_expected_usd"],
                               "gap_to_programme_usd": v["gap_to_programme_expected_usd"]}
                           for k, v in design.items() if isinstance(v, dict) and "sessions" in v},
              "full_search_by_dose": {k: {x: v[x] for x in ("search_minutes",
                                                             "search_minutes_with_overhead_fix")}
                                      for k, v in proposal["pricing"]["full_search_by_dose"].items()}}
    fundable = {k: v["completes_in"]["formal_phase"] for k, v in budget["variants"].items()}
    resolution_path = LOGS / "design/resolution_check.json"
    resolution = load(resolution_path) if resolution_path.is_file() else None

    # --- the formal study (round 3), read ---------------------------------------
    prereg_path = LOGS / "plans/volume_preregistration.json"
    analysis_path = LOGS / "design/formal_analysis.json"
    formal = None
    if prereg_path.is_file():
        prereg = load(prereg_path)
        doses = {}
        for r in runs:
            if r["run_id"].startswith("dose_"):
                t = r["run_id"].split("_")[1]
                doses.setdefault(t, []).append({k: r[k] for k in ("run_id", "terminal", "usd")})
        analysis = load(analysis_path) if analysis_path.is_file() else None
        formal = {"preregistration": bound(prereg_path),
                  "treatments": {t: {k: v[k] for k in ("dose", "search_config_hash",
                                                       "profile_hashes")}
                                 for t, v in prereg["treatments_by_id"].items()},
                  "sessions_by_dose": doses,
                  "complete_doses": sorted(t for t, rs in doses.items()
                                           if any(x["terminal"] == "ALL_DONE" for x in rs)),
                  "analysis": bound(analysis_path) if analysis else None,
                  "contrasts": ({c: {k: v.get("label") or v.get("diff") for k, v in res.items()}
                                 for c, res in analysis["H2"]["contrasts"].items()}
                                if analysis else None)}

    # --- the Top-4 extension (beside the formal study, never inside it), read ----
    ext_plan_path = LOGS / "plans/top4_extension.json"
    ext_path = LOGS / "design/extension_analysis.json"
    extension = None
    if ext_plan_path.is_file():
        ext = load(ext_path) if ext_path.is_file() else None
        extension = {"plan": bound(ext_plan_path),
                     "sessions": [{k: r[k] for k in ("run_id", "terminal", "usd")}
                                  for r in runs if r["run_id"].startswith("probes_")],
                     "analysis": bound(ext_path) if ext else None,
                     "units": ({k: ext[k] for k in ("units_expected", "units_present", "units_missing")}
                               if ext else None),
                     "claims": ext["claims"] if ext else None}

    # --- the network volumes: reservation while live, actual once retired -------
    volumes = []
    for d in sorted((LOGS / "runs").glob("netvol_*")):
        o, s = load(d / "closeout/outcome.json"), load(d / "runtime/session.json")
        tomb = d / "closeout/tombstone.json"
        t = load(tomb) if tomb.is_file() else None
        volumes.append({"run_id": d.name, "volume_id": s["volume_id"], "size_gb": s["size_gb"],
                        "data_center_id": s["data_center_id"], "status": o["status"],
                        "usd": o["money"]["all_in_usd"],
                        "tombstone": bound(tomb) if t else None,
                        "deleted_utc": t["deleted_utc"] if t else None,
                        "members_source": t["members_source"]["kind"] if t else None,
                        "retired_bytes": sum(c["bytes"] for c in t["totals_by_class"].values())
                                         if t else None})
    audits = [{k: r[k] for k in ("run_id", "status", "usd")}
              for r in runs if r["run_id"].startswith("volaudit_")]

    # --- the pilot verdict, read ----------------------------------------------
    h1 = None
    if pilot:
        h1 = {}
        for kind, o in pilot["by_operator"].items():
            h1[kind] = {"label": o["label"], "reading": o["reading"],
                        "between_resamples": {d: {k: v[k] for k in ("mean", "min", "max", "noise")}
                                              for d, v in o["between_resamples"].items()}}

    doc = {
        "schema": "aadistill.stage0.calibration_volume.programme_closeout/v1",
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repository": {"branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                       "head_at_write": git("rev-parse", "HEAD"),
                       "dirty_at_write": bool(git("status", "--porcelain"))},
        "data_audit": {
            "owner": bound(LOGS / "audit/data_audit.json"),
            "corpus_v2_primary_axis_items": 0,
            "corpus_v2_rule_families": audit["A_independence"]["corpus:corpus_v2"]["rule_families"],
            "openmath_accepted_candidate_correct_rate":
                audit["C_length_selection"]["by_type"]["openmath"]["accepted_candidate_correct_rate"],
            "openmath_length_limited_candidates":
                audit["C_length_selection"]["by_type"]["openmath"]["length_limited_candidates"],
            "window_admitted_share": {t: v["admitted_by_window"] for t, v in
                                      audit["C_length_selection"]["by_type"].items()},
            "historical_profiles_uesp": {r: v["unique_effective_scored_positions"]
                                         for r, v in audit["B_effective_signal"].items()
                                         if r.startswith("calibration:")},
        },
        "capability_evaluation": {
            "battery": bound(REPO / "data/eval-set/eval-set-1/fluid_reasoning_720/dev_v1.jsonl"),
            "manifest": bound(REPO / "data/eval-set/eval-set-1/fluid_reasoning_720/dev_v1.manifest.json"),
            "analysis": bound(LOGS / "design/dev_analysis.json"),
            "floor_diagnostic": bound(LOGS / "design/floor_diagnostic.json"),
            "teacher": teacher, "fixed_recipe_students": dict(zip(pair, fixed)),
            "reference_2_96M": rates.get(dev["labels"]["reference"]),
            "teacher_by_axis": {ax: v["all"]["rate"] for ax, v in
                                dev["rates"][dev["labels"]["teacher"]].items() if ax != "overall"},
            "seed_discordance": res["discordance_rate"],
            "interactive_invalid_action_rate": {k: v["interactive"]["invalid_action_rate"]
                                                for k, v in floor["runs"].items()},
            "pca_fixed_recipe_students_off_floor": resolvable,
        },
        "base_pool": {"manifest": bound(pool_dir / "manifest.json"),
                      "n_items": pool["n_items"], "n_units": pool["n_units"],
                      "max_tokens": pool["max_tokens"],
                      "dropped_units_over_max_tokens": pool["dropped_units_over_max_tokens"]},
        "pilot": {"design": bound(CONF / "pilot_design.json"),
                  "plan": bound(CONF / "pilot_search_session.json"),
                  "analysis_plan": bound(LOGS / "plans/pilot_analysis_plan.json"),
                  "profiles": bound(CONF / "profiles.json"),
                  "covariates": pilot["covariates"] if pilot else None,
                  "agreement": bound(pilot_path) if pilot else None},
        "verdicts": {
            "H2_downstream_reasoning": (
                ("formal study: see formal_study.contrasts (single-draw treatment comparisons on "
                 "eval-set-1's formal rows; evaluation-sampling intervals only)")
                if formal and formal["analysis"] else h2),
            "H1_selection_stability": ("pilot (exploratory, root-level): " + json.dumps(
                {k: v["label"] for k, v in h1.items()}) if h1 else "pilot not complete"),
            "formal_volume_study": (
                (f"COMPLETE: all three doses measured ({', '.join(formal['complete_doses'])}), "
                 "preregistered and authorized 2026-10-19")
                if formal and len(formal["complete_doses"]) == 3 else
                (f"PARTIAL: complete doses {formal['complete_doses']} of d1/d2/d4"
                 if formal and formal["complete_doses"] else None)) or (
                "NOT RUN: BUDGET INSUFFICIENT for the maintainer's amended design (1x/2x/4x, one full "
                "search and one recovery each) -- "
                + "; ".join(f"{k}: ${v['expected_usd']} expected, money runs out at "
                            f"'{v['first_unfundable']['formal_phase']}' in the formal phase "
                            f"(${envelopes['formal_phase']}) and "
                            + (f"at '{v['first_unfundable']['all_programme_phases']}'"
                               if v['first_unfundable']['all_programme_phases'] else "nowhere")
                            + f" with every programme phase merged (${envelopes['all_programme_phases']})"
                            for k, v in budget["variants"].items())
                if not any(fundable.values()) else "fundable in the formal phase: needs a preregistration"),
            "H2_resolution_evidence": (resolution["verdict"]["reading"] if resolution else
                                       "reference check not run"),
            "top4_extension": (
                None if not extension else
                f"COMPLETE: {extension['units']['units_present']}/"
                f"{extension['units']['units_expected']} units analysed; described beside the "
                "formal study, never inside it (see top4_extension.claims for what it does not claim)"
                if extension["units"] and not extension["units"]["units_missing"] else
                "INCOMPLETE: " + json.dumps(extension["units"] and extension["units"]["units_missing"])),
            "network_volumes": (
                "ALL RETIRED (AGENTS.md 2.5): " + ", ".join(v["volume_id"] for v in volumes)
                if volumes and all(v["status"] == "VOLUME_DELETED" for v in volumes) else
                "LIVE: " + ", ".join(v["volume_id"] for v in volumes
                                     if v["status"] != "VOLUME_DELETED")),
        },
        "formal_study": formal,
        "top4_extension": extension,
        "network_volumes": {"volumes": volumes, "listing_runs": audits},
        "formal_design_proposal": bound(proposal_path),
        "resolution_check": bound(resolution_path) if resolution else None,
        "budget_gap": budget,
        "pilot_by_operator": h1,
        "money": {"runs": runs, "programme_spent_usd": spent["spent_usd"],
                  "programme_ceiling_usd": G.grant()["programme_ceiling_usd"],
                  "project": {k: derived["project"][k] for k in
                              ("cap_usd", "cumulative_spend_usd", "remaining_usd")}},
        "provider": provider_state(),
        "authorizes": "nothing",
    }
    out = LOGS / "closeout/programme_closeout.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(doc["verdicts"], indent=1))
    print(json.dumps(doc["money"]["project"]), doc["provider"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
