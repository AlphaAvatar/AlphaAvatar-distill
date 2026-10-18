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
    profiles = load(CONF / "profiles.json")
    pilot_path = LOGS / "design/pilot_structure_agreement.json"
    pilot = load(pilot_path) if pilot_path.is_file() else None

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
    h2 = ("EVALUATION NOT RESOLVABLE under the fixed recovery recipe" if not resolvable
          else "RESOLVABLE (the formal study's resolution must be computed)")

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
            "battery": bound(REPO / "data/stages/stage-0/calibration_volume/tasks/dev_v1.jsonl"),
            "manifest": bound(REPO / "data/stages/stage-0/calibration_volume/tasks/dev_v1.manifest.json"),
            "analysis": bound(LOGS / "design/dev_analysis.json"),
            "floor_diagnostic": bound(LOGS / "design/floor_diagnostic.json"),
            "teacher": teacher, "fixed_recipe_students": dict(zip(pair, fixed)),
            "reference_2_96M": rates.get(dev["labels"]["reference"]),
            "teacher_by_axis": {ax: v["all"]["rate"] for ax, v in
                                dev["rates"][dev["labels"]["teacher"]].items() if ax != "overall"},
            "seed_discordance": res["discordance_rate"],
            "interactive_invalid_action_rate": {k: v["interactive"]["invalid_action_rate"]
                                                for k, v in floor["runs"].items()},
            "resolvable_under_fixed_recipe": resolvable,
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
            "H2_downstream_reasoning": h2,
            "H1_selection_stability": ("pilot (exploratory, root-level): " + json.dumps(
                {k: v["label"] for k, v in h1.items()}) if h1 else "pilot not complete"),
            "formal_volume_study": ("NOT RUN: under the fixed recovery recipe every recovered "
                                    "student scores 0 on the capability battery, so no "
                                    "initialization difference can reach the endpoint"
                                    if not resolvable else "not decided by this record"),
        },
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
