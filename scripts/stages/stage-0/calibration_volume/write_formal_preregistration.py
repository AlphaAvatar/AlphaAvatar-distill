#!/usr/bin/env python3
"""Freeze the formal Stage-0 volume study: preregistration, dose plans and price.

    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-0/calibration_volume/write_formal_preregistration.py \
        --bench <bench record> --price-per-hour 1.79 [--write]

Written once, before the first formal session, from committed inputs only:

* the maintainer's design (amendment 2026-10-18, authorization 2026-10-19:
  1x/2x/4x, one D1 full search and one fixed recovery per dose, eval-set-1,
  no stop rule on a low 1x);
* `formal_design.json` and the profiles it built (one nested draw);
* every search's identity, computed here exactly as the pod computes it;
* the search-cost bench: the CUDA-exact execution options it validated and the
  timings it measured, which re-price the study (`write_formal_proposal.py`'s
  per-operator basis, with DEPTH divided by the measured prefix-reuse speedup
  and the per-expansion overhead replaced by the measured one).

`plans/volume_preregistration.json` opens the grant's `formal_volume` phase.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.calibration_volume import governance as G  # noqa: E402
from stages.calibration_volume import write_formal_proposal as WP  # noqa: E402

CONF = REPO / "configs/stages/stage-0/calibration_volume"
PLANS = REPO / "logs/stages/stage-0/calibration_volume/plans"
DESIGN = CONF / "formal_design.json"
PROFILES = CONF / "profiles_formal.json"
PILOT_TELEMETRY = REPO / ("artifacts/stages/stage-0/calibration_volume/pilot_20261018_134131/"
                          "pilot_root/search/telemetry.jsonl")
STUDY_ID = "formal_v1"
VOLUME = {"id": "ugllmlv19t", "data_center": "EUR-IS-1", "size_gb": 300,
          "mount": "/runpod-volume",
          "record": "logs/stages/stage-0/calibration_volume/runs/netvol_20261019_051206"}
GPU = "NVIDIA A100-SXM4-80GB"
DISK_GB = 400
SAFETY_MARGIN_USD = 5.0
#: eval-set-1, EVERY item, per recovered student at 8,192 context on an A100:
#: fluid 720 measured 13.0-18.8 min; the 996 static capability/legacy prompts
#: bounded by the D1 confirmation battery's 950 (20.1-29.6 min, L40S).
EVAL_MINUTES = {"fluid_reasoning_720": 19.0, "stage3_capability_846+stage3_legacy_150": 30.0}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rel(p: Path) -> str:
    return str(p.relative_to(REPO))


def search_identities() -> dict:
    from stages.calibration_volume import search_session as SS
    out = {}
    doc = json.loads(PROFILES.read_text())
    for t in ("d1", "d2", "d4"):
        ps = SS.treatment_profiles(t, PROFILES)
        rid = f"stage0_calvol_{STUDY_ID}_{t}"
        s = SS.build_session(workdir=Path("/unused"), run_id=rid, profiles=ps, device="cuda",
                             prune_weights=False)
        meta = {p["profile_id"] + "@v1": p["metadata"] for p in doc["profiles"]}
        out[t] = {"dose": doc["treatments"][t]["dose"], "profiles": [p.qualified_id for p in ps],
                  "profile_hashes": {p.qualified_id: p.profile_hash for p in ps},
                  "profile_items_sha256": {
                      p.qualified_id: next(x["items_file_sha256"] for x in doc["profiles"]
                                           if x["profile_id"] + "@v1" == p.qualified_id)
                      for p in ps},
                  "covariates": {p.qualified_id: meta[p.qualified_id] for p in ps},
                  "search_run_id": rid, "search_config_hash": s.config.config_hash,
                  "measurement_protocol_id": s.evaluator.measurement_protocol_id}
    return out


def price(bench: dict, price_per_hour: float) -> dict:
    """The proposal's per-operator basis, re-priced by the bench's measurements."""
    speed = bench["stages"]["C_depth_speed"]
    #: the MEASURED full-round wall-clock ratio, fixed per-expansion costs
    #: (reference sketches, child build) included -- conservative: over the 8
    #: rounds of a real expansion those costs amortize and the scoring loop
    #: alone ran item_seconds_off / item_seconds_on faster
    depth_factor = 1.0 / float(speed["speedup"])
    over = bench["stages"]["D_overhead"]["device_validation"]["phases"]
    overhead_min = sum(v["seconds"] for v in over.values()) / 60 + 0.05   # + parent load
    prof_doc = json.loads(PROFILES.read_text())
    tokens = {f"{p['profile_id']}@v{p['version']}": p["metadata"]["tokens"]
              for p in prof_doc["profiles"]}
    pilot_doc = json.loads((CONF / "profiles.json").read_text())
    pilot_tokens = {f"{p['profile_id']}@v{p['version']}": p["metadata"]["tokens"]
                    for p in pilot_doc["profiles"]}
    m = WP.measured(PILOT_TELEMETRY, pilot_tokens)
    d2 = json.loads(WP.D2_ANALYSIS.read_text())["operator_timing"]["per_impl"]
    d2_root = WP.d2_root_minutes()
    n_exp = sum(v["expansions"] for v in d2.values())
    by_dose = defaultdict(list)
    for p in prof_doc["profiles"]:
        by_dose[p["metadata"]["dose"]].append(p)
    probes = json.loads(WP.PROBE_RECORD.read_text())["probes"]
    train_min = max(p["train_minutes"] for p in probes)
    sessions = []
    for dose, profs in sorted(by_dose.items()):
        t = max(p["metadata"]["tokens"] for p in profs)
        op_min, per_impl = 0.0, {}
        for impl, v in d2.items():
            root, basis = WP.root_minutes_at(impl, t, m, d2_root)
            mins = v["operator_minutes_total"] * root / d2_root[impl]
            if impl.startswith("depth."):
                mins *= depth_factor
                basis += f"; x {depth_factor:.3f} (prefix reuse, bench)"
            per_impl[impl] = {"minutes": round(mins, 1), "basis": basis}
            op_min += mins
        search_min = op_min + n_exp * overhead_min
        work = search_min + train_min + sum(EVAL_MINUTES.values()) + 10.0   # + pushes
        gpu_cost = WP.session_cost(work, price_per_hour)
        #: container disk is billed beside the GPU ($0.10/GB-month, 720 h/month,
        #: the dearer proration), over the same expected and hard windows
        disk_per_hour = DISK_GB * 0.10 / 720
        exp_h = gpu_cost["expected_usd"] / price_per_hour
        hard_h = gpu_cost["hard_usd"] / price_per_hour
        sessions.append({"treatment": f"d{dose}", "profile_tokens_max": t,
                         "per_impl": per_impl, "expansions": n_exp,
                         "search_minutes": round(search_min, 1),
                         "recovery_minutes": train_min, "eval_minutes": dict(EVAL_MINUTES),
                         "work_minutes": gpu_cost["work_minutes"],
                         "gpu_expected_usd": gpu_cost["expected_usd"],
                         "gpu_hard_usd": gpu_cost["hard_usd"],
                         "disk_usd_per_hour": round(disk_per_hour, 4),
                         "expected_usd": round(gpu_cost["expected_usd"] + exp_h * disk_per_hour, 2),
                         "hard_usd": round(gpu_cost["hard_usd"] + hard_h * disk_per_hour, 2)})
    return {"gpu": GPU, "price_per_hour": price_per_hour,
            "depth_speedup_measured": float(speed["speedup"]),
            "overhead_minutes_measured": round(overhead_min, 3),
            "basis": ("write_formal_proposal.py's per-operator minutes (pilot telemetry on the "
                      "same card, D2's 92-expansion trajectory), DEPTH divided by the bench's "
                      "full-round prefix-reuse speedup, overhead replaced by the bench's "
                      "device-validation expansion"),
            "sessions": sessions}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, type=Path)
    ap.add_argument("--price-per-hour", type=float, required=True)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    bench = json.loads(args.bench.read_text())
    b = bench["stages"]
    if not (b["B_depth_exactness"]["tables_bit_identical"]
            and b["B_depth_exactness"]["child_weights_bit_identical"]
            and b["C_depth_speed"]["tables_bit_identical"]):
        raise SystemExit("prefix reuse is not bit-identical on CUDA; it cannot be used")
    if not (b["D_overhead"]["same_state_and_digest"] and b["D_overhead"]["same_evaluation"]):
        raise SystemExit("device-side validation changed a state or a measurement")
    ids = search_identities()
    cost = price(bench, args.price_per_hour)
    grant = G.grant()
    env = G.remaining_envelope("formal_volume")
    left, rows = min(env["phase_remaining_usd"], env["programme_remaining_usd"]), []
    for s in cost["sessions"]:
        rows.append({"treatment": s["treatment"], "hard_usd": s["hard_usd"],
                     "expected_usd": s["expected_usd"], "left_before_usd": round(left, 2),
                     "fits": s["hard_usd"] <= left})
        left -= s["expected_usd"]
    cost["sequence"] = {"available_usd": min(env["phase_remaining_usd"],
                                             env["programme_remaining_usd"]),
                        "steps": rows, "expected_left_after_usd": round(left, 2),
                        "per_session_cap_usd": grant["per_session_cap_usd"],
                        "all_fit": all(r["fits"] for r in rows)
                        and all(s["hard_usd"] <= grant["per_session_cap_usd"]
                                for s in cost["sessions"]),
                        "split_4x": False}
    from stages.calibration_volume.write_formal_proposal import SESOI
    from shared.recipes import E1_KD_HEAVY_0860K as recipe
    from stages.phase_c1.autoinit_c1_driver import FROZEN_RECIPE
    from aadistill.evaluation import eval_registry as ER
    from shared import eval_sets as ES
    es = ER.entry(ES.registry(), "eval-set-1")
    prereg = {
        "schema": "aadistill.stage0.calibration_volume.preregistration/v1",
        "study_id": STUDY_ID,
        "status": ("PREREGISTERED before the first formal session. Frozen: changing a treatment, "
                   "a profile, the method, the recipe, the seed, the evaluation contract or the "
                   "analysis needs a new maintainer decision."),
        "decisions": ["logs/budget/decisions.md 2026-10-18 (Stage-0 volume design amended to one draw per dose; resolution check; formal study still NOT RUN)",
                      "logs/budget/decisions.md 2026-10-19 (Maintainer: eval-set-1 finalization and the Stage-0 volume study authorized ($90 new-spend hard cap))"],
        "question": json.loads(DESIGN.read_text())["question"],
        "design": {"file": rel(DESIGN), "sha256": sha(DESIGN)},
        "profiles_doc": rel(PROFILES), "profiles_doc_sha256": sha(PROFILES),
        "profile_items_dir": json.loads(DESIGN.read_text())["output"]["items_dir"],
        "fixed": {
            "method": "the forward baseline stage1_init.d1_target_aware.scored_v1@v1, resolved by shared.forward_baseline (operators, positions, support, KL references, beam width 6 delayed_prune warm-up 1, pareto@v2, micro-batch 3, length_sorted_v1, scored_v1) -- never edited to save cost",
            "teacher": {"id": "Qwen/Qwen3-4B-Thinking-2507",
                        "revision": "768f209d9ea81521153ed38c47d515654e938aea"},
            "geometry_and_state_evaluation": "the D-series target geometry and state_eval@v1 (stages.phase_a3.a3_session.path_spec, stages.phase_d1.d1_session)",
            "operator_order_and_profile_per_operator": "free search variables over the dose's two profiles (base, rh), as in D1"},
        "execution": {
            "depth_prefix_reuse": True, "validate_on_search_device": True,
            "prune_weights": False,
            "why": ("cost (maintainer 2026-10-19: optimize first, identity unchanged). Both "
                    "options are execution only; the bench showed both bit-identical on this "
                    "card (DEPTH tables, removal order, child bytes; state ids, digests and "
                    "measurements). prune_weights=False keeps every state's bytes on the "
                    "network volume so a resumed session restores instead of recomputing."),
            "validated_by": {"record": rel(args.bench) if args.bench.is_relative_to(REPO) else str(args.bench),
                             "sha256": sha(args.bench)},
            "hardware": GPU, "container_disk_gb": DISK_GB, "network_volume": VOLUME},
        "treatments_by_id": ids,
        "search": {"finalists_k": 1, "commit": "stage1_selection.commit of the ranking's Top-1 complete leaf"},
        "recovery": {"recipe_id": recipe.recipe_id, "config": rel(FROZEN_RECIPE),
                     "config_sha256": sha(FROZEN_RECIPE),
                     "seed": json.loads(FROZEN_RECIPE.read_text())["seed"],
                     "pack_sha256": recipe.pack_sha256,
                     "rule": "the frozen recipe unchanged except run identity, pack path and student_path (the C1 override set); the same seed for every dose"},
        "evaluation": {
            "eval_set_id": "eval-set-1", "eval_set_identity": es["identity_history"][-1]["manifest_sha256"],
            "eval_set_status": es["status"],
            "coverage": "EVERY item of every unit (720 + 846 + 150); masks decide formal rows, not generation",
            "protocols": "each component's scorer.formal_protocol (v2)",
            "action_protocol": "v2", "trained_context": 8192, "decoding": "greedy",
            "report": "scripts/shared/evaluation/eval_set_scoring.py report (per unit and capability; no pooled score)"},
        "analysis": {
            "H1": "descriptive: per dose, the committed Top-1's operator order and per-step profile; per operator FFN/ATTENTION per-layer kept-set Jaccard, DEPTH removed-set Jaccard, WIDTH subspace overlap, between doses (2x vs 1x, 4x vs 2x, 4x vs 1x), each beside chance; no stability claim",
            "H2": "eval-set-1 formal rows, every unit and capability separately: fluid static items over the four static axes and per axis, interactive separately; capability units, behaviour, safety, legacy (in-sample) each separately; knowledge reported as supplementary",
            "H3": "the 1x/2x/4x shape of H1 and H2, described",
            "separation": "structure (H1), the Top-1's step-0 state evaluation against the teacher (KL and the other state_eval@v1 terms, from the search journal), the recovery's own training record, and the recovered student's capabilities (H2) are reported side by side and never combined into one number",
            "statistics": {"sesoi": SESOI,
                           "contrasts": "task-paired 2x-1x, 4x-2x, 4x-1x per unit/axis",
                           "intervals": "rule-family cluster bootstrap (10,000 resamples, stratified by axis) for fluid; item-paired bootstrap for the capability units; intervals cover EVALUATION sampling only",
                           "labels": {"GO": "interval wholly above +SESOI", "NO-GO": "interval wholly inside (-SESOI, +SESOI)",
                                      "INCONCLUSIVE": "neither"},
                           "not_claimed": ["cross-seed or cross-draw stability", "statistical saturation",
                                           "a pure volume effect (rule-family diversity rises with dose)"]},
            "no_stop_rule": "maintainer 2026-10-19: a low 1x does not cancel 2x/4x; all three doses run"},
        "execution_order": ["d1", "d2", "d4"],
        "budget_rule": ("before each launch the session's hard cap must fit what is left of the "
                        "$90 after everything already spent and the volume's reservation, AND "
                        "the expected cost of the remaining doses plus a $%.0f margin must still "
                        "fit; otherwise no new paid session" % SAFETY_MARGIN_USD),
        "retry_policy": ("before formal measurement: ordinary failures repaired and retried (P12.1). "
                         "After it began: a failed session is preserved and the SAME dose resumes "
                         "from the network volume -- journal restore, an existing trained student "
                         "evaluated rather than retrained, finished evaluation units rescored "
                         "rather than regenerated. Never a second complete measurement of a dose."),
        "pricing": cost,
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "authorizes": "nothing by itself: each session runs only on its own one-use authorization"}
    print(json.dumps({"identities": {k: v["search_config_hash"][:16] for k, v in ids.items()},
                      "pricing": {s["treatment"]: (s["search_minutes"], s["expected_usd"], s["hard_usd"])
                                  for s in cost["sessions"]},
                      "sequence": cost["sequence"]}, indent=1))
    if not args.write:
        return 0
    PLANS.mkdir(parents=True, exist_ok=True)
    (PLANS / "volume_preregistration.json").write_text(json.dumps(prereg, indent=1) + "\n")
    for s in cost["sessions"]:
        t = s["treatment"]
        plan = {"schema": "aadistill.stage0.calibration_volume.session_plan/v1",
                "session_kind": "dose", "name": f"dose_{t}", "treatment": t,
                "purpose": f"Formal Stage-0 volume dose {t}: D1 full search, fixed recovery, every item of eval-set-1 (plans/volume_preregistration.json).",
                "teacher_revision": prereg["fixed"]["teacher"]["revision"],
                "teacher": prereg["fixed"]["teacher"],
                "profile_items_dir": prereg["profile_items_dir"],
                "preregistration_sha256": sha(PLANS / "volume_preregistration.json"),
                "setup_minutes": 20,
                "budget_phases": {"search": s["search_minutes"], "recovery": s["recovery_minutes"],
                                  "eval_set_1": sum(s["eval_minutes"].values()), "pushes": 10.0},
                "contingency_fraction": 0.2, "reserve_minutes": 30, "mirror_seconds": 120,
                "hardware": {"gpu": GPU, "container_disk_gb": DISK_GB, "network_volume": VOLUME},
                #: One dose's kept states fill most of the volume, so each later
                #: dose first retires the COMPLETE earlier ones' (maintainer
                #: 2026-10-19). Absent for the first dose: its plan hash is bound.
                **({"retire_states_of": prereg["execution_order"][:prereg["execution_order"].index(t)]}
                   if prereg["execution_order"].index(t) else {}),
                "authorizes": "nothing"}
        (CONF / f"dose_{t}_session.json").write_text(json.dumps(plan, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
