#!/usr/bin/env python3
"""A-MIX-6cat-sr, seeds 20260726 and 20260801: the two units' configs bound to the mixed pack and the packed search's
Top-1, their checkpoint cadence, both evaluation modes and their cost. $0.

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python -m stages.teacher_modes.mixed_training_plan --write

writes ``plans/mixed_training_plan.json`` (the units, which kind ``train_sr_mix`` binds and `a1_units.py` runs) and
``plans/mixed_training_budget.json``. Nothing here trains, launches or authorizes anything.

Maintainer 2026-11-09 (UTC), "Research A 新增 Hybrid Mixed Stage 3 与 Checkpoint 精简", sections I-VI:

* **Data**: the mixed pack (`mixed_packs.py`), bound through its freeze record by the formal consumer's check
  (`packed_training_plan.bind_pack`: the admitted manifest, documents, spec and every sequence re-derived).
* **Initialization**: the packed search's Top-1 (``1f5e4b42...``), fresh -- the same one A1-6cat-sr and A2-6cat-sr
  started from, read on the search volume. Never an A1 or A2 checkpoint.
* **Recipe**: A1/A2's, unchanged (`packed_training_plan.unit_config`): CE 0.25 + KD 1.0 at T 1 on the assistant
  positions with Top-200 + tail, 3 epochs, batch size 2, micro-batch 1, AdamW 5e-5 with cosine to 0.1, cross-context
  attention. The optimizer steps and the checkpoint cadence follow from THIS pack (T = ceil(3 x packed sequences / 2)).
* **Lineage**: one hybrid teacher in two modes, declared as ``modes`` with one source per mode
  (`aadistill.training.lineage`), so every pack, resume and stamp is checked as before.
* **Evaluation**: the same checkpoint twice on the frozen eval-set-1 v2 with nucleus_v1 -- thinking
  (``enable_thinking=True``) and non-thinking (``enable_thinking=False``) -- each in its own directory, every
  record keeping its think state (`EVALUATIONS`).

Cost. Seconds per optimizer step from the six Stage-3 -sr units' own L40S steps -- this recipe exactly (the 596M
student, the hybrid or Thinking-2507 4B teacher on the fly, Top-200 + tail KD, cross-context 8,192-position
sequences): `STEP_FIT`, refit by `fit_logs` from their archived train logs. Evaluation seconds per mode from A1's
(thinking) and A2's (non-thinking) measured evaluations; the upper case of either is twice the longest thinking
evaluation, because the mixed student's non-thinking generation length is unmeasured (a deadline, not a forecast).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.teacher_modes import mixed_packs as MP  # noqa: E402
from stages.teacher_modes import packed_training_plan as PTP  # noqa: E402

ARM = MP.ARM
KIND = "train_sr_mix"
SEEDS = PTP.SEEDS
PLAN_SCHEMA = "aadistill.stage2.teacher_modes.mixed_training_plan/v1"
BUDGET_SCHEMA = "aadistill.stage2.teacher_modes.mixed_training_budget/v1"
BOUND = REPO / "logs/stages/stage-2/teacher_modes/plans/mixed_training_plan.json"
BUDGET = REPO / "logs/stages/stage-2/teacher_modes/plans/mixed_training_budget.json"
#: one checkpoint evaluated in both of the teacher's modes; ``dir`` under the unit's durable directory
EVALUATIONS = ({"mode": "thinking", "template": {"enable_thinking": True}, "think_state": "self_opening",
                "dir": "evaluation_thinking", "described_against": "A1-6cat-sr"},
               {"mode": "non_thinking", "template": {"enable_thinking": False}, "think_state": "preclosed",
                "dir": "evaluation_non_thinking", "described_against": "A2-6cat-sr"})
#: the six -sr units' train logs, as their evidence archives hold them on the dev box (C1 attempt 2's extraction
#: carries every unit's log, the earlier sessions' included)
STEP_LOGS = tuple(f"/home/ecs-user/aad-artifacts/stage3_sr/a_train_sr_c1_20261108_104542/extracted/{u}/train/"
                  "train_log.jsonl" for u in ("A1-6cat-sr-s20260726", "A1-6cat-sr-s20260801", "A2-6cat-sr-s20260726",
                                              "A2-6cat-sr-s20260801", "C1-6cat-sr-s20260726", "C1-6cat-sr-s20260801"))
#: seconds per optimizer step = a + b x KD positions per step, least squares over the six -sr units' logged steps
#: after step 20 (`fit_logs` reproduces it)
STEP_FIT = {"a": 2.8908, "b": 9.0996e-05, "residual_sd": 0.031, "steps": 2308,
            "source": "the six Stage-3 -sr units' train_log.jsonl (STEP_LOGS), train_step rows after step 20"}
#: the fit is this recipe on this card, so expected is the fit itself; upper covers pod-to-pod variation
STEP_FACTORS = {"lower": 0.97, "expected": 1.0, "upper": 1.15}
#: measured formal evaluations (a1_units.json :: eval_seconds) per mode, seconds
EVAL_MEASURED = {"thinking": {"A1-6cat-sr-s20260726": 4105.6, "A1-6cat-sr-s20260801": 3628.3},
                 "non_thinking": {"A2-6cat-sr-s20260726": 1080.7, "A2-6cat-sr-s20260801": 961.7}}
EVAL_UPPER_S = 2 * max(EVAL_MEASURED["thinking"].values())
#: the dev-box backup the session's polls carry (token_matched_phase.backup_final_weights): it overlaps training,
#: so it costs pod minutes only when the closeout has to finish it -- at most the whole set at the measured
#: pod -> dev box 11.5 MB/s
BACKUP_MIN = {"lower": 0.0, "expected": 0.0, "upper": 30.0}
RECOVERIES = 2


def fit_logs(paths=STEP_LOGS, after: int = 20) -> dict:
    """`STEP_FIT` again: least squares of ``seconds`` on ``kd_positions`` over every log's train_step rows."""
    xs, ys = [], []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if '"train_step"' in line:
                r = json.loads(line)
                if r["step"] > after:
                    xs.append(r["kd_positions"])
                    ys.append(r["seconds"])
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    b = sum((p - mx) * (q - my) for p, q in zip(xs, ys)) / sum((p - mx) ** 2 for p in xs)
    a = my - b * mx
    return {"a": round(a, 4), "b": float(f"{b:.4e}"),
            "residual_sd": round(statistics.pstdev([q - a - b * p for p, q in zip(xs, ys)]), 3), "steps": len(xs)}


def seconds_per_step(kd_per_step: float, case: str) -> float:
    return round((STEP_FIT["a"] + STEP_FIT["b"] * kd_per_step) * STEP_FACTORS[case], 3)


def eval_seconds(mode: str, case: str) -> float:
    m = list(EVAL_MEASURED[mode].values())
    return {"lower": min(m), "expected": round(sum(m) / len(m), 1), "upper": EVAL_UPPER_S}[case]


def units(pack_dir: Path | None = None) -> list[dict]:
    fr = json.loads(MP.FREEZE.read_text())
    data = PTP.bind_pack(ARM, Path(pack_dir or MP.OUT / ARM), fr)
    init = PTP.initializations()["A1-6cat-sr"]           #: the packed search's Top-1, A1's and A2's own start
    out = []
    for seed in SEEDS:
        cfg = PTP.unit_config(ARM, seed, data, init)
        cfg["checkpoint"]["save_every"] = max(1, round(PTP.SAVE_TARGET_S / seconds_per_step(PTP.kd_per_step(data),
                                                                                          "expected")))
        out.append({"unit": f"{ARM}-s{seed}", "arm": ARM, "seed": seed, "kind": KIND, "config": cfg,
                    "effective_schedule": PTP.effective_schedule(cfg, data),
                    "evaluations": [dict(e) for e in EVALUATIONS],
                    "data": {k: data[k] for k in ("files_sha256", "packed_content_sha256", "data_manifest_sha256",
                                                  "data_manifest_content_sha256", "documents_file", "before_packing",
                                                  "packed_sequences", "validation_sequences", "packing_spec",
                                                  "context_strategy", "sources")},
                    "initialization": dict(init)})
    return out


def problems(us: list[dict]) -> dict:
    """`packed_training_plan.config_problems` (the trainer's validation, the lineage, nothing unset, the attention),
    and each unit's two evaluations: both modes, each its own directory and template."""
    res = PTP.config_problems(us)
    for u in us:
        ev = u["evaluations"]
        if sorted(e["mode"] for e in ev) != sorted(u["config"]["lineage"]["modes"]):
            res[u["unit"]]["problems"].append("the unit is not evaluated in every mode it trains")
        if len({e["dir"] for e in ev}) != len(ev) or any(e["dir"] == "evaluation" for e in ev):
            res[u["unit"]]["problems"].append("each evaluation needs its own directory")
    return res


def unit_seconds(u: dict, case: str) -> dict:
    es, cfg = u["effective_schedule"], u["config"]
    s = seconds_per_step(es["kd_positions_per_step"], case)
    n_val = len(range(0, es["optimizer_steps"] + 1, cfg["intervals"]["eval_every"])) + 1
    train = (es["optimizer_steps"] * s + n_val * es["validation"]["sequences"] * PTP.VAL_S[case]
             + len(es["save_points"]) * PTP.SAVE_S[case] + PTP.LOAD_S[case])
    ev = {e["mode"]: eval_seconds(e["mode"], case) for e in u["evaluations"]}
    return {"seconds_per_step": s, "train_s": round(train), "evaluation_s": ev,
            "unit_s": round(train + sum(ev.values()))}


def budget(us: list[dict]) -> dict:
    disk_h = PTP.RATE["disk_gb"] * 0.10 / (24 * 30.4375)
    per_unit = {u["unit"]: {c: unit_seconds(u, c) for c in ("lower", "expected", "upper")} for u in us}
    rows = {}
    for c in ("lower", "expected", "upper"):
        h = (sum(per_unit[u["unit"]][c]["unit_s"] for u in us) / 3600 + (PTP.SESSION_MIN[c] + BACKUP_MIN[c]) / 60)
        price = PTP.RATE["gpu_cap"] if c == "upper" else PTP.RATE["gpu_expected"]
        rows[c] = {"gpu_hours": round(h, 2), "usd": round(h * (price + disk_h), 2)}
    upper_train_h = max(per_unit[u["unit"]]["upper"]["train_s"] for u in us) / 3600
    driver_min = sum(per_unit[u["unit"]]["upper"]["unit_s"] for u in us) / 60 + BACKUP_MIN["upper"]
    cap = PTP.runner_cap(driver_min)
    es = us[0]["effective_schedule"]
    lost_min = es["save_points"][0] * seconds_per_step(es["kd_positions_per_step"], "upper") / 60
    rec_min = ((PTP.RUNNER["setup"] + PTP.RUNNER["transfer"]) * (1 + PTP.RUNNER["contingency"]) + PTP.RUNNER["closeout"]
               + PTP.RUNNER["recovery"] + PTP.RUNNER["termination"] + lost_min + PTP.LOAD_S["upper"] / 60)
    rec_usd = math.ceil(rec_min / 60 * (PTP.RATE["gpu_cap"] + disk_h) * 100) / 100
    arm = {"kind": KIND, "seeds": list(SEEDS), "session": rows,
           "unit_deadline_hours": math.ceil(upper_train_h * 4) / 4, "session_hard_cap": cap,
           "recovery": {"per_resumed_attempt_minutes": round(rec_min, 1), "usd_each": rec_usd,
                        "attempts_reserved": RECOVERIES,
                        "rule": "a resumed attempt pays the runner's fixed minutes, the training since the last save "
                                "point at the upper step time, and a reload; it never retrains a finished unit or "
                                "re-evaluates a scored mode"},
           "kind_cumulative_cap_usd": round(cap["hard_cap_usd"] + RECOVERIES * rec_usd, 2)}
    return {"per_unit": per_unit, "arms": {ARM: arm},
            "evaluation": "each unit's two formal evaluations (thinking, then non-thinking) run on the same pod right "
                          "after its training; their seconds are inside the unit's row",
            "backup": "the six closed -sr units' final weights and the mixed units' own reach the dev box on the "
                      "session's polls while it trains (token_matched_phase.backup_final_weights); the closeout "
                      f"finishes what is left, at most {BACKUP_MIN['upper']:.0f} minutes, priced in the upper case",
            "rates": {"gpu_expected_usd_per_hour": PTP.RATE["gpu_expected"],
                      "gpu_cap_usd_per_hour": PTP.RATE["gpu_cap"], "container_disk_usd_per_hour": round(disk_h, 4)}}


def capacity(us: list[dict]) -> dict:
    """What the session writes to the search volume: two fresh units (a save point twice over while saving, then
    the final save point; each evaluation package hardlinks the weights, `phase_c1.packaging`)."""
    ckpt = PTP.MODEL_BYTES + PTP.OPTIM_BYTES
    need = len(us) * (ckpt + PTP.MODEL_BYTES) + ckpt
    last = json.loads((REPO / "logs/stages/stage-2/teacher_modes/runs/a_train_sr_c1_20261108_104542/evidence/"
                       "C1-6cat-sr_units.json").read_text())["volume"]
    return {"need_bytes": need, "volume_gb": last["volume_gb"],
            "room_at_last_read_bytes": last["room_bytes"], "read_at": "a_train_sr_c1_20261108_104542's start",
            "written_since_bytes_upper": ckpt + 2 * PTP.MODEL_BYTES,
            "rule": "a1_units.volume_room measures the room on the pod (size sold less du -sb) and refuses before "
                    "training when it cannot hold the units; nothing on the volume is deleted for room"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true", help="write the plan and the budget records")
    ap.add_argument("--refit", action="store_true", help="refit STEP_FIT from the archived logs and print it")
    a = ap.parse_args(argv)
    if a.refit:
        print(json.dumps(fit_logs()))
        return 0
    us = units()
    probs = problems(us)
    if any(v["problems"] for v in probs.values()):
        print(json.dumps(probs, indent=1))
        return 2
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    b = budget(us)
    fr = json.loads(MP.FREEZE.read_text())["arms"][ARM]["compute"]
    src = json.loads(MP.SOURCE_FREEZE.read_text())["a1_a2_compute"]
    plan = {"schema": PLAN_SCHEMA, "study_id": "six_category_schema_restored@v1", "written_utc": now,
            "decided_by": "maintainer 2026-11-09 (UTC), 'Research A 新增 Hybrid Mixed Stage 3 与 Checkpoint 精简': "
                          "A-MIX-6cat-sr x seeds 20260726, 20260801, each evaluated in both modes",
            "freeze_record": {"path": str(MP.FREEZE.relative_to(REPO)), "sha256": MP._sha(MP.FREEZE)},
            "seed_order": list(SEEDS), "units_dir": PTP.UNITS_DIR,
            "checkpoint_policy": {"target_seconds_between_save_points": PTP.SAVE_TARGET_S,
                                  "save_every": {ARM: us[0]["config"]["checkpoint"]["save_every"]}, "keep_last": 1,
                                  "contents": ["model/ (fp32 weights, config)",
                                               "trainer_state.pt: AdamW state, step, consumed sequences, torch + CUDA "
                                               "RNG, config sha256, bound identity",
                                               "resume_identity.json: data, initialization, lineage stamp (its modes)"]},
            "units": us, "validation": probs,
            "comparisons": {
                "optimizer_steps": {"A-MIX": us[0]["effective_schedule"]["optimizer_steps"],
                                    "A1": src["optimizer_steps"]["A1-6cat-sr"], "A2": src["optimizer_steps"]["A2-6cat-sr"]},
                "training_flops_ratio": fr["training_flops_ratio"],
                "supervised_token_exposure": {"A-MIX": fr["supervised_token_exposure"],
                                              "A1": src["supervised_token_exposure"]["A1-6cat-sr"],
                                              "A2": src["supervised_token_exposure"]["A2-6cat-sr"]},
                "unique_supervised_tokens": {"A-MIX": us[0]["data"]["before_packing"]["training"]["unique_supervised_tokens"],
                                             **{k.split("-")[0]: v for k, v in src["unique_supervised_tokens"].items()}},
                "reading": "Mixed-thinking vs A1 and Mixed-non-thinking vs A2 are DESCRIPTIVE paired comparisons on "
                           "the same items: the mixed units hold about twice the unique data and A1 + A2's compute, so "
                           "a difference is not a pure mode-mixing effect. The registered A1 - A2 and A1 - C1 verdicts "
                           "are unchanged. The two modes of one mixed checkpoint are paired with identical weights."},
            "authorizes": "nothing"}
    budget_doc = {"schema": BUDGET_SCHEMA, "study_id": "six_category_schema_restored@v1", "written_utc": now,
                  "is_an_estimate": True, "step_model": {"fit": STEP_FIT, "factors": STEP_FACTORS},
                  "evaluation_model": {"measured_seconds": EVAL_MEASURED, "upper_seconds": EVAL_UPPER_S},
                  **b, "capacity": capacity(us), "authorizes": "nothing"}
    if a.write:
        BOUND.write_text(json.dumps(plan, indent=1, ensure_ascii=False) + "\n")
        BUDGET.write_text(json.dumps(budget_doc, indent=1, ensure_ascii=False) + "\n")
    arm = b["arms"][ARM]
    print(json.dumps({"save_every": plan["checkpoint_policy"]["save_every"], "steps": us[0]["effective_schedule"]["optimizer_steps"],
                      "per_unit": {k: {c: v[c]["unit_s"] for c in v} for k, v in b["per_unit"].items()},
                      "session": arm["session"], "cap": arm["session_hard_cap"], "kind_cap": arm["kind_cumulative_cap_usd"],
                      "deadline_h": arm["unit_deadline_hours"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
