#!/usr/bin/env python3
"""Stage 3 of six_category_schema_restored@v1: A1-6cat-sr, A2-6cat-sr and C1-6cat-sr, seeds 20260726 and 20260801 --
the six units' configs bound to their frozen data and initialization, their checkpoint cadence and their cost. $0.

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python -m stages.teacher_modes.packed_training_plan --write

writes ``plans/packed_training_plan.json`` (the units, which the training kinds bind and `a1_units.py` runs) and
``plans/packed_training_budget.json`` (lower / expected / upper per unit, the session caps, the checkpoint and volume
capacity, the funding position). Nothing here trains, launches or authorizes anything.

Binding (maintainer 2026-11-07, Stage 3 preparation, sections I-III). Each arm keeps its own data identity: the pack
the freeze record froze (`audit/six_category_schema_restored_freeze.json`), checked by `packed_packs.verify_pack`
(admitted manifest, documents, packing spec, the packed tokens re-derived) and file by file against the freeze
record, never copied or rebuilt. A1-6cat-sr and A2-6cat-sr start from the packed search's Top-1 on the search volume
(its committed ``search_record.json`` is the provenance record), C1-6cat-sr from Thinking-2507's own D4 Top-1
(fc28e0dd..., D4's search record), each with its own teacher's tokenizer, template, RoPE and online KD teacher. The
recipe is E1's with the registered objective, unchanged: CE 0.25 + KD 1.0 at T 1 on the assistant positions with
Top-200 + tail, 3 epochs, batch size 2, micro-batch 1, AdamW 5e-5 with cosine to 0.1; T = ceil(3 x packed sequences
/ 2), warmup max(10, round(0.05 T)), evaluation every max(25, T // 8) (`schedule`).

Checkpoints. One save point about every hour of the arm's expected step time (``save_every`` = 3,600 s / expected
seconds per step), instead of T // 2 -- an engineering interval, not a training term: saving reads no RNG and changes
no step. A save point holds the student weights, the AdamW state, the step and consumed-sequence position (from which
the trainer derives the epoch and its permutation), the torch and CUDA RNG states, the config sha256 and the bound
identity (data, initialization, lineage stamp); ``keep_last`` 1 keeps the newest save point and the final one, and a
save point is published (``latest.txt``) only after it is complete. The trainer refuses to resume onto other data, an
other initialization, lineage or config (`aadistill.training.train.Trainer.restore`; bitwise-exact resume is a core
test, tests/training/test_topk_kd.py).

Cost. Seconds per step from D4-Augmented's own L40S steps (the same 596M student geometry, a 4B Qwen3 teacher,
cross-context sequential packing over 8,192 positions -- the attention every -sr pack trains with): 3.0134 s +
8.554e-5 s per KD position per step, residual sd 0.043 s over 368 steps (`STEP_FIT`, refit by `fit_steps` from the
archive). Expected x1.05 for the Top-200 + tail support at assistant positions (unmeasured under cross-context);
upper x1.25 x1.15 (that support, and pod-to-pod variation). The first A1 session measures the real figure.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))
AUDIT = REPO / "logs/stages/stage-2/teacher_modes/audit/packing_audit.json"
SCHEMA = "aadistill.stage2.teacher_modes.packed_training_budget/v2"
PLAN_SCHEMA = "aadistill.stage2.teacher_modes.packed_training_plan/v2"
STUDY_ID = "six_category_schema_restored@v1"
SEEDS = (20260726, 20260801)
ARMS = ("A1-6cat-sr", "A2-6cat-sr", "C1-6cat-sr")
#: arm -> (teacher lineage, mode in the lineage block)
LINEAGE = {"A1-6cat-sr": ("hybrid", "thinking"), "A2-6cat-sr": ("hybrid", "non_thinking"),
           "C1-6cat-sr": ("thinking_2507", "thinking")}
FREEZE = REPO / "logs/stages/stage-2/teacher_modes/audit/six_category_schema_restored_freeze.json"
PACKS = "artifacts/stages/stage-2/teacher_modes/packs_6cat"
PACK_FILES = ("blocks.npz", "ladder.json", "audit.jsonl")
#: the packed search's committed records (a_packed_search_20261105_195011): its Top-1 is A1/A2's initialization
SEARCH_EVIDENCE = REPO / "logs/stages/stage-2/teacher_modes/runs/a_packed_search_20261105_195011/evidence"
#: where the units write, on the search volume (thprcgh7s9): one directory per unit, stable across attempts
OUT = "/runpod-volume/teacher_modes/stage3_sr"
UNITS_DIR = "stage3_sr"
BOUND = REPO / "logs/stages/stage-2/teacher_modes/plans/packed_training_plan.json"
BUDGET = REPO / "logs/stages/stage-2/teacher_modes/plans/packed_training_budget.json"
#: the checkpoint interval's target, seconds of training between save points
SAVE_TARGET_S = 3600.0

#: seconds per optimizer step = a + b x KD positions per step, fitted on D4-Augmented-s20260801's L40S steps (cross
#: context, 2 x 8,192 positions, the 596M student, a 4B Qwen3 teacher on the fly); `fit_steps` reproduces it
STEP_FIT = {"a": 3.0134, "b": 8.554e-05, "residual_sd": 0.043, "steps": 368, "mean": 3.539, "max": 4.55,
            "source": "artifacts/stages/stage-2/reasoning_data/h1pa/h1pa2_20261025_055042/store/"
                      "D4-Augmented-s20260801.evidence.tgz (sha256 680a598a...) :: training/trainer/train_log.jsonl, "
                      "train_step rows after step 20"}
STEP_FACTORS = {"lower": 1.0, "expected": 1.05, "upper": 1.25 * 1.15}
#: formal evaluation of one unit (all of eval-set-1, nucleus_v1, L40S): h1pa2's measured 2,653-3,917 s (A1 v2's
#: 2,936 / 3,198 s lie inside); upper 2x for generation to the full context (P18) -- a deadline, not a forecast
EVAL_S = {"lower": 2653.1, "expected": 3917.1, "upper": 7834.2}
#: in-training validation: seconds per validation sequence (student and teacher forward; A1 v2 measured 1.37 s
#: per sequence under document isolation, cross-context is cheaper); a save point's write to the volume (A1 v2:
#: 20-40 s for 5.9 GB); the student, teacher and data load per unit
VAL_S = {"lower": 0.8, "expected": 1.4, "upper": 2.0}
SAVE_S = {"lower": 20.0, "expected": 30.0, "upper": 60.0}
LOAD_S = {"lower": 30.0, "expected": 60.0, "upper": 180.0}
#: per session: create -> setup done (A1 v2 measured 8.0 min), ALL_DONE -> pod deleted (13.3 min)
SESSION_MIN = {"lower": 15.0, "expected": 22.0, "upper": 40.0}
RATE = {"gpu_expected": 1.09, "gpu_cap": 1.19, "disk_gb": 150}
#: the runner's own pricing of a session (study_engineering.launch_spec / aadistill.infrastructure.budget):
#: hard = (setup 20 + transfer 5 + driver minutes) x 1.05 + closeout 10.5 + recovery 10; billed to hard + 1.25
RUNNER = {"setup": 20.0, "transfer": 5.0, "contingency": 0.05, "closeout": 10.5, "recovery": 10.0, "termination": 1.25}
#: one resumed attempt after a failure: the runner's fixed minutes, the work since the last save point at the
#: upper step time, a reload; A2's session is the longest, so two
RECOVERIES = {"A1-6cat-sr": 1, "A2-6cat-sr": 2, "C1-6cat-sr": 1}
#: bytes on the volume: an fp32 save of the 596M student (2,384,234,968, measured) and its AdamW state for the
#: 440,467,456 trainable parameters (3,524,023,523, measured); the evaluation package copies the weights again
MODEL_BYTES, OPTIM_BYTES = 2_384_234_968, 3_524_023_523
KIND_OF = {"A1-6cat-sr": "train_sr_a1", "A2-6cat-sr": "train_sr_a2", "C1-6cat-sr": "train_sr_c1"}


def schedule(seqs: int) -> dict:
    """E1's builder rule on a pack's sequence count (the freeze record's compute block uses it as written; its
    ``save_every`` T // 2 is superseded by `save_every` for the formal units)."""
    T = math.ceil(3 * seqs / 2)
    return {"packed_sequences": seqs, "optimizer_steps": T, "warmup_steps": max(10, round(0.05 * T)),
            "eval_every": max(25, T // 8), "save_every": max(200, T // 2)}


def fit_steps(train_log: Path, after: int = 20) -> dict:
    """`STEP_FIT` again from a trainer log: least squares of ``seconds`` on ``kd_positions`` over train_step rows."""
    import statistics
    rows = [json.loads(x) for x in Path(train_log).read_text().splitlines() if '"train_step"' in x]
    rows = [r for r in rows if r["step"] > after]
    x, y = [r["kd_positions"] for r in rows], [r["seconds"] for r in rows]
    mx, my = sum(x) / len(x), sum(y) / len(y)
    b = sum((p - mx) * (q - my) for p, q in zip(x, y)) / sum((p - mx) ** 2 for p in x)
    a = my - b * mx
    return {"a": round(a, 4), "b": b, "residual_sd": round(statistics.pstdev([q - a - b * p for p, q in zip(x, y)]), 3),
            "steps": len(rows), "mean": round(my, 3), "max": max(y)}


def seconds_per_step(kd_per_step: float, case: str) -> float:
    return round((STEP_FIT["a"] + STEP_FIT["b"] * kd_per_step) * STEP_FACTORS[case], 3)


def save_every(kd_per_step: float) -> int:
    """Optimizer steps between save points: about one hour of the arm's expected step time."""
    return max(1, round(SAVE_TARGET_S / seconds_per_step(kd_per_step, "expected")))


def frozen() -> dict:
    return json.loads(FREEZE.read_text())


def initializations() -> dict:
    """arm -> its initialization: path, the single shard's sha256, the provenance record and its sha256, and where the
    bytes come from on a pod."""
    from aadistill.infrastructure.manifest import sha256_file
    from stages.teacher_modes import training_plan as TP
    rec = SEARCH_EVIDENCE / "search_record.json"
    prod = json.loads((SEARCH_EVIDENCE / "products.json").read_text())
    doc = json.loads(rec.read_text())
    if doc.get("status") != "COMPLETE":
        raise SystemExit(f"{rec.relative_to(REPO)} is {doc.get('status')!r}: no Top-1 before the search completes")
    top = prod["top1"]
    hybrid = {"student_path": top["path"], "model_safetensors_sha256": top["files_sha256"]["model.safetensors"],
              "files_sha256": dict(top["files_sha256"]), "state_id": top["state_id"],
              "artifact_digest": top["artifact_digest"],
              "provenance": {"record": str(rec.relative_to(REPO)), "record_sha256": sha256_file(rec)},
              "bytes_on_pod": "the search volume's copy (thprcgh7s9:/teacher_modes/search_packed/top1), read in place",
              "second_copy": prod.get("off_volume_copy", {}).get("relay", "") + ":" +
                             prod.get("off_volume_copy", {}).get("prefix", "")}
    h = json.loads(TP.H1PA.read_text())["initializations"]["d4_top1"]
    mat = json.loads((REPO / h["record"]).read_text())
    c1 = {"student_path": h["path"], "model_safetensors_sha256": h["model_safetensors_sha256"],
          "files_sha256": {f["name"]: f["sha256"] for f in mat["read_back"]["files"]},
          "state_id": Path(h["path"]).name, "artifact_digest": mat["read_back"]["identity"]["artifact_digest"],
          "provenance": {"record": str(TP.D4_SEARCH.relative_to(REPO)), "record_sha256": sha256_file(TP.D4_SEARCH)},
          "materialization_record": h["record"],
          "bytes_on_pod": "relayed into student_path at setup (C1_INIT_RELAY), verified by sha256",
          "relay_prefix": next(iter(mat["relay_metadata"])).rsplit("/", 1)[0]}
    return {"A1-6cat-sr": hybrid, "A2-6cat-sr": hybrid, "C1-6cat-sr": c1}


def bind_pack(arm: str, pack_dir: Path, fr: dict | None = None) -> dict:
    """The arm's pack as the units read it: `packed_packs.verify_pack` (admitted manifest, documents, packing spec,
    packed tokens re-derived), then every identity against the freeze record's -- or SystemExit. Returns the data
    facts the units bind."""
    from stages.teacher_modes import packed_packs as PP
    pack_dir = Path(pack_dir)
    man = PP.verify_pack(pack_dir)
    fr = (fr or frozen())["arms"][arm]
    lad = json.loads((pack_dir / "ladder.json").read_text())
    ident = fr["identity"]
    files = {n: PP.sha_file(pack_dir / n) for n in PACK_FILES}
    bad = [n for n in PACK_FILES if files[n] != ident["pack_files_sha256"][n]]
    if PP.sha_file(pack_dir / PP.MANIFEST) != ident["data_manifest_sha256"]:
        bad.append(PP.MANIFEST)
    if man["content_sha256"] != ident["content_sha256"]:
        bad.append("manifest content_sha256")
    if lad["data_admission"]["packed_content_sha256"] != ident["packed_content_sha256"]:
        bad.append("packed_content_sha256")
    if fr["status"] != "FROZEN":
        bad.append(f"status {fr['status']}")
    if bad:
        raise SystemExit(f"{arm}: the pack at {pack_dir} is not the frozen one: {bad}")
    tr = lad["train"]
    return {"files_sha256": files, "packed_content_sha256": ident["packed_content_sha256"],
            "data_manifest_sha256": ident["data_manifest_sha256"],
            "data_manifest_content_sha256": ident["content_sha256"],
            "documents_file": ident["documents_file"], "before_packing": fr["before_packing"],
            "packed_sequences": int(lad["n_blocks"]), "validation_sequences": int(lad["n_val_blocks"]),
            "rung": int(lad["rungs"][0]["target_supervised_tokens"]),
            "supervised_tokens_in_sequences": int(tr["supervised_tokens"]),
            "non_padding_tokens": int(tr["non_padding_tokens"]),
            "repeated_supervised_tokens": int(tr["repeated_supervised_tokens"]),
            "packing_spec": lad["packing_spec"], "context_strategy": lad["context_strategy"]}


#: the attention every -sr unit trains with (P19): standard causal attention over the whole packed sequence with
#: continuous positions -- the trainer's ``batch.context_strategy`` value for a sequential pack
CONTEXT_STRATEGY = "cross_context"
#: the packing spec's attention term of a sequential causal pack (packed_packs)
PACK_ATTENTION = "causal_continuous_positions_v1"


def attention_problems(cfg: dict, root: Path = REPO) -> list[str]:
    """Why a unit would not train with standard causal attention and continuous positions (empty when it would): the
    config declares `CONTEXT_STRATEGY`; the pack's ladder.json declares the same and a sequential causal packing;
    blocks.npz stores no per-document position ids; and the trainer's own reader (`ladder_blocks`) hands it sequences
    without position ids -- which the trainer accepts only under ``cross_context`` and then forwards with none, so
    the model builds its ordinary causal mask over the whole sequence (`aadistill.training.train.Trainer`). Run on
    the dev box when the plan is written and on the pod at launch admission (`a1_units.py`)."""
    import numpy as np
    from aadistill.data.ladder import ladder_blocks
    bad = []
    got = (cfg.get("batch") or {}).get("context_strategy")
    if got != CONTEXT_STRATEGY:
        bad.append(f"the config's batch.context_strategy is {got!r}, not {CONTEXT_STRATEGY!r}")
    dd = Path(root) / cfg["data_dir"]
    lad = json.loads((dd / "ladder.json").read_text())
    if lad.get("context_strategy") != CONTEXT_STRATEGY:
        bad.append(f"the pack's context_strategy is {lad.get('context_strategy')!r}")
    for key in ("packing_spec", "val_packing_spec"):
        spec = lad.get(key) or {}
        if spec.get("attention") != PACK_ATTENTION or spec.get("packing") != "sequential_truncate_v1":
            bad.append(f"the pack's {key} is {spec.get('packing')!r} / {spec.get('attention')!r}, not a sequential "
                       f"causal pack")
    with np.load(dd / "blocks.npz") as z:
        if "position_ids" in z.files:
            bad.append("blocks.npz stores per-document position ids")
    train, val, _ = ladder_blocks(dd, int(cfg["rung"]), int(cfg["val_blocks"]))
    if len(train) != 3 or len(val) != 3:
        bad.append(f"the trainer's reader gives {len(train)}- / {len(val)}-tuples: position ids reach the trainer")
    return bad


def kd_per_step(data: dict, batch: int = 2) -> float:
    """KD (and CE) positions per optimizer step: the assistant positions the sequences hold, per sequence x batch."""
    return data["supervised_tokens_in_sequences"] / data["packed_sequences"] * batch


def unit_config(arm: str, seed: int, data: dict, init: dict) -> dict:
    from stages.teacher_modes import training_plan as TP
    lineage, mode = LINEAGE[arm]
    t = TP.teachers()[lineage]
    e1 = json.loads(TP.E1.read_text())
    sch = TP.schedule(data["packed_sequences"])
    cfg = copy.deepcopy(e1)
    cfg.pop("_purpose", None)
    cfg.update(run_name=f"research_a_stage3_{arm}_s{seed}", seed=seed, data_dir=f"{PACKS}/{arm}", packing="ladder",
               student_path=init["student_path"], tokenizer_source=TP.TOKENIZERS[lineage],
               tokenizer_sha256=t["tokenizer_sha256"],
               teacher={"model_id": t["model_id"], "revision": t["revision"], "dtype": "bfloat16"},
               loss=dict(TP.OBJECTIVE), rung=data["rung"], val_blocks=data["validation_sequences"],
               out_dir=f"{OUT}/{arm}-s{seed}/train")
    cfg["schedule"] = {k: sch[k] for k in ("total_steps", "warmup_steps", "min_lr_frac")}
    cfg["intervals"] = {**cfg["intervals"], "eval_every": sch["eval_every"], "eval_blocks": data["validation_sequences"]}
    cfg["checkpoint"] = {"save_every": save_every(kd_per_step(data)), "keep_last": 1, "bind_identity": True}
    #: declared, not left to the trainer's default (maintainer 2026-11-07, Stage 3 authorization, section II)
    cfg["batch"] = {**cfg["batch"], "context_strategy": CONTEXT_STRATEGY}
    cfg["lineage"] = {"teacher": t, "mode": mode, "tokenizer": {"files_sha256": TP.tokenizer_files()[lineage]},
                      "initialization": {"root_teacher": {k: t[k] for k in ("model_id", "revision")},
                                         "model_safetensors_sha256": init["model_safetensors_sha256"],
                                         "provenance": dict(init["provenance"])},
                      "data": [{"files_sha256": dict(data["files_sha256"]), "mode": mode,
                                "teacher": {k: t[k] for k in ("model_id", "revision")},
                                "packed_content_sha256": data["packed_content_sha256"],
                                "data_manifest_content_sha256": data["data_manifest_content_sha256"]}]}
    return cfg


def effective_schedule(cfg: dict, data: dict) -> dict:
    """What a unit's config makes the trainer do, in the field's terms."""
    from aadistill.training.train import LR_SCHEDULE_ID
    T, ck = cfg["schedule"]["total_steps"], cfg["checkpoint"]
    saves = [s for s in range(ck["save_every"], T, ck["save_every"])] + [T]
    return {"epochs": 3, "packed_sequences": data["packed_sequences"], "optimizer_steps": T,
            "batch_size": cfg["batch"]["blocks_per_step"], "micro_batch_size": cfg["batch"]["micro_blocks"],
            "gradient_accumulation_steps": cfg["batch"]["blocks_per_step"] // cfg["batch"]["micro_blocks"],
            "sequence_length": cfg["block_len"], "learning_rate": cfg["optim"]["lr"],
            "warmup_steps": cfg["schedule"]["warmup_steps"], "lr_schedule": LR_SCHEDULE_ID,
            "min_lr_frac": cfg["schedule"]["min_lr_frac"],
            "data_order": f"per-epoch torch.randperm over the {data['packed_sequences']} packed sequences, seed "
                          f"{cfg['seed']} x 1,000,003 + epoch (aadistill.training.train.epoch_permutation)",
            "validation": {"every_steps": cfg["intervals"]["eval_every"], "sequences": cfg["intervals"]["eval_blocks"]},
            "save_points": saves, "keep_last": ck["keep_last"],
            "kd_positions_per_step": round(kd_per_step(data)),
            "supervised_token_exposure": 3 * data["supervised_tokens_in_sequences"],
            "non_padding_tokens_trained": 3 * data["non_padding_tokens"]}


def units(packs_dir: Path | None = None) -> list[dict]:
    """The six units, seed 20260726 first within each arm, bound to the frozen packs and the initializations."""
    fr, inits = frozen(), initializations()
    out = []
    for arm in ARMS:
        data = bind_pack(arm, Path(packs_dir or REPO / PACKS) / arm, fr)
        for seed in SEEDS:
            cfg = unit_config(arm, seed, data, inits[arm])
            out.append({"unit": f"{arm}-s{seed}", "arm": arm, "seed": seed, "kind": KIND_OF[arm], "config": cfg,
                        "effective_schedule": effective_schedule(cfg, data),
                        "data": {k: data[k] for k in ("files_sha256", "packed_content_sha256", "data_manifest_sha256",
                                                      "data_manifest_content_sha256", "documents_file",
                                                      "before_packing", "packed_sequences", "validation_sequences",
                                                      "packing_spec", "context_strategy")},
                        "initialization": {k: v for k, v in inits[arm].items()}})
    return out


def config_problems(us: list[dict]) -> dict:
    """Each unit's config through the trainer's own validation and the lineage validation (`training_plan.validate`),
    and nothing left unset."""
    from aadistill.training.train import validate_train_config
    from stages.teacher_modes import training_plan as TP
    res = TP.validate(us)
    for u in us:
        bad = res[u["unit"]]["problems"]
        try:
            validate_train_config(u["config"])
        except (ValueError, TypeError, KeyError) as exc:
            bad.append(f"trainer config: {exc}")
        if TP.UNSET in json.dumps(u["config"]) or "TO_BE_SET" in json.dumps(u["config"]):
            bad.append("an unset field remains")
        bad += [f"attention: {x}" for x in attention_problems(u["config"])]
        res[u["unit"]] = {"problems": bad}
    return res


# ---- cost -------------------------------------------------------------------------------------------------------

def unit_seconds(u: dict, case: str) -> dict:
    es, cfg = u["effective_schedule"], u["config"]
    s = seconds_per_step(es["kd_positions_per_step"], case)
    n_val = len(range(0, es["optimizer_steps"] + 1, cfg["intervals"]["eval_every"])) + 1
    train = (es["optimizer_steps"] * s + n_val * es["validation"]["sequences"] * VAL_S[case]
             + len(es["save_points"]) * SAVE_S[case] + LOAD_S[case])
    return {"seconds_per_step": s, "train_s": round(train), "evaluation_s": EVAL_S[case],
            "unit_s": round(train + EVAL_S[case])}


def runner_cap(driver_minutes: float) -> dict:
    """The runner's hard cap for a session of ``driver_minutes`` at the price ceiling (study_engineering.all_in)."""
    disk_h = RATE["disk_gb"] * 0.10 / (24 * 30.4375)
    hard = ((RUNNER["setup"] + RUNNER["transfer"] + driver_minutes) * (1 + RUNNER["contingency"])
            + RUNNER["closeout"] + RUNNER["recovery"])
    minutes = hard + RUNNER["termination"]
    return {"driver_minutes": round(driver_minutes), "billable_minutes": round(minutes, 1),
            "hard_cap_usd": math.ceil(minutes / 60 * (RATE["gpu_cap"] + disk_h) * 100) / 100}


def budget(us: list[dict], position: dict | None = None) -> dict:
    disk_h = RATE["disk_gb"] * 0.10 / (24 * 30.4375)
    per_unit, arms = {}, {}
    for u in us:
        per_unit[u["unit"]] = {c: unit_seconds(u, c) for c in ("lower", "expected", "upper")}
    for arm in ARMS:
        mine = [u for u in us if u["arm"] == arm]
        rows = {}
        for c in ("lower", "expected", "upper"):
            unit_h = sum(per_unit[u["unit"]][c]["unit_s"] for u in mine) / 3600
            h = unit_h + SESSION_MIN[c] / 60
            price = RATE["gpu_cap"] if c == "upper" else RATE["gpu_expected"]
            rows[c] = {"gpu_hours": round(h, 2), "usd": round(h * (price + disk_h), 2)}
        upper_train_h = max(per_unit[u["unit"]]["upper"]["train_s"] for u in mine) / 3600
        driver_min = sum(per_unit[u["unit"]]["upper"]["unit_s"] for u in mine) / 60
        cap = runner_cap(driver_min)
        es = mine[0]["effective_schedule"]
        lost_min = (es["save_points"][0] * seconds_per_step(es["kd_positions_per_step"], "upper")) / 60
        rec_min = (RUNNER["setup"] + RUNNER["transfer"]) * (1 + RUNNER["contingency"]) + RUNNER["closeout"] \
            + RUNNER["recovery"] + RUNNER["termination"] + lost_min + LOAD_S["upper"] / 60
        rec_usd = math.ceil(rec_min / 60 * (RATE["gpu_cap"] + disk_h) * 100) / 100
        n_rec = RECOVERIES[arm]
        arms[arm] = {"kind": KIND_OF[arm], "seeds": list(SEEDS), "session": rows,
                     "unit_deadline_hours": math.ceil(upper_train_h * 4) / 4,
                     "session_hard_cap": cap,
                     "recovery": {"per_resumed_attempt_minutes": round(rec_min, 1), "usd_each": rec_usd,
                                  "attempts_reserved": n_rec,
                                  "rule": "a resumed attempt pays the runner's fixed minutes, the training since the "
                                          "last save point at the upper step time, and a reload; it never retrains "
                                          "a finished unit or re-evaluates a scored one"},
                     "kind_cumulative_cap_usd": round(cap["hard_cap_usd"] + n_rec * rec_usd, 2)}
    total = {c: round(sum(a["session"][c]["usd"] for a in arms.values()), 2) for c in ("lower", "expected", "upper")}
    hours = {c: round(sum(a["session"][c]["gpu_hours"] for a in arms.values()), 2) for c in ("lower", "expected", "upper")}
    caps = round(sum(a["kind_cumulative_cap_usd"] for a in arms.values()), 2)
    return {"per_unit": per_unit, "arms": arms, "six_units_usd": total, "six_units_gpu_hours": hours,
            "stage3_cumulative_cap_usd": caps,
            "evaluation": "each unit's formal eval-set-1 evaluation runs on the same pod right after its training "
                          "(a1_units.py): its seconds are inside the unit's row and its cost inside the arm's session "
                          "cap; no separate evaluation session is priced",
            "rates": {"gpu_expected_usd_per_hour": RATE["gpu_expected"], "gpu_cap_usd_per_hour": RATE["gpu_cap"],
                      "container_disk_usd_per_hour": round(disk_h, 4)}}


def capacity(us: list[dict]) -> dict:
    """What Stage 3 writes to the search volume, and its high-water mark."""
    ckpt = MODEL_BYTES + OPTIM_BYTES
    unit_final = ckpt + MODEL_BYTES          #: the final save point + the evaluation package's copy of its weights
    n = len(us)
    return {"save_point_bytes": ckpt, "unit_after_evaluation_bytes": unit_final,
            "while_saving_bytes": 2 * ckpt,
            "stage3_high_water_bytes": (n - 1) * unit_final + 2 * ckpt + MODEL_BYTES,
            "stage3_after_bytes": n * unit_final,
            "rule": "keep_last 1: while a save point is written the previous one stays (2 x 5.91 GB), then is removed; "
                    "after a unit, its final save point (weights + AdamW state) and the evaluation package remain",
            "unmeasured": "the search's own footprint on the volume after its last level (221 GB by du before it); "
                          "the journal's checkpoints sum to 270.6 GB (decimal) -- the first session measures it "
                          "(a1_units.volume_room) and refuses a unit that would not fit"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true", help="write the plan and the budget records")
    ap.add_argument("--packs", default=None, help="the packs' directory (default: the repository's)")
    a = ap.parse_args(argv)
    us = units(Path(a.packs) if a.packs else None)
    probs = config_problems(us)
    if any(v["problems"] for v in probs.values()):
        print(json.dumps(probs, indent=1))
        return 2
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    b = budget(us)
    plan = {"schema": PLAN_SCHEMA, "study_id": STUDY_ID, "written_utc": now,
            "decided_by": "maintainer 2026-11-07 (Stage 3 preparation of six_category_schema_restored@v1): six units "
                          "A1-6cat-sr / A2-6cat-sr / C1-6cat-sr x seeds 20260726, 20260801",
            "freeze_record": {"path": str(FREEZE.relative_to(REPO)), "sha256": __import__("hashlib").sha256(
                FREEZE.read_bytes()).hexdigest()},
            "seed_order": list(SEEDS), "units_dir": UNITS_DIR,
            "checkpoint_policy": {"target_seconds_between_save_points": SAVE_TARGET_S,
                                  "save_every": {arm: next(u for u in us if u["arm"] == arm)["config"]["checkpoint"][
                                      "save_every"] for arm in ARMS},
                                  "keep_last": 1, "contents": ["model/ (fp32 weights, config)",
                                                               "trainer_state.pt: AdamW state, step, consumed "
                                                               "sequences, torch + CUDA RNG, config sha256, bound "
                                                               "identity", "resume_identity.json: data, "
                                                               "initialization, lineage stamp"]},
            "units": us, "validation": probs,
            "comparisons": {"A2_over_A1_optimizer_steps": round(
                us[2]["effective_schedule"]["optimizer_steps"] / us[0]["effective_schedule"]["optimizer_steps"], 4),
                "A2_over_A1_supervised_token_exposure": round(
                    us[2]["effective_schedule"]["supervised_token_exposure"]
                    / us[0]["effective_schedule"]["supervised_token_exposure"], 4),
                "A1_vs_A2": "matched in unique supervised tokens before packing and in category composition, NOT in "
                            "compute: A2's packed sequences, steps and training FLOPs are ~3.53x A1's, its supervised-"
                            "token exposure 0.88x (non-thinking turns carry long prompts and short answers)",
                "C1_vs_A1": "a secondary, cross-teacher comparison: C1's initialization came from its own root's D4 "
                            "search under forward baseline v1, not the packed search, so a difference is not a pure "
                            "teacher difference"},
            "authorizes": "nothing"}
    budget_doc = {"schema": SCHEMA, "study_id": STUDY_ID, "written_utc": now, "is_an_estimate": True,
                  "step_model": {"fit": STEP_FIT, "factors": STEP_FACTORS}, **b, "capacity": capacity(us),
                  "authorizes": "nothing"}
    if a.write:
        BOUND.write_text(json.dumps(plan, indent=1, ensure_ascii=False) + "\n")
        BUDGET.write_text(json.dumps(budget_doc, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"save_every": plan["checkpoint_policy"]["save_every"],
                      "arms": {k: {"session": v["session"], "cap": v["session_hard_cap"]["hard_cap_usd"],
                                   "kind_cap": v["kind_cumulative_cap_usd"], "deadline_h": v["unit_deadline_hours"]}
                               for k, v in b["arms"].items()},
                      "six": b["six_units_usd"], "hours": b["six_units_gpu_hours"],
                      "stage3_cap": b["stage3_cumulative_cap_usd"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
