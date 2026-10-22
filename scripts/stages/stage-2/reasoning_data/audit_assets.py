#!/usr/bin/env python3
"""The $0 audit of every asset Stage 2/3 could reuse, under the new target contract.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.reasoning_data.audit_assets

Writes `logs/stages/stage-2/reasoning_data/audit/asset_audit.json`. Every input
is read from its recorded location and checked against its recorded hash; every
figure is computed here, none is restated. Sections:

* `corpus_v2` -- what the historical recovery corpus holds, which of its
  SELECTED targets are wrong, and how its candidates fare under
  `aadistill.data.target_quality` with a verified-correct policy;
* `ladder` -- per historical rung: unique supervised tokens, real versus padding
  positions, terminal cuts, 3-epoch exposure and block passes, and the
  per-type packing waste (tool sessions);
* `base_pool_v1_fluid` -- the 519 verified Stage-0 fluid units: what they are,
  and whether each one is reconstructable from its raw teacher record;
* `teacher_cal` -- all 672 calibration sessions under the new contract;
* `recovery_scaling_history` -- the only fluid measurement of the E1 rungs;
* `isolation` -- every asset under `isolation_rules.json`.

Nothing here decides what trains; it states what the assets are.
"""
from __future__ import annotations

import json
import math
import statistics
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone

from stages.reasoning_data import assets as A
from stages.reasoning_data import isolation_check as IC
from stages.reasoning_data import process_check as PC

from aadistill.data import cleaning
from aadistill.data.target_quality import AcceptancePolicy, Check, assess, summarize
from stages.calibration_volume.tasks import registry as R
from stages.calibration_volume.tasks.common import build_instance

OUT = A.LOG_DIR / "audit/asset_audit.json"
EPOCHS = 3          # the frozen recipe's rule: ceil(blocks * 3 / blocks_per_step)
BLOCKS_PER_STEP = 2
#: corpus_v2's own candidate fields mapped onto the contract's ending categories.
CV2_ENDS = {"natural": ("eos",), "length": ("length", "context_limit", "over_budget"),
            "degeneration": ("degeneration",), "other": ("not_finished",)}
TC_ENDS = {"natural": ("eos",), "length": ("length_limited",),
           "degeneration": ("degeneration",)}
VERIFIED = AcceptancePolicy()
#: an episode with an illegal action is not a target, whatever its outcome
VERIFIED_EPISODE = AcceptancePolicy(reject_observations={"action_legality": ("illegal_action",)})


def _stats(xs):
    xs = sorted(xs)
    if not xs:
        return None
    return {"n": len(xs), "sum": int(sum(xs)), "mean": round(statistics.fmean(xs), 1),
            "p50": xs[len(xs) // 2], "p90": xs[min(len(xs) - 1, int(0.9 * len(xs)))],
            "max": xs[-1]}


def _cv2_stop(c: dict) -> str:
    if c.get("hit_cap"):
        return "context_limit"
    if c.get("over_budget"):
        return "over_budget"
    if not c.get("finished"):
        return "not_finished"
    if c.get("length_limited"):
        return "length"
    if c.get("degeneration"):
        return "degeneration"
    return "eos"


CV2_SCOPE = "corpus_v2's own correctness verdict (per-type checker, recorded at generation)"


def _cv2_answer(c: dict) -> Check:
    if c.get("correctness_verdict") == "unverifiable_slice":
        return Check("unverifiable", "no answer key for this slice", CV2_SCOPE)
    if c.get("correct") is True:
        return Check("correct", "", CV2_SCOPE)
    if c.get("correct") is False:
        return Check("incorrect", str(c.get("correctness_verdict") or ""), CV2_SCOPE)
    return Check("unverifiable", str(c.get("correctness_verdict") or ""), CV2_SCOPE)


def audit_corpus_v2() -> dict:
    m = A.corpus_v2_manifest()
    records, by_type_units = [], defaultdict(set)
    gen_all, gen_acc = Counter(), Counter()
    for ex in A.corpus_v2_candidates():
        for c in ex["candidates"]:
            proto = cleaning.check_serialization(c) if c.get("raw") is not None else "no_raw"
            rec = assess(ex["id"], int(c["index"]), stop_reason=_cv2_stop(c),
                         protocol_reason=proto, answer=_cv2_answer(c),
                         policy=VERIFIED, end_reasons=CV2_ENDS,
                         generated_tokens=c.get("new_tokens"))
            rec.unit_id = f"{ex['data_type']}::{ex['id']}"
            records.append(rec)
            by_type_units[ex["data_type"]].add(ex["id"])
            gen_all[ex["data_type"]] += int(c.get("new_tokens") or 0)
            if rec.accepted_for_training:
                gen_acc[ex["data_type"]] += int(c.get("new_tokens") or 0)
    contract = summarize(records, group_of=lambda r: r.unit_id.split("::")[0])

    selected = defaultdict(lambda: {"sessions": 0, "correct": Counter(), "supervised": [],
                                    "rendered": [], "prompt": []})
    for s in A.corpus_v2_sessions():
        row = selected[s["data_type"]]
        row["sessions"] += 1
        row["correct"][_cv2_answer(s).state] += 1
        row["supervised"].append(int(s["n_supervised_tokens"]))
        row["rendered"].append(int(s["n_rendered_tokens"]))
        row["prompt"].append(int(s["n_rendered_tokens"]) - int(s["n_supervised_tokens"]))
    sel = {}
    for t, row in sorted(selected.items()):
        sup = sum(row["supervised"])
        sel[t] = {"sessions": row["sessions"], "selected_target_answer": dict(row["correct"]),
                  "selected_wrong_share": round(row["correct"]["incorrect"] / row["sessions"], 4),
                  "supervised_tokens": _stats(row["supervised"]),
                  "rendered_tokens": _stats(row["rendered"]),
                  "rendered_per_supervised": round(sum(row["rendered"]) / sup, 4)}
    total_sup = sum(v["supervised_tokens"]["sum"] for v in sel.values())
    verified_sup = sum(s["supervised_tokens"]["sum"] * s["selected_target_answer"].get("correct", 0)
                       / s["sessions"] for s in sel.values())
    return {
        "manifest": {"teacher": m["teacher"], "engine": m["decoding"]["engine_version"],
                     "decoding": {k: m["decoding"][k] for k in ("n", "temperature", "top_p", "top_k", "min_p")},
                     "acceptance_basis": m["acceptance"]["basis"],
                     "correctness_computed_but_not_enforced": m["acceptance"]["correctness_computed_but_not_enforced"],
                     "session_limit_tokens": m["session_limit_tokens"],
                     "code_state": m["code_state"].get("git_commit")},
        "per_type_manifest": {t: {k: v[k] for k in ("examples", "accepted", "candidate_accept_rate",
                                                     "correctness_rate", "reject_reasons")}
                              for t, v in m["types"].items()},
        "selected_sessions": sel,
        "selected_supervised_tokens": total_sup,
        "selected_supervised_tokens_with_a_verified_correct_answer": int(verified_sup),
        "under_target_quality_contract": {
            "contract": "aadistill.data.target_quality/v2",
            "policy": "verified correct, naturally terminated, protocol valid (AcceptancePolicy())",
            "per_type": contract,
            "generated_tokens": {t: {"all_candidates": gen_all[t], "accepted_candidates": gen_acc[t],
                                     "accepted_share": round(gen_acc[t] / gen_all[t], 4) if gen_all[t] else None}
                                 for t in sorted(gen_all)},
            "reading": ("units with at least one candidate a verified-correct policy accepts; code and "
                        "tool_calling have no answer key, so none of their units can pass"),
        },
    }


def audit_ladder() -> dict:
    meta = A.ladder_meta()
    audit = A.ladder_audit()
    block_len = int(meta["block_len"])
    rungs = []
    for r in meta["rungs"]:
        n = int(r["n_blocks"])
        rows = audit[:n]
        single = sum(1 for b in rows if len(b["sessions"]) == 1)
        fills = sorted(b.get("unpadded_length", 0) / block_len for b in rows)
        discarded = sum((b.get("terminal_truncation") or {}).get("supervised_discarded", 0)
                        for b in rows)
        steps = math.ceil(n * EPOCHS / BLOCKS_PER_STEP)
        real = int(r["real_tokens"])
        rungs.append({
            "target_supervised_tokens": r["target_supervised_tokens"],
            "unique_supervised_tokens": r["actual_supervised_tokens"],
            "blocks": n, "sessions": r["n_sessions"],
            "real_tokens": real, "padding_tokens": r["padding_tokens"],
            "padding_share": round(r["padding_tokens"] / (n * block_len), 4),
            "single_session_blocks": single, "median_block_fill": round(fills[len(fills) // 2], 4),
            "terminal_truncations": r["terminal_truncations"],
            "supervised_tokens_discarded_by_cuts": int(discarded),
            "steps_at_3_epochs": steps, "block_passes": steps * BLOCKS_PER_STEP,
            "forward_positions": steps * BLOCKS_PER_STEP * block_len,
            "supervised_token_exposure": round(r["actual_supervised_tokens"] * steps * BLOCKS_PER_STEP / n),
            "epochs": round(steps * BLOCKS_PER_STEP / n, 4),
        })
    # Packing waste by the dominant type of a block, over the whole pack.
    per_type = defaultdict(lambda: {"blocks": 0, "fill": [], "sessions": 0})
    for b in audit:
        mix = Counter()
        for s in b["sessions"]:
            mix[s["data_type"]] += s["supervised_retained"]
        if not mix:
            continue
        dom = max(sorted(mix), key=lambda t: mix[t])
        row = per_type[dom]
        row["blocks"] += 1
        row["sessions"] += len(b["sessions"])
        row["fill"].append(b.get("unpadded_length", 0) / block_len)
    waste = {t: {"blocks": v["blocks"], "sessions_per_block": round(v["sessions"] / v["blocks"], 3),
                 "mean_fill": round(statistics.fmean(v["fill"]), 4)}
             for t, v in sorted(per_type.items())}
    return {"pack": "artifacts/shared/instruments/ladder_uniform_probe",
            "blocks_sha256": meta["outputs"]["blocks"], "packing": meta["packing"],
            "system_prompt_groups": meta["system_prompt_groups"], "block_len": block_len,
            "packing_efficiency": meta["packing_efficiency"],
            "frozen_recipe_rule": "steps = ceil(blocks * 3 / 2); 2 blocks per step; teacher and student forward all 8,192 positions (truncate_padding off)",
            "rungs": rungs, "waste_by_dominant_type": waste}


def audit_base_pool() -> dict:
    man = A.base_pool_manifest()
    items = [it for it in A.base_pool_items() if it.get("pool_class") == "primary"]
    units = defaultdict(list)
    for it in items:
        units[it["unit_id"]].append(it)
    tasks = A.calibration_tasks()
    tc = {r["task_id"]: r for r in A.teacher_cal_records()}
    by_axis, by_gen, fams = Counter(), Counter(), set()
    reconstructable, problems = 0, Counter()
    for uid, rows in units.items():
        it = rows[0]
        by_axis[it["axis"]] += 1
        by_gen[it["generator_id"]] += 1
        fams.add(f"{it['generator_id']}/{it['family_id']}")
        rec, task = tc.get(uid), tasks.get(uid)
        if rec is None or task is None:
            problems["missing_teacher_record_or_task"] += 1
            continue
        if any("output_token_ids" not in t or A.split_raw(t.get("raw", "")) is None
               for t in rec["turns"]):
            problems["turn_without_token_ids_or_think_close"] += 1
            continue
        reconstructable += 1
    return {
        "manifest_command": man.get("command"),
        "units": len(units), "items": len(items),
        "units_by_axis": dict(sorted(by_axis.items())),
        "units_by_generator": dict(sorted(by_gen.items())),
        "distinct_generator_families": len(fams),
        "scored_positions": sum(int(it["scored_positions"]) for it in items),
        "reasoning_positions": sum(int(it["reasoning_positions"]) for it in items),
        "final_answer_positions": sum(int(it["final_answer_positions"]) for it in items),
        "tokens_per_unit": _stats([sum(int(i["n_tokens"]) for i in rows) for rows in units.values()]),
        "reconstructable_from_teacher_record": reconstructable,
        "reconstruction_problems": dict(problems),
        "provenance_per_item": {
            "present": ["generator_id", "family_id", "axis", "difficulty", "task_id", "verified"],
            "absent_but_joinable": "seed and content_sha256 via task_id -> calibration_v2.jsonl; per-turn seed, stop reason and output token ids via task_id -> teacher_cal.jsonl",
            "absent": "teacher decoding parameters and revision per item (the generation record's summary holds them once)"},
        "seed_range": "calibration [2,000,000, 3,000,000): built before the Stage-2 training range existed",
        "reuse_as_training_data": (
            "NOT ADOPTED here. The units are reconstructable and pass R1-R4, but they were the "
            "CALIBRATION data the 12 initialization candidates were selected on; training on them "
            "needs its own identity (a new corpus id), the R5 note (calibration seeds, not the "
            "training range) and a maintainer decision"),
    }


def audit_teacher_cal() -> dict:
    recs = A.teacher_cal_records()
    tasks = A.calibration_tasks()
    from stages.calibration_volume import score_battery as SB
    scored = {s["task_id"]: s for s in SB.score(tasks, recs, "v2")}  # the protocol H1 generates under
    out = []
    for r in recs:
        stop = next((t["stop_reason"] for t in r["turns"] if t["stop_reason"] != "eos"), "eos")
        proto = None if all(A.split_raw(t.get("raw", "")) for t in r["turns"]) else "think_not_closed"
        s = scored.get(r["task_id"], {})
        obs = {}
        if r.get("interactive"):
            ok = bool(s.get("correct"))          # the replayed environment's success
            inv = int((s.get("env") or {}).get("invalid_actions", 0))
            obs["action_legality"] = Check("all_legal" if inv == 0 else "illegal_action",
                                           f"invalid_actions={inv}", "environment replay of every action")
            scope = "the environment's goal reached on replay"
        else:
            ok = bool(s.get("answer_correct"))   # the boxed answer alone, apart from termination
            scope = "the boxed final answer against the generator's gold"
            parts = A.split_raw(r["turns"][-1].get("raw", "")) if r["turns"] else None
            if ok and parts:
                task = tasks[r["task_id"]]
                obs["text_consistency"] = PC.text_consistency(
                    parts[0], s.get("extracted"), task["answer_type"],
                    prompt=" ".join(m.get("content") or "" for m in task["messages"]))
        ans = Check("correct" if ok else "incorrect", "" if ok else str(s.get("reason", "")), scope)
        rec = assess(r["task_id"], 0, stop_reason=stop, protocol_reason=proto, answer=ans,
                     steps=PC.NO_STEP_CHECKER, observations=obs,
                     policy=VERIFIED_EPISODE if r.get("interactive") else VERIFIED, end_reasons=TC_ENDS,
                     generated_tokens=sum(int(t["generated_tokens"]) for t in r["turns"]))
        rec.unit_id = f"{r['axis']}::{r['task_id']}"
        out.append(rec)
    gen, sup = Counter(), Counter()
    for r in recs:
        gen[r["axis"]] += sum(int(t["generated_tokens"]) for t in r["turns"])
    for it in A.base_pool_items():
        if it.get("pool_class") == "primary":
            sup[it["axis"]] += int(it["scored_positions"])
    yields = {a: {"generated_tokens": gen[a], "verified_supervised_positions_le_8192": sup[a],
                  "yield": round(sup[a] / gen[a], 4)} for a in sorted(gen)}
    return {"n_tasks": len(recs), "candidates_per_task": 1, "yield_by_axis": yields,
            "under_target_quality_contract": summarize(out, group_of=lambda x: x.unit_id.split("::")[0]),
            "overall": summarize(out)["all"],
            "note": "n = 1 per task, so unit and candidate acceptance coincide; base_pool_v1 additionally dropped 53 accepted units whose render exceeded 8,192 tokens"}


def audit_history() -> dict:
    rc = A.read_json(A.S0 / "design/resolution_check.json")
    rows = {}
    for name, v in rc["per_model"].items():
        rows[name] = {"static": v["static"]["correct"], "interactive": v["interactive"]["correct"],
                      "all": v["all"]["correct"], "n": v["all"]["n"]}
    return {"source": "logs/stages/stage-0/calibration_volume/design/resolution_check.json",
            "set": rc["set"], "fluid_correct": rows,
            "reading": "the E1 KD-heavy recipe on corpus_v2 rungs 0.86M-5.50M leaves PCA students at 0-10/720; "
                       "this is a property of that data+recipe, measured on a development-exposed set"}


def training_range_sample(per: int = 3) -> list[dict]:
    """A $0 sample of the R5 lineage: v2 generators, calibration families less the
    R9 mechanism-equivalent ones (as the task builder draws them), training seeds."""
    from stages.reasoning_data.build_training_tasks import MECHANISM_EXCLUDED
    lo, _ = next(r for r in IC.rules()["rules"] if r["id"] == "R5_shared_generator_exposure")["training_seed_range"]
    out = []
    for gen in R.GENERATORS_V2:
        fams = [f for f in R.family_split(gen)["calibration"]
                if f"{gen.generator_id}/{f}" not in MECHANISM_EXCLUDED]
        if not fams:                      # an evaluation-only generator: never sampled
            continue
        for d_i, diff in enumerate(("L1", "L2", "L3")):
            for i in range(per):
                fam = gen.family(fams[i % len(fams)])
                try:
                    t = build_instance(gen, fam, diff, lo + d_i * 100_000 + i)
                except Exception:   # an unsatisfiable draw is skipped, as the builders do
                    continue
                t.update(unit_id=t["task_id"], source="s0_generators_v2_calsplit",
                         source_split="calibration_families")
                out.append(t)
    return out


def audit_isolation() -> dict:
    sources = A.read_json(A.CONFIG_DIR / "sources.json")
    cv2 = [{"unit_id": s["id"], "messages": s["messages"], "source": s["source"]}
           for s in A.corpus_v2_sessions()]
    tasks = A.calibration_tasks()
    pool_ids = {it["unit_id"] for it in A.base_pool_items() if it.get("pool_class") == "primary"}
    bp = [{**tasks[u], "unit_id": u} for u in sorted(pool_ids) if u in tasks]
    sample = training_range_sample()
    return {
        "corpus_v2_sessions": IC.check_pool(cv2, historical=True),
        "base_pool_v1_fluid": IC.check_pool(bp, historical=True),
        "stage2_training_range_sample": {"n_generated": len(sample),
                                         "report": IC.check_pool(sample, sources=sources)},
    }


def code_state() -> dict:
    head = subprocess.run(["git", "-C", str(A.REPO), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(A.REPO), "status", "--porcelain"],
                                capture_output=True, text=True).stdout.strip())
    return {"git_commit": head, "dirty": dirty}


def main() -> int:
    doc = {
        "schema": "aadistill.stage2.reasoning_data.asset_audit/v1",
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code_state": code_state(),
        "inputs": {
            "corpus_v2_manifest": A.sha256_file(A.CORPUS_V2 / "manifest.json"),
            "ladder_json": A.sha256_file(A.LADDER / "ladder.json"),
            "base_pool_v1_manifest": A.sha256_file(A.BASE_POOL / "manifest.json"),
            "base_pool_v1_items": A.sha256_file(A.BASE_POOL / "items.jsonl"),
            "calibration_v2": A.sha256_file(A.CALIBRATION_V2),
        },
        "corpus_v2": audit_corpus_v2(),
        "ladder": audit_ladder(),
        "base_pool_v1_fluid": audit_base_pool(),
        "teacher_cal": audit_teacher_cal(),
        "recovery_scaling_history": audit_history(),
        "isolation": audit_isolation(),
        "authorizes": "nothing",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(OUT.relative_to(A.REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
