#!/usr/bin/env python3
"""A1's result (Research A v2, A1 phase D): both seeds on eval-set-1 v2 under nucleus_v1, every axis apart. $0, dev box.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.a1_analysis \\
        --units <extracted a1_evidence>/ --teacher <extracted v1 teacher_eval__teacher_H-T>/ [--out <json>]

Reported, never pooled into one score (P10.2, P10.3):
* the 570 formal static fluid items by axis -- task-contained deduction, inductive rule discovery, compositional
  generalization, abstract/analogical -- and over the four, per seed, with a Wilson 95% interval, the two seeds'
  mean and a paired bootstrap interval of that mean over items;
* the 144 formal interactive episodes, exploratory;
* rollout integrity on the formal fluid items: correct, usable (an answer extracted), natural termination,
  degeneration stops, context-limit hits (right-censored, never read as failures), reasoning and answer tokens
  (generated tokens through the generated ``</think>`` and after it);
* every other unit of eval-set-1 beside, as the formal report measured it;
* the teacher (the hybrid, thinking mode, v1's teacher evaluation on the same frozen set and decoding) and each
  seed's gap to it, by axis;
* the training side from each unit's own logs: final CE and KD loss, the validation losses, optimizer steps,
  and the data's supervised-token exposure and FLOPs (the admission record).

What this does not say: anything about non-thinking mode (A2 is not run), Research A's final GO / NO-GO, or a
teacher choice for Research B. A1's numbers change no frozen A2 / C1 protocol.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.reasoning_data.h1_analysis import FLUID, STATIC_AXES, capabilities, fluid_meta  # noqa: E402

FLUID_KEY = "__".join(FLUID)
OUT = REPO / "logs/stages/stage-2/teacher_modes/analyses/a1_result.json"
ADMISSION = REPO / "logs/stages/stage-2/teacher_modes/data/token_matched_admission.json"
DECODING_SHA = "912214c68db6acd8a42643b24e4aa7a420603884ca6b20ba845a099cfe5b00c4"
THINK_CLOSE = 151668
BOOT, BOOT_SEED = 10_000, 20261101


def wilson(c: int, n: int, z: float = 1.959964) -> tuple[float, float] | None:
    if not n:
        return None
    p = c / n
    d = 1 + z * z / n
    m = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(m - h, 4), round(m + h, 4)


def rate(rows: list[dict]) -> dict:
    n, c = len(rows), sum(bool(r.get("correct")) for r in rows)
    return {"n": n, "correct": c, "rate": round(c / n, 4) if n else None, "wilson95": wilson(c, n)}


def jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def evidence(d: Path) -> dict:
    """One evaluation directory: its fluid rows and generations, every row file, the formal report."""
    ev = d / "evaluation" if (d / "evaluation").is_dir() else d
    summary = json.loads((ev / f"generations/{FLUID_KEY}.summary.json").read_text())
    dec = (summary.get("identity") or {}).get("decoding_protocol", {}).get("sha256")
    if dec != DECODING_SHA:
        raise SystemExit(f"{d}: generations under decoding {dec}, not nucleus_v1")
    return {"rows": {f.name.removesuffix(".rows.jsonl"): jsonl(f) for f in sorted((ev / "rows").glob("*.rows.jsonl"))},
            "generations": {g["task_id"]: g for g in jsonl(ev / f"generations/{FLUID_KEY}.jsonl")},
            "formal_report": json.loads((ev / "formal_report.json").read_text())}


def rollout(rows: list[dict], gens: dict) -> dict:
    """Rollout integrity over the formal rows given (their generations): never evidence of reasoning by itself."""
    n = len(rows)
    nat = deg = ctx = usable = 0
    reasoning, answer = [], []
    for r in rows:
        g = gens.get(r["id"]) or {}
        turns = g.get("turns") or []
        nat += all(t.get("natural_termination") for t in turns) and bool(turns)
        deg += any(t.get("degeneration_triggered") for t in turns)
        ctx += any(t.get("context_limit_reached") for t in turns)
        usable += r.get("reason") != "unusable"
        for t in turns:
            o = t.get("output_token_ids") or []
            if THINK_CLOSE in o:
                k = o.index(THINK_CLOSE) + 1
                reasoning.append(k)
                answer.append(len(o) - k)
            else:
                answer.append(len(o))

    def stats(xs):
        return {"mean": round(statistics.fmean(xs), 1), "median": statistics.median(xs)} if xs else None
    return {"n": n, "usable_rate": round(usable / n, 4) if n else None,
            "natural_termination_rate": round(nat / n, 4) if n else None,
            "degeneration_stop_rate": round(deg / n, 4) if n else None,
            "context_limit_rate": round(ctx / n, 4) if n else None,
            "reasoning_tokens": stats(reasoning), "answer_tokens": stats(answer),
            "_": "context-limit hits are right-censored observations; a degeneration stop is the decoder's repetition stop"}


def fluid(ev: dict, items: dict) -> dict:
    rows = [r for r in ev["rows"][FLUID_KEY] if r.get("formal")]
    static = [r for r in rows if not items[r["id"]].get("interactive")]
    inter = [r for r in rows if items[r["id"]].get("interactive")]
    by = defaultdict(list)
    for r in static:
        by[items[r["id"]]["axis"]].append(r)
    return {"static_all": rate(static), "by_axis": {ax: rate(by[ax]) for ax in STATIC_AXES},
            "interactive_exploratory": rate(inter), "rollout_static": rollout(static, ev["generations"]),
            "rollout_interactive": rollout(inter, ev["generations"]),
            "item_correct": {r["id"]: bool(r.get("correct")) for r in static}}


def paired_mean(seeds: list[dict], axis: str | None) -> dict:
    """The two seeds' mean static correctness with a bootstrap 95% interval over items (paired across seeds)."""
    ids = sorted(set.intersection(*[set(s["item_correct"]) for s in seeds]))
    if axis is not None:
        ids = [i for i in ids if seeds[0]["_axis"][i] == axis]
    if not ids:
        return {"n_items": 0}
    vals = [statistics.fmean(s["item_correct"][i] for s in seeds) for i in ids]
    rng = random.Random(BOOT_SEED)
    boots = sorted(statistics.fmean(rng.choice(vals) for _ in vals) for _ in range(BOOT))
    return {"n_items": len(ids), "mean": round(statistics.fmean(vals), 4),
            "bootstrap95": (round(boots[int(0.025 * BOOT)], 4), round(boots[int(0.975 * BOOT) - 1], 4)),
            "seed_difference": round(abs(statistics.fmean(seeds[0]["item_correct"][i] for i in ids)
                                         - statistics.fmean(seeds[1]["item_correct"][i] for i in ids)), 4)
            if len(seeds) == 2 else None}


def training(unit_dir: Path) -> dict:
    log = unit_dir / "train" / "train_log.jsonl"
    if not log.is_file():
        return {"status": "no train log"}
    steps, evals, end = [], [], None
    for e in jsonl(log):
        if e.get("event") == "train_step":
            steps.append(e)
        elif e.get("event") == "eval_result":
            evals.append({k: e.get(k) for k in ("step", "val_loss", "val_ce", "val_kd") if k in e})
        elif e.get("event") == "run_end":
            end = e
    last = steps[-50:]

    def m(k):
        xs = [s[k] for s in last if isinstance(s.get(k), (int, float))]
        return round(statistics.fmean(xs), 5) if xs else None
    return {"optimizer_steps": steps[-1].get("step") if steps else None, "final_50_steps_mean": {"loss": m("loss"), "ce": m("ce"),
            "kd": m("kd")}, "seconds_per_step_median": statistics.median([s["seconds"] for s in steps if s.get("seconds")])
            if steps else None, "validation": evals, "run_end": end}


def analyse(units: dict[str, Path], teacher: Path | None) -> dict:
    items, _eval_only = fluid_meta()
    per = {}
    for name, d in sorted(units.items()):
        ev = evidence(d)
        f = fluid(ev, items)
        f["_axis"] = {i: items[i]["axis"] for i in f["item_correct"]}
        per[name] = {"fluid": f, "capabilities": capabilities(ev["formal_report"]), "training": training(d)}
    seeds = [per[n]["fluid"] for n in sorted(per)]
    combined = {"static_all": paired_mean(seeds, None) if len(seeds) == 2 else None,
                "by_axis": {ax: paired_mean(seeds, ax) for ax in STATIC_AXES} if len(seeds) == 2 else None}
    t = None
    if teacher is not None:
        tf = fluid(evidence(teacher), items)
        t = {"fluid": {k: v for k, v in tf.items() if k != "item_correct"},
             "capabilities": capabilities(evidence(teacher)["formal_report"])}
    gap = {}
    if t:
        for n, u in per.items():
            gap[n] = {"static_all": round(t["fluid"]["static_all"]["rate"] - u["fluid"]["static_all"]["rate"], 4),
                      **{ax: round((t["fluid"]["by_axis"][ax]["rate"] or 0) - (u["fluid"]["by_axis"][ax]["rate"] or 0), 4)
                         for ax in STATIC_AXES}}
    adm = json.loads(ADMISSION.read_text())["arms"]["A1"]
    for u in per.values():
        u["fluid"].pop("_axis")
        u["fluid"]["item_correct_count"] = sum(u["fluid"].pop("item_correct").values())
    return {"schema": "aadistill.stage2.teacher_modes.a1_result/v1",
            "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "set": "eval-set-1 v2 (frozen), scoring protocol v2, validity mask v1: 570 static + 144 interactive formal",
            "decoding": {"protocol_id": "nucleus_v1", "sha256": DECODING_SHA},
            "units": per, "two_seeds": combined, "teacher_thinking_mode": t, "teacher_minus_student": gap,
            "data": {k: adm.get(k) for k in ("sequences", "supervised_tokens", "documents", "tasks", "non_padding_tokens",
                                             "supervised_token_exposure", "non_padding_token_exposure", "training_flops",
                                             "optimizer_steps_at_3_epochs_batch_2")},
            "reading": ("static fluid correctness by axis is the primary reading; interactive episodes are exploratory; "
                        "usable rollout, loss, KL and lengths are never read as reasoning; nothing here compares thinking "
                        "with non-thinking (A2 is not run) or decides Research A"),
            "authorizes": "nothing"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--units", required=True, help="a directory holding A1-s<seed>/ unit directories")
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    units = {d.name: d for d in sorted(Path(a.units).glob("A1-s*")) if d.is_dir()}
    doc = analyse(units, Path(a.teacher) if a.teacher else None)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(json.dumps({n: {"static": u["fluid"]["static_all"], "interactive": u["fluid"]["interactive_exploratory"]}
                      for n, u in doc["units"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
