#!/usr/bin/env python3
"""Research A token-matched redesign: what the sealed teacher records already supply, in supervised tokens. $0.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.supply_audit \\
        --records <scr>/store/generation --identity <study identity> \\
        --rows artifacts/stages/stage-2/teacher_modes/audit/supply_rows.jsonl \\
        --write logs/stages/stage-2/teacher_modes/audit/supply_audit.json

Every (task, mode) the teacher sessions generated goes through admission.py's own verdict (`admit`: verified
correct, naturally terminated, protocol-valid, token-native documents) and its isolation recheck (R1, R2,
R9 against eval-set-1, then every rule over the rest). Nothing here re-decides admission; it counts what the
frozen verdict admits and why it refuses the rest, in tasks, documents, supervised tokens (the generated
tokens a document trains on) and context tokens (prompt + generation: the non-padding positions of an
isolated pack).

Rows (one per (task, mode), no text) go to --rows; the aggregate by mode x category x stratum x source x
family x slice goes to --write. Category is `fluid` for the P10.2 primary strata and `supporting` for the
rest: the strata Research A registered, named for what they are, never "knowledge".
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.reasoning_data import assets as A  # noqa: E402
from stages.reasoning_data import source_plan as SP  # noqa: E402
from stages.teacher_modes import admission as ADM  # noqa: E402

SCHEMA = "aadistill.stage2.teacher_modes.supply_audit/v1"


def reason_class(reasons: list[str]) -> str:
    """One refusal class per candidate, in the order the contract checks them."""
    r = " | ".join(reasons)
    for key, name in (("length_limited", "training-length-limited"), ("context_limit", "training-length-limited"),
                      ("degenerat", "degenerated"), ("repetit", "degenerated"),
                      ("document:", "not a token-native document"), ("mode's generation prompt", "prompt mode"),
                      ("protocol:", "protocol invalid"),
                      ("illegal_action", "illegal action"), ("answer:unverifiable", "unverifiable"),
                      ("answer:incorrect", "incorrect answer"), ("steps", "steps contradicted")):
        if key in r:
            return name
    return "other: " + r[:80] if r else "other"


def slice_of(stage: str | None, tid: str, t0: set[str]) -> str:
    if tid in t0:
        return "T0"
    if stage is None:
        return "?"
    return {"w00": "V", "w": "E", "x": "X"}["w00" if stage == "w00" else stage[0]]


def rows(store: Path, identity: str, repo: Path = ADM.REPO) -> tuple[list[dict], dict]:
    proto = ADM.protocol(repo)
    tasks, t0, order, stage = ADM.tasks_and_stages(proto, repo)
    recs = ADM.records(store, identity)
    verdicts = {k: ADM.admit(tasks[k[0]], k[1], v, proto) for k, v in recs.items()}
    flagged, iso = ADM.isolation_recheck({k: v for k, v in verdicts.items() if v["admitted"]}, tasks)
    spec = A.read_json(A.CONFIG_DIR / "corpus_spec.json")
    t0s = set(t0)
    out = []
    for (tid, mode), v in sorted(verdicts.items()):
        t = tasks[tid]
        stratum = t.get("stratum") or SP.stratum_of(t, spec)
        fam = f"{t['generator_id']}/{t.get('family_id')}" if t.get("generator_id") else (t.get("family_id") or t.get("source"))
        cands = v["candidates"]
        first = cands[0] if cands else {}
        admitted = bool(v["admitted"]) and (tid, mode) not in flagged
        docs = v["docs"] if admitted else []
        refusal = None
        if not admitted:
            refusal = ("isolation recheck" if (tid, mode) in flagged else
                       reason_class(cands[-1]["reasons"]) if cands else "no record")
        out.append({
            "task_id": tid, "mode": mode, "slice": slice_of(stage.get(tid), tid, t0s), "stage": stage.get(tid),
            "category": "fluid" if stratum.startswith("primary.") else "supporting", "stratum": stratum,
            "source": t.get("source"), "family": fam, "interactive": bool(t.get("interactive")) and
            str(t.get("interactive")) not in ("False", "0"),
            "candidates": len(cands), "generated_tokens": sum(c["generated_tokens"] for c in cands),
            "first": {"naturally_terminated": not first.get("length_limited") and
                      not any("degenerat" in x or "repetit" in x for x in first.get("reasons", [])),
                      "length_limited": bool(first.get("length_limited")), "correct": bool(first.get("correct")),
                      "accepted": bool(first.get("accepted"))},
            "admitted": admitted, "admitted_candidate": v.get("candidate") if admitted else None,
            "refusal": refusal, "documents": len(docs),
            "supervised_tokens": sum(len(d.output_ids) for d in docs),
            "context_tokens": sum(len(d) for d in docs),
            "doc_tokens": [len(d) for d in docs]})
    return out, {"isolation": iso, "n_flagged": len(flagged), "order_rows": len(order),
                 "order_generated": sum(1 for t in order if any((t["task_id"], m) in recs for m in ADM.MODES))}


def aggregate(rs: list[dict]) -> dict:
    def acc():
        return {"tasks": 0, "admitted_tasks": 0, "documents": 0, "supervised_tokens": 0, "context_tokens": 0,
                "generated_tokens": 0, "first_naturally_terminated": 0, "first_length_limited": 0,
                "first_correct": 0}

    def add(a, r):
        a["tasks"] += 1
        a["admitted_tasks"] += r["admitted"]
        a["documents"] += r["documents"]
        a["supervised_tokens"] += r["supervised_tokens"]
        a["context_tokens"] += r["context_tokens"]
        a["generated_tokens"] += r["generated_tokens"]
        a["first_naturally_terminated"] += r["first"]["naturally_terminated"]
        a["first_length_limited"] += r["first"]["length_limited"]
        a["first_correct"] += r["first"]["correct"]

    views = {"mode": lambda r: r["mode"], "mode/category": lambda r: f"{r['mode']}/{r['category']}",
             "mode/slice": lambda r: f"{r['mode']}/{r['slice']}",
             "mode/category/slice": lambda r: f"{r['mode']}/{r['category']}/{r['slice']}",
             "mode/stratum": lambda r: f"{r['mode']}/{r['stratum']}",
             "mode/stratum/source": lambda r: f"{r['mode']}/{r['stratum']}/{r['source']}",
             "mode/stratum/source/slice": lambda r: f"{r['mode']}/{r['stratum']}/{r['source']}/{r['slice']}",
             "mode/family": lambda r: f"{r['mode']}/{r['stratum']}/{r['family']}"}
    out = {}
    for name, key in views.items():
        g = defaultdict(acc)
        for r in rs:
            add(g[key(r)], r)
        out[name] = dict(sorted(g.items()))
    refusals = defaultdict(Counter)
    for r in rs:
        if not r["admitted"]:
            refusals[f"{r['mode']}/{r['category']}"][r["refusal"]] += 1
    out["refusals"] = {k: dict(v.most_common()) for k, v in sorted(refusals.items())}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", required=True)
    ap.add_argument("--identity", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--write", required=True)
    a = ap.parse_args(argv)
    rs, meta = rows(Path(a.records), a.identity)
    Path(a.rows).parent.mkdir(parents=True, exist_ok=True)
    with open(a.rows, "w") as fh:
        for r in rs:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")
    rec = {"schema": SCHEMA, "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "records": {"dir": a.records, "identity": a.identity}, "rows": {"path": a.rows, "sha256": A.sha256_file(Path(a.rows)),
                                                                         "n": len(rs)},
           **meta, "aggregate": aggregate(rs), "authorizes": "nothing"}
    Path(a.write).write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec["aggregate"]["mode/category"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
