#!/usr/bin/env python3
"""Isolation and length check of the external generators' validation samples. $0.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.reasoning_data.audit_external_sources \
        --samples <dir holding reasoning_gym.jsonl, reasoning_core.jsonl, prontoqa.jsonl,
                   mathematics_dataset.jsonl>

The samples are 5 instances per module and difficulty, generated at $0 from
pinned installs in an isolated scratch environment (versions and seeds inside
each row). This reads them -- it never imports a third-party generator -- and,
per candidate module of `configs/.../external_modules.json`, reports:

* prompt length in TEACHER tokens (the tokenizer every recovery used), the
  number the data-construction limit actually binds on;
* how often the gold answer appears verbatim in the prompt;
* the generator's own reproducibility across processes, and whether its own
  scorer gave partial credit (both recorded by the sampler);
* isolation rules R1 (held-out keys) and R2 (13-gram near-duplicates) against
  eval-set-1, run through `isolation_check.check_pool`.

Writes `logs/stages/stage-2/reasoning_data/audit/external_source_samples.json`
with the sha256 of every sample file it read.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from stages.reasoning_data import assets as A
from stages.reasoning_data import build_master_ladder as B
from stages.reasoning_data import isolation_check as IC

OUT = A.LOG_DIR / "audit/external_source_samples.json"
FILES = {"reasoning_gym": "reasoning_gym.jsonl", "reasoning_core": "reasoning_core.jsonl",
         "prontoqa": "prontoqa.jsonl", "mathematics_dataset": "mathematics_dataset.jsonl"}


def candidates() -> dict[tuple[str, str], dict]:
    doc = A.read_json(A.CONFIG_DIR / "external_modules.json")
    return {(m["source"], m["module"]): m for m in doc["modules"]
            if str(m.get("decision", "")).startswith("candidate")}


def module_key(source: str, row: dict, cands: dict) -> str | None:
    if source in ("prontoqa", "mathematics_dataset"):
        return next((k[1] for k in cands if k[0] == source
                     and (source == "prontoqa" or row["module"] in k[1])), None)
    return row["module"] if (source, row["module"]) in cands else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", required=True)
    args = ap.parse_args(argv)
    root = Path(args.samples)
    cands = candidates()
    tok = B.load_tokenizer()
    rows_by = defaultdict(list)
    shas = {}
    for source, name in FILES.items():
        path = root / name
        shas[name] = A.sha256_file(path)
        for row in A.jsonl(path):
            key = module_key(source, row, cands)
            if key is not None:
                rows_by[(source, key)].append(row)
    out, pool = {}, []
    for (source, mod), rows in sorted(rows_by.items()):
        tokens = defaultdict(list)
        for r in rows:
            tokens[r.get("difficulty", "?")].append(len(tok(r["prompt"], add_special_tokens=False).input_ids))
            pool.append({"unit_id": f"{source}:{mod}:{r.get('seed')}:{r.get('index')}",
                         "messages": [{"role": "user", "content": r["prompt"]}]})
        def scores(r):
            c = r.get("checks") or {}
            return [v for k in ("wrong_scores", "near_miss_scores")
                    for v in (c.get(k) or {}).values() if isinstance(v, (int, float))]
        partial = sum(1 for r in rows if any(0 < v < 1 for v in scores(r)))
        near_one = sum(1 for r in rows if any(0.8 <= v < 1 for v in scores(r)))
        out[f"{source}/{mod}"] = {
            "n": len(rows),
            "versions": sorted({str(r.get("version") or r.get("commit") or r.get("sha") or "") for r in rows}),
            "prompt_tokens": {d: {"p50": int(statistics.median(v)), "max": max(v)} for d, v in sorted(tokens.items())},
            "gold_in_prompt": sum(1 for r in rows if r.get("gold_in_prompt_word_boundary") or r.get("gold_in_prompt_substring")),
            "reproducible_same_hash_seed": sum(1 for r in rows if r.get("repro_cross_process_same_hashseed0") in (True, "true", 1)),
            "rows_where_the_library_scorer_gave_partial_credit": partial,
            "rows_where_a_wrong_answer_scored_at_least_0.8": near_one,
        }
    iso = IC.check_pool(pool)
    flagged = set(iso["results"]["R2_near_duplicate"]["flagged_ids"])
    doc = {
        "schema": "aadistill.stage2.reasoning_data.external_source_samples/v1",
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sample_files": shas,
        "sampler": "a $0 validation in an isolated scratch environment (reasoning-gym 0.1.25, reasoning-core 0.5.0, PrOntoQA 0a6412b6, mathematics_dataset 1.0.1 under sympy 1.5.1 / numpy 1.23.5); the rows carry their own version, seed and difficulty",
        "tokenizer": str(A.TOKENIZER_DIR.relative_to(A.REPO)),
        "_math_samples": "the mathematics_dataset samples are the single-step variants of the proposed composed modules (a length and isolation proxy)",
        "modules": out,
        "isolation": {"pass": iso["pass"], "violations": iso["violations"],
                      "R1": iso["results"]["R1_held_out_keys"],
                      "R2": {k: iso["results"]["R2_near_duplicate"][k] for k in ("checked", "flagged", "flagged_by_unit")},
                      "flagged_modules": sorted({f.split(":")[0] + "/" + f.split(":")[1] for f in flagged})},
        "authorizes": "nothing",
    }
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(OUT.relative_to(A.REPO), "isolation", iso["pass"], iso["violations"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
