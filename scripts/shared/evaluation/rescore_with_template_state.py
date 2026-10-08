#!/usr/bin/env python
"""Rescore saved generations under the corrected, template-aware scorer.

    PYTHONPATH=src python scripts/evaluation/rescore_with_template_state.py \
        --battery artifacts/eval/battery_v2 \
        --generations artifacts/eval/e2diag/ref_qwen3_0p6b_project \
        --label ref_qwen3_0p6b_project --think-preopened false \
        --out artifacts/eval/e2diag_rescored_v2/ref_qwen3_0p6b_project_battery.json

CPU only — generation is the paid part and it is already done. Nothing is
overwritten: the original scorings stay exactly where they were, and this writes
a **versioned derived artifact** next to them recording both the scorer version
and the template state it assumed, so the two can always be told apart.

`--think-preopened auto` re-renders one prompt with the model's own tokenizer and
reads the answer off the rendered text, which is where the fact actually lives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.evaluation.behavior import template_opens_think  # noqa: E402
from aadistill.infrastructure.env import code_state, library_versions  # noqa: E402
from aadistill.infrastructure.manifest import sha256_file  # noqa: E402

SCORER = REPO_ROOT / "scripts/evaluation/score_battery.py"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def resolve_state(mode: str, model: str | None, revision: str | None,
                  gen_dir: Path) -> tuple[bool, str]:
    if mode in ("true", "false"):
        return mode == "true", f"explicit:{mode}"
    # 1. the records may already carry it (anything written after 2026-08-04)
    for f in sorted(gen_dir.glob("*.generations.jsonl")):
        for line in f.open():
            rec = json.loads(line)
            if "think_preopened" in rec:
                return bool(rec["think_preopened"]), "record"
            break
        break
    # 2. otherwise ask the model's own template
    if not model:
        raise SystemExit("--think-preopened auto needs --model when the saved "
                         "records predate the field")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model, revision=revision)
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "x"}], tokenize=False,
        add_generation_prompt=True)
    return template_opens_think(rendered), f"template:{model}@{revision or 'main'}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", required=True, type=Path)
    ap.add_argument("--generations", required=True, type=Path)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--think-preopened", default="auto",
                    choices=("auto", "true", "false"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--original", type=Path, default=None,
                    help="the pre-fix scoring, reported side by side")
    args = ap.parse_args()

    preopened, source = resolve_state(args.think_preopened, args.model,
                                      args.revision, args.generations)
    print(f"think_preopened={preopened}  (from {source})")

    # Inject the state into a COPY of the generations. The originals are inputs
    # to a completed experiment and are never rewritten.
    staged = args.out.parent / f"_staged_{args.label}"
    staged.mkdir(parents=True, exist_ok=True)
    input_hashes = {}
    for f in sorted(args.generations.glob("*.generations.jsonl")):
        rows = []
        for line in f.open():
            rec = json.loads(line)
            rec["think_preopened"] = preopened
            rows.append(json.dumps(rec))
        (staged / f.name).write_text("\n".join(rows) + "\n")
        input_hashes[f.name] = sha256_file(f)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(SCORER), "--battery", str(args.battery),
           "--generations", str(staged), "--label", args.label,
           "--out", str(args.out),
           "--per-sample", str(args.out.with_suffix(".per_sample.jsonl"))]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT,
                   env={**__import__("os").environ,
                        "PYTHONPATH": str(REPO_ROOT / "src")})

    scored = json.loads(args.out.read_text())
    side = {}
    if args.original and args.original.is_file():
        orig = json.loads(args.original.read_text())["results"]
        for name, new in scored["results"].items():
            old = orig.get(name, {})
            side[name] = {
                "correct": {"before": old.get("pair_correct", old.get("correct")),
                            "after": new.get("pair_correct", new.get("correct"))},
                "protocol_valid_rate": {"before": old.get("protocol_valid_rate"),
                                        "after": new.get("protocol_valid_rate")},
                "answer_matches_ignoring_protocol": {
                    "before": old.get("answer_matches_ignoring_protocol"),
                    "after": new.get("answer_matches_ignoring_protocol")},
            }

    meta_path = args.out.with_name(args.out.stem + ".rescore_meta.json")
    meta_path.write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "label": args.label,
        "think_preopened": preopened,
        "think_preopened_source": source,
        "scorer_version": "template-aware (2026-08-04)",
        "generations_dir": str(args.generations),
        "input_generation_sha256": input_hashes,
        "output_sha256": sha256_file(args.out),
        "original_scoring": str(args.original) if args.original else None,
        "original_sha256": (sha256_file(args.original)
                            if args.original and args.original.is_file() else None),
        "before_after": side,
        "libraries": library_versions(),
        "code_state": code_state(REPO_ROOT),
        "note": ("Original scorings are preserved unmodified. Generation was not "
                 "re-run; only scoring changed."),
    }, indent=1))

    print(f"\n{'set':22s} {'correct':>16s} {'protocol_valid':>20s}")
    for name, d in side.items():
        c, p = d["correct"], d["protocol_valid_rate"]
        print(f"{name:22s} {str(c['before']):>7s} -> {str(c['after']):<6s} "
              f"{str(p['before']):>9s} -> {str(p['after']):<9s}")
    print(f"\nwrote {args.out}\n      {meta_path}")


if __name__ == "__main__":
    main()
