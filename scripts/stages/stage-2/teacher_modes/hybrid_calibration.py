#!/usr/bin/env python3
"""The common initialization's calibration: the hybrid's own trajectories on D4 Top-1's calibration prompts.

    # $0, dev box: the run_battery inputs (one file per system message), from the profiles' token ids
    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.calibration inputs
    # on the pod, after both modes' calibration groups are generated: the mode-balanced items
    #   calibration.items_from_records(...)  (stages.teacher_modes.teacher_session calls it)

D4 Top-1's fixed path read two Stage-0 profiles: ``calib.s0v_rh_d4@v1`` (75
items) for ATTENTION and ``calib.s0v_base_d4@v1`` (85 items) for DEPTH,
RESIDUAL_WIDTH and FFN. Their items are Thinking-2507 sessions as token ids.
Each item's PROMPT is recovered from its own ids -- the ids before its first
assistant target, decoded and split at the ChatML boundaries into its system
and user messages -- and the hybrid generates one trajectory per prompt in
each mode (the official hard switch, each mode's preset, the same seed).

An item of the hybrid profile is one naturally terminated trajectory exactly
as generated: ``ids`` = prompt_token_ids + output_token_ids, ``tags.assistant``
the positions that predict the generated tokens (position i predicts token
i+1), the source item's ``domain``, and ``subtype`` = the source subtype and
the mode, so DEPTH's subtype-then-domain average weighs the two modes equally
inside every domain. MODE BALANCE (d4_scale_design.json :: initialization):
equal scored positions from each mode, by whole items -- every item of the
mode with fewer scored positions, then the other mode's items in a seeded
order until its positions reach that total. Nothing is cut. Correctness is not
an admission criterion here: calibration measures the activations of the
teacher's own distribution, and a trajectory that did not terminate naturally
is not an item.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

PROFILES = {
    "calib.s0v_rh_d4@v1": "artifacts/stages/stage-0/calibration_volume/profiles_formal_v1/calib.s0v_rh_d4.items.jsonl",
    "calib.s0v_base_d4@v1": "artifacts/stages/stage-0/calibration_volume/profiles_formal_v1/calib.s0v_base_d4.items.jsonl",
}
#: the hybrid profiles the fixed path reads in place of each Stage-0 one
HYBRID = {"calib.s0v_rh_d4@v1": "calib.hybrid_modes_rh_d4@v1", "calib.s0v_base_d4@v1": "calib.hybrid_modes_base_d4@v1"}
SOURCE_TOKENIZER = ("Qwen/Qwen3-4B-Thinking-2507", "768f209d9ea81521153ed38c47d515654e938aea")
INPUTS_DIR = REPO / "logs/stages/stage-2/teacher_modes/plans/calibration_inputs"
BALANCE_SEED = 20261029
_BLOCK = re.compile(r"<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>\n?", re.S)


def items(profile: str) -> list[dict]:
    p = REPO / PROFILES[profile]
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def prompt_messages(item: dict, tokenizer) -> list[dict]:
    """The item's system and user messages, from its ids before the first assistant target."""
    tags = item["tags"]["assistant"]
    first = min(tags) if tags and isinstance(tags[0], int) else next(i for i, x in enumerate(tags) if x)
    text = tokenizer.decode(item["ids"][:first + 1], skip_special_tokens=False)
    head = text.rsplit("<|im_start|>assistant", 1)[0]
    msgs = [{"role": r, "content": c} for r, c in _BLOCK.findall(head)]
    if not msgs or msgs[-1]["role"] != "user" or any(m["role"] == "assistant" for m in msgs):
        raise ValueError(f"{item['item_id']}: not a single-turn system/user prompt")
    return msgs


def task_id(profile: str, item: dict) -> str:
    return "calib_" + hashlib.sha256(f"{profile}|{item['item_id']}".encode()).hexdigest()[:20]


def inputs(tokenizer_dir) -> list[dict]:
    """run_battery's task contract for every profile item, grouped by system message; written to INPUTS_DIR."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    by_system: dict[str, list[dict]] = defaultdict(list)
    for prof in PROFILES:
        for it in items(prof):
            msgs = prompt_messages(it, tok)
            system = msgs[0]["content"] if msgs[0]["role"] == "system" else "You are a helpful Assistant."
            by_system[system].append({
                "task_id": task_id(prof, it), "messages": [m for m in msgs if m["role"] != "system"],
                "generator_id": it.get("generator_id") or it["source"], "family_id": it.get("family_id") or it["source"],
                "axis": it["domain"], "difficulty": str(it.get("difficulty") or ""), "interactive": False,
                "calibration": {"profile": prof, "item_id": it["item_id"], "domain": it["domain"],
                                "subtype": it["subtype"]}})
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    files = []
    for k, (system, rows) in enumerate(sorted(by_system.items())):
        path = INPUTS_DIR / f"calibration_input_g{k}.jsonl"
        path.write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows))
        files.append({"file": path.name, "system": system, "tasks": len(rows),
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (INPUTS_DIR / "manifest.json").write_text(json.dumps(
        {"schema": "aadistill.stage2.teacher_modes.calibration_inputs/v1", "profiles": PROFILES,
         "profiles_sha256": {p: hashlib.sha256((REPO / f).read_bytes()).hexdigest() for p, f in PROFILES.items()},
         "source_tokenizer": {"model": SOURCE_TOKENIZER[0], "revision": SOURCE_TOKENIZER[1]},
         "files": files, "authorizes": "nothing"}, indent=1) + "\n")
    return files


def items_from_records(records: dict[str, list[dict]], tasks: dict[str, dict], stop_ids, *,
                       seed: int = BALANCE_SEED) -> dict:
    """Per hybrid profile: the mode-balanced items and their accounting.

    ``records`` maps a mode to its generation records (run_battery's episode
    records); ``tasks`` the calibration inputs by task id, which carry each
    prompt's source item (the record does not)."""
    stop_ids = {int(x) for x in stop_ids}
    out = {}
    for prof, hybrid in HYBRID.items():
        per_mode: dict[str, list[dict]] = {}
        dropped = {}
        for mode, recs in sorted(records.items()):
            rows, bad = [], 0
            for r in recs:
                cal = (tasks.get(r["task_id"]) or {}).get("calibration") or {}
                if cal.get("profile") != prof:
                    continue
                turn = r["turns"][0]
                p, o = turn.get("prompt_token_ids") or [], turn.get("output_token_ids") or []
                if not p or not o or int(o[-1]) not in stop_ids:
                    bad += 1
                    continue
                ids = list(p) + list(o)
                rows.append({"item_id": f"{cal['item_id']}|{mode}", "ids": ids, "n_tokens": len(ids),
                             "n_prediction_positions": len(ids) - 1,
                             "tags": {"assistant": list(range(len(p) - 1, len(ids) - 1))},
                             "domain": cal["domain"], "subtype": f"{cal['subtype']}|{mode}", "mode": mode,
                             "source": "hybrid_teacher_native", "source_item": cal["item_id"], "turn": 0})
            per_mode[mode], dropped[mode] = rows, bad
        scored = {m: sum(len(r["tags"]["assistant"]) for r in rows) for m, rows in per_mode.items()}
        small = min(scored, key=scored.get)
        chosen = list(per_mode[small])
        for m, rows in per_mode.items():
            if m == small:
                continue
            order = list(rows)
            random.Random(seed).shuffle(order)
            got = 0
            for r in order:
                if got >= scored[small]:
                    break
                chosen.append(r)
                got += len(r["tags"]["assistant"])
        out[hybrid] = {"source_profile": prof, "items": sorted(chosen, key=lambda r: r["item_id"]),
                       "scored_positions_available": scored, "not_naturally_terminated": dropped,
                       "scored_positions_used": {m: sum(len(r["tags"]["assistant"]) for r in chosen if r["mode"] == m)
                                                 for m in per_mode},
                       "items_used": {m: sum(r["mode"] == m for r in chosen) for m in per_mode},
                       "balance": f"every {small} item; the other mode's whole items in random.Random({seed}) order "
                                  "until its scored positions reach that total"}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("inputs")
    i.add_argument("--tokenizer", default=None, help="the Thinking-2507 tokenizer directory (default: the HF cache)")
    a = ap.parse_args(argv)
    from huggingface_hub.constants import HF_HUB_CACHE
    tok = a.tokenizer or (Path(HF_HUB_CACHE) / f"models--{SOURCE_TOKENIZER[0].replace('/', '--')}" / "snapshots"
                          / SOURCE_TOKENIZER[1])
    print(json.dumps(inputs(tok), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
