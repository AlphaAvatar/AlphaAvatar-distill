#!/usr/bin/env python3
"""`tool_pool@v1`'s assembly: families, splits, isolation, the length bound and the frozen order. $0 code.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.tool_pool \\
        --source toucan=<converted tasks.jsonl> [--source <name>=<converted tasks.jsonl> ...] \\
        --out data/stages/stage-2/teacher_modes/tool_pool_v1

The rules are six_category_design.json's (`task_pool`). Every source is a CANDIDATE until the maintainer admits it
(audit/tool_calling_source_review_v2.json). A source enters as CONVERTED task records (`tool_episodes.task_problems`
must pass: upstream assistant turns are never in the context, the backend is a replay table or none); each source's
converter is its own module, registered in `CONVERTERS` once it exists.

* family -- sha256 of the canonical tool set (sorted names with their canonical JSON schemas); an MCP task may name
  its server instead (``family_hint``);
* split -- by family, never by task: ``int(sha256('tool_pool@v1|' + family), 16) mod 100``: 0-2 validation,
  3-6 state_eval, 7-10 calibration, 11-12 reserve, 13-99 training -- so no evaluation, calibration or state-evaluation
  task shares a tool family with a training task (T4);
* T5 -- an exact or normalized-query duplicate (same prompt key and tool family) collapses to its first task in
  (source, task id) order;
* T1/T2 -- a task sharing a prompt key with a held-out item of eval-set-1 or of a named benchmark, or a tool family
  with eval-set-1's tool items, is excluded (`shared.eval_sets` spelling of the prompt key);
* length -- a task whose first rendered prompt exceeds ``PROMPT_BOUND`` tokens under any teacher's template is
  excluded for every teacher;
* order -- the training split's canonical order is a seeded shuffle (``ORDER_SEED``) of task ids, frozen with its
  sha256 before any generation; every teacher walks the same order.

Every exclusion is listed with its reason; nothing is silently dropped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.teacher_modes import tool_episodes as TE  # noqa: E402

POOL_ID = "tool_pool@v1"
ORDER_SEED = 20261103
PROMPT_BOUND = 6144
SPLIT_RANGES = (("validation", 0, 2), ("state_eval", 3, 6), ("calibration", 7, 10), ("reserve", 11, 12),
                ("training", 13, 99))
#: source id -> converter(path) -> iterable of task records, each written and checked against its source's real rows:
#: Toucan's SFT config (tool_sources_toucan.py); the Pivot pair and Nemotron-SFT-Agentic-v4 are not converted yet
#: (audit/tool_calling_source_review_v2.json). Every source remains a CANDIDATE until the maintainer admits it.
def _toucan(path: Path) -> Iterable[dict]:
    from stages.teacher_modes import tool_sources_toucan
    return tool_sources_toucan.convert(path)


CONVERTERS: dict[str, Callable[[Path], Iterable[dict]]] = {"toucan_sft": _toucan}
SCHEMA = "aadistill.stage2.teacher_modes.tool_pool/v1"


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def family_of(tools: Sequence[Mapping], hint: str | None = None) -> str:
    if hint:
        return "server:" + str(hint)
    sig = sorted((str(t.get("function", t)["name"]), canonical(t.get("function", t).get("parameters") or {}))
                 for t in tools)
    return hashlib.sha256(canonical(sig).encode()).hexdigest()[:20]


def split_of(family: str) -> str:
    v = int(hashlib.sha256(f"{POOL_ID}|{family}".encode()).hexdigest(), 16) % 100
    for name, lo, hi in SPLIT_RANGES:
        if lo <= v <= hi:
            return name
    raise AssertionError(v)


def prompt_key(task: Mapping) -> str:
    from shared.eval_sets import prompt_key as pk, prompt_text
    return pk(prompt_text(task))


def assemble(tasks: Iterable[dict], *, held_out_prompt_keys: set[str] = frozenset(),
             held_out_families: set[str] = frozenset(),
             prompt_lengths: Sequence[Callable[[Mapping], int]] = ()) -> dict:
    """The pool: admitted tasks with family and split set, the frozen training order, every exclusion."""
    seen, admitted, excluded = {}, [], []
    for t in sorted(tasks, key=lambda t: (str(t["source"].get("dataset")), t["task_id"])):
        t = {**t, "family": family_of(t["tools"], t.get("family_hint"))}
        t["split"] = split_of(t["family"])
        bad = TE.task_problems(t)
        if bad:
            excluded.append({"task_id": t["task_id"], "reason": "record: " + "; ".join(bad)})
            continue
        key = prompt_key(t)
        if key in held_out_prompt_keys:
            excluded.append({"task_id": t["task_id"], "reason": "T1/T2: a held-out prompt"})
            continue
        if t["family"] in held_out_families:
            excluded.append({"task_id": t["task_id"], "reason": "T1: a held-out tool family"})
            continue
        dup = seen.get((key, t["family"]))
        if dup is not None:
            excluded.append({"task_id": t["task_id"], "reason": f"T5: duplicates {dup}"})
            continue
        over = [n for n in (f(t) for f in prompt_lengths) if n > PROMPT_BOUND]
        if over:
            excluded.append({"task_id": t["task_id"], "reason": f"length: first prompt {max(over)} > {PROMPT_BOUND}"})
            continue
        seen[(key, t["family"])] = t["task_id"]
        admitted.append({**t, "content_keys": {"prompt": key, "tools": t["family"]}})
    from aadistill.data.sequence_packing import shuffled_order
    train = [t for t in admitted if t["split"] == "training"]
    order = [train[i]["task_id"] for i in shuffled_order([t["task_id"] for t in train], ORDER_SEED,
                                                         namespace=POOL_ID)]
    census = defaultdict(Counter)
    for t in admitted:
        census[t["split"]][t["category"]] += 1
    families = defaultdict(set)
    for t in admitted:
        families[t["family"]].add(t["split"])
    crossing = sorted(f for f, s in families.items() if len(s) > 1)
    if crossing:
        raise AssertionError(f"tool families in more than one split: {crossing[:3]}")
    return {"tasks": admitted, "excluded": excluded, "training_order": order,
            "training_order_sha256": hashlib.sha256(canonical(order).encode()).hexdigest(),
            "census": {s: dict(c) for s, c in sorted(census.items())},
            "by_source": dict(Counter(str(t["source"].get("dataset")) for t in admitted))}


def held_out(repo: Path = REPO) -> tuple[set[str], set[str]]:
    """eval-set-1's held-out prompt keys, and the tool families of its items that carry tools."""
    from shared import eval_sets as ES
    keys, fams = set(), set()
    for unit, rows in ES.component_items(roles=ES.HELD_OUT_ROLES, repo_root=repo).items():
        for r in rows:
            k = ES.KEYS["prompt"](r)
            if k:
                keys.add(k)
            if r.get("tools"):
                try:
                    fams.add(family_of(r["tools"]))
                except (KeyError, TypeError, AttributeError):
                    continue
    return keys, fams


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", action="append", default=[], help="name=converted tasks.jsonl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    tasks = []
    for spec in a.source:
        name, _, path = spec.partition("=")
        tasks += TE.read_tasks(path)
    keys, fams = held_out()
    pool = assemble(tasks, held_out_prompt_keys=keys, held_out_families=fams)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for split in {t["split"] for t in pool["tasks"]}:
        with (out / f"{split}.tasks.jsonl").open("w") as f:
            for t in pool["tasks"]:
                if t["split"] == split:
                    f.write(json.dumps(t, ensure_ascii=False) + "\n")
    (out / "pool.json").write_text(json.dumps({
        "schema": SCHEMA, "pool_id": POOL_ID, "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "census": pool["census"], "by_source": pool["by_source"], "excluded": pool["excluded"],
        "training_order": pool["training_order"], "training_order_sha256": pool["training_order_sha256"],
        "order_seed": ORDER_SEED, "prompt_bound": PROMPT_BOUND, "authorizes": "nothing"}, indent=1) + "\n")
    print(json.dumps({"census": pool["census"], "excluded": len(pool["excluded"])}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
