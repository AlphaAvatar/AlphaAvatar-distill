#!/usr/bin/env python3
"""`tool_pool@v1`'s assembly: atomic isolation, decontamination, the length bound and the frozen order. $0 code.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.tool_pool \\
        --source toucan=<converted tasks.jsonl> [--source <name>=<converted tasks.jsonl> ...] \\
        --out data/stages/stage-2/teacher_modes/tool_pool_v1

Every source is a CANDIDATE until the maintainer admits it (audit/tool_calling_source_review_v2.json). A source
enters as CONVERTED task records (`tool_episodes.task_problems` must pass: upstream assistant turns are never in the
context, the backend is a replay table or none); each source's converter is its own module, registered in
`CONVERTERS` once it exists.

* atoms -- a task's atomic tool identities: ``server:<slug>`` for every server its converter declares
  (``servers``: an MCP server owns every tool named ``<slug>-<tool>``), and ``tool:<identity>`` for every offered
  tool no declared server owns, its identity the sha256 of its bare name and canonical parameter schema;
* split -- per ATOM, never per task: ``int(sha256('tool_pool@v1|' + atom), 16) mod 100``: 0-2 validation,
  3-6 state_eval, 7-10 calibration, 11-12 reserve, 13-99 training. A task enters the split all its atoms share;
  a task whose atoms fall in different splits is refused, so no server or tool is ever in two splits (T4). A
  connected grouping is not an option: Toucan's multi-server tasks chain 99.8% of its tasks into one component;
* family -- the task's environment, sha256 of its sorted atoms (the T5 key and the documents' ``family``);
* T1 (eval-set-1) -- a task sharing a prompt key with a held-out item, a 13-gram near-duplicate of one (containment
  >= 0.5 against each held-out unit's index without its template n-grams, isolation_rules.json's protocol), or
  offering a tool whose bare name or identity is one of eval-set-1's tool items' is refused;
* T2 (public benchmarks, `tool_benchmarks`) -- the same prompt-key and 13-gram screens against BFCL v3, When2Call,
  API-Bank, tau-bench and ToolSandbox, and a tool IDENTITY match; a bare-name-only match with a benchmark tool is
  counted, not refused (names such as ``search`` recur across unrelated APIs);
* T4 (between splits) -- a task that is a 13-gram near-duplicate of a task of a split earlier in
  ``state_eval, calibration, validation, reserve, training`` is refused, so training never holds a reworded copy
  of an evaluation, calibration or state-evaluation task;
* T5 -- an exact or normalized-query duplicate (same prompt key and family) collapses to its first task in
  (source, task id) order;
* replay -- a task whose converter recorded a call with conflicting responses (``backend.conflicts``) is refused;
* length -- a task whose first rendered prompt exceeds ``PROMPT_BOUND`` tokens under ANY lineage and mode of
  tool_generation_protocol.json (each teacher's own tokenizer and official template, the tools rendered by it, the
  system text the session uses) is refused for every teacher (`length_checkers`);
* order -- the training split's canonical order is a seeded shuffle (``ORDER_SEED``) of task ids, frozen with its
  sha256 before any generation; every teacher walks the same order.

Every exclusion is listed with its reason and counted by reason, source, category and split; nothing is silently
dropped. The census registers every admitted task's verification levels (`tool_episodes.verification_levels`).
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
#: T4's priority: a task near-duplicating a task of an earlier split is refused
SPLIT_PRIORITY = ("state_eval", "calibration", "validation", "reserve", "training")
#: the repository's near-duplicate protocol (configs/stages/stage-2/reasoning_data/isolation_rules.json)
NGRAM, CONTAINMENT, MAX_DF = 13, 0.5, 5
PROTOCOL = REPO / "configs/stages/stage-2/teacher_modes/tool_generation_protocol.json"


#: source id -> converter(path) -> iterable of task records, each written and checked against its source's real rows:
#: Toucan's SFT config (tool_sources_toucan.py); the Pivot pair and Nemotron-SFT-Agentic-v4 are not converted yet
#: (audit/tool_calling_source_review_v2.json). Every source remains a CANDIDATE until the maintainer admits it.
def _toucan(path: Path) -> Iterable[dict]:
    from stages.teacher_modes import tool_sources_toucan
    return tool_sources_toucan.convert(path)


CONVERTERS: dict[str, Callable[[Path], Iterable[dict]]] = {"toucan_sft": _toucan}
SCHEMA = "aadistill.stage2.teacher_modes.tool_pool/v2"


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _fn(t: Mapping) -> Mapping:
    return t.get("function", t)


def tool_identity(bare_name: str, parameters) -> str:
    return hashlib.sha256(canonical([str(bare_name), parameters or {}]).encode()).hexdigest()[:20]


def bare_name(name: str, servers: Sequence[str]) -> str:
    """The tool's name without its declared server's ``<slug>-`` prefix (the longest that matches)."""
    s = max((x for x in servers if name.startswith(x + "-")), key=len, default=None)
    return name[len(s) + 1:] if s else name


def tool_identities(task: Mapping) -> list[tuple[str, str]]:
    """``(bare name, identity)`` of every offered tool."""
    servers = list(task.get("servers") or [])
    out = []
    for t in task.get("tools") or []:
        f = _fn(t)
        b = bare_name(str(f["name"]), servers)
        out.append((b, tool_identity(b, f.get("parameters"))))
    return out


def atoms_of(task: Mapping) -> list[str]:
    """The task's atomic identities: its declared servers, and every offered tool no declared server owns."""
    servers = sorted(set(task.get("servers") or []))
    atoms = {f"server:{s}" for s in servers}
    for t in task.get("tools") or []:
        f = _fn(t)
        name = str(f["name"])
        if not any(name.startswith(s + "-") for s in servers):
            atoms.add("tool:" + tool_identity(name, f.get("parameters")))
    return sorted(atoms)


def family_of(atoms: Sequence[str]) -> str:
    return hashlib.sha256(canonical(sorted(atoms)).encode()).hexdigest()[:20] if atoms else "no-tools"


def split_of(atom: str) -> str:
    v = int(hashlib.sha256(f"{POOL_ID}|{atom}".encode()).hexdigest(), 16) % 100
    for name, lo, hi in SPLIT_RANGES:
        if lo <= v <= hi:
            return name
    raise AssertionError(v)


def task_split(atoms: Sequence[str]) -> str | None:
    """The split every atom shares; None when they disagree. A task offering no tool has the atom ``no-tools``
    (which hashes to training)."""
    splits = {split_of(a) for a in atoms} or {split_of("no-tools")}
    return splits.pop() if len(splits) == 1 else None


def prompt_key(task: Mapping) -> str:
    from shared.eval_sets import prompt_key as pk, prompt_text
    return pk(prompt_text(task))


def user_text(task: Mapping) -> str:
    """Every user turn of the task (the text the 13-gram screens read)."""
    return "\n".join([str(m.get("content") or "") for m in task.get("messages") or [] if m.get("role") == "user"]
                     + [str(x) for x in task.get("later_user_turns") or []])


class Reference:
    """What a pool is isolated from: per named set, its prompt keys, a 13-gram index, tool identities and names."""

    def __init__(self):
        self.sets: dict[str, dict] = {}

    def add(self, name: str, *, prompts: Iterable[str] = (), prompt_units: Mapping[str, Sequence[str]] | None = None,
            tools: Iterable[tuple[str, object]] = (), refuse_tool_names: bool = False) -> Reference:
        from aadistill.evaluation.isolation import ngram_index
        from shared.eval_sets import prompt_key as pk
        units = dict(prompt_units or {})
        if prompts:
            units.setdefault(name, list(prompts))
        keys = {pk(t) for texts in units.values() for t in texts if t}
        index = set()
        for texts in units.values():
            index |= ngram_index([t for t in texts if t], NGRAM, max_document_frequency=MAX_DF)
        tools = list(tools)
        self.sets[name] = {"keys": keys - {None, ""}, "index": index,
                           "tool_ids": {tool_identity(n, p) for n, p in tools}, "tool_names": {n for n, _ in tools},
                           "refuse_tool_names": refuse_tool_names, "prompts": sum(map(len, units.values())),
                           "tools": len(tools)}
        return self

    def summary(self) -> dict:
        return {n: {"prompts": s["prompts"], "prompt_keys": len(s["keys"]), "ngrams": len(s["index"]),
                    "tools": s["tools"], "tool_names_refused": s["refuse_tool_names"]} for n, s in self.sets.items()}


def containment(grams: set, index: set) -> float:
    """The share of a text's 13-grams (`isolation.ngrams`) found in ``index``."""
    return len(grams & index) / len(grams) if grams else 0.0


def reason_class(reason: str) -> str:
    """An exclusion reason without its per-task detail (the census key)."""
    head = reason.split(":")[0]
    return reason if head in ("T1/T2", "T4") else head


def assemble(tasks: Iterable[dict], *, reference: Reference | None = None,
             prompt_lengths: Sequence[Callable[[Mapping], int]] = ()) -> dict:
    """The pool: admitted tasks with atoms, family and split set, the frozen training order, every exclusion."""
    from aadistill.evaluation.isolation import ngrams
    reference = reference or Reference()
    seen, candidates, excluded = {}, [], []
    name_hits = Counter()

    def refuse(t, reason, split=None):
        excluded.append({"task_id": t["task_id"], "reason": reason, "source": str(t["source"].get("dataset")),
                         "category": t.get("category"), "split": split or t.get("split") or "conflict"})

    for t in sorted(tasks, key=lambda t: (str(t["source"].get("dataset")), t["task_id"])):
        atoms = atoms_of(t)
        t = {**t, "atoms": atoms, "family": family_of(atoms), "split": task_split(atoms)}
        if t["split"] is None:
            refuse(t, "T4: its atoms fall in different splits")
            continue
        bad = TE.task_problems(t)
        if bad:
            refuse(t, "record: " + "; ".join(bad))
            continue
        if (t.get("backend") or {}).get("conflicts"):
            refuse(t, f"replay: {len(t['backend']['conflicts'])} call(s) recorded with different responses")
            continue
        key, grams, ids = prompt_key(t), ngrams(user_text(t), NGRAM), tool_identities(t)
        why = None
        for name, ref in reference.sets.items():
            if key in ref["keys"]:
                why = f"T1/T2: a {name} prompt"
            elif any(i in ref["tool_ids"] for _, i in ids):
                why = f"T1/T2: a {name} tool (name and schema)"
            elif any(b in ref["tool_names"] for b, _ in ids):
                if ref["refuse_tool_names"]:
                    why = f"T1/T2: a {name} tool name"
                else:
                    name_hits[name] += 1
            if why is None and ref["index"] and containment(grams, ref["index"]) >= CONTAINMENT:
                why = f"T1/T2: a 13-gram near-duplicate of a {name} prompt"
            if why:
                break
        if why:
            refuse(t, why)
            continue
        dup = seen.get((key, t["family"]))
        if dup is not None:
            refuse(t, f"T5: duplicates {dup}")
            continue
        over = [n for n in (f(t) for f in prompt_lengths) if n > PROMPT_BOUND]
        if over:
            refuse(t, f"length: first prompt {max(over)} > {PROMPT_BOUND}")
            continue
        seen[(key, t["family"])] = t["task_id"]
        candidates.append({**t, "content_keys": {"prompt": key, "tools": t["family"]}})
    admitted = near_duplicates_between_splits(candidates, refuse)
    from aadistill.data.sequence_packing import shuffled_order
    train = [t for t in admitted if t["split"] == "training"]
    order = [train[i]["task_id"] for i in shuffled_order([t["task_id"] for t in train], ORDER_SEED,
                                                         namespace=POOL_ID)]
    census, verification = defaultdict(Counter), defaultdict(Counter)
    atom_splits = defaultdict(set)
    for t in admitted:
        census[t["split"]][t["category"]] += 1
        verification[t["split"]]["+".join(TE.verification_levels(t))] += 1
        for a in t["atoms"]:
            atom_splits[a].add(t["split"])
    crossing = sorted(a for a, s in atom_splits.items() if len(s) > 1)
    if crossing:
        raise AssertionError(f"atoms in more than one split: {crossing[:3]}")
    by = Counter((reason_class(e["reason"]), e["source"], e["category"], e["split"]) for e in excluded)
    return {"tasks": admitted, "excluded": excluded, "training_order": order,
            "training_order_sha256": hashlib.sha256(canonical(order).encode()).hexdigest(),
            "census": {s: dict(c) for s, c in sorted(census.items())},
            "verification_levels": {s: dict(c) for s, c in sorted(verification.items())},
            "atoms": {"total": len(atom_splits),
                      "by_split": dict(Counter(next(iter(s)) for s in atom_splits.values()))},
            "excluded_by": [{"reason": r, "source": src, "category": c, "split": s, "tasks": n}
                            for (r, src, c, s), n in sorted(by.items())],
            "tool_name_only_matches_counted": dict(name_hits),
            "by_source": dict(Counter(str(t["source"].get("dataset")) for t in admitted))}


def near_duplicates_between_splits(candidates: list[dict], refuse) -> list[dict]:
    """T4: walk the splits in `SPLIT_PRIORITY`; a task near-duplicating an earlier split's admitted task is refused."""
    from aadistill.evaluation.isolation import ngrams
    by_split = defaultdict(list)
    for t in candidates:
        by_split[t["split"]].append(t)
    admitted, index, df = [], set(), Counter()
    for split in SPLIT_PRIORITY:
        keep = []
        for t in by_split.get(split, []):
            grams = ngrams(user_text(t), NGRAM)
            if index and containment(grams, index) >= CONTAINMENT:
                refuse(t, "T4: a 13-gram near-duplicate of an earlier split's task", split)
                continue
            keep.append((t, grams))
        admitted += [t for t, _ in keep]
        if split != SPLIT_PRIORITY[-1]:
            for _, grams in keep:
                df.update(grams)
            index = {g for g, c in df.items() if c <= MAX_DF}
    return sorted(admitted, key=lambda t: (str(t["source"].get("dataset")), t["task_id"]))


def eval_set_reference(ref: Reference, repo: Path = REPO) -> Reference:
    """eval-set-1's held-out units: their prompts (per unit; an interactive episode's opening is not a prompt key,
    `eval_sets.KEYS`) and the tools of its tool items, names refused."""
    from shared import eval_sets as ES
    units, tools = {}, []
    for unit, rows in ES.component_items(roles=ES.HELD_OUT_ROLES, repo_root=repo).items():
        units[f"eval-set-1/{unit}"] = [ES.prompt_text(r) for r in rows if ES.KEYS["prompt"](r)]
        for r in rows:
            for t in r.get("tools") or []:
                f = _fn(t)
                tools.append((str(f["name"]), f.get("parameters")))
    return ref.add("eval-set-1", prompt_units=units, tools=tools, refuse_tool_names=True)


def benchmark_reference(ref: Reference) -> Reference:
    """The pinned public benchmarks (`tool_benchmarks`): prompts and tool identities; names only counted."""
    from stages.teacher_modes import tool_benchmarks as TB
    for name in sorted(TB.BENCHMARKS):
        TB.fetch(name)
        ref.add(name, prompts=[t for _, t in TB.prompts(name)],
                tools=[(t["name"], t["parameters"]) for t in TB.tools(name)])
    return ref


def length_checkers(protocol: Mapping) -> list[Callable[[Mapping], int]]:
    """One first-prompt length function per lineage and mode of tool_generation_protocol.json: the teacher's own
    tokenizer at its pinned revision, its official template with the mode's kwargs and the task's tools, the
    system text the session uses -- exactly `run_battery.run_episodes`' first-turn rendering."""
    from transformers import AutoTokenizer
    system = protocol["execution"]["system_message_default"]
    out = []
    for lineage, li in sorted(protocol["lineages"].items()):
        tok = AutoTokenizer.from_pretrained(li["model"], revision=li["revision"])
        for mode, spec in sorted(li["modes"].items()):
            kwargs = dict(spec["template_kwargs"])

            def n(task, tok=tok, kwargs=kwargs):
                hist = [{"role": "system", "content": task.get("system") or system}, *task["messages"]]
                tools = task.get("tools")
                text = tok.apply_chat_template(hist, tokenize=False, add_generation_prompt=True, **kwargs,
                                               **({"tools": tools} if tools else {}))
                return len(tok(text, add_special_tokens=False).input_ids)
            n.__name__ = f"{lineage}:{mode}"
            out.append(n)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", action="append", default=[], help="name=converted tasks.jsonl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    tasks = []
    for spec in a.source:
        name, _, path = spec.partition("=")
        tasks += TE.read_tasks(path)
    proto = json.loads(PROTOCOL.read_text())
    ref = benchmark_reference(eval_set_reference(Reference()))
    checkers = length_checkers(proto)
    pool = assemble(tasks, reference=ref, prompt_lengths=checkers)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for split in {t["split"] for t in pool["tasks"]}:
        with (out / f"{split}.tasks.jsonl").open("w") as f:
            for t in pool["tasks"]:
                if t["split"] == split:
                    f.write(json.dumps(t, ensure_ascii=False) + "\n")
    (out / "pool.json").write_text(json.dumps({
        "schema": SCHEMA, "pool_id": POOL_ID, "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "census": pool["census"], "verification_levels": pool["verification_levels"], "atoms": pool["atoms"],
        "by_source": pool["by_source"], "excluded_by": pool["excluded_by"],
        "tool_name_only_matches_counted": pool["tool_name_only_matches_counted"], "excluded": pool["excluded"],
        "reference": ref.summary(), "length_checkers": [f.__name__ for f in checkers],
        "near_duplicate_protocol": {"n": NGRAM, "containment": CONTAINMENT, "max_document_frequency": MAX_DF},
        "training_order": pool["training_order"], "training_order_sha256": pool["training_order_sha256"],
        "order_seed": ORDER_SEED, "prompt_bound": PROMPT_BOUND, "authorizes": "nothing"}, indent=1) + "\n")
    print(json.dumps({"census": pool["census"], "excluded": len(pool["excluded"])}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
