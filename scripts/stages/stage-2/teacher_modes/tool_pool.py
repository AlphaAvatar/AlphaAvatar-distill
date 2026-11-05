#!/usr/bin/env python3
"""`tool_pool@v1`'s assembly: atomic isolation, decontamination, the length bound, frozen orders and its identity. $0.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.tool_pool \\
        --source toucan=<converted tasks.jsonl> [--source <name>=<converted tasks.jsonl> ...] \\
        --out data/stages/stage-2/teacher_modes/tool_pool_v1

Every source is a CANDIDATE until the maintainer admits it (audit/tool_calling_source_review_v2.json). A source
enters as CONVERTED task records (`tool_episodes.task_problems` must pass: upstream assistant turns are never in the
context, the backend is a replay table or none); each source's converter is its own module (`CONVERTERS`).

* atoms -- a task's atomic tool identities: its ``environment`` when the converter declares one (it owns every
  offered tool: the Pivot sets' connected tool components), else ``server:<slug>`` for every declared server (an
  MCP server owns every tool named ``<slug>-<tool>``) and ``tool:<identity>`` for every offered tool no declared
  server owns, its identity the sha256 of its bare name and canonical parameter schema;
* split -- per ATOM, never per task: ``int(sha256('tool_pool@v1|' + atom), 16) mod 100``: 0-2 validation,
  3-6 state_eval, 7-10 calibration, 11-12 reserve, 13-99 training. A task enters the split all its atoms share;
  a task whose atoms fall in different splits is refused (T4). A connected grouping of Toucan is not an option:
  its multi-server tasks chain 99.8% of it into one component;
* family -- the task's environment, sha256 of its sorted atoms (the T5 key and the documents' ``family``);
* T1 (eval-set-1), T2 (public benchmarks) and T3 (the excluded lineage sources xLAM-60k and Glaive-v2), all pinned
  and verified (`tool_benchmarks`) -- a task sharing a prompt key with a reference item, a 13-gram near-duplicate of
  one (containment >= 0.5 against the reference's index without its template n-grams, isolation_rules.json's
  protocol), or offering a tool with a reference tool's identity (bare name + schema) is refused. T1 also refuses a
  held-out tool's bare NAME. Signature matches (bare name + parameter names; xLAM's parameters are not a JSON
  Schema) and other bare-name matches are counted, not refused: on Toucan they were generic APIs offered by real
  MCP servers, not rows derived from the reference (``Reference.add(refuse_signatures=True)`` refuses them);
* T4 (between splits) -- walking ``state_eval, calibration, validation, reserve, training``, a task is refused when
  an earlier split's admitted task has its exact prompt key (13-gram screens drop template n-grams, so a prompt
  repeated across servers would pass them), offers one of its tool identities, or is its 13-gram near-duplicate;
* T5 -- an exact or normalized-query duplicate (same prompt key and family) collapses to its first task in
  (source, task id) order;
* schema -- a task offering a tool whose parameter schema uses a keyword the validator does not check
  (`aadistill.data.tool_calls.unchecked_keywords`) is refused: its calls could never pass schema validity;
* replay -- a task whose converter recorded a call with conflicting responses (``backend.conflicts``) is refused;
* length -- a task whose first rendered prompt exceeds ``PROMPT_BOUND`` tokens under ANY lineage and mode of
  tool_generation_protocol.json (each teacher's own tokenizer and official template, the tools rendered by it, the
  system text the session uses) is refused for every teacher (`length_checkers`); an admitted task records the
  longest of those lengths (``first_prompt_tokens``, the generation plan's prefill term);
* orders -- every split's canonical order is a seeded shuffle (``ORDER_SEED``) of its task ids, frozen with its
  sha256 before any generation; every teacher walks the same order (``training_order`` is the training split's).
  Episode shapes are tiers of that order (``SHAPE_TIERS``, schema v4): every full episode first, in its shuffle,
  then every single-step gold tool call in its own -- a cell reaches a decision sample only after the full
  episodes of its category, so a cell its full episodes fill is selected exactly as before. The shuffle key is
  per id, so a task keeps its place among the others when tasks are added;
* shapes -- a T5 duplicate collapses to a full episode before a single-step task of the same prompt and family;
* identity -- pool.json registers every split file's sha256, task count, sources and frozen order's sha256, every
  converted input's sha256, and ``content_sha256`` over files, orders and sources; `load_pool` re-verifies all of it
  before anything reads a task and refuses a missing file, a changed byte, a task in the wrong file, an unregistered
  source revision, an order that is not its registered one or a content sha256 that is not its parts';
  `tool_generation_plan.verified` then binds the plan to that content and to pool.json's exact bytes.

Every exclusion is listed with its reason and counted by reason, source, category and split; nothing is silently
dropped. The census registers every admitted task's verification levels (`tool_episodes.verification_levels`).
"""
from __future__ import annotations

import argparse
import gc
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
#: Toucan's SFT config (tool_sources_toucan.py) and the Pivot pair (tool_sources_pivot.py); Nemotron-SFT-Agentic-v4
#: is deferred (audit/tool_calling_source_review_v2.json). Every source remains a CANDIDATE until the maintainer
#: admits it.
def _toucan(path: Path) -> Iterable[dict]:
    from stages.teacher_modes import tool_sources_toucan
    return tool_sources_toucan.convert(path)


def _pivot(which: str) -> Callable[[Path], Iterable[dict]]:
    def convert(path: Path) -> Iterable[dict]:
        from stages.teacher_modes import tool_sources_pivot
        return tool_sources_pivot.convert(path, which)[0]
    return convert


CONVERTERS: dict[str, Callable[[Path], Iterable[dict]]] = {
    "toucan_sft": _toucan, "pivot_function_calling": _pivot("function_calling"),
    "pivot_conversational": _pivot("conversational")}


class PoolIdentityError(RuntimeError):
    """A pool whose files are missing, changed, misplaced or from an unregistered source."""
SCHEMA = "aadistill.stage2.teacher_modes.tool_pool/v4"
#: the order tier of each registered episode shape (`tool_episodes.EPISODE_SHAPES`): stronger verification first
SHAPE_TIERS = {TE.FULL_EPISODE: 0, TE.SINGLE_STEP: 1}


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


def param_names(parameters) -> tuple[str, ...]:
    """A tool's parameter names: a JSON Schema's properties, or xLAM's flat ``{name: {type, description}}`` map."""
    if not isinstance(parameters, Mapping):
        return ()
    if isinstance(parameters.get("properties"), Mapping):
        return tuple(sorted(parameters["properties"]))
    if parameters.get("type") == "object" or not parameters:
        return ()
    #: a flat map: every value a parameter's description (xLAM's tools may have a parameter named "type")
    if all(isinstance(v, Mapping) for v in parameters.values()):
        return tuple(sorted(parameters))
    return ()


def tool_signature(bare: str, parameters) -> str:
    return f"{bare}({','.join(param_names(parameters))})"


def atoms_of(task: Mapping) -> list[str]:
    """The task's atomic identities: its declared environment; else its declared servers and every offered tool no
    declared server owns."""
    if task.get("environment"):
        return [f"environment:{task['environment']}"]
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
    """What a pool is isolated from: per named set, its rule label (T1 / T2 / T3), prompt keys, a 13-gram index,
    tool identities, signatures and names."""

    def __init__(self):
        self.sets: dict[str, dict] = {}

    def add(self, name: str, *, label: str, prompts: Iterable[str] = (),
            prompt_units: Mapping[str, Sequence[str]] | None = None, tools: Iterable[tuple[str, object]] = (),
            refuse_tool_names: bool = False, refuse_signatures: bool = False) -> Reference:
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
        self.sets[name] = {"label": label, "keys": keys - {None, ""}, "index": index,
                           "tool_ids": {tool_identity(n, p) for n, p in tools}, "tool_names": {n for n, _ in tools},
                           "signatures": {tool_signature(n, p) for n, p in tools},
                           "refuse_tool_names": refuse_tool_names, "refuse_signatures": refuse_signatures,
                           "prompts": sum(map(len, units.values())), "tools": len(tools)}
        return self

    def summary(self) -> dict:
        return {n: {"label": s["label"], "prompts": s["prompts"], "prompt_keys": len(s["keys"]),
                    "ngrams": len(s["index"]), "tools": s["tools"], "tool_names_refused": s["refuse_tool_names"],
                    "signatures_refused": s["refuse_signatures"]} for n, s in self.sets.items()}


def containment(grams: set, index: set) -> float:
    """The share of a text's 13-grams (`isolation.ngrams`) found in ``index``."""
    return len(grams & index) / len(grams) if grams else 0.0


def reason_class(reason: str) -> str:
    """An exclusion reason without its per-task detail (the census key)."""
    head = reason.split(":")[0]
    return reason if head in ("T1", "T2", "T3", "T4") else head


def screen(t: Mapping, key: str, grams: set, reference: Reference, name_hits: Counter) -> str | None:
    """Why ``t`` is refused by a reference set (T1 / T2 / T3), or None."""
    tools = [(b, i, tool_signature(b, _fn(x).get("parameters")))
             for (b, i), x in zip(tool_identities(t), t.get("tools") or [])]
    for name, ref in reference.sets.items():
        lab = ref["label"]
        if key in ref["keys"]:
            return f"{lab}: a {name} prompt"
        if any(i in ref["tool_ids"] for _, i, _ in tools):
            return f"{lab}: a {name} tool (name and schema)"
        if any(sig in ref["signatures"] for _, _, sig in tools):
            if ref["refuse_signatures"]:
                return f"{lab}: a {name} tool (name and parameter names)"
            name_hits[f"{name}:signature"] += 1
        if any(b in ref["tool_names"] for b, _, _ in tools):
            if ref["refuse_tool_names"]:
                return f"{lab}: a {name} tool name"
            name_hits[name] += 1
        if ref["index"] and containment(grams, ref["index"]) >= CONTAINMENT:
            return f"{lab}: a 13-gram near-duplicate of a {name} prompt"
    return None


def assemble(tasks: Iterable[dict], *, reference: Reference | None = None,
             prompt_lengths: Sequence[Callable[[Mapping], int]] = (),
             split_rule: Callable[[Mapping, Sequence[str]], str | None] | None = None, pool_id: str = POOL_ID) -> dict:
    """The pool: admitted tasks with atoms, family and split set, every split's frozen order, every exclusion.
    ``split_rule(task, atoms)`` replaces the per-atom hash (`task_split`) for a source that registers its own
    whole-environment split (`extend`); every screen and the isolation between splits apply unchanged."""
    from aadistill.data.tool_calls import unchecked_keywords
    from aadistill.evaluation.isolation import ngrams
    reference = reference or Reference()
    seen, candidates, excluded = {}, [], []
    name_hits = Counter()

    def refuse(t, reason, split=None):
        excluded.append({"task_id": t["task_id"], "reason": reason, "source": str(t["source"].get("dataset")),
                         "category": t.get("category"), "split": split or t.get("split") or "conflict",
                         **({"episode_shape": TE.episode_shape(t)} if t.get("episode_shape") else {})})

    for t in sorted(tasks, key=lambda t: (str(t["source"].get("dataset")), SHAPE_TIERS[TE.episode_shape(t)],
                                          t["task_id"])):
        atoms = atoms_of(t)
        t = {**t, "atoms": atoms, "family": family_of(atoms),
             "split": split_rule(t, atoms) if split_rule else task_split(atoms)}
        if t["split"] is None:
            refuse(t, "T4: its atoms fall in different splits")
            continue
        bad = TE.task_problems(t)
        if bad:
            refuse(t, "record: " + "; ".join(bad))
            continue
        unchecked = sorted({k for x in t["tools"] for k in unchecked_keywords(_fn(x).get("parameters") or {})})
        if unchecked:
            refuse(t, f"schema: keywords the validator does not check {unchecked}")
            continue
        if (t.get("backend") or {}).get("conflicts"):
            refuse(t, f"replay: {len(t['backend']['conflicts'])} call(s) recorded with different responses")
            continue
        key, grams = prompt_key(t), ngrams(user_text(t), NGRAM)
        why = screen(t, key, grams, reference, name_hits)
        if why:
            refuse(t, why)
            continue
        dup = seen.get((key, t["family"]))
        if dup is not None:
            refuse(t, f"T5: duplicates {dup}")
            continue
        lengths = [f(t) for f in prompt_lengths]
        if lengths and max(lengths) > PROMPT_BOUND:
            refuse(t, f"length: first prompt {max(lengths)} > {PROMPT_BOUND}")
            continue
        seen[(key, t["family"])] = t["task_id"]
        candidates.append({**t, "content_keys": {"prompt": key, "tools": t["family"]},
                           "first_prompt_tokens": max(lengths) if lengths else None})
    admitted = isolate_between_splits(candidates, refuse)
    orders = frozen_orders(admitted, pool_id)
    census, verification, shapes = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    atom_splits, key_splits, tool_splits = defaultdict(set), defaultdict(set), defaultdict(set)
    for t in admitted:
        census[t["split"]][t["category"]] += 1
        shapes[t["split"]][f"{TE.episode_shape(t)}|{t['category']}"] += 1
        verification[t["split"]]["+".join(TE.verification_levels(t))] += 1
        key_splits[t["content_keys"]["prompt"]].add(t["split"])
        for a in t["atoms"]:
            atom_splits[a].add(t["split"])
        for _, i in tool_identities(t):
            tool_splits[i].add(t["split"])
    for what, m in (("atoms", atom_splits), ("prompt keys", key_splits), ("tool identities", tool_splits)):
        crossing = sorted(a for a, s in m.items() if len(s) > 1)
        if crossing:
            raise AssertionError(f"{what} in more than one split: {crossing[:3]}")
    by = Counter((reason_class(e["reason"]), e["source"], e["category"], e["split"]) for e in excluded)
    return {"tasks": admitted, "excluded": excluded, "orders": orders,
            "training_order": orders.get("training", []),
            "training_order_sha256": hashlib.sha256(canonical(orders.get("training", [])).encode()).hexdigest(),
            "census": {s: dict(c) for s, c in sorted(census.items())},
            "episode_shapes": {s: dict(sorted(c.items())) for s, c in sorted(shapes.items())},
            "verification_levels": {s: dict(c) for s, c in sorted(verification.items())},
            "atoms": {"total": len(atom_splits),
                      "by_split": dict(Counter(next(iter(s)) for s in atom_splits.values())),
                      "by_kind": dict(Counter(a.split(":")[0] for a in atom_splits))},
            "excluded_by": [{"reason": r, "source": src, "category": c, "split": s, "tasks": n}
                            for (r, src, c, s), n in sorted(by.items())],
            "tool_name_only_matches_counted": dict(name_hits),
            "by_source": {s: dict(c) for s, c in sorted(_by_source(admitted).items())}}


def _by_source(tasks) -> dict:
    out = defaultdict(Counter)
    for t in tasks:
        out[str(t["source"].get("dataset"))][t["split"]] += 1
    return out


def frozen_orders(tasks: list[dict], pool_id: str = POOL_ID) -> dict[str, list[str]]:
    """Every split's seeded shuffle of its task ids (training's namespace is the pool id, as in v1), tiered by
    episode shape (`SHAPE_TIERS`): the full episodes in their shuffle, then the single-step tasks in theirs."""
    from aadistill.data.sequence_packing import shuffled_order
    tier = {t["task_id"]: SHAPE_TIERS[TE.episode_shape(t)] for t in tasks}
    out = {}
    for split in sorted({t["split"] for t in tasks}):
        ids = sorted(t["task_id"] for t in tasks if t["split"] == split)
        ns = pool_id if split == "training" else f"{pool_id}|{split}"
        out[split] = sorted((ids[i] for i in shuffled_order(ids, ORDER_SEED, namespace=ns)), key=tier.get)
    return out


def isolate_between_splits(candidates: list[dict], refuse) -> list[dict]:
    """T4: walk the splits in `SPLIT_PRIORITY`; a task is refused when an earlier split's admitted task has its exact
    prompt key, one of its tool identities, or is its 13-gram near-duplicate."""
    from aadistill.evaluation.isolation import ngrams
    by_split = defaultdict(list)
    for t in candidates:
        by_split[t["split"]].append(t)
    admitted, index, df, keys, tools = [], set(), Counter(), set(), set()
    for split in SPLIT_PRIORITY:
        keep, last = [], split == SPLIT_PRIORITY[-1]
        for t in by_split.get(split, []):
            grams = ngrams(user_text(t), NGRAM)
            ids = {i for _, i in tool_identities(t)}
            if t["content_keys"]["prompt"] in keys:
                refuse(t, "T4: an earlier split's exact prompt", split)
            elif ids & tools:
                refuse(t, "T4: a tool identity of an earlier split", split)
            elif index and containment(grams, index) >= CONTAINMENT:
                refuse(t, "T4: a 13-gram near-duplicate of an earlier split's task", split)
            else:
                keep.append((t, None, None) if last else (t, grams, ids))   #: nothing reads the last split's
        admitted += [t for t, _, _ in keep]
        if last:
            break
        for t, grams, ids in keep:
            keys.add(t["content_keys"]["prompt"])
            tools |= ids
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
    return ref.add("eval-set-1", label="T1", prompt_units=units, tools=tools, refuse_tool_names=True)


def benchmark_reference(ref: Reference) -> Reference:
    """The pinned public benchmarks (T2) and lineage sources (T3), each verified against its manifest
    (`tool_benchmarks.fetch`): prompts and tool identities refused; tool signatures and names only counted -- a
    signature match on Toucan was generic APIs (get_definitions(word), get_current_weather(location), ...), not a
    derived row (audit/tool_pool_dry_run_v3.json)."""
    from stages.teacher_modes import tool_benchmarks as TB
    for name in sorted(TB.REFERENCES):
        TB.fetch(name)
        ref.add(name, label="T3" if name in TB.LINEAGE else "T2", prompts=[t for _, t in TB.prompts(name)],
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


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def content_sha256(files: Mapping, sources: Mapping) -> str:
    """The pool's content identity: every split file's sha256, its frozen order's sha256, and the sources."""
    return hashlib.sha256(canonical({"files": {k: v["sha256"] for k, v in files.items()},
                                     "orders": {k: v["order_sha256"] for k, v in files.items()},
                                     "sources": sources}).encode()).hexdigest()


def write_pool(pool: dict, out: Path, *, sources: dict, extra: dict) -> dict:
    """The split files and pool.json with the identity `load_pool` checks; returns pool.json's content."""
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for split in sorted({t["split"] for t in pool["tasks"]}):
        rows = [t for t in pool["tasks"] if t["split"] == split]
        path = out / f"{split}.tasks.jsonl"
        with path.open("w") as f:
            for t in rows:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        files[split] = {"file": path.name, "sha256": _sha_file(path), "tasks": len(rows),
                        "by_source": dict(Counter(str(t["source"].get("dataset")) for t in rows)),
                        "order_sha256": hashlib.sha256(canonical(pool["orders"][split]).encode()).hexdigest()}
    revisions = sorted({(str(t["source"].get("dataset")), str(t["source"].get("revision"))) for t in pool["tasks"]})
    doc = {"schema": SCHEMA, "pool_id": POOL_ID,
           "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "files": files, "sources": sources, "source_revisions": [list(x) for x in revisions],
           "content_sha256": content_sha256(files, sources),
           "census": pool["census"], "episode_shapes": pool["episode_shapes"], "shape_tiers": SHAPE_TIERS,
           "verification_levels": pool["verification_levels"], "atoms": pool["atoms"],
           "by_source": pool["by_source"], "excluded_by": pool["excluded_by"],
           "tool_name_only_matches_counted": pool["tool_name_only_matches_counted"], "excluded": pool["excluded"],
           **extra,
           "near_duplicate_protocol": {"n": NGRAM, "containment": CONTAINMENT, "max_document_frequency": MAX_DF},
           "orders": pool["orders"], "training_order": pool["training_order"],
           "training_order_sha256": pool["training_order_sha256"],
           "order_seed": ORDER_SEED, "prompt_bound": PROMPT_BOUND, "authorizes": "nothing"}
    (out / "pool.json").write_text(json.dumps(doc, indent=1) + "\n")
    return doc


#: a pool that is an existing pool plus new tasks (`extend`): the base's files, identity and frozen orders untouched
EXTENSION_SCHEMA = "aadistill.stage2.teacher_modes.tool_pool_extension/v1"


def extension_content_sha256(base_content: str, files: Mapping, sources: Mapping) -> str:
    """An extension's content identity: the base pool's content sha256, every addition file's sha256, every split's
    FULL order (base then addition) and the addition's sources."""
    return hashlib.sha256(canonical({"base": base_content, "files": {k: v["sha256"] for k, v in files.items()},
                                     "orders": {k: v["order_sha256"] for k, v in files.items()},
                                     "sources": sources}).encode()).hexdigest()


def _rel(p: Path) -> str:
    p = Path(p).resolve()
    return p.relative_to(REPO).as_posix() if p.is_relative_to(REPO) else str(p)


def extend(base_dir: Path, addition: Sequence[dict], out: Path, *, pool_id: str, sources: dict,
           reference: Reference | None = None, prompt_lengths: Sequence[Callable[[Mapping], int]] = (),
           extra: Mapping | None = None) -> dict:
    """Write ``out`` = the verified pool at ``base_dir`` plus ``addition``, without copying or re-ordering the base.

    The addition keeps the split each task declares (a whole environment per split, the source's registered rule)
    and passes every screen of `assemble` (record, schema keywords, replay conflicts, T1-T3, T5, the length bound);
    T4 then runs over the base and the addition together in `SPLIT_PRIORITY`, and must refuse nothing of the base --
    no base task changes split, order or identity. Each split's frozen order is the base's order followed by the
    addition's own seeded shuffle (namespace ``pool_id``), so every base task keeps its place and every import
    record bound to the base's content keeps describing the same tasks (`tool_generation_plan.import_records`)."""
    base_man, base_splits = load_pool(base_dir)
    base = [t for rows in base_splits.values() for t in rows]
    pool = assemble(addition, reference=reference, prompt_lengths=prompt_lengths,
                    split_rule=lambda t, atoms: t.get("split"), pool_id=pool_id)
    refused = []
    kept = isolate_between_splits(base + pool["tasks"], lambda t, why, split=None: refused.append(
        {"task_id": t["task_id"], "reason": why, "source": str(t["source"].get("dataset")),
         "category": t.get("category"), "split": split or t.get("split")}))
    base_ids = {t["task_id"] for t in base}
    lost = [r for r in refused if r["task_id"] in base_ids]
    if lost:
        raise AssertionError(f"T4 over base + addition would refuse {len(lost)} base task(s), e.g. {lost[:2]}")
    crossing = defaultdict(set)
    for t in base + pool["tasks"]:
        for a in t.get("atoms") or atoms_of(t):
            crossing[a].add(t["split"])
    shared = sorted(a for a, sp in crossing.items() if len(sp) > 1)
    if shared:
        raise AssertionError(f"atoms in more than one split across base + addition: {shared[:3]}")
    keep = {t["task_id"] for t in kept} - base_ids
    added = [t for t in pool["tasks"] if t["task_id"] in keep]
    excluded = pool["excluded"] + refused
    own = frozen_orders(added, pool_id)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    files, orders = {}, {}
    for split in sorted(set(base_man["orders"]) | set(own)):
        orders[split] = list(base_man["orders"].get(split, [])) + own.get(split, [])
        rows = [t for t in added if t["split"] == split]
        path = out / f"{split}.tasks.jsonl"
        with path.open("w") as f:
            for t in rows:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        files[split] = {"file": path.name, "sha256": _sha_file(path), "tasks": len(rows),
                        "by_source": dict(Counter(str(t["source"].get("dataset")) for t in rows)),
                        "order_sha256": hashlib.sha256(canonical(orders[split]).encode()).hexdigest()}
    revisions = sorted({tuple(x) for x in base_man.get("source_revisions") or []} |
                       {(str(t["source"].get("dataset")), str(t["source"].get("revision"))) for t in added})
    census = defaultdict(Counter)
    for split, rows in base_splits.items():
        for t in rows:
            census[split][t["category"]] += 1
    for t in added:
        census[t["split"]][t["category"]] += 1
    doc = {"schema": EXTENSION_SCHEMA, "pool_id": pool_id,
           "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "base": {"dir": _rel(base_dir), "pool_id": base_man["pool_id"], "content_sha256": base_man["content_sha256"],
                    "pool_json_sha256": _sha_file(Path(base_dir) / "pool.json")},
           "files": files, "sources": sources, "source_revisions": [list(x) for x in revisions],
           "content_sha256": extension_content_sha256(base_man["content_sha256"], files, sources),
           "census": {sp: dict(c) for sp, c in sorted(census.items())},
           "addition": {"census": pool["census"], "episode_shapes": pool["episode_shapes"],
                        "verification_levels": pool["verification_levels"], "atoms": pool["atoms"],
                        "by_source": pool["by_source"], "admitted": len(added),
                        "excluded_by": [{"reason": r, "tasks": n} for r, n in
                                        sorted(Counter(reason_class(e["reason"]) for e in excluded).items())],
                        "excluded": excluded,
                        "tool_name_only_matches_counted": pool["tool_name_only_matches_counted"]},
           **dict(extra or {}),
           "near_duplicate_protocol": {"n": NGRAM, "containment": CONTAINMENT, "max_document_frequency": MAX_DF},
           "orders": orders, "training_order": orders.get("training", []),
           "training_order_sha256": hashlib.sha256(canonical(orders.get("training", [])).encode()).hexdigest(),
           "order_seed": ORDER_SEED, "prompt_bound": PROMPT_BOUND, "authorizes": "nothing"}
    (out / "pool.json").write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def load_pool(pool_dir: Path, splits: Sequence[str] | None = None) -> tuple[dict, dict[str, list[dict]]]:
    """``(pool.json, {split: tasks})`` after verifying every file the caller reads against pool.json: present,
    byte-identical (sha256), its task count, every task in its own split and from a registered source revision, the
    split's order exactly its task ids; raises `PoolIdentityError` otherwise (fail closed)."""
    pool_dir = Path(pool_dir)
    man_path = pool_dir / "pool.json"
    if not man_path.is_file():
        raise PoolIdentityError(f"{man_path} does not exist")
    man = json.loads(man_path.read_text())
    if man.get("schema") == EXTENSION_SCHEMA:
        return _load_extension(pool_dir, man, splits)
    if man.get("schema") != SCHEMA or not isinstance(man.get("files"), dict):
        raise PoolIdentityError(f"{man_path} is not a {SCHEMA} manifest with registered files")
    if man.get("content_sha256") != content_sha256(man["files"], man.get("sources") or {}):
        raise PoolIdentityError("pool.json's content_sha256 is not its registered files, orders and sources")
    want = sorted(man["files"]) if splits is None else list(splits)
    revisions = {tuple(x) for x in man.get("source_revisions") or []}
    out = {}
    for split in want:
        rec = man["files"].get(split)
        if rec is None:
            raise PoolIdentityError(f"pool.json registers no {split!r} file")
        path = pool_dir / rec["file"]
        if not path.is_file():
            raise PoolIdentityError(f"{path} is missing")
        if _sha_file(path) != rec["sha256"]:
            raise PoolIdentityError(f"{path.name} differs from its registered sha256")
        rows = TE.read_tasks(path)
        if len(rows) != rec["tasks"]:
            raise PoolIdentityError(f"{path.name} holds {len(rows)} tasks, pool.json registers {rec['tasks']}")
        stray = [t["task_id"] for t in rows if t.get("split") != split
                 or (str(t["source"].get("dataset")), str(t["source"].get("revision"))) not in revisions]
        if stray:
            raise PoolIdentityError(f"{path.name}: {len(stray)} task(s) of another split or source, e.g. {stray[:3]}")
        order = man["orders"].get(split) or []
        if sorted(order) != sorted(t["task_id"] for t in rows) or \
                hashlib.sha256(canonical(order).encode()).hexdigest() != rec["order_sha256"]:
            raise PoolIdentityError(f"the {split} order is not the split's task ids as registered")
        out[split] = rows
    return man, out


def _load_extension(pool_dir: Path, man: dict, splits: Sequence[str] | None) -> tuple[dict, dict[str, list[dict]]]:
    """An extension read as one pool: the base verified first (`load_pool`, recursively) and bound by its content
    sha256 and pool.json bytes; then each addition file as `load_pool` checks a file; each split's rows are the base's
    then the addition's, and its order must be the base's order followed by the addition's ids."""
    if not isinstance(man.get("files"), dict) or not isinstance(man.get("base"), dict):
        raise PoolIdentityError(f"{pool_dir}/pool.json is not an extension manifest with a base and files")
    if man.get("content_sha256") != extension_content_sha256(man["base"].get("content_sha256"), man["files"],
                                                             man.get("sources") or {}):
        raise PoolIdentityError("the extension's content_sha256 is not its base, files, orders and sources")
    bdir = Path(man["base"]["dir"])
    bdir = bdir if bdir.is_absolute() else REPO / bdir
    if _sha_file(bdir / "pool.json") != man["base"].get("pool_json_sha256"):
        raise PoolIdentityError(f"the base pool.json at {bdir} is not the one the extension registers")
    want = sorted(man["files"]) if splits is None else list(splits)
    base_man, base_rows = load_pool(bdir, [s for s in want if s in json.loads((bdir / "pool.json").read_text())["files"]])
    if base_man["content_sha256"] != man["base"]["content_sha256"]:
        raise PoolIdentityError("the base pool's content is not the one the extension registers")
    revisions = {tuple(x) for x in man.get("source_revisions") or []}
    out = {}
    for split in want:
        rec = man["files"].get(split)
        if rec is None:
            raise PoolIdentityError(f"the extension registers no {split!r} file")
        path = Path(pool_dir) / rec["file"]
        if not path.is_file() or _sha_file(path) != rec["sha256"]:
            raise PoolIdentityError(f"{path.name} is missing or differs from its registered sha256")
        rows = TE.read_tasks(path)
        if len(rows) != rec["tasks"]:
            raise PoolIdentityError(f"{path.name} holds {len(rows)} tasks, the extension registers {rec['tasks']}")
        stray = [t["task_id"] for t in rows if t.get("split") != split
                 or (str(t["source"].get("dataset")), str(t["source"].get("revision"))) not in revisions]
        if stray:
            raise PoolIdentityError(f"{path.name}: {len(stray)} task(s) of another split or source, e.g. {stray[:3]}")
        base_order = base_man["orders"].get(split) or []
        order = man["orders"].get(split) or []
        tail = order[len(base_order):]
        if order[:len(base_order)] != base_order or sorted(tail) != sorted(t["task_id"] for t in rows) or \
                hashlib.sha256(canonical(order).encode()).hexdigest() != rec["order_sha256"]:
            raise PoolIdentityError(f"the {split} order is not the base's order followed by the addition's tasks")
        out[split] = list(base_rows.get(split, [])) + rows
    return man, out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", action="append", default=[], help="name=converted tasks.jsonl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    tasks, sources, shared = [], {}, {}
    for spec in a.source:
        name, _, path = spec.partition("=")
        rows = read_interned(Path(path), shared)
        tasks += rows
        sources[name] = {"converted_sha256": _sha_file(Path(path)), "tasks": len(rows),
                         "datasets": sorted({f"{t['source'].get('dataset')}@{t['source'].get('revision')}"
                                             for t in rows})}
    proto = json.loads(PROTOCOL.read_text())
    ref = benchmark_reference(eval_set_reference(Reference()))
    summary = ref.summary()
    checkers = length_checkers(proto)
    pool = assemble(tasks, reference=ref, prompt_lengths=checkers)
    del tasks, ref, shared
    write_pool(pool, Path(a.out), sources=sources,
               extra={"reference": summary, "length_checkers": [f.__name__ for f in checkers]})
    report = {"census": pool["census"], "excluded": len(pool["excluded"])}
    del pool
    gc.collect()
    load_pool(Path(a.out))                     #: what was written reads back under its own identity
    print(json.dumps(report, indent=1))
    return 0


def read_interned(path: Path, shared: dict) -> list[dict]:
    """A converted source's tasks, with identical tool lists and system texts held once (Toucan's 119,287 tasks
    carry 6,319 distinct tool lists): the records are unchanged, only their memory is shared."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            t = json.loads(line)
            key = canonical(t.get("tools"))
            t["tools"] = shared.setdefault(key, t.get("tools"))
            if t.get("system"):
                t["system"] = shared.setdefault("system:" + t["system"], t["system"])
            out.append(t)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
