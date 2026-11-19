"""Whether one evaluation set (one model x one mode) is complete, checked on its files and never on a report's word.

Phase 1 has eight sets: the three untrained arms and the teacher, each in both modes, at 8,192 tokens. A set is
complete only when all of these hold (`check`):

* **coverage** -- every registered unit of eval-set-1 has its generations file, holding every registered item id
  exactly once (`b_evidence.completeness`, from the set's own item files: never a typed count);
* **generation records** -- every record carries its raw output and the termination fields the scorers read
  (`TURN_FIELDS`), and passes the per-record
  protocol check (`teacher_parent_initialization.analysis.item_protocol`): each turn's sampling seed is the seed
  rule's for its unit, item and turn, prompt tokens plus generation allowance is the registered context, the rendered
  prompt carries the mode's think state, a natural termination ends with the unit's stop id, the fluid system turn is
  the registered message;
* **protocol summary** -- every unit has its summary and at least one runner summary, every key field
  (`b_evidence.KEY_FIELDS`) of every runner summary is recorded, and equals the registered value wherever the design,
  the decoding protocol or the teacher identity fixes one (`expected`). The set's protocol digest (sha256 over those
  fields, unit by unit) is recorded: the four models of a mode must share it (`mode_problems`);
* **formal report** -- ``formal_report.json`` is the scorer's report of exactly these generations (rescored here, $0):
  a ``{}`` or stale report is a missing part, every unit must be measured with nothing missing;
* **model identity (provenance)** -- every sealed part (``parts/part_NNNN.<generation>.jsonl``, fluid;
  ``parts/<unit>/...``, capability) names the generation that sealed it, and that generation's package record
  (``packages.json``, appended before anything is generated) names the expected weights (the arm's committed init
  record's ``model.safetensors`` sha256, or the teacher identity's three shards), this target and mode, and the
  registered protocol (`protocol_sha256`); every record the part holds is a registered item, well formed, under the
  per-record protocol (seed rule, context, the mode's think state, a teacher EOS id, the system turn). This holds
  whether or not a generations file exists yet: an attempt stopped with only ``parts/`` is checked the same way. And
  every record of every generations file is one a proven part holds.

`repair` turns what a check found into the smallest action, never choosing by outcome: a stale report is rescored;
every part whose provenance is wrong or unverifiable is isolated (moved under ``superseded/<generation>/`` with its
reason, never pooled) and the proven ones kept; a generations file with missing, extra or repeated items, or a record
no proven part holds, is superseded, so the runners compose the unit from the proven parts and generate only the items
none holds, under the same per-item seeds; a unit whose runner protocol cannot be shown is superseded whole. The same
item sealed twice in proven parts is a defect nothing here resolves: refused.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.single_operator_budget import common as C  # noqa: E402

SCHEMA = "aadistill.stage1.single_operator_budget.evaluation_completion/v1"
TEACHER = "teacher"
PACKAGES = "packages.json"
COMPLETION = "completion.json"
SUPERSEDED = "superseded"
#: summary fields that describe the engine and libraries: reported, never part of the protocol digest
NOT_PROTOCOL = ("architectural_context",)


class CompletionError(RuntimeError):
    pass


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").split("\n") if x.strip()]


def ev_dir(out: Path, target: str, mode: str) -> Path:
    return Path(out) / "evaluation" / target / C.mode_dirs()[mode]


def targets() -> list[str]:
    return list(C.design()["phase_1"]["evaluation_order"])


# ---- the registered protocol ------------------------------------------------------------------------------------------

def expected(mode: str) -> dict:
    """Per unit kind, the protocol fields (`b_evidence.protocol_fields` names) a registered source fixes: the
    decoding protocol file the design pins, the design's context, system message and degeneration interval, the
    mode's template kwargs, the teacher identity's chat template and tokenizer, and the fluid unit's task file."""
    from shared import eval_sets as ES
    d = C.design()
    ev = d["evaluation"]["protocol"]
    dec_path = REPO / ev["decoding"]["path"]
    if sha256_file(dec_path) != ev["decoding"]["sha256"]:
        raise CompletionError(f"{dec_path} is not the decoding protocol the design pins")
    dec = json.loads(dec_path.read_text())
    ident = json.loads((REPO / d["teacher"]["identity"]["path"]).read_text())
    ctx = int(ev["context"]["tokens"])
    common = {"decoding_protocol_id": dec["protocol_id"], "decoding_sha256": ev["decoding"]["sha256"],
              "sampling": dec["sampling"], "samples_per_item": dec["samples_per_item"],
              "seed_rule_base": dec["seed_rule"]["base"], "seed_rule_key": dec["seed_rule"]["key"],
              "seed_rule_derivation": dec["seed_rule"]["derivation"], "resolved_context": ctx,
              "context_source": "trained_block_len", "trained_context": ctx, "system_message": ev["system_message"],
              "chat_template_sha256": ident["chat_template_sha256"],
              "chat_template_kwargs": ev["modes"][mode]["template_kwargs"],
              "tokenizer_sha256": ident["tokenizer_files_sha256"],
              "degeneration_check_every": int(ev["degeneration_check_every"])}
    tasks = ES.unit_path(ev["eval_set"], "fluid_reasoning_720", "dev_v1")
    return {"fluid": {**common, "action_protocol": ev["action_protocol"], "seed_namespace": "fluid_reasoning_720__dev_v1",
                      "tasks_sha256": sha256_file(tasks)},
            "capability": dict(common), "capability_per_unit": {},
            "think_state": ev["modes"][mode]["think_state"]}


def protocol_sha256(mode: str) -> str:
    """The registered protocol of ``mode`` as one hash (`expected`): what a generation's package record names, so a
    part sealed under another protocol is never taken."""
    return hashlib.sha256(json.dumps(expected(mode), sort_keys=True, default=str).encode()).hexdigest()


def teacher_stop_ids() -> list[int]:
    """The teacher identity's EOS ids: a natural termination of a part's record must end with one of them."""
    ident = json.loads((REPO / C.design()["teacher"]["identity"]["path"]).read_text())
    return sorted(int(i) for i in ident["eos_token_ids"])


def expected_weights(target: str, out: Path) -> dict[str, str] | None:
    """The weights a set must have been generated from: the arm's committed init record (the copy in this evidence
    directory), or the teacher identity's shards. None when the arm has no COMPLETE init record here."""
    d = C.design()
    if target == TEACHER:
        ident = json.loads((REPO / d["teacher"]["identity"]["path"]).read_text())
        return {s["file"]: s["lfs_sha256"] for s in ident["weight_shards"]}
    p = Path(out) / target / "init_record.json"
    if not p.is_file():
        return None
    rec = json.loads(p.read_text())
    if rec.get("status") != "COMPLETE" or rec.get("arm") != target:
        return None
    return {"model.safetensors": rec["checkpoint"]["shards"][0]["sha256"]}


# ---- the check ----------------------------------------------------------------------------------------------------

#: what every generated turn must carry: its raw output and the termination fields the registered scorers read
#: (`score_battery.turn_components`, `usable_rollout.components`)
TURN_FIELDS = ("raw", "stop_reason", "natural_termination", "degeneration_triggered", "context_limit_reached",
               "generated_tokens")


def _turn_ok(t: dict) -> bool:
    return isinstance(t.get("raw"), str) and bool(t.get("stop_reason")) and all(k in t for k in TURN_FIELDS)


def _wellformed(key: str, rows: list[dict]) -> list[str]:
    from stages.teacher_parent_initialization import b_evidence as BE
    bad = []
    for r in rows:
        if key == BE.FLUID:
            turns = r.get("turns")
            ok = (isinstance(r.get("task_id"), str) and bool(r.get("end")) and bool(r.get("label"))
                  and isinstance(turns, list) and turns and all(_turn_ok(t) for t in turns))
            iid = r.get("task_id")
        else:
            ok = isinstance(r.get("id"), str) and _turn_ok(r)
            iid = r.get("id")
        if not ok:
            bad.append(str(iid))
    return bad


def _protocol_record(gens: Path, key: str) -> dict:
    """The unit's runner summaries' protocol fields (what the digest covers)."""
    from stages.teacher_parent_initialization import b_evidence as BE
    p = gens / f"{key}.summary.json"
    if not p.is_file():
        return {"summary": False, "runs": []}
    runs = BE.runner_summaries(json.loads(p.read_text()))
    return {"summary": True, "runs": [{k: v for k, v in BE.protocol_fields(r, key).items() if k not in NOT_PROTOCOL}
                                      for r in runs]}


def packages(d: Path) -> list[dict]:
    p = Path(d) / PACKAGES
    return json.loads(p.read_text()) if p.is_file() else []


#: a sealed part's name (`reasoning_inheritance.driver.PART_NAME`): its number and the generation that sealed it
PART_NAME = re.compile(r"^part_(\d{4,})(?:\.([A-Za-z0-9_-]+))?\.jsonl$")


def _parts(d: Path) -> list[tuple[str, str, Path]]:
    """(unit, path relative to the set, path) of every sealed part of the registered units."""
    from stages.teacher_parent_initialization import b_evidence as BE
    d, out = Path(d), []
    for key in BE.registered_units():
        root = d / "parts" if key == BE.FLUID else d / "parts" / key
        for p in sorted(root.glob("part_*.jsonl")) if root.is_dir() else ():
            out.append((key, str(p.relative_to(d)), p))
    return out


def part_inventory(d: Path) -> dict[str, str]:
    """Every sealed part now in the set, by relative path, with its sha256 (a package record keeps it: which parts
    existed before that generation ran)."""
    return {rel: sha256_file(p) for _k, rel, p in _parts(d)}


def _generation_problem(e: dict | None, gen: str | None, target: str, mode: str, weights, proto: str) -> str | None:
    if gen is None:
        return "the part names no generation (sealed before parts carried one)"
    if e is None:
        return f"no package record names generation {gen}"
    if weights is None:
        return "the weights this target must be evaluated from are unknown (no COMPLETE init record)"
    if e.get("weights_sha256") != weights:
        return f"generation {gen} ran other weights {e.get('weights_sha256')}"
    if e.get("target") != target or e.get("mode") != mode:
        return f"generation {gen} is {e.get('target')}/{e.get('mode')}'s, not {target}/{mode}'s"
    if e.get("protocol_sha256") != proto:
        return f"generation {gen} ran protocol {str(e.get('protocol_sha256'))[:12]}, not the registered one"
    return None


def provenance(d: Path, target: str, mode: str, out: Path) -> dict:
    """Each sealed part's provenance: the generation it names, that generation's package record (weights, target,
    mode, protocol), and the part's own records (registered items, well formed, the per-record protocol). Returns
    ``parts`` (per relative path: unit, generation, sha256, records, ``proven``, ``why``), ``proven_records`` (per
    unit, by item id) and ``sealed_twice`` (an item two proven parts hold)."""
    from stages.teacher_parent_initialization import analysis as TA
    from stages.teacher_parent_initialization import b_evidence as BE
    d = Path(d)
    entries = {e.get("generation"): e for e in packages(d) if e.get("generation")}
    weights, want = expected_weights(target, out), expected(mode)
    proto, stops, units = protocol_sha256(mode), teacher_stop_ids(), BE.registered_units()
    parts, proven, twice = {}, {k: {} for k in units}, []
    for key, rel, p in _parts(d):
        m = PART_NAME.match(p.name)
        gen = m.group(2) if m else None
        rows = _jl(p)
        why = _generation_problem(entries.get(gen), gen, target, mode, weights, proto)
        field, ids = units[key]
        got = [str(r.get(field) if field in r else r.get("id")) for r in rows]
        if why is None and (not rows or set(got) - set(ids)):
            why = f"holds {sorted(set(got) - set(ids))[:3] or 'no'} records of registered items"
        if why is None and _wellformed(key, rows):
            why = f"{len(_wellformed(key, rows))} records without raw output or termination"
        if why is None:
            kind = BE.kind_of(key)
            with tempfile.TemporaryDirectory() as tmp:
                shutil.copyfile(p, Path(tmp) / f"{key}.jsonl")
                item = TA.item_protocol(Path(tmp), None, key, {**want[kind], "stop_ids": stops}, want["think_state"])
            disagree = {n: v["first"] for n, v in (item.get("checks") or {}).items() if v["disagree"]}
            if disagree or not item.get("agrees"):
                why = f"its records are not the registered protocol: {disagree or item.get('unverifiable')}"
        parts[rel] = {"unit": key, "generation": gen, "sha256": sha256_file(p), "records": len(rows),
                      "proven": why is None, "why": why}
        if why is None:
            for i, r in zip(got, rows):
                if i in proven[key]:
                    twice.append(f"{key}/{i}")
                proven[key][i] = r
    return {"parts": parts, "proven_records": proven, "sealed_twice": sorted(set(twice))}


def check(d: Path, target: str, mode: str, out: Path, *, label: str | None = None) -> dict:
    """The five conditions of the module docstring for the set in ``d``. Returns ``complete``, ``problems`` (by
    condition) and, per unit, what a repair needs. The formal report is rescored here (``rescored``: the report these
    generations give; `repair` writes it)."""
    import shared.evaluation.eval_set_scoring as S
    from stages.teacher_parent_initialization import analysis as TA
    from stages.teacher_parent_initialization import b_evidence as BE
    d, gens = Path(d), Path(d) / "generations"
    want = expected(mode)
    units = BE.registered_units()
    comp = BE.completeness(gens, units) if gens.is_dir() else {k: {"present": False, "complete": False}
                                                               for k in units}
    problems: dict[str, list[str]] = {"coverage": [], "records": [], "protocol": [], "formal_report": [],
                                      "model_identity": []}
    per_unit, digest_body = {}, {}
    prov = provenance(d, target, mode, out)
    for key, c in comp.items():
        u = {"present": c.get("present", False), "complete": c.get("complete", False)}
        if not u["present"]:
            problems["coverage"].append(f"{key}: no generations")
            per_unit[key] = {**u, "action": "generate"}
            continue
        if not u["complete"]:
            problems["coverage"].append(f"{key}: {c['n_missing']} missing, {c['n_extra']} extra, "
                                        f"{c['n_duplicated']} repeated")
        rows = _jl(gens / f"{key}.jsonl")
        malformed = _wellformed(key, rows)
        if malformed:
            problems["records"].append(f"{key}: {len(malformed)} records without raw output or termination "
                                       f"({malformed[:3]})")
        prot = _protocol_record(gens, key)
        kind = BE.kind_of(key)
        reg = dict(want[kind])
        miss, differs = [], []
        if not prot["summary"]:
            miss.append("no summary")
        elif not prot["runs"]:
            miss.append("no runner summary")
        for i, run in enumerate(prot["runs"]):
            for f in BE.KEY_FIELDS[kind]:
                if f == "tasks_sha256":
                    continue        #: a resumed run submits only the missing tasks: checked on the sealed summary
                got = run.get(f, BE.NOT_RECORDED)
                if f == "seed_namespace" and kind == "capability":
                    got = (got or {}).get("derived") if isinstance(got, dict) else got
                if BE.empty(got, f):
                    miss.append(f"run {i}: {f} not recorded")
                elif f in reg and got != reg[f]:
                    differs.append(f"run {i}: {f} {got!r} != {reg[f]!r}")
        if kind == "fluid" and prot["summary"]:
            got_tasks = json.loads((gens / f"{key}.summary.json").read_text()).get("tasks_sha256")
            if BE.empty(got_tasks, "tasks_sha256"):
                miss.append("tasks_sha256 not recorded")
            elif got_tasks != reg["tasks_sha256"]:
                differs.append(f"tasks_sha256 {got_tasks!r} != {reg['tasks_sha256']!r}")
        stops = sorted({tuple(r.get("stop_ids") or ()) for r in prot["runs"]})
        ref = {**reg, "stop_ids": list(stops[0]) if len(stops) == 1 else None}
        item = TA.item_protocol(gens, None, key, ref, want["think_state"])
        disagree = {n: v["disagree"] for n, v in (item.get("checks") or {}).items() if v["disagree"]}
        if disagree or not item.get("agrees"):
            problems["records"].append(f"{key}: per-record protocol {disagree or 'not verifiable'} "
                                       f"(unverifiable: {item.get('unverifiable')})")
        if miss:
            problems["protocol"].append(f"{key}: missing {miss[:4]}")
        if differs:
            problems["protocol"].append(f"{key}: differs {differs[:4]}")
        #: what the protocol IS, not how many runs produced the unit: a resumed unit's runs are one protocol, and the
        #: task file a resumed run submits (its missing tasks only) is checked on the sealed summary above
        digest_body[key] = sorted({json.dumps({k: v for k, v in r.items() if k != "tasks_sha256"}, sort_keys=True,
                                              default=str) for r in prot["runs"]})
        field, held = units[key][0], prov["proven_records"][key]
        unproven = sum(1 for r in rows if held.get(str(r.get(field) if field in r else r.get("id"))) != r)
        if unproven:
            problems["model_identity"].append(f"{key}: {unproven} of its {len(rows)} records no proven part holds")
        action = None
        if not u["complete"] or unproven:
            action = "recompose"            #: missing, extra, repeated or unproven records: the proven parts, then the rest
        if malformed or miss or differs or disagree or not item.get("agrees"):
            action = "regenerate"           #: records or protocol that cannot be shown: the unit again, same seeds
        per_unit[key] = {**u, "generations_sha256": c.get("generations_sha256"), "records": len(rows),
                         "malformed": len(malformed), "protocol_missing": miss, "protocol_differs": differs,
                         "records_no_proven_part_holds": unproven,
                         "per_record": {"agrees": item.get("agrees"), "disagree": disagree,
                                        "unverifiable": item.get("unverifiable")},
                         "action": action}
    # model identity: every sealed part's generation, weights, mode and protocol (with or without a generations file)
    pk = packages(d)
    weights = expected_weights(target, out)
    if weights is None:
        problems["model_identity"].append(f"{target} has no COMPLETE init record in {out}")
    bad_parts = {rel: v["why"] for rel, v in prov["parts"].items() if not v["proven"]}
    if bad_parts:
        problems["model_identity"].append(f"{len(bad_parts)} sealed parts whose provenance is wrong or unverifiable "
                                          f"({sorted(set(bad_parts.values()))[:3]})")
    if prov["sealed_twice"]:
        problems["records"].append(f"items sealed twice in proven parts: {prov['sealed_twice'][:5]}")
    # the formal report: rescored from these generations
    rescored, stored = None, None
    if all(per_unit[k]["present"] for k in per_unit):
        rescored = S.report(gens, label or target, {}, C.design()["evaluation"]["protocol"]["eval_set"])
        p = d / "formal_report.json"
        stored = json.loads(p.read_text()) if p.is_file() else None
        strip = lambda r: {k: v for k, v in (r or {}).items() if k != "created_utc"}  # noqa: E731
        if stored is None or strip(stored) != strip(rescored):
            problems["formal_report"].append("absent" if stored is None else
                                             "empty ({})" if stored == {} else "not the report of these generations")
        bad = {k: v.get("status") for k, v in rescored["units"].items()
               if v.get("status") != "measured" or v.get("missing")}
        if bad:
            problems["formal_report"].append(f"units not measured whole: {bad}")
    else:
        problems["formal_report"].append("not scorable: units without generations")
    digest = hashlib.sha256(json.dumps(digest_body, sort_keys=True, default=str).encode()).hexdigest()
    complete = not any(problems.values())
    return {"schema": SCHEMA, "target": target, "mode": mode, "out": str(out), "complete": complete,
            "problems": {k: v for k, v in problems.items() if v}, "units": per_unit,
            "protocol_digest": digest, "packages": [{k: e.get(k) for k in ("generation", "attempt", "utc",
                                                                             "weights_sha256")} for e in pk],
            "parts": prov["parts"], "sealed_twice": prov["sealed_twice"],
            "expected_weights": weights, "think_state": want["think_state"], "rescored": rescored,
            "stored_report_matches": not problems["formal_report"]}


def brief(st: dict) -> dict:
    """A check without its rescored report (what records keep)."""
    return {k: v for k, v in st.items() if k != "rescored"}


def write(d: Path, st: dict) -> None:
    from stages.reasoning_inheritance.driver import write_json
    write_json(Path(d) / COMPLETION, brief(st))


# ---- the repair ---------------------------------------------------------------------------------------------------

def _supersede(d: Path, rel: str, attempt: str) -> str | None:
    src = Path(d) / rel
    if not src.exists():
        return None
    dest = Path(d) / SUPERSEDED / attempt / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))
    return rel


def _proven_generation(e: dict, target: str, mode: str, out: Path) -> bool:
    return _generation_problem(e, e.get("generation"), target, mode, expected_weights(target, out),
                               protocol_sha256(mode)) is None


def repair(d: Path, st: dict, attempt: str) -> list[dict]:
    """The smallest actions for what `check` found (module docstring), under ``superseded/<attempt>/``; returns what
    was done. A stale report needs nothing here (the caller writes the rescored one). Nothing proven is moved and
    nothing is chosen by its outcome; ``packages.json`` stays whole (it is what the proven parts are read against)."""
    from stages.reasoning_inheritance.driver import write_json
    from stages.teacher_parent_initialization import b_evidence as BE
    d, done = Path(d), []
    if st.get("sealed_twice"):
        raise CompletionError(f"{d.name}: items sealed twice in proven parts {st['sealed_twice'][:5]}; nothing here "
                              "chooses between them")
    isolated = {rel: v for rel, v in (st.get("parts") or {}).items() if not v["proven"]}
    for rel, v in isolated.items():
        _supersede(d, rel, attempt)
    if isolated:
        write_json(d / SUPERSEDED / attempt / "isolated_parts.json", isolated)
        done.append({"isolated_parts": len(isolated), "kept_parts": len(st["parts"]) - len(isolated),
                     "why": sorted({v["why"] for v in isolated.values()})})
    for key, u in st["units"].items():
        action = u.get("action")
        if action == "recompose":
            moved = [_supersede(d, f"generations/{key}{s}", attempt) for s in (".jsonl", ".summary.json")]
            done.append({"unit": key, "superseded": [m for m in moved if m],
                         "why": "composed again from the proven parts; the items none holds are generated"})
        elif action == "regenerate":
            moved = [_supersede(d, f"generations/{key}{s}", attempt) for s in (".jsonl", ".summary.json")]
            moved += [_supersede(d, rel, attempt) for k, rel, _p in _parts(d) if k == key]
            done.append({"unit": key, "superseded": [m for m in moved if m],
                         "why": "its runner protocol or records cannot be shown: generated again under the same seeds"})
    #: the fluid unit's run history keeps only the runs of proven generations (a run whose parts were isolated must
    #: not stand in the recomposed unit's summary)
    runs_p = d / "fluid_runs.json"
    if runs_p.is_file() and (any(v["unit"] == BE.FLUID for v in isolated.values())
                             or (st["units"].get(BE.FLUID) or {}).get("action")):
        entries = {e.get("generation"): e for e in packages(d) if e.get("generation")}
        runs = json.loads(runs_p.read_text())
        keep = [r for r in runs if r.get("attempt") in entries
                and _proven_generation(entries[r["attempt"]], st["target"], st["mode"], st["out"])]
        if len(keep) != len(runs):
            _supersede(d, "fluid_runs.json", attempt)
            if keep:
                write_json(runs_p, keep)
            done.append({"unit": BE.FLUID, "fluid_runs_kept": len(keep), "fluid_runs_superseded": len(runs) - len(keep)})
    return done


def needs_generation(d: Path) -> bool:
    from stages.teacher_parent_initialization import b_evidence as BE
    gens = Path(d) / "generations"
    return any(not (gens / f"{k}.jsonl").is_file() for k in BE.registered_units())


def rescore(d: Path, st: dict) -> None:
    """Write the report `check` rescored from these generations (a stale or ``{}`` report is replaced)."""
    from stages.reasoning_inheritance.driver import write_json
    if st.get("rescored") is not None:
        write_json(Path(d) / "formal_report.json", st["rescored"])


# ---- the phase ----------------------------------------------------------------------------------------------------

def init_problems(out: Path) -> dict[str, str]:
    """Each arm product (``<arm>/init``, ``<arm>/mapping``) whose init record in ``out`` does not show it durable:
    its manifest the required files, its relay copy verified with exactly them (`common.product_problems`)."""
    bad = {}
    for arm in C.design()["phase_1"]["build_order"]:
        p = Path(out) / arm / "init_record.json"
        bad.update(C.product_problems(json.loads(p.read_text()) if p.is_file() else None, arm))
    return bad


def mode_problems(sets: dict[tuple[str, str], dict]) -> list[str]:
    """The models of one mode must share one protocol digest."""
    bad = []
    for mode in sorted({m for _, m in sets}):
        digests = {t: st["protocol_digest"] for (t, m), st in sets.items() if m == mode}
        if len(set(digests.values())) > 1:
            bad.append(f"{mode}: the models were read under different protocols {digests}")
    return bad


def phase(out: Path, *, write_records: bool = False) -> dict:
    """Phase 1's completion on the files in ``out`` (a pod's evidence directory or the dev box's copy): the three
    initializations durable, the eight sets complete, one protocol per mode."""
    sets = {}
    for t in targets():
        for m in C.mode_dirs():
            st = check(ev_dir(out, t, m), t, m, out, label=t if t == TEACHER else f"{t}_untrained")
            if write_records and ev_dir(out, t, m).is_dir():
                write(ev_dir(out, t, m), st)
            sets[(t, m)] = st
    inits = init_problems(out)
    modes = mode_problems({k: v for k, v in sets.items() if v["complete"]})
    complete = not inits and not modes and all(v["complete"] for v in sets.values())
    return {"schema": SCHEMA.replace("evaluation_completion", "phase1_completion"), "complete": complete,
            "inits": {"problems": inits, "durable": not inits},
            "sets": {f"{t}/{m}": {"complete": v["complete"], "problems": v["problems"],
                                  "protocol_digest": v["protocol_digest"]} for (t, m), v in sets.items()},
            "modes": modes, "n_sets": len(sets), "n_complete": sum(v["complete"] for v in sets.values())}
