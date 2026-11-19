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
* **model identity** -- every package an attempt built for this set (``packages.json``, appended before anything is
  generated) names the expected weights: the arm's committed init record's ``model.safetensors`` sha256, or the
  teacher identity's three shards.

`repair` turns what a check found into the smallest action: a stale report is rescored; a generations file with
missing, extra or (identically) repeated items gives its valid records to the sealed parts and is superseded, so the
runners generate only the items no sealed record holds; a unit whose protocol or model cannot be shown is superseded
whole (records and parts kept under ``superseded/<attempt>/`` as evidence, never pooled) and generated again under the
same per-item seeds. Two different records sealed for one item are a defect nothing here resolves: refused.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
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
        digest_body[key] = prot["runs"]
        action = None
        if not u["complete"]:
            action = "recompose"            #: missing, extra or repeated items: the sealed records, then the rest
        if malformed or miss or differs or disagree or not item.get("agrees"):
            action = "regenerate"           #: records or protocol that cannot be shown: the unit again, same seeds
        per_unit[key] = {**u, "generations_sha256": c.get("generations_sha256"), "records": len(rows),
                         "malformed": len(malformed), "protocol_missing": miss, "protocol_differs": differs,
                         "per_record": {"agrees": item.get("agrees"), "disagree": disagree,
                                        "unverifiable": item.get("unverifiable")},
                         "action": action}
    # model identity
    pk = packages(d)
    weights = expected_weights(target, out)
    if weights is None:
        problems["model_identity"].append(f"{target} has no COMPLETE init record in {out}")
    elif any(per_unit[k]["present"] for k in per_unit):
        if not pk:
            problems["model_identity"].append("no package record: the weights these generations came from are unknown")
        for e in pk:
            if e.get("weights_sha256") != weights:
                problems["model_identity"].append(f"attempt {e.get('attempt')}: package weights "
                                                  f"{e.get('weights_sha256')} are not {weights}")
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
    return {"schema": SCHEMA, "target": target, "mode": mode, "complete": complete,
            "problems": {k: v for k, v in problems.items() if v}, "units": per_unit,
            "protocol_digest": digest, "packages": [{k: e.get(k) for k in ("attempt", "utc", "weights_sha256")}
                                                    for e in pk],
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


def _import_into_parts(d: Path, key: str, attempt: str) -> dict:
    """The generations file's records that no sealed part holds, given to the parts as one more sealed part: a
    registered item, once (identical repeats are one record), never a second, different record for an item."""
    from stages.teacher_parent_initialization import b_evidence as BE
    d = Path(d)
    field, ids = BE.registered_units()[key]
    reg = set(ids)
    fluid = key == BE.FLUID
    parts = d / "parts" if fluid else d / "parts" / key
    sealed = {}
    for p in sorted(parts.glob("part_*.jsonl")) if parts.is_dir() else ():
        for r in _jl(p):
            sealed[str(r.get(field) if field in r else r.get("id"))] = r
    take, conflicts = {}, []
    for r in _jl(d / "generations" / f"{key}.jsonl"):
        i = str(r.get(field) if field in r else r.get("id"))
        if i not in reg:
            continue
        for have in (sealed.get(i), take.get(i)):
            if have is not None and have != r:
                conflicts.append(i)
        if i not in sealed:
            take[i] = r
    if conflicts:
        raise CompletionError(f"{d.name}/{key}: two different records are sealed for {sorted(set(conflicts))[:5]}; "
                              "nothing here chooses between them")
    if take:
        parts.mkdir(parents=True, exist_ok=True)
        n = len(list(parts.glob("part_*.jsonl")))
        tmp = parts / f".part_{n:04d}.partial"
        tmp.write_text("".join(json.dumps(take[i], ensure_ascii=False) + "\n" for i in sorted(take)), encoding="utf-8")
        tmp.replace(parts / f"part_{n:04d}.jsonl")
    return {"unit": key, "records_given_to_parts": len(take), "attempt": attempt}


def repair(d: Path, st: dict, attempt: str) -> list[dict]:
    """The smallest actions for what `check` found (module docstring); returns what was done. A stale report needs
    nothing here (the caller writes the rescored one); a package that names other weights supersedes every unit."""
    from stages.teacher_parent_initialization import b_evidence as BE
    d, done = Path(d), []
    wrong_model = any("are not" in p or "unknown" in p for p in st["problems"].get("model_identity", []))
    for key, u in st["units"].items():
        action = "regenerate" if (wrong_model and u.get("present")) else u.get("action")
        if action == "recompose":
            done.append(_import_into_parts(d, key, attempt))
            moved = [_supersede(d, f"generations/{key}{s}", attempt) for s in (".jsonl", ".summary.json")]
            done.append({"unit": key, "superseded": [m for m in moved if m], "why": "recomposed from sealed parts"})
        elif action == "regenerate":
            parts = "parts" if key == BE.FLUID else f"parts/{key}"
            moved = [_supersede(d, f"generations/{key}{s}", attempt) for s in (".jsonl", ".summary.json")]
            if key == BE.FLUID:
                moved += [_supersede(d, f"parts/{p.name}", attempt) for p in sorted((d / "parts").glob("part_*.jsonl"))]
                moved.append(_supersede(d, "fluid_runs.json", attempt))
            else:
                moved.append(_supersede(d, parts, attempt))
            done.append({"unit": key, "superseded": [m for m in moved if m],
                         "why": "its protocol or model cannot be shown: generated again under the same seeds"})
    if wrong_model or (not st["packages"] and any(u.get("present") for u in st["units"].values())):
        moved = _supersede(d, PACKAGES, attempt)
        if moved:
            done.append({"superseded": [moved], "why": "the package record names other weights"})
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
    """Each arm without a COMPLETE init record whose relay copies all verified, in ``out``."""
    bad = {}
    for arm in C.design()["phase_1"]["build_order"]:
        p = Path(out) / arm / "init_record.json"
        rec = json.loads(p.read_text()) if p.is_file() else None
        if rec is None:
            bad[arm] = "no init record"
        elif rec.get("status") != "COMPLETE" or not all(v.get("verified") for v in (rec.get("relay") or {}).values()):
            bad[arm] = "init record not COMPLETE or a relay copy unverified"
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
