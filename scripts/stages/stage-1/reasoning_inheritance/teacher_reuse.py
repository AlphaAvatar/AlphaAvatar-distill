#!/usr/bin/env python3
"""Group T's evaluation, reused from the teacher run when -- and only when -- it is this round's protocol ($0, CPU).

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python \\
        scripts/stages/stage-1/reasoning_inheritance/teacher_reuse.py [--out logs/.../analyses/teacher_reuse.json]

`design.json :: evaluation.teacher_reuse` names the two archives of ``a_teacher_20261030_012648``'s H-T / H-N
evaluation by sha256 (the values its own session record secured). For each mode this

1. re-hashes the archive and copies it into ``artifacts/stages/stage-1/reasoning_inheritance/teacher_eval/`` (its
   ``.scratch`` home is disposable), unpacks it there and re-hashes every file against the unit descriptor;
2. checks every unit's recorded identity against THIS round's: the root snapshot at the pinned revision, the mode's
   template switch, the system message, the tokenizer and chat-template hashes, the stop ids, the 40,960-token
   context, the ``nucleus_v1`` decoder file and the unit's seed namespace, the action protocol, the degeneration
   check, and the eval-set item files the unit read;
3. checks every unit item by item (`item_checks`): the records cover the registered unit exactly (no item missing,
   extra or repeated), every request's seed is the decoding protocol's rule for its unit, item, turn and sample, and
   each item's prompt FIELDS equal this round's for the same item in every group that holds the unit (``--store``):
   the fluid unit's first user message (text, also compared with the registered item) and turn-0 prompt length; every
   other unit's prompt length in tokens, system source and think pre-opening, also against the registered items
   rendered on the dev box with the teacher's verified tokenizer. A field absent or invalid on either side is
   counted ``not recorded`` and never as equal (``think_preopened`` ``False`` is a value, not a gap);
4. reports the prompts' CONTENT identity apart from those fields (`content_identity`): verified from the text for the
   fluid unit; for the others, whose records carry no prompt text or ids, derived from the construction path --
   the same registered item files and the same rendering statements at the teacher's and this round's session
   commits (`construction_path`) -- and never from equal lengths alone;
5. rescores the generations with the CURRENT scorer (`shared.evaluation.eval_set_scoring.report`), which is the
   report the analysis reads.

A failed check, or a content identity not established, refuses the reuse: T is then a group to be generated, never a teacher result with a caveat. A field
the teacher's records never carried is listed under ``not_recorded`` with what stands in for it; it is never
counted as a passed check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

DESIGN = REPO / "configs/stages/stage-1/reasoning_inheritance/design.json"
DEST = REPO / "artifacts/stages/stage-1/reasoning_inheritance/teacher_eval"
OUT = REPO / "logs/stages/stage-1/reasoning_inheritance/analyses/teacher_reuse.json"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def unit_checks(name: str, s: dict, *, mode: str, d: dict, ident: dict, eval_files: dict) -> dict:
    """Each identity field of one unit's summary against this round's protocol."""
    ev = d["evaluation"]
    idn = s.get("identity") if "context_resolution" in (s.get("identity") or {}) else s
    dec = (idn.get("decoding_protocol") or {})
    model = str(idn.get("model") or "")
    stem = name.removesuffix(".summary.json")
    want_tasks = eval_files.get(stem)
    got_tasks = idn.get("tasks_sha256") or ((idn.get("prompts") or {}).get("sha256") if isinstance(idn.get("prompts"), dict) else None)
    c = {
        "root_revision": d["teacher"]["revision"] in model and "Qwen--Qwen3-4B/" in model,
        "context_40960": int((idn.get("context_resolution") or {}).get("resolved_context") or idn.get("context_len") or 0)
                         == int(ev["context"]["tokens"]),
        "decoder_nucleus_v1": dec.get("protocol_id") == "nucleus_v1" and dec.get("sha256") == sha(REPO / ev["decoding"]),
        "template_switch": (idn.get("chat_template_kwargs") or {}) == ev["modes"][mode]["template_kwargs"],
        "system_message": idn.get("system_message") == ev["system_message"],
        "chat_template_sha256": idn.get("chat_template_sha256") == ident["chat_template_sha256"],
        "tokenizer_sha256": idn.get("tokenizer_sha256") == ident["tokenizer_files_sha256"],
        "stop_ids": sorted(idn.get("stop_ids") or []) == sorted(ident["eos_token_ids"]),
        "degeneration_check_every": int(idn.get("degeneration_check_every") or
                                        (idn.get("sampling") or {}).get("degeneration_check_every") or 0)
                                    == int(ev["degeneration_check_every"]),
    }
    if stem.startswith("fluid_reasoning_720"):
        c["seed_namespace"] = idn.get("seed_namespace") == stem
        c["action_protocol"] = idn.get("action_protocol") == ev["action_protocol"]
    if want_tasks is not None and got_tasks is not None:
        c["eval_set_file"] = got_tasks == want_tasks
    return c


def registered_ids(eval_set: str, stem: str) -> list[str]:
    from shared import eval_sets as ES
    comp, unit = stem.split("__", 1)
    key = "task_id" if comp == "fluid_reasoning_720" else "id"
    return [it[key] for it in ES.items(eval_set, comp, unit)]


def this_rounds(stores: list[Path], mode: str, stem: str) -> dict:
    """Each of D, W and DW's records of the unit in this round (the last store holding it wins), by item id."""
    out = {}
    for g in ("D", "W", "DW"):
        for st in reversed(stores):
            f = st / "out" / "eval" / f"{g}_{mode}" / "generations" / f"{stem}.jsonl"
            if f.is_file():
                out[g] = {(r.get("task_id") or r["id"]): r
                          for r in (json.loads(l) for l in f.read_text().split("\n") if l.strip())}
                break
    return out


#: what one record's prompt is compared by, and what a valid value of each field is. A field that is absent or
#: invalid on either side is ``not recorded`` for that item: never equal, never a pass. ``False`` is a valid
#: ``think_preopened`` and is told apart from a missing one.
def _pos_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v > 0


def _text(v) -> bool:
    return isinstance(v, str) and bool(v.strip())


PROMPT_FIELDS = {
    #: the fluid unit: the prompt's text (its first user message) and the turn-0 prompt length
    True: {"first_user_message": _text, "prompt_tokens": _pos_int},
    #: every other unit: the records carry the prompt's length and two properties of it, never its text
    False: {"prompt_tokens": _pos_int, "system_source": lambda v: v in ("sample", "default", "none"),
            "think_preopened": lambda v: isinstance(v, bool)},
}


def prompt_fields(r: dict, fluid: bool) -> tuple[dict, list[str]]:
    """The record's prompt fields, and the names of those that are absent or invalid (never compared)."""
    if fluid:
        first = next((m.get("content") for m in r.get("messages") or [] if m.get("role") == "user"), None)
        t0 = (r.get("turns") or [{}])[0]
        vals = {"first_user_message": first, "prompt_tokens": t0.get("prompt_tokens")}
    else:
        vals = {k: r.get(k) for k in PROMPT_FIELDS[False]}
    return vals, [k for k, ok in PROMPT_FIELDS[fluid].items() if not ok(vals.get(k))]


def compare_prompts(theirs: dict, ours: dict, fluid: bool, fields: tuple[str, ...] | None = None) -> dict:
    """Item by item, the teacher's prompt fields against one other side's (``fields``: a subset to compare).

    ``compared`` counts the items where both sides carry every field validly; an item missing on either side, or
    with a field missing or invalid, is counted apart and never as equal."""
    keys = tuple(fields or PROMPT_FIELDS[fluid])
    out = {"fields": list(keys), "items": len(theirs), "compared": 0, "equal": 0, "differ": 0,
           "absent_on_the_other_side": 0, "not_recorded_teacher": 0, "not_recorded_other": 0, "fields_missing": {}}
    for i, r in theirs.items():
        o = ours.get(i)
        if o is None:
            out["absent_on_the_other_side"] += 1
            continue
        a, bad_a = r if isinstance(r, tuple) else prompt_fields(r, fluid)
        b, bad_b = o if isinstance(o, tuple) else prompt_fields(o, fluid)
        bad_a = [k for k in bad_a if k in keys]
        bad_b = [k for k in bad_b if k in keys]
        for k in bad_a + bad_b:
            out["fields_missing"][k] = out["fields_missing"].get(k, 0) + 1
        if bad_a or bad_b:
            out["not_recorded_teacher"] += int(bool(bad_a))
            out["not_recorded_other"] += int(bool(bad_b))
            continue
        out["compared"] += 1
        same = all(a[k] == b[k] for k in keys)
        out["equal"] += int(same)
        out["differ"] += int(not same)
    #: the check passes only when every item was compared on every field and none differs
    out["passes"] = out["items"] > 0 and out["compared"] == out["items"] and out["differ"] == 0
    return out


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)


#: the inputs a non-fluid unit's prompts were built from, at both commits: the registered item files and their
#: verified reader, and the staging code (``dose_driver.evaluate_all`` / ``legacy_prompts`` for the teacher; this
#: round's ``driver.capability`` copies the same ``ES.unit_path`` file and calls the same ``legacy_prompts``)
PROMPT_INPUTS = ("data/eval-set", "scripts/shared/eval_sets.py", "src/aadistill/evaluation/eval_registry.py",
                 "scripts/stages/stage-0/calibration_volume/dose_driver.py")
RUNNER = "scripts/shared/evaluation/uncapped_eval.py"


def render_block(commit: str) -> str | None:
    """uncapped_eval's statements that turn a staged sample into its prompt ids and prompt fields, at ``commit``:
    from the sample loop to the recorded ``think_preopened``."""
    src = _git("show", f"{commit}:{RUNNER}")
    if src.returncode != 0:
        return None
    lines = src.stdout.splitlines()
    try:
        a = next(i for i, l in enumerate(lines) if l.strip() == "for s in samples:")
        b = next(i for i in range(a, len(lines)) if '"think_preopened": preopened' in lines[i])
    except StopIteration:
        return None
    return "\n".join(lines[a:b + 1])


def construction_path(teacher_commit: str | None, our_commits: list[str]) -> dict:
    """Whether the teacher's and this round's non-fluid prompts were built from the same inputs by the same
    statements: a derivation from the commits that ran, not a record of the prompts themselves."""
    out = {"teacher_session_commit": teacher_commit, "this_rounds_session_commits": our_commits,
           "inputs_unchanged": {}, "render_statements_unchanged": {}}
    if not teacher_commit or not our_commits:
        out["established"] = False
        out["why_not"] = "a session commit is not recorded"
        return out
    ref = render_block(teacher_commit)
    for c in our_commits:
        out["inputs_unchanged"][c] = {p: _git("diff", "--quiet", teacher_commit, c, "--", p).returncode == 0
                                      for p in PROMPT_INPUTS}
        mine = render_block(c)
        out["render_statements_unchanged"][c] = ref is not None and mine is not None and ref == mine
    out["render_block_sha256"] = hashlib.sha256((ref or "").encode()).hexdigest() if ref else None
    out["established"] = all(all(v.values()) for v in out["inputs_unchanged"].values()) and all(
        out["render_statements_unchanged"].values())
    return out


def registered_samples(eval_set: str, stem: str, scratch: Path) -> list[dict]:
    """The samples a non-fluid unit's runner reads, staged exactly as both rounds staged them."""
    from shared import eval_sets as ES
    from stages.calibration_volume.dose_driver import legacy_prompts
    comp, unit = stem.split("__", 1)
    if comp == "stage3_legacy_150":
        p = legacy_prompts(scratch / "legacy_150.prompts.jsonl")
    else:
        p = ES.unit_path(eval_set, comp, unit)
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def rendered_fields(samples: list[dict], tok, template_kwargs: dict, system: str) -> dict:
    """Each registered sample rendered on the dev box by the runner's statements (`render_block`, checked unchanged):
    its prompt length, system source and think pre-opening, and the sha256 of its prompt ids."""
    from shared.evaluation import uncapped_eval as UE
    from aadistill.evaluation.behavior import template_opens_think
    out = {}
    for s in samples:
        turns = [m for m in s["messages"] if m["role"] != "assistant"]
        has_system = any(m.get("role") == "system" for m in turns)
        inject = bool(system) and not has_system
        if inject:
            turns = [{"role": "system", "content": system}] + turns
        prompt = tok.apply_chat_template(turns, tools=UE.tool_schemas(s), tokenize=False, add_generation_prompt=True,
                                         **template_kwargs)
        ids = tok(prompt, add_special_tokens=False).input_ids
        out[s["id"]] = ({"prompt_tokens": len(ids),
                         "system_source": "sample" if has_system else ("default" if inject else "none"),
                         "think_preopened": template_opens_think(prompt),
                         "prompt_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()}, [])
    return out


def content_identity(fluid: bool, text: dict | None, path: dict | None, rendered: dict | None) -> dict:
    """What the evidence says about the prompts' CONTENT, apart from their length.

    * ``verified_text``: the fluid records carry the prompt's text and it equals this round's and the registered
      item's;
    * ``derived_not_recorded``: the records carry no prompt text; the same registered item file and the same
      rendering statements ran at both commits (`construction_path`), and every recorded length, system source and
      think pre-opening equals the registered item's rendering. A derivation, not a byte comparison of the prompts;
    * ``not_established``: anything less. Equal lengths alone never establish it."""
    if fluid:
        ok = bool(text) and all(v["passes"] for v in text.values())
        return {"status": "verified_text" if ok else "not_established",
                "basis": "the first user message's text, against this round's and the registered item's"}
    ok = bool(path and path.get("established")) and bool(rendered) and all(v["passes"] for v in rendered.values())
    return {"status": "derived_not_recorded" if ok else "not_established",
            "basis": ("the same registered item file and the same rendering statements at the teacher's and this "
                      "round's session commits (construction_path), and every recorded prompt length, system source "
                      "and think pre-opening equal to the registered item's rendering on the dev box"),
            "limit": ("the records hold no prompt text or prompt ids: content identity is derived from the commits "
                      "that ran, not compared; equal lengths alone would not establish it")}


def item_checks(gens: Path, stem: str, *, mode: str, eval_set: str, protocol: dict, stores: list[Path],
                path: dict | None = None, render: dict | None = None) -> dict:
    """Coverage, seeds, prompt fields and prompt content identity of one unit's teacher records, item by item."""
    from shared.evaluation import decoding as DEC
    recs = [json.loads(l) for l in (gens / f"{stem}.jsonl").read_text().split("\n") if l.strip()]
    ids = [(r.get("task_id") or r["id"]) for r in recs]
    reg = registered_ids(eval_set, stem)
    seen: dict = {}
    for i in ids:
        seen[i] = seen.get(i, 0) + 1
    fluid = stem.startswith("fluid_reasoning_720")
    seeds_checked = seed_bad = seed_missing = 0
    for r in recs:
        i = r.get("task_id") or r["id"]
        for t in (r["turns"] if fluid else [r]):
            want = DEC.request_seed(protocol, stem, i, int(t.get("turn") or 0) if fluid else 0, 0)
            got = t.get("seed") if fluid else t.get("sampling_seed")
            if not isinstance(got, int) or isinstance(got, bool):
                seed_missing += 1
                continue
            seeds_checked += 1
            seed_bad += int(got != want)
    ours = this_rounds(stores, mode, stem)
    by_id = {(r.get("task_id") or r["id"]): r for r in recs}
    against = {g: compare_prompts(by_id, theirs, fluid) for g, theirs in ours.items()}
    out = {"registered": len(reg), "records": len(recs), "missing": sorted(set(reg) - set(seen))[:5],
           "n_missing": len(set(reg) - set(seen)), "n_extra": len(set(seen) - set(reg)),
           "n_repeated": sum(1 for v in seen.values() if v > 1),
           "seeds_checked": seeds_checked, "seeds_off_rule": seed_bad, "seeds_not_recorded": seed_missing,
           "prompt_fields": list(PROMPT_FIELDS[fluid]),
           "prompt_against_this_round": against}
    text = None
    if fluid:
        from shared import eval_sets as ES
        reg_items = {t["task_id"]: t for t in ES.items(eval_set, "fluid_reasoning_720", stem.split("__", 1)[1])}
        src = compare_prompts(by_id, {i: ({"first_user_message": next(
            (m.get("content") for m in t.get("messages") or [] if m.get("role") == "user"), None)}, [])
            for i, t in reg_items.items()}, True, fields=("first_user_message",))
        out["source_prompt_against_the_registered_item"] = src
        text = {**against, "registered": src}
        rendered = None
    else:
        rendered = None
        if render is not None:
            fields = tuple(PROMPT_FIELDS[False])
            rendered = {"T": compare_prompts(by_id, render, False, fields=fields)}
            rendered.update({g: compare_prompts({i: theirs[i] for i in theirs}, render, False, fields=fields)
                             for g, theirs in ours.items()})
            out["rendered_prompt_ids_sha256"] = hashlib.sha256(json.dumps(
                sorted((i, v[0]["prompt_ids_sha256"]) for i, v in render.items())).encode()).hexdigest()
        out["prompt_against_the_registered_rendering"] = rendered if rendered is not None else (
            "not run: no verified teacher tokenizer on this machine")
    out["prompt_content_identity"] = content_identity(fluid, text, path, rendered)
    out["checks"] = {
        "items_cover_the_registered_unit": out["n_missing"] == 0 and out["n_extra"] == 0 and out["n_repeated"] == 0,
        "seeds_follow_the_rule": seeds_checked > 0 and seed_bad == 0 and seed_missing == 0,
        #: equal prompt FIELDS -- for the non-fluid units a length and two properties, never the content
        "prompt_fields_equal_this_rounds": bool(against) and all(v["passes"] for v in against.values())}
    if fluid:
        out["checks"]["source_prompts_equal_the_registered_items"] = src["passes"]
    elif rendered is not None:
        out["checks"]["prompt_fields_equal_the_registered_rendering"] = all(v["passes"] for v in rendered.values())
    #: content identity is reported, and required for reuse, apart from the field checks
    out["checks"]["prompt_content_identity_established"] = out["prompt_content_identity"]["status"] != "not_established"
    return out


LOCAL = REPO / "logs/stages/stage-1/reasoning_inheritance/storage/local_checkpoints.json"


def session_commit(run_dir: Path) -> str | None:
    f = run_dir / "runtime" / "session.json"
    return (json.loads(f.read_text()).get("setup_environment") or {}).get("SESSION_COMMIT") if f.is_file() else None


def capability_attempts(stores: list[Path]) -> set[str]:
    """The attempts that generated this round's non-fluid units (each group's ``capability_run.<attempt>.json``)."""
    out = set()
    for st in stores:
        for f in (st / "out" / "eval").glob("*/capability_run.*.json"):
            rec = json.loads(f.read_text())
            if rec.get("status") == "complete" and rec.get("attempt"):
                out.add(rec["attempt"])
    return out


def teacher_tokenizer(path: Path | None, ident: dict):
    """The teacher's tokenizer from its dev-box copy, only if its files and chat template are the identity's."""
    from shared.evaluation import uncapped_eval as UE
    if path is None and LOCAL.is_file():
        path = Path((json.loads(LOCAL.read_text()).get("teacher_tokenizer") or {}).get("path") or "")
    if path is None or not path.is_dir():
        return None, {"path": str(path) if path else None, "used": False, "why": "not on this machine"}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(path))
    files = UE.tokenizer_files_sha256(str(path)) == ident["tokenizer_files_sha256"]
    tmpl = hashlib.sha256((tok.chat_template or "").encode()).hexdigest() == ident["chat_template_sha256"]
    check = {"path": str(path), "tokenizer_files_equal": files, "chat_template_equal": tmpl, "used": files and tmpl}
    return (tok if check["used"] else None), check


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--store", action="append", default=[],
                    help="this round's secured evidence (<run scr>/store), for the prompt comparison; a later one wins")
    ap.add_argument("--tokenizer", default=None,
                    help="the teacher's tokenizer files on this machine (default: local_checkpoints.json's), checked "
                         "against the teacher identity before the registered items are rendered with it")
    a = ap.parse_args(argv)
    from shared.evaluation import decoding as DEC
    stores = [Path(x) for x in a.store]
    from shared import eval_sets as ES
    import shared.evaluation.eval_set_scoring as S
    d = json.loads(DESIGN.read_text())
    tr = d["evaluation"]["teacher_reuse"]
    ident = json.loads((REPO / d["teacher"]["identity"]).read_text())
    m = ES.manifest(d["evaluation"]["eval_set"])
    eval_files = {"fluid_reasoning_720__dev_v1": sha(ES.unit_path(d["evaluation"]["eval_set"], "fluid_reasoning_720",
                                                                  "dev_v1"))}
    for u in m["components"]["stage3_capability_846"]["units"]:
        p = ES.unit_path(d["evaluation"]["eval_set"], "stage3_capability_846", u)
        if p is not None:
            eval_files[f"stage3_capability_846__{u}"] = sha(p)
    teacher_commit = session_commit(REPO / "logs/stages/stage-2/teacher_modes/runs" / tr["run"])
    ours = sorted({c for c in (session_commit(REPO / "logs/stages/stage-1/reasoning_inheritance/runs" / at)
                               for at in capability_attempts(stores)) if c})
    path = construction_path(teacher_commit, ours)
    tok, tok_check = teacher_tokenizer(Path(a.tokenizer) if a.tokenizer else None, ident)
    out = {"schema": "aadistill.stage1.reasoning_inheritance.teacher_reuse/v2", "run": tr["run"],
           "construction_path": path, "teacher_tokenizer_on_the_dev_box": tok_check, "modes": {}}
    ok_all = True
    for mode, arc in tr["archives"].items():
        src = REPO / arc["path"]
        got = sha(src)
        rec = {"archive": {"path": arc["path"], "sha256": got, "matches_record": got == arc["sha256"]}}
        dest = DEST / mode
        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True)
        shutil.copyfile(src, dest / src.name)
        with tarfile.open(dest / src.name) as t:
            t.extractall(dest / "unpacked", filter="data")
        desc = json.loads((dest / "unpacked" / "evaluation.descriptor.json").read_text())
        root = dest / "unpacked" / "evaluation"
        bad = [f["name"] for f in desc["files"] if sha(root / f["name"]) != f["sha256"]]
        rec["descriptor"] = {"files": len(desc["files"]), "mismatched": bad, "started_utc": desc.get("started_utc")}
        checks, items, not_recorded = {}, {}, {}
        protocol = DEC.load(REPO / d["evaluation"]["decoding"])
        kwargs = d["evaluation"]["modes"][mode]["template_kwargs"]
        for summ in sorted((root / "generations").glob("*.summary.json")):
            stem = summ.name.removesuffix(".summary.json")
            c = unit_checks(summ.name, json.loads(summ.read_text()), mode=mode, d=d, ident=ident, eval_files=eval_files)
            if "eval_set_file" not in c:
                not_recorded[summ.name] = {
                    "eval_set_file": "the summary names only the pod path of its prompt file, no hash",
                    "seed_namespace": "the summary carries the seed rule, not a per-unit namespace field",
                    "prompt_text_and_ids": "the records carry the prompt's length, system source and think "
                                           "pre-opening, never its text or token ids",
                    "stands_in": "item_checks: coverage of the registered unit, every seed recomputed, the prompt "
                                 "fields against this round's and against the registered items rendered on the dev "
                                 "box, and the construction path at both session commits (prompt_content_identity: "
                                 "derived, not compared)"}
            render = None
            if tok is not None and not stem.startswith("fluid_reasoning_720"):
                render = rendered_fields(registered_samples(d["evaluation"]["eval_set"], stem, dest),
                                         tok, kwargs, d["evaluation"]["system_message"])
            it = item_checks(root / "generations", stem, mode=mode, eval_set=d["evaluation"]["eval_set"],
                             protocol=protocol, stores=stores, path=path, render=render)
            items[summ.name] = it
            checks[summ.name] = {**c, **it["checks"]}
        rec["unit_checks"] = checks
        rec["item_checks"] = items
        rec["not_recorded"] = not_recorded
        failed = {u: [k for k, v in c.items() if not v] for u, c in checks.items() if not all(c.values())}
        rec["failed_checks"] = failed
        rep = S.report(root / "generations", f"T_{mode}", {}, d["evaluation"]["eval_set"], rows_out=dest / "rows")
        (dest / "formal_report.json").write_text(json.dumps(rep, indent=1) + "\n")
        rec["rescored_report"] = {"path": str((dest / "formal_report.json").relative_to(REPO)),
                                  "sha256": sha(dest / "formal_report.json"),
                                  #: the file carries its own created_utc, so its sha256 changes on every run
                                  "content_sha256_without_created_utc": hashlib.sha256(json.dumps(
                                      {k: v for k, v in rep.items() if k != "created_utc"}, sort_keys=True,
                                      default=str).encode()).hexdigest()}
        rec["reusable"] = bool(rec["archive"]["matches_record"] and not bad and not failed)
        ok_all &= rec["reusable"]
        out["modes"][mode] = rec
    out["runner_note"] = (
        "The teacher's runners (commit 6d0d0a7e) and this round's final attempt's (session commit 8f74bc94) differ. "
        "Fluid: run_battery.py e650539e... then, 805d3dbd... now: tool-task support (tools, env_factory, a task's own "
        "system text), engine metrics, splitting the task file on newlines only, an optional time box and submission "
        "order, the engine's max_num_seqs, and abort_request([id]) (commit 30494d3b); no fluid_reasoning_720 item "
        "carries tools or a system field, and the teacher run parsed all 720 lines. Capability: uncapped_eval.py "
        "cd4b1888... then, aef2658c... now (generation source digest 9eb5d477... then, 153e295c... now): an optional "
        "--max-num-seqs and abort_request([id]); degeneration.py (79354082...) and behavior.py are unchanged. In both, "
        "a record is taken at its stop and a stopped request's later output is dropped (`pending` no longer holds "
        "it), so the abort repair changes what the engine keeps running after a stop, not what a sealed record holds "
        "(the statements that build a non-fluid prompt are the same at both session commits: construction_path) "
        "(analysis.py :: stop_record_integrity checks this on every sealed degeneration stop). It does not make a "
        "rerun reproduce the teacher's records token for token: batch composition differs (bf16 batch "
        "nondeterminism), so any regeneration is a new sample under the same seeds.")
    out["reusable"] = ok_all
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({mode: {"reusable": r["reusable"], "failed": r["failed_checks"], "mismatched": r["descriptor"]["mismatched"]}
                      for mode, r in out["modes"].items()}, indent=1))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
