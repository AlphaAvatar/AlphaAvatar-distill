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
   extra or repeated), every request's seed is the decoding protocol's rule for its unit, item, turn and sample,
   and each item's prompt equals this round's for the same item -- the fluid unit's first user message (text, also
   compared with the registered item) and turn-0 prompt length, every other unit's prompt length in tokens, system
   source and think pre-opening (their records carry no prompt text: a length match, not a byte identity) --
   against every group of this round that holds the unit (``--store``);
4. rescores the generations with the CURRENT scorer (`shared.evaluation.eval_set_scoring.report`), which is the
   report the analysis reads.

A failed check refuses the reuse: T is then a group to be generated, never a teacher result with a caveat. A field
the teacher's records never carried is listed under ``not_recorded`` with what stands in for it; it is never
counted as a passed check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
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


def item_checks(gens: Path, stem: str, *, mode: str, eval_set: str, protocol: dict, stores: list[Path]) -> dict:
    """Coverage, seeds and prompt identity of one unit's teacher records, item by item."""
    from shared.evaluation import decoding as DEC
    recs = [json.loads(l) for l in (gens / f"{stem}.jsonl").read_text().split("\n") if l.strip()]
    ids = [(r.get("task_id") or r["id"]) for r in recs]
    reg = registered_ids(eval_set, stem)
    seen: dict = {}
    for i in ids:
        seen[i] = seen.get(i, 0) + 1
    fluid = stem.startswith("fluid_reasoning_720")
    seeds_checked = seed_bad = 0
    for r in recs:
        i = r.get("task_id") or r["id"]
        for t in (r["turns"] if fluid else [r]):
            want = DEC.request_seed(protocol, stem, i, int(t.get("turn") or 0) if fluid else 0, 0)
            seeds_checked += 1
            seed_bad += int((t.get("seed") if fluid else t.get("sampling_seed")) != want)
    ours = this_rounds(stores, mode, stem)
    by_id = {(r.get("task_id") or r["id"]): r for r in recs}

    def first_user(r):
        return next((m["content"] for m in r.get("messages") or [] if m.get("role") == "user"), None)

    def shape(r):
        #: fluid: the first user message and the turn-0 prompt length; other units carry no prompt text, only its
        #: length, the system source and whether the think block was pre-opened
        if fluid:
            return first_user(r), r["turns"][0].get("prompt_tokens")
        return r.get("prompt_tokens"), r.get("system_source"), r.get("think_preopened")
    source = None
    if fluid:
        from shared import eval_sets as ES
        reg_items = {t["task_id"]: t for t in ES.items(eval_set, "fluid_reasoning_720", stem.split("__", 1)[1])}
        bad = sum(first_user(r) != first_user(reg_items[i]) for i, r in by_id.items() if i in reg_items)
        source = {"compared": sum(i in reg_items for i in by_id), "differ": bad}
    prompt = {}
    for g, theirs in ours.items():
        n = bad = 0
        for i, r in by_id.items():
            o = theirs.get(i)
            if o is None:
                continue
            n += 1
            bad += int(shape(r) != shape(o))
        prompt[g] = {"compared": n, "differ": bad}
    out = {"registered": len(reg), "records": len(recs), "missing": sorted(set(reg) - set(seen))[:5],
           "n_missing": len(set(reg) - set(seen)), "n_extra": len(set(seen) - set(reg)),
           "n_repeated": sum(1 for v in seen.values() if v > 1),
           "seeds_checked": seeds_checked, "seeds_off_rule": seed_bad,
           "source_prompt_against_the_registered_item": source or (
               "not recorded: these records carry no prompt text; the unit's prompts were the registered item file, "
               "and the comparison with this round stands in for it"),
           "prompt_measure": ("first user message (text) and turn-0 prompt length in tokens" if fluid else
                              "prompt length in tokens, system source, think pre-opening (a length match, not a "
                              "byte identity)"),
           "prompt_against_this_round": prompt}
    out["checks"] = {
        "items_cover_the_registered_unit": out["n_missing"] == 0 and out["n_extra"] == 0 and out["n_repeated"] == 0,
        "seeds_follow_the_rule": seeds_checked > 0 and seed_bad == 0,
        "prompts_equal_this_rounds": bool(prompt) and all(v["compared"] == len(by_id) and v["differ"] == 0
                                                          for v in prompt.values())}
    if source is not None:
        out["checks"]["source_prompts_equal_the_registered_items"] = (source["compared"] == len(by_id)
                                                                      and source["differ"] == 0)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--store", action="append", default=[],
                    help="this round's secured evidence (<run scr>/store), for the prompt comparison; a later one wins")
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
    out = {"schema": "aadistill.stage1.reasoning_inheritance.teacher_reuse/v1", "run": tr["run"], "modes": {}}
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
        for summ in sorted((root / "generations").glob("*.summary.json")):
            stem = summ.name.removesuffix(".summary.json")
            c = unit_checks(summ.name, json.loads(summ.read_text()), mode=mode, d=d, ident=ident, eval_files=eval_files)
            if "eval_set_file" not in c:
                not_recorded[summ.name] = {
                    "eval_set_file": "the summary names only the pod path of its prompt file, no hash",
                    "seed_namespace": "the summary carries the seed rule, not a per-unit namespace field",
                    "stands_in": "item_checks: coverage of the registered unit, every seed recomputed, every prompt "
                                 "compared with this round's"}
            it = item_checks(root / "generations", stem, mode=mode, eval_set=d["evaluation"]["eval_set"],
                             protocol=protocol, stores=stores)
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
