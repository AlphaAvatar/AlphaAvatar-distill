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
3. rescores the generations with the CURRENT scorer (`shared.evaluation.eval_set_scoring.report`), which is the
   report the analysis reads.

A failed check refuses the reuse: T is then a group to be generated, never a teacher result with a caveat.
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
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
        checks = {}
        for summ in sorted((root / "generations").glob("*.summary.json")):
            checks[summ.name] = unit_checks(summ.name, json.loads(summ.read_text()), mode=mode, d=d, ident=ident,
                                            eval_files=eval_files)
        rec["unit_checks"] = checks
        failed = {u: [k for k, v in c.items() if not v] for u, c in checks.items() if not all(c.values())}
        rec["failed_checks"] = failed
        rep = S.report(root / "generations", f"T_{mode}", {}, d["evaluation"]["eval_set"], rows_out=dest / "rows")
        (dest / "formal_report.json").write_text(json.dumps(rep, indent=1) + "\n")
        rec["rescored_report"] = {"path": str((dest / "formal_report.json").relative_to(REPO)),
                                  "sha256": sha(dest / "formal_report.json")}
        rec["reusable"] = bool(rec["archive"]["matches_record"] and not bad and not failed)
        ok_all &= rec["reusable"]
        out["modes"][mode] = rec
    out["runner_note"] = ("fluid_reasoning_720 was generated by run_battery.py sha256 e650539e...; this round's runner "
                          "(5cb6529d...) differs only in tool-task support (tools, env_factory, a task's own system "
                          "text), engine metrics and splitting the task file on newlines only, none of which a "
                          "fluid_reasoning_720 item reaches (no item carries tools or a system field, and the teacher "
                          "run parsed all 720 lines). The capability runner's generation source digest (9eb5d477...) "
                          "and the degeneration rule (79354082...) are unchanged.")
    out["reusable"] = ok_all
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({mode: {"reusable": r["reusable"], "failed": r["failed_checks"], "mismatched": r["descriptor"]["mismatched"]}
                      for mode, r in out["modes"].items()}, indent=1))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
