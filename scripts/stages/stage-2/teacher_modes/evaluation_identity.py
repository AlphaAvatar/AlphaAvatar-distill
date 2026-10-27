#!/usr/bin/env python3
"""Each mode's evaluation identity: what a Research A generation is scored under ($0).

    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-2/teacher_modes/evaluation_identity.py \\
        --write configs/stages/stage-2/teacher_modes/evaluation_identity.json

A generation of this study is scored under one identity per mode, and every
part of it is bound by hash:

* the mode: the teacher revision, chat template, template kwargs, the
  generation prompt they produce and the think state it leaves;
* the instrument: eval-set-1's current identity and each component's formal
  scoring protocol and conformance vectors, and the decoding protocol;
* the scorer: the parser's think-state branch and the source files that score.

``preclosed`` is the hybrid non-thinking mode's state: an empty closed think
block in the prompt, no delimiter allowed in the continuation, the body the
answer. It applies only to a record that states ``think_state``; a record
without the field keeps the rule it was scored under, and no historical
result, item, gold answer or protocol changes (the maintainer's 2026-10-27
approval of this $0 scoring design; it authorizes no paid evaluation).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

IDENTITY = "configs/stages/stage-2/teacher_modes/teacher_identity.json"
DECODING = "configs/shared/evaluation/decoding/nucleus_v1.json"
#: the files whose code decides a verdict for a stored generation
SCORER_SOURCES = ("src/aadistill/evaluation/behavior.py", "src/aadistill/evaluation/strict_answer.py",
                  "src/aadistill/evaluation/capability.py", "src/aadistill/evaluation/usable_rollout.py",
                  "scripts/shared/evaluation/eval_set_scoring.py",
                  "scripts/stages/stage-0/calibration_volume/score_battery.py")
MODES = {"H-T": "thinking", "H-N": "non_thinking"}


def sha(path: str) -> str:
    return hashlib.sha256((REPO / path).read_bytes()).hexdigest()


def canonical_sha(doc) -> str:
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def instrument() -> dict:
    from shared import eval_sets as ES
    from shared.evaluation import eval_set_scoring as S

    eid = ES.default_formal()
    entry = next(e for e in ES.registry()["eval_sets"] if e["eval_set_id"] == eid)
    m, root = ES.manifest(eid), ES.set_root(eid)
    comps = {}
    for cname, comp in m["components"].items():
        comps[cname] = {"formal_protocol": S.formal_protocol(comp["scorer"]),
                        "conformance_sha256": {p: hashlib.sha256((root / rel).read_bytes()).hexdigest()
                                               for p, rel in S.conformance_files(comp["scorer"]).items()}}
    dec = json.loads((REPO / DECODING).read_text())
    return {"eval_set": {"eval_set_id": eid, "status": entry["status"],
                         "manifest_sha256": entry["identity_history"][-1]["manifest_sha256"],
                         "components": comps},
            "decoding": {"protocol_id": dec["protocol_id"], "path": DECODING, "sha256": sha(DECODING)}}


def build() -> dict:
    ident = json.loads((REPO / IDENTITY).read_text())
    shared = instrument()
    scorer = {"sources_sha256": {p: sha(p) for p in SCORER_SOURCES},
              "rule": "aadistill.evaluation.behavior.record_think_state: a record's stated think_state, "
                      "else the historical think_preopened rule"}
    modes = {}
    for arm, key in MODES.items():
        m = ident["modes"][key]
        block = {"mode": arm, "teacher": {"model": ident["model"], "revision": ident["revision"]},
                 "chat_template_sha256": ident["chat_template_sha256"],
                 "template_kwargs": m["template_kwargs"], "generation_suffix_ids": m["generation_suffix_ids"],
                 "think_state": m["think_state"], **shared, "scorer": scorer}
        modes[arm] = {**block, "identity_sha256": canonical_sha(block)}
    return {"schema": "aadistill.stage2.teacher_modes.evaluation_identity/v1",
            "_contract": __doc__.split("\n\n", 2)[2].strip(),
            "teacher_identity": {"path": IDENTITY, "sha256": sha(IDENTITY)},
            "modes": modes,
            "students": "a student is scored under the identity of the mode it was trained in; the teacher "
                        "under both",
            "authorizes": "nothing"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", default=None)
    a = ap.parse_args(argv)
    doc = build()
    text = json.dumps(doc, indent=1) + "\n"
    if a.write:
        Path(a.write).write_text(text)
        print(a.write)
    print(json.dumps({k: v["identity_sha256"] for k, v in doc["modes"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
