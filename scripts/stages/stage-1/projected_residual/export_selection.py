#!/usr/bin/env python3
"""Export D4 Top-1's structural decisions -- layers, query heads, FFN neurons -- as a committed record ($0).

    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-1/projected_residual/export_selection.py \\
        --write configs/stages/stage-1/projected_residual/d4_top1_selection.json

The projected-residual student keeps D4 Top-1's every selection decision and
changes only how width is reduced, so the comparison with D4-Augmented varies
the width mechanism and nothing chosen by a search. The decisions live in the
Stage-0 d4 search journal (out of tree, 61 MB); this copies exactly the three
lists out of D4 Top-1's state row, re-maps them to the student's layer order,
binds them to the journal's sha256 and the state's identity, and refuses
unless they describe the target geometry (`selection_problems`).

* ATTENTION ran first, on the 36-layer teacher: ``kept_heads`` is indexed by
  teacher layer.
* DEPTH ran second: its ``removal_order`` names the 8 teacher layers bypassed;
  the 28 kept teacher layers, ascending, are the student's layers in order.
* RESIDUAL_WIDTH ran third and is the step this study replaces.
* FFN ran last, on the 28-layer width-1024 state: ``kept_neurons`` is indexed
  by student layer, in teacher neuron ids (DEPTH and width change neither).
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

JOURNAL = Path("/home/ecs-user/aad-artifacts/stage0_calvol/dose_d4_20261020_212310/store/d4/search/search/states.jsonl")
JOURNAL_SHA256 = "c162a362e01ba6b10b7f7d6baccef31798d53a3d6f096cfe0d40c73358a0d358"
STATE_ID = "1301deb71685aa1ca5c7b6cf2c491ac4"
MATERIALIZATION = "logs/stages/stage-2/reasoning_data/assets/d4_top1_materialization.json"
PARENT = {"num_hidden_layers": 36, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
          "intermediate_size": 9728}


def build(journal: Path) -> dict:
    from aadistill.initialization.transforms.projected_residual import selection_problems

    sha = hashlib.sha256(journal.read_bytes()).hexdigest()
    if sha != JOURNAL_SHA256:
        raise SystemExit(f"journal sha256 {sha} is not the recorded {JOURNAL_SHA256}")
    row = next(json.loads(l) for l in journal.read_text().split("\n")
               if l.strip() and json.loads(l).get("state_id") == STATE_ID)
    steps = {s["kind"]: s for s in row["steps"]}
    order = [s["kind"] for s in row["steps"]]
    if order != ["ATTENTION", "DEPTH", "RESIDUAL_WIDTH", "FFN"]:
        raise SystemExit(f"unexpected operator order {order}")
    removed = steps["DEPTH"]["local_metrics"]["detail"]["removal_order"]
    kept_layers = [i for i in range(PARENT["num_hidden_layers"]) if i not in set(removed)]
    heads_by_teacher_layer = steps["ATTENTION"]["artifacts"]["kept_heads"]
    selection = {
        "kept_layers": kept_layers,
        "kept_heads": [sorted(heads_by_teacher_layer[i]) for i in kept_layers],
        "kept_neurons": [sorted(n) for n in steps["FFN"]["artifacts"]["kept_neurons"]],
    }
    target = {"num_hidden_layers": len(kept_layers), "num_attention_heads": 16, "num_key_value_heads": 8,
              "head_dim": 128, "intermediate_size": 3072}
    problems = selection_problems(selection, PARENT, target)
    if problems:
        raise SystemExit("; ".join(problems))
    mat = json.loads((REPO / MATERIALIZATION).read_text())["materialization_identity"]
    return {
        "schema": "aadistill.stage1.projected_residual.selection/v1",
        "source": {"journal": str(journal), "journal_sha256": sha, "state_id": STATE_ID,
                   "root_teacher": row["root_teacher_id"], "root_teacher_sha256": row["root_teacher_sha256"],
                   "d4_top1_checkpoint_sha256": mat["checkpoint_sha256"],
                   "d4_top1_artifact_digest": mat["artifact_digest"],
                   "steps": [{"kind": s["kind"], "impl_id": s["impl_id"], "profile_id": s["profile_id"],
                              "config_hash": s["config_hash"]} for s in row["steps"]]},
        "parent_geometry": PARENT, "target_geometry": target,
        "depth_removed_teacher_layers": sorted(removed), "depth_removal_order": removed,
        **selection,
        "content_sha256": hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest(),
        "_reading": ("FFN's neurons were chosen on the 1024-wide state after RESIDUAL_WIDTH; reusing them keeps "
                     "every selection fixed and is a declared bias toward D4's width mechanism, not a neutral "
                     "choice for the full-width student"),
        "authorizes": "nothing",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--journal", default=str(JOURNAL))
    ap.add_argument("--write", default=None)
    a = ap.parse_args(argv)
    doc = build(Path(a.journal))
    # one line per top-level key: the three index lists are 86k numbers and stay compact
    text = "{\n" + ",\n".join(f" {json.dumps(k)}: {json.dumps(v, separators=(',', ':'))}"
                               for k, v in doc.items()) + "\n}\n"
    if a.write:
        Path(a.write).write_text(text)
        print(a.write, doc["content_sha256"])
    else:
        print(text[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
