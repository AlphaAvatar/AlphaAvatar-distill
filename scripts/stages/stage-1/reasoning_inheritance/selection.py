#!/usr/bin/env python3
"""The anchor search's frozen selections, read from its journal into one committed record ($0, CPU).

    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-1/reasoning_inheritance/selection.py --write
    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-1/reasoning_inheritance/selection.py --check

`design.json :: anchor` names the packed search, its journal (by sha256) and the Top-1. This walks the Top-1's
ancestor chain in the journal (the last row of each state), requires it to be the anchor's path step for step, and
writes `configs/stages/stage-1/reasoning_inheritance/frozen_selection.json`:

* the FFN step's ``kept_neurons`` for each of the 36 root blocks (FFN ran before DEPTH, so every block has one);
* the DEPTH step's ``kept_layers`` / ``removed_layers`` / ``removal_order`` (root numbering);
* the ATTENTION step's ``kept_heads`` for the 28 kept blocks, mapped from child to ROOT numbering through
  ``kept_layers``;
* the width step's recorded diagnostics (``point_weights``, captured energy, eigenvalues, scale): the width
  transform itself was never stored, and is recomputed on the pod (`driver.py`) against these;
* every ancestor's identity: state id, artifact digest, single-shard sha256, config sha256, arch spec, parameters,
  and each step's ``config_hash`` (the fixed-path replay's input pin).

Nothing is selected here: the record repeats what the journal says, and ``--check`` refuses a committed record that a
fresh read of the journal does not reproduce byte for byte.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
DESIGN = REPO / "configs/stages/stage-1/reasoning_inheritance/design.json"
OUT = REPO / "configs/stages/stage-1/reasoning_inheritance/frozen_selection.json"
SCHEMA = "aadistill.stage1.reasoning_inheritance.frozen_selection/v1"


class SelectionError(RuntimeError):
    pass


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def journal_chain(journal: Path, state_id: str) -> list[dict]:
    """The state's ancestors, root-most first: the LAST journal row of each state (a resumed state is re-appended)."""
    last = {}
    for line in journal.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            last[r["state_id"]] = r
    chain, sid = [], state_id
    while sid in last:
        chain.append(last[sid])
        sid = last[sid]["parent_id"]
    if not chain:
        raise SelectionError(f"{state_id} is not in {journal}")
    return list(reversed(chain))


def identity(row: dict) -> dict:
    a = row["artifact"]
    return {"state_id": row["state_id"], "path_label": row["path_label"], "depth": row["depth"],
            "artifact_digest": row["artifact_digest"], "single_shard_sha256": a.get("single_shard_sha256"),
            "config_sha256": a["config_sha256"], "arch_signature": a["arch_signature"],
            "num_parameters": row["num_parameters"], "arch_spec": row["arch_spec"],
            "materialization_id": row["materialization"]["materialization_id"],
            "numerical_execution_fingerprint": row["materialization"]["numerical_execution_fingerprint"]}


def extract(design: dict | None = None, root: Path = REPO) -> dict:
    d = design or json.loads(DESIGN.read_text())
    a = d["anchor"]
    jp = root / a["journal"]["path"]
    got = sha256_file(jp)
    if got != a["journal"]["sha256"]:
        raise SelectionError(f"{jp} hashes to {got[:12]}, the anchor's journal is {a['journal']['sha256'][:12]}")
    sel_doc = json.loads((root / a["evidence_dir"] / "stage1_selection.json").read_text())
    if sel_doc["journal"]["sha256"] != got:
        raise SelectionError("stage1_selection.json names another journal")
    if sel_doc["selected"][0]["state_id"] != a["top1_state_id"]:
        raise SelectionError("the anchor's Top-1 is not stage1_selection.json's selected leaf")
    chain = journal_chain(jp, a["top1_state_id"])
    leaf = chain[-1]
    if (leaf["artifact_digest"], leaf["checkpoint_sha256"]) != (a["top1_artifact_digest"], a["top1_single_shard_sha256"]):
        raise SelectionError("the journal's Top-1 identity is not the anchor's")
    steps = leaf["steps"]
    want = [tuple(s) for s in a["path"]]
    have = [(s["impl_id"], s["profile_id"]) for s in steps]
    if have != want or len(chain) != len(steps):
        raise SelectionError(f"the Top-1's path is {have}, not the anchor's {want}")
    for i, (row, s) in enumerate(zip(chain, steps)):
        if row["steps"] != steps[:i + 1]:
            raise SelectionError(f"ancestor {row['state_id']} records other steps than the Top-1's prefix")
    ffn, depth, width, attn = steps
    n_root = len(ffn["artifacts"]["kept_neurons"])
    kept = [int(b) for b in depth["trace"]["kept_layers"]]
    removed = [int(b) for b in depth["trace"]["removed_layers"]]
    if sorted(kept + removed) != list(range(n_root)):
        raise SelectionError("kept and removed layers do not partition the root's blocks")
    heads_child = attn["artifacts"]["kept_heads"]
    if len(heads_child) != len(kept) or heads_child != attn["trace"]["kept_q_heads_per_layer"]:
        raise SelectionError("the ATTENTION step's head selection does not cover the kept blocks")
    diag = width["artifacts"]["projection_diagnostics"]
    return {
        "schema": SCHEMA,
        "source": {"journal": a["journal"], "stage1_selection": f"{a['evidence_dir']}/stage1_selection.json",
                   "stage1_selection_sha256": sha256_file(root / a["evidence_dir"] / "stage1_selection.json"),
                   "search_run": a["search_run"], "search_config_hash": sel_doc["search"]["config_hash"],
                   "root_teacher_id": leaf["root_teacher_id"], "root_teacher_sha256": leaf["root_teacher_sha256"],
                   "seed": leaf["seed"]},
        "n_root_blocks": n_root,
        "kept_blocks": kept,
        "removed_blocks": removed,
        "removal_order": [int(b) for b in depth["trace"]["removal_order"]],
        "ffn_neurons": {str(b): v for b, v in enumerate(ffn["artifacts"]["kept_neurons"])},
        "query_heads_kept_blocks": {str(kept[i]): v for i, v in enumerate(heads_child)},
        "width_recorded": {"d_parent": width["trace"]["d_parent"], "d_child": width["trace"]["d_child"],
                           "scale_compensation": width["trace"]["scale_compensation"],
                           "point_weights": diag["point_weights"],
                           "energy_captured_frac": diag["energy_captured_frac"],
                           "top_eigenvalue": diag["top_eigenvalue"],
                           "min_kept_eigenvalue": diag["min_kept_eigenvalue"],
                           "orthonormality_error": diag["orthonormality_error"],
                           "scoring_positions": width["trace"]["scoring_positions"],
                           "forward_context": width["trace"]["forward_context"]},
        "attention_recorded": {"head_scores_per_layer": attn["trace"]["head_scores_per_layer"],
                               "selection_margin_per_layer": attn["trace"]["selection_margin_per_layer"],
                               "calibration_tokens": attn["trace"]["calibration_tokens"]},
        "steps": [{"index": s["index"], "kind": s["kind"], "impl_id": s["impl_id"], "profile_id": s["profile_id"],
                   "profile_hash": s["profile_hash"], "config_hash": s["config_hash"],
                   "impl_signature_hash": s["impl_signature_hash"], "result_spec_hash": s["result_spec_hash"],
                   "seed": s["seed"], "wall_seconds": s["wall_seconds"]} for s in steps],
        "states": [identity(r) for r in chain],
        "leaf_evaluation": {k: leaf["evaluation"].get(k) for k in ("values", "measurement_protocol_id")}
        if isinstance(leaf.get("evaluation"), dict) else None,
        "authorizes": "nothing",
    }


def render(doc: dict) -> str:
    """Sorted, indented JSON with every list of numbers on one line, so a diff names the block that moved."""
    text = json.dumps(doc, indent=1, sort_keys=True)
    flat = re.sub(r"\[\s*([-0-9.eE+,\s]+?)\s*\]",
                  lambda m: "[" + ", ".join(x.strip() for x in m.group(1).split(",") if x.strip()) + "]", text)
    return flat + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    text = render(extract())
    if json.loads(text) != extract():
        raise SelectionError("the rendering does not round-trip")
    if a.write:
        OUT.write_text(text)
        print(f"{OUT.relative_to(REPO)}: {hashlib.sha256(text.encode()).hexdigest()}")
        return 0
    same = OUT.is_file() and OUT.read_text() == text
    print(("the committed record is the journal's" if same else "the committed record differs from the journal"))
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
