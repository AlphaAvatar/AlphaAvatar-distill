#!/usr/bin/env python3
"""A_teacher_parent's frozen selections, read from the packed search's journal into one committed record ($0, CPU).

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python \\
        scripts/stages/stage-1/teacher_parent_initialization/selection.py --write | --check

The packed search (`design.json :: source_search`) expanded the ROOT TEACHER with every operator under both profiles
before anything else. Four of those first-level states are this arm's selections or its reproduction gate:

* ``FFN(calib.hybrid_d4_base@v2)`` -- ``kept_neurons`` of all 36 root blocks (the same state is B's first step);
* ``DEPTH(calib.hybrid_d4_base@v2)`` -- ``kept_layers`` K, ``removed_layers`` and ``removal_order``; the record of each
  greedy round is checked here: round r removes nothing but the earlier rounds' choices, scores every remaining layer
  against the unbypassed teacher (``reference = parent_state``), and picks the lowest score;
* ``ATTENTION(calib.hybrid_d4_rh@v2)`` -- ``kept_heads`` of all 36 root blocks, keyed by ROOT index;
* ``RESIDUAL_WIDTH(calib.hybrid_d4_rh@v2)`` -- not a selection: its identity is the gate the pod's replay of the
  teacher statistics pass must reproduce bit for bit, and its 37-point diagnostics are recorded beside A's.

Every state must be the design's, have the search root as parent (the root's materialization, the session's
numerical fingerprint), the teacher's identity and the design's operator and profile. ``--check`` refuses a committed
record that a fresh read does not reproduce byte for byte. Nothing is selected here: the record repeats the journal.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.reasoning_inheritance.selection import identity, render, sha256_file  # noqa: E402

DESIGN = REPO / "configs/stages/stage-1/teacher_parent_initialization/design.json"
OUT = REPO / "configs/stages/stage-1/teacher_parent_initialization/selection.json"
SCHEMA = "aadistill.stage1.teacher_parent_initialization.selection/v1"
B_SELECTION = REPO / "configs/stages/stage-1/reasoning_inheritance/frozen_selection.json"
#: the design's arm key -> the journal step kind
KINDS = {"FFN": "FFN", "DEPTH": "DEPTH", "ATTENTION": "ATTENTION", "RESIDUAL_WIDTH": "RESIDUAL_WIDTH"}


class SelectionError(RuntimeError):
    pass


def last_rows(journal: Path) -> dict[str, dict]:
    """The LAST journal row of each state (a resumed state is re-appended)."""
    out = {}
    for line in journal.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["state_id"]] = r
    return out


def depth_rounds_problems(step: dict, n_root: int, n_keep: int) -> list[str]:
    """The DEPTH record against greedy accumulation on one fixed reference (pure, tested)."""
    bad = []
    a, tr = step["artifacts"], step["trace"]
    rounds = a["search_rounds"]
    order = [int(x) for x in tr["removal_order"]]
    if len(rounds) != n_root - n_keep or len(order) != len(rounds):
        bad.append(f"{len(rounds)} rounds / {len(order)} removals for {n_root} -> {n_keep}")
    for r, rd in enumerate(rounds):
        if [int(x) for x in rd["removed_before"]] != order[:r]:
            bad.append(f"round {r} starts from {rd['removed_before']}, not the earlier choices {order[:r]}")
        cands = [int(t["candidate"]) for t in rd["table"]]
        if sorted(cands) != sorted(set(range(n_root)) - set(order[:r])) or rd["n_candidates"] != n_root - r:
            bad.append(f"round {r} scored {len(cands)} candidates, not every remaining layer")
        best = min(rd["table"], key=lambda t: (t["score"], t["candidate"]))
        if r < len(order) and (int(rd["chosen"]) != order[r] or int(best["candidate"]) != order[r]):
            bad.append(f"round {r} chose {rd['chosen']}, the lowest score is {best['candidate']}")
    if sum(n_root - r for r in range(len(rounds))) != int(a["timing"]["candidate_subsets"]):
        bad.append(f"{a['timing']['candidate_subsets']} candidate subsets, not the greedy count")
    kept, removed = [int(b) for b in tr["kept_layers"]], [int(b) for b in tr["removed_layers"]]
    if sorted(removed) != sorted(order) or sorted(kept + removed) != list(range(n_root)) or len(kept) != n_keep:
        bad.append("kept and removed layers do not partition the root's blocks as the removal order says")
    if step["local_metrics"].get("reference") != "parent_state":
        bad.append(f"the KL reference is {step['local_metrics'].get('reference')!r}, not parent_state")
    return bad


def extract(design: dict | None = None, root: Path = REPO) -> dict:
    d = design or json.loads(DESIGN.read_text())
    src, arm = d["source_search"], d["arms"]["A_teacher_parent"]
    jp = root / src["journal"]["path"]
    got = sha256_file(jp)
    if got != src["journal"]["sha256"]:
        raise SelectionError(f"{jp} hashes to {got[:12]}, the design's journal is {src['journal']['sha256'][:12]}")
    rows = last_rows(jp)
    states, steps = {}, {}
    for key, spec in arm["selection"].items():
        r = rows.get(spec["state_id"])
        if r is None:
            raise SelectionError(f"{key}: state {spec['state_id']} is not in the journal")
        st = r["steps"][-1]
        problems = []
        if r["parent_id"] != src["root_state_id"] or r["depth"] != 1 or len(r["steps"]) != 1:
            problems.append(f"parent {r['parent_id']} at depth {r['depth']}: not a first-level state of the root")
        if r["materialization"]["parent_materialization_id"] != src["root_materialization_id"]:
            problems.append("its parent materialization is not the root's")
        if r["materialization"]["numerical_execution_fingerprint"] != src["numerical_execution_fingerprint"]:
            problems.append("another numerical execution fingerprint")
        if (r["root_teacher_id"], r["root_teacher_sha256"]) != (d["teacher"]["root_teacher_id"],
                                                                d["teacher"]["root_teacher_sha256"]):
            problems.append("another root teacher")
        if (st["kind"], st["impl_id"], st["profile_id"]) != (KINDS[key], spec["impl_id"], spec["profile_id"]):
            problems.append(f"{st['kind']} {st['impl_id']} {st['profile_id']}, not the design's")
        if r["validity"] != "measured" or r["invalid_reason"] or int(r["seed"]) != int(src["seed"]):
            problems.append(f"validity {r['validity']} / {r['invalid_reason']}, seed {r['seed']}")
        fc = st["trace"].get("forward_context")
        if fc != d["session"]["forward_context"]:
            problems.append(f"forward context {fc}, not the session's")
        if problems:
            raise SelectionError(f"{key} ({spec['state_id']}): " + "; ".join(problems))
        states[key] = {**identity(r), "parent_id": r["parent_id"],
                       "parent_materialization_id": r["materialization"]["parent_materialization_id"]}
        steps[key] = {k: st[k] for k in ("kind", "impl_id", "profile_id", "profile_hash", "config_hash",
                                         "impl_signature_hash", "result_spec_hash", "seed", "wall_seconds")}
    ffn, dep, att, wid = (rows[arm["selection"][k]["state_id"]]["steps"][-1]
                          for k in ("FFN", "DEPTH", "ATTENTION", "RESIDUAL_WIDTH"))
    n_root = len(ffn["artifacts"]["kept_neurons"])
    n_keep = int(d["target"]["geometry"]["num_hidden_layers"])
    bad = depth_rounds_problems(dep, n_root, n_keep)
    if bad:
        raise SelectionError("DEPTH: " + "; ".join(bad))
    heads = att["artifacts"]["kept_heads"]
    if len(heads) != n_root or heads != att["trace"]["kept_q_heads_per_layer"]:
        raise SelectionError("ATTENTION: kept_heads do not cover the root's blocks")
    q = int(d["target"]["geometry"]["num_attention_heads"])
    if {len(h) for h in heads} != {q}:
        raise SelectionError(f"ATTENTION: blocks keep {sorted({len(h) for h in heads})} heads, not {q}")
    b = json.loads(B_SELECTION.read_text())
    b_ffn_same = (b["states"][0]["state_id"] == arm["selection"]["FFN"]["state_id"]
                  and [b["ffn_neurons"][str(i)] for i in range(n_root)] == ffn["artifacts"]["kept_neurons"])
    if not b_ffn_same:
        raise SelectionError("FFN: the state is not B's first step, or its neurons are not B's")
    kept = [int(x) for x in dep["trace"]["kept_layers"]]
    layer_points = sorted({0, *kept})
    diag = wid["artifacts"]["projection_diagnostics"]
    rounds = dep["artifacts"]["search_rounds"]
    return {
        "schema": SCHEMA,
        "source": {"journal": src["journal"], "search_run": src["run"], "root_state_id": src["root_state_id"],
                   "root_teacher_id": d["teacher"]["root_teacher_id"],
                   "root_teacher_sha256": d["teacher"]["root_teacher_sha256"], "seed": int(src["seed"])},
        "n_root_blocks": n_root,
        "kept_blocks": kept,
        "removed_blocks": [int(x) for x in dep["trace"]["removed_layers"]],
        "removal_order": [int(x) for x in dep["trace"]["removal_order"]],
        "depth_record": {"rounds": [{"round": int(rd["round"]), "chosen": int(rd["chosen"]),
                                     "chosen_score": rd["chosen_score"], "n_candidates": int(rd["n_candidates"])}
                                    for rd in rounds],
                         "final_kl": dep["local_metrics"]["values"].get("op.depth.causal_kl.final"),
                         "evaluations": dep["local_metrics"]["values"].get("op.depth.causal_kl.evaluations"),
                         "reference": dep["local_metrics"]["reference"],
                         "packing": dep["trace"].get("packing"), "scoring_positions": dep["trace"]["scoring_positions"],
                         "checked": "depth_rounds_problems: greedy accumulation, every remaining layer scored, the "
                                    "lowest chosen, one parent_state reference"},
        "ffn_neurons": {str(i): v for i, v in enumerate(ffn["artifacts"]["kept_neurons"])},
        "ffn_is_b_first_step": b_ffn_same,
        "query_heads": {str(i): v for i, v in enumerate(heads)},
        "attention_record": {"head_scores_per_layer": att["trace"]["head_scores_per_layer"],
                             "selection_margin_per_layer": att["trace"]["selection_margin_per_layer"],
                             "calibration_tokens": att["trace"]["calibration_tokens"],
                             "retained_write_energy": att["local_metrics"]["values"]},
        "width": {"layer_points": layer_points, "points": layer_points + [n_root],
                  "point_weights": ([d["arms"]["A_teacher_parent"]["width_points"]["weights"]["embedding_output_point"]]
                                    + [d["arms"]["A_teacher_parent"]["width_points"]["weights"]["every_other_point"]]
                                    * (len(layer_points) - 1)
                                    + [d["arms"]["A_teacher_parent"]["width_points"]["weights"]["post_final_norm_point"]]),
                  "post_norm_point": n_root, "d_parent": int(wid["trace"]["d_parent"]),
                  "d_child": int(wid["trace"]["d_child"]),
                  "reproduction_gate": {"state_id": arm["selection"]["RESIDUAL_WIDTH"]["state_id"],
                                        "artifact_digest": states["RESIDUAL_WIDTH"]["artifact_digest"],
                                        "all_points_diagnostics": {k: v for k, v in diag.items()},
                                        "stats_tokens": wid["local_metrics"]["detail"].get("stats_tokens"),
                                        "scoring_positions": wid["trace"]["scoring_positions"]}},
        "states": states,
        "steps": steps,
        "authorizes": "nothing",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    text = render(extract())
    if a.write:
        OUT.write_text(text)
        print(f"wrote {OUT.relative_to(REPO)} ({len(text)} B)")
        return 0
    same = OUT.is_file() and OUT.read_text() == text
    print("selection reproduced from the journal" if same else "REFUSED: the committed selection is not the journal's")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
