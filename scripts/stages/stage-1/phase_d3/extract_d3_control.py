#!/usr/bin/env python3
"""The D3 control, D1-q2, as EVIDENCE: what its DEPTH chose and why. `$0`.

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python \
        scripts/stages/stage-1/phase_d3/extract_d3_control.py [--journal PATH] [--write]

Reads the completed D1 search's state journal (out of tree, in the durable
store), walks q2's ancestry by `parent_id`, and writes per step: the state id,
operator, profile, recorded config hash, the four checkpoint identities and the
step's execution knobs; for DEPTH, its kept and removed layers, its removal
order and EVERY greedy round's candidate table -- the parent-reference KL each
candidate scored. Every pin is cross-checked against the committed replay plan
the control is rebuilt from, so this record and the plan cannot describe two
different q2s.

What it is for: the treatment's DEPTH records the parent-reference KL of every
candidate it evaluates as a diagnostic. Round 0's candidates are the same
subsets the control scored, against the same parent, so that table is the
check that the treatment's parent is q2's FFN prefix and its reduction is q2's
-- and the removal order is the control's choice the treatment is compared to.
It reads no checkpoint bytes and decides nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
for _extra in ("src", "scripts", "scripts/stages/stage-1"):
    _p = str(REPO_ROOT / _extra)
    if _p not in sys.path:
        sys.path.insert(0, _p)

SCHEMA = "aadistill.phase_d3.control_record/v1"


class ControlEvidenceError(RuntimeError):
    """The journal does not describe the control the plan pins."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(journal: Path, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    from stages.phase_d1 import replay_specs as R
    from stages.phase_d3 import d3_design as D

    leaf = D.control_leaf(repo_root)
    _, plan_sha = D.control_plan(repo_root)
    states = R.load_states(journal)
    chain = R.resolve_ancestry(states, D.CONTROL_STATE_ID)
    if len(chain) != len(leaf.steps):
        raise ControlEvidenceError(
            f"the journal chain has {len(chain)} links and the plan {len(leaf.steps)}")
    steps, depth = [], None
    for pinned, row in zip(leaf.steps, chain):
        step = (row.get("steps") or [])[-1]
        art = row.get("artifact") or {}
        rec = {"index": pinned.index, "kind": pinned.kind,
               "impl_id": step.get("impl_id"), "profile_id": step.get("profile_id"),
               "state_id": row["state_id"],
               "config_hash": step.get("config_hash"),
               "artifact_digest": row.get("artifact_digest"),
               "weights_digest": art.get("weights_digest"),
               "single_shard_sha256": row.get("checkpoint_sha256"),
               "config_sha256": art.get("config_sha256"),
               "arch_signature": art.get("arch_signature"),
               "micro_batch_size": (step.get("trace") or {}).get("micro_batch_size"),
               "calibration_batch_packing": (step.get("trace") or {}).get(
                   "calibration_batch_packing"),
               "wall_seconds": step.get("wall_seconds")}
        for key, want in (("artifact_digest", pinned.expected_artifact_digest),
                          ("config_hash", pinned.expected_config_hash),
                          ("impl_id", pinned.impl_id),
                          ("profile_id", pinned.profile_id),
                          ("state_id", pinned.state_id)):
            if rec[key] != want:
                raise ControlEvidenceError(
                    f"step {pinned.index}: the journal's {key} is {rec[key]} and "
                    f"the plan pins {want}")
        steps.append(rec)
        if pinned.kind == D.TREATED_KIND:
            trace = step.get("trace") or {}
            arts = step.get("artifacts") or {}
            metrics = step.get("local_metrics") or {}
            rounds = [{"round": r["round"], "removed_before": r["removed_before"],
                       "chosen": r["chosen"], "chosen_score": r["chosen_score"],
                       "n_candidates": r["n_candidates"],
                       "table": [{"candidate": e["candidate"], "score": e["score"]}
                                 for e in r["table"]]}
                      for r in arts.get("search_rounds") or []]
            if [r["chosen"] for r in rounds] != trace.get("removal_order"):
                raise ControlEvidenceError(
                    "the DEPTH rounds' choices are not its recorded removal order")
            depth = {"step": pinned.index,
                     "reference": metrics.get("reference"),
                     "kept_layers": trace.get("kept_layers"),
                     "removed_layers": trace.get("removed_layers"),
                     "removal_order": trace.get("removal_order"),
                     "final_parent_reference_kl": (metrics.get("values") or {}).get(
                         "op.depth.causal_kl.final"),
                     "evaluations": (metrics.get("values") or {}).get(
                         "op.depth.causal_kl.evaluations"),
                     "scoring_positions": trace.get("scoring_positions"),
                     "reference_cache_mode": (arts.get("reference_cache") or {}).get(
                         "mode"),
                     "rounds": rounds}
    if depth is None:
        raise ControlEvidenceError("the control chain has no DEPTH step")
    final = {"artifact_digest": leaf.expected_artifact_digest,
             "weights_digest": leaf.expected_weights_digest,
             "single_shard_sha256": leaf.expected_single_shard_sha256,
             "arch_signature": leaf.expected_arch_signature}
    if steps[-1]["artifact_digest"] != final["artifact_digest"]:
        raise ControlEvidenceError("the chain's leaf is not the control's digest")
    return {
        "schema": SCHEMA,
        "control_state_id": D.CONTROL_STATE_ID,
        "path_label": leaf.path_label,
        "root_teacher_id": leaf.root_teacher_id,
        "root_teacher_sha256": leaf.root_teacher_sha256,
        "seed": leaf.seed, "num_parameters": leaf.num_parameters,
        "source": {"search_run_id": R.SOURCE_RUN_ID,
                   "search_session_commit": R.SOURCE_SESSION_COMMIT,
                   "journal_path": str(journal), "journal_sha256": _sha256(journal),
                   "replay_plan": D.CONTROL_PLAN, "replay_plan_sha256": plan_sha,
                   "reproduced_by": ("phase_d1_replay d1_replay_002: all four "
                                     "pinned steps and all four identities, "
                                     "differ: []")},
        "final": final, "steps": steps, "depth": depth,
        "_what_this_is": (
            "the control's recorded choices, extracted from the search journal "
            "and checked against the committed replay plan. Evidence only: it "
            "selects nothing and is read by the D3 structural comparison."),
    }


def main(argv: list[str] | None = None) -> int:
    from stages.phase_d1 import replay_specs as R
    from stages.phase_d3 import d3_design as D

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--journal", default=R.DEFAULT_JOURNAL)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    record = build(Path(args.journal))
    body = json.dumps(record, indent=1, sort_keys=True) + "\n"
    out = REPO_ROOT / D.CONTROL_RECORD
    if args.write:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(body)
        print(f"wrote {D.CONTROL_RECORD}")
        return 0
    same = out.is_file() and out.read_text() == body
    print(f"{D.CONTROL_RECORD}: {'current' if same else 'STALE (run with --write)'}")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
