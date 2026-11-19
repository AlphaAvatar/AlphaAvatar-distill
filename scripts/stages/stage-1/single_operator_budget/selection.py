#!/usr/bin/env python3
"""SOB1's reused selections and gates, read from the packed search's journal and TPI1's committed selection ($0, CPU).

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python \\
        scripts/stages/stage-1/single_operator_budget/selection.py --write | --check

Every arm is built on the original teacher with A's operators (`design.json :: operators`); what the journal already
holds is reused, and what is new is computed on the pod against a gate this record pins:

* **depth only** -- DEPTH resumes the journal's 36 -> 28 search (state ``fede45db...``) from its round 9. The record
  carries the eight committed rounds VERBATIM (every candidate's score), and the check that makes a resume legitimate,
  field by field (`resume_checks`): the same original teacher as the reference (``parent_state``: the unbypassed
  root, a first-level state of the search root), the same calibration profile, items, scoring positions and packed
  forward context, the same support and execution (micro-batch, packing, scored rows, prefix reuse), the operator's
  signature, the greedy accumulation and the tie rule (lowest score, ties to the lower index). On the pod the operator
  re-checks the rounds, re-scores round 8's chosen subset (equal to its recorded score, exactly) and scores rounds 9-20:
  28 + 27 + ... + 17 = 270 new candidate subsets. If a required field were missing, ``resume_checks.missing`` names it
  and ``minimal_recompute`` the first round that must be scored again.
* **FFN + attention** -- the attention heads are the journal's ``kept_heads`` at 16 / 8 on the teacher (TPI1's
  selection, gate: their child's digest); the FFN at 3,840 is new: the operator on the teacher under the journal's
  config hash, gated by the same run at 3,072 equalling the journal's FFN state (its digest).
* **residual width** -- the teacher residual statistics TPI1 already paid for, loaded from its sealed file
  (``teacher_residual_statistics.safetensors``) instead of a second teacher forward: TPI1's committed init record says
  the pass that produced them passed both of TPI1's gates (the replayed RESIDUAL_WIDTH child equal to the journal
  state's digest, P at 1,024 over all 37 points reproducing the recorded diagnostics), and this record binds that init
  record by sha256 and copies from it the file's sha256, bytes, tensor layout, statistics positions (tokens) and
  relay location. On the pod the file is checked against all of them and P at 1,024 recomputed from it must
  reproduce the recorded diagnostics; then P at 1,280 from the same statistics over all 37 residual points, weights
  9 / 1 / 8.

``--check`` refuses a committed record that a fresh read does not reproduce byte for byte.
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

from stages.reasoning_inheritance.selection import render, sha256_file  # noqa: E402
from stages.teacher_parent_initialization import selection as TSEL  # noqa: E402

DESIGN = REPO / "configs/stages/stage-1/single_operator_budget/design.json"
OUT = REPO / "configs/stages/stage-1/single_operator_budget/selection.json"
TPI1_SELECTION = REPO / "configs/stages/stage-1/teacher_parent_initialization/selection.json"
TPI1_DESIGN = REPO / "configs/stages/stage-1/teacher_parent_initialization/design.json"
#: TPI1's committed init record: the gates its statistics pass passed, and the sealed file's identity and relay copy
TPI1_INIT_RECORD = REPO / "logs/stages/stage-1/teacher_parent_initialization/runs/tpi_init_20261117_142049/evidence/init_record.json"
SCHEMA = "aadistill.stage1.single_operator_budget.selection/v1"
#: what a resumed round needs from the journal, per round, and what the step must carry
ROUND_FIELDS = ("round", "removed_before", "chosen", "chosen_score", "n_candidates", "table")
STEP_FIELDS = ("impl_id", "impl_signature_hash", "profile_id", "profile_hash", "config_hash", "seed")
TRACE_FIELDS = ("forward_context", "scoring_positions", "micro_batch_size", "calibration_batch_packing",
                "prediction_rows", "depth_prefix_reuse", "removal_order", "kept_layers", "packing")


class SelectionError(RuntimeError):
    pass


def resume_checks(row: dict, step: dict, *, design: dict, tpi1_design: dict, tpi1_sel: dict) -> dict:
    """Each condition a resume of the journal's DEPTH search needs, with what the journal says (pure)."""
    from aadistill.initialization.operators.depth.causal_kl_greedy import DepthCausalKLGreedyV1
    from aadistill.initialization.statistics.contribution import resumed_round_problems
    src = tpi1_design["source_search"]
    tr, lm, art = step.get("trace", {}), step.get("local_metrics", {}), step.get("artifacts", {})
    rounds = art.get("search_rounds") or []
    n_root = len(tpi1_sel["ffn_neurons"])
    missing = ([f"step.{k}" for k in STEP_FIELDS if k not in step] + [f"trace.{k}" for k in TRACE_FIELDS if k not in tr]
               + [f"round {r.get('round', i)}.{k}" for i, r in enumerate(rounds) for k in ROUND_FIELDS if k not in r])
    sig = DepthCausalKLGreedyV1().signature_hash
    op = design["operators"]["DEPTH"]
    checks = {
        "the reference is the original teacher": {
            "reference": lm.get("reference"), "parent_id": row.get("parent_id"), "depth": row.get("depth"),
            "root_state_id": src["root_state_id"],
            "ok": lm.get("reference") == "parent_state" and row.get("parent_id") == src["root_state_id"]
            and row.get("depth") == 1 and len(row.get("steps", [])) == 1},
        "the original teacher is the design's": {
            "root_teacher_id": row.get("root_teacher_id"), "root_teacher_sha256": row.get("root_teacher_sha256"),
            "ok": (row.get("root_teacher_id"), row.get("root_teacher_sha256"))
            == (design["teacher"]["root_teacher_id"], design["teacher"]["root_teacher_sha256"])},
        "the same operator and calibration profile": {
            "impl_id": step.get("impl_id"), "profile_id": step.get("profile_id"),
            "impl_signature_hash": step.get("impl_signature_hash"), "signature_now": sig,
            "ok": (step.get("impl_id"), step.get("profile_id")) == (op["impl_id"], op["profile_id"])
            and step.get("impl_signature_hash") == sig},
        "the same calibration items, positions and packing (one operator config)": {
            "config_hash": step.get("config_hash"), "tpi1_pinned": tpi1_sel["steps"]["DEPTH"]["config_hash"],
            "forward_context": tr.get("forward_context"), "scoring_positions": tr.get("scoring_positions"),
            "packing": tr.get("packing"),
            "ok": step.get("config_hash") == tpi1_sel["steps"]["DEPTH"]["config_hash"]
            and tr.get("forward_context") == design["calibration"]["forward_context"]
            and (tr.get("scoring_positions") or {}).get("n_items")
            == design["calibration"]["before_packing"][op["profile_id"]]["items"]},
        "the same execution (micro-batch, batch packing, scored rows, prefix reuse)": {
            k: tr.get(k) for k in ("micro_batch_size", "calibration_batch_packing", "prediction_rows",
                                   "depth_prefix_reuse")} | {
            "ok": (tr.get("micro_batch_size"), tr.get("calibration_batch_packing"), tr.get("prediction_rows"),
                   tr.get("depth_prefix_reuse")) == (1, "original_order_v1", "scored_v1", True)},
        "greedy accumulation, every surviving layer scored, the tie rule": {
            "problems": resumed_round_problems(rounds, n_root) + TSEL.depth_rounds_problems(step, n_root, n_root - 8),
            "rule": "lowest score; ties to the lower index (contribution.greedy_removal)"},
        "the removal order is TPI1's committed one": {
            "journal": [int(r["chosen"]) for r in rounds], "tpi1": tpi1_sel["removal_order"]},
    }
    checks["greedy accumulation, every surviving layer scored, the tie rule"]["ok"] = \
        not checks["greedy accumulation, every surviving layer scored, the tie rule"]["problems"]
    checks["the removal order is TPI1's committed one"]["ok"] = \
        checks["the removal order is TPI1's committed one"]["journal"] == tpi1_sel["removal_order"]
    first_bad = next((i for i, r in enumerate(rounds) if any(k not in r for k in ROUND_FIELDS)), None)
    return {"checks": checks, "missing": missing, "all_ok": not missing and all(c["ok"] for c in checks.values()),
            "minimal_recompute": (None if not missing else
                                  "the whole search from round 1 (a step or trace field is missing)"
                                  if any(m.startswith(("step.", "trace.")) for m in missing)
                                  else f"from round {first_bad + 1}: the rounds before it resume")}


def sealed_statistics(n_root: int, width_config_hash: str) -> dict:
    """The sealed teacher statistics, from TPI1's committed init record: refused unless that record is COMPLETE, its
    RESIDUAL_WIDTH and diagnostics gates passed, the statistics are the RESIDUAL_WIDTH operator's under the journal's
    config hash, hold 37 residual points (the 36 layer inputs and the post-final-norm output) and the file has a
    verified relay copy."""
    rec = json.loads(TPI1_INIT_RECORD.read_text())
    st, gates = rec.get("statistics") or {}, rec.get("gates") or {}
    relay = (rec.get("relay") or {}).get("statistics") or {}
    name = st.get("file")
    shape = ((st.get("tensors") or {}).get("residual_sum") or {}).get("shape")
    bad = [w for w, ok in (
        ("the record is not COMPLETE", rec.get("status") == "COMPLETE"),
        ("its RESIDUAL_WIDTH gate did not pass", (gates.get("RESIDUAL_WIDTH") or {}).get("equal") is True),
        ("its 37-point diagnostics did not reproduce", (rec.get("statistics_reproduction") or {}).get("equal") is True),
        ("the statistics are not RESIDUAL_WIDTH's under the journal's config hash",
         st.get("config_hash") == width_config_hash),
        (f"the statistics do not hold {n_root + 1} residual points", shape == [n_root + 1, shape and shape[-1]]),
        ("the relay copy is not verified at the recorded sha256",
         relay.get("verified") is True and (relay.get("files_sha256") or {}).get(name) == st.get("sha256")))
        if not ok]
    if bad:
        raise SelectionError(f"{TPI1_INIT_RECORD.name}: the sealed statistics cannot be reused: {bad}")
    return {"file": name, "sha256": st["sha256"], "bytes": int(st["bytes"]), "tensors": st["tensors"],
            "stats_tokens": int(st["stats_tokens"]), "points": st["points"], "profile_id": st["profile_id"],
            "operator": st["operator"], "config_hash": st["config_hash"],
            "relay": {"repo": relay["relay"], "path": f"{relay['prefix']}/{name}"},
            "tpi1_init_record": {"path": str(TPI1_INIT_RECORD.relative_to(REPO)), "sha256": sha256_file(TPI1_INIT_RECORD),
                                 "run_id": rec["run_id"]},
            "tpi1_gates": {"RESIDUAL_WIDTH": gates["RESIDUAL_WIDTH"],
                           "statistics_reproduction": rec["statistics_reproduction"]},
            "rule": "loaded on the pod in place of a second teacher statistics pass; its sha256, tensor layout and "
                    "residual_count (the statistics positions) must equal these values and P at 1,024 recomputed "
                    "from it must reproduce the recorded diagnostics before P at 1,280 is built"}


def extract(root: Path = REPO) -> dict:
    d = json.loads(DESIGN.read_text())
    td = json.loads(TPI1_DESIGN.read_text())
    ts = json.loads(TPI1_SELECTION.read_text())
    src = td["source_search"]
    jp = root / src["journal"]["path"]
    if sha256_file(jp) != src["journal"]["sha256"] or ts["source"]["journal"]["sha256"] != src["journal"]["sha256"]:
        raise SelectionError("the journal is not the one TPI1's design and selection bind")
    rows = TSEL.last_rows(jp)
    dep_id = ts["states"]["DEPTH"]["state_id"]
    row = rows[dep_id]
    step = row["steps"][-1]
    rc = resume_checks(row, step, design=d, tpi1_design=td, tpi1_sel=ts)
    if not rc["all_ok"]:
        bad = {k: v for k, v in rc["checks"].items() if not v["ok"]}
        raise SelectionError(f"the journal's DEPTH search cannot be resumed: missing {rc['missing']}; {sorted(bad)}")
    n_root = ts["n_root_blocks"]
    keep = int(d["arms"]["depth_only"]["geometry"]["num_hidden_layers"])
    resumed = len(step["artifacts"]["search_rounds"])
    new = [n_root - r for r in range(resumed, n_root - keep)]
    q = int(d["arms"]["ffn_attention"]["geometry"]["num_attention_heads"])
    if {len(v) for v in ts["query_heads"].values()} != {q}:
        raise SelectionError(f"TPI1's attention selection keeps {sorted({len(v) for v in ts['query_heads'].values()})} "
                             f"heads, not this arm's {q}")
    wpts = ts["width"]["reproduction_gate"]["all_points_diagnostics"]["point_weights"]
    if len(wpts) != n_root + 1 or (wpts[0], wpts[-1], set(wpts[1:-1])) != (9.0, 8.0, {1.0}):
        raise SelectionError(f"the 37-point weights are not 9 / 1 / 8: {wpts}")
    return {
        "schema": SCHEMA,
        "source": {"journal": src["journal"], "search_run": src["run"], "root_state_id": src["root_state_id"],
                   "tpi1_selection": {"path": str(TPI1_SELECTION.relative_to(REPO)),
                                      "sha256": sha256_file(TPI1_SELECTION)},
                   "root_teacher_id": d["teacher"]["root_teacher_id"],
                   "root_teacher_sha256": d["teacher"]["root_teacher_sha256"], "seed": int(src["seed"])},
        "n_root_blocks": n_root,
        "depth_only": {
            "state_id": dep_id, "step": {k: step[k] for k in STEP_FIELDS},
            "resume_checks": rc,
            "rounds": step["artifacts"]["search_rounds"],
            "resume": {"rounds_replayed": resumed, "first_new_round": resumed + 1,
                       "target_layers": keep, "new_rounds": len(new), "new_candidate_subsets": sum(new),
                       "candidates_per_new_round": new,
                       "rescore": {"round": resumed, "removed_before": step["artifacts"]["search_rounds"][-1]["removed_before"],
                                   "candidate": step["artifacts"]["search_rounds"][-1]["chosen"],
                                   "recorded": step["artifacts"]["search_rounds"][-1]["chosen_score"],
                                   "tolerance": 0.0,
                                   "why": "the operator scores round 8's chosen subset again and requires the recorded "
                                          "value exactly (DEPTH KL scores are bit-identical across sessions at a "
                                          "fixed execution on the L40S)"},
                       "journal_seconds_rounds_1_8": step["wall_seconds"]}},
        "ffn_attention": {
            "ffn": {"intermediate_size": int(d["arms"]["ffn_attention"]["geometry"]["intermediate_size"]),
                    "impl_id": ts["steps"]["FFN"]["impl_id"], "profile_id": ts["steps"]["FFN"]["profile_id"],
                    "config_hash": ts["steps"]["FFN"]["config_hash"],
                    "gate": {"intermediate_size": 3072, "state_id": ts["states"]["FFN"]["state_id"],
                             "artifact_digest": ts["states"]["FFN"]["artifact_digest"],
                             "rule": "the same operator run at 3,072 neurons per block, sharing the statistics pass, "
                                     "must build the journal's FFN state on the teacher bit for bit"}},
            "attention": {"num_attention_heads": q, "state_id": ts["states"]["ATTENTION"]["state_id"],
                          "artifact_digest": ts["states"]["ATTENTION"]["artifact_digest"],
                          "query_heads": ts["query_heads"],
                          "rule": "reused: the journal's kept_heads on the teacher; their child must equal the state's "
                                  "digest"}},
        "residual_width": {
            "hidden_size": int(d["arms"]["residual_width"]["geometry"]["hidden_size"]),
            "points": list(range(n_root + 1)), "point_weights": wpts, "post_norm_point": n_root,
            "gate": {"state_id": ts["width"]["reproduction_gate"]["state_id"],
                     "artifact_digest": ts["width"]["reproduction_gate"]["artifact_digest"],
                     "all_points_diagnostics": ts["width"]["reproduction_gate"]["all_points_diagnostics"],
                     "config_hash": ts["steps"]["RESIDUAL_WIDTH"]["config_hash"],
                     "rule": "P at 1,024 over all 37 points recomputed from the sealed statistics reproduces the "
                             "recorded diagnostics; the digest gate is TPI1's, passed by the pass that produced "
                             "the file (statistics.tpi1_gates)"},
            "statistics": sealed_statistics(n_root, ts["steps"]["RESIDUAL_WIDTH"]["config_hash"])},
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
