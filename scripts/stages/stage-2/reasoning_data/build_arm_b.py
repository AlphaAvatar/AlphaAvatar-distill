#!/usr/bin/env python3
"""Arm B's pack: verified sessions -> source cells -> nested ladder -> trainer pack. $0, CPU.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.reasoning_data.build_arm_b \
        --variant stage2.reasoning_master@v1 --sessions <adapter sessions.jsonl> ... \
        --reuse <selection.jsonl> ... --out <pack dir>

The end of the chain `generation_plan` -> `generation_driver` (records) ->
`generation_adapter` (verified, selected, one record per unit) -> here. Reused
targets (`build_supporting_tasks.reused_sessions`) join as whole units,
re-verified by their sha256. Every session is placed in its CELL
``<stratum>/<source>`` and the ladder interleaves cells by the variant's
source shares (`source_plan.allocation`), so a source's share is a share of
unique supervised positions exactly as a stratum's is. A session whose cell
the variant does not give a positive share is refused, never moved to another
cell: the corpus id names one allocation. Packing is `build_master_ladder`'s
(whole units only; each unit's target identity in ``targets.jsonl``).

``--validate-chain`` runs the same code on what exists at $0 -- the Stage-0
teacher records through the adapter, plus the reuse selections -- into a pack
labelled CHAIN VALIDATION (its cells' shares renormalized over the cells
present, its rung a size those cells can fill). It shows the mechanism end to
end on real teacher text; it is NOT arm B, whose generated cells need P1.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from stages.reasoning_data import assets as A
from stages.reasoning_data import build_master_ladder as B
from stages.reasoning_data import source_plan as SP

SALT = "stage2.reasoning_master/v1"
VALIDATION = A.LOG_DIR / "audit/arm_b_chain_validation.json"


def cell_shares(spec: dict, variant: str) -> dict[str, float]:
    return {f"{stratum}/{source}": share for stratum, cells in SP.allocation(spec, variant).items()
            for source, share in cells.items() if share > 0}


def place(sessions: list[dict], shares: dict[str, float]) -> list[dict]:
    """Each session in its cell; a session outside the variant's cells refuses."""
    out, bad = [], Counter()
    for s in sessions:
        cell = f"{s['stratum']}/{(s.get('attributes') or {}).get('source', '')}"
        if cell not in shares:
            bad[cell] += 1
            continue
        out.append({**s, "stratum": cell, "data_type": s.get("data_type") or s["stratum"]})
    if bad:
        raise SystemExit(f"sessions in cells the variant does not allocate: {dict(bad)}")
    return out


def build(sessions: list[dict], spec: dict, shares: dict[str, float], targets: list[int], tokenizer,
          *, drift: bool = False) -> dict:
    ladder_spec = {**{k: spec[k] for k in ("block_len", "reserve_tokens")}, "shares": shares,
                   "rung_targets": targets, "salt": SALT, "allow_composition_drift": drift}
    return B.build(place(sessions, shares), ladder_spec, tokenizer), ladder_spec


def _reused(paths) -> list[dict]:
    from stages.reasoning_data.build_supporting_tasks import reused_sessions
    return [s for p in paths for s in reused_sessions(Path(p))]


def validate_chain(reuse: list[Path], out: Path) -> dict:
    from stages.reasoning_data import generation_adapter as G
    spec = A.read_json(A.CONFIG_DIR / "corpus_spec.json")
    tok = B.load_tokenizer()
    tasks = {tid: {**t, "source": "s0_generators_v2_calsplit"} for tid, t in A.calibration_tasks().items()}
    got = G.adapt(tasks, G.stage0_records(), lambda t: SP.stratum_of(t, spec), tokenizer=tok)
    sessions = got["sessions"] + _reused(reuse)
    full = cell_shares(spec, "stage2.reasoning_master@v1")
    present = {f"{s['stratum']}/{(s.get('attributes') or {}).get('source', '')}" for s in sessions}
    total = sum(v for k, v in full.items() if k in present)
    shares = {k: v / total for k, v in full.items() if k in present}
    probe, _ = build(sessions, {**spec, "reserve_tokens": 0}, shares, [10 ** 9], tok, drift=True)
    # a rung the present cells can fill after the spec's reserve (the trainer's validation tail)
    target = int((probe["ladder"]["composition_capacity"] - spec["reserve_tokens"]) * 0.85)
    built, ladder_spec = build(sessions, spec, shares, [target], tok)
    meta = B.write_pack(out, built, ladder_spec, {"chain_validation": True, "not": "arm B",
                                                  "sessions": len(sessions)})
    rung = meta["rungs"][0]
    from aadistill.data.ladder import ladder_blocks
    train, val, stats = ladder_blocks(out, target, n_val=16)
    targets = list(A.jsonl(out / "targets.jsonl")) if (out / "targets.jsonl").exists() else []
    return {"schema": "aadistill.stage2.reasoning_data.arm_b_chain_validation/v1",
            "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "label": "CHAIN VALIDATION at $0 -- NOT arm B (its generated cells need P1)",
            "inputs": {"stage0_teacher_records": len(G.stage0_records()), "adapter_units_selected": len(got["selected"]),
                       "reuse_selections": [{"path": SP.repo_path(p), "sha256": A.sha256_file(p)} for p in reuse],
                       "sessions": len(sessions)},
            "cells_present": {k: round(v, 4) for k, v in sorted(shares.items())},
            "cells_absent_need_P1": sorted(set(full) - present),
            "rung": {k: rung.get(k) for k in ("target_supervised_tokens", "reachable", "n_blocks",
                                              "actual_supervised_tokens", "n_units", "n_sessions")},
            "excluded": meta["excluded_summary"],
            "trainer_reader_roundtrip": {"train_blocks": int(train[0].shape[0]), "val_blocks": int(val[0].shape[0]),
                                         "train_supervised_tokens": stats["train_supervised_tokens"],
                                         "val_disjoint_from_all_rungs": stats["val_disjoint_from_all_rungs"]},
            "targets_listed": len(targets),
            "targets_with_identity": sum(1 for t in targets if t["target"]),
            "outputs": meta["outputs"],
            "reading": ("every adapter session carries its record's identity into targets.jsonl; reused units "
                        "carry their corpus_v2 session sha; the packer admits whole units only"),
            "authorizes": "nothing"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="stage2.reasoning_master@v1")
    ap.add_argument("--sessions", nargs="*", default=[])
    ap.add_argument("--reuse", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--validate-chain", action="store_true")
    args = ap.parse_args(argv)
    if args.validate_chain:
        doc = validate_chain([Path(p) for p in args.reuse], Path(args.out))
        VALIDATION.write_text(json.dumps(doc, indent=1) + "\n")
        print(VALIDATION.relative_to(A.REPO), doc["rung"], doc["excluded"])
        return 0
    plan = SP.plan(args.variant)
    if not (plan["admitted"] and plan["chosen"]):
        raise SystemExit(f"{args.variant}: not the chosen, admitted variant ({plan['status']})")
    spec = A.read_json(A.CONFIG_DIR / "corpus_spec.json")
    sessions = [s for p in args.sessions for s in A.jsonl(Path(p))] + _reused(args.reuse)
    built, ladder_spec = build(sessions, spec, cell_shares(spec, args.variant), spec["rung_targets"],
                               B.load_tokenizer())
    meta = B.write_pack(Path(args.out), built, ladder_spec, {
        "corpus_id": args.variant, "sessions_sha256": {p: A.sha256_file(Path(p)) for p in args.sessions},
        "reuse_sha256": {p: A.sha256_file(Path(p)) for p in args.reuse}})
    print(json.dumps({r["target_supervised_tokens"]: r.get("n_blocks") for r in meta["rungs"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
