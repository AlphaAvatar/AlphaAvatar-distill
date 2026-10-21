#!/usr/bin/env python3
"""Build a nested training ladder from one master corpus, as a trainer-ready pack.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.reasoning_data.build_master_ladder \
        --sessions <sessions.jsonl> --spec <ladder spec .json> --out <pack dir>

Input contract (one row per renderable session; the future generation step
writes exactly this):

    {"id": str, "unit_id": str, "stratum": str, "family": str | null,
     "attributes": {str: str}, "messages": [...], "tools": [...] | null,
     "data_type": str}

A UNIT (a static task, a whole episode's turns, a conversation's turns) is
indivisible: nesting, the reserve and the composition are all decided on units.
The spec declares `shares` (by stratum), `rung_targets` (unique supervised
tokens), `reserve_tokens`, `salt`, `block_len`.

Output, readable by `aadistill.data.ladder.ladder_blocks` unchanged:
`blocks.npz` (input_ids / ce_mask / content_mask), `audit.jsonl`, and
`ladder.json` whose `rungs` carry the trainer's fields plus each rung's full
accounting. Each rung's increment is packed best-fit on its own
(`sessions.pack_best_fit`), so rung k is `blocks[:n_k]`; the reserve is packed
last, as the validation tail. No session is ever cut.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from stages.reasoning_data import assets as A

from aadistill.data.nested_ladder import LadderUnit, build_ladder
from aadistill.data.sessions import RenderedSession, pack_best_fit, render_session, render_system_block

PACK_SCHEMA = "aadistill.stage2.reasoning_data.pack/v1"


def load_tokenizer(path: Path = A.TOKENIZER_DIR):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(path))


def render_all(tokenizer, sessions: list[dict], *, block_len: int):
    """Render every session; a session that does not fit is excluded, never cut."""
    rendered, excluded, system_ids = {}, [], {}
    for s in sessions:
        try:
            r = render_session(tokenizer, s, block_len=block_len)
        except ValueError as e:
            excluded.append({"id": s["id"], "unit_id": s["unit_id"], "why": str(e)[:160]})
            continue
        r.source_id = str(s["unit_id"])
        r.meta.update({"stratum": s["stratum"], "family": s.get("family"),
                       "attributes": s.get("attributes") or {}})
        rendered[s["id"]] = r
        if r.system_key not in system_ids:
            block = render_system_block(tokenizer, r.system_text, r.tools)
            system_ids[r.system_key] = tokenizer(block, add_special_tokens=False).input_ids
    return rendered, excluded, system_ids


def split_counts(r: RenderedSession, think_close_id: int) -> dict:
    """Supervised tokens before and after the reasoning delimiter."""
    sup = [t for t, m in zip(r.body_ids, r.body_mask) if m]
    cut = sup.index(think_close_id) if think_close_id in sup else len(sup)
    return {"reasoning_tokens": cut, "answer_tokens": len(sup) - cut,
            "rendered_tokens": r.n_rendered_tokens, "supervised_tokens": len(sup)}


def ladder_units(rendered: dict[str, RenderedSession], think_close_id: int) -> list[LadderUnit]:
    by_unit: dict[str, list[RenderedSession]] = defaultdict(list)
    for r in rendered.values():
        by_unit[r.source_id].append(r)
    units = []
    for uid, rows in sorted(by_unit.items()):
        counts: dict[str, int] = defaultdict(int)
        for r in rows:
            for k, v in split_counts(r, think_close_id).items():
                counts[k] += v
        counts["sessions"] = len(rows)
        m = rows[0].meta
        units.append(LadderUnit(unit_id=uid, stratum=m["stratum"], size=counts["supervised_tokens"],
                                family=m["family"], counts=dict(counts),
                                attributes=dict(m["attributes"])))
    return units


def pack(ladder: dict, rendered: dict[str, RenderedSession], system_ids: dict, *,
         block_len: int, pad_id: int):
    """Pack each rung increment, then the reserve, best-fit; return blocks and cuts."""
    by_unit: dict[str, list[RenderedSession]] = defaultdict(list)
    for r in rendered.values():
        by_unit[r.source_id].append(r)
    blocks, cuts, prev = [], [], 0
    for row in ladder["rungs"]:
        if not row["reachable"]:
            cuts.append(None)
            continue
        inc = [s for uid in ladder["order"][prev:row["n_units"]] for s in by_unit[uid]]
        blocks.extend(pack_best_fit(inc, system_ids, block_len=block_len, pad_id=pad_id))
        cuts.append(len(blocks))
        prev = row["n_units"]
    tail = [s for uid in ladder["reserved"] for s in by_unit[uid]]
    val_start = len(blocks)
    blocks.extend(pack_best_fit(tail, system_ids, block_len=block_len, pad_id=pad_id))
    return blocks, cuts, val_start


def block_accounting(blocks, block_len: int) -> dict:
    real = sum(b.audit["unpadded_length"] for b in blocks)
    return {"blocks": len(blocks), "real_tokens": real,
            "padding_tokens": len(blocks) * block_len - real,
            "packing_efficiency": round(real / (len(blocks) * block_len), 4) if blocks else None,
            "supervised_tokens": sum(b.n_supervised for b in blocks),
            "sessions": sum(len(b.audit["sessions"]) for b in blocks),
            "multi_session_blocks": sum(1 for b in blocks if len(b.audit["sessions"]) > 1)}


def build(sessions: list[dict], spec: dict, tokenizer) -> dict:
    """Everything up to the bytes: rendering, ladder, packing, accounting."""
    block_len = int(spec["block_len"])
    rendered, excluded, system_ids = render_all(tokenizer, sessions, block_len=block_len)
    think_close = tokenizer.convert_tokens_to_ids("</think>")
    units = ladder_units(rendered, think_close)
    ladder = build_ladder(units, spec["shares"], spec["rung_targets"], salt=spec["salt"],
                          reserve=int(spec.get("reserve_tokens", 0)),
                          allow_composition_drift=bool(spec.get("allow_composition_drift", False)))
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    blocks, cuts, val_start = pack(ladder, rendered, system_ids, block_len=block_len, pad_id=pad_id)
    for row, cut in zip(ladder["rungs"], cuts):
        if cut is not None:
            row["n_blocks"] = cut
            row["packed"] = block_accounting(blocks[:cut], block_len)
    return {"ladder": ladder, "blocks": blocks, "val_start": val_start, "excluded": excluded,
            "n_rendered": len(rendered), "n_units": len(units), "pad_id": int(pad_id),
            "system_prompt_groups": len(system_ids),
            "validation_tail": block_accounting(blocks[val_start:], block_len)}


def write_pack(out: Path, built: dict, spec: dict, provenance: dict) -> dict:
    import numpy as np

    out.mkdir(parents=True, exist_ok=True)
    blocks = built["blocks"]
    np.savez(out / "blocks.npz",
             input_ids=np.asarray([b.input_ids for b in blocks], dtype=np.int32),
             ce_mask=np.asarray([b.ce_mask for b in blocks], dtype=bool),
             content_mask=np.asarray([b.content_mask for b in blocks], dtype=bool))
    cum = 0
    with open(out / "audit.jsonl", "w") as f:
        for i, b in enumerate(blocks):
            cum += b.n_supervised
            f.write(json.dumps({"block": i, "cumulative_supervised": cum, **b.audit}) + "\n")
    rungs = []
    for row in built["ladder"]["rungs"]:
        entry = {"target_supervised_tokens": row["target"], "reachable": row["reachable"],
                 "n_blocks": row.get("n_blocks"), "terminal_truncations": 0}
        if row["reachable"]:
            p = row["packed"]
            entry.update(actual_supervised_tokens=p["supervised_tokens"], n_sessions=p["sessions"],
                         real_tokens=p["real_tokens"], padding_tokens=p["padding_tokens"],
                         n_units=row["n_units"], accounting=row["accounting"])
        rungs.append(entry)
    files = {n: A.sha256_file(out / n) for n in ("blocks.npz", "audit.jsonl")}
    meta = {"schema": PACK_SCHEMA, "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "block_len": int(spec["block_len"]), "pad_id": built["pad_id"],
            "packing": "best-fit-decreasing per rung increment, never cut (sessions.pack_best_fit)",
            "declared_mixture": spec["shares"], "spec": spec, "rungs": rungs,
            "validation_tail": {"first_block": built["val_start"], **built["validation_tail"]},
            "order_sha256": hashlib.sha256(json.dumps(built["ladder"]["order"]).encode()).hexdigest(),
            "reserved_units": built["ladder"]["reserved"], "excluded_sessions": built["excluded"],
            "outputs": files, "provenance": provenance}
    (out / "ladder.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    spec = A.read_json(Path(args.spec))
    sessions = list(A.jsonl(Path(args.sessions)))
    built = build(sessions, spec, load_tokenizer())
    meta = write_pack(Path(args.out), built, spec, {"sessions_sha256": A.sha256_file(Path(args.sessions))})
    print(json.dumps({r["target_supervised_tokens"]: r.get("n_blocks") for r in meta["rungs"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
