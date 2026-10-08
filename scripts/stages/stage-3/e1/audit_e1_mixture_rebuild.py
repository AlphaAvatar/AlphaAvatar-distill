#!/usr/bin/env python
"""Rebuild Experiment 1's token ladder from the source corpus and verify it is
identical to the historical artifact, rung by rung and seed by seed.

    PYTHONPATH=src python scripts/data/audit_e1_mixture_rebuild.py \
        --rebuilt artifacts/audit/ladder_uniform_rebuild \
        --historical artifacts/stage3/ladder_uniform_probe \
        --seeds 20260726 20260801 \
        --out artifacts/audit/e1_mixture_rebuild.json

Why this exists
---------------
`interleave`, `order_blocks` and `block_token_mix` moved from
`scripts/data/build_token_ladder.py` into `src/aadistill/data/mixture.py` on
2026-08-04. Those three functions decide the session order, the block order and
therefore the contents of every rung of the ladder Experiment 1 trained on. If
the move perturbed any of them, all 25 Experiment 1 arms would be describing a
different dataset than their manifests claim, and the completed experiment would
lose its reproducibility.

This does not replay the ladder from the packed blocks — it rebuilds it from
`sessions.jsonl` through the relocated code, so the session interleave and the
packing are exercised too, not just the block reordering.

What is compared
----------------
1. **artifact hashes** — sha256 of `blocks.npz` and `audit.jsonl` against the
   values recorded in the historical `ladder.json`;
2. **array identity** — `input_ids`, `ce_mask`, `content_mask` element-for-element;
3. **block order** — the ordered session ids inside every block, which is what
   `order_blocks` actually determines;
4. **rung cuts** — every field of every rung entry: `n_blocks`,
   `actual_supervised_tokens`, `n_sessions`, `real_tokens`, `padding_tokens`,
   `terminal_truncations`, `token_mix`, `session_mix`;
5. **nesting** — rung k's blocks are an exact prefix of rung k+1's;
6. **per-seed training streams** — for each rung and each Experiment 1 seed, the
   deterministic block stream the trainer consumes
   (`stream_block_indices`) over that arm's full step budget. This is what makes
   the check "both seeds" rather than only "both packs": the pack is seed-free by
   design, and the seed enters at consumption time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.infrastructure.env import code_state  # noqa: E402
from aadistill.training.train import stream_block_indices  # noqa: E402

RUNG_FIELDS = ("target_supervised_tokens", "reachable", "n_blocks",
               "actual_supervised_tokens", "n_sessions", "real_tokens",
               "padding_tokens", "terminal_truncations", "token_mix",
               "session_mix", "token_counts")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def block_session_ids(audit_rows):
    """The ordered session ids inside each block — what `order_blocks` decides."""
    return [[s["session_id"] for s in row["sessions"]] for row in audit_rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuilt", required=True, type=Path)
    ap.add_argument("--historical", required=True, type=Path)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--configs", type=Path,
                    default=REPO_ROOT / "configs/stage3/e1")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    checks: dict[str, object] = {}
    failures: list[str] = []

    def record(name, ok, detail=None):
        checks[name] = {"ok": bool(ok), **({"detail": detail} if detail else {})}
        if not ok:
            failures.append(name)

    hist_meta = json.loads((args.historical / "ladder.json").read_text())
    new_meta = json.loads((args.rebuilt / "ladder.json").read_text())

    # --- 1. artifact hashes ------------------------------------------------
    for fname, key in (("blocks.npz", "blocks"), ("audit.jsonl", "audit")):
        h_hist = hist_meta["outputs"][key]
        h_new = sha256_file(args.rebuilt / fname)
        h_disk = sha256_file(args.historical / fname)
        record(f"hash::{fname}::rebuilt_matches_recorded", h_new == h_hist,
               {"recorded": h_hist, "rebuilt": h_new})
        record(f"hash::{fname}::historical_file_intact", h_disk == h_hist,
               {"recorded": h_hist, "on_disk": h_disk})

    record("corpus::sessions_sha256",
           new_meta["sessions_sha256"] == hist_meta["sessions_sha256"],
           {"historical": hist_meta["sessions_sha256"],
            "rebuilt": new_meta["sessions_sha256"]})

    # --- 2. array identity -------------------------------------------------
    a_hist = np.load(args.historical / "blocks.npz")
    a_new = np.load(args.rebuilt / "blocks.npz")
    record("arrays::key_set", sorted(a_hist.keys()) == sorted(a_new.keys()),
           {"historical": sorted(a_hist.keys()), "rebuilt": sorted(a_new.keys())})
    for key in sorted(set(a_hist.keys()) & set(a_new.keys())):
        same_shape = a_hist[key].shape == a_new[key].shape
        equal = bool(same_shape and np.array_equal(a_hist[key], a_new[key]))
        detail = {"shape_historical": list(a_hist[key].shape),
                  "shape_rebuilt": list(a_new[key].shape)}
        if same_shape and not equal:
            diff = np.argwhere(a_hist[key] != a_new[key])
            detail["n_differing_elements"] = int(len(diff))
            detail["first_difference_at"] = diff[0].tolist() if len(diff) else None
        record(f"arrays::{key}", equal, detail)

    # --- 3. block order ----------------------------------------------------
    hist_audit = [json.loads(l) for l in (args.historical / "audit.jsonl").open()
                  if l.strip()]
    new_audit = [json.loads(l) for l in (args.rebuilt / "audit.jsonl").open()
                 if l.strip()]
    record("order::n_blocks", len(hist_audit) == len(new_audit),
           {"historical": len(hist_audit), "rebuilt": len(new_audit)})
    ids_hist, ids_new = block_session_ids(hist_audit), block_session_ids(new_audit)
    first_bad = next((i for i, (x, y) in enumerate(zip(ids_hist, ids_new))
                      if x != y), None)
    record("order::block_session_sequence", first_bad is None,
           {"first_differing_block": first_bad,
            "historical": ids_hist[first_bad][:6] if first_bad is not None else None,
            "rebuilt": ids_new[first_bad][:6] if first_bad is not None else None})

    # --- 4. rung cuts ------------------------------------------------------
    hist_rungs = {r["target_supervised_tokens"]: r for r in hist_meta["rungs"]}
    new_rungs = {r["target_supervised_tokens"]: r for r in new_meta["rungs"]}
    record("rungs::same_targets", sorted(hist_rungs) == sorted(new_rungs),
           {"historical": sorted(hist_rungs), "rebuilt": sorted(new_rungs)})
    for target in sorted(set(hist_rungs) & set(new_rungs)):
        h, n = hist_rungs[target], new_rungs[target]
        bad = {f: {"historical": h.get(f), "rebuilt": n.get(f)}
               for f in RUNG_FIELDS if h.get(f) != n.get(f)}
        record(f"rungs::{target}::all_fields", not bad, bad or None)

    # --- 5. nesting --------------------------------------------------------
    # Nesting is claimed as prefix nesting on blocks *and* sessions. Comparing a
    # prefix of a list against itself would be vacuous, so test the substantive
    # properties: monotonic growth, and containment of each rung's session set in
    # the next rung's.
    ordered = [new_rungs[t] for t in sorted(new_rungs) if new_rungs[t]["reachable"]]
    nest_ok, nest_detail = True, {}
    for a, b in zip(ordered, ordered[1:]):
        na, nb = a["n_blocks"], b["n_blocks"]
        sess_a = {s for blk in ids_new[:na] for s in blk}
        sess_b = {s for blk in ids_new[:nb] for s in blk}
        step = {
            "monotonic_blocks": na <= nb,
            "monotonic_tokens": (a["actual_supervised_tokens"]
                                 <= b["actual_supervised_tokens"]),
            "sessions_contained": sess_a <= sess_b,
            "n_sessions_only_in_smaller": len(sess_a - sess_b),
        }
        if not (step["monotonic_blocks"] and step["monotonic_tokens"]
                and step["sessions_contained"]):
            nest_ok = False
            nest_detail[f"{a['target_supervised_tokens']}->"
                        f"{b['target_supervised_tokens']}"] = step
    record("rungs::nested_prefixes", nest_ok, nest_detail or None)

    # --- 6. per-seed training streams --------------------------------------
    stream_detail = {}
    stream_ok = True
    for cfg_path in sorted(args.configs.glob("e1_r*_pca.json")):
        cfg = json.loads(cfg_path.read_text())
        seed = cfg["seed"]
        if seed not in args.seeds:
            continue
        rung = int(cfg_path.name.split("_r")[1][:5].rstrip("k")) * 1000
        entry = next((r for r in new_rungs.values()
                      if abs(r["target_supervised_tokens"] - rung) < 1000), None)
        if entry is None:
            continue
        n_blocks = entry["n_blocks"]
        total = cfg["schedule"]["total_steps"] * cfg["batch"]["blocks_per_step"]
        a = stream_block_indices(n_blocks, seed, 0, total)
        b = stream_block_indices(n_blocks, seed, 0, total)
        deterministic = a == b
        # every consumed index must exist in the rebuilt rung
        in_range = all(0 <= i < n_blocks for i in a)
        # resume equivalence: any suffix re-derives from the counter alone
        mid = total // 2
        resume_ok = stream_block_indices(n_blocks, seed, mid, total - mid) == a[mid:]
        ok = deterministic and in_range and resume_ok
        stream_ok &= ok
        stream_detail[cfg_path.name] = {
            "seed": seed, "n_blocks": n_blocks, "blocks_consumed": total,
            "deterministic": deterministic, "indices_in_range": in_range,
            "resume_equivalent": resume_ok,
            "stream_sha256": hashlib.sha256(
                json.dumps(a).encode()).hexdigest()[:16],
        }
    record("streams::per_seed_deterministic_and_in_range", stream_ok, stream_detail)

    verdict = "pass" if not failures else "fail"
    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "verdict": verdict,
        "failures": failures,
        "rebuilt": str(args.rebuilt),
        "historical": str(args.historical),
        "seeds": args.seeds,
        "checks": checks,
        "code_state": code_state(REPO_ROOT),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))

    print(f"verdict: {verdict.upper()}   ({len(checks) - len(failures)}/{len(checks)} checks passed)")
    for name, res in checks.items():
        if not res["ok"]:
            print(f"  FAIL {name}: {json.dumps(res.get('detail'))[:300]}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
