#!/usr/bin/env python3
"""Variant 1's storage, from the build to the training session's closeout, in the drivers' real order ($0).

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python scripts/stages/stage-1/depth_merge/storage.py \\
        [--write]        # reads the relay's live headroom (LFS batch endpoint, nothing sent) and its inventory

`sequence` is the relay's net change over the chain, step by step, against what it holds NOW (whose bytes are
already in the quota, so the live headroom excludes them): the build puts the init and the statistics; the training
session runs SOB1's depth rotation from an empty state (`single_operator_budget.launch.rotation_need`: each save
point uploaded while the previous one is still there, the final uploaded while the last save point is there, then
the save points go; seed 1's final stays while seed 2 rotates); each session's closeout archive. The dev box's side
is each session's evidence bound (`session.evidence_bound_gib`). `need_bytes` -- the chain's peak plus the design's
margin -- is what the build session's launch gate asks the relay for (`session.relay_room`), so the init is never
built for a training session that would not fit.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.depth_merge import common as DC  # noqa: E402

GIB = 2 ** 30
RECORD = DC.LOGS / "plans/storage_plan.json"


def sizes(root: Path = REPO) -> dict:
    from stages.depth_merge import session as S
    return S.checkpoint_bytes(root)


def sequence(root: Path = REPO) -> dict:
    """The chain's relay steps (net bytes added vs now) and the dev box's bound (module docstring)."""
    from stages.depth_merge import session as S
    from stages.single_operator_budget import launch as SL
    d = DC.design()
    sz = sizes(root)
    st = d["storage"]
    build = int(sz["init_bf16"]) + int(st["statistics_bytes_bound"])
    archive = int(st["closeout_archive_bytes_bound"])
    src = json.loads((root / d["baseline"]["training_plan"]["path"]).read_text())
    rot = SL.rotation_need(src, {}, sz)
    steps, net = [], 0

    def add(what, delta):
        nonlocal net
        net += int(delta)
        steps.append({"step": what, "net_bytes": net, "net_gib": round(net / GIB, 2)})
    add("build: the init (bf16) and the statistics put", build)
    add("build closeout: its evidence archive", archive)
    for t in rot["trace"]:
        #: SOB1's depth units ARE this arm's units outside the declared fields (plan.py): the same rotation
        steps.append({"step": "training: " + t["step"].replace(DC.SOB1_DEPTH, "depth_merge"),
                      "net_bytes": build + archive + t["net_bytes"],
                      "net_gib": round((build + archive + t["net_bytes"]) / GIB, 2)})
    net = build + archive + rot["end_bytes"]
    add("training closeout: its evidence archive", archive)
    peak = max(s["net_bytes"] for s in steps)
    margin = int(float(st["relay_margin_gib"]) * GIB)
    dev = {k: S.evidence_bound_gib(k, S_variant()) for k in (S_variant().build_kind, *S_variant().train_kinds)}
    return {"relay": {"steps": steps, "peak_bytes": peak, "peak_gib": round(peak / GIB, 2), "end_bytes": net,
                      "end_gib": round(net / GIB, 2), "margin_bytes": margin, "need_bytes": peak + margin,
                      "need_gib": round((peak + margin) / GIB, 2),
                      "peak_at": max(steps, key=lambda s: s["net_bytes"])["step"]},
            "dev_box": {"evidence_bound_gib": dev, "peak_gib": max(dev.values()),
                        "rule": "one session's evidence, pull archive and teardown archive at a time (evidence_bound_gib)"},
            "sizes": {k: v for k, v in sz.items() if k != "rule"},
            "rule": "net bytes the relay must accept vs what it holds now, in the drivers' order (module docstring)"}


def S_variant():
    from stages.depth_merge import launch as L
    return L.VARIANT


def live(repo: str | None = None) -> dict:
    """The relay's live headroom (bisected, nothing sent) and its inventory total ($0, read-only)."""
    from huggingface_hub import HfApi
    from aadistill.runtime.hub_capacity import would_accept
    from shared.deployment import MAIN_RELAY
    from stages.single_operator_budget import launch as SL
    repo = repo or MAIN_RELAY
    tok = Path(os.path.expanduser("~/.cache/huggingface/token")).read_text().strip()
    lo, hi = 0, 128 * GIB
    while hi - lo > 16 * 2 ** 20:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if would_accept(repo, SL.chunks(mid), tok).accepted else (lo, mid)
    used = sum(int(o.size) for o in HfApi(token=tok).list_lfs_files(repo))
    return {"relay": repo, "headroom_bytes": lo, "headroom_gib": round(lo / GIB, 3), "refused_at_gib": round(hi / GIB, 3),
            "used_bytes": used, "used_gib": round(used / GIB, 3), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "rule": "LFS batch endpoint bisection with fresh oids in objects of <= 10 GiB (nothing stored); used = "
                    "the sum of list_lfs_files (the billing basis)"}


def plan(live_state: dict, root: Path = REPO) -> dict:
    seq = sequence(root)
    short = seq["relay"]["need_bytes"] - int(live_state["headroom_bytes"])
    return {"schema": "aadistill.dmr1.storage_plan/v1", "variant": "DMR1-M", "live": live_state, **seq,
            "relay_shortfall_bytes": max(short, 0), "relay_shortfall_gib": round(max(short, 0) / GIB, 2),
            "fits": short <= 0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    doc = plan(live())
    if a.write:
        RECORD.parent.mkdir(parents=True, exist_ok=True)
        RECORD.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: doc[k] for k in ("fits", "relay_shortfall_gib")} | {"peak": doc["relay"]["peak_gib"],
                      "need": doc["relay"]["need_gib"], "headroom": doc["live"]["headroom_gib"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
