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
margin -- is what a first build session's launch gate asks the relay for (`session.relay_room`), so the init is never
built for a training session that would not fit. A build TAKEOVER asks for what the chain still adds (`remaining`:
the chain less each product the attempt holds on the relay under this identity, at the chain's own count), against a
live headroom that already excludes those products -- the same margin, never the whole chain twice.
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
GROUPS_HELD = ("statistics", "init")


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


V2_RECORD = DC.RECURRENCE_LOGS / "plans/storage_plan.json"
#: relay objects with a verified copy on the dev box (re-hashed 2026-11-28 against their records), so removing the
#: relay copy deletes no artifact: the maintainer's decision, proposed for the revised variant 2's room
DUPLICATES = {
    "durable/stage1_single_operator_budget/residual_width/init/c853640b1ce2392c":
        "logs/stages/stage-1/single_operator_budget/runs/sob1_phase1_20261119_194259/evidence/residual_width/init_record.json",
    "durable/stage1_single_operator_budget/ffn_attention/init/4429c94994a5bd1f":
        "logs/stages/stage-1/single_operator_budget/runs/sob1_phase1_20261119_194259/evidence/ffn_attention/init_record.json"}


def v2_sequence(root: Path = REPO, copies: int | None = None) -> dict:
    """The revised variant 2's relay steps against what the relay holds now: the depth_loop_kept session, then the
    depth_loop_merged one, each its plan's rotation (`single_operator_budget.launch.rotation_need` under the plan's
    ``durability.savepoint_copies``, or ``copies``) and its closeout archive; the first session's final stays on the
    relay while the second rotates. No build: both initializations are on the relay already."""
    from stages.depth_merge import plan as PL
    from stages.single_operator_budget import launch as SL
    sz = sizes(root)
    st = DC.design()["storage"]
    archive = int(st["closeout_archive_bytes_bound"])
    steps, net = [], 0
    for arm in ("depth_loop_kept", "depth_loop_merged"):
        plan = json.loads(PL.out_path(arm).read_text())
        if copies is not None:
            plan = {**plan, "durability": {**plan["durability"], "savepoint_copies": int(copies)}}
        rot = SL.rotation_need(plan, {}, sz)
        for s in rot["trace"]:
            steps.append({"step": f"{arm}: {s['step']}", "net_bytes": net + s["net_bytes"],
                          "net_gib": round((net + s["net_bytes"]) / GIB, 2)})
        net += rot["end_bytes"] + archive
        steps.append({"step": f"{arm} closeout: its evidence archive", "net_bytes": net, "net_gib": round(net / GIB, 2)})
    peak = max(s["net_bytes"] for s in steps)
    margin = int(float(st["relay_margin_gib"]) * GIB)
    return {"steps": steps, "peak_bytes": peak, "peak_gib": round(peak / GIB, 2), "margin_bytes": margin,
            "need_bytes": peak + margin, "need_gib": round((peak + margin) / GIB, 2),
            "peak_at": max(steps, key=lambda s: s["net_bytes"])["step"],
            "savepoint_copies": copies if copies is not None else "the plans'"}


def v2_plan(live_state: dict, root: Path = REPO, dev_box: dict | None = None) -> dict:
    """The revised variant 2's storage: the need under each rotation, against the live headroom as it is and as it
    would be with the relay copies of `DUPLICATES` removed (their dev-box copies stay)."""
    head = int(live_state["headroom_bytes"])
    dup = {}
    for prefix, rec_path in DUPLICATES.items():
        rec = json.loads((root / rec_path).read_text())["relay"]["init"]
        dup[prefix] = {"bytes": int(sum((rec.get("bytes") or {}).values())), "record": rec_path,
                       "relay_sha256": rec["files_sha256"]}
    freed = sum(v["bytes"] for v in dup.values())
    options = {}
    for copies in (1, 2):
        seq = v2_sequence(root, copies)
        for label, room in (("as_is", head), ("duplicates_removed", head + freed)):
            options[f"copies_{copies}_{label}"] = {
                "need_gib": seq["need_gib"], "headroom_gib": round(room / GIB, 2),
                "fits": seq["need_bytes"] <= room, "margin_gib": round((room - seq["need_bytes"]) / GIB, 2),
                "peak_at": seq["peak_at"]}
    return {"schema": "aadistill.dmr1.v2_storage_plan/v1", "variant": "DMR1-R", "live": live_state,
            "dev_box_live": dev_box, "sequence_as_planned": v2_sequence(root), "duplicates": dup,
            "duplicates_bytes": freed, "duplicates_gib": round(freed / GIB, 2), "options": options,
            "kept": {"V1 seed 20260726 final (7.46 GiB)": "kept (maintainer 2026-11-28)",
                     "V1 seed 20260801 save point step 5,520 (19.50 GiB)": "kept: resumable (maintainer 2026-11-28)",
                     "V1 alpha-merged init (3.73 GiB)": "kept: V2-M's base", "SOB1 depth init (3.73 GiB)": "kept: V2-D's base"},
            "rule": "need = the two sessions' peak (net bytes vs now, the drivers' order) + the 1 GiB margin; headroom = "
                    "the live LFS bisection; nothing is deleted or bought by this plan"}


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


def counted(groups, root: Path = REPO) -> dict[str, int]:
    """How the chain counts each build product it starts with (the init at its bf16 size, the statistics at their
    bound): what a takeover that holds the group on the relay no longer adds."""
    count = {"init": int(sizes(root)["init_bf16"]), "statistics": int(DC.design()["storage"]["statistics_bytes_bound"])}
    return {g: count[g] for g in groups}


def remaining(held: dict, root: Path = REPO) -> dict:
    """What the chain still adds when a build takeover holds ``held`` (group -> its relay record, `build_driver.held`):
    every step of the chain includes the build's products, so each held group's count comes off every step -- the
    peak with them -- and the margin stays. The live headroom it is asked of already excludes the held bytes."""
    seq = sequence(root)["relay"]
    credit = counted(held, root)
    peak = seq["peak_bytes"] - sum(credit.values())
    return {"held": sorted(held), "credit_bytes": credit,
            "held_actual_bytes": {g: int(sum((r.get("bytes") or {}).values())) for g, r in held.items()},
            "remaining_peak_bytes": peak, "remaining_peak_gib": round(peak / GIB, 2), "peak_at": seq["peak_at"],
            "margin_bytes": seq["margin_bytes"], "need_bytes": peak + seq["margin_bytes"],
            "need_gib": round((peak + seq["margin_bytes"]) / GIB, 2), "first_run_need_bytes": seq["need_bytes"]}


def dev_box_live() -> dict:
    """The dev box's filesystem under the sessions' store (one filesystem holds the store, every --scr and the
    retained weights): its live free bytes, and what is left after the largest session evidence bound."""
    import shutil
    from stages.depth_merge import session as S
    store = DC.local_store(next(iter(DC.ARMS)))
    probe = next((q for q in (store, *store.parents) if q.exists()), Path("/"))
    free = shutil.disk_usage(probe).free
    peak = max(S.evidence_bound_gib(k, S_variant()) for k in S_variant().kinds) * GIB
    return {"path": str(probe), "free_bytes": free, "free_gib": round(free / GIB, 2),
            "peak_evidence_gib": round(peak / GIB, 2), "margin_after_peak_gib": round((free - peak) / GIB, 2),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "note": "the filesystem is shared with another project; its use moves without this project"}


def plan(live_state: dict, root: Path = REPO, dev_box: dict | None = None) -> dict:
    seq = sequence(root)
    short = seq["relay"]["need_bytes"] - int(live_state["headroom_bytes"])
    takeover = {}
    for label, groups in (("statistics_on_the_relay", ("statistics",)), ("both_products_on_the_relay", GROUPS_HELD)):
        r = remaining({g: {} for g in groups}, root)
        #: the headroom such a takeover finds: what it has now, less the held products' count (they went in since)
        found = int(live_state["headroom_bytes"]) - sum(r["credit_bytes"].values())
        takeover[label] = {"need_bytes": r["need_bytes"], "need_gib": r["need_gib"], "credit_bytes": r["credit_bytes"],
                           "headroom_then_gib": round(found / GIB, 2),
                           "shortfall_gib": round(max(r["need_bytes"] - found, 0) / GIB, 2)}
    return {"schema": "aadistill.dmr1.storage_plan/v1", "variant": "DMR1-M", "live": live_state,
            "dev_box_live": dev_box, **seq,
            "relay_shortfall_bytes": max(short, 0), "relay_shortfall_gib": round(max(short, 0) / GIB, 2),
            "fits": short <= 0, "relay_margin_after_first_run_need_gib": round(-short / GIB, 2),
            "takeover": takeover,
            "takeover_rule": "a build takeover asks only for what the chain still adds (session.build_need: the "
                             "chain less each product it holds on the relay under this identity, its files present, "
                             "at the chain's own count); its headroom is then the live headroom less those products, "
                             "so the margin is the same as the first run's"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--variant", choices=("1", "2"), default="1")
    a = ap.parse_args(argv)
    if a.variant == "2":
        doc = v2_plan(live(), dev_box=dev_box_live())
        if a.write:
            V2_RECORD.parent.mkdir(parents=True, exist_ok=True)
            V2_RECORD.write_text(json.dumps(doc, indent=1) + "\n")
        print(json.dumps({"options": doc["options"], "duplicates_gib": doc["duplicates_gib"]}, indent=1))
        return 0
    doc = plan(live(), dev_box=dev_box_live())
    if a.write:
        RECORD.parent.mkdir(parents=True, exist_ok=True)
        RECORD.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: doc[k] for k in ("fits", "relay_shortfall_gib", "relay_margin_after_first_run_need_gib")}
                     | {"peak": doc["relay"]["peak_gib"], "need": doc["relay"]["need_gib"],
                        "headroom": doc["live"]["headroom_gib"],
                        "takeover": {k: (v["need_gib"], v["headroom_then_gib"]) for k, v in doc["takeover"].items()},
                        "dev_box": {k: doc["dev_box_live"][k] for k in ("free_gib", "margin_after_peak_gib")}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
