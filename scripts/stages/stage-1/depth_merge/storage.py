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


def v2_sequence(root: Path = REPO) -> dict:
    """The revised variant 2's relay steps against what the relay holds now, under the registered two-copy rotation
    (`single_operator_budget.launch.rotation_need`: each new save point beside the previous until it hands off, the
    final beside the last): the depth_loop_kept session, then the depth_loop_merged one, each with its closeout
    archive; the first session's final stays on the relay while the second rotates. Per session: its own peak (from
    the bytes it starts with) and need (+ the margin), which its launch gate asks the live headroom for."""
    from stages.depth_merge import plan as PL
    from stages.single_operator_budget import launch as SL
    sz = sizes(root)
    st = DC.design()["storage"]
    archive = int(st["closeout_archive_bytes_bound"])
    margin = int(float(st["relay_margin_gib"]) * GIB)
    steps, net, sessions = [], 0, {}
    for arm in ("depth_loop_kept", "depth_loop_merged"):
        plan = json.loads(PL.out_path(arm).read_text())
        start = net
        rot = SL.rotation_need(plan, {}, sz)
        for s in rot["trace"]:
            steps.append({"step": f"{arm}: {s['step']}", "net_bytes": net + s["net_bytes"],
                          "net_gib": round((net + s["net_bytes"]) / GIB, 2)})
        net += rot["end_bytes"] + archive
        steps.append({"step": f"{arm} closeout: its evidence archive", "net_bytes": net, "net_gib": round(net / GIB, 2)})
        own_peak = max(rot["peak_bytes"], rot["end_bytes"] + archive)
        sessions[arm] = {"held_before_bytes": start, "own_peak_bytes": own_peak, "own_peak_gib": round(own_peak / GIB, 2),
                         "need_bytes": own_peak + margin, "need_gib": round((own_peak + margin) / GIB, 2),
                         "rule": "the session's launch gate: its own rotation peak + the margin, against the live "
                                 "headroom (which already excludes what earlier sessions left)"}
    peak = max(s["net_bytes"] for s in steps)
    return {"steps": steps, "peak_bytes": peak, "peak_gib": round(peak / GIB, 2), "margin_bytes": margin,
            "need_bytes": peak + margin, "need_gib": round((peak + margin) / GIB, 2),
            "peak_at": max(steps, key=lambda s: s["net_bytes"])["step"], "sessions": sessions,
            "rotation": "registered two-copy (no single-copy branch exists)"}


#: the relay's objects by group (`inventory`): why each stays, or what it would take to free it
INVENTORY_RULES = (
    ("durable/stage1_depth_merge/depth_merge/depth_merge-s20260801", "KEEP", "V1 seed 2's durable save point (step "
     "5,520): kept by the maintainer (2026-11-28); freeing it needs that decision"),
    ("durable/stage1_depth_merge/depth_merge/depth_merge-s20260726", "KEEP", "V1 seed 1's final: the M control's "
     "weights, their only copy (kept by the maintainer)"),
    ("durable/stage1_depth_merge/depth_merge/init", "KEEP", "V2-M's base (the only copy)"),
    ("durable/stage1_depth_merge/depth_merge/statistics", "KEEP", "V1's build statistics (the init's provenance)"),
    ("durable/stage1_single_operator_budget/depth_only/init", "KEEP", "V2-D's base: the pod stages it from here"),
    ("durable/stage1_single_operator_budget/residual_width/init", "PROPOSE", "a duplicate: the dev box holds a "
     "verified copy; no plan reads the relay copy"),
    ("durable/stage1_single_operator_budget/ffn_attention/init", "PROPOSE", "a duplicate: the dev box holds a "
     "verified copy; no plan reads the relay copy"),
    ("permanent_controls/", "KEEP", "permanent controls (protected)"),
    ("stage1/qwen3_0p6b_init_v0/", "KEEP", "the canonical 0.6B init and its random baseline (anchors)"),
    ("stage0_calvol/finalists/", "KEEP", "named by Stage-2 configs"),
    ("transfer/wheelhouse_", "KEEP", "the pods' environments (every session installs from them)"),
    ("transfer/stage3_sr/A-MIX-6cat-sr", "KEEP", "V2's training pack (relay input of every training session)"),
)


def inventory(root: Path = REPO) -> dict:
    """Every LFS object on the relay grouped by `INVENTORY_RULES` ($0, read-only): bytes per group and its rule; the
    rest is listed as unclassified with its bytes (not proposed: its consumers are not audited here)."""
    from huggingface_hub import HfApi
    from shared.deployment import MAIN_RELAY
    tok = Path(os.path.expanduser("~/.cache/huggingface/token")).read_text().strip()
    groups: dict[str, dict] = {}
    for f in HfApi(token=tok).list_lfs_files(MAIN_RELAY):
        name = f.filename or ""
        rule = next((r for r in INVENTORY_RULES if name.startswith(r[0])), None)
        key = rule[0] if rule else "/".join(name.split("/")[:3])
        g = groups.setdefault(key, {"bytes": 0, "objects": 0, "class": rule[1] if rule else "UNCLASSIFIED",
                                    "why": rule[2] if rule else "small or unrelated; consumers not audited here"})
        g["bytes"] += int(f.size)
        g["objects"] += 1
    for g in groups.values():
        g["gib"] = round(g["bytes"] / GIB, 3)
    total = sum(g["bytes"] for g in groups.values())
    by = {c: round(sum(g["bytes"] for g in groups.values() if g["class"] == c) / GIB, 2)
          for c in ("KEEP", "PROPOSE", "UNCLASSIFIED")}
    return {"groups": dict(sorted(groups.items(), key=lambda kv: -kv[1]["bytes"])), "total_gib": round(total / GIB, 2),
            "by_class_gib": by}


def v2_plan(live_state: dict, root: Path = REPO, dev_box: dict | None = None, inv: dict | None = None) -> dict:
    """The revised variant 2's storage under the registered rotation: each session's need against the live
    headroom, as it is and as it would be after each option, and the gap that remains (nothing is deleted)."""
    head = int(live_state["headroom_bytes"])
    seq = v2_sequence(root)
    dup = {}
    for prefix, rec_path in DUPLICATES.items():
        rec = json.loads((root / rec_path).read_text())["relay"]["init"]
        dup[prefix] = {"bytes": int(sum((rec.get("bytes") or {}).values())), "record": rec_path,
                       "relay_sha256": rec["files_sha256"]}
    freed_dup = sum(v["bytes"] for v in dup.values())
    v1_sp = int(sizes(root)["save_point"])
    options = {"as_is": 0, "duplicates_removed": freed_dup, "v1_seed2_save_point_freed": v1_sp,
               "duplicates_and_v1_seed2_save_point": freed_dup + v1_sp}
    rows = {}
    for label, freed in options.items():
        room = head + freed
        d_need = seq["sessions"]["depth_loop_kept"]["need_bytes"]
        #: V2-M starts with V2-D's final and archive held: the headroom then is lower by them
        held = seq["sessions"]["depth_loop_merged"]["held_before_bytes"]
        m_need = seq["sessions"]["depth_loop_merged"]["need_bytes"]
        rows[label] = {"freed_gib": round(freed / GIB, 2), "headroom_gib": round(room / GIB, 2),
                       "v2d": {"need_gib": round(d_need / GIB, 2), "fits": d_need <= room,
                               "gap_gib": round(max(0, d_need - room) / GIB, 2)},
                       "v2m": {"need_gib": round(m_need / GIB, 2), "headroom_then_gib": round((room - held) / GIB, 2),
                               "fits": m_need <= room - held, "gap_gib": round(max(0, m_need - (room - held)) / GIB, 2)},
                       "authorized": label == "as_is"}
    return {"schema": "aadistill.dmr1.v2_storage_plan/v2", "variant": "DMR1-R", "live": live_state,
            "dev_box_live": dev_box, "sequence": seq, "duplicates": dup, "duplicates_gib": round(freed_dup / GIB, 2),
            "options": rows, "inventory": inv,
            "rule": "the registered two-copy rotation: a valid, discoverable save point exists at every moment after "
                    "the first; need = a session's own peak + the 1 GiB margin, against the live headroom; nothing "
                    "is deleted or bought by this plan; every option but as_is needs the maintainer's decision"}


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
        doc = v2_plan(live(), dev_box=dev_box_live(), inv=inventory())
        if a.write:
            V2_RECORD.parent.mkdir(parents=True, exist_ok=True)
            V2_RECORD.write_text(json.dumps(doc, indent=1) + "\n")
        print(json.dumps({"options": doc["options"], "sessions": doc["sequence"]["sessions"],
                          "inventory_by_class": doc["inventory"]["by_class_gib"],
                          "groups": {k: (g["gib"], g["class"]) for k, g in doc["inventory"]["groups"].items()}},
                         indent=1))
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
