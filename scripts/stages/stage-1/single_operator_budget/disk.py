#!/usr/bin/env python3
"""Where a SOB1 session's bytes land on the dev box, and the peak when they coexist ($0; measured inputs).

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python scripts/stages/stage-1/single_operator_budget/disk.py \\
        [--kind sob1_phase1|sob1_train] [--arm <arm>] [--scr <the session's scr>] [--write | --check]

Phase 1 (`design.json :: cost_model.dev_box_disk`, every figure measured on TPI1's records): the evidence the poll
pulls (``<scr>/store/out``, generations and sealed parts both), one pull's archive in flight (``<scr>/store/pull_N.tgz``,
deleted once unpacked), the teardown collector's archive and its extracted copy (records and logs only), a resume
archive when the session must be resumed (``<scr>/resume_<run>.tgz``), then each arm's initialization and mapping
retained from the relay (``<local store>/<arm>/{init,mapping}/<digest16>/``; `relay.get` moves its cache blob, so a file
in flight is never held twice). The peak is at the end of retention: everything but the pull archive at once.

A training session (phase 2, `design.json :: cost_model.dev_box_disk.phase_2`) brings back, at the same kinds of
paths: the evidence of its two units (generations, sealed parts and training records; bound: every turn of both modes
at the full 8,192-token context; expected: TPI1's measured evidence per recovered unit), one pull's archive in flight,
the teardown collector's archive and its extracted copy, a resume archive on the failure path, then both seeds' fp32
finals retained from the relay (``<local store>/<arm>-s<seed>/final/<sha16>/``). `phase2_total` adds the three
sessions on the one filesystem: every arm's finals and evidence stay, so the last session's peak holds all six finals
(``--kind sob1_train --total``; ``plans/disk_peak_phase2.json``), beside phase 1's independent generations archive.

`check_room` is the launch's precheck: the free bytes of the filesystem each landing path is on, against the bound
peak of the paths it holds -- the filesystem the bytes actually land on, not a generic floor.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.single_operator_budget import common as C  # noqa: E402

GIB = 2 ** 30
OUT = C.LOGS / "plans" / "disk_peak.json"
OUT_PHASE2 = C.LOGS / "plans" / "disk_peak_phase2.json"
SCR_PLACEHOLDER = "<scr>"


def evidence_bytes() -> dict:
    """Phase 1's evidence, raw, expected and bound (module docstring)."""
    m = C.design()["cost_model"]["dev_box_disk"]
    tok = m["tokens"]
    n_models = len(C.design()["phase_1"]["evaluation_order"])
    per_model_bound = max([tok["teacher"], *tok["untrained_proxies"]]) * m["bound_margin"]
    tokens = {"expected": tok["teacher"] + sum(tok["untrained_proxies"]), "bound": per_model_bound * n_models}
    return {"tokens": tokens, "per_model_bound_tokens": per_model_bound,
            "raw": {k: v * m["bytes_per_generated_token"] for k, v in tokens.items()},
            "per_model_raw_bound": per_model_bound * m["bytes_per_generated_token"]}


def phase2_evidence_bytes() -> dict:
    """One recovered model's evidence on the dev box (both modes), bound and expected (module docstring)."""
    d = C.design()
    m = d["cost_model"]["dev_box_disk"]
    p2 = m["phase_2"]
    turns = sum(p2["turns_per_mode_bound"].values())
    bound = turns * p2["context_tokens"] * m["bytes_per_generated_token"] + p2["training_records_bytes_per_unit_bound"]
    return {"per_model_bound": bound, "per_model_expected": p2["expected_bytes_per_recovered_model"],
            "bound_tokens_per_model": turns * p2["context_tokens"], "turns_per_mode_bound": p2["turns_per_mode_bound"]}


def phase2_total(scr_root: str | None = None, *, free_bytes: int | None = None, archive_bytes: int = 0) -> dict:
    """The three training sessions on the one filesystem, in their order: each session's peak beside every earlier
    session's kept evidence and finals; the last one holds all six finals. ``free_bytes``: the filesystem's free bytes
    now (read when not given); ``archive_bytes``: phase 1's independent generations archive, kept beside them."""
    d = C.design()
    order = d["training"]["unit_order"]["arms"]
    swing = d["cost_model"]["dev_box_disk"]["phase_2"]["shared_filesystem_swing_bytes"]
    kept = {"bound": archive_bytes, "expected": archive_bytes}
    sessions = []
    for a in order:
        pl = plan("sob1_train", a, f"{scr_root or SCR_PLACEHOLDER}/<{a} run>")
        peak = {k: kept[k] + pl["peak_bytes"][k] for k in kept}
        sessions.append({"arm": a, "session_peak_bytes": pl["peak_bytes"], "kept_before_bytes": dict(kept),
                         "peak_with_earlier_sessions_bytes": peak,
                         "peak_with_earlier_sessions_gib": {k: round(v / GIB, 2) for k, v in peak.items()}})
        for k in kept:                          #: what stays: the evidence, the collector archives and the finals
            kept[k] += sum(i[f"bytes_{k}"] for i in pl["items"]
                           if "kept" in i["when"] or i["when"].startswith("retain"))
    peak = {k: max(s["peak_with_earlier_sessions_bytes"][k] for s in sessions) for k in kept}
    free = shutil.disk_usage(Path(os.path.expanduser(str(C.local_store()))).resolve().anchor).free \
        if free_bytes is None else int(free_bytes)
    need = {k: max(0, peak[k] + swing - free) for k in peak}
    return {"schema": "aadistill.stage1.single_operator_budget.disk_peak_phase2/v1", "order": order,
            "filesystem": "/ (one filesystem holds the repository, the local store and every landing path)",
            "sessions": sessions, "phase1_archive_bytes": archive_bytes, "peak_bytes": peak,
            "peak_gib": {k: round(v / GIB, 2) for k, v in peak.items()}, "free_bytes": free,
            "free_gib": round(free / GIB, 2), "shared_filesystem_swing_bytes": swing,
            "shortfall_with_swing_bytes": need, "shortfall_with_swing_gib": {k: round(v / GIB, 2) for k, v in need.items()},
            "rule": "the bound peak plus the measured swing of the other project's files, against the free bytes now",
            "inputs": "design.json :: cost_model.dev_box_disk (phase_2), structures.json :: checkpoint_bytes",
            "authorizes": "nothing"}


def plan(kind: str = "sob1_phase1", arm: str | None = None, scr: str | None = None) -> dict:
    d = C.design()
    m = d["cost_model"]["dev_box_disk"]
    st = C.read_json(C.STRUCTURES)["arms"]
    scr = scr or SCR_PLACEHOLDER
    store = str(C.local_store())
    items = []

    def add(name, path, when, bound, expected=None, consumer="", phases=("session", "retain")):
        items.append({"item": name, "path": path, "when": when, "bytes_bound": int(bound),
                      "bytes_expected": int(bound if expected is None else expected), "consumer": consumer,
                      "present_during": list(phases)})
    if kind == "sob1_phase1":
        ev = evidence_bytes()
        add("evidence", f"{scr}/store/out", "pulled on every poll, kept", ev["raw"]["bound"], ev["raw"]["expected"],
            "the completion check, the phase-1 comparison table, the analysis")
        add("pull archive in flight", f"{scr}/store/pull_<n>.tgz", "one poll, deleted once unpacked",
            ev["per_model_raw_bound"] / m["archive_ratio"], ev["per_model_raw_bound"] / m["archive_ratio"] / 4,
            "the poll's unpack", phases=("session",))
        coll = ev["raw"]["bound"] * m["records_share"]
        add("teardown collector archive + its extracted copy", f"{scr}/store/<archive>, {scr}/store/extracted",
            "teardown, kept", 2 * coll, 2 * ev["raw"]["expected"] * m["records_share"],
            "the runner's local hash verification")
        add("resume archive", f"{scr}/resume_<run>.tgz", "only when a session must be resumed",
            ev["raw"]["bound"] / m["archive_ratio"], 0, "stage-resume's upload", phases=("retain",))
        for a in d["phase_1"]["build_order"]:
            add(f"{a} initialization", f"{store}/{a}/init/<digest16>/", "retain, after closeout",
                st[a]["checkpoint_bytes"]["init_bf16"], consumer="the durable local copy (phase 2 reads the relay's)",
                phases=("retain",))
            add(f"{a} mapping", f"{store}/{a}/mapping/<digest16>/", "retain, after closeout",
                m["mapping_bytes_per_arm"], consumer="the durable local copy of the selection", phases=("retain",))
    else:
        if arm not in st:
            raise SystemExit(f"--arm {arm!r} is not an arm")
        ev = phase2_evidence_bytes()
        n = len(d["training"]["seeds"])
        add("evidence (two recovered units: generations, sealed parts, training records)", f"{scr}/store/out",
            "pulled on every poll, kept", n * ev["per_model_bound"], n * ev["per_model_expected"],
            "the completion check, the study's analysis")
        add("pull archive in flight", f"{scr}/store/pull_<n>.tgz", "one poll, deleted once unpacked",
            ev["per_model_bound"] / m["archive_ratio"], ev["per_model_expected"] / m["archive_ratio"] / 4,
            "the poll's unpack", phases=("session",))
        add("teardown collector archive + its extracted copy", f"{scr}/store/<archive>, {scr}/store/extracted",
            "teardown, kept", 2 * n * ev["per_model_bound"] * m["records_share"],
            2 * n * ev["per_model_expected"] * m["records_share"], "the runner's local hash verification")
        add("resume archive", f"{scr}/resume_<run>.tgz", "only when a session must be resumed",
            n * ev["per_model_bound"] / m["archive_ratio"], 0, "stage-resume's upload", phases=("retain",))
        for s in d["training"]["seeds"]:
            add(f"{arm}-s{s} final weights (fp32)", f"{store}/{arm}-s{s}/final/<sha16>/", "retain, after closeout",
                st[arm]["checkpoint_bytes"]["final_fp32"], consumer="the durable local copy", phases=("retain",))
    by_phase = {ph: {k: sum(i[f"bytes_{k}"] for i in items if ph in i["present_during"]) for k in ("bound", "expected")}
                for ph in ("session", "retain")}
    peak = {k: max(v[k] for v in by_phase.values()) for k in ("bound", "expected")}
    peak_phase = max(by_phase, key=lambda ph: by_phase[ph]["bound"])
    return {"schema": "aadistill.stage1.single_operator_budget.disk_peak/v1", "kind": kind, "arm": arm,
            "scr": scr, "local_store": store, "items": items, "by_phase": by_phase, "peak_phase": peak_phase,
            "peak_bytes": peak, "peak_gib": {k: round(v / GIB, 2) for k, v in peak.items()},
            "evidence": evidence_bytes() if kind == "sob1_phase1" else None,
            "inputs": "design.json :: cost_model.dev_box_disk", "authorizes": "nothing"}


def _filesystem_of(path: str) -> tuple[int, int, str]:
    """(device id, free bytes, the existing ancestor read) of the filesystem ``path`` lands on."""
    p = Path(os.path.expanduser(path)).resolve()
    while not p.exists():
        p = p.parent
    return os.stat(p).st_dev, shutil.disk_usage(p).free, str(p)


def check_room(kind: str, arm: str | None, scr: str) -> dict:
    """The bound peak against the free bytes of the filesystem(s) the landing paths are on."""
    pl = plan(kind, arm, scr)
    need: dict[int, dict] = {}
    for i in pl["items"]:
        if pl["peak_phase"] not in i["present_during"]:
            continue
        root = i["path"].split(",")[0].split("<")[0]
        dev, free, at = _filesystem_of(root)
        e = need.setdefault(dev, {"free_bytes": free, "read_at": at, "bound_bytes": 0, "items": []})
        e["bound_bytes"] += i["bytes_bound"]
        e["items"].append(i["item"])
    fs = list(need.values())
    for e in fs:
        e["fits"] = e["free_bytes"] >= e["bound_bytes"]
        e["margin_gib"] = round((e["free_bytes"] - e["bound_bytes"]) / GIB, 2)
    return {"kind": kind, "arm": arm, "scr": scr, "filesystems": fs, "fits": all(e["fits"] for e in fs),
            "peak_gib": pl["peak_gib"]}


def render(doc: dict) -> str:
    return json.dumps(doc, indent=1) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kind", default="sob1_phase1")
    ap.add_argument("--arm", default=None)
    ap.add_argument("--scr", default=None)
    ap.add_argument("--total", action="store_true", help="phase 2's three sessions together (plans/disk_peak_phase2.json)")
    ap.add_argument("--archive-bytes", type=int, default=0, help="phase 1's independent generations archive")
    ap.add_argument("--free-bytes", type=int, default=None, help="the free bytes the total is read against")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if a.total:
        doc = phase2_total(free_bytes=a.free_bytes, archive_bytes=a.archive_bytes)
        if a.write:
            OUT_PHASE2.write_text(render(doc))
            print(f"wrote {OUT_PHASE2.relative_to(REPO)}: peak {doc['peak_gib']} GiB, free {doc['free_gib']} GiB")
            return 0
        print(render(doc))
        return 0
    doc = plan(a.kind, a.arm)
    if a.write:
        OUT.write_text(render(doc))
        print(f"wrote {OUT.relative_to(REPO)}: peak {doc['peak_gib']} GiB")
        return 0
    if a.check:
        same = OUT.read_text() == render(doc)
        print("the disk plan is the derivation's" if same else "REFUSED: the committed disk plan is not the derivation's")
        return 0 if same else 1
    room = check_room(a.kind, a.arm, a.scr or str(C.local_store() / "<run_id>"))
    print(json.dumps({"peak_gib": doc["peak_gib"], "room": room}, indent=1))
    return 0 if room["fits"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
