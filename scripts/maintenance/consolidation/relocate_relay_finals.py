"""Move SOB1's relay-placed final weights to the dev box, then retire their redundant relay copies. $0 GPU.

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python \\
        scripts/maintenance/consolidation/relocate_relay_finals.py            # dry run: live checks, the plan
        ... --fetch      # each final into the local store, every file re-hashed on the disk, read back
        ... --retire     # after --fetch: re-hashed again, tombstone FIRST, relay copy removed, confirmed absent

The maintainer's brief of 2026-11-27 (storage option a) authorizes exactly this: the two final weights of SOB1's
``ffn_attention`` arm (seeds 20260726 and 20260801), placed on the relay by SOB1's design, move to the dev box's
persistent store; once each is verified there and registered, its relay copy is retired. Nothing else is touched:
no other protected weight, no precision or content change, no historical record rewritten. The finals stay kept --
only where they are kept changes -- so none of them ever needs retraining to be recovered.

What each step re-derives, never trusts:

* the final's manifest is the session's sealed ``final.json`` (committed evidence) and equals the retention record's
  (``storage/local_retention_phase2.json``); the relay holds every file at its prefix and the weights' LFS sha256 is
  the manifest's;
* the disk: what is still to fetch plus the reserve (DMR1's largest session evidence bound and the shared disk's
  margin, both below) must fit the filesystem's free bytes NOW, checked before each final;
* on the disk: every file re-hashed against the manifest (`relay.get` moves its download, one copy at a time), the
  safetensors header read and one tensor loaded finite, the config parsed;
* before a removal: the local copy re-hashed again; the tombstone written; `relay.remove` (permanent, its pointer
  with it) and the relay read back absent.

Records: the retention record's entry becomes the dev-box entry (with ``relocated``: from where, by whose decision,
what was checked); the relay rotation record gets the arm's finals; the project's tombstones get one entry per relay
copy. SOB1's design keeps its historical placement (frozen); its readers (`launch.retain`) honour the recorded
relocation.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

GIB = 2 ** 30
RUN_ID = "sob1_train_ffn_attention_20261124_033254"
ARM = "ffn_attention"
UNITS = ("ffn_attention-s20260726", "ffn_attention-s20260801")
DECISION = ("maintainer brief 2026-11-27 (DMR1 repair round 3), item 5: 「本轮明确授权采用存储方案 a：将 SOB1 FFN + "
            "Attention 的两份最终权重迁移到开发机，逐文件核验并完成登记后，退役 relay 上对应的冗余副本。」")
#: what the disk keeps free after each fetch: the largest evidence a DMR1 session brings back
#: (`depth_merge.session.evidence_bound_gib`, a training session) and the shared disk's margin (another project writes
#: to the same filesystem; 2 GiB, the swing it showed in an hour before its 2026-11-24 clean)
SHARED_DISK_MARGIN = 2 * GIB


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def paths():
    from stages.single_operator_budget import common as C
    from stages.single_operator_budget import launch as L
    return {"retention": REPO / L.LOCAL_RECORD_PHASE2, "rotation": REPO / L.ROTATION_RECORD,
            "tombstones": REPO / L.TOMBSTONES, "store": C.local_store(),
            "evidence": REPO / L.STUDY.run_dir(RUN_ID) / "evidence"}


def reserve_bytes() -> int:
    from stages.depth_merge import launch as DL
    from stages.depth_merge import session as DS
    return int(max(DS.evidence_bound_gib(k, DL.VARIANT) for k in DL.VARIANT.kinds) * GIB) + SHARED_DISK_MARGIN


def manifest(unit: str) -> tuple[dict, dict]:
    """(the sealed final record, its files without the pointer), checked against the retention record."""
    from stages.teacher_parent_initialization import relay as R
    p = paths()
    fin = json.loads((p["evidence"] / unit / "final.json").read_text())
    files = {k: v for k, v in fin["files_sha256"].items() if k != R.POINTER}
    entry = json.loads(p["retention"].read_text())["products"][f"{unit}/final"]
    if entry.get("files_sha256") != files or entry.get("prefix") != fin["prefix"] or not fin.get("verified"):
        raise SystemExit(f"{unit}: the sealed final record and the retention record do not name the same verified "
                         f"files at the same prefix")
    return fin, files


def relay_check(fin: dict, files: dict) -> dict:
    from stages.single_operator_budget import launch as L
    want = {f"{fin['prefix']}/{k}": v for k, v in files.items()}
    got = L._lfs_sha(list(want))
    bad = sorted(q for q, v in want.items() if q not in got or (q.endswith(".safetensors") and got[q] != v))
    return {"paths": len(want), "present": sum(q in got for q in want), "weights_lfs_sha256_match": not bad,
            "problems": bad, "utc": utc()}


def readable(dest: Path, files: dict) -> dict:
    """The local copy reads back: every file's sha256 the manifest's, the safetensors header parsed (names, dtypes,
    shapes), one tensor loaded and finite, the config parsed."""
    import torch
    from safetensors import safe_open
    from shared import study_engineering as SE
    on_disk = {k: (SE.sha256_file(dest / k) if (dest / k).is_file() else None) for k in files}
    out = {"files_on_disk_sha256": on_disk, "sha256_match": on_disk == files}
    with safe_open(str(dest / "model.safetensors"), framework="pt", device="cpu") as f:
        keys = sorted(f.keys())
        shapes = {k: list(f.get_slice(k).get_shape()) for k in keys}
        dtypes = sorted({str(f.get_slice(k).get_dtype()) for k in keys})
        probe = min(keys, key=lambda k: int(torch.tensor(shapes[k]).prod()) if shapes[k] else 1)
        t = f.get_tensor(probe)
    cfg = json.loads((dest / "config.json").read_text())
    out.update(n_tensors=len(keys), n_elements=int(sum(int(torch.tensor(s).prod()) if s else 1 for s in shapes.values())),
               dtypes=dtypes, probe_tensor=probe, probe_finite=bool(torch.isfinite(t).all().item()),
               config={k: cfg.get(k) for k in ("model_type", "num_hidden_layers", "hidden_size", "intermediate_size",
                                                 "num_attention_heads", "torch_dtype")}, utc=utc())
    out["ok"] = bool(out["sha256_match"] and out["probe_finite"] and keys)
    return out


def plan() -> dict:
    p = paths()
    rows = []
    for unit in UNITS:
        fin, files = manifest(unit)
        dest = p["store"] / unit / "final" / fin["prefix"].rsplit("/", 1)[-1]
        need = sum(int(fin["bytes"][k]) for k in files if not (dest / k).is_file())
        rows.append({"unit": unit, "prefix": fin["prefix"], "dest": str(dest), "bytes": sum(int(fin["bytes"][k]) for k in files),
                     "to_fetch_bytes": need, "relay": relay_check(fin, files)})
    free = shutil.disk_usage(p["store"]).free
    total = sum(r["to_fetch_bytes"] for r in rows)
    return {"units": rows, "free_bytes": free, "free_gib": round(free / GIB, 2), "to_fetch_gib": round(total / GIB, 2),
            "reserve_gib": round(reserve_bytes() / GIB, 2), "free_after_gib": round((free - total) / GIB, 2),
            "fits": free - total >= reserve_bytes(), "utc": utc()}


def fetch() -> dict:
    from stages.teacher_parent_initialization import relay as R
    p = paths()
    doc = json.loads(p["retention"].read_text())
    res = {}
    for unit in UNITS:
        key = f"{unit}/final"
        fin, files = manifest(unit)
        entry = doc["products"][key]
        dest = p["store"] / unit / "final" / fin["prefix"].rsplit("/", 1)[-1]
        if entry.get("kept_on") == "dev_box" and (entry.get("readable") or {}).get("ok"):
            res[unit] = "already on the dev box, verified"
            continue
        before = relay_check(fin, files)
        if not before["weights_lfs_sha256_match"] or before["present"] != before["paths"]:
            raise SystemExit(f"{unit}: the relay copy is not as recorded: {before}")
        need = sum(int(fin["bytes"][k]) for k in files if not (dest / k).is_file())
        free = shutil.disk_usage(p["store"]).free
        if free - need < reserve_bytes():
            raise SystemExit(f"{unit}: {need / GIB:.2f} GiB to fetch, {free / GIB:.2f} GiB free, the reserve is "
                             f"{reserve_bytes() / GIB:.2f} GiB: no room")
        dest.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        got = R.get({**fin, "files_sha256": files}, dest)
        rd = readable(dest, files)
        if not rd["ok"]:
            raise SystemExit(f"{unit}: the local copy does not read back as the manifest: {rd}")
        doc["products"][key] = {
            "kept_on": "dev_box", "path": str(dest), "files_sha256": files, "files_on_disk_sha256": rd["files_on_disk_sha256"],
            "verified": True, "relay": fin["relay"], "prefix": fin["prefix"], "producer_run": entry.get("producer_run", RUN_ID),
            "fetched": {**got, "seconds": round(time.time() - t0, 1), "free_bytes_before": free, "fetched_bytes": need},
            "readable": {k: v for k, v in rd.items() if k != "files_on_disk_sha256"},
            "relocated": {"from": "relay (design.json :: storage.phase_2_finals.placement, historical)",
                          "decision": DECISION, "relay_check_before": before, "utc": utc(),
                          "relay_copy": "kept until --retire"},
            "checked": "every file of the manifest re-hashed on the dev box's disk; the safetensors header and one "
                       "tensor read back"}
        doc["checked_utc"] = utc()
        p["retention"].write_text(json.dumps(doc, indent=1) + "\n")
        res[unit] = {"dest": str(dest), "seconds": doc["products"][key]["fetched"]["seconds"],
                     "free_gib_after": round(shutil.disk_usage(p["store"]).free / GIB, 2)}
    return res


def retire() -> dict:
    from shared import study_engineering as SE
    from stages.single_operator_budget import launch as L
    from stages.teacher_parent_initialization import relay as R
    from stages.teacher_parent_initialization import train_driver as TD
    p = paths()
    doc = json.loads(p["retention"].read_text())
    rot = json.loads(p["rotation"].read_text())
    row = rot["arms"][ARM]
    out = {}
    for unit in UNITS:
        key = f"{unit}/final"
        fin, files = manifest(unit)
        entry = doc["products"][key]
        if (row["finals"].get(unit) or {}).get("verified_absent"):
            out[unit] = "retired already"
            continue
        if entry.get("kept_on") != "dev_box" or not (entry.get("readable") or {}).get("ok"):
            raise SystemExit(f"{unit}: not verified on the dev box (run --fetch first); its relay copy stays")
        here = {k: SE.sha256_file(Path(entry["path"]) / k) for k in files}
        if here != files:
            raise SystemExit(f"{unit}: the dev box's copy does not re-hash to the record now; its relay copy stays")
        size = sum(int(fin["bytes"][k]) for k in files)
        tid = f"relay_{RUN_ID}_{unit}_final"
        tomb = {
            "canonical_id": tid, "historical_paths": [f"{fin['prefix']}/{k}" for k in sorted(files)],
            "store": "relay", "relay_path": fin["prefix"], "retention_tier": "REDUNDANT_REMOTE_COPY",
            "experiment": "SOB1", "seed": int(unit.rsplit("-s", 1)[1]),
            "scientific_role": f"{unit}'s final weights (fp32): the relay COPY; the dev box keeps the copy",
            "weights_sha256": files.get("model.safetensors"), "config_sha256": files.get("config.json"),
            "architecture": None, "size_bytes": size, "size_human": f"{size / GIB:.3f} GiB", "n_files": len(files),
            "members": [{"path": f"{fin['prefix']}/{k}", "sha256": v} for k, v in sorted(files.items())],
            "deletion_criterion": "byte_identical_duplicate",
            "deletion_criterion_meaning": "byte-identical to a copy that survives",
            "same_hash_copies": [{"location": entry["path"], "kind": "local",
                                  "verified": "every file re-hashed to the record just before the removal"}],
            "canonical_surviving_evidence": entry["path"],
            "frozen_results_location": "logs/stages/stage-1/single_operator_budget/results/study_completion.json",
            "reason_physical_weights_deleted": ("relocation to the dev box (storage option a): the final is kept on the "
                                                "dev box, verified file by file and registered; the relay copy is "
                                                "redundant and its room is DMR1 variant 1's chain"),
            "maintainer_decision": DECISION, "tombstone_written_utc": utc(), "deleted_utc": None}
        t_doc = json.loads(p["tombstones"].read_text())
        prior = next((x for x in t_doc["tombstones"] if x["canonical_id"] == tid), None)
        if prior is not None and any(prior.get(k) != tomb[k] for k in ("relay_path", "weights_sha256", "members")):
            raise SystemExit(f"{unit}: tombstone {tid} exists for other bytes; nothing is removed")
        if prior is None:
            L.write_tombstone(tomb, p["tombstones"])          #: first: the tombstone, before any byte goes
        there = relay_check(fin, files)
        got = R.run_online("remove", {"record": TD.with_pointer(fin),
                                      "message": f"{unit}: final weights, a redundant copy (tombstone {tid})"},
                           timeout_s=3600)
        ok = bool(got.get("verified_absent"))
        after = relay_check(fin, files)
        row["finals"][unit] = {"tombstone": tid, "local": entry["path"], "verified_absent": ok and after["present"] == 0,
                               "freed_bytes": got.get("freed_bytes"), "error": got.get("error"), "utc": utc(),
                               "relay_copy_present_before": there["present"] == there["paths"],
                               "relay_paths_present_after": after["present"], "continued_pending_tombstone": prior is not None,
                               "decision": DECISION}
        t_doc = json.loads(p["tombstones"].read_text())
        e = next(x for x in t_doc["tombstones"] if x["canonical_id"] == tid)
        att = [*((e.get("deletion") or {}).get("attempts") or []),
               {"utc": utc(), "verified_absent": ok, "relay_paths_present_after": after["present"], "error": got.get("error")}]
        e.update(deleted_utc=(e.get("deleted_utc") or utc()) if ok else None,
                 deletion={"status": "DELETED" if ok else "PENDING", "attempts": att,
                           "relay_remove": {k: got.get(k) for k in ("removed_paths", "lfs_left", "paths_left",
                                                                     "shared_oids_kept", "freed_bytes", "error")}})
        p["tombstones"].write_text(json.dumps(t_doc, indent=1, ensure_ascii=False) + "\n")
        entry["relocated"]["relay_copy"] = ({"retired": True, "tombstone": tid, "freed_bytes": got.get("freed_bytes"),
                                             "utc": utc()} if ok else {"retired": False, "error": got.get("error")})
        doc["checked_utc"] = utc()
        p["retention"].write_text(json.dumps(doc, indent=1) + "\n")
        out[unit] = row["finals"][unit]
    done = all((row["finals"].get(u) or {}).get("verified_absent") for u in UNITS)
    row["status"] = ("ROTATED: both relay copies retired after the finals moved to the dev box (maintainer 2026-11-27, "
                     "storage option a)" if done else "PARTIAL: see finals")
    row["checked_utc"] = utc()
    p["rotation"].write_text(json.dumps(rot, indent=1, ensure_ascii=False) + "\n")
    return {"status": row["status"], "finals": out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--retire", action="store_true")
    a = ap.parse_args(argv)
    if a.fetch:
        print(json.dumps({"fetch": fetch()}, indent=1))
    elif a.retire:
        print(json.dumps({"retire": retire()}, indent=1))
    else:
        print(json.dumps(plan(), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
