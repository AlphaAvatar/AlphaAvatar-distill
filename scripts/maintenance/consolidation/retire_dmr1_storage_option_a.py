#!/usr/bin/env python3
"""DMR1 storage option (a), the maintainer's decision of 2026-11-28: retire three relay objects. $0 GPU.

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python \\
        scripts/maintenance/consolidation/retire_dmr1_storage_option_a.py            # dry run: every check, the plan
        ... --retire     # tombstone FIRST, then `relay.remove` (permanent), then the relay read back

What it retires, and only that:

1. **V1 seed 20260801's durable save point, step 5,520** (model + trainer state, about 19.50 GiB): the maintainer
   abandoned resuming V1's seed 2 and accepted that this progress can no longer be resumed. No copy survives; the
   unit's evidence (its training log, its records, the stop decision) stays.
2. **The relay copy of SOB1's residual_width initialization** (``init/model.safetensors`` and ``init/config.json``).
3. **The relay copy of SOB1's ffn_attention initialization** (the same two files).

For 2 and 3 the dev box keeps the copy. Every file is re-hashed against the init record just before the removal, and
nothing is removed when one differs. Each init's ``mapping/`` files have no dev-box copy, so they are not redundant
and stay on the relay.

What each removal does (`teacher_parent_initialization.relay.remove`): it deletes the record's paths and its
pointer, then permanently deletes the LFS objects -- except an object another remaining path still references
(``shared_oids_kept``), so no other artifact loses its bytes. It then reads the paths and the LFS listing back.

Records:
* one tombstone per object in `logs/maintenance/inventories/checkpoint_tombstones.json`, written before the bytes go,
  with the protection the decision lifts;
* the checkpoint registry's entries of the two init copies (``relay_path`` cleared, the retirement noted);
* the V1 save point's record in the dev box's secured evidence gains its ``removed`` answer, as the rotation's own
  removals do;
* `logs/maintenance/storage/dmr1_storage_option_a_20261128.json` with the live headroom before and after.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

GIB = 2 ** 30
DECISION = ("maintainer brief 2026-11-28 (after the review of 277586ba): 「我确定放弃恢复 V1 的 seed 2，批准采用存储方案 (a)："
            "退役它的恢复存档，并删除 SOB1 两份已有本地核验副本的初始化在 HF 上的冗余副本。」")
V1_RUN = "dmr1_merge_train_20261127_201532"
V1_UNIT = "depth_merge-s20260801"
V1_TAG = "step_005520"
V1_EVIDENCE = Path("/home/ecs-user/aad-artifacts/stage1_depth_merge") / V1_RUN / "store/out" / V1_UNIT / "savepoints.json"
SOB1_INITS = {
    "residual_width": "logs/stages/stage-1/single_operator_budget/runs/sob1_phase1_20261119_194259/evidence/residual_width/init_record.json",
    "ffn_attention": "logs/stages/stage-1/single_operator_budget/runs/sob1_phase1_20261119_194259/evidence/ffn_attention/init_record.json",
}
LOCAL_SOB1 = Path("/home/ecs-user/aad-artifacts/stage1_single_operator_budget")
TOMBSTONES = REPO / "logs/maintenance/inventories/checkpoint_tombstones.json"
REGISTRY = REPO / "logs/maintenance/inventories/checkpoint_registry.json"
RECORD = REPO / "logs/maintenance/storage/dmr1_storage_option_a_20261128.json"


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def consumers(prefix: str) -> list[str]:
    """Code and configs that name the prefix (the consumer audit; records and prose are not consumers)."""
    got = subprocess.run(["git", "grep", "-l", prefix, "--", "src", "scripts", "configs", "tests"], cwd=REPO,
                         capture_output=True, text=True).stdout.split()
    return sorted(p for p in got if not p.endswith("retire_dmr1_storage_option_a.py"))


def relay_paths(prefix: str) -> dict[str, dict]:
    from huggingface_hub import HfApi
    from shared.deployment import MAIN_RELAY
    tok = Path("~/.cache/huggingface/token").expanduser().read_text().strip()
    from huggingface_hub.errors import RemoteEntryNotFoundError
    out = {}
    try:
        for i in HfApi(token=tok).list_repo_tree(MAIN_RELAY, path_in_repo=prefix, recursive=True, repo_type="model"):
            if hasattr(i, "size"):
                lfs = getattr(i, "lfs", None)
                out[i.path] = {"size": int(i.size), "lfs_sha256": lfs.sha256 if lfs else None}
    except RemoteEntryNotFoundError:
        return {}                                   #: the prefix is gone: nothing under it (a read, not a failure)
    return out


def lfs_present(oids) -> list[str]:
    """Which of these LFS objects the relay still bills (its LFS listing)."""
    from huggingface_hub import HfApi
    from shared.deployment import MAIN_RELAY
    tok = Path("~/.cache/huggingface/token").expanduser().read_text().strip()
    want = set(oids)
    return sorted(o.file_oid for o in HfApi(token=tok).list_lfs_files(MAIN_RELAY) if o.file_oid in want)


def targets() -> list[dict]:
    """The three objects, each with its relay record (prefix + files), what survives and the checks it needs."""
    from shared import study_engineering as SE
    from stages.teacher_parent_initialization import train_driver as TD
    out = []
    recs = json.loads(V1_EVIDENCE.read_text())
    sp = next(r for r in recs if r["tag"] == V1_TAG)
    if not sp.get("verified"):
        raise SystemExit(f"{V1_UNIT} {V1_TAG} is not a verified save point")
    out.append({"id": f"relay_{V1_RUN}_{V1_UNIT}_{V1_TAG}", "kind": "abandoned_resume_state",
                "record": TD.with_pointer(sp), "files_sha256": sp["files_sha256"], "bytes": sp["bytes"],
                "survivor": None, "experiment": "DMR1-M (V1)", "seed": 20260801,
                "role": f"{V1_UNIT}'s durable save point at step 5,520 (model fp32 + AdamW / RNG / data position): the "
                        "only copy; the unit stopped at step 6,590 by the maintainer's decision",
                "criterion": "maintainer_decision",
                "reason": ("storage option (a), the maintainer's decision of 2026-11-28: V1 seed 2's resume is "
                           "abandoned, so its resumable state has no consumer; its room is DMR1 variant 2's two-copy "
                           "rotation"),
                "meaning": "the maintainer abandoned resuming this unit and retired its resumable state",
                "frozen": f"logs/stages/stage-1/depth_merge/runs/{V1_RUN}/closeout/session_completion.json",
                "evidence_kept": f"the unit's training log, records and the stop decision ({V1_EVIDENCE.parent})"})
    for arm, rel in SOB1_INITS.items():
        rec = json.loads((REPO / rel).read_text())["relay"]["init"]
        digest16 = rec["prefix"].rsplit("/", 1)[1]
        local = LOCAL_SOB1 / arm / "init" / digest16
        out.append({"id": f"relay_sob1_phase1_{arm}_init_{digest16}", "kind": "redundant_remote_copy",
                    "record": {**rec}, "files_sha256": rec["files_sha256"], "bytes": rec["bytes"],
                    "survivor": str(local), "experiment": "SOB1", "seed": None,
                    "role": f"SOB1 {arm}'s training-pre initialization: the relay COPY of init/; the dev box keeps "
                            "the copy (mapping/ stays on the relay: it has no dev-box copy)",
                    "criterion": "byte_identical_duplicate", "meaning": "byte-identical to a copy that survives",
                    "reason": ("storage option (a): a redundant remote copy; the dev box keeps the verified copy, and "
                               "the room is DMR1 variant 2's"),
                    "frozen": "logs/stages/stage-1/single_operator_budget/results/study_completion.json",
                    "evidence_kept": rel})
    for t in out:
        t["size_bytes"] = int(sum(int(v) for v in t["bytes"].values()))
        t["consumers"] = consumers(t["record"]["prefix"])
    return out


def check(t: dict) -> dict:
    """Live: the relay holds every file of the record with the record's LFS sha256; a survivor re-hashes to it."""
    from shared import study_engineering as SE
    listed = relay_paths(t["record"]["prefix"])
    files = {f"{t['record']['prefix']}/{k}": v for k, v in t["files_sha256"].items()}
    missing = [p for p in files if p not in listed]
    differ = [p for p, v in files.items() if p in listed and listed[p]["lfs_sha256"] not in (None, v)]
    out = {"relay_files": len(files), "relay_missing": missing, "relay_lfs_differs": differ,
           "relay_also_under_prefix": sorted(set(listed) - set(files) - {f"{t['record']['prefix']}/record.json"})}
    if t["survivor"]:
        here = {k: SE.sha256_file(Path(t["survivor"]) / k) for k in t["files_sha256"]}
        out["survivor_rehash"] = {k: here[k] == v for k, v in t["files_sha256"].items()}
    out["ok"] = (not differ and (not t["survivor"] or all(out["survivor_rehash"].values())))
    return out


def tombstone(t: dict, chk: dict) -> dict:
    files = t["files_sha256"]
    return {"canonical_id": t["id"], "historical_paths": [f"{t['record']['prefix']}/{k}" for k in sorted(files)],
            "store": "relay", "relay_path": t["record"]["prefix"],
            "retention_tier": "REDUNDANT_REMOTE_COPY" if t["survivor"] else "RETIRED_BY_MAINTAINER_DECISION",
            "experiment": t["experiment"], "seed": t["seed"], "scientific_role": t["role"],
            "weights_sha256": files.get("model/model.safetensors") or files.get("init/model.safetensors"),
            "config_sha256": files.get("model/config.json") or files.get("init/config.json"), "architecture": None,
            "size_bytes": t["size_bytes"], "size_human": f"{t['size_bytes'] / GIB:.3f} GiB", "n_files": len(files),
            "members": [{"path": f"{t['record']['prefix']}/{k}", "sha256": v} for k, v in sorted(files.items())],
            "references_at_deletion": t["consumers"],
            "deletion_criterion": t["criterion"], "deletion_criterion_meaning": t["meaning"],
            "reason_physical_weights_deleted": t["reason"],
            **({"same_hash_copies": [{"location": t["survivor"], "kind": "local",
                                      "verified": "every file re-hashed to the record just before the removal"}],
                "canonical_surviving_evidence": t["survivor"]} if t["survivor"] else
               {"same_hash_copies": [], "canonical_surviving_evidence": t["evidence_kept"],
                "resumable_after_deletion": False}),
            "frozen_results_location": t["frozen"], "live_check_before_removal": chk,
            "protection_lifted": {"class": "registry protected / never_delete_clause" if t["survivor"]
                                  else "the maintainer's 2026-11-28 keep instruction", "maintainer_decision": DECISION},
            "maintainer_decision": DECISION, "tombstone_written_utc": utc(), "deleted_utc": None}


def _records(t: dict, got: dict, ok: bool) -> None:
    """The records beside the tombstone (idempotent): the V1 save point's ``removed`` answer in the dev box's secured
    evidence, or the registry entry of an init copy (its relay path cleared, the retirement noted)."""
    if not ok:
        return
    if t["kind"] == "abandoned_resume_state":
        recs = json.loads(V1_EVIDENCE.read_text())
        sp = next(r for r in recs if r["tag"] == V1_TAG)
        if not (sp.get("removed") or {}).get("verified_absent"):
            sp["removed"] = {**got, "verified_absent": True, "tombstone": t["id"], "decision": DECISION}
            sp.setdefault("remove_attempts", []).append({"verified_absent": True, "utc": utc(), "error": got.get("error")})
            V1_EVIDENCE.write_text(json.dumps(recs, indent=1) + "\n")
        return
    reg = json.loads(REGISTRY.read_text())
    arts = reg["scientific_artifacts"]
    for a in (arts.values() if isinstance(arts, dict) else arts):
        if isinstance(a, dict) and a.get("relay_path") == f"{t['record']['prefix']}/init/model.safetensors":
            a["relay_copy_retired"] = {"tombstone": t["id"], "utc": utc(), "decision": DECISION, "kept": t["survivor"]}
            a["relay_path"] = None
    REGISTRY.write_text(json.dumps(reg, indent=1, ensure_ascii=False) + "\n")


def headroom() -> dict:
    from stages.depth_merge import storage as ST
    return ST.live()


def retire(ts: list[dict]) -> dict:
    from stages.single_operator_budget import launch as L
    from stages.teacher_parent_initialization import relay as R
    out = {"before": headroom(), "objects": {}}
    for t in ts:
        chk = check(t)
        if not chk["ok"]:
            raise SystemExit(f"{t['id']}: the live check fails, nothing removed: {chk}")
        tomb = tombstone(t, chk)
        doc = json.loads(TOMBSTONES.read_text())
        prior = next((x for x in doc["tombstones"] if x["canonical_id"] == t["id"]), None)
        done = (prior is not None and (prior.get("deletion") or {}).get("status") == "DELETED"
                and not [p for p in relay_paths(t["record"]["prefix"])
                         if p.removeprefix(t["record"]["prefix"] + "/") in t["files_sha256"]]
                and not lfs_present(t["files_sha256"].values()))
        if done:                                    #: removed and recorded already: its answer is kept as captured
            out["objects"][t["id"]] = {"verified_absent": True,
                                       "freed_bytes": (prior["deletion"].get("relay_remove") or {}).get("freed_bytes"),
                                       "recorded": "earlier in this retirement", "left_under_prefix":
                                       prior["deletion"].get("relay_left_under_prefix")}
            _records(t, prior["deletion"].get("relay_remove") or {}, True)
            continue
        if prior is not None and prior.get("members") != tomb["members"]:
            raise SystemExit(f"{t['id']}: a tombstone exists for other bytes; nothing removed")
        if prior is None:
            L.write_tombstone(tomb, TOMBSTONES)                        #: first: the tombstone, before any byte goes
        got = R.run_online("remove", {"record": t["record"], "message": f"{t['id']} (tombstone; {DECISION[:40]}...)"},
                           timeout_s=3600)
        after = relay_paths(t["record"]["prefix"])
        left = [p for p in after if p.removeprefix(t["record"]["prefix"] + "/") in t["files_sha256"]]
        lfs_left = lfs_present(t["files_sha256"].values())
        ok = bool(got.get("verified_absent")) and not left and not lfs_left
        doc = json.loads(TOMBSTONES.read_text())
        e = next(x for x in doc["tombstones"] if x["canonical_id"] == t["id"])
        e.update(deleted_utc=utc() if ok else None,
                 deletion={"status": "DELETED" if ok else "PENDING", "utc": utc(),
                           "relay_remove": {k: got.get(k) for k in ("removed_paths", "lfs_objects_deleted", "lfs_left",
                                                                     "paths_left", "shared_oids_kept", "freed_bytes",
                                                                     "error")},
                           "relay_left_under_prefix": sorted(after), "lfs_objects_still_billed": lfs_left,
                           **({"continued": "a re-run after the first run's read-back failed (a 404 on the emptied "
                                            "prefix, read as an error): the first removal's answer -- its freed bytes "
                                            "-- was not captured; the absence of every path and LFS object is read "
                                            "here"} if prior is not None else {})})
        TOMBSTONES.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
        out["objects"][t["id"]] = {"verified_absent": ok, "freed_bytes": got.get("freed_bytes"),
                                   "shared_oids_kept": got.get("shared_oids_kept"), "left_under_prefix": sorted(after)}
        _records(t, got, ok)
    out["after"] = headroom()
    out["freed_bytes_reported"] = sum(int(v.get("freed_bytes") or 0) for v in out["objects"].values())
    out["headroom_gained_gib"] = round((out["after"]["headroom_bytes"] - out["before"]["headroom_bytes"]) / GIB, 3)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--retire", action="store_true")
    a = ap.parse_args(argv)
    ts = targets()
    plan = {t["id"]: {"size_gib": round(t["size_bytes"] / GIB, 3), "survivor": t["survivor"],
                      "consumers": t["consumers"], "check": check(t)} for t in ts}
    print(json.dumps(plan, indent=1))
    if not a.retire:
        print("DRY RUN: nothing written, nothing removed. Pass --retire.")
        return 0
    res = retire(ts)
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps({"schema": "aadistill.maintenance.relay_retirement/v1", "decision": DECISION,
                                  "plan": plan, **res, "utc": utc()}, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: res[k] for k in ("objects", "freed_bytes_reported", "headroom_gained_gib")}, indent=1))
    return 0 if all(v["verified_absent"] for v in res["objects"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
