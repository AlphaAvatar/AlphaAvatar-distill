#!/usr/bin/env python3
"""Close out ONE D3 structural session, off-pod, at `$0`.

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python \
        scripts/stages/stage-1/phase_d3/closeout_d3_structural.py \
        --run d3_structural_<utc> --scr <scr> [upload] [--write]

What a finished structural session owes the repository:

1. the SMALL evidence copied into the run directory: the driver's session
   record, the product record, every step's persisted result (DEPTH's round
   tables and both references' KL), the stage journal, the launcher log and
   the watchdog journals;
2. the OUTCOME RE-DERIVED here from those step records -- `depth_evidence` and
   `classify_structure` run again against the committed control record -- and
   required equal to what the pod recorded; a disagreement refuses the closeout;
3. for a new initialization, the PRODUCT re-identified from the bytes that came
   home, at all four identities, and (`upload`) made durable on the relay:
   uploaded once, its LFS oids checked, downloaded into an independent
   directory and re-identified again -- the bytes a probes pod would stage;
4. the MONEY (D2's closeout formula) and `provider_teardown_verified` only when
   the provider confirms the pod gone.

Without `--write` it prints what it would record and writes nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
for _extra in ("src", "scripts", "scripts/stages/stage-1"):
    _p = str(REPO_ROOT / _extra)
    if _p not in sys.path:
        sys.path.insert(0, _p)

RUNS_REL = "logs/stages/stage-1/phase_d3/runs"
AUDIT = "autoinit_d3_structural"
SCHEMA = "aadistill.phase_d3.structural_closeout/v1"
DURABILITY = "evidence/product_durability.json"
PRODUCT_FILES = ("model.safetensors", "config.json", "generation_config.json")


class CloseoutRefused(RuntimeError):
    pass


def _sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_first(scr: Path, name: str) -> Any:
    for path in (scr / "store" / name, scr / "relay" / name):
        if path.is_file():
            return (json.loads(path.read_text()) if name.endswith(".json")
                    else path.read_text())
    return None


def evidence_plan(run_dir: Path, scr: Path) -> list[tuple[Path, Path]]:
    store, ev = scr / "store", run_dir / "evidence"
    plan = [(store / n, ev / n) for n in (
        "d3_structural.json", "product.json", "steps.jsonl", "journal.jsonl")]
    plan += [(scr / "launch.log", run_dir / "runtime" / "launcher.log")]
    plan += [(p, run_dir / "runtime" / p.name)
             for p in sorted(scr.glob("watchdog_*.jsonl"))]
    plan += [(p, ev / "provider" / p.name)
             for p in sorted(scr.glob("create_raw_*.txt"))
             if '"id"' in p.read_text(errors="replace")]
    return plan


def rederive(record: dict[str, Any]) -> dict[str, Any]:
    """The outcome from the persisted step records, against the control."""
    from stages.phase_d3 import d3_design as D

    control = json.loads((REPO_ROOT / D.CONTROL_RECORD).read_text())
    steps = record.get("steps") or []
    leaf = D.control_leaf(REPO_ROOT)
    at = D.treated_index(leaf)
    if len(steps) != len(leaf.steps):
        raise CloseoutRefused(f"{len(steps)} step records, the path has "
                              f"{len(leaf.steps)}")
    depth = D.depth_evidence(steps[at], control)
    outcome = D.classify_structure(control=control,
                                   depth_kept=depth["kept_layers"],
                                   step_identities=steps)
    pod = (record.get("structural") or {}).get("outcome")
    if pod != outcome["outcome"]:
        raise CloseoutRefused(f"the pod recorded {pod!r} and the off-pod "
                              f"re-derivation gives {outcome['outcome']!r}")
    return {"depth": depth, "structural": outcome}


def product_bytes(scr: Path, product: dict[str, Any]) -> dict[str, Any]:
    from stages.phase_d1.behavioural_materialize import _verify_bytes_at

    digest = product["identities"]["artifact_digest"]
    directory = scr / "products" / digest
    want = {k: product["identities"][k] for k in (
        "artifact_digest", "weights_digest", "single_shard_sha256",
        "arch_signature")}
    got = _verify_bytes_at(directory, want, what="the D3 product")
    files = {n: {"sha256": _sha(directory / n),
                 "bytes": (directory / n).stat().st_size}
             for n in PRODUCT_FILES}
    bad = [n for n in PRODUCT_FILES
           if files[n]["sha256"] != product["files"][n]["sha256"]]
    if bad:
        raise CloseoutRefused(f"the arrived files differ from the pod's "
                              f"record: {bad}")
    return {"path": str(directory), "identities_checked": got["verified"],
            "files": files}


# --- relay durability ------------------------------------------------------------

def relay_prefix(run: str, digest: str) -> str:
    return f"stage1/phase_d3/{run}/products/{digest}"


def _token() -> str:
    return Path("~/.cache/huggingface/token").expanduser().read_text().strip()


def _relay() -> str:
    from shared.deployment import MAIN_RELAY

    return MAIN_RELAY


def remote_listing(prefix: str) -> dict[str, dict[str, Any]]:
    from huggingface_hub import HfApi

    api = HfApi(token=_token())
    try:
        items = list(api.list_repo_tree(_relay(), path_in_repo=prefix,
                                        repo_type="model", recursive=True,
                                        expand=True))
    except Exception as exc:                                 # noqa: BLE001
        if "404" in str(exc) or "not found" in str(exc).lower():
            return {}
        raise
    out = {}
    for item in items:
        if getattr(item, "size", None) is None:
            continue
        lfs = getattr(item, "lfs", None)
        out[item.path] = {"size_bytes": item.size,
                          "lfs_sha256": getattr(lfs, "sha256", None)
                          if lfs else None}
    return out


def upload_and_verify(run: str, scr: Path,
                      product: dict[str, Any]) -> dict[str, Any]:
    """Upload once, check the hub's oids, restore elsewhere, re-identify."""
    from huggingface_hub import HfApi, hf_hub_download

    from aadistill.runtime.hub_capacity import would_accept
    from stages.phase_d1.behavioural_materialize import _verify_bytes_at

    digest = product["identities"]["artifact_digest"]
    local = scr / "products" / digest
    prefix = relay_prefix(run, digest)
    sizes = [(local / n).stat().st_size for n in PRODUCT_FILES]
    room = would_accept(_relay(), sizes, _token())
    present = remote_listing(prefix)
    api = HfApi(token=_token())
    uploaded = []
    for name in PRODUCT_FILES:
        remote = f"{prefix}/{name}"
        have = present.get(remote)
        sha = _sha(local / name)
        if have and have["size_bytes"] == (local / name).stat().st_size and \
                have["lfs_sha256"] in (None, sha):
            uploaded.append({"remote": remote, "uploaded": False})
            continue
        if not room.accepted:
            raise CloseoutRefused(f"the relay refuses the product: {room}")
        t0 = time.time()
        api.upload_file(path_or_fileobj=str(local / name), path_in_repo=remote,
                        repo_id=_relay(), repo_type="model",
                        commit_message=f"{run}: D3 product {name} (durability)")
        uploaded.append({"remote": remote, "uploaded": True,
                         "seconds": round(time.time() - t0, 1)})
    listing = remote_listing(prefix)
    objects, problems = [], []
    for name in PRODUCT_FILES:
        remote = f"{prefix}/{name}"
        have = listing.get(remote)
        sha = _sha(local / name)
        ok = bool(have) and have["size_bytes"] == (local / name).stat().st_size \
            and have["lfs_sha256"] in (None, sha)
        objects.append({"remote": remote, "sha256": sha,
                        "lfs_sha256": (have or {}).get("lfs_sha256"),
                        "matches_local": ok})
        if not ok:
            problems.append(remote)
    if problems:
        raise CloseoutRefused(f"the relay does not hold the local bytes: "
                              f"{problems}")
    tmp = Path(tempfile.mkdtemp(prefix="d3-restore-"))
    try:
        into = tmp / "product"
        into.mkdir()
        for name in PRODUCT_FILES:
            got = hf_hub_download(_relay(), f"{prefix}/{name}",
                                  repo_type="model", token=_token(),
                                  cache_dir=str(tmp / "hf"))
            shutil.copy(got, into / name)
        want = {k: product["identities"][k] for k in (
            "artifact_digest", "weights_digest", "single_shard_sha256",
            "arch_signature")}
        restored = _verify_bytes_at(into, want, what="the relay-restored "
                                                     "D3 product")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"schema": "aadistill.phase_d3.product_durability/v1",
            "run_id": run, "relay": _relay(), "prefix": prefix,
            "files": {o["remote"].rsplit("/", 1)[1]: {"path": o["remote"],
                                                       "sha256": o["sha256"]}
                      for o in objects},
            "objects": objects, "uploaded": uploaded,
            "restored_identities_checked": restored["verified"],
            "restore_dir": "an independent temporary directory, deleted after",
            "checked_utc": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ")}


# --- the closeout ----------------------------------------------------------------------

def build(run: str, scr: Path, *, balance_after: float | None,
          disk_gb: int) -> tuple[dict[str, Any], list[tuple[Path, Path]],
                                 dict[str, Any]]:
    from stages.phase_d2.closeout_d2_screening import money
    from stages.phase_d3 import d3_design as D
    from stages.phase_d3.autoinit_d3_structural_launch import (
        PRODUCT_REQUIRED_FOR,
    )

    run_dir = REPO_ROOT / RUNS_REL / run
    session = json.loads((run_dir / "runtime" / "session.json").read_text())
    auth = json.loads((run_dir / "governance" / "authorization.json"
                       ).read_text())
    record = _read_first(scr, "d3_structural.json")
    if record is None:
        raise CloseoutRefused(f"no driver record under {scr}")
    reg = D.registration(REPO_ROOT)
    plan = evidence_plan(run_dir, scr)
    derived, refused = None, None
    if record.get("status") == "COMPLETE":
        try:
            derived = rederive(record)
        except (CloseoutRefused, D.D3DesignError) as exc:
            refused = str(exc)
    outcome = (derived or {}).get("structural", {}).get("outcome")
    product_doc = _read_first(scr, "product.json")
    product = None
    if outcome in PRODUCT_REQUIRED_FOR and product_doc:
        durability = run_dir / DURABILITY
        relay = (json.loads(durability.read_text())
                 if durability.is_file() else None)
        product = {"identities": product_doc["identities"],
                   "files": product_doc["files"],
                   "local": product_bytes(scr, product_doc),
                   "relay": ({"repo": relay["relay"], "prefix": relay["prefix"],
                              "files": relay["files"]} if relay else None),
                   "relay_verified": bool(relay and relay[
                       "restored_identities_checked"])}
    gone = bool(session.get("provider_confirms_gone"))
    state = session.get("final_pod_state") or {}
    doc = {
        "schema": SCHEMA, "run_id": run, "stage_id": "1",
        "experiment_id": D.EXPERIMENT_ID,
        "classification": (
            f"COMPLETE: {outcome}, re-derived off-pod from the step records"
            if outcome else
            f"INCOMPLETE: driver status {record.get('status')!r}"
            + (f"; re-derivation refused: {refused}" if refused else "")),
        "recorded_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "session_commit": (session.get("session_commit_check") or {}).get(
            "session_commit"),
        "authorization_sha256": auth["authorization_sha256"],
        "registration_hash": reg["registration_hash"],
        "treatment_path_hash": auth["contract_hash"],
        "driver_status": record.get("status"),
        "runner_terminal": session.get("terminal"),
        "structural_outcome": outcome,
        "rederivation_refused": refused,
        "depth": ({k: derived["depth"][k] for k in (
            "removal_order", "kept_layers", "control_removal_order",
            "control_kept_layers",
            "rounds_where_the_parent_reference_would_choose_differently",
            "round0_parent_diagnostic_vs_control")} if derived else None),
        "structural": (derived or {}).get("structural"),
        "teacher_reference": record.get("teacher_reference"),
        "execution": record.get("execution"),
        "root_state": (record.get("root_state") or {}).get("chosen_candidate"),
        "product": product,
        "provider_resource_created": bool(
            session.get("provider_resource_created")),
        "provider_teardown_verified": bool(
            gone and state.get("exists") is False
            and state.get("billing") is False),
        "teardown_evidence": {"provider_confirms_gone": gone,
                              "final_pod_state": state,
                              "teardown_gate": session.get("teardown_gate")},
        "pod": {"pod_id": session.get("pod_id"),
                "image_identity": session.get("image_identity"),
                "draws": session.get("draws")},
        "money": money(session, disk_gb=disk_gb, balance_after=balance_after),
        "required_products_secured": session.get("required_products_secured"),
        "artifact_loss_risk": session.get("artifact_loss_risk"),
        "what_this_authorizes": (
            "nothing. A STRUCTURAL_CHANGE makes the three-probe rung "
            "proposable; the probes need their own maintainer grant."),
    }
    return doc, plan, record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", nargs="?", default="record",
                    choices=("record", "upload"))
    ap.add_argument("--run", required=True)
    ap.add_argument("--scr", required=True)
    ap.add_argument("--balance-after", type=float, default=None)
    ap.add_argument("--disk-gb", type=int, default=120)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    scr = Path(args.scr)
    run_dir = REPO_ROOT / RUNS_REL / args.run
    if args.action == "upload":
        product = _read_first(scr, "product.json")
        if not product:
            raise SystemExit("no product record came home")
        out = upload_and_verify(args.run, scr, product)
        print(json.dumps(out, indent=1))
        if args.write:
            (run_dir / DURABILITY).parent.mkdir(parents=True, exist_ok=True)
            (run_dir / DURABILITY).write_text(json.dumps(out, indent=1) + "\n")
        return 0
    doc, plan, record = build(args.run, scr, balance_after=args.balance_after,
                              disk_gb=args.disk_gb)
    present = [(s, d) for s, d in plan if s.is_file()]
    print(json.dumps({"copy": len(present),
                      "missing": [str(s) for s, _ in plan
                                  if not s.is_file()][:12],
                      "structural_outcome": doc["structural_outcome"],
                      "rederivation_refused": doc["rederivation_refused"],
                      "money": doc["money"],
                      "provider_teardown_verified":
                          doc["provider_teardown_verified"]},
                     indent=1, default=str))
    if not args.write:
        return 0
    for src, dest in present:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    (run_dir / "closeout").mkdir(parents=True, exist_ok=True)
    (run_dir / "closeout" / "outcome.json").write_text(
        json.dumps(doc, indent=1, default=str) + "\n")
    print(f"wrote {run_dir.relative_to(REPO_ROOT)}/closeout/outcome.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
