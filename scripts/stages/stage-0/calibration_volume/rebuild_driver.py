#!/usr/bin/env python3
"""Rebuild a dose's retired finalists on the pod, every step digest-pinned.

    cd /workspace/aad && PYTHONPATH=src:scripts /opt/train/bin/python \
        scripts/stages/stage-0/calibration_volume/rebuild_driver.py \
        --plan <rebuild plan.json> --work <local dir> --dest <volume dir> --record <json>

The plan is `finalists.py`'s, resolved and checked at $0 where the journal lives.
Here each leaf is replayed by `materialize_fixed_path` under the Stage-0
search's own spec, adopted only when all four identities match exactly
(`replay_specs.adoption_matches`), and copied to `--dest/<state_id>/` with its
shard re-hashed there. A leaf already at its destination with the right shard
is not rebuilt. A digest mismatch is terminal for that leaf and is recorded,
never retried and never substituted. Runs on the GPU the search ran on: the
success criterion is byte equality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))


def shard_sha(d: Path) -> str | None:
    shards = sorted(Path(d).glob("*.safetensors"))
    if len(shards) != 1:
        return None
    h = hashlib.sha256()
    with shards[0].open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--record", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--check-only", action="store_true",
                    help="resolve the plan, the registries, every spec and the root state; "
                         "load no weights and write no checkpoint")
    args = ap.parse_args(argv)
    doc = json.loads(args.plan.read_text())
    record = {"schema": "aadistill.stage0.calibration_volume.rebuild/v1",
              "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
              "treatment": doc.get("treatment"), "leaves": [], "status": "RUNNING"}

    def save():
        args.record.write_text(json.dumps(record, indent=1, default=str) + "\n")

    try:
        from aadistill.initialization.planning.fixed_path import (FixedPathDigestMismatch,
                                                                  materialize_fixed_path)
        from aadistill.initialization.specs.arch import get_adapter
        from stages.calibration_volume import finalists as F
        from stages.phase_d1 import d1_session as D1S
        from stages.phase_d1 import replay_specs as R

        leaves = R.leaves_from_plan(doc)
        terms, prereg = F._registries(REPO)
        execution = F.search_execution(terms, prereg)
        record["execution_agreement"] = R.verify_execution_config(leaves, execution=execution)
        root = D1S.root_teacher_identity(REPO)
        specs = [(leaf, F.fixed_path_spec(leaf, device=args.device)) for leaf in leaves]
        root_state = R.root_state_from_plan(
            doc, leaves, base_config=R.teacher_config(root["repo_id"], root["revision"]),
            family=specs[0][1].family)
        overrides = dict(root_state["config_overrides"])
        record["root_state"] = root_state
        adapter = get_adapter(specs[0][1].family)
        record["spec_hashes"] = {leaf.state_id: spec.spec_hash for leaf, spec in specs}
        if record["spec_hashes"] != doc.get("spec_hashes", record["spec_hashes"]):
            raise RuntimeError(f"the specs built here {record['spec_hashes']} are not the plan's "
                               f"{doc.get('spec_hashes')}")
        save()
        if args.check_only:
            record["status"] = "CHECK_ONLY_OK"
            return 0
        for leaf, spec in specs:
            entry = {"state_id": leaf.state_id, "quality_position": leaf.quality_position,
                     "path_label": leaf.path_label, "spec_hash": spec.spec_hash}
            record["leaves"].append(entry)
            dest = args.dest / leaf.state_id
            if shard_sha(dest) == leaf.expected_single_shard_sha256:
                entry.update(adopted=True, already_on_volume=True)
                save()
                continue
            work = args.work / leaf.state_id
            shutil.rmtree(work, ignore_errors=True)
            work.mkdir(parents=True)
            t0 = time.time()
            try:
                results = materialize_fixed_path(
                    spec, adapter=adapter,
                    root_loader=lambda _s=spec: R.load_root_model(
                        _s, config_overrides=overrides, device=args.device),
                    workdir=work, repo_root=REPO, execution=execution,
                    on_step=lambda r, _e=entry: (_e.setdefault("steps", []).append(
                        {"index": r.index, "impl_id": r.impl_id,
                         "digest": r.identity.artifact_digest,
                         "seconds": round(float(r.seconds), 1)}), save()))
            except FixedPathDigestMismatch as exc:
                entry["digest_mismatch"] = {k: getattr(exc, k, None)
                                            for k in ("step_index", "label", "expected", "actual")}
                entry["seconds"] = round(time.time() - t0, 1)
                save()
                continue
            ok, detail = R.adoption_matches(leaf, results[-1].identity)
            entry.update(seconds=round(time.time() - t0, 1), identity=detail)
            if ok:
                tmp = args.dest / f".{leaf.state_id}.partial"
                shutil.rmtree(tmp, ignore_errors=True)
                shutil.copytree(results[-1].checkpoint_path, tmp)
                if shard_sha(tmp) != leaf.expected_single_shard_sha256:
                    raise RuntimeError(f"{leaf.state_id}: the copy on the volume does not hash "
                                       "to the adopted shard")
                shutil.rmtree(dest, ignore_errors=True)
                tmp.rename(dest)
            entry["adopted"] = bool(ok)
            shutil.rmtree(work, ignore_errors=True)
            save()
        record["status"] = ("COMPLETE" if all(e.get("adopted") for e in record["leaves"])
                            else "INCOMPLETE")
    except BaseException as exc:  # noqa: BLE001
        record["status"] = "FAILED"
        record["failure"] = {"type": type(exc).__name__, "message": str(exc)[:2000],
                             "traceback": traceback.format_exc()[-6000:]}
    finally:
        save()
    return 0 if record["status"] in ("COMPLETE", "CHECK_ONLY_OK") else 1


if __name__ == "__main__":
    raise SystemExit(main())
