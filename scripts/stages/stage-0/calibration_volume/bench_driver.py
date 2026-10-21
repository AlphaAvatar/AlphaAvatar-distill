#!/usr/bin/env python3
"""On-pod driver of the Stage-0 search-cost validation. Engineering only.

    cd /workspace/aad && PYTHONPATH=src:scripts /opt/train/bin/python \
        scripts/stages/stage-0/calibration_volume/bench_driver.py \
        --auth <authorization.json> --plan bench --status <file> [--volume /runpod-volume]

The maintainer's 2026-10-19 condition before any formal full search: find and
fix what makes the search slow without changing what it computes, and measure
the gain on the real card. Each stage answers one question on the real teacher
and the real 1x calibration profile, and writes its answer at once:

* B -- is DEPTH prefix reuse BIT-identical on CUDA? the whole greedy (several
  rounds, non-trailing removals) on a profile subset, both paths: every
  candidate score, the removal order and the child's bytes;
* C -- how much faster is it at the real 1x scale? one full round on the whole
  profile, both paths, tables compared again;
* D -- what does validating the save/reload on the device save? one real
  teacher-sized expansion with the historical host validation and one with
  `validate_on_search_device`, same state identity and same measurement;
* E -- what does the network volume sustain, and does a mirror come back
  bit-identical? D's checkpoints pushed and pulled with `volume_sync`.

No result here is science: nothing is selected, ranked for a study, trained or
evaluated on a benchmark.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

AUDIT = "artifacts/audit/stage0_calvol_bench"
DEPTH_IMPL = "depth.causal_kl_greedy_v1"
ATTENTION_IMPL = "attention.activation_importance_v1"


def mark(status: Path, name: str) -> None:
    with status.open("a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} MARKER:{name}\n")
    print(f"MARKER:{name}", flush=True)


def cpu_budget() -> dict:
    out = {"visible": os.cpu_count(), "affinity": len(os.sched_getaffinity(0))}
    try:
        q, p = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        out["cgroup"] = None if q == "max" else round(int(q) / int(p), 2)
    except (OSError, ValueError):
        pass
    import torch
    out["torch_threads"] = torch.get_num_threads()
    return out


def gpu() -> dict:
    try:
        line = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                               "--format=csv,noheader"], capture_output=True, text=True,
                              timeout=30).stdout.strip()
        return dict(zip(("name", "memory", "driver"), [x.strip() for x in line.split(",")]))
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}


def tables(outcome) -> list:
    return [(r["chosen"], [(e["candidate"], e["score"]) for e in r["table"]])
            for r in outcome.artifacts["search_rounds"]]


def depth_run(session, teacher, profile, items, n_remove: int, reuse: bool, workdir: Path):
    """One DEPTH application to the teacher, exactly as the search builds it."""
    import torch
    from aadistill.initialization.operators.base import OperatorContext, get_implementation
    from aadistill.initialization.planning.operator_config import hashed_operator_config
    from aadistill.initialization.specs.arch import get_adapter

    impl = get_implementation(DEPTH_IMPL)
    adapter = get_adapter(session.config.target_spec.family)
    parent = adapter.spec_of(teacher)
    target = parent.replace(num_hidden_layers=parent["num_hidden_layers"] - n_remove)
    config = hashed_operator_config(
        implementation=impl, policy=session.config.position_policy,
        support=session.config.distribution_support, items=items,
        objective=session.config.scoring_objective)
    ctx = OperatorContext(
        adapter=adapter, model=teacher, parent_spec=parent, target_spec=target,
        profile=profile, calibration_items=items, seed=session.config.seed,
        device="cuda", workdir=workdir, config=dict(config),
        execution=dataclasses.replace(session.evaluator.execution, depth_prefix_reuse=reuse),
        position_policy=session.config.position_policy,
        distribution_support=session.config.distribution_support,
        scoring_objective=session.config.scoring_objective)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = impl.execute(ctx)
    torch.cuda.synchronize()
    return out, round(time.time() - t0, 2), round(torch.cuda.max_memory_allocated() / 2**30, 2)


def same_weights(a, b) -> bool:
    import torch
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(
        torch.equal(sa[k].detach().cpu(), sb[k].detach().cpu()) for k in sa)


def sha_tree(root: Path) -> dict:
    out = {}
    for f in sorted(root.rglob("*")):
        if f.is_file():
            h = hashlib.sha256()
            with f.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 24), b""):
                    h.update(chunk)
            out[str(f.relative_to(root))] = h.hexdigest()
    return out


def main(argv=None) -> int:
    from stages.calibration_volume import governance as G
    from stages.calibration_volume import search_session as SS

    ap = argparse.ArgumentParser()
    ap.add_argument("--auth", required=True)
    ap.add_argument("--plan", default="bench")
    ap.add_argument("--status", required=True)
    ap.add_argument("--volume", default="")
    args = ap.parse_args(argv)
    plan = G.plan_doc(args.plan)
    out = REPO / AUDIT
    out.mkdir(parents=True, exist_ok=True)
    status = Path(args.status)
    record = {"schema": "aadistill.stage0.calibration_volume.bench_record/v1",
              "plan": args.plan, "plan_hash": G.plan_hash(args.plan),
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "gpu": gpu(), "cpu": cpu_budget(), "stages": {}, "status": "STARTED"}

    def save():
        (out / "bench.json").write_text(json.dumps(record, indent=1, default=str) + "\n")

    def stage(name, fn):
        t0 = time.time()
        try:
            record["stages"][name] = {"status": "ok", **fn()}
        except Exception as exc:  # noqa: BLE001 -- one failed question must not cost the others
            record["stages"][name] = {"status": "FAILED", "error": repr(exc)[:1500],
                                      "traceback": traceback.format_exc()[-4000:]}
        record["stages"][name]["seconds"] = round(time.time() - t0, 1)
        save()
        mark(status, f"STAGE:{name}:{record['stages'][name]['status']}")

    mark(status, "DRIVER_START")
    try:
        import torch
        from aadistill.initialization.calibration.items import prepare_calibration_items
        from aadistill.initialization.calibration.profiles import get_profile
        from stages.phase_d1 import autoinit_d1_driver as D1D
        from stages.phase_d1 import d1_session as D1S

        auth = G.load_for("bench", args.auth)
        auth.require_plan(G.plan_hash(args.plan))
        record["authorization_id"] = auth.authorization_id
        SS.register_stage0_profiles()
        profile = get_profile(plan["profile"])
        if profile.profile_hash != plan["profile_hash"]:
            raise RuntimeError(f"{plan['profile']} hashes to {profile.profile_hash}")
        items_all = prepare_calibration_items(profile.resolve(str(REPO)),
                                              profile_id=profile.qualified_id)
        session = SS.build_session(workdir=out / "unused", run_id=plan["run_label"],
                                   profiles=(profile,), device="cuda")
        teacher, teacher_id = D1D._load_root_teacher()
        record["root_teacher"] = D1S.verify_staged_teacher(teacher_id["path"])
        session.evaluator.prime_reference(teacher)
        record["calibration"] = {"profile": profile.qualified_id, "items": len(items_all),
                                 "tokens": int(sum(int(i["input_ids"].shape[1]) for i in items_all))}
        record["versions"] = {"torch": torch.__version__,
                              "transformers": __import__("transformers").__version__}
        save()

        def exact():
            sub = items_all[::int(plan["exact_subset_stride"])]
            n = int(plan["exact_remove"])
            off, t_off, m_off = depth_run(session, teacher, profile, sub, n, False, out / "w")
            on, t_on, m_on = depth_run(session, teacher, profile, sub, n, True, out / "w")
            equal = tables(off) == tables(on)
            return {"items": len(sub), "n_remove": n, "removal_order": on.trace["removal_order"],
                    "tables_bit_identical": equal,
                    "kept_layers_equal": off.trace["kept_layers"] == on.trace["kept_layers"],
                    "child_weights_bit_identical": same_weights(off.model, on.model),
                    "path_seconds": {"off": t_off, "on": t_on}, "peak_gib": {"off": m_off, "on": m_on},
                    "first_difference": None if equal else next(
                        ([a, b] for a, b in zip(tables(off), tables(on)) if a != b), None),
                    "timing_on": on.artifacts["timing"]}

        def speed():
            off, t_off, m_off = depth_run(session, teacher, profile, items_all, 1, False, out / "w")
            on, t_on, m_on = depth_run(session, teacher, profile, items_all, 1, True, out / "w")
            return {"items": len(items_all), "n_remove": 1,
                    "tables_bit_identical": tables(off) == tables(on),
                    "path_seconds": {"off": t_off, "on": t_on},
                    "speedup": round(t_off / t_on, 3) if t_on else None,
                    "peak_gib": {"off": m_off, "on": m_on},
                    "timing_off": off.artifacts["timing"], "timing_on": on.artifacts["timing"]}

        def overhead():
            from aadistill.initialization.planning.search import BeamSearch
            from aadistill.initialization.specs.arch import get_adapter
            adapter = get_adapter(session.config.target_spec.family)
            root = adapter.spec_of(teacher)
            want = session.config.target_spec
            #: a one-operator target: only the attention geometry differs, so
            #: the search is ONE teacher-sized ATTENTION expansion and a leaf
            target = root.replace(num_attention_heads=want["num_attention_heads"],
                                  num_key_value_heads=want["num_key_value_heads"])
            rid = D1S.root_teacher_identity()
            res = {}
            for label, on_device in (("host_validation", False), ("device_validation", True)):
                cfg = dataclasses.replace(session.config, target_spec=target,
                                          workdir=out / f"overhead_{label}", prune_weights=False)
                search = BeamSearch(
                    adapter=adapter, config=cfg, root_teacher_id=rid["root_teacher_id"],
                    root_teacher_sha256=rid["root_teacher_sha256"], root_loader=lambda: teacher,
                    calibration_loader=lambda p: items_all,
                    measurer=lambda m, d: session.evaluator.evaluate(m, d),
                    execution=session.evaluator.execution, numerics=session.numerics,
                    validate_on_search_device=on_device)
                result = search.run()
                leaf = result.complete_leaves[0]
                tel = [json.loads(l) for l in (cfg.workdir / "telemetry.jsonl").read_text().splitlines()]
                exp = next(t for t in tel if t["event"] == "expansion")
                res[label] = {"state_id": leaf.state_id, "artifact_digest": leaf.artifact_digest,
                              "evaluation": leaf.evaluation.values,
                              "validation": leaf.notes.get("validation"),
                              "operator_seconds": exp["operator_seconds"],
                              "phases": {k: v for k, v in exp.items() if k.endswith("_seconds")
                                         and isinstance(v, dict)},
                              "checkpoint": str(leaf.checkpoint_path)}
            a, b = res["host_validation"], res["device_validation"]
            res["same_state_and_digest"] = (a["state_id"] == b["state_id"]
                                            and a["artifact_digest"] == b["artifact_digest"])
            res["same_evaluation"] = a["evaluation"] == b["evaluation"]
            return res

        def volume():
            from stages.calibration_volume import volume_sync as VS
            mount = args.volume
            if not mount or subprocess.run(["mountpoint", "-q", mount]).returncode != 0:
                return {"skipped": f"no network volume mounted at {mount!r}"}
            src = Path(record["stages"]["D_overhead"]["device_validation"]["checkpoint"])
            dst = Path(mount) / "bench" / src.name
            t0 = time.time()
            push = VS.sync(src, dst)
            t_push = time.time() - t0
            back = out / "pulled" / src.name
            t0 = time.time()
            pull = VS.sync(dst, back)
            t_pull = time.time() - t0
            same = sha_tree(src) == sha_tree(back)
            t0 = time.time()
            sha_tree(dst)
            t_read = time.time() - t0
            subprocess.run(["rm", "-rf", str(Path(mount) / "bench")])
            gb = push["bytes"] / 1e9
            return {"bytes": push["bytes"], "push_seconds": round(t_push, 1),
                    "push_MBps": round(gb * 1e3 / t_push, 1) if t_push else None,
                    "pull_seconds": round(t_pull, 1),
                    "pull_MBps": round(gb * 1e3 / t_pull, 1) if t_pull else None,
                    "read_sha_seconds": round(t_read, 1), "round_trip_bit_identical": same,
                    "pull": pull}

        stage("B_depth_exactness", exact)
        stage("C_depth_speed", speed)
        stage("D_overhead", overhead)
        stage("E_volume", volume)
        bad = [k for k, v in record["stages"].items() if v["status"] != "ok"]
        record["status"] = "COMPLETE" if not bad else "INCOMPLETE"
        record["failed_stages"] = bad
    except BaseException as exc:  # noqa: BLE001
        record["status"] = "FAILED"
        record["failure"] = {"type": type(exc).__name__, "message": str(exc)[:2000],
                             "traceback": traceback.format_exc()[-6000:]}
    finally:
        record["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save()
        mark(status, "ALL_DONE" if record["status"] in ("COMPLETE", "INCOMPLETE") else "RUN_FAILED")
    return 0 if record["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
