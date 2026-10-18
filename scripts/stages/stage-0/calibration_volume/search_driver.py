#!/usr/bin/env python3
"""On-pod driver of one Stage-0 volume-treatment search (the D1 full search).

    cd /workspace/aad && PYTHONPATH=src:scripts /opt/train/bin/python \
        scripts/stages/stage-0/calibration_volume/search_driver.py \
        --treatment <id> --run-id <run> --auth <authorization.json> --status <file>

Stages, each journaled:

* A -- environment, registries, the treatment's profiles resolved from the
  preregistration and their bytes checked, the session built from the forward
  baseline (`search_session.build_session`), the authorization bound to the
  plan;
* B -- the root teacher, staged by setup and hash-verified against the frozen
  binding (`stages.phase_d1.d1_session.verify_staged_teacher`);
* C -- the beam search, exactly as the forward baseline defines it;
* D -- the preregistered finalist commit (`result.finalists(policy, k)` and
  `stage1_selection.commit`, the D1 committer), and a structural record of
  every expansion's operator decision for the H1 analysis.

The committed checkpoints are this session's PRODUCTS; the launcher secures
them off-pod before teardown, because the fixed-recovery probes train from
them.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

AUDIT = "artifacts/audit/stage0_calvol_search"
PREREG = REPO / "logs/stages/stage-0/calibration_volume/plans/volume_preregistration.json"


def mark(status: Path, name: str) -> None:
    with status.open("a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} MARKER:{name}\n")
    print(f"MARKER:{name}", flush=True)


def structural_record(states_path: Path) -> list[dict]:
    """Every expansion's operator decision, read from the search's own journal."""
    out = []
    for line in states_path.read_text().splitlines():
        s = json.loads(line)
        steps = s.get("steps") or []
        last = steps[-1] if steps else {}
        out.append({"state_id": s["state_id"], "parent_id": s.get("parent_id"),
                    "depth": s.get("depth"), "path": s.get("path_label"),
                    "applied_kinds": s.get("applied_kinds"),
                    "impl_ids": s.get("impl_ids"),
                    "profiles": s.get("calibration_profiles"),
                    "is_complete_leaf": s.get("is_complete_leaf"),
                    "artifact_digest": s.get("artifact_digest"),
                    "decision": last.get("artifacts"),
                    "evaluation_summary": {k: (s.get("evaluation") or {}).get(k)
                                           for k in ("objective", "value", "rank")}})
    return out


def width_projections(states_path: Path, teacher, out_dir: Path) -> list[dict]:
    """Recover each ROOT-level RESIDUAL_WIDTH child's projection, P = argmin |E_t P - E_s|.

    The operator records only diagnostics, not its basis, and the H1 comparison
    of PCA subspaces across resamples needs the basis. The child's input
    embedding is the teacher's projected (E_s = E_t P), so P is recovered from
    the normal equations on the GPU. Saved as float32, one file per root child.
    """
    import torch
    from safetensors.torch import load_file, save_file

    e_t = teacher.get_input_embeddings().weight.detach().float()
    gram = e_t.T @ e_t
    out = []
    for line in states_path.read_text().splitlines():
        s = json.loads(line)
        if s.get("depth") != 1 or s.get("applied_kinds") != ["RESIDUAL_WIDTH"]:
            continue
        ckpt = Path(s["checkpoint_path"])
        files = sorted(ckpt.glob("*.safetensors"))
        tensors = {}
        for f in files:
            tensors.update(load_file(str(f), device="cuda"))
        key = next(k for k in tensors if k.endswith("embed_tokens.weight"))
        e_s = tensors[key].float()
        proj = torch.linalg.solve(gram, e_t.T @ e_s)
        resid = (e_t @ proj - e_s).norm() / e_s.norm()
        name = f"width_projection_{s['state_id']}.safetensors"
        save_file({"P": proj.cpu().contiguous()}, str(out_dir / name))
        out.append({"state_id": s["state_id"], "profiles": s.get("calibration_profiles"),
                    "file": name, "relative_residual": float(resid),
                    "shape": list(proj.shape)})
        del tensors
    return out


def main(argv=None) -> int:
    from stages.calibration_volume import governance as G
    from stages.calibration_volume import search_session as SS

    ap = argparse.ArgumentParser()
    ap.add_argument("--treatment", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--auth", required=True)
    ap.add_argument("--status", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--plan", default="search", help="the session plan name the authorization binds")
    ap.add_argument("--treatments-doc", default=str(PREREG.relative_to(REPO)),
                    help="the document that freezes this treatment's profile hashes")
    ap.add_argument("--max-depth", type=int, default=None,
                    help="1 = the root-level pilot: no complete leaf, no commit")
    args = ap.parse_args(argv)

    out = REPO / AUDIT / args.treatment
    out.mkdir(parents=True, exist_ok=True)
    status = Path(args.status)
    record = {"schema": "aadistill.stage0.calibration_volume.search_record/v1",
              "treatment": args.treatment, "run_id": args.run_id,
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "stages": {}, "status": "STARTED"}

    def stage(name):
        record["stages"][name] = {"t0": time.time()}
        return record["stages"][name]

    def done(st, **kw):
        st.update(kw, seconds=round(time.time() - st.pop("t0"), 2), status="ok")

    def save():
        (out / "search_record.json").write_text(json.dumps(record, indent=1, default=str) + "\n")

    mark(status, f"DRIVER_START:{args.treatment}")
    try:
        st = stage("A_contract")
        prereg = json.loads((REPO / args.treatments_doc).read_text())
        table = prereg.get("treatments_by_id") or prereg["treatments"]
        treat = table[args.treatment]
        profiles = SS.treatment_profiles(args.treatment)
        for p in profiles:
            want = treat["profile_hashes"][p.qualified_id]
            if p.profile_hash != want:
                raise RuntimeError(f"{p.qualified_id} hashes to {p.profile_hash}, "
                                   f"the preregistration froze {want}")
            p.resolve(str(REPO))   # checks items_file_sha256 and content_sha256
        session = SS.build_session(workdir=out / "search", run_id=args.run_id,
                                   profiles=profiles, device=args.device,
                                   max_depth=args.max_depth,
                                   impl_profiles=treat.get("impl_profiles"))
        record["session"] = session.as_record()
        if not args.check_only:
            auth = G.load_for("search", args.auth)
            auth.require_plan(G.plan_hash(args.plan))
            record["authorization_id"] = auth.authorization_id
        done(st, config_hash=session.config.config_hash[:16])
        save()
        if args.check_only:
            record["status"] = "CHECK_ONLY_OK"
            return 0

        st = stage("B_teacher")
        from stages.phase_d1 import autoinit_d1_driver as D1D
        from stages.phase_d1 import d1_session as D1S
        teacher, teacher_id = D1D._load_root_teacher()
        record["root_teacher"] = D1S.verify_staged_teacher(teacher_id["path"])
        session.evaluator.prime_reference(teacher)
        done(st, revision=teacher_id.get("revision"))
        save()

        st = stage("C_beam_search")
        from aadistill.initialization.calibration.items import prepare_calibration_items
        from aadistill.initialization.planning.search import BeamSearch
        from aadistill.initialization.specs.arch import get_adapter
        root = D1S.root_teacher_identity()
        search = BeamSearch(
            adapter=get_adapter(session.config.target_spec.family),
            config=session.config,
            root_teacher_id=root["root_teacher_id"],
            root_teacher_sha256=root["root_teacher_sha256"],
            root_loader=lambda: teacher,
            calibration_loader=lambda prof: prepare_calibration_items(
                prof.resolve(str(REPO)), profile_id=prof.qualified_id),
            measurer=lambda model, digest: session.evaluator.evaluate(model, digest),
            execution=session.evaluator.execution,
            numerics=session.numerics)
        result = search.run()
        done(st, n_states=len(result.states), n_complete_leaves=len(result.complete_leaves))
        save()

        st = stage("D_commit")
        if args.max_depth is not None:
            (out / "structure.json").write_text(json.dumps(
                structural_record(out / "search" / "states.jsonl"), default=str) + "\n")
            record["width_projections"] = width_projections(
                out / "search" / "states.jsonl", teacher, out)
            done(st, k=0, note="root-level pilot: no finalist is committed")
            record["status"] = "COMPLETE"
            return 0
        from aadistill.initialization.planning import stage1_selection
        k = int(prereg["search"]["finalists_k"])
        policy = session.config.policy
        ranking = result.finalists(policy, k)
        if len(ranking.selected) != k:
            raise RuntimeError(f"the ranking selected {len(ranking.selected)} complete "
                               f"leaves; the preregistration commits {k}")
        path = stage1_selection.commit(
            search_config=session.config, ranking=ranking, suite=session.config.suite,
            policy=policy, profiles=session.config.profiles,
            journal_path=out / "search" / "states.jsonl", directory=out)
        committed = json.loads(Path(path).read_text())
        record["commit"] = {"selection_path": str(Path(path).relative_to(REPO)),
                            "selected_rows": committed["selected"]}
        (out / "structure.json").write_text(json.dumps(
            structural_record(out / "search" / "states.jsonl"), default=str) + "\n")
        try:
            record["width_projections"] = width_projections(
                out / "search" / "states.jsonl", teacher, out)
        except Exception as exc:  # noqa: BLE001 -- a diagnostic never fails the search
            record["width_projections_error"] = repr(exc)[:500]
        done(st, k=k)
        record["status"] = "COMPLETE"
    except BaseException as exc:  # noqa: BLE001
        record["status"] = "FAILED"
        record["failure"] = {"type": type(exc).__name__, "message": str(exc)[:2000],
                             "traceback": traceback.format_exc()[-6000:]}
    finally:
        record["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save()
        if not args.check_only:
            mark(status, f"SEARCH_DONE:{args.treatment}" if record["status"] == "COMPLETE"
                 else f"SEARCH_FAILED:{args.treatment}")
    return 0 if record["status"] in ("COMPLETE", "CHECK_ONLY_OK") else 1


if __name__ == "__main__":
    raise SystemExit(main())
