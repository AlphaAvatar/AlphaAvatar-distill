#!/usr/bin/env python3
"""The hybrid full search under the packed causal forward context (Research A packed round): session and driver.

    cd /workspace/aad && HF_HUB_OFFLINE=1 PYTHONPATH=src:scripts:scripts/stages/stage-1:scripts/stages/stage-0/calibration_volume \\
        /opt/train/bin/python scripts/stages/stage-2/teacher_modes/packed_full_search.py --authorization <a.json> \\
        --durable-root <volume>/teacher_modes --status <file> --deadline-minutes <m>

    # $0 on the dev box: the session, the identities and the packing of every set, nothing loaded
    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python -m stages.teacher_modes.packed_full_search --check-only \\
        --durable-root <scratch> --status <scratch>/status --deadline-minutes 1 [--inputs v1_dry_run]

The method is the forward baseline's version 2 (`shared.forward_baseline.selected()`), read and never retyped: D1's
four operators, beam.delayed_prune@v1 (width 6, warmup 1), Pareto v2, supervised_target_v1 positions, Top-200 + tail,
scored_v1 rows, DEPTH against the intact parent, state evaluation against the root -- now read in the packed causal
forward context (random shuffle -> sequential packing -> truncation -> standard causal attention). What this
experiment supplies (`packed_round_design.json`):

* the root -- the hybrid ``Qwen/Qwen3-4B@1cfa9a72``, every shard hashed before anything expands (`full_search`);
* the forward context's instance values -- sequence length 8,192 and its shuffle seed;
* the calibration -- the hybrid profiles v2 (v1's thirteen types in both modes plus ``tool_calling@v1``);
* the state evaluation -- ``state_eval_hybrid@v2`` (the same types, its own items, both modes' critical groups);
* the execution -- one packed sequence per forward, DEPTH prefix reuse, ``prune_weights`` false;
* the geometry and seed -- C3's 596M target and search seed, read from their owner, as in v2.

``--inputs v1_dry_run`` builds the same session over v1's profiles and suite, which exist, so the chain and every
identity can be checked at $0 before v2's tool-calling items exist. It is a structural check and never a search input.

The first state evaluation is the device admission, as in v2 (`full_search.FirstEvaluationCheck`): both modes'
critical groups present and the frozen epsilon rule measured on the device. A deadline stops the search with its
journal intact (INCOMPLETE) and a replacement session of the same search identity resumes it from the journal; a
partial ranking is never a Top-1. On completion: the commit, the full ranking, the Top-1 copied to
``<durable>/search_packed/top1`` with its sha256, and ``products.json``.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1", "scripts/stages/stage-0/calibration_volume"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.teacher_modes import full_search as FS  # noqa: E402

DESIGN = REPO / "configs/stages/stage-2/teacher_modes/packed_round_design.json"
SCHEMA = "aadistill.stage2.teacher_modes.packed_full_search_record/v1"
FINALISTS_K = 1


class PackedSearchError(RuntimeError):
    pass


def design() -> dict:
    return json.loads(DESIGN.read_text())


def inputs(which: str, d: dict | None = None) -> dict:
    """Where the profiles, the suite, its type table and its freeze record live, for ``which`` input set."""
    d = d or design()
    spec = d["full_search"]["inputs"][which]
    return {k: (REPO / v if isinstance(v, str) and k.endswith(("_doc", "_dir", "_record", "_types")) else v)
            for k, v in spec.items()}


def forward_context(baseline, d: dict | None = None):
    from shared import forward_baseline as FB
    d = d or design()
    fc = d["stage1_forward_context"]
    ctx = FB.forward_context(baseline, sequence_length=int(fc["sequence_length"]),
                             shuffle_seed=int(fc["shuffle_seed"]))
    if ctx is None:
        raise PackedSearchError(f"{baseline.qualified_id} declares no forward-context policy")
    return ctx


def build_session(*, workdir: Path, device: str = "cuda", which: str = "v2", repo_root: Path = REPO):
    from aadistill.initialization.planning.metrics import StateEvaluator
    from aadistill.initialization.planning.search import SearchConfig
    from aadistill.initialization.specs.metrics import ReferenceStrategy
    from shared import forward_baseline as FB
    from stages.phase_c3 import session as CS
    from stages.phase_d1 import d1_session as D1S
    from stages.teacher_modes import hybrid_state_eval as HSE
    d = design()
    src = inputs(which, d)
    if src.get("status") != "READY":
        raise PackedSearchError(f"input set {which!r} is {src.get('status')!r}: {src.get('why', '')}")
    ident = json.loads(FS.TEACHER.read_text())
    baseline = FB.selected(repo_root)
    D1S._register_frozen_operators()
    FB.register_operators(baseline)
    terms = FB.session_terms(baseline)
    if any(terms["operator_kl_reference"].values()):
        raise PackedSearchError("the forward baseline declares a non-parent operator KL reference")
    fc = forward_context(baseline, d)
    from stages.teacher_modes import packed_inputs as PI
    suite, items, manifest = HSE.load(src["suite_dir"], ident, src["suite_types"],
                                      critical_groups=(PI.CRITICAL_GROUPS_V2 if which == "v2" else None))
    execution = terms["execution"]
    evaluator = StateEvaluator(
        suite, items, device=device, vocab_size=int(ident["architecture"]["vocab_size"]),
        suite_content_sha256=manifest["content_sha256"], position_policy=terms["position_policy"],
        distribution_support=terms["distribution_support"], execution=execution, numerics=terms["numerics"],
        reference_strategy=ReferenceStrategy.RECOMPUTE, forward_context=fc)
    config = SearchConfig(
        run_id=d["full_search"]["search_run_id"], target_spec=CS._target_spec(), schedule=terms["schedule"],
        seed=CS.SEARCH_SEED, workdir=Path(workdir), profiles=FS.profiles(src["profiles_doc"]),
        policy=terms["ranking_policy"], suite=suite, position_policy=terms["position_policy"],
        distribution_support=terms["distribution_support"], measurement_protocol_id=evaluator.measurement_protocol_id,
        forward_context=fc, allowed_impls=terms["allowed_impls"], max_depth=None, impl_profiles=None,
        prune_weights=False, device=device)
    return FS.Session(evaluator=evaluator, config=config,
                      execution=dataclasses.replace(execution, depth_prefix_reuse=True),
                      numerics=terms["numerics"], suite_manifest=manifest, baseline=terms["baseline"])


def packing_record(session) -> dict:
    """Each set's packing identity and accounting under the session's forward context ($0)."""
    from aadistill.initialization.calibration.items import prepare_calibration_items
    fc = session.config.forward_context
    out = {}
    for prof in session.config.profiles:
        items = prepare_calibration_items(prof.resolve(str(REPO)), profile_id=prof.qualified_id)
        out[prof.qualified_id] = {"packing_identity": fc.packing_identity(items), **fc.report(items)}
    suite_items = [{"item_id": i.item_id, "input_ids": i.input_ids} for i in session.evaluator.items]
    out[session.config.suite.qualified_id] = {"packing_identity": fc.packing_identity(suite_items),
                                              **fc.report(suite_items)}
    return out


def session_record(session) -> dict:
    return {**session.as_record(), "forward_context": session.config.forward_context.as_dict(),
            "packing": packing_record(session)}


def persist_top1(row: dict, durable: Path) -> dict:
    """`full_search.persist_top1`'s rule, into this search's own directory."""
    import shutil
    src, dest = Path(row["checkpoint_path"]), Path(durable) / "search_packed" / "top1"
    dest.mkdir(parents=True, exist_ok=True)
    files = {}
    for n in FS.TOP1_FILES:
        if (src / n).is_file():
            shutil.copyfile(src / n, dest / n)
            files[n] = FS.sha(dest / n)
    if files.get("model.safetensors") != row.get("single_shard_sha256"):
        raise PackedSearchError(f"the Top-1 copy hashes to {files.get('model.safetensors')}, the commit names "
                                f"{row.get('single_shard_sha256')}")
    return {"path": str(dest), "files_sha256": files, "state_id": row["state_id"],
            "artifact_digest": row.get("artifact_digest"), "weights_digest": row.get("weights_digest"),
            "source_checkpoint_path": str(src)}


def main(argv=None, *, teacher_loader=None, repo: Path | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--authorization", default=None)
    ap.add_argument("--durable-root", required=True)
    ap.add_argument("--status", required=True)
    ap.add_argument("--deadline-minutes", type=float, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--inputs", default="v2", choices=("v2", "v1_dry_run"))
    ap.add_argument("--check-only", action="store_true", help="$0: the contract and the session, nothing loaded")
    a = ap.parse_args(argv)
    repo = Path(repo or REPO)
    if a.inputs != "v2" and not a.check_only:
        raise SystemExit("v1_dry_run is a structural check: it never feeds a search")
    d = design()
    durable, status = Path(a.durable_root), Path(a.status)
    work = durable / "search_packed"
    work.mkdir(parents=True, exist_ok=True)
    record = {"schema": SCHEMA, "search_run_id": d["full_search"]["search_run_id"], "inputs": a.inputs,
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stages": {}, "status": "STARTED"}

    def mark(m: str) -> None:
        with status.open("a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} MARKER:{m}\n")
        print(f"MARKER:{m}", flush=True)

    def save():
        (work / "search_record.json").write_text(json.dumps(record, indent=1, default=str) + "\n")

    mark("DRIVER_START")
    try:
        t0 = time.time()
        if not a.check_only:
            from stages.teacher_modes import token_matched_phase as TM
            if not a.authorization:
                raise PackedSearchError("a search needs its authorization")
            bound = TM.require_on_pod(Path(a.authorization), "packed_search", repo)
            record["authorization_id"] = bound.get("authorization_id")
        session = build_session(workdir=work / "search", device=a.device, which=a.inputs)
        record["session"] = session_record(session)
        frozen_path = inputs(a.inputs, d).get("suite_freeze_record")
        if frozen_path and Path(frozen_path).is_file() and a.inputs == "v2":
            frozen = json.loads(Path(frozen_path).read_text())["suite_identity"]["measurement_protocol_id"]
            if session.evaluator.measurement_protocol_id != frozen:
                raise PackedSearchError(f"the evaluator's protocol id {session.evaluator.measurement_protocol_id} is "
                                        f"not the frozen suite record's {frozen}")
        record["stages"]["A_contract"] = {"seconds": round(time.time() - t0, 2), "status": "ok"}
        save()
        mark("CONTRACT_OK")
        if a.check_only:
            record["status"] = "CHECK_ONLY_OK"
            return 0

        teacher, record["root_teacher"] = (teacher_loader or (lambda: FS.load_root(a.device)))()
        session.evaluator.prime_reference(teacher)
        save()
        mark("TEACHER_OK")

        from aadistill.initialization.calibration.items import prepare_calibration_items
        from aadistill.initialization.planning.search import BeamSearch, Deadline
        from aadistill.initialization.specs.arch import get_adapter
        adapter = get_adapter(session.config.target_spec.family)
        measurer = FS.FirstEvaluationCheck(session, adapter, scratch=work)
        rid = FS.root_identity()
        search = BeamSearch(
            adapter=adapter, config=session.config, root_teacher_id=rid["root_teacher_id"],
            root_teacher_sha256=rid["root_teacher_sha256"], root_loader=lambda: teacher,
            calibration_loader=lambda prof: prepare_calibration_items(prof.resolve(str(repo)), profile_id=prof.qualified_id),
            measurer=measurer, execution=session.execution, numerics=session.numerics,
            deadline=Deadline(seconds=float(a.deadline_minutes) * 60.0), validate_on_search_device=True)
        result = search.run()
        record["first_evaluation_check"] = measurer.record
        record["stages"]["C_beam_search"] = {"n_states": len(result.states),
                                             "n_complete_leaves": len(result.complete_leaves),
                                             "n_resumed": len(result.resumed)}
        save()
        mark("SEARCH_OK")

        from aadistill.initialization.planning import stage1_selection
        policy = session.config.policy
        ranking = result.finalists(policy, FINALISTS_K)
        if len(ranking.selected) != FINALISTS_K:
            raise PackedSearchError(f"the ranking selected {len(ranking.selected)} complete leaves, not {FINALISTS_K}")
        path = stage1_selection.commit(search_config=session.config, ranking=ranking, suite=session.config.suite,
                                       policy=policy, profiles=session.config.profiles,
                                       journal_path=work / "search" / "states.jsonl", directory=work)
        committed = json.loads(Path(path).read_text())
        full = result.finalists(policy, len(result.complete_leaves))
        record["commit"] = {"selection_path": str(path), "selected_rows": committed["selected"],
                            "pareto_order_of_complete_leaves": [s.state_id for s in full.selected]}
        top1 = persist_top1(committed["selected"][0], durable)
        record["top1"] = top1
        products = {"schema": "aadistill.stage2.teacher_modes.packed_full_search_products/v1",
                    "search_run_id": d["full_search"]["search_run_id"],
                    "root_teacher": {"root_teacher_id": rid["root_teacher_id"],
                                     "root_teacher_sha256": rid["root_teacher_sha256"]},
                    "forward_context": session.config.forward_context.as_dict(),
                    "top1": top1, "selection": str(path), "journal": str(work / "search" / "states.jsonl")}
        (work / "products.json").write_text(json.dumps(products, indent=1) + "\n")
        record["status"] = "COMPLETE"
    except BaseException as exc:  # noqa: BLE001
        from aadistill.initialization.planning.search import SearchDeadlineExceeded
        record["status"] = ("STOPPED_FOR_REVIEW" if isinstance(exc, FS.StopForReview) else
                            "INCOMPLETE" if isinstance(exc, SearchDeadlineExceeded) else "FAILED")
        record["failure"] = {"type": type(exc).__name__, "message": str(exc)[:2000],
                             "traceback": traceback.format_exc()[-6000:]}
    finally:
        record["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save()
        if not a.check_only:
            mark("ALL_DONE" if record["status"] == "COMPLETE" else f"RUN_FAILED:{record['status']}")
    return 0 if record["status"] in ("COMPLETE", "CHECK_ONLY_OK") else 1


if __name__ == "__main__":
    raise SystemExit(main())
