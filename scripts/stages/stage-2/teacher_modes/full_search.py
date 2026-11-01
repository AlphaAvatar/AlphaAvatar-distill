#!/usr/bin/env python3
"""The hybrid D4 full search (Research A v2, A1 phase B): its session, and its on-pod driver.

    cd /workspace/aad && HF_HUB_OFFLINE=1 PYTHONPATH=src:scripts:scripts/stages/stage-1:scripts/stages/stage-0/calibration_volume \\
        /opt/train/bin/python scripts/stages/stage-2/teacher_modes/full_search.py --authorization <a.json> \\
        --durable-root <volume>/teacher_modes --status <file> --deadline-minutes <m>

The method is the forward baseline's (`shared.forward_baseline`), read and never retyped: D4's four operators with
operator order and profile per operator free, beam.delayed_prune@v1 (width 6, warmup 1), Pareto v2, target-aware
scored positions, Top-200 + tail, scored_v1 rows, length-sorted packing. What this experiment supplies is what the
baseline leaves free:
* the root -- the hybrid ``Qwen/Qwen3-4B@1cfa9a72`` (`teacher_identity.json`), each weight shard hashed on the pod
  against its pinned sha256 before anything expands; its root id and shard-set hash are what every state's
  materialization identity, the commit and the A1/A2 lineage gate read;
* the calibration -- the two hybrid profiles (`hybrid_d4_profiles.json`, both modes in every cell);
* the state evaluation -- ``state_eval_hybrid@v1`` (`hybrid_state_eval.load`: one critical group per mode);
* the execution -- micro-batch 1 (declared before the run: micro-batch 3 measured out of memory in DEPTH on an L40S
  with this root), D4's DEPTH prefix reuse and on-device validation, ``prune_weights`` false;
* the geometry and seed -- C3's frozen target geometry (596M) and search seed (20260815), read from their owner.

THE FIRST STATE EVALUATION is also the device check (maintainer 2026-11-01: level 0's eight children are the real
search and the device admission at once). Before the search may continue: its values carry each mode's critical
group and ``state.critical_token_kl`` is their 50:50 mean; and the frozen epsilon rule
(`shared.epsilon_response.EPSILON_RESPONSE_V1`) is measured -- the child materialized, reloaded and measured
``repeats`` times, the range over the policy's objectives below the declared epsilon. Either failing stops the
search (STOPPED_FOR_REVIEW), its state on the volume.

A deadline (``--deadline-minutes``) stops the search with its journal intact (INCOMPLETE): a replacement session
of the same search identity resumes it; a partial ranking is never a Top-1. On completion: the commit (k = 1, the
first complete leaf in the Pareto order), the full ranking of every complete leaf, the Top-1's files copied to
``<durable>/search_a1/top1`` with their sha256, and ``products.json``.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1", "scripts/stages/stage-0/calibration_volume"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

TEACHER = REPO / "configs/stages/stage-2/teacher_modes/teacher_identity.json"
TYPES = REPO / "configs/stages/stage-2/teacher_modes/full_search_types.json"
PROFILES_DOC = REPO / "configs/stages/stage-2/teacher_modes/hybrid_d4_profiles.json"
SUITE_DIR = REPO / "artifacts/stages/stage-2/teacher_modes/state_eval_hybrid_v1"
SUITE_RECORD = REPO / "logs/stages/stage-2/teacher_modes/audit/state_eval_hybrid_freeze.json"
SEARCH_RUN_ID = "research_a_v2_hybrid_d4_full_search"
MICRO_BATCH = 1
FINALISTS_K = 1
SCHEMA = "aadistill.stage2.teacher_modes.full_search_record/v1"
TOP1_FILES = ("model.safetensors", "config.json", "generation_config.json")


class FullSearchError(RuntimeError):
    pass


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def root_identity(ident: dict | None = None) -> dict:
    """The hybrid root's id and shard-set hash (`aadistill.training.lineage.shard_set_sha256`'s rule, D4's)."""
    from aadistill.training.lineage import shard_set_sha256
    ident = ident or json.loads(TEACHER.read_text())
    shards = {s["file"]: s["lfs_sha256"] for s in ident["weight_shards"]}
    return {"root_teacher_id": f"{ident['model']}@{ident['revision']}", "root_teacher_sha256": shard_set_sha256(shards),
            "shard_sha256": shards}


def verify_root(path: Path, ident: dict | None = None) -> dict:
    """Every pinned shard of the staged snapshot hashed (its bytes, not its name) against its sha256."""
    rid = root_identity(ident)
    bad, got = [], {}
    for shard, want in rid["shard_sha256"].items():
        f = Path(path) / shard
        if not f.is_file():
            bad.append(f"{shard} is absent from {path}")
            continue
        got[shard] = sha(f)
        if got[shard] != want:
            bad.append(f"{shard} hashes to {got[shard][:12]}, the pin is {want[:12]}")
    if bad:
        raise FullSearchError("the staged root is not the hybrid teacher: " + "; ".join(bad))
    return {"verified": sorted(got), "root_teacher_id": rid["root_teacher_id"],
            "root_teacher_sha256": rid["root_teacher_sha256"]}


def profiles(doc_path: Path = PROFILES_DOC) -> tuple:
    """The two hybrid profiles, registered from their document; refused unless COMPLETE (no short cell)."""
    from aadistill.initialization.calibration.profiles import get_profile, load_profiles
    doc = json.loads(Path(doc_path).read_text())
    if not str(doc.get("status", "")).startswith("COMPLETE"):
        raise FullSearchError(f"{doc_path} is {doc.get('status')!r}: the profiles are not complete")
    load_profiles(doc)
    return tuple(get_profile(f"{p['profile_id']}@v{p['version']}") for p in doc["profiles"])


@dataclasses.dataclass
class Session:
    evaluator: object
    config: object
    execution: object
    numerics: object
    suite_manifest: dict
    baseline: dict

    def as_record(self) -> dict:
        return {"config_hash": self.config.config_hash,
                "measurement_protocol_id": self.evaluator.measurement_protocol_id,
                "profiles": [p.qualified_id for p in self.config.profiles],
                "profile_hashes": {p.qualified_id: p.profile_hash for p in self.config.profiles},
                "suite": {"suite_id": self.suite_manifest["suite_id"], "content_sha256": self.suite_manifest["content_sha256"],
                          "items_sha256": self.suite_manifest["items_sha256"]},
                "execution": {"evaluator": self.evaluator.execution.as_trace(), "operators": self.execution.as_trace()},
                "baseline": self.baseline, "seed": self.config.seed,
                "beam": {"width": self.config.schedule.width, "warmup_levels": self.config.schedule.warmup_levels},
                "prune_weights": self.config.prune_weights}


def build_session(*, workdir: Path, device: str = "cuda", suite_dir: Path = SUITE_DIR,
                  run_id: str = SEARCH_RUN_ID, repo_root: Path = REPO) -> Session:
    from aadistill.initialization.planning.metrics import StateEvaluator
    from aadistill.initialization.planning.search import SearchConfig
    from aadistill.initialization.specs.metrics import ReferenceStrategy
    from shared import forward_baseline as FB
    from stages.phase_c3 import session as CS
    from stages.phase_d1 import d1_session as D1S
    from stages.teacher_modes import hybrid_state_eval as HSE
    ident = json.loads(TEACHER.read_text())
    baseline = FB.selected(repo_root)
    D1S._register_frozen_operators()
    FB.register_operators(baseline)
    terms = FB.session_terms(baseline)
    if any(terms["operator_kl_reference"].values()):
        raise FullSearchError("the forward baseline declares a non-parent operator KL reference; none is passed here")
    suite, items, manifest = HSE.load(suite_dir, ident, TYPES)
    execution = dataclasses.replace(terms["execution"], micro_batch_size=MICRO_BATCH)
    evaluator = StateEvaluator(
        suite, items, device=device, vocab_size=int(ident["architecture"]["vocab_size"]),
        suite_content_sha256=manifest["content_sha256"], position_policy=terms["position_policy"],
        distribution_support=terms["distribution_support"], execution=execution, numerics=terms["numerics"],
        reference_strategy=ReferenceStrategy.RECOMPUTE)
    config = SearchConfig(
        run_id=run_id, target_spec=CS._target_spec(), schedule=terms["schedule"], seed=CS.SEARCH_SEED,
        workdir=Path(workdir), profiles=profiles(), policy=terms["ranking_policy"], suite=suite,
        position_policy=terms["position_policy"], distribution_support=terms["distribution_support"],
        measurement_protocol_id=evaluator.measurement_protocol_id, allowed_impls=terms["allowed_impls"],
        max_depth=None, impl_profiles=None, prune_weights=False, device=device)
    return Session(evaluator=evaluator, config=config,
                   execution=dataclasses.replace(execution, depth_prefix_reuse=True),
                   numerics=terms["numerics"], suite_manifest=manifest, baseline=terms["baseline"])


def critical_problems(values: dict) -> list[str]:
    """The first evaluation must carry both modes' critical groups and their 50:50 mean."""
    from stages.teacher_modes import hybrid_state_eval as HSE
    bad = []
    g = {m: values.get(f"state.critical_token_kl.group.{m}") for m in HSE.CRITICAL_GROUPS}
    if any(v is None for v in g.values()):
        return [f"the evaluation carries no critical group value for {[m for m, v in g.items() if v is None]}"]
    want = sum(g.values()) / len(g)
    if abs(values.get("state.critical_token_kl", float("nan")) - want) > 1e-9 * max(1.0, abs(want)):
        bad.append(f"state.critical_token_kl {values.get('state.critical_token_kl')} is not the groups' mean {want}")
    return bad


class FirstEvaluationCheck:
    """The search's measurer: the evaluator, with the first evaluation's device checks (module docstring)."""

    def __init__(self, session: Session, adapter, *, rule=None, scratch: Path):
        from shared.epsilon_response import EPSILON_RESPONSE_V1
        self.session, self.adapter, self.rule = session, adapter, rule or EPSILON_RESPONSE_V1
        self.scratch = Path(scratch)
        self.record: dict = {}

    def __call__(self, model, digest):
        ev = self.session.evaluator.evaluate(model, digest)
        if not self.record:
            self.record = self.check(model, digest, ev)
        return ev

    def check(self, model, digest, ev) -> dict:
        import torch
        objectives = [o.key for o in self.session.config.policy.objectives]
        out = {"artifact_digest": digest, "critical_problems": critical_problems(ev.values),
               "first_values": {k: ev.values[k] for k in objectives}}
        series = {k: [ev.values[k]] for k in objectives}
        d = self.scratch / "repeatability"
        for i in range(self.rule.repeats - 1):
            shutil.rmtree(d, ignore_errors=True)
            model.save_pretrained(d)
            again = type(model).from_pretrained(d, dtype=model.dtype).to(next(model.parameters()).device)
            again.config.use_cache = False
            e = self.session.evaluator.evaluate(again, digest)
            for k in objectives:
                series[k].append(e.values[k])
            del again
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        shutil.rmtree(d, ignore_errors=True)
        rng = max(max(v) - min(v) for v in series.values())
        out.update(repeats=self.rule.repeats, series=series, measured_range=rng, rule=self.rule.as_dict(),
                   decision=self.rule.evaluate(rng))
        problems = list(out["critical_problems"])
        if not out["decision"]["passes"]:
            problems.append(f"epsilon repeatability: range {rng} reaches the declared {self.rule.declared_epsilon}")
        out["passes"] = not problems
        (self.scratch / "first_evaluation_check.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
        if problems:
            raise StopForReview("; ".join(problems))
        return out


class StopForReview(RuntimeError):
    """A device check failed: the search stops with its state saved, for the maintainer."""


def persist_top1(row: dict, durable: Path) -> dict:
    """The Top-1's files copied beside the search with their sha256; the single shard must be the commit's."""
    src, dest = Path(row["checkpoint_path"]), Path(durable) / "search_a1" / "top1"
    dest.mkdir(parents=True, exist_ok=True)
    files = {}
    for n in TOP1_FILES:
        if (src / n).is_file():
            shutil.copyfile(src / n, dest / n)
            files[n] = sha(dest / n)
    if files.get("model.safetensors") != row.get("single_shard_sha256"):
        raise FullSearchError(f"the Top-1 copy hashes to {files.get('model.safetensors')}, the commit names "
                              f"{row.get('single_shard_sha256')}")
    return {"path": str(dest), "files_sha256": files, "state_id": row["state_id"],
            "artifact_digest": row.get("artifact_digest"), "weights_digest": row.get("weights_digest"),
            "source_checkpoint_path": str(src)}


def main(argv=None, *, teacher_loader=None, repo: Path | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--authorization", required=True)
    ap.add_argument("--durable-root", required=True)
    ap.add_argument("--status", required=True)
    ap.add_argument("--deadline-minutes", type=float, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--check-only", action="store_true", help="$0: the contract and the session, nothing loaded")
    a = ap.parse_args(argv)
    repo = Path(repo or REPO)
    durable, status = Path(a.durable_root), Path(a.status)
    work = durable / "search_a1"
    work.mkdir(parents=True, exist_ok=True)
    record = {"schema": SCHEMA, "search_run_id": SEARCH_RUN_ID, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "stages": {}, "status": "STARTED"}

    def mark(m: str) -> None:
        with status.open("a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} MARKER:{m}\n")
        print(f"MARKER:{m}", flush=True)

    def stage(name):
        record["stages"][name] = {"t0": time.time()}
        return record["stages"][name]

    def done(st, **kw):
        st.update(kw, seconds=round(time.time() - st.pop("t0"), 2), status="ok")

    def save():
        (work / "search_record.json").write_text(json.dumps(record, indent=1, default=str) + "\n")

    mark("DRIVER_START")
    try:
        st = stage("A_contract")
        if not a.check_only:
            from stages.teacher_modes import token_matched_phase as TM
            bound = TM.require_on_pod(Path(a.authorization), "search", repo)
            record["authorization_id"] = bound.get("authorization_id")
        session = build_session(workdir=work / "search", device=a.device)
        record["session"] = session.as_record()
        frozen = json.loads(SUITE_RECORD.read_text())["suite_identity"]["measurement_protocol_id"]
        if session.evaluator.measurement_protocol_id != frozen:
            raise FullSearchError(f"the evaluator's protocol id {session.evaluator.measurement_protocol_id} is not "
                                  f"the frozen suite record's {frozen}")
        done(st, config_hash=session.config.config_hash[:16])
        save()
        mark("CONTRACT_OK")
        if a.check_only:
            record["status"] = "CHECK_ONLY_OK"
            return 0

        st = stage("B_teacher")
        if teacher_loader is None:
            from huggingface_hub import snapshot_download

            from aadistill.initialization.specs.arch import get_adapter
            ident = json.loads(TEACHER.read_text())
            path = Path(snapshot_download(ident["model"], revision=ident["revision"], local_files_only=True))
            record["root_teacher"] = verify_root(path, ident)
            teacher = get_adapter("qwen3").load(str(path), dtype="bfloat16", device=a.device)
            teacher.config.use_cache = False
        else:
            teacher, record["root_teacher"] = teacher_loader()
        session.evaluator.prime_reference(teacher)
        done(st)
        save()
        mark("TEACHER_OK")

        st = stage("C_beam_search")
        from aadistill.initialization.calibration.items import prepare_calibration_items
        from aadistill.initialization.planning.search import BeamSearch, Deadline
        from aadistill.initialization.specs.arch import get_adapter
        adapter = get_adapter(session.config.target_spec.family)
        measurer = FirstEvaluationCheck(session, adapter, scratch=work)
        rid = root_identity()
        search = BeamSearch(
            adapter=adapter, config=session.config, root_teacher_id=rid["root_teacher_id"],
            root_teacher_sha256=rid["root_teacher_sha256"], root_loader=lambda: teacher,
            calibration_loader=lambda prof: prepare_calibration_items(prof.resolve(str(repo)), profile_id=prof.qualified_id),
            measurer=measurer, execution=session.execution, numerics=session.numerics,
            deadline=Deadline(seconds=float(a.deadline_minutes) * 60.0), validate_on_search_device=True)
        result = search.run()
        record["first_evaluation_check"] = measurer.record
        done(st, n_states=len(result.states), n_complete_leaves=len(result.complete_leaves), n_resumed=len(result.resumed))
        save()
        mark("SEARCH_OK")

        st = stage("D_commit")
        from aadistill.initialization.planning import stage1_selection
        policy = session.config.policy
        ranking = result.finalists(policy, FINALISTS_K)
        if len(ranking.selected) != FINALISTS_K:
            raise FullSearchError(f"the ranking selected {len(ranking.selected)} complete leaves, not {FINALISTS_K}")
        path = stage1_selection.commit(search_config=session.config, ranking=ranking, suite=session.config.suite,
                                       policy=policy, profiles=session.config.profiles,
                                       journal_path=work / "search" / "states.jsonl", directory=work)
        committed = json.loads(Path(path).read_text())
        full = result.finalists(policy, len(result.complete_leaves))
        record["commit"] = {"selection_path": str(path), "selected_rows": committed["selected"],
                            "pareto_order_of_complete_leaves": [s.state_id for s in full.selected]}
        record["root_teacher"].update(root_identity_hash=rid["root_teacher_sha256"])
        top1 = persist_top1(committed["selected"][0], durable)
        record["top1"] = top1
        products = {"schema": "aadistill.stage2.teacher_modes.full_search_products/v1", "search_run_id": SEARCH_RUN_ID,
                    "root_teacher": {"root_teacher_id": rid["root_teacher_id"],
                                     "root_teacher_sha256": rid["root_teacher_sha256"]},
                    "top1": top1, "selection": str(path), "journal": str(work / "search" / "states.jsonl")}
        (work / "products.json").write_text(json.dumps(products, indent=1) + "\n")
        done(st, k=FINALISTS_K)
        record["status"] = "COMPLETE"
    except BaseException as exc:  # noqa: BLE001
        from aadistill.initialization.planning.search import SearchDeadlineExceeded
        record["status"] = ("STOPPED_FOR_REVIEW" if isinstance(exc, StopForReview) else
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
