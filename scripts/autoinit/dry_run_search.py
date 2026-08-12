"""Run a complete AutoInitializer beam search on tiny models. Zero cost.

    PYTHONPATH=src python scripts/autoinit/dry_run_search.py \
        --out artifacts/autoinit/dryrun

Not a simulation. It builds a real (32-wide, 6-layer) Qwen3 teacher, runs the
real operators, writes real checkpoints, reloads them through the real
``from_pretrained`` path, hashes them, measures them against the real teacher and
ranks them with the real policy — then emits the same manifest a paid run would.
The only thing scaled down is the model.

The point is that the *mandatory cycle* is what gets exercised. A defect in
materialize -> reload -> hash -> validate -> measure costs nothing to find here
and has cost this project real money to find on a pod (STATE.md 0.6).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.autoinit.arch import ArchSpec, get_adapter  # noqa: E402
from aadistill.autoinit.calibration import CalibrationProfile, CalibrationSource  # noqa: E402
from aadistill.autoinit.cost import checkpoint_bytes  # noqa: E402
from aadistill.autoinit.manifest import build_manifest, verify_manifest, write_manifest  # noqa: E402
from aadistill.autoinit.metrics import StateEvalSuite, StateEvaluator, SuiteItem  # noqa: E402
from aadistill.autoinit.ranking import PARETO_V1, SCHEDULE_V1, BeamSchedule  # noqa: E402
from aadistill.autoinit.recovery import (  # noqa: E402
    E1_KD_HEAVY_0860K,
    SuccessiveHalvingPlan,
    admit_leaves,
    probe_configs,
)
from aadistill.autoinit.search import BeamSearch, SearchConfig  # noqa: E402
from aadistill.autoinit.artifact import identify_checkpoint  # noqa: E402
from aadistill.autoinit.state import make_control_state  # noqa: E402
from aadistill.infrastructure.env import code_state, hardware_report  # noqa: E402

TEACHER_GEOMETRY = dict(hidden_size=32, num_hidden_layers=6, intermediate_size=48,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                        vocab_size=128, tie_word_embeddings=True)
TARGET_GEOMETRY = dict(hidden_size=16, num_hidden_layers=4, intermediate_size=24,
                       num_attention_heads=2, num_key_value_heads=2, head_dim=8,
                       vocab_size=128, tie_word_embeddings=True)
DOMAINS = {"general": ("text",), "math": ("arith",)}


def build_teacher(seed: int):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(
        Qwen3Config(max_position_embeddings=256, rope_theta=5_000_000,
                    **TEACHER_GEOMETRY)).float().eval()
    with torch.no_grad():
        for module in model.modules():
            if module.__class__.__name__ == "Qwen3RMSNorm":
                module.weight.uniform_(0.5, 1.5)
    return model


def build_items(seed: int, n_per_subtype: int, seq_len: int):
    torch.manual_seed(seed)
    items = []
    for domain, (subtype,) in DOMAINS.items():
        for k in range(n_per_subtype):
            ids = torch.randint(0, TEACHER_GEOMETRY["vocab_size"], (1, seq_len))
            targets = ids[0, 1:]
            items.append({"item_id": f"{subtype}-{k}", "input_ids": ids,
                          "domain": domain, "subtype": subtype,
                          "tags": {"eos_like": targets == 0,
                                   "answer_like": targets % 17 == 0}})
    return items


def make_profiles(n: int):
    names = ["stage0_current", "domain_balanced", "reasoning_heavy"][:n]
    return tuple(
        CalibrationProfile(
            profile_id=f"dryrun.{name}", version=1,
            description=f"dry-run stand-in for calib.{name}",
            sources=tuple(CalibrationSource("dryrun", "local", d, 2) for d in DOMAINS),
            domain_weights={d: 1.0 / len(DOMAINS) for d in DOMAINS},
            token_budget=1, sample_rule="fixed order", seed=1000 + i)
        for i, name in enumerate(names))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/autoinit/dryrun")
    parser.add_argument("--beam-width", type=int, default=6)
    parser.add_argument("--warmup-levels", type=int, default=1)
    parser.add_argument("--profiles", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--items", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--searched-leaves", type=int, default=5)
    parser.add_argument("--resume", action="store_true",
                        help="keep the previous journal instead of starting clean")
    args = parser.parse_args()

    out = REPO_ROOT / args.out
    if out.exists() and not args.resume:
        shutil.rmtree(out)
    if args.resume and not (out / "search" / "states.jsonl").is_file():
        raise SystemExit(
            f"--resume needs a journal at {out / 'search' / 'states.jsonl'}; run the "
            "fresh pass first")
    adapter = get_adapter("qwen3")
    teacher = build_teacher(args.seed)
    target_spec = ArchSpec.of("qwen3", TARGET_GEOMETRY)

    suite = StateEvalSuite(
        suite_id="dryrun.state_eval", version=1, domains=tuple(DOMAINS),
        subtypes=DOMAINS, critical_tags=("eos_like", "answer_like"),
        description="synthetic held-out suite; structure mirrors the real one")
    eval_items = [SuiteItem(item_id=i["item_id"], input_ids=i["input_ids"],
                            domain=i["domain"], subtype=i["subtype"], tags=i["tags"])
                  for i in build_items(args.seed + 1, args.items, args.seq_len)]
    evaluator = StateEvaluator(suite, eval_items)
    evaluator.prime_reference(teacher)
    calibration = build_items(args.seed + 2, args.items, args.seq_len)

    schedule = BeamSchedule(
        schedule_id="dryrun.delayed_prune", version=1,
        description="mirrors SCHEDULE_V1: no pruning at level 0, then a fixed width",
        warmup_levels=args.warmup_levels, width=args.beam_width)
    config = SearchConfig(
        run_id="autoinit_dryrun", target_spec=target_spec,
        schedule=schedule, seed=args.seed, workdir=out / "search",
        profiles=make_profiles(args.profiles), policy=PARETO_V1, suite=suite,
        notes={"purpose": "zero-cost end-to-end validation of the mandatory cycle"})

    search = BeamSearch(
        adapter=adapter, config=config, root_teacher_id="dryrun-tiny-teacher",
        root_teacher_sha256="0" * 64, root_loader=lambda: teacher,
        calibration_loader=lambda profile: calibration,
        measurer=lambda model, sha: evaluator.evaluate(model, sha))
    result = search.run()

    leaves = result.complete_leaves

    # The canonical control is *injected*, not regenerated: a composite re-executed
    # inside the search is built from this run's calibration statistics, while the
    # retained incumbent was built from the original Stage-0 statistics. Same
    # algorithm, different input, different weights. Here the "retained" checkpoint
    # is a stand-in built once outside the search, which is the same relationship.
    control_dir = out / "canonical_control"
    control_model = adapter.build_model(
        adapter.build_config(teacher.config, target_spec), torch.float32, 4242)
    adapter.save(control_model, str(control_dir))
    control_artifact = identify_checkpoint(
        control_dir, adapter=adapter, spec=target_spec,
        num_parameters=adapter.param_count(target_spec))
    control = make_control_state(
        control_id="dryrun_canonical", artifact=control_artifact, spec=target_spec,
        target_spec=target_spec, num_parameters=adapter.param_count(target_spec),
        root_teacher_id="dryrun-tiny-teacher", root_teacher_sha256="0" * 64,
        description="stand-in for artifacts/stage1/qwen3_0p6b_init_v0/checkpoint",
        expected_single_file_sha256=control_artifact.single_shard_sha256)
    control.attach_evaluation(
        evaluator.evaluate(adapter.load(str(control_dir)),
                           control_artifact.artifact_digest))

    searched = min(args.searched_leaves, max(2, len(leaves)))
    top_n = result.top_n(PARETO_V1, min(searched, len(leaves)))

    plan = SuccessiveHalvingPlan(
        plan_id="autoinit.dryrun", recipe=E1_KD_HEAVY_0860K,
        searched_leaves=min(searched, len(leaves)),
        survivors=max(1, min(searched, len(leaves)) - 1),
        feasibility_min=0.0,
        survivor_rule="dry run only; the paid plan is preregistered separately",
        winner_rule="dry run only; the paid plan is preregistered separately",
        battery_asset_id="recovery.search_battery")
    admitted = admit_leaves([*top_n.selected, control], plan)
    probes = probe_configs(admitted, plan, rung=1)

    manifest = build_manifest(
        result, adapter=adapter, profiles=list(config.profiles), policy=PARETO_V1,
        teacher={"model_id": "dryrun-tiny-teacher", "sha256": "0" * 64,
                 "geometry": TEACHER_GEOMETRY,
                 "num_parameters": adapter.param_count(adapter.spec_of(teacher))},
        control={"state_id": control.state_id,
                 "provenance": control.provenance,
                 "artifact_digest": control.artifact_digest,
                 "single_shard_sha256": control.checkpoint_sha256,
                 "note": ("injected by artifact; a re-executed composite is not the "
                          "historical incumbent")},
        top_n=top_n,
        recovery_config={"plan": plan.as_dict(), "probes": probes},
        cost={"usd": 0.0, "hardware": "CPU dev box", "note": "zero-cost dry run"},
        environment={"code_state": code_state(str(REPO_ROOT)),
                     "hardware": hardware_report()},
        notes={"scope": "validates the pipeline, not the science: a 32-wide teacher "
                        "says nothing about which initialization path is better"})
    report = verify_manifest(manifest)
    write_manifest(out / "search_manifest.json", manifest)
    plan.freeze(out / "recovery_plan.json")

    # `artifacts/` is gitignored, and the full manifest is 240 KB of per-state
    # detail that regenerates at $0. A compact summary goes to `logs/` so the
    # repository carries evidence the pipeline ran without carrying its bulk.
    summary = {
        "schema": "aadistill.autoinit.dryrun_summary/v1",
        "generated_utc": manifest["generated_utc"],
        "purpose": ("zero-cost end-to-end validation of the mandatory "
                    "materialize -> reload -> hash -> validate -> measure cycle"),
        "scope_limit": ("a 32-wide teacher validates the pipeline, not the science; "
                        "no initialization-quality claim follows from it"),
        "command": "PYTHONPATH=src python scripts/autoinit/dry_run_search.py",
        "config_hash": result.config.config_hash,
        "manifest_hash": manifest["manifest_hash"],
        "manifest_verified": report,
        "teacher_geometry": TEACHER_GEOMETRY,
        "target_geometry": TARGET_GEOMETRY,
        "summary": result.summary(),
        "operator_registry_ledger_hashes": {
            k: v["signature_hash"]
            for k, v in manifest["operator_registry"]["implementations"].items()},
        "beam_ranking_policy_hash": PARETO_V1.policy_hash,
        "beam_schedule": schedule.as_dict(),
        "stats_cache": search.stats_cache.report(),
        "control": manifest["recovery_control"],
        "leaves": manifest["leaf_set"],
        "pruned": [
            {"state_id": s["state_id"], "path": s["path_label"],
             "reason": s["prune_reason"], "checkpoint_sha256": s["checkpoint_sha256"]}
            for s in manifest["states"] if s["validity"] == "pruned"],
        "full_manifest_path": str((out / "search_manifest.json").relative_to(REPO_ROOT)),
    }
    # Fresh and resume evidence are separate artifacts. Combining them produced a
    # summary that described itself as a fresh run while carrying n_resumed=24,
    # which is the sort of record that is worse than none.
    summary["run_kind"] = "resume" if args.resume else "fresh"
    summary["resume_evidence"] = {
        "n_resumed": len(result.resumed),
        "resumed_state_ids": sorted(result.resumed),
        "expected_resumed": ("0 for a fresh run; a fresh run starts from an empty "
                             "artifact directory"),
    }
    if not args.resume and result.resumed:
        raise SystemExit(
            f"a fresh run resumed {len(result.resumed)} states; the artifact "
            "directory was not empty and this summary would misdescribe itself")
    name = "autoinit_dryrun_resume.json" if args.resume else "autoinit_dryrun_fresh.json"
    summary_path = REPO_ROOT / "logs" / name
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")

    print(f"states {len(result.states)}  leaves {len(leaves)}  "
          f"pruned {len(manifest['state_index']['pruned'])}  "
          f"stats-cache {search.stats_cache.report()}")
    print(f"manifest verified: {report}")
    print("\ncomplete leaves, ranked:")
    for rank, state in enumerate(top_n.selected, 1):
        values = state.evaluation.values
        print(f"  {rank}. {state.path_label}")
        print(f"     impls  {' -> '.join(state.impl_ids)}")
        print(f"     artifact {state.artifact_digest[:16]}  params {state.num_parameters:,}"
              f"  {checkpoint_bytes(state.spec, adapter) / 2**20:.2f} MiB"
              f"  shards {len(state.artifact.shards)}")
        print(f"     teacher_kl {values['state.teacher_kl.equal_domain_mean']:.4f}"
              f"  critical {values.get('state.critical_token_kl', float('nan')):.4f}"
              f"  nll {values['state.nll.general']:.4f}")
    print(f"\npruned with reasons: {len(top_n.pruned)} at the leaf stage; "
          f"{len(manifest['state_index']['pruned'])} during the search")
    print(f"wrote {(out / 'search_manifest.json').relative_to(REPO_ROOT)}")
    print(json.dumps(result.summary(), indent=2))


if __name__ == "__main__":
    main()
