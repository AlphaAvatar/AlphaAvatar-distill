#!/usr/bin/env python3
"""The packing-optimization pilot, as one owned sequence.

    python scripts/pod/c3_packing_pilot_driver.py --out /workspace/out/packing

    1. obtain the verified pre-ATTENTION parent
         reuse it if durable bytes exist; otherwise replay the prefix ONCE at B1
    2. verify eea90c91...
    3. short three-layer throughput screen over P0/P1/P2/P3
    4. apply the screen gate (>= 1.10x over the same-session P0)
    5. apply the +/-5% B1 comparability rule against the prior full B1
    6. if a candidate advances, run ONE full 28-layer scorer for it
    7. apply the 1.25x adoption gate and compare head maps

Every stage persists as it completes. Selection at step 4 is by measured wall
time alone; no head-map or quality result chooses which packing advances.

`--toy` runs the identical sequence at a CPU geometry — same pilot functions,
same operator, same screen, same comparator, same records, nothing stubbed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

PILOT_DIR = "logs/stages/stage-1/phase_c3/pilots/packing-optimization/v1"


class PilotError(RuntimeError):
    """The pilot cannot proceed. Never read as a scientific result."""


def _say(msg: str) -> None:
    print(f"[packing {time.strftime('%H:%M:%S', time.gmtime())}] {msg}",
          flush=True)


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True,
                               ensure_ascii=False) + "\n")


def load_scope(repo: Path) -> dict:
    return json.loads((repo / PILOT_DIR / "scope.json").read_text())


def run(out_dir: Path, *, repo: Path, toy: bool, device: str,
        parent_dir: str | None = None, deadline=None) -> dict:
    import torch

    from aadistill.initialization.adapters import register_builtin_adapters
    from aadistill.initialization.adapters.qwen3 import QWEN3_ADAPTER
    from aadistill.initialization.operators.attention.gqa import causal_kl
    from aadistill.initialization.operators.register import (
        register_builtin_operators)
    from aadistill.initialization.planning.fixed_path import FixedPathSpec
    from aadistill.initialization.specs.arch import ArchSpec

    from experiments.calibration import register_builtin_profiles
    from experiments.phase_c3 import pilot
    from experiments.phase_c3.compare import compare_head_maps, speedup

    #: Three process-global registries, all empty in a fresh interpreter.
    register_builtin_adapters()
    register_builtin_operators()
    register_builtin_profiles()
    causal_kl.register(replace=True)

    scope = load_scope(repo)
    screen_cfg = scope["screen"]
    full_gate = float(scope["full_scorer_gate"]["threshold"])
    protocols = {p["protocol"]: p for p in scope["protocols"]}

    record: dict = {
        "schema": "aadistill.phase_c3.packing_pilot_result/v1",
        "toy": toy, "device": device,
        "screen": {"layers": screen_cfg["layers"],
                   "protocol_order": screen_cfg["protocol_order"],
                   "advance_threshold": screen_cfg["advance_threshold"]},
        "full_scorer_gate": full_gate,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stages": {},
    }
    result_path = out_dir / "packing_result.json"
    _write(result_path, record)

    def checkpoint(stage: str, payload) -> None:
        record["stages"][stage] = payload
        _write(result_path, record)

    # --- 1-2. the parent ----------------------------------------------------
    if toy:
        from c3_batching_pilot_driver import (          # noqa: F401
            TOY_PARENT, TOY_TARGET, _toy_items, _toy_root)

        items = _toy_items()
        calibration = {"calib.domain_balanced@v1": items,
                       "calib.reasoning_heavy@v2": items}

        def toy_spec(digest):
            steps = list(pilot.prefix_steps(pin_parent=False))
            last = steps[-1]
            steps[-1] = type(last)(impl_id=last.impl_id,
                                   profile_id=last.profile_id,
                                   expected_artifact_digest=digest,
                                   label="pre-ATTENTION parent")
            return FixedPathSpec(
                path_id="toy.packing-prefix", family="qwen3",
                target_spec=ArchSpec.of("qwen3", TOY_TARGET),
                steps=tuple(steps), root_repo_id="toy/root",
                root_revision="toy", device=device, seed=0)

        _say("toy: discovery pass to learn this machine's parent digest")
        found: list = []
        pilot.replay_prefix(toy_spec(None), adapter=QWEN3_ADAPTER,
                            root_loader=_toy_root,
                            workdir=out_dir / "discover", repo_root=repo,
                            calibration_items=calibration,
                            on_step=found.append)
        expected_digest = found[-1].identity.artifact_digest
        spec = toy_spec(expected_digest)
        weight_dtype = None
        root_loader = _toy_root
    else:
        spec = pilot.prefix_spec(repo_root=repo, device=device)
        calibration = None
        expected_digest = pilot.FROZEN_PARENT_DIGEST
        weight_dtype = torch.bfloat16

        def root_loader():
            from transformers import AutoModelForCausalLM

            return AutoModelForCausalLM.from_pretrained(
                spec.root_repo_id, revision=spec.root_revision,
                dtype=torch.bfloat16).to(device).eval()

    if parent_dir:
        #: REUSED. §17: do not replay 22 minutes to rebuild bytes that exist.
        #: The digest is still checked, from the FILES, before anything runs.
        from aadistill.initialization.specs.artifact import identify_checkpoint

        _say(f"stage 1/7: re-identifying a supplied parent at {parent_dir}")
        from transformers import AutoModelForCausalLM

        probe = AutoModelForCausalLM.from_pretrained(parent_dir,
                                                     dtype=weight_dtype)
        pspec = QWEN3_ADAPTER.spec_of(probe)
        ident = identify_checkpoint(
            parent_dir, adapter=QWEN3_ADAPTER, spec=pspec,
            num_parameters=QWEN3_ADAPTER.param_count(pspec))
        del probe
        if ident.artifact_digest != expected_digest:
            raise PilotError(
                f"the supplied parent identifies to {ident.artifact_digest}, "
                f"not {expected_digest}; refusing to score the wrong model")
        parent_path, prefix_seconds, replays = parent_dir, 0.0, 0
        parent_weights = ident.weights_digest
        checkpoint("parent", {"source": "reused", "replay_count": 0,
                              "path": parent_dir,
                              "artifact_digest": ident.artifact_digest,
                              "weights_digest": parent_weights})
        _say(f"  reused, verified {ident.artifact_digest[:16]}…")
    else:
        _say("stage 1/7: no durable parent supplied; replaying the prefix "
             "ONCE at B=1")
        t0 = time.monotonic()
        prefix: list = []

        def on_step(step):
            prefix.append(step)
            checkpoint("prefix_steps", [s.as_dict() for s in prefix])
            _say(f"  step {step.index} {step.kind} in {step.seconds:.1f}s, "
                 f"{step.identity.artifact_digest[:16]}…")

        pilot.replay_prefix(spec, adapter=QWEN3_ADAPTER,
                            root_loader=root_loader,
                            workdir=out_dir / "work", repo_root=repo,
                            calibration_items=calibration, on_step=on_step,
                            deadline=deadline)
        prefix_seconds = time.monotonic() - t0
        parent = prefix[-1]
        if parent.identity.artifact_digest != expected_digest:
            raise PilotError(
                f"the replayed parent is {parent.identity.artifact_digest}, "
                f"not {expected_digest}")
        parent_path = parent.checkpoint_path
        parent_weights = parent.identity.weights_digest
        replays = 1
        checkpoint("parent", {"source": "replayed", "replay_count": 1,
                              "wall_seconds": round(prefix_seconds, 3),
                              "path": parent_path,
                              "artifact_digest": expected_digest,
                              "weights_digest": parent_weights})
    _say(f"stage 2/7: parent verified {expected_digest[:16]}… "
         f"(replays this session: {replays})")

    # --- 3. the screen ------------------------------------------------------
    _say("stage 3/7: three-layer throughput screen")
    from c3_packing_screen import run as run_screen

    #: A toy run has no frozen mixture and a 3-layer geometry, so it supplies
    #: its own items and its own layers. THE REAL PATH PASSES NEITHER — the
    #: screen resolves the mixture and reads the layers from the record, and
    #: a $0 test asserts that this call site overrides nothing when `toy` is
    #: false.
    screen_kwargs = {"items": items, "layers": [0, 1]} if toy else {}
    screen = run_screen(parent_path, out_dir / "screen", repo=repo,
                        device=device, deadline=deadline, **screen_kwargs)
    checkpoint("screen", screen)

    # --- 4. the screen gate -------------------------------------------------
    gate = screen["screen_gate"]
    checkpoint("screen_gate", gate)
    if not gate["advances"]:
        record["verdict"] = "NO_BATCH_PACKING_CANDIDATE_WORTH_FULL_SCORER"
        record["recovery_triggered"] = False
        record["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                               time.gmtime())
        _write(result_path, record)
        _say(f"VERDICT {record['verdict']} — best "
             f"{gate['best']['protocol'] if gate['best'] else None} at "
             f"{gate['best']['screen_speedup'] if gate['best'] else 0}x")
        return record

    winner = gate["best"]["protocol"]
    _say(f"stage 4/7: {winner} advances at {gate['best']['screen_speedup']}x")

    # --- 5. the +/-5% B1 comparability rule --------------------------------
    comp = scope["b1_comparability"]
    prior = float(comp["prior_full_b1_seconds"])
    ref_name = screen_cfg["reference_protocol"]
    ref = screen["results"][ref_name]
    #: The prior full B1 scored 28 layers; this screen scored len(layers).
    #: Scale the prior to the screen's work to compare like with like.
    n_layers_full = int(scope.get("_full_layers", 28))
    scale = len(screen_cfg["layers"]) / n_layers_full
    expected = prior * scale
    ratio = ref["wall_seconds"] / expected
    comparable = abs(ratio - 1.0) <= float(comp["tolerance"])
    checkpoint("b1_comparability", {
        "prior_full_b1_seconds": prior, "screen_layers": screen_cfg["layers"],
        "full_layers": n_layers_full, "scale": scale,
        "expected_screen_seconds": round(expected, 3),
        "observed_screen_seconds": ref["wall_seconds"],
        "ratio": round(ratio, 4), "tolerance": comp["tolerance"],
        "comparable": comparable,
        "_meaning": ("if comparable, the prior full B1 may serve as the "
                     "full-run reference; if not, a new full B1 is required "
                     "before any full-scorer speed claim"),
        "_predeclared": True,
    })
    _say(f"stage 5/7: same-session P0 is {ratio:.4f}x the scaled prior B1 "
         f"-> {'comparable' if comparable else 'NOT comparable'}")

    # --- 6. one full scorer -------------------------------------------------
    todo = [winner] if comparable else [ref_name, winner]
    _say(f"stage 6/7: full scorer for {todo}")
    full: dict = {}
    for name in todo:
        spec_p = protocols[name]
        full[name] = _full_scorer(
            parent_path, spec_p, out_dir, repo=repo, device=device,
            dtype=weight_dtype, expected_digest=expected_digest,
            toy=toy, toy_target=(TOY_TARGET if toy else None),
            deadline=deadline)
        checkpoint(f"full_{name}", full[name])
        _say(f"  {name}: scorer {full[name]['scorer_seconds']:.1f}s")

    reference_seconds = (prior if comparable
                         else full[ref_name]["scorer_seconds"])
    sp = speedup(reference_seconds, full[winner]["scorer_seconds"])
    sp["reference_source"] = ("prior full B1 (comparable)" if comparable
                              else "this session's full B1")
    checkpoint("speed", sp)

    if sp["speedup"] < full_gate:
        record["verdict"] = "PACKED_BATCH_NOT_WORTH_ADOPTION"
        record["recovery_triggered"] = False
    else:
        b1_ev = _reference_landscape(repo, scope, full.get(ref_name))
        structural = compare_head_maps(
            b1_ev, full[winner]["causal_head_evidence"])
        checkpoint("structural", structural)
        record["verdict"] = (
            "PACKED_BATCH_STRUCTURALLY_EQUIVALENT_AND_FASTER"
            if structural["identical"]
            else "PACKED_BATCH_FASTER_AND_STRUCTURALLY_DIFFERENT_RECOVERY_TRIGGERED")
        record["recovery_triggered"] = not structural["identical"]
    record["winner"] = winner
    record["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write(result_path, record)
    _say(f"VERDICT {record['verdict']} (speedup {sp['speedup']}x)")
    return record


def _full_scorer(parent_path, protocol, out_dir, *, repo, device, dtype,
                 expected_digest, toy, toy_target, deadline):
    """One protocol's complete 28-layer scorer, through the real operator."""
    from aadistill.initialization.adapters.qwen3 import QWEN3_ADAPTER
    from aadistill.initialization.planning.fixed_path import FixedPathSpec
    from aadistill.initialization.specs.arch import ArchSpec
    from transformers import AutoModelForCausalLM

    from experiments.phase_c3 import pilot

    bs = protocol["calibration_forward_batch_size"]
    packing = protocol["calibration_batch_packing"]
    step = pilot.causal_step(bs, packing=packing)
    if toy:
        steps = tuple(pilot.prefix_steps(pin_parent=False)) + (step,)
        arm = FixedPathSpec(
            path_id=f"toy.packing-{protocol['protocol']}", family="qwen3",
            target_spec=ArchSpec.of("qwen3", toy_target), steps=steps,
            root_repo_id="toy/root", root_revision="toy", device=device,
            seed=0)
    else:
        arm = pilot.arm_spec(bs, repo_root=repo, device=device,
                             packing=packing)

    from aadistill.initialization.calibration.profiles import get_profile
    from aadistill.initialization.operators import get_implementation
    from aadistill.initialization.operators.base import OperatorContext

    profile = get_profile(step.profile_id)
    items = (pilot_toy_items() if toy else profile.resolve(repo))
    model = AutoModelForCausalLM.from_pretrained(
        parent_path, dtype=dtype).to(device).eval()
    parent_spec = QWEN3_ADAPTER.spec_of(model)
    target_spec = arm.target_spec
    ctx = OperatorContext(
        adapter=QWEN3_ADAPTER, model=model, parent_spec=parent_spec,
        target_spec=target_spec, profile=profile, calibration_items=items,
        seed=arm.seed, device=device, deadline=deadline,
        config={"n_calibration_items": len(items),
                "calibration_forward_batch_size": bs,
                "calibration_batch_packing": packing},
        workdir=out_dir / f"full_{protocol['protocol']}")
    out = get_implementation(pilot.CAUSAL_IMPL_ID).execute(ctx)
    del model
    return {
        "protocol": protocol["protocol"],
        "calibration_forward_batch_size": bs,
        "calibration_batch_packing": packing,
        "path_id": arm.path_id, "path_hash": arm.spec_hash,
        "scorer_seconds": out.trace["scorer_seconds"],
        "physical_forward_invocations":
            out.trace["physical_forward_invocations"],
        "padded_positions": out.trace["padded_positions"],
        "kept_heads": out.artifacts["kept_heads"],
        "causal_head_evidence": out.artifacts["causal_head_evidence"],
    }


def pilot_toy_items():
    from c3_batching_pilot_driver import _toy_items

    return _toy_items()


def _reference_landscape(repo: Path, scope: dict, fresh):
    """The B1 landscape to compare against: the committed one if it verifies.

    §13: prefer the previous B1 raw evidence when its committed sha256 still
    matches. If it cannot be verified, a new B1 full scorer is used rather
    than an approximation — which is why `fresh` is the fallback and not a
    reconstruction.
    """
    import hashlib

    if fresh is not None:
        return fresh["causal_head_evidence"]
    ref = scope["structural_comparison"]["reuse_if_verifiable"]
    path = repo / ref["file"]
    if not path.is_file():
        raise PilotError(
            f"the prior B1 evidence {ref['file']} is absent and no fresh B1 "
            "was run; refusing to approximate a reference landscape")
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != ref["sha256"]:
        raise PilotError(
            f"the prior B1 evidence hashes {got}, the record pins "
            f"{ref['sha256']}; refusing to compare against unverified bytes")
    doc = json.loads(path.read_text())
    for name, stage in doc.get("stages", {}).items():
        if not isinstance(stage, dict) or "step" not in stage:
            continue
        if stage.get("calibration_forward_batch_size") == 1:
            return stage["step"]["selection"]["causal_head_evidence"]
    raise PilotError("the prior evidence holds no B1 arm")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="")
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--parent", default=None,
                    help="a durable verified parent to reuse instead of "
                         "replaying the prefix")
    ap.add_argument("--device", default=None)
    ap.add_argument("--toy", action="store_true")
    ap.add_argument("--required-inputs", action="store_true",
                    help="print the mixtures the real path needs and exit")
    ap.add_argument("--check-inputs", action="store_true",
                    help="resolve every required mixture for real and exit")
    args = ap.parse_args(argv)

    #: Both delegate to the pilot module: the answer is a fact about the
    #: pilot's steps, and two copies is one place for it to be wrong.
    if args.required_inputs or args.check_inputs:
        from experiments.phase_c3 import pilot

        if args.required_inputs:
            for entry in pilot.required_profiles(Path(args.repo)):
                print(json.dumps(entry, sort_keys=True))
            return 0
        return pilot.check_inputs(Path(args.repo), say=_say)

    device = args.device
    if device is None:
        try:
            import torch

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    if not args.out:
        ap.error("--out is required unless --required-inputs/--check-inputs")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        run(out, repo=Path(args.repo), toy=args.toy, device=device,
            parent_dir=args.parent)
    except Exception as exc:      # noqa: BLE001 - a paid pod has died in one
        _write(out / "packing_failure.json", {
            "error": f"{type(exc).__name__}: {exc}",
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        _say(f"FAILED {type(exc).__name__}: {exc}")
        import traceback

        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "7")
    raise SystemExit(main())
