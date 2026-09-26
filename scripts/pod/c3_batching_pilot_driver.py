#!/usr/bin/env python3
"""The C3 batching-adoption pilot, as one owned sequence.

    python scripts/pod/c3_batching_pilot_driver.py --out /workspace/out/pilot

    1. replay the frozen pre-ATTENTION prefix ONCE, explicitly at B=1
    2. verify the parent is eea90c91...
    3. secure that one parent as the single source for both arms
    4. score causal-B1 from it
    5. score causal-B4 from it
    6. persist both score fields and the head-map evidence
    7. compute the measured speedup and the structural delta

The order of 4 and 5 is read from the pilot record, which froze it before
either result existed. Whichever arm runs first pays for allocator growth and
kernel autotuning the second does not, so the order is part of the protocol.

**Every stage writes its evidence as it completes.** A pilot that dies in
stage 5 still measured stage 4, and a collector that runs only on a clean
terminal state has already deleted verified work in this project.

`--toy` runs the identical sequence at a CPU geometry against a synthetic
root, so the whole path executes at $0 before it executes at $1.09/h. It is
not a rehearsal that stubs anything: the same three pilot functions, the same
operator, the same comparator, the same records.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

PILOT_DIR = "logs/stages/stage-1/phase_c3/pilots/batching-adoption/v1"


class PilotError(RuntimeError):
    """The pilot cannot proceed. Never read as a scientific result."""


def _say(msg: str) -> None:
    print(f"[pilot {time.strftime('%H:%M:%S', time.gmtime())}] {msg}", flush=True)


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True,
                               ensure_ascii=False) + "\n")


def load_scope(repo: Path) -> dict:
    return json.loads((repo / PILOT_DIR / "scope.json").read_text())


def required_profiles(repo: Path) -> list:
    """Every calibration mixture the REAL pilot path resolves, derived.

    DERIVED, not listed. Attempt a1 staged one mixture and died 24 minutes in
    because the prefix uses two — DEPTH and FFN on `domain_balanced@v1`,
    WIDTH on `reasoning_heavy@v2` — which is the exact non-uniformity a
    review had already corrected in the pilot module. A hand-written list is
    a second place for that fact to be wrong; this asks the steps.

    Each entry carries the profile's OWN `items_path` and `items_file_sha256`,
    so a stager verifies against the profile rather than against a constant
    copied beside it.
    """
    from aadistill.initialization.calibration.profiles import get_profile

    from experiments.calibration import register_builtin_profiles
    from experiments.phase_c3 import pilot

    register_builtin_profiles()
    seen, out = set(), []
    steps = list(pilot.prefix_steps()) + [pilot.causal_step(1)]
    for step in steps:
        if step.profile_id in seen:
            continue
        seen.add(step.profile_id)
        profile = get_profile(step.profile_id)
        out.append({"profile_id": step.profile_id,
                    "items_path": profile.items_path,
                    "items_file_sha256": profile.items_file_sha256,
                    "materialized": bool(profile.materialized)})
    return out


def check_inputs(repo: Path) -> int:
    """Resolve every required mixture for real, and say what is missing.

    This is the check attempt a1 did not have. The pod-side preflight it DID
    have runs the driver's `--toy` mode, which supplies `calibration_items`
    explicitly and therefore never resolves a profile at all — so the one
    gate standing before 24 minutes of GPU work could not see the one input
    that was absent.
    """
    from aadistill.initialization.calibration.profiles import (
        CalibrationError, get_profile)

    missing = []
    for entry in required_profiles(repo):
        try:
            items = get_profile(entry["profile_id"]).resolve(repo)
        except (CalibrationError, OSError) as exc:
            missing.append(f"{entry['profile_id']}: {exc}")
            _say(f"  MISSING {entry['profile_id']} -> {entry['items_path']}")
            continue
        _say(f"  ok {entry['profile_id']}: {len(items)} items from "
             f"{entry['items_path']}")
    if missing:
        _say("INPUTS UNAVAILABLE; the pilot would fail after the GPU work:")
        for m in missing:
            _say(f"  {m}")
        return 30
    return 0


# --- toy mode: the same sequence, at a geometry a CPU can finish -----------

TOY_PARENT = dict(hidden_size=32, num_hidden_layers=4, intermediate_size=64,
                  num_attention_heads=8, num_key_value_heads=2, head_dim=8,
                  vocab_size=128, tie_word_embeddings=True)
TOY_TARGET = dict(TOY_PARENT, num_hidden_layers=3, intermediate_size=48,
                  hidden_size=16, num_attention_heads=4)


def _toy_items():
    import torch

    g = torch.Generator().manual_seed(5)
    out = []
    for subtype, lens in (("general", (11, 15)), ("alpha", (9, 13)),
                          ("beta", (14, 10))):
        for k, n in enumerate(lens):
            out.append({"item_id": f"{subtype}/{k}",
                        "domain": "general" if subtype == "general" else "task",
                        "subtype": subtype,
                        "input_ids": torch.randint(0, 128, (1, n), generator=g)})
    return out


def _toy_root():
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(7)
    model = Qwen3ForCausalLM(
        Qwen3Config(max_position_embeddings=256, rope_theta=5_000_000,
                    **TOY_PARENT)).float().eval()
    with torch.no_grad():
        for m in model.modules():
            if m.__class__.__name__ == "Qwen3RMSNorm":
                m.weight.uniform_(0.5, 1.5)
    return model


def _load_checkpoint(path: str):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(path).eval()


def run(out_dir: Path, *, repo: Path, toy: bool, device: str,
        deadline=None) -> dict:
    """The seven steps, each persisting as it completes."""
    from aadistill.initialization.adapters import register_builtin_adapters
    from aadistill.initialization.adapters.qwen3 import QWEN3_ADAPTER
    from aadistill.initialization.operators.attention.gqa import causal_kl
    from aadistill.initialization.operators.register import (
        register_builtin_operators)
    from aadistill.initialization.planning.fixed_path import FixedPathSpec
    from aadistill.initialization.specs.arch import ArchSpec

    from experiments.phase_c3 import pilot
    from experiments.phase_c3.compare import compare_head_maps, speedup, verdict

    #: THREE PROCESS-GLOBAL REGISTRIES, ALL EMPTY IN A FRESH INTERPRETER.
    #: A pytest session hides every one of them, because some sibling test has
    #: always filled them first; this driver is the first code in this project
    #: to need all three in one bare process. The toy run found the second and
    #: third, in that order, each after the stage before it had succeeded —
    #: on a pod that would have been after the root teacher was downloaded and
    #: resident on the GPU.
    register_builtin_adapters()
    register_builtin_operators()
    #: EXPLICIT. Importing the module registers nothing, by design, so a
    #: driver that forgot this line would be refused by `FixedPathSpec` before
    #: anything ran — which is the point, but it should not happen here.
    causal_kl.register(replace=True)
    #: AND the calibration profiles, for the same reason and a worse failure:
    #: the registry is process-global and EMPTY in a fresh interpreter, so a
    #: driver that never fills it dies at `get_profile` — after the root
    #: teacher has been downloaded and loaded onto the GPU. A pytest session
    #: hides this completely, because some sibling test has always filled the
    #: registry first. This line is here because the toy run found it.
    from experiments.calibration import register_builtin_profiles

    register_builtin_profiles()

    scope = load_scope(repo)
    order = scope["arm_order"]
    threshold = float(scope["speed_gate"]["threshold"])
    sizes = {a["arm_id"]: a["calibration_forward_batch_size"]
             for a in scope["arms"]}
    if sorted(order) != sorted(sizes):
        raise PilotError(f"the record's arm_order {order} does not name its "
                         f"own arms {sorted(sizes)}")
    _say(f"predeclared arm order {order}, gate {threshold}x")

    record: dict = {
        "schema": "aadistill.phase_c3.batching_pilot_result/v1",
        "toy": toy, "device": device,
        "arm_order": order, "speed_gate_threshold": threshold,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stages": {},
    }
    result_path = out_dir / "pilot_result.json"
    _write(result_path, record)

    def checkpoint(stage: str, payload) -> None:
        record["stages"][stage] = payload
        _write(result_path, record)

    # --- 1. the prefix, ONCE, explicitly at B=1 ----------------------------
    if toy:
        items = _toy_items()
        calibration = {"calib.domain_balanced@v1": items,
                       "calib.reasoning_heavy@v2": items}
        root_loader = _toy_root

        def toy_spec(digest):
            steps = list(pilot.prefix_steps(pin_parent=False))
            last = steps[-1]
            steps[-1] = type(last)(impl_id=last.impl_id,
                                   profile_id=last.profile_id,
                                   expected_artifact_digest=digest,
                                   label="pre-ATTENTION parent")
            return FixedPathSpec(
                path_id="toy.shared-prefix", family="qwen3",
                target_spec=ArchSpec.of("qwen3", TOY_TARGET),
                steps=tuple(steps), root_repo_id="toy/root",
                root_revision="toy", device=device, seed=0)

        #: TOY ONLY, and the reason is the guard working. The real pilot knows
        #: its parent digest in advance -- `eea90c91…` is frozen -- so its
        #: prefix step is PINNED and `materialize_fixed_path_suffix` can see
        #: that the parent was gated. A toy parent's digest is whatever this
        #: machine produces, so there is one throwaway discovery pass first
        #: and then the real, gated one. Without it the toy path would have to
        #: skip the gate, and the gate is the thing being rehearsed.
        _say("toy: discovery pass to learn this machine's parent digest")
        discovered: list = []
        pilot.replay_prefix(toy_spec(None), adapter=QWEN3_ADAPTER,
                            root_loader=root_loader,
                            workdir=out_dir / "discover", repo_root=repo,
                            calibration_items=calibration,
                            on_step=discovered.append)
        expected_digest = discovered[-1].identity.artifact_digest
        spec = toy_spec(expected_digest)
    else:
        spec = pilot.prefix_spec(repo_root=repo, device=device)
        calibration = None          # resolved from the frozen profiles
        expected_digest = pilot.FROZEN_PARENT_DIGEST

        def root_loader():
            from transformers import AutoModelForCausalLM
            import torch

            return AutoModelForCausalLM.from_pretrained(
                spec.root_repo_id, revision=spec.root_revision,
                dtype=torch.bfloat16).to(device).eval()

    _say("stage 1/7: replaying the frozen prefix ONCE at B=1")
    t0 = time.monotonic()
    prefix: list = []

    def prefix_step_done(step) -> None:
        """PERSIST AS EACH STEP LANDS, not when all three have.

        Attempt a1 ran DEPTH and FFN for 24 minutes and recorded neither,
        because the prefix stage was written only after `replay_prefix`
        returned and the third step raised. A completed unit of work must
        survive the failure of a later one.
        """
        prefix.append(step)
        checkpoint("prefix_steps", [s.as_dict() for s in prefix])
        _say(f"  step {step.index} {step.kind} done in {step.seconds:.1f}s, "
             f"digest {step.identity.artifact_digest[:16]}…")

    pilot.replay_prefix(spec, adapter=QWEN3_ADAPTER, root_loader=root_loader,
                        workdir=out_dir / "work", repo_root=repo,
                        calibration_items=calibration,
                        on_step=prefix_step_done, deadline=deadline)
    prefix_seconds = time.monotonic() - t0
    if not prefix:
        raise PilotError("the prefix produced no steps")
    parent = prefix[-1]
    checkpoint("prefix", {
        "replay_count": 1,
        "_why_one": scope["_why_one"],
        "wall_seconds": round(prefix_seconds, 3),
        "execution": {"micro_batch_size":
                      pilot.PREFIX_EXECUTION.micro_batch_size},
        "steps": [s.as_dict() for s in prefix],
        "parent_artifact_digest": parent.identity.artifact_digest,
        "parent_weights_digest": parent.identity.weights_digest,
        "parent_checkpoint_path": parent.checkpoint_path,
    })
    _say(f"  prefix done in {prefix_seconds / 60:.1f} min, parent "
         f"{parent.identity.artifact_digest[:16]}…")

    # --- 2. verify -----------------------------------------------------------
    if parent.identity.artifact_digest != expected_digest:
        raise PilotError(
            f"the replayed parent is {parent.identity.artifact_digest} but the "
            f"pilot requires {expected_digest}; both arms would otherwise be "
            "scored against the wrong checkpoint")
    _say(f"stage 2/7: parent verified against {expected_digest[:16]}…")

    # --- 3-5. both arms, from that ONE parent ------------------------------
    arms: dict[str, dict] = {}
    for position, arm_id in enumerate(order, start=1):
        batch_size = sizes[arm_id]
        _say(f"stage {2 + position}/7: {arm_id} (B{batch_size}), position "
             f"{position} of {len(order)}")
        if toy:
            arm_spec = FixedPathSpec(
                path_id=f"toy.{arm_id}", family="qwen3",
                target_spec=ArchSpec.of("qwen3", TOY_TARGET),
                steps=tuple(spec.steps) + (pilot.causal_step(batch_size),),
                root_repo_id="toy/root", root_revision="toy",
                device=device, seed=0)
        else:
            arm_spec = pilot.arm_spec(batch_size, repo_root=repo, device=device)

        #: RELOADED per arm, from the same verified checkpoint. Not reused in
        #: memory: the first arm's allocator state, cached kernels and any
        #: fragmentation would otherwise be part of the second arm's
        #: conditions, which is a difference between the POSITIONS and not
        #: between B1 and B4.
        def parent_loader(path=parent.checkpoint_path):
            return _load_checkpoint(path)

        #: RESET, or the second arm inherits the first arm's peak and the
        #: two numbers stop being comparable -- `max_memory_allocated` is a
        #: high-water mark for the whole process, not for this arm.
        _reset_peak_vram(device)
        verified = pilot.verified_parent(prefix, arm_spec,
                                         expected_digest=expected_digest)
        results: list = []
        t0 = time.monotonic()
        _, evidence = pilot.run_arm(
            arm_spec, adapter=QWEN3_ADAPTER, parent_loader=parent_loader,
            workdir=out_dir / "work", verified=verified, repo_root=repo,
            calibration_items=calibration, on_step=results.append,
            deadline=deadline)
        arm_seconds = time.monotonic() - t0
        step = results[-1]
        causal = step.selection.get("causal_head_evidence")
        if causal is None:
            raise PilotError(
                f"{arm_id} produced no causal_head_evidence; the scientific "
                "evidence did not survive the executor")
        arms[arm_id] = {
            "arm_id": arm_id, "position": position,
            "calibration_forward_batch_size": batch_size,
            "path_id": arm_spec.path_id, "path_hash": arm_spec.spec_hash,
            "step": step.as_dict(),
            "suffix_evidence": evidence,
            #: THE GATE'S STATISTIC is the operator's own synchronized scorer
            #: clock, not this wall time, which also covers loading the
            #: parent and writing the child.
            "scorer_seconds": step.trace["scorer_seconds"],
            "arm_wall_seconds": round(arm_seconds, 3),
            "physical_forward_invocations":
                step.trace["physical_forward_invocations"],
            "item_forward_equivalents": step.trace["item_forward_equivalents"],
            "padded_positions": step.trace["padded_positions"],
            "peak_vram_bytes": _peak_vram(device),
            "child_artifact_digest": step.identity.artifact_digest,
            "child_weights_digest": step.identity.weights_digest,
            "child_checkpoint_path": step.checkpoint_path,
        }
        checkpoint(arm_id, arms[arm_id])
        _say(f"  {arm_id}: scorer {arms[arm_id]['scorer_seconds']:.1f}s, "
             f"{arms[arm_id]['physical_forward_invocations']} invocations")

    # --- 6-7. the measured comparison --------------------------------------
    b1, b4 = order[0], order[-1]
    ev = {a: arms[a]["step"]["selection"]["causal_head_evidence"]
          for a in (b1, b4)}
    by_size = {arms[a]["calibration_forward_batch_size"]: a for a in arms}
    sp = speedup(arms[by_size[1]]["scorer_seconds"],
                 arms[by_size[4]]["scorer_seconds"])
    structural = compare_head_maps(ev[by_size[1]], ev[by_size[4]])
    v = verdict(sp["speedup"], structural["identical"], threshold=threshold)
    checkpoint("comparison", {"speed": sp, "structural": structural,
                              "verdict": v})
    record["verdict"] = v
    record["recovery_triggered"] = v.endswith("RECOVERY_TRIGGERED")
    record["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write(result_path, record)

    #: TWO FILES, because they have different homes. `pilot_result.json`
    #: carries the raw per-item landscape -- 28 x 32 x 67 values at the real
    #: geometry, several MB -- and belongs with the run's artifacts. The
    #: SUMMARY is what a committed log record can hold: every decision, every
    #: identity, every distribution, and the sha256 of the file the raw values
    #: are in, so dropping them from git does not make them unverifiable.
    _write(out_dir / "pilot_summary.json",
           summarize(record, raw_sha256=_sha256(result_path)))
    _say(f"wrote pilot_result.json ({result_path.stat().st_size} B) and "
         f"pilot_summary.json")
    _say(f"VERDICT {v} (speedup {sp['speedup']}x, identical "
         f"{structural['identical']})")
    return record


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(record: Mapping[str, Any], *, raw_sha256: str) -> dict:
    """Everything but the 60k raw values, plus the hash of the file holding them.

    `per_item_kl` is the one field that cannot go in a committed record. It is
    also the one that must not be lost, so the summary names the artifact and
    its digest rather than quietly omitting it.
    """
    out = {k: v for k, v in record.items() if k != "stages"}
    out["raw_evidence"] = {"file": "pilot_result.json", "sha256": raw_sha256,
                           "holds": "per_item_kl [layer][head][item]"}
    stages = {}
    for name, payload in record.get("stages", {}).items():
        if not isinstance(payload, dict) or "step" not in payload:
            stages[name] = payload
            continue
        arm = dict(payload)
        step = dict(arm["step"])
        selection = dict(step.get("selection") or {})
        ev = dict(selection.get("causal_head_evidence") or {})
        n_items = len(ev.get("item_ids") or ())
        ev.pop("per_item_kl", None)
        ev["_per_item_kl"] = (f"omitted here; {n_items} values per (layer, "
                              f"head) in pilot_result.json")
        selection["causal_head_evidence"] = ev
        step["selection"] = selection
        arm["step"] = step
        stages[name] = arm
    out["stages"] = stages
    return out


def _reset_peak_vram(device: str) -> None:
    try:
        import torch

        if torch.device(device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
    except Exception:
        pass


def _peak_vram(device: str) -> int | None:
    try:
        import torch

        if torch.device(device).type != "cuda":
            return None
        return int(torch.cuda.max_memory_allocated(device))
    except Exception:
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="")
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--device", default=None)
    ap.add_argument("--toy", action="store_true")
    ap.add_argument("--required-inputs", action="store_true",
                    help="print the mixtures the real path needs, one JSON "
                         "object per line, and exit")
    ap.add_argument("--check-inputs", action="store_true",
                    help="resolve every required mixture for real and exit")
    args = ap.parse_args(argv)

    if args.required_inputs:
        for entry in required_profiles(Path(args.repo)):
            print(json.dumps(entry, sort_keys=True))
        return 0
    if args.check_inputs:
        return check_inputs(Path(args.repo))

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
        run(out, repo=Path(args.repo), toy=args.toy, device=device)
    except Exception as exc:              # noqa: BLE001 - see below
        #: BROAD, DELIBERATELY. A paid pod has died in this project inside an
        #: exception the driver did not name, and the session's evidence went
        #: with it. Everything completed so far is already on disk; this adds
        #: the reason and exits non-zero.
        _write(out / "pilot_failure.json", {
            "error": f"{type(exc).__name__}: {exc}",
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        _say(f"FAILED {type(exc).__name__}: {exc}")
        import traceback

        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "7")
    raise SystemExit(main())
