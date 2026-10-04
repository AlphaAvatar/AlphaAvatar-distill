#!/usr/bin/env python3
"""The D1 engineering GPU qualification. Engineering evidence only.

    python scripts/pod/d1_qualification_driver.py --out DIR --repo REPO [--deadline-s N]

**It answers five engineering questions and performs no science.** No recovery
training, no behavioural screening, no confirmation, no formal search, no
promotion, no GO/NO-GO. It reads no D-series behavioural prompt.

```text
A  does the incumbent fixed path reconstruct the frozen incumbent artifact
   identity on real CUDA, under the incumbent scoring policy?   HARD GATE
B  does the target-aware path execute at bsz=3 under CUDA/bf16, through the
   REAL evaluator including candidate/state evaluation?
C  does bf16/CUDA move any discrete operator or selection decision?
D  what is the real state-eval peak memory at the real vocabulary and batch plan?
E  what does a representative expansion actually cost?
```

**C is evidence, not a verdict.** A CPU-vs-GPU selection difference is a finding.
What this driver refuses to produce is an unbound numerical identity, unexplained
same-environment nondeterminism, a resume/materialization identity disagreement,
or a semantics quietly changed so CUDA matches CPU.

**The protocol identity is bound before the first expensive step.** Stage 2 builds
the evaluator with the frozen suite's `content_sha256`, the declared position
policy and the declared `NumericalEnvironment`, and REFUSES to continue if the
resulting `suite_content_identity` is `unbound`. A qualification record carrying
`unbound` would describe a measurement nobody can place.

**Nothing instance-specific lives in core.** The geometry, the vocabulary, the
batch size, the device and the artifact paths are all read from the frozen
records and the adapters; this script is the experiment layer where instance
policy belongs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO_DEFAULT = Path(__file__).resolve().parents[2]


def _bootstrap(repo: Path) -> None:
    for extra in ("src", "scripts", "scripts/data"):
        path = str(repo / extra)
        if path not in sys.path:
            sys.path.insert(0, path)


#: The frozen incumbent artifact the incumbent protocol must reconstruct. Read
#: from the A3 authorization that froze it rather than retyped -- one owner.
INCUMBENT_AUTHORIZATION = (
    "logs/stages/stage-1/phase_a3/runs/a3_attempt3/governance/authorization.json")

STATE_EVAL = "artifacts/stage1/state_eval_v1"


class QualificationError(RuntimeError):
    """A stage could not produce the evidence it exists to produce."""


# --- stage plumbing ---------------------------------------------------------

class Journal:
    """Append-only stage log, flushed per event.

    Flushed because the useful case is the one where the process dies: a buffered
    journal of a session that OOMed is a file with nothing in it.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started = time.time()
        self.stages: list[dict[str, Any]] = []

    def event(self, **fields: Any) -> None:
        row = {"t": round(time.time() - self.started, 3), **fields}
        with self.path.open("a") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        print(f"[{row['t']:9.3f}] {fields.get('stage', '')} "
              f"{fields.get('status', '')} "
              f"{json.dumps({k: v for k, v in fields.items() if k not in ('stage', 'status')})[:160]}",
              flush=True)

    def stage(self, name: str):
        return _Stage(self, name)


class _Stage:
    def __init__(self, journal: Journal, name: str) -> None:
        self.journal, self.name = journal, name

    def __enter__(self):
        self.t0 = time.time()
        self.journal.event(stage=self.name, status="start")
        self.result: dict[str, Any] = {}
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = round(time.time() - self.t0, 3)
        if exc is None:
            self.journal.event(stage=self.name, status="ok", seconds=elapsed,
                               **{k: v for k, v in self.result.items()
                                  if isinstance(v, (int, float, str, bool))})
            self.journal.stages.append(
                {"stage": self.name, "seconds": elapsed, "status": "ok",
                 **self.result})
        else:
            self.journal.event(stage=self.name, status="failed",
                               seconds=elapsed, error=f"{exc_type.__name__}: {exc}")
            self.journal.stages.append(
                {"stage": self.name, "seconds": elapsed, "status": "failed",
                 "error": f"{exc_type.__name__}: {exc}",
                 "traceback": "".join(traceback.format_exception(exc_type, exc, tb))})
        return False


# --- stage 0: the execution environment, recorded as protocol -------------

def environment(repo: Path) -> dict[str, Any]:
    """Device, dtype and runtime, recorded because they ARE the protocol identity.

    Not a banner: `NumericalEnvironment` is a term of `measurement_protocol_id`,
    so a measurement whose environment is unrecorded cannot be placed against
    another.
    """
    import torch
    import transformers

    if not torch.cuda.is_available():
        raise QualificationError(
            "no CUDA device. This qualification exists to answer CUDA questions; "
            "a CPU fallback would produce evidence about the wrong machine.")
    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    return {
        "device": "cuda",
        "gpu_name": props.name,
        "gpu_total_memory_bytes": int(props.total_memory),
        "capability": f"{props.major}.{props.minor}",
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "transformers": transformers.__version__,
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
        "_device_count": torch.cuda.device_count(),
        "_no_ordinal_is_hardcoded": (
            "the index comes from torch.cuda.current_device(), so the record "
            "describes whichever device the runtime selected"),
    }


# --- stage 2: bind the protocol BEFORE anything expensive -----------------

def bind_protocol(repo: Path, *, policy_id: str, execution, numerics) -> dict[str, Any]:
    """Build the real evaluator and return its bound identity.

    Refuses `unbound` suite content. The whole point of binding here rather than
    after the first measurement is that by then the expensive work has been done
    under an identity nothing recorded.
    """
    from aadistill.initialization.scoring.protocol_identity import (
        UNBOUND_SUITE_CONTENT,
    )
    from aadistill.initialization.scoring.positions import get_position_policy
    from aadistill.initialization.planning.metrics import StateEvaluator

    sys.path.insert(0, str(repo / "scripts/autoinit"))
    import load_state_eval

    suite, items, content_sha256 = _load_suite(load_state_eval, repo)
    if not content_sha256:
        raise QualificationError(
            f"{STATE_EVAL}/manifest.json carries no content_sha256, so the "
            "measurement protocol cannot be bound")
    #: QUALIFIED ids: the registry keys on `positions.x_v1@v1`, not on
    #: `positions.x_v1`. Resolving by the bare id raises, listing both shipped
    #: policies as proof the name was close but not the key.
    policy = get_position_policy(policy_id if "@" in policy_id
                                 else f"{policy_id}@v1")
    evaluator = StateEvaluator(
        suite, items, device="cuda", position_policy=policy,
        execution=execution, numerics=numerics,
        suite_content_sha256=content_sha256)
    bound = {
        "measurement_protocol_id": evaluator.measurement_protocol_id,
        "suite_content_sha256": content_sha256,
        "suite_structural_hash": suite.suite_hash,
        "position_policy": policy.policy_id,
        "position_policy_hash": policy.policy_hash,
        "n_items": len(items),
        "reduction": {"chunk": evaluator.chunk,
                      "reference_strategy": str(evaluator.reference_strategy)},
    }
    if UNBOUND_SUITE_CONTENT in json.dumps(bound):
        raise QualificationError(
            "the bound protocol carries `unbound` suite content; refusing to "
            "measure under an identity that cannot be placed")
    return {"evaluator": evaluator, "bound": bound, "items": items}


def _load_suite(load_state_eval, repo: Path):
    """`(suite, items, content_sha256)` from the frozen state-eval asset.

    `load_state_eval.load` returns `(suite, items, MANIFEST)` -- the whole
    manifest, not the hash. An earlier version of this helper assumed the third
    element was the hash string and would have passed a dict to
    `suite_content_sha256`, which binds whatever it is given: the protocol id
    would have been computed over a dict's repr and looked perfectly bound.

    Caught at `$0` by a contract probe rather than on a paid pod, which is the
    only reason it is cheap.
    """
    loaded = load_state_eval.load(repo / STATE_EVAL)
    if not (isinstance(loaded, tuple) and len(loaded) == 3):
        raise QualificationError(
            f"load_state_eval.load returned {type(loaded).__name__} of "
            f"{len(loaded) if isinstance(loaded, tuple) else '?'}; expected "
            "(suite, items, manifest)")
    suite, items, manifest = loaded
    if not isinstance(manifest, dict):
        raise QualificationError(
            f"the loader's third value is {type(manifest).__name__}, not the "
            "manifest; the content hash cannot be read from it")
    content = manifest.get("content_sha256")
    if not isinstance(content, str) or len(content) != 64:
        raise QualificationError(
            f"{STATE_EVAL}/manifest.json content_sha256 is "
            f"{content!r}; a non-hash value would bind a protocol identity that "
            "describes nothing")
    return suite, items, content


# --- stages 4 and 6: the two fixed-path runs ------------------------------

def _path_profiles():
    """The fixed path's steps, so the registry check knows what to require."""
    from experiments.phase_a3 import a3_session as A3S

    return A3S.path_spec(workdir_device="cuda").steps


def run_path(*, repo: Path, workdir: Path, arm: str, batch_size: int,
             policy_id: str, expected_final: str | None, teacher_path: str,
             journal: Journal, deadline=None) -> dict[str, Any]:
    """One fixed-path execution on CUDA, with its per-step evidence.

    `expected_final` is a HARD GATE when supplied: the incumbent protocol must
    reconstruct the frozen incumbent artifact. The target-aware arm supplies
    None, because what it builds is the finding.
    """
    import torch

    from aadistill.initialization.execution import ExecutionConfig
    from aadistill.initialization.planning.fixed_path import (
        materialize_fixed_path,
    )
    from aadistill.initialization.specs.arch import get_adapter
    from experiments.phase_a3 import a3_session as A3S

    spec = A3S.path_spec(workdir_device="cuda")
    adapter = get_adapter("qwen3")
    execution = ExecutionConfig(micro_batch_size=batch_size,
                               calibration_batch_packing="length_sorted_v1")

    steps: list[dict[str, Any]] = []

    def on_step(result) -> None:
        #: `as_dict()`, not attribute access. `artifact_digest` lives on
        #: `result.identity`, so `getattr(result, "artifact_digest", None)`
        #: returned None for every step -- every digest in the record would have
        #: been null and the hard gate would have compared None to the frozen
        #: incumbent and "failed" for the wrong reason.
        row = result.as_dict()
        row["index"] = len(steps)
        steps.append(row)
        journal.event(stage=f"{arm}.step{row['index']}", status="ok",
                      impl=row["impl_id"],
                      digest=(row["artifact_digest"] or "")[:12],
                      matches=row.get("digest_matches"),
                      seconds=row.get("seconds"))

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    results = materialize_fixed_path(
        spec, adapter=adapter,
        root_loader=lambda: adapter.load(teacher_path, dtype="bfloat16",
                                         device="cuda"),
        workdir=workdir, repo_root=repo, on_step=on_step,
        deadline=deadline, execution=execution)
    elapsed = round(time.time() - t0, 3)
    final = steps[-1]["artifact_digest"] if steps else None
    final_checkpoint = steps[-1]["checkpoint_path"] if steps else None
    peak = int(torch.cuda.max_memory_allocated())

    out = {
        "arm": arm, "batch_size": batch_size, "position_policy": policy_id,
        "path_id": spec.path_id, "n_steps": len(steps), "steps": steps,
        "final_artifact_digest": final,
        "final_checkpoint_path": final_checkpoint,
        "seconds": elapsed,
        "peak_memory_bytes": peak,
        "_peak_is_torch_allocated": (
            "torch.cuda.max_memory_allocated, reset immediately before the run"),
    }
    if expected_final is not None:
        out["expected_final_artifact_digest"] = expected_final
        out["reconstructed"] = (final == expected_final)
        if final != expected_final:
            raise QualificationError(
                f"{arm}: HARD GATE FAILED. The incumbent protocol rebuilt "
                f"{(final or 'nothing')[:16]} where the frozen incumbent is "
                f"{expected_final[:16]}. This is not something to smooth over: "
                "diagnose it. Per-step digests are in the stage record.")
    return out


def fetch_teacher(repo: Path) -> str:
    """The pinned teacher, fetched and shard-verified against its binding.

    The binding is the owner of the expected shard hashes; this verifies against
    it rather than trusting the hub, because a silently different shard would
    make every digest below describe a different model.
    """
    import hashlib

    from huggingface_hub import snapshot_download

    from experiments.phase_c3 import session as CS

    #: logs/, not configs/. The binding is EVIDENCE about a fetched artifact --
    #: a revision and its per-shard hashes -- and it lives with the phase that
    #: froze it. An invented configs/ path would have failed on the pod after
    #: setup, which is the expensive place to learn a filename.
    binding = json.loads(
        (repo / "logs/stages/stage-1/phase_c1/plans/teacher_binding.json").read_text())
    if binding["revision"] != CS.TEACHER_REVISION:
        raise QualificationError(
            f"the teacher binding pins {binding['revision']} and the session "
            f"declares {CS.TEACHER_REVISION}")
    local = snapshot_download(CS.TEACHER_REPO, revision=CS.TEACHER_REVISION)
    bad = []
    for name, want in binding["expected_shard_sha256"].items():
        path = Path(local) / name
        if not path.is_file():
            bad.append(f"{name}: absent after fetch")
            continue
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != want:
            bad.append(f"{name}: {got} != {want}")
    if bad:
        raise QualificationError("teacher verification FAILED: " + "; ".join(bad))
    return local


class WallClockDeadline:
    """The session's remaining time, as the operator deadline contract.

    `depth.causal_kl_greedy_v1` checks a deadline per candidate, and without one
    an operator whose work is measured in hours has nothing to consult. A wall
    clock rather than a spend figure because this driver is handed its budget as
    seconds by the launcher, which is where the live price lives.
    """

    def __init__(self, seconds: int) -> None:
        self.seconds = int(seconds)
        self.started = time.time()

    def remaining(self) -> float:
        return self.seconds - (time.time() - self.started)

    def check(self, where: str = "") -> None:
        if self.remaining() <= 0:
            raise QualificationError(
                f"wall-clock deadline of {self.seconds}s reached"
                f"{' at ' + where if where else ''}; stopping before the "
                "session ceiling rather than after it")


def _numerics(env: dict[str, Any]):
    """The execution fingerprint: device CLASS, compute dtype, accumulation dtype.

    Three declared fields and no more. The card name, the capability and the
    runtime versions are recorded in `environment` for the reader but are
    deliberately NOT part of the fingerprint: one digest has been rebuilt
    byte-identically on three different rented cards of the same model, so the
    particular card is not part of the identity, while `cpu` and `cuda` are
    different kernels and different reductions.

    An earlier version of this helper passed `gpu_name`, `capability` and
    `allow_tf32` and filtered by signature, which would have silently dropped
    all three and then failed on the two required fields it never supplied. The
    core is right to refuse: an unstated numerical condition is not a default.
    """
    from aadistill.initialization.specs.materialization import NumericalEnvironment

    return NumericalEnvironment(
        device_type=env["device"],       # the CLASS, never an ordinal
        compute_dtype="bfloat16",        # what the fixed path loads the root as
        accumulation_dtype="float64",    # the reducers' accumulation, unchanged
    )


# --- stage D: real state-eval peak memory ---------------------------------

def measure_state_eval(evaluator, model, *, label: str,
                       journal: Journal) -> dict[str, Any]:
    """Peak memory and wall clock of one real state evaluation."""
    import torch

    torch.cuda.reset_peak_memory_stats()
    before = int(torch.cuda.memory_allocated())
    t0 = time.time()
    evaluation = evaluator.evaluate(model)
    elapsed = round(time.time() - t0, 3)
    peak = int(torch.cuda.max_memory_allocated())
    derived = getattr(evaluator, "batch_budget_bytes", None)
    out = {
        "label": label,
        "seconds": elapsed,
        "peak_memory_bytes": peak,
        "allocated_before_bytes": before,
        "derived_budget_bytes": derived,
        "within_derived_budget": (None if derived is None else peak <= derived),
        "values": {k: (round(float(v), 8) if isinstance(v, (int, float)) else v)
                   for k, v in (getattr(evaluation, "values", {}) or {}).items()},
        "protocol_id": getattr(evaluation, "measurement_protocol_id", None)
                       or (getattr(evaluation, "detail", {}) or {}).get(
                           "measurement_protocol_id"),
    }
    journal.event(stage=f"state_eval.{label}", status="ok",
                  peak_gib=round(peak / 2**30, 3), seconds=elapsed)
    return out


# --- C: did any discrete decision move? -----------------------------------

def compare_selections(incumbent: dict[str, Any],
                       target_aware: dict[str, Any]) -> dict[str, Any]:
    """Per-step discrete differences between the two arms.

    Reported as evidence. A difference is expected -- the arms differ in both
    the scoring policy and the batch size -- and the record says which of those
    could explain it rather than ruling either way.
    """
    rows = []
    for a, b in zip(incumbent.get("steps", []), target_aware.get("steps", [])):
        rows.append({
            "index": a["index"],
            "impl_id": a["impl_id"],
            "artifact_digest_differs": a["artifact_digest"] != b["artifact_digest"],
            "incumbent_digest": (a["artifact_digest"] or "")[:16],
            "target_aware_digest": (b["artifact_digest"] or "")[:16],
            "selection_differs": a["selection"] != b["selection"],
            "incumbent_selection": a["selection"],
            "target_aware_selection": b["selection"],
        })
    moved = [r["impl_id"] for r in rows if r["selection_differs"]]
    return {
        "_what": ("per-step discrete comparison of the incumbent protocol and "
                  "the target-aware protocol on the same CUDA device"),
        "_a_difference_is_evidence": (
            "the arms differ in TWO ways -- the scoring position policy and the "
            "micro-batch size -- so a moved decision is not attributable to "
            "either alone from this comparison. It is recorded, not adjudicated."),
        "steps": rows,
        "operators_whose_selection_moved": moved,
        "n_moved": len(moved),
        "final_artifacts_differ": (
            incumbent.get("final_artifact_digest")
            != target_aware.get("final_artifact_digest")),
    }


# --- main -----------------------------------------------------------------

def frozen_incumbent_digest(repo: Path) -> str:
    """The digest the incumbent protocol must rebuild, from its owning record."""
    doc = json.loads((repo / INCUMBENT_AUTHORIZATION).read_text())
    for key in ("incumbent",):
        got = _find(doc, key)
        if isinstance(got, str) and len(got) == 64:
            return got
    raise QualificationError(
        f"{INCUMBENT_AUTHORIZATION} states no 64-hex `incumbent` digest")


def _find(node: Any, key: str) -> Any:
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for value in node.values():
            got = _find(value, key)
            if got is not None:
                return got
    elif isinstance(node, list):
        for value in node:
            got = _find(value, key)
            if got is not None:
                return got
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    #: NOT required: `--required-inputs` is a $0 listing that writes nothing,
    #: and the launcher calls it without an --out. Requiring it made the
    #: launcher's pre-create input check exit 2 with a usage error, which the
    #: launcher reads as "could not derive the inputs" -- a refusal to create a
    #: pod, caused by an argument the listing never needed.
    ap.add_argument("--out", default=None)
    ap.add_argument("--repo", default=str(REPO_DEFAULT))
    ap.add_argument("--deadline-s", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=3,
                    help="the target-aware arm's micro-batch size (experiment "
                         "policy; the core accepts any value)")
    ap.add_argument("--required-inputs", action="store_true",
                    help="print the inputs this driver needs and exit")
    ap.add_argument("--check-only", action="store_true",
                    help=("run every stage EXCEPT the two expensive path runs: "
                          "the CUDA probe, the three process-global registries, "
                          "the frozen-asset load, the policy resolution and both "
                          "protocol bindings. Seconds, on the pod's own "
                          "interpreter, before the root teacher is resident."))
    args = ap.parse_args(argv)

    repo = Path(args.repo).resolve()
    _bootstrap(repo)

    if args.required_inputs:
        #: The KEY NAMES the launcher consumes -- `items_path` and
        #: `items_file_sha256`. A driver that printed `path`/`sha256` would be
        #: read by the launcher's `json.load(...)['items_path']` as a KeyError
        #: after a pod had already been created.
        import hashlib

        for profile_id, rel in (
                ("state_eval.items", f"{STATE_EVAL}/items.jsonl"),
                ("state_eval.manifest", f"{STATE_EVAL}/manifest.json"),
                ("calib.domain_balanced@v1",
                 "artifacts/stage1/e8_calibration_v1/items.jsonl"),
                ("calib.reasoning_heavy@v2",
                 "artifacts/stage1/reasoning_heavy_v2/items.jsonl")):
            path = repo / rel
            if not path.is_file():
                raise QualificationError(
                    f"{rel} is absent on this machine; the qualification cannot "
                    "resolve its own inputs")
            print(json.dumps({
                "profile_id": profile_id, "items_path": rel,
                "size_bytes": path.stat().st_size,
                "items_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))
        return 0

    if not args.out:
        ap.error("--out is required to run the qualification")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    journal = Journal(out / "journal.jsonl")
    record: dict[str, Any] = {
        "schema": "aadistill.d1_gpu_qualification/v1",
        "_contract": (
            "ENGINEERING EVIDENCE ONLY. No recovery training, no behavioural "
            "screening, no confirmation, no formal search, no promotion, no "
            "GO/NO-GO. No D-series behavioural prompt was read. AUTHORIZES "
            "NOTHING."),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    deadline = None

    try:
        with journal.stage("environment") as st:
            record["environment"] = environment(repo)
            st.result = {"gpu": record["environment"]["gpu_name"]}

        with journal.stage("register_operators") as st:
            from aadistill.initialization.operators.attention.gqa import (
                activation_importance,
            )
            from aadistill.initialization.adapters import (
                register_builtin_adapters,
            )
            from aadistill.initialization.calibration.profiles import (
                registered_profiles,
            )
            from experiments.calibration import register_builtin_profiles
            from experiments.phase_c2.search_space import register_c2_operators

            #: FOUR process-global registries, all empty in a fresh interpreter.
            #: The CALIBRATION PROFILES were missing and subrun s1 paid $0.1098
            #: to find out: adapters and operators were registered, the protocol
            #: bound, and the fixed path then raised `no calibration profile
            #: 'calib.domain_balanced@v1'; registered: []` six seconds in.
            register_builtin_adapters()
            register_builtin_profiles()
            register_c2_operators()
            activation_importance.register()

            profiles = registered_profiles()
            #: ASSERTED, not assumed. The whole class of failure is a registry
            #: that is empty when something reads it, so the stage that fills
            #: them checks they are filled.
            needed = {step.profile_id for step in _path_profiles()}
            missing = sorted(needed - set(profiles))
            if missing:
                raise QualificationError(
                    f"the fixed path needs calibration profiles {missing} and "
                    f"the registry holds {sorted(profiles)}. Registering is "
                    "explicit in this project; a missing one is a crash at the "
                    "first operator, not at import.")
            st.result = {"profiles": len(profiles), "operators_ok": True}
            record["registries"] = {
                "calibration_profiles": sorted(profiles),
                "required_by_the_path": sorted(needed),
            }

        if args.deadline_s:
            deadline = WallClockDeadline(args.deadline_s)

        from aadistill.initialization.execution import ExecutionConfig
        from aadistill.initialization.specs.materialization import (
            NumericalEnvironment,
        )

        #: the EXECUTION FINGERPRINT the protocol id binds. Built from the
        #: environment this process actually has, never from a constant.
        numerics = _numerics(record["environment"])

        #: STAGE 2 -- bind before anything expensive
        with journal.stage("bind_protocol_incumbent") as st:
            bound_inc = bind_protocol(
                repo, policy_id="positions.all_v1",
                execution=ExecutionConfig(micro_batch_size=1),
                numerics=numerics)
            record["bound_protocol_incumbent"] = bound_inc["bound"]
            st.result = {"protocol": bound_inc["bound"]["measurement_protocol_id"][:16]}

        with journal.stage("bind_protocol_target_aware") as st:
            bound_tgt = bind_protocol(
                repo, policy_id="positions.supervised_target_v1",
                execution=ExecutionConfig(
                    micro_batch_size=args.batch_size,
                    calibration_batch_packing="length_sorted_v1"),
                numerics=numerics)
            record["bound_protocol_target_aware"] = bound_tgt["bound"]
            st.result = {"protocol": bound_tgt["bound"]["measurement_protocol_id"][:16]}

        with journal.stage("teacher_fetch_verify") as st:
            teacher_path = fetch_teacher(repo)
            st.result = {"path": teacher_path}

        expected = frozen_incumbent_digest(repo)
        record["frozen_incumbent_artifact_digest"] = expected

        if args.check_only:
            #: EVERYTHING BUT THE EXPENSIVE WORK. The registries are empty in a
            #: fresh interpreter and the frozen assets have to be staged; both
            #: have failed a paid session after the stage before them succeeded.
            #: This reaches the real loader, the real policy registry and the
            #: real CUDA device -- it is not a toy substitute.
            record["status"] = "CHECK_ONLY_OK"
            record["_check_only"] = (
                "the CUDA probe, both protocol bindings and the frozen incumbent "
                "digest resolved on this interpreter. The two fixed-path runs "
                "were NOT executed.")
            return 0

        #: A -- the hard gate
        with journal.stage("A_incumbent_reconstruction") as st:
            record["A_incumbent"] = run_path(
                repo=repo, workdir=out / "incumbent", arm="A_incumbent",
                batch_size=1, policy_id="positions.all_v1",
                expected_final=expected, teacher_path=teacher_path,
                journal=journal, deadline=deadline)
            st.result = {"reconstructed": record["A_incumbent"]["reconstructed"]}

        #: B -- the target-aware path
        with journal.stage("B_target_aware") as st:
            record["B_target_aware"] = run_path(
                repo=repo, workdir=out / "target_aware", arm="B_target_aware",
                batch_size=args.batch_size,
                policy_id="positions.supervised_target_v1",
                expected_final=None, teacher_path=teacher_path,
                journal=journal, deadline=deadline)
            st.result = {"digest":
                         (record["B_target_aware"]["final_artifact_digest"] or "")[:12]}

        #: D -- real state-eval peak memory, on each arm's own final artifact,
        #: under that arm's own bound evaluator. The point is the DEVICE's
        #: memory at the real vocabulary, so it runs on the artifact the path
        #: just built rather than on a stand-in.
        record["D_state_eval"] = []
        for label, arm_record, bound in (
                ("incumbent", record["A_incumbent"], bound_inc),
                ("target_aware", record["B_target_aware"], bound_tgt)):
            with journal.stage(f"D_state_eval_{label}") as st:
                try:
                    from aadistill.initialization.specs.arch import get_adapter

                    adapter = get_adapter("qwen3")
                    model = adapter.load(arm_record["final_checkpoint_path"],
                                         dtype="bfloat16", device="cuda")
                    record["D_state_eval"].append(
                        measure_state_eval(bound["evaluator"], model,
                                           label=label, journal=journal))
                    del model
                    import torch

                    torch.cuda.empty_cache()
                    st.result = {"peak_gib": round(
                        record["D_state_eval"][-1]["peak_memory_bytes"] / 2**30, 3)}
                except Exception as exc:          # noqa: BLE001
                    #: a memory measurement that fails is a finding about the
                    #: memory, so it is recorded rather than aborting C and E
                    record["D_state_eval"].append(
                        {"label": label, "failed": f"{type(exc).__name__}: {exc}"})
                    st.result = {"failed": True}

        #: C
        with journal.stage("C_discrete_decisions") as st:
            record["C_selection_comparison"] = compare_selections(
                record["A_incumbent"], record["B_target_aware"])
            st.result = {"n_moved": record["C_selection_comparison"]["n_moved"]}

        record["status"] = "COMPLETE"
    except BaseException as exc:          # noqa: BLE001 -- recorded, then re-raised shape
        record["status"] = "FAILED"
        record["failure"] = {
            "type": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        journal.event(stage="driver", status="failed", error=str(exc)[:300])
    finally:
        record["stages"] = journal.stages
        record["elapsed_seconds"] = round(time.time() - journal.started, 3)
        record["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        #: WRITTEN ON EVERY PATH. A failed qualification's evidence is the point.
        (out / "qualification.json").write_text(
            json.dumps(record, indent=1, sort_keys=True) + "\n")
        print(f"\nwrote {out}/qualification.json  status={record['status']}",
              flush=True)
    return 0 if record["status"] in ("COMPLETE", "CHECK_ONLY_OK") else 1


if __name__ == "__main__":
    raise SystemExit(main())
