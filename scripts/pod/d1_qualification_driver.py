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


class PartialPath(QualificationError):
    """A fixed-path arm failed, CARRYING the steps it had already completed.

    The point is that the caller files `partial` into the record before letting
    the failure propagate. A paid stage that completes expensive work and then
    fails must not report that work as absent.
    """

    def __init__(self, message: str, *, partial: dict[str, Any]) -> None:
        super().__init__(message)
        self.partial = partial


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


def release_intermediates(arm_record: dict[str, Any], *,
                          journal: Journal) -> dict[str, Any]:
    """Delete a completed arm's INTERMEDIATE checkpoints, keep its final.

    P8.4: artifacts follow consumers. Once an arm is recorded, nothing reads its
    intermediates -- stage C compares digests and selections that are already in
    the record, and stage D loads only `final_checkpoint_path`. Their bytes are
    pure disk pressure, and on a three-arm run they are the difference between
    fitting and not: each arm retains ~11.75 GiB, of which ~10.6 GiB is
    intermediate, so three arms would want ~65 GiB of a 60 GiB disk and the LAST
    arm would fail after the first two had already been paid for.

    The digests are what survive, which is the point -- an identity is evidence
    and a 6 GiB intermediate nobody reads is not.
    """
    import shutil

    keep = arm_record.get("final_checkpoint_path")
    freed, removed = 0, []
    for step in arm_record.get("steps") or []:
        path = step.get("checkpoint_path")
        if not path or path == keep:
            continue
        p = Path(path)
        if not p.is_dir():
            continue
        size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        shutil.rmtree(p, ignore_errors=True)
        freed += size
        removed.append(p.name)
        #: The record keeps the path it USED, and says the bytes are gone.
        step["checkpoint_released"] = True
    out = {"released": removed, "freed_bytes": freed, "kept": keep}
    if removed:
        journal.event(stage="release_intermediates", status="ok",
                      arm=arm_record.get("arm"), n=len(removed),
                      freed_gib=round(freed / 2**30, 3))
    return out


def unpinned(spec):
    """`spec` with every step's expected-artifact pin cleared.

    `dataclasses.replace`, so a field added to `FixedPathStep` or
    `FixedPathSpec` later is carried rather than silently dropped by a
    hand-written reconstruction. The `path_id` gains a suffix because an
    unpinned path is NOT the pinned one and must not be recorded under its id.
    """
    import dataclasses

    steps = tuple(dataclasses.replace(s, expected_artifact_digest=None)
                  for s in spec.steps)
    return dataclasses.replace(spec, steps=steps,
                               path_id=f"{spec.path_id}.unpinned")


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
    if expected_final is None:
        #: A3's spec pins every intermediate to the INCUMBENT's artifact, so it
        #: can replay the incumbent exactly. That is right for arm A and a
        #: contradiction for any other arm: an arm that changes the position
        #: policy or the calibration batch size is EXPECTED to build something
        #: else, and the pin stopped s2 at step 2 after 1,384 paid seconds for
        #: doing precisely its job. The pin is not the bug -- reusing a pinned
        #: spec to run an unpinned question is. Only the arm that claims to
        #: reproduce the incumbent keeps the pins.
        spec = unpinned(spec)
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
    try:
        materialize_fixed_path(
            spec, adapter=adapter,
            root_loader=lambda: adapter.load(teacher_path, dtype="bfloat16",
                                             device="cuda"),
            workdir=workdir, repo_root=repo, on_step=on_step,
            deadline=deadline, execution=execution)
    except BaseException as exc:
        #: EVERY completed step survives the failure of a later one. s2's arm B
        #: ran 1,384 paid seconds, completed three steps and recorded their
        #: digests -- and the record kept `batch_size: null, steps: []`, because
        #: the raise happened before the return dict was built. The digests were
        #: recoverable from the journal only by luck. An expensive unit of work
        #: that is finished must be persisted at the moment it finishes, not at
        #: the moment the stage happens to succeed.
        raise PartialPath(f"{arm}: {type(exc).__name__}: {exc}", partial={
            "arm": arm, "batch_size": batch_size, "position_policy": policy_id,
            "path_id": spec.path_id, "pinned": expected_final is not None,
            "n_steps": len(steps), "steps": steps,
            "seconds": round(time.time() - t0, 3),
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "failed_at_step": len(steps),
            "failure": f"{type(exc).__name__}: {exc}",
        }) from exc
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

def measure_state_eval(evaluator, model, *, label: str, teacher,
                       artifact_digest: str,
                       journal: Journal) -> dict[str, Any]:
    """Peak memory and wall clock of one real state evaluation.

    `teacher` and `artifact_digest` are REQUIRED, not optional with defaults.
    The evaluator refuses to measure without either, and a keyword default here
    would move that refusal from this signature to the middle of a paid run.
    """
    import torch

    torch.cuda.reset_peak_memory_stats()
    before = int(torch.cuda.memory_allocated())
    t0 = time.time()
    #: TWO contract defects cost stage D its whole measurement on s3, and both
    #: were readable at $0 from the signature:
    #:
    #:   `evaluate(self, model, artifact_digest, *, reference=..., runtime=...)`
    #:
    #: `evaluate(model)` raised TypeError, and the evaluator ALSO refuses
    #: without a primed reference -- "a candidate cannot be scored against a
    #: teacher that was not run". The teacher is the reference the distortion is
    #: measured against, so priming it is not setup, it is half the measurement.
    evaluator.prime_reference(teacher)
    evaluation = evaluator.evaluate(model, artifact_digest)
    elapsed = round(time.time() - t0, 3)
    peak = int(torch.cuda.max_memory_allocated())
    derived = getattr(evaluator, "batch_budget_bytes", None)
    #: THE CEILING IS NOT THE PREDICTION, and conflating them makes stage D
    #: unable to answer what it exists to answer. `batch_budget_bytes` is a
    #: configured ceiling on the two LOGIT BLOCKS; `batch_plan`'s
    #: `peak_logit_bytes` is this protocol's own derived prediction of them. The
    #: process peak includes neither alone -- it also holds the model and the
    #: reduction's transients. So all three are recorded, plus the DELTA the
    #: evaluation itself added, which is the quantity the prediction is about.
    plan: dict[str, Any] | None = None
    try:
        vocab = int(getattr(model.config, "vocab_size", 0) or 0)
        plan = dict(evaluator.batch_plan(
            vocab, next(model.parameters()).dtype.itemsize))
    except Exception as exc:                      # noqa: BLE001
        plan = {"failed": f"{type(exc).__name__}: {exc}"}
    predicted = (plan or {}).get("peak_logit_bytes")
    out = {
        "label": label,
        "seconds": elapsed,
        "peak_memory_bytes": peak,
        "allocated_before_bytes": before,
        "evaluation_delta_bytes": peak - before,
        "batch_plan": plan,
        "predicted_peak_logit_bytes": predicted,
        "delta_over_predicted_logits": (
            None if not predicted else round((peak - before) / predicted, 4)),
        "derived_budget_bytes": derived,
        "within_derived_budget": (None if derived is None else peak <= derived),
        "_what_each_bound_is": (
            "`derived_budget_bytes` is the configured CEILING on the logit "
            "blocks; `predicted_peak_logit_bytes` is this protocol's derived "
            "prediction of them; `peak_memory_bytes` is the whole process. "
            "`delta_over_predicted_logits` is the ratio that says whether the "
            "derivation describes the real allocation -- near 1 means the "
            "prediction is the allocation, well above 1 means the transients "
            "the bound excludes dominate it."),
        "values": {k: (round(float(v), 8) if isinstance(v, (int, float)) else v)
                   for k, v in (getattr(evaluation, "values", {}) or {}).items()},
        "protocol_id": getattr(evaluation, "measurement_protocol_id", None)
                       or (getattr(evaluation, "detail", {}) or {}).get(
                           "measurement_protocol_id"),
    }
    journal.event(stage=f"state_eval.{label}", status="ok",
                  peak_gib=round(peak / 2**30, 3), seconds=elapsed)
    return out


#: Which protocol gets state-evaluated, on whose artifact.
PROTOCOL_ARM = {"incumbent": "A_incumbent", "target_aware": "B_target_aware"}
FALLBACK_ORDER = ("A_incumbent", "B_target_aware", "B_batch_only")


def state_eval_plan(record: dict[str, Any], protocols):
    """`(label, arm_record, bound, source_arm, is_own)` per protocol, plus failures.

    State-eval memory is a property of the GEOMETRY, the vocabulary and the
    protocol's batch plan -- not of the weights. Both arms' finals are the same
    target `ArchSpec`, so one materialized artifact measures both protocols and
    the record states which it used. That is what lets a repair subrun measure
    stage D without repeating the arms whose findings are already complete.

    A protocol whose own arm did not materialize falls back to any arm that did,
    and the fallback is RECORDED rather than silent: the memory figure transfers
    across arms, the distortion VALUES do not.
    """
    plan, failures = [], []
    for label, bound in protocols:
        own = PROTOCOL_ARM[label]
        if (record.get(own) or {}).get("final_artifact_digest"):
            source = own
        else:
            source = next((k for k in FALLBACK_ORDER
                           if (record.get(k) or {}).get(
                               "final_artifact_digest")), None)
        if source is None:
            failures.append({
                "label": label,
                "failed": ("no materialized artifact exists in this subrun, so "
                           "there is nothing to state-evaluate")})
            continue
        plan.append((label, record[source], bound, source, source == own))
    return plan, failures


# --- C: did any discrete decision move? -----------------------------------

def _attribute(a: dict[str, Any], b: dict[str, Any],
               c: dict[str, Any] | None) -> str:
    """Which knob moved this step's selection: the POLICY or the BATCH SIZE.

    A differs from B in two knobs at once, so A-vs-B alone cannot say. C holds
    the policy at the incumbent's and moves only the batch size, which
    identifies the comparison.
    """
    if a["selection"] == b["selection"]:
        return "no difference between the incumbent and the target-aware arm"
    if c is None:
        return ("UNATTRIBUTED: the batch-only arm did not reach this step, so "
                "the policy and the batch size remain confounded here")
    if c["selection"] == a["selection"]:
        return ("THE POSITION POLICY moved it: holding the policy and changing "
                "only the batch size reproduced the incumbent's selection")
    if c["selection"] == b["selection"]:
        return ("THE CALIBRATION BATCH SIZE moved it: changing only the batch "
                "size, at the incumbent's own policy, already reproduced the "
                "target-aware selection -- so the policy is not implicated")
    return ("BOTH KNOBS, or an interaction: changing only the batch size "
            "produced a THIRD selection, equal to neither arm")


def compare_selections(incumbent: dict[str, Any],
                       target_aware: dict[str, Any],
                       batch_only: dict[str, Any] | None = None,
                       ) -> dict[str, Any]:
    """Per-step discrete differences between the arms, and what moved them.

    A difference is EVIDENCE, not a failure. What would be a failure is a
    difference nobody can explain -- so the comparison is three-armed and each
    moved selection is attributed to a knob rather than left ambiguous.
    """
    rows = []
    c_steps = {s["index"]: s for s in (batch_only or {}).get("steps", [])}
    for a, b in zip(incumbent.get("steps", []), target_aware.get("steps", [])):
        c = c_steps.get(a["index"])
        rows.append({
            "index": a["index"],
            "impl_id": a["impl_id"],
            "artifact_digest_differs": a["artifact_digest"] != b["artifact_digest"],
            "incumbent_digest": (a["artifact_digest"] or "")[:16],
            "target_aware_digest": (b["artifact_digest"] or "")[:16],
            "batch_only_digest": ((c or {}).get("artifact_digest") or "")[:16],
            "selection_differs": a["selection"] != b["selection"],
            "incumbent_selection": a["selection"],
            "target_aware_selection": b["selection"],
            "batch_only_selection": (c or {}).get("selection"),
            "attribution": _attribute(a, b, c),
        })
    moved = [r["impl_id"] for r in rows if r["selection_differs"]]
    return {
        "_what": ("per-step discrete comparison of three arms on the same CUDA "
                  "device: the incumbent protocol, the target-aware protocol at "
                  "the D1 batch size, and the incumbent policy at the D1 batch "
                  "size"),
        "_why_three_arms": (
            "the incumbent and target-aware arms differ in TWO knobs at once, "
            "the scoring position policy and the calibration micro-batch size. "
            "s2 measured a moved FFN selection between them and could attribute "
            "it to neither, which is most of the value of the finding lost. The "
            "third arm moves ONLY the batch size, so each difference is "
            "identified."),
        "steps": rows,
        "operators_whose_selection_moved": moved,
        "n_moved": len(moved),
        "attributions": {r["impl_id"]: r["attribution"] for r in rows
                         if r["selection_differs"]},
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
    ap.add_argument("--arms", default="A,B,Cattr",
                    help="which fixed-path arms to materialize: A (the pinned "
                         "incumbent hard gate), B (the target-aware D1 path), "
                         "Cattr (the attribution arm, incumbent policy at the "
                         "D1 batch size). A repair subrun that needs only stage "
                         "D should not repeat arms already measured.")
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

        def arm(key: str, *, workdir: str, batch_size: int, policy_id: str,
                expected_final: str | None) -> dict[str, Any]:
            """One fixed-path arm, whose partial evidence survives its failure."""
            with journal.stage(key) as st:
                try:
                    record[key] = run_path(
                        repo=repo, workdir=out / workdir, arm=key,
                        batch_size=batch_size, policy_id=policy_id,
                        expected_final=expected_final,
                        teacher_path=teacher_path, journal=journal,
                        deadline=deadline)
                except PartialPath as exc:
                    #: FILE IT, THEN RE-RAISE. The steps this arm completed are
                    #: paid evidence and they go into the record whether or not
                    #: the arm finished.
                    record[key] = exc.partial
                    raise
                st.result = {"digest": (record[key]["final_artifact_digest"]
                                        or "")[:12]}
                if expected_final is not None:
                    st.result["reconstructed"] = record[key]["reconstructed"]
                #: Immediately, not at closeout: the next arm needs the disk.
                record[key]["released_intermediates"] = release_intermediates(
                    record[key], journal=journal)
                return record[key]

        #: A -- the hard gate. Pinned, because this arm claims to reproduce the
        #: incumbent, and it is the only arm that claims that.
        if "A" in wanted:
            arm("A_incumbent", workdir="incumbent", batch_size=1,
                policy_id="positions.all_v1", expected_final=expected)

        #: WHICH ARMS THIS SUBRUN RUNS. A repair subrun that needs only stage D
        #: must not repeat 85 minutes of arms whose findings are already
        #: complete and recorded -- that is paying twice for one measurement.
        #: Default is every arm; the record states what ran.
        wanted = {a.strip() for a in (args.arms or "A,B,Cattr").split(",")
                  if a.strip()}
        record["arms_requested"] = sorted(wanted)

        #: B -- the intended D1 path: target-aware policy AT the D1 batch size.
        if "B" in wanted:
            arm("B_target_aware", workdir="target_aware",
                batch_size=args.batch_size,
                policy_id="positions.supervised_target_v1",
                expected_final=None)

        #: B_batch_only -- the ATTRIBUTION arm, and the reason it exists:
        #: A and B differ in TWO knobs at once, the position policy and the
        #: calibration batch size. s2 found the FFN top-k diverging between them
        #: and could not say which knob moved it, which makes the finding nearly
        #: useless -- `fixed_path`'s own docstring records that an operator
        #: selects differently at a different calibration forward batch size, so
        #: both explanations were live. This arm holds the policy at the
        #: incumbent's and moves ONLY the batch size, so the comparison is
        #: identified: matching A means the POLICY moved the selection, matching
        #: B means the BATCH SIZE did.
        if "Cattr" in wanted:
            arm("B_batch_only", workdir="batch_only",
                batch_size=args.batch_size, policy_id="positions.all_v1",
                expected_final=None)

        #: D -- real state-eval peak memory, PER BOUND PROTOCOL, on a real
        #: materialized artifact. The quantity is a property of the geometry, the
        #: vocabulary and the protocol's batch plan, not of the weights: both
        #: arms' finals ARE the same target `ArchSpec`, so one materialization
        #: measures both protocols and the record says which artifact it used.
        #: That is what lets a repair subrun re-measure D without repeating the
        #: arms whose findings are already complete.
        d_plan, d_failures = state_eval_plan(
            record, (("incumbent", bound_inc), ("target_aware", bound_tgt)))
        record.setdefault("D_state_eval", []).extend(d_failures)
        for label, arm_record, bound, source, is_own in d_plan:
            #: The attribution arm is not state-evaluated for its own sake: its
            #: artifact exists to identify a selection difference, and no
            #: consumer reads its memory. P8.4 -- artifacts follow consumers.
            with journal.stage(f"D_state_eval_{label}") as st:
                try:
                    from aadistill.initialization.specs.arch import get_adapter

                    adapter = get_adapter("qwen3")
                    model = adapter.load(arm_record["final_checkpoint_path"],
                                         dtype="bfloat16", device="cuda")
                    #: The REFERENCE teacher, loaded here because the distortion
                    #: is measured against it. Its bytes are part of the peak
                    #: this stage exists to report -- a state evaluation holds
                    #: the teacher, the candidate and two logit blocks at once,
                    #: and reporting the candidate's memory alone would describe
                    #: a measurement nobody runs.
                    teacher = adapter.load(teacher_path, dtype="bfloat16",
                                           device="cuda")
                    record["D_state_eval"].append(
                        measure_state_eval(
                            bound["evaluator"], model, label=label,
                            teacher=teacher,
                            artifact_digest=arm_record["final_artifact_digest"],
                            journal=journal))
                    record["D_state_eval"][-1]["measured_on"] = {
                        "arm": source, "is_the_protocols_own_arm": is_own,
                        "artifact_digest":
                            arm_record["final_artifact_digest"],
                        "_why_this_is_sound": (
                            "state-eval memory is a property of the geometry, "
                            "the vocabulary and the protocol's batch plan. Both "
                            "arms' finals are the same target ArchSpec, so the "
                            "figure does not depend on which arm's weights are "
                            "loaded -- only the distortion VALUES do, and those "
                            "are reported as this artifact's, not as the other "
                            "arm's."
                            if not is_own else
                            "this protocol's own arm")}
                    del model, teacher
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
            #: Only when both arms ran in THIS subrun. Comparing this run's
            #: arm against another run's recorded digests would be a comparison
            #: across two environments presented as one.
            if (record.get("A_incumbent") or {}).get("steps") and \
                    (record.get("B_target_aware") or {}).get("steps"):
                record["C_selection_comparison"] = compare_selections(
                    record["A_incumbent"], record["B_target_aware"],
                    record.get("B_batch_only"))
            else:
                record["C_selection_comparison"] = {
                    "_not_run": ("this subrun did not materialize both the "
                                 "incumbent and the target-aware arm, so there "
                                 "is no within-run comparison to make"),
                    "steps": [], "n_moved": 0}
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
