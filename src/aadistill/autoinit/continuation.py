"""Characterizing permanent controls that already exist, without retraining them.

The micro-preflight trained two permanent canonical controls, verified them
strictly against the Stage-0 attestation, and then stopped before Stage 3 because
the frozen battery could not render a tool prompt. The controls are valid; only
their characterization is outstanding. Re-running Stage 2 to reach Stage 3 would
train new controls, which is precisely what must not happen.

`PreflightPlan.advance_to(3, ...)` is right to refuse: it requires the blocking
predecessors to have passed **in this session**, and a session that imports a
checkpoint did not train it. So this is a different plan, not a relaxation of
that one:

    existing permanent controls
      -> strict import / evidence verification      (stage 0, CPU, fails closed)
      -> current generation + evaluation attestation (stage 1, boots the engine)
      -> real v2 tool + RAG generation smoke         (stage 2)
      -> characterize sa and sb                      (stage 3, the paid part)
      -> collect, provider-level teardown, STOP

Two identities are deliberately handled differently, because they are separate by
construction:

* **Recovery identity is imported, never re-attested.** A control's protocol was
  established when it was trained; nothing in a later session can change what
  already ran. The import gate's job is to prove the staged bytes are that same
  control.
* **Evaluation identity is attested fresh, every session.** `recovery_search_v2`,
  the scoring implementation digest and the generation source have all moved
  since the controls were trained, so the characterization binds to the protocol
  attested *now* — not to the one the controls were trained under.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..infrastructure.manifest import sha256_file, sha256_json
from .authorization import SpendAuthorization
from .recovery import (
    PreflightPlan,
    PreflightStage,
    RecoveryAdmissionError,
    RecoveryProbeIdentity,
    observe_recovery_protocol,
)


class ControlImportError(RecoveryAdmissionError):
    """A staged checkpoint is not the permanent control it claims to be."""


#: Every field of a permanent control's identity that the import gate checks.
#: Adding a field here is how a new material property becomes enforced; a field
#: that is recorded but unchecked is documentation, not a gate.
IMPORT_REQUIRED_FIELDS: tuple[str, ...] = (
    "weights_sha256",
    "probe_id",
    "control_binding",
    "observed_protocol_fingerprint",
    "seed",
    "initialization_artifact_digest",
)


@dataclass(frozen=True)
class ImportedControl:
    """One permanent control, re-established from its record and its bytes."""

    name: str
    checkpoint_dir: Path
    weights_sha256: str
    probe_id: str
    observed_protocol_fingerprint: str
    seed: int
    initialization_artifact_digest: str
    control_binding: dict[str, Any]
    reconstructed_from_run_evidence: bool
    evidence: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "checkpoint_dir": str(self.checkpoint_dir),
            "weights_sha256": self.weights_sha256,
            "probe_id": self.probe_id,
            "observed_protocol_fingerprint": self.observed_protocol_fingerprint,
            "seed": self.seed,
            "initialization_artifact_digest": self.initialization_artifact_digest,
            "control_binding": dict(self.control_binding),
            "reconstructed_from_run_evidence": self.reconstructed_from_run_evidence,
            "evidence": self.evidence,
        }


def import_permanent_control(name: str, *, record_path: str | Path,
                             checkpoint_dir: str | Path,
                             run_evidence_dir: str | Path | None = None,
                             repo_root: str | Path | None = None,
                             strict: bool = True) -> ImportedControl:
    """Bind staged bytes back to a recorded permanent control. Fails closed.

    `record_path` is the control's `*_probe_identity.json`, written by the
    session that trained it. `checkpoint_dir` is the staged
    `step_*/model` directory. `run_evidence_dir` holds that run's own
    `run_manifest.json` and `run_completion.json`, when they were collected.

    The gate is deliberately not "does a sidecar summary agree with itself".
    Where the run's own evidence exists, the strict reconstruction is **re-run
    against it** and required to reproduce the recorded protocol fingerprint —
    the same predicate the training session applied, applied again to the
    artifact being imported. A missing material field stops the import; so does
    a hash that does not match the bytes on disk.
    """
    record_path = Path(record_path)
    checkpoint_dir = Path(checkpoint_dir)
    problems: list[str] = []

    if not record_path.is_file():
        raise ControlImportError(
            f"{name}: no permanent-control record at {record_path}; an imported "
            "control with no recorded identity cannot be verified against "
            "anything")
    record = json.loads(record_path.read_text())

    # Where the writer actually puts each field. `verify_control` records the
    # protocol fingerprint inside `control_binding` and again inside
    # `observed_protocol`, never at the top level — so this reads both and
    # requires them to agree, which is a stronger check than one lookup.
    resolved = {
        "weights_sha256": record.get("weights_sha256"),
        "probe_id": record.get("probe_id"),
        "control_binding": record.get("control_binding"),
        "observed_protocol_fingerprint":
            (record.get("control_binding") or {}).get(
                "observed_protocol_fingerprint"),
        "seed": record.get("seed"),
        "initialization_artifact_digest":
            record.get("initialization_artifact_digest"),
    }
    from_observed = ((record.get("observed_protocol") or {})
                     .get("observed_protocol_fingerprint"))
    if from_observed and from_observed != resolved["observed_protocol_fingerprint"]:
        raise ControlImportError(
            f"{name}: the record's two copies of the protocol fingerprint "
            f"disagree ({from_observed} in observed_protocol, "
            f"{resolved['observed_protocol_fingerprint']} in control_binding); "
            "the record is internally inconsistent and cannot establish an "
            "identity")
    missing = [f for f in IMPORT_REQUIRED_FIELDS
               if resolved.get(f) in (None, "", {})]
    if missing:
        raise ControlImportError(
            f"{name}: the record is missing material identity field(s) "
            f"{missing}. Every field in IMPORT_REQUIRED_FIELDS must be present; "
            "a control whose identity was never fully recorded cannot be "
            "re-established from it.")
    if not record.get("protocol_verified"):
        raise ControlImportError(
            f"{name}: the record says protocol_verified is false. A control that "
            "did not verify when it was trained does not become one by being "
            "imported.")

    binding = resolved["control_binding"]
    for key in ("observed_protocol_fingerprint", "probe_id",
                "checkpoint_weights_sha256"):
        if not binding.get(key):
            problems.append(f"control_binding is missing {key}")

    # --- the bytes on disk -------------------------------------------------
    weights = checkpoint_dir / "model.safetensors"
    if not weights.is_file():
        raise ControlImportError(
            f"{name}: no weights at {weights}; nothing was staged to verify")
    observed_weights = sha256_file(weights)
    if observed_weights != resolved["weights_sha256"]:
        problems.append(
            f"weights sha256 {observed_weights} does not match the recorded "
            f"{resolved['weights_sha256']}: these are not the same checkpoint")
    if observed_weights != binding.get("checkpoint_weights_sha256"):
        problems.append(
            "weights sha256 does not match the control_binding, so the staged "
            "bytes are not the ones the binding was issued for")

    # --- the protocol, re-derived rather than trusted ----------------------
    reconstructed = False
    protocol_fingerprint = resolved["observed_protocol_fingerprint"]
    reconstruction: dict[str, Any] = {"attempted": False}
    if run_evidence_dir is not None:
        run_evidence_dir = Path(run_evidence_dir)
        if (run_evidence_dir / "run_manifest.json").is_file():
            reconstruction["attempted"] = True
            try:
                observed = observe_recovery_protocol(
                    run_evidence_dir, repo_root=repo_root, strict=strict)
            except RecoveryAdmissionError as exc:
                problems.append(
                    f"the imported run evidence does not reconstruct: {exc}"[:400])
            else:
                reconstructed = True
                actual = observed.protocol.fingerprint
                reconstruction.update({
                    "fingerprint": actual,
                    "seed": observed.seed,
                    "initialization_source": observed.initialization_source,
                    "pack_recomputed": observed.evidence.get(
                        "pack_blocks_sha256_recomputed"),
                    "step_accounting": observed.evidence.get("step_accounting"),
                    "missing_fields": observed.evidence.get("missing_fields"),
                })
                if actual != protocol_fingerprint:
                    problems.append(
                        f"the run evidence reconstructs protocol {actual} but the "
                        f"record says {protocol_fingerprint}")
                if observed.seed != resolved["seed"]:
                    problems.append(
                        f"the run evidence records seed {observed.seed}, the "
                        f"control record says {resolved['seed']}")
    if not reconstruction["attempted"]:
        reconstruction["why_not"] = (
            "no run_manifest.json was staged for this control; the import rests "
            "on the recorded identity and the weights hash alone")
        if strict:
            problems.append(
                "no run evidence was staged, so the protocol could not be "
                "re-derived. Stage run_manifest.json and run_completion.json "
                "beside the checkpoint, or accept a weaker import explicitly.")

    if binding.get("observed_protocol_fingerprint") != protocol_fingerprint:
        problems.append("control_binding disagrees with the recorded protocol "
                        "fingerprint")
    if binding.get("probe_id") != resolved["probe_id"]:
        problems.append("control_binding disagrees with the recorded probe id")

    if problems:
        raise ControlImportError(
            f"{name}: the staged artifact is not the permanent control it claims "
            "to be:\n  - " + "\n  - ".join(problems))

    return ImportedControl(
        name=name, checkpoint_dir=checkpoint_dir,
        weights_sha256=observed_weights, probe_id=resolved["probe_id"],
        observed_protocol_fingerprint=protocol_fingerprint,
        seed=int(resolved["seed"]),
        initialization_artifact_digest=resolved["initialization_artifact_digest"],
        control_binding=binding,
        reconstructed_from_run_evidence=reconstructed,
        evidence={
            "record": str(record_path),
            "record_sha256": sha256_file(record_path),
            "weights": str(weights),
            "weights_verified_against": ["record", "control_binding"],
            "reconstruction": reconstruction,
            "rule": ("every field in IMPORT_REQUIRED_FIELDS is checked; where "
                     "run evidence exists the strict reconstruction is re-run "
                     "and must reproduce the recorded fingerprint"),
        })


def probe_identity_of(control: ImportedControl,
                      protocol) -> RecoveryProbeIdentity:
    """The imported control's probe identity, rebuilt from the imported protocol."""
    return RecoveryProbeIdentity(
        protocol=protocol,
        initialization_artifact_digest=control.initialization_artifact_digest,
        seed=control.seed, label=control.name)


#: The continuation plan. Same machinery as `PREFLIGHT_PLAN_V1`, different
#: predecessors: nothing here trains anything, and the only expensive stage is
#: the characterization the preflight never reached.
CONTINUATION_PLAN_V1 = PreflightPlan(
    plan_id="autoinit.control_characterization_continuation",
    version=1,
    stages=(
        PreflightStage(
            stage=0, name="import the permanent controls", blocking=True,
            purpose=("bind the staged bytes back to the recorded permanent "
                     "identity before anything is measured against them"),
            produces=("per-control weights sha256 verified against the record "
                      "and the control_binding",
                      "strict RecoveryProtocolFingerprint reconstruction from the "
                      "imported run evidence",
                      "probe id, seed and initialization digest re-established",
                      "import evidence artifact"),
            stop_conditions=(
                "a staged weights hash differs from the record -> STOP",
                "the control_binding disagrees with the record -> STOP",
                "a material identity field is missing from the record -> STOP",
                "the imported run evidence does not reconstruct the recorded "
                "protocol fingerprint -> STOP",
                "no run evidence was staged -> STOP: the protocol cannot be "
                "re-derived from a sidecar summary alone")),
        PreflightStage(
            stage=1, name="current evaluation attestation", blocking=True,
            purpose=("recovery_search_v2, the scoring digest and the generation "
                     "source have all moved since these controls were trained, "
                     "so characterization binds to the protocol attested now"),
            produces=("runtime fingerprint", "generation source digest",
                      "engine-observed generation fields",
                      "materialized RecoveryGenerationProtocolFingerprint",
                      "RecoveryEvaluationProtocol over recovery_search_v2",
                      "frozen attested protocol artifact"),
            stop_conditions=(
                "the frozen assets do not match their preregistered constants",
                "a generation field cannot be materialized -> STOP",
                "the scoring contract digest differs from the pinned one")),
        PreflightStage(
            stage=2, name="v2 tool and RAG generation smoke", blocking=True,
            purpose=("execute the real generation path on the sets that broke it "
                     "before any permanent control is touched"),
            produces=("rollouts for the tool and rag sets of recovery_search_v2",
                      "observed generation fingerprint reconstructed from them",
                      "comparison against the Stage-1 attestation"),
            stop_conditions=(
                "generation raises -> STOP, and the full output is kept",
                "the observed generation protocol differs from the attested one "
                "-> STOP: characterization would not be comparable",
                "the smoke must cover the tool set; a smoke that renders no tool "
                "prompt is the failure mode this stage exists for")),
        PreflightStage(
            stage=3, name="characterize the imported controls", blocking=False,
            purpose=("materialize the frozen thresholds from the two permanent "
                     "controls, under the current evaluation identity"),
            produces=("sa and sb on recovery_search_v2",
                      "observed generation reconstruction per control",
                      "pooled_counts@v2 aggregate + per-seed counts and rates",
                      "materialized equivalence interval and feasibility floor",
                      "per-capability control baselines",
                      "measured battery-evaluation cost for Phase-A repricing"),
            stop_conditions=(
                "capability schema validation fails -> scoring defect, STOP",
                "a result's evaluation protocol is not comparable to the "
                "attestation -> STOP")),
    ))


#: The executable that actually runs this session. It is NOT the preflight set:
#: the continuation has its own launcher, its own driver and its own plan
#: module, and an authorization that digested the preflight's files would admit
#: an edited continuation driver without noticing. The shared infrastructure —
#: setup script, engine probe, watchdog, collector, authorization, generation —
#: is in both sets because both sessions execute it.
CONTINUATION_HARNESS_SOURCE_FILES_V1: tuple[str, ...] = (
    "scripts/pod/autoinit_continuation_launch.py",
    "scripts/pod/autoinit_continuation_driver.py",
    "scripts/pod/autoinit_preflight_launch.py",   # the launcher it subclasses
    "scripts/pod/autoinit_preflight_setup.sh",
    "scripts/pod/autoinit_engine_probe.py",
    "scripts/pod/watchdog.py",
    "scripts/pod/collect_artifacts.py",
    "src/aadistill/autoinit/authorization.py",
    "src/aadistill/autoinit/continuation.py",
    "src/aadistill/autoinit/generation.py",
)

#: The narrow authorization this session runs under. Characterization only: it
#: trains nothing, and Phase A is not expressible in this artifact at all.
CONTINUATION_AUTHORIZATION = SpendAuthorization(
    authorization_id="autoinit.control_characterization.2026-08-14",
    granted_utc="2026-08-14T00:00:00Z",
    granted_by="maintainer (session authorization)",
    plan_id=CONTINUATION_PLAN_V1.plan_id,
    plan_hash=CONTINUATION_PLAN_V1.plan_hash,
    # Priced by the launcher's own `make_plan`, re-run 2026-08-15 against the
    # fully offline setup and a MEASURED setup allowance (11 min; see
    # SETUP_MINUTES). One session: expected $1.3860, soft $1.5246, hard $1.6896.
    #
    # The cap is CUMULATIVE across the continuation, as the maintainer requires,
    # so the whole effort is bounded rather than each retry separately:
    #
    #   attempt 1  cold host + a test gate reading an unstaged battery  $0.6312
    #   attempt 2  three consecutive cold hosts                         $0.6367
    #   attempt 3  uv sync cannot install a registry-pinned wheel       $0.0700
    #   attempt 4  train env offline in 11 s; vLLM hung 76 min on PyPI  $1.3672
    #              ------------------------------------------------------------
    #   spent, zero driver stages reached                               $2.7051
    #   one newly-priced hard attempt                                   $1.6896
    #                                                                   =======
    #   cumulative hard                                                 $4.3947 -> $4.40
    #   cumulative expected = 2.7051 + 1.3860                         =  $4.0911 -> $4.10
    #
    # Raising the cap does not loosen the session: `make_plan` still prices one
    # run at soft $1.5246 / hard $1.6896, so a single launch cannot spend the
    # headroom that covers the four failures before it.
    expected_usd=4.10,
    hard_cap_usd=4.40,
    authorized_stages=(0, 1, 2, 3),
    stage_conditions={
        "0": "strict import of the two existing permanent controls; no training",
        "1": "current generation and evaluation attestation over recovery_search_v2",
        "2": "real v2 tool and RAG generation smoke",
        "3": "characterize the imported controls; the only paid measurement",
        "teardown": "collect, delete the pod, confirm from the provider, STOP",
    },
    scope_note=(
        "characterization of two ALREADY TRAINED permanent controls. This "
        "authorization does not permit training, retraining, or re-initializing "
        "any control: the driver has no training stage and the launcher "
        "materializes existing artifacts. A failed characterization is collected "
        "and torn down and remains a FAILED continuation; it does not authorize "
        "a retry that trains. Phase A remains separately unauthorized."),
    harness_source_files=CONTINUATION_HARNESS_SOURCE_FILES_V1,
)


@dataclass(frozen=True)
class ContinuationScope:
    """What the continuation session is permitted to do, stated once."""

    trains_anything: bool = False
    retrains_controls: bool = False
    reaches_phase_a: bool = False
    controls: tuple[str, ...] = ("preflight_ctl_r0860k_sa", "preflight_ctl_r0860k_sb")
    battery: str = "recovery_search_v2"
    notes: dict[str, str] = field(default_factory=lambda: {
        "recovery_identity": ("imported, never re-attested: a control's protocol "
                              "was established when it was trained"),
        "evaluation_identity": ("attested fresh this session: the battery, the "
                                "scoring digest and the generation source have "
                                "all moved"),
        "why_not_the_preflight_plan": (
            "PreflightPlan.advance_to(3) requires same-session predecessors and "
            "is right to; a session that imports a checkpoint did not train it"),
    })

    def as_dict(self) -> dict[str, Any]:
        return {"trains_anything": self.trains_anything,
                "retrains_controls": self.retrains_controls,
                "reaches_phase_a": self.reaches_phase_a,
                "controls": list(self.controls), "battery": self.battery,
                "notes": dict(self.notes),
                "plan_hash": CONTINUATION_PLAN_V1.plan_hash}


CONTINUATION_SCOPE = ContinuationScope()


def continuation_manifest() -> dict[str, Any]:
    """The continuation plan and its scope, as one hashable record."""
    payload = {"plan": CONTINUATION_PLAN_V1.as_dict(),
               "scope": CONTINUATION_SCOPE.as_dict(),
               "import_required_fields": list(IMPORT_REQUIRED_FIELDS)}
    payload["manifest_sha256"] = sha256_json(payload)
    return payload
