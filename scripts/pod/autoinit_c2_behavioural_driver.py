#!/usr/bin/env python3
"""The Phase-C2 behavioural selection: twelve probes, two rungs, one decision.

    /opt/train/bin/python scripts/pod/autoinit_c2_behavioural_driver.py \
        --audit-dir <dir> --b-workdir <dir> --campaign <id> ...

Stages, in the order the frozen protocol fixes them:

    P  prepare  — verify the teacher, then MATERIALIZE all six arms along their
                  pinned paths, each gated on its exact recorded identity
    S  screen   — six fresh recovery probes: five candidates plus B, one seed
    R  rank     — rank by paired delta against B, advance exactly ONE
    C  confirm  — six fresh probes: the advanced candidate and B, three seeds
    D  decide   — apply the frozen Phase-C rule, and STOP

**Standalone, composing C1's proven primitives.** An earlier draft subclassed
`C1Driver`, which was wrong in four concrete ways rather than one stylistic one:
it inherited C1's authorization, plan identity, recovery seeds and audit roots,
all of which describe a different experiment; its `complete()` resolved stage
letters through C1's session registry, which has no P/S/R/C/D; its `stage_c`
silently overrode C1's stage C; and `materialize_b` called `self.load_teacher()`,
which does not exist. What is genuinely reusable is the *machinery* — the
trainer, the evaluator, the packaging, the admission gate, the scorers, the
paired inference — and that is called here directly. Constants that are FROZEN
ASSETS rather than C1 session state (the evaluation tokenizer and its pinned
sidecar hashes, the recovery recipe, the pack, the teacher binding) are imported
from their one owner rather than re-typed into this file.

**The six arms are built here, not shipped here.** Each is a 1.19 GB
checkpoint. The scp path gives an asset 600 seconds against a dev-box uplink
that needs 1650 for one of them — the arithmetic that killed continuation
attempt 2 staging exactly this size — and the hub relay refuses 5.95 GB for
private-storage quota, asked directly at $0 and refused twice. So every arm is
materialized from the teacher along a path pinned at EVERY step to the artifact
digest the frozen record holds, which is the mechanism the replay proved by
reproducing all five byte-for-byte. It is not a search: no beam, no expansion,
no ranking, no selection, and a digest mismatch stops the session as a
scientific finding.

**Screening never decides.** It ranks and advances exactly one candidate, on a
battery whose own scorer refuses to emit a verdict. Only the confirmation rung,
on three disjoint seeds, may name a C2 incumbent — and `NO_GO` and
`INCONCLUSIVE` are results, not failures to retry.

**Confirmation cannot start early.** All six screening probes must be trained
AND scored and the winner mechanically determined first; otherwise the candidate
being confirmed was chosen from whoever happened to finish.

**Finished work is announced for durability the moment it exists.** C1 attempt
17 trained six probes over ten hours and lost every one when a later stage
failed. This driver does not push bytes itself: it writes each completed unit's
identity to its evidence, and the launcher's poll hook pulls the bytes off-pod
and re-identifies them at the destination while the session is still running.
That route is used instead of C1's Hugging Face relay because the relay failed
under an account-wide storage quota and preserved nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
for _p in ("src", "scripts", "scripts/autoinit", "scripts/pod"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

#: Frozen assets and proven free functions, from their one owner. Importing the
#: C1 driver module is inert — it defines constants and registers the builtin
#: profiles and adapters, which this session needs too — and `C1Driver` itself
#: is deliberately NOT imported.
from autoinit_c1_driver import (  # noqa: E402
    C1_PROBE_OVERRIDES, ENGINE_PROBE, FROZEN_RECIPE, PACK_DIR, TEACHER_BINDING,
    TOKENIZER_SIDECAR_SHA256, TOKENIZER_SOURCE, TRAINER, UNCAPPED_EVAL,
    _trainer_bytes, trained_model_dir,
)

from aadistill.infrastructure.manifest import sha256_file, sha256_json  # noqa: E402
from aadistill.initialization.planning.generation import (  # noqa: E402
    RecoveryEvaluationProtocol, declared_generation_protocol,
    observe_generation_protocol,
)
from aadistill.runtime.device_handoff import (  # noqa: E402
    complete_release, cuda_memory, require_headroom, require_released,
)
from experiments.phase_c1.packaging import build_evaluation_package  # noqa: E402
from experiments.phase_c1.scoring import c1_scoring_contract  # noqa: E402
from experiments.phase_c2 import behavioural as BH  # noqa: E402
from experiments.phase_c2 import behavioural_governance as BG  # noqa: E402
from experiments.phase_c2 import behavioural_decision as BD  # noqa: E402
from experiments.phase_c2 import behavioural_schedule as SCH  # noqa: E402
from experiments.phase_c2 import behavioural_continuation as BC  # noqa: E402
from experiments.phase_c2 import scoring as C2S  # noqa: E402
from experiments.source_sets import generation_source_digest  # noqa: E402

RUN_ID = "autoinit.v1.phase_c2.behavioural"
EVIDENCE_FILENAME = "c2_behavioural_evidence.json"

SUCCESS_MARKER = "C2_BEHAVIOURAL_ALL_DONE"
FAILURE_MARKER = "C2_BEHAVIOURAL_FAILED"
B_READY_MARKER = "C2_BEHAVIOURAL_B_READY"
PROBE_MARKER = "C2_BEHAVIOURAL_PROBE_DONE"
DURABLE_MARKER = "C2_BEHAVIOURAL_DURABLE_UNIT"
ADVANCED_MARKER = "C2_BEHAVIOURAL_ADVANCED"

C2_SCREENING_SCORER = REPO / "scripts/autoinit/score_c2_screening.py"
C1_SCORER = REPO / "scripts/autoinit/score_c1_confirmation.py"

#: The stages, in order. Own registry: C1's session registry describes C1's
#: stages and resolving a C2 letter through it raises.
STAGES = ("P", "S", "R", "C", "D")


class C2DriverError(RuntimeError):
    """This session cannot proceed on the evidence it has."""


class C2ProvenanceError(RuntimeError):
    """A scientific finding, not a retryable engineering failure."""


def _rel(p) -> str:
    """Repo-relative when it can be, absolute otherwise.

    `relative_to` raises outside the root and the audit root is redirectable —
    the `$0` rehearsal points it at a tmp tree. A path recorded for provenance
    must never be the thing that fails a stage.
    """
    try:
        return str(Path(p).relative_to(REPO))
    except ValueError:
        return str(p)


def scorer_argv(probe, *, battery: Path, gen_dir: Path, out: Path,
                per_sample: Path, generation_fingerprint: str,
                run_completion: Path | None = None) -> list[str]:
    """Which scorer runs for this probe, and how the arm is named to it.

    A free function so the mapping can be checked at `$0` instead of only on a
    paid pod. It carries three decisions that are easy to get quietly wrong:

    * **which entry point.** The screening rung gets its own, because C1's pins
      its battery by equality on purpose — "the production path cannot be aimed
      anywhere else" — and the rung it guards is the one that may name an
      incumbent. Loosening that pin so screening could borrow the entry point
      would weaken a guard for a consumer that does not need it weakened.
    * **how the arm is named.** Confirmation maps B to `incumbent` and the
      advanced candidate to `treatment`, which is the direction the estimand is
      defined in. Screening has SIX arms and C1's incumbent/treatment pair does
      not describe them, so it passes `--screening-arm` instead and nothing
      downstream can read a screening result as half of a confirmation pair.
    * **which fingerprint.** The OBSERVED one, reconstructed from this probe's
      own summaries, never the attested one. They are equal by the admission
      gate; the direction of provenance is the point.
    """
    screening = probe.rung == "screening"
    argv = [str(C2_SCREENING_SCORER if screening else C1_SCORER),
            "--generations", str(gen_dir), "--label", probe.probe_id,
            "--seed", str(probe.seed), "--out", str(out),
            "--per-sample", str(per_sample),
            "--battery", str(battery),
            "--init-digest", probe.initialization_artifact_digest,
            "--generation-fingerprint", generation_fingerprint]
    if screening:
        argv += ["--screening-arm", probe.arm]
    else:
        argv += ["--arm", BD.INCUMBENT_ARM if probe.arm == SCH.ANCHOR
                 else BD.TREATMENT_ARM]
    if run_completion is not None and Path(run_completion).is_file():
        argv += ["--trained-run", str(run_completion)]
    return argv


def say(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)


class C2BehaviouralDriver:
    """Twelve probes, two rungs, one decision — and no inherited experiment."""

    def __init__(self, a) -> None:
        self.a = a
        self.t0 = time.time()
        self.audit = Path(a.audit_dir)
        (self.audit / "probes").mkdir(parents=True, exist_ok=True)
        self.status = Path(a.status_path)

        self.rule = BD.decision_rule(REPO)
        self.proto = BH.protocol(REPO)["behavioural_selection"]

        self.candidates: list[dict[str, Any]] = []
        self.anchor: dict[str, Any] = {}
        self.arm_arch: dict[str, tuple[str, int]] = {}
        #: Which arms' BYTES this session must rebuild. `None` until the
        #: continuation manifest is read, and `None` afterwards when there is
        #: no manifest — a fresh campaign owes every arm. It is the same set
        #: the launcher priced before a pod existed, so stage P's GPU work and
        #: the budget that permitted it describe the same arms.
        self.arms_needed: set[str] | None = None
        self.screening: list[SCH.Probe] = []
        self.confirmation: list[SCH.Probe] = []
        self.ranked: list[dict[str, Any]] = []
        self.advanced: dict[str, Any] | None = None
        self.evaluation_protocol: RecoveryEvaluationProtocol | None = None
        self.teacher_path: str | None = None

        #: Keyed by probe_id, never by (arm, seed): the same arm appears in both
        #: rungs and B appears on four seeds, so a key that collided would let
        #: one rung's probe satisfy the other's completeness check.
        self.training: dict[str, dict] = {}
        self.scores: dict[str, dict] = {}
        self.durable: list[dict] = []
        self.completed: list[str] = []

        self.ev: dict[str, Any] = {
            "schema": "aadistill.autoinit.c2_behavioural_evidence/v1",
            "run_id": RUN_ID,
            #: THREE identities, and they are not interchangeable. `run_id`
            #: above is the plan id. `campaign` is the scientific experiment and
            #: the scope within which a completed probe may be reused.
            #: `run_attempt` is this invocation and its provider resource.
            "campaign": a.campaign,
            "run_attempt": getattr(a, "run_attempt", "") or None,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "stages": {},
            "durable_units": [],
            "probes_trained": 0,
            "probes_scored": 0,
        }

    # -- bookkeeping --------------------------------------------------------
    def mark(self, name: str) -> None:
        line = f"{datetime.now(timezone.utc):%FT%TZ} MARKER:{name}"
        print(line, flush=True)
        self.status.parent.mkdir(parents=True, exist_ok=True)
        with self.status.open("a") as f:
            f.write(line + "\n")

    def usd(self) -> float:
        return self.a.spent_usd + (time.time() - self.t0) / 3600 * self.a.rate

    def afford(self, minutes: float, what: str) -> bool:
        """Admission control, not a trip-wire.

        Asks whether the *whole* unit fits before starting it. A check that only
        refuses once the budget is already gone would let a 30-minute probe
        start on two minutes of headroom and then be killed mid-training, which
        spends the money and produces nothing.
        """
        projected = self.usd() + minutes / 60 * self.a.rate
        if projected > self.a.soft_stop_usd:
            say(f"SOFT STOP: {what} needs ~{minutes:.0f} min "
                f"(${projected:.2f} > ${self.a.soft_stop_usd:.2f}) — not starting")
            return False
        if projected > self.a.authorized_usd:
            say(f"AUTHORIZATION: {what} would reach ${projected:.2f} against an "
                f"authorized ${self.a.authorized_usd:.2f} — not starting")
            return False
        return True

    def child_env(self) -> dict:
        return {**os.environ, "PYTHONPATH": f"{REPO}/src",
                "AADISTILL_IMAGE_DIGEST": self.a.image_digest}

    def save(self) -> None:
        self.ev["elapsed_min"] = round((time.time() - self.t0) / 60, 2)
        self.ev["spend_usd"] = round(self.usd(), 4)
        self.ev["stages_completed"] = list(self.completed)
        self.ev["durable_units"] = self.durable
        self.ev["probes_trained"] = len(self.training)
        self.ev["probes_scored"] = len(self.scores)
        (self.audit / EVIDENCE_FILENAME).write_text(
            json.dumps(self.ev, indent=2, default=str) + "\n")

    def gate(self, name: str, argv: list[str], *, timeout: float,
             python: str = "/opt/train/bin/python") -> subprocess.CompletedProcess:
        out = subprocess.run([python, *argv], capture_output=True, text=True,
                             timeout=timeout, env=self.child_env())
        (self.audit / f"{name}.log").write_text(
            f"$ {python} {' '.join(argv)}\nrc={out.returncode}\n"
            f"--- stdout ---\n{out.stdout}\n--- stderr ---\n{out.stderr}\n")
        return out

    def complete(self, letter: str, **payload) -> None:
        """Record a finished stage and refuse an out-of-order execution."""
        if letter not in STAGES:
            raise C2DriverError(f"{letter!r} is not a C2 stage {STAGES}")
        expected = STAGES[len(self.completed)]
        if letter != expected:
            raise C2DriverError(
                f"stage {letter} completed but {expected} was next; the "
                f"protocol fixes the order {STAGES} and a rung that ran early "
                "was not the preregistered experiment")
        self.completed.append(letter)
        self.ev["stages"][letter] = {
            "passed": True,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "spend_usd": round(self.usd(), 4), **payload}
        self.mark(f"STAGE_PASSED:{letter}")
        self.save()

    def fail(self, letter: str, reason: str, **payload) -> None:
        self.ev["stages"][letter] = {
            "passed": False, "reason": str(reason)[-2000:],
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "spend_usd": round(self.usd(), 4), **payload}
        self.mark(f"STAGE_FAILED:{letter}")
        self.save()
        say(f"STAGE {letter} FAILED: {reason}")

    def runtime_identity(self) -> dict:
        import torch
        import transformers

        return {"image_digest": self.a.image_digest,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "cuda_runtime": getattr(torch.version, "cuda", None),
                "gpu": (torch.cuda.get_device_name(0)
                        if torch.cuda.is_available() else None),
                "driver": os.environ.get("NVIDIA_DRIVER_VERSION")}

    # -- continuation across a replacement resource -------------------------
    def restore_campaign(self) -> dict[str, Any]:
        """Adopt this campaign's restored probes. BEFORE the journal is read.

        A replacement resource has a fresh filesystem: nothing of the previous
        pod's audit directory exists and every `model_dir` it recorded points
        at nothing. The launcher has put the campaign's destination-verified
        probes under `--continuation-manifest`'s `pod_root`, and this turns
        them back into journal entries — at the NEW local paths, after
        re-identifying the bytes HERE.

        The re-identification is not a formality. The manifest is a document,
        the ack behind it is a record, and neither is a checkpoint; the only
        thing that makes a restored probe a measurement is that its bytes
        reproduce the identity its own campaign announced, on this machine, at
        the path this process will read. A mismatch RAISES: a continuation that
        admitted unverifiable bytes would be substituting for a measurement,
        which is the one failure the whole campaign contract exists to prevent.

        The committed screening ranking travels the same way, so a continuation
        confirms the candidate its campaign advanced rather than re-deciding.
        """
        path = Path(self.a.continuation_manifest or "")
        if not self.a.continuation_manifest or not path.is_file():
            #: No manifest at all. Distinguished from an empty one: the
            #: launcher always ships a manifest, so its absence means the
            #: restore step never ran and this is not a continuation.
            summary = {"manifest": None, "restored": [], "n": 0,
                        "arms_needed": None,
                        "_means": ("no continuation manifest; nothing to adopt "
                                   "and every arm is owed")}
            self.ev["continuation"] = summary
            return summary

        manifest = json.loads(path.read_text())
        if manifest.get("campaign_id") != self.a.campaign:
            raise C2DriverError(
                f"{path} is a continuation manifest for campaign "
                f"{manifest.get('campaign_id')!r}, not {self.a.campaign!r}. "
                "One experiment's probes may never be pooled into another.")

        #: THE ARMS THE LAUNCHER PRICED. Read before anything is restored, so
        #: stage P cannot build work the pre-provider budget did not fund. A
        #: manifest that names none is a fresh campaign in all but name and
        #: leaves every arm owed.
        remaining = manifest.get("remaining") or {}
        #: PRESENCE, not truthiness. An EMPTY list means this campaign owes no
        #: arm at all — every remaining probe is a restored checkpoint waiting
        #: to be scored — and treating that as "no manifest" would rebuild all
        #: six for work the budget did not fund and nothing will read.
        self.arms_needed = (set(remaining["arms_needed"])
                            if "arms_needed" in remaining
                            and remaining["arms_needed"] is not None
                            else None)

        (self.audit / "probes").mkdir(parents=True, exist_ok=True)
        restored: list[dict[str, Any]] = []
        for entry in manifest.get("probes", []):
            pid = entry["probe_id"]
            pod_path = Path(entry["pod_path"])
            identity = entry["identity"]
            #: FROM THE BYTES, at the path this pod will read them from.
            try:
                from aadistill.initialization.specs.arch import get_adapter
                from aadistill.runtime.leaf_durability import (
                    verify_transferred_leaf,
                )

                v = verify_transferred_leaf(pod_path, identity,
                                            adapter=get_adapter("qwen3"))
                ok = bool(v["matched"] and v["weights_digest_matched"]
                          and v["shard_matched"]
                          and v["config_matched"] is not False)
                why = "re-identified on this pod" if ok else (
                    f"artifact={v['matched']}, "
                    f"weights={v['weights_digest_matched']}, "
                    f"shard={v['shard_matched']}, config={v['config_matched']}")
            except Exception as exc:                              # noqa: BLE001
                ok, why = False, f"{type(exc).__name__}: {exc}"
            if not ok:
                raise C2DriverError(
                    f"{pid}: the restored bytes at {pod_path} do not reproduce "
                    f"the identity this campaign announced ({why}). A "
                    "continuation may consume its campaign's completed probes; "
                    "it may not accept bytes that cannot be shown to be them.")

            #: A journal entry with the NEW model_dir. The producing pod's
            #: absolute path is deliberately not carried in the manifest and is
            #: not reconstructed here: it does not exist on this machine, and a
            #: path that cannot be checked is not evidence.
            record = {
                "probe_id": pid, "campaign": self.a.campaign,
                "rung": entry.get("rung"), "arm": entry.get("arm"),
                "seed": entry.get("seed"),
                "model_dir": str(pod_path),
                "initialization_artifact_digest":
                    entry.get("initialization_artifact_digest"),
                "config_sha256": entry.get("config_sha256"),
                "complete": True,
                "restored_from": {
                    "campaign": manifest["campaign_id"],
                    "source_attempt": entry.get("source_attempt"),
                    "durable_path": entry.get("durable_path"),
                    "re_identified_on_this_pod": True,
                },
                "durable": {"identity": identity},
            }
            if entry.get("score"):
                #: The score's own evidence, restored beside it and re-pointed
                #: at this pod. The verdict reads the per-sample ROWS, so a
                #: score whose rows did not travel is not a usable score.
                score = self._restore_score_evidence(pid, pod_path,
                                                     dict(entry["score"]))
                record["score"] = score
            (self.audit / "probes" / f"{pid}.json").write_text(
                json.dumps(record, indent=2, default=str) + "\n")
            restored.append({"probe_id": pid, "pod_path": str(pod_path),
                             "scored": bool(entry.get("score"))})
            say(f"  restored {pid}: re-identified here, "
                f"{'scored' if entry.get('score') else 'TRAINED ONLY'}")

        ranking = manifest.get("committed_ranking")
        if ranking:
            #: The campaign's commitment, written where `committed_ranking`
            #: looks. Stage R will recompute, refuse on disagreement, and
            #: advance THIS candidate.
            (self.audit / "c2_screening_ranking.json").write_text(
                json.dumps(ranking, indent=2) + "\n")
            say(f"  campaign already advanced "
                f"{manifest['committed_candidate'][:12]}…; screening will not "
                "be re-decided")

        summary = {
            "manifest": str(path), "restored": restored, "n": len(restored),
            "committed_candidate": manifest.get("committed_candidate"),
            "arms_needed": (None if self.arms_needed is None
                            else sorted(self.arms_needed)),
            "_re_identified_here": (
                "every restored probe's bytes reproduced the identity its "
                "campaign announced, at the path on THIS pod. The manifest and "
                "the destination ack are records; the bytes are the evidence."),
        }
        self.ev["continuation"] = summary
        return summary

    def _restore_score_evidence(self, pid: str, pod_path: Path,
                                score: dict[str, Any]) -> dict[str, Any]:
        """Re-point a restored score at this pod's copies, and hash-check them.

        The score record's `result_path` and `per_sample_path` were the
        PRODUCING pod's. They are rewritten to the restored copies and verified
        against the hashes the score itself carries, so a truncated per-sample
        file is caught here rather than inside the decision rule.
        """
        moved: dict[str, str] = {}
        for key, name in (("result_path", "result.json"),
                          ("per_sample_path", "per_sample.jsonl")):
            src = pod_path / name
            if not src.is_file():
                raise C2DriverError(
                    f"{pid}: the restored score names {key} but {src} did not "
                    "arrive. The verdict reads these rows; a score without "
                    "them is not a usable score.")
            dest = self.audit / f"{pid}_{name}"
            dest.write_bytes(src.read_bytes())
            want = score.get(key.replace("_path", "_sha256"))
            got = sha256_file(dest)
            if want and want != got:
                raise C2DriverError(
                    f"{pid}: the restored {name} hashes to {got[:12]}… and the "
                    f"score records {want[:12]}…. The evidence changed in "
                    "transit and cannot decide anything.")
            moved[key] = str(dest)
        score.update(moved)
        return score

    # -- resume -------------------------------------------------------------
    def load_campaign_journal(self) -> dict[str, Any]:
        """Restore probes this CAMPAIGN already completed. Pre-registered rules.

        A probe may be reused only when all of this holds:

        * it was recorded under THIS campaign id. A replacement pod is a new
          RESOURCE and a new run attempt inside the same campaign, so it may
          continue with that campaign's completed probes; a probe from a
          DIFFERENT campaign is a different experiment and may never be pooled
          in, because the executed design would then be something nobody
          preregistered and the mixture would be invisible in the result.
        * its model directory is still present AND its identity still matches
          what was announced, re-derived from the bytes on disk. An entry whose
          weights are gone or changed is not a completed probe, it is a claim
          about one — which is how a replacement resource could substitute for
          a measurement.
        * it carries a score. A trained-but-unscored probe resumes at scoring,
          not at the verdict.

        A restored probe is NEVER retrained. Retraining a completed probe and
        keeping whichever result one prefers is the failure this rule exists to
        prevent, and it does not become acceptable because the first result was
        disappointing.
        """
        restored, rejected = [], []
        for path in sorted((self.audit / "probes").glob("*.json")):
            try:
                entry = json.loads(path.read_text())
            except json.JSONDecodeError as exc:
                rejected.append({"path": str(path), "why": f"unreadable: {exc}"})
                continue
            name = entry.get("probe_id") or path.stem
            if entry.get("campaign") and entry["campaign"] != self.a.campaign:
                rejected.append({"probe_id": name, "why": (
                    f"belongs to campaign {entry['campaign']!r}, not "
                    f"{self.a.campaign!r}; probes are not pooled across "
                    "experiments")})
                continue
            ok, why = self.reidentify(entry)
            if not ok:
                rejected.append({"probe_id": name, "why": why})
                continue
            self.training[name] = entry
            if entry.get("score"):
                self.scores[name] = entry["score"]
            restored.append({"probe_id": name, "scored": bool(entry.get("score"))})
        summary = {"restored": restored, "rejected": rejected,
                   "campaign": self.a.campaign}
        self.ev["resume"] = summary
        if restored or rejected:
            say(f"  resume: {len(restored)} probe(s) restored, "
                f"{len(rejected)} rejected")
        return summary

    def committed_ranking(self, path: Path) -> dict[str, Any] | None:
        """This CAMPAIGN's already-committed screening ranking, or None.

        A record from another campaign is not this campaign's commitment and is
        refused rather than ignored: reading it would let one experiment's
        selection decide another's confirmation, and silently ignoring it would
        leave a foreign ranking sitting in this run's evidence.
        """
        if not path.is_file():
            return None
        record = json.loads(path.read_text())
        if record.get("campaign") != self.a.campaign:
            raise C2DriverError(
                f"{path} holds a screening ranking committed under campaign "
                f"{record.get('campaign')!r}, not {self.a.campaign!r}. One "
                "experiment's selection may not decide another's confirmation.")
        if not (record.get("advanced") or {}).get("state_id"):
            raise C2DriverError(
                f"{path} exists but names no advanced candidate. A ranking "
                "record that cannot say what advanced is not a commitment, and "
                "overwriting it would destroy the only evidence of what the "
                "previous attempt did.")
        return record

    def assert_reuse_matches(self, probe: SCH.Probe) -> None:
        """A restored probe must BE the probe this rung asked for.

        Continuation inside one campaign is permitted; substitution is not. The
        journal's own record is self-consistent by construction, so the check
        that means anything compares it to the descriptor the schedule built
        from the frozen protocol and the six materialized arms: same rung, same
        arm, same seed, same initialization digest.

        An initialization digest that differs is the sharpest case. It would
        mean a probe trained from a different checkpoint is standing in for this
        arm's measurement — and every field here is derivable from the frozen
        protocol, so a mismatch is never ambiguous.
        """
        record = self.training.get(probe.probe_id) or {}
        for field, want in (
                ("rung", probe.rung), ("arm", probe.arm), ("seed", probe.seed),
                ("initialization_artifact_digest",
                 probe.initialization_artifact_digest)):
            got = record.get(field)
            if got != want:
                raise C2DriverError(
                    f"{probe.probe_id}: the restored probe records "
                    f"{field}={got!r} and this rung's descriptor requires "
                    f"{want!r}. A continuation may consume its campaign's "
                    "completed probes; it may not substitute a different "
                    "measurement for one of them.")
        if not record.get("config_sha256"):
            raise C2DriverError(
                f"{probe.probe_id}: the restored probe records no "
                "config_sha256, so the training configuration it ran under "
                "cannot be identified. An unidentifiable probe is not a "
                "completed one.")

    def reidentify(self, entry: Mapping[str, Any]) -> tuple[bool, str]:
        """Do the bytes on disk still ARE the probe this entry describes?"""
        durable = (entry.get("durable") or {})
        identity = durable.get("identity")
        if not identity:
            return False, "no recorded identity to check the bytes against"
        model_dir = Path(entry.get("model_dir") or "")
        if not model_dir.is_dir():
            return False, f"{model_dir} is gone; a record is not a checkpoint"
        try:
            from aadistill.initialization.specs.arch import get_adapter
            from aadistill.runtime.leaf_durability import identify_for_transfer

            now = identify_for_transfer(
                model_dir, adapter=get_adapter("qwen3"),
                arch_signature=identity["arch_signature"],
                num_parameters=identity["num_parameters"])
        except Exception as exc:                                  # noqa: BLE001
            return False, f"cannot re-identify: {type(exc).__name__}: {exc}"
        if now.artifact_digest != identity["artifact_digest"]:
            return False, (
                f"the bytes hash to {now.artifact_digest[:12]}… but the record "
                f"says {identity['artifact_digest'][:12]}…")
        return True, "re-identified"

    # -- durability ---------------------------------------------------------
    def announce_durable(self, unit_id: str, model_dir: Path, *,
                         kind: str, arch_signature: str, num_parameters: int,
                         **extra) -> dict:
        """Publish a finished unit's identity so the launcher can secure it.

        Not a push. This driver runs on the pod, and bytes written beside the
        process that made them are a copy that dies with the pod — which is
        exactly how C1 attempt 17 lost six probes. The launcher polls this
        evidence, pulls each announced unit off-pod, and re-identifies it from
        the bytes that land; the announcement is the identity that
        re-identification is checked against.

        The identity is rebuilt from the bytes on disk, not copied from
        whatever record asked for the announcement, so a probe whose weights
        were truncated on the way to disk is caught here rather than at the
        destination. It is built by `identify_for_transfer`, the same
        construction the destination re-identification uses, so the two cannot
        disagree about bookkeeping and report it as corruption.

        `arch_signature` and `num_parameters` come from the INITIALIZATION this
        unit was trained from. No file in a checkpoint carries them, and
        training changes weights rather than architecture, so the arm's own
        values are the right ones and are passed in rather than guessed.

        PRESERVATION IS NOT PERMISSION. Announcing a probe authorizes nothing
        about reusing it: whether a preserved checkpoint may be pooled, resumed
        or reused across attempts is a separate scientific decision, recorded in
        the payload so a later reader cannot mistake the one for the other.

        NEVER RAISES. A durability failure must not destroy the training it
        exists to protect.
        """
        payload: dict[str, Any] = {
            "unit_id": unit_id, "kind": kind, "campaign": self.a.campaign,
            "run_attempt": getattr(self.a, "run_attempt", "") or None,
            "announced_utc": datetime.now(timezone.utc).isoformat(),
            "path": str(model_dir), "identity": None, "identity_error": None,
            "authorizes": ("nothing. Preservation and reuse are separate "
                           "decisions: this checkpoint may not be pooled, "
                           "resumed or reused across formal attempts without "
                           "an explicit retry contract."),
            **extra,
        }
        try:
            from aadistill.initialization.specs.arch import get_adapter
            from aadistill.runtime.leaf_durability import identify_for_transfer

            d = Path(model_dir)
            ident = identify_for_transfer(
                d, adapter=get_adapter("qwen3"), arch_signature=arch_signature,
                num_parameters=num_parameters)
            #: Every field the destination re-identification compares. Recorded
            #: together, because a destination check that can only compare the
            #: fields it happens to find is not a check.
            payload["identity"] = {
                "artifact_digest": ident.artifact_digest,
                "weights_digest": ident.weights_digest,
                "config_sha256": ident.config_sha256,
                "single_shard_sha256": ident.single_shard_sha256,
                "arch_signature": ident.arch_signature,
                "num_parameters": ident.num_parameters,
                "tokenizer_sha256": ident.tokenizer_sha256,
                "total_bytes": ident.total_bytes,
            }
            (d / "c2_durable_unit.json").write_text(
                json.dumps(payload, indent=1) + "\n")
            self.mark(f"{DURABLE_MARKER}:{unit_id}")
            say(f"  {unit_id}: announced {ident.total_bytes / 2**30:.2f} GiB, "
                f"digest {ident.artifact_digest[:12]}…")
        except Exception as exc:                                  # noqa: BLE001
            payload["identity_error"] = f"{type(exc).__name__}: {exc}"
            say(f"  {unit_id}: NOT announced — {payload['identity_error']}")
        self.durable.append(payload)
        self.save()
        return payload

    # -- P: build the six arms, because none of them can be shipped here ----
    #: What a candidate's `durable_path` says when its BYTES were not built.
    #: Deliberately not an empty string: a path that is merely falsy gets
    #: passed to a loader by something that forgot to check, and this one names
    #: its own reason in any traceback that reaches it.
    NOT_MATERIALIZED = "<arm not materialized: no remaining probe needs it>"

    def arm_is_needed(self, arm: str) -> bool:
        """Does a REMAINING probe need this arm's bytes rebuilt?

        `self.arms_needed` comes from the continuation manifest, which the
        launcher derived from the campaign's verified state before a pod
        existed and PRICED. `None` means no manifest — a fresh campaign, or the
        `$0` rehearsal — and then every arm is needed.

        This is the parity that was missing: `remaining_work` charged a
        continuation for only the arms its remaining probes need, while this
        stage rebuilt all six unconditionally. The budget could therefore be
        below the GPU work the driver would actually do, which is the one
        direction a budget must never be wrong in.
        """
        return self.arms_needed is None or arm in self.arms_needed

    def require_materialized_arm(self, probe: SCH.Probe) -> None:
        """Refuse to train from an arm whose bytes were never built."""
        path = probe.initialization_path
        if path == self.NOT_MATERIALIZED or not path:
            raise C2DriverError(
                f"{probe.probe_id} is owed training from arm {probe.arm}, "
                "whose bytes this session did not materialize because the "
                "continuation manifest did not list it among the arms the "
                "remaining probes need. The budget and the work disagree; "
                "training from a path that was never built is not the "
                "alternative.")

    def stage_p(self) -> None:
        """Materialize the arms the remaining probes need, each digest-gated.

        Not a search. Each arm is a FIXED path whose every step carries the
        artifact digest attempt 3 recorded, and a mismatch stops the session as
        a scientific finding rather than being retried.

        The candidates are built rather than staged because neither transport
        can carry them: the scp path gives each asset 600 seconds against a
        dev-box uplink that needs 1650 for one of them — the arithmetic that
        killed continuation attempt 2 — and the hub relay refuses 5.95 GB for
        private-storage quota, measured at `$0` twice. B is built for a
        different reason: its bytes no longer exist. One mechanism for all of
        them, each identity-gated at the moment it is built.

        **A fresh campaign builds all six.** A continuation builds only the
        arms its remaining probes need, which is what its budget paid for. The
        frozen METADATA of every candidate is assembled either way — the
        schedule, the ranking and the tie-break read state ids, digests and the
        frozen ordering, none of which needs bytes — so a continuation that
        rebuilds two arms still ranks over the same five candidates in the same
        frozen order.
        """
        self.mark("STAGE_START:P")
        self.verify_teacher()

        #: The tie-break is "the frozen full-search ordering", so the rank must
        #: come from the selection document rather than from the order a
        #: builder happened to return. Asserted, not assumed: a silent
        #: reordering would change which candidate advances on a tie.
        from experiments.phase_c2.replay_specs import load_selection

        order = [s["state_id"] for s in load_selection(REPO)["selected"]]
        leaves = BG.candidate_leaves(REPO, device=self.a.device)
        if [leaf.state_id for leaf in leaves] != order:
            raise C2ProvenanceError(
                f"the builder returned {[l.state_id[:8] for l in leaves]} but "
                f"the frozen selection orders {[s[:8] for s in order]}. The "
                "screening tie-break is that ordering, so the two must agree.")

        built: list[str] = []
        for leaf in leaves:
            if self.arm_is_needed(leaf.state_id):
                path = self.materialize_arm(
                    leaf.state_id, leaf.spec,
                    required={"artifact_digest": leaf.artifact_digest,
                              "weights_digest": leaf.weights_digest,
                              "single_shard_sha256": leaf.single_shard_sha256,
                              "arch_signature": leaf.arch_signature,
                              "num_parameters": leaf.num_parameters},
                    bounded_minutes=float(leaf.bounded_minutes),
                    config_overrides=leaf.root_config_overrides)
                built.append(leaf.state_id)
            else:
                path = self.NOT_MATERIALIZED
            #: METADATA for every candidate, built or not. The frozen identity
            #: and the frozen rank come from the record, not from the bytes.
            self.candidates.append({
                "state_id": leaf.state_id,
                "artifact_digest": leaf.artifact_digest,
                "weights_digest": leaf.weights_digest,
                "arch_signature": leaf.arch_signature,
                "num_parameters": leaf.num_parameters,
                "durable_path": path,
                "rank_in_frozen_selection": len(self.candidates),
                "path_label": leaf.path_label,
                "materialized": path != self.NOT_MATERIALIZED,
            })
        say(f"  {len(built)} of {len(self.candidates)} candidate arms "
            f"materialized and identity-gated"
            + ("" if self.arms_needed is None else
               " (only those the remaining probes need)"))

        binding = BH.b_binding(REPO, device=self.a.device)
        say(f"  B construction {binding['construction']['spec_hash'][:12]}… == "
            f"frozen {binding['construction']['expected_spec_hash'][:12]}…")
        if self.arm_is_needed(SCH.ANCHOR):
            b_path = self.materialize_b(binding)
            materialized = True
        else:
            b_path, materialized = self.NOT_MATERIALIZED, False

        required = binding["required_identity"]
        self.anchor = {
            "state_id": SCH.ANCHOR,
            "artifact_digest": required["artifact_digest"],
            "durable_path": b_path,
            "arch_signature": required["arch_signature"],
            "num_parameters": required["num_parameters"],
        }
        #: A probe's architecture is its INITIALIZATION's — training changes
        #: weights, not shape — and no checkpoint file carries `arch_signature`
        #: or `num_parameters`. Held per arm so each announced probe can be
        #: re-identified at the destination against all six identity fields.
        self.arm_arch = {c["state_id"]: (c["arch_signature"],
                                         c["num_parameters"])
                         for c in self.candidates}
        self.arm_arch[SCH.ANCHOR] = (required["arch_signature"],
                                     required["num_parameters"])
        self.mark(B_READY_MARKER)
        self.complete("P", candidates=len(self.candidates),
                      arms_materialized=sorted(
                          built + ([SCH.ANCHOR] if materialized else [])),
                      arms_needed=(None if self.arms_needed is None
                                   else sorted(self.arms_needed)),
                      b_spec_hash=binding["construction"]["spec_hash"],
                      b_artifact_digest=self.anchor["artifact_digest"],
                      b_materialized=materialized)

    def verify_teacher(self) -> None:
        """The teacher, by shard hash. Shared frozen binding, not a C1 asset."""
        import hashlib

        from huggingface_hub import snapshot_download

        binding = json.loads(TEACHER_BINDING.read_text())
        local = snapshot_download(binding["repo_id"],
                                  revision=binding["revision"])
        bad = []
        for name, want in binding["expected_shard_sha256"].items():
            p = Path(local) / name
            if not p.is_file():
                bad.append(f"{name}: absent after fetch")
                continue
            got = hashlib.sha256(p.read_bytes()).hexdigest()
            if got != want:
                bad.append(f"{name}: {got} != {want}")
        if bad:
            raise C2DriverError("teacher verification FAILED: " + "; ".join(bad))
        self.teacher_path = local
        say(f"  teacher {binding['repo_id']}@{binding['revision'][:12]} verified, "
            f"{len(binding['expected_shard_sha256'])} shards")

    def reuse_arm(self, label: str, required: dict[str, Any]) -> str | None:
        """An arm already on disk, IF its bytes are the arm this path names.

        Returns the path when the checkpoint present re-identifies to the
        pinned `artifact_digest`, and `None` otherwise — including when it is
        absent, unreadable, or present with a different digest. A mismatch here
        is not an error: it means this directory holds something else, and the
        caller simply builds. What it must never do is return a path it has not
        checked.
        """
        for candidate in (Path(self.a.b_workdir) / label / "model",
                          Path(self.a.b_workdir) / label):
            if not (candidate / "config.json").is_file():
                continue
            try:
                from aadistill.initialization.specs.arch import get_adapter
                from aadistill.runtime.leaf_durability import (
                    identify_for_transfer,
                )

                ident = identify_for_transfer(
                    candidate, adapter=get_adapter("qwen3"),
                    arch_signature=required["arch_signature"],
                    num_parameters=required["num_parameters"])
            except Exception:                                     # noqa: BLE001
                continue
            if ident.artifact_digest == required["artifact_digest"]:
                return str(candidate)
        return None

    def materialize_arm(self, label: str, spec, *, required: dict[str, Any],
                        bounded_minutes: float,
                        config_overrides: Any = None) -> str:
        """Build one arm along its pinned path, then gate it on its identity.

        The one place an arm comes into existence, used by all six. A mismatch
        is a PROVENANCE finding and stops the session: the path is
        deterministic, so retrying would diverge identically, and an arm that is
        not the one the protocol names makes every delta measured against it
        meaningless.

        Only the fields the record actually carries are compared. A candidate
        leaf records no `config_sha256`, and comparing against a missing value
        would fail every arm for a field nobody recorded — which is a check
        that refuses correct work.
        """
        from aadistill.initialization.planning.fixed_path import (
            materialize_fixed_path,
        )
        from aadistill.initialization.specs.arch import get_adapter

        if self.teacher_path is None:
            raise C2DriverError(
                f"the teacher must be verified before {label} is built")
        if not self.afford(bounded_minutes, f"materializing {label}"):
            raise C2DriverError(
                f"budget refuses to materialize {label}; an arm that does not "
                "exist cannot be measured and no probe may start without all six")

        adapter = get_adapter("qwen3")
        workdir = Path(self.a.b_workdir) / label
        workdir.mkdir(parents=True, exist_ok=True)

        #: Resume rule R6: reuse an arm already on disk ONLY if its bytes
        #: re-identify to the digest this path is pinned to. Rebuilding all six
        #: costs ~156 minutes, so a restart that has them is worth honouring —
        #: but a copy that merely exists proves nothing, and trusting one is the
        #: single place a substituted initialization could enter the experiment.
        existing = self.reuse_arm(label, required)
        if existing is not None:
            say(f"  {label[:12]}… reused: {existing}")
            return existing

        #: The evidence-bound root pin, applied IN THE LOADER — which is where
        #: it belongs and the only place it can go: `materialize_fixed_path`
        #: takes no config-override parameter. The search reused one teacher
        #: object across all 108 expansions and `DepthCausalKLGreedyV1` leaves
        #: `use_cache = False` on the model it is handed, so paths expanded
        #: after any causal-KL DEPTH step began from a mutated root and that
        #: flag is serialized into every descendant's config, and therefore
        #: into its `artifact_digest`. A loader taking the hub default cannot
        #: reproduce those paths. Derived per path from attempt 3's own
        #: recorded step-0 config hash, never keyed on state ids.
        #: The root comes from the SPEC's own `root_repo_id`/`root_revision`,
        #: at the SPEC's dtype — byte-for-byte the loader that reproduced all
        #: five candidates in the replay. Reading the teacher from a local
        #: snapshot path or letting the dtype default instead would be a
        #: different root, and the digests would diverge four steps later with
        #: nothing in the record pointing at the cause. The verified teacher is
        #: checked to BE that root rather than substituted for it.
        def loader(_spec=spec, _ov=dict(config_overrides or {})):
            import torch
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                _spec.root_repo_id, dtype=torch.bfloat16,
                revision=_spec.root_revision).to(self.a.device).eval()
            for key, value in _ov.items():
                setattr(model.config, key, value)
            return model

        pinned = json.loads(TEACHER_BINDING.read_text())
        if (spec.root_repo_id, spec.root_revision) != (pinned["repo_id"],
                                                       pinned["revision"]):
            raise C2ProvenanceError(
                f"{label} is pinned to root {spec.root_repo_id}@"
                f"{spec.root_revision[:12]} but the verified teacher binding is "
                f"{pinned['repo_id']}@{pinned['revision'][:12]}. The root is "
                "part of the path's identity; building from a different one "
                "would diverge at the first step.")

        t0 = time.time()
        results = materialize_fixed_path(
            spec, adapter=adapter, root_loader=loader, workdir=workdir,
            repo_root=REPO)
        final = results[-1]

        observed = final.identity
        mismatched = [
            f"{field}: required {required[field]!r}, built "
            f"{getattr(observed, field)!r}"
            for field in ("artifact_digest", "weights_digest", "config_sha256",
                          "single_shard_sha256", "arch_signature",
                          "num_parameters")
            if required.get(field) is not None
            and getattr(observed, field) != required[field]
        ]
        if mismatched:
            raise C2ProvenanceError(
                f"the rebuilt {label} is not the arm the protocol names — "
                + "; ".join(mismatched)
                + ". NO PROBE MAY START. This is a provenance finding, not a "
                  "retryable engineering failure: the path is deterministic "
                  "and a retry would diverge identically.")
        say(f"  {label[:12]}… built and identity-gated in "
            f"{(time.time() - t0) / 60:.1f} min")
        self.announce_durable(
            label, Path(final.checkpoint_path), kind="arm_initialization",
            arch_signature=required["arch_signature"],
            num_parameters=required["num_parameters"])
        #: AFTER the gate and AFTER the announcement, never before: a
        #: provenance finding must keep every intermediate it diverged through,
        #: and a verified arm must be durable before anything is removed.
        released = self.release_intermediates(label, results, workdir)
        #: FAIL CLOSED HERE, at the caller. The helper stays non-raising — a
        #: cleanup error must not destroy a verified, announced arm — but the
        #: 120 GB provision is only correct BECAUSE these are released. If a
        #: release failed, the lifecycle assumption the bound rests on has been
        #: falsified, and continuing would build the next arm under a storage
        #: bound that no longer describes the program. attempt3 is what that
        #: looks like when it is discovered five arms later: ENOSPC, 32.8
        #: minutes of completed compute thrown away, and no verdict.
        if released["failed"]:
            raise C2DriverError(
                f"{label}: the arm is built, identity-gated and announced, but "
                f"its intermediates could not be released — {released['failed']}"
                ". The 120 GB provision is derived on the assumption that each "
                "arm's construction intermediates are freed when it is "
                "verified, so that assumption has now been falsified and the "
                "storage bound no longer describes this session. NO FURTHER "
                "ARM MAY BE BUILT under a bound that does not hold. The "
                "verified arm and its evidence are preserved.")
        return str(final.checkpoint_path)

    def probe_local_need_bytes(self) -> dict[str, Any]:
        """Local bytes ONE probe needs: its transient set plus what it retains.

        Derived from the recipe the trainer runs under, through the generic
        footprint -- not a constant, and not this model's parameter count. A
        different dtype, optimizer or `keep_last` moves it without an edit.
        """
        from aadistill.runtime import cost as COST

        tr = BH.training_dtypes(REPO)
        params = int(self.required_identity["num_parameters"]
                     if getattr(self, "required_identity", None)
                     else BH.candidate_manifest(REPO)[0]["num_parameters"])
        ck = COST.CheckpointFootprint(
            params, tr["save_dtype"],
            extra_bytes=int(tr["checkpoint_extra_bytes"]))
        transient = COST.training_working_set_bytes(
            params, weight_dtype=tr["weight_dtype"],
            grad_dtype=tr["grad_dtype"], moment_dtype=tr["moment_dtype"],
            n_moments=tr["n_moments"])
        retained = (1 + int(tr["keep_last"])) * ck.bytes
        return {"transient_bytes": transient, "retained_bytes": retained,
                "need_bytes": transient + retained,
                "checkpoint": ck.as_dict(), "keep_last": tr["keep_last"]}

    def require_probe_headroom(self, name: str) -> dict[str, Any]:
        """Refuse BEFORE training if the disk cannot hold this probe.

        The measurement the model is not. `storage_requirement` bounds the
        PROVISION; this bounds the next unit of work against what the
        filesystem actually reports, so a wrong derivation produces a refusal
        with every completed probe durable rather than an ENOSPC halfway
        through a checkpoint write.
        """
        from aadistill.runtime.leaf_durability import free_bytes_at

        need = self.probe_local_need_bytes()
        free = free_bytes_at(self.a.b_workdir)
        rec = {"probe": name, "free_bytes": free,
               "free_gib": round(free / 2**30, 3),
               "need_gib": round(need["need_bytes"] / 2**30, 3), **need}
        self.ev.setdefault("probe_headroom", []).append(rec)
        if free < need["need_bytes"]:
            raise C2DriverError(
                f"{name}: {free / 2**30:.2f} GiB free where this probe needs "
                f"{need['need_bytes'] / 2**30:.2f} GiB "
                f"({need['transient_bytes'] / 2**30:.2f} transient + "
                f"{need['retained_bytes'] / 2**30:.2f} retained). REFUSING "
                "BEFORE training rather than discovering it in the middle of a "
                "checkpoint write: every probe finished so far is durable and "
                "this stop preserves them. attempt5 learned this the other way "
                "and lost probe 11's compute and the campaign's verdict.")
        say(f"  {name}: {free / 2**30:.1f} GiB free, needs "
            f"{need['need_bytes'] / 2**30:.1f} GiB")
        return rec

    def release_acked_probe_workdirs(self) -> dict[str, Any]:
        """Release the local workdir of every probe the launcher has ACKED.

        THE ACKNOWLEDGEMENT BOUNDARY. `announce_durable` cannot authorize this:
        it runs BEFORE the transfer, so at announcement time the pod's copy is
        still the only one and deleting it would be a durability race. The
        launcher writes an ack only after the bytes arrived off-pod and
        re-identified there, which is exactly the condition R2 needs anyway.

        The final `model/` directory is kept until its probe is acked, and the
        intermediate checkpoint the recipe's `keep_last` retains goes with it:
        both live under the probe's out_dir and neither is needed locally once
        the bytes are durable elsewhere.

        NEVER RAISES, and that is not tolerance: a cleanup error must not kill
        a paid session mid-rung. The CALLER fails closed on a non-empty
        `failed`, because the storage bound assumes the release happened.
        """
        import shutil

        out = {"released": [], "failed": [], "freed_gib": 0.0, "kept": []}
        ack_dir = REPO / BC.RELEASE_ACK_REL
        try:
            acked = {f.stem for f in ack_dir.glob("*.json")}
        except OSError as exc:
            out["failed"].append(f"cannot read {ack_dir}: {exc}")
            self.ev.setdefault("probe_workdirs_released", []).append(out)
            return out
        freed = 0
        for name, rec in sorted(self.training.items()):
            d = rec.get("out_dir")
            if not d:
                continue
            local = (REPO / d) if not str(d).startswith("/") else Path(d)
            if not local.is_dir():
                continue
            if name not in acked:
                #: Not an error. The launcher pulls asynchronously, so a probe
                #: finished moments ago legitimately has no ack yet; it is
                #: released before the NEXT probe instead.
                out["kept"].append(name)
                continue
            try:
                size = sum(f.stat().st_size for f in local.rglob("*")
                           if f.is_file())
                shutil.rmtree(local)
                freed += size
                out["released"].append(name)
            except OSError as exc:                            # noqa: PERF203
                out["failed"].append(f"{name}: {exc}")
        out["freed_gib"] = round(freed / 2**30, 3)
        self.ev.setdefault("probe_workdirs_released", []).append(out)
        if out["released"] or out["failed"]:
            say(f"  released {len(out['released'])} probe workdir(s), "
                f"{out['freed_gib']:.2f} GiB"
                + (f" (FAILED: {out['failed']})" if out["failed"] else "")
                + (f"; {len(out['kept'])} not yet acked" if out["kept"] else ""))
        if out["failed"]:
            raise C2DriverError(
                f"probe workdir release failed: {out['failed']}. The storage "
                "bound this session runs under assumes each probe's local "
                "workdir is freed once its bytes are durable off-pod, so that "
                "assumption has now been falsified and NO FURTHER PROBE MAY "
                "BE TRAINED under a bound that does not hold. Every durable "
                "probe and its evidence are preserved.")
        return out

    def release_intermediates(self, label: str, results, workdir: Path) -> dict:
        """Delete this arm's intermediate steps, keeping the final checkpoint.

        THE DEFECT THIS EXISTS FOR. `materialize_fixed_path` writes every step
        of a four-step path to the arm's workdir and returns them all; nothing
        deleted them, so all six arms' full paths stayed resident for the whole
        of stage P. The storage derivation, meanwhile, charged
        `b_materialization_transient` ONCE, in as many words: "resident while
        that arm builds and released when it is verified". Nothing released
        them, so the bound described a program that did not exist.

        attempt3 died on exactly that: five arms rebuilt to their exact frozen
        digests, and B failed at `Writing model shards` with `No space left on
        device` after 32.8 minutes of completed compute, at `$2.50`.

        A resource bound has to follow the real free() call sites. This is that
        call site.

        NEVER RAISES, and that is not the same as tolerating failure. A cleanup
        error must not destroy a verified arm that is already announced and
        gated, so the error is recorded here and returned in `failed`. The
        CALLER then stops: `materialize_arm` refuses to build another arm,
        because the 120 GB bound is derived on the assumption these are freed,
        and a failed release falsifies it. Deciding here would conflate two
        separate things — whether this arm survives, and whether the next one
        may begin.
        """
        import shutil

        final = Path(results[-1].checkpoint_path).resolve()
        freed, removed, failed = 0, [], []
        for step in results[:-1]:
            p = Path(step.checkpoint_path).resolve()
            #: Two guards, because this deletes. Never the final, and never
            #: anything outside this arm's own workdir.
            if p == final or not p.is_relative_to(workdir.resolve()):
                continue
            try:
                size = sum(f.stat().st_size
                           for f in p.rglob("*") if f.is_file())
                shutil.rmtree(p)
                freed += size
                removed.append(step.impl_id)
            except OSError as exc:                                # noqa: PERF203
                failed.append(f"{step.impl_id}: {exc}")
        out = {"arm": label, "removed_steps": removed,
               "freed_gib": round(freed / 2**30, 3), "failed": failed,
               "kept": str(final)}
        self.ev.setdefault("intermediates_released", []).append(out)
        say(f"  {label[:12]}… released {len(removed)} intermediate(s), "
            f"{freed / 2**30:.2f} GiB"
            + (f" (FAILED: {failed})" if failed else ""))
        return out

    def materialize_b(self, binding: dict[str, Any]) -> str:
        """Build B from the frozen C1 treatment path, then gate on its identity.

        The construction comes from `baseline.frozen_baseline_spec`, which uses
        C1's own constructor, and `assert_frozen_construction` refuses anything
        whose spec hash is not what C1's preregistration froze. This adds the
        construction check to the shared materialization; the identity gate and
        the durability announcement are that one path's.
        """
        from experiments.phase_c2 import baseline as BL

        say("  B's bytes no longer exist — rebuilding from the frozen treatment path")
        spec = BL.frozen_baseline_spec(device=self.a.device)
        BL.assert_frozen_construction(spec)
        required = binding["required_identity"]
        path = self.materialize_arm(
            "incumbent_b", spec, required=required,
            bounded_minutes=self.a.b_build_minutes)

        (self.audit / "incumbent_b.identity.json").write_text(json.dumps({
            "schema": "aadistill.autoinit.c2_incumbent_b/v1",
            "construction": binding["construction"],
            "required_identity": required,
            "built_path": path,
            #: The built identity lives in the durable announcement, which is
            #: computed from the bytes on disk by the same construction the
            #: destination re-identification uses. Restating it here would be a
            #: second record of one fact.
            "announced": f"durable_units[{'incumbent_b'}]",
            "_not_a_probe": binding.get("_not_a_thirteenth_probe"),
        }, indent=2, default=str) + "\n")
        return path

    # -- the two rungs ------------------------------------------------------
    def probe_config(self, probe: SCH.Probe) -> Path:
        """Derive this probe's training config from the frozen recovery recipe.

        The override set is C1's, unchanged: the seed is the replicate and the
        initialization is the treatment, and everything else being identical is
        what makes two probes at the same seed comparable.
        """
        frozen = json.loads(FROZEN_RECIPE.read_text())
        name = probe.probe_id
        derived = {**frozen, "run_name": name,
                   "out_dir": f"artifacts/stage3/c2_behavioural/{name}",
                   "data_dir": PACK_DIR, "seed": probe.seed,
                   "student_path": probe.initialization_path,
                   "_purpose": (
                       f"Phase C2 {probe.rung} probe, arm {probe.arm}, seed "
                       f"{probe.seed}. Identical recovery; the only intended "
                       "difference between probes at one seed is the "
                       f"initialization. Derived from {FROZEN_RECIPE.name} by "
                       "overriding run identity, pack path, seed and "
                       "student_path.")}
        diff = sorted(k for k in set(frozen) | set(derived)
                      if frozen.get(k) != derived.get(k))
        if not set(diff) <= C1_PROBE_OVERRIDES:
            raise C2DriverError(
                f"{name}: the derived probe config differs from the frozen "
                f"recipe in {sorted(set(diff) - C1_PROBE_OVERRIDES)}, outside "
                f"the allowed override set {sorted(C1_PROBE_OVERRIDES)}")
        path = self.audit / "configs" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(derived, indent=2) + "\n")
        return path

    def release_device(self) -> dict:
        """Hand the card to the trainer, and prove the handoff before training.

        Not scope exit: a C1 attempt read a verdict saying 7.55 GiB was still
        allocated, started the trainer anyway and lost the probe. Both
        conditions are enforced — the release worked, and the card has room for
        the measured peak plus its observed overheads.
        """
        before = cuda_memory()
        import gc

        gc.collect()
        handoff = complete_release(before)
        (self.audit / "c2_device_handoff.json").write_text(
            json.dumps(handoff, indent=2, default=str) + "\n")
        require_released(handoff, what="the C2 recovery trainer")
        require_headroom(handoff["after"], need_bytes=_trainer_bytes(),
                         what="the C2 recovery trainer")
        say(f"  device handoff: {handoff.get('verdict', 'n/a')}")
        return handoff

    def train_one(self, name: str, config: Path) -> Path:
        """Spawn the recovery trainer. A HARDWARE SEAM.

        Everything around it — the budget check, the override check, the
        journal, the durability announcement, the completion count — is this
        driver's business, and the `$0` rehearsal replaces exactly this method
        and its two siblings. A harness that reimplemented the loop could not
        notice the loop being broken, which this programme has already done once.
        """
        rc = subprocess.run(
            ["/opt/train/bin/python", str(TRAINER), "--config", str(config)],
            capture_output=True, text=True,
            timeout=int(self.a.probe_train_minutes * 60 * 2),
            env=self.child_env())
        (self.audit / f"{name}_train_tail.log").write_text(
            (rc.stdout + rc.stderr)[-1500:])
        if rc.returncode != 0:
            raise C2DriverError(
                f"{name}: training failed rc={rc.returncode}; tail: "
                f"...{(rc.stdout + rc.stderr)[-1200:]}")
        return REPO / "artifacts/stage3/c2_behavioural" / name

    def generate_one(self, name: str, package: Path, gen_dir: Path,
                     battery: Path, sets) -> None:
        """Run the evaluator. A HARDWARE SEAM."""
        out = self.gate(
            f"{name}_generation",
            [str(UNCAPPED_EVAL), "--model", str(package), "--label", name,
             "--prompts", *[str(battery / f"{s}.jsonl") for s in sets],
             "--out-dir", str(gen_dir), "--diagnostics"],
            timeout=int(self.a.probe_battery_minutes * 60 * 3),
            python="/opt/vllm/bin/python")
        if out.returncode != 0:
            raise C2DriverError(
                f"{name}: generation rc={out.returncode}; tail: "
                f"...{(out.stdout + out.stderr)[-1200:]}")

    def attest(self, battery: Path) -> dict:
        """Attest the evaluation protocol for one rung's battery.

        Done ONCE PER RUNG, not once per session: the two rungs read different
        batteries, and an attestation naming the screening battery cannot
        certify a confirmation probe. The generation protocol itself is
        observed from a real engine probe, exactly as C1 does it.
        """
        sample = next(iter(self.training.values()))
        package = Path(self.a.eval_dir) / "_attestation_package"
        build_evaluation_package(
            Path(sample["model_dir"]), tokenizer_source=TOKENIZER_SOURCE,
            dest=package, expected_sidecar_sha256=TOKENIZER_SIDECAR_SHA256)
        probe_out = self.audit / f"engine_probe_{battery.name}.json"
        engine = self.gate(
            f"engine_probe_{battery.name}",
            [str(ENGINE_PROBE), "--model", str(package), "--out", str(probe_out),
             "--image-digest", self.a.image_digest],
            timeout=1800, python="/opt/vllm/bin/python")
        if engine.returncode != 0:
            raise C2DriverError(
                f"engine probe rc={engine.returncode}; tail: "
                f"...{(engine.stdout + engine.stderr)[-1200:]}")
        observed = json.loads(probe_out.read_text())

        gen = declared_generation_protocol().materialized(
            generation_source_digest=generation_source_digest(REPO)["digest"],
            degeneration_source_digest=sha256_file(
                REPO / "src/aadistill/evaluation/degeneration.py"))
        gen = gen.materialized(
            vllm_version=observed["vllm_version"],
            transformers_version=observed["transformers_version"],
            torch_version=observed["torch_version"],
            runtime_digest=observed["runtime_digest"], dtype=observed["dtype"],
            gpu_memory_utilization=observed["gpu_memory_utilization"],
            max_num_seqs=observed["max_num_seqs"],
            max_num_batched_tokens=observed["max_num_batched_tokens"],
            enforce_eager=observed["enforce_eager"],
            tokenizer_sha256=observed["tokenizer_sha256"],
            chat_template_sha256=observed["chat_template_sha256"],
            resolved_context=observed["resolved_context"],
            context_source=observed["context_source"],
            stop_token_ids=tuple(observed["stop_token_ids"]))
        gen.require_materialized(context=f"phase C2 battery {battery.name}")

        manifest = json.loads((battery / "manifest.json").read_text())
        contract = c1_scoring_contract(REPO)
        protocol = RecoveryEvaluationProtocol(
            generation=gen, scoring_contract=contract["contract"],
            scoring_digest=contract["digest"],
            battery_artifact=manifest["artifact"],
            battery_manifest_sha256=sha256_json(
                {k: v for k, v in manifest.items() if k != "manifest_sha256"}),
            battery_content_sha256=manifest["content_sha256"])
        self.evaluation_protocol = protocol

        attested = {
            "schema": "aadistill.autoinit.c2_attested_protocol/v1",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "battery": manifest["artifact"],
            "runtime": self.runtime_identity(),
            "generation_source_digest": generation_source_digest(REPO),
            "generation_protocol_fingerprint": gen.fingerprint,
            "scoring_contract": contract,
            "evaluation_protocol": protocol.as_dict(),
            "evaluation_protocol_hash": protocol.evaluation_protocol_hash,
            "tokenizer": {
                "source_rule": "the evaluated checkpoint",
                "packaged_from": _rel(TOKENIZER_SOURCE),
                "sidecar_sha256": dict(TOKENIZER_SIDECAR_SHA256),
                "observed_sha256": observed["tokenizer_sha256"],
                "observed_chat_template_sha256":
                    observed["chat_template_sha256"],
            },
            "_per_rung": ("attested once per battery. An attestation naming one "
                          "rung's battery cannot certify the other's probes."),
        }
        attested["report_sha256"] = sha256_json(attested)
        (self.audit / f"c2_attested_protocol_{battery.name}.json").write_text(
            json.dumps(attested, indent=2) + "\n")
        say(f"  attested {battery.name}: protocol "
            f"{protocol.evaluation_protocol_hash[:12]}…")
        return attested

    def admit_generation(self, name: str, gen_dir: Path, battery: Path) -> dict:
        """Refuse a probe whose generations were not produced under the protocol.

        The attestation says what the runtime is expected to do, once, from an
        engine probe; it is not evidence about any particular probe's rollouts.
        This reconstructs the protocol from THIS probe's raw per-set summaries
        and requires it to be comparable to the attested one.

        Fail-closed and BEFORE the scorer runs. Scoring first and recording the
        observed fingerprint afterwards let a result claim an identity before
        the evidence for it had been admitted, and never compared the two.
        """
        summaries = [json.loads(p.read_text())
                     for p in sorted(gen_dir.glob("*.json"))
                     if not p.name.endswith(".generations.jsonl")]
        observed_gen = observe_generation_protocol(summaries).protocol
        manifest = json.loads((battery / "manifest.json").read_text())
        contract = c1_scoring_contract(REPO)
        observed = RecoveryEvaluationProtocol(
            generation=observed_gen, scoring_contract=contract["contract"],
            scoring_digest=contract["digest"],
            battery_artifact=manifest["artifact"],
            battery_manifest_sha256=sha256_json(
                {k: v for k, v in manifest.items() if k != "manifest_sha256"}),
            battery_content_sha256=manifest["content_sha256"])
        record = {
            "probe_id": name,
            "generation_fingerprint": observed_gen.fingerprint,
            "evaluation_protocol_hash": observed.evaluation_protocol_hash,
            "attested_evaluation_protocol_hash":
                self.evaluation_protocol.evaluation_protocol_hash,
            "n_summaries": len(summaries),
        }
        try:
            observed.require_comparable(self.evaluation_protocol, context=name)
        except Exception as exc:                                  # noqa: BLE001
            record["comparable"] = False
            record["reason"] = str(exc)[-1500:]
            (self.audit / f"{name}_generation_admission.json").write_text(
                json.dumps(record, indent=2) + "\n")
            raise C2DriverError(
                f"{name}: the generations were not produced under the attested "
                "evaluation protocol, so this probe cannot be scored and no "
                f"later probe may be evaluated. {exc}") from exc
        record["comparable"] = True
        (self.audit / f"{name}_generation_admission.json").write_text(
            json.dumps(record, indent=2) + "\n")
        say(f"  {name}: generation protocol admitted "
            f"({record['evaluation_protocol_hash'][:12]}…)")
        return record

    def score_probe(self, probe: SCH.Probe, model_dir: Path, *,
                    battery: Path, run_completion: Path | None) -> dict:
        """Package, generate, ADMIT, then score. One probe, one rung's battery.

        Real production code, parameterized by the rung's battery — not a seam.
        The two hardware-bound calls inside it (`generate_one`, and the engine
        probe behind the attestation) are the seams; everything else here runs
        unchanged in the `$0` rehearsal.

        The battery decides which scorer runs. They are separate entry points on
        purpose: C1's pins its battery by equality so the rung that may name an
        incumbent cannot be aimed elsewhere, and the screening rung gets its own
        pinned entry point rather than a flag that would loosen that guard. Both
        apply the same metric contract, which is what the frozen protocol
        requires of the two rungs.
        """
        name = probe.probe_id
        screening = probe.rung == "screening"
        manifest = json.loads((battery / "manifest.json").read_text())

        package = Path(self.a.eval_dir) / name / "package"
        build_evaluation_package(
            model_dir, tokenizer_source=TOKENIZER_SOURCE, dest=package,
            expected_sidecar_sha256=TOKENIZER_SIDECAR_SHA256)

        gen_dir = Path(self.a.eval_dir) / name
        self.generate_one(name, package, gen_dir, battery, manifest["sets"])
        observed = self.admit_generation(name, gen_dir, battery)

        scored = self.audit / f"{name}_result.json"
        per_sample = self.audit / f"{name}_per_sample.jsonl"
        argv = scorer_argv(
            probe, battery=battery, gen_dir=gen_dir, out=scored,
            per_sample=per_sample,
            generation_fingerprint=observed["generation_fingerprint"],
            run_completion=run_completion)

        rc = self.gate(f"{name}_scoring", argv, timeout=1800,
                       python=sys.executable)
        if rc.returncode != 0:
            raise C2DriverError(
                f"{name}: scoring rc={rc.returncode}; tail: "
                f"...{(rc.stdout + rc.stderr)[-1200:]}")
        result = json.loads(scored.read_text())
        return {
            "probe_id": name, "rung": probe.rung, "arm": probe.arm,
            "seed": probe.seed,
            "result": result,
            "result_path": _rel(scored),
            "result_sha256": sha256_file(scored),
            "per_sample_path": _rel(per_sample),
            "per_sample_sha256": sha256_file(per_sample),
            "observed_generation": observed["generation_fingerprint"],
            "observed_evaluation_protocol_hash":
                observed["evaluation_protocol_hash"],
            "correct_overall": result["correct_overall"],
            "usable_rollout_rate": result["usable_rollout_rate"],
        }

    def persist_probe_record(self, name: str, record: Mapping[str, Any], *,
                             score: Mapping[str, Any] | None = None) -> None:
        """Write this probe's journal entry. ONE writer, called twice.

        Once when TRAINING finishes and before scoring starts, and again when
        the score exists. The early write is the whole point: scoring can fail
        after a checkpoint is already durable off-pod, and the descriptor —
        rung, arm, seed, initialization digest, config hash — is what proves
        those bytes are the measurement a later rung is asking for.

        Before this, the record was written only after a SUCCESSFUL score. So a
        real scoring failure left the destination holding verified bytes and an
        ack that carries only the checkpoint's identity, with nothing to say
        which probe it was; a replacement resource could not have shown it was
        this arm's measurement and would have had to refuse it. The rehearsal
        did not catch that because its fixture wrote a completed record up
        front, which is stronger than the production failure path.
        """
        (self.audit / "probes").mkdir(parents=True, exist_ok=True)
        payload = dict(record)
        if score is not None:
            payload["score"] = dict(score)
        (self.audit / "probes" / f"{name}.json").write_text(
            json.dumps(payload, indent=2, default=str) + "\n")

    def score_existing(self, probe: SCH.Probe, battery: Path,
                       attested: Any) -> tuple[dict, Any]:
        """Score a probe whose checkpoint already exists. NEVER trains.

        The preregistered policy says a probe trained but not validly scored
        resumes AT SCORING, and `run_rung` did not implement it: its only test
        was `name in self.scores`, so a restored trained-but-unscored probe
        fell through to `train_one` and was retrained — against R3, which
        forbids retraining a completed probe for any outcome.

        Budget admission charges the BATTERY only. Charging the trainer again
        for work that will not run would refuse affordable continuations.
        """
        name = probe.probe_id
        record = dict(self.training[name])
        model_dir = Path(record.get("model_dir") or "")
        if not model_dir.is_dir():
            raise C2DriverError(
                f"{name}: the journal records model_dir={model_dir} and it is "
                "not a directory on this pod. A trained probe resumes at "
                "scoring, and scoring reads the weights; there is nothing to "
                "score and retraining is forbidden.")
        if not self.afford(self.a.probe_battery_minutes,
                           f"{name} (scoring a restored checkpoint)"):
            raise C2DriverError(
                f"budget refuses scoring {name}; no probe is skipped to make "
                "progress and a partial rung decides nothing")
        if attested is None:
            attested = self.attest(battery)
        #: `run_completion` was the PRODUCING pod's and does not exist here.
        #: `scorer_argv` treats it as optional and omits `--trained-run` when
        #: it is absent, so the scorer runs on the checkpoint alone.
        score = self.score_probe(probe, model_dir, battery=battery,
                                 run_completion=None)
        record["scored_on_restored_checkpoint"] = True
        self.training[name] = record
        self.scores[name] = score
        self.persist_probe_record(name, record, score=score)
        self.save()
        self.mark(f"{PROBE_MARKER}:{name}")
        say(f"  {name}: NOT retrained — restored checkpoint scored, "
            f"correct_overall={score['correct_overall']:.4f}, "
            f"usable={score['usable_rollout_rate']:.4f}")
        return score, attested

    def run_rung(self, rung: str, probes: list[SCH.Probe], battery: Path) -> None:
        """Train and score one rung, announcing each probe as it completes.

        THREE states per probe, not two:

        1. **scored** — complete. Its descriptor is checked and it is skipped.
        2. **trained but not scored** — its checkpoint was restored and is
           real; it resumes at SCORING and is never retrained.
        3. **absent** — trained, announced for durability, then scored.
        """
        self.release_device()
        attested = None
        for probe in probes:
            name = probe.probe_id
            if name in self.scores:
                #: The identity the preregistration names, checked against THIS
                #: schedule's descriptor and not only against the journal's own
                #: copy of itself. `load_campaign_journal` already proved the
                #: bytes are what was announced; that is a claim about the file,
                #: not about whether it is the probe this rung is asking for. A
                #: continuation whose arm, seed or initialization differed would
                #: otherwise substitute one measurement for another silently.
                self.assert_reuse_matches(probe)
                say(f"  {name}: already complete in this campaign — not retrained")
                continue
            if name in self.training:
                #: State 2. Same descriptor check, then scoring only.
                self.assert_reuse_matches(probe)
                _, attested = self.score_existing(probe, battery, attested)
                continue
            #: BEFORE committing another probe's worth of disk: release every
            #: earlier probe whose bytes the launcher has confirmed durable.
            #: attempt5 had no such point, so twelve probes' workdirs
            #: accumulated and the trainer could not write probe 11 of 12.
            #: Fails CLOSED, because the storage bound this session runs under
            #: assumes the release happens.
            self.release_acked_probe_workdirs()
            #: AND MEASURE, rather than trust the model. Every storage bound in
            #: this repository is a derivation, and attempt5 proved a derivation
            #: can be wrong in the one direction that costs a session: its model
            #: said the pod had room and the trainer hit `No space left on
            #: device` writing probe 11 of 12, losing that probe's completed
            #: compute and the campaign's verdict. A refusal here costs a clean
            #: stop with every finished probe durable; an ENOSPC costs the run.
            self.require_probe_headroom(name)
            #: State 3. Training this probe needs its arm's BYTES, so an arm
            #: the remaining-work budget did not fund is a refusal rather than
            #: a train from a path that was never built.
            self.require_materialized_arm(probe)
            if not self.afford(self.a.probe_train_minutes
                               + self.a.probe_battery_minutes, name):
                raise C2DriverError(
                    f"budget refuses {name}; no probe is skipped to make "
                    "progress and a partial rung decides nothing")
            config = self.probe_config(probe)
            t0 = time.time()
            out_dir = self.train_one(name, config)
            model_dir = trained_model_dir(out_dir)
            record = {"probe_id": name, "campaign": self.a.campaign,
                      "rung": rung, "arm": probe.arm,
                      "seed": probe.seed, "out_dir": _rel(out_dir),
                      "model_dir": str(model_dir),
                      "initialization_artifact_digest":
                          probe.initialization_artifact_digest,
                      "config_sha256": sha256_file(config),
                      "train_minutes": round((time.time() - t0) / 60, 2),
                      "complete": True}
            self.training[name] = record
            #: THE DESCRIPTOR, DURABLE, BEFORE SCORING CAN FAIL. Written here
            #: rather than after a successful score so a scoring crash leaves
            #: bytes that can still be shown to be this probe.
            self.persist_probe_record(name, record)
            #: The bytes, the moment they exist, before anything else can fail.
            arch_signature, num_parameters = self.arm_arch[probe.arm]
            record["durable"] = self.announce_durable(
                name, model_dir, kind="probe", rung=rung, arm=probe.arm,
                seed=probe.seed, arch_signature=arch_signature,
                num_parameters=num_parameters,
                initialization_artifact_digest=
                    probe.initialization_artifact_digest)
            self.persist_probe_record(name, record)
            self.save()

            #: Attested once per rung, after the first probe exists — the
            #: attestation packages a real trained checkpoint.
            if attested is None:
                attested = self.attest(battery)

            run_completion = out_dir / "run_completion.json"
            score = self.score_probe(probe, model_dir, battery=battery,
                                     run_completion=run_completion)
            self.scores[name] = score
            self.persist_probe_record(name, record, score=score)
            self.save()
            self.mark(f"{PROBE_MARKER}:{name}")
            say(f"  {name}: trained and scored, correct_overall="
                f"{score['correct_overall']:.4f}, usable="
                f"{score['usable_rollout_rate']:.4f}")

    # -- S / R / C / D ------------------------------------------------------
    def stage_s(self) -> None:
        self.mark("STAGE_START:S")
        self.screening = SCH.screening_probes(
            self.candidates, self.anchor, self.proto["seeds"]["screening"])
        self.run_rung("screening", self.screening,
                      REPO / C2S.BATTERY_PATH)
        ok, why = SCH.screening_is_complete(
            self.screening, self.training, self.scores)
        if not ok:
            raise C2DriverError(why)
        self.complete("S", probes=len(self.screening), gate=why)

    def stage_r(self) -> None:
        """Rank and advance exactly one. NO VERDICT LEAVES THIS STAGE.

        Committed ONCE per campaign. Screening is mechanical and deterministic
        from six scores, so a continuation recomputing it should reach the same
        candidate — "should" being exactly the word that makes it worth
        checking. If a ranking is already committed under this campaign, the
        recomputation must agree with it and the committed candidate is the one
        that advances. A continuation may not rerun screening for another
        outcome, and this is where that is enforced rather than assumed.
        """
        self.mark("STAGE_START:R")
        ok, why = SCH.screening_is_complete(
            self.screening, self.training, self.scores)
        if not ok:
            raise C2DriverError(
                f"{why} Ranking a partial field selects on who finished first.")

        scores = {p.arm: self.scores[p.probe_id]["correct_overall"]
                  for p in self.screening}
        self.ranked = SCH.rank_screening(scores, self.candidates)
        SCH.assert_screening_emits_no_verdict({"ranked": self.ranked})
        self.advanced = SCH.advance_one(self.ranked)

        ranking_path = self.audit / "c2_screening_ranking.json"
        committed = self.committed_ranking(ranking_path)
        if committed is not None:
            was = committed["advanced"]["state_id"]
            if was != self.advanced["state_id"]:
                raise C2DriverError(
                    f"campaign {self.a.campaign!r} already committed "
                    f"{was} as its advanced candidate and this run's ranking "
                    f"advances {self.advanced['state_id']}. Screening commits "
                    "once: a continuation must confirm the candidate its own "
                    "campaign selected, and a second ranking that names "
                    "another one is a different experiment. Neither result is "
                    "discarded here — both are evidence and this goes to "
                    "review.")
            say(f"  screening already committed {was[:12]}… in this campaign; "
                "the recomputed ranking agrees and it is not re-decided")
            self.advanced = committed["advanced"]
            self.ranked = committed["ranked"]
        else:
            ranking_path.write_text(json.dumps({
                "schema": "aadistill.autoinit.c2_screening_ranking/v1",
                "campaign": self.a.campaign,
                "run_attempt": getattr(self.a, "run_attempt", "") or None,
                "battery": C2S.BATTERY_PATH,
                "seed": self.proto["seeds"]["screening"],
                "anchor_correct_overall": scores[SCH.ANCHOR],
                "ranked": self.ranked, "advanced": self.advanced,
                "may_not": list(C2S.SCREENING_MAY_NOT),
                "_committed_once": (
                    "the advanced candidate of this campaign. A later run "
                    "attempt of the same campaign must confirm THIS candidate "
                    "and may not rerun screening for another outcome."),
            }, indent=2) + "\n")
        self.mark(f"{ADVANCED_MARKER}:{self.advanced['state_id']}")
        say(f"  advanced {self.advanced['state_id'][:12]}…, delta vs B "
            f"{self.advanced['delta_vs_b']:+.4f}"
            + (" (tie broken on the frozen order)"
               if self.advanced["tie_broken"] else ""))
        self.complete("R", ranked=self.ranked, advanced=self.advanced,
                      emits_verdict=False)

    def stage_c(self) -> None:
        self.mark("STAGE_START:C")
        if self.advanced is None:
            raise C2DriverError("no candidate has advanced")
        chosen = next(c for c in self.candidates
                      if c["state_id"] == self.advanced["state_id"])
        self.confirmation = SCH.confirmation_probes(
            chosen, self.anchor, self.rule.seeds)
        self.run_rung("confirmation", self.confirmation,
                      REPO / self.proto["batteries"]["confirmation"]["path"])
        self.complete("C", probes=len(self.confirmation),
                      arm=self.advanced["state_id"])

    def stage_d(self) -> None:
        """Apply the frozen decision rule. The ONLY stage that may name one."""
        self.mark("STAGE_START:D")
        rows: dict[tuple[str, int], list[dict]] = {}
        for probe in self.confirmation:
            s = self.scores[probe.probe_id]
            arm = (BD.INCUMBENT_ARM if probe.arm == SCH.ANCHOR
                   else BD.TREATMENT_ARM)
            path = Path(s["per_sample_path"])
            path = path if path.is_absolute() else REPO / path
            rows[(arm, probe.seed)] = [
                json.loads(line) for line in path.open() if line.strip()]

        decision = BD.confirm(rows, rule=self.rule)
        decision["advanced_candidate"] = self.advanced["state_id"]
        decision["anchor"] = self.anchor["artifact_digest"]
        decision["probe_results"] = {
            k: {"result_sha256": v["result_sha256"],
                "per_sample_sha256": v["per_sample_sha256"]}
            for k, v in sorted(self.scores.items())}
        (self.audit / "c2_decision.json").write_text(
            json.dumps(decision, indent=2) + "\n")
        d = decision["decision"]
        say(f"DECISION: {decision['terminal_state']} · delta {d['delta']:+.4f} "
            f"· LCB {d['lcb_one_sided']:+.4f}")
        self.complete("D", terminal_state=decision["terminal_state"],
                      delta=d["delta"], lcb=d["lcb_one_sided"],
                      bootstrap_seed=decision["bootstrap_seed_used"])

    # -- run ----------------------------------------------------------------
    def run(self) -> int:
        self.mark("DRIVER_START")
        #: FIRST: adopt what the launcher restored from the durable
        #: destination. A replacement resource has a fresh filesystem, so
        #: without this the journal below would find nothing and the campaign's
        #: completed probes would look absent -- and the protocol forbids
        #: retraining them.
        self.restore_campaign()
        #: Then the journal, over whatever is now on this filesystem. A probe
        #: this campaign already finished is restored and never retrained; one
        #: from another campaign, or one whose bytes no longer match what was
        #: announced, is refused.
        self.load_campaign_journal()
        self.save()
        stages = (("P", self.stage_p), ("S", self.stage_s), ("R", self.stage_r),
                  ("C", self.stage_c), ("D", self.stage_d))
        for letter, fn in stages:
            say(f"stage {letter}")
            try:
                fn()
            except Exception as exc:                              # noqa: BLE001
                self.fail(letter, f"{type(exc).__name__}: {exc}",
                          traceback=traceback.format_exc()[-3000:])
                self.mark(FAILURE_MARKER)
                return 1
        self.mark(SUCCESS_MARKER)
        return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", required=True,
                    help=("the SCIENTIFIC campaign: one 12-probe behavioural "
                          "experiment. Stable across replacement resources, and "
                          "the scope within which a completed probe may be "
                          "reused"))
    ap.add_argument("--run-attempt", default="",
                    help=("this launcher invocation and provider resource, for "
                          "logs and evidence. NOT the campaign: when these were "
                          "the same string, a replacement resource became a new "
                          "campaign and had to refuse its predecessor's "
                          "destination-verified probes"))
    ap.add_argument("--continuation-manifest", default="",
                    help=("path on this pod to the campaign continuation "
                          "manifest the launcher staged. Names the verified "
                          "probes this CAMPAIGN already holds and where their "
                          "bytes were restored to. Absent means this is not a "
                          "continuation; empty means the campaign holds none"))
    ap.add_argument("--audit-dir", required=True)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--b-workdir", required=True,
                    help="where incumbent B is materialized before it is gated")
    ap.add_argument("--status-path", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--b-build-minutes", type=float, required=True)
    ap.add_argument("--probe-train-minutes", type=float, required=True)
    ap.add_argument("--probe-battery-minutes", type=float, required=True)
    ap.add_argument("--rate", type=float, required=True)
    ap.add_argument("--spent-usd", type=float, default=0.0)
    ap.add_argument("--soft-stop-usd", type=float, required=True)
    ap.add_argument("--authorized-usd", type=float, required=True)
    ap.add_argument("--image-digest", default="")
    return ap


def main() -> int:
    return C2BehaviouralDriver(build_parser().parse_args()).run()


if __name__ == "__main__":
    raise SystemExit(main())
