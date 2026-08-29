"""The behavioural continuation: finish Phase B without re-buying its search.

Phase-B attempt 5 completed Stage 0 and the joint P=2 search, emitted an
authoritative Top-5 and a durable Stage-1 selection artifact, and ran three
rung-1 `sa` probes. It then failed on a candidate-universe collision that the
identity-collapse amendment resolves. What is left is small: **one missing `sb`,
and at most two conditional `sc`**.

Running the full Phase-B session again to reach them would repurchase a 16.5 h
search that is already bought and already retained. So this is a different
session with a different plan, a different authorization type and a different
ceiling — and, most importantly, **an execution path from which the search is
mechanically unreachable**.

Not a flag. `--skip-search` on the existing driver would leave the search one
mistake away, and this project has already paid four times for code that only one
path reaches. `ContinuationAuthorization.runs_search` is `False` **by type**, the
plan declares no search stage, and a test asserts the continuation executable's
import closure never reaches `phase_a_search`.

What it imports instead is evidence, and it refuses to start if any of it has
moved: the Attempt-5 Stage-1 selection, the identity-collapse amendment, the
six-candidate universe identity, both strict reuse records, and the frozen
rung-1 result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..infrastructure.manifest import sha256_json
from ..infrastructure.source_identity import (
    CANONICAL_DIGEST_ALGORITHM, canonical_source_digest,
)
from .authorization import AuthorizationError
from .recovery import PreflightPlan, PreflightStage

SCHEMA = "aadistill.autoinit.continuation_authorization/v1"

#: The evidence the continuation stands on. Each is bound by hash into the
#: authorization and re-checked at Stage 0; a session whose inputs moved is not
#: the session that was authorized.
BOUND_EVIDENCE = (
    "stage1_selection_sha256",
    "identity_collapse_amendment_sha256",
    "collapsed_universe_identity",
    "historical_reuse_probes_dir_digest",
    "attempt5_reuse_probes_dir_digest",
    "rung1_selection_digest",
)

#: Stage numbers are the INHERITED ones, deliberately. `stage3`, `stage4` and
#: `stage5` are Phase A's frozen implementations and call `enter(3)`/`record(3)`
#: from inside their own bodies; numbering this plan 0,1,2,3,4 would order every
#: stage against a different stage's preconditions.
#:
#: There is no stage 2, and its absence is the point: Phase A's stage 2 is rung 1
#: on seed `sa`, which this session IMPORTS as completed evidence rather than
#: buying. A gap in the numbering is the honest way to say that.
CONTINUATION_PLAN_V1 = PreflightPlan(
    plan_id="autoinit.v1.phase_b_behavioural_continuation",
    version=2,
    stages=(
        PreflightStage(
            stage=0, name="attestation, comparability and evidence binding",
            blocking=True,
            purpose=("establish that this session measures under a runtime the "
                     "frozen Stage-3 thresholds still apply to, and that every "
                     "piece of completed evidence it intends to cite still hashes "
                     "to what the authorization named"),
            produces=("stage record", "continuation_b_stage0_binding.json"),
            stop_conditions=("generation_runtime_comparability@v2 holds; the executable "
                  "source digest matches the grant; the Attempt-5 Stage-1 "
                  "selection, identity-collapse amendment, six-candidate universe "
                  "identity, historical reuse record, Attempt-5 fresh-sa reuse "
                  "record and frozen rung-1 result all match. A comparability "
                  "failure TERMINATES: every cited observation would be lost at "
                  "once, and re-buying them is a different, larger session",)),
        PreflightStage(
            stage=1, name="import the completed behavioural state", blocking=True,
            purpose=("reconstruct the state Phase-B Stage 1 and rung 1 already "
                     "established, from retained artifacts rather than by "
                     "recomputation, and narrow SIX evidence candidates to the "
                     "THREE frozen finalists that may be probed"),
            produces=("stage record",),
            stop_conditions=("the six-candidate collapsed universe rebuilds to the bound "
                  "identity; exactly three finalists enter the probe stages and "
                  "they are the frozen ones; NO search runs, NO new sa probe is "
                  "purchased, and no searched non-survivor is materialized",)),
        # No stage 2. Phase A's stage 2 is rung 1 on sa; this session imports it.
        PreflightStage(
            stage=3, name="rung 2 on seed sb, then the frozen pooled decision",
            blocking=True,
            purpose=("buy the sb evidence that does not exist, cite the sb "
                     "evidence that does, and apply the frozen equivalence and "
                     "final-selection rule unchanged to the pooled two seeds"),
            produces=("stage record", "rung2_selection.json"),
            stop_conditions=("every advancing candidate has an sb observation; each was "
                  "either strictly reconstructed from retained evidence or newly "
                  "run here, and no (initialization, seed) contributes twice; the "
                  "decision comes from the frozen implementation and a tie_pending "
                  "names exactly which candidates lack a verified sc",)),
        PreflightStage(
            stage=4, name="conditional tie-break on seed sc", blocking=False,
            purpose=("resolve candidates inside the preregistered equivalence "
                     "interval after two seeds — only those, and only where no "
                     "verified sc already exists"),
            produces=("stage record",),
            stop_conditions=("no fourth seed, ever",)),
        PreflightStage(
            stage=5, name="final selection and report", blocking=True,
            purpose=("emit the terminal cross-phase result under the frozen "
                     "selection rule"),
            produces=("stage record", "phase_a_result.json"),
            stop_conditions=("winner=None with unresolved_equivalence is a RESULT. Stage-1 "
                  "ranking, search-side KL/NLL, the canonical Stage-1 NLL and "
                  "state-id ordering may NOT break a tie",)),
    ),
)


@dataclass
class ContinuationAuthorization:
    """What a maintainer permitted for the behavioural continuation.

    A distinct type for the same reason `PhaseBAuthorization` is: a grant sized
    for a session that includes a 16.5 h search must not be substitutable for one
    that does not, and vice versa. `runs_search` is `False` **by type** — there is
    no field to set — so no continuation artifact can ever authorize a search.
    """

    authorization_id: str
    granted_utc: str
    granted_by: str
    plan_id: str
    plan_hash: str
    science_plan_hash: str
    calibration_profile_hashes: dict[str, str]
    calibration_content_hashes: dict[str, str]
    #: The completed evidence this session cites, by hash.
    bound_evidence: dict[str, str]
    planning_floor_usd: float
    hard_cap_usd: float
    per_launch_hard_usd: float | None
    authorized_stages: tuple[int, ...]
    stage_conditions: dict[str, str]
    scope_note: str
    authorized_session_commit: str | None = None
    source_digest: str | None = None
    source_files: tuple[str, ...] = ()
    provenance_commit: str | None = None
    version: int = 1

    # --- what this type refuses to be ------------------------------------
    @property
    def runs_search(self) -> bool:
        """False by type. The continuation cannot purchase Stage 1 again."""
        return False

    @property
    def allows_phase_a(self) -> bool:
        return False

    @property
    def allows_phase_b(self) -> bool:
        """True in the sense of "is a Phase-B session", not "may search"."""
        return True

    @property
    def automatic_followon_start(self) -> bool:
        return False

    # --- aliases the shared session machinery requires -------------------
    @property
    def harness_source_files(self) -> tuple[str, ...]:
        return self.source_files

    @property
    def harness_source_digest(self) -> str | None:
        return self.source_digest

    def require_harness(self, repo_root: str | Path = ".") -> dict[str, Any]:
        return self.require_source(repo_root)

    def require_source(self, repo_root: str | Path = ".") -> dict[str, Any]:
        observed = continuation_source_digest(repo_root)
        if self.source_digest and observed["digest"] != self.source_digest:
            raise AuthorizationError(
                f"the continuation executable digests to {observed['digest']} but "
                f"the authorization was granted against {self.source_digest}")
        return observed

    def require_plan(self, plan_hash: str) -> None:
        if plan_hash != self.plan_hash:
            raise AuthorizationError(
                f"this authorization binds plan {self.plan_hash[:12]} and the "
                f"session runs {plan_hash[:12]}")

    def require_science_plan(self, science_plan_hash: str) -> None:
        """The frozen recovery-science plan, distinct from the session plan.

        Required by the inherited stage 0, which rebuilds the plan and binds it
        rather than trusting the launcher's string. Absent from the first draft
        of this type: it was written from what the continuation *needs* instead
        of what the machinery it subclasses *requires*, and stage 0 would have
        raised `AttributeError` on a paid pod.
        """
        if science_plan_hash != self.science_plan_hash:
            raise AuthorizationError(
                f"this authorization binds science plan "
                f"{self.science_plan_hash[:12]} and the session rebuilt "
                f"{science_plan_hash[:12]}; the selection rule moved")

    def require_calibration(self, profile: Any) -> None:
        """Both mixtures, by spec hash AND by content hash.

        **Bound as provenance, not consumed at runtime.** An earlier version of
        this docstring said the probe trainer consumes the calibration; it does
        not. The paid behavioural probes train from
        `artifacts/stage3/ladder_uniform_probe` and are scored on the
        `artifacts/stage3/recovery_search_v2` battery. Neither mixture is read by
        this session at all: the continuation runs no search, and the only
        calibration reference on the inherited path is a comment inside the
        `stage1` its stage map never binds.

        What these identities *do* is name the distribution under which the
        imported Phase-B Stage-1 result was produced. That is worth binding — a
        grant that names a mixture must be able to refuse a different one — and
        it is why the mixtures are **not** staged onto the pod. Reading the bind
        as a consume is what cost continuation attempt 1 `$0.2513`.
        """
        qualified = profile.qualified_id
        expected_spec = self.calibration_profile_hashes.get(qualified)
        expected_content = self.calibration_content_hashes.get(qualified)
        if expected_spec is None or expected_content is None:
            raise AuthorizationError(
                f"the authorization names no calibration {qualified!r}")
        if profile.profile_hash != expected_spec:
            raise AuthorizationError(
                f"{qualified} spec hash {profile.profile_hash[:12]} does not "
                f"match the granted {expected_spec[:12]}")
        if profile.content_sha256 != expected_content:
            raise AuthorizationError(
                f"{qualified} content hash {profile.content_sha256[:12]} does "
                f"not match the granted {expected_content[:12]}")

    def require_evidence(self, observed: dict[str, str]) -> None:
        """Every bound identity must still be what it was when granted."""
        moved = {k: (self.bound_evidence.get(k), observed.get(k))
                 for k in BOUND_EVIDENCE
                 if self.bound_evidence.get(k) != observed.get(k)}
        if moved:
            raise AuthorizationError(
                "the completed evidence this continuation cites has moved since "
                f"the grant: {sorted(moved)}. A session whose inputs changed is "
                "not the session that was authorized.")

    def require_stage(self, stage: int) -> None:
        if stage not in self.authorized_stages:
            raise AuthorizationError(
                f"stage {stage} is not authorized; permitted "
                f"{list(self.authorized_stages)}")

    def require_within_cap(self, projected_usd: float, *, what: str = "") -> None:
        if projected_usd > self.hard_cap_usd:
            raise AuthorizationError(
                f"{what or 'projected spend'} ${projected_usd:.4f} exceeds the "
                f"authorized hard cap ${self.hard_cap_usd:.4f}")

    def require_within_launch_limit(self, hard_usd: float, *, what: str = "") -> None:
        limit = self.per_launch_hard_usd
        if limit is not None and hard_usd > limit:
            raise AuthorizationError(
                f"{what or 'planned hard threshold'} ${hard_usd:.4f} exceeds the "
                f"per-launch limit ${limit:.4f}")

    def as_dict(self) -> dict[str, Any]:
        body = {
            "schema": SCHEMA,
            "authorization_id": self.authorization_id,
            "granted_utc": self.granted_utc,
            "granted_by": self.granted_by,
            "plan_id": self.plan_id,
            "continuation_plan_hash": self.plan_hash,
            "science_plan_hash": self.science_plan_hash,
            "calibration_profile_hashes": dict(self.calibration_profile_hashes),
            "calibration_content_hashes": dict(self.calibration_content_hashes),
            "bound_evidence": dict(self.bound_evidence),
            "planning_floor_usd": self.planning_floor_usd,
            "hard_cap_usd": self.hard_cap_usd,
            "per_launch_hard_usd": self.per_launch_hard_usd,
            "authorized_stages": list(self.authorized_stages),
            "stage_conditions": dict(self.stage_conditions),
            "scope_note": self.scope_note,
            "runs_search": self.runs_search,
            "phase_a_authorized": self.allows_phase_a,
            "automatic_followon_start": self.automatic_followon_start,
            "authorized_session_commit": self.authorized_session_commit,
            "source_digest": self.source_digest,
            "source_files": list(self.source_files),
            "provenance_commit": self.provenance_commit,
            "version": self.version,
        }
        body["authorization_sha256"] = sha256_json(body)
        return body

    @classmethod
    def load(cls, path: str | Path) -> "ContinuationAuthorization":
        raw = json.loads(Path(path).read_text())
        if raw.get("schema") != SCHEMA:
            raise AuthorizationError(
                f"{path} is {raw.get('schema')!r}, not a continuation "
                f"authorization ({SCHEMA!r}). A full Phase-B grant authorizes a "
                "search this session must not run.")
        stated = raw.get("authorization_sha256")
        check = {k: v for k, v in raw.items() if k != "authorization_sha256"}
        if stated != sha256_json(check):
            raise AuthorizationError(
                f"{path} does not match its own authorization_sha256; it has been "
                "edited since it was granted")
        if raw.get("runs_search"):
            raise AuthorizationError(
                "this artifact claims runs_search, which the continuation type "
                "cannot grant; refusing to load it")
        return cls(
            authorization_id=raw["authorization_id"], granted_utc=raw["granted_utc"],
            granted_by=raw["granted_by"], plan_id=raw["plan_id"],
            plan_hash=raw["continuation_plan_hash"],
            science_plan_hash=raw["science_plan_hash"],
            calibration_profile_hashes=dict(raw["calibration_profile_hashes"]),
            calibration_content_hashes=dict(raw["calibration_content_hashes"]),
            bound_evidence=dict(raw["bound_evidence"]),
            planning_floor_usd=float(raw["planning_floor_usd"]),
            hard_cap_usd=float(raw["hard_cap_usd"]),
            per_launch_hard_usd=raw.get("per_launch_hard_usd"),
            authorized_stages=tuple(raw["authorized_stages"]),
            stage_conditions=dict(raw["stage_conditions"]),
            scope_note=raw["scope_note"],
            authorized_session_commit=raw.get("authorized_session_commit"),
            source_digest=raw.get("source_digest"),
            source_files=tuple(raw.get("source_files") or ()),
            provenance_commit=raw.get("provenance_commit"),
            version=int(raw.get("version", 1)))


#: The two entry points whose import closure IS the continuation executable.
CONTINUATION_IMPORT_ROOTS = (
    "scripts/pod/autoinit_continuation_b_driver.py",
    "scripts/pod/autoinit_continuation_b_launch.py",
)

#: Runtime sources that no import reaches, because they are invoked as
#: subprocesses. Each is named with the line that runs it, so the claim "this
#: executes" is checkable rather than asserted:
#:   * `autoinit_preflight_setup.sh` — session_runner.py:548, `bash ...setup.sh`
#:   * `watchdog.py`                 — session_runner.py:379, detached poller
#:   * `collect_artifacts.py`        — session_runner.py:673, pod-side collection
CONTINUATION_RUNTIME_ONLY_FILES = (
    "scripts/pod/autoinit_preflight_setup.sh",
    "scripts/pod/collect_artifacts.py",
    "scripts/pod/watchdog.py",
)

#: What a paid CONTINUATION actually loads and executes — **derived from the real
#: import closure of the two roots, not curated by hand**.
#:
#: It therefore DOES include `search.py`, `ranking.py` and every operator module.
#: They are imported, because the `aadistill.autoinit` package `__init__` imports
#: `BeamSearch` for every consumer of the package. Excluding them would answer a
#: question nobody asked: the source digest records *what code is loaded*, and a
#: digest that omitted loaded files would let those files change under a grant
#: that claimed to pin the executable.
#:
#: Whether the search may RUN is a separate contract, enforced separately and more
#: strongly than an import list could: no search stage in the plan, `stage1()` and
#: `run_search()` that raise, `runs_search` False by type, no forbidden call site
#: in any of these files, and a whole-function test that drives the real stage map.
#:
#: The first version of this tuple conflated the two. It listed 25 files, omitting
#: eight that are loaded and including `phase_b.py`, which is not.
CONTINUATION_SOURCE_FILES_V2: tuple[str, ...] = (
    "scripts/pod/autoinit_continuation_b_driver.py",
    "scripts/pod/autoinit_continuation_b_launch.py",
    "scripts/pod/autoinit_phase_a_driver.py",
    "scripts/pod/autoinit_phase_a_launch.py",
    "scripts/pod/autoinit_preflight_setup.sh",
    "scripts/pod/autoinit_recovery_continuation_launch.py",
    "scripts/pod/autoinit_science_inputs.py",
    "scripts/pod/collect_artifacts.py",
    "scripts/pod/watchdog.py",
    "src/aadistill/__init__.py",
    "src/aadistill/autoinit/__init__.py",
    "src/aadistill/autoinit/adapters/__init__.py",
    "src/aadistill/autoinit/adapters/qwen3.py",
    "src/aadistill/autoinit/arch.py",
    "src/aadistill/autoinit/artifact.py",
    "src/aadistill/autoinit/authorization.py",
    "src/aadistill/autoinit/calibration.py",
    "src/aadistill/autoinit/cost.py",
    "src/aadistill/autoinit/datasets.py",
    "src/aadistill/autoinit/device.py",
    "src/aadistill/autoinit/device_handoff.py",
    "src/aadistill/autoinit/generation.py",
    "src/aadistill/autoinit/generation_compat.py",
    "src/aadistill/autoinit/identity_collapse.py",
    "src/aadistill/autoinit/leaf_durability.py",
    "src/aadistill/autoinit/manifest.py",
    "src/aadistill/autoinit/metrics.py",
    "src/aadistill/autoinit/operators/__init__.py",
    "src/aadistill/autoinit/operators/_common.py",
    "src/aadistill/autoinit/operators/attention.py",
    "src/aadistill/autoinit/operators/base.py",
    "src/aadistill/autoinit/operators/composite.py",
    "src/aadistill/autoinit/operators/depth.py",
    "src/aadistill/autoinit/operators/ffn.py",
    "src/aadistill/autoinit/operators/width.py",
    "src/aadistill/autoinit/phase_a.py",
    "src/aadistill/autoinit/phase_b_continuation.py",
    "src/aadistill/autoinit/ranking.py",
    "src/aadistill/autoinit/recovery.py",
    "src/aadistill/autoinit/recovery_continuation.py",
    "src/aadistill/autoinit/search.py",
    "src/aadistill/autoinit/state.py",
    "src/aadistill/autoinit/stats.py",
    "src/aadistill/autoinit/telemetry.py",
    "src/aadistill/data/__init__.py",
    "src/aadistill/data/extra_stream.py",
    "src/aadistill/infrastructure/__init__.py",
    "src/aadistill/infrastructure/artifact_gate.py",
    "src/aadistill/infrastructure/budget.py",
    "src/aadistill/infrastructure/log_relay.py",
    "src/aadistill/infrastructure/manifest.py",
    "src/aadistill/infrastructure/provider.py",
    "src/aadistill/infrastructure/remote.py",
    "src/aadistill/infrastructure/session.py",
    "src/aadistill/infrastructure/session_prechecks.py",
    "src/aadistill/infrastructure/session_runner.py",
    "src/aadistill/infrastructure/source_identity.py",
    "src/aadistill/init/__init__.py",
    "src/aadistill/init/collect.py",
    "src/aadistill/init/contribution.py",
    "src/aadistill/init/project.py",
    "src/aadistill/init/sandwich.py",
)
#: v2 -> v3 is an ALGORITHM change, not a content change. The file set is
#: identical; what moved is how the set becomes an identity. v2 used
#: `sha256_json(entries)`, which `session_commit_gate` — the first pre-provider
#: precheck — can never reproduce, because it re-derives the digest itself from
#: git blobs using the canonical line-join formula that Phase A and Phase B
#: already use. The version is bumped so a reader can tell "the algorithm was
#: corrected" from "someone edited the executable"; the same 61 files under the
#: two schemes produce different digests and would otherwise look like an edit.
#:
#: This is an executable/provenance identity only. No scientific protocol is
#: versioned by it.
CONTINUATION_SOURCE_SET_VERSION = 3
CONTINUATION_SOURCE_DIGEST_ALGORITHM = CANONICAL_DIGEST_ALGORITHM


def derive_continuation_closure(repo_root: str | Path = ".") -> tuple[str, ...]:
    """Import the roots in a clean interpreter; report the repo files loaded.

    Run in a subprocess because the answer is "what does a fresh process load",
    and this process has already imported half the repository for other reasons.
    A test compares this against the frozen tuple, so adding an import without
    updating the source set fails at `$0` rather than under a grant.
    """
    import subprocess
    import sys as _sys

    root = Path(repo_root).resolve()
    roots = [f"scripts/pod/{Path(r).name[:-3]}" for r in CONTINUATION_IMPORT_ROOTS]
    program = (
        "import sys, json, os\n"
        "sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts/pod')\n"
        "sys.path.insert(0, 'scripts/autoinit')\n"
        + "".join(f"import {Path(r).name}\n" for r in roots) +
        "root = os.path.abspath('.')\n"
        "out = set()\n"
        "for m in list(sys.modules.values()):\n"
        "    f = getattr(m, '__file__', None)\n"
        "    if not f: continue\n"
        "    p = os.path.abspath(f)\n"
        # `torch._classes` and `torch._ops` carry bare relative `__file__`s that
        # abspath resolves INTO the repo root. The isfile check drops them.
        "    if not p.startswith(root + os.sep) or not os.path.isfile(p): continue\n"
        "    rel = os.path.relpath(p, root)\n"
        "    if rel.startswith(('.venv', 'tests')): continue\n"
        "    out.add(rel)\n"
        "print(json.dumps(sorted(out)))\n")
    proc = subprocess.run([_sys.executable, "-c", program], cwd=root,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise AuthorizationError(
            "could not import the continuation roots to derive their closure:\n"
            + proc.stderr[-2000:])
    imported = json.loads(proc.stdout.strip().splitlines()[-1])
    return tuple(sorted(set(imported) | set(CONTINUATION_RUNTIME_ONLY_FILES)))

#: Modules whose presence in the continuation's import closure would mean a
#: search is reachable. Asserted by test.
#:
#: `aadistill.autoinit.search` is deliberately NOT listed, and the reason matters:
#: the `aadistill.autoinit` package `__init__` imports it for every consumer of
#: the package, so its presence says nothing about this session — which is also
#: why it appears in the source digest. What DOES say something is
#: `phase_a_search`, the module holding `run_phase_a_search`, the only entry point
#: that actually executes a search, and it is absent from the closure.
#:
#: Being loaded and being reachable are different claims. The compensating
#: guarantees, checked separately because an import list alone would be weaker
#: than it looks: no file in `CONTINUATION_SOURCE_FILES_V2` calls
#: `run_phase_a_search(` or constructs `BeamSearch(`; the plan declares no search
#: stage; `stage1()` and `run_search()` raise; and a whole-function test drives
#: the real stage map and counts zero search invocations.
FORBIDDEN_IN_CONTINUATION = (
    "phase_a_search",
    "autoinit_phase_b_driver",
)

#: Call sites that would execute a search.
FORBIDDEN_CALLS = ("run_phase_a_search(", "BeamSearch(")

#: The files that ARE the continuation — as opposed to the libraries it loads.
#: A search call site here would mean the continuation itself invokes a search.
CONTINUATION_OWN_PATH_FILES = (
    "scripts/pod/autoinit_continuation_b_driver.py",
    "scripts/pod/autoinit_continuation_b_launch.py",
    "src/aadistill/autoinit/phase_b_continuation.py",
)

#: The ONE loaded file that legitimately contains a search call site, and why it
#: cannot fire.
#:
#: `PhaseADriver.stage1` calls `run_phase_a_search(...)`. The continuation
#: subclasses that driver, so the line is loaded — and it is unreachable twice
#: over: `ContinuationDriver.stage1` overrides it with a raise, and the
#: continuation stage map never binds stage 1 to it at all. Neither fact is
#: provable by grep, so both are asserted by whole-function tests that drive the
#: real stage map with `BeamSearch` and `run_phase_a_search` replaced by
#: detonators.
#:
#: Pinning the set rather than skipping the check means a search call site
#: appearing in ANY other loaded file fails the gate, including one added to a
#: library module that today has none.
KNOWN_NEUTRALIZED_SEARCH_CALL_SITES = ("scripts/pod/autoinit_phase_a_driver.py",)


def search_call_site_owners(repo_root: str | Path = ".",
                            files: tuple[str, ...] | None = None) -> tuple[str, ...]:
    """Which loaded files contain a search call site. Declaration excluded."""
    root = Path(repo_root)
    declarer = "src/aadistill/autoinit/phase_b_continuation.py"
    declared = tuple(files) if files is not None else CONTINUATION_SOURCE_FILES_V2
    out = []
    for rel in declared:
        if rel == declarer or not (root / rel).is_file():
            continue
        text = (root / rel).read_text(errors="ignore")
        if any(call in text for call in FORBIDDEN_CALLS):
            out.append(rel)
    return tuple(sorted(out))


def continuation_source_digest(repo_root: str | Path = ".", *,
                               files: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Digest the declared continuation source. Fails closed on a gap."""
    from .phase_a import sha256_file

    root = Path(repo_root)
    declared = tuple(files) if files is not None else CONTINUATION_SOURCE_FILES_V2
    entries = []
    for rel in sorted(declared):
        path = root / rel
        if not path.is_file():
            raise AuthorizationError(
                f"declared continuation source {rel!r} is missing; refusing a "
                "digest over a smaller executable than the one that runs")
        entries.append({"path": rel, "sha256": sha256_file(path),
                        "bytes": path.stat().st_size})
    return {"set_version": CONTINUATION_SOURCE_SET_VERSION,
            "algorithm": CONTINUATION_SOURCE_DIGEST_ALGORITHM,
            "n_files": len(entries), "files": entries,
            # The SHARED formula, imported rather than re-typed. `session_commit_gate`
            # re-derives this value independently from the blobs at the launch
            # commit, so a producer-local variant is not a style question: it is a
            # value the gate cannot reproduce, and it refused every launch.
            "digest": canonical_source_digest(entries), "not_yet_covered": []}
