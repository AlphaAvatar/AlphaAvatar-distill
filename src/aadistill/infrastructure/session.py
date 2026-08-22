"""What a paid pod session IS, as one immutable declaration.

A session used to be a subclass of the micro-preflight launcher, retargeted by
mutating that module's globals before construction. Its real contract was the
union of three things, no one of which could be read from the others: its own
class attributes and overridden hooks, every ``self.a.<name>`` the base read, and
every line a shared shell script executed unconditionally. Nothing checked that a
session satisfied all three, and three paid pods were lost proving it:

===========================  ========  =================================================
session                        cost     died on
===========================  ========  =================================================
Phase-A attempt 1             $0.1075   ``SESSION_KIND`` leaked between two sessions
device canary attempt 1       $0.0603   the base read ``self.a.teacher_revision``; the
                                        subclass had never heard of it
device canary retry           $0.0637   the shared setup copies two assets out of
                                        ``$WS/assets``; the subclass had declared
                                        ``LOCAL_ASSETS = ()`` because it needed neither
===========================  ========  =================================================

Every one is the same failure: **a session inherited a requirement it never
declared.** This module is the replacement. A session states everything it is,
once, in a frozen object; :mod:`aadistill.infrastructure.session_runner` consumes
that object and nothing else. There is no base class to inherit from and no
module global to mutate, so "inherited a requirement it never declared" stops
being a class of defect rather than becoming rarer.

:meth:`SessionSpec.validate` refuses an incomplete declaration **before anything
is priced**, which is the only point at which refusing is free.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .budget import BudgetPlan, Phase, StepTime, plan_session


class SessionSpecError(ValueError):
    """A session declaration that cannot be executed as written."""


#: The trees a session may stage into. Same set the local-asset check uses:
#: everything else in the pod checkout is code the bundle put there.
WRITABLE_TREES: tuple[str, ...] = ("artifacts", "logs", "configs")


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _bad_repo_dir(dest: str) -> list[str]:
    """Why `dest` is not a repository-relative directory a session may write.

    Absolute paths are refused rather than normalized: the shell used to write
    `/workspace/aad/artifacts/...` literally, and a declaration that carries the
    pod's absolute layout is one that cannot be checked against the repository
    it claims to write into.
    """
    problems = []
    if dest.startswith("/"):
        problems.append("it is absolute; declare it relative to the repo root")
    if ".." in Path(dest).parts:
        problems.append("it escapes the repository with '..'")
    root = dest.strip("/").split("/")[0]
    if root not in WRITABLE_TREES:
        problems.append(f"{root!r} is not one of {list(WRITABLE_TREES)}")
    return problems


# ---------------------------------------------------------------------------
# what a session needs staged before its driver starts
# ---------------------------------------------------------------------------

#: The artifact store every declaration has always meant. Named once so a
#: session that needs a *different* repository says so in its manifest rather
#: than in a fetch block nobody declared.
MAIN_RELAY = "AlphaAvatar/aadistill-artifacts"


@dataclass(frozen=True)
class RelayInput:
    """One object the pod fetches from the artifact store.

    Declared so the launcher can refuse **at $0** rather than after setup has
    been paid for. Phase-A attempt 5 died at $0.6426 on a calibration file that
    was neither staged nor prechecked: stage 1 called ``resolve()`` and there was
    nothing to read.

    Until 2026-08-18 this type carried a *path and nothing else*, and ``dest``
    documented itself as "the setup script already knows where this goes". It
    did know: ``autoinit_preflight_setup.sh`` named three relay prefixes, ten
    filenames, four sha256 pins and a probe-to-ladder copy, unconditionally, for
    every session. So the declaration was an existence assertion and the staging
    was hidden — the same defect the ``LOCAL_ASSETS`` fix closed on the dev-box
    side, still open on the relay side. Four of the ten files were declared by no
    session at all, and the calibration that killed attempt 5 was declared by
    two sessions out of four while being staged for all four.

    Now the declaration IS the staging. ``dest`` means what it says, and the
    setup script names no relay path, no filename and no digest of its own.
    """

    path: str
    #: WHICH artifact repository holds it. Defaults to the main relay, so every
    #: existing declaration keeps its meaning unchanged.
    #:
    #: Added 2026-08-22 because the five Attempt-12 leaves cannot travel by the
    #: routes that existed. Pushing them by scp needs 1.99 MB/s to fit the
    #: launcher's 600 s per-asset timeout against a dev box observed at
    #: 0.44-0.72 MB/s — that is what ended continuation attempt 2 — and the main
    #: relay had 1.60 GiB of headroom against 5.55 GiB of leaves. A second
    #: private repo was measured to work (2026-08-13: "a 1 MiB write to a
    #: different private repo succeeded, so the limit binds per-repo"), which
    #: inverts the transfer: the pod PULLS at hub speed instead of the dev box
    #: PUSHING while a GPU bills.
    #:
    #: It lives on ``RelayInput`` rather than in a continuation-specific fetch
    #: block for the same reason ``dest`` does. A hidden second fetch path would
    #: be exactly the defect this type was created to remove: staging the
    #: manifest does not declare.
    repo: str = MAIN_RELAY
    #: Repository-relative DIRECTORY it is staged into on the pod. ``None`` means
    #: setup does not stage it — the continuation's two permanent controls arrive
    #: by ``--transport``, and this entry buys them the $0 precheck. It can no
    #: longer mean "the shell knows": a structural test forbids the shell from
    #: knowing.
    dest: str | None = None
    #: The frozen content hash, when one is pinned. Verified on the pod, after
    #: staging and before any science runs, at every destination it lands in.
    sha256: str | None = None
    #: A second directory the staged file is copied to. The recovery pack is read
    #: as ``ladder_uniform_probe`` by `p2_driver.py` and as ``ladder_uniform`` by
    #: the recovery corpus loader; the shell used to do this with a directory
    #: walk nobody had declared.
    also_stage_to: str | None = None

    @property
    def staged(self) -> bool:
        """Does setup put this file somewhere, or is it precheck-only?"""
        return bool(self.dest)

    def as_record(self) -> dict[str, Any]:
        """The whole truth about one input: source, destinations, digest.

        This is both what setup consumes and what the session record preserves.
        They are the same dict on purpose — a serialized identity that dropped
        the destination and the digest would describe a staging nobody could
        reproduce.
        """
        return {"repo": self.repo, "path": self.path, "dest": self.dest,
                "sha256": self.sha256, "also_stage_to": self.also_stage_to}


@dataclass(frozen=True)
class LocalAsset:
    """A dev-box-only artifact the launcher scp's, and setup installs.

    ``install_to`` is the whole point. The shared setup script used to name
    ``state_eval_v1`` and ``recovery_search_v2`` itself, so a session that needed
    neither still had them copied out of ``$WS/assets`` — and the device-canary
    retry, which had honestly declared it wanted none, died at $0.0637 when that
    unconditional ``cp`` found an empty directory under ``set -e``.
    """

    repo_path: str
    #: The name under ``$WS/assets`` after the launcher copies it.
    dest_name: str
    #: Directory in the repository on the pod that it is installed into.
    install_to: str

    def as_env_entry(self) -> str:
        """``name:install_to``, the form ``SESSION_ASSETS`` carries."""
        return f"{self.dest_name}:{self.install_to}"


@dataclass(frozen=True)
class SetupManifest:
    """Everything the shared setup script is allowed to know about a session.

    The runner builds setup's environment **entirely** from this. Nothing is
    injected that a session did not declare, which is what makes the canary
    retry's failure impossible rather than merely unlikely.
    """

    #: Extra environment beyond the ones the runner always sets.
    env: Mapping[str, str] = field(default_factory=dict)
    #: Variables the setup script requires this session to supply. Declared so a
    #: structural test can compare them against what the script actually reads.
    required_env: tuple[str, ...] = ()
    relay_inputs: tuple[RelayInput, ...] = ()
    local_assets: tuple[LocalAsset, ...] = ()
    #: Markers setup emits on the happy path, in order. Used for documentation
    #: and for the probe, not as a gate — the gate is ``SETUP_DONE``.
    setup_markers: tuple[str, ...] = ()
    #: Was ``--uv-max-s`` / ``--tests-max-s`` on every launcher's parser, read
    #: only by the base. A session-shaped constant is a manifest field, not an
    #: operational knob: the canary had to declare a teacher revision it never
    #: used because the base read one off its argument namespace.
    uv_max_seconds: int = 1500
    tests_max_seconds: int = 2700
    teacher_revision: str = ""
    #: Ignored by the pod's blocking test gate. Must stay equal to the pod
    #: simulator's list or the simulation runs a command the pod does not.
    test_ignores: tuple[str, ...] = ()

    def assets_env(self) -> str:
        return ",".join(a.as_env_entry() for a in self.local_assets)

    def test_ignores_env(self) -> str:
        return " ".join(f"--ignore={p}" for p in self.test_ignores)

    def staged_relay_inputs(self) -> tuple[RelayInput, ...]:
        """The subset setup stages. The rest exist for the $0 precheck only."""
        return tuple(r for r in self.relay_inputs if r.staged)

    def relay_env(self) -> str:
        """`SESSION_RELAY_INPUTS`: what setup fetches, and nothing else.

        JSON rather than the `name:dest` form `SESSION_ASSETS` uses, because an
        input carries four fields and one of them is a 64-character digest. The
        shell hands this straight to the python that does the fetching, so there
        is no shell parsing to get wrong.
        """
        return json.dumps([r.as_record() for r in self.staged_relay_inputs()],
                          separators=(",", ":"), sort_keys=True)


# ---------------------------------------------------------------------------
# how a session ends
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MarkerPolicy:
    """The terminal markers a driver may write, and what each one means."""

    success: str = "ALL_DONE"
    failure: tuple[str, ...] = ()
    #: Failure markers that still mean the blocking stages passed, so whatever
    #: they produced EXISTS and must be fetched. ``if terminal == "ALL_DONE"``
    #: deleted $2.82 of verified checkpoints on 2026-08-13 for want of this.
    incomplete: tuple[str, ...] = ()
    failure_note: str = ""

    def is_incomplete(self, terminal: str) -> bool:
        return terminal in self.incomplete

    def stage2_passed(self, terminal: str, driver_stages: Mapping[str, Any]) -> bool:
        return bool(driver_stages.get("2")
                    or terminal == self.success
                    or self.is_incomplete(terminal))


@dataclass(frozen=True)
class ArtifactPolicy:
    """What a session produced, and which of it must survive the pod."""

    audit_dirname: str
    evidence_filename: str
    archive_basename: str
    spec_success: str
    spec_failed: str
    report_names: tuple[str, ...] = ()
    #: Append-only streams a torn-down session may have left mid-write. A
    #: callable rather than a list because Phase A derives its nine probe streams
    #: from what was actually journalled: naming a fixed set would either miss a
    #: rung-2 probe or demand a rung-3 one that correctly never ran.
    event_streams: Callable[["SessionContext"], tuple[str, ...]] = \
        lambda ctx: ()
    #: Fetch what this session PRODUCED and cannot regenerate for free. Returns
    #: a list of records for the session evidence.
    fetch_products: Callable[["SessionContext"], list] = lambda ctx: []
    #: Did this session secure the products it OWES off-pod? Returns `(ok, why)`
    #: and feeds the `required_products_secured` teardown check.
    #:
    #: Separate from `fetch_products` returning cleanly, because `all([])` is
    #: True: a fetch that returned NOTHING passes `checkpoint_hashes_matched`
    #: while having secured nothing at all. Phase-A attempt 11 staged five
    #: selected leaves on the pod and lost every one with every check green.
    #:
    #: The default answers "this session owes no products", which is the honest
    #: answer for a run that produces no weights — not a reason to skip the gate.
    products_secured: Callable[..., tuple[bool, str]] = \
        lambda ctx, fetched: (True, "this session owes no off-pod products")
    #: Extra `(remote_path, local_filename)` pairs the log relay pulls while the
    #: driver runs, beyond the run log, the status file and the evidence. The
    #: preflight relays both controls' training streams; the continuation, which
    #: trains nothing, relays none. Returned as plain tuples so this module does
    #: not depend on the relay's own types.
    extra_relay_streams: Callable[["SessionContext"], tuple[tuple[str, str], ...]] = \
        lambda ctx: ()


@dataclass(frozen=True)
class TeardownPolicy:
    """When the pod is deleted, and what counts as proof that it is gone."""

    #: Tear down on every path the gate allows, success or not.
    always: bool = True
    #: Poll the provider until it stops billing. `--terminate-after` has never
    #: been observed to fire, so termination is a verification, not a 200.
    require_provider_confirmation: bool = True
    note: str = ""


@dataclass(frozen=True)
class BudgetSpec:
    """Everything needed to price the session, and nothing about running it.

    Held separately from the runner so a price can be reproduced without a pod,
    a provider or a network — which is how a refactor of the runner is proved
    behaviour-preserving.
    """

    arms: int
    steps_per_arm: int
    step_seconds: float
    step_source: str
    setup_minutes: float
    transfer_minutes: float
    other_phases: tuple[Phase, ...] = ()
    eval_minutes_per_arm: float = 0.0
    contingency_fraction: float = 0.10
    #: Folded in AFTER the contingency multiplier and BEFORE the soft stop, so a
    #: risk that materializes early is not paid for out of the later stages.
    soft_stop_reserves: tuple[Phase, ...] = ()
    artifact_recovery_reserve_minutes: float = 30.0
    below_floor_reason: str = ""

    def plan(self, *, price_per_hour: float, authorized_usd: float) -> BudgetPlan:
        return plan_session(
            price_per_hour=price_per_hour,
            authorized_usd=authorized_usd,
            arms=self.arms, steps_per_arm=self.steps_per_arm,
            step_time=StepTime(self.step_seconds, self.step_source),
            below_floor_reason=self.below_floor_reason,
            setup_minutes=self.setup_minutes,
            other_phases=self.other_phases,
            eval_minutes_per_arm=self.eval_minutes_per_arm,
            transfer_minutes=self.transfer_minutes,
            contingency_fraction=self.contingency_fraction,
            soft_stop_reserves=self.soft_stop_reserves,
            artifact_recovery_reserve_minutes=(
                self.artifact_recovery_reserve_minutes))


# ---------------------------------------------------------------------------
# what a spec callable may see
# ---------------------------------------------------------------------------

@dataclass
class SessionContext:
    """The runner's state, as much of it as a spec callable is allowed to touch.

    Deliberately not the runner itself. A callable that could reach the runner
    could mutate the flow, which is the property this design exists to remove.
    """

    scr: Any
    args: Any
    auth: Any
    evidence: dict
    say: Callable[[str], None]
    host: str = ""
    target: Any = None
    scp: tuple[str, ...] = ()
    stage2_passed: bool = False
    plan: BudgetPlan | None = None
    price: float | None = None
    image_digest: str = ""
    elapsed_minutes: float = 0.0
    spent_usd: float = 0.0


#: A precheck answers "can this session run at all", at $0, before a pod exists.
Precheck = Callable[[SessionContext], tuple[bool, str]]


# ---------------------------------------------------------------------------
# the specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SessionSpec:
    """One session, completely declared.

    The runner consumes this and nothing else. Everything a previous subclass
    expressed by overriding a hook, setting a class attribute or mutating a
    module global is a field here, which is what makes a session's contract
    readable in one place.
    """

    session_id: str
    schema: str
    description: str

    #: The artifact this session's spending is granted by, and the TYPE that
    #: loads it. The type is a field rather than a base-class choice, which is
    #: what makes "this launcher cannot start Phase A" a property instead of a
    #: promise: `SpendAuthorization.allows_phase_a` is a hard False, so a session
    #: naming that type cannot start Phase A whatever it is pointed at.
    authorization_path: str
    authorization_loader: Callable[[Any], Any]

    plan_id: str
    plan_hash: str

    budget: BudgetSpec
    setup: SetupManifest

    #: The command the pod runs, and the job id the detached start registers.
    driver_command: Callable[[SessionContext, BudgetPlan], str]
    driver_job_id: str

    status_path: str
    run_log_path: str

    markers: MarkerPolicy
    artifacts: ArtifactPolicy
    teardown: TeardownPolicy = field(default_factory=TeardownPolicy)

    #: `(ctx) -> (ok, message)`, all of them run before a pod is created. Phase
    #: A's two identity gates — the session-commit/harness/lineage check and the
    #: frozen science-plan check — are entries here rather than overridden
    #: methods, so a reader can count them.
    precheck: tuple[Precheck, ...] = ()

    #: Extra fields written into the session record at construction.
    evidence_fields: Mapping[str, Any] = field(default_factory=dict)

    #: Inputs the driver consumes but does not fetch. `(ctx) -> ok`.
    materialize_inputs: Callable[[SessionContext], bool] = lambda ctx: True

    # -- the refusal that has to happen before anything is priced ----------
    def validate(self) -> "SessionSpec":
        """Refuse an incomplete declaration. Returns self so it can be chained.

        Every check here is a defect that has actually reached a pod, or the
        direct analogue of one. It runs before pricing, before the provider is
        contacted, and before a pod exists — the only point at which refusing
        costs nothing.
        """
        problems: list[str] = []
        if not self.session_id.strip():
            problems.append("session_id is empty")
        if not self.schema.strip():
            problems.append("schema is empty; the session record would be untyped")
        if not self.authorization_path.strip():
            problems.append("no authorization_path: nothing would bound the spend")
        if self.authorization_loader is None:
            problems.append(
                "no authorization_loader: the artifact TYPE is what makes "
                "allows_phase_a a property rather than a promise")
        if not self.plan_hash.strip():
            problems.append(
                "no plan_hash: the authorization could not be bound to a plan, "
                "and an authorization that binds to nothing grants everything")
        if not self.status_path.strip():
            problems.append(
                "no status_path: the launcher probes this file to decide whether "
                "setup succeeded. Hardcoding another session's filename cost "
                "$0.1324 on a setup that had SUCCEEDED")
        if not self.run_log_path.strip():
            problems.append("no run_log_path: the relay would have nothing to pull")
        if not self.driver_job_id.strip():
            problems.append("no driver_job_id: the detached start could not be probed")
        if not callable(self.driver_command):
            problems.append("driver_command is not callable")

        if not self.markers.success.strip():
            problems.append("no success marker: the session could never complete")
        if not self.markers.failure:
            problems.append(
                "no failure markers: a failing driver would be polled until the "
                "limit instead of being classified")
        if set(self.markers.incomplete) - set(self.markers.failure):
            problems.append(
                "incomplete markers must be a subset of failure markers: an "
                "incomplete marker the poller does not recognise is not seen")

        a = self.artifacts
        if not a.evidence_filename.strip():
            problems.append("no evidence_filename: the relay would pull no evidence")
        if not a.audit_dirname.strip():
            problems.append("no audit_dirname: collection would have no root")
        if not a.spec_success.strip() or not a.spec_failed.strip():
            problems.append(
                "both artifact specs are required: a blocking failure has a "
                "SMALLER required set, and demanding the success set would block "
                "teardown on artifacts the run correctly refused to produce")
        if not a.archive_basename.strip():
            problems.append("no archive_basename: nothing to transfer")

        seen: set[str] = set()
        for asset in self.setup.local_assets:
            if asset.dest_name in seen:
                problems.append(f"duplicate local asset {asset.dest_name!r}")
            seen.add(asset.dest_name)
            if not asset.install_to.strip():
                problems.append(
                    f"local asset {asset.dest_name!r} declares no install_to, so "
                    "setup would not know where to put it")
        if any(":" in a.dest_name or "," in a.dest_name
               for a in self.setup.local_assets):
            problems.append(
                "a local asset name may contain neither ':' nor ',': "
                "SESSION_ASSETS is a comma-separated list of name:dest pairs")

        # Relay inputs. Unchecked until 2026-08-18, because the shell staged
        # them from its own hardcoded list and the declaration was decorative.
        # Now the declaration is the staging, so a malformed one is a pod that
        # fetches into the wrong place or verifies nothing.
        pairs: set[tuple[str, str | None]] = set()
        for r in self.setup.relay_inputs:
            if not r.path.strip():
                problems.append("a relay input declares no path")
            if (r.path, r.dest) in pairs:
                problems.append(
                    f"relay input {r.path!r} is declared twice for the same "
                    "destination")
            pairs.add((r.path, r.dest))
            if r.sha256 is not None and not _is_sha256(r.sha256):
                problems.append(
                    f"relay input {r.path!r} pins {r.sha256!r}, which is not a "
                    "sha256; a digest that cannot match verifies nothing")
            if r.also_stage_to and not r.dest:
                problems.append(
                    f"relay input {r.path!r} declares also_stage_to but no "
                    "dest, so setup would have nothing to copy from")
            for label, dest in (("dest", r.dest),
                                ("also_stage_to", r.also_stage_to)):
                if dest is None:
                    continue
                if not dest.strip():
                    problems.append(
                        f"relay input {r.path!r} declares an empty {label}; "
                        "omit it to mean 'setup does not stage this'")
                    continue
                problems.extend(
                    f"relay input {r.path!r} stages to {dest!r}: {why}"
                    for why in _bad_repo_dir(dest))

        if problems:
            raise SessionSpecError(
                f"{self.session_id}: incomplete session specification — "
                + "; ".join(problems))
        return self

    # -- the environment the shared setup script is given ------------------
    def setup_environment(self, *, session_commit: str, bundle: str) -> dict[str, str]:
        """Built ENTIRELY from the manifest. Nothing else reaches setup.

        Two failures live in this method's existence. `SESSION_KIND` leaked
        between two sessions sharing one setup script ($0.1075) because it was a
        module global; here it can only arrive through `setup.env`. And the
        shared script named two assets itself, so a session that wanted none got
        them anyway ($0.0637); here it reads `SESSION_ASSETS`.
        """
        env = {
            "SESSION_COMMIT": session_commit,
            "BUNDLE_NAME": bundle,
            "SESSION_STATUS": self.status_path,
            "SESSION_AUTH_PATH": self.authorization_path,
            "SESSION_PLAN_HASH": self.plan_hash,
            "SESSION_ASSETS": self.setup.assets_env(),
            "SESSION_RELAY_INPUTS": self.setup.relay_env(),
            "SESSION_TEST_IGNORES": self.setup.test_ignores_env(),
            "UV_MAX_S": str(self.setup.uv_max_seconds),
            "TESTS_MAX_S": str(self.setup.tests_max_seconds),
        }
        if self.setup.teacher_revision:
            env["TEACHER_REVISION"] = self.setup.teacher_revision
        env.update({k: str(v) for k, v in self.setup.env.items()})
        return env

    def as_dict(self) -> dict[str, Any]:
        """The declaration, for the session record. Callables are named, not called."""
        return {
            "session_id": self.session_id,
            "schema": self.schema,
            "description": self.description,
            "authorization_path": self.authorization_path,
            "authorization_type": getattr(self.authorization_loader, "__qualname__",
                                          str(self.authorization_loader)),
            "plan_id": self.plan_id,
            "plan_hash": self.plan_hash,
            "status_path": self.status_path,
            "run_log_path": self.run_log_path,
            "driver_job_id": self.driver_job_id,
            "markers": {"success": self.markers.success,
                        "failure": list(self.markers.failure),
                        "incomplete": list(self.markers.incomplete)},
            "artifacts": {"audit_dirname": self.artifacts.audit_dirname,
                          "evidence_filename": self.artifacts.evidence_filename,
                          "archive_basename": self.artifacts.archive_basename,
                          "spec_success": self.artifacts.spec_success,
                          "spec_failed": self.artifacts.spec_failed,
                          "report_names": list(self.artifacts.report_names)},
            "setup": {
                "env": dict(self.setup.env),
                "required_env": list(self.setup.required_env),
                # The whole record, not just the path. Until 2026-08-18 this was
                # `[r.path for r in ...]`, so the session record — the artifact a
                # later reader reproduces the run from — preserved neither the
                # destination nor the digest of a single science input.
                "relay_inputs": [r.as_record()
                                 for r in self.setup.relay_inputs],
                "local_assets": [a.as_env_entry()
                                 for a in self.setup.local_assets],
                "uv_max_seconds": self.setup.uv_max_seconds,
                "tests_max_seconds": self.setup.tests_max_seconds,
                "teacher_revision": self.setup.teacher_revision,
                "test_ignores": list(self.setup.test_ignores),
            },
            "n_prechecks": len(self.precheck),
            "teardown": {"always": self.teardown.always,
                         "require_provider_confirmation":
                             self.teardown.require_provider_confirmation},
        }


#: Every attribute the RUNNER reads off the argument namespace. Eighteen
#: operational knobs, down from the twenty-one the base class read: the three
#: that left — `teacher_revision`, `uv_max_s`, `tests_max_s` — are session-shaped
#: constants and became manifest fields. The attribute that killed device-canary
#: attempt 1 stops being an argument at all.
#:
#: Declared here so `tests/pod/test_session_argument_contract.py` can check every
#: launcher's REAL parser against it, rather than against a transcription.
RUNNER_ARGUMENT_CONTRACT: tuple[str, ...] = (
    "scr", "session_commit", "bundle", "out",
    "image", "gpu", "max_price", "disk_gb",
    "token_src", "runpod_config",
    "startup_limit_min", "create_attempts", "create_retry_seconds",
    "host_draws", "setup_timeout_s", "poll_seconds", "poll_limit_min",
    "settle_seconds",
)


def missing_arguments(args: Any) -> list[str]:
    """Which runner arguments a namespace does not carry.

    Device-canary attempt 1 was lost at $0.0603 because the base read three
    attributes the subclass's parser never defined, and nothing looked until a
    pod existed. This is that look, and it is free.
    """
    return [name for name in RUNNER_ARGUMENT_CONTRACT if not hasattr(args, name)]
