"""The structural checks the session specification must ship with.

Three paid pods were lost to one shape of defect — a session inheriting a
requirement it never declared — and the composition design in
`docs/SESSION_ARCHITECTURE.md` is only worth the churn if the properties it
claims are checked rather than asserted. Each test below corresponds to one line
of that document's "structural checks the replacement must ship with", or to one
of the three failures:

    Phase-A attempt 1        $0.1075   SESSION_KIND leaked between two sessions
                                       sharing one setup script
    device canary attempt 1  $0.0603   the base read `self.a.teacher_revision`;
                                       the subclass had never heard of it
    device canary retry      $0.0637   the shared setup copies two assets; the
                                       subclass had declared LOCAL_ASSETS = ()

The argument contract is checked in `test_device_canary_argument_contract.py` and
the setup-environment contract in `test_launcher_forwards_setup_env.py`, because
each belongs beside the failure it prevents. What is here is everything else.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest

from session_specs import (SESSION_LAUNCHERS, all_specs,
                           load_session_launcher, session_args)

REPO = Path(__file__).resolve().parents[2]
POD = REPO / "scripts/pod"
SETUP = POD / "autoinit_preflight_setup.sh"


def test_every_session_validates():
    """`validate()` runs before anything is priced, which is the only point at
    which refusing an incomplete declaration is free."""
    for name, _mod, _args, spec in all_specs():
        spec.validate()          # raises SessionSpecError if incomplete
        assert spec.session_id and spec.schema, name


@pytest.mark.parametrize("field,value,fragment", [
    ("status_path", "", "status_path"),
    ("run_log_path", "", "run_log_path"),
    ("driver_job_id", "", "driver_job_id"),
    ("plan_hash", "", "plan_hash"),
    ("authorization_path", "", "authorization_path"),
])
def test_validate_refuses_an_incomplete_declaration(field, value, fragment):
    """A spec missing any of these would create a pod and then fail on it."""
    from aadistill.infrastructure.session import SessionSpecError

    _name, _mod, _args, spec = all_specs()[0]
    with pytest.raises(SessionSpecError, match=fragment):
        dataclasses.replace(spec, **{field: value}).validate()


def test_validate_refuses_a_spec_that_could_never_finish_or_fail():
    from aadistill.infrastructure.session import (
        ArtifactPolicy, MarkerPolicy, SessionSpecError,
    )

    _name, _mod, _args, spec = all_specs()[0]
    with pytest.raises(SessionSpecError, match="no success marker"):
        dataclasses.replace(spec, markers=MarkerPolicy(
            success="", failure=("X",))).validate()
    with pytest.raises(SessionSpecError, match="no failure markers"):
        dataclasses.replace(spec, markers=MarkerPolicy(failure=())).validate()
    with pytest.raises(SessionSpecError, match="subset"):
        dataclasses.replace(spec, markers=MarkerPolicy(
            failure=("A",), incomplete=("B",))).validate()
    with pytest.raises(SessionSpecError, match="evidence_filename"):
        dataclasses.replace(spec, artifacts=ArtifactPolicy(
            "aud", "", "a.tar.gz", "s.json", "f.json")).validate()
    with pytest.raises(SessionSpecError, match="artifact specs"):
        dataclasses.replace(spec, artifacts=ArtifactPolicy(
            "aud", "ev.json", "a.tar.gz", "", "f.json")).validate()


@pytest.mark.parametrize("kwargs,fragment", [
    ({"path": ""}, "declares no path"),
    ({"dest": "/workspace/aad/artifacts/stage1/x"}, "it is absolute"),
    ({"dest": "artifacts/../../etc"}, "escapes the repository"),
    ({"dest": "src/aadistill"}, "not one of"),
    ({"dest": "  "}, "empty dest"),
    ({"sha256": "not-a-digest"}, "which is not a sha256"),
    ({"dest": None, "also_stage_to": "artifacts/stage1/y"}, "no.*dest"),
])
def test_validate_refuses_a_malformed_relay_input(kwargs, fragment):
    """Unchecked until 2026-08-18, because the shell staged from its own list
    and the declaration was decorative. Now the declaration IS the staging, so a
    malformed one is a pod that fetches into the wrong place or verifies
    nothing."""
    from aadistill.infrastructure.session import (
        RelayInput, SessionSpecError, SetupManifest,
    )

    _name, _mod, _args, spec = all_specs()[0]
    base = {"path": "stage1/x/model.safetensors", "dest": "artifacts/stage1/x"}
    bad = RelayInput(**{**base, **kwargs})
    manifest = dataclasses.replace(spec.setup, relay_inputs=(bad,))
    assert isinstance(manifest, SetupManifest)
    with pytest.raises(SessionSpecError, match=fragment):
        dataclasses.replace(spec, setup=manifest).validate()


def test_validate_refuses_the_same_input_declared_twice():
    from aadistill.infrastructure.session import RelayInput, SessionSpecError

    _name, _mod, _args, spec = all_specs()[0]
    r = RelayInput("stage1/x/model.safetensors", dest="artifacts/stage1/x")
    manifest = dataclasses.replace(spec.setup, relay_inputs=(r, r))
    with pytest.raises(SessionSpecError, match="declared twice"):
        dataclasses.replace(spec, setup=manifest).validate()


def setup_installed_asset_names() -> set[str]:
    """Asset names the setup script copies out of `$WS/assets` by name.

    Empty is the correct answer, and the reason this test exists: on
    2026-08-17 the script contained two `cp -r "$WS/assets/<name>"` lines, so a
    session that declared no local asset had two copied into it anyway and died
    at $0.0637 under `set -e`.
    """
    # Comments are stripped first. The script explains the two lines it used to
    # contain, and an extractor that reads the explanation as the offence would
    # make this test impossible to satisfy honestly.
    code = "\n".join(l for l in SETUP.read_text().splitlines()
                     if not l.lstrip().startswith("#"))
    return set(re.findall(r'\$WS/assets/([A-Za-z0-9_.-]+)"', code))


def test_the_setup_script_installs_only_what_a_session_declares():
    hardcoded = setup_installed_asset_names()
    assert not hardcoded, (
        f"autoinit_preflight_setup.sh names {sorted(hardcoded)} itself. A "
        "session that declares no local asset would have them installed anyway, "
        "which is the $0.0637 the device-canary retry paid.")
    assert "SESSION_ASSETS" in SETUP.read_text(), (
        "the setup script no longer reads the session's asset declaration")


def test_every_declared_asset_is_a_directory_the_launcher_can_actually_send():
    """A declaration the launcher cannot satisfy fails after ssh, not at $0."""
    for name, _mod, _args, spec in all_specs():
        for asset in spec.setup.local_assets:
            path = REPO / asset.repo_path
            if not path.exists():
                pytest.skip(f"{asset.repo_path} is a gitignored local artifact")
            root = asset.install_to.strip("/").split("/")[0]
            assert root in ("artifacts", "logs", "configs"), (
                f"{name} installs {asset.dest_name} to {asset.install_to}, "
                "outside the trees a session may write")


@pytest.mark.parametrize("name,extra", SESSION_LAUNCHERS,
                         ids=lambda v: v if isinstance(v, str) else "")
def test_no_launcher_mutates_another_modules_globals(name, extra):
    """The mechanism `SESSION_KIND` leaked through, removed by construction.

    Every session used to `import` the preflight launcher and assign to its
    module globals — `_preflight.AUTH_PATH`, `_preflight.STATUS`,
    `_preflight.LOCAL_ASSETS`, `_preflight.SpendAuthorization` — before
    constructing a subclass. Two sessions sharing one module is two sessions
    sharing one set of globals.
    """
    src = (POD / f"{name}.py").read_text()
    # `_preflight.` as an attribute access, not the substring — every session
    # legitimately names `autoinit_preflight_setup.sh` and its own status file.
    assert not re.search(r"(?<![A-Za-z0-9])_preflight\.", src), (
        f"{name} still retargets a shared module")
    assert "importlib.util.spec_from_file_location" not in src, (
        f"{name} loads another launcher as a module; sessions are specifications "
        "now and share the runner, not each other")
    assert not re.search(r"^\s*class \w+\(.*Preflight.*\):", src, re.M), (
        f"{name} subclasses a launcher again")


def test_the_runner_is_not_subclassed_anywhere():
    """One runner, no inheritance. That is the whole design."""
    for path in sorted(REPO.rglob("*.py")):
        if any(p in path.parts for p in (".git", ".venv", "__pycache__")):
            continue
        if path.name in ("session_runner.py", "test_session_architecture.py"):
            continue
        text = path.read_text(errors="ignore")
        assert not re.search(r"class \w+\(\s*SessionRunner\s*\)", text), (
            f"{path.relative_to(REPO)} subclasses SessionRunner")


def test_no_attempt_specific_grant_prose_in_executable_source():
    """A grant is a one-use decision and goes stale where code does not.

    `src/aadistill/autoinit/phase_a.py` carried attempt-7's grant — which attempt
    it covered, the cumulative spend at approval, what it did not authorize —
    inside the authorization constant, where it still read as current after the
    attempt was over. The schema stays; the grant arrives at issue time.
    """
    from aadistill.autoinit.phase_a import (
        GRANT_PROSE_REQUIRED, PHASE_A_AUTHORIZATION,
    )

    assert PHASE_A_AUTHORIZATION.granted_by == GRANT_PROSE_REQUIRED, (
        "the Phase-A authorization schema carries grant prose again")
    for attr in ("granted_utc", "science_plan_hash"):
        assert getattr(PHASE_A_AUTHORIZATION, attr) == "PLACEHOLDER", attr
    assert PHASE_A_AUTHORIZATION.authorized_session_commit is None
    assert PHASE_A_AUTHORIZATION.harness_source_digest is None

    # And no authorization CONSTANT in the core carries a grant. Checked on the
    # `granted_by` field rather than on free text: a module docstring listing
    # what past attempts cost is failure history, which AGENTS.md P11 requires
    # to stay, and is not a permission.
    from aadistill.autoinit.authorization import MICRO_PREFLIGHT_AUTHORIZATION
    from aadistill.autoinit.continuation import CONTINUATION_AUTHORIZATION

    for label, constant in (("micro-preflight", MICRO_PREFLIGHT_AUTHORIZATION),
                            ("continuation", CONTINUATION_AUTHORIZATION),
                            ("phase A", PHASE_A_AUTHORIZATION)):
        prose = constant.granted_by
        offenders = re.findall(
            r"attempt \d+[^.\n]{0,80}\$\d|\$\d[^.\n]{0,80}attempt \d+", prose)
        assert not offenders, (
            f"the {label} authorization constant names an attempt beside a "
            f"dollar figure: {offenders[:2]}. That is a one-use grant living in "
            "executable source.")


def test_the_issuer_refuses_to_issue_without_a_grant_document():
    """Issuing must not be possible by running the script."""
    src = (REPO / "scripts/autoinit/issue_phase_a_authorization.py").read_text()
    assert '"--grant", required=True' in src or '--grant", required=True' in src, (
        "the grant document is optional again")
    assert "no grant document at" in src
    assert "which this script\n" in src or "which this script " in src, (
        "the issuer no longer refuses a grant that asserts a derived identity")


def test_every_session_has_a_distinct_operational_identity():
    """Two sessions sharing any of these is the $0.1324 and the $0.1369.

    **Operational identity, not scientific identity.** This list used to include
    `plan_hash`, which conflated the two: a plan hash names *what science is
    being run*, while a status file, a log, an authorization path and a job id
    name *which run is running*. Those are different questions, and two sessions
    may legitimately answer the first identically.

    The recovery continuation is exactly that case — it runs the frozen Phase-A
    plan from Stage 2 rather than a plan of its own — and requiring global
    `plan_hash` uniqueness would have forced a *frozen scientific identity* to
    change in order to satisfy a test about *file names*. See
    `test_the_recovery_continuation_shares_the_science_and_not_the_session`,
    which pins that sharing as deliberate rather than leaving it merely
    unchecked.
    """
    specs = [spec for _n, _m, _a, spec in all_specs()]
    for field in ("session_id", "schema", "status_path", "run_log_path",
                  "authorization_path", "driver_job_id"):
        values = [getattr(s, field) for s in specs]
        assert len(set(values)) == len(values), f"sessions share {field}: {values}"


def test_the_recovery_continuation_shares_the_science_and_not_the_session():
    """The one intentional `plan_hash` collision, asserted rather than allowed.

    Dropping `plan_hash` from the uniqueness list above removes a check; this
    replaces it with a stronger and more specific one. The continuation must
    share the **full Phase-A** plan hash — nothing was rewritten to pretend
    Phase A always began at Stage 2 — while being a different operational
    session in every respect that decides what runs, what it costs, and what
    permits it.
    """
    by_name = {name: spec for name, _m, _a, spec in all_specs()}
    cont = by_name["autoinit_recovery_continuation_launch"]
    phase_a = by_name["autoinit_phase_a_launch"]

    # Same science, deliberately and exactly.
    assert cont.plan_hash == phase_a.plan_hash
    assert cont.plan_id == phase_a.plan_id
    assert cont.plan_hash == (
        "9377a2dc61f21790dd111d72a5de0e039ea1d31afef2d09e18c98a0b0cc2a0aa"), (
        "the frozen Phase-A session plan moved")

    # Different operational session, in every field that names a run.
    for field in ("session_id", "schema", "status_path", "run_log_path",
                  "authorization_path", "driver_job_id"):
        assert getattr(cont, field) != getattr(phase_a, field), field

    # Its own authorization TYPE and harness, not Phase A's.
    from aadistill.autoinit.recovery_continuation import (
        RECOVERY_CONTINUATION_HARNESS_FILES_V1, RecoveryContinuationAuthorization,
    )
    from aadistill.autoinit.phase_a import (
        PHASE_A_HARNESS_SOURCE_FILES_V1, PhaseAAuthorization,
    )
    assert cont.authorization_loader == RecoveryContinuationAuthorization.load
    assert phase_a.authorization_loader == PhaseAAuthorization.load
    assert (set(RECOVERY_CONTINUATION_HARNESS_FILES_V1)
            != set(PHASE_A_HARNESS_SOURCE_FILES_V1))
    assert ("scripts/autoinit/phase_a_search.py"
            not in RECOVERY_CONTINUATION_HARNESS_FILES_V1)

    # Its own budget: the Stage-1 search phase and both Stage-1 reserves are
    # gone, so it cannot be priced as if it were running one.
    plan = cont.budget.plan(price_per_hour=0.99, authorized_usd=16.7456)
    assert not [p for p in plan.breakdown if p.name == "stage1_beam_search"]
    assert plan.soft_stop_reserves == ()
    assert plan.hard_terminate_usd == pytest.approx(16.7456, abs=1e-4)
    full = phase_a.budget.plan(price_per_hour=0.99, authorized_usd=23.0484)
    assert plan.hard_terminate_usd < full.hard_terminate_usd

    # And it says so about itself.
    assert cont.evidence_fields["runs_a_search"] is False
    assert phase_a.evidence_fields.get("runs_a_search") is not False


def test_the_specs_are_frozen():
    """A spec that could be mutated after validation is a spec that can differ
    from the one that was validated."""
    from aadistill.infrastructure.session import SessionSpec

    _n, _m, _a, spec = all_specs()[0]
    assert dataclasses.is_dataclass(SessionSpec)
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.status_path = "/workspace/somewhere_else.status"


# ---------------------------------------------------------------------------
# The relay-side twin of the LOCAL_ASSETS defect, closed 2026-08-18.
#
# `SESSION_ASSETS` made the shared setup stop naming dev-box assets, and a test
# above pins that. Nothing did the same for the relay: the script named three
# prefixes, ten filenames, four sha256 pins and a probe-to-ladder copy, and ran
# them for every session. The declarations named at most three of the ten — the
# micro-preflight and the continuation consumed the calibration mixture without
# declaring it, and the device canary was handed the whole recovery pack it had
# never asked for. `RelayInput.dest` documented the hole in its own docstring:
# "None means the setup script already knows".
# ---------------------------------------------------------------------------

def setup_code() -> str:
    """The setup script with comments stripped.

    The script explains the hardcoded fetches it used to contain, and an
    extractor that read the explanation as the offence would make these tests
    impossible to satisfy honestly. Same treatment as
    `setup_installed_asset_names`.
    """
    return "\n".join(l for l in SETUP.read_text().splitlines()
                     if not l.lstrip().startswith("#"))


def test_the_shared_setup_names_no_relay_path_of_its_own():
    """A new hardcoded science fetch must fail here, not on a paid pod.

    This is the check that did not exist. Adding
    `fetch("e8_inputs_20260810/whatever", ...)` to the shared script passed every
    structural test in the repository until this one.
    """
    code = setup_code()
    prefixes = sorted(set(re.findall(
        r"['\"]((?:stage1|stage3|stage3_recovery_corpus_v2|e8_inputs_\d+|"
        r"permanent_controls|transfer)/[A-Za-z0-9_./-]*)['\"]", code)))
    # `transfer/` names the repo bundle and the two wheelhouses, which are
    # harness inputs every session takes identically, not science a session
    # selects. They are excluded by name so this test says what it means.
    science = [p for p in prefixes if not p.startswith("transfer/")]
    assert not science, (
        f"autoinit_preflight_setup.sh names the relay paths {science} itself. A "
        "session that does not declare them would be given them anyway, which is "
        "the shape of defect that cost the device-canary retry $0.0637 one layer "
        "down. Declare them in the session's `relay_inputs`.")


def test_the_shared_setup_pins_no_digest_of_its_own():
    """A digest in the shell is a digest no session's declaration carries."""
    digests = sorted(set(re.findall(r"\b[0-9a-f]{64}\b", setup_code())))
    assert not digests, (
        f"autoinit_preflight_setup.sh hardcodes the sha256 pins {digests}. They "
        "belong on the `RelayInput` that stages the file, so the session record "
        "preserves what was verified.")


def test_the_shared_setup_names_no_science_destination_of_its_own():
    """Where a science input lands is the session's declaration, not the shell's.

    A `glob` over `artifacts/stage1/*/checkpoint` is allowed and is not an
    exception: it adapts to whatever the session staged, and refuses when a
    session staged nothing.
    """
    code = setup_code()
    # `*` is inside the class, so a glob is matched whole and then excluded. A
    # class that omitted it would cut `artifacts/stage1/*/checkpoint` down to
    # `artifacts/stage1/` and report the glob as a literal.
    literals = sorted({m for m in re.findall(r"artifacts/[A-Za-z0-9_./*-]+", code)
                       if "*" not in m})
    assert not literals, (
        f"autoinit_preflight_setup.sh writes to {literals} by name; the "
        "destination is a field on the session's `RelayInput`.")
    assert "SESSION_RELAY_INPUTS" in code, (
        "the setup script no longer reads the session's science-input manifest")


@pytest.mark.parametrize("name,extra", SESSION_LAUNCHERS,
                         ids=lambda v: v if isinstance(v, str) else "")
def test_every_staged_input_declares_a_destination_and_the_env_carries_it(name, extra):
    """`dest=None` may mean "the driver stages it". It may not mean "the shell knows"."""
    import json

    mod = load_session_launcher(name)
    spec = mod.spec(session_args(mod, extra))
    env = spec.setup_environment(session_commit="0" * 40, bundle="aad.bundle")
    staged = spec.setup.staged_relay_inputs()
    assert staged, f"{name} stages no science input at all"
    carried = json.loads(env["SESSION_RELAY_INPUTS"])
    assert [r["path"] for r in carried] == [r.path for r in staged], (
        f"{name}'s SESSION_RELAY_INPUTS is not its staged declaration")
    for r in carried:
        assert r["dest"], f"{name} would ask setup to stage {r['path']} nowhere"
    # And the precheck-only ones are NOT handed to setup: setup receives what it
    # is meant to do, not the whole declaration to filter itself.
    unstaged = {r.path for r in spec.setup.relay_inputs if not r.staged}
    assert unstaged.isdisjoint({r["path"] for r in carried}), (
        f"{name} hands setup {unstaged & {r['path'] for r in carried}}, which "
        "the driver stages by another route")


@pytest.mark.parametrize("name,extra", SESSION_LAUNCHERS,
                         ids=lambda v: v if isinstance(v, str) else "")
def test_a_checkpoint_is_staged_with_the_files_it_cannot_load_without(name, extra):
    """Weights are not a checkpoint. `logs/autoinit_control_sb_packaging_repair.json`
    is the write-up of a control whose identity gates all passed and which could
    not be evaluated, because it shipped without its tokenizer."""
    mod = load_session_launcher(name)
    spec = mod.spec(session_args(mod, extra))
    staged = {r.path.rsplit("/", 1)[-1]
              for r in spec.setup.staged_relay_inputs()
              if r.path.startswith("stage1/qwen3_0p6b_init_v0/checkpoint/")}
    if not staged:
        pytest.skip(f"{name} stages no checkpoint")
    missing = {"config.json", "tokenizer.json", "tokenizer_config.json",
               "generation_config.json", "chat_template.jinja",
               "model.safetensors"} - staged
    assert not missing, (
        f"{name} stages the canonical init without {sorted(missing)}; the "
        "checkpoint would not load")


def test_the_calibration_pin_matches_the_registry_that_already_carried_it():
    """One hash, two homes, and they may not drift.

    `aadistill.autoinit.datasets.E8A_CALIBRATION` has carried this file's hash
    since E8a. The shared setup carried a second copy of it, and nothing compared
    them.
    """
    import sys

    sys.path.insert(0, str(POD))
    from autoinit_science_inputs import CALIBRATION_V1

    from aadistill.autoinit.datasets import E8A_CALIBRATION

    pins = {r.sha256 for r in CALIBRATION_V1 if r.sha256}
    assert pins == {E8A_CALIBRATION.content_sha256}, (
        f"the staging pin {pins} and the registry's "
        f"{E8A_CALIBRATION.content_sha256!r} disagree")
    dests = {r.dest for r in CALIBRATION_V1}
    assert E8A_CALIBRATION.path.startswith(tuple(d + "/" for d in dests)), (
        f"the calibration is staged into {dests} and read from "
        f"{E8A_CALIBRATION.path!r}")


# ---------------------------------------------------------------------------
# The staging block, EXECUTED. Not inspected, not simulated — run.
#
# Four paid pods have died inside lines no $0 path could reach. The block below
# is extracted from the shell file itself and run against a temporary tree with
# a stub relay, so a typo, a wrong key or a missing mkdir fails here for free.
# Rehearsing a copy would prove nothing about the file the pod executes.
# ---------------------------------------------------------------------------

def extract_staging_block() -> str:
    """The `FETCHEOF` heredoc body, verbatim from the shell script."""
    text = SETUP.read_text()
    start = text.index("<<'FETCHEOF'\n") + len("<<'FETCHEOF'\n")
    end = text.index("\nFETCHEOF", start)
    body = text[start:end]
    assert "SESSION_RELAY_INPUTS" in body and "hf_hub_download" in body
    return body


def run_staging(tmp_path, inputs, relay_bytes, monkeypatch):
    """Execute the real block with a stub `huggingface_hub`, in `tmp_path`."""
    import json
    import sys
    import types

    cache = tmp_path / "cache"
    cache.mkdir()

    fetched: list[str] = []

    def hf_hub_download(repo, path, repo_type=None, token=None):
        fetched.append(path)
        if path not in relay_bytes:
            raise FileNotFoundError(path)
        p = cache / path.replace("/", "__")
        p.write_bytes(relay_bytes[path])
        return str(p)

    stub = types.ModuleType("huggingface_hub")
    stub.hf_hub_download = hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", stub)

    repo = tmp_path / "aad"
    repo.mkdir()
    monkeypatch.setenv("REPO", str(repo))
    monkeypatch.setenv("HF_TOKEN", "stub-token")
    monkeypatch.setenv("SESSION_RELAY_INPUTS", json.dumps(inputs))
    # `time.sleep` would make the retry path cost 75 s of test wall clock.
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)

    code = compile(extract_staging_block(), str(SETUP) + ":FETCHEOF", "exec")
    exec(code, {"__name__": "__main__"})           # noqa: S102 — the point
    return repo, fetched


def test_the_staging_block_stages_exactly_what_it_is_given(tmp_path, monkeypatch):
    import hashlib

    payload = b"weights\n"
    digest = hashlib.sha256(payload).hexdigest()
    other = b"ladder\n"
    inputs = [
        {"path": "stage1/x/checkpoint/model.safetensors",
         "dest": "artifacts/stage1/x/checkpoint",
         "sha256": digest, "also_stage_to": None},
        {"path": "corpus/ladder/blocks.npz", "dest": "artifacts/stage3/probe",
         "sha256": hashlib.sha256(other).hexdigest(),
         "also_stage_to": "artifacts/stage3/mirror"},
    ]
    relay = {"stage1/x/checkpoint/model.safetensors": payload,
             "corpus/ladder/blocks.npz": other}
    repo, fetched = run_staging(tmp_path, inputs, relay, monkeypatch)

    assert fetched == [i["path"] for i in inputs], "fetched something else"
    assert (repo / "artifacts/stage1/x/checkpoint/model.safetensors"
            ).read_bytes() == payload
    # The mirror is the probe-to-ladder copy, now declared instead of walked.
    assert (repo / "artifacts/stage3/probe/blocks.npz").read_bytes() == other
    assert (repo / "artifacts/stage3/mirror/blocks.npz").read_bytes() == other
    # And nothing it was not given.
    staged = {p.relative_to(repo).as_posix()
              for p in repo.rglob("*") if p.is_file()}
    assert staged == {"artifacts/stage1/x/checkpoint/model.safetensors",
                      "artifacts/stage3/probe/blocks.npz",
                      "artifacts/stage3/mirror/blocks.npz"}


def test_the_staging_block_stages_nothing_when_a_session_declares_nothing(
        tmp_path, monkeypatch):
    """The canary retry's failure, at the relay layer. It must be a no-op."""
    repo, fetched = run_staging(tmp_path, [], {}, monkeypatch)
    assert fetched == []
    assert not [p for p in repo.rglob("*") if p.is_file()]


def test_the_staging_block_refuses_a_wrong_digest(tmp_path, monkeypatch):
    import hashlib

    inputs = [{"path": "a/b.bin", "dest": "artifacts/stage1/x",
               "sha256": hashlib.sha256(b"expected").hexdigest(),
               "also_stage_to": None}]
    with pytest.raises(SystemExit) as e:
        run_staging(tmp_path, inputs, {"a/b.bin": b"different"}, monkeypatch)
    assert "FROZEN ASSET MISMATCH" in str(e.value)


def test_the_staging_block_verifies_the_mirror_too(tmp_path, monkeypatch, capsys):
    """The old block pinned `ladder_uniform/blocks.npz` separately from the probe
    copy. Checking one landing site and not the other silently drops half of a
    frozen pin — and would miss a mirror that failed to copy at all.

    Asserted on the block's OUTPUT, not on its source text. The first version of
    this test grepped for `for rel in landed`, which `for rel in landed[:1]`
    still contains: the mutation that removes the second check left the gate
    green.
    """
    import hashlib

    inputs = [{"path": "a/b.bin", "dest": "artifacts/stage1/x",
               "sha256": hashlib.sha256(b"ok").hexdigest(),
               "also_stage_to": "artifacts/stage1/y"}]
    repo, _ = run_staging(tmp_path, inputs, {"a/b.bin": b"ok"}, monkeypatch)
    out = capsys.readouterr().out
    assert (repo / "artifacts/stage1/x/b.bin").is_file()
    assert (repo / "artifacts/stage1/y/b.bin").is_file()
    assert "artifacts/stage1/x/b.bin " in out, "the primary was not verified"
    assert "artifacts/stage1/y/b.bin " in out, "the mirror was not verified"
    assert "verified 2 digests" in out, (
        f"one file landed in two places and the block reports: {out!r}")


def test_the_staging_block_refuses_an_input_with_no_destination(
        tmp_path, monkeypatch):
    """Setup receives what it must stage. A dest-less entry reaching it means
    the precheck-only filter broke, and staging it somewhere invented would be
    worse than stopping."""
    inputs = [{"path": "a/b.bin", "dest": None, "sha256": None,
               "also_stage_to": None}]
    with pytest.raises(SystemExit) as e:
        run_staging(tmp_path, inputs, {"a/b.bin": b"x"}, monkeypatch)
    assert "no dest" in str(e.value)


def test_the_staging_block_gives_up_and_names_the_file(tmp_path, monkeypatch):
    inputs = [{"path": "missing/thing.bin", "dest": "artifacts/stage1/x",
               "sha256": None, "also_stage_to": None}]
    with pytest.raises(SystemExit) as e:
        run_staging(tmp_path, inputs, {}, monkeypatch)
    assert "FETCH FAILED missing/thing.bin" in str(e.value)


@pytest.mark.parametrize("name,extra", SESSION_LAUNCHERS,
                         ids=lambda v: v if isinstance(v, str) else "")
def test_each_real_session_manifest_stages_through_the_real_block(
        name, extra, tmp_path, monkeypatch):
    """End to end at $0: this session's actual declaration, through the actual
    shell code, landing the actual destinations — with the relay stubbed."""
    import hashlib
    import json

    mod = load_session_launcher(name)
    spec = mod.spec(session_args(mod, extra))
    env = spec.setup_environment(session_commit="0" * 40, bundle="aad.bundle")
    declared = json.loads(env["SESSION_RELAY_INPUTS"])

    # The stub relay serves arbitrary bytes, and each declared pin is restated
    # as the digest of what is actually served — a sha256 preimage is not
    # available and is not what this test is for. What it exercises is the real
    # declaration's paths, destinations and mirrors through the real code. That
    # the frozen digests are the RIGHT ones is a separate claim, checked by
    # `scripts/autoinit/verify_frozen_assets.py` and again on the pod.
    relay, expect = {}, set()
    for r in declared:
        body = f"content of {r['path']}".encode()
        relay[r["path"]] = body
        if r["sha256"]:
            r["sha256"] = hashlib.sha256(body).hexdigest()
        nm = r["path"].rsplit("/", 1)[-1]
        for into in (r["dest"], r["also_stage_to"]):
            if into:
                expect.add(f"{into}/{nm}")

    repo, fetched = run_staging(tmp_path, declared, relay, monkeypatch)
    landed = {p.relative_to(repo).as_posix()
              for p in repo.rglob("*") if p.is_file()}
    assert landed == expect, f"{name} staged {landed ^ expect} unexpectedly"
    assert len(fetched) == len(declared)
    for r in declared:
        if r["sha256"]:
            got = hashlib.sha256(
                (repo / f"{r['dest']}/{r['path'].rsplit('/', 1)[-1]}"
                 ).read_bytes()).hexdigest()
            assert got == r["sha256"]
