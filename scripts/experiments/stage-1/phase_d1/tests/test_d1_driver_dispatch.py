"""Every name the D1 driver reaches for, resolved at `$0`.

The driver's expensive stages cannot run on a CPU box — the environment probe
refuses, deliberately, because a CPU search answers a different question. That is
exactly the condition under which a NameError or a wrong import inside stage C
first appears at $1.09/h, one step after a passing preflight.

This round produced one: the driver imported `WallClockDeadline` from
`planning.search`, where the type is called `Deadline`. The import sits inside
`_run_search`, so nothing before the beam would have touched it.

So these resolve every symbol the driver's unreachable stages name, assert the
argument surface, and drive the refusals that must fire before any money is spent.
"""
from __future__ import annotations

import ast
import importlib
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[5]
for extra in ("src", "scripts", "scripts/pod", "scripts/autoinit"):
    path = str(REPO / extra)
    if path not in sys.path:
        sys.path.insert(0, path)

DRIVER = REPO / "scripts/pod/autoinit_d1_driver.py"


@pytest.fixture(scope="module")
def driver():
    return importlib.import_module("autoinit_d1_driver")


def _deferred_imports(source: str) -> list[tuple[str, tuple[str, ...]]]:
    """Every `from X import a, b` that sits INSIDE a function.

    Module-level imports fail at import time and are therefore already covered by
    importing the driver. The dangerous ones are the deferred ones: they are not
    executed until the stage that needs them, which on this driver means not until
    the pod is billing.
    """
    out: list[tuple[str, tuple[str, ...]]] = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.ImportFrom) and sub.module:
                out.append((sub.module, tuple(a.name for a in sub.names)))
            elif isinstance(sub, ast.Import):
                for alias in sub.names:
                    out.append((alias.name, ()))
    return out


class TestEveryDeferredImportResolves:
    """The failure mode that cost this round a defect, closed generically."""

    def test_there_are_deferred_imports_to_check(self):
        found = _deferred_imports(DRIVER.read_text())
        assert len(found) >= 6, (
            f"only {found} deferred imports found; either the driver changed "
            "shape or this parser stopped seeing them, and a parser that cannot "
            "see a real import is the same hazard as a missing check")

    @pytest.mark.parametrize("module,names", _deferred_imports(DRIVER.read_text()))
    def test_the_module_and_every_name_exist(self, module, names):
        mod = importlib.import_module(module)
        missing = []
        for name in names:
            if hasattr(mod, name):
                continue
            #: `from pkg import submodule` is legal and the submodule is NOT an
            #: attribute of the package until it is imported, which is what Python
            #: itself does here. Treating that as missing would flag every
            #: package-relative import in the file.
            try:
                importlib.import_module(f"{module}.{name}")
            except ImportError:
                missing.append(name)
        assert not missing, (
            f"{module} has no {missing}. This import sits inside a function the "
            "CUDA probe keeps unreachable on a CPU box, so it would first fail on "
            "a billing machine.")


class TestTheArgumentSurface:

    #: EVERY FLAG THE LAUNCH CHAIN EMITS. `--run-id` and `--status` joined it
    #: when the evidence layout moved: the run id is no longer derivable from
    #: the output directory's basename, and the status file is where the
    #: terminal marker the launcher polls for is appended.
    LAUNCH_FLAGS = [
        ["--check-only"],
        ["--authorization", "/tmp/a.json"],
        ["--arm", "all_positions"],
        ["--arm", "supervised_target"],
        ["--deadline-s", "60"],
        ["--device", "cuda"],
        ["--status", "/tmp/d1-flag-probe.status"],
    ]

    @pytest.mark.parametrize("flags", LAUNCH_FLAGS)
    def test_each_flag_the_launch_chain_passes_is_known(self, driver, flags,
                                                        tmp_path):
        """A flag the shell passes and the parser does not know is an immediate
        exit 2 on the pod, after setup has already run.

        Each is paired with an UNKNOWN flag: argparse exits 2 for the unknown one,
        which proves the parser was reached and that the known flag was accepted on
        the way. A flag the parser rejected would exit 2 as well -- so the control
        below asserts that a wholly valid set does NOT exit at parse time.

        `--run-id` is supplied because it is REQUIRED: without it argparse exits
        2 for the missing argument instead of for the unknown flag, and the
        probe would then pass even for a flag the parser rejects.
        """
        with pytest.raises(SystemExit) as exc:
            driver.main(["--out", str(tmp_path / "probe"),
                         "--run-id", "flag_probe", *flags,
                         "--nonexistent-flag"])
        assert exc.value.code == 2, flags

    def test_every_launch_chain_flag_is_covered(self):
        """The list above must be the flags the launcher really emits. A flag
        added to the command and not to this list is untested at the exact point
        argparse would reject it."""
        import shlex
        import types

        import autoinit_d1_launch as L

        args = L.build_parser().parse_args(
            ["--scr", "/tmp/x", "--session-commit", "d" * 40,
             "--bundle", "aad_autoinit_dddddddd.bundle", "--run-id", "probe"])
        cmd = L.driver_command(types.SimpleNamespace(args=args),
                               types.SimpleNamespace(soft_stop_seconds=600.0))
        emitted = {t for t in shlex.split(cmd) if t.startswith("--")}
        covered = {f[0] for f in self.LAUNCH_FLAGS} | {"--out", "--run-id"}
        assert emitted <= covered, emitted - covered

    def test_a_valid_flag_set_reaches_past_the_parser(self, driver, tmp_path):
        """The control for the probe above: these flags are accepted, and the
        driver proceeds to its first real check rather than exiting 2."""
        rc = driver.main(["--out", str(tmp_path / "r"), "--check-only",
                          "--run-id", "flag_probe",
                          "--status", str(tmp_path / "s"),
                          "--arm", "supervised_target", "--deadline-s", "60",
                          "--device", "cuda"])
        #: 1, not 2: the CUDA probe refused on this host, which is past parsing.
        assert rc == 1
        assert (tmp_path / "r" / "d1_search.json").is_file()

    def test_an_unknown_arm_is_refused_by_the_parser(self, driver):
        with pytest.raises(SystemExit):
            driver.main(["--out", "/tmp/x", "--run-id", "p",
                         "--arm", "whatever"])

    def test_a_run_id_is_required(self, driver, tmp_path):
        """It enters `SearchConfig.config_hash`, so a session that cannot name
        its run cannot be checked against the identity its grant binds."""
        with pytest.raises(SystemExit) as exc:
            driver.main(["--out", str(tmp_path / "r"), "--check-only"])
        assert exc.value.code == 2


class TestTheRefusalsThatMustFireBeforeMoney:

    def test_a_paid_session_without_an_authorization_is_refused(self, driver,
                                                               tmp_path,
                                                               monkeypatch):
        """`--check-only` may omit it; a real session may not. A search that
        cannot name its authorization is unauthorized work."""
        monkeypatch.setattr(driver, "probe_environment",
                            lambda: {"gpu_name": "fake", "capability": "8.9"})
        rc = driver.main(["--out", str(tmp_path / "run"), "--run-id", "dispatch_probe",
                    "--status", str(tmp_path / "s"), "--device", "cpu"])
        assert rc == 1
        record = json.loads((tmp_path / "run" / "d1_search.json").read_text())
        assert record["status"] == "FAILED"
        assert "authorization is required" in record["failure"]["message"]

    def test_the_cuda_probe_refuses_a_host_run(self, driver, tmp_path):
        """And the refusal is EVIDENCE, written on the failure path."""
        rc = driver.main(["--out", str(tmp_path / "run"), "--run-id", "dispatch_probe",
                    "--status", str(tmp_path / "s"), "--check-only"])
        assert rc == 1
        record = json.loads((tmp_path / "run" / "d1_search.json").read_text())
        assert record["status"] == "FAILED"
        assert "no CUDA device" in record["failure"]["message"]
        #: The stage ladder records WHERE it stopped, which a bare traceback does
        #: not.
        assert record["stages"][-1]["stage"] == "A"
        assert record["stages"][-1]["status"] == "failed"

    def test_a_control_arm_authorization_cannot_even_be_built(self):
        """A formal D1 search is treatment-only, so the refusal is at ISSUANCE.
        The control arm stays constructible as a $0 session for protocol-identity
        checks; what it may not be is a paid beam."""
        from experiments.phase_d1 import d1_authorization as A
        from experiments.phase_d1 import d1_session as S

        with pytest.raises(A.D1AuthorizationRefused, match="may not be issued"):
            A.build_payload(run_id="dispatch_probe", grant={"granted_by": "test"},
                            session_commit="b" * 40,
                            granted_utc="2026-10-06T00:00:00Z",
                            arm=S.CONTROL_ARM, live_rate=1.09)

    def test_the_driver_refuses_an_artifact_whose_arm_was_altered(
            self, driver, tmp_path):
        """The driver's own check, driven against an artifact that claims another
        arm -- which can only arise from editing one, and the self-hash catches
        that first. Both layers are asserted, in order."""
        from aadistill.infrastructure.manifest import sha256_json
        from experiments.phase_d1 import d1_authorization as A
        from experiments.phase_d1 import d1_session as S

        payload = A.build_payload(
            run_id="dispatch_probe", grant={"granted_by": "test"}, session_commit="b" * 40,
            granted_utc="2026-10-06T00:00:00Z", live_rate=1.09)
        payload["arm"] = S.CONTROL_ARM
        path = tmp_path / "edited.json"
        path.write_text(json.dumps(payload))
        #: FIRST the self-hash, because the artifact was edited.
        with pytest.raises(A.D1AuthorizationRefused,
                           match="authorization_sha256"):
            driver.load_authorization(str(path), arm=S.TREATMENT_ARM,
                                      design_hash=payload["design_hash"])
        #: Re-hashed, so the driver's arm check is what is left.
        payload.pop("authorization_sha256", None)
        payload["authorization_sha256"] = sha256_json(payload)
        rehashed = tmp_path / "rehashed.json"
        rehashed.write_text(json.dumps(payload))
        with pytest.raises(A.D1AuthorizationRefused, match="covers the"):
            driver.load_authorization(str(rehashed), arm=S.TREATMENT_ARM,
                                      design_hash=payload["design_hash"])

    def test_an_authorization_bound_to_another_design_revision_is_refused(
            self, driver, tmp_path):
        from experiments.phase_d1 import d1_authorization as A
        from experiments.phase_d1 import d1_session as S

        payload = A.build_payload(
            run_id="dispatch_probe", grant={"granted_by": "test"}, session_commit="b" * 40,
            granted_utc="2026-10-06T00:00:00Z", live_rate=1.09)
        path = tmp_path / "authorization.json"
        path.write_text(json.dumps(payload))
        with pytest.raises(A.D1AuthorizationRefused, match="different design"):
            driver.load_authorization(str(path), arm=S.TREATMENT_ARM,
                                      design_hash="0" * 64)

    def test_the_real_authorization_is_accepted(self, driver, tmp_path):
        """So the refusals above are not refusing everything."""
        from experiments.phase_d1 import d1_authorization as A
        from experiments.phase_d1 import d1_session as S

        payload = A.build_payload(
            run_id="dispatch_probe", grant={"granted_by": "test"}, session_commit="b" * 40,
            granted_utc="2026-10-06T00:00:00Z", live_rate=1.09)
        path = tmp_path / "authorization.json"
        path.write_text(json.dumps(payload))
        auth = driver.load_authorization(str(path), arm=S.TREATMENT_ARM,
                                        design_hash=payload["design_hash"])
        assert auth.allows_beam_search is True
        assert auth.allows_recovery is False


class TestTheDriverOwnsNoScience:
    """Every frozen number must come from the design or the session, not here."""

    FORBIDDEN = ("151936", "0.86", "E1_KD_HEAVY", "supervised_target_v1@v1",
                 "top_k=200", "micro_batch_size=3")

    def test_no_frozen_constant_is_typed_into_the_driver(self):
        source = DRIVER.read_text()
        hits = [token for token in self.FORBIDDEN if token in source]
        assert not hits, (
            f"the driver carries {hits}; a frozen scientific constant typed into "
            "an execution script is a second owner of the experiment's shape")

    def test_the_beam_and_the_space_are_not_named_here(self):
        source = DRIVER.read_text()
        assert "width=6" not in source and "allowed_impls=(" not in source, (
            "the driver names the beam or the space; both come from the session, "
            "which reads them from the design")

    def test_it_commits_and_does_not_decide(self, driver):
        """Stage D is commit_top_k. A promotion decision here would be a session
        doing work no authorization covers."""
        source = DRIVER.read_text()
        for forbidden in ("promote", "GO", "NO-GO", "verdict"):
            assert f'"{forbidden}"' not in source, forbidden
        assert "_commit" in dir(driver)


def _authorization_attributes(source: str) -> list[str]:
    """Every attribute this driver reads off the LOADED authorization.

    `auth` is the only name the driver binds to a `D1Authorization`, in
    `load_authorization` and in stage A. So an `auth.<name>` read is a
    cross-module vocabulary claim: it asserts that the authorization module
    calls a field by the name this driver typed.
    """
    return sorted({
        node.attr
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "auth"
    })


class TestEveryAuthorizationAttributeTheDriverReadsExists:
    """THE $0.24 DEFECT, closed as a class rather than as one name.

    `TestEveryDeferredImportResolves` above resolved every deferred *import*,
    because an unresolvable import inside stage C would first appear at
    $1.09/h. An attribute read is the same hazard through a different hole and
    was not covered: stage A built its audit record from
    `auth.session_commit`, the authorization declares that field as
    `authorized_session_commit`, and the first real pod raised

        AttributeError: 'D1Authorization' object has no attribute
                        'session_commit'

    1.69 s into stage A -- after setup had passed all eight markers, D1's own
    338-test pod gate had passed, and $0.24 was spent.

    Nothing at $0 could have caught it. The six pre-provider gates load the
    same authorization through the same loader, but none of them reads this
    attribute; the only code that does sits behind `if args.authorization`,
    and a `--check-only` rehearsal is explicitly permitted to omit
    `--authorization` and skip the whole stage. A CPU rehearsal that DID pass
    one could not have got past the identity comparison either, because
    `config_hash` binds `device` and the authorization was issued on `cuda`.

    So the check is static, against the real class's real attribute surface.
    """

    @staticmethod
    def _surface() -> set[str]:
        """The authorization's REAL attribute surface, from the real class.

        `dataclasses.fields` alone is not it -- `authorizes_d1_search` is a
        field and `require_harness` is a method, and the driver legitimately
        reads both kinds. `hasattr` on the class alone is not it either: a
        dataclass field with no default is not a class attribute, which makes
        `authorization_id`, `hard_cap_usd` and `authorized_stages` look absent
        when they are declared. Both halves, or the probe reports the wrong
        three names and misses the real one.
        """
        import dataclasses

        from experiments.phase_d1.d1_authorization import D1Authorization

        return ({f.name for f in dataclasses.fields(D1Authorization)}
                | {n for n in dir(D1Authorization) if not n.startswith("_")})

    def test_the_parser_sees_the_reads(self):
        found = _authorization_attributes(DRIVER.read_text())
        assert len(found) >= 8, (
            f"only {found} authorization reads found; a parser that cannot see "
            "a real read is the same hazard as no check at all")

    def test_every_read_name_is_declared_by_the_authorization(self):
        surface = self._surface()
        missing = [n for n in _authorization_attributes(DRIVER.read_text())
                   if n not in surface]
        assert not missing, (
            f"the driver reads auth.{missing} and D1Authorization declares no "
            "such attribute. Stage A runs only on a billing pod, so this first "
            "appears after setup is paid for")

    def test_the_probe_bites_on_the_name_that_cost_the_money(self):
        """Non-vacuity, against the exact text that failed."""
        surface = self._surface()
        assert "authorized_session_commit" in surface
        assert "session_commit" not in surface, (
            "the authorization now declares a bare `session_commit` too, so "
            "this probe can no longer distinguish the two vocabularies and the "
            "test above has stopped protecting anything")
        hypothetical = _authorization_attributes(
            "record = {'c': auth.session_commit}")
        assert [n for n in hypothetical if n not in surface] == [
            "session_commit"]
