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

    @pytest.mark.parametrize("flags", [
        ["--check-only"],
        ["--authorization", "/tmp/a.json"],
        ["--arm", "all_positions"],
        ["--arm", "supervised_target"],
        ["--deadline-s", "60"],
        ["--device", "cuda"],
    ])
    def test_each_flag_the_launch_chain_passes_is_known(self, driver, flags):
        """A flag the shell passes and the parser does not know is an immediate
        exit 2 on the pod, after setup has already run.

        Each is paired with an UNKNOWN flag: argparse exits 2 for the unknown one,
        which proves the parser was reached and that the known flag was accepted on
        the way. A flag the parser rejected would exit 2 as well -- so the control
        below asserts that a wholly valid set does NOT exit at parse time.
        """
        with pytest.raises(SystemExit) as exc:
            driver.main(["--out", "/tmp/d1-flag-probe", *flags,
                         "--nonexistent-flag"])
        assert exc.value.code == 2, flags

    def test_a_valid_flag_set_reaches_past_the_parser(self, driver, tmp_path):
        """The control for the probe above: these flags are accepted, and the
        driver proceeds to its first real check rather than exiting 2."""
        rc = driver.main(["--out", str(tmp_path / "r"), "--check-only",
                          "--arm", "supervised_target", "--deadline-s", "60",
                          "--device", "cuda"])
        #: 1, not 2: the CUDA probe refused on this host, which is past parsing.
        assert rc == 1
        assert (tmp_path / "r" / "d1_search.json").is_file()

    def test_an_unknown_arm_is_refused_by_the_parser(self, driver):
        with pytest.raises(SystemExit):
            driver.main(["--out", "/tmp/x", "--arm", "whatever"])


class TestTheRefusalsThatMustFireBeforeMoney:

    def test_a_paid_session_without_an_authorization_is_refused(self, driver,
                                                               tmp_path,
                                                               monkeypatch):
        """`--check-only` may omit it; a real session may not. A search that
        cannot name its authorization is unauthorized work."""
        monkeypatch.setattr(driver, "probe_environment",
                            lambda: {"gpu_name": "fake", "capability": "8.9"})
        rc = driver.main(["--out", str(tmp_path / "run"), "--device", "cpu"])
        assert rc == 1
        record = json.loads((tmp_path / "run" / "d1_search.json").read_text())
        assert record["status"] == "FAILED"
        assert "authorization is required" in record["failure"]["message"]

    def test_the_cuda_probe_refuses_a_host_run(self, driver, tmp_path):
        """And the refusal is EVIDENCE, written on the failure path."""
        rc = driver.main(["--out", str(tmp_path / "run"), "--check-only"])
        assert rc == 1
        record = json.loads((tmp_path / "run" / "d1_search.json").read_text())
        assert record["status"] == "FAILED"
        assert "no CUDA device" in record["failure"]["message"]
        #: The stage ladder records WHERE it stopped, which a bare traceback does
        #: not.
        assert record["stages"][-1]["stage"] == "A"
        assert record["stages"][-1]["status"] == "failed"

    def test_an_authorization_for_the_other_arm_is_refused(self, driver, tmp_path):
        from experiments.phase_d1 import d1_authorization as A
        from experiments.phase_d1 import d1_session as S

        payload = A.build_payload(
            grant={"granted_by": "test"}, session_commit="b" * 40,
            granted_utc="2026-10-05T00:00:00Z", arm=S.CONTROL_ARM)
        path = tmp_path / "authorization.json"
        path.write_text(json.dumps(payload))
        with pytest.raises(A.D1AuthorizationRefused, match="covers the"):
            driver.load_authorization(str(path), arm=S.TREATMENT_ARM,
                                      design_hash=payload["design_hash"])

    def test_an_authorization_bound_to_another_design_revision_is_refused(
            self, driver, tmp_path):
        from experiments.phase_d1 import d1_authorization as A
        from experiments.phase_d1 import d1_session as S

        payload = A.build_payload(
            grant={"granted_by": "test"}, session_commit="b" * 40,
            granted_utc="2026-10-05T00:00:00Z", arm=S.TREATMENT_ARM)
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
            grant={"granted_by": "test"}, session_commit="b" * 40,
            granted_utc="2026-10-05T00:00:00Z", arm=S.TREATMENT_ARM)
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
