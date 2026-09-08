"""Adapter resolution must not depend on who imported what first.

The registry used to be a module-level dict that the concrete adapter filled at
import time. Whether `get_adapter("qwen3")` resolved therefore depended on
whether something, anywhere, had already pulled that module in. With
`pytest-randomly` active this made whole test FILES fail at collection in one
run and pass in the next, and on a pod it would have meant a driver resolving an
adapter because of an unrelated import three modules away.

The order-independence tests run in **fresh subprocesses**. That is the only way
to check it: once this interpreter has imported anything, the registry may
already be populated, and an in-process assertion would be measuring the test
session rather than the contract.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def in_fresh_process(body: str) -> subprocess.CompletedProcess:
    """Run `body` in a new interpreter with nothing imported yet."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)], cwd=REPO,
        capture_output=True, text=True,
        env={"PYTHONPATH": "src:scripts", "PATH": "/usr/bin:/bin",
             "HOME": str(REPO)})


class TestNothingRegistersOnImport:
    def test_importing_the_adapters_package_registers_nothing(self):
        out = in_fresh_process("""
            import aadistill.initialization.adapters
            from aadistill.initialization.specs.arch import registered_families
            print(registered_families())
        """)
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "[]", (
            "importing the package must not populate the registry")

    def test_importing_the_concrete_adapter_registers_nothing(self):
        out = in_fresh_process("""
            import aadistill.initialization.adapters.qwen3
            from aadistill.initialization.specs.arch import registered_families
            print(registered_families())
        """)
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "[]"

    def test_importing_planning_and_operators_registers_nothing(self):
        out = in_fresh_process("""
            import aadistill.initialization.planning.search
            import aadistill.initialization.operators.depth
            import aadistill.initialization.operators.attention
            from aadistill.initialization.specs.arch import registered_families
            print(registered_families())
        """)
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "[]"

    def test_the_package_init_does_not_import_the_adapters(self):
        """The specific side effect the cutover removed; keep it removed."""
        init = (REPO / "src/aadistill/initialization/__init__.py").read_text()
        assert "import" not in init, (
            "aadistill.initialization.__init__ must stay inert; re-importing "
            "adapters here would restore exactly the coupling that was removed")


class TestResolutionRequiresExplicitBootstrap:
    def test_an_unregistered_family_refuses_and_says_what_to_call(self):
        out = in_fresh_process("""
            from aadistill.initialization.specs.arch import get_adapter
            try:
                get_adapter("qwen3")
            except KeyError as exc:
                print("REFUSED", exc)
        """)
        assert out.returncode == 0, out.stderr
        assert "REFUSED" in out.stdout
        assert "register_builtin_adapters" in out.stdout, (
            "the refusal must name the call that fixes it, or the next person "
            "re-adds an import-time side effect to make it go away")

    def test_explicit_registration_succeeds(self):
        out = in_fresh_process("""
            from aadistill.initialization.adapters import register_builtin_adapters
            from aadistill.initialization.specs.arch import get_adapter
            register_builtin_adapters()
            a = get_adapter("qwen3")
            print(a.family, a.adapter_version)
        """)
        assert out.returncode == 0, out.stderr
        assert out.stdout.split()[0] == "qwen3"

    @pytest.mark.parametrize("order", [
        ("aadistill.initialization.operators.depth",
         "aadistill.initialization.adapters"),
        ("aadistill.initialization.adapters",
         "aadistill.initialization.operators.depth"),
        ("aadistill.initialization.planning.search",
         "aadistill.initialization.adapters.qwen3"),
        ("aadistill.initialization.adapters.qwen3",
         "aadistill.initialization.planning.search"),
    ], ids=lambda o: f"{o[0].split('.')[-1]}_then_{o[1].split('.')[-1]}")
    def test_the_outcome_is_the_same_in_any_import_order(self, order):
        out = in_fresh_process(f"""
            import {order[0]}
            import {order[1]}
            from aadistill.initialization.specs.arch import (
                get_adapter, registered_families)
            print("before", registered_families())
            from aadistill.initialization.adapters import register_builtin_adapters
            register_builtin_adapters()
            print("after", get_adapter("qwen3").family)
        """)
        assert out.returncode == 0, out.stderr
        assert "before []" in out.stdout, "import order must not pre-register"
        assert "after qwen3" in out.stdout


class TestTheRegistryContract:
    def test_a_caller_can_own_its_registry(self):
        from aadistill.initialization.adapters import build_registry
        registry = build_registry()
        assert registry.families() == ["qwen3"]

    def test_two_registries_are_independent(self):
        from aadistill.initialization.adapters import build_registry
        from aadistill.initialization.specs.arch import AdapterRegistry
        assert build_registry().families() == ["qwen3"]
        assert AdapterRegistry().families() == [], (
            "a fresh registry must be empty; shared state would put us back "
            "where we started")

    def test_registering_the_same_family_twice_is_idempotent(self):
        from aadistill.initialization.adapters import QWEN3_ADAPTER
        from aadistill.initialization.specs.arch import AdapterRegistry
        registry = AdapterRegistry()
        first = registry.register(QWEN3_ADAPTER)
        second = registry.register(QWEN3_ADAPTER)
        assert first is second
        assert registry.families() == ["qwen3"]

    def test_a_different_version_for_the_same_family_is_refused(self):
        """A manifest records the adapter version it ran under."""
        from aadistill.initialization.adapters import Qwen3Adapter
        from aadistill.initialization.specs.arch import AdapterRegistry
        registry = AdapterRegistry()
        registry.register(Qwen3Adapter())
        other = Qwen3Adapter()
        other.adapter_version = "qwen3.other"
        with pytest.raises(ValueError, match="already registered at version"):
            registry.register(other)

    def test_replace_is_available_but_must_be_asked_for(self):
        from aadistill.initialization.adapters import Qwen3Adapter
        from aadistill.initialization.specs.arch import AdapterRegistry
        registry = AdapterRegistry()
        registry.register(Qwen3Adapter())
        other = Qwen3Adapter()
        other.adapter_version = "qwen3.other"
        registry.register(other, replace=True)
        assert registry.get("qwen3").adapter_version == "qwen3.other"

    def test_an_unknown_builtin_family_is_refused(self):
        from aadistill.initialization.adapters import build_registry
        with pytest.raises(KeyError, match="no builtin adapter"):
            build_registry(["not_a_family"])


class TestNoExperimentProfileRegistersAsASideEffect:
    """§3: bootstrapping adapters must not drag in experiment-instance policy."""

    def test_registering_adapters_registers_no_calibration_profile(self):
        out = in_fresh_process("""
            from aadistill.initialization.adapters import register_builtin_adapters
            from aadistill.initialization.calibration import profiles
            register_builtin_adapters()
            print(sorted(profiles.registered_profiles()))
        """)
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "[]", (
            "a concrete experiment profile appearing here would mean the "
            "adapter bootstrap carries experiment policy with it")

    def test_importing_the_core_registers_no_profile(self):
        out = in_fresh_process("""
            import aadistill.initialization.calibration.profiles as p
            import aadistill.initialization.operators.base          # noqa: F401
            import aadistill.initialization.planning.search         # noqa: F401
            print(sorted(p.registered_profiles()))
        """)
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "[]"
