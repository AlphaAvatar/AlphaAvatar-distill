"""What a paid C3 pod must prove before it spends an hour on anything.

`pytest tests/ $SESSION_TEST_IGNORES` runs on the pod as a blocking gate, and
the ignore complement leaves exactly this directory collectable. So these are
the checks a C3 session can fail in a way that costs money or invalidates the
result — nothing else belongs here, and the whole file must run in seconds.

The four that have precedent in this project's paid failures:

* **the launcher/driver CLI seam.** C1 attempt 7 cleared the pod test gate --
  the first attempt ever to -- and then died at `$0.4231` because the
  launcher emitted a flag the driver's parser did not define. argparse exited
  2 before a line of the driver ran.
* **the imports the driver needs AFTER training.** A missing import in stage
  H surfaces 70 minutes and nine probes into a session.
* **the frozen scientific identities**, because a pod that replays the wrong
  parent produces a complete, plausible, wrong result.
* **plan/executor agreement**, which is the defect this whole round exists to
  repair: at `b937aebb` the plan said three arms and the executor built two.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from experiments.phase_c3 import session as CS  # noqa: E402

DRIVER = REPO / "scripts/pod/autoinit_c3_driver.py"
LAUNCH = REPO / "scripts/pod/autoinit_c3_launch.py"
PREREG_PATH = REPO / "logs/stages/stage-1/phase_c3/plans/c3_preregistration.json"


@pytest.fixture(scope="module")
def registered():
    from aadistill.initialization.operators.attention.gqa import (
        activation_importance, causal_kl)

    CS.register_experimental_operators()
    yield
    for mod in (activation_importance, causal_kl):
        try:
            mod.unregister()
        except Exception:                            # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# the seam that cost $0.4231
# ---------------------------------------------------------------------------

def test_the_launcher_emits_a_command_the_driver_can_parse():
    """Parsed by the DRIVER'S OWN parser, which is the authority.

    A launcher flag the driver does not define is not a type error and not a
    lint failure: argparse exits 2, on the pod, after setup has been paid for.
    """
    #: From the function's RETURN expressions, not its text. Its docstring
    #: discusses `--stage all` at length precisely because that flag was
    #: REMOVED -- C1 attempt 7 died on it -- so scanning the prose reports the
    #: flag whose absence is the fix.
    import ast as _ast

    tree = _ast.parse(LAUNCH.read_text())
    fn = next(n for n in _ast.walk(tree)
              if isinstance(n, _ast.FunctionDef) and n.name == "driver_command")
    emitted = " ".join(
        _ast.unparse(node) for node in _ast.walk(fn)
        if isinstance(node, _ast.Return))
    flags = set(re.findall(r"--[a-z][a-z0-9-]+", emitted))
    assert flags, "driver_command emits no flags; the seam is untested"

    import importlib.util

    spec = importlib.util.spec_from_file_location("c3drv_parser", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO / "scripts/autoinit"))
    spec.loader.exec_module(mod)
    parser = mod.build_parser() if hasattr(mod, "build_parser") else None
    if parser is None:
        pytest.skip("the C3 driver exposes no build_parser to interrogate")
    known = {a for action in parser._actions for a in action.option_strings}
    unknown = sorted(flags - known)
    assert not unknown, (
        f"the launcher emits {unknown}, which the C3 driver's parser does not "
        f"define. argparse would exit 2 on the pod before the driver ran.")


def test_the_driver_imports_everything_it_needs_after_training():
    """A stage-H import error surfaces nine probes and ~10 hours in."""
    import importlib.util

    sys.path.insert(0, str(REPO / "scripts/autoinit"))
    spec = importlib.util.spec_from_file_location("c3drv_full", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                     # raises if anything is missing
    for needed in ("C3Driver", "C3DriverError"):
        assert hasattr(mod, needed), f"the driver exposes no {needed}"


# ---------------------------------------------------------------------------
# the science this pod must not have drifted from
# ---------------------------------------------------------------------------

def test_the_preregistration_on_this_pod_binds_itself():
    """A plan whose stamp does not verify is not a frozen plan."""
    doc = json.loads(PREREG_PATH.read_text())
    from aadistill.infrastructure.manifest import sha256_json

    stated = doc["preregistration_sha256"]
    got = sha256_json({k: v for k, v in doc.items()
                       if k != "preregistration_sha256"})
    assert got == stated, (
        f"the preregistration checked out on this pod does not bind itself: "
        f"stamp {stated[:16]}, body {got[:16]}")


def test_the_executor_builds_the_plan_s_three_arms(registered):
    """The b937aebb defect, asked on the machine that would execute it."""
    specs = CS.build_arm_specs(workdir_device="cpu")
    planned = [k for k in json.loads(PREREG_PATH.read_text())["arms"]
               if not k.startswith("_")]
    assert sorted(specs) == sorted(planned) == sorted(CS.arm_ids())
    assert len(specs) == 3
    assert len(specs) * len(CS.recovery_seeds()) == 9


def test_the_two_causal_arms_are_distinguishable_before_anything_is_built(registered):
    """If the hashed protocol stopped separating them, the session would run
    one treatment twice and report two. Cheaper to learn here than after the
    45-minute and 38-minute scorers have both run."""
    specs = CS.build_arm_specs(workdir_device="cpu")
    tails = {a: s.steps[-1].as_dict() for a, s in specs.items()}
    assert len(tails) == len({json.dumps(t, sort_keys=True)
                              for t in tails.values()}), (
        "two arms have identical final steps; they would materialize to the "
        "same digest")


def test_the_seeds_are_c3_s_own_and_not_c1_s():
    """Disjoint by construction, which is what makes the mistake detectable."""
    from experiments.phase_c1.isolation import derive_recovery_seeds

    assert CS.recovery_seeds() == (217230555, 1151307191, 2045359208)
    assert not set(CS.recovery_seeds()) & set(derive_recovery_seeds())


def test_the_digest_gates_are_the_frozen_ones():
    """A pod replaying the wrong parent produces a plausible wrong result."""
    assert CS.expected_parent_digest() == (
        "eea90c91346a0745b8b1b847503b48fe73c33bb9d75d92c196dc43598e91e722")
    assert CS.expected_incumbent_digest() == (
        "53e30566c5f795f1870d76c1fa6a970ddc507fa5459047f3010ffab8aa890342")


def test_the_primary_contrast_is_the_operator_isolation_one():
    """The verdict's owner, checked where it will be computed."""
    doc = json.loads(PREREG_PATH.read_text())
    for field in (doc["claim_boundary"]["primary_contrast"],
                  doc["estimand"]["primary"]["contrast"],
                  doc["decision_rule"]["_applies_to"]):
        assert "causal-B1" in field and "causal-B3" not in field


# ---------------------------------------------------------------------------
# the accelerator
# ---------------------------------------------------------------------------

def test_a_cuda_device_is_actually_present_and_usable():
    """A real GEMM, not `is_available()`. A driver that reports a device and
    cannot run a kernel fails at materialization, after the teacher fetch."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device on this machine (dev-box run)")
    a = torch.randn(64, 64, device="cuda:0", dtype=torch.bfloat16)
    out = (a @ a).float().sum().item()
    assert out == out, "the GEMM produced NaN"
    torch.cuda.synchronize()
