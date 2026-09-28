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


def code_without_comments(text: str) -> str:
    """Source with COMMENTS removed and string literals kept.

    The right lens for "does this literal value execute anywhere": a frozen
    digest IS a string literal, so `executable_source` -- which blanks every
    string -- would strip the very constant the check looks for. Comments are
    still dropped, so a note explaining why a superseded digest is gone does
    not read as that digest surviving.
    """
    import io
    import tokenize

    out = []
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.COMMENT:
            continue
        out.append(tok.string)
    return " ".join(out)


def executable_source(text: str) -> str:
    """Source with comments and string literals removed.

    A `#`-strip is not enough: the launcher's own docstring NAMES the two
    deleted session constants while explaining that they are gone, so a
    naive scan reports the very absence that is the fix.
    """
    import io
    import tokenize

    out = []
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.COMMENT:
            continue
        out.append('""' if tok.type == tokenize.STRING else tok.string)
    return " ".join(out)


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

def test_a_reported_cuda_device_can_actually_run_a_kernel():
    """A real GEMM, not `is_available()`.

    **It does not skip**, and that is deliberate. Written as
    `skip if not cuda.is_available()` it skipped on the CPU dev box, and the
    readiness sweep -- which runs this directory in a pod-like simulator --
    refused the record for an undeclared environment skip. Declaring the skip
    as EXPECTED would have been worse: the same nodeid PASSES on the pod, so
    a "must skip" expectation inverts exactly where it matters, and a gate
    can check THAT a test skipped but never WHY.

    So it asserts the implication instead: IF torch reports a device, a
    kernel must run on it. That is true in both environments, passes in
    both, and still catches the failure worth catching -- a driver that
    advertises a device and cannot execute, which surfaces at
    materialization, after the teacher fetch has been paid for.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        #: No device claimed, nothing to disprove. The pod's setup gate is
        #: what refuses a GPU session with no GPU; this test's subject is
        #: the consistency of the claim, not its presence.
        return
    a = torch.randn(64, 64, device="cuda:0", dtype=torch.bfloat16)
    out = (a @ a).float().sum().item()
    assert out == out, "a device was reported but its GEMM produced NaN"
    torch.cuda.synchronize()


def test_the_launcher_plan_hash_is_what_the_authorization_bound(registered):
    """`require_plan` compares these two; a mismatch is exit 98 on a pod.

    The launcher's `_plan_hash()` built a two-arm `C1IsolationPlan` from
    session constants the rewritten module does not have, on C1's three
    seeds. It raised at `$0`, which is the gate order working -- but had it
    merely *run*, it would have produced a hash the authorization does not
    carry, and the failure would have been `AUTHORIZATION_MISMATCH` after
    setup had been paid for. One owner now: the session contract hash.
    """
    import importlib.util

    auth_path = REPO / "logs/budget/approvals/autoinit_c3_authorization.json"
    if not auth_path.is_file():
        pytest.skip("no C3 authorization has been issued in this checkout")
    sys.path.insert(0, str(REPO / "scripts/pod"))
    sys.path.insert(0, str(REPO / "scripts/autoinit"))
    spec = importlib.util.spec_from_file_location("c3launch_ph", LAUNCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    bound = json.loads(auth_path.read_text())["bound"]["session_contract_hash"]
    assert mod._plan_hash() == bound, (
        f"the launcher computes plan hash {mod._plan_hash()[:16]} and the "
        f"authorization binds {bound[:16]}; the preflight would exit 98")


def test_the_launcher_expects_nine_probe_streams_on_c3_s_seeds():
    """Six streams would let a nine-probe session come home missing three."""
    import importlib.util

    sys.path.insert(0, str(REPO / "scripts/pod"))
    spec = importlib.util.spec_from_file_location("c3launch_ps", LAUNCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    streams = mod.probe_streams(None)
    assert len(streams) == 9, f"{len(streams)} probe streams, expected 9"
    for seed in CS.recovery_seeds():
        assert any(str(seed) in s for s in streams), f"no stream for seed {seed}"
    for stale in (1635674081, 1656475568, 696460635):
        assert not any(str(stale) in s for s in streams), (
            f"a probe stream names C1's seed {stale}")


def test_every_session_attribute_the_launcher_uses_exists():
    """Enumerated, not spot-checked.

    The launcher referenced four names the rewritten session does not have --
    EXPECTED_PARENT_DIGEST and EXPECTED_INCUMBENT_DIGEST became functions,
    INCUMBENT_ATTENTION and TREATMENT_ATTENTION were deleted with the two-arm
    design. Each raised only when the line ran, and two of them sat inside
    `spec()`, which the readiness sweep reaches and a spot check does not.
    Asking the module for every attribute the file names is the cheap
    version of that discovery.
    """
    code = executable_source(LAUNCH.read_text())
    used = sorted(set(re.findall(r"\bCS\.([A-Za-z_][A-Za-z0-9_]*)", code)))
    missing = [u for u in used if not hasattr(CS, u)]
    assert not missing, (
        f"the C3 launcher references session attributes that do not exist: "
        f"{missing}. Each would raise when its line ran -- on a pod, if the "
        f"line is inside a stage rather than a gate.")


def test_every_session_attribute_the_driver_uses_exists():
    """The same question of the driver, where a stale name costs more."""
    code = executable_source(DRIVER.read_text())
    used = sorted(set(re.findall(r"\bCS\.([A-Za-z_][A-Za-z0-9_]*)", code)))
    missing = [u for u in used if not hasattr(CS, u)]
    assert not missing, (
        f"the C3 driver references session attributes that do not exist: "
        f"{missing}. Stage H runs ten hours into a paid session.")


def test_the_launcher_uses_c3_s_own_readiness_contract_and_records():
    """A C3 session verified against C1's expectations proves nothing.

    Found one gate at a time across three $0 launcher aborts: the grant
    provenance reference, the incumbent digest, the readiness contract, the
    pricing provenance and the bundle pointer were each C1's. Each abort cost
    nothing and a full re-issue cycle, which is the expensive part.

    `RecordContract` carries the schema, the harness field, the record path
    and the permitted-post-sweep paths. Handing C3's session C1's contract
    would verify C3's readiness against C1's schema, look for the record at
    C1's path, and permit the wrong file to change after the sweep.
    """
    code = executable_source(LAUNCH.read_text())
    assert "c3_record_contract" in code, (
        "the launcher does not use C3's readiness contract")
    assert "c1_record_contract" not in code, (
        "the launcher still uses C1's readiness contract")
    #: And every governance path it names belongs to phase_c3.
    for const, want in (("PRICING", "phase_c3"), ("BUNDLE_POINTER", "phase_c3"),
                        ("PREREG", "phase_c3"), ("AUTH_POINTER", "c3")):
        m = re.search(rf"^{const} = .*$", LAUNCH.read_text(), re.M)
        assert m, f"{const} is no longer a module constant"
        assert want in m.group(0), f"{const} points outside C3: {m.group(0)}"


def test_the_incumbent_the_launcher_gates_on_is_c3_s():
    """c313d1b4 is weight_proxy_v0 -- the arm C1 already beat.

    Carrying C1's incumbent constant here would have gated C3 against the
    loser of the previous round while every message read correctly.
    """
    code = code_without_comments(LAUNCH.read_text())
    assert "53e30566c5f795f1870d76c1fa6a970ddc507fa5459047f3010ffab8aa890342" in code
    assert "c313d1b4081b9a3b410dddf7a29ebcaad8dd0759179d51e1d761238c1743a2a6" not in code
    assert CS.expected_incumbent_digest().startswith("53e30566")
