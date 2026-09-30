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

import ast
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


def source_minus_comment_lines(text: str) -> str:
    """Source with whole-line `#` comments dropped, SPACING PRESERVED.

    The tokenizer-based lenses join tokens with spaces, so
    `CS.register_experimental_operators()` becomes
    `CS . register_experimental_operators ( )` and a literal search for the
    call fails. When the check is about a call SITE rather than a bare name,
    this is the lens that works.
    """
    return "\n".join(l for l in text.splitlines()
                      if not l.lstrip().startswith("#"))


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
    #: Asserted, not skipped. A driver with no parser to interrogate does not
    #: make this check inapplicable -- it makes the seam unverifiable, which
    #: is the condition that cost C1 attempt 7 $0.4231. And an undeclared
    #: skip is what the readiness sweep refuses.
    assert hasattr(mod, "build_parser"), (
        "the C3 driver exposes no build_parser; the launcher/driver CLI seam "
        "cannot be checked, and that seam has already killed a paid session")
    parser = mod.build_parser()
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
    #: Imported, not import-or-skipped. This directory is the pod's blocking
    #: gate and the pod's science environment has torch by construction; a
    #: missing one is a broken image, which is a failure, not an
    #: inapplicable check. It is also the last undeclared skip the readiness
    #: sweep would have refused.
    import torch

    if not torch.cuda.is_available():
        #: No device claimed, nothing to disprove. The pod's setup gate is
        #: what refuses a GPU session with no GPU; this test's subject is
        #: the consistency of the claim, not its presence.
        return
    a = torch.randn(64, 64, device="cuda:0", dtype=torch.bfloat16)
    #: `.item()` already forces this stream to complete, so the GEMM has run
    #: by the time the value exists. An explicit `torch.cuda.synchronize()`
    #: adds nothing and carries a real hazard: it re-raises faults from
    #: EARLIER async work, so this check could fail for something another
    #: test did a thousand cases ago and abort a paid session. That is what
    #: `test_no_test_drains_a_real_gpu` exists to prevent, and it caught it.
    out = (a @ a).float().sum().item()
    assert out == out, "a device was reported but its GEMM produced NaN"


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


@pytest.mark.parametrize("target", ["launcher", "driver"])
def test_no_module_level_name_is_undefined(target):
    """A NameError inside a gate is only found by reaching that gate.

    `C3_HARNESS_SOURCE_FILES_V1` survived a rename in one line of
    `artifact_spec_gate` -- the ninth gate -- so four launcher invocations
    and four full re-issue cycles ran before anything reached it. Every one
    cost $0 and an hour.

    A static pass over module-level loads answers the same question without
    executing anything. Locals and comprehension targets are collected too,
    so this is deliberately permissive: it reports names nothing in the file
    could bind, which is the class that raises.
    """
    import builtins

    path = LAUNCH if target == "launcher" else DRIVER
    tree = ast.parse(path.read_text())
    #: Module globals Python provides, which nothing in the file assigns.
    bound = set(dir(builtins)) | {
        "__file__", "__name__", "__doc__", "__package__", "__spec__",
        "__loader__", "__builtins__", "__debug__"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.Global):
            bound.update(node.names)
    used = {n.id for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    undefined = sorted(used - bound)
    assert not undefined, (
        f"the C3 {target} loads names nothing binds: {undefined}. Each raises "
        f"only when its line runs -- inside a gate, or on a pod.")


def test_every_required_audit_artifact_is_one_the_driver_writes():
    """The spec and the driver must name the same files.

    `c1_evidence.json` survived the rename in the artifact spec while the
    driver wrote `c3_evidence.json`. The manifest gate then reported
    `MISSING session_evidence [final_required]` at teardown -- the point at
    which the evidence is still on the pod and still recoverable, but only if
    someone is watching. A required artifact nobody writes is a session that
    completes and comes home incomplete.

    Only literal patterns are checked: globbed ones name probe ids that do
    not exist until the run does.
    """
    spec = json.loads(
        (REPO / "configs/autoinit/c3_artifacts.json").read_text())
    driver_src = DRIVER.read_text()
    missing = []
    for entry in spec["entries"]:
        pattern = entry.get("pattern", "")
        if not entry.get("required") or "*" in pattern:
            continue
        if not pattern.startswith("audit/autoinit_c3/"):
            continue
        name = pattern.rsplit("/", 1)[-1]
        if name in driver_src:
            continue
        #: The per-arm records are built by f-string from the plan's arm ids,
        #: so the literal never appears. Reconstruct the same way the driver
        #: does rather than demanding a literal it has no reason to contain.
        if any(name == f"c3_{arm}_record.json" for arm in CS.arm_ids()):
            assert 'AUDIT / f"c3_{arm_id}_record.json"' in driver_src, (
                "the per-arm record filename is no longer built from arm_id")
            continue
        missing.append(pattern)
    assert not missing, (
        f"the success spec requires {missing}, which the C3 driver never "
        f"writes; the manifest gate would report them MISSING at teardown")


def test_the_driver_loads_a_c3_authorization_from_this_run():
    """A repository-level pointer can name another experiment's artifact.

    It named C1's, and C3Authorization.load refused it on schema -- the type
    working exactly as designed, eight minutes and $0.15 into a session. The
    run-scoped path is also the one the launcher's gates verified, so driver
    and launcher read the same artifact rather than two that agree by habit.
    """
    src = DRIVER.read_text()
    assert "autoinit_c1_authorization.json" not in src, (
        "the C3 driver still points at C1's authorization")
    assert "def authorization_path(self)" in src
    assert "governance/authorization.json" in src


def test_the_frozen_plan_can_actually_be_constructed(registered):
    """Constructing it is free, and it only ever ran on a paid pod.

    `C1IsolationPlan` holds EXACTLY two arms and `C1Arm.role` must be
    `incumbent` or `treatment`. Handing it C3's three arms with ids like
    `A_incumbent` raised `role must be 'incumbent' or 'treatment'` in
    `C3Driver.__init__` -- on a live L40S, after setup had completed, at
    `$0.17`. Nothing about that failure needed a GPU.

    So this builds the plan the way the driver does and asserts what it must
    contain: the PRIMARY contrast's two arms, in their C1 roles.
    """
    import importlib.util

    sys.path.insert(0, str(REPO / "scripts/autoinit"))
    spec = importlib.util.spec_from_file_location("c3drv_plan", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class FakeArgs:
        run_id = "preflight"
        rate = 1.0
        spent_usd = 0.0
        soft_stop_usd = 1.0
        authorized_usd = 1.0
        image_digest = ""

    #: `frozen_plan` and `primary_operands` are pure reads of the frozen
    #: plan plus the battery file -- they need no driver state, so they are
    #: exercised unbound rather than by constructing a whole C3Driver.
    drv = mod.C3Driver.__new__(mod.C3Driver)
    drv.seeds = list(CS.recovery_seeds())
    control, candidate = mod.C3Driver.primary_operands(drv)
    assert control == "A_incumbent", f"control arm is {control}"
    assert candidate == "B_causal_b1", f"candidate arm is {candidate}"

    plan = mod.C3Driver.frozen_plan(drv)
    assert len(plan.arms) == 2, "C1IsolationPlan holds exactly two arms"
    roles = [a.role for a in plan.arms]
    assert roles == ["incumbent", "treatment"], roles
    assert plan.arms[0].attention_impl_id == "attention.activation_importance_v1"
    assert plan.arms[1].attention_impl_id == "attention.causal_kl_v1"
    assert tuple(plan.seeds) == CS.recovery_seeds()


def test_the_verdict_operands_come_from_the_plan_not_from_list_order(registered):
    """A renamed or reordered arm must not silently move the verdict."""
    import importlib.util

    sys.path.insert(0, str(REPO / "scripts/autoinit"))
    spec = importlib.util.spec_from_file_location("c3drv_ops", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    #: BEHAVIOURAL, not a grep. This used to search the driver's source for
    #: `prereg["estimand"]["primary"]["contrast"]`, which asserted where the
    #: string appears rather than that the operands follow the plan -- and it
    #: broke the moment the resolution moved to its owner in `session.py`
    #: without the property changing at all. So the plan's own statement is
    #: rewritten and the operands are required to follow it.
    import copy as _copy

    real = CS.preregistration()
    flipped = _copy.deepcopy(real)
    flipped["estimand"]["primary"]["contrast"] = "causal-B3 - causal-B1"
    saved = CS._PREREG
    try:
        CS._PREREG = flipped
        assert CS.primary_operands() == ("B_causal_b1", "C_causal_b3"), (
            "the primary operands did not follow a rewritten contrast; they "
            "are coming from list order, not from the frozen plan")
    finally:
        CS._PREREG = saved
    assert CS.primary_operands() == ("A_incumbent", "B_causal_b1"), (
        "the real contrast no longer resolves to the incumbent/B1 pair")

    #: And the third arm is deliberately absent from the decision rule. Asked
    #: of the DRIVER, so the delegation to the owner is exercised too.
    drv = mod.C3Driver.__new__(mod.C3Driver)
    drv.seeds = list(CS.recovery_seeds())
    control, candidate = mod.C3Driver.primary_operands(drv)
    assert "C_causal_b3" not in (control, candidate), (
        "the B3 arm is in the decision rule; the preregistration makes it a "
        "secondary contrast that may not redefine the verdict")


def test_the_driver_validates_the_hash_the_authorization_actually_binds(registered):
    """Two quantities are called "plan hash"; only one is bound.

    The authorization records the SESSION CONTRACT hash -- the value the
    launcher's `_plan_hash()` computes and the preflight's `require_plan`
    checks. The driver was validating `self.plan.plan_hash`, the ISOLATION
    plan's, which is a different number for C3 because the isolation plan
    carries only the primary contrast's two arms. It raised "an
    authorization does not transfer to a plan that changed" on a live pod,
    at $0.22, after setup had completed.

    They coincided under C1, where one object produced both. Driving them
    apart is what made the mistake visible, and this pins the right one.
    """
    import importlib.util

    auth_path = REPO / "logs/budget/approvals/autoinit_c3_authorization.json"
    src = DRIVER.read_text()
    assert "self.auth.require_plan(CS.C3SessionContract().contract_hash)" in src, (
        "the driver validates something other than the session contract hash")
    assert "require_plan(self.plan.plan_hash)" not in src, (
        "the driver still validates the isolation plan's hash, which the "
        "authorization does not bind")

    if not auth_path.is_file():
        pytest.skip("no C3 authorization in this checkout")
    bound = json.loads(auth_path.read_text())["bound"]["session_contract_hash"]
    assert CS.C3SessionContract().contract_hash == bound, (
        "the session contract hash and the authorization's binding differ")

    #: And the isolation plan's hash is DIFFERENT, which is the whole point:
    #: if they were equal this test would pass for the wrong reason.
    sys.path.insert(0, str(REPO / "scripts/autoinit"))
    spec = importlib.util.spec_from_file_location("c3drv_hash", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    drv = mod.C3Driver.__new__(mod.C3Driver)
    drv.seeds = list(CS.recovery_seeds())
    assert mod.C3Driver.frozen_plan(drv).plan_hash != bound, (
        "the isolation plan and the session contract hash to the same value; "
        "while they coincide, checking either passes and this test proves "
        "nothing")


def test_every_identity_the_driver_asserts_is_one_the_authorization_binds(registered):
    """Both `require_*` arguments, checked against the issued artifact.

    Two ran on live pods before anything caught them, one line apart:
    `require_plan` was given the isolation plan's hash ($0.22) and
    `require_science_plan` was given C1's C0 digest ($0.18). Each is a
    constant that was right for C1, and each aborts AFTER setup.

    So this asks the artifact what it binds and the driver what it asserts,
    rather than trusting either in isolation.
    """
    auth_path = REPO / "logs/budget/approvals/autoinit_c3_authorization.json"
    src = DRIVER.read_text()

    assert "self.auth.require_plan(CS.C3SessionContract().contract_hash)" in src
    assert 'require_science_plan(\n            CS.preregistration()["preregistration_sha256"])' in src \
        or 'require_science_plan(CS.preregistration()["preregistration_sha256"])' in src, (
        "the driver does not assert C3's own preregistration as its science plan")
    assert "C0_PREREGISTRATION_SHA256" not in code_without_comments(src), (
        "C1's C0 science-plan digest is still reachable in the C3 driver")

    if not auth_path.is_file():
        pytest.skip("no C3 authorization in this checkout")
    auth = json.loads(auth_path.read_text())
    assert auth["phase_a_session_plan_hash"] == CS.C3SessionContract().contract_hash
    assert auth["phase_a_science_plan_hash"] == \
        CS.preregistration()["preregistration_sha256"]


def test_stage_C_registers_every_implementation_the_arms_name():
    """The DRIVER's stage C, not the session helper it should call.

    I tested `CS.register_experimental_operators()` and it passed, while the
    driver's stage C called `attention_activation.register()` alone -- C1 has
    one experimental operator, C3 has two. Stage D then died with
    "attention.causal_kl_v1 is not registered" on a live pod, after the
    teacher fetch, at $0.13. Testing the helper proved nothing about the
    caller: the seam is exactly where the blind spot was.
    """
    src = source_minus_comment_lines(DRIVER.read_text())
    assert "CS.register_experimental_operators()" in src, (
        "stage C does not register the set the arms name")
    assert "attention_activation.register(replace=True)" not in src, (
        "stage C still registers a single hard-coded operator")
    #: And it must PROVE every named implementation resolves afterwards,
    #: rather than assuming the call covered them.
    assert "for impl_id in CS.required_implementations():" in src
    assert "get_implementation(impl_id)" in src


def test_no_stage_indexes_the_arms_by_c1_s_literal_keys():
    """`self.arms["incumbent"]` is a KeyError on C3's arm ids.

    It raised in stage D on a live pod at $0.14, one stage after the
    registration fix let stage C pass. C3's arms are A_incumbent,
    B_causal_b1 and C_causal_b3 -- the plan's names, not roles.
    """
    src = source_minus_comment_lines(DRIVER.read_text())
    for bad in ('arms["incumbent"]', 'arms["treatment"]',
                "arms['incumbent']", "arms['treatment']"):
        assert bad not in src, (
            f"a stage indexes the arms with {bad}, which C3's arm ids do not "
            f"contain: {list(CS.arm_ids())}")
    #: and the incumbent is reached by the plan's own first arm id.
    assert "CS.arm_ids()[0]" in src


def test_the_replay_applies_the_execution_config_the_plan_pins():
    """An unhashed value that changes the answer is invisible until it isn't.

    `materialize_fixed_path` was called without `execution=`, so DEPTH ran at
    `DEFAULT_MICRO_BATCH_SIZE` while the frozen plan pins `micro_batch_size:
    1`. The parent came back `d6d8d7ee` instead of `eea90c91` -- after 28
    minutes of real GPU search, at $0.65, on the correct GPU under the
    correct runtime.

    It is not a performance knob. L40S GEMMs reduce shape-dependently, so the
    micro-batch changes which layer a near-tie picks; that run's round 7 chose
    layer 21 over 17 by a margin of 1.4e-03. Runtime knobs are deliberately
    not hashed into any state id, which is exactly why the plan must pin this
    one and the driver must apply it -- no digest can catch it until the
    digest itself comes out wrong.
    """
    src = source_minus_comment_lines(DRIVER.read_text())
    assert "execution=self.prefix_execution_config()" in src, (
        "the prefix replay does not pass the plan's pinned execution config; "
        "DEPTH would run at whatever the repository default happens to be")
    assert CS.prefix_execution() == {"micro_batch_size": 1}

    import importlib.util

    sys.path.insert(0, str(REPO / "scripts/autoinit"))
    spec = importlib.util.spec_from_file_location("c3drv_exec", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    drv = mod.C3Driver.__new__(mod.C3Driver)
    cfg = mod.C3Driver.prefix_execution_config(drv)
    assert cfg.micro_batch_size == 1, cfg

    #: And a plan that pinned nothing must be refused, not defaulted.
    import unittest.mock as _mock

    with _mock.patch.object(mod.CS, "prefix_execution", lambda: {}):
        with pytest.raises(mod.C3DriverError, match="pins no micro_batch_size"):
            mod.C3Driver.prefix_execution_config(drv)


def test_every_shelled_out_scorer_accepts_the_argv_the_driver_builds():
    """The seam one level down: the scripts the driver SHELLS OUT to.

    attempt66 trained all nine probes, then stage H exited 2:

        score_c1_confirmation.py: error: argument --arm: invalid choice:
        'A_incumbent' (choose from 'incumbent', 'treatment')

    The existing CLI-seam check covers launcher -> driver. This covers
    driver -> scorer, which is where a C1 constant can still hide: the
    scorer is a SEPARATE process, so nothing in the driver's imports, names
    or digests can reach an argparse constraint inside it.

    It runs the REAL scorer as a subprocess with the REAL argv. `--out` is
    under tmp_path and the paths do not exist, so the run fails -- that is
    fine and deliberate. What is asserted is that it does not fail with
    **exit 2**, which is argparse's own code for a malformed command line.
    A missing input fails later and differently.

    The scorer is NOT importable-and-patchable here on purpose: it is named
    by C1's frozen executable closure and preregistration, so this check
    must observe it, never adapt it.
    """
    import tempfile

    seeds = CS.recovery_seeds()
    scorer = REPO / "scripts/autoinit/score_c1_confirmation.py"
    #: Bound, not incidental: the superset assertion at the end reads this.
    exercised: set[str] = set()
    with tempfile.TemporaryDirectory() as tmp:
        for arm in CS.arm_ids():
            name = f"autoinit.v1.phase_c3.{arm}.{seeds[0]}"
            argv = [sys.executable, str(scorer),
                    "--generations", f"{tmp}/gen", "--label", name,
                    "--seed", str(seeds[0]), "--out", f"{tmp}/out.json",
                    "--per-sample", f"{tmp}/per.jsonl",
                    "--init-digest", "0" * 64,
                    "--trained-run", f"{tmp}/rc.json",
                    "--generation-fingerprint", "f" * 16]
            exercised |= {a for a in argv if a.startswith("--")}
            r = subprocess.run(argv, capture_output=True, text=True, timeout=120)
            assert r.returncode != 2, (
                f"the C1 scorer rejects the argv the C3 driver builds for "
                f"{arm}:\n{r.stderr[-600:]}")

    #: And the flags themselves must be ones the scorer defines. Read the
    #: driver's argv BY AST -- `C1_SCORER` appears in more than one gate, and
    #: an index-based slice silently swallowed the next call's flags.
    argv_lists = []
    for node in ast.walk(ast.parse(DRIVER.read_text())):
        if not isinstance(node, ast.List) or not node.elts:
            continue
        head = node.elts[0]
        if (isinstance(head, ast.Call) and isinstance(head.func, ast.Name)
                and head.func.id == "str" and head.args
                and isinstance(head.args[0], ast.Name)
                and head.args[0].id == "C1_SCORER"):
            argv_lists.append(node)
    assert argv_lists, "no argv list starting with the C1 scorer was found"

    defined = set(re.findall(r'add_argument\(\s*"(--[a-z-]+)"',
                             scorer.read_text()))
    assert defined, "no options were found in the scorer's source"
    passed = {e.value for node in argv_lists for e in node.elts
              if isinstance(e, ast.Constant) and isinstance(e.value, str)
              and e.value.startswith("--")}
    assert passed, "no scorer flags were found in the driver's argv"
    assert passed <= defined, (
        f"the driver passes flags the C1 scorer does not define: "
        f"{sorted(passed - defined)}")

    #: And the subprocess above must EXERCISE every flag the driver sends.
    #: Its argv is written out here rather than synthesized from the driver's
    #: (most of the driver's elements are runtime values, not constants), and
    #: a transcription that drifts is a check that stops covering what it
    #: claims to: reintroducing `--arm` in the driver left this test green,
    #: because this test was still sending its own older argv. So the
    #: transcription is now BOUND to the driver rather than trusted.
    assert passed <= exercised, (
        f"the driver sends {sorted(passed - exercised)} to the C1 scorer and "
        "this check never does, so it cannot prove the scorer accepts them. "
        "Add them to the argv above — that is the whole subject of this test.")


def test_the_driver_does_not_pass_c1s_arm_vocabulary():
    """C3 must not send an arm id through a field whose values are C1's.

    `--arm` is `choices=("incumbent", "treatment")`: C1's two ROLES. C3 has
    three ARMS. The scorer cannot be widened -- it is inside C1's frozen
    executable closure and preregistration, and a COMPLETED GO experiment
    binds that digest -- so the fix belongs on this side of the seam. C3
    loses nothing: `--label` is the probe id, which carries the arm, and the
    driver keys (arm, seed) off the TRAINING record, never off the scorer's
    output.

    Checked over EVERY call site by AST. The driver invokes the scorer more
    than once, and the first occurrence is a preflight that passes
    `--label preflight` -- which is exactly how the defect survived: the
    driver already had a scorer preflight gate, and it passed on attempt66
    because it omitted the one argument that mattered.
    """
    sites = []
    for node in ast.walk(ast.parse(DRIVER.read_text())):
        if not isinstance(node, ast.List) or not node.elts:
            continue
        head = node.elts[0]
        if (isinstance(head, ast.Call) and isinstance(head.func, ast.Name)
                and head.func.id == "str" and head.args
                and isinstance(head.args[0], ast.Name)
                and head.args[0].id == "C1_SCORER"):
            sites.append([e.value for e in node.elts
                          if isinstance(e, ast.Constant)
                          and isinstance(e.value, str)])
    assert len(sites) >= 2, (
        f"expected the preflight AND the real scoring call, found {len(sites)}")
    for flags in sites:
        assert "--arm" not in flags, (
            "the C3 driver still passes --arm to the C1 scorer; that flag "
            "only accepts C1's two role names and exits 2 on a C3 arm id")

    #: the arm is still recoverable from what IS passed, at the real call.
    src = source_minus_comment_lines(DRIVER.read_text())
    assert '"--label", name' in src
    assert 'name = record["probe_id"]' in src


def test_the_scorer_preflight_sends_what_the_real_call_sends():
    """A gate that supplies its own inputs cannot detect a wrong input.

    The driver's scorer preflight passed on attempt66 and the real call
    exited 2 forty minutes later. The difference was the argument set: the
    preflight sent a subset. So the preflight's flags must be a SUPERSET of
    the real call's, or it is not exercising the command that matters.
    """
    sites = []
    for node in ast.walk(ast.parse(DRIVER.read_text())):
        if not isinstance(node, ast.List) or not node.elts:
            continue
        head = node.elts[0]
        if (isinstance(head, ast.Call) and isinstance(head.func, ast.Name)
                and head.func.id == "str" and head.args
                and isinstance(head.args[0], ast.Name)
                and head.args[0].id == "C1_SCORER"):
            sites.append({e.value for e in node.elts
                          if isinstance(e, ast.Constant)
                          and isinstance(e.value, str)
                          and e.value.startswith("--")})
    preflight, *rest = sites
    real = set().union(*rest) if rest else set()
    missing = sorted(real - preflight)
    assert not missing, (
        f"the driver's scorer preflight omits {missing}, which the real "
        f"scoring call passes. A flag the preflight never sends is a flag "
        f"the preflight cannot prove the scorer accepts -- which is exactly "
        f"how --arm survived to stage H.")
