"""What a paid A3 pod must prove before it spends five hours on anything.

`pytest tests/ $SESSION_TEST_IGNORES` runs on the pod as a blocking gate and
A3's derived complement leaves exactly this directory collectable, so these are
the checks an A3 session can fail in a way that costs money or invalidates the
result. Nothing else belongs here and the whole file must run in seconds.

The three seams with precedent in this project's paid failures:

* **the launcher/driver CLI seam.** C1 attempt 7 cleared the pod test gate and
  then died at `$0.4231` because the launcher emitted a flag the driver's
  parser did not define. argparse exited 2 before a line of the driver ran.
* **the two SHELLED-OUT seams**, `/opt/train/bin/python` for the trainer and
  `/opt/vllm/bin/python` for the uncapped evaluator. Each is a separate
  interpreter with its own argparse, and each is first reached after real
  money has been spent -- the evaluator after three trainings.
* **the imports and attributes a late stage needs.** Stage H runs after three
  trainings and three evaluations; a name that does not exist there is ~4.5
  hours of L40S time for a traceback.

And one that is A3's own: **A-bsz3 is pinned NOWHERE**. The whole experiment is
the question of what it builds, so a digest literal for it anywhere in the
chain would decide the result in the code that is supposed to measure it.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib.util
import inspect
import io
import json
import re
import sys
import tokenize
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from stages.phase_a3 import a3_session as A3S  # noqa: E402

DRIVER = REPO / "scripts/stages/stage-1/phase_a3/autoinit_a3_driver.py"
LAUNCH = REPO / "scripts/stages/stage-1/phase_a3/autoinit_a3_launch.py"
DESIGN = REPO / "logs/stages/stage-1/phase_c3/plans/a3_design.json"


def attributes_of(path: Path, holder: str) -> set[str]:
    """Every `<holder>.<name>` an EXPRESSION in `path` evaluates.

    From the AST, not from a text lens, and that is the repair rather than a
    style preference. The idiomatic lens in this repository joins tokens with
    spaces, which turns `A3S.PROTOCOL_EXECUTIONS` into
    `A3S . PROTOCOL_EXECUTIONS` -- so a `\\bA3S\\.(\\w+)` scan over it matches
    NOTHING and reports a clean result for a file it never looked at. C3's
    copy of this check feeds exactly that lens and both of its parametrized
    cases are vacuous. Asking the syntax tree cannot be fooled that way, and
    it also ignores prose by construction: a docstring is a `Constant`, never
    an `Attribute`.
    """
    tree = ast.parse(path.read_text())
    return {n.attr for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
            and n.value.id == holder}


def code_without_comments(text: str) -> str:
    """Comments gone, string literals KEPT -- the lens for "is this literal
    value reachable", since a frozen digest *is* a string literal."""
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.COMMENT:
            continue
        out.append(tok.string)
    return " ".join(out)


def load(path: Path, name: str):
    sys.path.insert(0, str(REPO / "scripts/pod"))
    sys.path.insert(0, str(REPO / "scripts/autoinit"))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def argparse_flags(script: Path) -> set[str]:
    """Every option string an `add_argument` call in `script` declares.

    Parsed, not executed: the evaluator and the trainer run under OTHER
    interpreters (`/opt/vllm/bin/python`, `/opt/train/bin/python`) whose
    imports this one does not have, so importing them to call `build_parser`
    would fail for a reason that says nothing about the seam.
    """
    tree = ast.parse(script.read_text())
    flags: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if arg.value.startswith("-"):
                        flags.add(arg.value)
    return flags


@pytest.fixture(scope="module")
def registered():
    """The incumbent ATTENTION implementation, registered the way stage C
    does it. `FixedPathSpec.__post_init__` resolves every step's impl id, so
    `path_spec()` raises `KeyError` without this -- and the registry is
    process-global, so it is also unregistered afterwards rather than left
    for a sibling test to find already full."""
    from aadistill.initialization.operators.attention.gqa import (
        activation_importance,
    )

    activation_importance.register(replace=True)
    yield
    try:
        activation_importance.unregister()
    except Exception:                                    # noqa: BLE001
        pass


def flags_in_argv(fn_src: str) -> set[str]:
    """The `--flags` a source fragment passes to a subprocess.

    Either quote style: the fragments here come from `ast.unparse`, which
    emits single quotes, and a double-quoted pattern silently matched nothing.
    """
    return set(re.findall(r"""['"](--[a-z][a-z0-9-]*)['"]""", fn_src))


# ---------------------------------------------------------------------------
# the seam that cost $0.4231
# ---------------------------------------------------------------------------

def test_the_launcher_emits_a_command_the_driver_can_parse():
    """Parsed by the DRIVER'S OWN parser, which is the authority."""
    tree = ast.parse(LAUNCH.read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "driver_command")
    emitted = " ".join(ast.unparse(node) for node in ast.walk(fn)
                       if isinstance(node, ast.Return))
    flags = set(re.findall(r"--[a-z][a-z0-9-]+", emitted))
    assert flags, "driver_command emits no flags; the seam is untested"

    mod = load(DRIVER, "a3drv_parser")
    assert hasattr(mod, "build_parser"), (
        "the A3 driver exposes no build_parser; the seam that killed C1 "
        "attempt 7 cannot be checked")
    parser = mod.build_parser()
    known = {a for action in parser._actions for a in action.option_strings}
    unknown = sorted(flags - known)
    assert not unknown, (
        f"the launcher emits {unknown}, which the A3 driver's parser does not "
        f"define. argparse would exit 2 on the pod before the driver ran.")

    #: And every REQUIRED driver flag is one the launcher actually sends. A
    #: required flag nobody emits is the same argparse exit 2 from the other
    #: direction, and the launcher is the only caller.
    required = {action.option_strings[0] for action in parser._actions
                if action.required and action.option_strings}
    assert not sorted(required - flags), (
        f"the driver requires {sorted(required - flags)}, which the launcher "
        f"does not emit")


def test_the_trainer_seam_sends_flags_the_trainer_defines():
    """`/opt/train/bin/python` is a different interpreter and a different
    parser. This seam is reached once per probe, the first time ~25 minutes
    into the chain, and three times in total."""
    mod_src = DRIVER.read_text()
    tree = ast.parse(mod_src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "train_one")
    sent = flags_in_argv(ast.unparse(fn))
    assert sent, "train_one passes no flags; the seam is untested"

    trainer = next((ast.literal_eval(n.value)
                    for n in ast.walk(tree)
                    if isinstance(n, ast.Assign)
                    and any(getattr(t, "id", "") == "TRAINER" for t in n.targets)
                    and isinstance(n.value, ast.Constant)), None)
    #: TRAINER is built from a path expression rather than a bare literal in
    #: the driver, so resolve it from the module instead of the AST.
    drv = load(DRIVER, "a3drv_trainer_seam")
    script = Path(str(drv.TRAINER)) if trainer is None else REPO / trainer
    assert script.is_file(), f"the trainer script {script} does not exist"
    declared = argparse_flags(script)
    assert not sorted(sent - declared), (
        f"the driver sends {sorted(sent - declared)} to {script.name}, which "
        f"declares {sorted(declared)}")


def test_the_evaluator_seam_sends_flags_the_evaluator_defines():
    """`/opt/vllm/bin/python` is reached AFTER all three trainings.

    A plausible-looking `--battery/--out/--sets` would be an argparse exit 2
    roughly three hours and `$3.40` into the chain, with every probe trained
    and nothing measured.
    """
    tree = ast.parse(DRIVER.read_text())
    drv = load(DRIVER, "a3drv_eval_seam")
    for fn_name, script_attr in (("generate_one", "UNCAPPED_EVAL"),
                                 ("attest", "ENGINE_PROBE")):
        fn = next((n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == fn_name),
                  None)
        assert fn is not None, f"the driver has no {fn_name}"
        sent = flags_in_argv(ast.unparse(fn))
        assert sent, f"{fn_name} passes no flags; the seam is untested"
        script = Path(str(getattr(drv, script_attr)))
        assert script.is_file(), f"{script_attr} -> {script} does not exist"
        declared = argparse_flags(script)
        assert not sorted(sent - declared), (
            f"{fn_name} sends {sorted(sent - declared)} to {script.name}, "
            f"which declares {sorted(declared)}")


def test_the_scorer_seam_sends_flags_the_scorer_defines():
    """The THIRD shelled-out seam, reached after a probe has generated.

    `/opt/train/bin/python scripts/stages/stage-1/phase_c1/score_c1_confirmation.py` is a
    separate interpreter with its own parser, first reached ~4.5 hours into
    the chain. `--arm` is deliberately NOT sent: its choices are C1's two
    ROLES, and passing an experiment's arm id there is what ended C3's
    attempt66 one stage from a verdict.
    """
    tree = ast.parse(DRIVER.read_text())
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "stage_g"), None)
    assert fn is not None, "the driver has no stage_g"
    sent = flags_in_argv(ast.unparse(fn))
    assert "--generations" in sent, (
        f"stage_g does not invoke the scorer; flags seen: {sorted(sent)}")
    assert "--arm" not in sent, (
        "stage_g passes --arm to the C1 scorer, whose choices are C1's two "
        "roles; that ended C3 attempt66 one stage from its verdict")

    drv = load(DRIVER, "a3drv_score_seam")
    scorer = Path(str(drv.C1_SCORER))
    assert scorer.is_file(), f"C1_SCORER -> {scorer} does not exist"
    declared = argparse_flags(scorer)
    assert not sorted(sent - declared - {"--help"}), (
        f"stage_g sends {sorted(sent - declared - {'--help'})} to "
        f"{scorer.name}, which declares {sorted(declared)}")

    #: And every flag the scorer REQUIRES is one stage_g sends.
    required = set()
    for node in ast.walk(ast.parse(scorer.read_text())):
        if (isinstance(node, ast.Call)
                and getattr(node.func, "attr", "") == "add_argument"):
            names = [a.value for a in node.args
                     if isinstance(a, ast.Constant)
                     and str(a.value).startswith("-")]
            kw = {k.arg: getattr(k.value, "value", None) for k in node.keywords}
            if kw.get("required") and names:
                required.add(names[0])
    assert required, "the scorer declares nothing required; the check is vacuous"
    assert not sorted(required - sent), (
        f"the scorer requires {sorted(required - sent)}, which stage_g does "
        "not send")


def test_the_driver_imports_everything_it_needs_after_training():
    """A stage-H import error surfaces three probes and ~4.5 hours in."""
    mod = load(DRIVER, "a3drv_full")
    for needed in ("A3Driver", "A3DriverError", "build_parser"):
        assert hasattr(mod, needed), f"the driver exposes no {needed}"


@pytest.mark.parametrize("target", ["launcher", "driver"])
def test_no_module_level_name_is_undefined(target):
    """A NameError inside a gate is only found by reaching that gate."""
    import builtins

    path = LAUNCH if target == "launcher" else DRIVER
    tree = ast.parse(path.read_text())
    bound = set(dir(builtins)) | {
        "__file__", "__name__", "__doc__", "__package__", "__spec__",
        "__loader__", "__builtins__", "__debug__"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef)):
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
        f"the A3 {target} loads names nothing binds: {undefined}")


@pytest.mark.parametrize("target", ["launcher", "driver"])
def test_every_session_attribute_it_uses_exists(target):
    """Enumerated, not spot-checked. C3's launcher referenced four names its
    rewritten session did not have, two of them inside `spec()`."""
    path = LAUNCH if target == "launcher" else DRIVER
    used = sorted(attributes_of(path, "A3S"))
    assert len(used) > 5, (
        f"the {target} evaluates only {used} on the session module; the scan "
        f"is not looking at the file")
    missing = [u for u in used if not hasattr(A3S, u)]
    assert not missing, (
        f"the A3 {target} references session attributes that do not exist: "
        f"{missing}. Each raises only when its line runs.")


# ---------------------------------------------------------------------------
# the science this pod must not have drifted from
# ---------------------------------------------------------------------------

def test_the_design_on_this_pod_binds_itself():
    """A plan whose stamp does not verify is not a frozen plan."""
    design = A3S.design(REPO)
    assert design["design_sha256"] == A3S.design_hash(REPO)
    if DESIGN.is_file():
        doc = json.loads(DESIGN.read_text())
        assert doc["design_sha256"] == design["design_sha256"], (
            "the committed design document and the live design disagree on "
            "this pod; the authorization bound one of them")


def test_the_digest_gates_are_the_frozen_ones():
    """A pod replaying the wrong parent produces a plausible wrong result."""
    assert A3S.expected_parent_digest() == (
        "eea90c91346a0745b8b1b847503b48fe73c33bb9d75d92c196dc43598e91e722")
    assert A3S.expected_incumbent_digest() == (
        "53e30566c5f795f1870d76c1fa6a970ddc507fa5459047f3010ffab8aa890342")
    #: THE ASYMMETRY IS THE EXPERIMENT. A-bsz1 must rebuild the incumbent;
    #: A-bsz3 is gated nowhere, because what it builds is the finding.
    assert A3S.protocol_gate(A3S.REFERENCE_PROTOCOL) == (
        A3S.expected_incumbent_digest())
    assert A3S.protocol_gate(A3S.TREATMENT_PROTOCOL) is None


def test_no_executable_line_pins_a_bsz3_to_any_digest():
    """The one literal that must not exist anywhere in the chain.

    If A-bsz3 were compared against a pinned value, the chain would answer
    its own question in the code that is supposed to measure it -- and the
    answer would be whatever the author guessed. Both files DISCUSS the
    incumbent digest in prose, so the lens keeps literals and drops comments,
    and then asks where each one is used.
    """
    incumbent = A3S.expected_incumbent_digest()
    for path in (DRIVER, LAUNCH):
        src = code_without_comments(path.read_text())
        for hit in re.finditer(re.escape(incumbent), src):
            window = src[max(0, hit.start() - 300):hit.start()]
            assert "bsz3" not in window.lower(), (
                f"{path.name} pins the incumbent digest near a bsz3 "
                f"reference; A-bsz3 is measured, not gated")
    #: And the treatment's gate is resolved through the owner, not inlined.
    assert "protocol_gate" in code_without_comments(DRIVER.read_text()), (
        "the driver no longer resolves its digest gates through the session")


def test_the_seeds_are_attempt75_s_three_and_not_c1_s():
    """A3 trains on the controls' seeds BY DESIGN -- that is what makes the
    comparison paired. A fourth seed, or C1's, would unpair it."""
    from stages.phase_c1.isolation import derive_recovery_seeds

    assert A3S.recovery_seeds() == (217230555, 1151307191, 2045359208)
    assert not set(A3S.recovery_seeds()) & set(derive_recovery_seeds())
    assert len(A3S.probe_ids()) == 3, (
        f"{len(A3S.probe_ids())} probe ids; A3 trains exactly three")


def test_the_two_protocols_differ_in_execution_and_agree_in_identity(registered):
    """A3's signature property, asked on the machine that will execute it.

    The batching protocol is an EXECUTION knob: it changes how the forward
    passes are composed, not what the operator computes, so the two arms
    share a semantic state id. If they ever stopped sharing it, A3 would be
    comparing two different operators and the whole framing would be wrong;
    if their executions ever stopped differing, it would be running the
    incumbent twice and reporting a speedup of 1.0.
    """
    ref = A3S.PROTOCOL_EXECUTIONS[A3S.REFERENCE_PROTOCOL]
    trt = A3S.PROTOCOL_EXECUTIONS[A3S.TREATMENT_PROTOCOL]
    assert ref.micro_batch_size == 1
    assert trt.micro_batch_size == 3
    assert ref.calibration_batch_packing == "original_order_v1"
    assert trt.calibration_batch_packing == "length_sorted_v1"
    assert dataclasses.asdict(ref) != dataclasses.asdict(trt), (
        "the two protocols are executing identically; there is nothing to "
        "measure")

    spec = A3S.path_spec()
    tail = spec.steps[-1]
    assert tail.impl_id == A3S.ATTENTION_IMPL_ID, (
        f"the path's final step is {tail.impl_id}, not the incumbent operator")
    #: And the SPEC pins the reference arm's digest -- one path, two
    #: executions, and the gate that differs between them is applied by the
    #: driver through `protocol_gate`, not by the spec.
    assert tail.expected_artifact_digest in (
        None, A3S.expected_incumbent_digest())
    #: ONE operator id for both arms. The review forbade inventing an
    #: `attention.activation_importance_bsz3`, and this is where that would
    #: show up first.
    assert "bsz" not in tail.impl_id


def test_the_pod_computes_no_decision():
    """A3's comparison runs OFF POD at `$0`. attempt75 lost a ten-hour
    session in its on-pod stage I, and the repair was to delete the stage
    rather than to guard it -- so the absence is what gets checked."""
    assert A3S.AGGREGATION_IS_OFF_POD is True
    assert "I" not in A3S.STAGE_LETTERS, (
        f"the stage ladder {A3S.STAGE_LETTERS} contains a stage I")
    drv = load(DRIVER, "a3drv_no_decision")
    assert not hasattr(drv.A3Driver, "stage_i")
    src = code_without_comments(DRIVER.read_text())
    for forbidden in ("mcnemar", "bootstrap_ci", "noninferiority",
                      "paired_delta"):
        assert forbidden not in src.lower(), (
            f"the driver names {forbidden}; the decision belongs off pod")


def test_the_launcher_plan_hash_is_what_an_authorization_would_bind():
    """`require_plan` compares these two; a mismatch is exit 98 on a pod.

    **It does not skip when no authorization exists**, and that matters more
    than it looks. Written as "skip unless an issued artifact is on disk" it
    skipped in the launch-bound readiness sweep -- because the chain is
    grant → sweep → authorization → bundle, so at sweep time the artifact is
    one step in the future -- and the sweep reported
    `unexpected environment skips`, failing the record a funded launch rests
    on. The chain would have consumed a round per attempt, forever, for a
    test asking for something the order of operations guarantees is absent.

    So the DERIVATION is what gets asserted: the launcher's plan hash and the
    hash an authorization would carry come from the same owner, which is the
    C3 defect this exists to catch -- its launcher computed a hash from
    session constants the rewritten module no longer had, and the failure
    would have been `AUTHORIZATION_MISMATCH` after setup was paid for. An
    issued artifact, when one happens to exist, is compared as well.
    """
    from stages.phase_a3.a3_authorization_payload import (
        build_a3_authorization_payload,
    )

    contract = A3S.A3_SESSION_CONTRACT
    #: What the payload builder puts in `plan_hash`, asked of the builder.
    src = inspect.getsource(build_a3_authorization_payload)
    assert "plan_hash=contract.contract_hash" in src, (
        "the authorization's plan hash no longer comes from the session "
        "contract; the launcher and the artifact now have separate owners")
    sys.path.insert(0, str(REPO / "scripts/pod"))
    launcher = load(LAUNCH, "a3lau_plan")
    assert launcher.A3S.A3_SESSION_CONTRACT.contract_hash == (
        contract.contract_hash), (
        "the launcher resolves a different session contract than this test")

    #: Only authorizations issued against the CURRENT tree. A consumed
    #: historical chain bound the contract hash of the tree it ran on, and
    #: that is a fact about history rather than a defect: comparing today's
    #: contract to a closed attempt's authorization asserts that the contract
    #: may never change, which would make any repair to the design document
    #: look like a protocol violation. The container-disk repair moved the
    #: design hash -- and with it the contract hash -- while every science
    #: field stayed byte-identical.
    import subprocess

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                          capture_output=True, text=True,
                          check=True).stdout.strip()
    runs = (REPO / "logs/stages/stage-1/phase_a3/runs")
    issued = sorted(runs.glob("*/governance/authorization.json")) if (
        runs.is_dir()) else []
    checked = 0
    for path in issued:
        doc = json.loads(path.read_text())
        if doc.get("authorized_session_commit") != head:
            continue                      # a closed chain, bound to its tree
        bound = doc["bound"]["session_contract_hash"]
        checked += 1
        assert contract.contract_hash == bound, (
            f"{path.parent.parent.name}: the session contract hashes to "
            f"{contract.contract_hash[:16]} and that authorization binds "
            f"{bound[:16]}; the preflight would exit 98")
    #: On a pod there is exactly one, because the pod checks out the commit
    #: its own authorization was issued against.
    assert checked <= 1, (
        f"{checked} authorizations claim the current commit; a one-use chain "
        "is one authorization per attempt")


def test_every_required_audit_artifact_is_one_the_driver_writes():
    """A required artifact nobody writes is a session that completes and
    comes home incomplete -- C3 learned that at teardown, where the evidence
    is still on the pod and only recoverable if someone is watching."""
    spec = json.loads(
        (REPO / "configs/stages/stage-1/phase_a3/a3_artifacts.json").read_text())
    src = DRIVER.read_text()
    missing = []
    for entry in spec["entries"]:
        pattern = entry.get("pattern", "")
        if not entry.get("required") or "*" in pattern:
            continue
        if "/audit/" not in f"/{pattern}":
            continue
        name = pattern.rsplit("/", 1)[-1]
        if name not in src:
            missing.append(pattern)
    assert not missing, (
        f"the success spec requires {missing}, which the A3 driver never "
        f"writes; the manifest gate would report them MISSING at teardown")


def test_the_authorization_is_run_scoped_and_read_by_exactly_one_owner():
    """A repository-level pointer can name another experiment's artifact, and
    once did: C3's driver loaded C1's and was refused on schema eight minutes
    and `$0.15` into a session.

    A3 reads the artifact in ONE place -- the launcher, which hands the driver
    the derived figures as flags. That is deliberate and is not a gap: a
    driver that re-derived its own budget from the grant would be the second
    consumer of one derived quantity, which is how the reserves got
    double-counted before. So what gets checked here is that the path is
    run-scoped, that no repository-root pointer survives anywhere in the
    chain, and that the driver cannot plan past what it was handed.
    """
    launch_src = LAUNCH.read_text()
    drv_src = DRIVER.read_text()
    for stale in ("autoinit_c1_authorization.json",
                  "autoinit_c3_authorization.json",
                  "autoinit_a3_authorization.json"):
        assert stale not in launch_src + drv_src, (
            f"the A3 chain still names the repository-root pointer {stale}")
    assert "governance/authorization.json" in launch_src
    assert "def auth_path_for" in launch_src

    #: The one budget assertion the driver DOES own: it refuses a soft stop
    #: above the authorized ceiling, so a launcher that mis-derived the pair
    #: cannot be executed.
    mod = load(DRIVER, "a3drv_budget")
    #: Safe to call: the guard raises in `main` BEFORE `A3Driver` is
    #: constructed, so nothing touches the filesystem, the GPU or the clock.
    with pytest.raises(SystemExit) as exc:
        mod.main(["--image-digest", "sha256:x", "--rate", "1.09",
                  "--soft-stop-usd", "9.99", "--authorized-usd", "8.2525"])
    assert "exceeds the authorized" in str(exc.value)


# ---------------------------------------------------------------------------
# the accelerator
# ---------------------------------------------------------------------------

def test_a_reported_cuda_device_can_actually_run_a_kernel():
    """A real GEMM, not `is_available()`.

    Written as `skipif not cuda.is_available()` this skips on the CPU dev box
    and PASSES on the pod, so the readiness sweep refuses the record for an
    undeclared environment skip and a "must skip" expectation inverts exactly
    where it matters. So it asserts the IMPLICATION: if torch reports a
    device, a kernel must run on it. True in both environments.
    """
    import torch

    if not torch.cuda.is_available():
        return
    a = torch.randn(64, 64, device="cuda:0", dtype=torch.bfloat16)
    out = (a @ a).float().sum().item()
    assert out == out, "a device was reported but its GEMM produced NaN"


def test_the_path_is_the_frozen_incumbents_path_step_for_step(registered):
    """The defect that cost `$0.65` and would have cost the experiment.

    A3's reference protocol must rebuild the frozen incumbent, so its path has
    to BE the incumbent's path. It was not: the ATTENTION step took its
    profile from `steps[-1][1]` -- the profile of the step BEFORE it,
    `width.global_pca_v0`'s `calib.reasoning_heavy@v2` -- while the incumbent
    uses `calib.domain_balanced@v1`. A different calibration mixture gives
    different activation statistics, a different head map and different
    weights, so A-bsz1 built `7fbfadd0f0f6` against the frozen
    `53e30566c5f7`.

    Stage D's asymmetric gate caught it on a paid pod, which is what that gate
    is for. This check asks the same question for nothing, by comparing A3's
    spec to the arm C3 actually froze, step for step -- not by restating the
    profile, which is how two owners of one fact drift in the first place.
    """
    from stages.phase_c3 import session as CS

    CS.register_experimental_operators()
    try:
        incumbent = CS.build_arm_specs(workdir_device="cpu")["A_incumbent"]
    finally:
        pass

    a3 = A3S.path_spec(workdir_device="cpu")
    assert len(a3.steps) == len(incumbent.steps), (
        f"A3's path has {len(a3.steps)} steps and the frozen incumbent's has "
        f"{len(incumbent.steps)}")
    for i, (mine, theirs) in enumerate(zip(a3.steps, incumbent.steps)):
        assert (mine.impl_id, mine.profile_id) == (
            theirs.impl_id, theirs.profile_id), (
            f"step {i}: A3 runs {mine.impl_id}/{mine.profile_id} and the "
            f"frozen incumbent runs {theirs.impl_id}/{theirs.profile_id}. "
            "A-bsz1 cannot rebuild an artifact it does not build the same way.")

    #: And the frozen preregistration is what A3 reads it from, so the two
    #: cannot be made to agree by editing A3 alone.
    assert A3S.incumbent_attention_step() == (
        incumbent.steps[-1].impl_id, incumbent.steps[-1].profile_id)

    #: The tail carries NO digest in the SPEC -- the gate belongs to the run,
    #: because one path produces both protocols' artifacts.
    assert a3.steps[-1].expected_artifact_digest is None
    #: ...while the parent step does, and it is the frozen one.
    assert a3.steps[-2].expected_artifact_digest == A3S.expected_parent_digest()
