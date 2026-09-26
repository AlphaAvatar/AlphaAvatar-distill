"""The C3 pilot replays the frozen prefix at B=1, proved by RUNNING it.

The earlier version of this evidence built its own `materialize_fixed_path`
call with `execution=ExecutionConfig(1)` and asserted that the call it had
just written passed a 1. That is a fixture agreeing with itself: the pilot
could have forgotten the argument entirely and the evidence would still be
green.

So everything here goes through `pilot.replay_prefix`, the one function the
launcher and the driver also call, at toy scale on CPU. What is checked is not
that B=1 is possible but that the pilot's own path DOES it -- while
`DEFAULT_EXECUTION.micro_batch_size` is 4, which is what the frozen parent was
NOT built under.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from aadistill.initialization.adapters.qwen3 import QWEN3_ADAPTER  # noqa: E402
from aadistill.initialization.execution import DEFAULT_EXECUTION  # noqa: E402
from aadistill.initialization.operators.attention.gqa import causal_kl  # noqa: E402
from aadistill.initialization.operators.register import (  # noqa: E402
    register_builtin_operators)
from aadistill.initialization.planning.fixed_path import (  # noqa: E402
    FixedPathSpec, FixedPathStep)

from conftest import build_tiny_model  # noqa: E402
from experiments.phase_c3 import pilot  # noqa: E402

#: Toy, and reducible in every dimension the prefix touches.
PARENT = dict(hidden_size=32, num_hidden_layers=4, intermediate_size=64,
              num_attention_heads=8, num_key_value_heads=2, head_dim=8,
              vocab_size=128, tie_word_embeddings=True)
TARGET = dict(PARENT, num_hidden_layers=3, intermediate_size=48,
              hidden_size=16, num_attention_heads=4)


@pytest.fixture(autouse=True)
def registered():
    register_builtin_operators()
    causal_kl.register(replace=True)
    yield
    causal_kl.unregister()


def toy_items(n_per_subtype=2, vocab=128, seed=5):
    """Ragged, so a batch size above 1 would visibly pad."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for subtype, lens in (("general", (11, 15)), ("alpha", (9, 13)),
                          ("beta", (14, 10))):
        for k, length in enumerate(lens[:n_per_subtype]):
            out.append({
                "item_id": f"{subtype}/{k}",
                "domain": "general" if subtype == "general" else "task",
                "subtype": subtype,
                "input_ids": torch.randint(0, vocab, (1, length), generator=g),
            })
    return out


def toy_spec(batch_size: int) -> FixedPathSpec:
    """The pilot's own step list, at a geometry a CPU can execute.

    `pilot.prefix_steps()` and `pilot.causal_step()` are the REAL constructors
    -- only the target geometry and the root are toy, because the question is
    which execution config `replay_prefix` passes, not how big the model is.
    """
    from aadistill.initialization.specs.arch import ArchSpec

    steps = tuple(pilot.prefix_steps(pin_parent=False)) + (
        pilot.causal_step(batch_size),)
    return FixedPathSpec(
        path_id=f"toy.{pilot.ARM_IDS[batch_size]}", family="qwen3",
        target_spec=ArchSpec.of("qwen3", TARGET), steps=steps,
        root_repo_id="toy/root", root_revision="toy", device="cpu", seed=0)


def run(tmp_path, batch_size=1):
    items = toy_items()
    results = []
    pilot.replay_prefix(
        toy_spec(batch_size), adapter=QWEN3_ADAPTER,
        root_loader=lambda: build_tiny_model(PARENT),
        workdir=tmp_path / "work", repo_root=REPO,
        calibration_items={"calib.domain_balanced@v1": items,
                           "calib.reasoning_heavy@v2": items},
        on_step=results.append)
    return results


# --- the evidence, from the path the launcher uses -------------------------

def test_the_test_is_not_vacuous():
    """If the process default were already 1 this file would prove nothing."""
    assert DEFAULT_EXECUTION.micro_batch_size == 4, (
        "the pilot's B=1 pin is only meaningful while the default differs")


def test_every_prefix_step_actually_ran_at_batch_size_one(tmp_path):
    """RUN, not asserted about a fixture. Each prefix operator records the
    batch size it used, so this reads back what the executor really did."""
    results = run(tmp_path)
    assert [r.kind for r in results] == ["DEPTH", "FFN", "RESIDUAL_WIDTH",
                                         "ATTENTION"]
    for step in results[:3]:
        assert step.trace["micro_batch_size"] == 1, (
            f"{step.impl_id} replayed at {step.trace['micro_batch_size']}, not 1; "
            "the frozen parent was built by the one-item path")


def test_the_causal_step_is_the_only_one_that_varies(tmp_path):
    """The prefix stays at 1 even when the ATTENTION arm is B4."""
    results = run(tmp_path, batch_size=4)
    for step in results[:3]:
        assert step.trace["micro_batch_size"] == 1
    assert results[3].trace["calibration_forward_batch_size"] == 4


def test_the_causal_step_reads_its_own_config_not_the_execution(tmp_path):
    b1 = run(tmp_path / "a", batch_size=1)[3]
    b4 = run(tmp_path / "b", batch_size=4)[3]
    assert b1.trace["calibration_forward_batch_size"] == 1
    assert b4.trace["calibration_forward_batch_size"] == 4
    #: And the prefix underneath them was byte-identical, which is the whole
    #: point of pinning it: both arms build on the same parent.
    assert [s.identity.artifact_digest for s in run(tmp_path / "c")[:3]] == \
           [s.identity.artifact_digest for s in run(tmp_path / "d", 4)[:3]]


def test_replay_prefix_has_no_execution_parameter():
    """A caller able to pass one could undo the pin without touching the pilot."""
    import inspect

    params = inspect.signature(pilot.replay_prefix).parameters
    assert "execution" not in params, (
        "the execution mode is the pilot's decision, not the caller's")


def test_the_pin_is_a_literal_and_not_an_alias():
    """`PREFIX_EXECUTION = DEFAULT_EXECUTION` would track the default and
    silently become B4, which is exactly the coupling to avoid."""
    import ast

    tree = ast.parse((REPO / "scripts/experiments/phase_c3/pilot.py").read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and any(getattr(t, "id", None) == "PREFIX_EXECUTION"
                        for t in node.targets)):
            assert isinstance(node.value, ast.Call), "not a constructed config"
            kwargs = {k.arg: k.value for k in node.value.keywords}
            assert isinstance(kwargs.get("micro_batch_size"), ast.Constant)
            assert kwargs["micro_batch_size"].value == 1
            return
    pytest.fail("PREFIX_EXECUTION is not assigned in the pilot module")


# --- the arm constructor builds the frozen path, not a copy of it ----------

def test_both_arms_come_from_one_constructor(tmp_path):
    """"Identical by construction" has to be a property of the code."""
    b1, b4 = pilot.arm_spec(1, repo_root=REPO), pilot.arm_spec(4, repo_root=REPO)
    assert [s.as_dict() for s in b1.steps[:3]] == [s.as_dict() for s in b4.steps[:3]]
    assert b1.target_spec == b4.target_spec
    assert b1.spec_hash != b4.spec_hash, "the arms must be distinct states"


def test_the_arm_target_is_read_from_the_committed_record():
    """Not retyped: a target typed into the pilot could drift from the path
    the frozen prefix was built for."""
    import json

    doc = json.loads((REPO / pilot.C1_PATH_RECORD).read_text())
    spec = pilot.arm_spec(1, repo_root=REPO).target_spec
    assert spec.as_dict() == doc["path"]["target_spec"]
    assert spec.spec_hash == doc["path"]["target_spec_hash"]


def test_the_root_is_the_pinned_commit_the_frozen_path_used():
    """A MOVING reference here would replay from a different checkpoint.

    This module first hardcoded `root_revision = "main"`. The frozen path
    pins a commit sha; `main` is whatever the hub points at on the day the
    pilot runs, and the parent digest gate would only have caught it after
    the replay was paid for -- the same shape as the profile defect.
    """
    import json

    record = json.loads((REPO / pilot.C1_PATH_RECORD).read_text())["path"]
    repo_id, revision = pilot.root_binding(REPO)
    assert (repo_id, revision) == (record["root_repo_id"],
                                   record["root_revision"])
    assert revision != "main" and len(revision) == 40
    spec = pilot.arm_spec(1, repo_root=REPO)
    assert (spec.root_repo_id, spec.root_revision) == (repo_id, revision)


def test_a_moving_root_reference_is_refused(tmp_path):
    """The guard fires, shown rather than assumed."""
    import json

    doc = json.loads((REPO / pilot.C1_PATH_RECORD).read_text())
    doc["path"]["root_revision"] = "main"
    fake = tmp_path / pilot.C1_PATH_RECORD
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="not a pinned commit sha"):
        pilot.root_binding(tmp_path)


def test_the_parent_is_pinned_at_the_width_step_by_default():
    spec = pilot.arm_spec(1, repo_root=REPO)
    pinned = [s for s in spec.steps if s.expected_artifact_digest]
    assert len(pinned) == 1
    assert pinned[0].impl_id == "width.global_pca_v0"
    assert pinned[0].expected_artifact_digest == pilot.FROZEN_PARENT_DIGEST


def test_the_arm_path_ids_name_the_arms_not_the_batch_size_twice():
    assert pilot.arm_spec(1, repo_root=REPO).path_id.endswith("causal-B1")
    assert pilot.arm_spec(4, repo_root=REPO).path_id.endswith("causal-B4")
    #: The step LABEL is still shared -- the arm name lives in the path id.
    for bs in (1, 4):
        assert pilot.arm_spec(bs, repo_root=REPO).steps[3].label == \
            pilot.CAUSAL_STEP_LABEL


def test_a_step_may_not_be_added_twice_for_one_kind():
    """The spec's own guard, asserted so the pilot cannot grow a second
    ATTENTION step without something turning red."""
    from aadistill.initialization.planning.fixed_path import FixedPathError
    from aadistill.initialization.specs.arch import ArchSpec

    steps = tuple(pilot.prefix_steps()) + (pilot.causal_step(1),
                                           pilot.causal_step(4))
    with pytest.raises(FixedPathError, match="repeats kind"):
        FixedPathSpec(path_id="toy", family="qwen3",
                      target_spec=ArchSpec.of("qwen3", TARGET), steps=steps,
                      root_repo_id="toy/root", root_revision="toy")


def test_an_unknown_arm_is_refused():
    with pytest.raises(KeyError):
        pilot.arm_spec(2, repo_root=REPO)


def test_the_prefix_steps_carry_no_config(tmp_path):
    """A config on a prefix step would move the B1/B4 boundary and change the
    parent both arms are supposed to share."""
    for step in pilot.arm_spec(4, repo_root=REPO).steps[:3]:
        assert not step.config
        assert "config" not in step.as_dict()


def test_a_prefix_step_gaining_a_config_would_change_its_hash():
    """The premise of the test above, so it cannot pass vacuously."""
    from aadistill.infrastructure.manifest import sha256_json

    bare = pilot.prefix_steps()[0]
    loaded = FixedPathStep(impl_id=bare.impl_id, profile_id=bare.profile_id,
                           config={"anything": 1})
    assert sha256_json(bare.as_dict()) != sha256_json(loaded.as_dict())


def test_the_arm_constructor_refuses_before_the_operator_is_registered():
    """A launcher that forgets `causal_kl.register()` must fail HERE, at $0.

    IN A SUBPROCESS, because the registry is process-global: in this session a
    sibling test's fixture has already filled it, so an in-process check would
    pass no matter what `arm_spec` does. `FixedPathSpec.__post_init__` resolves
    every step's implementation, so the refusal is structural -- but only a
    clean process can show it.
    """
    import subprocess

    probe = (
        "import sys; sys.path[:0] = ['src', 'scripts'];"
        "from aadistill.initialization.operators.register import"
        " register_builtin_operators;"
        "register_builtin_operators();"
        "from experiments.phase_c3 import pilot;"
        "\ntry:\n"
        "    pilot.arm_spec(1)\n"
        "    print('NO_REFUSAL')\n"
        "except Exception as exc:\n"
        "    print(type(exc).__name__)\n"
    )
    done = subprocess.run([sys.executable, "-c", probe], cwd=REPO,
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() != "NO_REFUSAL", (
        "arm_spec built a path for an unregistered operator; a launcher that "
        "forgot to register would discover it on the pod")
    assert "Error" in done.stdout or "Unknown" in done.stdout, done.stdout


def test_the_arm_constructor_succeeds_once_it_is_registered():
    """The other half, so the refusal above cannot be passing for a wrong
    reason -- a typo in the probe would also "refuse"."""
    import subprocess

    probe = (
        "import sys; sys.path[:0] = ['src', 'scripts'];"
        "from aadistill.initialization.operators.register import"
        " register_builtin_operators;"
        "register_builtin_operators();"
        "from aadistill.initialization.operators.attention.gqa import causal_kl;"
        "causal_kl.register();"
        "from experiments.phase_c3 import pilot;"
        "print(pilot.arm_spec(4).steps[3].impl_id)"
    )
    done = subprocess.run([sys.executable, "-c", probe], cwd=REPO,
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "attention.causal_kl_v1"
