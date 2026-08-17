"""The materialize -> reload -> validate -> measure boundary, on two devices.

Phase-A attempt 6 reached stage 1 on a paid pod and died with

    index is on cuda:0, different from other tensors on cpu
    (wrapper_CUDA__index_select)

`_validate` forwards the produced child and the canonical reload through **one**
input. That input used to be built on `SearchConfig.device`, and so was the
reload — but the produced child is whatever the operator built, and
`ChildBuilder` -> `build_student` sets the dtype and deliberately does not place
the model. On a GPU run the child therefore sits on the host and the probe
indexed CPU embedding weights with a CUDA index.

**No zero-cost run could see it.** Every one passes `device="cpu"`, where the
produced model and `config.device` coincide. That is why this test does not use
a real device: it records placement instead of performing it, so the two
devices can differ on a CPU-only box. Under the old assumption the probe is
built with `torch.tensor(..., device="cuda:0")` and this fails at once on a box
with no GPU — which is the point. Under the fix the probe is built where the
models actually are.

Scope is this boundary. The Phase-A rehearsal is not extended.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import aadistill.autoinit  # noqa: F401,E402
from aadistill.autoinit.arch import ArchSpec  # noqa: E402
from aadistill.autoinit.metrics import StateEvaluation  # noqa: E402
from aadistill.autoinit.ranking import PARETO_V1, BeamSchedule  # noqa: E402
from aadistill.autoinit.search import BeamSearch, SearchConfig, model_device  # noqa: E402
from aadistill.autoinit.state import StateValidity  # noqa: E402
# `fake_family` is imported INSIDE the helpers below, never at module scope.
# Importing it registers the toy kinds globally, and
# `test_registry.py::test_the_kind_set_is_open` asserts they are absent. Two
# things keep this file out of that test's way, and both are needed: the lazy
# import keeps COLLECTION clean, and the filename sorts after `test_registry`
# so the registration happens after that test has run. `test_search.py` uses
# the same two, for the same reason. Undoing the registrations afterwards is
# NOT an option — a module is imported once, so unregistering what its import
# performed leaves `test_search.py` without the kinds it needs.


#: A device name that is REAL enough for torch to reject it on this box. That
#: rejection is the regression: the old code hands it to `torch.tensor`.
ELSEWHERE = "cuda:0"


def placement_recording_adapter():
    """An adapter that records where it was asked to put a model, and always
    puts it on the host.

    The dev box has no GPU, so performing the placement is impossible and faking
    a torch device is worse than useless. What matters at this boundary is
    *which device each step asks for*, and that is observable without one.
    """
    from fake_family import ToyAdapter

    class PlacementRecordingAdapter(ToyAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.load_devices: list[str] = []
            self.moved_to: list[str] = []

        def load(self, path, dtype=None, device="cpu"):
            self.load_devices.append(str(device))
            model = ToyAdapter.load(self, path, dtype=dtype, device="cpu")
            moved = self.moved_to

            def recording_to(target, *a, **k):
                moved.append(str(target))
                return model

            model.to = recording_to      # shadows the bound method on this instance
            return model

    return PlacementRecordingAdapter()


def toy_config():
    from fake_family import ToyConfig

    return ToyConfig(d_model=8, n_experts=2, expert_width=4, vocab_size=16)


def build_search(tmp_path, adapter, device: str):
    from fake_family import TOY_FAMILY
    from test_search import _toy_profile, _toy_suite

    # `register_all()` is deliberately NOT called. It registers the toy MOE
    # kinds globally, and `test_registry.py::test_the_kind_set_is_open` asserts
    # they are absent — this file sorts before that one. Nothing here dispatches
    # an operator: the target spec equals the root spec, so
    # `_assert_target_reachable` has no differing field to find an
    # implementation for, and `_materialize_and_measure` is called directly.
    spec = ArchSpec.of(TOY_FAMILY, {"d_model": 8, "n_experts": 2,
                                    "expert_width": 4, "vocab_size": 16})
    config = SearchConfig(
        run_id="device_boundary", target_spec=spec,
        schedule=BeamSchedule("toy.beam", 1, "toy", warmup_levels=0, width=2),
        seed=7, workdir=tmp_path / "wd", profiles=(_toy_profile(),),
        policy=PARETO_V1, suite=_toy_suite(), device=device)
    measured: list[tuple[str, str]] = []

    def measurer(model, digest):
        measured.append((str(model_device(model)), digest))
        return StateEvaluation(
            artifact_digest=digest, suite_id=config.suite.suite_id,
            suite_hash=config.suite.suite_hash, reference="root_teacher",
            values={k: 0.0 for k in config.suite.required_metrics()}, positions=8)

    search = BeamSearch(
        adapter=adapter, config=config, root_teacher_id="toy",
        root_teacher_sha256="b" * 64,
        root_loader=lambda: adapter.build_model(toy_config(), torch.float32, 1),
        calibration_loader=lambda p: [{"input_ids": torch.arange(8).reshape(1, 8)}],
        measurer=measurer)
    return search, config, spec, measured


def test_the_cycle_validates_where_the_models_are_and_measures_where_the_search_is(
        tmp_path):
    """The whole lifecycle, with the produced model NOT on `config.device`.

    Under the old same-config-device-input assumption this raises immediately —
    `torch.tensor([[1, 2, 3, 4, 5]], device="cuda:0")` on a box with no GPU —
    before any assertion below is reached.
    """
    adapter = placement_recording_adapter()
    search, config, spec, measured = build_search(tmp_path, adapter, ELSEWHERE)

    produced = adapter.build_model(toy_config(), torch.float32, 3)
    assert str(model_device(produced)) == "cpu", (
        "the fixture no longer reproduces the real situation: an operator's "
        "child is built unplaced, which is why it is on the host")
    assert str(config.device) == ELSEWHERE, "the two devices must differ"

    state = search.root_state()
    search._materialize_and_measure(state, produced, spec)

    checks = state.notes["validation"]
    # Validated on the produced model's device, measured on the search device.
    assert checks["validation_device"] == "cpu"
    assert checks["measurement_device"] == ELSEWHERE
    assert adapter.load_devices == ["cpu"], (
        f"the canonical reload was loaded onto {adapter.load_devices}; a reload "
        "placed on the search device cannot be compared against a host-resident "
        "child on one input")
    assert adapter.moved_to == [ELSEWHERE], (
        f"the reload was moved to {adapter.moved_to}; the measurement must "
        "happen on the search device, not wherever validation happened")

    # And the save/reload check itself still did its job, on one backend.
    assert checks["finite"] is True
    assert checks["reload_max_logit_diff"] == pytest.approx(0.0, abs=1e-5)
    assert state.validity is StateValidity.MEASURED
    assert measured and measured[0][0] == "cpu", (
        "the measurer received the reload; its device is whatever `.to` was "
        "asked for, which the recording adapter does not perform")


def test_model_device_reads_the_weights_not_the_config(tmp_path):
    """`model_device` is the whole fix in one function, so it is pinned.

    Asking the config where a model is was the bug. Ask the weights.
    """
    adapter = placement_recording_adapter()
    produced = adapter.build_model(toy_config(), torch.float32, 5)
    assert model_device(produced) == next(produced.parameters()).device

    class Parameterless(torch.nn.Module):
        pass

    assert model_device(Parameterless()) == torch.device("cpu")
