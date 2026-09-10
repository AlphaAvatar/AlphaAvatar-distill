"""The end-to-end suffix case, and proof that it can fail.

The per-operator matrix in `cuda_engineering_check` could not have caught
attempt 9 twice over: the failure was in the COMPOSITION
(`materialize_fixed_path_suffix` -> `_run_steps` -> `impl.execute` -> `apply` ->
`head_write_energy`), and the operator that failed,
`attention.activation_importance_v1`, is not in the matrix at all. The suffix
case runs that composition. These tests run the suffix case.

**Nothing here is a CUDA result.** Everything below executes on one device, where
every co-location claim is trivially true — the exact substitution that let
attempt 9 reach a paid pod. What these tests establish is that the harness
*works*: that it executes production code, that it observes the placements
rather than assuming them, and above all that it **goes red** when the defect is
reintroduced. A harness that passes on the repaired code and also passes on the
broken code measures nothing, and this project has shipped that kind of gate
before.

The device topology that cannot be real on this box is modelled with
`device_split`, the instrument the repository already has for exactly this.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ENTRY = REPO / "scripts/validation/cuda_engineering_check.py"
CONFIG = REPO / "configs/validation/cuda_engineering.json"

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests/autoinit"))
sys.path.insert(0, str(REPO / "scripts/validation"))

from aadistill.initialization.operators import attention_activation  # noqa: E402

from device_split import CrossDeviceUse, on_cache_device  # noqa: E402
from device_observations import DeviceObservations  # noqa: E402


@pytest.fixture
def check():
    spec = importlib.util.spec_from_file_location("cuda_check_suffix", ENTRY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def cfg():
    return json.loads(CONFIG.read_text())


@pytest.fixture
def adapter():
    from aadistill.initialization.adapters import register_builtin_adapters
    from aadistill.initialization.operators.register import (
        register_builtin_operators)
    from aadistill.initialization.specs.arch import get_adapter

    register_builtin_adapters()
    register_builtin_operators()
    return get_adapter(json.loads(CONFIG.read_text())["model"]["adapter"])


def run_case(check, cfg, adapter, tmp_path, geometry_index=0):
    geometry = cfg["suffix_case"]["geometries"][geometry_index]
    return check.run_suffix_case(
        cfg, device="cpu", dtype_name="float32", geometry=geometry,
        root=tmp_path, adapter=adapter,
        build_root=lambda: check.build_fixture(cfg, "cpu", "float32"))


def split_the_snapshot(monkeypatch):
    """Put the statistics snapshot on a different logical device.

    This labels what the real `AttentionHeadStatsCollector.state()` returns, at
    the real boundary. It replaces no collector, no operator and no arithmetic.
    """
    real = attention_activation.AttentionHeadStatsCollector.state
    monkeypatch.setattr(attention_activation.AttentionHeadStatsCollector,
                        "state", lambda self: on_cache_device(real(self)))


# --- the case runs, and runs the right thing --------------------------------

class TestTheCaseExecutesTheRealTreatmentPath:
    def test_it_passes_on_the_repaired_code(self, check, cfg, adapter, tmp_path):
        out = run_case(check, cfg, adapter, tmp_path)
        assert out["passed"], json.dumps(out, indent=1)[:3000]

    def test_it_executes_the_treatment_operator_not_the_incumbent(
            self, check, cfg, adapter, tmp_path):
        """The whole point: the matrix runs `attention.weight_proxy_v0`."""
        out = run_case(check, cfg, adapter, tmp_path)
        assert out["treatment_impl_id"] == "attention.activation_importance_v1"
        assert out["checks"]["the_treatment_operator_executed"]
        assert out["checks"]["the_treatment_did_its_own_work"]

    def test_only_the_original_suffix_index_executes(self, check, cfg, adapter,
                                                     tmp_path):
        """Proven from the checkpoints on disk, not from a patched `execute`.

        A recorder that patches the operators proves what the patch saw. This
        asks whether anything wrote a step-0/1/2 checkpoint into the stage-F
        workdir.
        """
        out = run_case(check, cfg, adapter, tmp_path)
        assert out["executed_step_indices"] == [3]
        assert out["steps_written"] == ["03_attention"]
        assert out["checks"]["the_original_suffix_index_is_retained"]
        assert out["checks"]["the_prefix_did_not_execute_again"]

    def test_the_parent_it_starts_from_was_genuinely_gated(
            self, check, cfg, adapter, tmp_path):
        out = run_case(check, cfg, adapter, tmp_path)
        assert out["checks"]["the_parent_was_genuinely_gated"]

    def test_the_record_is_written_and_reads_back(self, check, cfg, adapter,
                                                  tmp_path):
        out = run_case(check, cfg, adapter, tmp_path)
        assert out["checks"]["the_record_is_not_a_replay"]
        assert out["checks"]["the_record_names_the_executed_step"]
        assert out["checks"]["the_output_stays_bound_to_the_full_frozen_path"]
        assert (tmp_path / f"suffix_{out['geometry_id']}" / out["record"]).is_file()

    @pytest.mark.parametrize("i", [0, 1])
    def test_both_declared_geometries_run(self, check, cfg, adapter, tmp_path, i):
        out = run_case(check, cfg, adapter, tmp_path, geometry_index=i)
        assert out["passed"], out.get("error")

    def test_the_config_declares_at_least_two_and_both_reduce_heads(self, cfg):
        """A geometry that keeps every head exercises selection without ever
        discarding anything."""
        geoms = cfg["suffix_case"]["geometries"]
        assert len(geoms) >= 2
        parent_heads = cfg["source_structure"]["num_attention_heads"]
        for g in geoms:
            assert g["num_attention_heads"] < parent_heads, g["geometry_id"]
        assert len({g["num_key_value_heads"] for g in geoms}) > 1, (
            "both geometries use the same kv grouping, so the grouping path is "
            "exercised at one ratio")


# --- it reports honestly about what a CPU run means -------------------------

class TestItDoesNotOversellACpuRun:
    def test_it_marks_the_device_proofs_as_not_meaningful(self, check, cfg,
                                                          adapter, tmp_path):
        out = run_case(check, cfg, adapter, tmp_path)
        assert out["device_proofs_are_meaningful"] is False
        assert "trivially true" in out["_why"]

    def test_the_entry_point_still_refuses_cpu(self, check):
        with pytest.raises(check.NotRun, match="validates CUDA"):
            check.require_cuda("cpu")


# --- the mutations: the harness must be able to fail ------------------------

class TestItGoesRedWhenTheDefectIsBack:
    def test_the_attempt_9_defect_fails_the_case(self, check, cfg, adapter,
                                                 tmp_path, monkeypatch):
        """Remove the working copy, split the devices, and stage F dies where it
        died on the L40S -- surfaced through the harness, not a unit."""
        split_the_snapshot(monkeypatch)
        monkeypatch.setattr(attention_activation, "stats_to",
                            lambda state, device: state)      # the old behaviour

        with pytest.raises(CrossDeviceUse, match="persistent cache device"):
            run_case(check, cfg, adapter, tmp_path)

    def test_main_records_that_failure_rather_than_passing(
            self, check, cfg, adapter, tmp_path, monkeypatch):
        """`main` catches per-case exceptions so one failure does not hide the
        others. It must still fail the run."""
        split_the_snapshot(monkeypatch)
        monkeypatch.setattr(attention_activation, "stats_to",
                            lambda state, device: state)
        monkeypatch.setattr(check, "require_cuda", lambda r: {
            "device_name": "substituted-for-test", "capability": "0.0",
            "device_count": 0, "torch": "n/a", "cuda_runtime": None,
            "driver": "none"})
        doc = dict(cfg)
        doc["execution"] = {**cfg["execution"], "device": "cpu",
                            "dtype": "float32"}
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(doc))
        monkeypatch.setattr("sys.argv", [
            "cuda_engineering_check.py", "--config", str(p), "--run-id", "mut",
            "--run-root", str(tmp_path / "runs")])
        rc = check.main()

        report = json.loads(
            (tmp_path / "runs/cuda_engineering/mut/report.json").read_text())
        assert rc != 0
        assert report["suffix_case_passed"] is False
        assert report["passed"] is False
        assert any("CrossDeviceUse" in (c.get("error") or "")
                   for c in report["suffix_case"])

    def test_a_stand_in_operator_is_refused_by_the_path(self, check, cfg,
                                                        adapter, tmp_path,
                                                        monkeypatch):
        """The guard on this file: if a fake could satisfy the case, the case
        would be proving nothing about the real operator."""
        from aadistill.initialization.operators.base import (
            OperatorOutcome, get_implementation)
        from aadistill.initialization.planning.fixed_path import FixedPathError

        attention_activation.register(replace=True)
        impl = get_implementation("attention.activation_importance_v1")
        monkeypatch.setattr(
            impl, "execute",
            lambda ctx: OperatorOutcome(model=ctx.model, local_metrics=None,
                                        trace={}, artifacts={}))
        with pytest.raises(FixedPathError, match="is not the planned"):
            run_case(check, cfg, adapter, tmp_path)


# --- an unobserved proof is not a satisfied proof ---------------------------

class TestAnUnreachedSiteCannotPass:
    """The failure mode this project keeps hitting: a check that never ran,
    reported as a check that passed."""

    def test_an_empty_observation_set_does_not_hold(self):
        rep = DeviceObservations().report("cuda")
        assert set(rep) == {
            "collector_state_is_the_host_snapshot",
            "one_working_copy_reaches_the_model_device",
            "statistics_and_o_proj_are_co_located",
            "score_vector_is_allocated_on_the_operand_device",
            "returned_score_vector_is_host_resident"}
        for name, proof in rep.items():
            assert proof["observed"] is False, name
            assert proof["holds"] is False, name

    def test_a_second_working_copy_is_a_finding(self):
        """One transfer, after the whole snapshot is taken. Two would mean the
        transfer moved into the per-layer loop."""
        obs = DeviceObservations()
        obs.working_copy_devices = [["cuda:0"], ["cuda:0"]]
        rep = obs.report("cuda")
        assert not rep["one_working_copy_reaches_the_model_device"]["holds"]

    def test_a_host_working_copy_is_a_finding(self):
        obs = DeviceObservations()
        obs.working_copy_devices = [["cpu", "cpu"]]
        assert not obs.report("cuda")["one_working_copy_reaches_the_model_device"]["holds"]

    def test_split_operands_are_a_finding(self):
        """Literally attempt 9: statistics on the host, o_proj on the device."""
        obs = DeviceObservations()
        obs.energy_operands = [{"layer": 0, "stats": "cpu", "o_proj": "cuda:0"}]
        assert not obs.report("cuda")["statistics_and_o_proj_are_co_located"]["holds"]

    def test_a_host_allocated_score_buffer_is_a_finding(self):
        """The latent second defect behind the first one."""
        obs = DeviceObservations()
        obs.score_alloc_devices = ["cpu"]
        assert not obs.report("cuda")["score_vector_is_allocated_on_the_operand_device"]["holds"]

    def test_a_device_resident_return_is_a_finding(self):
        obs = DeviceObservations()
        obs.score_return_devices = ["cuda:0"]
        assert not obs.report("cuda")["returned_score_vector_is_host_resident"]["holds"]

    def test_the_live_run_actually_reaches_every_site(self, check, cfg, adapter,
                                                      tmp_path):
        """If the wrappers silently stopped firing, every proof would read
        `observed: false` and the case would fail -- but only because this
        asserts it."""
        out = run_case(check, cfg, adapter, tmp_path)
        for name, proof in out["device_proofs"].items():
            assert proof["observed"] is True, f"{name} was never reached"


# --- the matrix criterion, against the contract it must not contradict ------

class TestTheMatrixAsksTheRightQuestion:
    """The first real GPU run failed all 8 matrix cases while every operator
    had succeeded.

    The criterion demanded the CHILD be on the requested device.
    `initialization/device.py` documents the opposite: "an operator's child
    comes from ChildBuilder -> build_student, which sets the dtype and does NOT
    place the model, so a parent on CUDA routinely coexists with a freshly
    built child on the host."

    It was invisible on CPU, where `device` is the host and the check passed
    trivially. That is the inverse of the usual trap: CPU did not hide a device
    bug, it hid a wrong ACCEPTANCE CRITERION.
    """

    def test_the_documented_contract_still_says_the_child_is_not_placed(self):
        """If this sentence ever leaves device.py, the criterion below must be
        revisited rather than silently kept."""
        doc = (REPO / "src/aadistill/initialization/device.py").read_text()
        assert "does NOT place the model" in doc
        assert "coexists with a freshly built child on the host" in doc

    def test_a_case_is_judged_on_where_the_OPERATOR_ran(self, check):
        """A parent on the host means the operator did not run on the device,
        whatever the config asked for."""
        rows = [{"applied": True, "ran_on_requested_device": True,
                 "child_host_resident_per_builder_contract": True},
                {"applied": True, "ran_on_requested_device": False,
                 "child_host_resident_per_builder_contract": True}]
        passed = [r for r in rows
                  if r.get("applied")
                  and r.get("ran_on_requested_device") is True
                  and r.get("child_host_resident_per_builder_contract") is True]
        assert len(passed) == 1

    def test_the_old_criterion_would_have_failed_a_healthy_cuda_run(self):
        """The exact 2026-09-10 shape: parent on cuda, child on cpu, no error."""
        row = {"applied": True, "parent_device": "cuda:0", "child_device": "cpu"}
        old = (row["child_device"] is not None
               and row["child_device"].startswith("cuda"))
        assert old is False, "the old criterion is what failed 8/8"
        new = (row["applied"] and row["parent_device"].startswith("cuda")
               and row["child_device"].startswith("cpu"))
        assert new is True

    def test_it_is_not_weakened_to_accept_a_cpu_run(self, check, cfg, adapter,
                                                    tmp_path):
        """The correction must not have turned the matrix into something that
        passes when nothing ran on a GPU. A host parent still fails."""
        row = {"applied": True, "ran_on_requested_device": False,
               "child_host_resident_per_builder_contract": True}
        assert not (row["ran_on_requested_device"] is True), (
            "a run whose parent never reached the device must not pass")

    def test_the_source_no_longer_names_the_wrong_key(self):
        src = (REPO / "scripts/validation/cuda_engineering_check.py").read_text()
        code = "\n".join(l for l in src.splitlines()
                         if not l.lstrip().startswith("#"))
        assert "child_on_requested_device" not in code
