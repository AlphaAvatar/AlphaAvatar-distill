"""The CUDA validation entry point, executed end to end without a GPU.

Every line after `require_cuda` is unreachable on this host, and unreachable
code is where this project's paid sessions have died: four pods have now failed
inside lines that no $0 gate had ever executed. So these tests substitute the
device gate -- and only the device gate -- and run the whole body on CPU.

**This is not a CUDA pass and cannot be read as one.** It asserts the harness's
mechanics: that the config is honoured, that the operator ids are checked before
any device work, that a failing operator is recorded rather than allowed to hide
the others, that the report has the shape the run layout declares. The device
verdict is exactly what these tests replace, which is why the entry point writes
no report at all on NOT RUN and why nothing here asserts `passed is True` as
evidence about CUDA.

The one thing that cannot be checked here is the thing the check exists for: a
tensor on the wrong device is unobservable with one device. That is the residual
risk, and it is why the GPU run is still owed.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ENTRY = REPO / "scripts/validation/cuda_engineering_check.py"
CONFIG = REPO / "configs/validation/cuda_engineering.json"


@pytest.fixture
def check():
    spec = importlib.util.spec_from_file_location("cuda_check_under_test", ENTRY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def cpu_config(tmp_path):
    """The real config, with the device swapped. Nothing else changes."""
    doc = json.loads(CONFIG.read_text())
    doc["execution"]["device"] = "cpu"
    doc["execution"]["dtype"] = "float32"
    out = tmp_path / "cpu_config.json"
    out.write_text(json.dumps(doc))
    return out


def run(check, monkeypatch, cfg: Path, tmp_path, run_id="t1", argv_extra=()):
    """Run main() with the device gate substituted, and return (rc, report)."""
    monkeypatch.setattr(check, "require_cuda",
                        lambda requested: {"device_name": "substituted-for-test",
                                           "capability": "0.0", "device_count": 0,
                                           "torch": "n/a", "cuda_runtime": None,
                                           "driver": "none"})
    monkeypatch.setattr("sys.argv", [
        "cuda_engineering_check.py", "--config", str(cfg),
        "--run-id", run_id, "--run-root", str(tmp_path / "runs"), *argv_extra])
    rc = check.main()
    report = tmp_path / "runs/cuda_engineering" / run_id / "report.json"
    return rc, (json.loads(report.read_text()) if report.is_file() else None)


class TestTheDeviceGate:
    def test_no_cuda_is_not_run_and_writes_nothing(self, check, tmp_path,
                                                   monkeypatch):
        """A CPU pass under a GPU label would be worse than no check."""
        monkeypatch.setattr("sys.argv", [
            "cuda_engineering_check.py", "--config", str(CONFIG),
            "--run-id", "nr", "--run-root", str(tmp_path)])
        rc = check.main()
        assert rc == 3, "absent CUDA must be NOT RUN, distinct from a failure"
        assert not list(tmp_path.rglob("report.json")), (
            "an artifact saying 'not run' eventually gets read as a result; "
            "there must be no report at all")

    def test_a_non_cuda_config_is_refused(self, check, cpu_config, tmp_path,
                                          monkeypatch):
        """The entry point validates CUDA; it must not quietly accept cpu."""
        monkeypatch.setattr("sys.argv", [
            "cuda_engineering_check.py", "--config", str(cpu_config),
            "--run-id", "x", "--run-root", str(tmp_path)])
        assert check.main() == 3

    def test_the_real_config_asks_for_cuda_and_bfloat16(self):
        doc = json.loads(CONFIG.read_text())
        assert doc["execution"]["device"] == "cuda"
        assert doc["execution"]["dtype"] == "bfloat16"


class TestTheBodyRuns:
    """Everything the device gate normally hides."""

    def test_it_completes_and_writes_a_report(self, check, monkeypatch,
                                              cpu_config, tmp_path):
        rc, report = run(check, monkeypatch, cpu_config, tmp_path)
        assert report is not None, "the run must produce its declared report role"
        assert rc in (0, 1)

    def test_every_geometry_and_operator_is_attempted(self, check, monkeypatch,
                                                      cpu_config, tmp_path):
        cfg = json.loads(cpu_config.read_text())
        _, report = run(check, monkeypatch, cpu_config, tmp_path)
        expected = len(cfg["target_geometries"]) * len(cfg["operators"])
        assert report["n_cases"] == expected, (
            "each declared operator must be attempted for each declared "
            "geometry; one geometry can pass by shape coincidence")
        assert len(report["geometries"]) >= 2, "the directive asks for two"

    def test_the_report_declares_it_is_not_science(self, check, monkeypatch,
                                                   cpu_config, tmp_path):
        _, report = run(check, monkeypatch, cpu_config, tmp_path)
        assert report["scientific_use"] is False
        assert report["authorizes"] == "nothing"

    def test_the_environment_role_is_written(self, check, monkeypatch,
                                             cpu_config, tmp_path):
        run(check, monkeypatch, cpu_config, tmp_path)
        env = tmp_path / "runs/cuda_engineering/t1/environment.json"
        assert env.is_file(), "a declared required role must exist after the run"
        assert "torch" in json.loads(env.read_text())

    def test_artifacts_stay_inside_the_run_root(self, check, monkeypatch,
                                                cpu_config, tmp_path):
        run(check, monkeypatch, cpu_config, tmp_path)
        root = (tmp_path / "runs/cuda_engineering/t1").resolve()
        written = [p for p in (tmp_path / "runs").rglob("*") if p.is_file()]
        assert written, "the run must actually write something"
        for p in written:
            assert root in p.resolve().parents, f"{p} escaped the run root"


class TestFailuresAreRecordedNotHidden:
    def test_an_unknown_operator_is_refused_before_any_device_work(
            self, check, monkeypatch, tmp_path):
        doc = json.loads(CONFIG.read_text())
        doc["execution"]["device"] = "cpu"
        doc["operators"] = [{"operator": "depth", "impl_id": "depth.does_not_exist"}]
        cfg = tmp_path / "bad.json"
        cfg.write_text(json.dumps(doc))
        rc, report = run(check, monkeypatch, cfg, tmp_path, run_id="bad")
        assert rc == 3, "a typo must cost nothing, not fail partway through"
        assert report is None

    def test_a_failing_operator_is_recorded_and_the_others_still_run(
            self, check, monkeypatch, cpu_config, tmp_path):
        """One operator raising must not hide whether the rest also fail."""
        original = check.run_one

        def explode(impl_id, *a, **k):
            if impl_id.startswith("width."):
                raise RuntimeError("synthetic device fault")
            return original(impl_id, *a, **k)

        monkeypatch.setattr(check, "run_one", explode)
        rc, report = run(check, monkeypatch, cpu_config, tmp_path, run_id="part")
        assert rc == 1, "a failing case must not report success"
        failed = [r for r in report["results"] if not r.get("applied")]
        assert failed, "the failure must appear in the report"
        assert all(r["impl_id"].startswith("width.") for r in failed), (
            "only the operator that was made to fail should have failed; "
            f"got {[r['impl_id'] for r in failed]}")
        assert all("synthetic device fault" in r["error"] for r in failed)
        applied = [r["impl_id"] for r in report["results"] if r.get("applied")]
        assert any(i.startswith("attention.") for i in applied), (
            "the operators AFTER the failing one must still have been attempted")

    def test_the_traceback_is_kept(self, check, monkeypatch, cpu_config, tmp_path):
        def explode(impl_id, *a, **k):
            raise RuntimeError("synthetic device fault")
        monkeypatch.setattr(check, "run_one", explode)
        _, report = run(check, monkeypatch, cpu_config, tmp_path, run_id="tb")
        assert all("traceback" in r for r in report["results"])
