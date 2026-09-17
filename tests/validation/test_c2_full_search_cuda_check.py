"""The C2 full-search CUDA check, EXECUTED at $0 on CPU.

This file is not a substitute for the GPU validation and cannot be: the whole
point of that run is `--device cuda`, and a CPU execution is the one
substitution that has previously hidden a device-placement bug here. What this
does is execute the CHECK SCRIPT's own lines so a paid pod does not die in a
typo -- which is how four paid pods in this repository have died.

It has already paid for itself. Executing these stages found three real defects
before any pod existed:

* `size_report(REPO)["leaves"]` -- a key that does not exist, in the LAST
  statement of stage D, after the whole search had succeeded;
* stage M resolving an adapter from an empty registry, because stage D happened
  to reach one through the driver's import chain and stage M does not;
* `int(exc.code)` raising ValueError on the string-coded refusal, so the line
  meant to REPORT a refusal crashed instead of reporting it.

Two more came from the pod itself, one per producer of non-source input, and
both are now closed by a derivation rather than by a longer list:

* a1 ($0.0299) died in stage A -- the cost table pools over committed telemetry
  and none of `logs/` was shipped;
* a2 ($0.0208) got stage A to PASS with the space derived as 578 leaves, then
  died in stage B -- the two real calibration mixtures resolve item files in the
  out-of-tree artifact store, which a list naming only the first producer could
  not have covered.

Every one of those is in a line a $0 run reaches and no static check does.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CHECK = REPO / "scripts/validation/c2_full_search_cuda_check.py"
CONFIG = REPO / "configs/validation/c2_full_search_cuda.json"

for extra in ("src", "scripts", "scripts/autoinit", "scripts/pod"):
    if str(REPO / extra) not in sys.path:
        sys.path.insert(0, str(REPO / extra))


@pytest.fixture(scope="module")
def check():
    spec = importlib.util.spec_from_file_location("c2_cuda_check", CHECK)
    module = importlib.util.module_from_spec(spec)
    sys.modules["c2_cuda_check"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cfg():
    return json.loads(CONFIG.read_text())


# --- the stages, executed ---------------------------------------------------

@pytest.fixture(scope="module")
def driver_result(check, cfg, tmp_path_factory):
    """Stage D, really run: real driver, real joint space, real BeamSearch."""
    return check.stage_driver(cfg, "cpu", "float32",
                              tmp_path_factory.mktemp("stage-d"))


def test_stage_d_drives_the_real_driver_to_a_zero_return(driver_result):
    assert driver_result["driver_returncode"] == 0


def test_stage_d_materializes_states_and_reloads_every_one(driver_result):
    """A vacuous pass is the failure mode: no states means nothing was placed."""
    assert driver_result["state_dirs"] > 0
    errored = [p for p in driver_result["placements"] if "error" in p]
    assert not errored, f"states failed to reload: {errored}"
    assert len(driver_result["placements"]) == driver_result["state_dirs"]


def test_stage_d_derives_the_joint_space_rather_than_declaring_it(driver_result):
    """578 = 576 decomposed + 2 single-step COMPOSITE, from the registry.

    Read through `size_report`, so a wrong key fails here instead of on a pod.
    """
    assert driver_result["space_leaves"] == 578


def test_stage_d_observes_the_space_arguments_the_driver_forwarded(driver_result):
    fwd = driver_result["forwarded"]
    #: NONE, both of them. A pinned impl->profile mapping is Search-1's
    #: restriction and a conditional candidate is its baseline fallback; the
    #: joint re-search has neither, and these are the arguments that say so.
    assert fwd["impl_profiles"] is None
    assert fwd["conditional_candidates"] is None
    assert fwd["top_n"] == 5
    assert fwd["run_id"] == "autoinit.v1.phase_c2.full_search"
    assert len(fwd["allowed_impls"]) == 6
    assert fwd["profiles"] == ["calib.domain_balanced@v1",
                               "calib.reasoning_heavy@v2"]


def test_stage_d_asserts_the_device_it_was_given_rather_than_a_default():
    """The assertion is in the wrapper, and it is the reason this file exists.

    The driver's CPU test overrides `device` to `"cpu"`; this check must not,
    or the GPU run would certify a host-resident search. Verified by source,
    because a CPU execution cannot distinguish "forwarded cuda" from
    "forwarded cpu" -- only that the forwarded value is what was asked for.
    """
    body = CHECK.read_text()
    assert 'assert kwargs.get("device") == device' in body
    wrapped = body.split("def wrapped(**kwargs):", 1)[1].split("\n    phase_a", 1)[0]
    assert '"device"' not in wrapped.split("return real_search", 1)[1], (
        "the wrapper overrides `device` in the substituted kwargs, which is "
        "exactly the substitution this validation exists to avoid")


@pytest.fixture(scope="module")
def geometry_result(check, cfg, tmp_path_factory):
    """Stage M, really run, at the REAL 1024x28 target geometry."""
    return check.stage_geometry(cfg, "cpu", "bfloat16",
                                tmp_path_factory.mktemp("stage-m"))


def test_stage_m_builds_the_real_student_geometry(geometry_result):
    assert geometry_result["parameters"] == 596_049_920
    assert geometry_result["target_geometry"]["hidden_size"] == 1024
    assert geometry_result["target_geometry"]["num_hidden_layers"] == 28


def test_stage_m_survives_a_canonical_reload_unchanged(geometry_result):
    assert (geometry_result["parameters_after_reload"]
            == geometry_result["parameters"])
    assert geometry_result["reloaded_placement"]["dtypes"] == ["torch.bfloat16"]
    assert geometry_result["tie_word_embeddings"] is True


def test_stage_m_reads_the_rope_base_through_both_field_layouts(geometry_result):
    """transformers moved `rope_theta` into a nested `rope_parameters` dict.

    On 5.x a flat `config.rope_theta` read RAISES, and across a mixed pair it
    silently returns the 4.x class default -- the 500x misread that reports a
    wrong NLL instead of failing. The repo's `stored_rope_base` handles both,
    and this asserts the value survives build -> save -> reload.
    """
    assert geometry_result["rope_theta_from_the_loaded_model"] == 5_000_000.0
    assert (geometry_result["rope_theta_from_the_loaded_model"]
            == geometry_result["rope_theta_from_the_teacher_config"])


def test_stage_m_used_the_config_and_not_the_teacher_weights(geometry_result):
    """What keeps this validation cheap, asserted rather than assumed."""
    assert geometry_result["teacher_id"] == "Qwen/Qwen3-4B-Thinking-2507"
    assert geometry_result["teacher_revision"] == (
        "768f209d9ea81521153ed38c47d515654e938aea")
    assert geometry_result["_weights_not_downloaded"] == "the pinned CONFIG only"


def test_stage_m_runs_ALONE_in_a_fresh_process(tmp_path):
    """The registry check that cannot be made in this process.

    Mutation-verified the hard way, twice. Deleting the module-level
    `register_builtin_adapters()` call left every test above GREEN, because by
    the time stage M runs there the adapter registry has already been filled --
    by stage D's import of the driver, or by any sibling test in the session.
    The registries are process-global, so a test sharing a process with
    something that populates them can never detect a missing registration.

    Moving to a fresh process, stage D DISABLED, is also the real configuration
    under which the defect bites: a check that dies in `get_adapter` before
    building anything, reporting what looks like a missing model rather than a
    missing call.

    **What this does NOT pin.** Removing any ONE of the three module-level
    registration calls still passes, even here -- measured, not assumed. All
    three import chains transitively reach `register_builtin_adapters`, so
    dropping one is a no-op mutation and a test claiming otherwise would be
    claiming coverage it does not have. Removing all three DOES fail. That is
    the property worth holding: the check must not depend on some other stage,
    test or import having registered things for it. The calls stay explicit
    anyway, for the reason the driver gives for its own -- relying on a
    transitive import to populate a registry is how a paid pod was lost, and
    "somebody else probably called it" is not a contract.
    """
    import os
    import subprocess

    config = json.loads(CONFIG.read_text())
    config["stages"]["driver"]["enabled"] = False
    #: bf16 on CPU, and the capability floor cannot be met without a GPU, so
    #: this drives the stage function directly rather than `main`. What matters
    #: is that the process has imported NOTHING else first.
    local = tmp_path / "geometry_only.json"
    local.write_text(json.dumps(config))

    program = (
        "import json, sys, importlib.util\n"
        f"sys.path[:0] = {[str(REPO / e) for e in ('src', 'scripts', 'scripts/autoinit', 'scripts/pod')]!r}\n"
        f"spec = importlib.util.spec_from_file_location('c', {str(CHECK)!r})\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "sys.modules['c'] = m\n"
        "spec.loader.exec_module(m)\n"
        f"cfg = json.loads(open({str(local)!r}).read())\n"
        f"out = m.stage_geometry(cfg, 'cpu', 'bfloat16', __import__('pathlib').Path({str(tmp_path / 'w')!r}))\n"
        "print('PARAMS', out['parameters'])\n"
    )
    r = subprocess.run([sys.executable, "-c", program], capture_output=True,
                       text=True, timeout=1800,
                       env={**os.environ, "HF_HUB_OFFLINE": "1"})
    assert r.returncode == 0, (
        "stage M cannot run in a fresh process, so it depends on another "
        f"stage or another test having registered something for it:\n"
        f"{r.stdout[-2000:]}\n{r.stderr[-3000:]}")
    assert "PARAMS 596049920" in r.stdout, r.stdout[-2000:]


# --- the inputs the driver reads and is not source --------------------------

def test_the_declared_inputs_cover_BOTH_producers_the_driver_reads_from():
    """Derived from code, not typed beside it -- because a list is per-producer.

    The targeted regression for two observed pod failures, one per producer:

    * a1 ($0.0299) reached a real L40S, passed the capability floor and died in
      stage A: `full_search_space` pools its cost table over the committed
      telemetry and correctly refused to price from a partial history when none
      of `logs/` was shipped;
    * a2 ($0.0208) got stage A to PASS with the space derived as 578 leaves,
      then died in stage B: the two real calibration mixtures resolve item
      files in the out-of-tree artifact store, and I had declared the first
      producer's inputs and not the second's.

    So the assertion is about the DERIVATION covering both, which closes the
    class. A hand-written list would have been corrected twice and could still
    be short a third time.
    """
    from aadistill.initialization.calibration.profiles import get_profile
    from experiments.phase_c2 import full_search_space as FS

    import importlib.util
    spec = importlib.util.spec_from_file_location("c2_inputs", CHECK)
    module = importlib.util.module_from_spec(spec)
    sys.modules["c2_inputs"] = module
    spec.loader.exec_module(module)
    derived = module.declared_inputs()

    expected = []
    for _name, telemetry, result in FS.TELEMETRY_SOURCES:
        expected.append(telemetry)
        if result:
            expected.append(result)
    for qualified in FS.PROFILE_IDS:
        expected.append(str(get_profile(qualified).items_path))

    assert sorted(derived) == sorted(dict.fromkeys(expected))
    #: BOTH producers are represented. Either alone is one of the two failures.
    assert any(x.startswith("logs/") for x in derived), "no telemetry input"
    assert any(x.startswith("artifacts/") for x in derived), (
        "no calibration item file -- this is the a2 failure")
    for rel in derived:
        assert (REPO / rel).is_file(), rel

    #: And the config must NOT carry a competing list.
    assert "paths" not in json.loads(CONFIG.read_text())["required_repo_inputs"], (
        "a path list beside the derivation is a second owner of one fact, and "
        "it is what was short twice")


def test_a_missing_declared_input_refuses_BEFORE_cuda(check, cfg, tmp_path,
                                                      monkeypatch):
    """And it refuses with the CAUSE, not with a search-space error.

    On the pod the missing file surfaced as `FullSearchSpaceError: ... refusing
    to price from a partial history` inside stage A -- true, but it sent the
    diagnosis looking at the cost model rather than at the ship set.
    """
    broken = {**cfg, "required_repo_inputs": {
        "paths": ["logs/stages/stage-1/phase_b/runs/attempt5/nope.jsonl"]}}
    with pytest.raises(AssertionError, match="declared repository input"):
        check.require_repo_inputs(broken)


def test_a_missing_input_still_writes_a_report_rather_than_a_traceback(
        check, tmp_path, monkeypatch):
    """A launcher reads the report. A refusal that never reaches it looks like
    a crash instead of a missing file.

    This branch was genuinely absent when the input check was added: `main`
    caught only `SystemExit`, so the AssertionError would have escaped.
    """
    config = json.loads(CONFIG.read_text())
    config["required_repo_inputs"]["paths"] = ["logs/definitely/absent.jsonl"]
    local = tmp_path / "cfg.json"
    local.write_text(json.dumps(config))
    monkeypatch.setattr(sys, "argv", [
        "check", "--run-id", "missing-input", "--config", str(local),
        "--out", str(tmp_path / "out")])
    assert check.main() == check.FAILED
    doc = json.loads((tmp_path / "out"
                      / "c2_full_search_cuda_report.json").read_text())
    assert doc["verdict"] == "FAIL"
    assert "declared repository input" in doc["reason"]
    #: And it did NOT get as far as reporting a device.
    assert "device" not in doc


# --- the refusals -----------------------------------------------------------

def test_a_non_cuda_request_is_a_failure_not_a_quiet_cpu_run(check, tmp_path,
                                                             monkeypatch):
    """And it REPORTS, rather than raising inside the reporting line."""
    monkeypatch.setattr(sys, "argv", [
        "check", "--run-id", "refusal", "--device", "cpu",
        "--out", str(tmp_path / "out")])
    assert check.main() == check.FAILED
    doc = json.loads((tmp_path / "out"
                      / "c2_full_search_cuda_report.json").read_text())
    assert doc["verdict"] == "FAIL"
    assert "only meaningful on cuda" in doc["reason"]
    assert doc["authorizes"] == "nothing"
    assert doc["scientific_use"] is False


def test_no_cuda_device_is_NOT_RUN_rather_than_a_pass(check, tmp_path,
                                                      monkeypatch):
    """An environment that cannot host the check has not failed it.

    Skipped where CUDA exists: there the branch is unreachable, and forcing it
    would test a monkeypatch rather than the check.
    """
    import torch

    if torch.cuda.is_available():
        pytest.skip("this box has CUDA, so the NOT RUN branch cannot be reached")
    monkeypatch.setattr(sys, "argv", [
        "check", "--run-id", "notrun", "--out", str(tmp_path / "out")])
    assert check.main() == check.NOT_RUN
    doc = json.loads((tmp_path / "out"
                      / "c2_full_search_cuda_report.json").read_text())
    assert doc["verdict"] == "NOT RUN"


def test_the_capability_floor_refuses_below_it_and_does_not_lower_the_dtype(
        check, cfg):
    """bf16 and cc>=8.0 are the formal search's configuration, not a preference.

    A check that fell back to fp16 on an older card would validate something
    the search will not run.
    """
    need = cfg["capability_requirement"]
    assert need["compute_capability_min"] == "8.0"
    assert need["native_bf16"] is True
    assert cfg["dtype"] == "bfloat16"

    ok = {"capability": "8.9", "bf16_supported": True, "free_gib": 44.0}
    check.require_capability(ok, need)              # must not raise

    for bad, why in (
            ({**ok, "capability": "7.5"}, "capability"),
            ({**ok, "bf16_supported": False}, "bf16"),
            ({**ok, "free_gib": 1.0}, "GiB")):
        with pytest.raises(AssertionError, match=why):
            check.require_capability(bad, need)


# --- what it must never do --------------------------------------------------

def test_the_check_trains_nothing_and_measures_no_behaviour():
    """Its verdict is engineering evidence. A behavioural path here would let a
    search-authorized session produce a promotion signal."""
    body = CHECK.read_text().lower()
    for forbidden in ("correct_overall", "usable_rollout", "screening_battery",
                      "confirmation_battery", "recovery_probe"):
        #: Mentioned in prose is fine; imported or called is not.
        assert f"import {forbidden}" not in body
        assert f"{forbidden}(" not in body
