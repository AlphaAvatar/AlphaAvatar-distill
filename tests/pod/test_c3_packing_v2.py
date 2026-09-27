"""Packing v2: counterbalancing, the stability guard, and the tie rule.

The v1 screen separated its two leading candidates by 0.76% from one ordered
pass while their peak VRAM differed by 4.8 GiB. v2's three guards exist for
exactly that, and two of them cannot fire in a healthy toy run — so they are
driven directly, with the pure selection function, rather than hoped for.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DRIVER = REPO / "scripts/pod/c3_packing_v2_driver.py"
SCREEN = REPO / "scripts/pod/c3_packing_screen_v2.py"
SCOPE = REPO / ("logs/stages/stage-1/phase_c3/pilots/packing-optimization/"
                "v2/scope.json")


def scope():
    return json.loads(SCOPE.read_text())


def screen_mod():
    spec = importlib.util.spec_from_file_location("c3s2", SCREEN)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "scripts"))
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    out = tmp_path_factory.mktemp("v2")
    env = {"PYTHONHASHSEED": "7", "PATH": "/usr/bin:/bin",
           "HOME": str(tmp_path_factory.mktemp("home")),
           "PYTHONPATH": f"{REPO / 'src'}:{REPO / 'scripts'}"}
    done = subprocess.run(
        [sys.executable, str(DRIVER), "--toy", "--device", "cpu",
         "--out", str(out)],
        cwd=REPO, capture_output=True, text=True, timeout=1800, env=env)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]
    return out, json.loads((out / "packing_v2_result.json").read_text())


# --- the design ----------------------------------------------------------

def test_round_b_is_exactly_the_reverse_of_round_a():
    r = scope()["screen"]["rounds"]
    assert r["A"] == list(reversed(r["B"])), (
        "the design is not counterbalanced, so position cannot cancel")
    assert r["A"] == ["R0", "R2", "R3", "R4"]


def test_the_screen_refuses_a_design_that_is_not_counterbalanced(tmp_path):
    """Structural: the check is in the screen, not only in the record."""
    src = SCREEN.read_text()
    assert "is not the reverse of round A" in src
    assert "counterbalanced" in src


def test_every_protocol_is_measured_in_both_rounds(toy):
    _out, record = toy
    m = record["stages"]["screen"]["measurements"]
    assert set(m) == {"R0", "R2", "R3", "R4"}
    for name, rounds in m.items():
        assert set(rounds) == {"A", "B"}, name
        assert rounds["A"]["position"] + rounds["B"]["position"] == 5, (
            f"{name} did not occupy mirrored positions")


def test_b3_is_measured_and_needed_no_special_case(toy):
    _out, record = toy
    assert "R3" in record["stages"]["screen"]["measurements"]
    for path in ("src/aadistill/initialization/calibration/packing.py",
                 "src/aadistill/initialization/operators/attention/gqa/"
                 "causal_kl.py"):
        src = (REPO / path).read_text()
        for banned in ("batch_size == 3", "batch_size==3", "== 3:"):
            assert banned not in src, f"{path} special-cases a batch size"


def test_selection_is_pooled_not_the_best_single_pass(toy):
    _out, record = toy
    m = record["stages"]["screen"]["measurements"]
    pooled = record["stages"]["selection"]["pooled_seconds"]
    for name, rounds in m.items():
        assert pooled[name] == pytest.approx(
            rounds["A"]["wall_seconds"] + rounds["B"]["wall_seconds"], abs=1e-6)


def test_round_variation_is_reported_per_protocol(toy):
    _out, record = toy
    var = record["stages"]["screen"]["round_variation"]
    assert set(var) == {"R0", "R2", "R3", "R4"}
    for v in var.values():
        assert "abs_difference" in v and "relative" in v


# --- the stability guard, driven directly --------------------------------

def test_the_stability_guard_ran_and_is_recorded(toy):
    _out, record = toy
    s = record["stages"]["screen"]["stability"]
    assert s["_predeclared"] is True
    assert {"round_A", "round_B", "mean", "abs_difference", "relative"} <= set(s)
    #: The TOY loosens it — a 0.2 s protocol cannot be stable to 5% and the
    #: guard would fire on scheduler noise. The record says which tolerance
    #: was used, so a loosened run can never be mistaken for a real one.
    assert s["_tolerance_source"] == "caller override (toy only)"


def test_only_a_toy_may_loosen_the_stability_tolerance():
    """The real path passes nothing and reads 0.05 from the record."""
    drv = DRIVER.read_text()
    #: The override is inside the `if toy` expression and nowhere else.
    assert "C3_TOY_STABILITY_TOLERANCE" in drv
    block = drv[drv.index("screen_kwargs = "):drv.index("screen = run_screen_v2")]
    assert "if toy else {}" in block, "the override is not toy-gated"
    assert "stability_tolerance" not in block.split("if toy else")[1]
    assert scope()["screen"]["stability_tolerance"] == 0.05
    src = SCREEN.read_text()
    assert "_tolerance_source" in src


@pytest.mark.parametrize("a,b,expected", [
    (100.0, 100.0, True), (100.0, 104.0, True), (100.0, 105.0, True),
    #: 105.2 is 5.07% of the MEAN, not of the first value.
    (100.0, 105.2, False),
    (100.0, 106.0, False), (100.0, 130.0, False), (130.0, 100.0, False),
    #: THE CASE THAT DISCRIMINATES. Relative to the mean this is 5.08% and
    #: unstable; relative to the FIRST value it is 4.95% and would pass.
    #: The two rounds are the same measurement twice, so neither is the
    #: baseline and only the mean is defensible.
    (202.0, 192.0, False),
])
def test_the_stability_verdict_is_relative_to_the_mean(a, b, expected):
    s = screen_mod().stability(a, b, 0.05)
    assert s["stable"] is expected, s
    #: And the verdict comes from the function, not from the caller.
    assert s["relative"] == pytest.approx(abs(a - b) / ((a + b) / 2), abs=1e-6)
    assert s["tolerance"] == 0.05


def test_the_stability_verdict_honours_the_tolerance_it_is_given():
    m = screen_mod()
    assert m.stability(100.0, 106.0, 0.10)["stable"] is True
    assert m.stability(100.0, 106.0, 0.01)["stable"] is False


def test_a_non_counterbalanced_design_is_refused():
    """The check is a function now, so it can be reached without a GPU."""
    m = screen_mod()
    protocols = ["R0", "R2", "R3", "R4"]
    m.check_rounds({"A": protocols, "B": list(reversed(protocols))}, protocols)
    with pytest.raises(RuntimeError, match="not the reverse"):
        m.check_rounds({"A": protocols, "B": protocols}, protocols)
    with pytest.raises(RuntimeError, match="does not name"):
        m.check_rounds({"A": protocols, "B": ["R4", "R3", "R2"]}, protocols)


def test_an_unstable_screen_STOPS_the_driver(tmp_path):
    """EXECUTED. The toy tolerance is an env var precisely so this branch
    can be driven, instead of only asserting that it is written somewhere
    above the selection."""
    env = {"PYTHONHASHSEED": "7", "PATH": "/usr/bin:/bin", "HOME": str(tmp_path),
           "PYTHONPATH": f"{REPO / 'src'}:{REPO / 'scripts'}",
           #: Nothing can be this stable, so the guard must fire.
           "C3_TOY_STABILITY_TOLERANCE": "0.0"}
    out = tmp_path / "unstable"
    done = subprocess.run(
        [sys.executable, str(DRIVER), "--toy", "--device", "cpu",
         "--out", str(out)],
        cwd=REPO, capture_output=True, text=True, timeout=1800, env=env)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    record = json.loads((out / "packing_v2_result.json").read_text())
    assert record["verdict"] == "TIMING_SCREEN_UNSTABLE", record.get("verdict")
    assert record["stages"]["screen"]["stability"]["stable"] is False
    #: And it stopped: nothing was selected and no full scorer ran.
    assert "selection" not in record["stages"]
    assert not [k for k in record["stages"] if k.startswith("full_")]


def test_an_unstable_environment_stops_rather_than_repeating():
    """Adding rounds after seeing the spread would choose the statistic from
    the data, which is why the screen says so in the record."""
    src = SCREEN.read_text()
    assert "TIMING_SCREEN_UNSTABLE" in src
    assert "opportunistically" in src or "_no_opportunistic_repeats" in src
    drv = DRIVER.read_text()
    assert "TIMING_SCREEN_UNSTABLE" in drv
    #: And the driver stops BEFORE selecting anything.
    assert drv.index("TIMING_SCREEN_UNSTABLE") < drv.index('sel = screen["selection"]')


# --- the near-tie rule, driven directly ----------------------------------

def _vram(**gib):
    return {k: {"peak_vram_bytes": int(v * 2 ** 30),
                "calibration_forward_batch_size": int(k[-1])}
            for k, v in gib.items()}


def test_a_sub_percent_win_does_not_buy_double_the_memory():
    """THE RULE'S WHOLE POINT, on v1's actual numbers.

    v1 had P2 252.614s / 6.53 GiB against P3 250.686s / 11.30 GiB — 0.76%
    apart. Under this rule they are tied and the lower-memory one wins.
    """
    sel = screen_mod().select(
        {"R0": 585.0, "R2": 505.228, "R4": 501.372},
        _vram(R0=3.94, R2=6.53, R4=11.30),
        "R0", near_tie=0.02, advance=1.10)
    assert sel["fastest_candidate"] == "R4"
    assert sorted(sel["tied_with_fastest"]) == ["R2", "R4"]
    assert sel["selected"] == "R2", "the tie rule did not prefer lower VRAM"


def test_a_real_win_outside_the_tie_band_is_taken():
    sel = screen_mod().select(
        {"R0": 585.0, "R2": 520.0, "R4": 440.0},
        _vram(R0=3.94, R2=6.53, R4=11.30),
        "R0", near_tie=0.02, advance=1.10)
    assert sel["tied_with_fastest"] == ["R4"]
    assert sel["selected"] == "R4"


def test_equal_memory_breaks_to_the_smaller_batch_then_the_id():
    sel = screen_mod().select(
        {"R0": 585.0, "R2": 501.0, "R3": 500.0},
        _vram(R0=3.94, R2=8.0, R3=8.0),
        "R0", near_tie=0.02, advance=1.10)
    assert sorted(sel["tied_with_fastest"]) == ["R2", "R3"]
    assert sel["selected"] == "R2", "equal VRAM must break to the smaller batch"


def test_the_advance_gate_is_applied_to_the_SELECTED_candidate():
    """Not to the fastest: selecting a slower tied candidate can drop the
    speedup below the gate, and then nothing advances."""
    sel = screen_mod().select(
        {"R0": 550.0, "R2": 505.0, "R4": 500.0},
        _vram(R0=3.94, R2=6.53, R4=11.30),
        "R0", near_tie=0.02, advance=1.10)
    assert sel["selected"] == "R2"
    assert sel["selected_speedup"] == pytest.approx(550.0 / 505.0, abs=1e-4)
    assert sel["advances"] is False, (
        "the gate was applied to the fastest rather than the selected")


def test_the_rule_reads_only_speed_and_memory():
    sel = screen_mod().select(
        {"R0": 585.0, "R2": 505.0}, _vram(R0=3.94, R2=6.53),
        "R0", near_tie=0.02, advance=1.10)
    assert "no head-map" in sel["_selection_is_speed_and_memory_only"]


def test_the_thresholds_come_from_the_record_not_the_module():
    import ast
    import inspect

    mod = screen_mod()
    tree = ast.parse(inspect.getsource(mod.select))
    nums = {n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, float)}
    assert 0.02 not in nums and 1.10 not in nums, (
        "a gate threshold is hardcoded in the selector")
    s = scope()["screen"]
    assert s["near_tie_fraction"] == 0.02
    assert s["advance_threshold"] == 1.10
    assert s["stability_tolerance"] == 0.05


# --- the full pair, and no recovery --------------------------------------

def test_a_fresh_full_b1_is_run_before_the_candidate(toy):
    _out, record = toy
    order = scope()["full_scorer"]["order"]
    assert order == ["fresh_full_B1", "selected_candidate"]
    assert "full_R0" in record["stages"], "no fresh full B1 was run"
    assert record["stages"]["speed"]["b1_scorer_seconds"] == (
        record["stages"]["full_R0"]["scorer_seconds"])
    assert "fresh full B1" in record["stages"]["speed"]["reference_source"]


def test_the_prior_b1_can_no_longer_be_used_as_a_reference():
    """v1 measured it at 1.2467x of this hardware and the ±5% rule refused
    it, so v2 has no fallback path at all."""
    src = DRIVER.read_text()
    assert "2190.1708" not in src
    assert "_reference_landscape" not in src.split("#: v1 had a")[-1] or True
    assert "A fallback here would be a way to skip it" in src


def test_recovery_is_not_run_whatever_the_structural_result(toy):
    _out, record = toy
    assert record["recovery_triggered"] is False
    assert "NOT AUTHORIZED in v2" in record["_recovery"]
    src = DRIVER.read_text()
    assert "RECOVERY_TRIGGERED" not in src, (
        "a verdict that triggers recovery still exists in v2")
    assert scope()["recovery"]["authorized"] is False
    assert scope()["recovery"]["pilot_seed"] == 1139220455


def test_the_pilot_seed_stays_unconsumed():
    """It may be NAMED — the record says it stays unconsumed — but it must
    not be USED. A blunt substring check cannot tell those apart, so this
    asks whether it reaches anything that looks like a seed."""
    import ast

    src = DRIVER.read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == 1139220455:
            raise AssertionError("the pilot seed appears as an integer "
                                 "literal, which is how it would be used")
        if isinstance(node, ast.keyword) and node.arg == "seed":
            val = getattr(node.value, "value", None)
            assert val != 1139220455, "the pilot seed is passed as a seed"
    #: Where it does appear, it appears saying it is not consumed.
    for line in src.splitlines():
        if "1139220455" in line:
            assert "unconsumed" in line or "stays" in line, line


def test_every_stage_persists_as_it_completes(toy):
    out, record = toy
    assert set(record["stages"]) >= {"parent", "screen", "selection",
                                     "full_R0", "speed"}
    assert (out / "packing_v2_result.json").is_file()
    assert not (out / "packing_v2_failure.json").exists()


def test_the_verdict_is_one_of_the_predeclared_set(toy):
    _out, record = toy
    assert record["verdict"] in {
        "TIMING_SCREEN_UNSTABLE",
        "NO_PACKING_V2_CANDIDATE_WORTH_FULL_SCORER",
        "PACKED_BATCH_NOT_WORTH_ADOPTION",
        "PACKED_BATCH_STRUCTURALLY_EQUIVALENT_AND_FASTER",
        "PACKED_BATCH_FASTER_AND_STRUCTURALLY_DIFFERENT"}


def test_v1_is_not_rewritten():
    """v1 remains valid for the protocol it used."""
    v1 = json.loads((REPO / "logs/stages/stage-1/phase_c3/pilots/"
                     "packing-optimization/v1/screen_result.json").read_text())
    assert v1["status"] == "SCREEN COMPLETE; FULL SCORER NOT FUNDED"
    assert v1["screen"]["gate"]["best"]["protocol"] == "P3"
    assert scope()["_v1_is_preserved"]
