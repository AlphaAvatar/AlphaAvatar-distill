"""The packing-optimization pilot, EXECUTED at toy scale.

Same seven steps the pod runs: parent, verify, three-layer screen, screen
gate, the +/-5% comparability rule, one full scorer, the adoption gate. Same
`score_heads`, same comparator, same committed records.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DRIVER = REPO / "scripts/pod/c3_packing_pilot_driver.py"
SCREEN = REPO / "scripts/pod/c3_packing_screen.py"
SCOPE = REPO / ("logs/stages/stage-1/phase_c3/pilots/packing-optimization/"
                "v1/scope.json")


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    out = tmp_path_factory.mktemp("packing")
    env = {"PYTHONHASHSEED": "7", "PATH": "/usr/bin:/bin",
           "HOME": str(tmp_path_factory.mktemp("home")),
           "PYTHONPATH": f"{REPO / 'src'}:{REPO / 'scripts'}"}
    done = subprocess.run(
        [sys.executable, str(DRIVER), "--toy", "--device", "cpu",
         "--out", str(out)],
        cwd=REPO, capture_output=True, text=True, timeout=1800, env=env)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]
    return out, json.loads((out / "packing_result.json").read_text())


def scope():
    return json.loads(SCOPE.read_text())


def test_the_whole_sequence_runs_in_a_bare_interpreter(toy):
    _out, record = toy
    assert record["verdict"]
    assert set(record["stages"]) >= {"parent", "screen", "screen_gate",
                                     "b1_comparability"}


def test_the_parent_is_obtained_once_and_verified(toy):
    _out, record = toy
    assert record["stages"]["parent"]["replay_count"] == 1
    assert record["stages"]["parent"]["source"] == "replayed"
    assert record["stages"]["parent"]["artifact_digest"]


def test_the_screen_covers_every_protocol_in_the_predeclared_order(toy):
    _out, record = toy
    s = record["stages"]["screen"]
    assert s["protocol_order"] == scope()["screen"]["protocol_order"]
    assert set(s["results"]) == set(s["protocol_order"])
    for i, name in enumerate(s["protocol_order"], start=1):
        assert s["results"][name]["position"] == i


def test_every_protocol_is_warmed_identically(toy):
    _out, record = toy
    for r in record["stages"]["screen"]["results"].values():
        assert r["warmup"]["warmup_forwards"] >= 1


def test_the_screen_reports_the_metrics_the_return_owes(toy):
    _out, record = toy
    for r in record["stages"]["screen"]["results"].values():
        for field in ("wall_seconds", "physical_forward_invocations",
                      "invocations_per_minute", "valid_positions_per_second",
                      "executed_positions", "executed_padded_positions",
                      "padding_over_valid_full_corpus", "peak_vram_bytes",
                      "n_groups"):
            assert field in r, f"{r['protocol']} does not report {field}"


def test_b1_issues_the_most_invocations_and_pads_nothing(toy):
    _out, record = toy
    res = record["stages"]["screen"]["results"]
    p0 = res["P0"]
    assert all(p0["physical_forward_invocations"]
               >= r["physical_forward_invocations"] for r in res.values())
    assert p0["padding_over_valid_full_corpus"] == 0.0


def test_selection_is_by_wall_time_alone(toy):
    """No head-map or quality result chooses which packing advances."""
    _out, record = toy
    gate = record["stages"]["screen_gate"]
    res = record["stages"]["screen"]["results"]
    ranked = sorted((n for n in res if n != gate["reference_protocol"]),
                    key=lambda n: res[n]["wall_seconds"])
    assert [c["protocol"] for c in gate["ranked"]] == ranked
    assert gate["best"]["protocol"] == ranked[0]
    assert "wall time ALONE" in gate["_selection"]


def test_the_screen_gate_threshold_comes_from_the_record(toy):
    _out, record = toy
    assert record["stages"]["screen_gate"]["advance_threshold"] == (
        scope()["screen"]["advance_threshold"]) == 1.10


def test_the_comparability_rule_is_applied_and_predeclared(toy):
    _out, record = toy
    c = record["stages"]["b1_comparability"]
    assert c["_predeclared"] is True
    assert c["tolerance"] == scope()["b1_comparability"]["tolerance"] == 0.05
    assert c["prior_full_b1_seconds"] == 2190.1708
    assert c["full_layers"] == 28
    assert isinstance(c["comparable"], bool)
    #: A toy geometry cannot be comparable to an L40S full run, and the rule
    #: must therefore have demanded a fresh reference.
    assert c["comparable"] is False
    assert "full_P0" in record["stages"], (
        "not comparable, yet no fresh B1 was run")


def test_the_full_gate_uses_the_reference_the_rule_selected(toy):
    _out, record = toy
    sp = record["stages"]["speed"]
    assert sp["b1_scorer_seconds"] == record["stages"]["full_P0"][
        "scorer_seconds"]
    assert "this session" in sp["reference_source"]


def test_the_verdict_is_one_of_the_predeclared_set(toy):
    _out, record = toy
    assert record["verdict"] in {
        "NO_BATCH_PACKING_CANDIDATE_WORTH_FULL_SCORER",
        "PACKED_BATCH_NOT_WORTH_ADOPTION",
        "PACKED_BATCH_STRUCTURALLY_EQUIVALENT_AND_FASTER",
        "PACKED_BATCH_FASTER_AND_STRUCTURALLY_DIFFERENT_RECOVERY_TRIGGERED"}


def test_the_full_scorer_runs_the_real_operator(toy):
    _out, record = toy
    full = record["stages"]["full_P3"]
    assert full["kept_heads"] and full["causal_head_evidence"]
    assert full["path_hash"] and full["path_id"]
    assert full["calibration_batch_packing"] == "length_sorted_v1"


def test_the_screen_cannot_produce_a_head_map(toy):
    """A three-layer landscape is incomplete by construction."""
    _out, record = toy
    assert "kept_heads" not in json.dumps(record["stages"]["screen"])


# --- the real path must not be the overridden one ------------------------

def test_the_driver_overrides_nothing_when_it_is_not_a_toy():
    """`items` and `layers` exist for the toy geometry. A pod must take the
    branch that resolves the mixture and reads the record."""
    src = DRIVER.read_text()
    assert 'screen_kwargs = {"items": items, "layers": [0, 1]} if toy else {}' \
        in src, "the screen override is no longer toy-only"


def test_the_screen_resolves_and_PREPARES_the_mixture():
    """It went straight from `resolve` to `item_lengths` and died on
    `item 0 has no input_ids to measure`."""
    src = SCREEN.read_text()
    assert "prepare_calibration_items" in src
    assert "def resolve_items" in src


@pytest.mark.skipif(not (REPO / "artifacts/stage1").is_dir(),
                    reason="artifacts/ is gitignored and not built here")
def test_the_real_resolve_path_yields_prepared_items():
    """Executed against the REAL frozen mixture, because the toy run takes
    the other branch and would never notice this breaking."""
    import importlib.util

    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "scripts"))
    from experiments.calibration import register_builtin_profiles

    register_builtin_profiles()
    spec = importlib.util.spec_from_file_location("c3screen", SCREEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    items = mod.resolve_items(REPO, "calib.domain_balanced@v1")
    assert len(items) == 67
    assert all("input_ids" in i for i in items)
    from aadistill.initialization.calibration.packing import item_lengths

    assert sum(item_lengths(items)) == 59_830


def test_the_screen_layers_are_predeclared_and_span_the_depth():
    s = scope()
    assert s["screen"]["layers"] == [0, 13, 27]
    assert max(s["screen"]["layers"]) == s["_full_layers"] - 1


def test_both_drivers_answer_the_input_modes_their_launchers_call():
    """The packing launcher asked its own driver for `--required-inputs`,
    got an argparse error, and refused to create a pod. Correct — and only
    because the refusal was fail-closed. Both drivers now delegate to the
    pilot module, which owns the fact."""
    env = {"PYTHONHASHSEED": "7", "PATH": "/usr/bin:/bin",
           "PYTHONPATH": f"{REPO / 'src'}:{REPO / 'scripts'}"}
    rows = {}
    for name in ("c3_batching_pilot_driver.py", "c3_packing_pilot_driver.py"):
        done = subprocess.run(
            [sys.executable, str(REPO / "scripts/pod" / name),
             "--required-inputs"],
            cwd=REPO, capture_output=True, text=True, timeout=300, env=env)
        assert done.returncode == 0, f"{name}: {done.stderr[-800:]}"
        rows[name] = [json.loads(l) for l in done.stdout.splitlines() if l.strip()]
        assert len(rows[name]) == 2, rows[name]
    #: One answer, not two implementations of it.
    assert list(rows.values())[0] == list(rows.values())[1]


def test_each_launcher_asks_its_own_driver():
    """A launcher pointed at the other pilot's driver would derive the right
    mixtures by luck and the wrong pilot's anything else."""
    pairs = {
        "c3_batching_pilot_launch.sh": "c3_batching_pilot_driver.py",
        "c3_packing_pilot_launch.sh": "c3_packing_pilot_driver.py",
    }
    for launcher, driver in pairs.items():
        src = (REPO / "scripts/pod" / launcher).read_text()
        assert driver in src, f"{launcher} does not name {driver}"
        other = [d for d in pairs.values() if d != driver][0]
        assert other not in src, f"{launcher} names the other pilot's driver"
