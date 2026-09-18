"""The full-suite state-eval certification, driven for real and then mutated.

The check runs on a paid pod against the complete frozen suite. Everything here
runs at `$0` against a structurally-real toy suite and a deterministic toy
model, for two purposes:

1. **execution** — every line of every stage runs, so none of them first
   executes on a billing pod. Two attribute errors inside `PARETO_V1.rank` were
   found this way, each of which would have been a pod;
2. **mutation** — each predeclared assertion is shown to FIRE. A gate that has
   never refused anything is not a gate, and this project has shipped one:
   `require_headroom` was written for an OOM, named that OOM in its error, and
   still did not fire.

What this canNOT do is answer the question the pod answers. On the host both
implementations coincide -- `_reduce_on_device` is false for host tensors -- so
the kernel-agreement question is untouched here by construction.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CHECK = REPO / "scripts/validation/c2_state_eval_certification_check.py"
CONFIG = REPO / "configs/validation/c2_state_eval_certification.json"


def _module():
    spec = importlib.util.spec_from_file_location("c2_cert_check", CHECK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def M():
    return _module()


@pytest.fixture(scope="module")
def cfg():
    return json.loads(CONFIG.read_text())


@pytest.fixture
def toy(M):
    suite, items, info = M.toy_suite()
    teacher, _ = M.toy_teacher(info["vocab"], "cpu")
    return suite, items, teacher


# --- 1. the predeclared constants are what the maintainer set --------------


def test_the_predeclared_constants_are_the_reviewed_ones(M):
    """Read from the policy, not restated, and ten times below epsilon."""
    assert M.PARETO_EPSILON == 1e-4
    assert M.RANKED_ABS_TOLERANCE == 1e-5
    assert M.RANKED_ABS_TOLERANCE * 10 <= M.PARETO_EPSILON, (
        "the predeclared target must sit at least 10x below epsilon")
    assert set(M.RANKED_KEYS) == {
        "state.teacher_kl.equal_domain_mean",
        "state.teacher_kl.worst_domain",
        "state.critical_token_kl"}


def test_epsilon_is_read_from_the_policy_not_written_down(M):
    """If PARETO_V1's epsilon moved, the check must move with it."""
    from aadistill.initialization.planning.ranking import PARETO_V1

    assert M.PARETO_EPSILON == min(PARETO_V1.epsilon.values())
    assert M.RANKED_KEYS == tuple(o.key for o in PARETO_V1.objectives)


def test_the_discredited_threshold_is_not_used_anywhere(M):
    """`0.007782` is a disclosure trigger, not a decision threshold.

    The check's prose names it as the thing NOT being used, and its report
    explains the correction to a reader, so a text search is the wrong
    instrument -- it flagged both. What must not exist is the NUMBER, as a
    literal the code computes with, and that is an AST question.
    """
    import ast

    tree = ast.parse(CHECK.read_text())
    numbers = [node.value for node in ast.walk(tree)
               if isinstance(node, ast.Constant)
               and isinstance(node.value, float)]
    assert 0.007782 not in numbers, (
        "0.007782 appears as a numeric literal, so something computes with it")
    #: and the real boundary is not a literal either -- it is read from the
    #: policy, which is the property that keeps the two from drifting apart.
    assert 1e-4 not in numbers, (
        "epsilon is written down as a literal; it must come from PARETO_V1")


# --- 2. the toy suite is structurally real --------------------------------


def test_the_toy_suite_has_the_structure_the_aggregation_needs(M):
    """Several domains, a two-sub-type domain, and a tag matching nothing.

    A single-domain suite would never exercise the unweighted two-level mean,
    and a suite where every tag matches would never reach the branch that omits
    an absent critical-token class -- which the real suite needs, since
    `tool_close` covers 28 of 74,022 positions.
    """
    suite, items, info = M.toy_suite()
    assert len(suite.domains) >= 2
    assert any(len(subs) >= 2 for subs in suite.subtypes.values())
    assert "never_present" in suite.critical_tags
    assert all(int(item.tags["never_present"].sum()) == 0 for item in items)
    assert any(int(item.tags["eos"].sum()) == 1 for item in items), (
        "a rare tag covering exactly one position is what the real suite has")
    declared = {(d, s) for d, subs in suite.subtypes.items() for s in subs}
    assert {(i.domain, i.subtype) for i in items} == declared


def test_the_toy_teacher_is_deterministic(M):
    """Stage F asserts forward reproducibility; a random toy would fail it for
    the wrong reason."""
    import torch

    teacher, _ = M.toy_teacher(96, "cpu")
    ids = torch.randint(0, 96, (1, 12))
    a = teacher(ids).logits
    b = teacher(ids).logits
    assert torch.equal(a, b)


# --- 3. the whole thing runs, end to end ----------------------------------


def test_every_stage_runs_and_the_verdict_is_reached(M, cfg, toy):
    suite, items, teacher = toy
    report = {"stages": {}}
    report["stages"]["fingerprint"] = M.stage_fingerprint(
        teacher, items, report, "cpu")
    ev = M.stage_evaluate(cfg, suite, items, teacher, report, "cpu")
    report["stages"]["evaluate"] = ev
    pareto = M.stage_pareto(cfg, ev, report)

    assert report["stages"]["fingerprint"]["bitwise_reproducible"] is True
    assert len(ev["candidates"]) == len(cfg["evaluate"]["magnitudes"])
    #: On the host both implementations are the same code path, so this is the
    #: one place the expected drift is exactly zero -- and that is a statement
    #: about the rehearsal, not about the certification.
    assert ev["worst_ranked_absolute_drift"] == 0.0
    assert pareto["identical_decisions"] is True
    assert pareto["boundary_cases_identical"] is True
    assert len(pareto["boundary_cases"]) == 5
    #: every candidate carried every ranked objective and every tag
    for row in ev["candidates"]:
        for key in M.RANKED_KEYS:
            assert row["metrics"][key]["absolute"] == 0.0
        assert all(row["tagged_positions_identical"].values())
        assert row["logits_verified_identical"] is True


def test_the_fingerprints_prove_both_passes_saw_one_forward(M, cfg, toy):
    """The mechanism, not just its result: pass 2 verifies pass 1's digests."""
    suite, items, teacher = toy
    prints: dict = {}
    row = M._evaluate_both(suite, items, teacher, 0.35, 1, 512, prints, "cpu")
    assert row["logits_verified_identical"] is True
    #: One digest per item -- the reference and the candidate call forward the
    #: SAME ids, so they share a key and the second overwrites the first with
    #: an identical digest.
    assert len(prints) == len(items)
    #: ONE forward per item per pass, since the candidate reuses the
    #: reference. Subrun s1 forwarded twice per item per pass -- four 4B
    #: forwards per item per candidate -- which doubled the pod time and
    #: buried the reduction's share of it.
    assert row["forwards_per_pass"] == len(items)
    assert row["reference_forwards"] == {"old": len(items), "new": len(items)}
    assert row["candidate_reused_the_reference"] == {
        "old": len(items), "new": len(items)}
    assert row["candidate_extra_forwards"] == {"old": 0, "new": 0}


# --- 4. every predeclared assertion FIRES ---------------------------------


def test_a_ranked_objective_drift_above_the_target_stops_the_run(M, cfg, toy,
                                                                 monkeypatch):
    """The assertion the whole certification turns on."""
    suite, items, teacher = toy
    real = M._evaluate_both

    def drifted(*args, **kwargs):
        row = real(*args, **kwargs)
        key = M.RANKED_KEYS[0]
        row["metrics"][key]["absolute"] = 5e-5      # above 1e-5, below epsilon
        return row

    monkeypatch.setattr(M, "_evaluate_both", drifted)
    with pytest.raises(AssertionError) as exc:
        M.stage_evaluate(cfg, suite, items, teacher, {"stages": {}}, "cpu")
    message = str(exc.value)
    assert "PREDECLARED" in message and "STOP and report" in message
    assert M.RANKED_KEYS[0] in message


def test_a_drift_below_the_target_does_not_stop_it(M, cfg, toy, monkeypatch):
    """The other direction, so the test above is not passing on any drift."""
    suite, items, teacher = toy
    real = M._evaluate_both

    def barely(*args, **kwargs):
        row = real(*args, **kwargs)
        row["metrics"][M.RANKED_KEYS[0]]["absolute"] = 9e-6
        return row

    monkeypatch.setattr(M, "_evaluate_both", barely)
    out = M.stage_evaluate(cfg, suite, items, teacher, {"stages": {}}, "cpu")
    assert out["worst_ranked_absolute_drift"] == 9e-6
    assert out["ranked_drift_below_epsilon_by"] > 10


def test_a_metric_emitted_by_only_one_side_stops_the_run(M, cfg, toy,
                                                         monkeypatch):
    """A changed measurement, even when every shared number agrees."""
    suite, items, teacher = toy
    real = M._evaluate_both

    def missing(*args, **kwargs):
        row = real(*args, **kwargs)
        row["metrics"]["state.nll.general"] = {
            "present_in_old": True, "present_in_new": False, "absolute": None}
        return row

    monkeypatch.setattr(M, "_evaluate_both", missing)
    with pytest.raises(AssertionError, match="emitted by only one"):
        M.stage_evaluate(cfg, suite, items, teacher, {"stages": {}}, "cpu")


def test_a_tag_covering_different_positions_stops_the_run(M, cfg, toy,
                                                          monkeypatch):
    suite, items, teacher = toy
    real = M._evaluate_both

    def retagged(*args, **kwargs):
        row = real(*args, **kwargs)
        row["tagged_positions_identical"]["eos"] = False
        return row

    monkeypatch.setattr(M, "_evaluate_both", retagged)
    with pytest.raises(AssertionError, match="different number of positions"):
        M.stage_evaluate(cfg, suite, items, teacher, {"stages": {}}, "cpu")


def test_a_diagnostic_drift_above_the_scale_bound_stops_the_run(M, cfg, toy,
                                                                monkeypatch):
    """Nothing prunes on diagnostics, but a drift above the expected error
    scale is a finding rather than rounding."""
    suite, items, teacher = toy
    real = M._evaluate_both

    def drifted(*args, **kwargs):
        row = real(*args, **kwargs)
        entry = row["metrics"]["state.top1_agreement"]
        #: ABSOLUTE now, because the bound is absolute-with-a-unit-floor. A
        #: relative-only bound failed subrun s1 on a near-zero diagnostic.
        entry["absolute"] = 1e-2
        entry["old"] = 0.5
        return row

    monkeypatch.setattr(M, "_evaluate_both", drifted)
    with pytest.raises(AssertionError, match="beyond the scale bound"):
        M.stage_evaluate(cfg, suite, items, teacher, {"stages": {}}, "cpu")


def test_a_non_reproducible_forward_stops_the_run(M, toy):
    """Stage F is what makes "identical logits" a measurement."""
    import torch
    from types import SimpleNamespace

    _, items, _ = toy
    counter = {"n": 0}

    class _Drifting:
        def __call__(self, ids):
            counter["n"] += 1
            g = torch.Generator().manual_seed(counter["n"])
            return SimpleNamespace(
                logits=torch.randn(1, int(ids.shape[1]), 96, generator=g))

    with pytest.raises(AssertionError, match="NOT bitwise reproducible"):
        M_ = _module()
        M_.stage_fingerprint(_Drifting(), items, {}, "cpu")


def test_a_fingerprint_mismatch_between_passes_stops_the_run(M, toy):
    """If the second pass sees different logits, the comparison is void.

    Driven through the MECHANISM rather than by appending to `mismatches`: a
    pre-poisoned store means the verifying pass recomputes a digest that does
    not match what was recorded, which is exactly what a non-reproducible
    forward would look like. Forcing the flag would have tested the assertion
    and not the thing that sets it.
    """
    suite, items, teacher = toy
    prints: dict = {}
    #: A first pass, to learn the real keys.
    M._evaluate_both(suite, items, teacher, 0.35, 1, 512, prints, "cpu")
    assert prints, "the recording pass stored nothing, so nothing is verified"
    poisoned = {key: "0" * 64 for key in prints}
    with pytest.raises(AssertionError, match="did not agree with the fingerprint"):
        M._evaluate_both(suite, items, teacher, 0.35, 1, 512, poisoned, "cpu")


def test_a_teacher_that_drifts_between_candidates_is_caught(M, cfg, toy):
    """The case the first fingerprinting design missed entirely.

    Each candidate ran two passes and the recording pass wrote
    unconditionally, so a teacher whose output changed BETWEEN candidates was
    silently re-recorded rather than noticed. The store is now a run-wide
    invariant, and this is the scenario that distinguishes the two designs.
    """
    from types import SimpleNamespace

    import torch

    suite, items, _ = toy
    calls = {"n": 0}
    g = torch.Generator().manual_seed(3)
    table = torch.randn(96, 96, generator=g, dtype=torch.float32)

    class _DriftsAfterAWhile:
        def __call__(self, ids):
            calls["n"] += 1
            rows = table[ids[0]]
            #: stable for the first candidate's two passes, then different
            #: stable for candidate 1's two passes (one reference forward per
            #: item per pass, so 2 * len(items) calls), then different
            if calls["n"] > 2 * len(items):
                rows = rows + 1.0
            return SimpleNamespace(logits=rows[None, ...])

    prints: dict = {}
    teacher = _DriftsAfterAWhile()
    M._evaluate_both(suite, items, teacher, 0.05, 1, 512, prints, "cpu")
    with pytest.raises(AssertionError,
                       match="did not agree with the fingerprint"):
        M._evaluate_both(suite, items, teacher, 0.35, 1, 512, prints, "cpu")


def test_a_pass_that_skips_an_item_is_caught(M, toy):
    """`verify=True` requires the key to be KNOWN, not merely to agree."""
    suite, items, teacher = toy
    #: An empty store means the verifying pass sees nothing it recognises. That
    #: is what a recording pass which forwarded fewer items would leave behind.
    verifier = M._Fingerprinting(teacher, {}, verify=True)
    verifier(items[0].input_ids)
    assert verifier.mismatches == [
        f"{verifier.mismatches[0].split(':')[0]}:unseen-in-the-recording-pass"]


def test_a_pareto_flip_at_the_boundary_stops_the_run(M, cfg, toy, monkeypatch):
    """The decision-level failure the boundary cases exist to find."""
    suite, items, teacher = toy
    ev = M.stage_evaluate(cfg, suite, items, teacher, {"stages": {}}, "cpu")
    #: A drift the size of epsilon itself: applied in the closing direction at
    #: the boundary, it must change a decision.
    ev["worst_ranked_absolute_drift"] = M.PARETO_EPSILON
    with pytest.raises(AssertionError, match="changes a PARETO_V1 decision"):
        M.stage_pareto(cfg, ev, {"stages": {}})


def test_differing_selected_ids_stop_the_run(M, cfg, toy, monkeypatch):
    """The direct comparison, not only the boundary construction."""
    suite, items, teacher = toy
    ev = M.stage_evaluate(cfg, suite, items, teacher, {"stages": {}}, "cpu")
    #: INVERT the order on one side. Making the first candidate better would
    #: change nothing -- it already dominates, being the one closest to the
    #: teacher -- so the mutation has to make the LAST candidate the best, and
    #: reasoning that "a big change must change something" is exactly how a
    #: vacuous mutation gets recorded as a caught one.
    for key in M.RANKED_KEYS:
        ev["candidates"][-1]["_new_values"][key] = -1.0
    with pytest.raises(AssertionError,
                       match="selects different ids|front membership"):
        M.stage_pareto(cfg, ev, {"stages": {}})


# --- 4b. everything the pod needs is DECLARED ------------------------------


def test_every_import_resolves_under_a_declared_ship_path(M, cfg):
    """The failure class that killed two paid subruns, one per producer.

    The launcher's `DEFAULT_SHIP` does not include `scripts/autoinit`, where
    `load_state_eval` and `phase_a_frozen` live, and cannot include the suite
    itself -- it is in the gitignored artifact store. An unshipped import fails
    on the pod AFTER it starts billing, so the ship list is declared in the
    config and every module the check imports is required to resolve under it.
    """
    import ast

    roots = [REPO / rel for rel in cfg["ship_paths"]]
    assert all(root.exists() for root in roots), (
        f"a declared ship path is missing: "
        f"{[str(r) for r in roots if not r.exists()]}")

    tree = ast.parse(CHECK.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])

    #: Resolved by ASKING the interpreter which modules are built in or
    #: standard, rather than by listing them -- a hand-kept list is how
    #: `__future__` ended up reported as an unshipped dependency.
    import sys as _sys

    stdlib = set(getattr(_sys, "stdlib_module_names", ())) | set(
        _sys.builtin_module_names)
    provided_by_the_image = {"torch", "transformers", "numpy", "safetensors"}
    unresolved = []
    for name in sorted(names - stdlib - provided_by_the_image):
        if any((root / name).exists() or (root / f"{name}.py").exists()
               for root in roots):
            continue
        unresolved.append(name)
    assert not unresolved, (
        f"{unresolved} is imported by the check and resolves under no declared "
        "ship path, so it would be missing on the pod")


def test_the_suite_root_is_shipped(M, cfg):
    """The suite is gitignored, so it travels only if it is named."""
    root = cfg["suite"]["root"]
    assert root in cfg["ship_paths"], (
        f"{root} is the suite this certification is ABOUT and it is not in the "
        "ship list; it is gitignored, so nothing else carries it")
    assert (REPO / root / "items.jsonl").is_file()
    assert (REPO / root / "manifest.json").is_file()


# --- 4c. the report survives every exit path ------------------------------


def test_the_report_is_written_even_when_a_stage_raises(tmp_path, monkeypatch):
    """The defect that cost subrun s1 its evidence.

    `--out` defaulted to `None`, so the check measured all three candidates for
    `$0.4925`, failed on its last assertion, wrote nothing, and the launcher
    collected zero files. The measurement survives only as a stdout tail. A
    correct measurement that lives in memory is not evidence.
    """
    import subprocess
    import sys

    module = _module()
    assert module.REPORT_NAME.endswith(".json")

    #: Driven through `main()` in a SUBPROCESS, because that is the code path
    #: the pod runs and the one that owns the writing. A monkeypatched
    #: in-process call would test a different function.
    out = tmp_path / "collected"
    env = {**dict(**__import__("os").environ),
           "PYTHONPATH": f"{REPO}/src:{REPO}/scripts",
           "HF_HUB_OFFLINE": "1"}
    bad_config = tmp_path / "bad.json"
    cfg = json.loads(CONFIG.read_text())
    cfg["suite"]["items"] = 79          # the tree has 80; load_suite refuses
    bad_config.write_text(json.dumps(cfg))
    r = subprocess.run(
        [sys.executable, str(CHECK), "--config", str(bad_config),
         "--run-id", "written-on-failure", "--device", "cpu", "--out", str(out)],
        capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode != 0, "the poisoned config should not have passed"
    written = out / module.REPORT_NAME
    assert written.is_file(), (
        f"no report at {written} after a failing run; stdout was:\n{r.stdout}")
    report = json.loads(written.read_text())
    assert report["verdict"] in ("FAIL", "NOT RUN")
    assert "predeclared" in report, (
        "a failed run must still record what it was measuring against")


def test_the_default_out_is_the_directory_the_launcher_collects(M):
    """`cuda_engineering_launch.py` scp's `<repo>/artifacts/validation` back and
    passes no `--out`, so the default is what decides whether anything is
    collected at all."""
    import argparse

    parser_default = None
    for line in CHECK.read_text().splitlines():
        if '"--out"' in line:
            parser_default = line
            break
    assert parser_default is not None
    assert "artifacts/validation" in parser_default, (
        f"--out does not default to the collected directory: {parser_default!r}")
    launcher = (REPO / "scripts/validation/cuda_engineering_launch.py").read_text()
    assert "artifacts/validation" in launcher, (
        "the launcher no longer collects that directory; the default is stale")
    assert "--out" not in launcher.split("def validate")[1].split("def ")[0], (
        "the launcher now passes --out itself, so the default is not what "
        "decides this any more -- update this test rather than assuming")
    del argparse


# --- 5. the suite identity is checked before any work ---------------------


def test_a_suite_that_is_not_the_frozen_one_is_refused(M, cfg, tmp_path):
    """A certification of a smaller suite would be a different claim."""
    bad = dict(cfg)
    bad["suite"] = dict(cfg["suite"])
    bad["suite"]["prediction_positions"] = 2032
    with pytest.raises(AssertionError, match="prediction positions"):
        M.load_suite(bad)


def test_the_real_frozen_suite_satisfies_the_declared_identity(M, cfg):
    """The counts in the config are the tree's, checked at `$0` rather than
    discovered on a pod."""
    suite, items, info = M.load_suite(cfg)
    assert info["positions"] == 74022
    assert info["items"] == 80
    assert suite.suite_hash.startswith(cfg["suite"]["suite_hash_prefix"])
    assert len(suite.critical_tags) == 4


def test_a_wrong_suite_hash_is_refused(M, cfg):
    bad = dict(cfg)
    bad["suite"] = dict(cfg["suite"])
    bad["suite"]["suite_hash_prefix"] = "0" * 16
    with pytest.raises(AssertionError, match="not the frozen suite"):
        M.load_suite(bad)


# --- 6. the old path is the production path, not a copy -------------------


def test_the_host_path_evaluator_overrides_exactly_one_method(M):
    """An equivalence claim needs the thing it is equivalent to.

    `HostPathEvaluator` restores the pre-optimization behaviour by putting one
    `.cpu()` back, so the AGGREGATION both sides run is byte-identical
    production code. A re-implemented aggregation would be a third thing that
    agrees with neither -- and this project has certified a defective line with
    a stub that matched the consumer instead of the producer.
    """
    from aadistill.initialization.planning.metrics import StateEvaluator

    own = {name for name, value in vars(M.HostPathEvaluator).items()
           if callable(value) and not name.startswith("__")}
    assert own == {"_reference_for"}, (
        f"the certification evaluator overrides {sorted(own)}; every "
        "additional override is aggregation that is no longer production code. "
        "`__init__` is excluded by the name filter and is allowed: it only "
        "stores the host flag, the candidate and the timing counters.")
    assert issubclass(M.HostPathEvaluator, StateEvaluator)


def test_the_host_path_really_reduces_on_the_host(M, toy):
    """The override has to CHANGE the device, or both passes are the new path.

    On the host this is trivially true for both, so what is asserted is the
    mechanism: the subclass's reference comes back on the CPU whatever device
    it was asked for.
    """
    suite, items, teacher = toy
    ev = M.HostPathEvaluator(suite, items, device="cpu", chunk=512, host=True)
    ev.prime_reference(teacher)
    ref = ev._reference_for(items[0])
    assert ref.device.type == "cpu"
    #: and it timed the forward it just did, which is the other thing the
    #: override exists for
    assert ev.forwards == 1 and ev.forward_seconds > 0.0


def test_the_perturbation_is_bitwise_identical_on_both_sides(M, toy):
    """If it were not, the comparison would be measuring the perturbation."""
    import torch

    _, items, teacher = toy
    a = M._Perturbed(teacher, 0.35, 1, to_host=False)
    b = M._Perturbed(teacher, 0.35, 1, to_host=True)
    ids = items[0].input_ids
    assert torch.equal(a(ids).logits, b(ids).logits)


def test_a_different_magnitude_gives_a_different_candidate(M, toy):
    """Guard on the guard above: identical outputs for different inputs would
    make the equality test vacuous."""
    import torch

    _, items, teacher = toy
    a = M._Perturbed(teacher, 0.05, 1, to_host=False)
    b = M._Perturbed(teacher, 1.0, 1, to_host=False)
    ids = items[0].input_ids
    assert not torch.equal(a(ids).logits, b(ids).logits)
