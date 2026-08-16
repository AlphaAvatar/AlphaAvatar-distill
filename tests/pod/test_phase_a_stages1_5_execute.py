"""Execute the REAL `PhaseADriver` stages 1-5 locally, at $0, on CPU.

`test_phase_a_stage0_executes.py` closed stage 0 after three paid pods died in
it. Stages 1-5 were still **scripted** in the lifecycle rehearsal, so their
bodies had never run anywhere. Executing them here found two more defects that
were guaranteed to fire on a paid pod:

* `stage1` read `manifest["suite_hash"]` from `state_eval_v1/manifest.json`.
  That key does not exist. The `KeyError` would have been raised *after* the
  whole GPU beam search completed — the most expensive point in the session.
* `depth.causal_kl_greedy_v1` cached the unbypassed reference logits for the
  entire `calib.domain_balanced@v1` mixture, unconditionally, in float32:
  59,763 positions x 151,936 vocabulary = **33.8 GiB** of host RAM per
  invocation. Its first contact with the real mixture was this rehearsal, and
  the OOM killer took it.

What is substituted, and nothing else:

* the **teacher and target geometry** — a 32-wide/6-layer Qwen3 teacher and a
  16-wide/4-layer target, at the REAL 151,936 vocabulary so the production
  calibration token ids are valid. This is the expensive model boundary the
  driver's own DI surface exists for.
* the **item counts** — one state_eval item per declared (domain, sub-type), and
  the first few calibration items rather than all 67. Both are cost, not logic:
  at the real 151,936 vocabulary a single suite pass is 85 s and a single depth
  invocation over the whole mixture is tens of minutes on CPU. The suite OBJECT
  is the real one, so the structural hash stage 1 pins is the real one, and the
  calibration items are the real ones — `DOMAIN_BALANCED_V1.resolve()` runs, with
  both of its hash checks, and its output is what the operators consume.
  `logs/autoinit_phase_a_full_mixture_depth.json` records the same operator run
  against the **complete** 59,763-position mixture, which is the case that OOMed.
* the **vLLM engine probe** at stage 0 — no GPU here.
* the **probe training subprocess and the generation battery** — the
  irreducibly expensive boundary. The stub's shape is taken from a REAL
  `score_recovery_search.py` artifact rather than invented.

The calibration mixture is NOT substituted. `run_phase_a_search` calls
`DOMAIN_BALANCED_V1.resolve(REPO)` against the real frozen items file, hash
verified both ways, exactly as the pod does.
"""

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/autoinit"))

AUTH = REPO / "logs/autoinit_phase_a_authorization.json"
STAGE3_PROBE = REPO / "logs/autoinit_stage3_complete/engine_probe.json"
STATE_EVAL = REPO / "artifacts/stage1/state_eval_v1"
CALIBRATION = REPO / "artifacts/stage1/e8_calibration_v1/items.jsonl"
#: A real scored battery result. The stub copies its shape instead of guessing.
REAL_SCORED = (REPO / "logs/autoinit_stage3_complete"
               / "preflight_ctl_r0860k_sa_recovery_search.json")

pytestmark = pytest.mark.skipif(
    not (AUTH.is_file() and STAGE3_PROBE.is_file() and CALIBRATION.is_file()
         and (STATE_EVAL / "manifest.json").is_file() and REAL_SCORED.is_file()
         and (REPO / "logs/autoinit_phase_a_recovery_plan_frozen.json").is_file()
         and (REPO / "artifacts/stage3/recovery_search_v2/manifest.json").is_file()),
    reason="needs the issued authorization, the frozen plan, the staged "
           "state_eval + calibration + battery, and a real scored result")

TEACHER_GEOMETRY = dict(hidden_size=32, num_hidden_layers=6, intermediate_size=48,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                        vocab_size=151936, tie_word_embeddings=True)
TARGET_GEOMETRY = dict(hidden_size=16, num_hidden_layers=4, intermediate_size=24,
                       num_attention_heads=2, num_key_value_heads=2, head_dim=8,
                       vocab_size=151936, tie_word_embeddings=True)


class Args:
    stage = "all"
    image_digest = "sha256:rehearsal"
    rate = 0.99
    spent_usd = 0.20
    soft_stop_usd = 19.68
    authorized_usd = 20.02
    search_minutes = 180.0
    probe_train_minutes = 61.55
    probe_battery_minutes = 9.82


def load_driver(tmp_path: Path):
    spec = importlib.util.spec_from_file_location(
        "phase_a_driver_s15", REPO / "scripts/pod/autoinit_phase_a_driver.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phase_a_driver_s15"] = mod
    spec.loader.exec_module(mod)
    mod.WS = tmp_path
    mod.STATUS = tmp_path / "phase_a.status"
    mod.AUDIT = tmp_path / "audit"
    mod.SEARCH_WORKDIR = tmp_path / "search"
    return mod


def toy_teacher(seed: int = 4242):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(
        Qwen3Config(max_position_embeddings=4096, rope_theta=5_000_000,
                    **TEACHER_GEOMETRY)).float().eval()
    with torch.no_grad():
        for module in model.modules():
            if module.__class__.__name__ == "Qwen3RMSNorm":
                module.weight.uniform_(0.5, 1.5)
    return model


def suite_subset():
    """The real suite object, with one (the shortest) item per declared pair.

    The pairs are all kept because `StateEvaluator` refuses a subset that drops
    one — "a silently absent sub-type reweights its domain".
    """
    from load_state_eval import load

    suite, items, manifest = load(STATE_EVAL)
    shortest: dict[tuple[str, str], object] = {}
    for item in items:
        key = (item.domain, item.subtype)
        if (key not in shortest
                or item.input_ids.shape[1] < shortest[key].input_ids.shape[1]):
            shortest[key] = item
    return suite, [shortest[k] for k in sorted(shortest)], manifest


def fake_scored(label: str, seed: int, *, usable: int, correct: int) -> dict:
    """A battery result shaped like a real one, with the counts dialled."""
    result = copy.deepcopy(json.loads(REAL_SCORED.read_text()))
    result["label"], result["seed"] = label, seed
    n, n_scorable = result["n"], result["n_scorable"]
    result.update(usable=usable, usable_scorable=min(usable, n_scorable),
                  correct=correct,
                  usable_rollout_rate=round(usable / n, 4),
                  correct_overall=round(correct / n_scorable, 4),
                  correct_given_usable=round(correct / usable, 4) if usable else 0.0)
    share = usable / max(sum(c["n"] for c in result["per_capability"].values()), 1)
    for cap in result["per_capability"].values():
        cap["usable"] = min(cap["n"], int(round(cap["n"] * share)))
        cap["usable_rollout_rate"] = round(cap["usable"] / cap["n"], 4) if cap["n"] else 0.0
    result["_battery_minutes"] = 9.4
    return result


def calibration_subset(n: int):
    """The real frozen mixture, resolved and hash-checked, then truncated.

    `resolve()` is the production call — it re-derives both the file hash and
    the mixture content hash — and `as_operator_items` is the production
    adaptation. Only the count is reduced, and only because a full-mixture
    depth pass is tens of minutes of CPU at a 151,936 vocabulary.
    """
    from aadistill.autoinit.calibration import DOMAIN_BALANCED_V1
    from phase_a_search import as_operator_items

    return as_operator_items(DOMAIN_BALANCED_V1.resolve(REPO))[:n]


def build(tmp_path, monkeypatch, *, separated=False, n_suite_items=None,
          n_calibration_items=6):
    """The real driver with only the four boundaries substituted."""
    import phase_a_search

    mod = load_driver(tmp_path)
    mod.PhaseADriver.__init__(
        driver := mod.PhaseADriver.__new__(mod.PhaseADriver), Args())

    # -- boundary 1: the vLLM engine probe (no GPU here) ------------------
    real_gate = mod.PhaseADriver.gate.__get__(driver)

    def gate(name, argv, *, timeout, python="/opt/train/bin/python"):
        if name == "engine_probe":
            (mod.AUDIT / "engine_probe.json").write_text(STAGE3_PROBE.read_text())
            return subprocess.CompletedProcess(argv, 0, "", "")
        return real_gate(name, argv, timeout=timeout, python=sys.executable)

    driver.gate = gate

    # -- boundary 2: the teacher, the target geometry and the suite size --
    teacher = toy_teacher()
    suite, items, manifest = suite_subset()
    if n_suite_items:
        items = items[:n_suite_items]
    control_dir = tmp_path / "canonical_control"
    from aadistill.autoinit.arch import ArchSpec, get_adapter

    adapter = get_adapter("qwen3")
    target_spec = ArchSpec.of("qwen3", TARGET_GEOMETRY)
    adapter.save(adapter.build_model(
        adapter.build_config(teacher.config, target_spec), torch.float32, 4242),
        str(control_dir))

    real_search = phase_a_search.run_phase_a_search
    calibration = calibration_subset(n_calibration_items)

    def wrapped(*, workdir, state_eval, top_n, device, repo_root):
        # `repo_root` is passed through UNCHANGED: it is what
        # `DOMAIN_BALANCED_V1.resolve()` reads.
        return real_search(
            workdir=workdir, state_eval=state_eval, top_n=top_n,
            device="cpu", repo_root=repo_root,
            teacher_id="rehearsal-tiny-teacher",
            canonical_init=str(control_dir), canonical_sha256=None,
            teacher_loader=lambda: teacher,
            target_geometry=TARGET_GEOMETRY,
            suite_bundle=(suite, items, manifest),
            calibration_items=calibration)

    monkeypatch.setattr(phase_a_search, "run_phase_a_search", wrapped)

    # -- boundary 3: probe training. Nothing else in `run_probe` is stubbed:
    #    the journal, the resume binding and the config derivation are real.
    real_run = subprocess.run

    def fake_run(argv, **kwargs):
        argv = [str(a) for a in argv]
        if any(a.endswith("train_stage3.py") for a in argv):
            config = json.loads(Path(argv[argv.index("--config") + 1]).read_text())
            out = mod.REPO / f"artifacts/stage3/phase_a/{config['run_name']}"
            (out / "step000" / "model").mkdir(parents=True, exist_ok=True)
            (out / "latest.txt").write_text("step000\n")
            return subprocess.CompletedProcess(argv, 0, "trained", "")
        return real_run(argv, **kwargs)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    # -- boundary 4: generation + scoring ---------------------------------
    #
    # `separated` spreads the candidates by more than the frozen equivalence
    # interval (0.011695) on the plan's PRIMARY metric, which is
    # `correct_overall` — NOT `usable_rollout_rate`, which is the feasibility
    # gate. Spreading the wrong one leaves every finalist tied and the tie-break
    # rung runs anyway; that is how this first read as a driver defect. The
    # default keeps every candidate identical so the selection TIES and the
    # conditional third seed is owed. Both branches of stage 4 are then
    # reachable, and which one runs is a property of the numbers rather than of
    # a flag inside the driver.
    seen: dict[str, int] = {}

    def battery(label, model_dir, seed):
        state = label.split(".")[-2]
        if not separated:
            usable, correct = 74, 2
        else:
            rank = seen.setdefault(state, len(seen))
            # correct_overall = (2 + 6r)/170: 0.0118 .. 0.1882, steps of 0.0353.
            # usable_rollout_rate stays above the 0.30 feasibility floor.
            usable, correct = 74 + 9 * rank, 2 + 6 * rank
        return fake_scored(label, seed, usable=usable, correct=correct)

    driver.battery = battery
    return driver, mod


def run_stages(driver, mod, monkeypatch, tmp_path, stages=range(6)):
    """Stages in order, redirecting the trained-probe output tree after stage 1.

    Stages 0 and 1 read the REAL repository — the frozen assets, the scoring
    contract, the preregistration, the battery and `state_eval_v1` all live
    there. Only `run_probe`'s *output* path is redirected, and only once the
    stages that read the repo are done, so a rehearsal cannot litter
    `artifacts/stage3/phase_a/` or collide with a real run's outputs.
    """
    codes = {}
    for stage in stages:
        if stage == 2:
            monkeypatch.setattr(mod, "REPO", tmp_path / "pod_repo")
        codes[stage] = getattr(driver, f"stage{stage}")()
    return codes


@pytest.fixture(scope="module")
def driven(tmp_path_factory):
    """One real stage 0 -> 5 orchestration run, reused by the assertions."""
    tmp = tmp_path_factory.mktemp("phase_a_1_5")
    monkeypatch = pytest.MonkeyPatch()
    driver, mod = build(tmp, monkeypatch, separated=True)
    try:
        codes = run_stages(driver, mod, monkeypatch, tmp)
    finally:
        monkeypatch.undo()
    return driver, mod, codes


# --- stage 1: the search, with the production calibration -------------------


def test_every_stage_passes_end_to_end(driven):
    driver, mod, codes = driven
    failed = {s: driver.ev["stages"].get(str(s), {}).get("reason")
              for s, ok in codes.items() if not ok}
    assert not failed, failed


def test_the_production_calibration_resolves_to_what_the_operators_consume(driven):
    """The attempt-5 failure and the one behind it, both at $0.

    Attempt 5 died because the frozen mixture was never staged. The defect
    behind it was that the mixture's stored key is `ids` (a list) while every
    operator reads `i["input_ids"]` (a `[1, T]` LongTensor) — `dry_run_search.py`
    built its own items already in the operator shape, so no zero-cost run had
    ever fed a real one to an operator.
    """
    resolved = calibration_subset(6)
    assert len(resolved) == 6
    for item in resolved:
        assert item["input_ids"].shape[0] == 1
        assert item["input_ids"].dtype is torch.int64
        assert item["input_ids"].shape[1] == len(item["ids"])
        assert int(item["input_ids"].max()) < TEACHER_GEOMETRY["vocab_size"]


def test_stage1_ran_the_real_search_on_the_production_calibration(driven):
    driver, mod, _ = driven
    summary = json.loads((mod.AUDIT / "search_result.json").read_text())
    assert summary["calibration_profile"] == "calib.domain_balanced@v1", (
        "the search did not use the frozen mixture; a rehearsal that injects "
        "its own items proves nothing about the file the pod stages")
    assert summary["summary"]["n_complete_leaves"] >= 5
    assert len(driver.leaves) == driver.plan.searched_leaves
    for leaf in driver.leaves:
        leaf.require_recovery_admissible()
    assert driver.control_state.provenance == "retained_canonical"


def test_stage1_binds_the_suite_to_the_preregistered_identity(driven):
    """The check that was a `KeyError`. Both halves must be the pinned values."""
    driver, mod, _ = driven
    staged = json.loads((STATE_EVAL / "manifest.json").read_text())
    assert staged["content_sha256"] == mod.STATE_EVAL_CONTENT_SHA256
    summary = json.loads((mod.AUDIT / "search_result.json").read_text())
    assert summary["suite_hash"] == mod.STATE_EVAL_SUITE_HASH


@pytest.mark.parametrize("pin,fragment", [
    ("STATE_EVAL_SUITE_HASH", "suite structure"),
    ("STATE_EVAL_CONTENT_SHA256", "declares content"),
])
def test_stage1_refuses_a_suite_that_is_not_the_preregistered_one(
        pin, fragment, tmp_path, monkeypatch):
    """Both halves of the binding must fail the stage, not pass it silently.

    Parameterized because the check that this replaced was a `KeyError` that
    could never have refused anything — it crashed before comparing.
    """
    # Two calibration items: these tests are about the gate, not the search,
    # and the search is the expensive part.
    driver, mod = build(tmp_path, monkeypatch, separated=True,
                        n_calibration_items=2)
    assert driver.stage0() is True
    monkeypatch.setattr(mod, pin, "f" * 64)
    assert driver.stage1() is False
    assert fragment in driver.ev["stages"]["1"]["reason"]


# --- stages 2-3: the rungs --------------------------------------------------


def test_rung1_ran_six_probes_and_selected_two_searched_survivors(driven):
    driver, mod, _ = driven
    stage2 = driver.ev["stages"]["2"]
    assert stage2["n_probes"] == 6, "5 searched leaves + the injected control"
    assert len(driver.rung1["selected_searched"]) == driver.plan.survivors
    assert driver.rung1["auto_advanced_control"], (
        "the control advances unconditionally; without it rung 2 is not a "
        "baseline comparison")
    assert len(driver.rung1["advancing"]) == driver.plan.survivors + 1


def test_every_probe_config_differs_only_in_the_allowed_overrides(driven):
    """The experiment's core claim: only the initialization differs."""
    driver, mod, _ = driven
    frozen = json.loads(mod.FROZEN_RECIPE.read_text())
    configs = sorted((mod.AUDIT / "configs").glob("*.json"))
    assert len(configs) == 9, "6 rung-1 probes + 3 rung-2 probes"
    for path in configs:
        derived = json.loads(path.read_text())
        diff = {k for k in set(frozen) | set(derived)
                if frozen.get(k) != derived.get(k)}
        assert diff <= mod.PROBE_OVERRIDES, sorted(diff - mod.PROBE_OVERRIDES)
        assert derived["seed"] in (*driver.plan.seeds, driver.plan.tie_break_seed)


def test_rung1_seeds_are_sa_and_rung2_seeds_are_sb(driven):
    driver, mod, _ = driven
    probes = [json.loads(p.read_text())
              for p in sorted((mod.AUDIT / "probes").glob("*.json"))]
    by_rung: dict[int, set] = {}
    for probe in probes:
        by_rung.setdefault(probe["rung"], set()).add(probe["seed"])
    assert by_rung[1] == {driver.plan.seeds[0]}
    assert by_rung[2] == {driver.plan.seeds[1]}


def test_the_leaf_retention_record_accounts_for_all_five_leaves(driven):
    driver, mod, _ = driven
    retention = json.loads((mod.AUDIT / "leaf_retention.json").read_text())
    assert retention["n_leaves"] == 5
    assert retention["n_advancing"] + retention["n_rejected"] == 5, (
        "the counts must partition the SEARCHED leaves; the control is not "
        "one of them")
    assert retention["n_advancing"] == 2
    assert retention["n_control_advancing"] == 1
    tiers = {e["canonical_id"]: e["retention_tier"] for e in retention["entries"]}
    assert sum(t == "TIER_1_ACTIVE_CANONICAL" for t in tiers.values()) == 1
    for entry in retention["entries"]:
        # A rejected leaf keeps its evidence, not its bytes.
        assert entry["artifact_digest"] and entry["sa_probe_id"]
        assert entry["sa_result"]["usable_rollout_rate"] is not None
        assert entry["physically_present_on_pod_until_teardown"] is True
        if not entry["permanent_checkpoint_retained"]:
            assert entry["retention_tier"] == "TIER_4_DISPOSABLE"


def test_rung2_pools_both_seeds_per_finalist(driven):
    driver, mod, _ = driven
    rung2 = json.loads((mod.AUDIT / "rung2_selection.json").read_text())
    assert rung2["ranked"], "no finalist survived the gates"
    for row in rung2["ranked"]:
        assert set(row["seeds"]) == set(driver.plan.seeds[:2]), (
            "a finalist ranked on one seed is not a two-seed comparison")


# --- stage 4: both branches -------------------------------------------------


def test_stage4_does_not_run_when_the_finalists_are_separated(driven):
    driver, mod, _ = driven
    stage4 = driver.ev["stages"]["4"]
    assert driver.plan.primary_metric == "correct_overall", (
        "separation is judged on the PRIMARY metric; usable_rollout_rate is "
        "the feasibility gate and separating it proves nothing")
    assert stage4["ran"] is False
    assert stage4["reason_not_run"] == "resolved"
    assert driver.rung2["needs_tie_break_seed"] is False


def test_stage4_runs_the_tie_break_rung_when_finalists_are_equivalent(
        tmp_path, monkeypatch):
    """The conditional third seed, executed. Identical counts across candidates
    put every finalist inside the equivalence interval."""
    driver, mod = build(tmp_path, monkeypatch, n_calibration_items=2)
    codes = run_stages(driver, mod, monkeypatch, tmp_path, stages=range(5))
    assert all(codes.values()), {s: driver.ev["stages"].get(str(s), {}).get("reason")
                                 for s, ok in codes.items() if not ok}

    assert driver.rung2["needs_tie_break_seed"] is True, (
        "identical probe results must leave the finalists tied")
    assert driver.ev["stages"]["4"]["ran"] is True
    assert driver.ev["stages"]["4"]["n_probes"] == len(
        driver.rung2["tie_break_candidates"])

    rung3 = [json.loads(p.read_text())
             for p in sorted((mod.AUDIT / "probes").glob("*.json"))
             if json.loads(p.read_text())["rung"] == 3]
    assert rung3, "no rung-3 probe was journalled"
    assert {p["seed"] for p in rung3} == {driver.plan.tie_break_seed}
    assert all(".rung3." in p["probe_id"] and p["probe_id"].endswith(".sc")
               for p in rung3)

    assert driver.stage5() is True
    report = json.loads((mod.AUDIT / "phase_a_result.json").read_text())
    assert report["tie_break_ran"] is True
    assert report["decision_status"] == "unresolved_equivalence", (
        "a tie surviving seed sc is a RESULT; it must not name a winner")
    assert report["winner"] is None


# --- stage 5: the report ----------------------------------------------------


def test_stage5_writes_a_report_bound_to_the_plan_and_the_protocol(driven):
    driver, mod, _ = driven
    report = json.loads((mod.AUDIT / "phase_a_result.json").read_text())
    assert report["science_plan_hash"] == driver.plan.plan_hash
    assert report["evaluation_protocol_hash"] == (
        "250f72efbd43b86a475e8dda293b45f07ee61a4d858e147f4a5bd7681c32c2e4")
    assert report["equivalence_interval"] == pytest.approx(0.011695296982299022)
    assert report["feasibility_floor"] == pytest.approx(0.30)
    assert report["decision_status"] in (
        "resolved", "tie_pending", "unresolved_equivalence")
    assert report["capability_schema_enforced"] is True
    # Behaviour and correctness stay on separate axes.
    assert set(report["axes"]) == {"behaviour", "capability", "diagnostic"}
    assert report["report_sha256"]


def test_the_run_records_that_no_followon_is_reachable(driven):
    driver, mod, _ = driven
    driver.finish(True, failed=[])
    assert driver.ev["followon_started"] is False
    assert driver.ev["followon_reachable_from_this_driver"] is False
    assert driver.ev["retrains_permanent_controls"] is False


# --- resume -----------------------------------------------------------------


def test_a_journalled_probe_is_restored_but_only_for_the_same_run(driven):
    """Nine probes at ~71 min is a long run on hardware that has failed here."""
    driver, mod, _ = driven
    descriptors = mod.probe_configs(
        [*driver.leaves, driver.control_state], driver.plan, rung=1)
    restored = driver.restore_probe(descriptors[0])
    assert restored is not None and restored["resumed"] is True

    for moved in ({"seed": 999999},
                  {"student_artifact_digest": "0" * 64}):
        assert driver.restore_probe({**descriptors[0], **moved}) is None, (
            f"a probe was adopted after {sorted(moved)} moved; that result "
            "does not describe this run")
