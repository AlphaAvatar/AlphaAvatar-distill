"""Execute the REAL `ContinuationDriver` end to end, at $0, on CPU.

The behavioural continuation exists to finish Phase B without re-buying its
16.5 h P=2 search. That makes two things worth proving before a grant is issued,
and neither is provable by inspection:

1. **The frozen decision path actually runs.** Stage 1 imports completed
   evidence rather than recomputing it, exactly one `sb` is bought, the pooled
   rule decides, and a `tie_pending` reaches the conditional `sc` and terminates.
2. **The search cannot execute.** Not "is not called" — cannot be. `BeamSearch`
   and `run_phase_a_search` are replaced by detonators for the whole run.

Writing this file found five defects that were each guaranteed to fire on a paid
pod, and that every cheaper gate had passed:

* `stage_bind` never called `super().stage0()`. `self.plan` and
  `self.evaluation_protocol` would have been unset when stage 3 read them —
  `AttributeError` after the session had paid for setup and a probe.
* `enter()` advanced `PHASE_A_PLAN_V1`, whose stage 1 is the search, so every
  stage was ordered against the wrong plan's preconditions.
* `ContinuationAuthorization` had no `require_science_plan`. The inherited stage
  0 calls it unconditionally. The type had been written from what the
  continuation *needs* instead of what the machinery it subclasses *requires*.
* `restore_probe` was inherited strict, comparing raw `evaluation_protocol_hash`.
  The eight citable probes carry TWO raw hashes that differ by host driver patch
  alone, which `generation_runtime_comparability@v2` declares non-material. The
  session would have re-bought all eight — roughly nine times its ceiling —
  while reporting success.
* `build_universe` required all SIX evidence candidates to be staged as bytes,
  though only three are ever probed: ~3.6 GiB of checkpoints transferred to read
  their filenames, and three non-survivors left one filter away from a probe.

What is substituted, and nothing else:

* the **teacher and target geometry** — toy Qwen3 at the REAL 151,936 vocabulary.
* the **frozen-assets gate and the vLLM engine probe** — no GPU here; the engine
  probe's output is a REAL recorded artifact, so everything downstream of it
  (the generation protocol, the comparability rule, the protocol hash) runs for
  real.
* the **probe training subprocess and the generation battery** — the irreducibly
  expensive boundary, shaped from a REAL scored artifact.

The evidence artifacts are toy, and deliberately so: this test must be able to
mutate them. Their SHAPE is taken from the real amendment, the real reuse records
and real probe records, including the two-distinct-protocol-hash property that
the reuse defect above turned on.
"""

import copy
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/pod"))
sys.path.insert(0, str(REPO / "scripts/autoinit"))

STAGE3_PROBE = REPO / "logs/autoinit_stage3_complete/engine_probe.json"
FROZEN_PLAN = REPO / "logs/autoinit_phase_a_recovery_plan_frozen.json"
REAL_SCORED = (REPO / "logs/autoinit_stage3_complete"
               / "preflight_ctl_r0860k_sa_recovery_search.json")
REAL_AMENDMENT = REPO / "logs/autoinit_phase_b_identity_collapse_amendment.json"
PRICING = REPO / "logs/autoinit_behavioural_continuation_pricing.json"

pytestmark = pytest.mark.skipif(
    not (STAGE3_PROBE.is_file() and FROZEN_PLAN.is_file() and REAL_SCORED.is_file()
         and REAL_AMENDMENT.is_file()
         and (REPO / "artifacts/stage1/state_eval_v1/manifest.json").is_file()),
    reason="needs the frozen plan, a recorded engine probe, a real scored "
           "battery result and the identity-collapse amendment")

TEACHER_GEOMETRY = dict(hidden_size=32, num_hidden_layers=6, intermediate_size=48,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                        vocab_size=151936, tie_word_embeddings=True)
TARGET_GEOMETRY = dict(hidden_size=16, num_hidden_layers=4, intermediate_size=24,
                       num_attention_heads=2, num_key_value_heads=2, head_dim=8,
                       vocab_size=151936, tie_word_embeddings=True)

#: The three ACTIVE finalists and the three searched non-survivors. Kept apart
#: here for the same reason the driver keeps them apart: the second group is
#: evidence, and must never reach a probe.
FINALIST_LEAVES = ("fe9683e6a9c783bbc6fe276a78c851c6",
                   "85bde4ded2c31953f802e39cf2252c87")
NON_SURVIVORS = ("ab7632b00788f825e252ea8d5ff4be30",
                 "bf5ae3b6ae00c3dbb963805135b3838f",
                 "cca699c93f34dad7e94a5d13a25b2bc2")
CONTROL_STATE = "control-qwen3_0p6b_init_v0"
CONTROL_COLLAPSED = "control-qwen"

#: The two raw protocol hashes the real evidence carries. They differ by host
#: NVIDIA driver patch (580.178.04 vs 580.159.03) and are comparable under
#: `generation_runtime_comparability@v2`. Reproduced here because a test that
#: gave every imported probe one hash could not catch the reuse defect.
PROTOCOL_HISTORICAL = "7327e880645492ec308fce62054b4845dce3de12e3079bb17a32e7d22898aaaa"
PROTOCOL_ATTEMPT5 = "250f72efbd43b86a475e8dda293b45f07ee61a4d858e147f4a5bd7681c32c2e4"

#: The REAL frozen plan's seeds, read from the artifact rather than invented.
#: Guessing them made every imported probe fail `restore_probe`'s seed check, so
#: all three sb probes were re-bought and the test read as a driver defect.
_FROZEN = json.loads(FROZEN_PLAN.read_text())
_PLAN = _FROZEN.get("plan", _FROZEN)
SEED_SA, SEED_SB = _PLAN["seeds"]
SEED_SC = _PLAN["tie_break_seed"]


class Args:
    stage = "all"
    image_digest = "sha256:rehearsal"
    rate = 0.99
    spent_usd = 0.20
    soft_stop_usd = 6.50
    authorized_usd = 8.07
    search_minutes = 0.0
    search_deadline_minutes = 0.0
    probe_train_minutes = 61.55
    probe_battery_minutes = 9.82
    rung2_probes = 3
    tie_break_probes = 2


def load_continuation(tmp_path: Path):
    spec = importlib.util.spec_from_file_location(
        "continuation_b_driver_wf",
        REPO / "scripts/pod/autoinit_continuation_b_driver.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["continuation_b_driver_wf"] = mod
    spec.loader.exec_module(mod)
    mod.WS = tmp_path
    mod.STATUS = tmp_path / "continuation_b.status"
    # NOT `mod.bind_status_file()`: that writes the shared parent global and
    # nothing would undo it. `build()` monkeypatches `parent.STATUS` instead, so
    # the rebinding is scoped to the test and reverted with the fixture.
    mod.AUDIT = tmp_path / "audit"
    mod.AUDIT.mkdir(parents=True, exist_ok=True)
    return mod


def toy_teacher(seed: int = 4242):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    return Qwen3ForCausalLM(
        Qwen3Config(max_position_embeddings=4096, rope_theta=5_000_000,
                    **TEACHER_GEOMETRY)).float().eval()


def build_checkpoints(tmp_path: Path) -> dict[str, dict]:
    """Toy bytes for the THREE finalists only — which is the point.

    The three non-survivors get no directory at all. If the driver ever demands
    their bytes again, this fixture cannot satisfy it.
    """
    from transformers import AutoConfig

    from aadistill.autoinit.arch import ArchSpec, get_adapter
    from aadistill.autoinit.artifact import identify_checkpoint

    adapter = get_adapter("qwen3")
    target = ArchSpec.of("qwen3", TARGET_GEOMETRY)
    teacher = toy_teacher()
    out = {}
    for i, state_id in enumerate((*FINALIST_LEAVES, CONTROL_COLLAPSED)):
        directory = tmp_path / "ckpt" / state_id
        adapter.save(adapter.build_model(
            adapter.build_config(teacher.config, target), torch.float32, 100 + i),
            str(directory))
        spec = adapter.spec_from_config(AutoConfig.from_pretrained(str(directory)))
        artifact = identify_checkpoint(directory, adapter=adapter, spec=spec,
                                       num_parameters=adapter.param_count(spec))
        out[state_id] = {"dir": directory, "digest": artifact.artifact_digest}
    return out


def scored(label: str, seed: int, *, usable: int, correct: int) -> dict:
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
        cap["usable_rollout_rate"] = (round(cap["usable"] / cap["n"], 4)
                                      if cap["n"] else 0.0)
    result["_battery_minutes"] = 9.4
    return result


def probe_record(state_id: str, short: str, rung: int, seed_name: str, seed: int,
                 digest: str, protocol: str, *, usable: int, correct: int) -> dict:
    probe_id = f"autoinit.v1.phase_a.rung{rung}.{short}.{seed_name}"
    return {"probe_id": probe_id, "state_id": state_id, "rung": rung, "seed": seed,
            "is_control": state_id == CONTROL_STATE,
            "student_artifact_digest": digest,
            "evaluation_protocol_hash": protocol,
            "complete": True, "resumed": False,
            "train_minutes": 61.5, "battery_minutes": 9.4,
            "result": scored(probe_id, seed, usable=usable, correct=correct)}


def write_evidence(tmp_path: Path, ckpts: dict, *, tie: bool) -> dict:
    """Toy evidence with the REAL structure, including the real reuse inventory.

    `tie` controls only the battery numbers, so which branch of stage 4 runs is
    a property of the measurements rather than of a flag inside the driver.
    """
    from aadistill.autoinit.identity_collapse import collapse, universe_identity
    from aadistill.infrastructure.manifest import sha256_json

    logs = tmp_path / "evidence"
    (logs / "historical").mkdir(parents=True, exist_ok=True)
    (logs / "attempt5").mkdir(parents=True, exist_ok=True)

    digest_of = {
        FINALIST_LEAVES[0]: ckpts[FINALIST_LEAVES[0]]["digest"],
        FINALIST_LEAVES[1]: ckpts[FINALIST_LEAVES[1]]["digest"],
        CONTROL_COLLAPSED: ckpts[CONTROL_COLLAPSED]["digest"],
    }
    for i, state_id in enumerate(NON_SURVIVORS):
        digest_of[state_id] = f"{i:064x}"

    candidates = []
    for state_id in (*FINALIST_LEAVES, *NON_SURVIVORS):
        roles = (["searched", "imported_finalist"]
                 if state_id in NON_SURVIVORS[2:] else ["searched"])
        candidates.append({"state_id": state_id, "artifact_digest": digest_of[state_id],
                           "roles": roles,
                           "checkpoint_path": str(ckpts[state_id]["dir"])
                           if state_id in ckpts else str(tmp_path / "absent" / state_id)})
    candidates.append({"state_id": CONTROL_COLLAPSED,
                       "artifact_digest": digest_of[CONTROL_COLLAPSED],
                       "roles": ["control"],
                       "checkpoint_path": str(ckpts[CONTROL_COLLAPSED]["dir"])})

    collapsed = collapse([{"state_id": c["state_id"],
                           "artifact_digest": c["artifact_digest"], "role": r,
                           "checkpoint_path": c["checkpoint_path"]}
                          for c in candidates for r in c["roles"]])
    identity = universe_identity(collapsed)

    rung1 = {
        "computed_by": "SuccessiveHalvingPlan.select_rung1_survivors",
        "selected_searched": list(FINALIST_LEAVES),
        "auto_advanced_control": [CONTROL_STATE],
        "advancing": [CONTROL_STATE, *FINALIST_LEAVES],
        "rule": "top-2 searched by correct_overall, control advances unconditionally",
        "all_exclusions": [{"state_id": s, "reason": "not in the top 2"}
                           for s in NON_SURVIVORS],
    }
    amendment = {"schema": "aadistill.autoinit.identity_collapse_amendment/v1",
                 "collapsed_universe": {"distinct_candidates": len(collapsed),
                                        "universe_identity": identity,
                                        "candidates": candidates},
                 "rung1_selection": rung1}
    amendment["amendment_sha256"] = sha256_json(amendment)
    (logs / "amendment.json").write_text(json.dumps(amendment, indent=2))

    selection = {"schema": "aadistill.autoinit.stage1_selection/v1",
                 "top_n": 5, "selected": list(FINALIST_LEAVES)}
    selection["selection_sha256"] = sha256_json(selection)
    (logs / "stage1_selection.json").write_text(json.dumps(selection, indent=2))

    # -- the REAL reuse inventory -------------------------------------------
    # historical: 85bde sa/sb/sc, cca699 sa/sb/sc, control sa/sb
    # attempt 5 : ab7632 sa, bf5ae3 sa, fe9683 sa
    # so: every candidate has sa; sb is missing for fe9683 ONLY; sc is missing
    # for fe9683 and the control.
    spread = {FINALIST_LEAVES[0]: 0, FINALIST_LEAVES[1]: 1, CONTROL_STATE: 2}

    def counts(state_id: str) -> tuple[int, int]:
        if tie:
            return 74, 2
        rank = spread[state_id]
        return 74 + 9 * rank, 2 + 6 * rank

    historical_admitted, attempt5_reusable = [], []
    for state_id, short, seeds, where, protocol in (
            (FINALIST_LEAVES[1], "85bde4ded2c3", ("sa", "sb", "sc"), "historical",
             PROTOCOL_HISTORICAL),
            (NON_SURVIVORS[2], "cca699c93f34", ("sa", "sb", "sc"), "historical",
             PROTOCOL_HISTORICAL),
            (CONTROL_STATE, CONTROL_COLLAPSED, ("sa", "sb"), "historical",
             PROTOCOL_HISTORICAL),
            (NON_SURVIVORS[0], "ab7632b00788", ("sa",), "attempt5", PROTOCOL_ATTEMPT5),
            (NON_SURVIVORS[1], "bf5ae3b6ae00", ("sa",), "attempt5", PROTOCOL_ATTEMPT5),
            (FINALIST_LEAVES[0], "fe9683e6a9c7", ("sa",), "attempt5", PROTOCOL_ATTEMPT5)):
        for seed_name in seeds:
            rung = {"sa": 1, "sb": 2, "sc": 3}[seed_name]
            seed = {"sa": SEED_SA, "sb": SEED_SB, "sc": SEED_SC}[seed_name]
            key = CONTROL_COLLAPSED if state_id == CONTROL_STATE else short
            usable, correct = counts(state_id) if state_id in spread else (74, 2)
            # The control is keyed by its COLLAPSED id in `digest_of`, and by its
            # full state id everywhere else. Looking it up by state id silently
            # yielded a zero digest, which `restore_probe` correctly refused.
            digest_key = (CONTROL_COLLAPSED if state_id == CONTROL_STATE
                          else state_id)
            record = probe_record(state_id, key, rung, seed_name, seed,
                                  digest_of.get(digest_key, "0" * 64), protocol,
                                  usable=usable, correct=correct)
            (logs / where / f"{record['probe_id']}.json").write_text(
                json.dumps(record, indent=2))
            (historical_admitted if where == "historical"
             else attempt5_reusable).append(f"{key}/{seed_name}")

    (logs / "historical_reuse.json").write_text(json.dumps({
        "reuse_verified": True, "probes_dir_digest": "h" * 64,
        "admitted_reusable_probes": sorted(historical_admitted)}, indent=2))
    (logs / "attempt5_reuse.json").write_text(json.dumps({
        "reuse_verified": True, "probes_dir_digest": "a" * 64,
        "reusable_probes": sorted(attempt5_reusable)}, indent=2))

    return {"logs": logs, "amendment": amendment, "selection": selection,
            "rung1": rung1, "identity": identity,
            "evidence": {
                "stage1_selection_sha256": selection["selection_sha256"],
                "identity_collapse_amendment_sha256": amendment["amendment_sha256"],
                "collapsed_universe_identity": identity,
                "historical_reuse_probes_dir_digest": "h" * 64,
                "attempt5_reuse_probes_dir_digest": "a" * 64,
                "rung1_selection_digest": sha256_json({
                    "selected_searched": rung1["selected_searched"],
                    "auto_advanced_control": rung1["auto_advanced_control"],
                    "advancing": rung1["advancing"]})}}


def make_auth(evidence: dict, tmp_path: Path):
    from aadistill.autoinit.calibration import DOMAIN_BALANCED_V1, REASONING_HEAVY_V2
    from aadistill.autoinit.phase_b_continuation import (
        CONTINUATION_PLAN_V1, ContinuationAuthorization, continuation_source_digest,
    )

    frozen = json.loads(FROZEN_PLAN.read_text())
    science = frozen.get("plan_hash") or frozen["plan"]["plan_hash"]
    return ContinuationAuthorization(
        authorization_id="test-continuation-b", granted_utc="2026-08-29T00:00:00Z",
        granted_by="whole-function test", plan_id=CONTINUATION_PLAN_V1.plan_id,
        plan_hash=CONTINUATION_PLAN_V1.plan_hash, science_plan_hash=science,
        calibration_profile_hashes={p.qualified_id: p.profile_hash
                                    for p in (DOMAIN_BALANCED_V1, REASONING_HEAVY_V2)},
        calibration_content_hashes={p.qualified_id: p.content_sha256
                                    for p in (DOMAIN_BALANCED_V1, REASONING_HEAVY_V2)},
        bound_evidence=dict(evidence), planning_floor_usd=5.4784,
        hard_cap_usd=8.0691, per_launch_hard_usd=8.0691,
        authorized_stages=(0, 1, 3, 4, 5),
        stage_conditions={}, scope_note="test",
        source_digest=continuation_source_digest(REPO)["digest"])


class Detonator:
    """Anything that reaches a search fails the test loudly, not silently."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    def __call__(self, *a, **k):
        self.calls += 1
        raise AssertionError(
            f"the behavioural continuation reached {self.name}. Phase-B stage 1 "
            "is complete and retained; this session must not be able to buy it "
            "again under a ceiling that does not price it.")


def build(tmp_path, monkeypatch, *, tie: bool):
    """The real continuation driver with only the irreducible boundaries stubbed."""
    import phase_a_search

    import autoinit_phase_a_driver as parent

    mod = load_continuation(tmp_path)
    ckpts = build_checkpoints(tmp_path)
    ev = write_evidence(tmp_path, ckpts, tie=tie)

    monkeypatch.setattr(mod, "AMENDMENT", ev["logs"] / "amendment.json")
    monkeypatch.setattr(mod, "STAGE1_SELECTION", ev["logs"] / "stage1_selection.json")
    monkeypatch.setattr(mod, "HISTORICAL_REUSE", ev["logs"] / "historical_reuse.json")
    monkeypatch.setattr(mod, "ATTEMPT5_REUSE", ev["logs"] / "attempt5_reuse.json")
    monkeypatch.setattr(mod, "HISTORICAL_PROBES", ev["logs"] / "historical")
    monkeypatch.setattr(mod, "ATTEMPT5_PROBES", ev["logs"] / "attempt5")
    monkeypatch.setattr(parent, "AUDIT", mod.AUDIT)
    monkeypatch.setattr(parent, "STATUS", mod.STATUS)

    # -- THE GUARANTEE: no search may execute, for the whole run ------------
    beam = Detonator("BeamSearch")
    search = Detonator("run_phase_a_search")
    monkeypatch.setattr(phase_a_search, "run_phase_a_search", search)
    monkeypatch.setattr(parent, "run_phase_a_search", search, raising=False)
    import aadistill.autoinit as autoinit_pkg
    monkeypatch.setattr(autoinit_pkg, "BeamSearch", beam)
    monkeypatch.setattr("aadistill.autoinit.search.BeamSearch", beam)

    # -- the toy target geometry -------------------------------------------
    import phase_a_frozen
    monkeypatch.setattr(phase_a_frozen, "TARGET_GEOMETRY", TARGET_GEOMETRY)

    driver = mod.ContinuationDriver.__new__(mod.ContinuationDriver)
    mod.ContinuationDriver.__init__(driver, Args())
    driver.auth = make_auth(ev["evidence"], tmp_path)

    # -- boundary: frozen-asset gate and vLLM engine probe (no GPU here) ----
    real_gate = parent.PhaseADriver.gate.__get__(driver)

    def gate(name, argv, *, timeout, python="/opt/train/bin/python"):
        if name == "engine_probe":
            (mod.AUDIT / "engine_probe.json").write_text(STAGE3_PROBE.read_text())
            return subprocess.CompletedProcess(argv, 0, "", "")
        if name == "frozen_assets":
            return subprocess.CompletedProcess(argv, 0, "", "")
        return real_gate(name, argv, timeout=timeout, python=sys.executable)

    driver.gate = gate

    # -- boundary: probe training ------------------------------------------
    real_run = subprocess.run
    trained: list[str] = []

    def fake_run(argv, **kwargs):
        argv = [str(a) for a in argv]
        if any(a.endswith("train_stage3.py") for a in argv):
            config = json.loads(Path(argv[argv.index("--config") + 1]).read_text())
            trained.append(config["run_name"])
            out = parent.REPO / f"artifacts/stage3/phase_a/{config['run_name']}"
            ckpts_dir = out / "checkpoints"
            (ckpts_dir / "step000" / "model").mkdir(parents=True, exist_ok=True)
            (ckpts_dir / "latest.txt").write_text("step000\n")
            return subprocess.CompletedProcess(argv, 0, "trained", "")
        return real_run(argv, **kwargs)

    monkeypatch.setattr(parent.subprocess, "run", fake_run)

    # -- boundary: generation + scoring -------------------------------------
    evaluated: list[str] = []

    def battery(label, model_dir, seed):
        evaluated.append(label)
        state = label.split(".")[-2]
        rank = {"fe9683e6a9c7": 0, "85bde4ded2c3": 1, CONTROL_COLLAPSED: 2}.get(state, 0)
        if tie:
            return scored(label, seed, usable=74, correct=2)
        return scored(label, seed, usable=74 + 9 * rank, correct=2 + 6 * rank)

    driver.battery = battery

    # A spy, not a stub: the REAL `restore_probe` runs and its return value is
    # recorded. Asserting against the seeded files instead would read the
    # unmodified imports and could never see a citation.
    real_restore = driver.restore_probe
    cited: list[dict] = []

    def spy(descriptor):
        out = real_restore(descriptor)
        if out is not None:
            cited.append(out)
        return out

    driver.restore_probe = spy
    driver._cited = cited
    driver._trained = trained
    driver._evaluated = evaluated
    driver._detonators = {"BeamSearch": beam, "run_phase_a_search": search}
    driver._evidence = ev
    driver._ckpts = ckpts
    return driver, mod, ev


def run_all(driver, mod, monkeypatch, tmp_path):
    """Every continuation stage, in the driver's own order."""
    monkeypatch.setattr(mod, "REPO", REPO)
    codes = {}
    stages = {0: driver.stage_bind, 1: driver.stage_import, 3: driver.stage3,
              4: driver.stage4, 5: driver.stage5}
    import autoinit_phase_a_driver as parent
    for stage, fn in sorted(stages.items()):
        if stage == 3:
            monkeypatch.setattr(parent, "REPO", tmp_path / "pod_repo")
        codes[stage] = fn()
        if not codes[stage]:
            break
    return codes


@pytest.fixture(scope="module")
def resolved(tmp_path_factory):
    """Path A: the pooled rule separates the finalists and no `sc` is owed."""
    tmp = tmp_path_factory.mktemp("continuation_resolved")
    mp = pytest.MonkeyPatch()
    driver, mod, ev = build(tmp, mp, tie=False)
    try:
        codes = run_all(driver, mod, mp, tmp)
    finally:
        mp.undo()
    return driver, mod, ev, codes


@pytest.fixture(scope="module")
def tied(tmp_path_factory):
    """Path B: the finalists are equivalent, so the conditional `sc` fires."""
    tmp = tmp_path_factory.mktemp("continuation_tied")
    mp = pytest.MonkeyPatch()
    driver, mod, ev = build(tmp, mp, tie=True)
    try:
        codes = run_all(driver, mod, mp, tmp)
    finally:
        mp.undo()
    return driver, mod, ev, codes


# --- A. resolved at sb ------------------------------------------------------

def test_every_continuation_stage_passes_end_to_end(resolved):
    driver, _, _, codes = resolved
    assert codes == {0: True, 1: True, 3: True, 4: True, 5: True}, {
        k: driver.results.get(k, {}).get("reason") for k, v in codes.items() if not v}


def test_the_evidence_universe_is_six_and_exactly_three_finalists_advance(resolved):
    """The boundary this session is built around."""
    driver, _, _, _ = resolved
    assert len(driver.evidence_universe) == 6
    assert len(driver.finalists) == 3
    advancing = {s.state_id for s in driver.finalists}
    assert advancing == {CONTROL_STATE, *FINALIST_LEAVES}
    assert not advancing & set(NON_SURVIVORS), (
        "a searched non-survivor reached the probe stages; rung 1 is complete "
        "and frozen and is not recomputed here")


def test_the_three_non_survivors_never_reach_a_probe(resolved):
    driver, _, _, _ = resolved
    for label in driver._evaluated:
        for loser in NON_SURVIVORS:
            assert loser[:12] not in label, f"{loser[:12]} was probed"


def test_exactly_one_new_sb_probe_is_bought(resolved):
    """The whole economic claim of the continuation, asserted mechanically."""
    driver, _, _, _ = resolved
    assert len(driver._trained) == 1, driver._trained
    assert "fe9683e6a9c7" in driver._trained[0]
    assert ".sb" in driver._trained[0] or "rung2" in driver._trained[0]


def test_no_new_sa_probe_is_ever_bought(resolved):
    driver, _, _, _ = resolved
    assert not [t for t in driver._trained if "rung1" in t or t.endswith(".sa")]


def test_the_reused_sb_probes_are_cited_not_rerun(resolved):
    """Two of the three sb observations are cited; only fe9683's is bought."""
    driver, _, _, _ = resolved
    cited_sb = {r["state_id"] for r in driver._cited if r["seed"] == SEED_SB}
    assert cited_sb == {FINALIST_LEAVES[1], CONTROL_STATE}, cited_sb
    assert FINALIST_LEAVES[0] not in cited_sb, (
        "fe9683 had no sb evidence and must have been newly run")
    assert all(r.get("resumed") for r in driver._cited)


def test_imported_probes_are_cited_under_a_protocol_hash_that_is_not_this_runs(
        resolved):
    """The defect that would have re-bought all eight probes.

    The citable evidence carries two raw `evaluation_protocol_hash` values that
    differ from each other, and from this session's, by host NVIDIA driver patch
    alone — which `generation_runtime_comparability@v2` declares non-material.
    The inherited `restore_probe` compares that field exactly, so it would have
    cited nothing; the override compares student identity and seed and defers
    comparability to the rule.

    Asserted as "differs from this run's" rather than "equals a constant",
    because that is the property that makes reuse work on a pod whose driver
    patch nobody controls.
    """
    driver, _, _, _ = resolved
    cited = [r for r in driver._cited if r.get("imported_evidence")]
    assert cited, "no imported probe was cited at all"
    mine = driver.evaluation_protocol.evaluation_protocol_hash
    citing = {r["evaluation_protocol_hash"] for r in cited}
    assert citing, "no protocol hashes recorded on the cited probes"
    assert mine not in citing, (
        "the cited evidence happens to share this run's raw protocol hash, so "
        "this test would also pass under the inherited strict check and proves "
        "nothing about comparability")


def test_the_divergent_protocol_sa_probe_still_reaches_the_pooled_row(resolved):
    """`fe9683`'s only prior observation carries the OTHER protocol hash.

    Pooling reads the journal directly rather than through `restore_probe`, so
    this is a separate path from the citation above and needs its own assertion.
    """
    driver, mod, _, _ = resolved
    report = json.loads((mod.AUDIT / "phase_a_result.json").read_text())
    row = next(r for r in report["pooled_rows"]
               if r["state_id"] == FINALIST_LEAVES[0])
    assert sorted(row["seeds"]) == sorted([SEED_SA, SEED_SB]), row["seeds"]


def test_no_sc_runs_when_the_finalists_are_separated(resolved):
    driver, _, _, _ = resolved
    assert driver.ev["stages"]["4"]["ran"] is False
    assert not [t for t in driver._trained if "rung3" in t or t.endswith(".sc")]


def test_the_report_is_written_under_the_frozen_selection_rule(resolved):
    driver, mod, _, _ = resolved
    report = json.loads((mod.AUDIT / "phase_a_result.json").read_text())
    assert report["decision_status"]
    assert report["science_plan_hash"] == driver.plan.plan_hash
    assert report["rung1_selection"]["selected_searched"] == list(FINALIST_LEAVES)


# --- B. tie_pending -> sc ---------------------------------------------------

def test_the_tie_break_runs_only_the_missing_sc_probes(tied):
    """`85bde.../sc` exists and is cited; `fe9683` and the control are bought."""
    driver, _, _, codes = tied
    assert codes.get(3) is True, driver.results.get(3, {}).get("reason")
    assert driver.ev["stages"]["4"]["ran"] is True
    sc_trained = [t for t in driver._trained if "rung3" in t or t.endswith(".sc")]
    assert len(sc_trained) <= 2, sc_trained
    assert not any("85bde4ded2c3" in t for t in sc_trained), (
        "85bde had a verified sc and must have been cited, not re-bought")


def test_the_tie_break_never_reaches_a_fourth_seed(tied):
    driver, mod, _, _ = tied
    seeds = {json.loads(p.read_text())["seed"]
             for p in (mod.AUDIT / "probes").glob("*.json")}
    assert seeds <= {SEED_SA, SEED_SB, SEED_SC}, seeds


def test_the_tie_path_reaches_a_terminal_result(tied):
    driver, mod, _, codes = tied
    assert codes.get(5) is True
    report = json.loads((mod.AUDIT / "phase_a_result.json").read_text())
    assert report["decision_status"] in {
        "resolved", "unresolved_equivalence", "winner_selected",
        "control_retained", "tie_pending"}
    assert report["tie_break_ran"] is True


# --- C. evidence mismatch ---------------------------------------------------

@pytest.mark.parametrize("field", [
    "stage1_selection_sha256",
    "identity_collapse_amendment_sha256",
    "collapsed_universe_identity",
    "historical_reuse_probes_dir_digest",
    "attempt5_reuse_probes_dir_digest",
    "rung1_selection_digest",
])
def test_a_moved_identity_fails_before_any_probe_is_reachable(
        field, tmp_path, monkeypatch):
    """Every bound identity, one at a time. None may be discovered late."""
    driver, mod, ev = build(tmp_path, monkeypatch, tie=False)
    driver.auth.bound_evidence = {**ev["evidence"], field: "f" * 64}
    monkeypatch.setattr(mod, "REPO", REPO)
    assert driver.stage_bind() is False, (
        f"{field} moved and stage 0 passed anyway")
    assert not driver._trained, "a probe was bought despite unbound evidence"
    from aadistill.autoinit.recovery import RecoveryAdmissionError
    with pytest.raises(RecoveryAdmissionError):
        driver.stage3()


def test_a_finalist_whose_BYTES_moved_fails_before_it_is_probed(
        tmp_path, monkeypatch):
    """The amendment still hashes correctly; the checkpoint no longer matches it.

    A different failure from the parametrized one above and reached later: the
    grant binds the amendment, and the amendment binds a digest, but only
    re-deriving the identity from the staged bytes can catch a checkpoint that
    was replaced after the grant was issued.
    """
    driver, mod, ev = build(tmp_path, monkeypatch, tie=False)
    monkeypatch.setattr(mod, "REPO", REPO)
    assert driver.stage_bind() is True
    victim = ev["logs"]
    ckpt = driver._ckpts[FINALIST_LEAVES[0]]["dir"] / "config.json"
    config = json.loads(ckpt.read_text())
    config["bos_token_id"] = 999999
    ckpt.write_text(json.dumps(config))
    assert driver.stage_import() is False, (
        "a finalist's bytes changed and the continuation probed it anyway")
    assert not driver._trained
    assert victim.is_dir()


def test_one_state_id_with_two_digests_is_refused_not_merged():
    """Collapse decides on materialized identity, and fails closed on conflict."""
    from aadistill.autoinit.identity_collapse import IdentityCollapseError, collapse

    with pytest.raises(IdentityCollapseError, match="broken identity"):
        collapse([{"state_id": "s", "artifact_digest": "a" * 64, "role": "searched"},
                  {"state_id": "s", "artifact_digest": "b" * 64,
                   "role": "imported_finalist"}])


# --- D. the search is unreachable -------------------------------------------

def test_no_detonator_was_ever_touched(resolved, tied):
    """Neither path reached a search, with both entry points live-armed."""
    for driver, *_ in (resolved, tied):
        for name, det in driver._detonators.items():
            assert det.calls == 0, f"{name} was invoked {det.calls} times"


def test_the_continuation_stage_map_contains_no_search_stage(resolved):
    driver, mod, _, _ = resolved
    import inspect
    source = inspect.getsource(mod.ContinuationDriver.run)
    assert "self.stage1" not in source
    assert "run_search" not in source


@pytest.mark.parametrize("method", ["stage1", "run_search"])
def test_the_inherited_search_entry_points_refuse(method, tmp_path, monkeypatch):
    from aadistill.autoinit.recovery import RecoveryAdmissionError

    driver, _, _ = build(tmp_path, monkeypatch, tie=False)
    with pytest.raises(RecoveryAdmissionError, match="continuation"):
        getattr(driver, method)()


def test_the_authorization_cannot_be_made_to_permit_a_search(tmp_path, monkeypatch):
    """`runs_search` is False BY TYPE — there is no field to set."""
    from dataclasses import fields

    from aadistill.autoinit.phase_b_continuation import ContinuationAuthorization

    driver, _, ev = build(tmp_path, monkeypatch, tie=False)
    assert driver.auth.runs_search is False
    assert "runs_search" not in {f.name for f in fields(ContinuationAuthorization)}
    with pytest.raises(AttributeError):
        driver.auth.runs_search = True


def test_the_parent_driver_contract_is_fully_satisfied():
    """Every `self.auth.*` the inherited machinery calls must exist.

    `require_science_plan` was missing and the inherited stage 0 calls it
    unconditionally. Enumerated from the parent's source rather than listed, so
    a method the parent starts calling tomorrow fails here and not on a pod.
    """
    import re

    from aadistill.autoinit.phase_b_continuation import ContinuationAuthorization

    source = (REPO / "scripts/pod/autoinit_phase_a_driver.py").read_text()
    required = set(re.findall(r"self\.auth\.([a-z_]+)", source))
    assert required, "the probe found no auth calls; it is broken"
    missing = sorted(required - set(dir(ContinuationAuthorization)))
    assert not missing, f"the inherited driver calls {missing}, which this type lacks"


# --- pricing binds to the executable ----------------------------------------

@pytest.mark.skipif(not PRICING.is_file(), reason="pricing has not been derived")
def test_the_priced_probe_inventory_is_what_the_executable_actually_buys(
        resolved, tied):
    """Two independent derivations of the same number, required to agree.

    The pricing script reads the reuse records and the amendment. The driver
    reaches its purchases through `probe_configs`, `restore_probe` and the frozen
    pooled rule, which share no code with the pricing. A ceiling that priced
    fewer probes than the executable buys is the failure mode that matters, so
    the agreement is asserted rather than assumed.
    """
    priced = json.loads(PRICING.read_text())
    resolved_driver, *_ = resolved
    tied_driver, *_ = tied

    assert priced["evidence"]["missing_sb"] == ["fe9683e6a9c7"]
    assert set(priced["evidence"]["missing_sc_worst_case"]) == {
        "fe9683e6a9c7", CONTROL_COLLAPSED}
    assert priced["total"]["low_probes"] == len(resolved_driver._trained), (
        "the floor prices a different number of probes than the resolved-at-sb "
        "path actually buys")
    assert priced["total"]["hard_probes"] == len(tied_driver._trained), (
        "the CEILING prices a different number of probes than the tie path "
        "actually buys; a session can exceed a ceiling derived this way")
    assert priced["total"]["hard_usd"] <= 8.0691


# --- the launcher's operational envelope ------------------------------------

def test_the_poll_lifetime_is_derived_from_THIS_sessions_plan():
    """A launcher that stops polling leaves a pod alive and billing.

    The inverse error is just as real: polling for the full Phase-B lifetime
    over a session bounded at ~5 h means a HUNG pod bills for 32 h before
    anything gives up, which silently converts an $8.07 ceiling into a much
    larger exposure. Asserted as a relation to the session's own hard-terminate
    bound rather than against a constant.
    """
    import autoinit_continuation_b_launch as L
    from autoinit_phase_b_launch import phase_b_poll_limit_minutes

    args = L.build_parser().parse_args(
        ["--scr", "/tmp/x", "--session-commit", "d" * 40, "--bundle", "b.bundle"])
    mine = L.continuation_poll_limit_minutes(args)
    plan = L.continuation_budget(args).plan(price_per_hour=args.max_price,
                                            authorized_usd=float("inf"))
    assert mine > plan.hard_terminate_minutes, (
        "the launcher would stop polling before its own session could finish")
    assert mine < phase_b_poll_limit_minutes(args), (
        "the continuation inherited the full Phase-B polling lifetime, which is "
        "priced around a 16.5 h search this session does not run")


def test_the_continuation_budget_strips_the_search_and_its_reserves():
    import autoinit_continuation_b_launch as L

    args = L.build_parser().parse_args(
        ["--scr", "/tmp/x", "--session-commit", "d" * 40, "--bundle", "b.bundle"])
    budget = L.continuation_budget(args)
    names = " ".join(p.name for p in budget.other_phases)
    assert "search" not in names and "beam" not in names
    assert budget.soft_stop_reserves == ()
    assert budget.arms == args.rung2_probes + args.tie_break_probes == 3


# --- executable-source provenance -------------------------------------------

def test_the_frozen_source_set_IS_the_real_import_closure():
    """Provenance answers "what is loaded", not "what is permitted".

    The first version of this set was curated by hand around the second
    question. It listed 25 files: it omitted eight that the continuation
    genuinely loads — `search.py`, `ranking.py` and all six operator modules,
    which the `aadistill.autoinit` package `__init__` imports for every consumer
    — and included `phase_b.py`, which is never loaded at all. A digest that
    omits loaded files lets them change under a grant that claims to pin the
    executable.

    Derived in a subprocess and compared, so adding an import without updating
    the set fails here rather than under a grant.
    """
    from aadistill.autoinit.phase_b_continuation import (
        CONTINUATION_RUNTIME_ONLY_FILES, CONTINUATION_SOURCE_FILES_V2,
        derive_continuation_closure,
    )

    derived = derive_continuation_closure(REPO)
    assert derived == tuple(sorted(CONTINUATION_SOURCE_FILES_V2)), (
        "declared but not loaded: "
        f"{sorted(set(CONTINUATION_SOURCE_FILES_V2) - set(derived))}; "
        "loaded but not declared: "
        f"{sorted(set(derived) - set(CONTINUATION_SOURCE_FILES_V2))}")
    for runtime_only in CONTINUATION_RUNTIME_ONLY_FILES:
        assert runtime_only in CONTINUATION_SOURCE_FILES_V2
        assert (REPO / runtime_only).is_file()


def test_the_loaded_modules_a_search_lives_in_are_covered_by_the_digest():
    """Explicitly: these ARE in the digest, and that is correct."""
    from aadistill.autoinit.phase_b_continuation import CONTINUATION_SOURCE_FILES_V2

    for loaded in ("src/aadistill/autoinit/search.py",
                   "src/aadistill/autoinit/ranking.py",
                   "src/aadistill/autoinit/operators/depth.py",
                   "scripts/pod/autoinit_phase_a_driver.py"):
        assert loaded in CONTINUATION_SOURCE_FILES_V2, loaded


def test_only_the_known_neutralized_file_holds_a_search_call_site():
    """A call site appearing anywhere else fails, including in a library."""
    from aadistill.autoinit.phase_b_continuation import (
        CONTINUATION_OWN_PATH_FILES, KNOWN_NEUTRALIZED_SEARCH_CALL_SITES,
        search_call_site_owners,
    )

    assert search_call_site_owners(REPO, files=CONTINUATION_OWN_PATH_FILES) == ()
    assert search_call_site_owners(REPO) == tuple(
        sorted(KNOWN_NEUTRALIZED_SEARCH_CALL_SITES))


def test_the_neutralized_call_site_is_never_bound_into_the_stage_map(resolved):
    """`PhaseADriver.stage1` holds the call site; the map never reaches it."""
    _, mod, _, _ = resolved
    driver = mod.ContinuationDriver
    assert driver.stage1 is not mod.PhaseADriver.stage1, (
        "the continuation did not override the method holding the search call site")


def test_importing_this_driver_does_not_rebind_the_shared_status_global():
    """Hidden global state, caught by another session's test.

    `mark()` reads `autoinit_phase_a_driver.STATUS`. Binding it at import time
    means importing this module silently redirects Phase B's markers too — which
    is exactly what happened: `test_phase_b_driver_and_launcher.py` failed in the
    full suite and passed alone. Rebinding now happens on entry.
    """
    import importlib.util

    import autoinit_phase_a_driver as parent

    before = parent.STATUS
    spec = importlib.util.spec_from_file_location(
        "continuation_b_import_probe",
        REPO / "scripts/pod/autoinit_continuation_b_driver.py")
    probe = importlib.util.module_from_spec(spec)
    sys.modules["continuation_b_import_probe"] = probe
    spec.loader.exec_module(probe)
    try:
        assert parent.STATUS == before, (
            "importing the continuation driver rebound the shared STATUS global; "
            "any other session importing it in the same process now writes its "
            "markers where this launcher polls")
    finally:
        sys.modules.pop("continuation_b_import_probe", None)
        parent.STATUS = before
