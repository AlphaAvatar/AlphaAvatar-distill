"""`c1_confirmation_scoring@v1`: the numbers must not move, and the pins must hold.

Two jobs.

**The admission gate.** C1 needs a new scoring *binding* — the frozen scorer's
battery pins are module constants and its result builder requires a `metrics` key
the C1 manifest does not carry. It must not acquire new *semantics*, because the
C0 power analysis and the SESOI were computed under `recovery_search_scoring@v2`.
The C1 scorer imports every rule and restates only the loop over sets, so the
restatement is checked against real retained `recovery_search_v2` generations
rather than reviewed. The mutation tests below break each load-bearing rule in
turn and require the gate to notice; a gate that cannot fail is not a gate.

**The production path.** The C1 CLI must accept the frozen C1 battery, must not
ask it for `metrics`, and must refuse every way of being pointed at something
else — a moved hash, a wrong set, a short battery, a duplicate id, a foreign id,
a missing set.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "autoinit"))

import score_c1_confirmation as C1S  # noqa: E402
import verify_c1_scoring_equivalence as EQ  # noqa: E402

from aadistill.autoinit.c1_scoring import (  # noqa: E402
    C1_BATTERY_CONTENT_SHA256, C1_BATTERY_MANIFEST_SHA256, C1_BATTERY_SETS,
    C1_METRIC_CONTRACT, C1_N_PROMPTS, C1_N_SCORABLE_PROMPTS,
    C1_SCORING_FILES_V1, C1ScoringError, c1_scoring_contract,
    validate_c1_battery,
)

C1_BATTERY = REPO / "artifacts/stage3/c1_confirmation_v1"
SCORER = REPO / "scripts/autoinit/score_c1_confirmation.py"
HISTORICAL = EQ.find_generations()


# --- identity ---------------------------------------------------------------

def test_the_frozen_assets_are_untouched():
    """Neither historical asset may move because C1 needed a scorer."""
    from aadistill.autoinit.recovery import recovery_scoring_contract

    assert recovery_scoring_contract(REPO)["digest"] == (
        "808080a7c5d88d5a66760fd0d7eeabc5451c096ad0819f8c5663a0b8224660be")
    manifest, manifest_sha = C1S.battery_manifest(C1_BATTERY)
    assert manifest_sha == C1_BATTERY_MANIFEST_SHA256
    assert manifest["content_sha256"] == C1_BATTERY_CONTENT_SHA256


def test_the_c1_contract_is_a_new_name_not_a_new_metric():
    c = c1_scoring_contract(REPO)
    assert c["contract"] == "c1_confirmation_scoring@v1"
    assert c["semantic_parent"] == "recovery_search_scoring@v2"
    assert c["digest"] != c["semantic_parent_digest"]


def test_the_c1_closure_covers_the_three_files_v2_omits():
    """V2 omits three files that decide numbers. Do not repeat the hole."""
    from aadistill.autoinit.recovery import RECOVERY_SCORING_FILES_V2

    holes = {"scripts/autoinit/audit_tool_scoring.py",
             "src/aadistill/data/tools.py",
             "src/aadistill/data/verify.py"}
    assert not (holes & set(RECOVERY_SCORING_FILES_V2))   # the historical hole
    assert holes <= set(C1_SCORING_FILES_V1)              # not repeated here
    assert "scripts/autoinit/score_c1_confirmation.py" in C1_SCORING_FILES_V1
    assert "src/aadistill/autoinit/c1_scoring.py" in C1_SCORING_FILES_V1


def test_every_declared_scoring_source_exists_and_a_missing_one_refuses():
    c1_scoring_contract(REPO)
    with pytest.raises(C1ScoringError):
        c1_scoring_contract(REPO, files=(*C1_SCORING_FILES_V1, "src/nope.py"))


def test_the_metric_contract_pins_the_frozen_denominators():
    assert C1_METRIC_CONTRACT["correct_overall"]["denominator_value"] == 850
    assert C1_METRIC_CONTRACT["usable_rollout_rate"]["denominator_value"] == 950
    assert C1_METRIC_CONTRACT["behaviour_only_sets"] == ["code"]
    assert len(C1_METRIC_CONTRACT["scorable_sets"]) == 6
    assert "no_weighted_scalar" in C1_METRIC_CONTRACT


# --- the historical numerical-equivalence gate ------------------------------

@pytest.mark.skipif(not HISTORICAL, reason="no retained historical generations")
def test_the_committed_equivalence_record_says_identical():
    rec = json.loads((REPO / "logs/phase_c1_scoring_equivalence.json").read_text())
    assert rec["verdict"] == "IDENTICAL"
    assert rec["total_differences"] == 0
    assert rec["n_cases"] >= 3
    assert all(c["identical"] for c in rec["cases"])


@pytest.mark.skipif(not HISTORICAL, reason="no retained historical generations")
def test_one_historical_probe_scores_identically_through_both_paths(tmp_path):
    """The gate itself, live, on one real probe."""
    gen = HISTORICAL[0]
    manifest, _ = C1S.battery_manifest(EQ.HISTORICAL_BATTERY)
    frozen, frozen_rows = EQ.frozen_scores(gen, gen.name, 20260726, tmp_path)
    c1 = C1S.score_battery(
        battery=EQ.HISTORICAL_BATTERY, gen_dir=gen, label=gen.name, seed=20260726,
        sets=manifest["sets"], scorable_sets=set(manifest["scorable_sets"]),
        behaviour_only=set(manifest["behaviour_only_sets"]))
    assert EQ.compare(frozen, frozen_rows, c1, c1["per_sample"]) == []


def _mutation_detected(monkeypatch, tmp_path, mutate) -> str:
    """Did breaking a rule change observable behaviour? A raise counts.

    The first mutation attempted here returned no differences and looked inert.
    It was not: with `TOOL_STRUCTURAL_GATE` emptied, the row loop stops writing
    the three structural fields and the frozen `summarize` — which still iterates
    its own unpatched tuple — raises `KeyError`. A mutation caught by a crash is
    caught; asserting only on the diff list would have recorded a real detection
    as a blind spot.
    """
    gen = HISTORICAL[0]
    manifest, _ = C1S.battery_manifest(EQ.HISTORICAL_BATTERY)
    frozen, frozen_rows = EQ.frozen_scores(gen, gen.name, 20260726, tmp_path)
    mutate(monkeypatch)
    try:
        c1 = C1S.score_battery(
            battery=EQ.HISTORICAL_BATTERY, gen_dir=gen, label=gen.name,
            seed=20260726, sets=manifest["sets"],
            scorable_sets=set(manifest["scorable_sets"]),
            behaviour_only=set(manifest["behaviour_only_sets"]))
    except Exception as exc:                                   # noqa: BLE001
        return f"raised {type(exc).__name__}: {exc}"
    diffs = EQ.compare(frozen, frozen_rows, c1, c1["per_sample"])
    return "; ".join(diffs[:3])


@pytest.mark.skipif(not HISTORICAL, reason="no retained historical generations")
def test_mutation_dropping_the_tool_structural_gate_is_caught(monkeypatch, tmp_path):
    """A tool rollout that emits unparseable bytes must not count as usable.

    The chosen probe has 9 tool rows where all five generic components pass and
    the structural gate fails, so the rule is load-bearing on this evidence.
    """
    detected = _mutation_detected(
        monkeypatch, tmp_path, lambda mp: mp.setattr(C1S, "TOOL_STRUCTURAL_GATE", ()))
    assert detected, "removing the tool structural gate changed nothing"


@pytest.mark.skipif(not HISTORICAL, reason="no retained historical generations")
def test_the_equivalence_gate_cannot_cover_correct_implies_usable(monkeypatch,
                                                                  tmp_path):
    """A real coverage hole, asserted rather than assumed.

    `correct_but_unusable` is **0 on all 15 retained probes**: no historical
    rollout was ever scored correct while being unusable, so the implication
    never fires and the equivalence gate cannot discriminate it. Breaking the
    rule therefore produces no difference — which is a fact about the evidence,
    not a licence. The rule is covered directly below instead, and the hole is
    recorded in `logs/phase_c1_scoring_equivalence.json`.
    """
    def mutate(mp):
        def naive(*, usable, scorer_correct, scorable=True):
            return {"usable": bool(usable), "scorable": bool(scorable),
                    "scorer_correct": bool(scorer_correct),
                    "correct": bool(scorer_correct) and bool(scorable),
                    "correct_but_unusable": False}
        mp.setattr(C1S, "score_recovery_row", naive)
    assert _mutation_detected(monkeypatch, tmp_path, mutate) == ""


def test_correct_implies_usable_is_enforced_by_the_frozen_row_contract():
    """C1 imports this function unmodified; this is where the rule is covered."""
    from aadistill.autoinit.recovery import score_recovery_row

    unusable = score_recovery_row(usable=False, scorer_correct=True, scorable=True)
    assert unusable["correct"] is False
    assert unusable["correct_but_unusable"] is True
    assert score_recovery_row(usable=True, scorer_correct=True,
                              scorable=True)["correct"] is True
    assert score_recovery_row(usable=True, scorer_correct=True,
                              scorable=False)["correct"] is False


@pytest.mark.skipif(not HISTORICAL, reason="no retained historical generations")
def test_code_is_excluded_by_set_membership_not_by_the_scorer(monkeypatch, tmp_path):
    """Forcing a `code` verdict cannot reach the rows; membership is the guard."""
    real = C1S.scorer_correct
    detected = _mutation_detected(
        monkeypatch, tmp_path,
        lambda mp: mp.setattr(
            C1S, "scorer_correct",
            lambda name, rec, s: (True, {"forced": True}) if name == "code"
            else real(name, rec, s)))
    assert detected == "", (
        "a forced code scorer changed a number, which would mean code is "
        "reaching the correctness path")


# --- the production CLI on the frozen C1 battery ----------------------------

def _c1_generations(dest: Path, *, drop_set=None, drop_one=None,
                    duplicate=False, foreign=False) -> Path:
    """Synthetic C1 generations built from REAL historical record shapes."""
    dest.mkdir(parents=True, exist_ok=True)
    src_dir = HISTORICAL[0] if HISTORICAL else None
    for name in C1_BATTERY_SETS:
        if name == drop_set:
            continue
        template = None
        if src_dir is not None:
            p = src_dir / f"{name}.generations.jsonl"
            if p.is_file():
                template = json.loads(p.open().readline())
        if template is None:                      # no historical evidence here
            template = {"raw": "<think>\n</think>\nno.", "think_preopened": True,
                        "natural_termination": True, "context_limit_reached": False,
                        "degeneration_triggered": False, "generated_tokens": 4,
                        "stop_reason": "eos"}
        ids = [json.loads(line)["id"]
               for line in (C1_BATTERY / f"{name}.jsonl").open() if line.strip()]
        if drop_one == name:
            ids = ids[:-1]
        with (dest / f"{name}.generations.jsonl").open("w") as f:
            for i in ids:
                f.write(json.dumps({**template, "id": i}) + "\n")
            if duplicate and name == "gsm8k":
                f.write(json.dumps({**template, "id": ids[0]}) + "\n")
            if foreign and name == "gsm8k":
                f.write(json.dumps({**template, "id": "not-in-this-battery"}) + "\n")
    return dest


def _run(gen: Path, out: Path, extra=()) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCORER), "--generations", str(gen),
         "--label", "probe", "--seed", "1635674081", "--out", str(out),
         "--per-sample", str(out.with_suffix(".jsonl")), *extra],
        capture_output=True, text=True, timeout=1800,
        env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"})


def test_the_production_scorer_accepts_the_frozen_c1_battery(tmp_path):
    gen = _c1_generations(tmp_path / "gen")
    out = tmp_path / "result.json"
    rc = _run(gen, out)
    assert rc.returncode == 0, rc.stdout + rc.stderr
    result = json.loads(out.read_text())
    assert result["schema"] == "aadistill.autoinit.c1_confirmation_result/v1"
    assert result["scoring_contract"]["contract"] == "c1_confirmation_scoring@v1"
    assert result["n"] == C1_N_PROMPTS == 950
    assert result["n_scorable"] == C1_N_SCORABLE_PROMPTS == 850
    assert result["battery"]["content_sha256"] == C1_BATTERY_CONTENT_SHA256
    assert result["capability_schema_enforced"] is True
    assert sorted(result["per_capability"]) == sorted(
        C1_METRIC_CONTRACT["scorable_sets"])
    rows = [json.loads(x) for x in out.with_suffix(".jsonl").open() if x.strip()]
    assert len(rows) == 950
    assert sum(1 for r in rows if r["scorable"]) == 850
    assert all(not r["scorable"] for r in rows if r["set"] == "code")


def test_the_missing_metrics_key_is_expected_and_does_not_fail(tmp_path):
    """The battery defines the examples; the contract defines the semantics."""
    manifest, _ = C1S.battery_manifest(C1_BATTERY)
    assert "metrics" not in manifest
    rc = _run(_c1_generations(tmp_path / "gen"), tmp_path / "r.json")
    assert rc.returncode == 0, rc.stdout + rc.stderr
    result = json.loads((tmp_path / "r.json").read_text())
    assert result["battery"]["has_metrics_key"] is False
    assert result["metric_contract"]["defined_by"] == "source, not the battery manifest"


def test_code_never_enters_the_correctness_denominator(tmp_path):
    rc = _run(_c1_generations(tmp_path / "gen"), tmp_path / "r.json")
    assert rc.returncode == 0, rc.stdout + rc.stderr
    result = json.loads((tmp_path / "r.json").read_text())
    assert "code" not in result["per_capability"]
    assert result["per_set"]["code"]["scorable"] is False
    assert result["n_scorable"] == result["n"] - result["per_set"]["code"]["n"]


@pytest.mark.parametrize("kwargs,expect", [
    ({"drop_set": "tool"}, "no generations for"),
    ({"drop_one": "rag"}, "have no generation"),
    ({"duplicate": True}, "duplicate generation"),
    ({"foreign": True}, "is not in the frozen set"),
])
def test_the_scorer_refuses_an_incomplete_or_foreign_generation_set(
        tmp_path, kwargs, expect):
    rc = _run(_c1_generations(tmp_path / "gen", **kwargs), tmp_path / "r.json")
    assert rc.returncode != 0
    assert expect in (rc.stdout + rc.stderr)


# --- the pins, mutated ------------------------------------------------------

def _manifest(**over) -> dict:
    m, _ = C1S.battery_manifest(C1_BATTERY)
    return {**m, **over}


def test_a_moved_manifest_hash_refuses():
    with pytest.raises(C1ScoringError, match="manifest"):
        validate_c1_battery(_manifest(), manifest_sha256="0" * 64)


def test_a_moved_content_hash_refuses():
    with pytest.raises(C1ScoringError, match="content"):
        validate_c1_battery(_manifest(content_sha256="0" * 64),
                            manifest_sha256=C1_BATTERY_MANIFEST_SHA256)


def test_a_changed_set_count_refuses():
    m = _manifest()
    m["sets"] = {**m["sets"], "gsm8k": {**m["sets"]["gsm8k"], "n": 149}}
    with pytest.raises(C1ScoringError, match="n=149"):
        validate_c1_battery(m, manifest_sha256=C1_BATTERY_MANIFEST_SHA256)


def test_a_changed_scorable_membership_refuses():
    with pytest.raises(C1ScoringError, match="scorable sets"):
        validate_c1_battery(_manifest(scorable_sets=["gsm8k"]),
                            manifest_sha256=C1_BATTERY_MANIFEST_SHA256)


def test_code_must_be_the_only_behaviour_only_set():
    with pytest.raises(C1ScoringError, match="behaviour-only"):
        validate_c1_battery(_manifest(behaviour_only_sets=["code", "tool"]),
                            manifest_sha256=C1_BATTERY_MANIFEST_SHA256)


def test_a_missing_set_refuses():
    m = _manifest()
    m["sets"] = {k: v for k, v in m["sets"].items() if k != "code"}
    with pytest.raises(C1ScoringError, match="battery sets"):
        validate_c1_battery(m, manifest_sha256=C1_BATTERY_MANIFEST_SHA256)


def test_pointing_the_c1_pins_at_recovery_search_v2_refuses():
    """Mutation: the historical battery must not satisfy the C1 pins."""
    manifest, manifest_sha = C1S.battery_manifest(EQ.HISTORICAL_BATTERY)
    with pytest.raises(C1ScoringError):
        validate_c1_battery(manifest, manifest_sha256=manifest_sha)


def test_the_scorer_refuses_the_historical_battery_end_to_end(tmp_path):
    rc = _run(_c1_generations(tmp_path / "gen"), tmp_path / "r.json",
              extra=("--battery", str(EQ.HISTORICAL_BATTERY)))
    assert rc.returncode != 0
    assert "frozen C1 confirmation battery" in (rc.stdout + rc.stderr)


def test_the_frozen_scorer_is_never_pointed_at_the_c1_battery():
    """The driver and the launcher must invoke only the C1 scorer for C1."""
    for rel in ("scripts/pod/autoinit_c1_driver.py",):
        p = REPO / rel
        if not p.is_file():
            continue
        src = p.read_text()
        assert "score_recovery_search.py" not in src, rel
