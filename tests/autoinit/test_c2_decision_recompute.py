"""The archive-side recompute of the C2 verdict, driven for real.

This exists because the recompute is load-bearing for C2's acceptance: the
verdict attempt13 produced could not be established from committed evidence,
and this script is what establishes it once probe 11's rows are reconstructed.
Its refusal path runs on the real archive today; its SUCCESS path cannot, until
those rows exist. That is exactly the shape of code that has failed on a paid
pod repeatedly in this campaign, so it is driven here against a synthetic
archive instead of being trusted.

Both directions are fixtured. A recompute that hardcoded NO_GO would satisfy a
NO_GO-only test, and NO_GO is the outcome under review.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments.phase_c2 import behavioural_decision as BD  # noqa: E402
from experiments.phase_c2 import behavioural_schedule as SCH  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "recompute_c2", REPO / "scripts/autoinit/recompute_c2_behavioural_decision.py")
RC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RC)

N_PROMPTS, N_SCORABLE = 950, 850
#: THE FROZEN STRATA AND THEIR REAL SIZES, read off an archived probe rather
#: than invented: `decision_inputs` validates every scorable row's set against
#: the frozen list, and a fixture with made-up stratum names fails inside the
#: production code for the wrong reason. The unscorable hundred are `code`,
#: which is deliberately NOT a scorable stratum.
SCORABLE_STRATA = (("gsm8k", 150), ("math_verified", 150), ("multihop", 150),
                   ("rag", 150), ("knowledge", 150), ("tool", 100))
UNSCORABLE_STRATUM, N_UNSCORABLE = "code", 100
CANDIDATE = "1a2b5b030e7e4e202fda3a810ed53c5f"


def _rows(seed: int, correct_rate: float, usable_rate: float) -> list[dict]:
    """One probe's rows in the schema `decision_inputs` actually reads."""
    rng = random.Random(seed)
    out = []
    #: Prompt ids are STABLE across probes and strata-qualified, as the real
    #: ones are (`gsm8k-test-00051`). `decision_inputs` refuses a prompt set
    #: that differs between probes, so a per-probe id scheme would fail there.
    for stratum, n in SCORABLE_STRATA:
        for j in range(n):
            out.append({
                "id": f"{stratum}-test-{j:05d}", "set": stratum,
                "scorable": True,
                "correct": rng.random() < correct_rate,
                "usable": rng.random() < usable_rate,
            })
    for j in range(N_UNSCORABLE):
        out.append({
            "id": f"mbpp-test-{j:05d}", "set": UNSCORABLE_STRATUM,
            "scorable": False, "correct": False,
            "usable": rng.random() < usable_rate,
        })
    assert len(out) == N_PROMPTS
    assert sum(1 for r in out if r["scorable"]) == N_SCORABLE
    return out


#: The protocol identities a real scorer writes into `result.json`. The
#: recompute builds `confirm`'s `protocols` from these, and the gate refuses a
#: field that cannot show they match -- so a fixture omitting them is refused
#: for a reason that has nothing to do with the arithmetic under test.
BATTERY = {"artifact": "c1_confirmation_v1", "content_sha256": "a" * 64}
SCORING = {"contract": "c1_confirmation_scoring@v1", "digest": "b" * 64}
METRIC = {"contract": "c1_confirmation_scoring@v1", "schema": "CAPABILITY_V1"}
FINGERPRINT = "c" * 64


def _archive(root: Path, *, treatment_correct: float, incumbent_correct: float,
             usable: float = 0.95, fingerprints: dict | None = None) -> Path:
    """A complete six-probe confirmation archive, laid out as the real one is.

    Probes are spread across attempt directories on purpose: the real campaign
    has the candidate's third seed under `attempt12` and the incumbent's under
    `attempt13`, and the walker has to collapse them into one experiment.

    `fingerprints` overrides individual probes' generation protocol, keyed
    `(arm, seed)`, to build the mixed-protocol field the real campaign has.
    """
    store = root / "c2-behavioural-12probe-v1"
    rule = BD.decision_rule(REPO)
    for i, seed in enumerate(sorted(rule.seeds)):
        for arm, rate in ((SCH.ANCHOR, incumbent_correct),
                          (CANDIDATE, treatment_correct)):
            attempt = f"attempt{5 + (i * 2) + (0 if arm == SCH.ANCHOR else 1)}"
            pid = f"confirmation.{arm}.s{seed}"
            d = store / attempt / pid
            d.mkdir(parents=True, exist_ok=True)
            #: NOT `hash(arm)`: `hash` is per-process randomized, so a
            #: fixture seeded from it draws different rows every run and
            #: a directional assertion becomes a coin flip. A stable
            #: digest keeps the same archive for the same inputs.
            arm_salt = int(hashlib.sha256(arm.encode()).hexdigest()[:8], 16)
            rows = _rows(seed + arm_salt % 1000, rate, usable)
            (d / "per_sample.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rows))
            scorable = [r for r in rows if r["scorable"]]
            (d / "result.json").write_text(json.dumps({
                "probe_id": pid, "rung": "confirmation", "arm": arm,
                "seed": seed,
                "correct_overall": sum(r["correct"] for r in scorable) / len(scorable),
                "usable_rollout_rate": sum(r["usable"] for r in rows) / len(rows),
                "battery": dict(BATTERY),
                "scoring_contract": dict(SCORING),
                "metric_contract": dict(METRIC),
                "generation_protocol_fingerprint": (fingerprints or {}).get(
                    (arm, seed), FINGERPRINT),
            }, indent=1) + "\n")
            (d / "probe_record.json").write_text(json.dumps({
                "probe_id": pid, "rung": "confirmation", "arm": arm,
                "seed": seed, "complete": True,
            }, indent=1) + "\n")
    return store


def test_it_refuses_a_partial_field_rather_than_deciding_on_it(tmp_path):
    """Five probes is not the preregistered experiment.

    This is the state the real archive was in: eleven complete probes and one
    scored on a pod whose rows were never collected. A recompute that averaged
    the five it could read would have produced a number and concealed the gap.
    """
    store = _archive(tmp_path, treatment_correct=0.30, incumbent_correct=0.30)
    victim = next(iter(sorted(store.glob(f"*/confirmation.{CANDIDATE}.s*"))))
    (victim / "per_sample.jsonl").unlink()

    out = tmp_path / "decision.json"
    rc = RC.main(["--store", str(store), "--out", str(out)])
    assert rc == 2, "it decided on a partial field"
    assert not out.exists(), "it wrote a record for a field it could not read"


def test_a_worse_candidate_recomputes_to_NO_GO(tmp_path):
    """The outcome under review, and the full criteria the rule actually uses.

    `ucb_one_sided < SESOI` is the NO_GO criterion; the LCB is a GO criterion.
    The record has to carry both, because a reader checking this verdict needs
    the bound the rule read and not a neighbouring one.
    """
    store = _archive(tmp_path, treatment_correct=0.26, incumbent_correct=0.34)
    out = tmp_path / "decision.json"
    assert RC.main(["--store", str(store), "--out", str(out)]) == 0

    rec = json.loads(out.read_text())
    assert rec["terminal_state"] == "NO_GO"
    d = rec["decision"]
    assert d["ucb_one_sided"] < rec["rule"]["sesoi"], (
        "NO_GO was reached without the criterion that defines it")
    for field in ("delta", "lcb_one_sided", "ucb_one_sided", "sesoi",
                  "criteria"):
        assert field in d, field
    for field in ("per_seed_delta_correct", "usable_pooled_delta",
                  "usable_per_seed_delta", "catastrophic_violations",
                  "bootstrap", "bootstrap_seed_used", "probe_fingerprints"):
        assert field in rec, field
    assert len(rec["per_seed_delta_correct"]) == 3
    assert rec["bootstrap_seed_used"] == BD.decision_rule(REPO).bootstrap_seed
    assert len(rec["probe_fingerprints"]) == 6
    for f in rec["probe_fingerprints"].values():
        assert f["n_rows"] == N_PROMPTS
        assert len(f["per_sample_sha256"]) == 64


def test_a_clearly_better_candidate_recomputes_to_GO(tmp_path):
    """The other direction, so the recompute is not a NO_GO generator.

    Without this, a script that returned NO_GO unconditionally would pass every
    other test in this file -- and NO_GO is precisely the answer under review.
    """
    store = _archive(tmp_path, treatment_correct=0.50, incumbent_correct=0.20)
    out = tmp_path / "decision.json"
    assert RC.main(["--store", str(store), "--out", str(out)]) == 0

    rec = json.loads(out.read_text())
    d = rec["decision"]
    assert rec["terminal_state"] == "GO", (
        f"a candidate 30 points better did not reach GO: {d}")
    assert d["lcb_one_sided"] > 0 and d["delta"] >= rec["rule"]["sesoi"]
    assert sum(1 for x in rec["per_seed_delta_correct"] if x > 0) >= 2


def test_the_walker_prefers_the_copy_that_carries_the_rows(tmp_path):
    """One probe, two attempt directories: the scored one wins.

    Probe 11 exists under `attempt12`, which trained it, and a session that
    scores it writes its evidence beside that same copy. But a probe CAN appear
    twice -- trained under one attempt and secured under another -- and reading
    the rowless copy would make a complete archive look incomplete.
    """
    store = _archive(tmp_path, treatment_correct=0.30, incumbent_correct=0.30)
    rich = next(iter(sorted(store.glob(f"*/confirmation.{CANDIDATE}.s*"))))
    poor = store / "attempt99" / rich.name
    poor.mkdir(parents=True)
    (poor / "probe_record.json").write_text((rich / "probe_record.json").read_text())

    found = RC.confirmation_probes(store)
    assert len(found) == 6, sorted(found)
    chosen = found[rich.name]
    assert chosen["has_rows"] and chosen["has_result"], (
        f"the walker took the rowless copy at {chosen['attempt']}")


def test_it_refuses_the_mixed_protocol_field_the_REAL_archive_has(tmp_path):
    """C2's own shape: six readable probes, three generation protocols.

    This is the defect that closed C2 without promotion. Every row file is
    valid, every probe is complete, the walker finds all six and the arithmetic
    runs — so nothing in the recompute before the gate can tell that the third
    seed's pair was measured under two different protocols. The paired interval
    over such a field has no estimand, and a script that printed a verdict for
    it would be publishing a number for a comparison nobody made.

    Refusal must be a clean exit with nothing written, not a traceback: this
    path runs against the real archive.
    """
    store = _archive(tmp_path, treatment_correct=0.26, incumbent_correct=0.34,
                     fingerprints={
                         (SCH.ANCHOR, sorted(BD.decision_rule(REPO).seeds)[2]):
                             "d" * 64,
                         (CANDIDATE, sorted(BD.decision_rule(REPO).seeds)[2]):
                             "e" * 64,
                     })
    out = tmp_path / "decision.json"
    rc = RC.main(["--store", str(store), "--out", str(out)])
    assert rc == 3, f"a mixed-protocol field produced exit {rc}"
    assert not out.exists(), (
        "it published a decision record for a field spanning three protocols")

    #: And the same archive with ONE protocol decides normally, so the refusal
    #: is attributable to the mixing and not to the fixture.
    ok = tmp_path / "uniform"
    ok.mkdir()
    store2 = _archive(ok, treatment_correct=0.26, incumbent_correct=0.34)
    out2 = ok / "decision.json"
    assert RC.main(["--store", str(store2), "--out", str(out2)]) == 0
    assert json.loads(out2.read_text())["terminal_state"] == "NO_GO"
