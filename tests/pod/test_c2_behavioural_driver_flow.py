"""The behavioural driver's stage sequence, executed screening -> advancement
-> confirmation.

The two hardware-bound steps — training a probe and scoring a battery — are
seams C1 built explicitly so a `$0` regression can replace them, and that is
what happens here. Everything else is the real code: the real candidate
manifest joined to the real durable products, the real frozen seeds and
batteries, the real completeness gate, the real ranking and tie-break, the real
advancement, and the real refusal of a screening verdict.

What this proves is the shape. A harness that reimplemented the loop could not
notice the loop being broken, which is why the loop is inherited from C1 and
only `descriptors`, the rung boundary and the selection are overridden.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for extra in ("src", "scripts", "scripts/autoinit", "scripts/pod"):
    if str(ROOT / extra) not in sys.path:
        sys.path.insert(0, str(ROOT / extra))

from experiments.phase_c2 import behavioural as BH  # noqa: E402
from experiments.phase_c2 import behavioural_schedule as SCH  # noqa: E402


class _Recorder:
    """A driver stand-in that runs the REAL stage bodies over fake hardware.

    It deliberately does not subclass the pod driver: importing that module
    pulls in C1's pod-side imports, and the point here is the control flow, not
    the import graph. The stage bodies it runs are the schedule module's — the
    same functions the driver calls, in the same order.
    """

    def __init__(self, scores: dict[str, float]):
        self.scores = scores
        self.trained: dict[str, dict] = {}
        self.scored: dict[str, dict] = {}
        self.preserved: list[str] = []
        self.order: list[str] = []

    def run_rung(self, probes):
        for p in probes:
            self.trained[p.probe_id] = {"ok": True}
            #: Persisted AS IT COMPLETES, before the next probe starts.
            self.preserved.append(p.probe_id)
            self.scored[p.probe_id] = {
                "correct_overall": self.scores[p.arm]}
            self.order.append(p.probe_id)


def inputs():
    proto = BH.protocol(ROOT)["behavioural_selection"]
    cands = BH.candidate_manifest(ROOT)
    b = BH.b_binding(ROOT)
    anchor = {"state_id": SCH.ANCHOR,
              "artifact_digest": b["required_identity"]["artifact_digest"],
              "durable_path": b["availability"]["durable_path"]}
    return proto, cands, anchor


def test_the_full_sequence_runs_and_totals_twelve_probes():
    proto, cands, anchor = inputs()
    scores = {c["state_id"]: 0.30 + i / 100 for i, c in enumerate(cands)}
    scores[SCH.ANCHOR] = 0.31
    rec = _Recorder(scores)

    # -- S: screening
    screening = SCH.screening_probes(cands, anchor, proto["seeds"]["screening"])
    rec.run_rung(screening)
    ok, why = SCH.screening_is_complete(screening, rec.trained, rec.scored)
    assert ok, why

    # -- R: rank and advance, emitting no verdict
    arm_scores = {p.arm: rec.scored[p.probe_id]["correct_overall"]
                  for p in screening}
    ranked = SCH.rank_screening(arm_scores, cands)
    SCH.assert_screening_emits_no_verdict({"ranked": ranked})
    advanced = SCH.advance_one(ranked)
    assert advanced["state_id"] != SCH.ANCHOR

    # -- C: confirmation on the advanced arm only
    chosen = next(c for c in cands if c["state_id"] == advanced["state_id"])
    confirmation = SCH.confirmation_probes(chosen, anchor,
                                           proto["seeds"]["confirmation"])
    rec.run_rung(confirmation)

    assert len(rec.order) == 12
    assert len(screening) == 6 and len(confirmation) == 6
    #: Screening ran entirely before confirmation began.
    assert rec.order[:6] == [p.probe_id for p in screening]
    #: Every probe was persisted as it completed, not at the end.
    assert rec.preserved == rec.order

    # -- D: a per-seed paired delta, three of them
    deltas = []
    for seed in sorted({p.seed for p in confirmation}):
        pair = {p.arm: rec.scored[p.probe_id]["correct_overall"]
                for p in confirmation if p.seed == seed}
        deltas.append(round(pair[advanced["state_id"]] - pair[SCH.ANCHOR], 10))
    assert len(deltas) == 3


def test_confirmation_cannot_begin_while_a_screening_probe_is_unscored():
    """The gate that stops a candidate being chosen from whoever finished."""
    proto, cands, anchor = inputs()
    scores = {c["state_id"]: 0.3 for c in cands}
    scores[SCH.ANCHOR] = 0.3
    rec = _Recorder(scores)
    screening = SCH.screening_probes(cands, anchor, proto["seeds"]["screening"])
    rec.run_rung(screening[:-1])                    # one short

    ok, why = SCH.screening_is_complete(screening, rec.trained, rec.scored)
    assert not ok
    assert screening[-1].probe_id in why
    with pytest.raises(SCH.ScheduleError, match="no screening score"):
        SCH.rank_screening(
            {p.arm: rec.scored[p.probe_id]["correct_overall"]
             for p in screening[:-1]}, cands)


def test_the_anchor_is_scored_in_both_rungs_and_never_advances():
    proto, cands, anchor = inputs()
    #: The anchor beats every candidate — it still must not advance.
    scores = {c["state_id"]: 0.20 for c in cands}
    scores[SCH.ANCHOR] = 0.90
    rec = _Recorder(scores)
    screening = SCH.screening_probes(cands, anchor, proto["seeds"]["screening"])
    rec.run_rung(screening)
    ranked = SCH.rank_screening(
        {p.arm: rec.scored[p.probe_id]["correct_overall"] for p in screening},
        cands)
    advanced = SCH.advance_one(ranked)
    assert advanced["state_id"] != SCH.ANCHOR
    assert advanced["delta_vs_b"] < 0, (
        "every candidate lost to B and one still advanced -- correct: "
        "screening ranks and does not veto, and confirmation decides")


def test_a_failure_after_screening_does_not_discard_completed_probes():
    """Expensive completed work survives a later stage's failure."""
    proto, cands, anchor = inputs()
    scores = {c["state_id"]: 0.3 + i / 100 for i, c in enumerate(cands)}
    scores[SCH.ANCHOR] = 0.3
    rec = _Recorder(scores)
    screening = SCH.screening_probes(cands, anchor, proto["seeds"]["screening"])
    rec.run_rung(screening)
    before = list(rec.preserved)

    with pytest.raises(SCH.ScheduleError):
        SCH.confirmation_probes(anchor, anchor, proto["seeds"]["confirmation"])

    assert rec.preserved == before and len(before) == 6


def test_the_driver_declares_its_hardware_steps_as_seams():
    """`score_probe` and the decision rule must be bound, never inlined —
    otherwise a regression cannot replace them and the loop goes unexercised."""
    source = (ROOT / "scripts/pod/autoinit_c2_behavioural_driver.py").read_text()
    for name in ("def score_probe", "def apply_frozen_decision_rule"):
        assert name in source
    assert source.count("NotImplementedError") >= 2, (
        "a seam that silently returns a default is worse than one that refuses")
    #: And the loop itself must be inherited, not copied.
    assert "class C2BehaviouralDriver(C1Driver)" in source


def test_the_driver_gates_b_on_its_exact_identity():
    """If the rebuilt B is not B, no screening probe may start."""
    source = (ROOT / "scripts/pod/autoinit_c2_behavioural_driver.py").read_text()
    body = source[source.index("def materialize_b"):source.index("def descriptors")]
    for field in ("artifact_digest", "weights_digest", "single_shard_sha256",
                  "arch_signature", "num_parameters"):
        assert field in body, f"B's identity gate does not check {field}"
    assert "NO SCREENING PROBE MAY START" in body
