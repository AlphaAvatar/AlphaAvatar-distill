"""B→C survives teardown in BOTH branches, and the beam's ranking is untouched.

Zero cost, CPU only. The hole under test is narrow and was real:
`run_phase_a_search` commits the beam ranking before the baseline resolves —
correctly — and its summary serializes an imported candidate as identity and
provenance only. So a run could rebuild B, evaluate it on the frozen suite, and
lose B's cheap-metric result at teardown, leaving a reviewer with a ranking over
C and no number for the baseline the question is asked against.

Every state here is a REAL `InitializationState` with a REAL `StateEvaluation`
attached through `attach_evaluation`, which refuses a measurement whose artifact
digest does not match. Fakes are used only where resolution is under test and
the metric plumbing is not.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for _extra in ("src", "scripts", "scripts/autoinit"):
    if str(REPO / _extra) not in sys.path:
        sys.path.insert(0, str(REPO / _extra))

from aadistill.initialization.planning.ranking import PARETO_V1  # noqa: E402
from aadistill.initialization.specs.artifact import (  # noqa: E402
    CheckpointIdentity, ShardRecord,
)
from aadistill.initialization.specs.metrics import (  # noqa: E402
    StateEvalSuite, StateEvaluation,
)
from aadistill.initialization.specs.state import make_retained_state  # noqa: E402
from experiments.phase_c2 import baseline as B  # noqa: E402
from experiments.phase_c2 import comparison as C  # noqa: E402

SUITE = StateEvalSuite(
    suite_id="toy.state_eval", version=1, domains=("general",),
    subtypes={"general": ("text",)}, critical_tags=("eos_like",),
    description="toy stand-in; structure mirrors state_eval_v1")

#: The three objectives PARETO_V1 ranks on, all minimized.
KL, WORST, CRIT = PARETO_V1.required_metrics()


def identity(shard_sha: str, config_sha: str = "c" * 64) -> CheckpointIdentity:
    return CheckpointIdentity(
        path=f"/states/{shard_sha[:8]}",
        shards=(ShardRecord("model.safetensors", shard_sha, 1192099840),),
        index_sha256=None, config_sha256=config_sha,
        arch_signature=B.B_ARCH_SIGNATURE, tokenizer_sha256=None,
        num_parameters=B.B_NUM_PARAMETERS)


def state(state_id: str, *, kl: float, worst: float, crit: float,
          shard_sha: str, provenance: str = "retained_imported",
          config_sha: str = "c" * 64):
    """A real measured state: identified, then evaluated on its own digest."""
    artifact = identity(shard_sha, config_sha)
    s = make_retained_state(
        state_id=state_id, artifact=artifact, spec=_spec(), target_spec=_spec(),
        num_parameters=B.B_NUM_PARAMETERS, root_teacher_id="toy",
        root_teacher_sha256="0" * 64, description="test",
        provenance=provenance)
    s.attach_evaluation(StateEvaluation(
        artifact_digest=artifact.artifact_digest,
        suite_id=SUITE.suite_id, suite_hash=SUITE.suite_hash,
        reference="root_teacher",
        values={KL: kl, WORST: worst, CRIT: crit},
        positions=1234, detail={"per_domain_kl": {"general": kl}},
        measured_utc="2026-09-16T00:00:00Z", runtime={"gpu": "toy"}))
    return s


def _spec():
    from aadistill.initialization.specs.arch import ArchSpec

    return ArchSpec.of("qwen3", dict(
        hidden_size=1024, num_hidden_layers=28, intermediate_size=3072,
        num_attention_heads=16, num_key_value_heads=8, head_dim=128,
        vocab_size=151936, tie_word_embeddings=True))


def b_state(**kw):
    return state("b-state", kl=4.00, worst=4.40, crit=6.10,
                 shard_sha="b" * 64, **kw)


def c_states():
    return [state("c-better", kl=3.90, worst=4.30, crit=6.00, shard_sha="1" * 64),
            state("c-worse", kl=4.20, worst=4.60, crit=6.30, shard_sha="2" * 64)]


def build(baseline, candidates, outcome):
    return C.build(baseline=baseline, baseline_outcome=outcome,
                   candidates=candidates, suite=SUITE, policy=PARETO_V1,
                   run_id="autoinit.v1.phase_c2.search1",
                   config_hash="f" * 64,
                   selection_record="artifacts/autoinit/phase_c2_search/"
                                    "stage1_selection.json",
                   search_summary="artifacts/audit/autoinit_phase_c2/"
                                  "c2_search_summary.json")


SEARCHED = {"resolution": "searched", "rebuilt": False}
REBUILT = {"resolution": "rebuilt", "rebuilt": True, "rebuilds": 1,
           "identity_matches": {"artifact_digest": True, "weights_digest": True},
           "construction": {"spec_hash": B.B_SPEC_HASH, "verified": True}}


# --- 1 and 2: both branches persist the SAME metric schema -----------------

def metric_shape(block: dict) -> set:
    return set(block["evaluation"])


def test_the_searched_branch_persists_bs_metrics():
    record = build(b_state(), c_states(), SEARCHED)
    b = record["baseline"]
    assert b["resolution"] == "searched" and b["rebuilt"] is False
    #: The COMPLETE serialized evaluation, not a summary of it: a reviewer must
    #: be able to reproduce the comparison from this record alone.
    assert b["evaluation"]["values"] == {KL: 4.00, WORST: 4.40, CRIT: 6.10}
    assert b["evaluation"]["suite_hash"] == SUITE.suite_hash
    assert b["evaluation"]["positions"] == 1234
    assert b["evaluation"]["detail"]["per_domain_kl"] == {"general": 4.00}
    assert b["evaluation"]["artifact_digest"] == b["identity"]["artifact_digest"]


def test_the_rebuilt_branch_persists_the_identical_schema():
    searched = build(b_state(), c_states(), SEARCHED)
    rebuilt = build(b_state(), c_states(), REBUILT)

    assert metric_shape(rebuilt["baseline"]) == metric_shape(searched["baseline"])
    assert set(rebuilt["baseline"]) == set(searched["baseline"])
    assert rebuilt["baseline"]["evaluation"] == searched["baseline"]["evaluation"]
    #: Only the provenance fields differ, which is the point of recording them.
    assert rebuilt["baseline"]["rebuilt"] is True
    assert rebuilt["baseline"]["frozen_identity_matches"]["weights_digest"] is True
    assert rebuilt["baseline"]["construction"]["spec_hash"] == B.B_SPEC_HASH


def test_a_rebuilt_bs_evaluation_cannot_vanish_from_the_evidence():
    """The hole itself. An unmeasured baseline must REFUSE, not serialize.

    Before this record existed, a rebuilt B's evaluation lived only in memory
    and in the generic summary's identity-only entry. If it is ever absent the
    right outcome is a loud failure, because a comparison written without it
    would be a comparison with a hole exactly where the baseline belongs.
    """
    unmeasured = make_retained_state(
        state_id="b-unmeasured", artifact=identity("d" * 64), spec=_spec(),
        target_spec=_spec(), num_parameters=B.B_NUM_PARAMETERS,
        root_teacher_id="toy", root_teacher_sha256="0" * 64,
        description="never evaluated")
    assert unmeasured.evaluation is None
    with pytest.raises(C.ComparisonError, match="no state_eval result"):
        build(unmeasured, c_states(), REBUILT)

    #: And the measured case carries every value through to the written bytes.
    record = build(b_state(), c_states(), REBUILT)
    assert json.loads(json.dumps(record))["baseline"]["evaluation"]["values"][KL] \
        == 4.00


def test_the_candidates_carry_their_metrics_too():
    record = build(b_state(), c_states(), SEARCHED)
    assert [c["state_id"] for c in record["candidates"]] == ["c-better", "c-worse"]
    for candidate in record["candidates"]:
        assert set(candidate["evaluation"]["values"]) == {KL, WORST, CRIT}
        assert candidate["evaluation"]["suite_hash"] == SUITE.suite_hash


# --- the comparison is the run's own policy, and its verdict is derived ----

def test_the_suite_and_policy_identities_are_recorded():
    record = build(b_state(), c_states(), SEARCHED)
    assert record["suite"] == {"id": SUITE.qualified_id,
                               "hash": SUITE.suite_hash}
    assert record["policy"]["id"] == PARETO_V1.qualified_id
    assert record["policy"]["hash"] == PARETO_V1.policy_hash
    assert [o["key"] for o in record["policy"]["objectives"]] == list(
        PARETO_V1.required_metrics())
    assert record["policy"]["epsilon"] == {k: float(v) for k, v
                                           in PARETO_V1.epsilon.items()}


def test_the_verdict_is_computed_from_the_fronts_not_asserted():
    """A dominating candidate, a dominated one, and a derived conclusion."""
    record = build(b_state(), c_states(), SEARCHED)
    comparison = record["comparison"]
    assert comparison["baseline_front"] == 1
    assert comparison["candidates_in_a_better_front"] == ["c-better"]
    assert comparison["candidates_in_a_worse_front"] == ["c-worse"]
    assert comparison["verdict"] == "CANDIDATE_IN_A_BETTER_FRONT_THAN_BASELINE"

    #: Per objective, with the policy's own epsilon and direction.
    deltas = next(c for c in record["candidates"]
                  if c["state_id"] == "c-better")["objectives"][KL]
    assert deltas["direction"] == "minimize"
    assert deltas["candidate_better"] is True
    assert deltas["delta_candidate_minus_baseline"] == pytest.approx(-0.10)
    assert deltas["within_epsilon"] is False


def test_a_baseline_nothing_beats_gets_the_opposite_verdict():
    """The verdict must be able to come out the other way, or it is a constant."""
    strong = state("b-strong", kl=3.00, worst=3.30, crit=5.00,
                   shard_sha="e" * 64)
    record = build(strong, c_states(), SEARCHED)
    assert record["comparison"]["baseline_front"] == 0
    assert record["comparison"]["candidates_in_a_better_front"] == []
    assert record["comparison"]["verdict"] == \
        "BASELINE_IN_A_BETTER_FRONT_THAN_EVERY_CANDIDATE"


def test_metrics_that_tie_within_epsilon_share_a_front():
    tied = state("c-tied", kl=4.00005, worst=4.40005, crit=6.10005,
                 shard_sha="3" * 64)
    record = build(b_state(), [tied], SEARCHED)
    assert record["comparison"]["candidates_in_the_same_front"] == ["c-tied"]
    assert record["comparison"]["verdict"] == \
        "BASELINE_AND_CANDIDATES_SHARE_A_FRONT"
    assert record["candidates"][0]["objectives"][KL]["within_epsilon"] is True


# --- 5: a duplicate B is not compared twice --------------------------------

def test_a_selected_b_is_excluded_rather_than_compared_with_itself():
    """The beam may well select B. Comparing B with B would place an identical
    pair in one front and read as a finding."""
    baseline = b_state()
    duplicate = state("b-state", kl=4.00, worst=4.40, crit=6.10,
                      shard_sha="b" * 64)
    record = build(baseline, [duplicate, *c_states()], SEARCHED)

    assert record["excluded_as_duplicate_of_baseline"] == ["b-state"]
    assert [c["state_id"] for c in record["candidates"]] == \
        ["c-better", "c-worse"]
    assert "b-state" not in [c["state_id"] for c in record["candidates"]]
    #: One appearance in the fronts, not two.
    flat = [sid for front in record["comparison"]["fronts"] for sid in front]
    assert flat.count("b-state") == 1


def test_a_duplicate_by_digest_under_another_id_is_also_excluded():
    """Identity is the artifact, not the name. A rebuilt B carries a different
    `state_id` by design and the same digest by construction."""
    baseline = b_state()
    same_bytes = state(B.REBUILT_STATE_ID, kl=4.00, worst=4.40, crit=6.10,
                       shard_sha="b" * 64)
    assert same_bytes.artifact_digest == baseline.artifact_digest
    record = build(baseline, [same_bytes], SEARCHED)
    assert record["excluded_as_duplicate_of_baseline"] == [B.REBUILT_STATE_ID]
    assert record["candidates"] == []
    assert record["comparison"]["verdict"] == "NO_CANDIDATE_TO_COMPARE"


# --- 4: the beam's ranking artifact is not mutated -------------------------

def test_the_comparison_cites_the_beam_ranking_and_never_writes_it(tmp_path):
    """`stage1_selection.json` is the beam's durability boundary.

    Inserting a state the beam did not generate would make the search's own
    record describe a candidate set it never produced. So the comparison is a
    separate file, and this test proves the bytes of the ranking artifact are
    untouched by writing one.
    """
    selection = tmp_path / "stage1_selection.json"
    original = json.dumps({"schema": "aadistill.autoinit.stage1_selection/v1",
                           "selected": [{"state_id": "c-better"}]}, indent=1)
    selection.write_text(original)
    before = selection.read_bytes()

    record = build(b_state(), c_states(), REBUILT)
    written = C.commit(record, tmp_path)

    assert selection.read_bytes() == before, (
        "the post-search comparison mutated the beam's ranking artifact")
    assert written.name == "c2_baseline_comparison.json"
    assert written != selection
    assert record["cites"]["beam_ranking"].endswith("stage1_selection.json")
    #: Atomic, and no temp file left behind.
    assert not list(tmp_path.glob("*.partial"))
    assert json.loads(written.read_text())["record_sha256"] == \
        record["record_sha256"]


def test_the_record_is_self_hashing_and_says_what_it_is_not():
    record = build(b_state(), c_states(), SEARCHED)
    from aadistill.infrastructure.manifest import sha256_json

    stated = record["record_sha256"]
    assert stated == sha256_json({k: v for k, v in record.items()
                                  if k != "record_sha256"})
    assert record["schema"] == C.SCHEMA
    assert "NOT behavioural recovery evidence" in record["not_behavioural_evidence"]
    assert "hypothesis" in record["not_behavioural_evidence"].lower()


# --- resolution: one place decides which state is B -----------------------

@dataclass
class FakeStep:
    kind: str
    impl_id: str
    profile_id: str


@dataclass
class FakeLeaf:
    steps: tuple
    artifact_digest: str
    state_id: str = "f" * 32

    def is_complete_leaf(self) -> bool:
        return True


@dataclass
class FakeResult:
    leaves: list


@dataclass
class FakeFound:
    result: FakeResult
    imported: list


def test_resolution_finds_the_searched_leaf():
    leaf = FakeLeaf(steps=tuple(FakeStep(*t) for t in B.B_PATH),
                    artifact_digest=B.B_ARTIFACT_DIGEST)
    found = FakeFound(result=FakeResult([leaf]), imported=[])
    assert C.resolve_baseline_state(found=found, outcome=SEARCHED) is leaf


def test_resolution_finds_the_single_injected_rebuild():
    rebuilt = state(B.REBUILT_STATE_ID, kl=4.0, worst=4.4, crit=6.1,
                    shard_sha="b" * 64)
    found = FakeFound(result=FakeResult([]), imported=[rebuilt])
    assert C.resolve_baseline_state(found=found, outcome=REBUILT) is rebuilt


@pytest.mark.parametrize("outcome,imported,leaves,match", [
    (SEARCHED, [], [], "cannot both be true"),
    (REBUILT, [], [], "0 injected states"),
    ({"resolution": None}, [], [], "must be"),
])
def test_resolution_refuses_rather_than_guessing(outcome, imported, leaves,
                                                 match):
    found = FakeFound(result=FakeResult(leaves), imported=imported)
    with pytest.raises(C.ComparisonError, match=match):
        C.resolve_baseline_state(found=found, outcome=outcome)


def test_two_injected_rebuilds_are_refused():
    """Their identities collide by construction; two of them is a defect."""
    pair = [state(B.REBUILT_STATE_ID, kl=4.0, worst=4.4, crit=6.1,
                  shard_sha="b" * 64),
            state(B.REBUILT_STATE_ID, kl=4.0, worst=4.4, crit=6.1,
                  shard_sha="b" * 64)]
    found = FakeFound(result=FakeResult([]), imported=pair)
    with pytest.raises(C.ComparisonError, match="2 injected states"):
        C.resolve_baseline_state(found=found, outcome=REBUILT)
