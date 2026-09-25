"""The Phase-C2 Search-1 space: every identity re-derived, and the price checked.

Zero cost, CPU only. Two things are being tested and they are different:

1. **The declared space is what committed evidence says it is.** Every constant
   in `experiments.phase_c2.search_space` is recomputed here from the file it
   came from. A search configured from an identity someone typed in is a search
   for a path nobody ran, and this repository has pinned the wrong hash before.

2. **The cost model reproduces the one search it has evidence for.** Phase-B
   attempt 5's real beams go through the same branching code and must predict
   every level's expansion count exactly. A model that cannot reproduce the past
   cannot bound the future, and `children_max x mean` — which could not — is
   what authorized a 1.91-7.51 h projection for a run that took 9.08 h.

The generic capability the space needs, `SearchConfig.impl_profiles`, is
exercised both as a unit and by running `run_phase_a_search` for real at toy
scale. The second is not redundant: `attention.activation_importance_v1` has
never run inside `BeamSearch` in this project's history — C1 drove it through
`fixed_path` only — so every line of that combination is code no test and no
paid pod has executed.
"""

import json
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for extra in ("src", "scripts", "scripts/autoinit"):
    if str(REPO / extra) not in sys.path:
        sys.path.insert(0, str(REPO / extra))

from experiments.phase_c2 import search_space as SS  # noqa: E402


@pytest.fixture
def registered():
    """Register what the space names, then leave the registry as it was.

    `attention.activation_importance_v1` is deliberately not a shipped default.
    Leaking it would make any sibling test that enumerates the registry depend
    on whether this file ran first.
    """
    from aadistill.initialization.operators.attention.gqa import activation_importance as attention_activation

    SS.register_c2_operators()
    try:
        yield
    finally:
        attention_activation.unregister()


# --- 1. the identities, re-derived -----------------------------------------

def test_the_phase_b_winner_is_the_path_the_selection_record_names():
    record = json.loads((
        REPO / "logs/stages/stage-1/phase_b/runs/attempt5/stage1_selection.json"
    ).read_text())
    winner = next(s for s in record["selected"]
                  if s["state_id"] == SS.PHASE_B_WINNER_STATE_ID)
    assert winner["artifact_digest"] == SS.PHASE_B_WINNER_ARTIFACT_DIGEST
    #: The incumbent's three held mixtures are read off its path, not assumed.
    assert winner["path"] == (
        "DEPTH(calib.domain_balanced@v1)->FFN(calib.domain_balanced@v1)->"
        "RESIDUAL_WIDTH(calib.reasoning_heavy@v2)->ATTENTION(calib.none@v1)")


def test_the_held_profiles_are_the_incumbents_own():
    """C2 holds DEPTH/FFN/WIDTH at the Phase-B winner's mixtures. Derived."""
    record = json.loads((
        REPO / "logs/stages/stage-1/phase_b/runs/attempt5/stage1_selection.json"
    ).read_text())
    winner = next(s for s in record["selected"]
                  if s["state_id"] == SS.PHASE_B_WINNER_STATE_ID)
    held = {}
    for step in winner["path"].split("->"):
        kind, _, rest = step.partition("(")
        held[kind] = rest.rstrip(")")

    assert SS.C2_IMPL_PROFILES["depth.causal_kl_greedy_v1"] == (held["DEPTH"],)
    assert SS.C2_IMPL_PROFILES["ffn.activation_importance_v0"] == (held["FFN"],)
    assert SS.C2_IMPL_PROFILES["width.global_pca_v0"] == (
        held["RESIDUAL_WIDTH"],)
    #: ATTENTION is the one that branches, and it cannot inherit the incumbent's
    #: `calib.none@v1`: that sentinel belongs to `weight_proxy_v0`, which
    #: declares CalibrationNeed.NONE. The replacement consumes activation stats.
    assert held["ATTENTION"] == "calib.none@v1"
    assert SS.C2_IMPL_PROFILES["attention.activation_importance_v1"] == (
        SS.C2_PROFILE_IDS)


def test_the_c1_treatment_is_the_baseline_and_is_a_leaf_of_this_search():
    prereg = json.loads((
        REPO / "logs/stages/stage-1/phase_c1/plans/execution_preregistration.json"
    ).read_text())
    assert prereg["fixed_path"]["treatment_spec_hash"] == SS.C1_TREATMENT_SPEC_HASH
    assert prereg["fixed_path"]["treatment_path_label"] == SS.C1_TREATMENT_PATH_LABEL
    assert (prereg["treatment_operator"]["impl_id"]
            in SS.C2_ALLOWED_IMPLS)

    #: Reachable: its order is a permutation the search may take and its
    #: ATTENTION mixture is one of the two branches, so the search RE-DERIVES the
    #: baseline rather than importing a checkpoint. There is nothing to compare
    #: against if the search cannot reach it.
    applied, attention_profile = set(), None
    for step in SS.C1_TREATMENT_PATH_LABEL.split("->"):
        kind, _, rest = step.partition("(")
        applied.add(kind)
        if kind == "ATTENTION":
            attention_profile = rest.rstrip(")")
    assert applied == {"DEPTH", "FFN", "RESIDUAL_WIDTH", "ATTENTION"}
    assert attention_profile in SS.C2_IMPL_PROFILES[
        "attention.activation_importance_v1"]


def test_the_allowed_implementations_are_one_per_kind_and_exclude_the_anchors(
        registered):
    from aadistill.initialization.operators.base import get_implementation

    kinds = [get_implementation(i).kind for i in SS.C2_ALLOWED_IMPLS]
    assert sorted(kinds) == ["ATTENTION", "DEPTH", "FFN", "RESIDUAL_WIDTH"]
    #: The historical control, the positional heuristic and the one-step
    #: composite are anchors and alternatives, not C2 branches. Absence from
    #: `allowed_impls` is the only way to say so: `_allowed_impl_ids` falls back
    #: to the whole registry when it is None.
    for excluded in ("attention.weight_proxy_v0", "depth.positional_v0",
                     "composite.stage1_sandwich_v0"):
        assert excluded not in SS.C2_ALLOWED_IMPLS


def test_both_branch_profiles_are_actually_materialized():
    from aadistill.initialization.calibration.profiles import get_profile
    from experiments.calibration import register_builtin_profiles

    register_builtin_profiles()
    for qualified_id in SS.C2_PROFILE_IDS:
        profile = get_profile(qualified_id)
        assert profile.materialized, f"{qualified_id} is declared, not built"
        assert profile.content_sha256


def test_the_teacher_geometry_is_the_one_the_phase_a_journal_recorded():
    """Re-derived, not transcribed.

    A level-0 ATTENTION child's `arch_spec` is the teacher in every field its
    operator did not modify, and `steps[0].trace.q_heads` is `[parent, target]`
    — so the one field it did modify is recoverable too.
    """
    journal = (REPO
               / "logs/stages/stage-1/phase_a/runs/attempt7/states.jsonl")
    rows = [json.loads(line) for line in journal.read_text().splitlines()
            if line.strip()]
    child = next(r for r in rows
                 if r.get("depth") == 1 and r["steps"][0]["kind"] == "ATTENTION")
    derived = dict(child["arch_spec"])
    derived["num_attention_heads"] = child["steps"][0]["trace"]["q_heads"][0]
    assert derived == SS.TEACHER_GEOMETRY

    #: And the four fields that differ from the target are exactly the four
    #: kinds' business — which is why the path length is four and why the order
    #: search is a permutation of four operators.
    assert set(child["remaining_differences"]) == {
        "hidden_size", "intermediate_size", "num_hidden_layers"}


def test_the_cost_table_is_recomputed_from_the_committed_telemetry():
    """The table is a cache of this arithmetic, not a set of remembered numbers."""
    import collections
    import statistics

    result = json.loads((REPO / SS.PHASE_B_RESULT).read_text())
    level_of = {sid: L["level"] for L in result["levels"]
                for sid in L["generated"]}
    phases = ("materialize_seconds", "identify_seconds",
              "canonical_reload_seconds", "validation_seconds",
              "state_evaluation_seconds")

    def seconds(row, key):
        value = row.get(key) or {}
        return float(value.get("seconds", 0.0)) if isinstance(value, dict) \
            else float(value or 0)

    grouped = collections.defaultdict(list)
    for line in (REPO / SS.PHASE_B_TELEMETRY).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        total = (row["operator_seconds"] + row["parent_load_seconds"]
                 + sum(seconds(row, k) for k in phases)) / 60
        where = "root" if level_of.get(row["state_id"]) == 0 else "deeper"
        grouped[(row["impl_id"], where)].append(total)

    for (impl_id, where), values in grouped.items():
        entry = SS.MEASURED_MINUTES[impl_id]
        assert entry[f"{where}_max"] == pytest.approx(max(values), abs=0.01)
        assert entry[f"{where}_mean"] == pytest.approx(
            statistics.mean(values), abs=0.01)

    #: And nothing in the table is a figure no expansion produced.
    assert set(SS.MEASURED_MINUTES) == {impl for impl, _ in grouped}


def test_the_only_unmeasured_cost_is_named_as_such(registered):
    b = SS.bound(SS.c2_search1_space())
    assert b.unmeasured == ("attention.activation_importance_v1",)
    #: It has no measured row, so it cannot be priced by accident.
    assert "attention.activation_importance_v1" not in SS.MEASURED_MINUTES
    assert SS.minutes_for("attention.activation_importance_v1", root=True) == (
        pytest.approx(SS.MEASURED_MINUTES[SS.ATTENTION_ACTIVATION_PROXY_IMPL]
                      ["root_max"] * SS.ATTENTION_ACTIVATION_PROXY_FACTOR))


# --- 2. the back-test -------------------------------------------------------

def test_the_model_reproduces_phase_b_attempt_5_level_by_level(registered):
    back = SS.replay_phase_b()
    assert back["every_level_exact"], back["levels"]
    assert back["predicted_expansions"] == back["observed_expansions"] == 82
    #: Minutes come from the same telemetry, so agreement is a check on the
    #: BRANCHING and on the root/deeper split, not a tautology: a model that put
    #: an expansion at the wrong level would use the wrong column.
    assert back["predicted_minutes"] == pytest.approx(
        back["observed_minutes"], rel=0.01)


def test_the_bound_contains_the_depth_early_trajectory(registered):
    space = SS.c2_search1_space()
    b = SS.bound(space)
    early = SS.trajectory(space, prefer_depth=True)
    late = SS.trajectory(space, prefer_depth=False)
    assert b.min_minutes <= early["minutes"] <= b.max_minutes
    assert b.min_minutes <= late["minutes"] <= b.max_minutes
    #: The cost risk is beam composition, not per-node cost: deferring DEPTH
    #: roughly doubles the search while changing no unit price.
    assert late["minutes"] > 1.8 * early["minutes"]


def test_a_narrower_beam_costs_strictly_less(registered):
    """The bound responds to the knob it is supposed to respond to.

    A `bound()` that returned the same number for every width would be a
    constant wearing a model's clothing, and would have looked fine in the two
    assertions above.
    """
    space = SS.c2_search1_space()
    maxima = [SS.bound(space, beam_width=w).max_minutes for w in (3, 4, 5, 6)]
    assert maxima == sorted(maxima)
    assert len(set(maxima)) == 4


def test_the_predicted_level_shape_is_the_one_the_report_quotes(registered):
    """Five, then eighteen. The numbers a maintainer is asked to authorize."""
    space = SS.c2_search1_space()
    rows = SS.trajectory(space, prefer_depth=True)["levels"]
    assert [r["generated"] for r in rows[:2]] == [5, 18]
    b = SS.bound(space)
    assert (b.min_expansions, b.max_expansions) == (41, 53)


def test_the_priced_ceiling_covers_the_structural_worst_case(registered):
    space = SS.c2_search1_space()
    plan = SS.price(space, price_per_hour=SS.PRICE_PER_HOUR_LAST_QUOTED,
                    authorized_usd=10_000.0)
    worst_search = SS.bound(space).max_minutes
    overhead = sum(m for _, m in SS.SESSION_PHASE_MINUTES)
    #: The hard-terminate point must still fit the worst search PLUS the whole
    #: non-search session PLUS the teardown reserve. A ceiling that only covers
    #: the expected path is what leaves nothing for artifact recovery.
    assert plan.hard_terminate_minutes >= worst_search + overhead + 30.0
    assert plan.expected_minutes < plan.soft_stop_minutes < \
        plan.hard_terminate_minutes


def test_pricing_refuses_an_authorization_the_ceiling_would_exceed(registered):
    from aadistill.infrastructure.budget import BudgetError

    with pytest.raises(BudgetError):
        SS.price(SS.c2_search1_space(),
                 price_per_hour=SS.PRICE_PER_HOUR_LAST_QUOTED,
                 authorized_usd=1.0)


# --- 3. the generic capability ---------------------------------------------

def test_a_no_calibration_implementation_is_offered_exactly_once(registered):
    """The bug the cost model had, now pinned in the shared function."""
    from aadistill.initialization.operators.base import get_implementation
    from aadistill.initialization.planning.search import expansion_profiles
    from experiments.calibration import (
        DOMAIN_BALANCED_V1, REASONING_HEAVY_V2, register_builtin_profiles)

    register_builtin_profiles()
    both = (DOMAIN_BALANCED_V1, REASONING_HEAVY_V2)

    offered = expansion_profiles(
        get_implementation("attention.weight_proxy_v0"), both)
    assert [p.qualified_id for p in offered] == ["calib.none@v1"]

    offered = expansion_profiles(
        get_implementation("ffn.activation_importance_v0"), both)
    assert [p.qualified_id for p in offered] == [
        "calib.domain_balanced@v1", "calib.reasoning_heavy@v2"]

    offered = expansion_profiles(
        get_implementation("ffn.activation_importance_v0"), both,
        {"ffn.activation_importance_v0": ("calib.reasoning_heavy@v2",)})
    assert [p.qualified_id for p in offered] == ["calib.reasoning_heavy@v2"]


def _toy_search_config(tmp_path, impl_profiles, allowed_impls):
    """A real `SearchConfig`/`BeamSearch` pair over a 32-wide teacher."""
    from aadistill.initialization.specs.arch import ArchSpec, get_adapter
    from aadistill.initialization.planning.ranking import PARETO_V1, SCHEDULE_V1
    from aadistill.initialization.planning.search import BeamSearch, SearchConfig
    from aadistill.initialization.specs.metrics import StateEvalSuite
    from experiments.calibration import (
        DOMAIN_BALANCED_V1, REASONING_HEAVY_V2, register_builtin_profiles)

    register_builtin_profiles()
    suite = StateEvalSuite(
        suite_id="toy.state_eval", version=1, domains=("general",),
        subtypes={"general": ("text",)}, critical_tags=("eos_like",),
        description="toy")
    config = SearchConfig(
        run_id="toy.c2", target_spec=ArchSpec.of("qwen3", dict(
            hidden_size=16, num_hidden_layers=4, intermediate_size=24,
            num_attention_heads=2, num_key_value_heads=2, head_dim=8,
            vocab_size=128, tie_word_embeddings=True)),
        schedule=SCHEDULE_V1, seed=1, workdir=tmp_path,
        profiles=(DOMAIN_BALANCED_V1, REASONING_HEAVY_V2),
        policy=PARETO_V1, suite=suite, device="cpu",
        allowed_impls=allowed_impls, impl_profiles=impl_profiles)
    return BeamSearch(
        adapter=get_adapter("qwen3"), config=config,
        root_teacher_id="toy", root_teacher_sha256="0" * 64,
        root_loader=lambda: pytest.fail("the loader must not be needed"),
        calibration_loader=lambda profile: (),
        measurer=lambda model, digest: pytest.fail("nothing is measured here"))


@pytest.mark.parametrize("restriction,why", [
    ({"ffn.no_such_impl_v9": ("calib.domain_balanced@v1",)},
     "a typo'd impl_id would restrict nothing and run the factorial silently"),
    ({"attention.weight_proxy_v0": ("calib.domain_balanced@v1",)},
     "a NONE-calibration implementation cannot be restricted meaningfully"),
    ({"ffn.activation_importance_v0": ()},
     "an empty list removes a kind from the space but not from allowed_impls"),
    ({"ffn.activation_importance_v0": ("calib.stage0_current@v1",)},
     "a profile this search does not branch over names nothing reachable"),
])
def test_an_impl_profile_restriction_that_lies_is_refused(
        registered, tmp_path, restriction, why):
    from aadistill.initialization.planning.search import SearchError

    with pytest.raises(SearchError):
        _toy_search_config(tmp_path, restriction, None)


def test_a_valid_restriction_changes_the_branching_and_the_config_hash(
        registered, tmp_path):
    allowed = SS.C2_ALLOWED_IMPLS
    from aadistill.initialization.operators.base import get_implementation

    unrestricted = _toy_search_config(tmp_path / "a", None, allowed)
    restricted = _toy_search_config(tmp_path / "b", dict(SS.C2_IMPL_PROFILES),
                                    allowed)

    width = get_implementation("width.global_pca_v0")
    assert len(unrestricted.profiles_for(width)) == 2
    assert [p.qualified_id for p in restricted.profiles_for(width)] == [
        "calib.reasoning_heavy@v2"]
    attention = get_implementation("attention.activation_importance_v1")
    assert len(restricted.profiles_for(attention)) == 2

    #: Two searches that reach different leaves must not share an identity.
    assert unrestricted.config.config_hash != restricted.config.config_hash
    assert restricted.config.as_dict()["impl_profiles"] == {
        k: sorted(v) for k, v in sorted(SS.C2_IMPL_PROFILES.items())}
    #: And the key is ABSENT, not null, when nothing is restricted — so every
    #: search recorded before this field existed still hashes to the value in
    #: its committed `search_result.json`.
    assert "impl_profiles" not in unrestricted.config.as_dict()


def test_the_default_is_byte_identical_to_what_earlier_searches_ran(
        registered, tmp_path):
    """`impl_profiles=None` must not move any existing search's identity.

    Not merely "the value is None": the KEY must be absent, because
    `config_hash` is `sha256_json(as_dict())` and a new key with a null value
    still moves the digest. Phase A's and Phase B's config hashes are recorded
    in their committed `search_result.json`, and a field added for a later
    experiment must not make those records unverifiable against this code.
    """
    config = _toy_search_config(tmp_path, None, None).config
    rendered = config.as_dict()
    assert "impl_profiles" not in rendered
    #: The hash is a pure function of that dict, so absence is what preserves it.
    from aadistill.infrastructure.manifest import sha256_json

    assert config.config_hash == sha256_json(rendered)


# --- 4. the combination no pod has executed --------------------------------

def test_the_c2_shaped_search_runs_for_real_at_toy_scale(registered, tmp_path):
    """`run_phase_a_search` with C2's restriction and C2's ATTENTION operator.

    The reason this is here and not only as unit assertions above: in this
    project four paid pods have died inside lines no test had executed, and this
    combination — `attention.activation_importance_v1` expanded by `BeamSearch`,
    under a per-implementation profile restriction — is code that has never run
    anywhere. C1 drove that operator through `fixed_path`, which shares no
    expansion, statistics-cache or materialization path with the beam.

    Only the model is scaled down. Real operators, real checkpoints on disk, the
    real `from_pretrained` reload, real hashing, real measurement.
    """
    from aadistill.initialization.specs.arch import ArchSpec, get_adapter
    from phase_a_search import run_phase_a_search
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from aadistill.initialization.calibration.profiles import (
        CalibrationProfile, CalibrationSource)
    from aadistill.initialization.specs.metrics import StateEvalSuite, SuiteItem

    teacher_geometry = dict(hidden_size=32, num_hidden_layers=6,
                            intermediate_size=48, num_attention_heads=4,
                            num_key_value_heads=2, head_dim=8, vocab_size=128,
                            tie_word_embeddings=True)
    target_geometry = dict(hidden_size=16, num_hidden_layers=4,
                           intermediate_size=24, num_attention_heads=2,
                           num_key_value_heads=2, head_dim=8, vocab_size=128,
                           tie_word_embeddings=True)

    torch.manual_seed(4242)
    teacher = Qwen3ForCausalLM(Qwen3Config(
        max_position_embeddings=256, rope_theta=5_000_000,
        **teacher_geometry)).float().eval()
    with torch.no_grad():
        for module in teacher.modules():
            if module.__class__.__name__ == "Qwen3RMSNorm":
                module.weight.uniform_(0.5, 1.5)

    def items(seed, n=2, seq_len=24):
        torch.manual_seed(seed)
        out = []
        for k in range(n):
            ids = torch.randint(0, 128, (1, seq_len))
            targets = ids[0, 1:]
            #: The tags must actually MATCH something. All-zero tags gave every
            #: state no value for `state.critical_token_kl`, PARETO_V1 dropped
            #: all five as ineligible, and the search ended after level 0 with no
            #: leaf and no dead end — which reads exactly like a broken operator.
            out.append({"item_id": f"text-{k}", "domain": "general",
                        "subtype": "text", "input_ids": ids,
                        "tags": {"eos_like": targets == 0,
                                 "answer_like": targets % 17 == 0}})
        return out

    def profile(name, seed):
        return CalibrationProfile(
            profile_id=name, version=1, description="toy",
            sources=(CalibrationSource("toy", "local", "general", 2),),
            domain_weights={"general": 1.0}, token_budget=1,
            sample_rule="fixed order", seed=seed)

    held, branched = profile("toy.held", 1), profile("toy.branched", 2)
    suite = StateEvalSuite(
        suite_id="toy.state_eval", version=1, domains=("general",),
        subtypes={"general": ("text",)},
        critical_tags=("eos_like", "answer_like"), description="toy")
    suite_items = [SuiteItem(item_id=i["item_id"], input_ids=i["input_ids"],
                             domain=i["domain"], subtype=i["subtype"],
                             tags=i["tags"]) for i in items(99)]

    adapter = get_adapter("qwen3")
    control = tmp_path / "control"
    adapter.save(adapter.build_model(
        adapter.build_config(teacher.config,
                             ArchSpec.of("qwen3", target_geometry)),
        torch.float32, 4242), str(control))

    #: C2's shape: three kinds pinned to one mixture, ATTENTION over both.
    restriction = {
        "depth.causal_kl_greedy_v1": (held.qualified_id,),
        "ffn.activation_importance_v0": (held.qualified_id,),
        "width.global_pca_v0": (held.qualified_id,),
        "attention.activation_importance_v1": (held.qualified_id,
                                               branched.qualified_id),
    }
    found = run_phase_a_search(
        workdir=tmp_path / "search", state_eval=tmp_path / "unused", top_n=3,
        device="cpu", repo_root=tmp_path, teacher_id="toy",
        canonical_init="control", canonical_sha256=None,
        teacher_loader=lambda: teacher, target_geometry=target_geometry,
        suite_bundle=(suite, suite_items, {"teacher_sha256": "0" * 64}),
        calibration_items={held.qualified_id: items(7),
                           branched.qualified_id: items(31)},
        profiles=(held, branched),
        allowed_impls=SS.C2_ALLOWED_IMPLS,
        impl_profiles=restriction,
        run_id="toy.phase_c2.search1",
        purpose="toy execution of the C2 Search-1 shape")

    assert found.summary["run_id"] == "toy.phase_c2.search1"
    assert found.summary["summary"]["n_complete_leaves"] > 0

    #: The operator under test actually ran, and produced its own metric.
    states = list(found.result.states.values())
    attention_steps = [step for state in states for step in state.steps
                       if step.impl_id == "attention.activation_importance_v1"]
    assert attention_steps, "the C2 ATTENTION operator never expanded"
    assert any("op.attention.retained_write_energy_mean"
               in (step.local_metrics.values if step.local_metrics else {})
               for step in attention_steps)

    #: The restriction held where it was applied and not where it was not.
    for state in states:
        for step in state.steps:
            allowed = restriction.get(step.impl_id)
            if allowed:
                assert step.profile_id in allowed, (
                    f"{step.impl_id} ran under {step.profile_id}, which its "
                    "restriction excludes")
    assert {step.profile_id for state in states for step in state.steps
            if step.impl_id == "attention.activation_importance_v1"} == {
        held.qualified_id, branched.qualified_id}, (
        "ATTENTION did not branch over both mixtures")

    #: No excluded implementation appeared, which is what keeps the historical
    #: control an anchor rather than a competitor.
    ran = {step.impl_id for state in states for step in state.steps}
    assert ran <= set(SS.C2_ALLOWED_IMPLS), sorted(ran - set(SS.C2_ALLOWED_IMPLS))
