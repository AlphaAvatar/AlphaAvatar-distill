"""The historical root state a D1 replay must start from, derived not assumed.

THE FAILURE THIS CLOSES. The replay's fourth paid attempt reached q2's step 0,
ran `ffn.activation_importance_v0` for 64 s on an L40S, and produced

    expected 292e36f13829545b465ac2a19fa43d5f7c63d319bd841b269553a2717b3995f0
    actual   524493fde06f5b4139738affb4613347786cbba0f045dd562d06f7ccf8df4de9

The driver carried a comment asserting that D1's search had one root state and
therefore needed no reconstruction. The pre-launch check behind that comment had
compared each level-0 expansion's OPERATOR `config_hash` -- which is uniform by
construction, one value per calibration profile -- and not the MODEL config. A
child's config is `parent.config.to_dict()` with the spec applied, so every
field of the root's live config reaches every descendant's `config_sha256` and
therefore its `artifact_digest`. The search's driver sets
`use_cache = False` on the loaded teacher; the replay loaded the teacher as
published, with `use_cache: true`.

The repair is not a better comment. It is that the root state is now COMPUTED
against the recorded identities before any weights load:

    candidate set from the mechanism
      -> each candidate's reconstructed step-0 config sha
      -> the recorded sha picks exactly one, or it is material

These tests run on CPU, build no model, download nothing, read neither the 67 MB
journal nor the staged plan, and so are safe in a pod's test gate -- which is
where a previous version of a replay regression failed, for $0.1849, by reading
an out-of-tree dev-box path.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
for _extra in ("src", "scripts", "scripts/experiments/stage-1", "scripts/pod"):
    if str(REPO / _extra) not in sys.path:
        sys.path.insert(0, str(REPO / _extra))


@pytest.fixture(scope="module")
def R():
    from aadistill.initialization.adapters import register_builtin_adapters

    register_builtin_adapters()
    from experiments.phase_d1 import replay_specs as module

    return module


@pytest.fixture(scope="module")
def toy_base():
    """A tiny config of the same family. No weights, no download, no device.

    Small enough to be obviously not D1's teacher, which is the point: the
    mechanism under test is "a root field reaches a child's config sha", and
    that is a property of the adapter and the serializer, not of one geometry.
    """
    from transformers import Qwen3Config

    return Qwen3Config(
        hidden_size=32, num_hidden_layers=4, intermediate_size=64,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        vocab_size=128, tie_word_embeddings=True, use_cache=True)


#: A toy geometry per step position. Only the FIELDS matter; the operator names
#: are labels here and nothing resolves them.
TOY_GEOMETRY = (
    {"hidden_size": 32, "num_hidden_layers": 4, "intermediate_size": 48,
     "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
     "vocab_size": 128, "tie_word_embeddings": True},
    {"hidden_size": 32, "num_hidden_layers": 3, "intermediate_size": 48,
     "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
     "vocab_size": 128, "tie_word_embeddings": True},
)


def _leaf(R, base, *, overrides, geometry=TOY_GEOMETRY, corrupt=None):
    """A toy leaf whose recorded config shas are the ones `overrides` produces.

    `corrupt` replaces the recorded sha at one step index, so a test can say
    "step 1 disagrees" without hand-writing a hash.
    """
    steps = []
    for index, fields in enumerate(geometry):
        sha = R._child_config_sha256(base, overrides=overrides,
                                     arch_fields=fields, family="qwen3")
        if corrupt is not None and index == corrupt:
            sha = "f" * 64
        steps.append(R.ReplayStep(
            index=index, kind="TOY", impl_id=f"toy.step_{index}",
            profile_id="calib.toy@v1", seed=7,
            state_id=f"toystate{index:024d}",
            expected_artifact_digest="a" * 64,
            expected_single_shard_sha256="b" * 64,
            expected_config_sha256=sha,
            arch_spec=tuple(sorted(fields.items()))))
    return R.ReplayLeaf(
        state_id="toyleaf" + "0" * 25, quality_position=1,
        path_label="TOY->TOY", steps=tuple(steps),
        root_teacher_id="toy/teacher@rev", root_teacher_sha256="c" * 64,
        target_spec_hash="d" * 64, seed=7, num_parameters=1,
        expected_artifact_digest="a" * 64, expected_weights_digest="e" * 64,
        expected_single_shard_sha256="b" * 64, expected_arch_signature="9" * 64)


class TestTheCandidateSetIsDerivedFromTheMechanism:
    """Two things can put the root in a state other than as-published, and the
    candidate set is exactly those two. If either mechanism moves, the set is
    stale and these fail rather than silently narrowing the derivation."""

    def test_the_session_driver_still_mutates_the_root_it_loads(self):
        src = (REPO / "scripts/pod/autoinit_d1_driver.py").read_text()
        assert "model.config.use_cache = False" in src, (
            "the search driver no longer sets use_cache on the root it loads, "
            "so ROOT_CANDIDATES describes a mechanism that is gone")

    def test_an_operator_still_mutates_the_model_it_is_handed_in_place(self):
        src = (REPO / "src/aadistill/initialization/operators/depth/"
                      "causal_kl_greedy.py").read_text()
        assert "model.config.use_cache = False" in src, (
            "the in-place config mutation the candidate set is derived from is "
            "gone; a historical identity that depended on it still needs a "
            "candidate for it, so the set must be re-derived, not shortened")

    def test_only_two_config_fields_are_assigned_in_place_anywhere_in_core(self):
        """The candidate set covers ONE field. That is only sound while the
        fields a model's live config can be assigned are accounted for. A third
        one would make the set incomplete, and a replay would resolve a root
        state that is right about `use_cache` and wrong about the new field --
        this bug again, one field over.

        Two exist. `use_cache` is the one operators really assign, and the
        candidate set varies it. `num_hidden_layers` is assigned only inside
        the adapter's `set_blocks`, behind `update_config`, and the next test
        is that no caller ever enables it."""
        found: dict[str, set[str]] = {}
        root = REPO / "src/aadistill/initialization"
        for path in sorted(root.rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text())):
                if not isinstance(node, ast.Assign):
                    continue
                for target in node.targets:
                    #: `<something>.config.<field> = ...`
                    if (isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Attribute)
                            and target.value.attr == "config"):
                        found.setdefault(target.attr, set()).add(
                            str(path.relative_to(REPO)))
        assert set(found) == {"use_cache", "num_hidden_layers"}, (
            f"the in-place config assignments in core are now {sorted(found)}; "
            "ROOT_CANDIDATES accounts for use_cache and the adapter's guarded "
            "layer count, and a new one needs a derivation of its own")
        assert found["num_hidden_layers"] == {
            "src/aadistill/initialization/adapters/qwen3.py"}

    def test_nothing_lets_the_adapter_rewrite_a_handed_models_layer_count(self):
        """`set_blocks(..., update_config=True)` on a model an operator was
        handed would leave the parent's config describing a geometry it no
        longer has, and the next expansion off a SHARED root would build its
        child from it. The one caller passes False, and says why."""
        hits = []
        for path in sorted((REPO / "src/aadistill").rglob("*.py")):
            src = path.read_text()
            if "def set_blocks" in src:
                src = src[:src.index("def set_blocks")] + \
                    src[src.index("def set_blocks"):].split("\n    def ", 1)[-1]
            for node in ast.walk(ast.parse(path.read_text())):
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "set_blocks"):
                    enabled = [kw for kw in node.keywords
                               if kw.arg == "update_config"
                               and not (isinstance(kw.value, ast.Constant)
                                        and kw.value.value is False)]
                    declared = any(kw.arg == "update_config"
                                   for kw in node.keywords)
                    if enabled or not declared:
                        hits.append(f"{path.relative_to(REPO)}:{node.lineno}")
        assert not hits, (
            f"set_blocks is called without update_config=False at {hits}; a "
            "handed model's layer count can now be rewritten in place")

    def test_non_root_parents_cannot_carry_sibling_state(self):
        """The derivation is confined to the ROOT because every other parent is
        reloaded from its checkpoint. If `_load_state_model` started caching,
        a mid-path parent could carry a sibling's mutation and the candidate
        set would need to cover positions, not just the root."""
        src = (REPO / "src/aadistill/initialization/planning/search.py").read_text()
        body = src[src.index("def _load_state_model"):]
        body = body[:body.index("\n    def ", 1)]
        assert "self.adapter.load(state.checkpoint_path" in body
        assert "cache" not in body.lower(), (
            "a non-root parent is being cached; sibling side effects can now "
            "reach a mid-path expansion and the root-state derivation is no "
            "longer sufficient")


class TestTheDerivationResolvesToExactlyOneRootState:

    def test_it_finds_the_state_the_recorded_identities_were_produced_under(
            self, R, toy_base):
        for name, overrides in R.ROOT_CANDIDATES:
            leaf = _leaf(R, toy_base, overrides=overrides)
            out = R.derive_root_state([leaf], base_config=toy_base)
            assert out["chosen_candidate"] == name
            assert out["config_overrides"] == dict(overrides)

    def test_the_losing_candidate_is_recorded_with_what_it_produced(
            self, R, toy_base):
        """Evidence, not an assertion. A derivation that recorded only its
        winner would be indistinguishable from a declaration."""
        leaf = _leaf(R, toy_base, overrides={"use_cache": False})
        out = R.derive_root_state([leaf], base_config=toy_base)
        losers = [c for c in out["evidence"]
                  if not c["explains_every_step_0"]]
        assert losers, "every candidate explained the record; nothing was ruled out"
        for cand in losers:
            for step in cand["steps"]:
                assert len(step["reconstructed_config_sha256"]) == 64
                assert step["reconstructed_config_sha256"] != \
                    step["recorded_config_sha256"]

    def test_every_later_step_is_checked_under_the_chosen_root(self, R, toy_base):
        leaf = _leaf(R, toy_base, overrides={"use_cache": False})
        out = R.derive_root_state([leaf], base_config=toy_base)
        assert len(out["lineage_verified"]) == len(TOY_GEOMETRY) - 1
        assert all(s["matches"] for s in out["lineage_verified"])


class TestAnUnresolvableDerivationIsMaterialAndNotPapered(
        object):
    """The three ways this can fail, all of which must raise rather than guess.

    The maintainer's stop condition is narrow: a mismatch AFTER the mechanism
    demonstrably reproduces the historical state. These are the cases where it
    demonstrably does not, and none of them may be repaired by widening the
    candidate set until something fits."""

    def test_no_candidate_explaining_step_zero_raises(self, R, toy_base):
        leaf = _leaf(R, toy_base, overrides={"use_cache": False}, corrupt=0)
        with pytest.raises(R.RootStateUndetermined) as exc:
            R.derive_root_state([leaf], base_config=toy_base)
        assert "no derived root state reproduces" in str(exc.value)

    def test_a_later_step_that_disagrees_raises_and_names_it(self, R, toy_base):
        """Step 0 picks the candidate; the rest confirm it. A path whose step 0
        agrees and whose step 1 does not means some execution state outside the
        candidate set reached that checkpoint -- which is exactly the class of
        bug this whole file exists for, one position along."""
        leaf = _leaf(R, toy_base, overrides={"use_cache": False}, corrupt=1)
        with pytest.raises(R.RootStateUndetermined) as exc:
            R.derive_root_state([leaf], base_config=toy_base)
        message = str(exc.value)
        assert "not 1 later one" in message
        assert "toy.step_1" in message

    def test_an_ambiguous_derivation_is_refused(self, R, toy_base, monkeypatch):
        """Two candidates that both explain the record mean the recorded
        identity does not discriminate, so the check has not determined
        anything and must not report that it has."""
        monkeypatch.setattr(R, "ROOT_CANDIDATES", (
            ("first", {"use_cache": False}),
            ("second_same_state", {"use_cache": False}),
        ))
        leaf = _leaf(R, toy_base, overrides={"use_cache": False})
        with pytest.raises(R.RootStateUndetermined) as exc:
            R.derive_root_state([leaf], base_config=toy_base)
        assert "does not pick one" in str(exc.value)

    def test_a_plan_with_no_step_zero_config_identity_is_refused(self, R, toy_base):
        """A plan that did not carry the fingerprint would send a pod to a root
        nothing checked -- the state before this repair."""
        leaf = _leaf(R, toy_base, overrides={"use_cache": False})
        doc = R.describe([leaf], {"chosen_candidate": "x"})
        doc["leaves"][0]["steps"][0]["expected_config_sha256"] = ""
        with pytest.raises(R.ReplaySourceError) as exc:
            R.leaves_from_plan(doc)
        assert "root state cannot be derived or checked" in str(exc.value)

    def test_the_pod_rederives_the_plans_root_state_and_refuses_disagreement(
            self, R, toy_base):
        leaf = _leaf(R, toy_base, overrides={"use_cache": False})
        doc = R.describe(
            [leaf], {"chosen_candidate": "teacher_as_published",
                     "config_overrides": {}})
        rebuilt = R.leaves_from_plan(doc)
        with pytest.raises(R.RootStateUndetermined) as exc:
            R.root_state_from_plan(doc, rebuilt, base_config=toy_base)
        assert "the plan ships root state" in str(exc.value)

    def test_an_agreeing_plan_passes_and_says_so(self, R, toy_base):
        leaf = _leaf(R, toy_base, overrides={"use_cache": False})
        state = R.derive_root_state([leaf], base_config=toy_base)
        doc = R.describe([leaf], state)
        out = R.root_state_from_plan(doc, R.leaves_from_plan(doc),
                                     base_config=toy_base)
        assert out["chosen_candidate"] == state["chosen_candidate"]
        assert out["agreed_with_plan"] is True


class TestTheDriverUsesTheDerivedRootAndNothingElse:
    """The derivation is worthless if the loader ignores it. The first version
    of this driver had its own `load_root` with a bare `from_pretrained`."""

    def test_the_driver_has_no_loader_of_its_own(self):
        src = (REPO / "scripts/pod/autoinit_d1_replay_driver.py").read_text()
        tree = ast.parse(src)
        bare = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "from_pretrained"
        ]
        assert not bare, (
            "the replay driver calls from_pretrained directly; the root state "
            "must come from replay_specs.load_root_model, which is where the "
            "derived overrides are applied")

    def test_the_driver_derives_before_it_materializes(self):
        """Order matters for money: deriving costs two config hashes, and
        materializing a wrong root costs a GPU minute per path at best."""
        src = (REPO / "scripts/pod/autoinit_d1_replay_driver.py").read_text()
        assert src.index("root_state_from_plan") < src.index(
            "materialize_fixed_path(")

    def test_load_root_model_applies_the_derived_overrides(self, R, monkeypatch):
        """The loader, exercised. No weights: the call is intercepted and the
        config object it returns is the whole subject."""
        import transformers

        class _Config:
            use_cache = True

        class _Model:
            def __init__(self):
                self.config = _Config()
                self.moved_to = None
                self.evaled = False

            def to(self, device):
                self.moved_to = device
                return self

            def eval(self):
                self.evaled = True
                return self

        built = {}

        class _Auto:
            @staticmethod
            def from_pretrained(repo_id, **kwargs):
                built["repo_id"] = repo_id
                built["kwargs"] = kwargs
                return _Model()

        monkeypatch.setattr(transformers, "AutoModelForCausalLM", _Auto)

        class _Spec:
            root_repo_id = "toy/teacher"
            root_revision = "r" * 40
            device = "cpu"

        model = R.load_root_model(_Spec(), config_overrides={"use_cache": False})
        assert model.config.use_cache is False
        assert model.moved_to == "cpu" and model.evaled
        assert built["repo_id"] == "toy/teacher"
        assert built["kwargs"]["revision"] == "r" * 40


class TestTheRealD1DerivationIsCommittedAndUnique:
    """The staged plan. ASSERTED, not skipped.

    This began as `if not path.is_file(): pytest.skip(...)`, reasoning that a
    pod's gate might run before the plan was staged. The pod-like sweep then
    reported it as an **unexpected environment skip** -- and the sweep was
    right twice over. The plan is a declared `LocalAsset`, the pod's driver had
    already loaded it and reached its first operator, and what actually
    produced the skip was a defect in the SIMULATOR: `staged_files` collected a
    local asset's contents only `if tree.is_dir()`, so a one-file asset
    contributed nothing and fell into the hidden complement.

    The skip was therefore load-bearing in the worst way: it made a wrong
    model of the pod look fine. Asserting instead means the next session that
    stages a file and cannot see it gets told, rather than passing quietly.
    """

    def test_the_plan_is_staged_where_the_driver_reads_it(self):
        path = REPO / "artifacts/stage1/d1_replay_plan.json"
        assert path.is_file(), (
            f"{path.relative_to(REPO)} is not here. The driver is invoked with "
            "`--plan` pointing at exactly this path, so a pod without it dies "
            "after setup has been paid for; resolve it with "
            "`autoinit_d1_replay_launch.py --write-plan`, which costs $0.")

    def test_the_plan_carries_a_unique_derivation(self):
        path = REPO / "artifacts/stage1/d1_replay_plan.json"
        state = json.loads(path.read_text())["root_state"]
        assert state is not None, "the plan ships no root-state derivation"
        explain = [c for c in state["evidence"] if c["explains_every_step_0"]]
        assert len(explain) == 1
        assert explain[0]["candidate"] == state["chosen_candidate"]
        assert state["lineage_verified"]
        assert all(s["matches"] for s in state["lineage_verified"])
