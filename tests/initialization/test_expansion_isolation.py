"""One expansion must not change what a sibling expansion produces.

    same parent artifact + same operator + same profile + same protocol
    = same child artifact,  whatever sibling ran before it

THE DEFECT. `BeamSearch` caches the root model and hands the same object to
every level-0 expansion, because reloading a teacher per expansion is minutes of
GPU each. An operator that mutates the config of the model it is handed
therefore leaves that mutation visible to every later sibling — and
`build_config` carries a parent's whole `to_dict()` into its child, so the
mutation reaches the child's `config.json`, its `config_sha256` and its
`artifact_digest`. The identity then depends on **expansion order**.

It was found by a replay, not by a search: a reconstruction of two finalists
could not start from the published teacher, because the root the search had
expanded from was not the teacher as published. That replay cost a GPU minute
and $0.26 to discover, and it only recovered because the mutation happened to be
one documented boolean that a journal could resolve.

These tests assert the invariant at the level the search actually operates:
through the real `build_config`, with a model whose config really is mutated by
the thing it is handed to.
"""
from __future__ import annotations

import pytest

from aadistill.initialization.planning.isolation import (
    config_snapshot, isolate_parent_config, restore_config,
)


@pytest.fixture
def model():
    """A real config on a minimal object. No weights, no device, no family
    knowledge in the test: the quantity under test is `config.to_dict()`."""
    from transformers import Qwen3Config

    class _Model:
        def __init__(self):
            self.config = Qwen3Config(
                hidden_size=32, num_hidden_layers=4, intermediate_size=64,
                num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                vocab_size=128, use_cache=True)

    return _Model()


class TestAMutationIsContainedToTheExpansionThatMadeIt:

    def test_the_parent_is_as_it_was_afterwards(self, model):
        with isolate_parent_config(model):
            model.config.use_cache = False
            assert model.config.use_cache is False, (
                "the operator must still see its own mutation; this contains "
                "it, it does not prevent it")
        assert model.config.use_cache is True

    def test_what_was_restored_is_reported(self, model):
        with isolate_parent_config(model) as touched:
            model.config.use_cache = False
        assert touched == {"use_cache": {"was": True, "became": False}}

    def test_a_record_dict_can_be_supplied(self, model):
        """So a telemetry payload is built without a second copy."""
        payload: dict = {}
        with isolate_parent_config(model, payload):
            model.config.use_cache = False
        assert payload["use_cache"]["became"] is False

    def test_an_untouched_parent_reports_nothing(self, model):
        with isolate_parent_config(model) as touched:
            pass
        assert touched == {}

    def test_restoration_happens_even_when_the_operator_raises(self, model):
        """The moment a shared parent is MOST likely to be left in a state no
        sibling expects is when an operator dies part-way through its own
        mutations."""
        with pytest.raises(RuntimeError):
            with isolate_parent_config(model) as touched:
                model.config.use_cache = False
                raise RuntimeError("the operator died mid-expansion")
        assert model.config.use_cache is True
        assert touched["use_cache"]["became"] is False

    def test_several_fields_at_once(self, model):
        with isolate_parent_config(model) as touched:
            model.config.use_cache = False
            model.config.attention_dropout = 0.5
        assert model.config.use_cache is True
        assert model.config.attention_dropout == 0.0
        assert set(touched) == {"use_cache", "attention_dropout"}

    def test_a_field_the_operator_added_is_reported_and_kept(self, model):
        """Not removed. Deleting an attribute a config added for its own
        reasons is more dangerous than recording it, and a config that grows a
        field is not the failure mode this protects against."""
        with isolate_parent_config(model) as touched:
            model.config.something_new = 7
        assert model.config.something_new == 7
        assert touched["something_new"]["added"] is True


class TestTheInvariantItself:
    """The property, stated as the thing a search depends on: a child's config
    identity must not depend on which sibling ran first."""

    def _child_sha(self, parent):
        import json
        import tempfile
        from pathlib import Path

        from aadistill.infrastructure.manifest import sha256_json
        from aadistill.initialization.adapters import register_builtin_adapters
        from aadistill.initialization.specs.arch import ArchSpec, get_adapter

        register_builtin_adapters()
        spec = ArchSpec.of("qwen3", {
            "hidden_size": 32, "num_hidden_layers": 4, "intermediate_size": 48,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
            "vocab_size": 128, "tie_word_embeddings": True})
        config = get_adapter("qwen3").build_config(parent.config, spec)
        with tempfile.TemporaryDirectory() as tmp:
            config.save_pretrained(tmp)
            return sha256_json(json.loads((Path(tmp) / "config.json").read_text()))

    def test_without_isolation_a_sibling_changes_the_next_childs_identity(
            self, model):
        """The defect, demonstrated. This is NOT the behaviour being shipped —
        it is why the next test matters."""
        first = self._child_sha(model)
        model.config.use_cache = False               # an earlier sibling
        second = self._child_sha(model)
        assert first != second, (
            "if these are equal, a config field no longer reaches a child's "
            "identity and this whole module is testing nothing")

    def test_with_isolation_the_order_of_siblings_does_not_matter(self, model):
        before = self._child_sha(model)
        with isolate_parent_config(model):
            model.config.use_cache = False           # a mutating sibling
            _ = self._child_sha(model)               # its own child differs
        after = self._child_sha(model)
        assert after == before, (
            "a later sibling's child identity still depends on which operator "
            "ran before it")


class TestItKnowsNothingAboutWhatItIsIsolating:
    """P3. A field name, a family or an operator in here would make the next
    search's isolation a fork of this one."""

    def test_no_field_or_family_vocabulary_in_the_module(self):
        from pathlib import Path

        from aadistill.initialization.planning import isolation

        src = Path(isolation.__file__).read_text()
        code = "\n".join(line for line in src.splitlines()
                         if not line.lstrip().startswith("#"))
        _, _, rest = code.partition('"""')
        _, _, body = rest.partition('"""')
        for word in ("qwen", "num_hidden_layers", "depth.", "ffn.",
                     "phase_", "d1", "causal_kl"):
            assert word not in body.lower(), (
                f"{word!r} is in the isolation code; the quantity it protects "
                "is whatever to_dict() said, not a named field")

    def test_use_cache_appears_only_in_the_docstring(self):
        """It is the mutation that exposed the defect, so it belongs in the
        explanation. A mechanism that only restores THAT field would miss the
        next one."""
        from pathlib import Path

        from aadistill.initialization.planning import isolation

        src = Path(isolation.__file__).read_text()
        code = "\n".join(line for line in src.splitlines()
                         if not line.lstrip().startswith("#"))
        _, _, rest = code.partition('"""')
        _, _, body = rest.partition('"""')
        assert "use_cache" not in body


class TestItDoesNotBreakOnThingsThatAreNotModels:
    """A search may legitimately be handed a bare module in a test, and
    isolation refusing to run would make the search untestable."""

    def test_an_object_with_no_config(self):
        class _Bare:
            pass

        bare = _Bare()
        assert config_snapshot(bare) is None
        with isolate_parent_config(bare) as touched:
            pass
        assert touched == {}

    def test_a_config_that_cannot_serialize(self):
        class _Hostile:
            def to_dict(self):
                raise TypeError("not serializable")

        class _Model:
            config = _Hostile()

        assert config_snapshot(_Model()) is None
        assert restore_config(_Model(), None) == {}

    def test_a_config_that_is_a_plain_namespace(self):
        class _Model:
            class config:                         # noqa: N801
                @staticmethod
                def to_dict():
                    return {"a": 1}

        #: A mapping comes back, so the snapshot is taken; nothing mutates it,
        #: so nothing is restored.
        assert config_snapshot(_Model()) == {"a": 1}


class TestTheSearchActuallyUsesIt:

    def test_the_expansion_is_wrapped_and_the_diff_is_recorded(self):
        """A mechanism with no production caller protects nothing."""
        import ast
        from pathlib import Path

        from aadistill.initialization.planning import search

        src = Path(search.__file__).read_text()
        assert "isolate_parent_config(parent_model" in src, (
            "the search no longer isolates the parent it hands to an operator")
        tree = ast.parse(src)
        withs = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.With)
            and any(isinstance(item.context_expr, ast.Call)
                    and getattr(item.context_expr.func, "id", "")
                    == "isolate_parent_config"
                    for item in node.items)
        ]
        assert withs, "isolate_parent_config is imported but never entered"
        executed = [
            node for w in withs for node in ast.walk(w)
            if isinstance(node, ast.Call)
            and getattr(node.func, "attr", "") == "execute"
        ]
        assert executed, (
            "the operator call is outside the isolation block, so the mutation "
            "it makes is not contained")
        assert "parent_config_restored=dict(config_touched)" in src, (
            "the diff is undone but not recorded; a silently restored "
            "mutation is one nobody knows about")
