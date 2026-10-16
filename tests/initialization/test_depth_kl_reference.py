"""A DECLARED KL reference for DEPTH: the intact parent (default) or the root.

`scoring.kl_reference` lets a KL-scored operator's config declare which model
its candidates are compared with. The historical reference -- the parent the
operator is handed -- stays the default and is never written into a config, so
no recorded hash moves. ``root_reference/v1`` reads the stored lineage root.

Pinned here against independent computations:

* the declaration rules (default omitted, unknown/explicit-default refused,
  only accepting operators, never beside a consumed objective);
* the config identity (absent == historical bytes; declared forks the hash);
* DEPTH's arithmetic under the root reference: every decision score is the
  independent KL(root || parent-with-S-bypassed), every parent diagnostic the
  independent KL(parent || parent-with-S-bypassed), the choice follows the
  root and never the diagnostic;
* parity: where the parent IS the root, both references choose the same layers
  with the same scores;
* fail-closed store handling: missing, another mixture, another profile,
  another execution protocol, no execution binding;
* the fixed-path mechanism: stores built while the root is held, bound to the
  execution, checked when supplied to a suffix.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from support.toy import (  # noqa: E402
    TEACHER_GEOMETRY, build_tiny_model, make_items, make_profile,
)
from aadistill.initialization.calibration.items import (  # noqa: E402
    prepare_calibration_items,
)
from aadistill.initialization.execution import ExecutionConfig  # noqa: E402
from aadistill.initialization.operators.base import (  # noqa: E402
    ContractViolation, OperatorContext, OperatorError, get_implementation,
)
from aadistill.initialization.operators.depth import causal_kl_greedy  # noqa: E402
from aadistill.initialization.planning.operator_config import (  # noqa: E402
    hashed_operator_config, operator_config_hash, root_reference_needed,
)
from aadistill.initialization.scoring.kl_reference import (  # noqa: E402
    KL_REFERENCE_KEY, PARENT_STATE, ROOT_REFERENCE_V1, KLReferenceError,
    declared_kl_reference, requests_root_reference, validate_kl_reference,
)
from aadistill.initialization.scoring.objective import (  # noqa: E402
    FORM_REFERENCE_CONFIDENCE_WEIGHTED_KL, KL_REFERENCE_ROOT,
    ScoringObjective,
)
from aadistill.initialization.scoring.positions import (  # noqa: E402
    SUPERVISED_TARGET_V1, policy_config,
)
from aadistill.initialization.scoring.reference_store import (  # noqa: E402
    ReferenceStoreError, build_root_reference_store, root_reference_identity,
    store_execution,
)
from aadistill.initialization.scoring.support import (  # noqa: E402
    FULL_VOCAB_V1, reference_topk_tail, sketch_forward_kl, sketch_reference,
)
from aadistill.initialization.specs.arch import ArchSpec, get_adapter  # noqa: E402
from aadistill.initialization.statistics.contribution import (  # noqa: E402
    bypassed_blocks, domain_balanced_score,
)

K = 8
GEOMETRY = dict(TEACHER_GEOMETRY)
ROOT_ID = {"teacher_id": "toy/root@abc", "teacher_sha256": "ab" * 32}
PROFILE_HASH = "p" * 64
DECLARED = {KL_REFERENCE_KEY: ROOT_REFERENCE_V1}
OBJECTIVE = ScoringObjective("objective.toy_reference_confidence", 1,
                             FORM_REFERENCE_CONFIDENCE_WEIGHTED_KL,
                             kl_reference=KL_REFERENCE_ROOT)


@pytest.fixture(scope="module", autouse=True)
def _registered():
    from aadistill.initialization.adapters import register_builtin_adapters
    from aadistill.initialization.operators.register import (
        register_builtin_operators,
    )
    register_builtin_adapters()
    register_builtin_operators()
    yield


@pytest.fixture(scope="module")
def root():
    m = build_tiny_model(GEOMETRY)
    m.config.use_cache = False
    return m


@pytest.fixture(scope="module")
def parent():
    """A model that is NOT the root, standing in for a compressed parent."""
    m = build_tiny_model(GEOMETRY, seed=19)
    m.config.use_cache = False
    return m


def _items(seq_lens=(20, 14, 17, 11)):
    raw = make_items(n_per_subtype=2, seq_len=max(seq_lens))
    for item, n in zip(raw, seq_lens):
        item["input_ids"] = item["input_ids"][:, :n]
        m = n - 1
        item["tags"] = {k: v[:m] for k, v in item["tags"].items()}
        item["tags"]["assistant"] = torch.arange(m) >= 4
    return prepare_calibration_items(raw, profile_id="toy@v1")


def _execution(bsz):
    return ExecutionConfig(
        micro_batch_size=bsz,
        calibration_batch_packing=("length_sorted_v1" if bsz > 1
                                   else "original_order_v1"))


def _store(model, items, *, bsz=1, bind=True, profile_hash=PROFILE_HASH,
           execution_bsz=None):
    support = reference_topk_tail(K)
    ex = _execution(bsz if execution_bsz is None else execution_bsz)
    return build_root_reference_store(
        model, items, support=support,
        identity=root_reference_identity(
            root_identity=ROOT_ID, profile_hash=profile_hash, support=support,
            execution=(store_execution(
                micro_batch_size=ex.micro_batch_size,
                packing=ex.calibration_batch_packing, device="cpu",
                model=model) if bind else None)),
        micro_batch_size=bsz,
        packing=_execution(bsz).calibration_batch_packing, device="cpu")


class _Profile:
    profile_hash = PROFILE_HASH


def _ctx(model, store, *, declared=True, bsz=1, rows="all_v1", items=None):
    items = items if items is not None else _items()
    support = reference_topk_tail(K)
    config = {"n_calibration_items": len(items),
              **policy_config(SUPERVISED_TARGET_V1),
              "distribution_support": support.as_dict(),
              **(DECLARED if declared else {})}
    ex = _execution(bsz)
    return OperatorContext(
        adapter=get_adapter("qwen3"), model=model,
        parent_spec=ArchSpec.of("qwen3", GEOMETRY),
        target_spec=ArchSpec.of("qwen3", {**GEOMETRY, "num_hidden_layers": 4}),
        profile=_Profile(), calibration_items=items, seed=0, device="cpu",
        config=config, distribution_support=support,
        execution=ExecutionConfig(
            micro_batch_size=ex.micro_batch_size,
            calibration_batch_packing=ex.calibration_batch_packing,
            prediction_rows=rows),
        position_policy=SUPERVISED_TARGET_V1, root_reference=store), items


def _depth(model, store, **kw):
    ctx, items = _ctx(model, store, **kw)
    return causal_kl_greedy.DepthCausalKLGreedyV1().execute(ctx), items


def _independent_kl(reference, model, items, skip):
    """KL(reference || model-with-skip-bypassed) from first principles: the
    reference's Top-K + tail sketch, the bypassed logits, the mean over the
    supervised positions, then the sub-type and domain rule."""
    per_subtype: dict[str, list[float]] = {}
    for item in items:
        ids = item["input_ids"]
        with torch.no_grad():
            ref = reference(ids).logits[0, :-1].float()
            with bypassed_blocks(model, set(skip)):
                cand = model(ids).logits[0, :-1].float()
        sk = sketch_reference(ref, ids[0, 1:], top_k=K)
        kl = sketch_forward_kl(sk.support_indices, sk.support_log_probs,
                               sk.tail_log_prob, cand, has_tail=True)
        scored = SUPERVISED_TARGET_V1.weights(
            item, axis="prediction").weights > 0
        per_subtype.setdefault(item["subtype"], []).append(
            float(kl[scored].double().mean()))
    domains: dict[str, set] = {}
    for item in items:
        domains.setdefault(item["domain"], set()).add(item["subtype"])
    return domain_balanced_score(
        {k: sum(v) / len(v) for k, v in per_subtype.items()},
        {d: sorted(s) for d, s in domains.items()})[0]


# --- 1. the declaration ---------------------------------------------------------

class TestTheDeclaration:

    def test_absent_means_the_historical_parent(self):
        assert declared_kl_reference(None) == PARENT_STATE
        assert declared_kl_reference({}) == PARENT_STATE
        assert declared_kl_reference(DECLARED) == ROOT_REFERENCE_V1

    @pytest.mark.parametrize("value", [PARENT_STATE, "root", "teacher",
                                       "root_reference/v2", None])
    def test_unknown_and_the_spelled_default_are_refused(self, value):
        with pytest.raises(KLReferenceError):
            declared_kl_reference({KL_REFERENCE_KEY: value})

    def test_only_an_accepting_operator_may_be_given_one(self):
        depth = get_implementation("depth.causal_kl_greedy_v1")
        ffn = get_implementation("ffn.activation_importance_v0")
        assert depth.accepts_kl_reference and not ffn.accepts_kl_reference
        assert validate_kl_reference(depth, DECLARED,
                                     objective_consumed=False) == ROOT_REFERENCE_V1
        with pytest.raises(KLReferenceError, match="does not accept"):
            validate_kl_reference(ffn, DECLARED, objective_consumed=False)
        assert not requests_root_reference(ffn, DECLARED)

    def test_a_declaration_beside_a_consumed_objective_is_a_conflict(self):
        depth = get_implementation("depth.causal_kl_greedy_v1")
        with pytest.raises(KLReferenceError, match="not resolved by precedence"):
            validate_kl_reference(depth, DECLARED, objective_consumed=True)

    def test_the_flag_is_not_part_of_any_signature(self):
        """Adding the capability moved no frozen implementation identity."""
        depth = get_implementation("depth.causal_kl_greedy_v1")
        assert "accepts_kl_reference" not in depth.signature()

    def test_one_rule_says_who_needs_the_root(self):
        depth = get_implementation("depth.causal_kl_greedy_v1")
        ffn = get_implementation("ffn.activation_importance_v0")
        assert root_reference_needed(depth, None, DECLARED)
        assert not root_reference_needed(depth, None, None)
        assert not root_reference_needed(depth, None, {})
        assert not root_reference_needed(ffn, None, DECLARED)
        #: The objective's own rule is untouched.
        assert root_reference_needed(depth, OBJECTIVE)


# --- 2. the config identity -----------------------------------------------------

class TestTheConfigIdentity:

    def _config(self, impl_id, declared=None, objective=None):
        return hashed_operator_config(
            implementation=get_implementation(impl_id),
            policy=SUPERVISED_TARGET_V1, support=reference_topk_tail(K),
            items=_items(), declared=declared, objective=objective)

    def test_absent_is_the_historical_config_byte_for_byte(self):
        from aadistill.initialization.planning.operator_config import (
            distribution_support_config, position_policy_config,
        )
        impl = get_implementation("depth.causal_kl_greedy_v1")
        items = _items()
        plain = self._config("depth.causal_kl_greedy_v1")
        assert KL_REFERENCE_KEY not in plain
        #: The pre-existing contributors and nothing else.
        assert plain == {
            "n_calibration_items": len(items),
            **position_policy_config(impl, SUPERVISED_TARGET_V1, items),
            **distribution_support_config(reference_topk_tail(K))}

    def test_the_declaration_forks_the_hash_and_only_by_its_key(self):
        plain = self._config("depth.causal_kl_greedy_v1")
        declared = self._config("depth.causal_kl_greedy_v1", DECLARED)
        assert operator_config_hash(plain) != operator_config_hash(declared)
        assert {k: v for k, v in declared.items()
                if k != KL_REFERENCE_KEY} == plain

    def test_refusals_happen_where_every_caller_builds_the_config(self):
        with pytest.raises(ValueError, match="does not accept"):
            self._config("ffn.activation_importance_v0", DECLARED)
        with pytest.raises(ValueError, match="historical default"):
            self._config("depth.causal_kl_greedy_v1",
                         {KL_REFERENCE_KEY: PARENT_STATE})
        with pytest.raises(ValueError, match="precedence"):
            self._config("depth.causal_kl_greedy_v1", DECLARED,
                         objective=OBJECTIVE)


# --- 3. DEPTH's arithmetic --------------------------------------------------------

class TestDepthUnderTheRootReference:

    def test_every_decision_score_is_the_independent_root_kl(self, root, parent):
        out, items = _depth(parent, _store(root, _items()))
        for rnd in out.artifacts["search_rounds"][:2]:
            removed = set(rnd["removed_before"])
            for entry in rnd["table"][:3]:
                want = _independent_kl(root, parent, items,
                                       removed | {entry["candidate"]})
                assert entry["score"] == pytest.approx(want, rel=1e-4, abs=1e-6)

    def test_every_parent_diagnostic_is_the_independent_parent_kl(
            self, root, parent):
        out, items = _depth(parent, _store(root, _items()))
        cands = out.artifacts["reference_candidates"]
        #: Every candidate of EVERY round carries the diagnostic.
        n_evaluated = sum(r["n_candidates"]
                          for r in out.artifacts["search_rounds"])
        assert len(cands) == n_evaluated
        assert {c["round"] for c in cands.values()} == set(
            range(len(out.artifacts["search_rounds"])))
        for key in list(cands)[:4] + list(cands)[-3:]:
            skip = {int(x) for x in key.split(",")}
            want = _independent_kl(parent, parent, items, skip)
            assert cands[key]["parent_reference_kl"] == pytest.approx(
                want, rel=1e-4, abs=1e-6)
        assert out.artifacts["parent_kl_diagnostic"] == "every round's candidates"

    def test_the_choice_follows_the_root_and_never_the_diagnostic(
            self, root, parent):
        out, _ = _depth(parent, _store(root, _items()))
        cands = out.artifacts["reference_candidates"]
        disagreements = 0
        for rnd in out.artifacts["search_rounds"]:
            removed = sorted(rnd["removed_before"])
            table = {e["candidate"]: e["score"] for e in rnd["table"]}
            #: The decision table IS the root-reference KL ...
            for c, s in table.items():
                key = ",".join(str(x) for x in sorted(removed + [c]))
                assert cands[key]["root_reference_kl"] == s
            best = min(table.items(), key=lambda kv: (kv[1], kv[0]))[0]
            assert rnd["chosen"] == best
            by_parent = min(
                table, key=lambda c: (cands[",".join(str(x) for x in sorted(
                    removed + [c]))]["parent_reference_kl"], c))
            disagreements += by_parent != rnd["chosen"]
        #: The two rankings genuinely differ on this fixture, so agreement
        #: above is the selection reading the root, not a coincidence.
        assert disagreements >= 1

    def test_it_records_the_reference_it_actually_read(self, root, parent):
        out, _ = _depth(parent, _store(root, _items()))
        assert out.local_metrics.reference == "root_reference"
        assert out.local_metrics.detail[KL_REFERENCE_KEY] == ROOT_REFERENCE_V1
        assert out.trace[KL_REFERENCE_KEY] == ROOT_REFERENCE_V1
        assert out.artifacts["reference_cache"]["mode"] == "root_reference_store"
        assert out.artifacts["reference_counters"]["reference_hits"] > 0
        assert "op.depth.parent_kl_diagnostic.final" in out.local_metrics.values

    def test_at_the_root_both_references_choose_the_same_layers(self, root):
        """NUMERICAL PARITY: where the parent IS the root the stored root
        sketch is the sketch DEPTH computes for itself, so the declared root
        reference reproduces the historical parent reference exactly."""
        for bsz in (1, 3):
            historical, _ = _depth(root, None, declared=False, bsz=bsz)
            declared, _ = _depth(root, _store(root, _items(), bsz=bsz), bsz=bsz)
            assert declared.trace["removal_order"] == \
                historical.trace["removal_order"]
            for ra, rb in zip(historical.artifacts["search_rounds"],
                              declared.artifacts["search_rounds"]):
                for ea, eb in zip(ra["table"], rb["table"]):
                    assert eb["score"] == pytest.approx(ea["score"], rel=1e-6,
                                                        abs=1e-9)
            for rec in declared.artifacts["reference_candidates"].values():
                assert rec["parent_reference_kl"] == pytest.approx(
                    rec["root_reference_kl"], rel=1e-6, abs=1e-9)

    def test_batch_one_and_three_agree(self, root, parent):
        one, _ = _depth(parent, _store(root, _items()), bsz=1)
        three, _ = _depth(parent, _store(root, _items(), bsz=3), bsz=3)
        assert one.trace["removal_order"] == three.trace["removal_order"]
        for ra, rb in zip(one.artifacts["search_rounds"],
                          three.artifacts["search_rounds"]):
            for ea, eb in zip(ra["table"], rb["table"]):
                assert eb["score"] == pytest.approx(ea["score"], rel=1e-4,
                                                    abs=1e-6)

    @pytest.mark.parametrize("bsz", [1, 3])
    def test_scored_rows_choose_identically(self, root, parent, bsz):
        store = _store(root, _items(), bsz=bsz)
        base, _ = _depth(parent, store, bsz=bsz)
        rows, _ = _depth(parent, store, bsz=bsz, rows="scored_v1")
        assert rows.trace["kept_layers"] == base.trace["kept_layers"]
        for key, rec in base.artifacts["reference_candidates"].items():
            got = rows.artifacts["reference_candidates"][key]
            for f in ("root_reference_kl", "parent_reference_kl"):
                assert got[f] == pytest.approx(rec[f], rel=1e-5, abs=1e-8)

    def test_it_is_deterministic(self, root, parent):
        store = _store(root, _items())
        a, _ = _depth(parent, store)
        b, _ = _depth(parent, store)
        assert a.trace == b.trace
        assert a.artifacts["reference_candidates"] == \
            b.artifacts["reference_candidates"]


class TestTheHistoricalParentPathIsUnchanged:

    def test_no_declaration_records_no_reference_fields(self, parent):
        out, _ = _depth(parent, None, declared=False)
        assert out.local_metrics.reference == "parent_state"
        assert KL_REFERENCE_KEY not in out.trace
        assert KL_REFERENCE_KEY not in out.local_metrics.detail
        assert "reference_candidates" not in out.artifacts
        assert "parent_reference_cache" not in out.artifacts
        assert "op.depth.parent_kl_diagnostic.final" not in \
            out.local_metrics.values

    def test_an_undeclared_invocation_ignores_a_handed_store(self, root,
                                                             parent):
        """The parent path does not start reading the root because a store
        happens to be present: the declaration decides, not the context."""
        plain, _ = _depth(parent, None, declared=False)
        with_store, _ = _depth(parent, _store(root, _items()), declared=False)
        assert with_store.trace == plain.trace
        assert with_store.local_metrics.reference == "parent_state"


# --- 4. refusals --------------------------------------------------------------------

class TestRefusals:

    def test_no_store_is_a_contract_violation(self, parent):
        ctx, _ = _ctx(parent, None)
        with pytest.raises(ContractViolation, match="ROOT"):
            causal_kl_greedy.DepthCausalKLGreedyV1().execute(ctx)

    def test_a_store_of_another_mixture_is_refused(self, root, parent):
        ctx, _ = _ctx(parent, _store(root, _items()[:2]))
        with pytest.raises(ReferenceStoreError, match="another"):
            causal_kl_greedy.DepthCausalKLGreedyV1().execute(ctx)

    def test_a_store_of_another_profile_is_refused(self, root, parent):
        ctx, _ = _ctx(parent, _store(root, _items(), profile_hash="q" * 64))
        with pytest.raises(ReferenceStoreError, match="profile_hash"):
            causal_kl_greedy.DepthCausalKLGreedyV1().execute(ctx)

    def test_a_store_without_an_execution_binding_is_refused(self, root,
                                                             parent):
        ctx, _ = _ctx(parent, _store(root, _items(), bind=False))
        with pytest.raises(ReferenceStoreError, match="no execution"):
            causal_kl_greedy.DepthCausalKLGreedyV1().execute(ctx)

    def test_a_store_of_another_execution_protocol_is_refused(self, root,
                                                              parent):
        ctx, _ = _ctx(parent, _store(root, _items(), execution_bsz=3), bsz=1)
        with pytest.raises(ReferenceStoreError, match="micro_batch_size"):
            causal_kl_greedy.DepthCausalKLGreedyV1().execute(ctx)

    def test_full_vocabulary_cannot_carry_a_root_reference(self, root, parent):
        import dataclasses

        ctx, _ = _ctx(parent, _store(root, _items()))
        ctx = dataclasses.replace(
            ctx, distribution_support=FULL_VOCAB_V1,
            config={k: v for k, v in ctx.config.items()
                    if k != "distribution_support"})
        with pytest.raises(OperatorError, match="full_vocab_v1"):
            causal_kl_greedy.DepthCausalKLGreedyV1().apply(ctx)


# --- 5. the fixed-path mechanism ------------------------------------------------------

PROFILE_ID = "test.klref@v1"


@pytest.fixture
def registered_profile():
    from aadistill.initialization.calibration.profiles import (
        register_profile, unregister_profile,
    )
    p = make_profile("klref")
    register_profile(p, replace=True)
    yield p
    unregister_profile(p.profile_id)


def _spec(*, declared=True, objective=None, ffn_declares=False):
    from aadistill.initialization.planning.fixed_path import (
        FixedPathSpec, FixedPathStep,
    )
    return FixedPathSpec(
        path_id="test.klref", family="qwen3",
        target_spec=ArchSpec.of("qwen3", {**GEOMETRY, "num_hidden_layers": 4,
                                          "intermediate_size": 24}),
        steps=(FixedPathStep("depth.causal_kl_greedy_v1", PROFILE_ID,
                             config=DECLARED if declared else None),
               FixedPathStep("ffn.activation_importance_v0", PROFILE_ID,
                             config=DECLARED if ffn_declares else None)),
        root_repo_id="test/teacher", root_revision="deadbeef",
        position_policy=SUPERVISED_TARGET_V1,
        distribution_support=reference_topk_tail(K),
        scoring_objective=objective)


class TestTheFixedPath:

    def test_construction_refuses_what_cannot_run(self):
        from aadistill.initialization.planning.fixed_path import FixedPathError
        with pytest.raises(FixedPathError, match="does not accept"):
            _spec(ffn_declares=True)
        with pytest.raises(FixedPathError, match="precedence"):
            _spec(objective=OBJECTIVE)

    def test_the_declaration_is_in_the_spec_identity(self):
        assert _spec().spec_hash != _spec(declared=False).spec_hash
        assert _spec().as_dict()["steps"][0]["config"] == DECLARED
        assert "config" not in _spec(declared=False).as_dict()["steps"][0]

    def test_the_store_is_built_from_the_root_and_bound_to_the_execution(
            self, registered_profile, tmp_path):
        from aadistill.initialization.adapters.qwen3 import QWEN3_ADAPTER
        from aadistill.initialization.planning.fixed_path import (
            materialize_fixed_path,
        )
        seen: list = []

        def on_step(result):
            seen.append(result)

        root_identity = {"teacher_id": "test/teacher@deadbeef",
                         "teacher_sha256": "cd" * 32}
        out = materialize_fixed_path(
            _spec(), adapter=QWEN3_ADAPTER,
            root_loader=lambda: build_tiny_model(GEOMETRY), workdir=tmp_path,
            calibration_items={PROFILE_ID: make_items()},
            execution=_execution(3), root_identity=root_identity,
            on_step=on_step)
        depth = out[0]
        assert depth.trace[KL_REFERENCE_KEY] == ROOT_REFERENCE_V1
        assert depth.local_metrics["op.depth.parent_kl_diagnostic.final"] >= 0
        #: The round tables and both references' records reach the persisted
        #: step result; the cache decisions travel as operations, not selection.
        record = depth.as_dict()
        assert len(record["selection"]["search_rounds"]) == 2
        assert record["selection"]["reference_candidates"]
        ops = record["operations"]
        assert ops["reference_cache"]["mode"] == "root_reference_store"
        execution = ops["reference_cache"]["identity_hash"]
        assert execution and ops["parent_reference_cache"]["mode"] in (
            "cached", "recomputed")
        assert ops["parent_kl_diagnostic"] == "every round's candidates"
        #: The FFN step reports no operations of this kind.
        assert "reference_candidates" not in out[1].as_dict()["selection"]
        #: The FFN step reads no reference and was handed none.
        assert KL_REFERENCE_KEY not in out[1].trace
        assert len(seen) == 2

    def test_a_suffix_without_the_root_needs_the_store(
            self, registered_profile, tmp_path):
        from aadistill.initialization.adapters.qwen3 import QWEN3_ADAPTER
        from aadistill.initialization.planning.fixed_path import (
            FixedPathError, _run_steps,
        )
        model = build_tiny_model(GEOMETRY)
        with pytest.raises(FixedPathError, match="declared KL reference"):
            _run_steps(_spec(), adapter=QWEN3_ADAPTER, model=model,
                       indices=[0], parent_spec=QWEN3_ADAPTER.spec_of(model),
                       parent_digest="f" * 64, workdir=tmp_path,
                       repo_root=".", stats_cache=None,
                       calibration_items={PROFILE_ID: make_items()},
                       on_step=None, deadline=None)

    def test_a_supplied_store_of_another_root_is_refused(
            self, registered_profile, tmp_path):
        from aadistill.initialization.adapters.qwen3 import QWEN3_ADAPTER
        from aadistill.initialization.planning.fixed_path import (
            FixedPathError, _run_steps,
        )
        model = build_tiny_model(GEOMETRY)
        items = prepare_calibration_items(make_items(), profile_id=PROFILE_ID)
        support = reference_topk_tail(K)
        store = build_root_reference_store(
            model, items, support=support,
            identity=root_reference_identity(
                root_identity={"teacher_id": "another/root"},
                profile_hash=registered_profile.profile_hash, support=support,
                execution=store_execution(micro_batch_size=3,
                                          packing="length_sorted_v1",
                                          device="cpu", model=model)),
            micro_batch_size=3, packing="length_sorted_v1", device="cpu")
        with pytest.raises(FixedPathError, match="declares root"):
            _run_steps(_spec(), adapter=QWEN3_ADAPTER, model=model,
                       indices=[0], parent_spec=QWEN3_ADAPTER.spec_of(model),
                       parent_digest="f" * 64, workdir=tmp_path,
                       repo_root=".", stats_cache=None,
                       calibration_items={PROFILE_ID: make_items()},
                       on_step=None, deadline=None,
                       execution=_execution(3),
                       root_identity={"teacher_id": "test/teacher@deadbeef"},
                       root_references={
                           registered_profile.profile_hash: store})
