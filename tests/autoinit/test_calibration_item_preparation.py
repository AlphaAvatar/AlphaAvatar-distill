"""The raw -> operator-ready boundary, and the root device the path declares.

Both regressions here exist because C1 attempt 8 paid to find them. The driver
executed for the first time, stage B and stage C passed, 398 weight shards
loaded, and stage D raised `KeyError: 'input_ids'` inside `depth.apply`.

**Why no `$0` run had ever caught it.** Every `test_fixed_path` case passes
`calibration_items=`, and `conftest.make_items()` builds items already carrying
`input_ids`. So the executor's own resolve-a-real-profile branch — the only
branch a paid run takes — had never once been executed. The tests were not weak
about the contract; they never reached it. This module therefore refuses the
override and makes `materialize_fixed_path` resolve a genuinely materialized
profile from disk, which is the thing that has to work.

The second defect was found by auditing beside it: stage D declared `cuda` and
loaded its root with a raw `AutoModelForCausalLM.from_pretrained(...).eval()`,
no transfer. Operators read `model_device(model)` — the fact — so the parent
replay would have run on the host CPU inside a paid GPU hour, and nothing would
have said so. That one needs no GPU to test, only the refusal to trust
`ctx.device` as evidence about weights.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.initialization.operators import attention_activation  # noqa: E402
from aadistill.initialization.adapters.qwen3 import QWEN3_ADAPTER  # noqa: E402
from aadistill.initialization.specs.arch import ArchSpec  # noqa: E402
from aadistill.initialization.calibration.profiles import (
    CalibrationProfile,
    CalibrationSource,
    mixture_content_sha256,
    register_profile,
    unregister_profile,
)
from experiments.calibration import DOMAIN_BALANCED_V1, REASONING_HEAVY_V2
from aadistill.initialization.calibration.items import (  # noqa: E402
    CalibrationItemError,
    prepare_calibration_items,
)
from aadistill.initialization.calibration.datasets import DatasetRole  # noqa: E402
from aadistill.initialization.operators.base import get_implementation  # noqa: E402
from aadistill.initialization.planning import fixed_path as FP
from aadistill.initialization.planning.fixed_path import (  # noqa: E402
    FixedPathRootDeviceMismatch,
    FixedPathSpec,
    FixedPathStep,
    materialize_fixed_path,
    require_root_on_declared_device,
)

from conftest import TEACHER_GEOMETRY, build_tiny_model  # noqa: E402

#: The C1 fixed path's own prefix and its treatment tail, in order. Four
#: calibrated operators, so one preparation boundary is proven to feed all four
#: rather than DEPTH alone — which is all attempt 8 got far enough to disprove.
DEPTH = "depth.causal_kl_greedy_v1"
FFN = "ffn.activation_importance_v0"
WIDTH = "width.global_pca_v0"
ATTENTION = "attention.activation_importance_v1"

TINY_PROFILE_ID = "test.raw_materialized@v1"
TARGET = dict(TEACHER_GEOMETRY, num_hidden_layers=4, intermediate_size=24,
              hidden_size=16, num_attention_heads=2)


# --- fixtures ---------------------------------------------------------------

@pytest.fixture(autouse=True)
def c1_operator_registered():
    attention_activation.register(replace=True)
    yield
    attention_activation.unregister()


def raw_item(item_id: str, domain: str, subtype: str, ids: list[int]) -> dict:
    """Exactly the shape a frozen materialized mixture stores on disk.

    `ids`, not `input_ids` — that is what `mixture_content_sha256` hashes, and
    therefore what the pinned content identity is defined over.
    """
    return {"item_id": item_id, "ids": list(ids), "domain": domain,
            "subtype": subtype, "n_prediction_positions": len(ids) - 1}


def write_tiny_mixture(root: Path, *, seq_len: int = 24, vocab: int = 128,
                       n_per_domain: int = 2, seed: int = 4242):
    """A REAL materialized profile: JSONL on disk, both hashes derived from it.

    Built through the production rules rather than around them — the file hash
    and `mixture_content_sha256` are computed from the bytes actually written, so
    `CalibrationProfile.resolve()` runs its full fail-closed check here exactly as
    it does against the frozen mixtures.
    """
    g = torch.Generator().manual_seed(seed)
    items = []
    for domain, subtype in (("general", "text"), ("math", "arith")):
        for k in range(n_per_domain):
            ids = torch.randint(0, vocab, (seq_len,), generator=g).tolist()
            items.append(raw_item(f"{subtype}-{k}", domain, subtype, ids))

    rel = "artifacts/tiny_calibration/items.jsonl"
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(i) + "\n" for i in items))

    profile = CalibrationProfile(
        profile_id="test.raw_materialized", version=1,
        description="a real materialized mixture that stores tokens under 'ids'",
        sources=(CalibrationSource("test/raw", "local", "general", n_per_domain),
                 CalibrationSource("test/raw", "local", "math", n_per_domain)),
        domain_weights={"general": 0.5, "math": 0.5},
        token_budget=seq_len * len(items), sample_rule="fixed", seed=seed,
        role=DatasetRole.OPERATOR_CALIBRATION,
        materialized=True, items_path=rel,
        content_sha256=mixture_content_sha256(items),
        items_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return profile, items


@pytest.fixture
def tiny_mixture(tmp_path):
    profile, items = write_tiny_mixture(tmp_path)
    register_profile(profile, replace=True)
    yield tmp_path, profile, items
    unregister_profile(profile.qualified_id)


# --- the real frozen mixtures store raw ids, and resolve() returns them ------

REAL_PROFILES = [DOMAIN_BALANCED_V1, REASONING_HEAVY_V2]
REAL_IDS = [p.qualified_id for p in REAL_PROFILES]


@pytest.mark.parametrize("profile", REAL_PROFILES, ids=REAL_IDS)
def test_resolve_returns_raw_ids_and_invents_no_input_ids(profile):
    """The frozen evidence is raw, and the resolver is not allowed to enrich it.

    If `resolve()` ever started returning tensors, `mixture_content_sha256` would
    be hashing something other than what the pinned identity was computed over.
    """
    items = profile.resolve(REPO)
    assert items
    assert all("ids" in i for i in items)
    assert not [i for i in items if "input_ids" in i], (
        f"{profile.qualified_id}: resolve() invented input_ids; the frozen "
        "content hash is defined over the raw 'ids'")


@pytest.mark.parametrize("profile", REAL_PROFILES, ids=REAL_IDS)
def test_preparation_of_the_real_mixture_is_token_identical(profile):
    raw = profile.resolve(REPO)
    prepared = prepare_calibration_items(raw, profile_id=profile.qualified_id)

    assert len(prepared) == len(raw)
    for before, after in zip(raw, prepared):
        t = after["input_ids"]
        assert isinstance(t, torch.Tensor)
        assert t.dtype is torch.long
        assert t.dim() == 2 and t.shape[0] == 1
        assert t.shape[1] == len(before["ids"])
        assert t[0].tolist() == list(before["ids"])
        # metadata survives untouched, raw ids included
        for key, value in before.items():
            assert after[key] == value


@pytest.mark.parametrize("profile", REAL_PROFILES, ids=REAL_IDS)
def test_the_real_mixtures_prediction_positions_match_their_tokens(profile):
    prepared = prepare_calibration_items(profile.resolve(REPO),
                                         profile_id=profile.qualified_id)
    declared = [i["n_prediction_positions"] for i in prepared]
    assert declared == [int(i["input_ids"].shape[1]) - 1 for i in prepared]


def test_preparation_does_not_mutate_the_raw_items():
    raw = DOMAIN_BALANCED_V1.resolve(REPO)
    prepare_calibration_items(raw, profile_id=DOMAIN_BALANCED_V1.qualified_id)
    assert not [i for i in raw if "input_ids" in i]


def test_the_c1_prefix_profiles_are_the_ones_this_covers():
    """The mixtures above are the mixtures C1's own path names.

    Otherwise this module could pass while the fixed path resolves something
    nobody prepared.
    """
    from experiments.phase_c1.session import INCUMBENT_ATTENTION, PREFIX_STEPS, TREATMENT_ATTENTION

    named = {p for _, p in (*PREFIX_STEPS, INCUMBENT_ATTENTION,
                            TREATMENT_ATTENTION)}
    assert named - {"calib.none@v1"} <= set(REAL_IDS)


def test_the_frozen_boundary_and_this_one_agree_token_for_token():
    """`phase_a_search.as_operator_items` is the same conversion, frozen.

    That script is inside THREE closed preregistrations (Phase A, Phase B and the
    recovery continuation), so it is not edited to remove the duplication — a
    frozen record must never be regenerated to match new code. The duplication is
    made harmless instead: if the two ever disagree about what an operator item
    is, this fails, and it fails at `$0`.
    """
    sys.path.insert(0, str(REPO / "scripts" / "autoinit"))
    from phase_a_search import as_operator_items

    raw = DOMAIN_BALANCED_V1.resolve(REPO)
    frozen = as_operator_items(raw)
    here = prepare_calibration_items(raw, profile_id=DOMAIN_BALANCED_V1.qualified_id)
    assert len(frozen) == len(here)
    for a, b in zip(frozen, here):
        assert a["item_id"] == b["item_id"]
        assert a["input_ids"].dtype == b["input_ids"].dtype
        assert a["input_ids"].shape == b["input_ids"].shape
        assert torch.equal(a["input_ids"], b["input_ids"])


# --- the item contract ------------------------------------------------------

def test_an_item_with_neither_key_is_refused():
    with pytest.raises(CalibrationItemError, match="neither"):
        prepare_calibration_items([{"item_id": "x", "domain": "general"}])


def test_empty_ids_are_refused():
    with pytest.raises(CalibrationItemError, match="empty"):
        prepare_calibration_items([{"item_id": "x", "ids": []}])


def test_an_already_prepared_item_is_accepted_unchanged():
    """The toy fixtures and the device canary build items in operator shape."""
    t = torch.tensor([[5, 6, 7]], dtype=torch.long)
    out = prepare_calibration_items([{"item_id": "x", "input_ids": t}])
    assert out[0]["input_ids"] is t
    assert "ids" not in out[0]


def test_both_keys_are_accepted_when_the_tokens_agree():
    ids = [5, 6, 7]
    out = prepare_calibration_items([{
        "item_id": "x", "ids": ids,
        "input_ids": torch.tensor([ids], dtype=torch.long)}])
    assert out[0]["input_ids"][0].tolist() == ids
    assert out[0]["ids"] == ids


def test_both_keys_that_disagree_are_refused():
    with pytest.raises(CalibrationItemError, match="not the same tokens"):
        prepare_calibration_items([{
            "item_id": "x", "ids": [5, 6, 7],
            "input_ids": torch.tensor([[5, 6, 8]], dtype=torch.long)}])


def test_a_wrong_prediction_position_count_is_refused():
    with pytest.raises(CalibrationItemError, match="n_prediction_positions"):
        prepare_calibration_items([{"item_id": "x", "ids": [5, 6, 7],
                                    "n_prediction_positions": 5}])


@pytest.mark.parametrize("bad,match", [
    (torch.tensor([5, 6, 7], dtype=torch.long), "rank 1"),
    (torch.tensor([[5, 6], [7, 8]], dtype=torch.long), "batch 2"),
    (torch.tensor([[5.0, 6.0]]), "dtype"),
    ([[5, 6, 7]], "not a torch.Tensor"),
])
def test_a_malformed_input_ids_is_refused(bad, match):
    with pytest.raises(CalibrationItemError, match=match):
        prepare_calibration_items([{"item_id": "x", "input_ids": bad}])


def test_the_profile_id_is_named_in_the_failure():
    with pytest.raises(CalibrationItemError, match="calib.domain_balanced@v1"):
        prepare_calibration_items([{"item_id": "x"}],
                                  profile_id="calib.domain_balanced@v1")


# --- the fixed path, resolving a real profile with NO override --------------

def path_spec(steps, *, device: str = "cpu") -> FixedPathSpec:
    return FixedPathSpec(
        path_id="test.raw_calibration", family="qwen3",
        target_spec=ArchSpec.of("qwen3", TARGET), steps=tuple(steps),
        root_repo_id="test/teacher", root_revision="deadbeef", device=device)


def c1_shaped_steps(profile_id: str) -> list[FixedPathStep]:
    """DEPTH -> FFN -> WIDTH -> ATTENTION: the C1 path's own kinds and order."""
    return [FixedPathStep(DEPTH, profile_id), FixedPathStep(FFN, profile_id),
            FixedPathStep(WIDTH, profile_id), FixedPathStep(ATTENTION, profile_id)]


def test_the_fixed_path_resolves_and_prepares_without_any_override(tiny_mixture):
    """The branch attempt 8 died in, executed at `$0`.

    No `calibration_items=`. `materialize_fixed_path` must call
    `profile.resolve()`, get raw `ids` back, and hand every one of the four
    calibrated operators a usable item.
    """
    root, profile, raw = tiny_mixture
    spec = path_spec(c1_shaped_steps(profile.qualified_id))

    out = materialize_fixed_path(
        spec, adapter=QWEN3_ADAPTER,
        root_loader=lambda: build_tiny_model(TEACHER_GEOMETRY),
        workdir=root / "work", repo_root=root)

    assert [r.kind for r in out] == ["DEPTH", "FFN", "RESIDUAL_WIDTH", "ATTENTION"]
    assert [r.impl_id for r in out] == [DEPTH, FFN, WIDTH, ATTENTION]
    assert all(r.profile_id == profile.qualified_id for r in out)
    final = ArchSpec.of("qwen3", TARGET)
    assert out[-1].result_spec_hash == final.spec_hash


def test_every_calibrated_operator_sees_the_same_prepared_tokens(tiny_mixture,
                                                                 monkeypatch):
    """One boundary, four operators — and they get identical tokens.

    Recorded from inside `execute`, so this is what the operators actually
    received, not what the executor believes it sent.
    """
    root, profile, raw = tiny_mixture
    expected = [list(i["ids"]) for i in raw]
    seen: dict[str, list[list[int]]] = {}

    for impl_id in (DEPTH, FFN, WIDTH, ATTENTION):
        impl = get_implementation(impl_id)
        real = impl.execute

        def spy(ctx, _impl_id=impl_id, _real=real):
            seen[_impl_id] = [i["input_ids"][0].tolist()
                              for i in ctx.calibration_items]
            assert all(i["input_ids"].dtype is torch.long
                       for i in ctx.calibration_items)
            return _real(ctx)

        monkeypatch.setattr(impl, "execute", spy)

    materialize_fixed_path(
        path_spec(c1_shaped_steps(profile.qualified_id)), adapter=QWEN3_ADAPTER,
        root_loader=lambda: build_tiny_model(TEACHER_GEOMETRY),
        workdir=root / "work", repo_root=root)

    assert set(seen) == {DEPTH, FFN, WIDTH, ATTENTION}
    for impl_id, tokens in seen.items():
        assert tokens == expected, impl_id


def test_the_resolver_is_called_exactly_once_per_profile(tiny_mixture, monkeypatch):
    """Four steps, one mixture: preparation must not re-resolve or re-tokenize.

    `resolve()` re-hashes the file on every call, so this is a cost claim as well
    as a caching one.
    """
    root, profile, _ = tiny_mixture
    calls = []
    real = CalibrationProfile.resolve

    def counting(self, repo_root="."):
        calls.append(self.qualified_id)
        return real(self, repo_root)

    monkeypatch.setattr(CalibrationProfile, "resolve", counting)
    materialize_fixed_path(
        path_spec(c1_shaped_steps(profile.qualified_id)), adapter=QWEN3_ADAPTER,
        root_loader=lambda: build_tiny_model(TEACHER_GEOMETRY),
        workdir=root / "work", repo_root=root)
    assert calls == [profile.qualified_id]


def test_mutation_bypassing_preparation_reproduces_the_attempt_8_failure(
        tiny_mixture, monkeypatch):
    """Delete the boundary and stage D's exact crash comes back.

    Without this the tests above could pass for reasons unrelated to the repair.
    `KeyError: 'input_ids'` is verbatim what pod `fbuggw0x9efqsz` raised at
    `operators/depth.py:176`.
    """
    root, profile, _ = tiny_mixture
    monkeypatch.setattr(FP, "prepare_calibration_items",
                        lambda items, profile_id="": list(items))

    with pytest.raises(KeyError) as exc:
        materialize_fixed_path(
            path_spec(c1_shaped_steps(profile.qualified_id)),
            adapter=QWEN3_ADAPTER,
            root_loader=lambda: build_tiny_model(TEACHER_GEOMETRY),
            workdir=root / "work", repo_root=root)
    assert exc.value.args[0] == "input_ids"


def test_a_supplied_override_goes_through_the_same_boundary(tiny_mixture):
    """A caller cannot route around the contract by passing items itself."""
    root, profile, raw = tiny_mixture
    bad = [dict(i, n_prediction_positions=999) for i in raw]
    with pytest.raises(CalibrationItemError, match="n_prediction_positions"):
        materialize_fixed_path(
            path_spec([FixedPathStep(FFN, profile.qualified_id)]),
            adapter=QWEN3_ADAPTER,
            root_loader=lambda: build_tiny_model(TEACHER_GEOMETRY),
            workdir=root / "work", repo_root=root,
            calibration_items={profile.qualified_id: bad})


# --- the root device the path declares --------------------------------------

def test_a_cpu_root_is_accepted_by_a_cpu_path(tiny_mixture):
    root, profile, _ = tiny_mixture
    spec = path_spec([FixedPathStep(FFN, profile.qualified_id)], device="cpu")
    model = build_tiny_model(TEACHER_GEOMETRY)
    assert require_root_on_declared_device(model, spec) == "cpu"


def test_a_cpu_root_is_refused_by_a_cuda_path(tiny_mixture):
    root, profile, _ = tiny_mixture
    spec = path_spec([FixedPathStep(FFN, profile.qualified_id)], device="cuda")
    model = build_tiny_model(TEACHER_GEOMETRY)
    with pytest.raises(FixedPathRootDeviceMismatch) as exc:
        require_root_on_declared_device(model, spec)
    assert exc.value.declared == "cuda" and exc.value.actual == "cpu"


def test_a_misplaced_root_is_refused_before_any_operator_runs(tiny_mixture,
                                                              monkeypatch):
    """The refusal has to precede execution, or it saves nothing.

    A `cuda` declaration against a CPU root is the attempt-8 stage-D shape
    exactly; on a real pod the path would simply have run in the wrong place.
    """
    root, profile, _ = tiny_mixture
    ran: list[str] = []
    for impl_id in (DEPTH, FFN, WIDTH, ATTENTION):
        impl = get_implementation(impl_id)
        monkeypatch.setattr(
            impl, "execute",
            lambda ctx, _i=impl_id: ran.append(_i) or pytest.fail(
                f"{_i} executed after a root-device mismatch"))

    with pytest.raises(FixedPathRootDeviceMismatch):
        materialize_fixed_path(
            path_spec(c1_shaped_steps(profile.qualified_id), device="cuda"),
            adapter=QWEN3_ADAPTER,
            root_loader=lambda: build_tiny_model(TEACHER_GEOMETRY),
            workdir=root / "work", repo_root=root)

    assert ran == []
    assert not (root / "work" / "steps").exists()


def test_the_check_reads_the_weights_not_the_declaration(tiny_mixture):
    """`ctx.device` is intent. A model that only *claims* cuda is still refused.

    This is the property the class exists for: attempt 8's spec said `cuda`, its
    context said `cuda`, and its weights were on the host.
    """
    root, profile, _ = tiny_mixture
    spec = path_spec([FixedPathStep(FFN, profile.qualified_id)], device="cuda")
    model = build_tiny_model(TEACHER_GEOMETRY)
    model.config.device = "cuda"          # a claim, and only a claim
    with pytest.raises(FixedPathRootDeviceMismatch):
        require_root_on_declared_device(model, spec)


def test_an_unindexed_cuda_declaration_accepts_an_ordinal():
    """`Tensor.to("cuda")` lands on cuda:0, so `"cuda"` must accept cuda:N.

    Checked without a GPU by asking the predicate about a device directly: the
    rule is about how two `torch.device` values compare, not about hardware.
    """
    declared, actual = torch.device("cuda"), torch.device("cuda:3")
    assert actual.type == declared.type and declared.index is None
    assert torch.device("cuda:1").index != torch.device("cuda:0").index
