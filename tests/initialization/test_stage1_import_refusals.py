"""The Stage-1 importer's fail-closed behaviour, on a search built here.

`import_stage1_result` is reusable core: it rebuilds selected leaves and a
control from committed evidence, from BYTES, and refuses everything else. Those
refusals are what make a continuation safe to start, so they belong in the core
suite — and they must not need a historical experiment's evidence to prove.

They used to. The only coverage was
`tests/initialization/test_stage1_import.py`, which read attempt 12's real
records and its five preserved 1.11-GiB checkpoints off a host-local store under
a `skipif`. That made the CORE suite's result depend on whether an old
experiment's bytes were still on this machine; archiving them turned core green
into core-green-with-skips, which is not the same statement. The attempt-12
integration now lives in
`scripts/experiments/stage-1/phase_a/tests/test_phase_a_attempt12_import.py`,
where the historical evidence IS the subject.

What is here instead is the smallest search that exercises each refusal: two toy
leaves and a toy control, 32-wide and 2-layer, saved and re-read through the real
`save`/`identify_checkpoint` path so the digests are genuinely recomputed from
files. Two leaves rather than five, because the ordering check needs exactly two
to swap; tiny rather than real, because nothing about a refusal depends on a
checkpoint being 1.11 GiB.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.initialization.planning.stage1_import import (  # noqa: E402
    Stage1ImportError,
    import_stage1_result,
)
from aadistill.initialization.specs.arch import ArchSpec, get_adapter  # noqa: E402
from aadistill.initialization.specs.artifact import identify_checkpoint  # noqa: E402

from support.toy import build_tiny_model  # noqa: E402

#: The "target" every leaf must BE. Deliberately not the teacher geometry: the
#: wrong-geometry refusal has to have something to be wrong about.
TARGET = dict(hidden_size=32, num_hidden_layers=2, intermediate_size=48,
              num_attention_heads=4, num_key_value_heads=2, head_dim=8,
              vocab_size=128, tie_word_embeddings=True)

CONFIG_HASH = "c" * 64


def _write_leaf(root: Path, state_id: str, *, seed: int, adapter, geometry=None):
    """A real checkpoint on disk, plus the journal row and durability record.

    Saved through the adapter and identified through `identify_checkpoint`, so
    every digest in the records below was recomputed from the bytes just written
    — which is the property the importer relies on.
    """
    geometry = dict(geometry or TARGET)
    model = build_tiny_model(geometry, seed=seed)
    out = root / state_id
    adapter.save(model, str(out))
    spec = ArchSpec.of("qwen3", geometry)
    art = identify_checkpoint(out, adapter=adapter, spec=spec,
                             num_parameters=adapter.param_count(spec))
    row = {
        "state_id": state_id,
        "parent_id": None,
        "root_teacher_id": "toy/teacher",
        "root_teacher_sha256": "de" * 32,
        "arch_spec": geometry,
        "arch_spec_hash": spec.spec_hash,
        "num_parameters": adapter.param_count(spec),
        "seed": 7,
        "path_label": state_id,
        "applied_kinds": ["DEPTH"],
        "validity": "measured",
        "checkpoint_path": str(out),
        "artifact": {k: v for k, v in art.as_dict().items() if k != "path"},
        "evaluation": {
            "artifact_digest": art.artifact_digest,
            "suite_id": "toy.suite", "suite_hash": "s" * 64,
            "reference": "teacher", "values": {"state.teacher_kl": 0.5},
            "positions": 10,
        },
    }
    record = {
        "state_id": state_id,
        "artifact_digest": art.artifact_digest,
        "single_shard_sha256": art.single_shard_sha256,
        "arch_signature": art.arch_signature,
        "num_parameters": art.num_parameters,
    }
    return row, record


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    """A two-leaf Stage-1 result, a control, and the files the importer reads."""
    root = tmp_path_factory.mktemp("stage1")
    adapter = get_adapter("qwen3")
    store = root / "store"
    store.mkdir()

    rows, records = [], []
    for index, sid in enumerate(("leaf-a", "leaf-b")):
        row, rec = _write_leaf(store, sid, seed=11 + index, adapter=adapter)
        rows.append(row)
        records.append(rec)

    control_dir = root / "control"
    control = build_tiny_model(TARGET, seed=99)
    adapter.save(control, str(control_dir))
    control_spec = ArchSpec.of("qwen3", TARGET)
    control_art = identify_checkpoint(
        control_dir, adapter=adapter, spec=control_spec,
        num_parameters=adapter.param_count(control_spec))

    states = root / "states.jsonl"
    states.write_text("".join(json.dumps(r) + "\n" for r in rows))

    search_result = {
        "config_hash": CONFIG_HASH,
        "top_n": {"selected": [{"state_id": r["state_id"]} for r in rows]},
        "control": {
            "state_id": "control",
            "provenance": "retained_canonical",
            "artifact_digest": control_art.artifact_digest,
            "single_shard_sha256": control_art.single_shard_sha256,
            "frozen_sha256_verified": True,
        },
    }
    durability = {"leaves": records}
    return {
        "root": root, "store": store, "states": states, "adapter": adapter,
        "search_result": search_result, "durability": durability,
        "control_dir": control_dir,
        "control_sha256": control_art.single_shard_sha256,
    }


def run_import(s, **over):
    kw = dict(
        search_result=s["search_result"], states_path=s["states"],
        checkpoint_store=s["store"], durability=s["durability"],
        adapter=s["adapter"], expected_config_hash=CONFIG_HASH,
        target_geometry=TARGET, control_dir=s["control_dir"],
        control_sha256=s["control_sha256"], control_id="toy_control_v0")
    kw.update(over)
    return import_stage1_result(**kw)


# --- it works on a well-formed search --------------------------------------

def test_a_well_formed_result_imports(synthetic):
    """The premise. A refusal test proves nothing if nothing ever passes."""
    out = run_import(synthetic)
    assert [s.state_id for s in out.leaves] == ["leaf-a", "leaf-b"], (
        "order is the ranking")
    for state in out.leaves:
        assert state.validity.value == "measured"
        state.require_recovery_admissible()


def test_the_control_is_rebuilt_from_its_own_checkpoint(synthetic):
    """Never from the journal: a re-executed composite is not the incumbent."""
    out = run_import(synthetic)
    assert out.control.provenance == "retained_canonical"
    assert out.control.artifact.single_shard_sha256 == synthetic["control_sha256"]


def test_the_control_arrives_unmeasured_and_the_gate_says_so(synthetic):
    """Its Stage-1 evaluation is not in the evidence, so a continuation must
    measure it rather than invent one. The gate refusing is the guarantee."""
    from aadistill.initialization.specs.state import StateError

    out = run_import(synthetic)
    assert out.verification["control_is_unmeasured"] is True
    with pytest.raises(StateError, match="hash-bound measurements"):
        out.control.require_recovery_admissible()


# --- and refuses everything else -------------------------------------------

def test_a_different_search_is_refused(synthetic):
    with pytest.raises(Stage1ImportError, match="config_hash"):
        run_import(synthetic, expected_config_hash="0" * 64)


def test_a_reordered_selection_is_refused(synthetic):
    """The order IS the ranking, so swapping two leaves is a different result."""
    dur = synthetic["durability"]
    swapped = {**dur, "leaves": [dur["leaves"][1], dur["leaves"][0]]}
    with pytest.raises(Stage1ImportError, match="order matters"):
        run_import(synthetic, durability=swapped)


def test_a_missing_leaf_is_refused(synthetic, tmp_path):
    empty = tmp_path / "store"
    (empty / "leaf-a").mkdir(parents=True)
    with pytest.raises(Stage1ImportError):
        run_import(synthetic, checkpoint_store=empty)


def test_a_substituted_leaf_is_refused(synthetic, tmp_path):
    """Right id, wrong bytes: the digest comes from the file, not the name.

    By SYMLINK, as the historical version had to be — copying real leaves filled
    the disk there. Here it is cheap either way, and the symlink keeps the two
    tests the same shape.
    """
    fake = tmp_path / "swapped"
    fake.mkdir()
    (fake / "leaf-a").symlink_to(synthetic["store"] / "leaf-b")
    (fake / "leaf-b").symlink_to(synthetic["store"] / "leaf-b")
    with pytest.raises(Stage1ImportError, match="identifies as|shard"):
        run_import(synthetic, checkpoint_store=fake)


def test_a_digest_claim_that_the_bytes_contradict_is_refused(synthetic):
    dur = synthetic["durability"]
    lying = {**dur, "leaves": [{**dur["leaves"][0], "artifact_digest": "0" * 64},
                               dur["leaves"][1]]}
    with pytest.raises(Stage1ImportError, match="identifies as"):
        run_import(synthetic, durability=lying)


def test_a_wrong_target_geometry_is_refused(synthetic):
    """A leaf that is not the target size would rank well and deploy never."""
    assert TARGET["hidden_size"] == 32, "the mutation below must differ"
    other = {**TARGET, "hidden_size": 64}
    with pytest.raises(Stage1ImportError, match="is not the target"):
        run_import(synthetic, target_geometry=other)


def test_a_wrong_control_hash_is_refused(synthetic):
    with pytest.raises(Exception):
        run_import(synthetic, control_sha256="0" * 64)


def test_a_journal_without_the_states_is_refused(synthetic, tmp_path):
    empty = tmp_path / "states.jsonl"
    empty.write_text("")
    with pytest.raises(Stage1ImportError, match="no record for"):
        run_import(synthetic, states_path=empty)


def test_an_evaluation_measured_on_another_artifact_is_refused(synthetic,
                                                              tmp_path):
    """Metrics bind to artifacts, and importing does not relax that.

    The import checks this AND `attach_evaluation` checks it again — deliberate
    redundancy, which is why removing either alone changes no behaviour. This
    asserts the property rather than one of its two guards.
    """
    rows = [json.loads(line) for line
            in synthetic["states"].read_text().splitlines() if line.strip()]
    #: leaf-a's row claiming leaf-b's evaluation.
    rows[0]["evaluation"] = dict(rows[1]["evaluation"])
    swapped = tmp_path / "states.jsonl"
    swapped.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(Stage1ImportError, match="measured on"):
        run_import(synthetic, states_path=swapped)
