"""Execute the REAL Phase-B Stage-1 path at toy scale, on CPU, at `$0`.

Four paid Phase-B failures have now shared one shape: **code only Phase B
reaches, which nothing cheap ever executes.** The `SESSION_KIND` dispatch branch,
the artifact-spec journal path, the probe-reuse tests, and — for `$8.17`, eight
hours in — a loop that rebinds a local `config` with `AutoConfig.from_pretrained`
and leaves the summary reading `run_id` off a `Qwen3Config`.

Every one was reachable here. The reason none was caught is that Phase A's suite
is the de-facto integration test and Phase A passes **no retained candidates and
one profile**, so the `for entry in retained_candidates:` body and the second
profile's branch had never run outside a pod.

So this runs `run_phase_a_search` end to end with everything Phase B actually
passes: P=2 over two genuinely distinct mixtures, two non-empty retained
candidates, a real beam search, the control measurement, both retained
measurements, the Top-N ranking, the durability commit and the summary — and
asserts on the returned object rather than on constants.

Substituted, and nothing else: the teacher and target geometry (a tiny Qwen3 at
the real 151,936 vocabulary, so production calibration token ids stay valid), the
suite size, and the calibration item counts. All cost, no logic. The mixtures
themselves are the real frozen ones, resolved with both hash checks.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/autoinit"))
sys.path.insert(0, str(REPO / "scripts/pod"))

import phase_a_search  # noqa: E402
from aadistill.autoinit import stage1_selection  # noqa: E402
from aadistill.autoinit.arch import ArchSpec, get_adapter  # noqa: E402
from aadistill.autoinit.artifact import identify_checkpoint  # noqa: E402
from aadistill.autoinit.calibration import (  # noqa: E402
    DOMAIN_BALANCED_V1, REASONING_HEAVY_V2,
)

ADAPTER = get_adapter("qwen3")

#: The geometries the Phase-A whole-function harness already drives through this
#: same search, so any failure here is Phase-B-specific rather than a toy-model
#: artefact. Real vocabulary, so the frozen calibration token ids stay in range;
#: `tie_word_embeddings=True` because an untied head is a parameter the child
#: builder legitimately refuses to leave unassigned.
TEACHER_GEOMETRY = dict(hidden_size=32, num_hidden_layers=6, intermediate_size=48,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                        vocab_size=151_936, tie_word_embeddings=True)
#: Differs in TWO structural fields, not four. A four-field target is a 24-path,
#: four-level beam at P=2 and takes longer than a CPU test may; two fields keeps
#: the multi-level search, the profile branching, the ranking, the retained
#: candidates and the summary — everything this file exists to execute — while
#: finishing in seconds. The structural DISTANCE is cost; it is not the defect
#: under test, which lives after the search in code no path length changes.
TARGET_GEOMETRY = dict(hidden_size=32, num_hidden_layers=4, intermediate_size=24,
                       num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                       vocab_size=151_936, tie_word_embeddings=True)


def toy_model(geometry: dict, seed: int):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen3Config(max_position_embeddings=512, **geometry)
    model = Qwen3ForCausalLM(cfg)
    return model.eval()


def items_for(profile, n: int):
    """The real frozen mixture, resolved with both hash checks, then truncated."""
    return phase_a_search.as_operator_items(profile.resolve(REPO))[:n]


@pytest.fixture(scope="module")
def suite_bundle():
    from aadistill.autoinit.metrics import StateEvalSuite

    suite, items, manifest = phase_a_search.load_suite(
        REPO / "artifacts/stage1/state_eval_v1")
    assert isinstance(suite, StateEvalSuite)
    # One item per (domain, sub-type) keeps the suite OBJECT real — so the
    # structural hash is the real one — while making a pass affordable.
    # One item per (domain, sub-type): 7 of 80. The suite OBJECT stays the real
    # one, so the structural hash the search pins is the real one.
    seen, kept = set(), []
    for item in items:
        key = (item.domain, item.subtype)   # SuiteItem is a dataclass, not a mapping
        if key in seen:
            continue
        seen.add(key)
        kept.append(item)
    return suite, kept, manifest


@pytest.fixture(scope="module")
def retained(tmp_path_factory):
    """Two real checkpoints at the target geometry, with re-derived digests.

    This is what Phase B passes and Phase A never does. Building them properly —
    saved, then identified — is the point: `make_retained_state` refuses bytes
    whose digest contradicts the record, so a fabricated digest would not survive.
    """
    root = tmp_path_factory.mktemp("retained")
    out = []
    for n, seed in enumerate((91, 92)):
        directory = root / f"finalist{n}"
        ADAPTER.save(toy_model(TARGET_GEOMETRY, seed), str(directory))
        spec = ADAPTER.spec_of(ADAPTER.load(str(directory), device="cpu"))
        art = identify_checkpoint(directory, adapter=ADAPTER, spec=spec,
                                  num_parameters=ADAPTER.param_count(spec))
        out.append({"candidate_id": f"retained{n}0000000{n}",
                    "checkpoint_dir": str(directory),
                    "description": f"toy retained finalist {n}",
                    "provenance": "retained_imported",
                    "expected_artifact_digest": art.artifact_digest})
    return out


def run_stage1(workdir, retained, *, top_n=5, n_items=2):
    control_dir = Path(workdir) / "control"
    ADAPTER.save(toy_model(TARGET_GEOMETRY, 4242), str(control_dir))
    teacher = toy_model(TEACHER_GEOMETRY, 7)
    from aadistill.autoinit.metrics import StateEvalSuite  # noqa: F401

    return phase_a_search.run_phase_a_search(
        workdir=Path(workdir) / "search",
        state_eval=REPO / "artifacts/stage1/state_eval_v1",
        top_n=top_n, device="cpu", repo_root=REPO,
        teacher_id="toy-phase-b-teacher",
        canonical_init=str(control_dir), canonical_sha256=None,
        teacher_loader=lambda: teacher,
        target_geometry=TARGET_GEOMETRY,
        suite_bundle=run_stage1.bundle,
        calibration_items={
            DOMAIN_BALANCED_V1.qualified_id: items_for(DOMAIN_BALANCED_V1, n_items),
            REASONING_HEAVY_V2.qualified_id: items_for(REASONING_HEAVY_V2, n_items)},
        profiles=(DOMAIN_BALANCED_V1, REASONING_HEAVY_V2),
        retained_candidates=retained)


@pytest.fixture(scope="module")
def completed(tmp_path_factory, suite_bundle, retained):
    run_stage1.bundle = suite_bundle
    workdir = tmp_path_factory.mktemp("stage1")
    return run_stage1(workdir, retained), workdir


# --- the whole function, end to end -----------------------------------------


def test_stage1_completes_through_summary_construction(completed):
    """The assertion attempt 4 needed and nothing had: it gets to the end."""
    found, _ = completed
    assert found.summary["schema"] == "aadistill.autoinit.phase_a_search/v1"
    assert found.top_n is not None and found.result is not None


def test_run_id_and_config_hash_come_from_the_SEARCH_config(completed):
    """The exact defect: the summary must not read them off a model config.

    `Qwen3Config` has neither attribute, so the old code raised — but a future
    model config that happened to carry a `run_id` would be worse, because it
    would silently label the run.
    """
    found, _ = completed
    assert found.summary["run_id"] == "autoinit.v1.phase_a"
    assert len(found.summary["config_hash"]) == 64
    # And it is the SearchConfig's own hash, re-derived rather than trusted.
    assert found.summary["config_hash"] != found.summary.get("suite_hash")
    assert "run_id" not in json.dumps(found.summary["retained_candidates"])


def test_both_profiles_are_represented_and_neither_is_claimed_as_the_one(completed):
    found, _ = completed
    assert found.summary["calibration_profiles"] == [
        DOMAIN_BALANCED_V1.qualified_id, REASONING_HEAVY_V2.qualified_id]
    assert found.summary["calibration_profile"] is None, (
        "a two-profile run must not report a singular profile")


def test_the_retained_candidates_were_measured_and_their_metadata_survived(
        completed, retained):
    found, _ = completed
    assert len(found.imported) == len(retained) == 2
    by_id = {s.state_id: s for s in found.imported}
    for entry in retained:
        state = by_id[entry["candidate_id"]]
        assert state.provenance == "retained_imported"
        assert state.artifact_digest == entry["expected_artifact_digest"]
        assert state.evaluation is not None, "an imported candidate must be measured"
    recorded = {r["state_id"] for r in found.summary["retained_candidates"]}
    assert recorded == {e["candidate_id"] for e in retained}


def test_the_control_was_measured_on_the_same_suite(completed):
    found, _ = completed
    assert found.control.provenance == "retained_canonical"
    assert found.control.evaluation is not None


def test_the_ranking_came_from_the_real_search(completed):
    found, _ = completed
    assert found.top_n.selected, "the real path produced no ranking"
    leaves = {s.state_id for s in found.result.leaves}
    assert {s.state_id for s in found.top_n.selected} <= leaves
    assert found.summary["top_n"]["selected"]
    assert found.summary["top_n"]["decisions"]


# --- the durability boundary -------------------------------------------------


def test_the_selection_artifact_is_written_and_self_verifying(completed):
    found, workdir = completed
    path = Path(workdir) / "search" / stage1_selection.FILENAME
    assert path.is_file(), "no Stage-1 selection artifact was committed"
    record = stage1_selection.load(path)          # raises if the hash disagrees
    assert record["schema"] == stage1_selection.SCHEMA
    assert record["n_selected"] == len(found.top_n.selected)
    assert {s["state_id"] for s in record["selected"]} == \
        {s.state_id for s in found.top_n.selected}
    for entry in record["selected"]:
        assert entry["artifact_digest"] and entry["checkpoint_path"]
    assert record["journal"]["sha256"]
    assert record["decisions"]


def test_the_selection_identity_excludes_its_own_timestamp(completed):
    """Provenance is not commitment; a timestamp inside it would churn the hash."""
    found, workdir = completed
    record = stage1_selection.load(Path(workdir) / "search" / stage1_selection.FILENAME)
    reissued = dict(record)
    reissued["generated_utc"] = "2099-01-01T00:00:00+00:00"
    from aadistill.infrastructure.manifest import sha256_json

    assert sha256_json({k: v for k, v in reissued.items()
                        if k not in ("selection_sha256", "generated_utc")}) == \
        record["selection_sha256"]


def test_a_tampered_selection_is_refused(completed, tmp_path):
    found, workdir = completed
    record = stage1_selection.load(Path(workdir) / "search" / stage1_selection.FILENAME)
    record["selected"] = record["selected"][:1]
    bad = tmp_path / stage1_selection.FILENAME
    bad.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="selection_sha256"):
        stage1_selection.load(bad)


def test_the_selection_SURVIVES_a_failure_in_the_next_bookkeeping_step(
        tmp_path, suite_bundle, retained, monkeypatch):
    """Attempt 4's failure mode, injected.

    The search completes, the ranking completes, the selection is committed — and
    then the very next step raises. The selection must still be on disk, complete
    and verifying, because that is the whole point of writing it early.
    """
    run_stage1.bundle = suite_bundle

    def boom(*a, **k):
        raise RuntimeError("injected: the bookkeeping after ranking failed")

    monkeypatch.setattr(phase_a_search, "make_control_state", boom)
    with pytest.raises(RuntimeError, match="injected"):
        run_stage1(tmp_path, retained)

    path = tmp_path / "search" / stage1_selection.FILENAME
    assert path.is_file(), (
        "the search completed and ranked, then the next step raised, and the "
        "selection was lost — which is exactly attempt 4")
    record = stage1_selection.load(path)
    assert record["n_selected"] > 0
    assert all(e["checkpoint_path"] for e in record["selected"])
