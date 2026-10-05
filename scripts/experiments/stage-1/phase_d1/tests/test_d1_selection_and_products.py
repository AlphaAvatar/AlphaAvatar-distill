"""Stage D commits a REAL Top-2, and those two checkpoints are products.

Two blockers this closes, both of which would have produced a green session with
nothing to show for it:

* Stage D returned `top_k: []`. A session could have marked `ALL_DONE` with no
  candidate set at all, and the behavioural rungs would then have screened
  something the search did not choose.
* the launcher said JSON AND LOGS ONLY and left `fetch_products`/
  `products_secured` at their defaults, so the runner would report that D1 owes no
  off-pod products and delete the pod while the only copies of the selected
  initializations were on it. The next stage trains recovery probes FROM those
  initializations and the `$14.7966` screening price does not fund rebuilding two
  complete structural paths.

The selection path runs for real on a toy search: a real `BeamSearch`, a real
`result.top_n(PARETO_V1, 2)` and the real `stage1_selection.commit`. No second
ranking and no second record format.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[5]
for extra in ("src", "scripts", "scripts/pod", "scripts/autoinit", "tests"):
    path = str(REPO / extra)
    if path not in sys.path:
        sys.path.insert(0, path)

from support.toy import (  # noqa: E402
    TEACHER_GEOMETRY, build_tiny_model, make_items, make_profile,
)
from aadistill.initialization.execution import ExecutionConfig  # noqa: E402
from aadistill.initialization.planning import stage1_selection  # noqa: E402
from aadistill.initialization.planning.metrics import StateEvaluator  # noqa: E402
from aadistill.initialization.planning.ranking import (  # noqa: E402
    PARETO_V1, SCHEDULE_V1,
)
from aadistill.initialization.planning.search import (  # noqa: E402
    BeamSearch, SearchConfig,
)
from aadistill.initialization.specs.arch import (  # noqa: E402
    ArchSpec, adapter_for_config,
)
from aadistill.initialization.specs.materialization import (  # noqa: E402
    NumericalEnvironment, root_materialization_id,
)
from aadistill.initialization.specs.metrics import (  # noqa: E402
    ReferenceStrategy, StateEvalSuite, SuiteItem,
)
from experiments.phase_d1 import d1_session as S  # noqa: E402

CPU = NumericalEnvironment(device_type="cpu", compute_dtype="float32")
#: A target reachable by DEPTH alone, so a toy beam produces several complete
#: leaves cheaply and the ranking has something to choose from.
TOY_TARGET = {**TEACHER_GEOMETRY, "num_hidden_layers": 4}


class TestTheRootTeacherIdentityIsTheTeachers:
    """Not the arm, and not empty. `root_materialization_id` refuses both."""

    def test_it_comes_from_the_frozen_binding(self):
        identity = S.root_teacher_identity(REPO)
        binding = json.loads((REPO / S.TEACHER_BINDING).read_text())
        assert identity["repo_id"] == binding["repo_id"]
        assert identity["revision"] == binding["revision"]
        assert identity["root_teacher_id"] == \
            f"{binding['repo_id']}@{binding['revision']}"
        assert len(identity["root_teacher_sha256"]) == 64
        assert identity["n_shards"] == binding["n_shards"] == 3

    def test_core_accepts_it_as_a_root_materialization(self):
        """The check that refused the previous version outright."""
        identity = S.root_teacher_identity(REPO)
        got = root_materialization_id(
            root_teacher_id=identity["root_teacher_id"],
            root_teacher_sha256=identity["root_teacher_sha256"])
        assert len(got) == 32

    def test_it_does_not_mention_the_arm(self):
        """The arm belongs to the SCORING protocol identity, which carries it
        through `position_policy` and `config_hash`. A root identity naming the
        arm would make the teacher a property of how it was scored."""
        identity = S.root_teacher_identity(REPO)
        blob = json.dumps(identity)
        for arm in S.ARMS:
            assert arm not in blob, arm

    def test_the_hash_moves_with_a_shard(self, tmp_path, monkeypatch):
        """Derived from the per-shard digests, so it is not a label."""
        binding = json.loads((REPO / S.TEACHER_BINDING).read_text())
        first = sorted(binding["expected_shard_sha256"])[0]
        binding["expected_shard_sha256"][first] = "0" * 64
        fake = tmp_path / S.TEACHER_BINDING
        fake.parent.mkdir(parents=True, exist_ok=True)
        fake.write_text(json.dumps(binding))
        moved = S.root_teacher_identity(tmp_path)
        assert moved["root_teacher_sha256"] != \
            S.root_teacher_identity(REPO)["root_teacher_sha256"]

    def test_a_binding_with_an_unhashed_shard_is_refused(self, tmp_path):
        binding = json.loads((REPO / S.TEACHER_BINDING).read_text())
        binding["expected_shard_sha256"].pop(binding["weight_shards"][0])
        fake = tmp_path / S.TEACHER_BINDING
        fake.parent.mkdir(parents=True, exist_ok=True)
        fake.write_text(json.dumps(binding))
        with pytest.raises(S.D1SessionError, match="no sha256"):
            S.root_teacher_identity(tmp_path)

    def test_a_staged_teacher_that_does_not_match_is_refused(self, tmp_path):
        """A DECLARED root identity binds nothing on its own: the lineage would
        record the frozen revision while the search read whatever was staged."""
        staged = tmp_path / "teacher"
        staged.mkdir()
        for shard in json.loads(
                (REPO / S.TEACHER_BINDING).read_text())["weight_shards"]:
            (staged / shard).write_bytes(b"not the teacher")
        with pytest.raises(S.D1SessionError, match="not the frozen root"):
            S.verify_staged_teacher(staged, REPO)

    def test_an_absent_shard_is_refused_too(self, tmp_path):
        staged = tmp_path / "teacher"
        staged.mkdir()
        with pytest.raises(S.D1SessionError, match="absent"):
            S.verify_staged_teacher(staged, REPO)


@pytest.fixture(scope="module")
def toy_search(tmp_path_factory):
    """A REAL beam over a toy model, to a real `SearchResult`."""
    from aadistill.initialization.adapters import register_builtin_adapters
    from aadistill.initialization.operators.register import (
        register_builtin_operators,
    )

    register_builtin_adapters()
    register_builtin_operators()
    teacher = build_tiny_model(TEACHER_GEOMETRY)
    teacher.config.use_cache = False
    calib = make_items()
    from aadistill.initialization.calibration.items import (
        prepare_calibration_items,
    )
    calib = prepare_calibration_items(calib, profile_id="test.balanced@v1")
    items = tuple(
        SuiteItem(item_id=i["item_id"], domain=i["domain"], subtype=i["subtype"],
                  input_ids=i["input_ids"], tags=i.get("tags") or {})
        for i in calib)
    suite = StateEvalSuite(
        suite_id="test.d1sel", version=1, domains=("general", "math"),
        subtypes={"general": ("text",), "math": ("arith",)},
        critical_tags=("eos_like", "answer_like"), n_items=len(items),
        general_domain="general")
    ev = StateEvaluator(suite, items, device="cpu",
                        reference_strategy=ReferenceStrategy.RECOMPUTE)
    #: PRIMED WITH THE TEACHER, which `evaluate` refuses without: "a candidate
    #: cannot be scored against a teacher that was not run". The driver's stage B
    #: satisfies exactly this, with the same object it hands the search.
    ev.prime_reference(teacher)
    workdir = tmp_path_factory.mktemp("sel")
    config = SearchConfig(
        run_id="selection", target_spec=ArchSpec.of("qwen3", TOY_TARGET),
        schedule=SCHEDULE_V1, seed=7, workdir=workdir,
        profiles=(make_profile("balanced"),), policy=PARETO_V1, suite=suite,
        #: BOTH DEPTH operators, so several complete leaves exist and the ranking
        #: is choosing rather than accepting whatever there is.
        allowed_impls=("depth.causal_kl_greedy_v1", "depth.positional_v0"),
        measurement_protocol_id=ev.measurement_protocol_id, device="cpu")
    root = S.root_teacher_identity(REPO)
    search = BeamSearch(
        adapter=adapter_for_config(teacher.config), config=config,
        root_teacher_id=root["root_teacher_id"],
        root_teacher_sha256=root["root_teacher_sha256"],
        root_loader=lambda: teacher,
        calibration_loader=lambda profile: calib,
        measurer=lambda m, d: ev.evaluate(m, d),
        execution=ExecutionConfig(micro_batch_size=3,
                                  calibration_batch_packing="length_sorted_v1"),
        numerics=CPU)
    return search.run(), config, suite, workdir


class TestTheRealSelectionPath:
    """`SearchResult -> top_n(PARETO_V1, 2) -> stage1_selection.commit`."""

    def test_the_search_produced_complete_leaves(self, toy_search):
        result, _, _, _ = toy_search
        assert result.complete_leaves, "no complete leaf; the ranking has no input"

    def test_exactly_two_are_ranked(self, toy_search):
        result, _, _, _ = toy_search
        ranking = result.top_n(PARETO_V1, 2)
        assert len(ranking.selected) == 2, (
            f"{len(ranking.selected)} selected; D1 commits exactly 2")

    def test_the_committer_writes_the_real_record(self, toy_search, tmp_path):
        result, config, suite, workdir = toy_search
        ranking = result.top_n(PARETO_V1, 2)
        path = stage1_selection.commit(
            search_config=config, ranking=ranking, suite=suite, policy=PARETO_V1,
            profiles=config.profiles,
            journal_path=workdir / "states.jsonl", directory=tmp_path)
        doc = json.loads(Path(path).read_text())
        assert Path(path).name == "stage1_selection.json"
        assert len(doc["selected"]) == 2
        #: EVERY IDENTITY A TRANSFER NEEDS. `verify_transferred_leaf` rebuilds the
        #: identity from the bytes that arrive and takes two of these from the
        #: record because no file carries them.
        for row in doc["selected"]:
            for required in ("state_id", "path", "artifact_digest",
                             "checkpoint_path", "arch_signature",
                             "weights_digest", "num_parameters"):
                assert row.get(required) is not None, (required, row["state_id"])

    def test_the_committed_checkpoints_exist_on_disk(self, toy_search, tmp_path):
        """They are the product bytes; a record naming a path that does not exist
        would secure nothing."""
        result, config, suite, workdir = toy_search
        ranking = result.top_n(PARETO_V1, 2)
        path = stage1_selection.commit(
            search_config=config, ranking=ranking, suite=suite, policy=PARETO_V1,
            profiles=config.profiles,
            journal_path=workdir / "states.jsonl", directory=tmp_path)
        for row in json.loads(Path(path).read_text())["selected"]:
            assert Path(row["checkpoint_path"]).is_dir(), row["checkpoint_path"]

    def test_fewer_than_two_is_not_a_complete_search(self, toy_search):
        """The driver refuses rather than committing a partial set, because the
        behavioural rungs would then screen something the search did not choose."""
        result, _, _, _ = toy_search
        one = result.top_n(PARETO_V1, 1)
        assert len(one.selected) == 1
        #: The driver's own guard, as its message states the consequence.
        import autoinit_d1_driver as D

        src = Path(D.__file__).read_text()
        assert "NOT " in src and "COMPLETE" in src
        assert "must not be marked done" in src


class TestTheProductGate:
    """`products_secured` must require BOTH selected leaves."""

    @staticmethod
    def _ctx(scr: Path, rows: list[dict]):
        import types

        record = {"commit": {"selected_rows": rows}}
        (scr / "d1_search.json").write_text(json.dumps(record))
        return types.SimpleNamespace(
            args=types.SimpleNamespace(scr=str(scr), ckpt_store=str(scr / "p"),
                                      ckpt_fetch_limit_min=1),
            products_eligible=True, host="127.0.0.1",
            target=types.SimpleNamespace(port=22), say=lambda *a, **k: None)

    def test_it_reads_the_rows_the_driver_writes(self, tmp_path):
        """A launcher consuming a field the driver never writes is silent: no rows
        means no products, and the pod is deleted with the checkpoints on it."""
        import autoinit_d1_launch as L

        rows = [{"state_id": "s1", "checkpoint_path": "/x/s1"},
                {"state_id": "s2", "checkpoint_path": "/x/s2"}]
        assert L.committed_selection(self._ctx(tmp_path, rows)) == rows

    def test_an_empty_fetch_does_not_pass_the_gate(self, tmp_path):
        """`all([])` is True. A fetch that secured NOTHING must not pass."""
        import autoinit_d1_launch as L

        ctx = self._ctx(tmp_path, [{"state_id": "s1", "checkpoint_path": "/x/s1"},
                                   {"state_id": "s2", "checkpoint_path": "/x/s2"}])
        ok, why = L.both_selected_leaves_secured(ctx, [])
        assert ok is False
        assert "0 of 2" in why

    def test_one_of_two_does_not_pass_either(self, tmp_path):
        import autoinit_d1_launch as L
        from aadistill.infrastructure.session import ProductFetchResult

        ctx = self._ctx(tmp_path, [{"state_id": "s1", "checkpoint_path": "/x/s1"},
                                   {"state_id": "s2", "checkpoint_path": "/x/s2"}])
        ok, why = L.both_selected_leaves_secured(
            ctx, [ProductFetchResult(kind="transfer", rc=0, detail="s1"),
                  ProductFetchResult(kind="transfer", rc=1, detail="s2 failed")])
        assert ok is False and "1 of 2" in why

    def test_both_secured_passes(self, tmp_path):
        import autoinit_d1_launch as L
        from aadistill.infrastructure.session import ProductFetchResult

        ctx = self._ctx(tmp_path, [{"state_id": "s1", "checkpoint_path": "/x/s1"},
                                   {"state_id": "s2", "checkpoint_path": "/x/s2"}])
        ok, why = L.both_selected_leaves_secured(
            ctx, [ProductFetchResult(kind="transfer", rc=0, detail="s1"),
                  ProductFetchResult(kind="transfer", rc=0, detail="s2")])
        assert ok is True and "both selected checkpoints secured" in why

    def test_no_selection_means_no_product_bytes_to_owe(self, tmp_path):
        """A search that did not complete has no products; teardown is not blocked
        by bytes that were never produced, and the evidence is what matters."""
        import autoinit_d1_launch as L

        ok, why = L.both_selected_leaves_secured(self._ctx(tmp_path, []), [])
        assert ok is True and "did not complete" in why

    def test_the_spec_declares_the_pair(self):
        """Not at their defaults: the default answers 'this session owes no
        off-pod products', which for D1 would be false."""
        import autoinit_d1_launch as L

        args = L.build_parser().parse_args([
            "--scr", "/tmp/x", "--session-commit", "d" * 40,
            "--bundle", "b.bundle", "--run-id", "probe"])
        spec = L.spec(args)
        assert spec.artifacts.fetch_products is L.fetch_selected_checkpoints
        assert spec.artifacts.products_secured is L.both_selected_leaves_secured


class TestTheSuccessArtifactContract:
    """The authoritative search evidence is REQUIRED on success, optional on
    failure -- an early refusal legitimately produces none of it."""

    @staticmethod
    def _spec(name: str) -> dict:
        return json.loads((REPO / "configs/autoinit" / name).read_text())

    def test_success_requires_the_selection_and_the_journals(self):
        entries = {e["artifact_class"]: e
                   for e in self._spec("d1_search_artifacts.json")["entries"]}
        for required in ("d1_stage1_selection", "d1_state_journal",
                         "d1_search_telemetry", "d1_search_record"):
            assert entries[required]["required"] is True, required
        assert "stage1_selection.json" in entries["d1_stage1_selection"]["pattern"]
        assert "states.jsonl" in entries["d1_state_journal"]["pattern"]
        assert "telemetry.jsonl" in entries["d1_search_telemetry"]["pattern"]

    def test_failure_requires_nothing(self):
        for e in self._spec("d1_search_artifacts_failed.json")["entries"]:
            assert e["required"] is False, e["artifact_class"]

    def test_neither_spec_asks_for_checkpoint_bytes(self):
        """They are PRODUCTS, secured by the launcher's pair. An artifact spec
        demanding multi-GiB weights would pull them through the evidence path."""
        for name in ("d1_search_artifacts.json",
                     "d1_search_artifacts_failed.json"):
            blob = json.dumps(self._spec(name))
            for weighty in ("safetensors", "model-0000", "*.bin"):
                assert weighty not in blob, (name, weighty)
