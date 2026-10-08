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


#: THE LAYOUT `SessionRunner` ACTUALLY CREATES. Every context below is built
#: through this helper, which writes the reports where the runner's
#: `collect_and_teardown` scp's them -- `<scr>/store/<report_name>` -- and
#: nowhere else.
#:
#: The previous fixture wrote `<scr>/d1_search.json`, a path the runner never
#: creates, so the whole class tested a layout that does not exist. It passed,
#: and on a real session `committed_selection` would have found nothing and
#: `products_secured` would have read that as "no products were owed".
def _store_ctx(scr: Path, *, selection: dict | None = None,
               evidence: dict | None = None):
    import types

    store = scr / "store"
    store.mkdir(parents=True, exist_ok=True)
    if selection is not None:
        (store / "stage1_selection.json").write_text(json.dumps(selection))
    if evidence is not None:
        (store / "d1_search.json").write_text(json.dumps(evidence))
    return types.SimpleNamespace(
        args=types.SimpleNamespace(scr=str(scr), ckpt_store=str(scr / "p"),
                                   ckpt_fetch_limit_min=1),
        products_eligible=True, host="127.0.0.1", evidence={},
        target=types.SimpleNamespace(port=22), say=lambda *a, **k: None)


def _committed(rows: list[dict]) -> dict:
    """A selection record that passes `stage1_selection.load`'s own hash check."""
    from aadistill.infrastructure.manifest import sha256_json

    body = {"schema": "aadistill.stage1_selection/v1", "selected": rows,
            "n_selected": len(rows), "decisions": []}
    body["selection_sha256"] = sha256_json(body)
    body["generated_utc"] = "2026-10-05T00:00:00Z"
    return body


def _ok_evidence(rows: list[dict]) -> dict:
    return {"status": "COMPLETE", "commit": {"selected_rows": rows},
            "stages": [{"stage": "D", "name": "commit_top_k", "status": "ok"}]}


def _top_k() -> int:
    """THE DESIGN'S width, read rather than typed.

    This was a two-element literal, and the Top-2 -> Top-4 widening of
    2026-10-07 turned four of these tests red for the right reason: the gate
    reads `design.behavioural_design.top_k` and the fixture had its own copy.
    One owner.
    """
    from experiments.phase_d1 import d1_session as D1S

    return int(D1S.design()["behavioural_design"]["top_k"])


K = _top_k()
ROWS = [{"state_id": f"s{i}", "checkpoint_path": f"/x/s{i}"}
        for i in range(1, K + 1)]


class TestTheProductGateOnTheRealStoreLayout:
    """`products_secured` must require EVERY selected leaf, and must never read
    a path miss as "nothing was owed".

    The count comes from the design, so this class follows a retention-width
    decision instead of pinning one.
    """

    def test_it_reads_the_authoritative_record_from_the_runners_store(self,
                                                                     tmp_path):
        """From `<scr>/store/stage1_selection.json`, which is what the committer
        writes and what the runner fetches -- not the driver's summary and not
        `<scr>/`."""
        import autoinit_d1_launch as L

        ctx = _store_ctx(tmp_path, selection=_committed(ROWS),
                         evidence=_ok_evidence(ROWS))
        assert L.committed_selection(ctx) == ROWS

    def test_a_selection_in_the_old_place_is_not_found(self, tmp_path):
        """The regression. A record at `<scr>/` is invisible to the real runner
        layout, and that must REFUSE rather than report nothing owed."""
        import autoinit_d1_launch as L

        (tmp_path / "d1_search.json").write_text(
            json.dumps(_ok_evidence(ROWS)))
        ctx = _store_ctx(tmp_path)
        with pytest.raises(L.SelectionUnreadable):
            L.committed_selection(ctx)

    def test_success_with_a_missing_selection_fails_the_gate(self, tmp_path):
        """`success + missing selection -> False`. The driver reached
        commit_top_k and the record did not come home: the checkpoints exist and
        this launcher cannot name them."""
        import autoinit_d1_launch as L

        ctx = _store_ctx(tmp_path, evidence=_ok_evidence(ROWS))
        ok, why = L.both_selected_leaves_secured(ctx, [])
        assert ok is False
        assert "PRODUCT GATE FAILURE" in why
        #: It must name the real condition -- the record was not fetched -- and
        #: not the "search did not complete" verdict that would allow teardown.
        #: Asserted on the OUTCOME rather than on the absence of a phrase: the
        #: message deliberately quotes that verdict in order to reject it.
        assert "reached commit_top_k" in why
        assert "stage1_selection.json" in why

    def test_an_unparseable_selection_fails_the_gate(self, tmp_path):
        import autoinit_d1_launch as L

        store = tmp_path / "store"
        store.mkdir()
        (store / "stage1_selection.json").write_text("{not json")
        (store / "d1_search.json").write_text(json.dumps(_ok_evidence(ROWS)))
        ok, why = L.both_selected_leaves_secured(_store_ctx(tmp_path), [])
        assert ok is False and "PRODUCT GATE FAILURE" in why

    def test_a_selection_failing_its_own_hash_fails_the_gate(self, tmp_path):
        """`stage1_selection.load` verifies `selection_sha256`. An edited record
        is not a selection."""
        import autoinit_d1_launch as L

        doc = _committed(ROWS)
        doc["selected"] = [{**ROWS[0], "state_id": "tampered"}, ROWS[1]]
        ctx = _store_ctx(tmp_path, selection=doc, evidence=_ok_evidence(ROWS))
        ok, why = L.both_selected_leaves_secured(ctx, [])
        assert ok is False and "PRODUCT GATE FAILURE" in why

    def test_the_two_records_must_agree_about_which_leaves(self, tmp_path):
        """Two records of one decision that name different leaves settle
        nothing, and neither may be used to decide which bytes are products."""
        import autoinit_d1_launch as L

        other = [{"state_id": "zz1", "checkpoint_path": "/x/zz1"},
                 {"state_id": "zz2", "checkpoint_path": "/x/zz2"}]
        ctx = _store_ctx(tmp_path, selection=_committed(ROWS),
                         evidence=_ok_evidence(other))
        with pytest.raises(L.SelectionUnreadable, match="disagree"):
            L.committed_selection(ctx)

    def test_an_empty_fetch_does_not_pass_the_gate(self, tmp_path):
        """`success + 0/K -> False`. `all([])` is True; a fetch that secured
        NOTHING must not pass."""
        import autoinit_d1_launch as L

        ctx = _store_ctx(tmp_path, selection=_committed(ROWS),
                         evidence=_ok_evidence(ROWS))
        ok, why = L.both_selected_leaves_secured(ctx, [])
        assert ok is False
        assert f"0 of {K}" in why

    def test_one_short_does_not_pass_either(self, tmp_path):
        """`success + (K-1)/K -> False`."""
        import autoinit_d1_launch as L
        from aadistill.infrastructure.session import ProductFetchResult

        ctx = _store_ctx(tmp_path, selection=_committed(ROWS),
                         evidence=_ok_evidence(ROWS))
        fetched = [ProductFetchResult(kind="transfer", rc=0, detail=r["state_id"])
                   for r in ROWS[:-1]]
        fetched.append(ProductFetchResult(kind="transfer", rc=1,
                                          detail=f"{ROWS[-1]['state_id']} failed"))
        ok, why = L.both_selected_leaves_secured(ctx, fetched)
        assert ok is False and f"{K - 1} of {K}" in why

    def test_every_selected_leaf_secured_passes(self, tmp_path):
        """`K rows exist and all verify -> PASS`."""
        import autoinit_d1_launch as L
        from aadistill.infrastructure.session import ProductFetchResult

        ctx = _store_ctx(tmp_path, selection=_committed(ROWS),
                         evidence=_ok_evidence(ROWS))
        ok, why = L.both_selected_leaves_secured(
            ctx, [ProductFetchResult(kind="transfer", rc=0,
                                     detail=r["state_id"]) for r in ROWS])
        assert ok is True

    def test_a_selection_of_the_wrong_size_fails_the_gate(self, tmp_path):
        """`commit_top_k exists with != K selected rows -> PRODUCT GATE
        FAILURE`. One leaf secured out of one would otherwise read as complete."""
        import autoinit_d1_launch as L
        from aadistill.infrastructure.session import ProductFetchResult

        one = [ROWS[0]]
        ctx = _store_ctx(tmp_path, selection=_committed(one),
                         evidence=_ok_evidence(one))
        ok, why = L.both_selected_leaves_secured(
            ctx, [ProductFetchResult(kind="transfer", rc=0, detail="s1")])
        assert ok is False
        assert "PRODUCT GATE FAILURE" in why
        assert f"design commits {K}" in why

    def test_failure_before_commit_owes_no_checkpoint_products(self, tmp_path):
        """`failure before commit -> no checkpoint product obligation`.

        Established from the driver's OWN evidence showing it never reached
        stage D -- not inferred from a missing file, which is the distinction
        this whole class turns on.
        """
        import autoinit_d1_launch as L

        ctx = _store_ctx(tmp_path, evidence={
            "status": "FAILED",
            "stages": [{"stage": "A", "name": "environment", "status": "failed"}]})
        ok, why = L.both_selected_leaves_secured(ctx, [])
        assert ok is True
        assert "did not reach commit_top_k" in why

    def test_no_evidence_at_all_is_an_unknown_not_a_pass(self, tmp_path):
        """Nothing came home, so whether products are owed is UNKNOWN. An
        unknown must not be reported as 'nothing was owed'."""
        import autoinit_d1_launch as L

        ok, why = L.both_selected_leaves_secured(_store_ctx(tmp_path), [])
        assert ok is False and "PRODUCT GATE FAILURE" in why

    def test_the_fetcher_reports_a_failure_when_it_cannot_read(self, tmp_path):
        """`fetch_products` must not swallow it: the runner's
        `checkpoint_hashes_matched` reads these entries."""
        import autoinit_d1_launch as L

        ctx = _store_ctx(tmp_path, evidence=_ok_evidence(ROWS))
        fetched = L.fetch_selected_checkpoints(ctx)
        assert len(fetched) == 1
        assert fetched[0].ok is False
        assert "d1_selection_unreadable" in ctx.evidence

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
