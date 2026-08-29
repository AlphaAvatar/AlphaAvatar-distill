"""The behavioural continuation produces no checkpoints, and must not claim to.

The continuation launcher was written by deriving from Phase B's, and it
inherited Phase B's **product contract** along with the parts it actually
needed:

    fetch_products   = fetch_selected_leaves
    products_secured = selected_leaves_secured
    precheck        += ckpt_store_capacity_gate

All three exist to rescue SEARCH OUTPUT. `fetch_selected_leaves` is the
attempt-11 fix: that session produced five measured stage-1 leaves and lost
every one, because the only product fetch returned early when stage 2 did not
pass. `ckpt_store_capacity_gate` refuses to launch unless the dev-box checkpoint
store can hold those five bf16 leaves — 5 x 1.110 GiB plus 6 GiB of working
room, **11.55 GiB**.

This session runs no search. Its three finalists are inputs, already durable on
this box and on the relay, and its only irreplaceable product is a ~12 MiB
artifact archive carrying the newly bought probe journal, the generations, the
per-sample rows, the pooled decision and the report.

**What each inherited piece actually did.** The capacity gate was a real,
blocking defect: it refused every launch, asking for a thousand times the
session's need on a filesystem the session does not write. The two callables
were **inert rather than fatal** — `selected_leaf_records` reads
`<scr>/store/selected_leaf_durability.json`, the continuation driver never
writes one, and `selected_leaf_durability.json` is not among the continuation's
fetched reports, so both returned empty and `selected_leaves_secured` passed
vacuously. `test_the_inherited_callables_were_inert_not_fatal` pins that, so the
record says what was true rather than the worst reading of it.

That distinction is why they still had to go. A declaration that is wrong but
currently harmless is the one that survives review, and it only stays harmless
for as long as nothing downstream starts writing a durability report.

These tests pin the repaired contract: no leaf fetch, no checkpoint owed at
teardown, a capacity gate on the filesystem the collector actually writes, an
archive that still carries the science, and the two invariants the repair must
not have disturbed.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/pod"))
sys.path.insert(0, str(REPO / "scripts/autoinit"))

CAPACITY_RECORD = REPO / "logs/autoinit_continuation_b_capacity.json"


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        name, REPO / "scripts/pod" / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cont():
    return load("continuation_b_launch_pc", "autoinit_continuation_b_launch.py")


@pytest.fixture(scope="module")
def phase_a():
    return load("phase_a_launch_pc", "autoinit_phase_a_launch.py")


@pytest.fixture(scope="module")
def args(cont):
    return cont.build_parser().parse_args(
        ["--session-commit", "0" * 40, "--bundle", "b.tar.gz"])


class Ctx:
    """The slice of `SessionContext` the product callables and gate touch."""

    def __init__(self, scr, *, ckpt_store=None, stage2_passed=True):
        self.scr = Path(scr)
        self.stage2_passed = stage2_passed
        self.host = "pod.invalid"
        self.scp = ("scp",)
        self.evidence: dict = {}
        self.say = lambda m: None
        self.target = None

        class A:
            pass
        self.args = A()
        self.args.scr = str(scr)
        self.args.ckpt_store = str(ckpt_store or (Path(scr) / "ckpt_store"))
        self.args.ckpt_fetch_limit_min = 1
        self.args.fetch_finalists = True


# --- the guard, and the mutation that must trip it --------------------------

def assert_no_leaf_products(policy, phase_a) -> None:
    """The one assertion both the real spec and the mutation are put through.

    Written once and called twice on purpose. A guard that only ever sees the
    passing case is not known to be able to fail — the hole
    `mutate-new-gates-before-trusting-them` describes — so
    `test_reconnecting_the_leaf_fetch_is_caught` feeds it the mutated policy and
    requires it to raise.
    """
    if policy.fetch_products is phase_a.fetch_selected_leaves:
        raise AssertionError(
            "the continuation's ArtifactPolicy fetches stage-1 selected leaves; "
            "this session runs no search and produces none")
    if policy.products_secured is phase_a.selected_leaves_secured:
        raise AssertionError(
            "the continuation's ArtifactPolicy judges teardown by whether five "
            "stage-1 selected leaves came home; it creates none")
    fetched = policy.fetch_products(None)
    if fetched:
        raise AssertionError(
            f"the continuation claims to fetch {len(fetched)} product(s) off the "
            "pod; its checkpoints are inputs, not outputs")


# 1. no selected-leaf fetch_products on the success policy.
def test_the_success_policy_has_no_selected_leaf_fetch(cont, phase_a, args):
    assert_no_leaf_products(cont.spec(args).artifacts, phase_a)


# 2. the session owes no checkpoint product at teardown.
def test_the_continuation_owes_no_checkpoint_product_at_teardown(
        cont, phase_a, args, tmp_path):
    """And says so EXPLICITLY. `all([])` is True, so a fetch that returned
    nothing would pass `checkpoint_hashes_matched` while securing nothing —
    which is why `products_secured` is asked separately and must answer."""
    from aadistill.infrastructure.artifact_gate import (
        GATE_ORDER, evaluate_teardown,
    )

    policy = cont.spec(args).artifacts
    ctx = Ctx(tmp_path)
    ok, why = policy.products_secured(ctx, policy.fetch_products(ctx))
    assert ok, why
    assert "owes no off-pod products" in why, why

    # The gate still ASKS, and still fails closed when nobody answers.
    assert "required_products_secured" in GATE_ORDER
    passing = {name: True for name in GATE_ORDER}
    assert evaluate_teardown(passing).allowed
    del passing["required_products_secured"]
    assert not evaluate_teardown(passing).allowed


# 3. reconnecting the leaf fetch fails.
def test_reconnecting_the_leaf_fetch_is_caught(cont, phase_a, args):
    from dataclasses import replace

    real = cont.spec(args).artifacts
    for mutation in (
            {"fetch_products": phase_a.fetch_selected_leaves},
            {"products_secured": phase_a.selected_leaves_secured},
            {"fetch_products": lambda ctx: [{"artifact": "stage1_selected_leaf"}]},
    ):
        with pytest.raises(AssertionError):
            assert_no_leaf_products(replace(real, **mutation), phase_a)


def test_the_inherited_callables_were_inert_not_fatal(phase_a, tmp_path):
    """The severity claim, checked rather than repeated.

    STATE.md read the inherited pair as "at teardown they would hunt the pod for
    five leaf directories that were never created there". Driven for real
    against a continuation-shaped scratch, they do not: `selected_leaf_records`
    reads `<scr>/store/selected_leaf_durability.json`, the continuation driver
    writes none and the continuation does not fetch one, so both return empty
    and the secured check passes vacuously.

    The capacity gate was the blocker. The callables were a wrong declaration
    that happened to be inert — which is exactly the kind that survives review,
    and stays inert only while nothing downstream starts writing that report.
    """
    (tmp_path / "store").mkdir()
    ctx = Ctx(tmp_path)
    assert phase_a.selected_leaf_records(ctx) == []
    assert phase_a.fetch_selected_leaves(ctx) == []
    ok, why = phase_a.selected_leaves_secured(ctx, [])
    assert ok and "did not stage any selected leaves" in why


# 4. the capacity gate checks the resolved `args.scr` filesystem.
def test_the_capacity_gate_is_wired_and_checks_the_scratch_filesystem(
        cont, phase_a, args, tmp_path):
    assert cont.continuation_artifact_capacity_gate in cont.spec(args).precheck
    assert phase_a.ckpt_store_capacity_gate not in cont.spec(args).precheck, (
        "the continuation still carries Phase B's five-leaf capacity gate")

    # A scratch and a checkpoint store on DIFFERENT paths, so "which one did it
    # measure" has an observable answer.
    scr, store = tmp_path / "scratch", tmp_path / "elsewhere"
    scr.mkdir()
    store.mkdir()
    ctx = Ctx(scr, ckpt_store=store)

    seen: list[Path] = []
    real_usage = cont.shutil.disk_usage

    def spy(path):
        seen.append(Path(path))
        return real_usage(path)

    ok, why = cont.continuation_artifact_capacity_gate(
        type("C", (), {"args": ctx.args, "evidence": ctx.evidence})())
    ev = ctx.evidence["precheck"]["continuation_artifact_store"]
    assert ev["destination"] == str(scr / "store"), ev
    assert not ev["destination"].startswith(str(store)), (
        "the gate measured the checkpoint store, not the scratch filesystem")
    assert ev["produces_checkpoint_products"] is False
    assert isinstance(ok, bool) and why


def test_the_capacity_gate_measures_the_directory_the_collector_writes(
        cont, tmp_path, monkeypatch):
    """`SessionRunner.collect_and_teardown` scp's the archive to `<scr>/store`
    and extracts a second full copy beside it. Both are on the scratch
    filesystem, which is why that is the one measured."""
    runner = (REPO / "src/aadistill/infrastructure/session_runner.py").read_text()
    assert 'store = self.scr / "store"' in runner, (
        "the collector no longer writes <scr>/store; the gate is measuring a "
        "filesystem nothing uses")
    assert 'extract = store / "extracted"' in runner

    scr = tmp_path / "scratch"
    scr.mkdir()
    ctx = type("C", (), {"args": type("A", (), {"scr": str(scr)})(),
                         "evidence": {}})()

    class Short:
        free = 4 << 20
    monkeypatch.setattr(cont.shutil, "disk_usage", lambda p: Short)
    ok, why = cont.continuation_artifact_capacity_gate(ctx)
    assert not ok
    assert "free" in why and "archive" in why


def test_the_capacity_requirement_is_measured_not_asserted(cont):
    """A written constant is a guess wearing a gate's clothing unless something
    says where the number came from. This one is `du -sb` over every retained
    session store on this box, restricted to the search-free ones."""
    assert CAPACITY_RECORD.is_file(), (
        f"{CAPACITY_RECORD.name} is missing; the frozen requirement has no "
        "provenance")
    doc = json.loads(CAPACITY_RECORD.read_text())
    req = doc["requirement"]

    assert req["measured_bytes"] == cont.CONTINUATION_STORE_MEASURED_BYTES
    assert req["safety_factor"] == cont.CONTINUATION_STORE_SAFETY_FACTOR
    assert req["working_margin_bytes"] == cont.CONTINUATION_SCRATCH_MARGIN_BYTES
    assert req["required_bytes"] == cont.continuation_artifact_bytes_required()

    # The bound IS the largest search-free store in the recorded table, and the
    # table is not empty of the class it claims to summarize.
    search_free = [r for r in doc["table"] if not r["ran_a_search"]]
    assert len(search_free) >= 5, search_free
    assert max(r["store_bytes"] for r in search_free) == req["measured_bytes"]

    # Every store bigger than the bound ran a search, and carries the journal
    # that made it bigger. That is what makes them incomparable, not an opinion.
    for row in doc["table"]:
        if row["store_bytes"] > req["measured_bytes"]:
            assert row["ran_a_search"] and row["search_journal_bytes"] > 0, row

    # Strictly smaller than what it replaces, and by three orders of magnitude.
    assert req["required_bytes"] < req["replaces"]["required_bytes"] / 10


# 5. the success archive carries the real continuation evidence.
def test_the_success_archive_carries_the_probes_and_the_final_selection(cont, args):
    from collect_artifacts import load_specs

    policy = cont.spec(args).artifacts
    specs = load_specs(str(REPO / policy.spec_success))
    by_class = {s.artifact_class: s for s in specs}

    # The irreplaceable science: a newly bought probe cannot be re-derived for
    # free, and the pooled decision and report are what the session is FOR.
    for name in ("probe_journal", "probe_config", "per_sample", "generations",
                 "rung2_selection", "phase_a_result", "phase_a_evidence",
                 "continuation_binding", "attested_protocol"):
        assert name in by_class, f"{name} is not collected on success"
        assert by_class[name].required, f"{name} is collected but not required"

    assert by_class["probe_journal"].min_matches >= 12, (
        "11 imported citations plus the one mandatory sb")

    # And no weights. Temporary probe training checkpoints are not promoted to
    # permanent products: nothing downstream reads them.
    assert not any("checkpoint" in s.pattern or s.pattern.endswith(".safetensors")
                   for s in specs), [s.pattern for s in specs]

    # No search artifact, on the success path either. This session cannot make
    # one, and requiring it would fail a SUCCESSFUL run at teardown.
    for s in specs:
        assert "states.jsonl" not in s.pattern
        assert "search_result" not in s.pattern
        assert "stage1_selection" not in s.pattern
        assert "leaf_retention" not in s.pattern


# 6. the failure archive still supports diagnosis.
def test_the_failure_archive_still_supports_diagnosis(cont, args):
    from collect_artifacts import load_specs

    policy = cont.spec(args).artifacts
    specs = load_specs(str(REPO / policy.spec_failed))
    by_class = {s.artifact_class: s for s in specs}

    # The evidence file is the one thing a stage-0 death still produces, and it
    # is the only thing required — a required-but-absent artifact would turn a
    # recoverable failure into a collection error.
    assert by_class["phase_a_evidence"].required
    assert sum(s.required for s in specs) == 1, [s.artifact_class for s in specs
                                                 if s.required]

    # But a probe bought before the failure must still be rescued: it is the
    # only irreplaceable thing this session can create, and losing it means the
    # next attempt pays for it again.
    for name in ("probe_journal", "probe_config", "per_sample", "generations",
                 "continuation_binding", "session_evidence", "gate_log"):
        assert name in by_class, f"{name} is unrecoverable after a failure"


# 7. the no-search guarantee survives the repair.
def test_the_no_search_guarantee_is_intact(cont, args):
    from aadistill.autoinit.phase_b_continuation import (
        CONTINUATION_OWN_PATH_FILES, KNOWN_NEUTRALIZED_SEARCH_CALL_SITES,
        search_call_site_owners,
    )

    assert search_call_site_owners(REPO, files=CONTINUATION_OWN_PATH_FILES) == ()
    owners = search_call_site_owners(REPO)
    assert set(owners) == set(KNOWN_NEUTRALIZED_SEARCH_CALL_SITES), sorted(owners)

    spec = cont.spec(args)
    assert cont.no_search_gate in spec.precheck
    assert spec.evidence_fields["runs_search"] is False
    assert spec.evidence_fields["stage1_imported_not_recomputed"] is True


# 8. six evidence candidates, three finalists.
def test_the_six_to_three_boundary_is_intact():
    """Six candidates carry the evidence; three of them may be probed.

    Rebuilt by the real `build_evidence_universe`, not read from the
    preregistration, so a repair that moved the boundary would fail here rather
    than agree with a document it also wrote.
    """
    import autoinit_continuation_b_driver as drv

    universe = drv.ContinuationDriver.build_evidence_universe(
        drv.ContinuationDriver)
    assert len(universe) == drv.EXPECTED_EVIDENCE_UNIVERSE == 6, universe
    assert drv.EXPECTED_ACTIVE_FINALISTS == 3
    assert len({c.state_id for c in universe}) == 6, universe

    # The frozen amendment identity, re-derived, still matches the grant's.
    observed = drv.ContinuationDriver.observed_evidence()
    prereg = json.loads(
        (REPO / "logs/autoinit_continuation_b_preregistration.json").read_text())
    assert (observed["collapsed_universe_identity"]
            == prereg["evidence_universe"]["universe_identity"])
    assert prereg["evidence_universe"]["distinct_candidates"] == 6
    assert prereg["active_finalists"]["count"] == 3
    assert len(prereg["active_finalists"]["state_ids"]) == 3
    assert len(prereg["active_finalists"]["searched_non_survivors"]["state_ids"]) == 3
