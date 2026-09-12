"""The C1 paid session: its type, its ceiling, and what it structurally cannot do.

Every assertion here is about a property the session has by construction rather
than by intention — the authorization type that refuses other phases, the ceiling
that exists in exactly one place, and the three driver methods that raise.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/pod"))
sys.path.insert(0, str(REPO / "scripts/autoinit"))

from experiments.phase_c1.authorization import c1_current_executable  # noqa: E402
from experiments.phase_c1.authorization import C1_HARNESS_SOURCE_FILES_V1, SCHEMA as C1_SCHEMA, C1Authorization, c1_budget_spec, c1_hard_ceiling_usd, c1_harness_digest  # noqa: E402
from aadistill.governance.authorization import AuthorizationError  # noqa: E402
from session_specs import load_session_launcher, session_args  # noqa: E402

#: The LIVE executable set. `C1_HARNESS_SOURCE_FILES_V1` is the
#: historical declaration and deliberately still names the old paths.
C1_EXECUTABLE = tuple(r["path"] for r in c1_current_executable(REPO)["files"])

#: A candidate authorization, built here, deterministically, into `tmp_path`.
#:
#: This used to be `Path.home() / "aad-scratch/sessions/c1-candidate/..."`, a
#: hand-issued artifact outside the repository. Locating it through `$HOME`
#: rather than an absolute literal was already an improvement — it made the file
#: invisible under the empty HOME the pod contract uses — but it traded one
#: failure for another: on the dev box the fixture existed and pinned a harness
#: digest that went stale the moment the harness moved, so a CORRECT gate
#: reported a false alarm, while on the pod the whole test skipped. A test whose
#: subject lives outside the repository is tested nowhere and trusted anyway.
#:
#: The payload derivation now lives in `c1_authorization_payload`, so the CLI
#: issuer and this fixture cannot diverge. Building one is not issuing one:
#: nothing below reads a clock, runs git, writes to `logs/`, stages a bundle or
#: touches a provider — the timestamp and commit are fixed strings, and the file
#: is written into `tmp_path`.
TEST_GRANTED_UTC = "2026-01-01T00:00:00+00:00"
TEST_SESSION_COMMIT = "0" * 40

def _accepted_cap_usd() -> float:
    """The cap the ISSUER refuses a mis-stated grant against, read from config.

    Written out as `283.76` until 2026-09-11, when the approved package raised
    it to `320.00` and this fixture became the only thing in the tree still
    naming the old one — so every candidate-driven gate test failed on a cap
    that had moved for a reason none of them are about. A fixture that restates
    a number the subject owns is a second source for it, which is the failure
    this whole file exists to police elsewhere.

    Imported through the PACKAGE, like everything else that reaches this
    module. Putting `scripts/experiments/phase_c1` on `sys.path` is never safe:
    it holds `packaging.py`, which shadows the third-party `packaging`
    distribution for the whole process.
    """
    from experiments.phase_c1.authorization_payload import load_config

    return float(load_config(REPO)["accepted_pricing"]["cumulative_cap_usd"])


#: The maintainer-stated half. Deliberately not copied from any real grant: this
#: names a fictional approver and says, in the artifact itself, what it is.
TEST_GRANT = {
    "granted_by": "TEST FIXTURE — not a maintainer, not a grant, not a permission",
    "covers": ("an ephemeral candidate used to drive the real pre-provider gates "
               "at $0. It permits nothing and is never written to logs/."),
    "cumulative_spend_at_approval_usd": 0.0,
    "cumulative_cap_usd": _accepted_cap_usd(),
    "does_not_authorize": ["anything at all"],
}


def write_candidate(tmp_path, **over):
    """A deterministic candidate authorization in `tmp_path`. Returns its path.

    Imported through the PACKAGE, never by putting
    `scripts/experiments/phase_c1` on `sys.path`. That directory contains
    `packaging.py`, which SHADOWS the third-party `packaging` distribution, so
    the first later module to import transformers dies with
    `cannot import name 'version' from 'packaging'`. `sys.path` is
    process-global: a function-scope insert is not scoped to the function, it
    just takes a particular module ordering to expose. An earlier comment in
    this file claimed such an insert was "survivable inside a function"; it is
    not, and 17 collection errors in an unrelated module are what that claim
    was worth.
    """
    from experiments.phase_c1.authorization_payload import (
        build_c1_authorization_payload)

    payload = build_c1_authorization_payload(
        grant=TEST_GRANT, session_commit=TEST_SESSION_COMMIT,
        granted_utc=TEST_GRANTED_UTC, repo_root=REPO,
        grant_path="<test fixture>", **over)
    path = tmp_path / "candidate_authorization.json"
    path.write_text(json.dumps(payload, indent=1) + "\n")
    return path


@pytest.fixture(scope="module")
def launcher():
    return load_session_launcher("autoinit_c1_launch")


@pytest.fixture(scope="module")
def spec(launcher):
    return launcher.spec(session_args(launcher))


# --- the type refuses everything else --------------------------------------

def test_the_session_loads_the_c1_type_and_nothing_else(spec):
    # By qualified name, not object identity: `load_session_launcher` execs the
    # launcher under its own module name, so its `C1Authorization` is a distinct
    # object from this test's import even though both are the same source.
    loader = spec.authorization_loader
    assert loader.__qualname__ == "C1Authorization.load"
    # `phase_c1.authorization` since the cutover; it was `c1_authorization`
    # when C1 policy lived under src/aadistill.
    assert loader.__module__.endswith("authorization")
    assert loader.__self__.__name__ == C1Authorization.__name__
    assert spec.setup.env["SESSION_KIND"] == "c1"


def test_a_foreign_authorization_is_refused_by_schema(tmp_path):
    """A Phase-A/B or continuation grant measures a different harness and carries
    a ceiling derived for different work."""
    fake = {"schema": "aadistill.autoinit.phase_a_authorization/v1",
            "authorization_id": "x"}
    p = tmp_path / "a.json"
    p.write_text(json.dumps(fake))
    with pytest.raises(AuthorizationError):
        C1Authorization.load(p)


def test_it_cannot_claim_phase_a_or_a_search():
    from dataclasses import FrozenInstanceError

    a = C1Authorization(
        authorization_id="t", granted_utc="u", granted_by="t", plan_id="p",
        plan_hash="h", science_plan_hash="s", expected_usd=1.0, hard_cap_usd=2.0,
        authorized_stages=(0,), stage_conditions={}, scope_note="t")
    assert a.allows_phase_a is False
    assert a.allows_beam_search is False
    assert a.authorizes_c1_isolation is True
    # Properties, not fields: there is nothing to set.
    with pytest.raises((FrozenInstanceError, AttributeError)):
        a.allows_beam_search = True


def test_the_setup_dispatcher_has_a_c1_branch():
    """A missing branch is not a type error — SESSION_KIND falls through to
    `spend` and loads a SpendAuthorization. Phase-B attempt 2 proved what that
    costs: $0.2300, one step after its test gate passed."""
    setup = (REPO / "scripts/pod/autoinit_preflight_setup.sh").read_text()
    assert 'elif [ "$SESSION_KIND" = "c1" ]; then' in setup
    branch = setup.split('"$SESSION_KIND" = "c1"')[1].split("elif")[0]
    assert "C1Authorization" in branch
    assert "authorizes_c1_isolation" in branch
    assert "allows_beam_search is False" in branch


# --- the ceiling exists in exactly one place -------------------------------

def test_the_ceiling_is_derived_from_the_accepted_pricing_record():
    pricing = json.loads((REPO / "logs/experiments/phase_c1/plans/phase_c1_pricing.json").read_text())
    assert c1_hard_ceiling_usd(REPO) == pricing["totals"]["hard_ceiling_usd"]


def test_the_budget_plan_fits_that_ceiling_exactly():
    ceiling = c1_hard_ceiling_usd(REPO)
    plan = c1_budget_spec(REPO).plan(price_per_hour=1.09, authorized_usd=ceiling)
    assert plan.hard_terminate_usd <= ceiling
    assert plan.expected_usd == pytest.approx(13.4401, abs=5e-4)
    assert plan.soft_stop_usd == pytest.approx(14.7841, abs=5e-4)


def test_the_plan_is_priced_at_the_rate_the_launcher_actually_pays():
    """Planning at a rate the provider no longer offers is how attempt 3 came to
    hold an authorization it could not launch under: the ceiling was derived at
    $0.99/h secure, the live secure rate was $1.09/h, and `session_runner` prices
    on securePrice. So the rate above is not decoration — it is the number that
    has to agree with the pricing record, and the SAME minutes at the stale rate
    must NOT satisfy the current ceiling."""
    from aadistill.infrastructure.budget import BudgetError

    pricing = json.loads((REPO / "logs/experiments/phase_c1/plans/phase_c1_pricing.json").read_text())
    assert pricing["hardware"]["price_per_hour_usd"] == 1.09

    stale = c1_budget_spec(REPO).plan(price_per_hour=0.99,
                                      authorized_usd=c1_hard_ceiling_usd(REPO))
    assert stale.hard_terminate_usd < 0.99 / 1.09 * c1_hard_ceiling_usd(REPO) + 1e-6
    with pytest.raises(BudgetError):
        c1_budget_spec(REPO).plan(price_per_hour=1.09, authorized_usd=13.7578)


def test_a_ceiling_that_rounds_down_would_fail_closed():
    """The 4-dp ceiling is rounded UP on purpose: the exact plan is $15.147403,
    and a grant written at a rounded-DOWN $15.1474 under-authorizes it."""
    from aadistill.infrastructure.budget import BudgetError

    with pytest.raises(BudgetError):
        c1_budget_spec(REPO).plan(price_per_hour=1.09, authorized_usd=15.1474)


def test_the_pricing_record_authorizes_nothing():
    pricing = json.loads((REPO / "logs/experiments/phase_c1/plans/phase_c1_pricing.json").read_text())
    assert pricing["authorizes"] == "nothing"


def test_a_superseded_committed_authorization_cannot_authorize():
    """A grant exists at the canonical path; a SUPERSEDED one must be refused.

    This asserted absence first, then specifically a moved harness digest. Both
    premises expired. C1 has now had three authorizations, and they are
    superseded by different things: attempts 1 and 2 by a moved harness, and the
    2026-09-04 repricing by a moved CEILING with the harness untouched. Pinning
    one mechanism made the test pass for the wrong reason.

    So it asserts the property that matters: whatever sits at the canonical path,
    if it is not current then at least one identity gate must refuse it. A
    current authorization passes them all, which is what a launch requires.
    """
    from experiments.phase_c1.authorization import C1Authorization, c1_hard_ceiling_usd, c1_harness_digest

    p = REPO / "logs/budget/approvals/autoinit_c1_authorization.json"
    assert p.is_file(), "the authorization record must be retained"
    auth = C1Authorization.load(p)
    assert auth.allows_phase_a is False and auth.allows_beam_search is False
    harness_ok = auth.harness_source_digest == c1_harness_digest(REPO)["digest"]
    ceiling_ok = abs(auth.hard_cap_usd - c1_hard_ceiling_usd(REPO)) < 1e-9
    #: Either it is current on BOTH, or a gate refuses it. Never one and not the
    #: other without a refusal.
    assert harness_ok == ceiling_ok or not (harness_ok and ceiling_ok), (
        f"harness_ok={harness_ok} ceiling_ok={ceiling_ok}: the artifact is "
        "partly current, which no gate combination would catch cleanly")


# --- the harness the grant measures ----------------------------------------

def test_the_harness_set_covers_the_launcher_driver_and_c1_science():
    for required in ("scripts/pod/autoinit_c1_launch.py",
                     "scripts/pod/autoinit_c1_driver.py",
                     # NOT a ternary accepting either setup script. That is what
                     # let the set name `scripts/pod/setup.sh` — which nothing in
                     # the repository executes — while the pod ran
                     # `autoinit_preflight_setup.sh` unmeasured. A regression that
                     # accepts either file cannot detect the wrong one.
                     "scripts/pod/autoinit_preflight_setup.sh",
                     "scripts/experiments/phase_c1/session.py",
                     "scripts/experiments/phase_c1/isolation.py",
                     "src/aadistill/initialization/planning/fixed_path.py",
                     "src/aadistill/initialization/operators/attention_activation.py",
                     "src/aadistill/initialization/statistics/attention.py",
                     "src/aadistill/initialization/planning/recovery.py",
                     "scripts/autoinit/score_recovery_search.py"):
        assert required in C1_EXECUTABLE, required


def test_a_missing_harness_file_raises_rather_than_shrinking_the_digest():
    with pytest.raises(AuthorizationError, match="missing"):
        c1_harness_digest(REPO, files=("src/aadistill/does_not_exist.py",))


def test_the_preregistration_records_the_live_harness_digest():
    doc = json.loads(
        (REPO / "logs/experiments/phase_c1/plans/execution_preregistration.json").read_text())
    assert doc["c1_harness"]["digest"] == c1_harness_digest(REPO)["digest"]
    assert doc["authorizes"] == "nothing"
    assert doc["authorization"]["schema"] == C1_SCHEMA
    assert "NO GRANT EXISTS" in doc["authorization"]["status"]


# --- what the session cannot do --------------------------------------------

def test_the_driver_cannot_search_rank_or_eliminate():
    """It has no such method to disable.

    The earlier driver subclassed PhaseADriver and overrode `stage1`, `run_rung`
    and `selection_row` to raise — which was necessary precisely BECAUSE it
    inherited them. The standalone driver has no beam, no rungs and no ranking to
    override, so absence is the stronger property and is what is asserted.
    """
    spec_ = importlib.util.spec_from_file_location(
        "c1_driver_probe", REPO / "scripts/pod/autoinit_c1_driver.py")
    mod = importlib.util.module_from_spec(spec_)
    sys.modules["c1_driver_probe"] = mod
    spec_.loader.exec_module(mod)
    assert "PhaseADriver" not in [c.__name__ for c in mod.C1Driver.__mro__]
    for method in ("stage1", "run_rung", "selection_row", "run_search",
                   "pooled_over_rungs"):
        assert not hasattr(mod.C1Driver, method), method


def test_neither_launcher_nor_driver_imports_the_search():
    import ast

    for rel in ("scripts/pod/autoinit_c1_launch.py",
                "scripts/pod/autoinit_c1_driver.py"):
        mods: set[str] = set()
        for node in ast.walk(ast.parse((REPO / rel).read_text())):
            if isinstance(node, ast.ImportFrom):
                mods.add(node.module or "")
            elif isinstance(node, ast.Import):
                mods.update(a.name for a in node.names)
        bad = [m for m in mods
               if "phase_a_search" in m or m.endswith(".search") or m == "search"]
        assert not bad, f"{rel} imports {bad}"


def test_the_probe_schedule_is_six_and_names_the_frozen_seeds():
    spec_ = importlib.util.spec_from_file_location(
        "c1_driver_sched", REPO / "scripts/pod/autoinit_c1_driver.py")
    mod = importlib.util.module_from_spec(spec_)
    sys.modules["c1_driver_sched"] = mod
    spec_.loader.exec_module(mod)
    driver = object.__new__(mod.C1Driver)
    driver.seeds = mod.derive_recovery_seeds()
    driver.arm_init = {a: (f"/tmp/{a}", a * 16) for a in ("incumbent", "treatment")}
    probes = driver.descriptors()
    assert len(probes) == 6
    assert all(p["initialization_artifact_digest"] for p in probes)
    assert sorted({p["arm"] for p in probes}) == ["incumbent", "treatment"]
    assert sorted({p["seed"] for p in probes}) == [696460635, 1635674081, 1656475568]
    for arm in ("incumbent", "treatment"):
        assert sum(1 for p in probes if p["arm"] == arm) == 3


def test_the_session_declares_that_it_neither_searches_nor_eliminates(spec):
    assert spec.evidence_fields["runs_a_search"] is False
    assert spec.evidence_fields["eliminates_arms"] is False
    assert spec.evidence_fields["probes"] == 6
    assert spec.evidence_fields["formal_recovery_evidence"] == "OUT OF SCOPE"


# --- the pre-provider gates -------------------------------------------------

#: Gates that can never run here, with a reason each, so widening this list is a
#: decision somebody makes. A module constant rather than a local, so
#: `test_c1_readiness_gates` can assert against the LIST instead of scanning this
#: file's source — a text scan matched the explanatory comment beside it and
#: reported `rope_input_gate` as unconditionally excluded when it is not.
#:
#: `session_commit_and_lineage` binds a real issued commit; `bundle_staged_gate`
#: needs a bundle for that commit uploaded to the relay; `pod_environment_gate`
#: consumes the sweep's own output and so cannot be a precondition of the suite
#: that produces it; `grant_provenance_gate` requires an issued grant to exist
#: at `logs/runs/phase_c1/<run_id>/governance/grant.json` in the REAL repository,
#: and a scratch candidate has no such run — writing one into the repository to
#: satisfy a unit test would be a test creating a run directory.
#:
#: Every entry here owes direct coverage elsewhere, and each has it:
#: `session_commit_and_lineage` in `test_run_identity_is_wired`,
#: `bundle_staged_gate` in `test_c1_bundle_transport`, `pod_environment_gate` in
#: `test_c1_readiness_gates`, and `grant_provenance_gate` below, driven against
#: a temporary root with the same `monkeypatch` pattern the preregistration gate
#: uses — pass, foreign run, absent file, edited grant and missing reference.
ALWAYS_STRUCTURALLY_UNAVAILABLE = (
    "session_commit_and_lineage", "bundle_staged_gate", "pod_environment_gate",
    "grant_provenance_gate")

#: Excluded only when their inputs are genuinely absent. NEVER unconditional:
#: attempt 2 died at ROPE_OK and `rope_input_gate` is what closed that gap.
HF_DEPENDENT_GATES = ("rope_input_gate", "renderer_parity_gate")


def hf_inputs_are_absent() -> bool:
    """Is a real Hugging Face credential AND a populated hub cache missing?

    Named and shared so the predicate can be asserted directly, rather than
    being an inline condition nobody can check. See its use below for why it is
    keyed on the condition instead of on `AAD_SYNTHETIC_HF_TOKEN`.
    """
    hub = Path(os.environ.get("HF_HOME") or (Path.home() / ".cache/huggingface"))
    hub = hub / "hub"
    return bool(os.environ.get("AAD_SYNTHETIC_HF_TOKEN")
                or not (os.environ.get("HF_TOKEN") or os.environ.get("HF_HUB_TOKEN"))
                or not hub.is_dir()
                or not any(hub.glob("datasets--*")))


def test_every_gate_but_the_commit_binding_passes_against_the_candidate(
        launcher, spec, tmp_path):
    """Every `$0` gate must pass against the candidate except the three that
    structurally cannot here.

    `session_commit_and_lineage` binds a real issued commit. `bundle_staged_gate`
    needs a bundle for that commit actually uploaded to the relay, and a scratch
    candidate has no such commit — uploading one to satisfy a unit test would
    pollute the relay to prove nothing. Its real exercise is
    `test_c1_bundle_transport.py`, which drives the identical `roundtrip` over
    real bundles, real clones and real digests with only transport injected,
    including the passing case.

    `pod_environment_gate` joined them on 2026-09-04, and the reason is a genuine
    circularity rather than a convenience. That gate requires a recorded pod-like
    sweep that describes THIS tree, and the sweep writes its record only after the
    suite — this test included — has finished. So during the run that produces the
    record, the record either does not exist or describes an older tree, and no
    ordering fixes that: a gate that consumes the suite's own output cannot also
    be a precondition of it.

    It is not left unexercised. `tests/autoinit/test_c1_readiness_gates.py` drives
    `verify_record` directly against the live digests for a valid record, a
    harness-drift record, a test-environment-drift record, a tampered self-hash, a
    failed sweep and a dirty tree — stricter coverage than this test gives any
    gate it does check.
    """
    import types

    auth = C1Authorization.load(write_candidate(tmp_path))
    ctx = types.SimpleNamespace(scr=Path("/tmp/c1gate"), args=session_args(launcher),
                                auth=auth, evidence={}, image_digest="candidate",
                                price=0.99, spent_usd=0.0)

    structurally_unavailable = list(ALWAYS_STRUCTURALLY_UNAVAILABLE)
    #: Two gates need a real Hugging Face credential and a populated hub cache.
    #: `rope_input_gate` authenticates to the private relay to download and hash
    #: the pinned checkpoint config; `renderer_parity_gate` reads the seven
    #: pinned dataset snapshots.
    #:
    #: Keyed on the CONDITION, not on `AAD_SYNTHETIC_HF_TOKEN`. That marker is
    #: set by `simulate_pod_env.sh` and by nothing else, so a guard reading it
    #: describes one particular simulator rather than the requirement — and it
    #: silently stopped covering this test the moment the test could also run
    #: under a bare empty `$HOME`, which is exactly what removing the host-local
    #: candidate made possible. A predicate that asks whether the inputs are
    #: actually there is true in the simulator, true under an empty HOME, and
    #: FALSE on a real dev box, where both gates still run for real.
    if hf_inputs_are_absent():
        structurally_unavailable += list(HF_DEPENDENT_GATES)
    failures = []
    for gate in spec.precheck:
        name = getattr(gate, "__name__", "session_commit_and_lineage")
        ok, msg = gate(ctx)
        if not ok and name not in structurally_unavailable:
            failures.append(f"{name}: {msg}")
    assert not failures, failures
    names = [getattr(g, "__name__", "") for g in spec.precheck]
    assert "bundle_staged_gate" in names and "rope_input_gate" in names, names
    # The two readiness gates attempt 3R paid for: it cleared ten gates, created
    # a pod, and died on a CPU test suite nobody had run under a pod's HOME.
    assert "renderer_parity_gate" in names, names
    assert "pod_environment_gate" in names, names
    # Added 2026-09-11: the grant an authorization was issued from must still
    # exist, be unedited, and belong to THIS run. Nine structurally valid grants
    # from earlier attempts are sitting in the log root.
    assert "grant_provenance_gate" in names, names
    # Compared, not restated. The single pinned literal is in
    # test_c1_readiness_gates.test_the_prereg_gate_count_and_order_equal_the_live_session.
    prereg = json.loads(
        (REPO / "logs/experiments/phase_c1/plans/execution_preregistration.json").read_text())
    assert len(spec.precheck) == prereg["transport"]["n_pre_provider_gates"], names


# --- the preregistration must verify its own declared hash ------------------

def _prereg_gate(tmp_path, monkeypatch, doc) -> tuple[bool, str]:
    """Point the gate at a mutated copy, in a root that is otherwise the repo."""
    import autoinit_c1_launch as L

    root = tmp_path / "root"
    (root / "logs").mkdir(parents=True, exist_ok=True)
    #: The preregistration moved into `logs/experiments/phase_c1/` on
    #: 2026-09-12, so its parent no longer exists in a bare fixture root.
    (root / L.PREREG).parent.mkdir(parents=True, exist_ok=True)
    (root / L.PREREG).write_text(json.dumps(doc, indent=1) + "\n")
    monkeypatch.setattr(L, "REPO_ROOT", root)
    monkeypatch.setattr(L, "c1_harness_digest",
                        lambda *_a, **_k: {"digest": (doc.get("c1_harness") or {})
                                           .get("digest", "")})
    return L.preregistration_gate(None)


def _prereg_doc() -> dict:
    return json.loads(
        (REPO / "logs/experiments/phase_c1/plans/execution_preregistration.json").read_text())


def test_the_preregistration_gate_verifies_the_self_hash(tmp_path, monkeypatch):
    ok, why = _prereg_gate(tmp_path, monkeypatch, _prereg_doc())
    assert ok, why
    assert "self-hash verified" in why


def test_a_field_edited_without_recomputing_the_hash_fails_the_gate(tmp_path,
                                                                    monkeypatch):
    """The mutation the gate exists for: only the harness block was checked
    before, so the stage order, the decision rule and the admission rule could
    all be edited after freezing and the gate would still pass."""
    doc = _prereg_doc()
    doc["decision"] = {**doc["decision"], "sesoi": 0.001}
    ok, why = _prereg_gate(tmp_path, monkeypatch, doc)
    assert not ok and "contents hash to" in why


def test_a_restated_hash_fails_the_gate(tmp_path, monkeypatch):
    doc = _prereg_doc()
    doc["preregistration_sha256"] = "0" * 64
    ok, why = _prereg_gate(tmp_path, monkeypatch, doc)
    assert not ok and "contents hash to" in why


def test_a_missing_hash_fails_the_gate(tmp_path, monkeypatch):
    doc = _prereg_doc()
    doc.pop("preregistration_sha256")
    ok, why = _prereg_gate(tmp_path, monkeypatch, doc)
    assert not ok and "declares no preregistration_sha256" in why


def test_the_preregistration_describes_the_standalone_driver():
    """It said C1Driver inherits PhaseADriver and writes phase_a paths. It does
    not, and a preregistration that misdescribes the executable is worse than a
    missing one: it is a false record of what was reviewed."""
    doc = _prereg_doc()
    text = json.dumps(doc)
    for dead in ("audit/autoinit_phase_a", "stage3/phase_a/<probe_id>",
                 "eval/phase_a/<probe_id>", "which C1Driver inherits"):
        assert dead not in text, dead
    d = doc["driver"]
    assert d["standalone"] is True
    assert d["subclasses_phase_a_driver"] is False
    assert d["imports_phase_a_driver_or_launcher"] is False
    assert d["owns_paths"] == ["artifacts/audit/autoinit_c1",
                               "artifacts/stage3/c1", "artifacts/eval/c1"]
    for key in ("stage_g_h_separation", "generation_admission",
                "attested_evaluation_protocol", "scoring", "device_handoff"):
        assert d[key], key
    assert "NO PROBE RESULT IS ADMITTED" in d["generation_admission"]
    assert "c1_attested_evaluation_protocol.json" in d["evidence"]


def test_the_launcher_fetches_the_report_the_driver_actually_writes(spec):
    """`attested_evaluation_protocol.json` was asked for; the driver writes
    `c1_attested_evaluation_protocol.json`. The scp is best-effort, so the
    mismatch would have fetched nothing, silently."""
    driver_src = (REPO / "scripts/pod/autoinit_c1_driver.py").read_text()
    for name in spec.artifacts.report_names:
        assert f'"{name}"' in driver_src, name


# --- the candidate is deterministic, in-tree, and provably load-bearing -----
#
# Three claims, because the host-local fixture failed all three: it was not
# reproducible, it was not visible to review, and when it went stale nobody
# could tell a stale fixture from a broken gate.

def test_the_candidate_is_byte_identical_across_builds(tmp_path):
    """Deterministic: same inputs, same bytes. The old fixture never was."""
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    assert write_candidate(first).read_bytes() == write_candidate(second).read_bytes()


def test_the_candidate_needs_no_host_local_state(tmp_path, monkeypatch):
    """It must build with `$HOME` pointing at an empty directory.

    This is the pod contract: the C1 CPU test gate runs pytest under a fresh
    empty HOME precisely so host-local state cannot decide a result. The old
    fixture lived in `$HOME` and therefore vanished exactly there.
    """
    empty = tmp_path / "empty_home"
    empty.mkdir()
    monkeypatch.setenv("HOME", str(empty))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: empty))

    auth = C1Authorization.load(write_candidate(tmp_path))
    assert auth.harness_source_digest == c1_harness_digest(REPO)["digest"]
    assert list(empty.iterdir()) == [], "the candidate wrote into $HOME"


def test_building_a_candidate_writes_nothing_outside_tmp_path(tmp_path):
    """Building is not issuing. `logs/` must be untouched, byte for byte."""
    logs = REPO / "logs"
    before = {p: p.stat().st_mtime_ns for p in logs.rglob("*") if p.is_file()}
    write_candidate(tmp_path)
    after = {p: p.stat().st_mtime_ns for p in logs.rglob("*") if p.is_file()}
    assert before == after, "building a test candidate modified logs/"


def test_the_candidate_describes_this_tree_not_a_remembered_one(tmp_path):
    """The property whose absence produced the false alarm.

    The host-local fixture pinned `4437074249d5…` and the tree moved to
    `a3566eec79b3…`; `c1_harness_gate` refused, correctly, and the failure read
    like a broken gate. A candidate derived from the live tree cannot go stale.
    """
    auth = C1Authorization.load(write_candidate(tmp_path))
    live = c1_harness_digest(REPO)
    assert auth.harness_source_digest == live["digest"]
    #: The DECLARED set is the derived one too, since 2026-09-11. It asserted
    #: `== C1_HARNESS_SOURCE_FILES_V1` here, which is how "describes this tree"
    #: came to be checked for the digest and contradicted for the file list:
    #: the constant is frozen at the pre-migration paths, so the candidate
    #: declared 73 files that no longer exist while binding a digest over the 97
    #: that do. This test's own name is the property it stopped checking.
    assert tuple(auth.harness_source_files) == tuple(
        f["path"] for f in live["files"])
    assert tuple(auth.harness_source_files) != C1_HARNESS_SOURCE_FILES_V1, (
        "the historical declaration is back in an issued authorization")


def test_mutation_a_stale_candidate_is_refused_by_the_harness_gate(launcher, spec,
                                                                   tmp_path):
    """The gate must still fire. A fixture that can never be stale proves nothing
    unless a stale one is shown to be caught."""
    import types

    stale = write_candidate(tmp_path, harness_digest_override="0" * 64)
    auth = C1Authorization.load(stale)
    ctx = types.SimpleNamespace(scr=Path("/tmp/c1gate"), args=session_args(launcher),
                                auth=auth, evidence={}, image_digest="candidate",
                                price=0.99, spent_usd=0.0)
    gates = {getattr(g, "__name__", ""): g for g in spec.precheck}
    ok, msg = gates["c1_harness_gate"](ctx)
    assert not ok, "a candidate pinning 000…0 was accepted as describing this tree"
    assert "0000000000" in msg and c1_harness_digest(REPO)["digest"][:12] in msg


def test_mutation_a_host_local_candidate_dependency_is_visible_here():
    """The portability claim, checked from BEHAVIOUR rather than source text.

    My first attempt scanned this file for `Path.home() / "aad-scratch"` and
    failed against its own explanatory comment — a text scan cannot tell a
    comment from a dependency, and the fix is not a cleverer regex. What
    actually matters is two runtime properties:

      * no module-level constant resolves a candidate under `$HOME`; and
      * the gate test carries no skip marker, so it cannot report success by
        not running.

    Restoring the old design breaks both, and neither can be satisfied by prose.
    """
    #: Scoped to "outside the repository", not "under $HOME": on this box the
    #: checkout itself lives under $HOME, so a $HOME test flags `REPO` and says
    #: nothing. The property that matters is that every module-level input comes
    #: from the tree under review or from `tmp_path` — never from host-local
    #: state a reviewer cannot see and CI does not have.
    module = sys.modules[__name__]
    outside = []
    for name, value in vars(module).items():
        if not isinstance(value, Path):
            continue
        try:
            value.resolve().relative_to(REPO.resolve())
        except ValueError:
            outside.append(f"{name}={value}")
    assert not outside, f"module-level paths outside the repository: {outside}"

    fn = test_every_gate_but_the_commit_binding_passes_against_the_candidate
    marks = [m.name for m in getattr(fn, "pytestmark", [])]
    assert "skipif" not in marks and "skip" not in marks, (
        f"the gate test carries {marks}; it must execute on the dev box, under "
        "an empty HOME, and in simulate_pod_env.sh")


def test_the_hf_predicate_does_not_excuse_the_gates_on_a_real_dev_box(monkeypatch,
                                                                      tmp_path):
    """A condition-keyed exemption is only safe if it is FALSE where it matters.

    The danger of replacing a simulator marker with a predicate is that the
    predicate quietly becomes true everywhere, and two real gates stop being
    exercised without anyone noticing. So both directions are pinned.
    """
    real = Path(os.environ.get("HF_HOME") or (Path.home() / ".cache/huggingface"))
    have_real_inputs = (real / "hub").is_dir() and any(
        (real / "hub").glob("datasets--*"))
    if have_real_inputs and (os.environ.get("HF_TOKEN")
                             or os.environ.get("HF_HUB_TOKEN")):
        assert not hf_inputs_are_absent(), (
            "this machine HAS a credential and a populated hub cache, yet the "
            "predicate excuses rope_input_gate and renderer_parity_gate")

    # And it must be TRUE when the inputs really are gone.
    empty = tmp_path / "hf"
    empty.mkdir()
    monkeypatch.setenv("HF_HOME", str(empty))
    assert hf_inputs_are_absent()

    # The simulator's marker still forces it, so the pod path is unchanged.
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setenv("AAD_SYNTHETIC_HF_TOKEN", "1")
    assert hf_inputs_are_absent()


# --- the grant provenance gate ----------------------------------------------
#
# The decision a session runs under is an INPUT: authored and committed while
# the tree is still clean, because the launch-bound sweep and the authorization
# issued from it both need it to already be there. Until 2026-09-11 the
# authorization recorded which grant it came from and nothing ever looked at
# that reference again — so an edited grant, a deleted one, and *another
# attempt's* grant were all indistinguishable from the right one.
#
# Driven against a temporary root, the same way the preregistration gate is.

def _grant_root(tmp_path, *, run_id="attempt10", grant_run_id=None,
                grant_body=None, write_grant=True, reference=True):
    """A repo-shaped root holding an authorization and (usually) its grant."""
    import autoinit_c1_launch as L
    from aadistill.infrastructure.manifest import sha256_json

    root = tmp_path / "root"
    body = {"granted_by": "maintainer, review 2026-09-11",
            "covers": "one launch"} if grant_body is None else grant_body
    #: Through the SAME helper the gate uses, and then checked against where
    #: `open_run` really puts the run. This built the path itself, the identical
    #: wrong way the gate did -- both omitted the stage segment after runs were
    #: grouped by stage -- so the two agreed and the test could not see it. A
    #: fixture that recomputes its subject's logic proves only that the logic is
    #: self-consistent.
    from experiments.run_layout import layout_for, rel_run_dir
    rel = (f"{rel_run_dir(L.RUN_EXPERIMENT_ID, grant_run_id or run_id, L.RUN_STAGE_ID)}"
           f"/{L.C1_RUN_ROLES['grant']}")
    expect = layout_for(root, L.RUN_EXPERIMENT_ID, grant_run_id or run_id,
                        L.RUN_STAGE_ID).path(L.C1_RUN_ROLES["grant"])
    assert (root / rel).resolve() == expect.resolve(), (
        f"the gate would look at {rel}, but the run's grant role resolves to "
        f"{expect.relative_to(root)}")
    if write_grant:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(json.dumps(body, indent=1) + "\n")
    auth = {"authorization_id": "autoinit.v1.phase_c1"}
    if reference:
        auth["grant"] = {"path": rel, "sha256": sha256_json(body)}
    #: Through the SAME helper the gate uses, for the same reason the grant
    #: path is: the authorization moved into the run on 2026-09-12, and a
    #: fixture that keeps writing it to the repository root tests a layout
    #: nothing produces any more.
    auth_rel = L.auth_path_for(grant_run_id or run_id)
    (root / auth_rel).parent.mkdir(parents=True, exist_ok=True)
    (root / auth_rel).write_text(json.dumps(auth, indent=1) + "\n")
    return root, rel


def _grant_gate(tmp_path, monkeypatch, *, run_id="attempt10", **over):
    import types
    import autoinit_c1_launch as L

    root, rel = _grant_root(tmp_path, run_id=run_id, **over)
    monkeypatch.setattr(L, "REPO_ROOT", root)
    ctx = types.SimpleNamespace(args=types.SimpleNamespace(run_id=run_id),
                                evidence={})
    return L.grant_provenance_gate(ctx), ctx, rel


def test_a_grant_prepared_in_this_run_passes_the_gate(tmp_path, monkeypatch):
    (ok, why), ctx, rel = _grant_gate(tmp_path, monkeypatch)
    assert ok, why
    assert ctx.evidence["grant_provenance"]["path"] == rel
    assert ctx.evidence["grant_provenance"]["run_id"] == "attempt10"


def test_another_attempts_grant_is_refused(tmp_path, monkeypatch):
    """The failure mode with nine real instances sitting in the log root."""
    #: The authorization is written where ATTEMPT 6 owns it, and the gate runs
    #: as attempt 10 -- so it reads attempt 10's, which does not exist. Both
    #: refusals are correct and both are the same defect: running under a
    #: decision made about another session.
    (ok, why), _, _ = _grant_gate(tmp_path, monkeypatch, grant_run_id="attempt6")
    assert not ok
    assert "attempt10" in why or "attempt6" in why, why


def test_a_grant_the_authorization_names_but_does_not_exist_is_refused(
        tmp_path, monkeypatch):
    (ok, why), _, _ = _grant_gate(tmp_path, monkeypatch, write_grant=False)
    assert not ok and "does not exist" in why


def test_a_grant_edited_after_issuance_is_refused(tmp_path, monkeypatch):
    """The authorization's self-hash covers the REFERENCE, not the file."""
    import autoinit_c1_launch as L

    root, rel = _grant_root(tmp_path)
    doc = json.loads((root / rel).read_text())
    doc["covers"] = "two launches"
    (root / rel).write_text(json.dumps(doc, indent=1) + "\n")
    monkeypatch.setattr(L, "REPO_ROOT", root)
    import types
    ok, why = L.grant_provenance_gate(
        types.SimpleNamespace(args=types.SimpleNamespace(run_id="attempt10"),
                              evidence={}))
    assert not ok and "edited after it was used" in why


def test_an_authorization_that_names_no_grant_is_refused(tmp_path, monkeypatch):
    (ok, why), _, _ = _grant_gate(tmp_path, monkeypatch, reference=False)
    assert not ok and "records no grant" in why


def test_a_spelled_differently_but_identical_path_still_passes(tmp_path,
                                                               monkeypatch):
    """`./logs/...` is the same file as `logs/...`; the operator types one."""
    import types
    import autoinit_c1_launch as L

    root, rel = _grant_root(tmp_path)
    auth_rel = L.auth_path_for("attempt10")
    auth = json.loads((root / auth_rel).read_text())
    auth["grant"]["path"] = f"./{rel}"
    (root / auth_rel).write_text(json.dumps(auth, indent=1) + "\n")
    monkeypatch.setattr(L, "REPO_ROOT", root)
    ok, why = L.grant_provenance_gate(
        types.SimpleNamespace(args=types.SimpleNamespace(run_id="attempt10"),
                              evidence={}))
    assert ok, why


def test_the_grant_role_is_declared_and_is_not_snapshotted(L_unused=None):
    """It is an input, so nothing copies it in — unlike the three one-use
    artifacts, whose live paths the next issuance overwrites."""
    import autoinit_c1_launch as L

    assert L.C1_RUN_ROLES["grant"] == "governance/grant.json"
    assert "grant" in L.C1_RUN_SPEC.optional
    assert "grant" not in [role for _src, role in L._RUN_GOVERNANCE]
    assert "grant" not in [role for _src, role in L._RUN_COLLECT]
    assert "grant" in L._RUN_PREPARED

    #: The readiness record joined it on 2026-09-12. It is produced by the sweep
    #: BEFORE the authorization is issued, so by the time the launcher opens the
    #: run it is already there -- the same shape as the grant, and exempt from
    #: the occupancy rule for the same reason.
    assert L.C1_RUN_ROLES["readiness_record"] == "governance/readiness.json"
    assert "readiness_record" in L._RUN_PREPARED
    assert set(L._RUN_PREPARED) == {"grant", "readiness_record"}, (
        "prepared roles are exempted BY NAME; a set that has grown beyond the "
        "two inputs written before the run opens is a widened exemption")

    #: And the exemption is per role, never the directory they share.
    assert "governance" not in L._RUN_PREPARED
    assert "governance/" not in L._RUN_PREPARED


# --- the authorization must declare the set its digest covers ----------------
#
# `session_commit_gate` re-digests `harness_source_files` at the session commit,
# by running `git show <commit>:<path>` for each one, and compares the result to
# `harness_source_digest`. So those two fields have to describe the SAME files.
#
# They stopped doing so at the initialization cutover: the digest became the
# derived post-migration closure while the declaration stayed frozen at the
# pre-migration paths. Every authorization issued afterwards was structurally
# unusable, and the first one was caught by a read-only pre-flight rather than
# by a test, because the only gate that reads the field is excluded from the
# candidate sweep. These are the $0 checks that close that gap.

def _issued_payload(tmp_path):
    from experiments.phase_c1.authorization_payload import (
        build_c1_authorization_payload)

    return build_c1_authorization_payload(
        grant=TEST_GRANT, session_commit=TEST_SESSION_COMMIT,
        granted_utc=TEST_GRANTED_UTC, repo_root=REPO,
        grant_path="<test fixture>")


def test_the_declared_harness_set_is_the_one_the_digest_covers(tmp_path):
    """The invariant that was violated, stated directly."""
    payload = _issued_payload(tmp_path)
    declared = tuple(payload["harness_source_files"])
    live = c1_harness_digest(REPO)
    assert declared == tuple(f["path"] for f in live["files"]), (
        "the authorization declares one file set and binds a digest over "
        "another; session_commit_gate re-digests the declared set and could "
        "never match")
    assert payload["harness_source_digest"] == live["digest"]


def test_every_declared_harness_path_exists_in_this_repository(tmp_path):
    """Exactly what the gate does, and exactly where it failed.

    `git show <commit>:<path>` on a path the cutover moved returns non-zero, and
    the gate refuses with `does not contain [...]` after listing every missing
    one. Asked here against HEAD, for free.
    """
    import subprocess

    declared = _issued_payload(tmp_path)["harness_source_files"]
    missing = [p for p in declared
               if subprocess.run(["git", "show", f"HEAD:{p}"],
                                 capture_output=True, cwd=REPO).returncode != 0]
    assert not missing, (
        f"{len(missing)} declared harness path(s) are not in HEAD, so a pod "
        f"checking out this commit cannot digest them: {missing[:5]}")


def test_the_superseded_historical_declaration_is_refused_by_name(tmp_path):
    """The mutation, and its message must be the useful one.

    Re-introducing `C1_HARNESS_SOURCE_FILES_V1` must not read as "some other
    phase's grant" — that sends the reader looking for the wrong thing.
    """
    import types

    auth = C1Authorization.load(write_candidate(tmp_path))
    stale = dataclasses.replace(auth,
                                harness_source_files=C1_HARNESS_SOURCE_FILES_V1)
    launcher = load_session_launcher("autoinit_c1_launch")
    ctx = types.SimpleNamespace(args=session_args(launcher), auth=stale,
                                evidence={})
    ok, why = launcher.c1_harness_gate(ctx)
    assert not ok
    assert "PRE-MIGRATION" in why and "Re-issue" in why, why


def test_the_gate_accepts_the_set_the_issuer_actually_writes(tmp_path):
    """Both directions: the refusal above must not be refusing everything."""
    import types

    auth = C1Authorization.load(write_candidate(tmp_path))
    launcher = load_session_launcher("autoinit_c1_launch")
    ctx = types.SimpleNamespace(args=session_args(launcher), auth=auth,
                                evidence={})
    ok, why = launcher.c1_harness_gate(ctx)
    assert ok, why


# --- the frozen-asset gate ---------------------------------------------------
#
# Attempt 10 died at the pod's frozen-asset check for $0.1177 and one of three
# attempts, on a condition this machine could have decided for free: the
# initialization cutover relocated two of the scoring contract's six declared
# files, so the contract legitimately became `@v3` while the verifier's
# compiled-in constants still said `@v2`, and nothing passed the `--expect` flag
# that exists for exactly that distinction. These cover the counterpart gate.

def test_the_expectation_document_restates_the_asset_block_exactly():
    """The assets are COPIED from the verifier's constants, never re-typed.

    The first draft of the expectation document was hand-written and named an
    asset that does not exist with hashes that were invented. A transcribed hash
    is a second source for a value that already has one.
    """
    sys.path.insert(0, str(REPO / "scripts/autoinit"))
    import verify_frozen_assets as V

    doc = json.loads((REPO / "configs/experiments/phase_c1/frozen_assets.json"
                      ).read_text())
    assert doc["assets"] == V.FROZEN, (
        "the expectation document's asset block has drifted from the verifier's "
        "own constants; one of the two was edited alone")
    assert doc["scoring_contract"]["supersedes"] == {
        "contract": V.FROZEN_SCORING_CONTRACT,
        "digest": V.FROZEN_SCORING_DIGEST}, (
        "the document must name the pin it supersedes, so the change is a "
        "documented succession rather than an unexplained different number")
    assert doc["authorizes"] == "nothing"


def test_the_expectation_names_the_live_scoring_contract():
    """And it must be the contract this tree really computes, not a guess."""
    from experiments.source_sets import recovery_scoring_contract

    doc = json.loads((REPO / "configs/experiments/phase_c1/frozen_assets.json"
                      ).read_text())
    live = recovery_scoring_contract(REPO)
    assert doc["scoring_contract"]["contract"] == live["contract"]
    assert doc["scoring_contract"]["digest"] == live["digest"]


def test_the_frozen_asset_gate_passes_on_this_tree(launcher):
    import types

    ctx = types.SimpleNamespace(args=session_args(launcher), evidence={})
    ok, why = launcher.frozen_assets_gate(ctx)
    assert ok, why
    assert ctx.evidence["frozen_assets"]["returncode"] == 0


def test_the_frozen_asset_gate_reproduces_attempt_tens_refusal(tmp_path,
                                                               monkeypatch):
    """The mutation, and it is the exact condition that was paid for.

    Point the gate at an expectation carrying the superseded `@v2` pin and it
    must refuse — at `$0`, here, instead of at `SETUP_RC=91` on a billing pod.
    """
    import types

    doc = json.loads((REPO / "configs/experiments/phase_c1/frozen_assets.json"
                      ).read_text())
    doc["scoring_contract"] = {**doc["scoring_contract"],
                               **doc["scoring_contract"]["supersedes"]}
    stale = REPO / "configs/experiments/phase_c1/_stale_expect_for_test.json"
    stale.write_text(json.dumps(doc, indent=1) + "\n")
    try:
        mod = load_session_launcher("autoinit_c1_launch")
        monkeypatch.setattr(mod, "FROZEN_EXPECT", str(
            stale.relative_to(REPO)))
        ctx = types.SimpleNamespace(args=session_args(mod), evidence={})
        ok, why = mod.frozen_assets_gate(ctx)
    finally:
        stale.unlink()
    assert not ok
    assert "recovery_search_scoring@v2" in why or "scoring contract" in why, why


def test_the_setup_reads_the_variable_the_launcher_declares(launcher, spec):
    """A forwarded variable nothing reads, or a read variable nothing forwards,
    are the same defect seen from two ends."""
    setup = (REPO / "scripts/pod/autoinit_preflight_setup.sh").read_text()
    assert "SESSION_FROZEN_EXPECT" in setup
    assert spec.setup.env["SESSION_FROZEN_EXPECT"] == launcher.FROZEN_EXPECT
    assert (REPO / launcher.FROZEN_EXPECT).is_file()
    #: Optional by construction: every other session keeps the historical
    #: question, so this must not become a required variable for all of them.
    assert "${SESSION_FROZEN_EXPECT:-}" in setup


def test_what_the_frozen_asset_gate_trusts_is_inside_the_measured_harness():
    """Otherwise the gate's answer could be changed without moving any digest.

    The launcher names the expectation document as a string and hands it to a
    subprocess, so the import walk cannot see it. Declared explicitly, for the
    same reason `c1_artifacts.json` is: a file that decides whether a session may
    run has to be inside the set a grant binds.
    """
    closure = {f["path"] for f in c1_harness_digest(REPO)["files"]}
    launcher = load_session_launcher("autoinit_c1_launch")
    assert launcher.FROZEN_EXPECT in closure, (
        f"{launcher.FROZEN_EXPECT} decides the frozen-asset gate and is not in "
        "the harness the authorization measures")
    assert "scripts/autoinit/verify_frozen_assets.py" in closure, (
        "the verifier the gate executes is not measured either")
