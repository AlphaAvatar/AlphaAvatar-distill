"""Behavioural launch checks that can only run on the DEV BOX. Not pod-safe.

Everything here needs something a pod does not have: the out-of-tree durable
store at `/home/ecs-user/aad-artifacts`, or the launcher's full `SessionSpec`,
which hashes a plan derived by joining the frozen selection to that store's
destination-verified products.

That is why they are HERE and not in `tests/c2_behavioural_preflight/`. That
directory is the pod selection: `ignores_for_selection` derives the pod's
`pytest tests/ --ignore ...` complement so that exactly that directory is
collectable, and everything in it runs inside the paid session's blocking
TESTS_OK gate, on a container with no `/home/ecs-user`.

The distinction is not theoretical. `test_the_durable_store_can_hold_twelve_probes`
lived in the pod selection and asserted `/home/ecs-user/aad-artifacts` is a
directory — which is true on the dev box, true under `simulate_pod_env.sh`
(which isolates `$HOME` as an environment variable and does not hide absolute
paths outside the repository), and false on a pod. The launch-bound readiness
sweep would therefore have been green about a gate that fails a minute or two
into a billing pod. Its real production caller is `destination_gate`, which runs
on the dev box before a pod exists and is tested in the pod selection against
temporary stores.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in ("src", "scripts", "scripts/autoinit", "scripts/pod"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from experiments.phase_c2 import behavioural as BH  # noqa: E402
from experiments.phase_c2 import behavioural_governance as BG  # noqa: E402

import autoinit_c2_behavioural_launch as L  # noqa: E402


def test_the_durable_store_can_hold_twelve_probes():
    """A durability mechanism needs a backend with room for what it protects.

    C1 attempt 18 ran a correct preservation mechanism on six probes and
    preserved none of them, because every upload was refused for quota. The
    mechanism behaved perfectly and nothing survived.
    """
    store = Path("/home/ecs-user/aad-artifacts")
    assert store.is_dir(), "the out-of-tree artifact store must exist"
    need = 12 * 1.11 * 2**30
    assert shutil.disk_usage(store).free > need


def test_the_campaign_id_is_in_the_plan_an_authorization_binds():
    """Renaming the campaign must not be a way to obtain a second permission.

    `plan_payload` joins the frozen selection to the dev box's durable products,
    so it cannot run on a pod — which is the whole reason this assertion is not
    beside the rest of the campaign checks.
    """
    payload = BG.plan_payload(REPO)
    assert payload["campaign_id"] == BG.CAMPAIGN_ID
    #: And the hash really covers it.
    from aadistill.infrastructure.manifest import sha256_json

    assert BG.plan_hash(REPO) == sha256_json(payload)
    assert BG.plan_hash(REPO) != sha256_json(
        {**payload, "campaign_id": "some-other-campaign"})


def test_the_continuation_gate_is_in_the_launchers_precheck_chain():
    """A gate nothing calls protects nothing.

    Through the REAL parser and the REAL `spec()`: a hand-built namespace is how
    a launcher comes to be tested against arguments it never receives, and one
    died at `$0.0603` on exactly that. `spec()` hashes the plan, which is what
    makes this dev-box-only.
    """
    args = L.build_parser().parse_args(
        ["--scr", "/tmp/probe", "--session-commit", "abc123", "--bundle", "b",
         "--run-id", "attempt1", "--max-price", "1.09"])
    names = [getattr(g, "__name__", "") for g in L.spec(args).precheck]
    assert "campaign_continuation_gate" in names
    assert "destination_gate" in names
    #: BEFORE the one gate that touches the network, which must stay last.
    assert names.index("campaign_continuation_gate") < names.index(
        "bundle_staged_gate")
    assert names[-1] == "bundle_staged_gate"


def test_the_replacement_resource_handoff_is_wired_into_the_session():
    """A mechanism with no production caller protects nothing.

    Found by mutation: deleting `materialize_inputs=restore_campaign_probes`
    from the spec left every continuation test green, because they call the
    restore step directly. Tests prove a mechanism works; only this proves the
    session uses it.

    `materialize_inputs` is the right seam and not an arbitrary one: the runner
    calls it AFTER setup and BEFORE the driver starts, and tears the pod down
    if it returns `False`. A probe restored after the driver had begun would be
    a probe the campaign journal had already decided was absent — and the
    protocol forbids retraining it.
    """
    args = L.build_parser().parse_args(
        ["--scr", "/tmp/probe", "--session-commit", "abc123", "--bundle", "b",
         "--run-id", "attempt1", "--max-price", "1.09"])
    spec = L.spec(args)
    assert spec.materialize_inputs is L.restore_campaign_probes, (
        "the session does not declare the continuation restore step, so a "
        "replacement resource would start its driver on a fresh filesystem "
        "with none of its campaign's verified probes")
    #: And the driver is told where to find what that step staged.
    class _Ctx:
        image_digest = "sha256:abc"
        price = 1.09
        spent_usd = 0.0
        args = L.build_parser().parse_args(
            ["--scr", "/tmp/probe", "--session-commit", "abc123",
             "--bundle", "b", "--run-id", "attempt1", "--max-price", "1.09"])

        class auth:
            hard_cap_usd = 32.7097
            campaign_id = BG.CAMPAIGN_ID

    command = L.driver_command(_Ctx(), type("P", (), {"soft_stop_usd": 30.0})())
    assert f"--continuation-manifest {L.RESTORE_MANIFEST}" in command


def test_the_session_plan_carries_the_one_canonical_budget():
    """The launcher's SessionSpec prices the same decomposition as the proposal.

    The parity itself is asserted in the pod selection, where it costs nothing.
    What this adds is that the spec a LAUNCH actually builds — not `budget()`
    called directly — carries it.
    """
    args = L.build_parser().parse_args(
        ["--scr", "/tmp/probe", "--session-commit", "abc123", "--bundle", "b",
         "--run-id", "attempt1", "--max-price", "1.09"])
    spec = L.spec(args)
    plan = spec.budget.plan(price_per_hour=BG.QUOTED_RATE_USD_PER_HOUR,
                            authorized_usd=10_000.0)
    #: AT THE LAUNCHER'S OWN WORK INPUTS. This compared the spec's plan to the
    #: decomposition at the FULL-session defaults, which is the same thing only
    #: while the campaign owes all twelve probes. attempt5 completed ten, so
    #: the launcher now prices the remainder under R10 and the two figures
    #: legitimately differ -- and the assertion made that correct behaviour a
    #: failure. One MODEL is what must hold: the spec a launch builds carries
    #: the canonical decomposition, whatever work it is fed.
    work = L.budget_work(args)
    d = BH.session_decomposition(REPO, **work)
    assert plan.hard_terminate_minutes == d["hard_minutes"]
    assert plan.expected_minutes == d["expected_minutes"]
    assert plan.soft_stop_minutes == d["soft_stop_minutes"]
    #: and at the full-session defaults it is still the proposal's ceiling
    full = BH.session_decomposition(
        REPO, materialization_minutes=BG.materialization_minutes(
            REPO)["total_minutes"])
    assert float(BG.ceiling(
        REPO, gpu_rate_usd_per_hour=BG.QUOTED_RATE_USD_PER_HOUR
    )["hard_ceiling"]["minutes"]) == full["hard_minutes"]
    assert spec.plan_hash == BG.plan_hash(REPO)
