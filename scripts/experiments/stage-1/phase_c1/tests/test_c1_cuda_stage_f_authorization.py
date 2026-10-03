"""What the 2026-09-10 cuda-stage-f authorization actually says.

Split from `tests/validation/test_cuda_engineering_launch.py` by the 2026-10-03
convergence round. The launcher's MECHANISM — how it reads a resource contract,
carries prior spend into both ceilings, refuses an exhausted campaign and refuses
a validation with no ledger — stays in core and is proved there against an
authorization and ledger built under `tmp_path`.

These three are facts about a closed record: its withdrawn count-based clauses,
its teardown reserve sitting inside its ceiling, and what it declined to
authorize. A core test asserting them made the core suite depend on this
directory still existing.

The count-based clauses are HISTORY. Stage F ran under them and they were
withdrawn on 2026-09-17 because attempt count is not budget; the launcher no
longer reads them and an authorization written after the withdrawal omits them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "src"))

AUTH = (REPO / "logs/stages/stage-1/phase_c1/validations/cuda-stage-f/v1"
        / "authorization.json")

pytestmark = pytest.mark.skipif(
    not AUTH.is_file(),
    reason="the cuda-stage-f authorization is not staged in this checkout")


def test_it_reads_the_engineering_authorization_not_a_c1_grant():
    doc = json.loads(AUTH.read_text())
    assert doc["schema"] == "aadistill.engineering_validation_authorization/v1"
    rc = doc["resource_contract"]
    #: The count-based clauses are HISTORY: stage F ran under them, and they
    #: were withdrawn on 2026-09-17 because attempt count is not budget. They
    #: are asserted here as a fact about a closed record, not as a contract
    #: the launcher still enforces -- it no longer reads them, and an
    #: authorization written after the withdrawal omits them.
    assert rc["provider_resources_max"] == 1
    assert rc["provider_create_attempts_max"] == 1
    assert rc["retries_or_replacement_pods"] == 0
    assert rc["engineering_soft_cap_usd"] == 0.25
    assert rc["total_resource_cost_ceiling_usd"] == 0.40
    assert rc["teardown_reserve_usd"] == 0.15


def test_the_reserve_sits_inside_the_ceiling():
    rc = json.loads(AUTH.read_text())["resource_contract"]
    assert rc["teardown_reserve_usd"] < rc["total_resource_cost_ceiling_usd"]


def test_it_authorizes_nothing_formal():
    doc = json.loads(AUTH.read_text())
    joined = " ".join(doc["does_not_authorize"]).lower()
    for forbidden in ("attempt-10", "formal c1 bundle", "confirmation battery",
                      "formal c1 decision", "merging"):
        assert forbidden in joined, forbidden
    assert doc["formal_c1_status_unchanged"]["formal_treatment"] == "UNMEASURED"
    assert doc["formal_c1_status_unchanged"]["attempt_9"] == "NO DECISION"
