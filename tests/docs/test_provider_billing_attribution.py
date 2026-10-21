"""Billing evidence does not attribute the pods it lists.

`provider_billing.mentioned` asks whether a billed pod is known to the
repository. Committing a billing snapshot -- or a reconciliation listing the
unattributed pods -- once made every pod in it read as known: the 53 pods the
2026-10-21 ledger called "named by no record" reported as 0 the next time the
report ran. A pod named only by the billing evidence must stay unattributed; a
pod named by any other record is mentioned.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "provider_billing_under_test", REPO / "scripts/maintenance/consolidation/provider_billing.py")
PB = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PB)


def _repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "logs/budget/provider").mkdir(parents=True)
    (tmp_path / "logs/stages/x/closeout").mkdir(parents=True)
    (tmp_path / "logs/budget/provider/runpod_billing_2026-01-01.json").write_text(
        json.dumps({"pods": [{"podId": "podlistedonly"}, {"podId": "podinaledger"}]}))
    (tmp_path / "logs/stages/x/closeout/budget_reconciliation.json").write_text(
        json.dumps({"pods_only_mentioned": ["podlistedonly"]}))
    (tmp_path / "logs/ledger.md").write_text("the pod podinaledger ran the pilot\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    return tmp_path


def test_a_pod_named_only_by_billing_evidence_is_not_mentioned(tmp_path):
    root = _repo(tmp_path)
    assert PB.mentioned(root, ["podlistedonly", "podinaledger"]) == {"podinaledger"}


def test_no_ids_is_an_empty_set(tmp_path):
    assert PB.mentioned(_repo(tmp_path), []) == set()
