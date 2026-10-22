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


def _commit(root: Path, msg: str) -> None:
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t",
                    "commit", "-q", "-m", msg], check=True)


def test_a_new_reconciliation_anywhere_leaves_open_pods_open(tmp_path):
    """The H1 budget position -- a reconciliation at a path the exclusion list
    did not know -- made every OPEN pod it listed read as mentioned. A report
    that cites the snapshot it was derived from attributes nothing, wherever it
    is committed and whatever it is called."""
    root = _repo(tmp_path)
    _commit(root, "evidence")
    ids = ["podlistedonly", "podinaledger"]
    before = PB.mentioned(root, ids)
    report = root / "logs/stages/stage-9/some_experiment/budget/position_v7.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({
        "snapshot": {"path": "logs/budget/provider/runpod_billing_2026-01-01.json", "sha256": "0" * 64},
        "unnamed": {"pods": ["podlistedonly"], "status": "OPEN"}}))
    (root / "docs").mkdir()
    (root / "docs/budget note.md").write_text(
        "Per the provider snapshot (schema aadistill.provider_billing_snapshot/v1), "
        "podlistedonly is still OPEN.\n")
    _commit(root, "a new reconciliation")
    assert PB.mentioned(root, ids) == before == {"podinaledger"}


def test_a_record_that_does_not_cite_the_billing_evidence_still_attributes(tmp_path):
    root = _repo(tmp_path)
    (root / "logs/stages/x/launch.log").write_text("created pod podlistedonly for the pilot\n")
    _commit(root, "a launch log")
    assert PB.mentioned(root, ["podlistedonly"]) == {"podlistedonly"}
