"""Staging leaves on the pod is not durability — they have to leave it.

The first version of this closure copied the five selected leaves to
`artifacts/audit/autoinit_phase_a/selected_leaves` on the pod and verified their
digests there. Useful staging, and not durability: the artifact specs do not name
`selected_leaves`, `selected_leaf_durability.json` was not a fetched report, and
`fetch_finalists` returns immediately when `stage2_passed` is false. A Stage-2
failure could therefore still delete all five with the pod — the exact class the
closure exists to prevent, with every other check green.

These pin the transfer half: the report comes home, the fetch is NOT gated on
Stage 2, the leaves land in the dev-box checkpoint store rather than the archive,
they are re-identified from local bytes after transfer, and teardown fails closed
when Stage 1 produced leaves that are not verified off-pod.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


def load_launcher():
    spec = importlib.util.spec_from_file_location(
        "phase_a_launch_leaf", REPO / "scripts/pod/autoinit_phase_a_launch.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phase_a_launch_leaf"] = mod
    spec.loader.exec_module(mod)
    return mod


class Ctx:
    """The slice of `SessionContext` these callables touch."""

    def __init__(self, scr, *, stage2_passed=False, ckpt_store=None):
        self.scr = Path(scr)
        self.stage2_passed = stage2_passed
        self.host = "pod.invalid"
        self.scp = ("scp", "-o", "StrictHostKeyChecking=no")
        self.evidence: dict = {}
        self.say = lambda m: None
        self.target = None

        class A:
            pass
        self.args = A()
        self.args.ckpt_store = str(ckpt_store or (self.scr / "store_out"))
        self.args.ckpt_fetch_limit_min = 1
        self.args.fetch_finalists = True
        self.args.stage_leaves_to_relay = False


def write_report(scr: Path, state_ids) -> None:
    store = scr / "store"
    store.mkdir(parents=True, exist_ok=True)
    (store / "selected_leaf_durability.json").write_text(json.dumps({
        "schema": "aadistill.autoinit.selected_leaf_durability/v1",
        "n_leaves": len(state_ids),
        "leaves": [{"state_id": s, "artifact_digest": f"digest-{s}",
                    "single_shard_sha256": f"shard-{s}", "total_bytes": 1,
                    "arch_signature": "arch", "num_parameters": 1,
                    "weights_digest": f"w-{s}"} for s in state_ids]}))


# --- the report itself ------------------------------------------------------

def test_the_durability_report_is_a_fetched_report():
    """It must come home BEFORE products: the leaf fetch reads it to learn which
    five leaves exist and what they must hash to."""
    mod = load_launcher()
    args = mod.build_parser().parse_args(
        ["--scr", "/tmp/x", "--session-commit", "0" * 40, "--bundle", "b"])
    art = mod.spec(args).artifacts
    assert mod.SELECTED_LEAF_REPORT in art.report_names
    assert art.report_names.index(mod.SELECTED_LEAF_REPORT) < \
        art.report_names.index("phase_a_result.json")


def test_the_leaves_are_not_in_the_artifact_tarball():
    """The collector keeps the downloaded archive AND its extracted copy while
    verifying, so five incompressible 1.11 GiB safetensors would roughly double
    the temporary local footprint on a box already short of disk."""
    for spec_name in ("configs/autoinit/phase_a_artifacts.json",
                      "configs/autoinit/phase_a_artifacts_failed.json"):
        text = (REPO / spec_name).read_text()
        assert "selected_leaves" not in text, (
            f"{spec_name} names selected_leaves; the leaves must travel by the "
            "product transfer path, not inside the archive")


# --- the fetch --------------------------------------------------------------

def test_the_stage1_fetch_is_not_gated_on_stage_2(tmp_path, monkeypatch):
    """The attempt-11 defect. `fetch_finalists` returns early when stage 2 did
    not pass, and stage 2 failed six seconds after stage 1 succeeded."""
    mod = load_launcher()
    write_report(tmp_path, ["aaa", "bbb"])
    ctx = Ctx(tmp_path, stage2_passed=False)

    calls = []

    def fake_scp(argv, **kw):
        calls.append(argv)
        dest = Path(argv[-1]); dest.mkdir(parents=True, exist_ok=True)
        class R:
            returncode = 0
            stderr = b""
        return R()

    monkeypatch.setattr(mod.subprocess, "run", fake_scp)
    monkeypatch.setattr(mod, "verify_transferred_leaf", None, raising=False)
    out = mod.fetch_selected_leaves(ctx)

    assert len(calls) == 2, "the stage-1 leaves were not fetched"
    assert len(out) == 2
    # And they went to the checkpoint store, not the archive.
    for argv in calls:
        assert str(Path(ctx.args.ckpt_store) / "phase_a") in argv[-1]


def test_no_report_means_stage_1_never_staged_anything(tmp_path):
    """No leaves owed, nothing fetched, and that is not a failure."""
    mod = load_launcher()
    ctx = Ctx(tmp_path)
    (tmp_path / "store").mkdir(parents=True, exist_ok=True)
    assert mod.fetch_selected_leaves(ctx) == []
    ok, why = mod.selected_leaves_secured(ctx, [])
    assert ok and "did not stage" in why


# --- the teardown gate ------------------------------------------------------

def test_teardown_is_refused_when_a_staged_leaf_is_not_off_pod(tmp_path):
    mod = load_launcher()
    write_report(tmp_path, ["aaa", "bbb", "ccc"])
    ctx = Ctx(tmp_path)
    fetched = [{"artifact": "stage1_selected_leaf", "state_id": "aaa",
                "rc": 0, "matched": True}]
    ok, why = mod.selected_leaves_secured(ctx, fetched)
    assert not ok
    assert "bbb" in why and "ccc" in why


def test_teardown_is_refused_when_a_leaf_arrived_but_did_not_verify(tmp_path):
    """Transferred is not verified. A truncated shard scp's with rc=0."""
    mod = load_launcher()
    write_report(tmp_path, ["aaa"])
    ctx = Ctx(tmp_path)
    ok, _ = mod.selected_leaves_secured(
        ctx, [{"artifact": "stage1_selected_leaf", "state_id": "aaa",
               "rc": 0, "matched": False}])
    assert not ok


def test_teardown_is_allowed_when_every_staged_leaf_is_verified(tmp_path):
    mod = load_launcher()
    write_report(tmp_path, ["aaa", "bbb"])
    ctx = Ctx(tmp_path)
    ok, why = mod.selected_leaves_secured(
        ctx, [{"artifact": "stage1_selected_leaf", "state_id": s,
               "rc": 0, "matched": True} for s in ("aaa", "bbb")])
    assert ok and "verified off-pod" in why


def test_the_gate_check_is_wired_and_fails_closed_when_absent():
    """`required_products_secured` is in GATE_ORDER, and an unreported check
    counts as False — which is what makes it fail closed."""
    from aadistill.infrastructure.artifact_gate import GATE_ORDER, evaluate_teardown

    assert "required_products_secured" in GATE_ORDER
    passing = {name: True for name in GATE_ORDER}
    assert evaluate_teardown(passing).allowed

    del passing["required_products_secured"]
    assert not evaluate_teardown(passing).allowed, (
        "an unreported products-secured check was treated as passed")


def test_the_phase_a_spec_supplies_the_secured_callable():
    mod = load_launcher()
    args = mod.build_parser().parse_args(
        ["--scr", "/tmp/x", "--session-commit", "0" * 40, "--bundle", "b"])
    art = mod.spec(args).artifacts
    assert art.products_secured is mod.selected_leaves_secured


def test_a_session_that_owes_no_products_still_answers_the_gate():
    """The default must be an explicit "owes none", not a skipped check."""
    sys.path.insert(0, str(REPO / "scripts" / "pod"))
    from aadistill.infrastructure.session import ArtifactPolicy

    art = ArtifactPolicy(audit_dirname="x", evidence_filename="e.json",
                         archive_basename="a.tar.gz", spec_success="s",
                         spec_failed="f")
    ok, why = art.products_secured(None, [])
    assert ok and why


# --- the $0 capacity gate ---------------------------------------------------

def test_the_capacity_gate_is_a_precheck_and_measures_the_dev_box(tmp_path):
    """The driver's headroom check runs on the POD and proves only that the pod
    can stage them. This one asks whether the destination can hold them, before
    a pod exists."""
    mod = load_launcher()
    args = mod.build_parser().parse_args(
        ["--scr", "/tmp/x", "--session-commit", "0" * 40, "--bundle", "b"])
    assert mod.ckpt_store_capacity_gate in mod.spec(args).precheck

    ctx = Ctx(tmp_path, ckpt_store=tmp_path / "store")
    ok, why = mod.ckpt_store_capacity_gate(ctx)
    ev = ctx.evidence["precheck"]["ckpt_store"]
    assert ev["leaf_bytes"] == mod.SELECTED_LEAF_BYTES
    assert ev["required_bytes"] > ev["leaf_bytes"], "no working-room margin"
    assert isinstance(ok, bool) and why


def test_the_capacity_gate_refuses_when_the_destination_is_short(
        tmp_path, monkeypatch):
    mod = load_launcher()
    ctx = Ctx(tmp_path, ckpt_store=tmp_path / "store")

    class Usage:
        free = 1 << 20
    monkeypatch.setattr(mod.shutil, "disk_usage", lambda p: Usage)
    ok, why = mod.ckpt_store_capacity_gate(ctx)
    assert not ok
    assert "free" in why and "attempt-11" in why


def test_the_leaf_size_is_measured_not_guessed():
    """5 x 1.110 GiB, from attempt 11's own search record."""
    mod = load_launcher()
    assert mod.SELECTED_LEAF_BYTES == 5 * 1_192_099_840
    assert abs(mod.SELECTED_LEAF_BYTES / 2**30 - 5.55) < 0.01


def test_the_runner_reports_every_check_the_gate_orders():
    """`GATE_ORDER` ⊆ the keys the runner actually puts in its teardown state.

    An unreported check counts as False, so adding one to `GATE_ORDER` without
    reporting it would block every teardown — and *removing* the report while
    leaving the check would do the same silently. Asserted as a set relation so
    it covers the next check as well as this one.
    """
    import ast

    from aadistill.infrastructure.artifact_gate import GATE_ORDER

    src = (REPO / "src/aadistill/infrastructure/session_runner.py").read_text()
    tree = ast.parse(src)
    reported: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = {k.value for k in node.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)}
        # The teardown state is the dict that carries the gate's own vocabulary.
        if "artifact_manifest_created" in keys:
            reported |= keys
    missing = sorted(set(GATE_ORDER) - reported)
    assert not missing, (
        f"the runner never reports {missing}, so the gate would read them as "
        "False and refuse every teardown")
