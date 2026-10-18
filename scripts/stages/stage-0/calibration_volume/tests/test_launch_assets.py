"""Every local asset a search session ships lands where its consumer reads it.

Setup copies an asset to `<install_to>/<dest_name>`; the driver resolves the
state-eval suite and every profile's `items_path` by repository path. A name
that differs from the directory the consumer reads is a refusal on a billing
pod, so it is checked here at $0.
"""
from __future__ import annotations

import json
from pathlib import Path

from stages.calibration_volume import governance as G
from stages.calibration_volume import launch

REPO = Path(__file__).resolve().parents[5]
PLANS = sorted(p.name[: -len("_session.json")] for p in
               (REPO / "configs/stages/stage-0/calibration_volume").glob("*_session.json")
               if json.loads(p.read_text()).get("session_kind") == "search")


def test_search_assets_land_on_the_paths_their_consumers_read():
    assert PLANS
    for name in PLANS:
        plan = G.plan_doc(name)
        landed = {str(Path(a.install_to) / a.dest_name) for a in launch.search_assets(plan)}
        assert plan["profile_items_dir"] in landed, (name, landed)
        doc = json.loads((REPO / plan["profiles_doc"]).read_text())
        for t in plan["treatments"]:
            for pid in doc["treatments"][t["treatment"]]["profiles"]:
                prof = next(p for p in doc["profiles"]
                            if f"{p['profile_id']}@v{p['version']}" == pid)
                assert str(Path(prof["items_path"]).parent) in landed, prof["items_path"]
        from stages.phase_d1 import d1_session as D1S
        assert str(D1S.STATE_EVAL_ROOT) in landed


BATTERY_PLANS = sorted(p.name[: -len("_session.json")] for p in
                       (REPO / "configs/stages/stage-0/calibration_volume").glob("*_session.json")
                       if json.loads(p.read_text()).get("session_kind") == "battery")


def test_every_battery_run_resolves_to_a_pinned_model_and_a_hashed_task_file():
    """What `battery_driver` resolves on the pod, checked at $0: each run's model is
    the teacher, a pinned `reference:` revision or a `student:` whose relay files are
    staged into its directory, and each task file still hashes to the plan."""
    import hashlib
    assert BATTERY_PLANS
    for name in BATTERY_PLANS:
        plan = G.plan_doc(name)
        staged = {(r.path, r.dest) for r in launch.battery_relay_inputs(name)}
        refs = {r["label"]: r for r in plan.get("references", [])}
        students = {s["label"]: s for s in plan.get("students", [])}
        for role, t in plan["tasks"].items():
            path = REPO / t["path"]
            if not path.is_file():          # a historical plan's spelling; not re-launchable
                continue
            assert hashlib.sha256(path.read_bytes()).hexdigest() == t["sha256"], (name, role)
        for run in plan["runs"]:
            model = run["model"]
            if model == "teacher":
                assert len(plan["teacher"]["revision"]) == 40
            elif model.startswith("reference:"):
                ref = refs[model.split(":", 1)[1]]
                assert len(ref["revision"]) == 40 and run.get("trained_context") is None, ref
            else:
                s = students[model.split(":", 1)[1]]
                assert {(f["path"], s["dir"]) for f in s["relay_files"]} <= staged, s["label"]
                assert any(f["path"].endswith("model.safetensors") for f in s["relay_files"])


def test_the_resolution_plan_reads_the_registered_fluid_set():
    from shared import eval_sets as ES
    plan = G.plan_doc("resolution")
    unit = ES.unit_path("eval-set-1", "fluid_reasoning_720", "dev_v1")
    assert REPO / plan["tasks"]["dev"]["path"] == unit
    ident = ES.manifest("eval-set-1")["files"]["fluid_reasoning_720/dev_v1.jsonl"]
    assert plan["tasks"]["dev"]["sha256"] == ident["sha256"]


def test_a_launch_whose_bundle_is_not_on_the_relay_is_refused_at_zero_cost():
    """Resolution attempt 1 paid for a 404 on an unstaged bundle (C1 attempt 1's
    failure). The gate refuses a non-canonical name and a missing object."""
    import argparse

    class Relay:
        def __init__(self, present):
            self.present = present

        def get_paths_info(self, repo, paths, repo_type):
            return [type("Info", (), {"size": 1})()] if self.present else []

    commit = "990ec4068927a23e86bbbd8355fd149eed8805e1"
    good = argparse.Namespace(bundle="aad_autoinit_990ec406.bundle", session_commit=commit)
    assert launch.bundle_on_relay(good, api=Relay(True))[0]
    ok, why = launch.bundle_on_relay(good, api=Relay(False))
    assert not ok and "stage_bundle" in why
    alias = argparse.Namespace(bundle="resolution.bundle", session_commit=commit)
    assert not launch.bundle_on_relay(alias, api=Relay(True))[0]
