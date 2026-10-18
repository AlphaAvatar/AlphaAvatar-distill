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
