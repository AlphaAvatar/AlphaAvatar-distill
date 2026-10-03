"""The 2026-09-10 cuda-stage-f campaign's booked spend, as recorded.

Split from `tests/validation/test_cuda_engineering_launch.py` by the 2026-10-03
convergence round. The engineering launcher's ACCOUNTING -- that a campaign's
prior spend is carried in and subtracted from both the hard and soft ceilings, and
that an exhausted campaign refuses at construction rather than at create time --
stays in core and is proved there against a campaign built under `tmp_path`.

This asks the historical question: does the launcher, pointed at C1's real
cuda-stage-f ledger, start from the dollars that campaign actually booked? It
reads an old campaign's bytes, so it belongs with the experiment that owns them.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

ENTRY = REPO / "scripts/validation/cuda_engineering_launch.py"
AUTH = REPO / "logs/stages/stage-1/phase_c1/validations/cuda-stage-f/v1/authorization.json"
CAMPAIGN = AUTH.parent / "campaign.json"


@pytest.mark.skipif(not CAMPAIGN.is_file(),
                    reason="the cuda-stage-f campaign record is not staged here")
def test_the_launcher_starts_from_what_earlier_subruns_booked(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("cuda_eng_launch", ENTRY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cuda_eng_launch"] = mod
    spec.loader.exec_module(mod)

    monkeypatch.setattr(mod, "read_api_key", lambda p: "test-key")
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    fake_cli = tmp_path / "runpodctl"
    fake_cli.write_text("#!/bin/sh\nexit 0\n")
    fake_cli.chmod(0o755)
    monkeypatch.setattr(mod, "provider_cli_candidates", lambda: (str(fake_cli),))
    monkeypatch.setattr(mod, "RunPodProvider", lambda key: types.SimpleNamespace(
        _gql=lambda q: {"data": {}}))

    booked = json.loads(CAMPAIGN.read_text())["booked_usd"]
    #: The launcher takes a namespace, and `args()` in the core file builds one
    #: from the module's own defaults. Only the two fields this test cares about
    #: are overridden; everything else is stage F's declared profile.
    eng = mod.Engineering(types.SimpleNamespace(
        run_id="t-run", scr=str(tmp_path / "scr"), execution_sha="deadbeef",
        image="img", remote_python="python", disk_gb=20, startup_limit_min=1.0,
        run_limit_min=1.0, min_minutes=25.0, dry_run=False,
        authorization=str(AUTH),
        experiment_id=mod.DEFAULT_EXPERIMENT_ID, stage_id=mod.DEFAULT_STAGE_ID,
        check="scripts/validation/cuda_engineering_check.py",
        check_config="configs/validation/cuda_engineering.json",
        ship=[], gpu=[]))
    assert eng.booked_usd == booked > 0, "no prior spend was carried in"
    assert eng.remaining_total == pytest.approx(eng.hard_usd - booked)
    assert eng.remaining_soft == pytest.approx(eng.soft_usd - booked)
