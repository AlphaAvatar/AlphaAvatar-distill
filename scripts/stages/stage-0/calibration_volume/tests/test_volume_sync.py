"""The search mirror starts before the search creates its directory. Dose d2
(dose_d2_20261019_135821) began with nothing on the volume to pull, so the
mirror's first pass found no source, exited 1, and the search ran for hours
with every state on the pod's local disk only.
"""
from __future__ import annotations

from stages.calibration_volume import volume_sync as VS


def test_a_mirror_started_before_its_source_exists_waits_for_it(tmp_path, monkeypatch):
    src, dst, stop = tmp_path / "search", tmp_path / "volume", tmp_path / "mirror.stop"

    def next_pass(_seconds):
        if not src.exists():
            (src / "states" / "s0").mkdir(parents=True)
            (src / "states" / "s0" / "model.safetensors").write_bytes(b"x" * 10)
        else:
            stop.write_text("stop\n")

    monkeypatch.setattr(VS.time, "sleep", next_pass)
    assert VS.main(["push", "--src", str(src), "--dst", str(dst),
                    "--loop", "120", "--until", str(stop)]) == 0
    assert (dst / "states" / "s0" / "model.safetensors").read_bytes() == b"x" * 10


def test_a_single_pass_without_a_source_still_reports_it(tmp_path):
    missing = tmp_path / "missing"
    assert VS.main(["pull", "--src", str(missing), "--dst", str(tmp_path / "d")]) == 0
    assert VS.main(["push", "--src", str(missing), "--dst", str(tmp_path / "d")]) == 1
