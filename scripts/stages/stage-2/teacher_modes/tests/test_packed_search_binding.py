"""The packed Full Search's binding (maintainer 2026-11-05, 'Packed Full Search 正式授权'): a formal input set refuses a
missing freeze record and any content, protocol or calibration identity that differs from it; the kind's cap is
cumulative over every attempt; the search attaches only its granted volume, never a protected one; and the closeout
counts the Top-1's off-volume copy only when the relay itself holds it."""
from __future__ import annotations

import json
from types import SimpleNamespace

from stages.teacher_modes import full_search as FS
from stages.teacher_modes import packed_full_search as PFS
from stages.teacher_modes import token_matched_phase as TM

CONTENT, PROTOCOL = "c" * 64, "p" * 32


def inputs_fixture(tmp_path, *, record=True, content=CONTENT, protocol=PROTOCOL):
    items = tmp_path / "calib.items.jsonl"
    items.write_text('{"a": 1}\n')
    prof = tmp_path / "profiles.json"
    prof.write_text(json.dumps({"profiles": [{"profile_id": "calib.x", "version": 2, "items_path": str(items),
                                              "items_file_sha256": FS.sha(items)}]}))
    rec = tmp_path / "freeze.json"
    if record:
        rec.write_text(json.dumps({"suite_identity": {"content_sha256": content, "measurement_protocol_id": protocol}}))
    d = {"full_search": {"formal_inputs": "v3", "inputs": {"v3": {
        "status": "READY", "profiles_doc": str(prof), "suite_freeze_record": str(rec),
        "suite_content_sha256": CONTENT, "measurement_protocol_id": PROTOCOL}}}}
    session = SimpleNamespace(suite_manifest={"content_sha256": CONTENT},
                              evaluator=SimpleNamespace(measurement_protocol_id=PROTOCOL))
    return d, session, items


def test_a_formal_input_set_is_its_frozen_identity_or_refuses(tmp_path):
    d, session, items = inputs_fixture(tmp_path)
    assert PFS.input_problems(session, "v3", d) == []
    #: a missing freeze record refuses outright (nothing is skipped)
    (tmp_path / "a").mkdir()
    d2, s2, _ = inputs_fixture(tmp_path / "a", record=False)
    assert any("no freeze record" in b for b in PFS.input_problems(s2, "v3", d2))
    #: the record names another protocol
    (tmp_path / "b").mkdir()
    d3, s3, _ = inputs_fixture(tmp_path / "b", protocol="q" * 32)
    assert any("measurement_protocol_id" in b for b in PFS.input_problems(s3, "v3", d3))
    #: the evaluator built here, the suite's manifest, a calibration items file
    session.evaluator.measurement_protocol_id = "x" * 32
    session.suite_manifest["content_sha256"] = "y" * 64
    items.write_text('{"a": 2}\n')
    bad = PFS.input_problems(session, "v3", d)
    assert len(bad) == 3 and any("evaluator" in b for b in bad) and any("manifest" in b for b in bad) \
        and any("calib.x@v2" in b for b in bad)


def test_the_design_binds_the_formal_search_to_v3():
    d = PFS.design()
    v3 = d["full_search"]["inputs"]["v3"]
    assert PFS.formal_inputs(d) == "v3" and v3["status"] == "READY" and v3["critical_groups"] == "v2"
    rec = json.loads((TM.REPO_ROOT / v3["suite_freeze_record"]).read_text())["suite_identity"]
    assert (rec["content_sha256"], rec["measurement_protocol_id"]) == (v3["suite_content_sha256"],
                                                                       v3["measurement_protocol_id"])
    assert v3["suite_freeze_record"] in TM.KIND_RECORDS["packed_search"]
    assert "suite_v3" in TM.DRIVERS["packed_search"][2]


def grant_root(tmp_path, spent_usd: float | None):
    g = json.loads((TM.REPO_ROOT / TM.STUDY.grant).read_text())
    (tmp_path / TM.STUDY.grant).parent.mkdir(parents=True)
    (tmp_path / TM.STUDY.grant).write_text(json.dumps(g))
    if spent_usd is not None:
        run = tmp_path / TM.STUDY.logs / "runs" / "a_packed_search_1"
        (run / "governance").mkdir(parents=True)
        (run / "closeout").mkdir()
        (run / "governance/authorization.json").write_text(json.dumps(
            {"plan_id": TM.STUDY.plan_id, "stage_conditions": {"kind": "packed_search"}}))
        (run / "closeout/outcome.json").write_text(json.dumps({"money": {"all_in_usd": spent_usd}}))
    return tmp_path


def test_the_search_cap_is_cumulative_over_attempts(tmp_path):
    assert TM.kind_cap_problems("packed_search", 55.72, grant_root(tmp_path / "a", None)) == []
    root = grant_root(tmp_path / "b", 20.0)
    assert TM.kind_cap_problems("packed_search", 35.72, root) == []
    bad = TM.kind_cap_problems("packed_search", 35.73, root)
    assert bad and "cumulative" in bad[0] and "35.72" in bad[0]


def test_the_search_attaches_only_its_granted_volume():
    g = json.loads((TM.REPO_ROOT / TM.STUDY.grant).read_text())["packed_search"]
    ctx = lambda vid: SimpleNamespace(args=SimpleNamespace(network_volume_id=vid))  # noqa: E731
    assert TM.the_search_volume_is_the_granted_one(ctx(g["network_volume"]["id"]))[0]
    assert not TM.the_search_volume_is_the_granted_one(ctx("4rhubprswo"))[0]
    assert not TM.the_search_volume_is_the_granted_one(ctx("someothervol"))[0]
    assert "4rhubprswo" in g["protected_volumes"]


def test_a_copy_the_relay_does_not_hold_is_not_secured():
    ctx = SimpleNamespace(evidence={})
    got = TM.top1_off_volume(ctx, {"top1": {"files_sha256": {"model.safetensors": "a" * 64}},
                                   "off_volume_copy": {"verified": False, "error": "quota"}})
    assert got == {"top1_off_volume/model.safetensors": False} and ctx.evidence["top1_off_volume"]["error"] == "quota"
