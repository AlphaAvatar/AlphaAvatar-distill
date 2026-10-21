"""aadistill.evaluation.eval_registry — identity, protection and per-capability reports.

Synthetic registries only: the mechanism must refuse every way a set's bytes can
drift from its authorized identity, and must never turn an unsupported, withheld
or unrun unit into a zero.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aadistill.evaluation import eval_registry as ER
from aadistill.infrastructure.manifest import sha256_file


def _make(tmp: Path, *, status="candidate", authorization="maintainer decision X") -> Path:
    root = tmp / "es"
    (root / "unit").mkdir(parents=True)
    (root / "unit" / "items.jsonl").write_text('{"id": "a"}\n{"id": "b"}\n')
    (root / "README.md").write_text("docs\n")
    (root / "audit").mkdir()
    (root / "audit" / "notes.json").write_text("{}\n")
    item = root / "unit" / "items.jsonl"
    manifest = {"schema": ER.MANIFEST_SCHEMA, "eval_set_id": "es",
                "files": {"unit/items.jsonl": {"sha256": sha256_file(item),
                                               "bytes": item.stat().st_size}},
                "documentation": ["README.md", "audit/*"],
                "withheld": {"unit/secret.jsonl": {"sha256": "0" * 64, "bytes": 1}}}
    (root / "manifest.json").write_text(json.dumps(manifest))
    reg = {"schema": ER.REGISTRY_SCHEMA, "default_formal_measurement": "es",
           "eval_sets": [{"eval_set_id": "es", "root": "es", "manifest": "es/manifest.json",
                          "status": status,
                          "identity_history": [{"manifest_sha256": sha256_file(root / "manifest.json"),
                                                "authorization": authorization}]}]}
    path = tmp / "registry.json"
    path.write_text(json.dumps(reg))
    return path


def test_intact_set_verifies_and_loads(tmp_path):
    reg = _make(tmp_path)
    assert ER.verify(reg, "es") == []
    assert ER.require_intact(reg, "es")["eval_set_id"] == "es"


@pytest.mark.parametrize("mutate, needle", [
    (lambda r: (r / "unit" / "items.jsonl").write_text('{"id": "a"}\n{"id": "c"}\n'), "sha256 differs"),
    (lambda r: (r / "unit" / "items.jsonl").write_text('{"id": "a"}\n'), "bytes"),
    (lambda r: (r / "unit" / "items.jsonl").unlink(), "missing content file"),
    (lambda r: (r / "unit" / "extra.jsonl").write_text("{}\n"), "unlisted file unit/extra.jsonl"),
    (lambda r: (r / "unit" / "secret.jsonl").write_text("x"), "declared withheld but present"),
    (lambda r: (r / "manifest.json").write_text((r / "manifest.json").read_text() + " "),
     "!= authorized"),
])
def test_every_drift_is_refused(tmp_path, mutate, needle):
    reg = _make(tmp_path)
    mutate(tmp_path / "es")
    problems = ER.verify(reg, "es")
    assert any(needle in p for p in problems), problems
    with pytest.raises(ER.EvalSetIntegrityError):
        ER.require_intact(reg, "es")


def test_documentation_may_change_without_a_new_identity(tmp_path):
    reg = _make(tmp_path)
    (tmp_path / "es" / "README.md").write_text("rewritten\n")
    (tmp_path / "es" / "audit" / "later_review.json").write_text("{}\n")
    assert ER.verify(reg, "es") == []


def test_protected_identity_must_cite_an_authorization(tmp_path):
    assert any("cites no authorization" in p
               for p in ER.verify(_make(tmp_path, authorization=" "), "es"))
    draft = tmp_path / "d"
    draft.mkdir()
    assert ER.verify(_make(draft, status="draft", authorization=""), "es") == []


def test_withheld_copy_is_checked_by_identity(tmp_path):
    f = tmp_path / "copy.jsonl"
    f.write_text("abc")
    ident = {"sha256": sha256_file(f), "bytes": 3}
    assert ER.withheld_intact(f, ident)
    f.write_text("abd")
    assert not ER.withheld_intact(f, ident)
    assert not ER.withheld_intact(tmp_path / "absent", ident)


def test_capability_table_never_pools_or_zeroes():
    declared = {"cap.a": ["u1", "u2"], "cap.b": ["u3"], "cap.c": ["u4"], "cap.d": ["u5"]}
    outcomes = {"u1": {"status": "measured", "rate": 0.5},
                "u2": {"status": "unsupported", "reason": "no tool interface"},
                "u3": {"status": "measured", "rate": 0.0},
                "u4": {"status": "withheld"}}
    t = ER.capability_table(declared, outcomes)
    assert t["cap.a"]["status"] == "partial"
    assert "rate" not in t["cap.a"]                      # no pooled number exists
    assert t["cap.a"]["units"]["u2"] == {"status": "unsupported", "reason": "no tool interface"}
    assert t["cap.b"]["status"] == "measured"
    assert t["cap.c"]["status"] == "withheld"
    assert t["cap.d"]["status"] == "not_run"
    with pytest.raises(ValueError):
        ER.capability_table({"x": ["u"]}, {"u": {"status": "zero"}})
