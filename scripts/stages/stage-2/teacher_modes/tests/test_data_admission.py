"""Data selection, proportion validation, admission / freeze and packing are separate (maintainer 2026-11-04):

1. the proportion validation runs alone and packs nothing;
2. the packing runs on any valid documents and never calls the validation;
3. data outside its proportions can be packed but is never admitted, and the workflow refuses before packing;
4. frozen data keeps its counted size and proportions through the packing, fragments and all;
5. a changed document, an edited manifest or another packing spec is refused;
6. the quotas are the registered owners' numbers, and the validation itself is the generic core function.

The designs are the real files with their numbers scaled down (`scaled`); the structure and the tolerance are kept."""
from __future__ import annotations

import json

import pytest

from aadistill.data.token_documents import TokenDocument, pack_sequential
from stages.teacher_modes import packed_packs as PP
from stages.teacher_modes import packed_training_plan as PTP

EOS = 151645
PER, FLUID, UNIT = 1000, 6000, 250          #: a type's target, fluid's, one example's supervised tokens
TYPES = ("supporting.math_new", "supporting.math_gsm8k", "supporting.code_algorithmic", "rag_evidence",
         "multihop_qa")


@pytest.fixture()
def scaled(tmp_path, monkeypatch):
    """The real designs with A1-6cat's numbers scaled to a six-type target of PER and a fluid target of FLUID."""
    tm = json.loads(PP.TOKEN_MATCHED.read_text())
    tm["scale"].update(per_mode_total=FLUID + 6 * PER, categories={"base_counterpart": 6 * PER, "fluid": FLUID})
    six = json.loads(PP.SIX.read_text())
    six["base_category"]["per_type"] = PER
    design = json.loads(PP.DESIGN.read_text())
    design["stage3_packing"]["sequence_length"] = 700                      #: fragments for sure
    for name, doc in (("tm.json", tm), ("six.json", six), ("design.json", design)):
        (tmp_path / name).write_text(json.dumps(doc))
    monkeypatch.setattr(PP, "TOKEN_MATCHED", tmp_path / "tm.json")
    monkeypatch.setattr(PP, "SIX", tmp_path / "six.json")
    monkeypatch.setattr(PP, "DESIGN", tmp_path / "design.json")
    return tmp_path


def doc(doc_id, n, meta=None):
    return TokenDocument(doc_id, (5, 6, 7), tuple([9] * (n - 1) + [EOS]), meta or {})


def selection(sizes=None, tool=(("c1", 500, 2), ("c2", 500, 2)), approvals=()):
    """A1-6cat's selection: ``sizes`` {data type: [supervised tokens per example]}; ``tool``: per tool cell
    (category, target, episodes of UNIT)."""
    sizes = sizes or {**{t: [UNIT] * 4 for t in TYPES}, "primary.x": [500] * 12}
    train, dtype, unit_of, quota, k = [], {}, {}, {}, 0
    for t, ns in sizes.items():
        for n in ns:
            d = doc(f"{t}-{k}|H-T|t0", n)
            train.append(d)
            dtype[d.doc_id], unit_of[d.doc_id], k = t, d.doc_id.split("|")[0], k + 1
    for cat, target, episodes in tool:
        for j in range(episodes):
            d = doc(f"tc1-{cat}-{j}|H-T|t0", UNIT, {"category": cat, "task_id": f"tc1-{cat}-{j}"})
            train.append(d)
            dtype[d.doc_id], unit_of[d.doc_id] = PP.TOOL_TYPE, f"tc1-{cat}-{j}"
        got = UNIT * episodes
        quota[f"training/{cat}"] = {"split": "training", "category": cat, "unit": "supervised_tokens",
                                    "target": float(target), "selected": got, "episodes": episodes,
                                    "status": "REACHED" if got >= target else "SHORTFALL"}
    val = [doc("val-0|H-T|t0", 40)]
    dtype[val[0].doc_id], unit_of[val[0].doc_id] = TYPES[0], "val-0"
    return {"arm": "A1-6cat", "mode": "H-T", "teacher": "hybrid", "train": train, "validation": val,
            "data_type": dtype, "unit_of": unit_of, "missing_tasks_read_from_records": [], "sources": {"toy": 1},
            "tool": {"quota": {"cells": quota, "rule": "toy"}, "train": 4, "validation": 0,
                     "approved_shortfalls": list(approvals)}}


def frozen(scaled, sel=None):
    sel = sel or selection()
    adm = PP.admit("A1-6cat", sel, PP.proportions("A1-6cat", sel), scaled / "out")
    return sel, adm


def refuse(monkeypatch, *names):
    def boom(*a, **k):
        raise AssertionError("must not be called")
    for mod, name in names:
        monkeypatch.setattr(mod, name, boom)


class TestSeparation:
    def test_the_proportion_validation_runs_alone_and_packs_nothing(self, scaled, monkeypatch):
        from aadistill.data import token_documents as TD
        refuse(monkeypatch, (TD, "pack_sequential"), (TD, "write_pack"), (PP, "pack"))
        sel = selection()
        rep = PP.proportions("A1-6cat", sel)
        assert rep["passed"], rep["failures"]
        assert rep["counted"] == "unique supervised tokens before packing"
        row = rep["quotas"]["gsm8k"]
        assert (row["target"], row["amount"], row["status"]) == (PER, 4 * UNIT, "OK")
        assert rep["quotas"]["tool_calling"]["bounds"] == [PER - 2 * UNIT, PER + 2 * UNIT]   #: one unit per cell
        assert rep["shares"]["fluid_of_total"]["ok"] and PP.admission_verdict("A1-6cat", sel, rep)["admitted"]

    def test_the_packing_runs_on_any_valid_documents_without_the_validation(self, scaled, monkeypatch):
        sel, adm = frozen(scaled)
        from aadistill.data import mixture
        refuse(monkeypatch, (mixture, "validate_proportions"), (PP, "proportions"), (PP, "registered_quotas"),
               (PP, "admission_verdict"), (PP, "type_counts"))
        odd = [doc(f"x{k}", 50 + 37 * k) for k in range(30)]                 #: any documents, no proportions at all
        assert pack_sequential(odd, 256, PP.PAD_ID, order_seed=1).stats()["documents"] == 30
        rec = PP.pack(sel["train"], sel["validation"], adm["path"], scaled / "out" / "A1-6cat")
        assert rec["n_blocks"] > 0 and rec["data_admission"]["manifest"] == PP.MANIFEST

    def test_data_outside_its_proportions_packs_but_is_never_admitted(self, scaled, monkeypatch):
        sizes = {**{t: [UNIT] * 4 for t in TYPES}, "supporting.math_gsm8k": [UNIT] * 2, "primary.x": [500] * 12}
        sel = selection(sizes)
        assert pack_sequential(sel["train"], 700, PP.PAD_ID, order_seed=1).stats()["sequences"] > 0
        rep = PP.proportions("A1-6cat", sel)
        assert not rep["passed"] and rep["quotas"]["gsm8k"]["status"] == "SHORTFALL"
        with pytest.raises(PP.AdmissionError, match="gsm8k: SHORTFALL") as e:
            PP.admit("A1-6cat", sel, rep, scaled / "out")
        assert not e.value.verdict["admitted"] and not (scaled / "out").exists()
        monkeypatch.setattr(PP, "select", lambda *a, **k: sel)
        with pytest.raises(PP.AdmissionError):
            PP.build("A1-6cat", tool_docs=None, out=scaled / "out")
        assert not (scaled / "out").exists()                                  #: refused before packing

    def test_a_short_tool_cell_is_refused_even_inside_every_proportion(self, scaled):
        """One tool category over its cell, another under: the tool type sits at its target and every proportion
        holds, the short cell does not -- refused, unless the maintainer approved that cell's SHORTFALL."""
        tool = (("c1", 490, 2), ("c2", 510, 2))
        sizes = None
        sel = selection(sizes, tool)
        rep = PP.proportions("A1-6cat", sel)
        assert rep["passed"], rep["failures"]
        with pytest.raises(PP.AdmissionError, match="tool cell training/c2: SHORTFALL"):
            PP.admit("A1-6cat", sel, rep, scaled / "out")
        appr = {"split": "training", "mode": "H-T", "category": "c2", "approved_by": "maintainer (test)"}
        adm = PP.admit("A1-6cat", selection(sizes, tool, [appr]), rep, scaled / "out")
        assert adm["manifest"]["approved_exceptions"] == [{**appr, "cell": "training/c2"}]

    def test_a_proportion_failure_is_admitted_only_by_a_recorded_exception(self, scaled):
        sizes = {**{t: [UNIT] * 4 for t in TYPES}, "supporting.math_gsm8k": [UNIT] * 2, "primary.x": [500] * 12}
        sel = selection(sizes)
        rep = PP.proportions("A1-6cat", sel)
        design = json.loads(PP.DESIGN.read_text())
        design["data_admission"]["approved_exceptions"] = [
            {"arm": "A1-6cat", "quota": q, "approved_by": "maintainer (test)"} for q in ("gsm8k", "supporting", "total")]
        PP.DESIGN.write_text(json.dumps(design))
        with pytest.raises(PP.AdmissionError, match="fluid_of_total"):     #: its share moved too: not approved
            PP.admit("A1-6cat", sel, rep, scaled / "out")
        design["data_admission"]["approved_exceptions"].append(
            {"arm": "A1-6cat", "quota": "fluid_of_total", "approved_by": "maintainer (test)"})
        PP.DESIGN.write_text(json.dumps(design))
        adm = PP.admit("A1-6cat", sel, rep, scaled / "out")
        assert {e["quota"] for e in adm["manifest"]["approved_exceptions"]} == {"gsm8k", "supporting", "total",
                                                                                "fluid_of_total"}


class TestFrozenThroughPacking:
    def test_the_packing_keeps_the_counted_size_and_proportions(self, scaled):
        sel, adm = frozen(scaled)
        before = (scaled / "out/A1-6cat" / PP.MANIFEST).read_bytes()
        rec = PP.pack(sel["train"], sel["validation"], adm["path"], scaled / "out" / "A1-6cat")
        assert rec["train"]["fragments"] > 0 and rec["train"]["repeated_supervised_tokens"] > 0
        assert rec["train"]["supervised_tokens"] > rec["train"]["unique_supervised_tokens"]   #: exposure > size
        man = PP.verify_pack(scaled / "out" / "A1-6cat")
        assert (scaled / "out/A1-6cat" / PP.MANIFEST).read_bytes() == before          #: the manifest is untouched
        n = man["before_packing"]["training"]["unique_supervised_tokens"]
        assert rec["rungs"][0]["target_supervised_tokens"] == n == 6 * PER + FLUID
        assert man["proportions"]["counts"]["tool_calling"] == PER and man["proportions"]["passed"]
        assert man["before_packing"]["training"]["tool_calling_tokens"] == PER

    def test_the_workflow_runs_the_four_steps_in_order(self, scaled, monkeypatch):
        calls, sel = [], selection()
        monkeypatch.setattr(PP, "select", lambda *a, **k: (calls.append("select"), sel)[1])
        for name in ("proportions", "admit", "pack"):
            real = getattr(PP, name)
            monkeypatch.setattr(PP, name, lambda *a, _r=real, _n=name, **k: (calls.append(_n), _r(*a, **k))[1])
        rec = PP.build("A1-6cat", tool_docs=None, out=scaled / "out")
        assert calls == ["select", "proportions", "admit", "pack"]
        assert rec["data_manifest"]["sha256"] == rec["data_admission"]["manifest_sha256"]


class TestRefusals:
    def test_a_changed_or_missing_document_is_refused(self, scaled):
        sel, adm = frozen(scaled)
        d = sel["train"][0]
        changed = [TokenDocument(d.doc_id, d.prompt_ids, d.output_ids[:-2] + (8, EOS)), *sel["train"][1:]]
        with pytest.raises(PP.PackBuildError, match="not the frozen manifest's"):
            PP.pack(changed, sel["validation"], adm["path"], scaled / "p")
        with pytest.raises(PP.PackBuildError, match="not the frozen manifest's"):
            PP.pack(sel["train"][1:], sel["validation"], adm["path"], scaled / "p")
        assert not (scaled / "p").exists()

    def test_an_edited_manifest_is_refused(self, scaled):
        sel, adm = frozen(scaled)
        p = scaled / "out/A1-6cat" / PP.MANIFEST
        man = json.loads(p.read_text())
        for edit in ("count", "admitted"):
            m = json.loads(json.dumps(man))
            if edit == "count":
                m["before_packing"]["training"]["unique_supervised_tokens"] += 1
            else:
                m["admitted"] = False
            p.write_text(json.dumps(m))
            with pytest.raises(PP.PackBuildError, match="content_sha256|not an admitted"):
                PP.pack(sel["train"], sel["validation"], p, scaled / "p")

    def test_another_packing_spec_or_manifest_refuses_the_formal_binding(self, scaled):
        sel, adm = frozen(scaled)
        out = scaled / "out" / "A1-6cat"
        PP.pack(sel["train"], sel["validation"], adm["path"], out)
        PP.verify_pack(out)
        design = json.loads(PP.DESIGN.read_text())
        design["stage3_packing"]["shuffle_seed"] += 1                       #: the pack is not the design's any more
        PP.DESIGN.write_text(json.dumps(design))
        with pytest.raises(PP.PackBuildError, match="packing_spec"):
            PP.verify_pack(out)
        with pytest.raises(PP.PackBuildError, match="packing_spec"):
            PTP.bind([u for u in PTP.packed_units() if u["arm"] == "A1-6cat"], scaled / "out")
        design["stage3_packing"]["shuffle_seed"] -= 1
        PP.DESIGN.write_text(json.dumps(design))
        man = json.loads((out / PP.MANIFEST).read_text())                   #: re-sealed, but not the packed manifest
        man["sources"] = {"toy": 2}
        man["content_sha256"] = PP.canonical_sha({k: v for k, v in man.items() if k != "content_sha256"})
        (out / PP.MANIFEST).write_text(json.dumps(man))
        with pytest.raises(PP.PackBuildError, match="not the one it was packed from"):
            PP.verify_pack(out)

    def test_a_pack_without_an_admitted_manifest_is_refused(self, scaled):
        sel = selection()
        from aadistill.data.token_documents import write_pack
        tr = pack_sequential(sel["train"], 700, PP.PAD_ID, order_seed=1)
        v = pack_sequential(sel["validation"], 700, PP.PAD_ID, order_seed=1)
        write_pack(tr, v, scaled / "old" / "A1-6cat", data_type=sel["data_type"])
        with pytest.raises(PP.PackBuildError, match="no admitted data manifest"):
            PTP.bind([u for u in PTP.packed_units() if u["arm"] == "A1-6cat"], scaled / "old")


class TestRegisteredQuotas:
    def test_the_quotas_are_the_owners_numbers_and_no_new_tolerance(self):
        tm = json.loads(PP.TOKEN_MATCHED.read_text())["scale"]
        six = json.loads(PP.SIX.read_text())["base_category"]
        largest = {t: 1000 for t in PP.SIX_NAMES}
        q, shares, rule = PP.registered_quotas("A2-6cat", largest, 7)
        assert q["gsm8k"] == {"target": six["per_type"], "below": 1000, "above": 1000}
        assert q["code"]["below"] == 2000 and q["tool_calling"]["below"] == 7000          #: one unit per cell
        assert q["fluid"] == {"target": tm["categories"]["fluid"], "below": 0, "above": None}
        assert q["total"]["above"] == tm["tolerance"]["total_over_target_max_fraction"] * tm["per_mode_total"]
        assert shares[0]["max_deviation_points"] == tm["tolerance"]["category_share_deviation_max_points"] == 0.5
        assert rule["tolerance"] == tm["tolerance"]
        c1, _, _ = PP.registered_quotas("C1-6cat", largest, 7)
        assert {k: v["target"] for k, v in c1.items() if v["below"] == v["above"] == 0} == {
            "competition_math": 494130, "gsm8k": 494582, "multihop_reading": 492790, "rag_reading": 493581,
            "code": 492837, "fluid": tm["categories"]["fluid"]}
        assert c1["tool_calling"]["target"] == six["C1_6cat"]["tool_calling_target"]
        assert "share_check" not in json.loads(PP.DESIGN.read_text())["stage3_packing"]   #: packing owns no share

    def test_the_withdrawn_check_would_have_passed_a_short_tool_type(self):
        """A2-6cat's projected schema_arguments case: the tool type 1.5% short -- inside the withdrawn 2% check,
        refused by the registered rule (the supporting category is below its target)."""
        largest = {t: 900 for t in PP.SIX_NAMES}
        q, shares, _ = PP.registered_quotas("A2-6cat", largest, 7)
        five = {"competition_math": 493404, "gsm8k": 493495, "rag_reading": 493276, "multihop_reading": 493296,
                "code": 493701}
        counts = {**five, "tool_calling": round(493417.83 * 0.985), "fluid": 2963139}
        from aadistill.data.mixture import validate_proportions
        rep = validate_proportions(counts, q, shares)
        assert rep["quotas"]["supporting"]["status"] == "SHORTFALL" and not rep["passed"]
        supporting = sum(v for k, v in counts.items() if k != "fluid")
        assert abs(counts["tool_calling"] / (supporting / 6) - 1) < 0.02          #: what share_check allowed
