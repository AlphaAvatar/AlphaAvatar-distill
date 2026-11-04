"""Data selection, proportion validation, admission / freeze and packing are separate (maintainer 2026-11-04):

1. the proportion validation runs alone and packs nothing;
2. the packing runs on any valid documents and never calls the validation;
3. data outside its proportions can be packed but is never admitted, and the workflow refuses before packing;
4. frozen data keeps its counted size and proportions through the packing, fragments and all;
5. a changed document, an edited manifest or another packing spec is refused;
6. the quotas are the registered owners' numbers, and the validation itself is the generic core function;
7. the admission takes only the proportion report of the selection it admits (maintainer 2026-11-04, P1): another
   selection's report, an edited report, or content, mask, category, split, tool-cell or quota-design drift after
   the validation refuses -- without the admission validating anything again;
8. the formal binding verifies the ACTUAL packed tokens and supervision (P1): one changed token, a changed mask, a
   replaced file or a forged pack with a re-sealed identity refuses; an untouched pack binds;
9. truncate=True holds through the binding (maintainer review of 2468c1aa): an overlong document packs truncated with
   prefix restarts, its unique supervised tokens still count whole, the pack is re-derived from the frozen
   documents and binds, and tampering with the truncated placement or the frozen documents refuses.

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
        approved = selection(sizes, tool, [appr])
        adm = PP.admit("A1-6cat", approved, PP.proportions("A1-6cat", approved), scaled / "out")
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


def short_gsm8k():
    return selection({**{t: [UNIT] * 4 for t in TYPES}, "supporting.math_gsm8k": [UNIT] * 2, "primary.x": [500] * 12})


class TestReportBinding:
    def test_a_passing_report_of_another_selection_is_refused(self, scaled):
        a, b = selection(), short_gsm8k()
        rep_a = PP.proportions("A1-6cat", a)
        assert rep_a["passed"] and not PP.proportions("A1-6cat", b)["passed"]
        with pytest.raises(PP.AdmissionError, match="validated another selection") as e:
            PP.admit("A1-6cat", b, rep_a, scaled / "out")
        assert not e.value.verdict["admitted"] and not (scaled / "out").exists()      #: no manifest frozen

    @pytest.mark.parametrize("drift", ["tokens", "mask", "data type", "tool category", "split", "unit",
                                       "tool target", "source", "quota design"])
    def test_drift_after_the_validation_is_refused(self, scaled, drift):
        from aadistill.data.token_documents import MaskedDocument
        sel = selection()
        rep = PP.proportions("A1-6cat", sel)
        d = sel["train"][0]
        if drift == "tokens":                                                   #: same length, one token changed
            sel["train"][0] = TokenDocument(d.doc_id, d.prompt_ids, d.output_ids[:-2] + (8, EOS), d.meta)
        elif drift == "mask":                         #: same ids, same supervised count, one target moved by one
            m = list(d.target_mask)
            m[len(d.prompt_ids) - 1], m[len(d.prompt_ids)] = True, False
            sel["train"][0] = MaskedDocument(d.doc_id, d.ids, tuple(m))
            assert sum(m) == sum(d.target_mask)
        elif drift == "data type":
            sel["data_type"][d.doc_id] = "supporting.math_gsm8k"
        elif drift == "tool category":
            t = next(i for i, x in enumerate(sel["train"]) if x.meta.get("category"))
            x = sel["train"][t]
            sel["train"][t] = TokenDocument(x.doc_id, x.prompt_ids, x.output_ids, {**x.meta, "category": "c2"})
        elif drift == "split":
            sel["train"].append(sel["validation"].pop())
        elif drift == "unit":
            sel["unit_of"][d.doc_id] = "another-task"
        elif drift == "tool target":
            sel["tool"]["quota"]["cells"]["training/c1"]["target"] += 1
        elif drift == "source":
            sel["sources"] = {"toy": 2}
        else:
            six = json.loads(PP.SIX.read_text())
            six["base_category"]["per_type"] = PER + 1
            PP.SIX.write_text(json.dumps(six))
        with pytest.raises(PP.AdmissionError, match="validated another selection"):
            PP.admit("A1-6cat", sel, rep, scaled / "out")
        assert not (scaled / "out").exists()

    def test_an_edited_report_or_another_arms_report_is_refused(self, scaled):
        sel = short_gsm8k()
        rep = PP.proportions("A1-6cat", sel)
        forged = json.loads(json.dumps(rep))
        forged["quotas"]["gsm8k"]["status"], forged["passed"], forged["failures"] = "OK", True, []
        with pytest.raises(PP.AdmissionError, match="edited after its validation"):
            PP.admit("A1-6cat", sel, forged, scaled / "out")
        other = {**rep, "arm": "A2-6cat"}
        other["report_sha256"] = PP.canonical_sha({k: v for k, v in other.items() if k != "report_sha256"})
        with pytest.raises(PP.AdmissionError, match="'A2-6cat'"):
            PP.admit("A1-6cat", sel, other, scaled / "out")

    def test_the_admission_binds_without_validating_again(self, scaled, monkeypatch):
        sel = selection()
        rep = PP.proportions("A1-6cat", sel)
        from aadistill.data import mixture
        refuse(monkeypatch, (mixture, "validate_proportions"), (PP, "proportions"), (PP, "registered_quotas"),
               (PP, "type_counts"))
        adm = PP.admit("A1-6cat", sel, rep, scaled / "out")
        assert adm["manifest"]["selection_sha256"] == rep["selection_sha256"] == PP.selection_identity("A1-6cat", sel)


def built(scaled):
    sel, adm = frozen(scaled)
    out = scaled / "out" / "A1-6cat"
    rec = PP.pack(sel["train"], sel["validation"], adm["path"], out)
    return out, rec


def rewrite_blocks(out, edit):
    import numpy as np
    with np.load(out / "blocks.npz") as z:
        arr = {k: z[k].copy() for k in z.files}
    edit(arr)
    np.savez(out / "blocks.npz", **arr)
    return arr


def a1_units():
    return [u for u in PTP.packed_units() if u["arm"] == "A1-6cat"]


class TestPackContent:
    def test_an_untouched_pack_binds_with_its_content_identity(self, scaled):
        out, rec = built(scaled)
        PP.verify_pack(out)
        unit = PTP.bind(a1_units(), scaled / "out")["units"][0]
        data = unit["config"]["lineage"]["data"][0]
        assert data["packed_content_sha256"] == rec["data_admission"]["packed_content_sha256"]
        assert data["data_manifest_content_sha256"] == rec["data_admission"]["content_sha256"]

    @pytest.mark.parametrize("where", ["training", "validation"])
    def test_one_changed_non_padding_token_refuses_the_binding(self, scaled, where):
        out, rec = built(scaled)
        files = {n: (out / n).read_bytes() for n in ("ladder.json", "audit.jsonl", PP.MANIFEST)}
        n_train = rec["n_blocks"]

        def edit(arr):
            rows = range(n_train) if where == "training" else range(n_train, len(arr["input_ids"]))
            b = next(r for r in rows if arr["content_mask"][r].any())
            j = int(arr["content_mask"][b].nonzero()[0][-1])                   #: a non-padding position
            arr["input_ids"][b, j] = arr["input_ids"][b, j] + 1
        rewrite_blocks(out, edit)
        assert all((out / n).read_bytes() == v for n, v in files.items())     #: every other file unchanged
        with pytest.raises(PP.PackBuildError, match="packed_content_sha256"):
            PP.verify_pack(out)
        with pytest.raises(PP.PackBuildError, match="packed_content_sha256"):
            PTP.bind(a1_units(), scaled / "out")

    def test_a_changed_supervision_mask_or_a_replaced_file_refuses(self, scaled):
        out, rec = built(scaled)
        good = (out / "blocks.npz").read_bytes()
        rewrite_blocks(out, lambda a: a["ce_mask"].__setitem__((0, int(a["ce_mask"][0].nonzero()[0][0])), False))
        with pytest.raises(PP.PackBuildError, match="packed_content_sha256"):
            PP.verify_pack(out)
        (out / "blocks.npz").write_bytes(good)
        PP.verify_pack(out)
        sel = selection()                                                       #: another packing of the same data
        from aadistill.data.token_documents import write_pack
        tr = pack_sequential(sel["train"], 700, PP.PAD_ID, order_seed=12345)
        v = pack_sequential(sel["validation"], 700, PP.PAD_ID, order_seed=12345)
        write_pack(tr, v, scaled / "other", data_type=sel["data_type"])
        (out / "blocks.npz").write_bytes((scaled / "other" / "blocks.npz").read_bytes())
        with pytest.raises(PP.PackBuildError, match="packed_content_sha256"):
            PP.verify_pack(out)

    @pytest.mark.parametrize("forgery", ["token", "swapped sequences", "reordered sequences, audit rewritten to match"])
    def test_a_forged_pack_with_a_resealed_identity_is_refused(self, scaled, forgery):
        """The recorded identity rewritten to match the edit: the sequences re-derived from the frozen documents
        still refuse it -- the identity alone never makes a pack correct. The last forgery keeps every document
        intact and its placements consistent."""
        why = "not the packing of the manifest's documents"
        out, rec = built(scaled)
        n_train = rec["n_blocks"]

        def edit(arr):
            if forgery == "token":
                j = int(arr["ce_mask"][0].nonzero()[0][0])
                arr["input_ids"][0, j] += 1
            else:
                for k in ("input_ids", "ce_mask", "content_mask"):
                    arr[k][[0, 1]] = arr[k][[1, 0]]
        arr = rewrite_blocks(out, edit)
        if forgery.startswith("reordered"):
            rows = [json.loads(x) for x in (out / "audit.jsonl").read_text().splitlines()]
            rows[0]["sessions"], rows[1]["sessions"] = rows[1]["sessions"], rows[0]["sessions"]
            (out / "audit.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        lad = json.loads((out / "ladder.json").read_text())
        lad["data_admission"]["packed_content_sha256"] = PP.packed_content_sha256(
            arr["input_ids"], arr["ce_mask"], arr["content_mask"], n_train)
        (out / "ladder.json").write_text(json.dumps(lad))
        with pytest.raises(PP.PackBuildError, match=why):
            PP.verify_pack(out)


def overlong():
    """A1-6cat's selection with one fluid example of 1,000 supervised tokens (two of 500 merged: the same total),
    longer than the scaled 700-token sequence."""
    return selection({**{t: [UNIT] * 4 for t in TYPES}, "primary.x": [1000] + [500] * 10})


class TestTruncation:
    def test_an_overlong_document_packs_truncated_and_binds(self, scaled):
        sel = overlong()
        big = next(d for d in sel["train"] if len(d) > 700)
        adm = PP.admit("A1-6cat", sel, PP.proportions("A1-6cat", sel), scaled / "out")
        out = scaled / "out" / "A1-6cat"
        rec = PP.pack(sel["train"], sel["validation"], adm["path"], out)
        st = rec["train"]
        assert st["truncated_supervised_tokens"] == len(big) - 700 and st["fragments"] > 0      #: truncate + restart
        n = adm["manifest"]["before_packing"]["training"]["unique_supervised_tokens"]
        assert n == st["unique_supervised_tokens"] == 6 * PER + FLUID == rec["rungs"][0]["target_supervised_tokens"]
        rows = [json.loads(x) for x in (out / "audit.jsonl").read_text().splitlines()]
        final = [m for r in rows for m in r["sessions"] if m["session_id"] == big.doc_id and m["final"]]
        assert [(m["length"], m["document_length"]) for m in final] == [(700, len(big))]          #: placed truncated
        assert PP.verify_pack(out)["documents_file"]["name"] == PP.DOCUMENTS
        unit = PTP.bind(a1_units(), scaled / "out")["units"][0]
        assert unit["config"]["lineage"]["data"][0]["packed_content_sha256"] == rec["data_admission"]["packed_content_sha256"]

    def test_tampering_with_the_truncated_placement_is_refused(self, scaled):
        sel = overlong()
        big = next(d for d in sel["train"] if len(d) > 700)
        adm = PP.admit("A1-6cat", sel, PP.proportions("A1-6cat", sel), scaled / "out")
        out = scaled / "out" / "A1-6cat"
        rec = PP.pack(sel["train"], sel["validation"], adm["path"], out)
        rows = [json.loads(x) for x in (out / "audit.jsonl").read_text().splitlines()]
        b, m = next((r["block"], m) for r in rows for m in r["sessions"] if m["session_id"] == big.doc_id and m["final"])

        def edit(arr):
            arr["input_ids"][b, m["start"] + m["length"] - 1] += 1                #: the last token placed
        arr = rewrite_blocks(out, edit)
        with pytest.raises(PP.PackBuildError, match="packed_content_sha256"):
            PP.verify_pack(out)
        lad = json.loads((out / "ladder.json").read_text())                       #: re-sealed: re-derivation refuses
        lad["data_admission"]["packed_content_sha256"] = PP.packed_content_sha256(
            arr["input_ids"], arr["ce_mask"], arr["content_mask"], rec["n_blocks"])
        (out / "ladder.json").write_text(json.dumps(lad))
        with pytest.raises(PP.PackBuildError, match="not the packing of the manifest's documents"):
            PP.verify_pack(out)

    def test_changed_frozen_documents_are_refused(self, scaled):
        import numpy as np
        sel = overlong()
        adm = PP.admit("A1-6cat", sel, PP.proportions("A1-6cat", sel), scaled / "out")
        out = scaled / "out" / "A1-6cat"
        PP.pack(sel["train"], sel["validation"], adm["path"], out)
        f = out / PP.DOCUMENTS
        with np.load(f) as z:
            arr = {k: z[k].copy() for k in z.files}
        arr["training_ids"][-2] += 1                            #: a tail token beyond any truncation point
        np.savez(f, **arr)
        with pytest.raises(PP.PackBuildError, match="frozen documents are missing or not the admitted ones"):
            PP.verify_pack(out)
        man = json.loads((out / PP.MANIFEST).read_text())       #: the manifest re-pointed and re-sealed
        man["documents_file"]["sha256"] = PP.sha_file(f)
        man["content_sha256"] = PP.canonical_sha({k: v for k, v in man.items() if k != "content_sha256"})
        (out / PP.MANIFEST).write_text(json.dumps(man))
        with pytest.raises(PP.PackBuildError, match="not the one it was packed from"):
            PP.verify_pack(out)

    def test_frozen_documents_must_match_the_manifests_rows(self, scaled):
        """The manifest's document rows are the admission's record: frozen documents re-pointed by a re-sealed
        manifest and ladder -- the packed sequences untouched -- still refuse, document by document."""
        import numpy as np
        sel = overlong()
        adm = PP.admit("A1-6cat", sel, PP.proportions("A1-6cat", sel), scaled / "out")
        out = scaled / "out" / "A1-6cat"
        PP.pack(sel["train"], sel["validation"], adm["path"], out)
        f = out / PP.DOCUMENTS
        with np.load(f) as z:
            arr = {k: z[k].copy() for k in z.files}
        arr["training_ids"][-2] += 1
        np.savez(f, **arr)
        man = json.loads((out / PP.MANIFEST).read_text())
        man["documents_file"]["sha256"] = PP.sha_file(f)
        man["content_sha256"] = PP.canonical_sha({k: v for k, v in man.items() if k != "content_sha256"})
        (out / PP.MANIFEST).write_text(json.dumps(man))
        lad = json.loads((out / "ladder.json").read_text())
        lad["data_admission"].update(manifest_sha256=PP.sha_file(out / PP.MANIFEST), content_sha256=man["content_sha256"])
        (out / "ladder.json").write_text(json.dumps(lad))
        with pytest.raises(PP.PackBuildError, match="tokens or supervision are not the admitted ones"):
            PP.verify_pack(out)
