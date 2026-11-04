"""The tool pool's isolation (atoms, eval-set-1, benchmarks, between splits), its length bound, Toucan's reading of
steps, errors and conflicts, the lineage's own generation kind, and the six-type packs counted before packing."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from stages.teacher_modes import tool_pool as TP

TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]


def tool(name, params=None):
    return {"type": "function", "function": {"name": name, "parameters": params or {"type": "object"}}}


def task(tid, text="a question", servers=(), tools=None, **over):
    t = {"task_id": tid, "category": "tool_selection", "source": {"dataset": "toy"}, "family": None, "split": None,
         "servers": list(servers), "tools": tools if tools is not None else TOOLS,
         "messages": [{"role": "user", "content": text}], "backend": {"kind": "replay", "table": {}},
         "final_check": {"kind": "calls_tools", "values": ["get_weather"]}, "max_turns": 4}
    t.update(over)
    return t


def server_in(split, avoid=()):
    """A server name whose atom hashes to ``split``."""
    k = 0
    while True:
        name = f"srv{k}"
        if TP.split_of(f"server:{name}") == split and name not in avoid:
            return name
        k += 1


def long_text(seed: str) -> str:
    return " ".join(f"{seed}{i}" for i in range(30))


class TestAtoms:
    def test_a_servers_tasks_share_one_split_and_a_split_never_shares_a_server(self):
        a, b = server_in("training"), server_in("state_eval")
        tasks = [task(f"tc1-a{k}", f"q{k}", [a], [tool(f"{a}-t{k % 3}")]) for k in range(6)]
        tasks += [task(f"tc1-b{k}", f"r{k}", [b], [tool(f"{b}-u{k}")]) for k in range(3)]
        pool = TP.assemble(tasks)
        assert {t["split"] for t in pool["tasks"] if t["task_id"].startswith("tc1-a")} == {"training"}
        assert {t["split"] for t in pool["tasks"] if t["task_id"].startswith("tc1-b")} == {"state_eval"}
        assert pool["atoms"]["total"] == 2

    def test_a_task_whose_servers_fall_in_different_splits_is_refused(self):
        a, b = server_in("training"), server_in("validation")
        pool = TP.assemble([task("tc1-x", "q", [a, b], [tool(f"{a}-t"), tool(f"{b}-t")]),
                            task("tc1-y", "r", [a], [tool(f"{a}-t")])])
        reasons = {e["task_id"]: e["reason"] for e in pool["excluded"]}
        assert reasons == {"tc1-x": "T4: its atoms fall in different splits"}
        assert [t["task_id"] for t in pool["tasks"]] == ["tc1-y"]

    def test_tools_no_server_owns_are_their_own_atoms(self):
        t = task("tc1-z", servers=["s"], tools=[tool("s-get"), tool("other", {"type": "object", "x": 1})])
        atoms = TP.atoms_of(t)
        assert "server:s" in atoms and any(x.startswith("tool:") for x in atoms) and len(atoms) == 2
        assert TP.tool_identities(t)[0][0] == "get"                  # the bare name drops the server's slug

    def test_the_old_combined_family_split_a_server_and_atoms_do_not(self):
        # v1 hashed the joined "a|b" as one family, so a's tasks could land in two splits; per-atom splits cannot
        a = server_in("training")
        k = 0
        while True:
            b = server_in("training", avoid=(a,) + tuple(f"srv{i}" for i in range(k)))
            if TP.split_of(f"server:{a}|{b}") != "training":
                break
            k = int(b[3:]) + 1
        pool = TP.assemble([task("tc1-1", "q", [a], [tool(f"{a}-t")]),
                            task("tc1-2", "r", [a, b], [tool(f"{a}-t"), tool(f"{b}-u")])])
        assert {t["split"] for t in pool["tasks"]} == {"training"} and len(pool["tasks"]) == 2


class TestDecontamination:
    def test_eval_set_tool_names_and_identities_and_benchmark_identities_are_refused(self):
        ref = TP.Reference()
        ref.add("eval-set-1", label="T1", prompts=["held out"], tools=[("get_weather", {"type": "object"})],
                refuse_tool_names=True)
        ref.add("bench", label="T2", prompts=["bench q"], tools=[("lookup", {"type": "object"}), ("search", {"a": 1})])
        s = server_in("training")
        pool = TP.assemble([task("tc1-held", "held   OUT", [s], [tool(f"{s}-x")]),
                            task("tc1-name", "q1", [s], [tool(f"{s}-get_weather", {
                                "type": "object", "properties": {"z": {"type": "string"}}})]),
                            task("tc1-ident", "q2", [s], [tool(f"{s}-lookup")]),
                            task("tc1-nameonly", "q3", [s], [tool(f"{s}-search")]),
                            task("tc1-ok", "q4", [s], [tool(f"{s}-x")])], reference=ref)
        reasons = {e["task_id"]: e["reason"] for e in pool["excluded"]}
        assert reasons["tc1-held"] == "T1: a eval-set-1 prompt"
        assert reasons["tc1-name"] == "T1: a eval-set-1 tool name"
        assert reasons["tc1-ident"] == "T2: a bench tool (name and schema)"
        assert "tc1-nameonly" not in reasons
        assert pool["tool_name_only_matches_counted"] == {"bench": 1, "bench:signature": 1}     #: counted, not refused

    def test_a_13_gram_near_duplicate_of_a_held_out_prompt_is_refused(self):
        ref = TP.Reference().add("bench", label="T2", prompts=[long_text("w")])
        s = server_in("training")
        near = long_text("w").replace("w29", "something else")
        pool = TP.assemble([task("tc1-near", near, [s], [tool(f"{s}-x")]),
                            task("tc1-far", long_text("v"), [s], [tool(f"{s}-x")])], reference=ref)
        assert {e["task_id"]: e["reason"] for e in pool["excluded"]} == {
            "tc1-near": "T2: a 13-gram near-duplicate of a bench prompt"}

    def test_training_never_keeps_a_near_duplicate_of_a_state_eval_task(self):
        a, b = server_in("state_eval"), server_in("training")
        pool = TP.assemble([task("tc1-se", long_text("u"), [a], [tool(f"{a}-x")]),
                            task("tc1-tr", long_text("u") + " extra words", [b], [tool(f"{b}-y")])])
        assert {e["task_id"]: e["reason"] for e in pool["excluded"]} == {
            "tc1-tr": "T4: a 13-gram near-duplicate of an earlier split's task"}

    def test_an_exact_prompt_never_crosses_splits_even_when_templated_away(self):
        # one prompt repeated across more servers than the 13-gram index keeps (document frequency > 5): only the
        # exact prompt key catches it; the earliest split in the priority keeps it, the others refuse it
        se, cal, tr = server_in("state_eval"), server_in("calibration"), server_in("training")
        text = long_text("same")
        tasks = [task(f"tc1-se{k}", text, [se], [tool(f"{se}-a{k}")]) for k in range(7)]
        tasks += [task("tc1-cal", text, [cal], [tool(f"{cal}-b")]), task("tc1-tr", text, [tr], [tool(f"{tr}-c")])]
        pool = TP.assemble(tasks)
        reasons = {e["task_id"]: e["reason"] for e in pool["excluded"]}
        assert reasons["tc1-cal"] == reasons["tc1-tr"] == "T4: an earlier split's exact prompt"
        assert {t["split"] for t in pool["tasks"]} == {"state_eval"}

    def test_a_tool_identity_never_crosses_splits(self):
        a, b = server_in("state_eval"), server_in("training")
        pool = TP.assemble([task("tc1-a", "first", [a], [tool(f"{a}-get_weather")]),
                            task("tc1-b", "second", [b], [tool(f"{b}-get_weather")])])
        assert {e["task_id"]: e["reason"] for e in pool["excluded"]} == {
            "tc1-b": "T4: a tool identity of an earlier split"}

    def test_t3_refuses_a_lineage_tools_signature_and_unchecked_schemas_are_refused(self):
        ref = TP.Reference().add("xlam", label="T3", prompts=["x"], refuse_signatures=True,
                                 tools=[("live_giveaways", {"type": {"type": "str", "description": "kind"}})])
        s = server_in("training")
        sig = tool(f"{s}-live_giveaways", {"type": "object", "properties": {"type": {"type": "string"}}})
        odd = tool(f"{s}-odd", {"type": "object", "properties": {"x": {"type": "string", "if": {}}}})
        pool = TP.assemble([task("tc1-sig", "q1", [s], [sig]), task("tc1-odd", "q2", [s], [odd])], reference=ref)
        reasons = {e["task_id"]: e["reason"] for e in pool["excluded"]}
        assert reasons["tc1-sig"] == "T3: a xlam tool (name and parameter names)"
        assert reasons["tc1-odd"] == "schema: keywords the validator does not check ['if']"
        #: the pool's default: a signature match is counted, not refused (generic APIs share signatures)
        ref = TP.Reference().add("xlam", label="T3", prompts=["x"],
                                 tools=[("live_giveaways", {"type": {"type": "str", "description": "kind"}})])
        pool = TP.assemble([task("tc1-sig", "q1", [s], [sig])], reference=ref)
        assert not pool["excluded"] and pool["tool_name_only_matches_counted"] == {"xlam:signature": 1, "xlam": 1}

    def test_conflicting_recorded_responses_are_refused_not_chosen(self):
        s = server_in("training")
        t = task("tc1-c", "q", [s], [tool(f"{s}-x")],
                 backend={"kind": "replay", "table": {}, "conflicts": [{"call": "{}", "responses": 2}]})
        assert TP.assemble([t])["excluded"][0]["reason"].startswith("replay:")

    def test_exclusions_are_counted_by_reason_source_category_and_split(self):
        a, b = server_in("training"), server_in("validation")
        pool = TP.assemble([task("tc1-x", "q", [a, b], [tool(f"{a}-t"), tool(f"{b}-t")])])
        assert pool["excluded_by"] == [{"reason": "T4: its atoms fall in different splits", "source": "toy",
                                        "category": "tool_selection", "split": "conflict", "tasks": 1}]


class TestLengthBound:
    def test_main_runs_the_teachers_length_check(self, tmp_path, monkeypatch):
        s = server_in("training")
        src = tmp_path / "tasks.jsonl"
        src.write_text("".join(json.dumps(task(f"tc1-{k}", f"q{k}", [s], [tool(f"{s}-x")])) + "\n" for k in range(3)))
        seen = []
        monkeypatch.setattr(TP, "eval_set_reference", lambda ref: ref)
        monkeypatch.setattr(TP, "benchmark_reference", lambda ref: ref)

        def checkers(proto):
            seen.append(sorted(proto["lineages"]))
            return [lambda t: 10 ** 6 if t["task_id"] == "tc1-1" else 10]
        monkeypatch.setattr(TP, "length_checkers", checkers)
        assert TP.main(["--source", f"toy={src}", "--out", str(tmp_path / "pool")]) == 0
        rec = json.loads((tmp_path / "pool/pool.json").read_text())
        assert seen == [["hybrid", "thinking_2507"]]
        assert [e["task_id"] for e in rec["excluded"]] == ["tc1-1"]
        assert rec["excluded"][0]["reason"] == f"length: first prompt {10 ** 6} > {TP.PROMPT_BOUND}"

    def test_the_checkers_render_each_teachers_template(self):
        proto = json.loads(TP.PROTOCOL.read_text())
        cache = Path.home() / ".cache/huggingface/hub"
        for li in proto["lineages"].values():
            if not (cache / f"models--{li['model'].replace('/', '--')}/snapshots/{li['revision']}").is_dir():
                pytest.skip("the teachers' tokenizers are not cached here")
        fs = {f.__name__: f for f in TP.length_checkers(proto)}
        assert sorted(fs) == ["hybrid:H-N", "hybrid:H-T", "thinking_2507:T2507"]
        t = task("tc1-r", "weather in Paris?")
        n = {k: f(t) for k, f in fs.items()}
        assert n["hybrid:H-N"] > n["hybrid:H-T"] > 50                 # the non-thinking prompt pre-closes the think
        assert n["hybrid:H-T"] < fs["hybrid:H-T"](task("tc1-r", "weather in Paris?", tools=TOOLS + [tool("second")]))


class TestToucan:
    def row(self, msgs, sub="single-turn-original", targets="get_weather", tools=None):
        return {"uuid": "u", "subset_name": sub, "target_tools": targets,
                "tools": json.dumps(tools or [tool("ws-get_weather"), tool("ws-get_time")]), "messages": json.dumps(msgs)}

    def call(self, name, **args):
        return {"role": "tool_call", "content": repr({"name": name, "arguments": json.dumps(args)})}

    def test_parallel_means_calls_in_one_step_not_a_call_count(self):
        from stages.teacher_modes import tool_sources_toucan as TS
        U, A = {"role": "user", "content": "q"}, {"role": "assistant", "content": "ok"}
        seq = TS.task_of(self.row([U, self.call("ws-get_weather", city="a"), {"role": "tool_response", "content": "1"},
                                   A, self.call("ws-get_weather", city="b"), {"role": "tool_response", "content": "2"},
                                   A]))
        par = TS.task_of(self.row([U, self.call("ws-get_weather", city="a"), self.call("ws-get_weather", city="b"),
                                   {"role": "tool_response", "content": "1"}, {"role": "tool_response", "content": "2"},
                                   A]))
        assert seq["category"] == "mcp_agent" and par["category"] == "multiple_parallel"
        assert par["source"]["recorded"]["max_calls_per_step"] == 2 and seq["source"]["recorded"]["steps"] == 2

    def test_an_error_null_field_is_not_an_error_and_errors_are_classified(self):
        from stages.teacher_modes import tool_sources_toucan as TS
        assert TS.error_kind('{"results": [1], "error": null}') is None
        assert TS.error_kind("The error bars show...") is None
        assert TS.error_kind('{"error": "Invalid API key"}') == "tool_failure"
        assert TS.error_kind("Error generating text: Payment Required") == "tool_failure"
        assert TS.error_kind("Error executing tool f: 1 validation error for fArguments\nx\n  Field required") == \
            "invalid_arguments"
        assert TS.error_kind("Error executing tool f: 1 validation error for fOutput\nresult") == "tool_failure"

    def test_recovery_needs_a_corrected_successful_retry(self):
        from stages.teacher_modes import tool_sources_toucan as TS
        U, A = {"role": "user", "content": "q"}, {"role": "assistant", "content": "ok"}
        bad = {"role": "tool_response", "content": "Error executing tool g: 1 validation error for gArguments"}
        fixed = TS.task_of(self.row([U, self.call("ws-get_weather", city=1), bad, A,
                                     self.call("ws-get_weather", city="a"), {"role": "tool_response", "content": "s"},
                                     A]))
        failed = TS.task_of(self.row([U, self.call("ws-get_weather", city="a"),
                                      {"role": "tool_response", "content": '{"error": "quota"}'}, A]))
        ok = TS.task_of(self.row([U, self.call("ws-get_weather", city="a"),
                                  {"role": "tool_response", "content": '{"v": 1, "error": null}'}, A]))
        assert (fixed["category"], fixed["category_detail"]) == ("error_recovery", "corrected_retry")
        assert (failed["category"], failed["category_detail"]) == ("error_recovery", "tool_failure")
        assert (ok["category"], ok["category_detail"]) == ("tool_selection", None)

    def test_conflicting_responses_are_recorded_and_left_out_of_the_table(self):
        from stages.teacher_modes import tool_sources_toucan as TS
        U, A = {"role": "user", "content": "q"}, {"role": "assistant", "content": "ok"}
        t = TS.task_of(self.row([U, self.call("ws-get_weather", city="a"), {"role": "tool_response", "content": "1"},
                                 A, self.call("ws-get_weather", city="a"), {"role": "tool_response", "content": "2"},
                                 A]))
        assert t["backend"]["table"] == {} and t["backend"]["conflicts"][0]["responses"] == 2

    def test_servers_come_from_known_slugs_and_verification_is_registered(self):
        from stages.teacher_modes import tool_sources_toucan as TS
        U = {"role": "user", "content": "q"}
        t = TS.task_of(self.row([U], sub="irrelevant", targets="", tools=[tool("ws-get_weather"), tool("loose")]),
                       known_slugs={"ws"})
        assert t["servers"] == ["ws"] and t["unattributed_tools"] == ["loose"]
        #: the irrelevant subset's no-call check: the tool decision, and no call to have executed
        assert t["source"]["verification"] == {"levels": ["protocol", "schema_validity", "tool_selection"],
                                               "step_gold": False, "task_success_verifier": None}


class TestLineageKinds:
    def test_each_lineage_runs_under_its_own_kind(self, tmp_path, monkeypatch):
        from stages.teacher_modes import token_matched_phase as TM
        from stages.teacher_modes import tool_session as TS
        seen = []

        def refuse(path, kind, root):
            seen.append(kind)
            raise RuntimeError("stop here")
        monkeypatch.setattr(TM, "require_on_pod", refuse)
        for lineage in ("hybrid", "thinking_2507"):
            assert TS.main(["--authorization", str(tmp_path / "a.json"), "--lineage", lineage,
                            "--durable-root", str(tmp_path), "--status", str(tmp_path / "s")]) == 3
        assert seen == ["tool_generation", "tool_generation_t2507"]

    #: which gate each lineage's kind takes: test_t2507_device_admission.py::test_only_the_hybrid_takes_the_generation_scoped_gate


class TestSixTypePacks:
    def docs(self, by_type):
        from aadistill.data.token_documents import TokenDocument
        out, dtype = [], {}
        for t, (n, size) in by_type.items():
            for k in range(n):
                d = TokenDocument(f"{t}:{k}", (5, 6), tuple([7] * (size - 1) + [0]))
                out.append(d)
                dtype[d.doc_id] = t
        return out, dtype

    def tool_file(self, path, n, size, mode="T2507"):
        rows = [{"doc_id": f"tc1-{k}|{mode}|t0", "task_id": f"tc1-{k}", "split": "training", "category": "c",
                 "prompt_ids": [5, 6], "output_ids": [7] * (size - 1) + [0]} for k in range(n)]
        rows += [{"doc_id": f"tc1-v{k}|{mode}|t0", "task_id": f"tc1-v{k}", "split": "validation", "category": "c",
                  "prompt_ids": [5], "output_ids": [7, 0]} for k in range(2)]
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return [f"tc1-{k}" for k in reversed(range(n))]

    def cells(self, train=100.0, val=2.0):
        return {"training": [{"split": "training", "category": "c", "unit": "supervised_tokens", "target": train}],
                "validation": [{"split": "validation", "category": None, "unit": "tasks", "target": val}]}

    def test_tool_episodes_are_cut_whole_per_cell_in_the_frozen_order(self, tmp_path):
        from stages.teacher_modes import packed_packs as PP
        order = self.tool_file(tmp_path / "t.jsonl", 10, 30)
        orders = {"training": order, "validation": ["tc1-v1", "tc1-v0"]}
        train, val, q = PP.tool_documents(tmp_path / "t.jsonl", "T2507", cells=self.cells(), orders=orders)
        assert [d.doc_id.split("|")[0] for d in train] == order[:4]
        assert q["cells"]["training/c"]["selected"] == 120 and q["cells"]["training/c"]["status"] == "REACHED"
        assert [d.doc_id.split("|")[0] for d in val] == ["tc1-v1", "tc1-v0"]
        _, _, q = PP.tool_documents(tmp_path / "t.jsonl", "T2507", cells=self.cells(train=10 ** 6), orders=orders)
        assert q["cells"]["training/c"]["status"] == "SHORTFALL"             #: recorded, never filled from elsewhere
        with pytest.raises(PP.PackBuildError, match="frozen order"):
            PP.tool_documents(tmp_path / "t.jsonl", "T2507", cells=self.cells(), orders={**orders, "training": []})

    def test_validation_keeps_every_document_and_reports_coverage(self, tmp_path, monkeypatch):
        from stages.teacher_modes import packed_packs as PP
        six = {"openmath": (2, 50), "gsm8k": (2, 50), "code": (2, 50), "rag_evidence": (2, 50),
               "multihop_qa": (2, 50)}
        train, dtype = self.docs({**six, "primary.x": (6, 100)})
        val, vtype = self.docs({"openmath": (40, 50)})
        vtype = {f"v{k}": t for k, t in vtype.items()}
        from aadistill.data.token_documents import TokenDocument
        val = [TokenDocument(f"v{d.doc_id}", d.prompt_ids, d.output_ids) for d in val]
        monkeypatch.setattr(PP, "c1_documents", lambda: (train, val, {**dtype, **vtype}))
        from stages.teacher_modes import tool_generation_plan as TGP
        order = self.tool_file(tmp_path / "t.jsonl", 4, 50)
        plan = {"lineages": {"thinking_2507": {"modes": {"T2507": {"cells": [
            *self.cells()["training"], *self.cells()["validation"]]}}}}}
        monkeypatch.setattr(TGP, "for_consumers", lambda **k: (plan, {"training": order,
                                                                      "validation": ["tc1-v0", "tc1-v1"]}))
        design = json.loads(PP.DESIGN.read_text())
        design["stage3_packing"]["sequence_length"] = 64       #: fluid examples of 102 tokens pack truncated
        (tmp_path / "design.json").write_text(json.dumps(design))
        monkeypatch.setattr(PP, "DESIGN", tmp_path / "design.json")
        tm, six = json.loads(PP.TOKEN_MATCHED.read_text()), json.loads(PP.SIX.read_text())  #: C1's numbers, scaled
        tm["scale"].update(per_mode_total=1200, categories={"base_counterpart": 600, "fluid": 600})
        six["base_category"]["C1_6cat"].update(tool_calling_target=100, kept_byte_for_byte={
            "competition_math (openmath)": 100, "gsm8k": 100, "multihop_qa": 100, "rag_evidence": 100, "code": 100})
        for name, d in (("tm.json", tm), ("six.json", six)):
            (tmp_path / name).write_text(json.dumps(d))
        monkeypatch.setattr(PP, "TOKEN_MATCHED", tmp_path / "tm.json")
        monkeypatch.setattr(PP, "SIX", tmp_path / "six.json")
        rec = PP.build("C1-6cat", tool_docs=tmp_path / "t.jsonl", out=tmp_path / "out")
        assert rec["n_val_blocks"] > 16                                       # no 16-sequence cap
        man = rec["manifest"]
        assert man["validation_coverage"]["training_types_without_validation"] == [
            "code", "gsm8k", "multihop_qa", "primary.x", "rag_evidence"]
        assert man["proportions"]["shares"]["fluid_of_total"]["fraction"] == 0.5 and man["proportions"]["passed"]
        assert man["tool_calling"]["quota"]["cells"]["training/c"]["episodes"] == 2
        assert len(man["documents"]["validation"]) == 42 and PP.verify_pack(tmp_path / "out" / "C1-6cat") == man


class TestPoolIdentity:
    def pool(self, tmp_path):
        a, b = server_in("training"), server_in("state_eval")
        pool = TP.assemble([task(f"tc1-{k}", f"q{k}", [a if k % 2 else b], [tool(f"{a if k % 2 else b}-t{k}")])
                            for k in range(6)])
        TP.write_pool(pool, tmp_path / "pool", sources={"toy": {"converted_sha256": "0" * 64, "tasks": 6}}, extra={})
        return tmp_path / "pool"

    def test_load_pool_verifies_every_file_and_fails_closed(self, tmp_path):
        d = self.pool(tmp_path)
        man, splits = TP.load_pool(d)
        assert set(splits) == {"training", "state_eval"} and man["files"]["training"]["tasks"] == 3
        assert man["content_sha256"] and all(len(v["sha256"]) == 64 for v in man["files"].values())
        f = d / "training.tasks.jsonl"
        good = f.read_text()
        f.write_text(good.replace('"q1"', '"q9"'))                            #: one changed byte
        with pytest.raises(TP.PoolIdentityError, match="sha256"):
            TP.load_pool(d)
        f.unlink()
        with pytest.raises(TP.PoolIdentityError, match="missing"):
            TP.load_pool(d)
        f.write_text(good)
        man_doc = json.loads((d / "pool.json").read_text())
        man_doc["source_revisions"] = [["other", "r"]]
        (d / "pool.json").write_text(json.dumps(man_doc))
        with pytest.raises(TP.PoolIdentityError, match="another split or source"):
            TP.load_pool(d)

    def test_the_session_refuses_a_plan_of_another_pool(self, tmp_path, monkeypatch):
        from stages.teacher_modes import tool_session as TS
        d = self.pool(tmp_path)
        proto = json.loads(TS.PROTOCOL.read_text())
        proto["pool"] = {**proto["pool"], "dir": str(d), "plan": str(tmp_path / "plan.json")}
        proto["lineages"]["hybrid"]["splits"] = ["training", "state_eval"]
        (tmp_path / "plan.json").write_text(json.dumps({"schema": "aadistill.stage2.teacher_modes.tool_generation_plan/v1",
                                                        "pool": {"content_sha256": "x"}, "protocol_sha256": "y",
                                                        "lineages": {"hybrid": {}}}))
        from stages.teacher_modes import tool_generation_plan as TGP
        with pytest.raises(TGP.PlanIdentityError, match="another pool content"):
            TS.checked_plan(proto, "hybrid")


class TestReferenceCache:
    def test_a_changed_or_missing_registered_file_refuses_the_reference(self, tmp_path, monkeypatch):
        import hashlib
        from stages.teacher_modes import tool_benchmarks as TB
        d = tmp_path / f"toolsandbox@{TB.REFERENCES['toolsandbox'][2][:8]}"
        (d / "a").mkdir(parents=True)
        (d / "a/x.py").write_text("content = 1\n")
        man = {"benchmark": "toolsandbox", "host": "github", "repository": TB.REFERENCES["toolsandbox"][1],
               "revision": TB.REFERENCES["toolsandbox"][2],
               "files": {"a/x.py": hashlib.sha256(b"content = 1\n").hexdigest(), "_tarball_sha256": "t"}}
        (d / "manifest.json").write_text(json.dumps(man))
        assert TB.fetch("toolsandbox", root=tmp_path)["revision"] == TB.REFERENCES["toolsandbox"][2]
        (d / "a/x.py").write_text("content = 2\n")
        with pytest.raises(TB.ReferenceCacheError, match="missing or changed"):
            TB.fetch("toolsandbox", root=tmp_path)
        (d / "a/x.py").unlink()
        with pytest.raises(TB.ReferenceCacheError):
            TB.fetch("toolsandbox", root=tmp_path)


class TestPivot:
    def row(self, tid, step, items, action, *, conv=False):
        info = {"turn": 1, "step": step}
        return {"trajectory_id": tid, ("meta_info" if conv else "info"): info,
                "responses_create_params": {"input": items, "tools": [
                    {"type": "function", "name": "lookup", "parameters": {
                        "type": "object", "properties": {"id": {"type": "string"}, "n": {"type": "integer"}},
                        "required": ["id"]}}]},
                "expected_action": action}

    def test_trajectories_rebuild_from_their_first_step_with_recorded_outputs(self, tmp_path):
        from stages.teacher_modes import tool_sources_pivot as PV
        sys_, user = {"role": "system", "content": "policy"}, {"role": "user", "content": "find A1"}
        call = {"type": "function_call", "name": "lookup", "arguments": json.dumps({"id": "A1"})}
        fc = {"type": "function_call", "name": "lookup", "arguments": json.dumps({"id": "A1"}), "call_id": "c1"}
        out = {"type": "function_call_output", "call_id": "c1", "output": "found"}
        rows = [self.row(1, 0, [sys_, user], call, conv=True),                               # 0-based steps
                self.row(1, 1, [sys_, user, fc, out], {"type": "message", "content": "ok"}, conv=True),
                self.row(2, 0, [sys_, user, {"role": "assistant", "content": "hi"}], call, conv=True),  # expert text
                self.row(3, 0, [sys_, user], call, conv=True)]                                # no recorded output
        path = tmp_path / "train.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        tasks, audit = PV.convert(path, "conversational")
        assert [t["source"]["row"] for t in tasks] == ["1"]
        t = tasks[0]
        assert t["messages"] == [{"role": "user", "content": "find A1"}] and t["system"] == "policy"
        assert t["gold_steps"] == [[[{"name": "lookup", "arguments": {"id": "A1"}}]], [[]]]
        assert list(t["backend"]["table"].values()) == ["found"] and t["category"] == "tool_selection"
        assert t["environment"].startswith("pivot-conversational:")
        assert t["source"]["verification"]["levels"] == ["protocol", "schema_validity", "recorded_execution",
                                                         "step_gold"]
        assert audit["skipped"] == {"no first-step row (system and user only) for turn 1": 1,
                                    "first gold call without a recorded output, not structured (a tool_selection "
                                    "decision step: outside the approved single-step scope)": 1}

    def test_a_structured_unanswered_first_call_becomes_a_single_step_task_only(self, tmp_path):
        """Conversational only, a structured first call only (schema_arguments), the same context, the first call
        as the only gold, no backend; an answered first call with an unanswered later one stays skipped."""
        from stages.teacher_modes import tool_episodes as TE
        from stages.teacher_modes import tool_sources_pivot as PV
        sys_, user = {"role": "system", "content": "policy"}, {"role": "user", "content": "find A1, two of them"}
        structured = {"type": "function_call", "name": "lookup", "arguments": json.dumps({"id": "A1", "n": 2})}
        plain = {"type": "function_call", "name": "lookup", "arguments": json.dumps({"id": "B2"})}
        fc = {"type": "function_call", "name": "lookup", "arguments": json.dumps({"id": "B2"}), "call_id": "c1"}
        out = {"type": "function_call_output", "call_id": "c1", "output": "found"}
        rows = [self.row(7, 0, [sys_, user], structured, conv=True),                       # -> single step
                self.row(7, 1, [sys_, user, {**structured, "call_id": "c7"}], {"type": "message", "content": "x"},
                         conv=True),                                                       # its call never answered
                self.row(8, 0, [sys_, user], plain, conv=True),                            # answered first call ...
                self.row(8, 1, [sys_, user, fc, out], structured, conv=True)]              # ... later one is not
        path = tmp_path / "train.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        tasks, audit = PV.convert(path, "conversational")
        (t,) = tasks
        assert t["episode_shape"] == TE.SINGLE_STEP and t["category"] == "schema_arguments"
        assert t["task_id"] == TE.task_id_of(*PV.DATASETS["conversational"], "7", "turn1-step1")
        assert t["gold_steps"] == [[[{"name": "lookup", "arguments": {"id": "A1", "n": 2}}]]] and t["max_turns"] == 1
        assert t["backend"] == {"kind": "none"} and t["messages"] == [{"role": "user", "content": user["content"]}]
        assert t["source"]["verification"] == {"levels": ["protocol", "schema_validity", "step_gold"],
                                               "step_gold": True, "recorded_execution": None,
                                               "task_success_verifier": None}
        assert TE.task_problems({**t, "split": "training"}) == []
        assert audit["episode_shapes"] == {TE.SINGLE_STEP: 1}
        assert audit["skipped"] == {"a gold call without a recorded output": 1}
        fc_rows = [self.row(9, 1, [sys_, user], structured)]                                # Function-Calling
        path.write_text("".join(json.dumps(r) + "\n" for r in fc_rows))
        tasks, audit = PV.convert(path, "function_calling")
        assert tasks == [] and audit["skipped"] == {"a gold call without a recorded output": 1}
