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
        tasks += [task(f"tc1-b{k}", f"r{k}", [b], [tool(f"{b}-t{k}")]) for k in range(3)]
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
        ref.add("eval-set-1", prompts=["held out"], tools=[("get_weather", {"type": "object"})], refuse_tool_names=True)
        ref.add("bench", prompts=["bench q"], tools=[("lookup", {"type": "object"}), ("search", {"a": 1})])
        s = server_in("training")
        pool = TP.assemble([task("tc1-held", "held   OUT", [s], [tool(f"{s}-x")]),
                            task("tc1-name", "q1", [s], [tool(f"{s}-get_weather", {"other": 1})]),
                            task("tc1-ident", "q2", [s], [tool(f"{s}-lookup")]),
                            task("tc1-nameonly", "q3", [s], [tool(f"{s}-search")]),
                            task("tc1-ok", "q4", [s], [tool(f"{s}-x")])], reference=ref)
        reasons = {e["task_id"]: e["reason"] for e in pool["excluded"]}
        assert reasons["tc1-held"] == "T1/T2: a eval-set-1 prompt"
        assert reasons["tc1-name"] == "T1/T2: a eval-set-1 tool name"
        assert reasons["tc1-ident"] == "T1/T2: a bench tool (name and schema)"
        assert "tc1-nameonly" not in reasons and pool["tool_name_only_matches_counted"] == {"bench": 1}

    def test_a_13_gram_near_duplicate_of_a_held_out_prompt_is_refused(self):
        ref = TP.Reference().add("bench", prompts=[long_text("w")])
        s = server_in("training")
        near = long_text("w").replace("w29", "something else")
        pool = TP.assemble([task("tc1-near", near, [s], [tool(f"{s}-x")]),
                            task("tc1-far", long_text("v"), [s], [tool(f"{s}-x")])], reference=ref)
        assert {e["task_id"]: e["reason"] for e in pool["excluded"]} == {
            "tc1-near": "T1/T2: a 13-gram near-duplicate of a bench prompt"}

    def test_training_never_keeps_a_near_duplicate_of_a_state_eval_task(self):
        a, b = server_in("state_eval"), server_in("training")
        pool = TP.assemble([task("tc1-se", long_text("u"), [a], [tool(f"{a}-x")]),
                            task("tc1-tr", long_text("u") + " extra words", [b], [tool(f"{b}-y")])])
        assert {e["task_id"]: e["reason"] for e in pool["excluded"]} == {
            "tc1-tr": "T4: a 13-gram near-duplicate of an earlier split's task"}

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
        assert t["source"]["verification"] == {"levels": ["protocol", "schema_validity", "recorded_execution",
                                                          "tool_selection"],
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

    def test_both_tool_kinds_pass_the_l40s_teacher_gate(self, monkeypatch):
        from stages.teacher_modes import l40s_check as LC
        from stages.teacher_modes import token_matched_phase as TM
        gates = []
        monkeypatch.setattr(LC, "load_config", lambda p: {"gates": {"teacher": ["generation"]}})
        monkeypatch.setattr(LC, "gate", lambda ops, root: gates.append(ops) or ["unmeasured"])
        for k in TM.TOOL_GENERATION_KINDS.values():
            with pytest.raises(TM.SE.AuthorizationError):
                TM.kind_conditions(k)
        assert gates == [["generation"], ["generation"]]


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

    def test_tool_episodes_are_cut_whole_at_the_target_in_the_frozen_order(self, tmp_path):
        from stages.teacher_modes import packed_packs as PP
        order = self.tool_file(tmp_path / "t.jsonl", 10, 30)
        train, val, q = PP.tool_documents(tmp_path / "t.jsonl", "T2507", target=100, order=order)
        assert [d.doc_id.split("|")[0] for d in train] == order[:4] and q["selected_supervised_tokens"] == 120
        assert len(val) == 2
        with pytest.raises(PP.PackBuildError):
            PP.tool_documents(tmp_path / "t.jsonl", "T2507", target=10 ** 6, order=order)

    def test_shares_are_counted_before_packing_and_refuse_outside_the_check(self):
        from stages.teacher_modes import packed_packs as PP
        check = {"per_type_relative": 0.02, "fluid_fraction": [0.49, 0.51]}
        six = {"openmath": (2, 50), "gsm8k": (2, 50), "code": (2, 50), "rag_evidence": (2, 50),
               "multihop_qa": (2, 50), PP.TOOL_TYPE: (2, 50)}
        docs, dtype = self.docs({**six, "primary.x": (6, 100)})
        s = PP.shares("C1-6cat", docs, dtype, check)
        assert s["fluid_fraction"] == 0.5 and set(s["six_types"].values()) == {100}
        docs, dtype = self.docs({**six, PP.TOOL_TYPE: (3, 50), "primary.x": (7, 100)})
        with pytest.raises(PP.PackBuildError, match="tool_calling"):
            PP.shares("C1-6cat", docs, dtype, check)

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
        monkeypatch.setattr(PP, "tool_target", lambda arm: 100)
        order = self.tool_file(tmp_path / "t.jsonl", 4, 50)
        (tmp_path / "pool").mkdir()
        (tmp_path / "pool/pool.json").write_text(json.dumps({"training_order": order}))
        design = json.loads(PP.DESIGN.read_text())
        design["stage3_packing"]["sequence_length"] = 64
        (tmp_path / "design.json").write_text(json.dumps(design))
        monkeypatch.setattr(PP, "DESIGN", tmp_path / "design.json")
        rec = PP.build("C1-6cat", tool_docs=tmp_path / "t.jsonl", out=tmp_path / "out", pool_dir=tmp_path / "pool")
        assert rec["n_val_blocks"] > 16                                       # no 16-sequence cap
        assert rec["validation_coverage"]["training_types_without_validation"] == [
            "code", "gsm8k", "multihop_qa", "primary.x", "rag_evidence"]
        assert rec["proportions_before_packing"]["fluid_fraction"] == 0.5
        assert rec["tool_documents"]["quota"]["episodes"] == 2
