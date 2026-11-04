"""Tool-calling episodes through the real episode loop with a scripted engine, and what a verified episode becomes."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from aadistill.evaluation.behavior import split_generation, template_opens_think
from stages.calibration_volume.run_battery import run_episodes
from stages.teacher_modes import tool_episodes as TE

EOS = 0
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "w", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]


def task(**over):
    t = {"task_id": "tc1-test", "category": "multi_turn", "source": {"dataset": "toy"}, "family": "fam1",
         "split": "training", "tools": TOOLS, "messages": [{"role": "user", "content": "weather in Paris?"}],
         "backend": {"kind": "replay", "table": {json.dumps({"name": "get_weather", "arguments": {"city": "Paris"}},
                                                            sort_keys=True, separators=(",", ":")): "sunny"}},
         "max_turns": 4, "final_check": {"kind": "contains_all", "values": ["sunny"]}}
    t.update(over)
    return t


class Tok:
    """Characters as ids; renders tools, structured tool calls and tool messages like the Qwen3 template does."""

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, tools=None):
        out = (f"<tools>{json.dumps(tools)}</tools>" if tools else "")
        for m in msgs:
            body = m.get("content") or ""
            for c in m.get("tool_calls") or []:
                body += "<tool_call>" + json.dumps(c["function"]) + "</tool_call>"
            out += f"<|{m['role']}|>{body}\n"
        return out + "<|assistant|><think>\n"

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[ord(c) for c in text])

    def decode(self, ids, skip_special_tokens=False):
        return "".join("<|im_end|>" if i == EOS else chr(i) for i in ids)


class Engine:
    def __init__(self, replies):
        self.replies, self.queue, self.prompts = list(replies), [], []

    def add_request(self, rid, prompt, params):
        self.prompts.append("".join(chr(i) for i in prompt["prompt_token_ids"]))
        ids = [ord(c) for c in self.replies.pop(0)] + [EOS]
        self.queue.append(SimpleNamespace(request_id=rid, finished=True,
                                          outputs=[SimpleNamespace(token_ids=ids, finish_reason="stop")]))

    def has_unfinished_requests(self):
        return bool(self.queue)

    def step(self):
        out, self.queue = self.queue, []
        return out

    def abort_request(self, rid):
        pass


def run(tasks, replies):
    eng = Engine(replies)
    eps, n, _ = run_episodes(tasks, tok=Tok(), eng=eng, make_sampling=lambda n, seed: n, limit=10 ** 6, ctx=10 ** 6,
                             stop_ids=[EOS], mode="generate", seed_base=1, system="SYS", check_every=10 ** 9,
                             degeneration_check=lambda ids: None, split_generation=split_generation,
                             template_opens_think=template_opens_think, label="t", log=lambda *a, **k: None,
                             env_factory=TE.ToolEnv.factory)
    return eps, eng


CALL = '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'


def test_a_two_turn_episode_runs_verifies_and_becomes_documents():
    eps, eng = run([task()], ["need the tool</think>\n" + CALL, "done</think>\nIt is sunny in Paris."])
    e = eps["tc1-test"]
    assert e["end"] == "episode_done" and len(e["turns"]) == 2
    assert e["turns"][0]["tool_calls"] == [{"name": "get_weather", "arguments": {"city": "Paris"}}]
    assert e["turns"][0]["responses"] == ["sunny"]
    assert "<|tool|>sunny" in eng.prompts[1] and "<tools>" in eng.prompts[0]
    assert "rerender_reproduces_generation" in e["turns"][1]
    v = TE.verify(e["task"], e["turns"], thinking=True)
    assert v.admitted, v.problems
    docs = TE.documents(e["task"], e["turns"], mode="H-T", stop_ids=[EOS])
    assert [d.doc_id for d in docs] == ["tc1-test|H-T|t0", "tc1-test|H-T|t1"]
    assert docs[0].output_ids[-1] == EOS


def test_an_off_script_call_ends_the_episode_and_is_refused():
    bad = '<tool_call>{"name": "get_weather", "arguments": {"city": "Rome"}}</tool_call>'
    eps, _ = run([task()], ["x</think>\n" + bad])
    e = eps["tc1-test"]
    assert e["end"] == "episode_done" and e["env"].off_script
    assert TE.verify(e["task"], e["turns"], thinking=True).checks["recorded_execution"] is False


def test_later_user_turns_arrive_after_an_answer():
    eps, eng = run([task(later_user_turns=["and tomorrow?"], final_check={"kind": "none"})],
                   ["a</think>\nNo call needed.", "b</think>\nStill none."])
    e = eps["tc1-test"]
    assert len(e["turns"]) == 2 and "<|user|>and tomorrow?" in eng.prompts[1]


def test_non_tool_tasks_keep_their_rendering():
    eps, eng = run([{"task_id": "plain", "messages": [{"role": "user", "content": "hi"}]}], ["x</think>\nhello"])
    assert eps["plain"]["end"] == "answered" and "<tools>" not in eng.prompts[0]
    assert "rerender_reproduces_generation" not in eps["plain"]["turns"][0]


def test_task_records_are_checked():
    assert TE.task_problems(task()) == []
    assert any("upstream assistant" in p for p in TE.task_problems(
        task(messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}])))
    assert any("backend" in p for p in TE.task_problems(task(backend={"kind": "live_api"})))
    with pytest.raises(TE.ToolTaskError):
        TE.ToolEnv(task(category="bogus"))


def test_items_carry_the_tool_calling_critical_tags():
    p = TE.TOOL_PROTOCOL
    prompt = [5, 6, 7]
    out = [11, p["think_close_id"], 12, p["open_id"], 13, p["close_id"], EOS]
    turn = {"stop_reason": "eos", "prompt_token_ids": prompt, "output_token_ids": out}
    it = TE.item_of_turn(task(), turn, 0, mode="H-T", thinking=True, domain="supporting", subtype="tool_calling@v1")
    base = len(prompt) - 1
    assert it["tags"]["tool_call_open"] == [base + 3] and it["tags"]["tool_call_close"] == [base + 5]
    assert it["tags"]["think_close"] == [base + 1] and it["tags"]["eos"] == [base + 6]
    assert it["subtype"] == "tool_calling@v1|H-T" and it["critical_group"] == "H-T"
    assert TE.item_of_turn(task(), turn, 0, mode="H-N", thinking=False, domain="supporting",
                           subtype="tool_calling@v1") is None


def test_the_real_template_renders_the_env_turn_as_a_tool_call():
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", revision="1cfa9a7208912126459214e8b04321603b3df60c",
                                            local_files_only=True)
    except Exception:
        pytest.skip("the hybrid's tokenizer is not cached here")
    env = TE.ToolEnv(task())
    obs = env.step("<think>\nneed it\n</think>\n\n" + CALL)
    msgs = [{"role": "user", "content": "weather in Paris?"}, {"role": "assistant", **obs["assistant"]}, *obs["messages"]]
    s = tok.apply_chat_template(msgs, tools=TOOLS, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    assert '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>' in s
    assert "<tool_response>\nsunny\n</tool_response>" in s
    assert tok.convert_tokens_to_ids(["<tool_call>", "</tool_call>"]) == [p for p in (TE.TOOL_PROTOCOL["open_id"],
                                                                                     TE.TOOL_PROTOCOL["close_id"])]


def test_the_cli_resolves_the_tool_env_and_records_tool_episodes():
    from stages.calibration_volume.run_battery import episode_record, resolve_env_factory
    assert resolve_env_factory("stages.teacher_modes.tool_episodes:ToolEnv.factory") is TE.ToolEnv.factory
    eps, _ = run([task()], ["x</think>\n" + CALL, "y</think>\nIt is sunny."])
    rec = episode_record("tc1-test", eps["tc1-test"], label="t", mode="generate", action_protocol="v1")
    assert rec["category"] == "multi_turn" and rec["split"] == "training" and rec["env_result"]["done"]
    assert rec["turns"][0]["prompt_token_ids"] and rec["turns"][0]["responses"] == ["sunny"]


def test_admission_turns_verified_episodes_into_documents_and_items():
    from stages.teacher_modes import tool_admission as TA
    recs = []
    for t in (task(), task(task_id="tc1-cal", split="calibration")):
        eps, _ = run([t], ["x</think>\n" + CALL, "y</think>\nIt is sunny."])  # one episode at a time
        recs += [{"task_id": tid, "turns": e["turns"]} for tid, e in eps.items()]
    res = TA.admit([task(), task(task_id="tc1-cal", split="calibration")], recs, mode="H-T", stop_ids=[EOS])
    assert [d["doc_id"] for d in res["documents"]] == ["tc1-test|H-T|t0", "tc1-test|H-T|t1"]
    assert res["census"]["training|multi_turn"]["admitted"] == 1
    # the fake engine's ids carry no </think> token id, so the calibration item is refused by the item protocol
    assert res["census"]["calibration|multi_turn"]["refused_by_check"].get("item_protocol") == 1


def test_the_pool_splits_by_atom_dedupes_isolates_and_freezes_its_order():
    from stages.teacher_modes import tool_pool as TP
    tools_b = [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}]
    base = [task(task_id=f"tc1-{k}", tools=TOOLS if k % 2 else tools_b,
                 messages=[{"role": "user", "content": f"question {k}"}]) for k in range(40)]
    dup = task(task_id="tc1-zz", messages=[{"role": "user", "content": "QUESTION   1"}])
    long_ = task(task_id="tc1-long", messages=[{"role": "user", "content": "x" * 50}])
    held = task(task_id="tc1-held", messages=[{"role": "user", "content": "held out"}])
    pool = TP.assemble(base + [dup, long_, held], reference=TP.Reference().add("eval-set-1", label="T1", prompts=["held out"]),
                       prompt_lengths=[lambda t: len(TP.prompt_key(t)) if t["task_id"] != "tc1-long" else 10 ** 6])
    reasons = {e["task_id"]: e["reason"] for e in pool["excluded"]}
    assert reasons["tc1-zz"].startswith("T5") and reasons["tc1-held"].startswith("T1") and \
        reasons["tc1-long"].startswith("length")
    fam = {t["family"]: t["split"] for t in pool["tasks"]}
    assert len(fam) == 2 and all(TP.split_of(t["atoms"][0]) == t["split"] for t in pool["tasks"])
    again = TP.assemble(list(reversed(base)), prompt_lengths=())
    assert again["training_order"] == TP.assemble(base)["training_order"]


def test_the_packed_kinds_are_registered_and_refuse_without_a_grant_cap():
    from stages.teacher_modes import token_matched_phase as TM
    new = ("tool_generation", "tool_generation_t2507", "packed_search", "train_6cat_a1", "train_6cat_a2", "train_6cat_c1")
    assert set(new) <= set(TM.STUDY.kinds) and set(new) <= set(TM.DRIVERS) and set(new) <= set(TM.KIND_RECORDS)
    for k in new:
        assert TM.kind_cap_problems(k, 1.0), k                      # the grant names no cap for any of them
        assert all(isinstance(r, str) for r in TM.KIND_RECORDS[k])
    assert TM.main(["issue", "--kind", "packed_search", "--run-id", "x", "--expected-usd", "1",
                    "--hard-cap-usd", "1", "--dry-run"]) == 11
    for s in TM.STUDY.scripts:
        assert (TM.REPO_ROOT / s).is_file(), s


def test_binding_fills_a_built_packs_files_and_steps(tmp_path, monkeypatch):
    from aadistill.data.token_documents import TokenDocument
    from stages.teacher_modes import packed_packs as PP
    from stages.teacher_modes import packed_training_plan as PTP
    docs = [TokenDocument(f"d{i}", (5, 6, 7), (8, 9, 0)) for i in range(30)]
    sel = {"arm": "C1-6cat", "mode": "T2507", "teacher": "thinking_2507", "train": docs[:25], "validation": docs[25:],
           "data_type": {d.doc_id: "t" for d in docs}, "unit_of": {d.doc_id: d.doc_id for d in docs},
           "missing_tasks_read_from_records": [], "sources": {}, "tool": {"quota": None, "approved_shortfalls": []}}
    monkeypatch.setattr(PP, "admission_verdict", lambda *a: {"admitted": True, "unapproved": [],
                                                             "approved_exceptions": []})  #: test_data_admission's
    rep = {"arm": "C1-6cat", "selection_sha256": PP.selection_identity("C1-6cat", sel)}   #: this selection's report
    rep["report_sha256"] = PP.canonical_sha(rep)
    adm = PP.admit("C1-6cat", sel, rep, tmp_path)
    rec = PP.pack(sel["train"], sel["validation"], adm["path"], tmp_path / "C1-6cat")  #: a built, admitted pack
    plan = PTP.bind(PTP.packed_units(), tmp_path)
    c1 = [u for u in plan["units"] if u["arm"] == "C1-6cat"]
    a1 = [u for u in plan["units"] if u["arm"] == "A1-6cat"]
    assert all(isinstance(u["config"]["schedule"]["total_steps"], int) for u in c1)
    assert all(u["config"]["lineage"]["data"][0]["files_sha256"]["blocks.npz"] != PTP.TO_BUILD for u in c1)
    assert all(u["config"]["lineage"]["data"][0]["packed_content_sha256"]
               == rec["data_admission"]["packed_content_sha256"] for u in c1)
    assert all(u["config"]["schedule"]["total_steps"] == PTP.TO_BUILD for u in a1)    # its pack is not built
    assert plan["seed_order"] == [20260726, 20260801]


def test_a_toucan_row_becomes_task_material_only():
    from stages.teacher_modes import tool_sources_toucan as TS
    row = {"uuid": "u1", "subset_name": "multi-turn", "target_tools": "Weather Server::get_weather",
           "tools": json.dumps(TOOLS),
           "messages": json.dumps([
               {"role": "user", "content": "weather in Paris?"},
               {"role": "assistant", "content": "Let me check."},
               {"role": "tool_call", "content": "{'name': 'get_weather', 'arguments': '{\"city\": \"Paris\"}'}"},
               {"role": "tool_response", "content": "sunny"},
               {"role": "assistant", "content": "It is sunny."},
               {"role": "user", "content": "and London?"}])}
    t = TS.task_of(row)
    assert t["messages"] == [{"role": "user", "content": "weather in Paris?"}]        # no upstream assistant turn
    assert t["later_user_turns"] == ["and London?"] and t["category"] == "multi_turn"
    assert list(t["backend"]["table"].values()) == ["sunny"]
    assert t["servers"] == [] and t["unattributed_tools"] == ["get_weather"]       # no "<slug>-" prefix to read
    t = {**t, "family": "f", "split": "training"}
    assert TE.task_problems(t) == []
    env = TE.ToolEnv(t)
    assert env.step(CALL)["responses"] == ["sunny"]
    fc = TE.final_check(t)
    assert fc([{"raw": "x</think>" + CALL}]) and not fc([{"raw": "x</think>no call"}])


# --- single_step_gold_tool_call@v1 (maintainer 2026-11-04, review of 647eea0e, item 4: a $0 implementation) ---------

SA_TOOLS = [{"type": "function", "function": {"name": "book", "description": "b", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}, "nights": {"type": "integer"},
                                     "rooms": {"type": "array", "items": {"type": "string"}}},
    "required": ["city", "nights"], "additionalProperties": False}}}]
GOLD = {"name": "book", "arguments": {"city": "Paris", "nights": 2}}
#: the teacher writes the arguments in ITS order: the gold matches after canonicalization, the target stays its own
SCALL = '<tool_call>{"name": "book", "arguments": {"nights": 2, "city": "Paris"}}</tool_call>'


def single(**over):
    t = {"task_id": "tc1-single", "category": "schema_arguments", "episode_shape": TE.SINGLE_STEP,
         "source": {"dataset": "toy"}, "family": "fam1", "split": "training", "tools": SA_TOOLS,
         "messages": [{"role": "user", "content": "book Paris for two nights"}], "backend": {"kind": "none"},
         "gold_steps": [[[GOLD]]], "final_check": {"kind": "none"}, "max_turns": 1}
    t.update(over)
    return t


class TestSingleStepGoldToolCall:
    def test_the_teachers_call_ends_the_sample_and_is_verified_against_the_gold_only(self):
        eps, eng = run([single()], ["plan</think>\n" + SCALL, "never asked"])
        e = eps["tc1-single"]
        assert e["end"] == "episode_done" and len(e["turns"]) == 1 and len(eng.prompts) == 1
        assert e["turns"][0]["responses"] == [] and e["turns"][0]["tool_calls"] == [GOLD]
        v = TE.verify(e["task"], e["turns"], thinking=True)
        assert v.admitted, v.problems
        assert {k: v.checks[k] for k in ("protocol", "schema_validity", "step_gold")} == dict.fromkeys(
            ("protocol", "schema_validity", "step_gold"), True)
        assert v.checks["recorded_execution"] is v.checks["tool_selection"] is v.checks["task_success"] is None
        assert v.verified_levels() == TE.verification_levels(e["task"]) == ["protocol", "schema_validity", "step_gold"]
        (d,) = TE.documents(e["task"], e["turns"], mode="H-T", stop_ids=[EOS])
        assert list(d.output_ids) == list(e["turns"][0]["output_token_ids"]) and d.output_ids[-1] == EOS
        assert '"nights": 2, "city": "Paris"' in "".join(chr(i) for i in d.output_ids[:-1])  #: the teacher's, not the gold's
        assert d.meta["episode_shape"] == TE.SINGLE_STEP and not any(d.target_mask[:len(d.prompt_ids)])

    @pytest.mark.parametrize("reply,mode,level", [
        ('x</think>\n<tool_call>{"name": "book", "arguments": {"city": "Rome", "nights": 2}}</tool_call>', True,
         "step_gold"),
        ('x</think>\n<tool_call>{"name": "book", "arguments": {"city": "Paris", "nights": "two"}}</tool_call>', True,
         "schema_validity"),
        ("x</think>\nI booked it.", True, "protocol"),                                  #: an answer, no call
        ("x</think>\n" + SCALL + "\nDone, booked!", True, "protocol"),                   #: text after the call
        ("x</think>\n" + SCALL + SCALL, True, "step_gold"),                              #: two calls, gold is one
        ("x</think>\n" + SCALL, False, "protocol"),                                     #: </think> in non-thinking
        (SCALL, True, "protocol")])                                                      #: think block never closed
    def test_each_failure_refuses(self, reply, mode, level):
        eps, _ = run([single()], [reply])
        v = TE.verify(eps["tc1-single"]["task"], eps["tc1-single"]["turns"], thinking=mode)
        assert not v.admitted and v.checks[level] is False, v.problems

    def test_non_thinking_mode_admits_a_bare_call(self):
        eps, _ = run([single()], [SCALL])
        assert TE.verify(eps["tc1-single"]["task"], eps["tc1-single"]["turns"], thinking=False).admitted

    def test_a_degeneration_stop_is_refused(self):
        turn = {"raw": "x</think>\n" + SCALL, "stop_reason": "degeneration", "responses": []}
        assert TE.verify(single(), [turn], thinking=True).checks["protocol"] is False

    @pytest.mark.parametrize("over,why", [
        ({"category": "multi_turn"}, "registered for"), ({"category": "error_recovery"}, "registered for"),
        ({"category": "mcp_agent"}, "registered for"), ({"category": "tool_selection"}, "registered for"),
        ({"max_turns": 2}, "max_turns"), ({"backend": {"kind": "replay", "table": {}}}, "never executed"),
        ({"final_check": {"kind": "contains_all", "values": ["ok"]}}, "final check"),
        ({"later_user_turns": ["and then?"]}, "later user"),
        ({"gold_steps": [[[GOLD, GOLD]]]}, "exactly one call"), ({"gold_steps": [[[GOLD]], None]}, "one step"),
        ({"gold_steps": [[[{"name": "pay", "arguments": {}}]]]}, "offered tool"),
        ({"episode_shape": "multi_step_gold@v1"}, "not registered")])
    def test_the_shape_is_refused_outside_its_registration(self, over, why):
        assert TE.task_problems(single()) == []
        bad = TE.task_problems(single(**over))
        assert any(why in b for b in bad), bad

    def test_the_registered_shape_is_the_executed_one(self):
        from stages.teacher_modes import packed_packs as PP
        shapes = json.loads(PP.SIX.read_text())["protocol"]["episode_shapes"]
        assert set(shapes) - {"_contract"} == set(TE.EPISODE_SHAPES)
        reg = shapes[TE.SINGLE_STEP]
        assert reg["categories"] == list(TE.EPISODE_SHAPES[TE.SINGLE_STEP]["categories"]) == ["schema_arguments"]
        assert reg["max_turns"] == TE.EPISODE_SHAPES[TE.SINGLE_STEP]["max_turns"] == 1
        unverified = sorted(k for k, x in reg["verification"].items() if x is None)
        assert unverified == ["recorded_execution", "task_success", "tool_selection"]
        assert not set(unverified) & set(TE.verification_levels(single()))

    def test_admission_counts_the_shape_and_its_verified_levels(self):
        from stages.teacher_modes import tool_admission as TA
        recs = []
        for t in (single(), single(task_id="tc1-cal", split="calibration")):
            eps, _ = run([t], [SCALL])                                             #: non-thinking: a bare call
            recs += [{"task_id": tid, "turns": e["turns"]} for tid, e in eps.items()]
        res = TA.admit([single(), single(task_id="tc1-cal", split="calibration")], recs, mode="H-N", stop_ids=[EOS])
        c = res["census"]["training|schema_arguments"]
        assert c["admitted"] == 1 and c["admitted_by_shape"] == {TE.SINGLE_STEP: 1}
        assert c["admitted_by_verification"] == {"protocol+schema_validity+step_gold": 1}
        (it,) = res["items"]
        assert it["episode_shape"] == TE.SINGLE_STEP and it["split"] == "calibration"
        assert it["tags"]["eos"] == [it["n_tokens"] - 2] and len(it["tags"]["assistant"]) == len(SCALL) + 1

    def test_the_pool_tiers_single_steps_after_full_episodes(self):
        from stages.teacher_modes import tool_pool as TP
        full = [task(task_id=f"tc1-f{k}", category="schema_arguments", messages=[{"role": "user", "content": f"q {k}"}])
                for k in range(30)]
        ones = [single(task_id=f"tc1-s{k}", tools=TOOLS, gold_steps=[[[{"name": "get_weather",
                                                                       "arguments": {"city": "Paris"}}]]],
                       messages=[{"role": "user", "content": f"single {k}"}]) for k in range(30)]
        dup = single(task_id="tc1-a-dup", tools=TOOLS, gold_steps=[[[{"name": "get_weather",
                                                                       "arguments": {"city": "Paris"}}]]],
                     messages=[{"role": "user", "content": "q 3"}])       #: sorts first, same prompt and family as f3
        pool = TP.assemble(full + ones + [dup])
        reasons = {e["task_id"]: e["reason"] for e in pool["excluded"]}
        assert reasons["tc1-a-dup"] == "T5: duplicates tc1-f3"                  #: the full episode is kept
        order = pool["training_order"]
        shape = {t["task_id"]: TE.episode_shape(t) for t in pool["tasks"]}
        tiers = [shape[t] for t in order]
        assert tiers == sorted(tiers, key=TP.SHAPE_TIERS.get) and set(tiers) == set(TP.SHAPE_TIERS)
        assert [t for t in order if shape[t] == TE.FULL_EPISODE] == TP.assemble(full)["training_order"]
        assert pool["episode_shapes"]["training"][f"{TE.SINGLE_STEP}|schema_arguments"] == 30


# --- V1's tail rule on every turn that makes a call (tool_verification@v2, maintainer review of 2468c1aa) ----------

class TestV1TailRule:
    def test_text_after_a_full_episodes_call_is_refused(self):
        eps, _ = run([task()], ["x</think>\n" + CALL + "\nLet me check that.", "y</think>\nIt is sunny."])
        v = TE.verify(eps["tc1-test"]["task"], eps["tc1-test"]["turns"], thinking=True)
        assert not v.admitted and v.checks["protocol"] is False
        assert any("turn 0: text after its last call" in p for p in v.problems)

    def test_an_answer_turn_and_whitespace_after_a_call_are_not_subject_to_it(self):
        eps, _ = run([task()], ["x</think>\n" + CALL + "\n  \n", "y</think>\nIt is sunny, and more prose follows."])
        assert TE.verify(eps["tc1-test"]["task"], eps["tc1-test"]["turns"], thinking=True).admitted
        assert TE.text_after_last_call("no call, just an answer") is None
        assert TE.text_after_last_call(CALL + "<|im_end|>") == ""
        assert TE.text_after_last_call(CALL + " more " + CALL + "\n<|endoftext|>") == "\n"

    def test_the_registered_version_is_the_executed_one(self):
        from stages.teacher_modes import packed_packs as PP
        ver = json.loads(PP.SIX.read_text())["protocol"]["verification"]
        assert ver["version"] == TE.VERIFICATION_VERSION == "tool_verification@v2"
        from stages.teacher_modes import tool_admission as TA
        assert TA.admit([], [], mode="H-T", stop_ids=[EOS])["verification_version"] == TE.VERIFICATION_VERSION
