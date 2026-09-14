"""The generation path must render every tool prompt in the frozen battery.

C1 attempt 17 trained all six formal probes over ten hours, entered the
confirmation evaluation, and raised on the first tool prompt before generating a
single token:

    ValueError: Tools should either be a JSON schema, or a callable function
    with type hints and a docstring suitable for auto-conversion to a schema.

$11.19 and no measurement. The battery stores `tools` as a JSON string;
`apply_chat_template` needs a list.

Two renderers over one battery is what hid it. `scripts/data/battery_render.py`
parses the string; `scripts/evaluation/uncapped_eval.py` did not. The
pre-provider renderer-parity gate reported 7/7 groups PASS over 190 prompts and
was telling the truth about the SCORING renderer — which is not the one that
generates.

So this exercises the generation path's own conversion, over the real frozen
battery and the real frozen tokenizer, on CPU. It is deliberately small: the
root cause is known and the protection needed is a regression around the
renderer that failed, not another readiness mechanism.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BATTERY = REPO / "artifacts/stage3/c1_confirmation_v1"
TOKENIZER = REPO / "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint"


def evaluator():
    """The real module, loaded by path — it is a script, not a package member."""
    spec = importlib.util.spec_from_file_location(
        "uncapped_eval", REPO / "scripts/evaluation/uncapped_eval.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("uncapped_eval", mod)
    spec.loader.exec_module(mod)
    return mod


def samples(group: str) -> list[dict]:
    #: ASSERTED, never skipped. Both the battery and the evaluation tokenizer
    #: are staged assets — `c1_confirmation_v1` as a local asset and the
    #: tokenizer through `C1_EVAL_TOKENIZER` — so a pod that lacks either cannot
    #: evaluate anything, and a preflight check that skipped would be a check
    #: the pod silently did not run.
    path = BATTERY / f"{group}.jsonl"
    assert path.is_file(), f"{path} was not staged; the session has no battery"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def tokenizer():
    assert (TOKENIZER / "tokenizer.json").is_file(), (
        f"{TOKENIZER} was not staged; the evaluation cannot render a prompt")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(TOKENIZER))
    #: Assert the identity rather than that the call returned something: a
    #: config-only directory yields a 1-token vocab instead of raising, and this
    #: repository has been bitten by that twice.
    assert tok.vocab_size > 100_000, f"loaded a {tok.vocab_size}-token vocab"
    return tok


def test_all_hundred_tool_samples_parse_to_a_list_of_schemas():
    """Property 1 and 2: every one parses, and to something a template accepts."""
    ev = evaluator()
    rows = samples("tool")
    assert len(rows) == 100, len(rows)
    for s in rows:
        parsed = ev.tool_schemas(s)
        assert isinstance(parsed, list) and parsed, (s["id"], parsed)
        for t in parsed:
            assert isinstance(t, dict), (s["id"], t)
            assert t.get("name") and "parameters" in t, (s["id"], t)


def test_every_tool_prompt_renders_through_the_generation_path():
    """Property 3: the real call that failed, over all 100, on the real template.

    Not a reimplementation of the call — the same `apply_chat_template` with the
    same arguments the evaluator passes.
    """
    ev = evaluator()
    tok = tokenizer()
    for s in samples("tool"):
        turns = [m for m in s["messages"] if m["role"] != "assistant"]
        prompt = tok.apply_chat_template(turns, tools=ev.tool_schemas(s),
                                         tokenize=False,
                                         add_generation_prompt=True)
        assert prompt and isinstance(prompt, str), s["id"]
        #: The tools really reached the rendered prompt; a template that
        #: silently dropped them would render fine and evaluate nothing.
        assert ev.tool_schemas(s)[0]["name"] in prompt, s["id"]


def test_the_repair_leaves_every_non_tool_group_untouched():
    """Property 4: byte-identical prompts where there were no tools to parse."""
    ev = evaluator()
    tok = tokenizer()
    checked = 0
    for group in ("gsm8k", "knowledge", "math_verified", "multihop", "rag",
                  "code"):
        for s in samples(group)[:5]:
            assert ev.tool_schemas(s) == s.get("tools")
            turns = [m for m in s["messages"] if m["role"] != "assistant"]
            before = tok.apply_chat_template(turns, tools=s.get("tools"),
                                             tokenize=False,
                                             add_generation_prompt=True)
            after = tok.apply_chat_template(turns, tools=ev.tool_schemas(s),
                                            tokenize=False,
                                            add_generation_prompt=True)
            assert before == after, (group, s["id"])
            checked += 1
    assert checked >= 25, checked


def test_the_two_renderers_now_agree_on_what_a_tool_schema_is():
    """Property 5: the same interpretation, taken from the renderer that had it.

    This is the assertion the parity gate could not make, because it compares
    one renderer against a frozen expectation rather than the two renderers
    against each other.
    """
    ev = evaluator()
    for s in samples("tool"):
        raw = s.get("tools")
        battery_side = json.loads(raw) if isinstance(raw, str) else raw
        assert ev.tool_schemas(s) == battery_side, s["id"]

    source = (REPO / "scripts/data/battery_render.py").read_text()
    assert "json.loads(tools) if isinstance(tools, str) else tools" in source, (
        "the battery renderer's conversion moved; the generation path copied "
        "its semantics and must follow it")


def test_the_generation_path_no_longer_passes_the_raw_field():
    """Mutation guard: the exact line attempt 17 died on must not come back."""
    source = (REPO / "scripts/evaluation/uncapped_eval.py").read_text()
    assert 'tools=s.get("tools")' not in source, (
        "apply_chat_template is being handed the raw field again")
    assert "tools=tool_schemas(s)" in source
