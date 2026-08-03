"""End-to-end CPU dress rehearsal of the recovery-corpus production path.

Drives the real `build_recovery_corpus.py` -> `build_token_ladder.py` ->
`validate_corpus_gate.py` chain with a stub decode backend, so every step that
will run on the paid pod is exercised here first: budget derivation, acceptance,
session construction, rendering, packing, the ladder cut, and the gate's own
checks. Only the teacher's weights are absent (P8 — start cheap).

The stub deliberately produces a *mix*: candidates that terminate naturally,
candidates that run past their budget, and sessions long enough to force
terminal truncation, so the rejection and truncation paths are covered rather
than merely reachable.
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.rollout.engines import Engine, _caps  # noqa: E402

TEACHER = "Qwen/Qwen3-4B-Thinking-2507"
REVISION = "768f209d9ea81521153ed38c47d515654e938aea"
BLOCK_LEN = 8192


def load_script(name: str, relative: str):
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(
            TEACHER, revision=REVISION, local_files_only=True)
    except Exception:
        pytest.skip("teacher tokenizer not in local HF cache")


WORDS = ("consider the quantity carefully then divide subtract multiply and "
         "compare each intermediate result against the stated constraint "
         "before concluding anything final about the value we seek").split()


class StubEngine(Engine):
    """Well-formed teacher-shaped completions, varying with the seed."""

    name = "stub"

    def __init__(self, tokenizer):
        self.tok = tokenizer

    def _raw_generate(self, prompts, *, max_new_tokens, stop_ids, greedy,
                      temperature, top_p, top_k, min_p=0.0, seed, logprobs=False):
        caps = _caps(max_new_tokens, len(prompts))
        rows = []
        for i, cap in enumerate(caps):
            rng = random.Random((seed or 0) + i * 7919)
            # Non-repeating prose: a cyclic filler would trip the degeneration
            # detector and reject every candidate for the wrong reason.
            # The 9000 arm always overruns an 8,192-token budget, so the
            # length-limited rejection path is covered rather than merely reachable.
            n_words = rng.choice([60, 140, 260, 520, 9000])
            body = " ".join(rng.choice(WORDS) for _ in range(n_words))
            text = f"{body}\n</think>\n\nThe answer is {rng.randint(1, 99)}.<|im_end|>"
            ids = self.tok(text, add_special_tokens=False).input_ids
            if len(ids) > cap:
                rows.append((ids[:cap], "length"))
            else:
                rows.append((ids, "stop"))
        return rows


@pytest.fixture(scope="module")
def pipeline(tokenizer, tmp_path_factory):
    """Run builder -> ladder -> gate once; the tests assert on the artifacts."""
    out = tmp_path_factory.mktemp("corpus")
    corpus_dir, packed_dir = out / "gate", out / "packed"

    builder = load_script("build_recovery_corpus",
                          "scripts/rollout/build_recovery_corpus.py")
    builder.build_engine = lambda args, tok: StubEngine(tok)

    argv = [
        "build_recovery_corpus.py",
        "--model", f"{TEACHER}@{REVISION}",
        "--limit-per-type", "8",
        "--n", "4",
        "--block-len", str(BLOCK_LEN),
        "--batch-size", "8",
        "--engine", "stub",
        "--out", str(corpus_dir),
    ]
    # `--engine` is choice-constrained; the stub is injected, so widen it here.
    original_parse = builder.argparse.ArgumentParser.parse_args

    def parse(self, args=None, namespace=None):
        for action in self._actions:
            if action.dest == "engine":
                action.choices = None
        return original_parse(self, args, namespace)

    builder.argparse.ArgumentParser.parse_args = parse
    try:
        sys.argv = argv
        builder.main()
    finally:
        builder.argparse.ArgumentParser.parse_args = original_parse

    ladder = load_script("build_token_ladder", "scripts/data/build_token_ladder.py")
    sys.argv = [
        "build_token_ladder.py",
        "--sessions", str(corpus_dir / "sessions.jsonl"),
        "--model", f"{TEACHER}@{REVISION}",
        "--block-len", str(BLOCK_LEN),
        "--ladder", "1000,3000",
        "--out", str(packed_dir),
    ]
    ladder.main()

    return corpus_dir, packed_dir


def test_corpus_was_produced(pipeline):
    corpus, _ = pipeline
    manifest = json.loads((corpus / "manifest.json").read_text())
    sessions = [json.loads(l) for l in open(corpus / "sessions.jsonl") if l.strip()]
    assert manifest["complete"] is True
    assert manifest["decoding"]["n"] == 4
    assert manifest["decoding"]["temperature"] == 0.6
    assert manifest["decoding"]["top_p"] == 0.95
    assert manifest["decoding"]["top_k"] == 20
    assert manifest["decoding"]["min_p"] == 0.0
    assert manifest["session_limit_tokens"] == BLOCK_LEN
    assert manifest["turn_expansion"]["enabled"] is True
    assert sessions, "no session survived the stub run"


def test_all_four_candidates_are_retained_with_reasons(pipeline):
    corpus, _ = pipeline
    records = [json.loads(l) for l in open(corpus / "candidates.jsonl") if l.strip()]
    assert records
    for r in records:
        assert len(r["candidates"]) == 4
        seeds = [c["seed"] for c in r["candidates"]]
        assert len(set(seeds)) == 4, "candidate seeds collided"
        for c in r["candidates"]:
            assert {"raw", "tokens", "reason", "accepted", "correctness_verdict",
                    "length_limited", "finished", "hit_cap"} <= set(c)
        # Rejected candidates are kept, not dropped.
        assert all(isinstance(c["reason"], str) for c in r["candidates"])


def test_no_public_target_fallback(pipeline):
    corpus, _ = pipeline
    sessions = [json.loads(l) for l in open(corpus / "sessions.jsonl") if l.strip()]
    for s in sessions:
        assert "target_source" not in s
        assert s["messages"][-1]["role"] == "assistant"
        assert s["messages"][-1]["reasoning_content"]
        assert s["candidate_index"] is not None
    records = [json.loads(l) for l in open(corpus / "candidates.jsonl") if l.strip()]
    kept = {s["id"] for s in sessions}
    for r in records:
        if r["selected_index"] is None:
            assert r["id"] not in kept, "a prompt with no accepted candidate leaked in"


def test_length_limited_candidates_are_rejected(pipeline):
    corpus, _ = pipeline
    records = [json.loads(l) for l in open(corpus / "candidates.jsonl") if l.strip()]
    saw = False
    for r in records:
        for c in r["candidates"]:
            if c["length_limited"]:
                saw = True
                assert not c["accepted"]
                assert c["reason"] == "length_limited"
    assert saw, "stub did not exercise the length-limited path"


def test_every_session_carries_the_mandatory_system_prompt(pipeline, tokenizer):
    from aadistill.data.sessions import SYSTEM_DEFAULT

    corpus, _ = pipeline
    sessions = [json.loads(l) for l in open(corpus / "sessions.jsonl") if l.strip()]
    for s in sessions:
        assert s["messages"][0]["role"] == "system"
        assert s["messages"][0]["content"] == SYSTEM_DEFAULT
        assert sum(m["role"] == "system" for m in s["messages"]) == 1


def test_every_session_fits_the_limit(pipeline):
    corpus, _ = pipeline
    sessions = [json.loads(l) for l in open(corpus / "sessions.jsonl") if l.strip()]
    for s in sessions:
        assert s["n_rendered_tokens"] <= BLOCK_LEN
        assert s["n_supervised_tokens"] > 0


def test_ladder_is_nested_and_type_balanced(pipeline):
    _, packed = pipeline
    ladder = json.loads((packed / "ladder.json").read_text())
    reachable = [r for r in ladder["rungs"] if r["reachable"]]
    assert reachable
    for a, b in zip(reachable, reachable[1:]):
        assert a["n_blocks"] <= b["n_blocks"]
        assert a["actual_supervised_tokens"] <= b["actual_supervised_tokens"]
    for r in reachable:
        assert r["actual_supervised_tokens"] >= r["target_supervised_tokens"]


def test_packed_blocks_load_and_are_well_formed(pipeline):
    from aadistill.data.sessions import load_packed_blocks

    _, packed = pipeline
    ids, ce, content = load_packed_blocks(packed)
    audit = [json.loads(l) for l in open(packed / "audit.jsonl") if l.strip()]
    assert ids.shape[0] == len(audit)
    assert ids.shape[1] == BLOCK_LEN
    for i, row in enumerate(audit):
        n = row["unpadded_length"]
        assert not ce[i][n:].any()
        assert not content[i][n:].any()
        assert content[i][:n].all()
        assert int(ce[i].sum()) == row["supervised_tokens"]


def test_gate_validator_passes_on_the_dress_rehearsal(pipeline):
    corpus, packed = pipeline
    gate = load_script("validate_corpus_gate", "scripts/data/validate_corpus_gate.py")
    sys.argv = [
        "validate_corpus_gate.py",
        "--corpus", str(corpus),
        "--packed", str(packed),
        "--model", f"{TEACHER}@{REVISION}",
        "--block-len", str(BLOCK_LEN),
        "--skip-logits",
    ]
    with pytest.raises(SystemExit) as exc:
        gate.main()
    report = json.loads((corpus / "gate_report.json").read_text())
    failures = [(t, n) for t, row in report["types"].items()
                for n, ok in row.items() if not ok]
    assert exc.value.code == 0, f"gate failed: {failures}"
    assert report["gate"] == "pass"


def test_turn_expansion_produces_multi_turn_context_examples(pipeline):
    """A multi-turn source becomes one example per eligible turn."""
    corpus, _ = pipeline
    records = [json.loads(l) for l in open(corpus / "candidates.jsonl") if l.strip()]
    by_source = {}
    for r in records:
        by_source.setdefault(r["source_id"], []).append(r)
    # Every example id encodes the turn it targets.
    for r in records:
        assert r["id"] == f"{r['source_id']}#t{r['turn_index']}"
        assert r["turn_index"] >= 1
    expanded = [r for r in records if r["n_context_assistant_turns"] > 0]
    assert expanded, "no turn-expanded example in the sample"
    for r in expanded:
        # A turn-expanded example's prompt must already contain prior assistant
        # turns as context.
        assert r["rendered_prompt"].count("<|im_start|>assistant") >= 1


def test_only_the_teacher_turn_is_supervised(pipeline, tokenizer):
    """Preceding original assistant turns are context, never training targets."""
    from aadistill.data.sessions import render_session

    corpus, _ = pipeline
    sessions = [json.loads(l) for l in open(corpus / "sessions.jsonl") if l.strip()]
    multi = [s for s in sessions if s["n_context_assistant_turns"] > 0]
    assert multi, "no multi-turn session survived to check"
    for s in multi:
        r = render_session(tokenizer, s, block_len=BLOCK_LEN)
        supervised = tokenizer.decode(
            [t for t, m in zip(r.body_ids, r.body_mask) if m])
        # The teacher's own answer is supervised ...
        assert s["messages"][-1]["content"][:40] in supervised
        # ... and every earlier original assistant response is not.
        for prior in s["messages"][:-1]:
            if prior["role"] == "assistant" and len(prior.get("content", "")) > 40:
                assert prior["content"][:40] not in supervised
        # Exactly one assistant segment is supervised.
        assert supervised.count("<|im_end|>") == 1


def test_session_order_anchor_replays_an_existing_pack(pipeline, tmp_path):
    """`--session-order` must reproduce the anchor's session order.

    The Experiment 2 cleaning arm re-cuts a pack after dropping sessions, and
    the largest-remainder interleave is a global function of the session set —
    so without an anchor the rung prefix holds a substantially different prompt
    set and the comparison stops being about target quality.
    """
    corpus, packed = pipeline

    anchor_order = []
    with (packed / "audit.jsonl").open() as f:
        for line in f:
            for s in json.loads(line)["sessions"]:
                anchor_order.append(s["session_id"])
    anchor_file = tmp_path / "anchor.txt"
    anchor_file.write_text("\n".join(reversed(anchor_order)) + "\n")

    out = tmp_path / "anchored"
    ladder = load_script("build_token_ladder", "scripts/data/build_token_ladder.py")
    sys.argv = [
        "build_token_ladder.py",
        "--sessions", str(corpus / "sessions.jsonl"),
        "--model", f"{TEACHER}@{REVISION}",
        "--block-len", str(BLOCK_LEN),
        "--ladder", "1000,3000",
        "--session-order", str(anchor_file),
        "--out", str(out),
    ]
    ladder.main()

    replayed = []
    with (out / "audit.jsonl").open() as f:
        for line in f:
            for s in json.loads(line)["sessions"]:
                replayed.append(s["session_id"])
    assert sorted(replayed) == sorted(anchor_order)
    # Packing groups by system prompt and blocks are then mixture-ordered, so
    # the emitted order is not the anchor verbatim; what must hold is that the
    # anchor drove session order, i.e. the reversed anchor is closer to the
    # replay than the original order is.
    rank = {sid: i for i, sid in enumerate(reversed(anchor_order))}
    replay_rank = {sid: i for i, sid in enumerate(replayed)}
    reversed_corr = sum(abs(rank[s] - replay_rank[s]) for s in replayed)
    forward_rank = {sid: i for i, sid in enumerate(anchor_order)}
    forward_corr = sum(abs(forward_rank[s] - replay_rank[s]) for s in replayed)
    assert reversed_corr < forward_corr


def test_session_order_anchor_is_optional_and_default_free(pipeline):
    """Omitting the anchor keeps the seed-free stratified interleave."""
    _, packed = pipeline
    ladder = json.loads((packed / "ladder.json").read_text())
    assert "interleave" in ladder["ordering"]
