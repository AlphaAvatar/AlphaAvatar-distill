"""The token-id sidecar wrapper changes nothing the generator records.

`scripts/shared/evaluation/uncapped_eval_token_ids.py` runs the in-tree
generator's `main()` with a read-only decode hook. These tests drive the REAL
generator (only the engine and the tokenizer/config loaders are stubbed) twice
-- plain and wrapped -- and require:

* byte-identical records and summaries apart from wall-clock fields, and the
  same reconstructed generation fingerprint;
* sidecar ids equal to the ids the engine emitted, including a request the
  degeneration stop truncated, each verified against its record;
* a run whose generator digest is not the expected one is left unhooked and
  says so, and the tokenizer loader is restored either way.

The engine's scheduling and kernels are not exercised -- a stub cannot -- which
is why the wrapper adds no work inside the engine loop at all.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
GENERATOR = REPO / "scripts/shared/evaluation/uncapped_eval.py"
WRAPPER = REPO / "scripts/shared/evaluation/uncapped_eval_token_ids.py"

HARNESS = r'''
import json, runpy, sys, types, importlib.util
from pathlib import Path
mode, generator, wrapper, truth_path = sys.argv[1:5]
args = sys.argv[5:]

import transformers

class Tok:
    eos_token_id = 2
    chat_template = "stub-template"
    def apply_chat_template(self, turns, tools=None, tokenize=False,
                            add_generation_prompt=True, **kw):
        return " ".join(f"<{m['role']}> {m['content']}" for m in turns) + " <a>"
    def __call__(self, text, add_special_tokens=False):
        return types.SimpleNamespace(
            input_ids=[3 + (len(w) % 40) for w in text.split()])
    def convert_tokens_to_ids(self, t):
        return 2
    def decode(self, ids, skip_special_tokens=False):
        return " ".join("<|im_end|>" if i == 2 else f"w{i}" for i in ids)

STUB_TOK = classmethod(lambda cls, *a, **k: Tok())
transformers.AutoTokenizer.from_pretrained = STUB_TOK
transformers.AutoConfig.from_pretrained = classmethod(
    lambda cls, *a, **k: types.SimpleNamespace(
        max_position_embeddings=4096, eos_token_id=2, rope_scaling=None))
def _no_gen_config(*a, **k):
    raise OSError("no generation config in the stub")
transformers.GenerationConfig.from_pretrained = _no_gen_config

TRUTH = {}

class SamplingParams:
    def __init__(self, **kw):
        self.kw = kw

class Engine:
    def __init__(self):
        self.plan, self.out, self.alive, self.order = {}, {}, [], 0
    def add_request(self, rid, prompt, params):
        n, i = params.kw["max_tokens"], self.order
        self.order += 1
        if i % 3 == 0:
            plan = [10 + (i + k) % 50 for k in range(12 + i)] + [2]
        elif i % 3 == 1:
            plan = [7, 8, 9] * 1000
        else:
            plan = [1000 + k for k in range(10 * n)]
        self.plan[rid], self.out[rid] = plan[:n], []
        self.alive.append(rid)
    def has_unfinished_requests(self):
        return bool(self.alive)
    def abort_request(self, rid):
        self.alive.remove(rid)
    def step(self):
        outs = []
        for rid in list(self.alive):
            self.out[rid].append(self.plan[rid][len(self.out[rid])])
            fin = len(self.out[rid]) == len(self.plan[rid])
            if fin:
                self.alive.remove(rid)
            reason = (("stop" if self.out[rid][-1] == 2 else "length")
                      if fin else None)
            o = types.SimpleNamespace(token_ids=list(self.out[rid]),
                                      finish_reason=reason)
            outs.append(types.SimpleNamespace(request_id=rid, finished=fin,
                                              outputs=[o]))
            TRUTH[rid] = list(self.out[rid])
        return outs

class LLM:
    def __init__(self, model, **kw):
        self.llm_engine = Engine()

vllm = types.ModuleType("vllm")
vllm.__version__ = "stub"
vllm.LLM, vllm.SamplingParams = LLM, SamplingParams
sys.modules["vllm"] = vllm

rc = None
try:
    if mode == "plain":
        sys.argv = [generator, *args]
        runpy.run_path(generator, run_name="__main__")
    else:
        spec = importlib.util.spec_from_file_location("ue_ids", wrapper)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        rc = mod.main(args)
except SystemExit as e:
    rc = e.code
Path(truth_path).write_text(json.dumps({
    "rc": rc, "truth": TRUTH,
    "loader_restored": transformers.AutoTokenizer.__dict__["from_pretrained"]
                       is STUB_TOK}))
'''


def _prompts(tmp: Path) -> list[Path]:
    files = []
    for stem, n in (("set_a", 4), ("set_b", 3)):
        path = tmp / f"{stem}.jsonl"
        rows = [{"id": f"{stem}-{i}", "group": "general",
                 "prompt_text": f"question {i} about {stem}",
                 "messages": [{"role": "user",
                               "content": f"question {i} about {stem}"}]}
                for i in range(n)]
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        files.append(path)
    return files


def _run(tmp: Path, mode: str, out: Path, extra: list[str]) -> dict:
    harness = tmp / "harness.py"
    harness.write_text(HARNESS)
    truth = tmp / f"truth_{out.name}.json"
    args = [*extra, "--model", "stub-model", "--label", "arm",
            "--context-len", "1600", "--check-every", "256",
            "--prompts", *map(str, _prompts(tmp)), "--out-dir", str(out)]
    proc = subprocess.run(
        [sys.executable, str(harness), mode, str(GENERATOR), str(WRAPPER),
         str(truth), *args],
        capture_output=True, text=True, timeout=600, cwd=str(tmp))
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(truth.read_text())


TIMING = {"seconds", "ttft_seconds"}
SUMMARY_TIMING = {"wave_seconds", "throughput", "engine", "libraries"}


def _records(path: Path) -> list[dict]:
    return [{k: v for k, v in json.loads(l).items() if k not in TIMING}
            for l in path.read_text().splitlines() if l.strip()]


def _summary(path: Path) -> dict:
    return {k: v for k, v in json.loads(path.read_text()).items()
            if k not in SUMMARY_TIMING}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    sys.path.insert(0, str(REPO / "scripts"))
    from shared.source_sets import generation_source_digest
    digest = generation_source_digest(REPO)["digest"]
    tmp = tmp_path_factory.mktemp("token_ids")
    plain = _run(tmp, "plain", tmp / "plain", [])
    wrapped = _run(tmp, "wrapped", tmp / "wrapped",
                   ["--expect-generation-source-digest", digest])
    stale = _run(tmp, "wrapped", tmp / "stale",
                 ["--expect-generation-source-digest", "0" * 64])
    return tmp, digest, plain, wrapped, stale


def test_generator_outputs_are_unchanged_by_the_wrapper(runs):
    tmp, _digest, plain, wrapped, _stale = runs
    assert plain["rc"] == wrapped["rc"] == 0
    for stem in ("set_a", "set_b"):
        assert (_records(tmp / "plain" / f"{stem}.generations.jsonl")
                == _records(tmp / "wrapped" / f"{stem}.generations.jsonl"))
        assert (_summary(tmp / "plain" / f"{stem}.json")
                == _summary(tmp / "wrapped" / f"{stem}.json"))
    #: Admission reads every *.json as a summary: the sidecar must not be one.
    assert sorted(p.name for p in (tmp / "wrapped").glob("*.json")) == [
        "set_a.json", "set_b.json"]


def test_the_reconstructed_generation_fingerprint_is_the_plain_runs(runs):
    from aadistill.initialization.planning.generation import (
        RecoveryGenerationProtocolFingerprint as FP,
    )
    tmp = runs[0]

    def fingerprint(d: Path) -> str:
        sums = [json.loads((d / f"{s}.json").read_text())
                for s in ("set_a", "set_b")]
        return FP.from_run_summaries(sums, strict=False).fingerprint

    assert fingerprint(tmp / "plain") == fingerprint(tmp / "wrapped")


def test_sidecar_ids_are_the_engines_and_verify_against_records(runs):
    tmp, digest, _plain, wrapped, _stale = runs
    truth = wrapped["truth"]
    stops = {"eos": 0, "degeneration": 0, "context_limit": 0}
    for stem in ("set_a", "set_b"):
        lines = (tmp / "wrapped" / f"{stem}.token_ids.jsonl").read_text().splitlines()
        header, rows = json.loads(lines[0]), [json.loads(l) for l in lines[1:]]
        assert header["available"] and header["exact"], header
        assert header["generation_source_digest"] == digest
        assert header["verification"]["n_problems"] == 0
        records = {r["id"]: r for r in _records(
            tmp / "wrapped" / f"{stem}.generations.jsonl")}
        assert sorted(r["id"] for r in rows) == sorted(records)
        for row in rows:
            assert row["output_token_ids"] == truth[row["request_id"]]
            rec = records[row["id"]]
            assert len(row["output_token_ids"]) == rec["generated_tokens"]
            stops[rec["stop_reason"]] += 1
    #: All three ways a record ends are covered, including the truncated one.
    assert all(stops.values()), stops


def test_a_stale_generator_runs_unhooked_and_says_why(runs):
    tmp, _digest, _plain, _wrapped, stale = runs
    assert stale["rc"] == 0
    header = json.loads((tmp / "stale" / "set_a.token_ids.jsonl")
                        .read_text().splitlines()[0])
    assert header["available"] is False and header["exact"] is False
    assert "not the expected" in header["reason_unavailable"]
    assert (_records(tmp / "plain" / "set_a.generations.jsonl")
            == _records(tmp / "stale" / "set_a.generations.jsonl"))


def test_the_tokenizer_loader_is_restored(runs):
    _tmp, _digest, _plain, wrapped, stale = runs
    assert wrapped["loader_restored"] and stale["loader_restored"]


def test_verify_rows_catches_a_wrong_id_list():
    sys.path.insert(0, str(WRAPPER.parent))
    import uncapped_eval_token_ids as W
    rec = {"id": "x", "generated_tokens": 3, "prompt_tokens": 2,
           "raw": "w5 w6 <|im_end|>", "stop_reason": "eos"}
    good = {"x": {"output_token_ids": [5, 6, 2], "prompt_token_ids": [1, 1],
                  "raw_sha256": W._sha(rec["raw"])}}
    assert W.verify_rows([rec], good, [2])["exact"]
    bad = {"x": {**good["x"], "output_token_ids": [5, 6, 7]}}
    out = W.verify_rows([rec], bad, [2])
    assert not out["exact"] and any("EOS" in p for p in out["problems"])
