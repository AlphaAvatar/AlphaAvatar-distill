#!/usr/bin/env python3
"""One vLLM engine run in its own process, on a GPU no earlier check still holds ($0 on a host).

    # the child, in the vLLM environment (it imports vllm and the standard library only)
    /opt/vllm/bin/python scripts/shared/validation/vllm_check.py --request <req.json> --out <out.json>

    # the parent, from any interpreter that has NOT initialized CUDA
    from shared.validation import vllm_check as VC
    result = VC.run_isolated("/opt/vllm/bin/python", request, workdir)

An engineering validation that trains or runs a teacher in PyTorch and then
starts vLLM in the same session must not let the first hold GPU memory while
the second allocates its KV cache: vLLM sizes its cache from the memory it
finds free, so a teacher still resident in the parent shrinks the cache or
refuses the engine, and the vLLM measurement then describes the leftover
rather than the engine. The boundary is a process boundary:

* the PyTorch checks run in a child that EXITS before vLLM starts (the caller's
  job: a parent that imported torch.cuda holds a context of its own);
* `run_isolated` reads the GPU with ``nvidia-smi`` before the engine starts and
  refuses while any compute process is listed or more than
  ``max_used_mib`` is in use; the reading is recorded either way;
* the engine is this file, in the vLLM interpreter, and it reads token ids and
  returns token ids (no text round trip): what a request's ``prompt_token_ids``
  were, what the engine generated, and the log-probabilities it reported.

A request's ``max_tokens`` of ``null`` means the full remaining context,
``max_model_len - len(prompt_token_ids)`` (P18); a bounded value is the
caller's declared engineering choice and is recorded as given. The model's own
``generation_config.json`` supplies no defaults (``generation_config="vllm"``):
every sampling parameter is the request's. Nothing here scores, selects or
judges; the caller compares what comes back.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "aadistill.vllm_check/v1"
#: what a clear GPU may still show in use (the driver's own reservation; an idle L40S/A100 reads a few MiB)
MAX_USED_MIB = 1024


def gpu_state(index: int = 0) -> dict:
    """The GPU as ``nvidia-smi`` reports it now: compute processes and memory. Never raises."""
    out: dict = {"read_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    try:
        apps = subprocess.run(["nvidia-smi", f"--id={index}", "--query-compute-apps=pid,used_memory",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60)
        mem = subprocess.run(["nvidia-smi", f"--id={index}", "--query-gpu=name,memory.used,memory.total,driver_version",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        return {**out, "known": False, "error": f"{type(exc).__name__}: {exc}"}
    if apps.returncode or mem.returncode:
        return {**out, "known": False, "error": (apps.stderr + mem.stderr)[-400:]}
    procs = [dict(zip(("pid", "used_mib"), (int(x) for x in line.split(",")))) for line in apps.stdout.splitlines()
             if line.strip()]
    name, used, total, driver = (x.strip() for x in mem.stdout.strip().split(","))
    return {**out, "known": True, "name": name, "driver": driver, "used_mib": int(used), "total_mib": int(total),
            "compute_processes": procs}


def clear_problems(state: dict, max_used_mib: int = MAX_USED_MIB) -> list[str]:
    """Why the GPU is not clear for an engine (empty when it is). An unknown reading is not clear."""
    if not state.get("known"):
        return [f"the GPU state is unknown ({state.get('error')})"]
    bad = []
    if state["compute_processes"]:
        bad.append(f"compute processes still on the GPU: {state['compute_processes']}")
    if state["used_mib"] > max_used_mib:
        bad.append(f"{state['used_mib']} MiB in use, above the {max_used_mib} MiB a clear GPU may show")
    return bad


def run_isolated(vllm_python: str, request: dict, workdir: Path, *, timeout_s: float = 3600,
                 max_used_mib: int = MAX_USED_MIB, gpu_index: int = 0, script: Path | None = None) -> dict:
    """Run one engine request in the vLLM interpreter after confirming the GPU is clear.

    Returns ``{"status": "PASS" | "FAIL", "gpu_before", "gpu_after", "returncode", "result", "problems"}``;
    ``result`` is the child's output document. Refuses (FAIL, no engine started) on a GPU that is
    not clear or cannot be read."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    before = gpu_state(gpu_index)
    rec: dict = {"gpu_before": before, "max_used_mib": max_used_mib}
    bad = clear_problems(before, max_used_mib)
    if bad:
        return {**rec, "status": "FAIL", "problems": bad, "engine_started": False}
    req_p, out_p = workdir / "vllm_request.json", workdir / "vllm_result.json"
    req_p.write_text(json.dumps(request) + "\n")
    script = script or Path(__file__).resolve()
    t = time.time()
    try:
        r = subprocess.run([vllm_python, str(script), "--request", str(req_p), "--out", str(out_p)],
                           capture_output=True, text=True, timeout=timeout_s)
        rc, err = r.returncode, r.stderr[-4000:]
    except subprocess.TimeoutExpired as exc:
        rc, err = None, f"timeout after {timeout_s}s: {str(exc.stderr or '')[-2000:]}"
    rec.update(engine_started=True, returncode=rc, seconds=round(time.time() - t, 1), stderr_tail=err,
               gpu_after=gpu_state(gpu_index))
    result = json.loads(out_p.read_text()) if out_p.is_file() else None
    problems = [] if rc == 0 and result else [f"the engine exited {rc}"]
    return {**rec, "status": "PASS" if not problems else "FAIL", "problems": problems, "result": result}


def _logprob_rows(rows) -> list:
    """vLLM's per-position {token_id: Logprob} into JSON: [[token_id, logprob, rank], ...] per position."""
    out = []
    for pos in rows or []:
        out.append(None if pos is None else
                   [[int(t), float(lp.logprob), getattr(lp, "rank", None)] for t, lp in pos.items()])
    return out


def engine(request: dict) -> dict:
    """The child: build the engine, run every request, report token ids and log-probabilities."""
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    t0 = time.time()
    llm = LLM(model=request["model"], revision=request.get("revision"), tokenizer=request.get("tokenizer"),
              tokenizer_revision=request.get("tokenizer_revision"), dtype=request.get("dtype", "bfloat16"),
              max_model_len=int(request["max_model_len"]), seed=int(request.get("seed", 0)),
              gpu_memory_utilization=float(request.get("gpu_memory_utilization", 0.85)),
              enforce_eager=bool(request.get("enforce_eager", False)), generation_config="vllm")
    load_s = time.time() - t0
    prompts, params = [], []
    for r in request["requests"]:
        s = dict(r.get("sampling") or {})
        if s.get("max_tokens") is None:
            s["max_tokens"] = int(request["max_model_len"]) - len(r["prompt_token_ids"])
        prompts.append(TokensPrompt(prompt_token_ids=list(r["prompt_token_ids"])))
        params.append(SamplingParams(**s))
    t1 = time.time()
    outs = llm.generate(prompts, sampling_params=params, use_tqdm=False)
    gen_s = time.time() - t1
    import torch
    rows = []
    for r, p, o in zip(request["requests"], params, outs):
        rows.append({"id": r["id"], "prompt_token_ids": list(o.prompt_token_ids),
                     "max_tokens": p.max_tokens,
                     "prompt_logprobs": _logprob_rows(o.prompt_logprobs),
                     "outputs": [{"token_ids": list(c.token_ids), "finish_reason": c.finish_reason,
                                  "stop_reason": c.stop_reason, "cumulative_logprob": c.cumulative_logprob,
                                  "logprobs": _logprob_rows(c.logprobs)} for c in o.outputs]})
    return {"schema": SCHEMA, "versions": {"vllm": vllm.__version__, "torch": torch.__version__,
                                           "cuda": torch.version.cuda},
            "device": torch.cuda.get_device_name(0), "engine_args": {k: request.get(k) for k in (
                "model", "revision", "tokenizer", "tokenizer_revision", "dtype", "max_model_len", "seed",
                "gpu_memory_utilization", "enforce_eager")} | {"generation_config": "vllm"},
            "load_seconds": round(load_s, 1), "generate_seconds": round(gen_s, 1), "requests": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--request", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    doc = engine(json.loads(Path(a.request).read_text()))
    Path(a.out).write_text(json.dumps(doc) + "\n")
    print(json.dumps({"requests": len(doc["requests"]), "load_seconds": doc["load_seconds"],
                      "generate_seconds": doc["generate_seconds"], "vllm": doc["versions"]["vllm"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
