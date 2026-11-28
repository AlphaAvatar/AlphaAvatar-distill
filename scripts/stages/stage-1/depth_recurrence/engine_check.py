#!/usr/bin/env python3
"""DMR1 variant 2's engine validation (design ``engine_validation``): on the pod, in the first V2 session, after setup
and before any training step. A FAIL stops the session before anything trains.

    /opt/train/bin/python scripts/stages/stage-1/depth_recurrence/engine_check.py run --plan <training plan> \\
        --student <the staged recurrent init dir> --work <dir> --vllm-python /opt/vllm/bin/python
    (internal) ... engine_check.py vllm --model <dir> --prompts <json> --out <json>     # under the vLLM env
    (internal) ... engine_check.py hf --model <dir> --prompts <json> --out <json> [--continuations <json>]
                   [--shared-cache-fault]

Two models, one weight file: the RECURRENT one (the staged init: SOB1's depth weights + the 36-logical-layer
recurrent config) and its CONTROL (the same weights under ``base_config.json``: a stock 16-layer Qwen3), so each
engine path's bf16 noise is measured in the same session on the same bytes. Inputs: the first ``n_prompts`` validation
sequences of the training pack, their first ``prompt_tokens`` tokens as prompts.

* ``hf_cache`` -- HF in bf16 on the GPU: incremental decoding with the cache vs one full no-cache forward over the same
  tokens (the next ``new_tokens`` positions of the pack); the cache must hold one entry per logical layer.
* ``vllm_generation`` -- vLLM (the out-of-tree recurrent model through the plugin, the evaluation runner's engine
  settings: CUDA graphs and compilation as there) generates ``new_tokens`` greedy tokens per prompt with their top-20
  log-probabilities; HF re-scores prompt + continuation without a cache.
* ``sensitivity`` -- the HF cache check on the recurrent model with a deliberately WRONG cache (every logical layer of
  a block writing the block's first entry): it must FAIL the gate, else the gate cannot see a cache bug.

Metrics per check: top-1 agreement (the reference's argmax is the compared path's token) and the mean absolute
difference of the compared token's log-probability. The gate (fixed before any GPU reading, the design's): PASS iff for
both checks the recurrent model's mean |dlogp| <= 3 x the control's + 0.02 and its top-1 agreement >= the control's -
0.02, AND the faulty cache FAILS the hf_cache gate. Each model runs in its own subprocess, so no GPU memory outlives
the check.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

SCHEMA = "aadistill.stage1.depth_recurrence.engine_check/v1"
GATE = {"dlogp_factor": 3.0, "dlogp_slack": 0.02, "top1_slack": 0.02}
INPUTS = {"n_prompts": 4, "prompt_tokens": 384, "new_tokens": 64, "top_logprobs": 20}
BASE_CONFIG = REPO / "configs/stages/stage-1/depth_recurrence/base_config.json"


class EngineCheckError(RuntimeError):
    pass


def write_json(p: Path, doc) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".partial")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    os.replace(tmp, p)


# ---- the gate ------------------------------------------------------------------------------------------------------

def passes(rec: dict, ctrl: dict, g: dict = GATE) -> bool:
    """One check's gate: the recurrent path's metrics against the control's (module docstring)."""
    return (rec["mean_abs_dlogp"] <= g["dlogp_factor"] * ctrl["mean_abs_dlogp"] + g["dlogp_slack"]
            and rec["top1_agreement"] >= ctrl["top1_agreement"] - g["top1_slack"])


def verdict(m: dict, g: dict = GATE) -> dict:
    """``m``: {"hf_cache": {"recurrent", "control", "shared_cache_fault"}, "vllm_generation": {"recurrent", "control"}}
    -> the gate's reading, check by check."""
    hf, vl = m["hf_cache"], m["vllm_generation"]
    rows = {"hf_cache": passes(hf["recurrent"], hf["control"], g),
            "vllm_generation": passes(vl["recurrent"], vl["control"], g),
            "sensitivity_fault_fails": not passes(hf["shared_cache_fault"], hf["control"], g),
            "cache_entries": hf["recurrent"].get("cache_entries") == hf["recurrent"].get("logical_layers")}
    return {"status": "PASS" if all(rows.values()) else "FAIL", "checks": rows, "gate": g,
            "rule": "PASS iff hf_cache and vllm_generation pass against the control, the shared-cache fault fails "
                    "hf_cache, and the recurrent cache holds one entry per logical layer"}


# ---- HF ------------------------------------------------------------------------------------------------------------

def load_hf(model_dir: Path, device: str, dtype=None):
    import torch
    from transformers import AutoModelForCausalLM
    from aadistill.initialization.adapters.families import register_custom_model_families
    register_custom_model_families()
    dtype = dtype or (torch.bfloat16 if device.startswith("cuda") else torch.float32)
    m = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=dtype).to(device).eval()
    return m


class shared_cache_fault:
    """Context manager: every logical layer still writes its own entry, but each iteration of a looped block READS the
    block's first logical-layer entry (the keys and values of another iteration: the bug one cache entry per logical
    layer exists to prevent, with every shape still right)."""

    def __init__(self, model):
        self.model = model

    def __enter__(self):
        from aadistill.initialization.adapters import recurrent_depth as RD
        sched = list(self.model.config.layer_to_block)
        first = {b: sched.index(b) for b in set(sched)}
        self.RD, self.orig = RD, RD.LogicalLayerCache

        class Shared(self.orig):
            __slots__ = ("first",)

            def __init__(self, cache, layer):
                super().__init__(cache, layer)
                self.first = first[sched[int(layer)]]

            def update(self, key_states, value_states, layer_idx=None, *args, **kwargs):
                out = self.cache.update(key_states, value_states, self.layer, *args, **kwargs)
                if self.first == self.layer:
                    return out
                shared = self.cache.layers[self.first]
                return shared.keys, shared.values
        RD.LogicalLayerCache = Shared
        return self

    def __exit__(self, *exc):
        self.RD.LogicalLayerCache = self.orig
        return False


def _agreement(ref_logits, cmp_logits, next_tokens) -> dict:
    """Top-1 agreement and the mean |dlogp| of each position's next token, reference vs compared ([N, V] each)."""
    import torch
    lr, lc = torch.log_softmax(ref_logits.float(), -1), torch.log_softmax(cmp_logits.float(), -1)
    tok = torch.as_tensor(next_tokens, device=lr.device)
    rows = torch.arange(tok.shape[0], device=lr.device)
    d = (lr[rows, tok] - lc[rows, tok]).abs()
    top1 = (lr.argmax(-1) == lc.argmax(-1)).float()
    return {"positions": int(tok.shape[0]), "top1_agreement": float(top1.mean()), "mean_abs_dlogp": float(d.mean()),
            "max_abs_dlogp": float(d.max())}


def hf_cache_check(model, sequences: list[list[int]], prompt_tokens: int, fault: bool = False) -> dict:
    """Cached incremental decoding vs one no-cache forward over each sequence's positions ``prompt_tokens - 1`` ..
    ``len - 2`` (each predicting the sequence's own next token)."""
    import torch
    from transformers import DynamicCache
    dev = next(model.parameters()).device
    refs, cmps, nxt, entries = [], [], [], None
    ctx = shared_cache_fault(model) if fault else None
    with torch.no_grad():
        if ctx:
            ctx.__enter__()
        try:
            for seq in sequences:
                ids = torch.as_tensor([seq], device=dev)
                full = model(ids, use_cache=False).logits[0]
                cache = DynamicCache(config=model.config)
                out = model(ids[:, :prompt_tokens], past_key_values=cache, use_cache=True)
                steps = [out.logits[0, -1]]
                for t in range(prompt_tokens, len(seq) - 1):
                    out = model(ids[:, t:t + 1], past_key_values=cache, use_cache=True)
                    steps.append(out.logits[0, -1])
                #: the entries this forward actually wrote: one per logical layer, or fewer when layers share one
                entries = sum(int(cache.get_seq_length(i) > 0) for i in range(len(cache.layers)))
                refs.append(full[prompt_tokens - 1:len(seq) - 1])
                cmps.append(torch.stack(steps))
                nxt += list(seq[prompt_tokens:])
        finally:
            if ctx:
                ctx.__exit__(None, None, None)
    got = _agreement(torch.cat(refs), torch.cat(cmps), nxt)
    return {**got, "cache_entries": entries, "logical_layers": int(model.config.num_hidden_layers)}


def hf_rescore(model, prompts: list[list[int]], continuations: list[dict]) -> dict:
    """HF without a cache on prompt + vLLM's greedy continuation: per generated position, HF's argmax vs vLLM's token
    and |HF logp - vLLM logp| of that token."""
    import torch
    dev = next(model.parameters()).device
    top1, dl = [], []
    with torch.no_grad():
        for p, c in zip(prompts, continuations):
            toks = list(c["token_ids"])
            ids = torch.as_tensor([list(p) + toks], device=dev)
            lp = torch.log_softmax(model(ids, use_cache=False).logits[0].float(), -1)
            pos = torch.arange(len(p) - 1, len(p) - 1 + len(toks), device=dev)
            tok = torch.as_tensor(toks, device=dev)
            top1 += (lp[pos].argmax(-1) == tok).float().tolist()
            dl += (lp[pos, tok] - torch.as_tensor(c["logprobs"], device=dev)).abs().tolist()
    n = max(1, len(dl))
    return {"positions": len(dl), "top1_agreement": sum(top1) / n, "mean_abs_dlogp": sum(dl) / n,
            "max_abs_dlogp": max(dl) if dl else 0.0}


def hf_main(a) -> int:
    import torch
    prompts = json.loads(Path(a.prompts).read_text())
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    m = load_hf(Path(a.model), dev)
    seqs = prompts["sequences"]
    out = {"model": a.model, "device": dev, "load_seconds": round(time.time() - t0, 1),
           "hf_cache": hf_cache_check(m, seqs, int(prompts["prompt_tokens"]))}
    if a.shared_cache_fault:
        out["shared_cache_fault"] = hf_cache_check(m, seqs, int(prompts["prompt_tokens"]), fault=True)
    if a.continuations:
        cont = json.loads(Path(a.continuations).read_text())
        out["vllm_rescore"] = hf_rescore(m, [s[:int(prompts["prompt_tokens"])] for s in seqs], cont["outputs"])
    if dev == "cuda":
        out["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    out["seconds"] = round(time.time() - t0, 1)
    write_json(Path(a.out), out)
    return 0


# ---- vLLM ----------------------------------------------------------------------------------------------------------

def vllm_main(a) -> int:
    """Under the vLLM env: greedy continuations with their top-k log-probabilities, the runner's engine settings."""
    from vllm import LLM, SamplingParams
    prompts = json.loads(Path(a.prompts).read_text())
    k, n = int(prompts["top_logprobs"]), int(prompts["new_tokens"])
    p_len = int(prompts["prompt_tokens"])
    t0 = time.time()
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=int(a.max_model_len),
              gpu_memory_utilization=float(a.gpu_mem_util), skip_tokenizer_init=True, seed=0)
    init_s = round(time.time() - t0, 1)
    sp = SamplingParams(temperature=0.0, max_tokens=n, logprobs=k, detokenize=False, ignore_eos=True)
    res = llm.generate([{"prompt_token_ids": s[:p_len]} for s in prompts["sequences"]], sp)
    outs = []
    for r in res:
        o = r.outputs[0]
        ids = list(o.token_ids)
        outs.append({"token_ids": ids, "logprobs": [float(o.logprobs[t][tok].logprob) for t, tok in enumerate(ids)]})
    write_json(Path(a.out), {"model": a.model, "engine_init_seconds": init_s, "seconds": round(time.time() - t0, 1),
                             "outputs": outs})
    return 0


# ---- the orchestration on the pod ----------------------------------------------------------------------------------

def control_dir(student: Path, work: Path) -> Path:
    """The same weight file under the stock 16-layer config (``base_config.json``)."""
    d = Path(work) / "control"
    d.mkdir(parents=True, exist_ok=True)
    w = d / "model.safetensors"
    if not w.exists():
        os.symlink(Path(student).resolve() / "model.safetensors", w)
    shutil.copyfile(BASE_CONFIG, d / "config.json")
    return d


def pack_prompts(u: dict, n: int = INPUTS["n_prompts"],
                 length: int = INPUTS["prompt_tokens"] + INPUTS["new_tokens"]) -> list:
    """The first ``n`` validation sequences of the unit's training pack, their first ``length`` tokens."""
    from aadistill.data.ladder import ladder_blocks
    c = u["config"]
    data_dir = Path(c["data_dir"]) if Path(c["data_dir"]).is_absolute() else REPO / c["data_dir"]
    _train, val, _stats = ladder_blocks(data_dir, c["rung"], n_val=c.get("val_blocks", 16))
    return [[int(x) for x in val[0][i][:length].tolist()] for i in range(n)]


def _run(cmd: list[str], log: Path, timeout_s: float) -> None:
    with log.open("a") as fh:
        fh.write(f"$ {' '.join(cmd)}\n")
        fh.flush()
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout_s, cwd=str(REPO)).returncode
    if rc != 0:
        raise EngineCheckError(f"{cmd[2] if len(cmd) > 2 else cmd} exited {rc} (log {log})")


def run(plan: dict, student: Path, work: Path, vllm_python: str, train_python: str = sys.executable,
        timeout_s: float = 3600.0) -> dict:
    """The whole check (module docstring): prompts, the control dir, vLLM for both models, HF for both, the gate."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    log = work / "engine_check.log"
    t0 = time.time()
    prompts = {**INPUTS, "sequences": pack_prompts(plan["units"][0])}
    write_json(work / "prompts.json", prompts)
    models = {"recurrent": Path(student), "control": control_dir(Path(student), work)}
    me = str(Path(__file__).resolve())
    for name, d in models.items():
        _run([vllm_python, me, "vllm", "--model", str(d), "--prompts", str(work / "prompts.json"),
              "--out", str(work / f"vllm_{name}.json")], log, timeout_s)
    for name, d in models.items():
        _run([train_python, me, "hf", "--model", str(d), "--prompts", str(work / "prompts.json"),
              "--continuations", str(work / f"vllm_{name}.json"), "--out", str(work / f"hf_{name}.json")]
             + (["--shared-cache-fault"] if name == "recurrent" else []), log, timeout_s)
    hf = {n: json.loads((work / f"hf_{n}.json").read_text()) for n in models}
    vl = {n: json.loads((work / f"vllm_{n}.json").read_text()) for n in models}
    metrics = {"hf_cache": {"recurrent": hf["recurrent"]["hf_cache"], "control": hf["control"]["hf_cache"],
                            "shared_cache_fault": hf["recurrent"]["shared_cache_fault"]},
               "vllm_generation": {n: hf[n]["vllm_rescore"] for n in models}}
    doc = {"schema": SCHEMA, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "inputs": INPUTS,
           "models": {n: str(d) for n, d in models.items()}, "metrics": metrics, **verdict(metrics),
           "timing_seconds": {"total": round(time.time() - t0, 1),
                              **{f"vllm_{n}_engine_init": v["engine_init_seconds"] for n, v in vl.items()},
                              **{f"hf_{n}": h["seconds"] for n, h in hf.items()}},
           "hf_peak_allocated_gib": {n: h.get("peak_allocated_gib") for n, h in hf.items()}}
    write_json(work / "engine_check.json", doc)
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("vllm")
    v.add_argument("--model", required=True)
    v.add_argument("--prompts", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--max-model-len", type=int, default=1024)
    v.add_argument("--gpu-mem-util", type=float, default=0.85)
    h = sub.add_parser("hf")
    h.add_argument("--model", required=True)
    h.add_argument("--prompts", required=True)
    h.add_argument("--out", required=True)
    h.add_argument("--continuations", default=None)
    h.add_argument("--shared-cache-fault", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--plan", required=True)
    r.add_argument("--student", required=True)
    r.add_argument("--work", required=True)
    r.add_argument("--vllm-python", default="/opt/vllm/bin/python")
    a = ap.parse_args(argv)
    if a.cmd == "vllm":
        return vllm_main(a)
    if a.cmd == "hf":
        return hf_main(a)
    doc = run(json.loads(Path(a.plan).read_text()), Path(a.student), Path(a.work), a.vllm_python)
    print(json.dumps({"status": doc["status"], "checks": doc["checks"]}))
    return 0 if doc["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
