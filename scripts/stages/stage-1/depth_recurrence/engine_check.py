#!/usr/bin/env python3
"""DMR1 variant 2's engine validation (design ``engine_validation``): on the pod, in the first V2 session, after setup
and before any training step. A FAIL stops the session before anything trains.

    /opt/train/bin/python scripts/stages/stage-1/depth_recurrence/engine_check.py run --plan <training plan> \\
        --student <the staged recurrent init dir> --work <dir> --vllm-python /opt/vllm/bin/python
    ... engine_check.py adopt --record <a session's engine_check.json> --arm depth_loop_kept --write   # dev box, $0
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

**Complete or FAIL.** Before any threshold is read, the record must hold what the protocol registers
(`completeness_problems`):
- every check has exactly ``n_prompts x new_tokens`` scored positions and finite metrics;
- vLLM returned ``n_prompts`` outputs of ``new_tokens`` tokens each, ended by length (``ignore_eos``), with one
  finite log-probability per token;
- every prompt is ``prompt_tokens`` long;
- the cache entries are one per logical layer for both models.

An empty, missing or partial result is a FAIL, never a relative pass of zeros.

**What a record is bound to** (`binding`):
- the recurrent config (the schedule) and the execution the plan carries;
- the control config;
- the inputs, the training pack's identity and the gate;
- the implementation: docstring-free ast digests of the recurrent model's HF and vLLM files, the module they import
  and the plugin path, the plugin's entry points, and `code_identity` of this check's own probe functions;
- the environment: the train and vLLM environments' versions and the GPU.

The validated weights are recorded but are NOT part of the binding: D's and M's weights are the experiment's variable,
and the check validates the execution path for this schedule, not a weight file. A PASS of one arm therefore covers
every arm whose plan binds the same schedule, control, inputs, gate and implementation, run under the same
environment.

**Reuse.** `record_problems` re-derives everything before a record is reused: the binding against the plan it is
asked for (and the live environment on a pod), the completeness, and the verdict recomputed from the saved metrics.
The stored status string is never trusted. `adopt` writes the canonical copy V2-M's plan names, only for a record
with no problem.
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

SCHEMA = "aadistill.stage1.depth_recurrence.engine_check/v2"
GATE = {"dlogp_factor": 3.0, "dlogp_slack": 0.02, "top1_slack": 0.02}
INPUTS = {"n_prompts": 4, "prompt_tokens": 384, "new_tokens": 64, "top_logprobs": 20}
BASE_CONFIG = REPO / "configs/stages/stage-1/depth_recurrence/base_config.json"
CANONICAL = "logs/stages/stage-1/depth_recurrence/engine_check.json"
#: the execution path the check validates (module docstring: the implementation half of the binding)
IMPLEMENTATION = ("src/aadistill/initialization/adapters/recurrent_depth.py",
                  "src/aadistill/initialization/adapters/recurrent_depth_vllm.py",
                  "src/aadistill/initialization/adapters/projected_residual_vllm.py",
                  "scripts/shared/rollout/vllm_plugin_path.py",
                  "scripts/shared/rollout/vllm_plugin/aadistill_vllm_plugin-1.0.dist-info/entry_points.txt")
CHECKS = (("hf_cache", "recurrent"), ("hf_cache", "control"), ("hf_cache", "shared_cache_fault"),
          ("vllm_generation", "recurrent"), ("vllm_generation", "control"))


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


def _finite(x) -> bool:
    import math
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def completeness_problems(m: dict, outputs: dict, inputs: dict = INPUTS, logical_layers: dict | None = None) -> list:
    """Why ``m`` (the metrics) and ``outputs`` (each model's vLLM output summary, `summarize`) are not the registered
    protocol's complete result (empty when they are). ``logical_layers``: {"recurrent": n, "control": n}."""
    bad = []
    n = int(inputs["n_prompts"]) * int(inputs["new_tokens"])
    for check, model in CHECKS:
        r = ((m or {}).get(check) or {}).get(model)
        if not isinstance(r, dict):
            bad.append(f"{check}.{model}: missing")
            continue
        if r.get("positions") != n:
            bad.append(f"{check}.{model}: {r.get('positions')!r} scored positions, the protocol's {n}")
        for k in ("top1_agreement", "mean_abs_dlogp", "max_abs_dlogp"):
            if not _finite(r.get(k)):
                bad.append(f"{check}.{model}: {k} = {r.get(k)!r} is not a finite number")
        if _finite(r.get("top1_agreement")) and not 0.0 <= r["top1_agreement"] <= 1.0:
            bad.append(f"{check}.{model}: top-1 agreement {r['top1_agreement']} outside [0, 1]")
        if _finite(r.get("mean_abs_dlogp")) and r["mean_abs_dlogp"] < 0:
            bad.append(f"{check}.{model}: a negative mean |dlogp|")
    for model in ("recurrent", "control"):
        r = ((m or {}).get("hf_cache") or {}).get(model) or {}
        want = (logical_layers or {}).get(model, r.get("logical_layers"))
        if r.get("cache_entries") is None or r.get("cache_entries") != want:
            bad.append(f"hf_cache.{model}: {r.get('cache_entries')!r} cache entries for {want!r} logical layers")
        o = (outputs or {}).get(model)
        if not isinstance(o, dict):
            bad.append(f"vllm.{model}: no output summary")
            continue
        if o.get("n_outputs") != int(inputs["n_prompts"]):
            bad.append(f"vllm.{model}: {o.get('n_outputs')!r} outputs for {inputs['n_prompts']} prompts")
        if o.get("lengths") != [int(inputs["new_tokens"])] * int(inputs["n_prompts"]):
            bad.append(f"vllm.{model}: output lengths {o.get('lengths')!r}, the protocol's {inputs['new_tokens']} each")
        if o.get("logprob_lengths") != o.get("lengths"):
            bad.append(f"vllm.{model}: log-probabilities {o.get('logprob_lengths')!r} for tokens {o.get('lengths')!r}")
        if o.get("finish_reasons") != ["length"] * int(inputs["n_prompts"]):
            bad.append(f"vllm.{model}: finish reasons {o.get('finish_reasons')!r}, the protocol's length stop")
        if o.get("all_logprobs_finite") is not True:
            bad.append(f"vllm.{model}: a log-probability is not finite")
        if o.get("prompt_lengths") != [int(inputs["prompt_tokens"])] * int(inputs["n_prompts"]):
            bad.append(f"vllm.{model}: prompt lengths {o.get('prompt_lengths')!r}")
    return bad


def verdict(m: dict, outputs: dict, g: dict = GATE, inputs: dict = INPUTS, logical_layers: dict | None = None) -> dict:
    """The gate's reading, check by check, after `completeness_problems`: an incomplete result is FAIL."""
    incomplete = completeness_problems(m, outputs, inputs, logical_layers)
    if incomplete:
        return {"status": "FAIL", "checks": {"complete": False}, "completeness_problems": incomplete, "gate": g,
                "rule": "an incomplete result is FAIL before any threshold is read"}
    hf, vl = m["hf_cache"], m["vllm_generation"]
    rows = {"complete": True,
            "hf_cache": passes(hf["recurrent"], hf["control"], g),
            "vllm_generation": passes(vl["recurrent"], vl["control"], g),
            "sensitivity_fault_fails": not passes(hf["shared_cache_fault"], hf["control"], g),
            "cache_entries": hf["recurrent"].get("cache_entries") == hf["recurrent"].get("logical_layers")}
    return {"status": "PASS" if all(rows.values()) else "FAIL", "checks": rows, "gate": g,
            "rule": "PASS iff the result is complete, hf_cache and vllm_generation pass against the control, the "
                    "shared-cache fault fails hf_cache, and the recurrent cache holds one entry per logical layer"}


def summarize(vllm_doc: dict, prompts: dict) -> dict:
    """One model's vLLM outputs as `completeness_problems` reads them."""
    import math
    outs = (vllm_doc or {}).get("outputs") or []
    return {"n_outputs": len(outs), "lengths": [len(o.get("token_ids") or []) for o in outs],
            "logprob_lengths": [len(o.get("logprobs") or []) for o in outs],
            "finish_reasons": [o.get("finish_reason") for o in outs],
            "all_logprobs_finite": bool(outs) and all(isinstance(x, (int, float)) and math.isfinite(x)
                                                      for o in outs for x in (o.get("logprobs") or [])),
            "prompt_lengths": [len(s[:int(prompts["prompt_tokens"])]) for s in prompts.get("sequences") or []]}


def source_identity(paths=IMPLEMENTATION, root: Path = REPO) -> dict:
    """Each file's identity: a Python file's ast dumped without positions and with its docstrings dropped (RI-LM1's
    `code_identity` rule, at file level: the vLLM model module cannot be imported off a pod), any other file's bytes."""
    import ast
    import hashlib
    from stages.reasoning_inheritance.layer_merge import _DUMP_FORM
    out = {}
    for rel in paths:
        src = (Path(root) / rel).read_bytes()
        if rel.endswith(".py"):
            tree = ast.parse(src)
            for node in ast.walk(tree):
                body = getattr(node, "body", None)
                if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)) and body
                        and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant)
                        and isinstance(body[0].value.value, str)):
                    node.body = body[1:] or [ast.Pass()]
            out[rel] = hashlib.sha256(ast.dump(tree, include_attributes=False, **_DUMP_FORM).encode()).hexdigest()
        else:
            out[rel] = hashlib.sha256(src).hexdigest()
    return out


def check_code() -> str:
    """`code_identity` of the probe itself: what it measures and how it decides (not the CLI or the orchestration)."""
    from stages.reasoning_inheritance.layer_merge import code_identity
    return code_identity(passes, completeness_problems, verdict, summarize, shared_cache_fault, _agreement,
                         hf_cache_check, hf_rescore, vllm_main, pack_prompts, control_dir)["sha256"]


def expected_binding(plan: dict, root: Path = REPO) -> dict:
    """What a record must be bound to for ``plan``'s arm (module docstring); the environment is compared apart."""
    import hashlib
    u = plan["units"][0]
    c = u["config"]
    return {"recurrent_config_sha256": u["initialization"]["recurrent_config"]["sha256"],
            "execution": u["initialization"]["execution"],
            "control_config_sha256": hashlib.sha256(BASE_CONFIG.read_bytes()).hexdigest(),
            "inputs": INPUTS, "pack": {"data_dir": c["data_dir"], "rung": c["rung"], "val_blocks": c.get("val_blocks")},
            "gate": GATE, "implementation": source_identity(root=root), "check_code": check_code()}


def environment(vllm_python: str | None = None) -> dict:
    """The versions the check ran under: the train environment's (in-process) and, with ``vllm_python``, the vLLM
    environment's (a subprocess, seconds), and the GPU."""
    import torch
    import transformers
    env = {"train": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "cuda": torch.version.cuda,
                     "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}
    if vllm_python:
        got = subprocess.run([vllm_python, "-c", "import json, torch, vllm; print(json.dumps({'vllm': vllm.__version__, "
                              "'torch': torch.__version__}))"], capture_output=True, text=True, timeout=600)
        env["vllm"] = json.loads(got.stdout.strip().splitlines()[-1]) if got.returncode == 0 else {
            "error": got.stderr[-400:]}
    return env


def record_problems(rec: dict, plan: dict, env: dict | None = None, root: Path = REPO) -> list[str]:
    """Why ``rec`` may not stand as ``plan``'s arm's engine validation (empty when it may): its schema, its binding
    against the plan's (and, given ``env``, the live environment), its completeness, and its verdict re-derived from
    the saved metrics -- the stored status is never trusted."""
    if not isinstance(rec, dict) or rec.get("schema") != SCHEMA:
        return [f"not an engine-check record of {SCHEMA}"]
    bad = []
    want, got = expected_binding(plan, root), rec.get("binding") or {}
    for k, v in want.items():
        if got.get(k) != v:
            bad.append(f"binding.{k} is not this plan's (the record validated another {k})")
    if env is not None and got.get("environment") != env:
        bad.append(f"the record ran under {got.get('environment')}, this pod is {env}")
    again = verdict(rec.get("metrics") or {}, rec.get("vllm_outputs") or {}, GATE, INPUTS,
                    {"recurrent": int(plan["units"][0]["initialization"]["execution"]["n_logical_layers"]),
                     "control": int(plan["units"][0]["initialization"]["execution"]["n_parameter_blocks"])})
    bad += [f"incomplete: {x}" for x in again.get("completeness_problems", [])]
    if again["status"] != "PASS":
        bad.append(f"the saved metrics re-derive {again['status']} ({again['checks']})")
    if rec.get("status") != again["status"]:
        bad.append(f"the record says {rec.get('status')!r}, its metrics say {again['status']!r}")
    return bad


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
    if not dl:                          #: nothing scored: no metric exists (completeness refuses it, never a zero)
        return {"positions": 0, "top1_agreement": None, "mean_abs_dlogp": None, "max_abs_dlogp": None}
    return {"positions": len(dl), "top1_agreement": sum(top1) / len(dl), "mean_abs_dlogp": sum(dl) / len(dl),
            "max_abs_dlogp": max(dl)}


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
        outs.append({"token_ids": ids, "logprobs": [float(o.logprobs[t][tok].logprob) for t, tok in enumerate(ids)],
                     "finish_reason": o.finish_reason})
    write_json(Path(a.out), {"model": a.model, "engine_init_seconds": init_s, "seconds": round(time.time() - t0, 1),
                             "outputs": outs})
    return 0


# ---- the orchestration on the pod ----------------------------------------------------------------------------------

def _sha(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with Path(p).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def adopt(record: Path, arm: str, root: Path = REPO) -> dict:
    """The canonical copy (`CANONICAL`) of a session's record, for the plans that reuse it: written byte for byte, and
    only when `record_problems` finds nothing against ``arm``'s committed plan (the environment is the pod's to check)."""
    from stages.depth_merge import plan as PL
    rec = json.loads(Path(record).read_text())
    plan = json.loads(PL.out_path(arm).read_text())
    bad = record_problems(rec, plan, root=root)
    if bad:
        raise EngineCheckError(f"{record} cannot be adopted for {arm}: {bad}")
    dest = Path(root) / CANONICAL
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(Path(record).read_bytes())
    return {"adopted": str(dest.relative_to(root)), "sha256": _sha(dest), "from": str(record), "arm": arm}


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
    outputs = {n: summarize(v, prompts) for n, v in vl.items()}
    ex = plan["units"][0]["initialization"]["execution"]
    import hashlib
    binding = {**expected_binding(plan), "environment": environment(vllm_python),
               "prompts_sha256": hashlib.sha256((work / "prompts.json").read_bytes()).hexdigest()}
    doc = {"schema": SCHEMA, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "inputs": INPUTS,
           "binding": binding, "arm": plan["arm"],
           "validated_weights_sha256": {"model.safetensors": _sha(Path(student) / "model.safetensors"),
                                        "rule": "recorded, not bound: the weights are the experiment's variable"},
           "models": {n: str(d) for n, d in models.items()}, "metrics": metrics, "vllm_outputs": outputs,
           **verdict(metrics, outputs, GATE, INPUTS, {"recurrent": int(ex["n_logical_layers"]),
                                                      "control": int(ex["n_parameter_blocks"])}),
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
    ad = sub.add_parser("adopt")
    ad.add_argument("--record", required=True)
    ad.add_argument("--arm", required=True)
    ad.add_argument("--write", action="store_true")
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
    if a.cmd == "adopt":
        if not a.write:
            from stages.depth_merge import plan as PL
            bad = record_problems(json.loads(Path(a.record).read_text()), json.loads(PL.out_path(a.arm).read_text()))
            print(json.dumps({"problems": bad}))
            return 0 if not bad else 1
        print(json.dumps(adopt(Path(a.record), a.arm)))
        return 0
    doc = run(json.loads(Path(a.plan).read_text()), Path(a.student), Path(a.work), a.vllm_python)
    print(json.dumps({"status": doc["status"], "checks": doc["checks"]}))
    return 0 if doc["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
