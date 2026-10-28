#!/usr/bin/env python3
"""B-E, the projected-residual student's engineering validation: every check, one record.

    # on the pod (the B-E session's driver command), with the real teacher
    cd /workspace/aad && HF_HUB_OFFLINE=1 PYTHONPATH=src:scripts:scripts/stages/stage-1 /opt/train/bin/python \\
        scripts/stages/stage-1/projected_residual/engineering_validation.py --run-id <run> --out <evidence> \\
        --work <scratch> --status <status file> --device cuda:0 --vllm-python /opt/vllm/bin/python
    # the same orchestration on the host, on a random small teacher ($0)
    PYTHONPATH=src:scripts python scripts/stages/stage-1/projected_residual/engineering_validation.py \\
        --run-id host --out <dir> --tiny

It answers what a CPU cannot -- CUDA numerics, real memory, real step time,
the inference engine -- and produces no scientific result: nothing it trains
is kept as a student, nothing is scored, and the design is not changed by what
it measures. Every threshold, the probe and the numeric identity are fixed in
advance in `CONFIG` (configs/stages/stage-1/projected_residual/engineering_validation.json).

THE PROCESS BOUNDARY. This process (the parent) never imports torch. It runs
every PyTorch check in a child process (``--phase torch``, the same
interpreter) that EXITS -- the teacher, the students and the trainer live and
die there -- and only then starts vLLM in its own interpreter through
`shared.validation.vllm_check.run_isolated`, which refuses while nvidia-smi
lists any compute process or more than the configured MiB in use, and records
the reading. A teacher left resident beside the engine would shrink its KV
cache or refuse it, and the vLLM check would then describe the leftover.

The checks (statuses re-derived by `verdict` from the evidence and the config alone):

1. ``initialization`` -- design.json's materialization on the real teacher;
   the declared basis's captured energy on every layer's input roles and on
   the diagnostic roles (``residual_in``, ``attn_out``, ``ffn_out``). PASS:
   counted parameters equal the cost model's.
2. ``structural_fold`` -- materialize's CPU float32 check that the fold is the
   same function. PASS: max |diff| / max |logit| <= the config's bound.
3. ``fold_cuda`` -- factored against folded stock Qwen3, BOTH on the L40S,
   float32 and bfloat16, over the config's fixed eval-set-1 probe (ids
   re-rendered and hash-checked). PASS: design.json :: fold_acceptance on that
   device, the card the config names, TF32 off, sdpa on both. Mean |diff| and
   argmax agreement are diagnostics.
4. ``trainer_smoke`` -- the recipe's shape (D4-Augmented's E1 config for the
   smoke seed plus ``\\.projection$``; 8,192, batch 2, micro-batch 1, bf16
   autocast, gradient checkpointing, the teacher on the fly) on synthetic
   sequences: step time, peak memory, orthogonality drift. PASS: finite losses,
   a gradient on every projection, the tied embedding unchanged.
5. ``resume`` -- uninterrupted against half + checkpoint + reload + half.
   PASS: bit-identical, or (kernel nondeterminism) no further apart than two
   uninterrupted runs are.
6. ``checkpoint_identity`` -- save, reload, save: identical shard bytes; the
   folded model saved and reloaded as a stock ``Qwen3ForCausalLM`` computes
   the folded model's logits.
7. ``gpu_isolation`` -- the PyTorch child exited before the engine started, the
   parent never imported torch, and the GPU read clear (GPU only).
8. ``vllm`` -- the saved folded checkpoint in vLLM 0.27.1 at the formal
   evaluation's resolved context: loads; echoed prompt ids; finite
   log-probabilities; finish reasons; and its teacher-forced log-probabilities
   against the HF bfloat16 reference on the same ids and card, within the
   config's fixed bounds (GPU only). Engineering, not a measurement: no score,
   no GO.

On the host (``--tiny``) the PyTorch checks run on the CPU; ``fold_cuda`` keeps
its numbers as a rehearsal and is NOT_RUN, and the GPU-only checks are NOT_RUN.
Every check runs even after an earlier one fails, so one session reports all of
them. Weights live only under ``--work`` and are deleted at the end.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from shared.validation import vllm_check as VC  # noqa: E402  (standard library only)

CONFIG = "configs/stages/stage-1/projected_residual/engineering_validation.json"
SCHEMA = "aadistill.stage1.projected_residual.engineering_validation/v2"
DIAGNOSTICS = ("residual_input", "read_write")
CHECKS = ("initialization", "structural_fold", "fold_cuda", "trainer_smoke", "resume", "checkpoint_identity",
          "gpu_isolation", "vllm")
#: what only a GPU can answer; NOT_RUN is a pass for these on the host rehearsal and nowhere else
GPU_ONLY = ("fold_cuda", "gpu_isolation", "vllm")
#: what the PyTorch child writes under --work; removed when the run ends
WORK_DIRS = ("init", "smoke", "resume", "control", "ident")
TORCH_CHILD_TIMEOUT_S = 3 * 3600
VLLM_TIMEOUT_S = 1800


def _load(p: Path):
    return json.loads(p.read_text()) if p.is_file() else None


def ids_sha(ids) -> str:
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode()).hexdigest()


def shard_hashes(d: Path) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.glob("*.safetensors"))}


def smoke_config(out: Path, *, steps: int, seq_len: int, device: str, seed: str) -> dict:
    """D4-Augmented's E1 config for the smoke seed, changed only where a smoke must differ."""
    h1pa = json.loads((REPO / "configs/stages/stage-2/reasoning_data/h1pa.json").read_text())
    e1 = json.loads((REPO / h1pa["recipe"]["e1_configs"][seed]["path"]).read_text())
    cfg = {k: v for k, v in e1.items() if not k.startswith("_")}
    cfg.update({"run_name": "pw900_engineering_smoke", "trainable_patterns": e1["trainable_patterns"]
                + ["\\.projection$"], "block_len": seq_len, "device": device, "out_dir": str(out),
                "schedule": {"total_steps": steps, "warmup_steps": 1, "min_lr_frac": e1["schedule"]["min_lr_frac"]},
                "checkpoint": {"save_every": 0, "keep_last": 1},
                "intervals": {"log_every": 1, "eval_every": 0, "eval_blocks": 0}})
    return cfg


def synthetic_blocks(n: int, seq_len: int, vocab: int, seed: int = 0):
    import torch
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (n, seq_len), generator=g)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    mask[:, seq_len // 4:] = True
    return ids, mask


# ---- the PyTorch child ------------------------------------------------------------------------------------

def apply_numeric_identity(cfg: dict) -> dict:
    """The config's numeric identity, set before any CUDA work; returns what is in force."""
    import torch
    ni = cfg["numeric_identity"]
    torch.backends.cuda.matmul.allow_tf32 = bool(ni["tf32"])
    torch.backends.cudnn.allow_tf32 = bool(ni["tf32"])
    torch.use_deterministic_algorithms(True, warn_only=ni["deterministic_algorithms"] == "warn_only")
    return {"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "attn_implementation": ni["attn_implementation"]}


def render_probe(cfg: dict) -> tuple[list[dict], list[list[int]]]:
    """The config's eval-set-1 items, rendered as the evaluation route renders them, as token ids.

    Read through `shared.eval_sets` (which refuses a set whose bytes differ from
    its authorized identity). Returns the evidence rows (ids' hash and length,
    never text) and the ids."""
    from transformers import AutoTokenizer

    from shared import eval_sets as ES

    p = cfg["probe"]
    tok = AutoTokenizer.from_pretrained(p["tokenizer"]["model"], revision=p["tokenizer"]["revision"])
    rows, ids_all = [], []
    for want in p["items"]:
        it = next(x for x in ES.items(p["eval_set_id"], want["component"], want["unit"])
                  if (x.get("task_id") or x.get("id")) == want["item_id"])
        msgs = [{"role": "system", "content": p["system_message"]}, *it["messages"]]
        ids = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
                  add_special_tokens=False).input_ids
        rows.append({"id": f"{want['component']}/{want['unit']}/{want['item_id']}", "tokens": len(ids),
                     "token_ids_sha256": ids_sha(ids), "expected_sha256": want["token_ids_sha256"]})
        ids_all.append(list(ids))
    return rows, ids_all


def tiny_probe(cfg: dict, vocab: int) -> tuple[list[dict], list[list[int]]]:
    import torch
    g = torch.Generator().manual_seed(7)
    ids_all = [torch.randint(0, vocab, (n,), generator=g).tolist() for n in cfg["probe"]["tiny_lengths"]]
    rows = [{"id": f"tiny/{i}", "tokens": len(x), "token_ids_sha256": ids_sha(x), "expected_sha256": None}
            for i, x in enumerate(ids_all)]
    return rows, ids_all


def torch_phase(out: Path, work: Path, cfg: dict, *, device: str, tiny: bool, steps: int) -> dict:
    """Every PyTorch check, in this (child) process; writes ``torch.json`` after each one,
    ``hf_reference.json`` and the engine request ``vllm_request.prepared.json``."""
    applied = apply_numeric_identity(cfg)
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from aadistill.initialization.adapters.projected_residual import (
        fold_to_qwen3, projection_orthogonality, projections, register_auto_classes)
    from aadistill.training.train import Trainer
    from stages.projected_residual import materialize as M

    register_auto_classes()
    cuda = device.startswith("cuda")
    design = {} if tiny else json.loads((REPO / cfg["design"]).read_text())
    attn = cfg["numeric_identity"]["attn_implementation"]
    rec = {"tiny": tiny, "device": device, "device_name": torch.cuda.get_device_name(device) if cuda else "cpu",
           "capability": list(torch.cuda.get_device_capability(device)) if cuda else None,
           "torch": torch.__version__, "transformers": transformers.__version__, "cuda_runtime": torch.version.cuda,
           "numeric_identity_applied": applied, "measurements": {}, "errors": {}}
    meas = rec["measurements"]

    def save():
        (out / "torch.json").write_text(json.dumps(rec, indent=1) + "\n")

    def attempt(name, fn):
        try:
            meas[name] = fn()
        except Exception:  # every check still runs
            rec["errors"][name] = traceback.format_exc()[-3000:]
        finally:
            if cuda:
                torch.cuda.empty_cache()
            save()

    save()
    student_dir = work / "init/model"

    def initialization():
        t0 = time.time()
        m = M.run(design, work / "init", device=device, tiny=tiny, diagnostic_statistics=DIAGNOSTICS)
        shutil.copy2(work / "init/materialization.json", out / "materialization.json")
        return {"num_parameters": m["num_parameters"], "num_parameters_cost_model": m["num_parameters_cost_model"],
                "seconds": round(time.time() - t0, 1), "calibration_positions": m["calibration_positions"],
                "energy_captured_min": {k: round(min(L[k] for L in m["per_layer"]), 6) for k in m["per_layer"][0]
                                        if k.endswith(("_energy_captured", "_declared_basis"))},
                "structural_fold_check": m["structural_fold_check"], "weights": m["weights"],
                "config_sha256": m["config_sha256"], "timings": m["timings"]}

    attempt("initialization", initialization)

    def fresh(dtype=torch.float32):
        return AutoModelForCausalLM.from_pretrained(student_dir, dtype=dtype, attn_implementation=attn)

    probe_ids: list[list[int]] = []

    def probe():
        rows, ids = tiny_probe(cfg, AutoConfig.from_pretrained(student_dir).vocab_size) if tiny else render_probe(cfg)
        probe_ids.extend(ids)
        return {"rows": rows, "positions": sum(r["tokens"] for r in rows)}

    attempt("probe", probe)

    def fold_cuda():
        st = fresh().to(device)
        folded = fold_to_qwen3(st, dtype=torch.float32)
        if hasattr(folded, "set_attn_implementation"):
            folded.set_attn_implementation(attn)
        else:
            folded.config._attn_implementation = attn
        r = M.fold_numerics(st, folded, probe_ids, device=device, factor=cfg["fold_acceptance"]["factor"])
        r.update(device_name=rec["device_name"], attn_implementation={
            "factored": st.config._attn_implementation, "folded": folded.config._attn_implementation},
            dtypes={"factored": str(next(st.parameters()).dtype), "folded": str(next(folded.parameters()).dtype)})
        del st, folded
        return r

    attempt("fold_cuda", fold_cuda)

    if tiny:
        teacher, seq_len = M.tiny_inputs()[0], 64
    else:
        teacher = AutoModelForCausalLM.from_pretrained(design["teacher"]["model"],
                                                       revision=design["teacher"]["revision"],
                                                       dtype=torch.bfloat16, attn_implementation=attn)
        h1pa = json.loads((REPO / "configs/stages/stage-2/reasoning_data/h1pa.json").read_text())
        seq_len = int(json.loads((REPO / h1pa["recipe"]["e1_configs"][cfg["smoke"]["e1_seed"]]["path"])
                                 .read_text())["block_len"])
    seed = cfg["smoke"]["e1_seed"]
    try:
        blocks = synthetic_blocks(2 * steps, seq_len, AutoConfig.from_pretrained(student_dir).vocab_size)
    except Exception:
        blocks = None
        rec["errors"]["blocks"] = traceback.format_exc()[-2000:]
    full: dict = {}

    def train(out_dir: Path, n: int):
        #: `device=` is the Trainer's own argument; it does not read cfg["device"], and its default is the CPU
        tr = Trainer(smoke_config(out_dir, steps=steps, seq_len=seq_len, device=device, seed=seed), fresh(),
                     blocks, teacher=teacher, device=device)
        times, losses = [], []
        for _ in range(n):
            s = time.time()
            m = tr.step_once()
            if cuda:
                torch.cuda.synchronize(device)
            times.append(time.time() - s)
            losses.append(float(m.get("loss", float("nan"))) if isinstance(m, dict) else float("nan"))
        return tr, times, losses

    def trainer_smoke():
        s0 = fresh()
        emb0, orth0 = s0.model.embed_tokens.weight.detach().clone(), projection_orthogonality(s0)
        del s0
        if cuda:
            torch.cuda.reset_peak_memory_stats(device)
        tr, times, losses = train(work / "smoke", steps)
        grads = [float(p.grad.norm()) if p.grad is not None else 0.0 for p in projections(tr.student)]
        orth1 = projection_orthogonality(tr.student)
        full.update({n: p.detach().cpu().clone() for n, p in tr.student.named_parameters()})
        warm = times[2:] or times
        r = {"steps": steps, "sequence_length": seq_len, "requested_device": device,
             "trainer_device": str(tr.device), "student_device": str(next(tr.student.parameters()).device),
             "seconds_per_step_mean_after_2": round(sum(warm) / len(warm), 4), "seconds_per_step": times,
             "peak_memory_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 2) if cuda else None,
             "peak_reserved_gib": round(torch.cuda.max_memory_reserved(device) / 2**30, 2) if cuda else None,
             "losses": losses, "losses_finite": all(math.isfinite(x) for x in losses),
             "projections_with_a_gradient": sum(g > 0 for g in grads), "projections": len(grads),
             "embedding_unchanged": bool(torch.equal(tr.student.model.embed_tokens.weight.detach().cpu(),
                                                     emb0.cpu())),
             "orthogonality_rel_fro_max": {"before": max(x["rel_fro"] for x in orth0),
                                           "after": max(x["rel_fro"] for x in orth1)},
             "sigma_range_after": [min(x["sigma_min"] for x in orth1), max(x["sigma_max"] for x in orth1)]}
        del tr
        return r

    attempt("trainer_smoke", trainer_smoke)

    def worst(student) -> float:
        return max(float((p.detach().cpu() - full[n]).abs().max()) for n, p in student.named_parameters())

    def resume():
        if not full:
            raise RuntimeError("no uninterrupted run to compare with: the trainer smoke failed")
        half = steps // 2
        part = Trainer(smoke_config(work / "resume", steps=steps, seq_len=seq_len, device=device, seed=seed),
                       fresh(), blocks, teacher=teacher, device=device)
        for _ in range(half):
            part.step_once()
        ckpt = part.save_checkpoint()
        del part
        resumed = Trainer(smoke_config(work / "resume", steps=steps, seq_len=seq_len, device=device, seed=seed),
                          AutoModelForCausalLM.from_pretrained(ckpt / "model", dtype=torch.float32,
                                                               attn_implementation=attn), blocks, teacher=teacher,
                          device=device)
        resumed.restore(ckpt)
        for _ in range(steps - half):
            resumed.step_once()
        r = {"resumed_at_step": half, "max_abs_param_diff": worst(resumed.student),
             "control_max_abs_param_diff": None}
        del resumed
        if r["max_abs_param_diff"] != 0.0:     # the device's own run-to-run difference
            ctl, _, _ = train(work / "control", steps)
            r["control_max_abs_param_diff"] = worst(ctl.student)
            del ctl
        return r

    attempt("resume", resume)
    del teacher

    def checkpoint_identity():
        a, b, f = work / "ident/a", work / "ident/b", work / "ident/folded"
        s1 = fresh()
        s1.save_pretrained(a)
        AutoModelForCausalLM.from_pretrained(a, dtype=torch.float32).save_pretrained(b)
        folded = fold_to_qwen3(s1, dtype=torch.float32)
        folded.save_pretrained(f)
        stock = AutoModelForCausalLM.from_pretrained(f, dtype=torch.float32)
        x = torch.tensor([probe_ids[0][:256]])
        with torch.no_grad():
            diff = float((stock(x, use_cache=False).logits - folded(x, use_cache=False).logits).abs().max())
        return {"resave_shards_identical": shard_hashes(a) == shard_hashes(b), "shards": shard_hashes(a),
                "folded_class": type(stock).__name__, "folded_reload_max_abs_diff": diff,
                "folded_shards": shard_hashes(f),
                "folded_config_sha256": hashlib.sha256((f / "config.json").read_bytes()).hexdigest()}

    attempt("checkpoint_identity", checkpoint_identity)

    def reference():
        """The saved folded bytes as stock Qwen3 on `device` in bf16: what the engine is compared with."""
        from shared.evaluation.uncapped_eval import resolve_context, resolve_stop_ids
        f = work / "ident/folded"
        stock = AutoModelForCausalLM.from_pretrained(f, dtype=torch.bfloat16, attn_implementation=attn).to(device)
        stock.eval()
        rows = []
        with torch.no_grad():
            for i, ids in enumerate(probe_ids):
                lp = torch.log_softmax(stock(torch.tensor([ids], device=device), use_cache=False)
                                       .logits.float()[0], -1)[:-1]
                nxt = torch.tensor(ids[1:], device=device)
                top2 = lp.topk(2, dim=-1)
                rows.append({"id": meas["probe"]["rows"][i]["id"], "positions": len(ids) - 1,
                             "realized_logprob": [round(float(v), 6) for v in lp.gather(1, nxt[:, None])[:, 0]],
                             "argmax": [int(v) for v in top2.indices[:, 0]],
                             "top1_margin": [round(float(v), 6) for v in top2.values[:, 0] - top2.values[:, 1]]})
        (out / "hf_reference.json").write_text(json.dumps({"dtype": "bfloat16", "attn_implementation":
                                                           stock.config._attn_implementation,
                                                           "device_name": rec["device_name"], "prompts": rows}) + "\n")
        ctx = resolve_context(AutoConfig.from_pretrained(f), None, cfg["vllm"]["context"]["trained_context"])
        if tiny:
            stops, tokenizer = [], {"tokenizer": None, "tokenizer_revision": None}
        else:
            tm, trev = design["teacher"]["model"], design["teacher"]["revision"]
            stops = resolve_stop_ids(tm, AutoTokenizer.from_pretrained(tm, revision=trev),
                                     AutoConfig.from_pretrained(tm, revision=trev), trev)
            tokenizer = {"tokenizer": tm, "tokenizer_revision": trev}
        v = cfg["vllm"]
        request = {"model": str(f), **tokenizer, "dtype": cfg["numeric_identity"]["dtype"],
                   "max_model_len": int(ctx["resolved_context"]), "gpu_memory_utilization": v["gpu_memory_utilization"],
                   "seed": int(v["sampling"]["seed"]),
                   "requests": [{"id": r["id"], "prompt_token_ids": ids,
                                 "sampling": {**v["sampling"], "stop_token_ids": stops}}
                                for r, ids in zip(meas["probe"]["rows"], probe_ids)]}
        (out / "vllm_request.prepared.json").write_text(json.dumps(request) + "\n")
        del stock
        return {"context": ctx, "stop_token_ids": stops, "dtype": "bfloat16",
                "attn_implementation": attn, "prompts": len(rows)}

    attempt("reference", reference)
    return rec


# ---- the engine -------------------------------------------------------------------------------------------

def vllm_phase(out: Path, cfg: dict, vllm_python: str, *, script: Path | None = None,
               timeout_s: float = VLLM_TIMEOUT_S) -> dict:
    """The prepared request through `run_isolated`; ``vllm.json`` records the run (the engine's
    own output is ``vllm/vllm_result.json``, its request ``vllm/vllm_request.json``)."""
    prepared = _load(out / "vllm_request.prepared.json")
    if prepared is None:
        rec = {"status": "FAIL", "engine_started": False,
               "problems": ["the PyTorch child prepared no engine request (its reference check failed)"]}
    else:
        rec = VC.run_isolated(vllm_python, prepared, out / "vllm", timeout_s=timeout_s,
                              max_used_mib=int(cfg["vllm"]["max_used_mib_before_engine"]), script=script)
    rec = {k: v for k, v in rec.items() if k != "result"}
    (out / "vllm.json").write_text(json.dumps(rec, indent=1) + "\n")
    return rec


# ---- the verdict, from the evidence and the config alone ---------------------------------------------------

def _not_measured(name: str, t: dict, iso: dict) -> dict:
    err = (t.get("errors") or {}).get(name)
    rc = (iso.get("torch_child") or {}).get("returncode")
    last = err.strip().splitlines()[-1] if err else f"nothing recorded (the PyTorch child exited {rc})"
    return {"status": "FAIL", "why": f"not measured: {last}"}


def _probe_problems(t: dict, cfg: dict) -> list[str]:
    rows = ((t.get("measurements") or {}).get("probe") or {}).get("rows")
    if rows is None:
        return ["no probe was rendered"]
    want = [f"{i['component']}/{i['unit']}/{i['item_id']}" for i in cfg["probe"]["items"]]
    if [r["id"] for r in rows] != want:
        return ["the probe's items are not the config's"]
    bad = [r["id"] for r, i in zip(rows, cfg["probe"]["items"]) if r["token_ids_sha256"] != i["token_ids_sha256"]]
    return [f"rendered ids differ from the fixed probe for {bad}"] if bad else []


def _device_problems(t: dict, cfg: dict) -> list[str]:
    ni, applied, bad = cfg["numeric_identity"], t.get("numeric_identity_applied") or {}, []
    if ni["device_class"] not in str(t.get("device_name")):
        bad.append(f"ran on {t.get('device_name')!r}, not the {ni['device_class']} the config names")
    if applied.get("matmul_allow_tf32") is not ni["tf32"] or applied.get("cudnn_allow_tf32") is not ni["tf32"]:
        bad.append(f"TF32 was {applied.get('matmul_allow_tf32')}/{applied.get('cudnn_allow_tf32')}, not {ni['tf32']}")
    return bad


def _vllm_check(out: Path, cfg: dict, t: dict) -> dict:
    v, res = _load(out / "vllm.json") or {}, _load(out / "vllm/vllm_result.json")
    req, ref = _load(out / "vllm/vllm_request.json"), _load(out / "hf_reference.json")
    if not v.get("engine_started") or v.get("returncode") != 0 or not res or not req or not ref:
        return {"status": "FAIL", "why": f"the engine did not complete: {v.get('problems') or 'no result'}",
                "returncode": v.get("returncode"), "stderr_tail": (v.get("stderr_tail") or "")[-800:]}
    crit, problems = cfg["vllm"]["criteria"], _probe_problems(t, cfg)
    want_ctx = (((t.get("measurements") or {}).get("reference") or {}).get("context") or {}).get("resolved_context")
    if not (res["engine_args"]["max_model_len"] == req["max_model_len"] == want_ctx):
        problems.append(f"the engine ran at {res['engine_args']['max_model_len']}, not the resolved {want_ctx}")
    by_id = {r["id"]: r for r in ref["prompts"]}
    deltas, confident, agree, finish = [], 0, 0, []
    if len(res["requests"]) != len(req["requests"]):
        problems.append(f"{len(res['requests'])} results for {len(req['requests'])} requests")
    for q, r in zip(req["requests"], res["requests"]):
        if r["id"] != q["id"] or r["prompt_token_ids"] != q["prompt_token_ids"]:
            problems.append(f"{q['id']}: the engine's prompt ids are not the request's")
            continue
        outs = r["outputs"]
        finish += [o["finish_reason"] for o in outs]
        if not outs or not outs[0]["token_ids"] or outs[0]["finish_reason"] not in ("stop", "length"):
            problems.append(f"{q['id']}: no tokens or no finish reason")
        lps = [x[1] for o in outs for pos in o["logprobs"] or [] if pos for x in pos]
        lps += [x[1] for pos in r["prompt_logprobs"] if pos for x in pos]
        if not all(math.isfinite(x) for x in lps):
            problems.append(f"{q['id']}: a non-finite log-probability")
        h, ids = by_id.get(q["id"]), q["prompt_token_ids"]
        if h is None or h["positions"] != len(ids) - 1 or len(r["prompt_logprobs"]) != len(ids):
            problems.append(f"{q['id']}: the reference and the engine do not cover the same positions")
            continue
        for i in range(1, len(ids)):
            pos = {int(tk): lp for tk, lp, _ in r["prompt_logprobs"][i] or []}
            if ids[i] not in pos:
                problems.append(f"{q['id']}: position {i} reports no log-probability for its token")
                break
            deltas.append(abs(pos[ids[i]] - h["realized_logprob"][i - 1]))
            if h["top1_margin"][i - 1] >= crit["confident_margin_nats"]:
                confident += 1
                agree += max(pos, key=pos.get) == h["argmax"][i - 1]
    med = statistics.median(deltas) if deltas else None
    rate = agree / confident if confident else None
    measured = {"positions": len(deltas), "median_abs_realized_logprob_delta": med,
                "max_abs_realized_logprob_delta": max(deltas) if deltas else None,
                "p99_abs_realized_logprob_delta": sorted(deltas)[int(0.99 * (len(deltas) - 1))] if deltas else None,
                "confident_positions": confident, "confident_top1_agreement": rate, "finish_reasons": finish,
                "generated_tokens": [len(o["token_ids"]) for r in res["requests"] for o in r["outputs"]],
                "max_model_len": res["engine_args"]["max_model_len"], "load_seconds": res.get("load_seconds"),
                "generate_seconds": res.get("generate_seconds"), "versions": res.get("versions"),
                "device": res.get("device")}
    if med is None or med > crit["median_abs_realized_logprob_delta_max"]:
        problems.append(f"median |delta| {med} exceeds {crit['median_abs_realized_logprob_delta_max']} nats")
    if confident < crit["confident_positions_min"]:
        problems.append(f"{confident} confident positions, fewer than {crit['confident_positions_min']}")
    elif rate < crit["confident_top1_agreement_min"]:
        problems.append(f"confident top-1 agreement {rate:.4f} below {crit['confident_top1_agreement_min']}")
    return {"status": "PASS" if not problems else "FAIL", "problems": problems, "criteria": crit, **measured,
            "scientific_use": False}


def _same_device(requested, *used) -> bool:
    """Every recorded device is the requested one's type (``cuda:0`` and ``cuda`` agree)."""
    if not requested or not all(used):
        return False
    return all(str(u).split(":")[0] == str(requested).split(":")[0] for u in used)


def verdict(out_dir, cfg: dict) -> dict:
    """Every check's status, re-derived from the evidence files and the config alone (no GPU)."""
    out = Path(out_dir)
    t, iso = _load(out / "torch.json") or {}, _load(out / "isolation.json") or {}
    meas, tiny, checks = t.get("measurements") or {}, bool(t.get("tiny")), {}

    m = meas.get("initialization")
    checks["initialization"] = _not_measured("initialization", t, iso) if m is None else {
        **{k: v for k, v in m.items() if k not in ("structural_fold_check", "weights")},
        "status": "PASS" if m["num_parameters"] == m["num_parameters_cost_model"] else "FAIL"}

    sf, bound = (m or {}).get("structural_fold_check"), cfg["structural_fold"]["max_relative_logit_diff"]
    checks["structural_fold"] = _not_measured("initialization", t, iso) if sf is None else {
        **sf, "criterion": f"max |diff| / max |logit| <= {bound} (CPU, float32)",
        "status": "PASS" if sf.get("relative") is not None and sf["relative"] <= bound else "FAIL"}

    f, factor = meas.get("fold_cuda"), float(cfg["fold_acceptance"]["factor"])
    if f is None:
        checks["fold_cuda"] = _not_measured("fold_cuda", t, iso)
    else:
        met = f["max_abs_logit_diff_bf16"] <= factor * f["bf16_vs_fp32_same_model"]
        row = {k: f[k] for k in ("max_abs_logit_diff_fp32", "max_abs_logit_diff_bf16", "bf16_vs_fp32_same_model",
                                 "diagnostics", "positions", "device_name", "attn_implementation", "dtypes")}
        row["criterion"] = cfg["fold_acceptance"]["criterion"]
        if tiny:
            checks["fold_cuda"] = {**row, "status": "NOT_RUN", "rehearsal_criterion_met": met,
                                   "why": "the host rehearsal ran this code on the CPU; it is not a CUDA result"}
        else:
            problems = _device_problems(t, cfg) + _probe_problems(t, cfg)
            want = cfg["numeric_identity"]["attn_implementation"]
            if set(f["attn_implementation"].values()) != {want}:
                problems.append(f"attention ran as {f['attn_implementation']}, not {want}")
            if not met:
                problems.append(f"folded-bf16 {f['max_abs_logit_diff_bf16']:.6g} exceeds {factor} x "
                                f"{f['bf16_vs_fp32_same_model']:.6g}")
            checks["fold_cuda"] = {**row, "problems": problems, "status": "PASS" if not problems else "FAIL"}

    s = meas.get("trainer_smoke")
    checks["trainer_smoke"] = _not_measured("trainer_smoke", t, iso) if s is None else {
        **{k: v for k, v in s.items() if k not in ("losses", "seconds_per_step")},
        "first_loss": s["losses"][0], "last_loss": s["losses"][-1],
        #: the trainer ran where the session asked (a CPU smoke on a GPU pod measures nothing B-E is for)
        "on_the_requested_device": _same_device(s.get("requested_device"), s.get("trainer_device"),
                                                s.get("student_device")),
        "status": "PASS" if (s["losses_finite"] and s["projections_with_a_gradient"] == s["projections"] > 0
                             and s["embedding_unchanged"]
                             and _same_device(s.get("requested_device"), s.get("trainer_device"),
                                              s.get("student_device"))) else "FAIL"}

    r = meas.get("resume")
    if r is None:
        checks["resume"] = _not_measured("resume", t, iso)
    else:
        d, ctl = r["max_abs_param_diff"], r.get("control_max_abs_param_diff")
        ok = d == 0.0 or (ctl is not None and ctl > 0.0 and d <= ctl)
        checks["resume"] = {**r, "status": "PASS" if ok else "FAIL", "criterion": cfg["resume"]["criterion"],
                            "basis": "bit-identical" if d == 0.0 else
                            ("within the device's run-to-run difference" if ok else "differs beyond it")}

    c = meas.get("checkpoint_identity")
    checks["checkpoint_identity"] = _not_measured("checkpoint_identity", t, iso) if c is None else {
        **c, "status": "PASS" if (c["resave_shards_identical"] and c["folded_reload_max_abs_diff"] == 0.0
                                  and c["folded_class"] == "Qwen3ForCausalLM") else "FAIL"}

    v = _load(out / "vllm.json") or {}
    if tiny or v.get("status") == "NOT_RUN":
        why = v.get("why") or "the host rehearsal has no GPU"
        checks["gpu_isolation"] = {"status": "NOT_RUN", "applicable": False, "why": f"not applicable: {why}"}
        checks["vllm"] = {"status": "NOT_RUN", "why": why}
    else:
        child = iso.get("torch_child") or {}
        problems = []
        if child.get("returncode") is None:
            problems.append("the PyTorch child did not exit before the engine")
        if iso.get("parent_imported_torch") is not False:
            problems.append(f"the parent process imported torch ({iso.get('parent_imported_torch')})")
        problems += VC.clear_problems(v.get("gpu_before") or {}, int(cfg["vllm"]["max_used_mib_before_engine"]))
        if not v.get("engine_started"):
            problems.append("the engine was not started")
        checks["gpu_isolation"] = {"status": "PASS" if not problems else "FAIL", "problems": problems,
                                   "torch_child": child, "gpu_before": v.get("gpu_before"),
                                   "gpu_after": v.get("gpu_after"), "criterion": cfg["gpu_isolation"]["criterion"]}
        checks["vllm"] = _vllm_check(out, cfg, t)
    passed = all(ch["status"] == "PASS" or (tiny and k in GPU_ONLY and ch["status"] == "NOT_RUN")
                 for k, ch in checks.items())
    return {"checks": checks, "passed": passed, "tiny": tiny}


# ---- the parent ------------------------------------------------------------------------------------------

def _version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def _marker(status: str | None, passed: bool) -> None:
    if status:
        with Path(status).open("a") as fh:
            fh.write(f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} "
                     f"MARKER:{'ALL_DONE' if passed else 'RUN_FAILED'}\n")


def run(out: Path, work: Path, *, run_id: str, device: str, tiny: bool, steps: int | None = None,
        vllm_python: str = "/opt/vllm/bin/python", status: str | None = None, config_path: str = CONFIG,
        vllm_script: Path | None = None) -> dict | None:
    """The parent: the PyTorch child, then the engine, then the verdict; one terminal marker."""
    out, work = Path(out), Path(work)
    cfg_p = Path(config_path) if Path(config_path).is_absolute() else REPO / config_path
    passed, report = False, None
    try:
        out.mkdir(parents=True, exist_ok=True)
        work.mkdir(parents=True, exist_ok=True)
        cfg = json.loads(cfg_p.read_text())
        steps = steps or int(cfg["smoke"]["tiny_steps" if tiny else "steps"])
        iso: dict = {"parent_imported_torch_at_start": "torch" in sys.modules}
        cmd = [sys.executable, str(Path(__file__).resolve()), "--phase", "torch", "--run-id", run_id,
               "--out", str(out), "--work", str(work), "--device", device, "--steps", str(steps),
               "--config", str(cfg_p)] + (["--tiny"] if tiny else [])
        env = {**os.environ, "CUBLAS_WORKSPACE_CONFIG": cfg["numeric_identity"]["cublas_workspace_config"]}
        t0 = time.time()
        child = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            log, _ = child.communicate(timeout=TORCH_CHILD_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            child.kill()
            log, _ = child.communicate()
        (out / "torch_child.log").write_text((log or "")[-400_000:])
        iso["torch_child"] = {"pid": child.pid, "returncode": child.returncode, "seconds": round(time.time() - t0, 1)}
        if tiny or not device.startswith("cuda"):
            (out / "vllm.json").write_text(json.dumps({"status": "NOT_RUN", "engine_started": False,
                                                       "why": "the host rehearsal has no GPU and no engine"}) + "\n")
        else:
            vllm_phase(out, cfg, vllm_python, script=vllm_script)
        iso["parent_imported_torch"] = "torch" in sys.modules
        (out / "isolation.json").write_text(json.dumps(iso, indent=1) + "\n")
        v = verdict(out, cfg)
        t, vrec = _load(out / "torch.json") or {}, _load(out / "vllm.json") or {}
        vres = _load(out / "vllm/vllm_result.json") or {}
        report = {"schema": SCHEMA, "run_id": run_id, "study": "B-E", "tiny": tiny, "device": device,
                  "config": {"path": config_path, "sha256": hashlib.sha256(cfg_p.read_bytes()).hexdigest()},
                  "environment": {"python": platform.python_version(), "torch": t.get("torch") or _version("torch"),
                                  "transformers": t.get("transformers") or _version("transformers"),
                                  "cuda_runtime": t.get("cuda_runtime"), "gpu": t.get("device_name"),
                                  "capability": t.get("capability"),
                                  "driver": (vrec.get("gpu_before") or {}).get("driver"),
                                  "vllm": (vres.get("versions") or {}).get("vllm")},
                  "checks": v["checks"], "passed": v["passed"], "scientific_use": False, "authorizes": "nothing",
                  "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
        passed = v["passed"]
    finally:
        for d in WORK_DIRS:
            shutil.rmtree(work / d, ignore_errors=True)
        _marker(status, passed)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", required=True, help="the evidence directory (small files only)")
    ap.add_argument("--work", default=None, help="scratch for weights, emptied at the end (default <out>_work)")
    ap.add_argument("--status", default=None, help="the session's status file; one terminal marker is appended")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--vllm-python", default="/opt/vllm/bin/python")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--phase", choices=("all", "torch"), default="all", help="torch: the child (internal)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    work = Path(a.work) if a.work else out.with_name(out.name + "_work")
    if a.phase == "torch":
        cfg = json.loads(Path(a.config if Path(a.config).is_absolute() else REPO / a.config).read_text())
        torch_phase(out, work, cfg, device=a.device, tiny=a.tiny, steps=a.steps)
        return 0
    rep = run(out, work, run_id=a.run_id, device=a.device, tiny=a.tiny, steps=a.steps, vllm_python=a.vllm_python,
              status=a.status, config_path=a.config)
    if rep:
        print(json.dumps({k: c["status"] for k, c in rep["checks"].items()} | {"passed": rep["passed"]}, indent=1))
    return 0 if rep and rep["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
