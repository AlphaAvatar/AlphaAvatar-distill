#!/usr/bin/env python3
"""B-E, the projected-residual student's engineering validation: every check, one record.

    # on the GPU the grant names (L40S), with the real teacher
    PYTHONPATH=src:scripts python scripts/stages/stage-1/projected_residual/engineering_validation.py \\
        --out <dir> --device cuda:0
    # the same driver end to end on the host, on a random small teacher ($0)
    PYTHONPATH=src:scripts python scripts/stages/stage-1/projected_residual/engineering_validation.py \\
        --out <dir> --tiny

It answers what a CPU cannot -- CUDA numerics, real memory, real step time,
the inference engine -- and produces no scientific result: nothing it trains
is kept as a student, nothing is evaluated on eval-set-1, and the design is
not changed by what it measures. The checks, each with its pass criterion:

1. ``initialization`` -- design.json's materialization on the real teacher
   (W0, per-layer statistics, bases, student), with the declared basis's
   captured energy on every layer's input roles AND on the other statistics'
   roles, the sub-layers' outputs included. PASS: it completes and the student's
   counted parameters equal the cost model's.
2. ``fold`` -- the student against its folded stock-Qwen3 form, float32 and
   bfloat16. PASS: the bf16 difference is at most twice the student's own
   bf16-vs-fp32 difference (design.json :: evaluation.fold_acceptance).
3. ``trainer_smoke`` -- ``--steps`` optimizer steps at the recipe's shape
   (D4-Augmented's E1 config for seed 20260726 plus the ``\\.projection$``
   pattern; sequence length, batch size, micro-batch, bf16 autocast, gradient
   checkpointing, the teacher on the fly) on synthetic token blocks. Records
   step time and peak memory; PASS: finite losses, a gradient on every
   projection, the tied embedding unchanged, and the projections'
   orthogonality drift reported.
4. ``resume`` -- the same steps uninterrupted against half, a checkpoint, a
   reload and the other half. PASS: identical parameters; otherwise the
   largest difference is recorded and the check FAILS.
5. ``checkpoint_identity`` -- save, reload, save again: identical shard bytes;
   the folded model saved and reloaded as a stock ``Qwen3ForCausalLM``
   computes the folded model's logits.
6. ``vllm`` -- the folded checkpoint loads in vLLM and generates (GPU only;
   NOT_RUN on the host).

Every check runs even after an earlier one fails, so one session reports all
of them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

DESIGN = "configs/stages/stage-1/projected_residual/design.json"
H1PA = "configs/stages/stage-2/reasoning_data/h1pa.json"
SMOKE_SEED = "20260726"
DIAGNOSTICS = ("residual_input", "read_write")


def shard_hashes(d: Path) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.glob("*.safetensors"))}


def smoke_config(out: Path, *, steps: int, seq_len: int, device: str) -> dict:
    """D4-Augmented's E1 config for the smoke seed, changed only where a smoke must differ."""
    e1 = json.loads((REPO / json.loads((REPO / H1PA).read_text())["recipe"]["e1_configs"][SMOKE_SEED]["path"])
                    .read_text())
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


def run(out: Path, *, device: str, tiny: bool, steps: int) -> dict:
    import torch
    from transformers import AutoModelForCausalLM

    from aadistill.initialization.adapters.projected_residual import (
        fold_to_qwen3, projection_orthogonality, projections, register_auto_classes)
    from aadistill.training.train import Trainer
    from stages.projected_residual import materialize as M

    register_auto_classes()
    design = {} if tiny else json.loads((REPO / DESIGN).read_text())
    rec = {"schema": "aadistill.stage1.projected_residual.engineering_validation/v1", "tiny": tiny,
           "device": device, "checks": {}}
    if device.startswith("cuda"):
        rec["gpu"] = torch.cuda.get_device_name(device)
    checks = rec["checks"]

    # 1-2. initialization and fold
    t0 = time.time()
    m = M.run(design, out / "init", device=device, tiny=tiny, diagnostic_statistics=DIAGNOSTICS)
    fc = m["fold_check"]
    checks["initialization"] = {
        "status": "PASS" if m["num_parameters"] == m["num_parameters_cost_model"] else "FAIL",
        "num_parameters": m["num_parameters"], "seconds": round(time.time() - t0, 1),
        "energy_captured_min": {k: round(min(L[k] for L in m["per_layer"]), 6)
                                for k in m["per_layer"][0] if k.endswith(("_energy_captured", "_declared_basis"))}}
    checks["fold"] = {**fc, "criterion": "max_abs_logit_diff_bf16 <= 2 x bf16_vs_fp32_same_model",
                      "status": "PASS" if fc["max_abs_logit_diff_bf16"] <= 2 * fc["bf16_vs_fp32_same_model"]
                      else "FAIL"}

    # the teacher for KD
    if tiny:
        teacher = M.tiny_inputs()[0]
        seq_len = 64
    else:
        teacher = AutoModelForCausalLM.from_pretrained(design["teacher"]["model"],
                                                       revision=design["teacher"]["revision"],
                                                       dtype=torch.bfloat16)
        seq_len = int(design["architecture"].get("sequence_length", 8192))
    student_dir = out / "init/model"

    def fresh():
        return AutoModelForCausalLM.from_pretrained(student_dir, dtype=torch.float32)

    vocab = fresh().config.vocab_size
    blocks = synthetic_blocks(2 * steps, seq_len, vocab)

    # 3. trainer smoke
    try:
        student = fresh()
        emb0 = student.model.embed_tokens.weight.detach().clone()
        orth0 = projection_orthogonality(student)
        tr = Trainer(smoke_config(out / "smoke", steps=steps, seq_len=seq_len, device=device), student, blocks,
                     teacher=teacher)
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
        times, losses, grads = [], [], None
        for _ in range(steps):
            s = time.time()
            metrics = tr.step_once()
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            times.append(time.time() - s)
            losses.append(float(metrics.get("loss", float("nan"))) if isinstance(metrics, dict) else float("nan"))
            if grads is None:
                grads = [float(p.grad.norm()) if p.grad is not None else 0.0 for p in projections(tr.student)]
        orth1 = projection_orthogonality(tr.student)
        warm = times[2:] or times
        checks["trainer_smoke"] = {
            "steps": steps, "sequence_length": seq_len, "seconds_per_step_mean_after_2": round(sum(warm) / len(warm), 4),
            "peak_memory_gib": (round(torch.cuda.max_memory_allocated(device) / 2**30, 2)
                                if device.startswith("cuda") else None),
            "losses_finite": all(math.isfinite(x) for x in losses), "first_loss": losses[0], "last_loss": losses[-1],
            "projections_with_a_gradient": sum(g > 0 for g in grads or []), "projections": len(grads or []),
            "embedding_unchanged": bool(torch.equal(tr.student.model.embed_tokens.weight.detach().cpu(), emb0.cpu())),
            "orthogonality_rel_fro_max": {"before": max(x["rel_fro"] for x in orth0),
                                          "after": max(x["rel_fro"] for x in orth1)},
            "sigma_range_after": [min(x["sigma_min"] for x in orth1), max(x["sigma_max"] for x in orth1)]}
        c = checks["trainer_smoke"]
        c["status"] = "PASS" if (c["losses_finite"] and c["projections_with_a_gradient"] == c["projections"]
                                 and c["embedding_unchanged"]) else "FAIL"
        full = {n: p.detach().cpu().clone() for n, p in tr.student.named_parameters()}
        del tr, student
    except Exception as exc:  # every check still runs
        checks["trainer_smoke"] = {"status": "FAIL", "error": repr(exc)}
        full = None

    # 4. resume
    try:
        half = steps // 2
        part = Trainer(smoke_config(out / "resume", steps=steps, seq_len=seq_len, device=device), fresh(), blocks,
                       teacher=teacher)
        for _ in range(half):
            part.step_once()
        ckpt = part.save_checkpoint()
        del part
        resumed = Trainer(smoke_config(out / "resume", steps=steps, seq_len=seq_len, device=device),
                          AutoModelForCausalLM.from_pretrained(ckpt / "model", dtype=torch.float32), blocks,
                          teacher=teacher)
        resumed.restore(ckpt)
        for _ in range(steps - half):
            resumed.step_once()
        worst = (max(float((p.detach().cpu() - full[n]).abs().max()) for n, p in resumed.student.named_parameters())
                 if full else None)
        checks["resume"] = {"resumed_at_step": half, "max_abs_param_diff": worst,
                            "status": "PASS" if worst == 0.0 else "FAIL"}
        del resumed
    except Exception as exc:
        checks["resume"] = {"status": "FAIL", "error": repr(exc)}

    # 5. checkpoint identity
    try:
        a, b, f = out / "ident/a", out / "ident/b", out / "ident/folded"
        s1 = fresh()
        s1.save_pretrained(a)
        AutoModelForCausalLM.from_pretrained(a, dtype=torch.float32).save_pretrained(b)
        folded = fold_to_qwen3(s1, dtype=torch.float32)
        folded.save_pretrained(f)
        stock = AutoModelForCausalLM.from_pretrained(f, dtype=torch.float32)
        probe = blocks[0][:1, :min(seq_len, 256)]
        with torch.no_grad():
            diff = float((stock(probe, use_cache=False).logits - folded(probe, use_cache=False).logits).abs().max())
        same = shard_hashes(a) == shard_hashes(b)
        checks["checkpoint_identity"] = {"resave_shards_identical": same, "shards": shard_hashes(a),
                                         "folded_class": type(stock).__name__, "folded_reload_max_abs_diff": diff,
                                         "status": "PASS" if same and diff == 0.0
                                         and type(stock).__name__ == "Qwen3ForCausalLM" else "FAIL"}
    except Exception as exc:
        checks["checkpoint_identity"] = {"status": "FAIL", "error": repr(exc)}

    # 6. vLLM on the folded checkpoint
    if tiny or not device.startswith("cuda"):
        checks["vllm"] = {"status": "NOT_RUN", "why": "GPU only"}
    else:
        code = ("import json,sys,time; from vllm import LLM, SamplingParams; t=time.time(); "
                f"llm=LLM({str(out / 'ident/folded')!r}, tokenizer={design['teacher']['model']!r}, "
                f"tokenizer_revision={design['teacher']['revision']!r}, dtype='bfloat16', max_model_len={seq_len}); "
                "o=llm.generate(['Hello'], SamplingParams(max_tokens=16, temperature=0)); "
                "print(json.dumps({'load_and_generate_s': round(time.time()-t,1), "
                "'tokens': len(o[0].outputs[0].token_ids)}))")
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=1800)
        last = (r.stdout.strip().splitlines() or [""])[-1]
        checks["vllm"] = {"returncode": r.returncode, "result": last, "stderr_tail": r.stderr[-1500:],
                          "status": "PASS" if r.returncode == 0 and '"tokens": 16' in last else "FAIL"}
    rec["materialization"] = {k: m[k] for k in ("geometry", "num_parameters", "weights", "config_sha256", "timings")}
    rec["per_layer"] = m["per_layer"]
    rec["passed"] = all(c["status"] in ("PASS", "NOT_RUN") for c in checks.values())
    (out / "engineering_validation.json").write_text(json.dumps(rec, indent=1) + "\n")
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--steps", type=int, default=20)
    a = ap.parse_args(argv)
    rec = run(Path(a.out), device=a.device, tiny=a.tiny, steps=a.steps)
    print(json.dumps({k: v.get("status") for k, v in rec["checks"].items()} | {"passed": rec["passed"]}, indent=1))
    return 0 if rec["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
