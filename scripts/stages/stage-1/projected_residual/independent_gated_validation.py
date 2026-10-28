#!/usr/bin/env python3
"""The independent-gated study's engineering validation: every check, one record.

    # on the pod, with the real teacher, D4 Top-1 and the training pack staged by setup
    cd /workspace/aad && HF_HUB_OFFLINE=1 PYTHONPATH=src:scripts:scripts/stages/stage-1 /opt/train/bin/python \\
        scripts/stages/stage-1/projected_residual/independent_gated_validation.py --run-id <run> --out <evidence> \\
        --work <scratch> --status <status file> --device cuda:0 --vllm-python /opt/vllm/bin/python \\
        --pack-root <dir holding e1_2p96M_plus_fluid/> --control-init <D4 Top-1 dir>
    # the same orchestration on the host, on a random small teacher ($0)
    PYTHONPATH=src:scripts:scripts/stages/stage-1 python \\
        scripts/stages/stage-1/projected_residual/independent_gated_validation.py --run-id host --out <dir> --tiny

What a CPU cannot answer, on the card the session runs on: the 1.123B
student's real initialization and its approximation error, CUDA numerics,
native generation, the training step of BOTH arms under the new KD protocol
(peak memory, step time), CUDA resume, checkpoint identity, and the vLLM
out-of-tree model against the Hugging Face reference. Nothing is scored and
nothing trained here is kept as a student. Every threshold is in `CONFIG`.

THE PROCESS BOUNDARY (as the historical B-E): this parent never imports torch;
every PyTorch check runs in a child (``--phase torch``) that EXITS, and only
then does vLLM start, through `shared.validation.vllm_check.run_isolated` on a
GPU it reads clear. Training runs go through the formal entry point,
``scripts/shared/training/train_stage3.py``, on each arm's derived recipe
(`recovery_driver.derived_recipe`) under the config's validation-only overrides
-- on the host too, with a tiny local teacher and a tiny pack of the same
format, so the host rehearsal executes the real path.

The checks (statuses re-derived by `verdict` from the evidence and the config):

1. ``initialization`` -- materialize.py's independent path on the real teacher:
   PASS when the counted parameters equal the design's and the cost model's and
   every initialization-error figure is finite (the figures are recorded).
2. ``numerics_cuda`` -- the student in float32 and bfloat16 on the card over the
   probe: finite; the bf16-vs-fp32 differences recorded (GPU only).
3. ``native_generation`` -- HF greedy decoding with the KV cache equals decoding
   without it, in float32 on the device (bfloat16 agreement recorded).
4. ``smoke_projected`` / ``smoke_control`` -- each arm's recipe, 20 steps on the
   real pack: every step finite, the trainable count the design's, every gate
   with a gradient at step 1 (projected), peak reserved memory leaving the
   configured headroom; median step time recorded.
5. ``resume`` -- the projected recipe, uninterrupted against killed + resumed:
   bit-identical, or no further apart than two uninterrupted runs.
6. ``checkpoint_identity`` -- the saved model reloads to identical logits and
   re-saves to identical shard bytes.
7. ``gpu_isolation`` -- the parent never imported torch, the child exited, the
   GPU read clear before the engine (GPU only).
8. ``vllm_admission`` -- `shared.validation.engine_admission` on the projected
   smoke checkpoint: the out-of-tree vLLM model against the HF bf16 reference,
   judged against the reference's own bf16-vs-fp32 difference (GPU only).
9. ``inference_throughput`` -- greedy decode tokens per second of both arms'
   smoke checkpoints under vLLM, the weight bytes and the KV cache per token:
   measurements for pricing, PASS when both engines produced tokens (GPU only).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from shared.validation import vllm_check as VC  # noqa: E402  (standard library only)

CONFIG = "configs/stages/stage-1/projected_residual/independent_gated_validation.json"
SCHEMA = "aadistill.stage1.projected_residual.independent_gated_validation/v1"
PROJECTED, CONTROL = "Projected-1123M-topk200", "D4-Augmented-topk200"
CHECKS = ("initialization", "numerics_cuda", "native_generation", "smoke_projected", "smoke_control", "resume",
          "checkpoint_identity", "gpu_isolation", "vllm_admission", "inference_throughput")
GPU_ONLY = ("numerics_cuda", "gpu_isolation", "vllm_admission", "inference_throughput")
TRAIN_PY = sys.executable
TRAINER = REPO / "scripts/shared/training/train_stage3.py"
TORCH_CHILD_TIMEOUT_S = 3 * 3600


def _load(p: Path):
    return json.loads(p.read_text()) if Path(p).is_file() else None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _rd():
    from stages.projected_residual import recovery_driver as RD
    return RD


# ---- the host rehearsal's inputs (a tiny teacher, a tiny control, a tiny pack) --------------------------

def tiny_assets(work: Path) -> dict:
    """A random small teacher and control student saved to disk, and a tiny pack in the ladder format."""
    import numpy as np
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from stages.projected_residual.materialize import tiny_inputs

    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    from aadistill.models.teacher import tokenizer_hash

    teacher, selection, items, arch = tiny_inputs()
    work.mkdir(parents=True, exist_ok=True)
    teacher.save_pretrained(work / "teacher")
    #: a 97-entry word-level tokenizer: the tiny models' vocabulary, so the formal entry point's
    #: tokenizer contract and teacher/student equality check run as they do on the pod
    tk = Tokenizer(models.WordLevel({f"t{i}": i for i in range(97)}, unk_token="t0"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="t0", eos_token="t1", pad_token="t1")
    fast.save_pretrained(work / "teacher")
    torch.manual_seed(3)
    ctl = Qwen3ForCausalLM(Qwen3Config(vocab_size=97, hidden_size=16, num_hidden_layers=4, intermediate_size=48,
                                       num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                                       tie_word_embeddings=True, max_position_embeddings=256))
    ctl.save_pretrained(work / "control_init")
    gp, pf = [90, 91, 92], [93, 94]
    g = torch.Generator().manual_seed(5)
    n, length = 24, 64
    ids = torch.randint(0, 89, (n, length), generator=g)
    ce = torch.zeros(n, length, dtype=torch.bool)
    for b in range(n):
        ids[b, 10:13] = torch.tensor(gp)
        ids[b, 13:15] = torch.tensor(pf)
        ce[b, 13:40] = True
    pack = work / "pack" / "e1_2p96M_plus_fluid"
    pack.mkdir(parents=True, exist_ok=True)
    np.savez(pack / "blocks.npz", input_ids=ids.numpy().astype(np.int32), ce_mask=ce.numpy(),
             content_mask=np.ones((n, length), dtype=bool))
    (pack / "audit.jsonl").write_text("".join(json.dumps(
        {"block": b, "sessions": [{"data_type": ("a", "b")[b % 2], "supervised_retained": 27}]}) + "\n"
        for b in range(n)))
    (pack / "ladder.json").write_text(json.dumps({"block_len": length, "rungs": [
        {"target_supervised_tokens": 8 * 27, "n_blocks": 8, "reachable": True}]}) + "\n")
    return {"teacher": work / "teacher", "control_init": work / "control_init", "pack_root": work / "pack",
            "template": {"generation_prompt": gp, "prefill": pf}, "items": items, "arch": arch,
            "tokenizer_sha256": tokenizer_hash(fast)}


# ---- one training run through the formal entry point -----------------------------------------------------

def run_training(cfg: dict, out: Path, *, env: dict, kill_after_save: int | None = None, resume: bool = False,
                 timeout_s: float = 3600, console: Path | None = None) -> dict:
    """train_stage3.py on ``cfg`` (written beside ``out``); returns the train log's events.

    The process's console goes to ``console`` (default ``out/train.log``): the
    validation points it into the evidence, which outlives the work directory."""
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = out / "config.json"
    cfg_path.write_text(json.dumps(cfg, indent=1) + "\n")
    log = Path(cfg["out_dir"]) / "train_log.jsonl"
    cmd = [TRAIN_PY, str(TRAINER), "--config", str(cfg_path)] + (["--resume", "latest"] if resume else [])
    console = Path(console or out / "train.log")
    console.parent.mkdir(parents=True, exist_ok=True)
    with console.open("a") as lf:
        proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT)
        t0, killed = time.time(), False
        while proc.poll() is None:
            if time.time() - t0 > timeout_s:
                proc.kill()
                raise TimeoutError(f"training exceeded {timeout_s}s")
            if kill_after_save is not None and log.is_file() and any(
                    r.get("event") == "checkpoint_saved" and r.get("step") == kill_after_save
                    for r in _events(log)):
                proc.send_signal(signal.SIGKILL)
                proc.wait()
                killed = True
                break
            #: short enough to land between a tiny host run's save point and its next step
            time.sleep(0.02)
    events = _events(log) if log.is_file() else []
    return {"returncode": proc.returncode, "killed_after_save": kill_after_save if killed else None,
            "events": events, "seconds": round(time.time() - t0, 1)}


def _events(log: Path) -> list[dict]:
    rows = []
    for line in Path(log).read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def _final_model(cfg: dict) -> Path:
    d = Path(cfg["out_dir"]) / "checkpoints"
    return d / (d / "latest.txt").read_text().strip() / "model"


def smoke_summary(run: dict, cfg: dict, steady_from: int) -> dict:
    steps = [r for r in run["events"] if r.get("event") == "train_step"]
    start = next((r for r in run["events"] if r.get("event") == "run_start"), {})
    stats = [r for r in run["events"] if r.get("event") == "param_stats"]
    secs = [r["seconds"] for r in steps if r["step"] > steady_from]
    return {"returncode": run["returncode"], "steps": len(steps), "planned": cfg["schedule"]["total_steps"],
            "finite": bool(steps) and all(math.isfinite(r["loss"]) and math.isfinite(r["grad_norm"]) for r in steps),
            "trainable_params": start.get("trainable_params"), "target_policy": start.get("target_policy"),
            "median_step_seconds": statistics.median(secs) if secs else None,
            "peak_allocated_gib": max((r.get("gpu_mem_gb") or 0) for r in steps) if steps else None,
            "peak_reserved_gib": max((r.get("gpu_mem_reserved_gb") or 0) for r in steps) if steps else None,
            "kd_target_in_support": [r.get("kd_target_in_support") for r in steps],
            "kd_teacher_tail_mass": [r.get("kd_teacher_tail_mass") for r in steps],
            "loss": [r["loss"] for r in steps], "kd": [r.get("kd") for r in steps],
            "first_param_stats": stats[0]["groups"] if stats else None, "seconds": run["seconds"]}


def smoke_config(arm: str, scfg: dict, *, work: Path, student: Path, pack_root: Path, device: str, steps: int,
                 save_every: int, teacher: dict | None = None, template: dict | None = None,
                 block_len: int | None = None, rung: int | None = None, tokenizer_sha256: str | None = None) -> dict:
    """The arm's derived recipe for the smoke seed, changed only by the validation-only overrides.

    On the pod the schedule is derived from the staged pack and checked against the
    study's record; the host rehearsal's tiny pack is not that pack, so there the
    record's schedule is taken and the tiny shape overrides it."""
    RD = _rd()
    cfg = RD.derived_recipe(arm, scfg["seed"], run_name=f"validation_{arm}", purpose="engineering validation smoke",
                            out_dir=str(work / "run"), data_dir=str(pack_root / "e1_2p96M_plus_fluid"),
                            student_path=str(student), pack_root=None if teacher is not None else pack_root)
    cfg["schedule"].update(total_steps=steps, warmup_steps=1)
    cfg["checkpoint"].update(save_every=save_every, keep_last=2)
    cfg["intervals"].update(eval_every=0, log_every=1)
    cfg["device"] = device
    if cfg.get("monitor"):
        cfg["monitor"]["every"] = 1
    if teacher is not None:                                   # the host rehearsal only
        cfg.update(teacher=teacher, block_len=block_len, rung=rung, val_blocks=2)
        cfg["loss"]["exclude_template_prefill"] = template
        cfg["intervals"]["eval_blocks"] = 2
        cfg["tokenizer_sha256"] = tokenizer_sha256
        cfg["tokenizer_source"] = teacher["model_id"]
    return cfg


# ---- the PyTorch child -----------------------------------------------------------------------------------

def torch_phase(out: Path, work: Path, cfg: dict, *, device: str, tiny: bool, pack_root: Path | None,
                control_init: Path | None) -> dict:
    import torch
    from transformers import AutoModelForCausalLM

    from aadistill.initialization.adapters.families import register_custom_model_families
    from aadistill.models.cost_model import kv_cache_bytes_per_token
    from shared.validation import engine_admission as EA
    from stages.projected_residual import materialize as M
    from stages.projected_residual.engineering_validation import apply_numeric_identity, render_probe, tiny_probe

    register_custom_model_families()
    doc = {"schema": SCHEMA + "#torch", "device": device, "tiny": tiny, "started_utc": _now(), "checks": {}}
    path = out / "torch.json"

    def save():
        path.write_text(json.dumps(doc, indent=1, default=str) + "\n")

    def check(name, fn):
        t0 = time.time()
        try:
            doc["checks"][name] = {"completed": True, **fn()}
        except Exception as exc:  # every check runs; a failure is recorded, never fatal to the rest
            doc["checks"][name] = {"completed": False, "error": f"{type(exc).__name__}: {exc}"[:1500],
                                   "traceback": traceback.format_exc()[-3000:]}
        doc["checks"][name]["seconds"] = round(time.time() - t0, 1)
        save()

    if device.startswith("cuda"):
        doc["numeric_identity"] = apply_numeric_identity(cfg)
        doc["device_name"] = torch.cuda.get_device_name(0)
        doc["device_total_gib"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
    design = json.loads((REPO / cfg["design"]).read_text())
    assets = tiny_assets(work / "tiny") if tiny else None
    if tiny:
        doc["probe"], probes = tiny_probe({"probe": {"tiny_lengths": [17, 33, 41]}}, 97)
        mat_design = {"architecture": {"projection_sharing": "independent",
                                       "projection_activation": "learnable_silu_residual"}}
        pack_root, control_init = assets["pack_root"], assets["control_init"]
        teacher = {"model_id": str(assets["teacher"]), "revision": None, "dtype": "float32"}
    else:
        doc["probe"], probes = render_probe(cfg)
        mat_design, teacher = design, None
    doc["control_init"] = str(control_init) if control_init else None
    save()
    init_dir = work / "init"
    env = {**os.environ, "PYTHONPATH": f"{REPO / 'src'}:{REPO / 'scripts'}:{REPO / 'scripts/stages/stage-1'}"}
    if tiny:
        env.pop("CUDA_VISIBLE_DEVICES", None)

    def initialization():
        rec = M.run(mat_design, init_dir, device=device, tiny=tiny, probes=probes)
        return {"num_parameters": rec["num_parameters"], "num_parameters_cost_model": rec["num_parameters_cost_model"],
                "geometry": rec["geometry"], "initialization_error": rec["initialization_error"],
                "per_layer": rec["per_layer"], "timings": rec["timings"], "weights": rec["weights"],
                "kv_cache_bytes_per_token_bf16": kv_cache_bytes_per_token(rec["geometry"]),
                "weight_bytes_fp32": sum((init_dir / "model" / w["file"]).stat().st_size for w in rec["weights"])}
    check("initialization", initialization)
    model_dir = init_dir / "model"

    def numerics():
        out_ = {}
        logits = {}
        for dt in (torch.float32, torch.bfloat16):
            m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dt, attn_implementation="sdpa").to(device).eval()
            with torch.no_grad():
                logits[str(dt)] = [m(torch.tensor([ids], device=device), use_cache=False).logits[0].float().cpu()
                                   for ids in probes]
            del m
        f, b = logits["torch.float32"], logits["torch.bfloat16"]
        d = torch.cat([(x - y).abs().flatten() for x, y in zip(f, b)])
        agree = torch.cat([(x.argmax(-1) == y.argmax(-1)) for x, y in zip(f, b)]).float().mean()
        out_.update(max_abs_logit_diff_bf16_vs_fp32=float(d.max()), mean_abs_logit_diff_bf16_vs_fp32=float(d.mean()),
                    argmax_agreement_bf16_vs_fp32=float(agree),
                    finite=bool(all(torch.isfinite(x).all() for x in f + b)))
        return out_
    check("numerics_cuda", numerics)

    def native_generation():
        res = {}
        n = cfg["native_generation"]["new_tokens"]
        for dt in (torch.float32, torch.bfloat16):
            m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dt, attn_implementation="sdpa").to(device).eval()
            same = []
            for ids in probes:
                x = torch.tensor([ids], device=device)
                a = m.generate(x, max_new_tokens=n, do_sample=False, use_cache=True)
                b = m.generate(x, max_new_tokens=n, do_sample=False, use_cache=False)
                same.append(bool(torch.equal(a, b)))
            res[str(dt).replace("torch.", "")] = same
            del m
        return {"cache_equals_no_cache": res}
    check("native_generation", native_generation)

    scfg = cfg["smoke"]
    steps = scfg["tiny_steps"] if tiny else scfg["steps"]
    kw = {} if not tiny else {"teacher": teacher, "template": assets["template"], "block_len": 64, "rung": 8 * 27,
                              "tokenizer_sha256": assets["tokenizer_sha256"]}

    def smoke(arm, student, tag):
        c = smoke_config(arm, scfg, work=work / tag, student=student, pack_root=pack_root, device=device,
                         steps=steps, save_every=steps, **kw)
        run = run_training(c, work / tag, env=env, console=out / "logs" / f"{tag}.log")
        return {"config": c, **smoke_summary(run, c, scfg["steady_state_from_step"] if not tiny else 1),
                "final_model": str(_final_model(c)) if run["returncode"] == 0 else None}
    check("smoke_projected", lambda: smoke(PROJECTED, model_dir, "smoke_projected"))
    check("smoke_control", lambda: smoke(CONTROL, control_init, "smoke_control"))

    def resume():
        rc = cfg["resume"]
        n, at = (4, 2) if tiny else (rc["steps"], rc["save_at"])
        def c(tag, save_every):
            return smoke_config(PROJECTED, scfg, work=work / tag, student=model_dir, pack_root=pack_root,
                                device=device, steps=n, save_every=save_every, **kw)
        cx, cy = c("resume_x", n), c("resume_y", at)
        logs = out / "logs"
        x = run_training(cx, work / "resume_x", env=env, console=logs / "resume_x.log")
        y1 = run_training(cy, work / "resume_y", env=env, kill_after_save=at, console=logs / "resume_y.log")
        y2 = run_training(cy, work / "resume_y", env=env, resume=True, console=logs / "resume_y.log")
        from safetensors.torch import load_file
        wx, wy = load_file(str(_final_model(cx) / "model.safetensors")), load_file(str(_final_model(cy) / "model.safetensors"))
        diff = max(float((wx[k].double() - wy[k].double()).abs().max()) for k in wx)
        out_ = {"x_returncode": x["returncode"], "killed_after_save": y1["killed_after_save"],
                "resume_returncode": y2["returncode"], "keys_equal": sorted(wx) == sorted(wy),
                "max_abs_param_diff": diff, "bit_identical": diff == 0.0,
                "resume_logged": any(r.get("event") == "resume_loaded" for r in y2["events"])}
        if diff != 0.0:
            cx2 = c("resume_x2", n)
            run_training(cx2, work / "resume_x2", env=env, console=logs / "resume_x2.log")
            w2 = load_file(str(_final_model(cx2) / "model.safetensors"))
            out_["run_to_run_max_abs_param_diff"] = max(float((wx[k].double() - w2[k].double()).abs().max()) for k in wx)
        return out_
    check("resume", resume)

    def trained_or_init() -> Path:
        """The smoke's checkpoint, or -- when the smoke produced none (an OOM is a measurement) -- the init."""
        got = (doc["checks"].get("smoke_projected") or {}).get("final_model")
        return Path(got) if got else model_dir

    def checkpoint_identity():
        src = trained_or_init()
        m = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32).to(device).eval()
        again = work / "ident" / "model"
        m.save_pretrained(again)
        m2 = AutoModelForCausalLM.from_pretrained(again, dtype=torch.float32).to(device).eval()
        with torch.no_grad():
            same = all(torch.equal(m(torch.tensor([i], device=device)).logits, m2(torch.tensor([i], device=device)).logits)
                       for i in probes)
        import hashlib
        h = lambda d: {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.glob("*.safetensors"))}  # noqa: E731
        return {"reload_logits_identical": bool(same), "resaved_shards_identical": h(src) == h(again),
                "model_type": m.config.model_type}
    check("checkpoint_identity", checkpoint_identity)

    def admission_reference():
        src = str(trained_or_init())
        ref = EA.hf_reference(src, probes, device=device)
        (out / "hf_reference.json").write_text(json.dumps(ref) + "\n")
        return {"model_dir": src, "prompts": len(ref["prompts"])}
    if not tiny:
        check("admission_reference", admission_reference)
    doc["finished_utc"] = _now()
    doc["probes"] = probes
    save()
    return doc


# ---- the vLLM phase (parent) -------------------------------------------------------------------------------

def vllm_phase(out: Path, cfg: dict, vllm_python: str, t: dict) -> dict:
    from shared.rollout.vllm_plugin_path import vllm_plugin_pythonpath
    from shared.validation import engine_admission as EA

    os.environ["PYTHONPATH"] = vllm_plugin_pythonpath(REPO, os.environ.get("PYTHONPATH"))
    res = {"started_utc": _now()}
    probes = t.get("probes") or []
    ea = cfg["engine_admission"]
    proj = (t["checks"].get("admission_reference") or {}).get("model_dir")
    ctl = (t["checks"].get("smoke_control") or {}).get("final_model") or t.get("control_init")
    ref = _load(out / "hf_reference.json")
    from shared.pod.teacher_cache import local_snapshot
    teacher = json.loads((REPO / cfg["design"]).read_text())["teacher"]
    tok = local_snapshot(teacher["model"], teacher["revision"])
    if proj and ref:
        r = VC.run_isolated(vllm_python, EA.vllm_request(proj, probes, ea["max_model_len"], tokenizer=tok),
                            out / "vllm_admission")
        res["admission_run"] = {k: r.get(k) for k in ("status", "gpu_before", "gpu_after", "returncode", "seconds",
                                                       "problems", "stderr_tail", "engine_started")}
        res["admission"] = EA.compare(ref, r.get("result"), {k: ea[k] for k in EA.CRITERIA_KEYS})
    tp = cfg["throughput"]
    prompts = [probes[i % len(probes)] for i in range(tp["prompts"])] if probes else []
    for name, model in (("projected", proj), ("control", ctl)):
        if not model:
            continue
        req = {"model": model, "tokenizer": tok, "dtype": "bfloat16", "max_model_len": ea["max_model_len"],
               "seed": 0, "requests": [{"id": f"t{i}", "prompt_token_ids": p,
                                        "sampling": {"temperature": 0.0, "max_tokens": tp["new_tokens"],
                                                     "ignore_eos": True}} for i, p in enumerate(prompts)]}
        r = VC.run_isolated(vllm_python, req, out / f"vllm_throughput_{name}")
        doc = r.get("result") or {}
        gen = sum(len(o["token_ids"]) for q in doc.get("requests", []) for o in q["outputs"])
        res[f"throughput_{name}"] = {"status": r.get("status"), "generated_tokens": gen,
                                     "generate_seconds": doc.get("generate_seconds"),
                                     "load_seconds": doc.get("load_seconds"),
                                     "decode_tokens_per_second": (gen / doc["generate_seconds"]
                                                                  if doc.get("generate_seconds") else None),
                                     "versions": doc.get("versions"), "device": doc.get("device"),
                                     "problems": r.get("problems")}
    res["finished_utc"] = _now()
    (out / "vllm.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    return res


# ---- the verdict, from the evidence and the config alone -----------------------------------------------------

def verdict(out_dir, cfg: dict) -> dict:
    out = Path(out_dir)
    t, v, iso = _load(out / "torch.json") or {}, _load(out / "vllm.json") or {}, _load(out / "isolation.json") or {}
    tiny = bool(t.get("tiny"))
    c = t.get("checks") or {}
    exp = cfg["expected"]
    total = t.get("device_total_gib")
    head = cfg["memory"]["headroom_gib_min"]

    def st(ok, why=""):
        return {"status": "PASS" if ok else "FAIL", **({"why": why} if why and not ok else {})}

    def done(name):
        return (c.get(name) or {}).get("completed", False)

    res = {}
    i = c.get("initialization") or {}
    finite = lambda x: all(math.isfinite(v) for v in _numbers(x))  # noqa: E731
    want_n = None if tiny else exp["num_parameters"]
    res["initialization"] = st(done("initialization") and i.get("num_parameters") == i.get("num_parameters_cost_model")
                               and (want_n is None or i.get("num_parameters") == want_n)
                               and finite(i.get("initialization_error")), str(i.get("error") or i.get("num_parameters")))
    n = c.get("numerics_cuda") or {}
    res["numerics_cuda"] = ({"status": "NOT_RUN"} if tiny else st(done("numerics_cuda") and n.get("finite"),
                                                                  str(n.get("error"))))
    g = (c.get("native_generation") or {}).get("cache_equals_no_cache") or {}
    res["native_generation"] = st(done("native_generation") and bool(g.get("float32")) and all(g["float32"]),
                                  str(g or (c.get("native_generation") or {}).get("error")))
    for name, arm in (("smoke_projected", PROJECTED), ("smoke_control", CONTROL)):
        s = c.get(name) or {}
        ok = (done(name) and s.get("returncode") == 0 and s.get("steps") == s.get("planned") and s.get("finite"))
        if not tiny:
            ok = ok and s.get("trainable_params") == exp["trainable_parameters"][arm]
            ok = ok and total is not None and (s.get("peak_reserved_gib") or 1e9) <= total - head
        if arm == PROJECTED and ok:
            gates = (s.get("first_param_stats") or {}).get("gates") or {}
            ok = bool(gates.get("grad_norm")) and gates["grad_norm"] > 0
        res[name] = st(ok, str(s.get("error") or {k: s.get(k) for k in ("returncode", "steps", "trainable_params",
                                                                         "peak_reserved_gib")}))
    r = c.get("resume") or {}
    rr = r.get("run_to_run_max_abs_param_diff")
    res["resume"] = st(done("resume") and r.get("killed_after_save") is not None and r.get("resume_returncode") == 0
                       and r.get("resume_logged") and r.get("keys_equal")
                       and (r.get("bit_identical") or (rr is not None and r["max_abs_param_diff"] <= 2 * rr)),
                       str(r.get("error") or {k: r.get(k) for k in ("max_abs_param_diff", "run_to_run_max_abs_param_diff")}))
    k = c.get("checkpoint_identity") or {}
    res["checkpoint_identity"] = st(done("checkpoint_identity") and k.get("reload_logits_identical")
                                    and k.get("resaved_shards_identical"), str(k.get("error") or k))
    if tiny:
        for name in ("gpu_isolation", "vllm_admission", "inference_throughput"):
            res[name] = {"status": "NOT_RUN"}
    else:
        clear = VC.clear_problems(((v.get("admission_run") or {}).get("gpu_before")) or {})
        res["gpu_isolation"] = st(iso.get("parent_imported_torch") is False and iso.get("child_returncode") == 0
                                  and not clear, str(clear or iso))
        a = v.get("admission") or {}
        res["vllm_admission"] = st(a.get("status") == "PASS", str(a.get("problems") or "no admission"))
        tps = [v.get(f"throughput_{x}") or {} for x in ("projected", "control")]
        res["inference_throughput"] = st(all(x.get("generated_tokens") for x in tps), str(tps))
    ok = all(res[x]["status"] in ("PASS",) or (tiny and res[x]["status"] == "NOT_RUN" and x in GPU_ONLY)
             for x in CHECKS)
    return {"schema": SCHEMA, "tiny": tiny, "checks": res, "status": "PASS" if ok else "FAIL",
            "measured": {"device": t.get("device_name"), "device_total_gib": total,
                         "projected": _measured(c.get("smoke_projected")), "control": _measured(c.get("smoke_control")),
                         "initialization_error": (c.get("initialization") or {}).get("initialization_error"),
                         "numerics_cuda": {k: n.get(k) for k in n if k.endswith("fp32")},
                         "vllm": {x: v.get(x) for x in ("admission", "throughput_projected", "throughput_control")}},
            "scientific_use": False}


def _measured(s: dict | None) -> dict:
    s = s or {}
    return {k: s.get(k) for k in ("median_step_seconds", "peak_allocated_gib", "peak_reserved_gib", "trainable_params",
                                  "target_policy")}


def _numbers(x):
    if isinstance(x, bool) or x is None:
        return
    if isinstance(x, (int, float)):
        yield float(x)
    elif isinstance(x, dict):
        for v in x.values():
            yield from _numbers(v)
    elif isinstance(x, list):
        for v in x:
            yield from _numbers(v)


# ---- the parent ------------------------------------------------------------------------------------------------

def run(out: Path, work: Path, *, run_id: str, device: str, tiny: bool, vllm_python: str | None,
        pack_root: Path | None, control_init: Path | None) -> dict:
    cfg = json.loads((REPO / CONFIG).read_text())
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(Path(__file__).resolve()), "--phase", "torch", "--run-id", run_id, "--out", str(out),
           "--work", str(work), "--device", device] + (["--tiny"] if tiny else []) + (
        ["--pack-root", str(pack_root)] if pack_root else []) + (["--control-init", str(control_init)] if control_init else [])
    t0 = time.time()
    child = subprocess.run(cmd, cwd=REPO, timeout=TORCH_CHILD_TIMEOUT_S)
    iso = {"parent_imported_torch": "torch" in sys.modules, "child_returncode": child.returncode,
           "child_seconds": round(time.time() - t0, 1)}
    (out / "isolation.json").write_text(json.dumps(iso, indent=1) + "\n")
    t = _load(out / "torch.json") or {}
    if not tiny and vllm_python:
        vllm_phase(out, cfg, vllm_python, t)
    rep = {"run_id": run_id, "written_utc": _now(), **verdict(out, cfg)}
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
    for d in (work / x for x in ("init", "smoke_projected", "smoke_control", "resume_x", "resume_y", "resume_x2",
                                  "ident", "tiny")):
        shutil.rmtree(d, ignore_errors=True)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default=None)
    ap.add_argument("--status", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--phase", choices=("all", "torch"), default="all")
    ap.add_argument("--vllm-python", default=None)
    ap.add_argument("--pack-root", default=None)
    ap.add_argument("--control-init", default=None)
    a = ap.parse_args(argv)
    out, work = Path(a.out), Path(a.work or Path(a.out) / "_work")
    pack = Path(a.pack_root) if a.pack_root else None
    ctl = Path(a.control_init) if a.control_init else None
    cfg = json.loads((REPO / CONFIG).read_text())
    if a.phase == "torch":
        torch_phase(out, work, cfg, device=a.device, tiny=a.tiny, pack_root=pack, control_init=ctl)
        return 0
    rep = run(out, work, run_id=a.run_id, device=a.device, tiny=a.tiny, vllm_python=a.vllm_python,
              pack_root=pack, control_init=ctl)
    if a.status:
        Path(a.status).write_text(json.dumps({"terminal": "RUN_DONE", "status": rep["status"]}) + "\n")
    print(json.dumps({"status": rep["status"], "checks": {k: v["status"] for k, v in rep["checks"].items()}}, indent=1))
    return 0 if rep["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
