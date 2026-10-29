#!/usr/bin/env python3
"""Research A's L40S engineering check: the phase's formal code paths on diagnostic inputs, measured on the card.

    # on the pod (phase.py launch --kind check), as the launcher runs it
    cd /workspace/aad && HF_HUB_OFFLINE=1 PYTHONPATH=src:scripts:scripts/stages/stage-1 /opt/train/bin/python \\
        scripts/stages/stage-2/teacher_modes/l40s_check.py run --authorization <authorization.json> \\
        --out artifacts/audit/stage2_teacher_modes_phase/<run> --work /workspace/a_check --status <file> \\
        --student artifacts/stages/stage-2/reasoning_data/h1pa/pca_original/20260726
    # the dev box, after the shared closeout: the verified archive's verdict inputs committed under the run,
    # the verdict re-derived, and the record the teacher and recovery kinds bind (`gate`)
    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-2/teacher_modes/l40s_check.py record \\
        --run-id <run> --scr <scratch>

Never formal (its artifact claims engineering_validation only). It answers, on
the card the phase runs on, what no CPU and no A100 record can: whether each
GPU operation of Research A fits an L40S 48 GB with the free margin
``free_memory_wanted_gib`` (4 GiB), how long it takes, and how fast the teacher
generates -- the measurements the teacher session and the recovery units are
priced from. Every criterion is fixed in advance in
``configs/stages/stage-2/teacher_modes/l40s_check.json``. The inputs are
DIAGNOSTIC (build_tasks.py diagnostic, the Stage-0 d4 profiles, the A-E
stand-in student): no eval-set-1 item is read, and nothing built here is
training data, a calibration profile, an initialization or a result.

PHASES, in the config's ``order`` (the parent never initializes CUDA; the
engine runs through the runner's own interpreter, the PyTorch phases in child
processes of this one; ``nvidia-smi`` is sampled every ``sampler_seconds``):

1. ``generation`` -- `teacher_session.run_stage` on the diagnostic tasks, each
   mode: the teacher session's runner, engine settings, limit and per-mode
   template and sampling. Generated tokens, wall seconds, tokens per second,
   the engine's KV cache (its log), the device's peak used memory.
2. ``evaluation_engine`` -- the runner in eval mode at the teacher's supported
   context (40,960), nucleus_v1, the full remaining context per item: the
   engine the teacher's evaluation builds. It starts or it does not; its KV
   capacity and maximum concurrency at 40,960; the peak used memory.
3. ``train`` -- the formal recipe (the seed's E1 config with
   `recovery_units.recipe_overrides`: Top-200 + tail KD at the target
   positions, kd_scope assistant, document isolation, the hybrid on the fly)
   for ``steps`` optimizer steps per mode on that mode's documents packed
   isolated at 8,192, micro-batch 1: peak allocated and reserved memory,
   seconds per step, non-padding tokens per second.
4. ``initialization`` -- D4 Top-1's fixed path on the hybrid root with the
   Stage-0 profiles (`initialization.materialize`), at the forward baseline's
   execution and, only after a CUDA out-of-memory exit, at the design's
   fallback micro-batch: each step's seconds and peak memory.

``verdict(out, config)`` derives every status from the evidence files and the
config alone. An out-of-memory error or a margin below the wanted one is a
FINDING about that operation -- reported with the memory management already
tried -- not a reason to shorten a sequence, drop data or lower a context; only
an operation that does not fit after every such option is evidence for an A100,
for that operation alone (d4_scale_design.json :: execution.hardware).

``gate(operations)`` is what a formal session's issue (and its driver on the
pod) asks of the committed record, FAIL-CLOSED: the phase identity and the
check config as they are now, the run's committed check authorization and
shared closeout, the committed evidence hashing to the record and re-deriving
exactly its verdict and the pod's report, and every operation that session
runs (the config's ``gates``) MEASURED with the wanted margin. A smaller
margin always refuses (maintainer, 2026-10-30: no exception is approved, and
none can be declared from here); it is answered by a protocol-neutral memory
optimization and a new check, never by another card. An exception, if the
maintainer ever approves one, is a new mechanism citing that decision.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

CONFIG = "configs/stages/stage-2/teacher_modes/l40s_check.json"
PROTOCOL = "configs/stages/stage-2/teacher_modes/generation_protocol.json"
H1PA = "configs/stages/stage-2/reasoning_data/h1pa.json"
RECORD = "logs/stages/stage-2/teacher_modes/assets/l40s_check.json"
#: under the check run's directory: the verdict's inputs (the evidence's top-level JSON files), committed
EVIDENCE_COPY = "check_evidence"
SCHEMA = "aadistill.stage2.teacher_modes.l40s_check/v1"
MODES = ("H-T", "H-N")
KV_TOKENS = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens")
KV_CONCURRENCY = re.compile(r"Maximum concurrency for\s*([\d,]+)\s*tokens per request:\s*([\d.]+)x")


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_config(path: str | Path = CONFIG) -> dict:
    return json.loads((REPO / path).read_text())


def _write(out: Path, name: str, doc) -> None:
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / f"{name}.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")


def _read(out: Path, name: str):
    p = Path(out) / f"{name}.json"
    return json.loads(p.read_text()) if p.is_file() else None


def _sha(p: Path) -> str | None:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).is_file() else None


def _plain(doc):
    """`doc` as its JSON file reads back (what a comparison with a written verdict compares)."""
    return json.loads(json.dumps(doc, default=str))


def diagnostic_tasks(cfg: dict) -> list[dict]:
    d = cfg["diagnostic_tasks"]
    p = REPO / d["path"]
    if hashlib.sha256(p.read_bytes()).hexdigest() != d["sha256"]:
        raise SystemExit(f"{d['path']}: not the diagnostic tasks the check config binds")
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def authorize(path: Path, repo: Path) -> dict:
    from stages.teacher_modes import phase as PH
    return PH.require_on_pod(path, "check", repo)


def contract(cfg: dict, authorization: Path, authorize=authorize, repo: Path = REPO) -> dict:
    """The teacher session's contract shape on the diagnostic tasks (`teacher_session.run_stage` reads it)."""
    bound = authorize(authorization, repo)
    tasks = diagnostic_tasks(cfg)
    return {**bound, "protocol": json.loads((repo / PROTOCOL).read_text()), "protocol_path": Path(PROTOCOL),
            "tasks": {t["task_id"]: t for t in tasks}, "ids": [t["task_id"] for t in tasks],
            "launch_identity_sha256": bound["identity"]}


# ---- the device, read from outside every process that uses it ----------------------------------------------

class Sampler:
    """``nvidia-smi`` every `every` seconds; per phase the peak used and the total memory (MiB)."""

    def __init__(self, every: float, query=None):
        self.every, self.query = float(every), query or self._smi
        self.peaks: dict[str, dict] = {}
        self._phase, self._stop = None, threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    @staticmethod
    def _smi() -> tuple[int, int] | None:
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=20)
            used, total = (int(x) for x in r.stdout.strip().splitlines()[0].split(","))
            return used, total
        except Exception:  # noqa: BLE001 -- a missing sample is recorded as none, never as zero
            return None

    def _loop(self) -> None:
        while not self._stop.wait(self.every):
            self.sample()

    def sample(self) -> None:
        got, name = self.query(), self._phase
        if got is None or name is None:
            return
        p = self.peaks.setdefault(name, {"used_mib_peak": 0, "total_mib": got[1], "samples": 0})
        p["used_mib_peak"] = max(p["used_mib_peak"], got[0])
        p["samples"] += 1

    def start(self) -> "Sampler":
        self._thread.start()
        return self

    def phase(self, name: str | None) -> None:
        self.sample()
        self._phase = name
        self.sample()

    def stop(self) -> None:
        self._stop.set()


def kv_lines(logs: list[Path]) -> dict:
    """The engine's own report of its KV cache, from its logs; absent lines are recorded absent."""
    out = {"kv_cache_tokens": None, "max_concurrency": None, "at_tokens_per_request": None}
    for log in logs:
        text = log.read_text(errors="replace") if log.is_file() else ""
        m = KV_TOKENS.search(text)
        if m and out["kv_cache_tokens"] is None:
            out["kv_cache_tokens"] = int(m.group(1).replace(",", ""))
        m = KV_CONCURRENCY.search(text)
        if m and out["max_concurrency"] is None:
            out["at_tokens_per_request"] = int(m.group(1).replace(",", ""))
            out["max_concurrency"] = float(m.group(2))
    return out


def _turns(recs: dict) -> list[dict]:
    return [t for rs in recs.values() for r in rs for t in r.get("turns", [])]


# ---- 1. generation ------------------------------------------------------------------------------------------

def phase_generation(cfg: dict, c: dict, out: Path, run, mark) -> dict:
    from stages.teacher_modes import teacher_session as TS
    g = cfg["generation"]
    ids = c["ids"][:int(g["tasks"])]
    shards = out / "generation"
    shards.mkdir(parents=True, exist_ok=True)
    rec = {"tasks": len(ids), "modes": {}}
    for mode in g["modes"]:
        t0 = time.time()
        got = TS.run_stage(c, shards, {"stage": "diag", "modes": (mode,), "ids": ids}, run, mark,
                           candidates=int(g["candidates"]))
        secs = time.time() - t0
        turns = _turns(got["records"].get(mode, {}))
        tokens = sum(int(t.get("generated_tokens") or 0) for t in turns)
        stops: dict[str, int] = {}
        for t in turns:
            stops[str(t.get("stop_reason"))] = stops.get(str(t.get("stop_reason")), 0) + 1
        rec["modes"][mode] = {"complete": got["complete"], "turns": len(turns), "generated_tokens": tokens,
                              "wall_seconds": round(secs, 1),
                              "tokens_per_second": round(tokens / secs, 1) if secs > 0 else None,
                              "stop_reasons": dict(sorted(stops.items())),
                              "engine": kv_lines(sorted(shards.glob(f"diag_{mode}_*.log")))}
    return rec


# ---- 2. the evaluation engine at the supported context ----------------------------------------------------

def phase_engine(cfg: dict, c: dict, out: Path, run) -> dict:
    from stages.reasoning_data import generation_driver as GD
    from stages.teacher_modes import teacher_session as TS
    e, proto = cfg["evaluation_engine"], c["protocol"]
    d = out / "evaluation_engine"
    d.mkdir(parents=True, exist_ok=True)
    files = TS.write_inputs(c, d, "engine", c["ids"][:int(e["tasks"])])
    ctx = int(proto["teacher_evaluation"]["context"])
    rows, logs, t0 = [], [], time.time()
    rcs = []
    for f in files:
        label = f"engine_{Path(f['file']).stem}"
        cmd = [GD.VLLM_PY, GD.RUNNER, "--model", proto["teacher"]["model"], "--revision", proto["teacher"]["revision"],
               "--label", label, "--tasks", str(d / "inputs" / f["file"]), "--mode", "eval",
               "--out", str(d / f"{label}.jsonl"), "--trained-context", str(ctx),
               "--decoding", str(REPO / e["decoding"]), "--seed-namespace", e["seed_namespace"],
               "--system", f["system"], "--action-protocol", proto["execution"]["action_protocol"],
               "--chat-template-kwargs", json.dumps(proto["modes"][e["mode"]]["template_kwargs"], sort_keys=True)]
        log = d / f"{label}.log"
        rcs.append(run(cmd, log, lambda: None))
        logs.append(log)
        p = d / f"{label}.jsonl"
        rows += [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.is_file() else []
    turns = [t for r in rows for t in r.get("turns", [])]
    return {"context": ctx, "mode": e["mode"], "tasks": sum(f["tasks"] for f in files), "returncodes": rcs,
            "started": bool(rcs) and all(rc == 0 for rc in rcs), "episodes": len(rows),
            "generated_tokens": sum(int(t.get("generated_tokens") or 0) for t in turns),
            "context_limit_reached": sum(bool(t.get("context_limit_reached")) for t in turns),
            "wall_seconds": round(time.time() - t0, 1), "engine": kv_lines(logs)}


# ---- 3. the formal training step --------------------------------------------------------------------------

def train_config(cfg: dict, mode: str, student: str, out_dir: Path) -> tuple[dict, list[str]]:
    """The seed's E1 config with the formal recipe's declared changes, and where/how long the check runs."""
    from stages.reasoning_data import h1pa_driver as HD
    from stages.reasoning_data import unit_driver as UD
    from stages.teacher_modes import recovery_units as RU
    tr = cfg["train"]
    e1_ref = json.loads((REPO / H1PA).read_text())["recipe"]["e1_configs"][tr["e1_seed"]]
    e1 = json.loads((REPO / e1_ref["path"]).read_text())
    if HD.canonical_sha(e1) != e1_ref["canonical_sha256"]:
        raise SystemExit(f"{e1_ref['path']}: not the E1 config H1-PA records")
    d = copy.deepcopy(e1)
    for key, value in RU.recipe_overrides(json.loads((REPO / PROTOCOL).read_text())).items():
        *parents, leaf = key.split(".")
        target = d
        for part in parents:
            target = target.setdefault(part, {})
        target[leaf] = copy.deepcopy(value)
    d.update(run_name=f"l40s_check_train_{mode}", _purpose="L40S check (engineering, never formal)",
             student_path=str(student), out_dir=str(out_dir), device="cuda:0")
    d["schedule"].update(total_steps=int(tr["steps"]), warmup_steps=1)
    d["checkpoint"]["save_every"] = 0
    d["intervals"].update(log_every=1, eval_every=0, eval_blocks=0)
    a, b = UD._flat(e1), UD._flat(d)
    return d, sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def mode_documents(shards: Path, identity: str, mode: str, stop_ids) -> list:
    """Every sealed turn of `mode` that forms a token document; correctness is never read."""
    from aadistill.data.token_documents import DocumentError, document_from_turn
    from stages.teacher_modes import admission as ADM
    docs = []
    for (tid, m), recs in sorted(ADM.records(shards, identity).items()):
        if m != mode:
            continue
        for k, t in enumerate(recs[0]["turns"]):
            try:
                docs.append(document_from_turn(t, doc_id=f"{tid}|{mode}|t{k}", stop_ids=stop_ids))
            except DocumentError:
                continue
    return docs


def phase_train(args) -> dict:
    """Child process: the formal training step on one mode's diagnostic documents."""
    import torch
    from transformers import AutoModelForCausalLM

    from aadistill.data.token_documents import pack_isolated
    from aadistill.models.teacher import load_teacher
    from aadistill.training.train import Trainer

    cfg, proto = load_config(args.config), json.loads((REPO / PROTOCOL).read_text())
    tr, mode, dev = cfg["train"], args.mode, "cuda:0"
    docs = mode_documents(Path(args.out) / "generation", args.identity, mode, proto["teacher"]["stop_ids"])
    p = pack_isolated(docs, int(tr["sequence_length"]), int(tr["pad_id"]), order_seed=int(tr["order_seed"]))
    tcfg, changed = train_config(cfg, mode, args.student, Path(args.work) / f"train_{mode}")
    t = tcfg["teacher"]
    teacher, _tok, teacher_identity = load_teacher(t["model_id"], t["revision"], dtype=t["dtype"], device=dev)
    student = AutoModelForCausalLM.from_pretrained(args.student, dtype=torch.float32)
    trainer = Trainer(tcfg, student, p.blocks(), teacher=teacher, device=dev, out_dir=Path(args.work) / f"train_{mode}")
    torch.cuda.reset_peak_memory_stats(dev)
    steps, secs = [], []
    for _ in range(int(tr["steps"])):
        t0 = time.time()
        m = trainer.step_once()
        torch.cuda.synchronize(dev)
        secs.append(round(time.time() - t0, 4))
        steps.append({k: m.get(k) for k in ("step", "loss", "ce", "kd", "grad_norm", "ce_targets", "kd_positions",
                                            "executed_nonpad_tokens", "context_strategy")})
    warm = secs[int(tr["warmup_steps_excluded"]):] or secs
    nonpad = [s["executed_nonpad_tokens"] or 0 for s in steps[int(tr["warmup_steps_excluded"]):]]
    rec = {"mode": mode, "device": torch.cuda.get_device_name(dev), "torch": torch.__version__,
           "documents": len(docs), "pack": p.stats(), "config_changed_keys": changed,
           "kd_support": tcfg["loss"].get("kd_support"), "steps": steps, "seconds_per_step": secs,
           "seconds_per_step_mean_after_warmup": round(sum(warm) / len(warm), 4),
           "nonpad_tokens_per_second": round(sum(nonpad) / sum(warm), 1) if sum(warm) else None,
           "memory": {"allocated_gib": round(torch.cuda.max_memory_allocated(dev) / 2**30, 2),
                      "reserved_gib": round(torch.cuda.max_memory_reserved(dev) / 2**30, 2),
                      "total_gib": round(torch.cuda.get_device_properties(dev).total_memory / 2**30, 2)},
           "teacher": {k: teacher_identity.get(k) for k in ("model_id", "revision") if k in teacher_identity}}
    _write(args.out, f"train_{mode}", rec)
    return rec


# ---- 4. the initialization path ---------------------------------------------------------------------------

def phase_init(args) -> int:
    """Child process: D4 Top-1's fixed path on the hybrid with the Stage-0 profiles at one micro-batch."""
    from aadistill.initialization.calibration.profiles import load_profiles
    from stages.teacher_modes import initialization as IN
    cfg = load_config(args.config)
    ic = cfg["initialization"]
    for p, sha in ic["items"].items():
        if hashlib.sha256((REPO / p).read_bytes()).hexdigest() != sha:
            raise SystemExit(f"{p}: not the Stage-0 items the check config binds")
    doc = json.loads((REPO / ic["profiles_document"]).read_text())
    keep = set(ic["profiles"].values())
    load_profiles({**doc, "profiles": [p for p in doc["profiles"] if f"{p['profile_id']}@v{p['version']}" in keep]})
    spec = IN.fixed_path_spec("cuda:0", profile_of=ic["profiles"])
    ex = IN.execution(args.micro_batch_size)
    work = Path(args.work) / f"init_mb{ex.micro_batch_size}"
    steps: list = []
    t0 = time.time()
    rec = {"execution": ex.as_trace(), "path": [s.label for s in spec.steps]}
    try:
        res = IN.materialize(spec, ex, work, "cuda:0", steps)
        rec.update(out_of_memory=False, steps=steps, seconds=round(time.time() - t0, 1),
                   final={"num_parameters": res[-1].identity.num_parameters,
                          "artifact_digest": res[-1].identity.artifact_digest})
        rc = 0
    except Exception as exc:  # noqa: BLE001 -- recorded, then the exit status tells the parent what happened
        oom = IN.is_out_of_memory(exc)
        rec.update(out_of_memory=oom, steps=steps, seconds=round(time.time() - t0, 1), error=str(exc)[:2000],
                   traceback=traceback.format_exc()[-3000:])
        rc = IN.OOM_EXIT if oom else 1
    _write(args.out, f"initialization_mb{ex.micro_batch_size}", rec)
    shutil.rmtree(work, ignore_errors=True)
    return rc


# ---- the parent ---------------------------------------------------------------------------------------------

def _child(args, phase: str, extra: list[str], log: Path, timeout: float) -> int:
    cmd = [sys.executable, str(Path(__file__).resolve()), "child", "--phase", phase, "--config", args.config,
           "--out", args.out, "--work", args.work, "--student", args.student, "--identity", args.identity, *extra]
    with log.open("a") as lf:
        try:
            return subprocess.run(cmd, cwd=REPO, stdout=lf, stderr=subprocess.STDOUT, timeout=timeout,
                                  env={**os.environ, "PYTHONPATH": "src:scripts:scripts/stages/stage-1"}).returncode
        except subprocess.TimeoutExpired:
            return 124


def run(args, *, runner=None, authorize=authorize, sampler: Sampler | None = None) -> dict:
    from stages.reasoning_data import generation_driver as GD
    from stages.teacher_modes import initialization as IN
    runner = runner or GD.run_battery
    cfg, out, status = load_config(args.config), Path(args.out), Path(args.status)
    out.mkdir(parents=True, exist_ok=True)

    def mark(m: str) -> None:
        with status.open("a") as fh:
            fh.write(f"{utc()} MARKER:{m}\n")
        print(f"MARKER:{m}", flush=True)

    run_doc = {"schema": SCHEMA, "started_utc": utc(), "config_sha256": hashlib.sha256((REPO / args.config).read_bytes()).hexdigest()}
    try:
        c = contract(cfg, Path(args.authorization), authorize)
    except BaseException as exc:  # noqa: BLE001
        run_doc["contract"] = {"status": "failed", "error": repr(exc)}
        _write(out, "run", run_doc)
        mark("RUN_FAILED")
        return run_doc
    args.identity = c["launch_identity_sha256"]
    run_doc["contract"] = {"status": "ok", "identity": c["identity"], "harness": c.get("harness"),
                           "authorization_id": c.get("authorization_id")}
    _write(out, "run", run_doc)
    mark("CONTRACT_OK")
    sampler = (sampler or Sampler(cfg["sampler_seconds"])).start()
    to = cfg["timeouts_s"]
    try:
        for phase in cfg["order"]:
            sampler.phase(phase)
            try:
                if phase == "generation":
                    _write(out, "generation", phase_generation(cfg, c, out, runner, mark))
                elif phase == "evaluation_engine":
                    _write(out, "evaluation_engine", phase_engine(cfg, c, out, runner))
                elif phase == "train":
                    rcs = {m: _child(args, "train", ["--mode", m], out / "train.log", to["train"]) for m in MODES}
                    _write(out, "train_exits", rcs)
                elif phase == "initialization":
                    sizes, rcs = [IN.execution().micro_batch_size,
                                  int(json.loads((REPO / "configs/stages/stage-2/teacher_modes/d4_scale_design.json")
                                                 .read_text())["initialization"]["execution"]["on_out_of_memory"])], {}
                    for k in dict.fromkeys(sizes):
                        rcs[k] = _child(args, "init", ["--micro-batch-size", str(k)], out / "initialization.log",
                                        to["initialization"])
                        if rcs[k] != IN.OOM_EXIT:
                            break
                    _write(out, "initialization_exits", {str(k): v for k, v in rcs.items()})
            except Exception as exc:  # noqa: BLE001 -- every later phase still runs; the verdict reads the gap
                _write(out, f"{phase}_error", {"error": repr(exc), "traceback": traceback.format_exc()[-4000:]})
            mark(f"PHASE_DONE:{phase}")
    finally:
        sampler.phase(None)
        sampler.stop()
        _write(out, "device", sampler.peaks)
    report = verdict(out, cfg)
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    mark("ALL_DONE" if report["complete"] else "RUN_FAILED")
    return report


# ---- the verdict, from the evidence alone -------------------------------------------------------------------

def _margin(total_gib: float | None, peak_gib: float | None, want: float) -> dict:
    if total_gib is None or peak_gib is None:
        return {"total_gib": total_gib, "peak_gib": peak_gib, "free_gib": None, "fits_with_margin": None}
    free = round(total_gib - peak_gib, 2)
    return {"total_gib": total_gib, "peak_gib": peak_gib, "free_gib": free, "fits_with_margin": free >= want}


def _smi(device: dict, phase: str) -> tuple[float | None, float | None]:
    p = (device or {}).get(phase)
    if not p or not p.get("samples"):
        return None, None
    return round(p["total_mib"] / 1024, 2), round(p["used_mib_peak"] / 1024, 2)


def verdict(out: str | Path, cfg: dict) -> dict:
    """Every status from the evidence files and the config alone (the dev box re-runs it on the secured copy)."""
    out, want = Path(out), float(cfg["free_memory_wanted_gib"])
    device = _read(out, "device") or {}
    ops: dict[str, dict] = {}

    def absent(phase: str, ran: str | None = None) -> dict:
        """No evidence: FAILED when the phase raised or its child ran, else NOT_RUN."""
        err = _read(out, f"{phase}_error")
        if err or (ran and (out / f"{ran}.json").is_file()):
            return {"status": "FAILED", **({"error": err.get("error")} if err else {})}
        return {"status": "NOT_RUN"}
    gen = _read(out, "generation")
    if gen:
        total, peak = _smi(device, "generation")
        ops["generation"] = {"status": "MEASURED" if all(m["complete"] for m in gen["modes"].values()) else "FAILED",
                             "memory": _margin(total, peak, want), "management": "the engine's own allocation "
                             "(gpu_memory_utilization 0.90 of the card, the runner's setting)",
                             "tokens_per_second": {m: v["tokens_per_second"] for m, v in gen["modes"].items()},
                             "engine": {m: v["engine"] for m, v in gen["modes"].items()}}
    else:
        ops["generation"] = absent("generation")
    eng = _read(out, "evaluation_engine")
    if eng:
        total, peak = _smi(device, "evaluation_engine")
        ops["evaluation_engine"] = {"status": "MEASURED" if eng["started"] else "FAILED", "context": eng["context"],
                                    "memory": _margin(total, peak, want), "engine": eng["engine"],
                                    "management": "the engine's KV capacity at the full context; a lower context is "
                                                  "never an option (P18)"}
    else:
        ops["evaluation_engine"] = absent("evaluation_engine")
    for mode in MODES:
        t = _read(out, f"train_{mode}")
        ops[f"train_{mode}"] = ({"status": "MEASURED", "memory": _margin(t["memory"]["total_gib"],
                                                                        t["memory"]["reserved_gib"], want),
                                 "allocated_gib": t["memory"]["allocated_gib"],
                                 "seconds_per_step": t["seconds_per_step_mean_after_warmup"],
                                 "nonpad_tokens_per_second": t["nonpad_tokens_per_second"],
                                 "documents": t["documents"], "management": "micro-batch 1 and activation "
                                 "checkpointing already (the recipe's); Top-K rows chunked by 512"}
                                if t else {**absent("train", "train_exits"),
                                           "exit": (_read(out, "train_exits") or {}).get(mode),
                                           "out_of_memory": bool((_read(out, f"train_{mode}_error") or {})
                                                                 .get("out_of_memory"))})
    tried = []
    for p in sorted(out.glob("initialization_mb*.json")):
        r = json.loads(p.read_text())
        mem = [s.get("device_memory") or {} for s in r.get("steps", [])]
        peak = max((m.get("reserved_gib") or 0 for m in mem), default=None)
        total = next((m.get("total_gib") for m in mem if m.get("total_gib")), None)
        tried.append({"micro_batch_size": r["execution"]["micro_batch_size"], "out_of_memory": r["out_of_memory"],
                      "completed": "final" in r, "seconds": r.get("seconds"),
                      "steps": [{"impl_id": s["impl_id"], "seconds": s["seconds"], **(s.get("device_memory") or {})}
                                for s in r.get("steps", [])],
                      "memory": _margin(total, peak, want) if peak else _margin(None, None, want)})
    done = [t for t in tried if t["completed"]]
    ops["initialization"] = {**({"status": "MEASURED"} if done else {"status": "FAILED"} if tried else
                                absent("initialization", "initialization_exits")),
                             "attempts": tried, "exits": _read(out, "initialization_exits"),
                             "management": "the design's micro-batch fallback (initialization.execution)"}
    if done:
        ops["initialization"]["memory"] = done[-1]["memory"]
    complete = all(o["status"] == "MEASURED" for o in ops.values())
    short = sorted(k for k, o in ops.items() if (o.get("memory") or {}).get("fits_with_margin") is False)
    unfit = sorted(k for k, o in ops.items() if o["status"] == "FAILED")
    return {"schema": SCHEMA, "card": cfg["card"], "free_memory_wanted_gib": want, "operations": ops,
            "complete": complete, "fit_with_margin": complete and not short,
            "passed": complete, "checks": {k: {"status": o["status"]} for k, o in ops.items()},
            "_passed": "every operation measured (the shared closeout's re-derivation reads passed and checks); "
                       "whether each fits with the wanted margin is fit_with_margin and below_margin",
            "below_margin": short, "failed": unfit,
            "a100_evidence": ([k for k in unfit if k.startswith("train_") and ops[k].get("out_of_memory")]
                              + (["initialization"] if "initialization" in unfit and len(tried) >= 2
                                 and all(t["out_of_memory"] for t in tried) else [])),
            "_a100_evidence": ("an operation that failed out of memory after every declared memory management "
                               "option; anything else that failed is an engineering failure to diagnose, not a "
                               "reason for another card"),
            "reads_no_eval_set": True, "formal": False, "authorizes": "nothing"}


def record(run_id: str, scr: Path, repo: Path = REPO) -> dict:
    """The dev box's record of a closed check: the evidence archive the shared closeout verified, its verdict
    inputs committed under the run (``EVIDENCE_COPY``), the verdict re-derived from that copy beside the pod's,
    and the run's identities -- what `gate` re-checks."""
    from stages.teacher_modes import phase as PH
    repo = Path(repo)
    run = repo / PH.STUDY.run_dir(run_id)
    outcome = _read(run / "closeout", "outcome") or {}
    tgz, ev = Path(scr) / "store" / PH.SE.EVIDENCE_ARCHIVE, outcome.get("evidence") or {}
    if outcome.get("kind") != "check" or not ev.get("verified") or _sha(tgz) != ev.get("sha256"):
        raise SystemExit(f"{run_id}: no closed check whose verified evidence archive is {tgz}")
    copy = run / EVIDENCE_COPY
    shutil.rmtree(copy, ignore_errors=True)
    copy.mkdir(parents=True)
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(tgz) as t:
            t.extractall(tmp, filter="data")
        for q in sorted(Path(tmp).glob("*.json")):
            shutil.copyfile(q, copy / q.name)
    run_doc = _read(copy, "run") or {}
    v = _plain(verdict(copy, load_config(repo / CONFIG)))
    rec = {"schema": f"{SCHEMA}_record", "run_id": run_id, "written_utc": utc(),
           "identity": (run_doc.get("contract") or {}).get("identity"),
           "authorization_id": (run_doc.get("contract") or {}).get("authorization_id"),
           "harness": (run_doc.get("contract") or {}).get("harness"), "config_sha256": run_doc.get("config_sha256"),
           "evidence_archive_sha256": ev["sha256"], "evidence": str(copy.relative_to(repo)),
           "evidence_files": {q.name: _sha(q) for q in sorted(copy.glob("*.json"))},
           "verdict": v, "agrees_with_the_pod": v == _read(copy, "report"),
           "use": ("what the teacher session and the recovery units are priced from (phase_forecast.py), and what "
                   "their artifacts bind once `gate` admits their operations (phase.kind_conditions)"),
           "authorizes": "nothing"}
    dest = repo / RECORD
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    return rec


def gate(operations, repo: Path = REPO) -> list[str]:
    """Why the committed record does NOT admit a session that runs `operations` (empty: it does). Fail-closed:
    anything unmeasured, failed, unverifiable or not this phase's refuses."""
    from stages.teacher_modes import phase as PH
    repo = Path(repo)
    if not (repo / RECORD).is_file():
        return [f"{RECORD} does not exist: the L40S check (kind check) and its dev-box record come first"]
    rec = json.loads((repo / RECORD).read_text())
    if rec.get("schema") != f"{SCHEMA}_record":
        return [f"{RECORD} is not a check record"]
    bad = []
    if rec.get("config_sha256") != _sha(repo / CONFIG):
        bad.append("the check config is not the one the check ran under")
    if rec.get("identity") != PH.SE.identity_hash(PH.STUDY, repo):
        bad.append("the check ran under another phase identity than the files hold now")
    try:
        run = repo / PH.STUDY.run_dir(str(rec.get("run_id") or ""))
        auth = PH.PhaseAuthorization.load(repo / PH.STUDY.authorization_path(rec["run_id"]))
    except Exception as exc:  # noqa: BLE001 -- every reason is a refusal
        return bad + [f"run {rec.get('run_id')!r} has no committed check authorization ({exc})"]
    if (auth.stage_conditions or {}).get("kind") != "check" or auth.authorization_id != rec.get("authorization_id"):
        bad.append(f"run {rec['run_id']}'s committed authorization is not the check the record names")
    out = _read(run / "closeout", "outcome") or {}
    ev = out.get("evidence") or {}
    if (out.get("kind") != "check" or ev.get("verified") is not True or ev.get("sha256") != rec.get("evidence_archive_sha256")
            or (out.get("verdict_rederived") or {}).get("agrees_with_the_pod") is not True):
        bad.append(f"run {rec['run_id']}'s shared closeout did not verify this evidence archive and its verdict")
    copy = run / EVIDENCE_COPY
    if not rec.get("evidence_files") or {q.name: _sha(q) for q in sorted(copy.glob("*.json"))} != rec["evidence_files"]:
        return bad + ["the committed evidence is not the evidence the record hashed"]
    run_doc = _read(copy, "run") or {}
    if ((run_doc.get("contract") or {}).get("identity"), (run_doc.get("contract") or {}).get("authorization_id"),
            run_doc.get("config_sha256")) != (rec.get("identity"), rec.get("authorization_id"), rec.get("config_sha256")):
        bad.append("the evidence's own contract names other identities than the record")
    v = _plain(verdict(copy, load_config(repo / CONFIG)))
    if v != rec.get("verdict"):
        bad.append("the record's verdict is not the one its evidence derives")
    if v != _read(copy, "report"):
        bad.append("the pod's verdict and the dev box's re-derivation disagree")
    for op in operations:
        o = v["operations"].get(op) or {}
        mem = o.get("memory") or {}
        if o.get("status") != "MEASURED":
            bad.append(f"{op}: {o.get('status') or 'not in the evidence'}")
        elif mem.get("fits_with_margin") is not True:
            bad.append(f"{op}: free memory {mem.get('free_gib')} GiB, not the {v['free_memory_wanted_gib']} GiB wanted "
                       "(a protocol-neutral memory optimization and a new check; no exception is approved)")
    return bad


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--authorization", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--work", required=True)
    r.add_argument("--status", required=True)
    r.add_argument("--student", required=True)
    r.add_argument("--config", default=CONFIG)
    c = sub.add_parser("child")
    c.add_argument("--phase", required=True, choices=("train", "init"))
    for k in ("--config", "--out", "--work", "--student", "--identity"):
        c.add_argument(k, required=True)
    c.add_argument("--mode", default=None)
    c.add_argument("--micro-batch-size", type=int, default=None)
    rec = sub.add_parser("record")
    rec.add_argument("--run-id", required=True)
    rec.add_argument("--scr", required=True, help="the check launch's scratch (its store holds the evidence archive)")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.cmd == "child":
        if a.phase == "init":
            return phase_init(a)
        from stages.teacher_modes import initialization as IN
        try:
            phase_train(a)
            return 0
        except Exception as exc:  # noqa: BLE001 -- recorded for the verdict; the exit status for the parent
            oom = IN.is_out_of_memory(exc)
            _write(a.out, f"train_{a.mode}_error", {"out_of_memory": oom, "error": str(exc)[:2000],
                                                    "traceback": traceback.format_exc()[-3000:]})
            return IN.OOM_EXIT if oom else 1
    if a.cmd == "record":
        rec = record(a.run_id, Path(a.scr))
        from stages.teacher_modes import phase as PH
        print(json.dumps({**{k: rec["verdict"][k] for k in ("complete", "fit_with_margin", "below_margin", "failed",
                                                            "a100_evidence")},
                          "admits": {k: gate(ops) or "yes" for k, ops in load_config()["gates"].items()},
                          "_": f"{PH.CHECK_RECORD} written; commit it with {rec['evidence']}"}, indent=1))
        return 0
    a.identity = ""
    report = run(a)
    return 0 if report.get("complete") else 3


if __name__ == "__main__":
    raise SystemExit(main())
