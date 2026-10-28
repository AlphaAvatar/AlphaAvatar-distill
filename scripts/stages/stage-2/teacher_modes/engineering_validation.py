#!/usr/bin/env python3
"""A-E, Research A's engineering validation on the hybrid teacher: every check, one record. Never formal.

    # on the pod the grant names (A100-SXM4-80GB), as the launcher runs it
    cd /workspace/aad && HF_HUB_OFFLINE=1 PYTHONPATH=src:scripts:scripts/stages/stage-1 /opt/train/bin/python \\
        scripts/stages/stage-2/teacher_modes/engineering_validation.py --run-id <run> --out <evidence dir> \\
        --work <scratch dir> --status <status file> --device cuda:0 --vllm-python /opt/vllm/bin/python \\
        --student artifacts/stages/stage-2/reasoning_data/h1pa/pca_original/20260726
    # the same driver end to end on a host, on random small models at the real vocabulary ($0)
    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-2/teacher_modes/engineering_validation.py \\
        --run-id <run> --out <dir> --work <dir> --status <file> --tiny

It answers what a host cannot -- the real hybrid teacher's numerics in both
official modes, the engine, real memory and real step time -- and produces no
scientific result: no document it builds is training data, no correctness is
reported, nothing it measures changes the design. Every criterion is fixed in
advance in ``configs/stages/stage-2/teacher_modes/engineering_validation.json``
(bound into the study identity by the issuer). The checks:

1. ``prompt_identity`` -- four fixed T0 tasks and four eval-set-1 items rendered
   with the official hard switch only (``apply_chat_template(...,
   add_generation_prompt=True, enable_thinking=...)``, the system-message rule):
   the chat template's sha256 is teacher_identity.json's, every prompt ends in
   its mode's generation_suffix_ids and leaves its mode's think state, the
   token ids are hashed; on the pod the teacher's small files and weight shards
   are checked against the identity.
2. ``generation`` -- the teacher under vLLM in both modes, token ids in, each
   mode's official preset, the same recorded seed per (task, turn) in both
   modes, the allowance rule of each kind (T0: A1's data-construction limit;
   eval items: the full remaining context, P18). PASS: the engine echoed every
   prompt's ids, every turn ended on a stop id or is recorded at its limit, no
   H-N output holds a think delimiter, every turn's think state is its mode's,
   and each mode yields at least one document (``document_from_turn`` +
   ``prompt_mode_problems``).
3. ``teacher_forward`` -- those documents packed ``document_isolated``
   (``pack_isolated``, restarting position ids, no attention mask,
   ``use_cache=False``), the teacher on each packed sequence against each
   document alone at every target position, fp32 (TF32 off) and bf16, both
   modes; and the engine's teacher-forced log-probabilities of the same ids
   against the HF forward. PASS: bf16 packed-vs-alone <= ``bf16_k`` x the
   documents' own bf16-vs-fp32 difference; fp32 packed-vs-alone <=
   ``fp32_fraction_of_bf16_reference`` x it; the engine's mean |difference| <=
   ``engine_mean_k`` x the HF bf16 one.
4. ``assistant_target_kl`` -- the trainer's ``kd_scope`` 'assistant' prediction
   mask equals the documents' generated-token layout exactly (no system, user,
   header, prefill, padding or cross-document position), position ids restart
   at every document and at the pad run, KL(teacher || teacher) on the mask is
   0, the stand-in student's KD on it is finite and counts the target tokens.
5. ``trainer_smoke`` -- ``steps`` optimizer steps of H1-PA's recipe with the
   declared changes (kd_scope assistant, batch.context_strategy
   document_isolated, the hybrid teacher on the fly) on the stand-in student,
   at 8,192: seconds per step after warmup and peak memory recorded; PASS:
   finite losses, every step document_isolated with KD positions equal to the
   CE targets, the derived config differing from E1's exactly in the declared
   keys, the student's bytes the declared ones.
6. ``preclosed_scoring`` -- the eval records through eval-set-1's frozen
   scorers under their declared think state (config ``preclosed_scoring``), and
   evaluation_identity.json regenerated to the committed document.

The process boundary: this parent never initializes CUDA. The PyTorch work runs
in child processes of the same interpreter (``--phase teacher``, ``--phase
train``) that exit before the next GPU user starts; the engine runs through
``shared.validation.vllm_check.run_isolated``, which refuses unless the GPU
shows no compute process. The engine runs first, so the GPU is read again,
recorded, and must be clear before each torch phase. Every check runs after an
earlier one fails. ``verdict(out, config)`` derives every status from the
evidence files and the config alone; the parent writes its result as
``report.json`` and the dev box re-runs it on the secured copy.
"""
from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from shared.validation import vllm_check as VC  # noqa: E402

CONFIG = "configs/stages/stage-2/teacher_modes/engineering_validation.json"
SCHEMA = "aadistill.stage2.teacher_modes.engineering_validation/v1"
MODES = ("H-T", "H-N")
EV = {k: f"{k}.json" for k in ("run", "environment", "prompts", "generation", "documents", "engine_logprobs",
                               "teacher_forward", "hf_logprobs", "masks", "trainer_smoke", "scoring", "phases")}
REPORT = "report.json"
#: checks that cannot run without the engine, NOT_RUN on a host only
ENGINE_CHECKS = ("generation",)
#: adapter output that is correctness, never stored or reported
CORRECTNESS_KEYS = ("correct", "extracted", "answer_correct", "success", "extraction", "gold",
                    "answer_matches_ignoring_protocol")


# ---- small helpers -------------------------------------------------------------------------------------

def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_config(path: str | Path = CONFIG) -> dict:
    p = Path(path)
    return json.loads((p if p.is_absolute() else REPO / p).read_text())


def _write(out: Path, key: str, doc) -> None:
    (Path(out) / EV[key]).write_text(json.dumps(doc, indent=1, default=str) + "\n")


def _read(out: Path, key: str):
    p = Path(out) / EV[key]
    return json.loads(p.read_text()) if p.is_file() else None


def _json(path: str) -> dict:
    return json.loads((REPO / path).read_text())


def identity(cfg: dict) -> dict:
    return _json(cfg["teacher_identity"])


def mode_spec(cfg: dict, mode: str) -> dict:
    return identity(cfg)["modes"][cfg["modes"][mode]["identity_mode"]]


def sizes(cfg: dict, tiny: bool) -> dict:
    t = cfg["tiny"]
    return {"sequence_length": t["sequence_length"] if tiny else cfg["teacher_forward"]["sequence_length"],
            "max_model_len": t["max_model_len"] if tiny else cfg["generation"]["engine"]["max_model_len"],
            "t0_session_limit": t["t0_session_limit"] if tiny else cfg["generation"]["t0_session_limit"],
            "steps": t["steps"] if tiny else cfg["trainer_smoke"]["steps"],
            "warmup": t["warmup_steps_excluded"] if tiny else cfg["trainer_smoke"]["warmup_steps_excluded"]}


def request_seed(base: int, task_id: str, turn: int) -> int:
    from stages.calibration_volume.run_battery import request_seed as rs
    return rs(base, task_id, turn)


# ---- tokenizer -----------------------------------------------------------------------------------------

class _Enc(dict):
    @property
    def input_ids(self):
        return self["input_ids"]


class StandInTokenizer:
    """The hybrid template's structure over a character vocabulary, at the real
    special-token ids: a HOST stand-in when the teacher tokenizer is not cached.
    Never used on a pod; its chat template is not the teacher's and the record says so."""

    kind = "stand_in"
    chat_template = "stand-in: <|im_start|>{role}\\n{content}<|im_end|>\\n ... hybrid generation prompt"
    VOCAB = {"<|im_start|>": 151644, "<|im_end|>": 151645, "<|endoftext|>": 151643, "<think>": 151667,
             "</think>": 151668, "assistant": 77091, "\n\n": 271, "\n": 198}

    def __call__(self, text: str, add_special_tokens: bool = False) -> _Enc:
        keys = sorted(self.VOCAB, key=len, reverse=True)
        ids, i = [], 0
        while i < len(text):
            k = next((k for k in keys if text.startswith(k, i)), None)
            if k is None:
                ids.append(1000 + ord(text[i]) % 50000)
                i += 1
            else:
                ids.append(self.VOCAB[k])
                i += len(k)
        return _Enc(input_ids=ids)

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False, enable_thinking=True, **_):
        s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
        if add_generation_prompt:
            s += "<|im_start|>assistant\n" + ("" if enable_thinking else "<think>\n\n</think>\n\n")
        return s

    def decode(self, ids, skip_special_tokens=False) -> str:
        inv = {v: k for k, v in self.VOCAB.items()}
        return "".join(inv.get(int(i), chr((int(i) - 1000) % 50000)) for i in ids)


def tokenizer(cfg: dict, tiny: bool):
    """The teacher's tokenizer from the HF cache (the pod runs offline); on a host
    without the cached files, the stand-in."""
    from transformers import AutoTokenizer
    t = cfg["teacher"]
    try:
        tok = AutoTokenizer.from_pretrained(t["model"], revision=t["revision"], local_files_only=tiny)
        tok.kind = "teacher"
        return tok
    except OSError:
        if not tiny:
            raise
        return StandInTokenizer()


# ---- 1. prompts ----------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _unit(component: str, unit: str) -> tuple[dict, str, str]:
    """(items by id, formal protocol, eval set id) of one registered unit, read once and verified."""
    from shared import eval_sets as ES
    from shared.evaluation import eval_set_scoring as S
    eid = ES.default_formal()
    comp = ES.manifest(eid)["components"][component]
    field = comp["units"][unit].get("id_field", "id")
    return {r[field]: r for r in ES.items(eid, component, unit)}, S.formal_protocol(comp["scorer"]), eid


def eval_item(cfg: dict, spec: dict) -> dict:
    items, protocol, eid = _unit(spec["component"], spec["unit"])
    return {"item": items[spec["id"]], "protocol": protocol, "eval_set_id": eid}


def t0_tasks(cfg: dict) -> list[dict]:
    t0 = cfg["t0"]
    path = REPO / t0["list"]
    design = _json(cfg["design"])
    declared = {x["path"]: x["sha256"] for x in design["tasks"]["T0"]["lists"]}
    if declared.get(t0["list"]) != t0["list_sha256"] or sha_file(path) != t0["list_sha256"]:
        raise RuntimeError(f"{t0['list']} is not design.json's T0 list")
    rows = {r["task_id"]: r for r in (json.loads(l) for l in path.read_text().split("\n") if l.strip())}
    return [rows[t] for t in t0["task_ids"]]


def teacher_files(cfg: dict) -> dict:
    """The cached teacher's small files and weight shards against teacher_identity.json."""
    from huggingface_hub import hf_hub_download
    ident, t = identity(cfg), cfg["teacher"]
    get = lambda name: Path(hf_hub_download(t["model"], name, revision=t["revision"]))  # noqa: E731
    files = {}
    for name, want in ident["files_sha256"].items():
        try:
            files[name] = {"observed": sha_file(get(name)), "expected": want}
        except Exception as exc:  # noqa: BLE001 -- README/LICENSE are not staged; recorded, not required
            files[name] = {"observed": None, "expected": want, "error": f"{type(exc).__name__}"}
    shards = {s["file"]: {"observed": sha_file(get(s["file"])), "expected": s["lfs_sha256"]}
              for s in ident["weight_shards"]}
    return {"files": files, "shards": shards}


def render_prompts(cfg: dict, tok, *, tiny: bool) -> dict:
    from aadistill.data.sessions import split_system
    from aadistill.data.token_documents import generation_suffix
    from aadistill.evaluation.behavior import template_think_state

    sources = [("t0", t["task_id"], t["messages"], {"axis": t["axis"], "family_id": t["family_id"]})
               for t in t0_tasks(cfg)]
    for spec in cfg["eval_items"]:
        it = eval_item(cfg, spec)["item"]
        msgs = it["messages"] if isinstance(it["messages"], list) else json.loads(it["messages"])
        sources.append(("eval", spec["id"], msgs, {"component": spec["component"], "unit": spec["unit"]}))
    rows, suffixes = [], {}
    for mode in MODES:
        kw = mode_spec(cfg, mode)["template_kwargs"]
        suffixes[mode] = {"derived": list(generation_suffix(tok, kw)), "expected": mode_spec(cfg, mode)[
            "generation_suffix_ids"], "think_state": cfg["modes"][mode]["think_state"], "template_kwargs": kw}
        for kind, sid, msgs, meta in sources:
            system, body = split_system(msgs)
            text = tok.apply_chat_template([{"role": "system", "content": system}, *body], tokenize=False,
                                           add_generation_prompt=True, **kw)
            ids = list(tok(text, add_special_tokens=False)["input_ids"])
            rows.append({"key": f"{mode}:{kind}:{sid}", "mode": mode, "kind": kind, "task_id": sid,
                         "system": system, "prompt_text": text, "prompt_token_ids": ids,
                         "prompt_sha256": sha_json(ids), "think_state": template_think_state(text), **meta})
    return {"tokenizer": {"kind": getattr(tok, "kind", "teacher"),
                          "chat_template_sha256": hashlib.sha256(tok.chat_template.encode()).hexdigest(),
                          "expected_chat_template_sha256": identity(cfg)["chat_template_sha256"]},
            "suffix": suffixes, "teacher_files": None if tiny else teacher_files(cfg), "rows": rows}


# ---- 2. generation -------------------------------------------------------------------------------------

def expected_sampling(cfg: dict, row: dict, size: dict) -> dict:
    g = cfg["generation"]
    preset = _json(cfg["design"])["generation"]["sampling"][row["mode"]]
    allowance = size["t0_session_limit"] - len(row["prompt_token_ids"]) if row["kind"] == "t0" else None
    return {**preset, "max_tokens": allowance, "seed": request_seed(g["seed_base"], row["task_id"], 0),
            "stop_token_ids": list(g["stop_ids"]), "detokenize": False}


def teacher_source(cfg: dict, args) -> str:
    """The teacher's local directory: the cached snapshot at the pinned revision (the pod runs offline,
    so the revision is fixed by the path itself), or the host run's tiny teacher."""
    if args.tiny:
        return str(Path(args.work) / "tiny_teacher")
    from huggingface_hub import snapshot_download
    t = cfg["teacher"]
    return snapshot_download(t["model"], revision=t["revision"], local_files_only=True)


def engine_request(cfg: dict, args, size: dict, requests: list[dict]) -> dict:
    e, t = cfg["generation"]["engine"], cfg["teacher"]
    path = teacher_source(cfg, args)
    return {"model": path, "tokenizer": path, "teacher": {"model": t["model"], "revision": t["revision"],
                                                          "path": path, "tiny": bool(args.tiny)},
            "dtype": e["dtype"], "max_model_len": size["max_model_len"], "seed": e["seed"],
            "gpu_memory_utilization": e["gpu_memory_utilization"], "enforce_eager": e["enforce_eager"],
            "requests": requests}


def run_engine(cfg: dict, args, req: dict, name: str) -> dict:
    if args.tiny and not args.engine_script:
        return {"status": "NOT_RUN", "why": "a host run has no vLLM engine", "engine_started": False}
    return VC.run_isolated(args.vllm_python, req, Path(args.work) / name,
                           timeout_s=cfg["timeouts_s"]["engine"],
                           script=Path(args.engine_script) if args.engine_script else None)


def generate(cfg: dict, args, prompts: dict, size: dict) -> dict:
    reqs = [{"id": f"gen:{r['key']}", "prompt_token_ids": r["prompt_token_ids"],
             "sampling": expected_sampling(cfg, r, size)} for r in prompts["rows"]]
    req = engine_request(cfg, args, size, reqs)
    return {"request": req, "engine": run_engine(cfg, args, req, "engine_generate"),
            "engine_stand_in": bool(args.engine_script)}


def synthetic_output(cfg: dict, mode: str) -> list[int]:
    s, (open_, close) = cfg["generation"]["synthetic_turn_ids"], cfg["generation"]["think_ids"]
    end = cfg["generation"]["stop_ids"][0]
    if mode == "H-T":
        return [open_, 198, *s["reasoning"], 198, close, 271, *s["answer"], end]
    return [*s["answer"], end]


def turns(cfg: dict, prompts: dict, gen: dict) -> list[dict]:
    """One turn per prompt: the engine's, or -- when the engine produced none -- a synthetic one."""
    res = {r["id"]: r for r in (((gen or {}).get("engine") or {}).get("result") or {}).get("requests") or []}
    out = []
    for row in prompts["rows"]:
        r = res.get(f"gen:{row['key']}")
        o = (r or {}).get("outputs") or [{}]
        ids = o[0].get("token_ids") if r else synthetic_output(cfg, row["mode"])
        out.append({"key": row["key"], "mode": row["mode"], "kind": row["kind"], "task_id": row["task_id"],
                    "prompt_token_ids": row["prompt_token_ids"], "output_token_ids": list(ids or []),
                    "finish_reason": o[0].get("finish_reason") if r else "stop",
                    "stop_reason": o[0].get("stop_reason") if r else None, "think_state": row["think_state"],
                    "synthetic": r is None})
    return out


# ---- 3. documents --------------------------------------------------------------------------------------

def documents(cfg: dict, prompts: dict, turn_rows: list[dict]) -> tuple[dict, dict]:
    """(T0 documents per mode, refusals). Eval turns never become documents."""
    from aadistill.data.token_documents import DocumentError, document_from_turn, prompt_mode_problems
    docs, refused = {m: [] for m in MODES}, []
    for t in turn_rows:
        if t["kind"] != "t0":
            continue
        try:
            d = document_from_turn(t, doc_id=t["key"], stop_ids=cfg["generation"]["stop_ids"],
                                   meta={"mode": t["mode"], "synthetic": t["synthetic"]})
        except DocumentError as exc:
            refused.append({"doc_id": t["key"], "reason": str(exc)})
            continue
        bad = prompt_mode_problems(d, prompts["suffix"][t["mode"]]["expected"])
        if bad:
            refused.append({"doc_id": t["key"], "reason": "; ".join(bad)})
            continue
        docs[t["mode"]].append(d)
    return docs, refused


def documents_doc(docs: dict, refused: list) -> dict:
    return {"documents": [{"doc_id": d.doc_id, "mode": m, "prompt_len": len(d.prompt_ids),
                           "output_len": len(d.output_ids), "content_sha256": d.content_sha256(),
                           "synthetic": bool(d.meta.get("synthetic"))} for m in MODES for d in docs[m]],
            "refused": refused}


def pack_sha(packed) -> str:
    return sha_json([packed.input_ids.tolist(), packed.target_mask.tolist(), packed.position_ids.tolist()])


def masks_doc(cfg: dict, docs: dict, size: dict) -> dict:
    """Each mode's isolated pack and the trainer's KD positions on it."""
    from aadistill.data.token_documents import pack_isolated
    from aadistill.training.train import prediction_mask
    tf, out = cfg["teacher_forward"], {}
    for mode in MODES:
        if not docs[mode]:
            out[mode] = {"documents": 0}
            continue
        p = pack_isolated(docs[mode], size["sequence_length"], tf["pad_id"], order_seed=tf["order_seed"])
        pm = prediction_mask(p.target_mask, cfg["assistant_target_kl"]["kd_scope"], p.content_mask)
        by = {d.doc_id: d for d in docs[mode]}
        seqs = []
        for b, ms in enumerate(p.members):
            seqs.append({"members": [{"doc_id": m["doc_id"], "start": m["start"], "length": m["length"],
                                      "prompt_len": len(by[m["doc_id"]].prompt_ids)} for m in ms],
                         "content_len": int(p.content_mask[b].sum()), "position_ids": p.position_ids[b].tolist(),
                         "predicted": [int(i) for i in pm[b].nonzero().flatten()],
                         "targets": int(p.target_mask[b].sum())})
        out[mode] = {"documents": len(docs[mode]), "sequence_length": size["sequence_length"],
                     "pack_sha256": pack_sha(p), "refused": p.refused, "sequences": seqs}
    return {"kd_scope": cfg["assistant_target_kl"]["kd_scope"], "modes": out}


def docs_payload(docs: dict) -> dict:
    return {m: [{"doc_id": d.doc_id, "prompt_ids": list(d.prompt_ids), "output_ids": list(d.output_ids),
                 "meta": dict(d.meta)} for d in docs[m]] for m in MODES}


def load_docs(work: Path) -> dict:
    from aadistill.data.token_documents import TokenDocument
    raw = json.loads((Path(work) / "documents_full.json").read_text())
    return {m: [TokenDocument(d["doc_id"], tuple(d["prompt_ids"]), tuple(d["output_ids"]), d["meta"])
                for d in raw[m]] for m in MODES}


def engine_scoring(cfg: dict, args, docs: dict, size: dict) -> dict:
    """The engine's log-probability of every target token, teacher-forced over each document's own ids."""
    reqs = [{"id": f"score:{d.doc_id}", "prompt_token_ids": list(d.ids),
             "sampling": {"max_tokens": 1, "temperature": 0.0, "prompt_logprobs": 1, "detokenize": False,
                          "seed": 0}} for m in MODES for d in docs[m]]
    req = engine_request(cfg, args, size, reqs)
    res = run_engine(cfg, args, req, "engine_score")
    lps, by = {}, {f"score:{d.doc_id}": d for m in MODES for d in docs[m]}
    for r in ((res.get("result") or {}).get("requests") or []):
        d = by[r["id"]]
        rows = r.get("prompt_logprobs") or []
        got = []
        for i in range(len(d.prompt_ids), len(d)):
            entry = rows[i] if i < len(rows) and rows[i] else []
            got.append(next((lp for tid, lp, _ in entry if tid == d.ids[i]), None))
        lps[d.doc_id] = got
    slim = {k: v for k, v in res.items() if k != "result"}
    slim["versions"] = (res.get("result") or {}).get("versions")
    return {"engine": slim, "engine_stand_in": bool(args.engine_script), "target_logprobs": lps}


# ---- 4-5. torch phases (children) ----------------------------------------------------------------------

def _device(args) -> str:
    return "cpu" if args.tiny else args.device


def _load_teacher(args, dtype):
    import torch
    from transformers import AutoModelForCausalLM
    cfg = load_config(args.config)
    tf = cfg["teacher_forward"]
    m = AutoModelForCausalLM.from_pretrained(teacher_source(cfg, args), dtype=dtype,
                                             attn_implementation=tf["attn_implementation"])
    return m.to(_device(args)).eval()


def _student_dir(args) -> Path:
    return Path(args.work) / "tiny_student" if args.tiny else Path(args.student)


def phase_teacher(args) -> dict:
    """Teacher forward (packed vs alone, fp32/bf16), HF target log-probs, KL(teacher||teacher), student KD."""
    import torch
    from transformers import AutoModelForCausalLM

    from aadistill.data.token_documents import pack_isolated
    from aadistill.training.train import kd_forward_kl, prediction_mask

    cfg = load_config(args.config)
    tf, size = cfg["teacher_forward"], sizes(cfg, args.tiny)
    torch.backends.cuda.matmul.allow_tf32 = tf["fp32"]["allow_tf32"]
    torch.backends.cudnn.allow_tf32 = tf["fp32"]["allow_tf32"]
    torch.set_float32_matmul_precision(tf["fp32"]["float32_matmul_precision"])
    dev = _device(args)
    docs = load_docs(args.work)
    t32 = _load_teacher(args, torch.float32)
    t16 = copy.deepcopy(t32).to(torch.bfloat16).eval()
    student = AutoModelForCausalLM.from_pretrained(_student_dir(args), dtype=torch.float32).to(dev).eval()
    rec = {"device": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu",
           "torch": torch.__version__, "attn_implementation": getattr(t32.config, "_attn_implementation", None),
           "tf32": {"matmul": torch.backends.cuda.matmul.allow_tf32, "cudnn": torch.backends.cudnn.allow_tf32},
           "float32_matmul_precision": torch.get_float32_matmul_precision(), "modes": {}}
    lps = {}
    ac = (torch.autocast(device_type="cuda" if dev.startswith("cuda") else "cpu", dtype=torch.bfloat16))
    with torch.no_grad():
        for mode in MODES:
            if not docs[mode]:
                rec["modes"][mode] = {"documents": 0}
                continue
            p = pack_isolated(docs[mode], size["sequence_length"], tf["pad_id"], order_seed=tf["order_seed"])
            by = {d.doc_id: d for d in docs[mode]}
            seqs = []
            for b, ms in enumerate(p.members):
                ids, pos = p.input_ids[b:b + 1].to(dev), p.position_ids[b:b + 1].to(dev)
                P32 = t32(ids, position_ids=pos, use_cache=False).logits[0]
                P16 = t16(ids, position_ids=pos, use_cache=False).logits[0]
                pm = prediction_mask(p.target_mask[b:b + 1], cfg["assistant_target_kl"]["kd_scope"],
                                     p.content_mask[b:b + 1]).to(dev)
                kl_self, kl_n = kd_forward_kl(P16[None], P16[None], pm)
                with ac:
                    S = student(ids, position_ids=pos, use_cache=False).logits
                kd, kd_n = kd_forward_kl(S, P16[None], pm)
                del S
                members = []
                for m in ms:
                    d = by[m["doc_id"]]
                    s, k = m["start"], m["length"]
                    alone = torch.tensor([d.ids], device=dev)
                    A32 = t32(alone, use_cache=False).logits[0]
                    A16 = t16(alone, use_cache=False).logits[0]
                    sel = torch.tensor(d.target_mask[1:], device=dev)
                    nxt = torch.tensor(d.ids[1:], device=dev)[sel]
                    p32, p16, a32, a16 = (P32[s:s + k - 1][sel], P16[s:s + k - 1][sel].float(),
                                          A32[:k - 1][sel], A16[:k - 1][sel].float())
                    members.append({"doc_id": d.doc_id, "targets": int(sel.sum()),
                                    "d_fp32": float((p32 - a32).abs().max()), "d_bf16": float((p16 - a16).abs().max()),
                                    "ref_bf16_alone": float((a16 - a32).abs().max()),
                                    "ref_bf16_packed": float((p16 - p32).abs().max())})
                    ar = torch.arange(len(nxt), device=dev)
                    lps[d.doc_id] = {"fp32": torch.log_softmax(a32, -1)[ar, nxt].tolist(),
                                     "bf16": torch.log_softmax(a16, -1)[ar, nxt].tolist()}
                    del A32, A16, p32, p16, a32, a16
                seqs.append({"index": b, "members": members, "kl_self": float(kl_self), "kl_self_positions": kl_n,
                             "student_kd": float(kd), "student_kd_positions": kd_n})
                del P32, P16
                if dev.startswith("cuda"):
                    torch.cuda.empty_cache()
            rec["modes"][mode] = {"documents": len(docs[mode]), "pack_sha256": pack_sha(p), "sequences": seqs}
    _write(args.out, "hf_logprobs", lps)
    _write(args.out, "teacher_forward", rec)
    return rec


def smoke_config(cfg: dict, args) -> tuple[dict, list[str]]:
    """H1-PA's E1 config for seed 20260726 with the declared changes; returns it and the changed keys."""
    from stages.reasoning_data import unit_driver as UD
    ts, size, t = cfg["trainer_smoke"], sizes(cfg, args.tiny), cfg["teacher"]
    e1 = _json(ts["e1_config"])
    d = copy.deepcopy(e1)
    d.update(run_name=f"ae_trainer_smoke_{args.run_id}", _purpose="A-E trainer smoke (engineering, never formal)",
             student_path=str(_student_dir(args)), out_dir=str(Path(args.work) / "smoke"), device=_device(args))
    d["teacher"] = {**e1["teacher"], "model_id": t["model"], "revision": t["revision"]}
    d["loss"]["kd_scope"] = cfg["assistant_target_kl"]["kd_scope"]
    d["batch"]["context_strategy"] = "document_isolated"
    d["schedule"].update(total_steps=size["steps"], warmup_steps=1)
    d["checkpoint"]["save_every"] = 0
    d["intervals"].update(log_every=1, eval_every=0, eval_blocks=0)
    if args.tiny:
        d["block_len"] = size["sequence_length"]
    a, b = UD._flat(e1), UD._flat(d)
    return d, sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def phase_train(args) -> dict:
    import torch
    from transformers import AutoModelForCausalLM

    from aadistill.data.token_documents import pack_isolated
    from aadistill.training.train import Trainer

    cfg = load_config(args.config)
    tf, size = cfg["teacher_forward"], sizes(cfg, args.tiny)
    dev = _device(args)
    docs = load_docs(args.work)
    every = [d for m in MODES for d in docs[m]]
    p = pack_isolated(every, size["sequence_length"], tf["pad_id"], order_seed=tf["order_seed"])
    tcfg, changed = smoke_config(cfg, args)
    teacher = _load_teacher(args, torch.bfloat16)
    student = AutoModelForCausalLM.from_pretrained(_student_dir(args), dtype=torch.float32)
    tr = Trainer(tcfg, student, p.blocks(), teacher=teacher, device=dev, out_dir=Path(args.work) / "smoke")
    if dev.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(dev)
    steps, secs = [], []
    for _ in range(size["steps"]):
        t0 = time.time()
        m = tr.step_once()
        if dev.startswith("cuda"):
            torch.cuda.synchronize(dev)
        secs.append(round(time.time() - t0, 4))
        steps.append({k: m.get(k) for k in ("step", "loss", "ce", "kd", "grad_norm", "ce_targets", "kd_positions",
                                            "executed_nonpad_tokens", "context_strategy")})
    warm = secs[size["warmup"]:] or secs
    rec = {"device": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu",
           "torch": torch.__version__, "sequence_length": size["sequence_length"],
           "pack": {**p.stats(), "pack_sha256": pack_sha(p)},
           "documents_synthetic": sorted({bool(d.meta.get("synthetic")) for d in every}),
           "config_sha256": sha_json(tcfg), "config_changed_keys": changed,
           "requested_device": dev, "trainer_device": str(tr.device),
           "student_device": str(next(tr.student.parameters()).device),
           "steps": steps, "seconds_per_step": secs,
           "seconds_per_step_mean_after_warmup": round(sum(warm) / len(warm), 4),
           "peak_memory_gib": (round(torch.cuda.max_memory_allocated(dev) / 2**30, 2)
                               if dev.startswith("cuda") else None)}
    _write(args.out, "trainer_smoke", rec)
    return rec


# ---- 6. scoring ----------------------------------------------------------------------------------------

def structural_ok(cfg: dict, mode: str, out_ids: list[int]) -> bool:
    open_, close = cfg["generation"]["think_ids"]
    if not out_ids or out_ids[-1] not in cfg["generation"]["stop_ids"]:
        return False
    if mode == "H-N":
        return open_ not in out_ids and close not in out_ids
    return out_ids[0] == open_ and out_ids.count(open_) == 1 and out_ids.count(close) == 1


def eval_records(cfg: dict, tok, turn_rows: list[dict]) -> list[dict]:
    """Each eval turn as run_battery records it: raw decoded with special tokens, the prompt's think state."""
    rows = []
    for t in turn_rows:
        if t["kind"] != "eval":
            continue
        spec = next(s for s in cfg["eval_items"] if s["id"] == t["task_id"])
        ids = t["output_token_ids"]
        rows.append({"key": t["key"], "mode": t["mode"], "component": spec["component"], "unit": spec["unit"],
                     "id": spec["id"], "raw": tok.decode(ids, skip_special_tokens=False),
                     "think_state": t["think_state"], "natural_termination": bool(ids) and ids[-1] in
                     cfg["generation"]["stop_ids"], "generated_tokens": len(ids),
                     "structural_ok": structural_ok(cfg, t["mode"], ids), "synthetic": t["synthetic"]})
    return rows


def score_record(cfg: dict, r: dict) -> dict:
    """One record through the frozen scorer path; correctness is dropped."""
    from aadistill.evaluation.behavior import record_think_state, split_generation
    from aadistill.evaluation.strict_answer import protocol_valid
    from shared.evaluation import eval_set_scoring as S

    stop = "eos" if r["natural_termination"] else "length"
    if r["component"] == "fluid_reasoning_720":
        turn = {"raw": r["raw"], "answer": split_generation(r["raw"], think_state=r["think_state"])["answer"],
                "natural_termination": r["natural_termination"], "degeneration_triggered": False,
                "context_limit_reached": not r["natural_termination"], "think_state": r["think_state"],
                "generated_tokens": r["generated_tokens"], "stop_reason": stop}
        rec, state = {"turns": [turn]}, record_think_state(turn)
    else:
        rec = {"raw": r["raw"], "think_state": r["think_state"], "stop_reason": stop,
               "natural_termination": r["natural_termination"], "degeneration_triggered": False}
        state = record_think_state(rec)
    it = eval_item(cfg, {"component": r["component"], "unit": r["unit"], "id": r["id"]})
    out = S.ADAPTERS[r["component"]](r["unit"], it["item"], rec, it["protocol"])
    reason = str(out.get("reason") or "")
    adapter_protocol_failure = reason == "unusable" if r["component"] == "fluid_reasoning_720" \
        else reason.startswith("protocol:")
    ok, why = protocol_valid(r["raw"], think_state=state)
    row = {"key": r["key"], "mode": r["mode"], "read_think_state": state, "protocol_valid": [ok, why],
           "adapter": ({"class": "protocol_failure", "reason": reason} if adapter_protocol_failure
                       else {"class": "scored"}), "structural_ok": r["structural_ok"]}
    if r["mode"] == "H-N":
        row["historical_rule_without_think_state"] = list(protocol_valid(r["raw"]))
        row["delimiter_inserted"] = list(protocol_valid("</think>" + r["raw"], think_state="preclosed"))
    assert not any(k in row for k in CORRECTNESS_KEYS)
    return row


def scoring_doc(cfg: dict, records: list[dict]) -> dict:
    return {"records": records, "results": [score_record(cfg, r) for r in records]}


# ---- verdict -------------------------------------------------------------------------------------------

def _st(problems: list[str], **measured) -> dict:
    return {"status": "FAIL" if problems else "PASS", "problems": problems, **measured}


def check_prompt_identity(out: Path, cfg: dict, run: dict) -> dict:
    pr = _read(out, "prompts")
    if not pr:
        return _st(["no prompts evidence"])
    bad, tk = [], pr["tokenizer"]
    stand_in = tk["kind"] == "stand_in"
    if stand_in and not run["tiny"]:
        bad.append("a stand-in tokenizer on a pod run")
    if not stand_in and tk["chat_template_sha256"] != identity(cfg)["chat_template_sha256"]:
        bad.append(f"chat template {tk['chat_template_sha256'][:12]} is not the identity's")
    for mode, s in pr["suffix"].items():
        if s["expected"] != mode_spec(cfg, mode)["generation_suffix_ids"]:
            bad.append(f"{mode}: the recorded expected suffix is not teacher_identity.json's")
        if s["derived"] != s["expected"]:
            bad.append(f"{mode}: the template's generation prompt {s['derived']} is not {s['expected']}")
    keys = {}
    for r in pr["rows"]:
        suf = mode_spec(cfg, r["mode"])["generation_suffix_ids"]
        if r["prompt_token_ids"][-len(suf):] != suf:
            bad.append(f"{r['key']}: the prompt does not end in its mode's generation prompt")
        if r["think_state"] != cfg["modes"][r["mode"]]["think_state"]:
            bad.append(f"{r['key']}: the prompt leaves {r['think_state']}")
        if sha_json(r["prompt_token_ids"]) != r["prompt_sha256"]:
            bad.append(f"{r['key']}: the recorded token-id hash does not match")
        keys.setdefault((r["kind"], r["task_id"]), set()).add(r["mode"])
    if any(v != set(MODES) for v in keys.values()) or len(keys) != len(cfg["t0"]["task_ids"]) + len(cfg["eval_items"]):
        bad.append("not every task and item is rendered in both modes")
    tfiles = pr.get("teacher_files")
    if not run["tiny"]:
        if not tfiles:
            bad.append("the teacher's files were not checked")
        else:
            for name, f in {**tfiles["files"], **tfiles["shards"]}.items():
                if f["observed"] is not None and f["observed"] != f["expected"]:
                    bad.append(f"teacher file {name} is not the identity's")
            missing = [n for n, f in tfiles["files"].items() if f["observed"] is None and n not in
                       ("README.md", "LICENSE")] + [n for n, f in tfiles["shards"].items() if f["observed"] is None]
            if missing:
                bad.append(f"teacher files not found: {missing}")
    return _st(bad, prompts=len(pr["rows"]), tokenizer=tk["kind"],
               prompt_tokens={m: [len(r["prompt_token_ids"]) for r in pr["rows"] if r["mode"] == m] for m in MODES})


def check_generation(out: Path, cfg: dict, run: dict) -> dict:
    from aadistill.data.token_documents import DocumentError, document_from_turn, prompt_mode_problems
    gen, pr, dd = _read(out, "generation"), _read(out, "prompts"), _read(out, "documents")
    if not gen or not pr:
        return _st(["no generation evidence"])
    eng = gen["engine"]
    if gen.get("engine_stand_in") and not run["tiny"]:
        return _st(["a stand-in engine on a pod run"])
    if eng.get("status") == "NOT_RUN":
        return {"status": "NOT_RUN", "why": eng.get("why"), "problems": []}
    res = ((eng.get("result") or {}).get("requests")) if eng.get("status") == "PASS" else None
    if not res:
        return _st([f"the engine did not complete: {eng.get('problems')}"])
    size = sizes(cfg, run["tiny"])
    g, by = cfg["generation"], {r["id"]: r for r in res}
    rows = {r["key"]: r for r in pr["rows"]}
    bad, per = [], {m: {"turns": 0, "ended_on_stop": 0, "at_limit": 0, "output_tokens": [], "ht_opened_think": 0}
                    for m in MODES}
    seeds = {}
    for q in gen["request"]["requests"]:
        key = q["id"].removeprefix("gen:")
        row, r = rows[key], by.get(q["id"])
        if r is None:
            bad.append(f"{key}: no engine result")
            continue
        if not (r["prompt_token_ids"] == q["prompt_token_ids"] == row["prompt_token_ids"]):
            bad.append(f"{key}: the engine's prompt ids are not the rendered ones")
        if q["sampling"] != expected_sampling(cfg, row, size):
            bad.append(f"{key}: the request's sampling is not the mode's preset and allowance rule")
        seeds.setdefault(row["task_id"], set()).add(q["sampling"]["seed"])
        limit = size["max_model_len"] - len(q["prompt_token_ids"]) if row["kind"] == "eval" \
            else q["sampling"]["max_tokens"]
        if r.get("max_tokens") != limit:
            bad.append(f"{key}: the engine's max_tokens {r.get('max_tokens')} is not the allowance {limit}")
        o = r["outputs"][0]
        ids = o["token_ids"]
        p = per[row["mode"]]
        p["turns"] += 1
        p["output_tokens"].append(len(ids))
        if ids and ids[-1] in g["stop_ids"]:
            p["ended_on_stop"] += 1
        elif o.get("finish_reason") == "length" and len(ids) == limit:
            p["at_limit"] += 1
        else:
            bad.append(f"{key}: ended on neither a stop id nor its limit ({o.get('finish_reason')})")
        if row["mode"] == "H-N" and any(i in g["think_ids"] for i in ids):
            bad.append(f"{key}: an H-N output holds a think delimiter")
        if row["mode"] == "H-T" and ids[:1] == [g["think_ids"][0]]:
            p["ht_opened_think"] += 1
        if row["think_state"] != cfg["modes"][row["mode"]]["think_state"]:
            bad.append(f"{key}: think state {row['think_state']}")
    if any(len(v) != 1 for v in seeds.values()):
        bad.append("a task's seed differs between the modes")
    built = {m: 0 for m in MODES}
    shas = {d["doc_id"]: d["content_sha256"] for d in (dd or {}).get("documents", [])}
    for q in gen["request"]["requests"]:
        key = q["id"].removeprefix("gen:")
        row, r = rows[key], by.get(q["id"])
        if row["kind"] != "t0" or r is None:
            continue
        turn = {"prompt_token_ids": r["prompt_token_ids"], "output_token_ids": r["outputs"][0]["token_ids"]}
        try:
            d = document_from_turn(turn, doc_id=key, stop_ids=g["stop_ids"])
        except DocumentError:
            continue
        if prompt_mode_problems(d, mode_spec(cfg, row["mode"])["generation_suffix_ids"]):
            continue
        built[row["mode"]] += 1
        if shas.get(key) != d.content_sha256():
            bad.append(f"{key}: the recorded document is not the engine's turn")
    bad += [f"{m}: no document" for m in MODES if not built[m]]
    for m in MODES:
        ot = per[m].pop("output_tokens")
        per[m]["output_tokens_mean"] = round(sum(ot) / len(ot), 1) if ot else None
        per[m]["output_tokens_max"] = max(ot, default=None)
        per[m]["documents"] = built[m]
    return _st(bad, per_mode=per, vllm=(eng.get("result") or {}).get("versions"),
               refused=(dd or {}).get("refused"))


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def check_teacher_forward(out: Path, cfg: dict, run: dict) -> dict:
    tf, hf, el, mk = (_read(out, k) for k in ("teacher_forward", "hf_logprobs", "engine_logprobs", "masks"))
    if not tf:
        ph = (_read(out, "phases") or {}).get("teacher") or {}
        return _st([f"no teacher_forward evidence (phase: {ph.get('problems') or ph.get('returncode')})"])
    c, bad, per = cfg["teacher_forward"], [], {}
    for mode in MODES:
        m = tf["modes"].get(mode) or {}
        mem = [x for s in m.get("sequences", []) for x in s["members"]]
        if not mem:
            bad.append(f"{mode}: no document forwarded")
            continue
        if m["pack_sha256"] != ((mk or {}).get("modes", {}).get(mode) or {}).get("pack_sha256"):
            bad.append(f"{mode}: the teacher phase packed something other than the recorded pack")
        ref = max(x["ref_bf16_alone"] for x in mem)
        d16, d32 = max(x["d_bf16"] for x in mem), max(x["d_fp32"] for x in mem)
        row = {"documents": len(mem), "targets": sum(x["targets"] for x in mem), "ref_bf16_alone": ref,
               "ref_bf16_packed": max(x["ref_bf16_packed"] for x in mem), "d_bf16": d16, "d_fp32": d32,
               "bf16_limit": c["bf16_k"] * ref, "fp32_limit": c["fp32_fraction_of_bf16_reference"] * ref}
        if not ref > 0:
            bad.append(f"{mode}: the bf16 reference is {ref}: bf16 was not exercised")
        if d16 > row["bf16_limit"]:
            bad.append(f"{mode}: bf16 packed-vs-alone {d16:.4g} > {c['bf16_k']} x {ref:.4g}")
        if d32 > row["fp32_limit"]:
            bad.append(f"{mode}: fp32 packed-vs-alone {d32:.4g} > {c['fp32_fraction_of_bf16_reference']} x {ref:.4g}")
        # the engine against the HF forward on the same target tokens
        eng = (el or {}).get("engine") or {}
        if (el or {}).get("engine_stand_in") and not run["tiny"]:
            bad.append("a stand-in engine on a pod run")
        if eng.get("status") == "NOT_RUN" and run["tiny"]:
            row["engine"] = {"status": "NOT_RUN", "why": eng.get("why")}
        else:
            de, dh = [], []
            for x in mem:
                v, h = (el or {}).get("target_logprobs", {}).get(x["doc_id"]), (hf or {}).get(x["doc_id"])
                if not v or not h or any(a is None for a in v) or len(v) != len(h["fp32"]):
                    bad.append(f"{x['doc_id']}: no engine log-probability for every target")
                    continue
                de += [abs(a - b) for a, b in zip(v, h["fp32"])]
                dh += [abs(a - b) for a, b in zip(h["bf16"], h["fp32"])]
            if de:
                lim = c["engine_mean_k"] * _mean(dh)
                srt = sorted(de)
                row["engine"] = {"mean_abs": _mean(de), "max_abs": srt[-1], "p99_abs": srt[int(0.99 * (len(srt) - 1))],
                                 "hf_bf16_mean_abs": _mean(dh), "limit": lim}
                if _mean(de) > lim:
                    bad.append(f"{mode}: the engine's mean |logprob difference| {_mean(de):.4g} > "
                               f"{c['engine_mean_k']} x the HF bf16 {_mean(dh):.4g}")
        per[mode] = row
    return _st(bad, per_mode=per, attn_implementation=tf.get("attn_implementation"), tf32=tf.get("tf32"),
               device=tf.get("device"))


def check_assistant_target_kl(out: Path, cfg: dict, run: dict) -> dict:
    mk, tf = _read(out, "masks"), _read(out, "teacher_forward")
    if not mk:
        return _st(["no mask evidence"])
    bad, n = [], 0
    for mode in MODES:
        m = mk["modes"].get(mode) or {}
        if not m.get("sequences"):
            bad.append(f"{mode}: no packed document")
            continue
        for b, s in enumerate(m["sequences"]):
            want = sorted(t for x in s["members"] for t in range(x["start"] + x["prompt_len"] - 1,
                                                                 x["start"] + x["length"] - 1))
            if s["predicted"] != want:
                extra = sorted(set(s["predicted"]) - set(want))[:5]
                bad.append(f"{mode}/{b}: KD positions are not the generated-token layout (extra {extra})")
            pos = s["position_ids"]
            for x in s["members"]:
                if pos[x["start"]:x["start"] + x["length"]] != list(range(x["length"])):
                    bad.append(f"{mode}/{b}: {x['doc_id']}'s positions do not restart at 0")
            if pos[s["content_len"]:] != list(range(len(pos) - s["content_len"])):
                bad.append(f"{mode}/{b}: the pad run's positions do not restart")
            if len(s["predicted"]) != s["targets"]:
                bad.append(f"{mode}/{b}: {len(s['predicted'])} KD positions for {s['targets']} targets")
            n += len(s["predicted"])
            seq = (((tf or {}).get("modes", {}).get(mode) or {}).get("sequences") or [None] * (b + 1))
            ts = seq[b] if b < len(seq) else None
            if ts is None:
                bad.append(f"{mode}/{b}: no teacher-phase KL evidence")
                continue
            if ts["kl_self"] != 0.0 or ts["kl_self_positions"] != s["targets"]:
                bad.append(f"{mode}/{b}: KL(teacher||teacher) {ts['kl_self']} over {ts['kl_self_positions']}")
            if not math.isfinite(ts["student_kd"]) or ts["student_kd_positions"] != s["targets"]:
                bad.append(f"{mode}/{b}: student KD {ts['student_kd']} over {ts['student_kd_positions']} positions")
    return _st(bad, kd_positions=n, kd_scope=mk.get("kd_scope"))


def check_trainer_smoke(out: Path, cfg: dict, run: dict) -> dict:
    ts, env = _read(out, "trainer_smoke"), _read(out, "environment") or {}
    if not ts:
        ph = (_read(out, "phases") or {}).get("train") or {}
        return _st([f"no trainer_smoke evidence (phase: {ph.get('problems') or ph.get('returncode')})"])
    c, size, bad = cfg["trainer_smoke"], sizes(cfg, run["tiny"]), []
    declared = sorted(c["declared_changes"] + (cfg["tiny"]["extra_declared_changes"] if run["tiny"] else []))
    if ts["config_changed_keys"] != declared:
        bad.append(f"the smoke config differs from E1's in {ts['config_changed_keys']}, not the declared {declared}")
    if len(ts["steps"]) != size["steps"]:
        bad.append(f"{len(ts['steps'])} steps, not {size['steps']}")
    for s in ts["steps"]:
        if not all(isinstance(s[k], (int, float)) and math.isfinite(s[k]) for k in ("loss", "ce", "kd")):
            bad.append(f"step {s['step']}: a loss is not finite")
        if s["context_strategy"] != "document_isolated":
            bad.append(f"step {s['step']}: context strategy {s['context_strategy']}")
        if not (s["kd_positions"] == s["ce_targets"] > 0):
            bad.append(f"step {s['step']}: {s['kd_positions']} KD positions against {s['ce_targets']} CE targets")
    #: `Trainer(device=...)` is the only way the trainer reaches the GPU (it ignores cfg["device"])
    used = [str(ts.get(k) or "") for k in ("trainer_device", "student_device")]
    if not all(u and u.split(":")[0] == str(ts.get("requested_device") or "?").split(":")[0] for u in used):
        bad.append(f"the smoke ran on {used}, not the requested {ts.get('requested_device')}")
    stu = (env.get("inputs") or {}).get("student") or {}
    if not run["tiny"] and stu.get("model_safetensors_sha256") != c["student"]["model_safetensors_sha256"]:
        bad.append("the stand-in student is not the declared checkpoint")
    return _st(bad, seconds_per_step_mean_after_warmup=ts["seconds_per_step_mean_after_warmup"],
               peak_memory_gib=ts["peak_memory_gib"], sequence_length=ts["sequence_length"], pack=ts["pack"],
               documents_synthetic=ts["documents_synthetic"], device=ts["device"])


def check_preclosed_scoring(out: Path, cfg: dict, run: dict) -> dict:
    from stages.teacher_modes import evaluation_identity as E
    sc = _read(out, "scoring")
    if not sc:
        return _st(["no scoring evidence"])
    bad = []
    again = [score_record(cfg, r) for r in sc["records"]]
    if again != sc["results"]:
        bad.append("re-scoring the stored records does not reproduce the recorded results")
    well = {m: 0 for m in MODES}
    for r, x in zip(sc["records"], again):
        want = cfg["modes"][r["mode"]]["think_state"]
        if x["read_think_state"] != want:
            bad.append(f"{r['key']}: the scorer read {x['read_think_state']}, not {want}")
        if r["structural_ok"]:
            well[r["mode"]] += 1
            if not x["protocol_valid"][0] or x["adapter"]["class"] == "protocol_failure":
                bad.append(f"{r['key']}: a well-formed record is a protocol violation ({x['protocol_valid'][1]}, "
                           f"{x['adapter']})")
        elif r["mode"] == "H-N" and any(t in r["raw"] for t in ("<think>", "</think>")) and x["protocol_valid"][0]:
            bad.append(f"{r['key']}: an H-N reply holding a delimiter was accepted")
        if r["mode"] == "H-N" and r["structural_ok"]:
            if x["historical_rule_without_think_state"] != [False, "think_delimiters_invalid"]:
                bad.append(f"{r['key']}: without think_state the reply is {x['historical_rule_without_think_state']}")
            if x["delimiter_inserted"][0]:
                bad.append(f"{r['key']}: a </think> inserted into the reply was accepted")
    bad += [f"{m}: no well-formed record" for m in MODES if not well[m]]
    committed = json.loads((REPO / cfg["evaluation_identity"]).read_text())
    if E.build() != committed:
        bad.append("evaluation_identity.json does not regenerate to the committed document")
    return _st(bad, records=len(sc["records"]), well_formed=well,
               identity_sha256={m: committed["modes"][m]["identity_sha256"] for m in MODES},
               synthetic=sorted({r["synthetic"] for r in sc["records"]}))


CHECKS = {"prompt_identity": check_prompt_identity, "generation": check_generation,
          "teacher_forward": check_teacher_forward, "assistant_target_kl": check_assistant_target_kl,
          "trainer_smoke": check_trainer_smoke, "preclosed_scoring": check_preclosed_scoring}


def verdict(out: str | Path, config: dict) -> dict:
    """Every check's status from the evidence files in `out` and the fixed thresholds in `config` alone."""
    out = Path(out)
    run = _read(out, "run") or {}
    cfg_sha = sha_json(config)
    checks = {}
    for name, fn in CHECKS.items():
        try:
            checks[name] = fn(out, config, run)
        except Exception as exc:  # noqa: BLE001 -- a malformed record is a failed check, never a crash
            checks[name] = _st([f"the check could not be derived: {type(exc).__name__}: {exc}"])
    allowed = {"PASS"} | ({"NOT_RUN"} if run.get("tiny") else set())
    ok = all(c["status"] in (allowed if n in ENGINE_CHECKS else {"PASS"}) for n, c in checks.items())
    return {"schema": SCHEMA, "run_id": run.get("run_id"), "study": "A-E", "tiny": run.get("tiny"),
            "device": run.get("device"), "started_utc": run.get("started_utc"),
            "config": {"path": run.get("config"), "sha256": run.get("config_sha256"),
                       "matches_verdict_config": run.get("config_sha256") == cfg_sha},
            "environment": _read(out, "environment"), "checks": checks,
            "passed": bool(ok and run.get("config_sha256") == cfg_sha),
            "scientific_use": False, "authorizes": "nothing"}


# ---- the parent ----------------------------------------------------------------------------------------

def tiny_models(cfg: dict, work: Path) -> None:
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    for name, seed in (("tiny_teacher", 0), ("tiny_student", 1)):
        torch.manual_seed(seed)
        m = Qwen3ForCausalLM(Qwen3Config(**cfg["tiny"]["model"], tie_word_embeddings=True))
        m.save_pretrained(work / name)


def run_phase(args, name: str) -> dict:
    """One torch phase in a child of this interpreter, on a GPU nothing else holds."""
    rec = {"started_utc": utc()}
    if not args.tiny:
        rec["gpu_before"] = VC.gpu_state()
        bad = VC.clear_problems(rec["gpu_before"])
        if bad:
            return {**rec, "problems": bad, "returncode": None}
    else:
        rec["gpu_before"] = {"status": "NOT_APPLICABLE", "why": "a host run"}
    argv = [sys.executable, str(Path(__file__).resolve()), "--phase", name, "--run-id", args.run_id,
            "--out", str(args.out), "--work", str(args.work), "--device", args.device, "--config", args.config]
    argv += ["--tiny"] if args.tiny else ["--student", str(args.student)]
    t = time.time()
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=load_config(args.config)["timeouts_s"][name])
        rc, so, se = r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired as exc:
        rc, so, se = None, str(exc.stdout or ""), f"timeout: {str(exc.stderr or '')}"
    (Path(args.work) / f"phase_{name}.log").write_text(so + "\n--- stderr ---\n" + se)
    return {**rec, "returncode": rc, "seconds": round(time.time() - t, 1), "stdout_tail": so[-2000:],
            "stderr_tail": se[-4000:], "problems": [] if rc == 0 else [f"the {name} phase exited {rc}"]}


def environment(args, gen: dict | None) -> dict:
    import torch
    import transformers
    env = {"python": platform.python_version(), "torch": torch.__version__,
           "transformers": transformers.__version__, "cuda": torch.version.cuda,
           "gpu": VC.gpu_state() if not args.tiny else {"status": "NOT_APPLICABLE"},
           "vllm": ((((gen or {}).get("engine") or {}).get("result") or {}).get("versions")), "inputs": {}}
    sd = _student_dir(args) / "model.safetensors"
    env["inputs"]["student"] = {"path": str(_student_dir(args)),
                                "model_safetensors_sha256": sha_file(sd) if sd.is_file() else None}
    return env


def run(args) -> dict:
    out, work = Path(args.out), Path(args.work)
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    _write(out, "run", {"schema": SCHEMA, "run_id": args.run_id, "study": "A-E", "tiny": bool(args.tiny),
                        "device": _device(args), "config": args.config, "config_sha256": sha_json(cfg),
                        "started_utc": utc(), "argv": sys.argv[1:]})
    size, phases, gen, tok, turn_rows = sizes(cfg, args.tiny), {}, None, None, []

    def step(name, fn):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 -- every check still runs
            phases.setdefault("errors", {})[name] = traceback.format_exc()[-3000:]
            print(f"[{name}] {type(exc).__name__}: {exc}", flush=True)
            return None

    if args.tiny:
        step("tiny_models", lambda: tiny_models(cfg, work))
    tok = step("tokenizer", lambda: tokenizer(cfg, args.tiny))
    prompts = step("prompts", lambda: render_prompts(cfg, tok, tiny=args.tiny))
    if prompts:
        _write(out, "prompts", prompts)
        gen = step("generation", lambda: generate(cfg, args, prompts, size))
        if gen:
            _write(out, "generation", gen)
        turn_rows = turns(cfg, prompts, gen)
        built = step("documents", lambda: documents(cfg, prompts, turn_rows))
        if built:
            docs, refused = built
            for m in MODES:      # a mode the engine left without a document gets synthetic ones
                if not docs[m]:
                    syn = [t | {"output_token_ids": synthetic_output(cfg, m), "synthetic": True}
                           for t in turn_rows if t["mode"] == m and t["kind"] == "t0"]
                    docs[m] = documents(cfg, prompts, syn)[0][m]
            _write(out, "documents", documents_doc(docs, refused))
            (work / "documents_full.json").write_text(json.dumps(docs_payload(docs)))
            step("masks", lambda: _write(out, "masks", masks_doc(cfg, docs, size)))
            el = step("engine_scoring", lambda: engine_scoring(cfg, args, docs, size))
            if el:
                _write(out, "engine_logprobs", el)
            phases["teacher"] = run_phase(args, "teacher")
            phases["train"] = run_phase(args, "train")
        recs = step("eval_records", lambda: eval_records(cfg, tok, turn_rows))
        if recs is not None:
            step("scoring", lambda: _write(out, "scoring", scoring_doc(cfg, recs)))
    env = step("environment", lambda: environment(args, gen))
    if env:
        _write(out, "environment", env)
    _write(out, "phases", phases)
    report = verdict(out, cfg)
    (out / REPORT).write_text(json.dumps(report, indent=1, default=str) + "\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", required=True, help="the evidence directory (small files only)")
    ap.add_argument("--work", required=True, help="scratch: models, documents, engine requests, phase logs")
    ap.add_argument("--status", default=None, help="the session status file the terminal marker is appended to")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--vllm-python", default="/opt/vllm/bin/python")
    ap.add_argument("--student", default=None, help="the stand-in student's checkpoint directory")
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--tiny", action="store_true", help="host path: random small models, no engine")
    ap.add_argument("--phase", choices=("teacher", "train"), default=None, help=argparse.SUPPRESS)
    ap.add_argument("--engine-script", default=None, help=argparse.SUPPRESS)  # a stand-in engine, tests only
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.phase:
        {"teacher": phase_teacher, "train": phase_train}[args.phase](args)
        return 0
    if not args.tiny and not args.student:
        raise SystemExit("--student is required on a pod run")
    if not args.status:
        raise SystemExit("--status is required")
    passed = False
    try:
        rep = run(args)
        passed = bool(rep["passed"])
        print(json.dumps({k: v["status"] for k, v in rep["checks"].items()} | {"passed": passed}, indent=1))
    finally:
        with Path(args.status).open("a") as fh:
            fh.write(f"{utc()} MARKER:{'ALL_DONE' if passed else 'RUN_FAILED'}\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
