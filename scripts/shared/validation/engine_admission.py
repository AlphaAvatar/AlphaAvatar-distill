"""Is an inference engine computing the same function as the Hugging Face reference implementation?

    # the reference, in a PyTorch child that exits (the GPU must be clear before vLLM starts)
    ref = EA.hf_reference(model_dir, probes, device="cuda:0")
    # the engine, in its own interpreter, through shared.validation.vllm_check.run_isolated
    res = VC.run_isolated(vllm_python, EA.vllm_request(model_dir, probes, max_model_len), workdir)
    verdict = EA.compare(ref, res["result"], criteria)

For a model the engine runs through its own implementation (an out-of-tree
model, not a format conversion), agreement with the reference implementation is
a measurement on the evaluation card. The yardstick is measured in the same
session, never a fixed number: on the same token ids, the reference's own
bfloat16-vs-float32 difference of the realized next-token log-probabilities is
the size of disagreement bfloat16 alone produces for THIS model, and an engine
computing the same function in bfloat16 should disagree with the reference's
bfloat16 run by about that much and not more. A fixed absolute bound (the
historical folded model's 0.1 nat) cannot tell a model that is merely
bfloat16-sensitive from an engine that computes something else.

``compare`` admits the engine when, over every teacher-forced position:

* median and p99 of |engine - HF bf16| are each at most ``factor`` x the median
  and p99 of |HF bf16 - HF fp32|;
* at positions where the float32 reference leads its runner-up by at least
  ``confident_margin_nats``, the engine's argmax equals the reference's in at
  least ``confident_top1_agreement_min`` of them (with at least
  ``confident_positions_min`` such positions).

The probe is the engineering diagnostic text of the caller's config
(``probe_ids``): written for engine numerics, isolated from every held-out
evaluation unit, tokenized as plain text, so that debugging an implementation
never re-reads a formal evaluation set and the probe depends on no teacher's
chat template.

Nothing here scores, selects or decides anything scientific.
"""
from __future__ import annotations

import math
import statistics
from pathlib import Path

SCHEMA = "aadistill.engine_admission/v1"
CRITERIA_KEYS = ("factor", "confident_margin_nats", "confident_top1_agreement_min", "confident_positions_min")


def ids_sha(ids: list[int]) -> str:
    import hashlib
    import json
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode()).hexdigest()


def probe_ids(probe: dict, tokenizer_dir: str | Path) -> tuple[list[dict], list[list[int]]]:
    """The config's diagnostic texts as token ids (plain text, no chat template, no added special tokens).

    Returns the evidence rows (id, length, the ids' sha256 and the configured one;
    never the text) and the ids. A row whose hash is not the configured one is the
    caller's to refuse."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    rows, ids_all = [], []
    for t in probe["texts"]:
        ids = tok(t["text"], add_special_tokens=False).input_ids
        rows.append({"id": t["id"], "tokens": len(ids), "token_ids_sha256": ids_sha(ids),
                     "expected_sha256": t["token_ids_sha256"]})
        ids_all.append(list(ids))
    return rows, ids_all


def hf_reference(model_dir: str | Path, probes: list[list[int]], *, device: str) -> dict:
    """Teacher-forced reference on each probe: realized log-probs in bf16 and fp32, fp32 argmax and margin."""
    import torch
    from transformers import AutoModelForCausalLM

    from aadistill.initialization.adapters.families import register_custom_model_families

    register_custom_model_families()
    out = {"schema": SCHEMA, "model_dir": str(model_dir), "device": device, "prompts": []}
    runs = {}
    for dtype in (torch.float32, torch.bfloat16):
        m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype, attn_implementation="sdpa").to(device).eval()
        rows = []
        with torch.no_grad():
            for ids in probes:
                x = torch.tensor([ids], device=device)
                lp = torch.log_softmax(m(x, use_cache=False).logits[0, :-1].float(), -1)
                tgt = x[0, 1:]
                top2 = lp.topk(2, dim=-1)
                rows.append({"realized": lp.gather(-1, tgt[:, None]).squeeze(-1).tolist(),
                             "argmax": top2.indices[:, 0].tolist(),
                             "margin": (top2.values[:, 0] - top2.values[:, 1]).tolist()})
        runs[str(dtype).replace("torch.", "")] = rows
        del m
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    if str(device).startswith("cuda"):
        out["device_name"] = torch.cuda.get_device_name(torch.device(device))
        out["tf32_matmul"] = bool(torch.backends.cuda.matmul.allow_tf32)
    for i, ids in enumerate(probes):
        f, b = runs["float32"][i], runs["bfloat16"][i]
        out["prompts"].append({"id": f"p{i}", "positions": len(ids) - 1, "fp32_realized": f["realized"],
                               "bf16_realized": b["realized"], "fp32_argmax": f["argmax"], "fp32_margin": f["margin"]})
    return out


def vllm_request(model_dir: str | Path, probes: list[list[int]], max_model_len: int, *, tokenizer: str | Path,
                 prompt_logprobs: int = 1, gen_tokens: int = 32) -> dict:
    """A vllm_check request: each probe teacher-forced (prompt log-probs) and greedily continued.

    ``tokenizer`` is a local directory holding the tokenizer files: a trained
    checkpoint carries none, and the engine runs offline."""
    return {"model": str(model_dir), "tokenizer": str(tokenizer), "dtype": "bfloat16",
            "max_model_len": int(max_model_len), "seed": 0,
            "requests": [{"id": f"p{i}", "prompt_token_ids": list(ids),
                          "sampling": {"temperature": 0.0, "max_tokens": gen_tokens,
                                       "prompt_logprobs": prompt_logprobs, "logprobs": 1}}
                         for i, ids in enumerate(probes)]}


def _q(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[int(q * (len(s) - 1))]


def compare(reference: dict, result: dict | None, criteria: dict) -> dict:
    """The admission verdict; ``status`` PASS or FAIL with every measured quantity beside it."""
    missing = [k for k in CRITERIA_KEYS if k not in criteria]
    if missing:
        raise ValueError(f"criteria lack {missing}")
    problems: list[str] = []
    if not result or not result.get("requests"):
        return {"schema": SCHEMA, "status": "FAIL", "problems": ["the engine returned no result"]}
    by_id = {p["id"]: p for p in reference["prompts"]}
    eng, base, confident, agree, finite = [], [], 0, 0, True
    for r in result["requests"]:
        h = by_id.get(r["id"])
        ids = r["prompt_token_ids"]
        if h is None or h["positions"] != len(ids) - 1 or len(r["prompt_logprobs"]) != len(ids):
            problems.append(f"{r['id']}: the reference and the engine do not cover the same positions")
            continue
        for i in range(1, len(ids)):
            pos = {int(t): float(lp) for t, lp, _ in (r["prompt_logprobs"][i] or [])}
            if ids[i] not in pos:
                problems.append(f"{r['id']}: position {i} reports no log-probability for its token")
                break
            finite &= all(math.isfinite(v) for v in pos.values())
            eng.append(abs(pos[ids[i]] - h["bf16_realized"][i - 1]))
            base.append(abs(h["bf16_realized"][i - 1] - h["fp32_realized"][i - 1]))
            if h["fp32_margin"][i - 1] >= criteria["confident_margin_nats"]:
                confident += 1
                agree += max(pos, key=pos.get) == h["fp32_argmax"][i - 1]
    if not finite:
        problems.append("a non-finite engine log-probability")
    measured = {"positions": len(eng)}
    if eng:
        measured.update(
            engine_vs_hf_bf16={"median": statistics.median(eng), "p99": _q(eng, 0.99), "max": max(eng)},
            hf_bf16_vs_fp32={"median": statistics.median(base), "p99": _q(base, 0.99), "max": max(base)},
            confident_positions=confident, confident_top1_agreement=(agree / confident if confident else None))
        f = float(criteria["factor"])
        for stat in ("median", "p99"):
            got, yard = measured["engine_vs_hf_bf16"][stat], measured["hf_bf16_vs_fp32"][stat]
            if got > f * yard:
                problems.append(f"{stat} |engine - HF bf16| {got:.4g} exceeds {f} x the bf16 baseline {yard:.4g}")
        if confident < criteria["confident_positions_min"]:
            problems.append(f"{confident} confident positions, fewer than {criteria['confident_positions_min']}")
        elif measured["confident_top1_agreement"] < criteria["confident_top1_agreement_min"]:
            problems.append(f"confident top-1 agreement {measured['confident_top1_agreement']:.4f} below "
                            f"{criteria['confident_top1_agreement_min']}")
    else:
        problems.append("no comparable positions")
    return {"schema": SCHEMA, "status": "PASS" if not problems else "FAIL", "problems": problems,
            "criteria": dict(criteria), **measured, "scientific_use": False}
