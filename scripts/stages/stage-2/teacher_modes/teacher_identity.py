#!/usr/bin/env python3
"""Pin the hybrid teacher's identity and both of its native modes ($0, read-only hub calls).

    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-2/teacher_modes/teacher_identity.py \\
        --write configs/stages/stage-2/teacher_modes/teacher_identity.json

Reads the official `Qwen/Qwen3-4B` repository at a pinned commit: every small
file's sha256 from the local copy, every weight shard's LFS sha256 from the
hub's own metadata (the 8 GB of weights are not downloaded), the chat template's
sha256 under the rule every runner in this repository records
(`sha256(tokenizer.chat_template)`), the vocabulary hash the tokenizer contract
uses, and -- per mode -- the exact token ids the official template appends as
its generation prompt, which is what a recorded prompt is checked against.

The two modes are the template's HARD switch only (`enable_thinking`); the
`/think` and `/no_think` soft switches are out of this round. The sampling
presets are the model card's own recommendations, quoted with where they come
from. `Qwen/Qwen3-4B-Thinking-2507` is set beside it as the historical teacher,
not as an arm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

MODEL = "Qwen/Qwen3-4B"
REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
HISTORICAL = ("Qwen/Qwen3-4B-Thinking-2507", "768f209d9ea81521153ed38c47d515654e938aea")
FILES = ("config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json",
         "vocab.json", "merges.txt", "README.md", "LICENSE")
SYSTEM = "You are a helpful Assistant."
MODES = {
    "thinking": {"template_kwargs": {"enable_thinking": True},
                 "official_sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
                 "sampling_source": "model card README: 'For thinking mode, use Temperature=0.6, TopP=0.95, "
                                    "TopK=20, and MinP=0 (the default setting in generation_config.json)'"},
    "non_thinking": {"template_kwargs": {"enable_thinking": False},
                     "official_sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0},
                     "sampling_source": "model card README: 'For non-thinking mode, we suggest using "
                                        "Temperature=0.7, TopP=0.8, TopK=20, and MinP=0'"},
}
SPECIAL = ("<think>", "</think>", "<|im_start|>", "<|im_end|>", "<|endoftext|>")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build() -> dict:
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer

    from aadistill.data.token_documents import generation_suffix
    from aadistill.evaluation.behavior import template_think_state
    from aadistill.models.teacher import tokenizer_hash
    from shared.evaluation.uncapped_eval import tokenizer_files_sha256

    info = HfApi().model_info(MODEL, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise SystemExit(f"hub returned {info.sha}, not the pinned {REVISION}")
    local = {f: Path(hf_hub_download(MODEL, f, revision=REVISION)) for f in FILES}
    snapshot = local["config.json"].parent
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    cfg = json.loads(local["config.json"].read_text())
    gen = json.loads(local["generation_config.json"].read_text())
    shards = sorted((s.rfilename, s.size, getattr(s.lfs, "sha256", None) if s.lfs else None)
                    for s in info.siblings if s.rfilename.endswith(".safetensors"))
    probe = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "probe"}]
    modes = {}
    for name, m in MODES.items():
        suffix = generation_suffix(tok, m["template_kwargs"])
        prompt = tok.apply_chat_template(probe, tokenize=False, add_generation_prompt=True,
                                         **m["template_kwargs"])
        modes[name] = {**m, "generation_suffix_ids": list(suffix),
                       "generation_suffix_text": tok.decode(list(suffix)),
                       "think_state": template_think_state(prompt),
                       "supervised": ("every generated token: the model's own <think>, its reasoning, </think>, "
                                      "the answer and <|im_end|>" if name == "thinking" else
                                      "every generated token: the answer and <|im_end|>; the template's "
                                      "empty <think>\\n\\n</think>\\n\\n is prompt context, never a target")}
    hist = {}
    try:
        htok = AutoTokenizer.from_pretrained(HISTORICAL[0], revision=HISTORICAL[1], local_files_only=True)
        hcfg = json.loads(Path(hf_hub_download(HISTORICAL[0], "config.json", revision=HISTORICAL[1])).read_text())
        hist = {"model": HISTORICAL[0], "revision": HISTORICAL[1],
                "chat_template_sha256": hashlib.sha256((htok.chat_template or "").encode()).hexdigest(),
                "vocabulary_sha256": tokenizer_hash(htok),
                "generation_suffix_text": htok.decode(list(generation_suffix(htok))),
                "rope_theta": hcfg.get("rope_theta"), "max_position_embeddings": hcfg.get("max_position_embeddings")}
    except Exception as exc:                      # noqa: BLE001 - recorded, not hidden
        hist = {"model": HISTORICAL[0], "revision": HISTORICAL[1], "unavailable": f"{type(exc).__name__}: {exc}"}
    return {
        "schema": "aadistill.stage2.teacher_modes.teacher_identity/v1",
        "model": MODEL, "revision": REVISION, "hub_last_modified": str(info.last_modified),
        "license": "apache-2.0 (LICENSE, sha256 below)",
        "files_sha256": {f: sha256(p) for f, p in local.items()},
        "weight_shards": [{"file": f, "size_bytes": n, "lfs_sha256": h} for f, n, h in shards],
        "chat_template_sha256": hashlib.sha256((tok.chat_template or "").encode()).hexdigest(),
        "_chat_template_rule": "sha256 of tokenizer.chat_template, the rule run_battery.py and uncapped_eval.py record",
        "tokenizer_files_sha256": tokenizer_files_sha256(str(snapshot)),
        "vocabulary_sha256": tokenizer_hash(tok),
        "vocabulary_size": len(tok.get_vocab()),
        "special_token_ids": {t: tok.convert_tokens_to_ids(t) for t in SPECIAL},
        "eos_token_ids": gen.get("eos_token_id"),
        "pad_token_id": gen.get("pad_token_id"),
        "architecture": {k: cfg.get(k) for k in (
            "architectures", "hidden_size", "num_hidden_layers", "intermediate_size", "num_attention_heads",
            "num_key_value_heads", "head_dim", "vocab_size", "tie_word_embeddings", "rope_theta",
            "rope_scaling", "max_position_embeddings", "rms_norm_eps", "torch_dtype")},
        "native_context": {"tokens": 32768, "source": "model card README: 'Context Length: 32,768 natively'"},
        "system_message": SYSTEM,
        "_system_message": "both modes use the same base system message (sessions.SYSTEM_DEFAULT); nothing in it "
                           "names a mode, so prompt wording is not a variable",
        "switch": "hard switch only: tokenizer.apply_chat_template(..., enable_thinking=True|False); no /think or "
                  "/no_think soft switch, no hand-built prefix",
        "modes": modes,
        "historical_teacher": hist,
        "_historical_teacher": "the frozen teacher of every earlier round; NOT an arm of this study",
        "authorizes": "nothing",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", default=None)
    a = ap.parse_args(argv)
    doc = build()
    text = json.dumps(doc, indent=1) + "\n"
    if a.write:
        Path(a.write).write_text(text)
        print(a.write)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
