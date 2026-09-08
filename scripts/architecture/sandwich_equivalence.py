#!/usr/bin/env python3
"""Hash every parameter `init_student` produces, so a refactor can be checked.

    PYTHONPATH=src:scripts python scripts/architecture/sandwich_equivalence.py --out <file>
    # refactor
    PYTHONPATH=src:scripts python scripts/architecture/sandwich_equivalence.py --compare <file>

Routing `sandwich.py`'s model-family access through the adapter touches 28 sites
in the Stage-1 initialization transform. Nothing about that is supposed to change
what it computes, and "supposed to" is not a standard that has served this
project well: an initialization that is subtly wrong produces a checkpoint that
trains, evaluates and looks plausible.

So the refactor is checked rather than reasoned about. This builds toy teacher and
student models under a fixed seed, runs the real `init_student`, and hashes every
resulting parameter. Two runs across the refactor must agree bit for bit.

Three geometries, because one can pass by a shape coincidence: equal width with an
identity projection (which must reproduce the teacher exactly), a width
compression, and a depth-and-width compression.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import torch  # noqa: E402

SEED = 20260908

#: The same toy teacher `tests/init/test_stage1.py` uses, including the
#: randomized RMSNorm weights -- a fresh model has every norm at 1.0, which
#: would make norm folding a no-op and hide any mistake in it.
TEACHER = dict(vocab_size=128, hidden_size=32, num_hidden_layers=4,
               intermediate_size=48, num_attention_heads=4,
               num_key_value_heads=2, head_dim=8, tie_word_embeddings=True,
               max_position_embeddings=256)

CASES = {
    #: Equal width, identity projection: the transform must reproduce the
    #: teacher exactly, which is the strongest single check available here.
    "identity_equal_width": dict(geometry={}, identity_proj=True, kept=None),
    "width_compression": dict(
        geometry=dict(hidden_size=16, intermediate_size=24),
        identity_proj=False, kept=None),
    "depth_and_width": dict(
        geometry=dict(hidden_size=16, intermediate_size=24,
                      num_hidden_layers=2),
        identity_proj=False, kept=[0, 3]),
}


def build_teacher(seed: int = 7):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(Qwen3Config(**TEACHER)).float().eval()
    with torch.no_grad():
        for m in model.modules():
            if m.__class__.__name__ == "Qwen3RMSNorm":
                m.weight.uniform_(0.5, 1.5)
    return model


def collect_stats(model, seed: int = 11, n_seqs: int = 8, seq_len: int = 64):
    from aadistill.initialization.statistics.collect import ActivationStatsCollector

    torch.manual_seed(seed)
    collector = ActivationStatsCollector(model)
    for _ in range(n_seqs):
        collector.process(torch.randint(0, model.config.vocab_size, (1, seq_len)))
    collector.close()
    return collector.state()


def run_case(name: str, case: dict) -> dict:
    from aadistill.initialization.adapters import register_builtin_adapters
    from aadistill.initialization.specs.arch import get_adapter
    from aadistill.initialization.transforms.sandwich import init_student

    from aadistill.models.student import build_student, build_student_config

    register_builtin_adapters()
    adapter = get_adapter("qwen3")

    def fresh():
        teacher = build_teacher()
        state = collect_stats(teacher)
        geo = {**{k: getattr(teacher.config, k) for k in
                  ("hidden_size", "num_hidden_layers", "intermediate_size",
                   "num_attention_heads", "num_key_value_heads", "head_dim",
                   "tie_word_embeddings")},
               **case["geometry"]}
        cfg = build_student_config(teacher.config, geo)
        return teacher, build_student(cfg, torch.float32, seed=123), state

    teacher, student, state = fresh()
    proj = (torch.eye(TEACHER["hidden_size"], dtype=torch.float64)
            if case["identity_proj"] else None)
    kwargs = {}
    try:
        init_student(teacher, student, state, proj_override=proj,
                     kept_layers=case["kept"], adapter=adapter)
        kwargs["adapter_passed"] = True
    except TypeError:
        # Before the refactor `init_student` takes no adapter.
        teacher, student, state = fresh()
        init_student(teacher, student, state, proj_override=proj,
                     kept_layers=case["kept"])
        kwargs["adapter_passed"] = False

    params = {n: hashlib.sha256(
        p.detach().to(torch.float64).contiguous().numpy().tobytes()).hexdigest()
        for n, p in sorted(student.named_parameters())}
    return {"case": name, "n_params": len(params), **kwargs,
            "digest": hashlib.sha256(
                "".join(f"{n}:{h}\n" for n, h in params.items()).encode()).hexdigest(),
            "parameters": params}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", default=None)
    args = ap.parse_args()

    results = {name: run_case(name, case) for name, case in CASES.items()}
    for name, r in results.items():
        print(f"  {name:22} {r['n_params']:>3} params  {r['digest'][:16]}…  "
              f"(adapter injected: {r['adapter_passed']})")

    if args.compare:
        before = json.loads(Path(args.compare).read_text())
        bad = []
        for name, r in results.items():
            was = before["results"][name]
            if was["digest"] != r["digest"]:
                moved = [n for n, h in r["parameters"].items()
                         if was["parameters"].get(n) != h]
                bad.append((name, moved))
        if bad:
            print("\nPARAMETERS MOVED:")
            for name, moved in bad:
                print(f"  {name}: {len(moved)} parameter(s), e.g. {moved[:4]}")
            return 1
        print("\nIDENTICAL: every parameter matches, across all cases.")
        return 0

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"seed": SEED, "teacher": TEACHER, "results": results}, indent=1) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
