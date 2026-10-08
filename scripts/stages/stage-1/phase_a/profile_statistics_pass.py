"""Split the activation-statistics pass into GPU forward vs CPU float64 accumulate.

Every Phase-A cost is a range because of this one unmeasured quantity: the model
runs on the GPU while `X^T X` accumulates in float64 on the CPU, and only the
CPU-only end-to-end rate has ever been measured. One profile collapses a 3.6x
spread, which is why it is a gate rather than a nice-to-have.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.infrastructure.manifest import sha256_json  # noqa: E402

TEACHER = "Qwen/Qwen3-4B-Thinking-2507"
REVISION = "768f209d9ea81521153ed38c47d515654e938aea"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=59_763,
                    help="the domain_balanced@v1 mixture size")
    ap.add_argument("--seq-len", type=int, default=892)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    #: Test-only override, recorded in the output. See
    #: measure_state_repeatability.py for why it exists.
    ap.add_argument("--teacher", default=TEACHER)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    def sync() -> None:
        # The split this measures is GPU-forward vs CPU-accumulate, so on CPU
        # there is nothing to synchronize and the numbers are a smoke run, not
        # the gate. Guarded rather than assumed: an unguarded
        # `torch.cuda.synchronize()` makes the script unrunnable anywhere it
        # could have been tested for free.
        if str(args.device).startswith("cuda"):
            torch.cuda.synchronize()

    kwargs = {"revision": REVISION} if args.teacher == TEACHER else {}
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher, dtype=torch.bfloat16,
        output_hidden_states=True, **kwargs).to(args.device).eval()
    hidden = model.config.hidden_size
    n_seq = max(1, args.tokens // args.seq_len)

    gpu_s, cpu_s = [], []
    torch.manual_seed(0)
    for _ in range(args.repeats):
        gram = torch.zeros(hidden, hidden, dtype=torch.float64)
        g = c = 0.0
        for _ in range(n_seq):
            ids = torch.randint(0, model.config.vocab_size, (1, args.seq_len),
                                device=args.device)
            sync()
            t = time.time()
            with torch.no_grad():
                states = model(ids).hidden_states[-1][0].float()
            sync()
            g += time.time() - t
            host = states.cpu().double()
            t = time.time()
            gram += host.T @ host
            c += time.time() - t
        gpu_s.append(g)
        cpu_s.append(c)

    total = [a + b for a, b in zip(gpu_s, cpu_s)]
    out = {
        "schema": "aadistill.autoinit.statistics_split/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "teacher": args.teacher, "revision": REVISION, "device": args.device,
        "is_real_teacher": args.teacher == TEACHER,
        "is_gate_measurement": (args.teacher == TEACHER
                                and str(args.device).startswith("cuda")),
        "tokens": args.tokens, "seq_len": args.seq_len, "sequences": n_seq,
        "hidden_size": hidden, "repeats": args.repeats,
        "gpu_forward_seconds": [round(x, 3) for x in gpu_s],
        "cpu_float64_accumulate_seconds": [round(x, 3) for x in cpu_s],
        "total_seconds": [round(x, 3) for x in total],
        "gpu_fraction": round(sum(gpu_s) / sum(total), 4) if sum(total) else None,
        "seconds_per_1k_tokens": round(
            sum(total) / args.repeats / (n_seq * args.seq_len) * 1000, 4),
        "note": ("one statistics pass over the real mixture; the CPU term does "
                 "not shrink with a faster GPU, which is what the range was about"),
    }
    out["report_sha256"] = sha256_json(out)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in
                      ("gpu_fraction", "seconds_per_1k_tokens")}, indent=2))


if __name__ == "__main__":
    main()
