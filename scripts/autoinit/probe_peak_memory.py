"""Peak resident GPU memory on the widest operator. Replaces arithmetic.

The 14.3 GiB figure in the cost model is derived, not observed. DEPTH at full
width holds the teacher and a full-width child at once and is the widest point
of the search; if it exceeds 40 GiB on a 48 GB L40S, the hardware plan is wrong
and Phase A must be re-priced before it is booked.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.autoinit.arch import ArchSpec, get_adapter  # noqa: E402
from aadistill.infrastructure.manifest import sha256_json  # noqa: E402

TEACHER = "Qwen/Qwen3-4B-Thinking-2507"
REVISION = "768f209d9ea81521153ed38c47d515654e938aea"
DEPTH_ONLY_LAYERS = 28


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=892)
    ap.add_argument("--device", default="cuda")
    #: Test-only override, recorded in the output. See
    #: measure_state_repeatability.py for why it exists.
    ap.add_argument("--teacher", default=TEACHER)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    on_cuda = str(args.device).startswith("cuda")
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()
    adapter = get_adapter("qwen3")
    kwargs = {"revision": REVISION} if args.teacher == TEACHER else {}
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, dtype=torch.bfloat16, **kwargs).to(args.device).eval()
    base = AutoConfig.from_pretrained(args.teacher, **kwargs)
    spec = adapter.spec_from_config(base).replace(
        num_hidden_layers=DEPTH_ONLY_LAYERS)
    child_cfg = adapter.build_config(base, spec)
    child = AutoModelForCausalLM.from_config(child_cfg).to(
        args.device, dtype=torch.bfloat16).eval()

    ids = torch.randint(0, base.vocab_size, (1, args.seq_len), device=args.device)
    with torch.no_grad():
        teacher(ids)
        child(ids)
    # There is no accelerator peak to read on CPU. Rather than making the script
    # unrunnable off a GPU — which is what kept its one real defect hidden until
    # a paid pod — a CPU run executes every other step and records nulls, marked
    # so it can never be mistaken for the gate measurement.
    if on_cuda:
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        device_name = torch.cuda.get_device_properties(0).name
        device_total = round(
            torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    else:
        peak = reserved = None
        device_name, device_total = "cpu", None

    out = {
        "schema": "aadistill.autoinit.peak_memory/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "operator": "DEPTH at full width (teacher + full-width child resident)",
        "teacher_layers": base.num_hidden_layers,
        "child_layers": DEPTH_ONLY_LAYERS,
        "seq_len": args.seq_len, "dtype": "bfloat16",
        "peak_allocated_bytes": peak, "peak_reserved_bytes": reserved,
        "peak_gib": None if peak is None else round(peak / 2**30, 2),
        "peak_reserved_gib": (None if reserved is None
                              else round(reserved / 2**30, 2)),
        "device": device_name, "device_total_gib": device_total,
        "is_real_teacher": args.teacher == TEACHER,
        "is_gate_measurement": on_cuda and args.teacher == TEACHER,
        "derived_estimate_gib": 14.3,
        "gate": "fails Stage 1 above 40 GiB on a 48 GB L40S",
    }
    out["report_sha256"] = sha256_json(out)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in
                      ("peak_gib", "peak_reserved_gib", "device")}, indent=2))


if __name__ == "__main__":
    main()
