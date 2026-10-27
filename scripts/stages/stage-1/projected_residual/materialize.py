#!/usr/bin/env python3
"""Materialize the projected-residual student: W0 -> per-layer statistics -> bases -> student.

    PYTHONPATH=src:scripts python scripts/stages/stage-1/projected_residual/materialize.py \\
        --design configs/stages/stage-1/projected_residual/design.json --out <dir> --device cuda:0

GPU-preferred (the 4B teacher and a 1.5B full-width parent); it also runs on a
host for a smoke test with ``--tiny`` (a random small teacher and selection), which
exercises every line below except the hub download. Steps, each recorded:

1. load the teacher at its pinned revision; build ``W0`` = the teacher with the
   selection's layers, query heads and FFN neurons kept, width unchanged;
2. stream the calibration profile's items through ``W0`` and accumulate each
   layer's float64 second moments of the declared statistic's roles over every
   real token;
3. ``P_l`` = the top-``internal_size`` eigenbasis per layer; per-role captured
   energy recorded, for the declared statistic's roles and for every
   ``diagnostic_statistics`` role (the other statistics' inputs, and the
   sub-layers' outputs under ``read_write``) under the declared basis: how well
   that basis fits each layer's input AND output subspaces;
4. the projected student (``W0``'s weights with ``P_l`` folded in), saved with
   ``save_pretrained``, its weights hashed;
5. the fold check: the student against its folded stock-Qwen3 form on a fixed
   probe, float32 and bfloat16, so the evaluation route is measured before use.

Nothing here trains, evaluates or selects; the selection is an input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))


def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def tiny_inputs():
    """A random small teacher, a valid selection and token items: the host smoke path."""
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    teacher = Qwen3ForCausalLM(Qwen3Config(vocab_size=97, hidden_size=64, num_hidden_layers=6,
                                           intermediate_size=96, num_attention_heads=8, num_key_value_heads=2,
                                           head_dim=8, tie_word_embeddings=True, max_position_embeddings=256))
    sel = {"kept_layers": [0, 1, 3, 5], "kept_heads": [[0, 1, 4, 5]] * 4,
           "kept_neurons": [list(range(0, 96, 2))] * 4}
    g = torch.Generator().manual_seed(1)
    items = [{"ids": torch.randint(0, 97, (int(n),), generator=g).tolist()} for n in (17, 33, 9, 41)]
    return teacher, sel, items, {"internal_size": 16, "statistic": "sublayer_inputs"}


def run(design: dict, out: Path, *, device: str, tiny: bool,
        diagnostic_statistics: tuple[str, ...] = ()) -> dict:
    import torch
    from transformers import AutoModelForCausalLM

    from aadistill.initialization.adapters import build_registry
    from aadistill.initialization.transforms.projected_residual import (
        STATISTICS, SublayerStatsCollector, materialize_projected, projection_basis, select_full_width)
    from aadistill.models.cost_model import parameter_counts
    from aadistill.initialization.adapters.projected_residual import fold_to_qwen3

    adapter = build_registry(["qwen3"]).get("qwen3")
    t0 = time.time()
    timings = {}
    if tiny:
        teacher, selection, items, arch = tiny_inputs()
        teacher = teacher.to(device)
    else:
        arch = design["architecture"]
        tid, rev = design["teacher"]["model"], design["teacher"]["revision"]
        teacher = AutoModelForCausalLM.from_pretrained(tid, revision=rev, dtype=torch.float32).to(device).eval()
        sel_doc = json.loads((REPO / design["selection"]["path"]).read_text())
        if sel_doc["content_sha256"] != design["selection"]["content_sha256"]:
            raise SystemExit("the selection record is not the design's")
        selection = {k: sel_doc[k] for k in ("kept_layers", "kept_heads", "kept_neurons")}
        prof = design["calibration"]
        path = REPO / prof["items_path"]
        if file_sha256(path) != prof["items_file_sha256"]:
            raise SystemExit(f"{path}: not the profile's items")
        items = [json.loads(l) for l in path.read_text().split("\n") if l.strip()]
    timings["load_s"] = round(time.time() - t0, 1)
    spec = adapter.spec_of(teacher).replace(num_hidden_layers=len(selection["kept_layers"]),
                                           num_attention_heads=len(selection["kept_heads"][0]),
                                           intermediate_size=len(selection["kept_neurons"][0]))
    w0 = select_full_width(adapter, teacher, selection, spec, dtype=torch.float32)
    del teacher
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    w0 = w0.to(device)
    roles = STATISTICS[arch["statistic"]]
    extra = tuple(r for s in diagnostic_statistics for r in STATISTICS[s] if r not in roles)
    extra = tuple(dict.fromkeys(extra))
    t1 = time.time()
    col = SublayerStatsCollector(adapter, w0, roles + extra)
    positions = 0
    for it in items:
        ids = torch.tensor([it["ids"]], device=device)
        col.process(w0, ids)
        positions += ids.shape[1]
    col.close()
    st = col.state()
    timings["statistics_s"] = round(time.time() - t1, 1)
    t2 = time.time()
    bases, diags = [], []
    for i in range(len(adapter.blocks(w0))):
        p, dg = projection_basis({r: st[r][i] for r in roles}, int(arch["internal_size"]))
        for r in extra:                       # the declared basis against the other roles' moments
            m = st[r][i][0].double().to(p.device)
            dg[f"{r}_energy_under_declared_basis"] = float(torch.trace(p.T @ m @ p) / torch.trace(m))
        bases.append(p)
        diags.append({"layer": i, **dg})
    timings["bases_s"] = round(time.time() - t2, 1)
    student = materialize_projected(adapter, w0, bases, dtype=torch.float32)
    out.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(out / "model")
    shards = sorted((out / "model").glob("*.safetensors"))
    # the fold check, on the first item's first 256 tokens
    probe = torch.tensor([items[0]["ids"][:256]], device="cpu")
    student_cpu = student.to("cpu")
    folded = fold_to_qwen3(student_cpu, dtype=torch.float32)
    with torch.no_grad():
        a = student_cpu(probe, use_cache=False).logits
        b = folded(probe, use_cache=False).logits
        fold_fp32 = float((a - b).abs().max())
        a16 = student_cpu.to(torch.bfloat16)(probe, use_cache=False).logits.float()
        b16 = folded.to(torch.bfloat16)(probe, use_cache=False).logits.float()
        fold_bf16 = float((a16 - b16).abs().max())
        bf16_vs_fp32 = float((a16 - a).abs().max())
    geom = {k: getattr(student.config, k) for k in ("vocab_size", "hidden_size", "internal_size",
                                                    "intermediate_size", "num_hidden_layers", "num_attention_heads",
                                                    "num_key_value_heads", "head_dim", "tie_word_embeddings")}
    counted = sum(p.numel() for p in student.parameters())
    record = {
        "schema": "aadistill.stage1.projected_residual.materialization/v1",
        "tiny_smoke": tiny,
        "teacher": None if tiny else design["teacher"],
        "selection_content_sha256": None if tiny else design["selection"]["content_sha256"],
        "statistic": arch["statistic"], "roles": list(roles), "diagnostic_roles": list(extra),
        "calibration_positions": positions,
        "geometry": geom, "num_parameters": counted,
        "num_parameters_cost_model": parameter_counts(geom)["total"],
        "weights": [{"file": p.name, "sha256": file_sha256(p)} for p in shards],
        "config_sha256": file_sha256(out / "model" / "config.json"),
        "per_layer": diags,
        "fold_check": {"probe_tokens": int(probe.shape[1]), "max_abs_logit_diff_fp32": fold_fp32,
                       "max_abs_logit_diff_bf16": fold_bf16, "bf16_vs_fp32_same_model": bf16_vs_fp32},
        "timings": timings, "device": device,
    }
    (out / "materialization.json").write_text(json.dumps(record, indent=1) + "\n")
    return record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--design", default="configs/stages/stage-1/projected_residual/design.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tiny", action="store_true", help="host smoke test on a random small teacher")
    a = ap.parse_args(argv)
    design = json.loads((REPO / a.design).read_text()) if not a.tiny else {}
    rec = run(design, Path(a.out), device=a.device, tiny=a.tiny)
    print(json.dumps({k: rec[k] for k in ("num_parameters", "num_parameters_cost_model", "fold_check",
                                          "timings")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
