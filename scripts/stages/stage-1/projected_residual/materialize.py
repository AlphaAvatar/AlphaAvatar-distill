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
5. the STRUCTURAL fold check, on the CPU in float32: the student against its
   folded stock-Qwen3 form on the first calibration item's first 256 tokens.
   It asks whether the fold computes the same function (a transposed or
   misplaced projection moves logits by their own magnitude; a correct fold by
   float32 rounding), not how the evaluation device rounds it.

A design with ``architecture.projection_sharing = "independent"`` (four
projections per layer, optionally gated) takes, in step 3, one basis per site
from the activation that site maps (``INDEPENDENT_BASES``: the down maps from
the normalized sub-layer inputs, the up maps from the sub-layer outputs), and in
step 5 replaces the fold check -- a gated model does not fold -- with the
INITIALIZATION ERROR: ``aadistill.evaluation.function_error`` of the student
against ``W0`` (what the projections lose) and against the teacher (what the
selection and the projections lose together), in float32 on the materializing
device, over every real position of the calibration items (in-sample) and of
the probe prompts (``--probe``; out of sample).

Device numerics are a separate measurement: `fold_numerics` runs both forms ON
the evaluation device, float32 and bfloat16, over a fixed probe, and returns
what design.json :: evaluation.fold_acceptance compares. A CPU bfloat16 figure
is never a stand-in for it (the B-E driver runs it on the L40S).

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
        diagnostic_statistics: tuple[str, ...] = (), probes: list[list[int]] | None = None) -> dict:
    import torch
    from transformers import AutoModelForCausalLM

    from aadistill.initialization.adapters import build_registry
    from aadistill.initialization.transforms.projected_residual import (
        INDEPENDENT_BASES, STATISTICS, SublayerStatsCollector, independent_bases, materialize_projected,
        projection_basis, select_full_width)
    from aadistill.models.cost_model import parameter_counts
    from aadistill.initialization.adapters.projected_residual import fold_to_qwen3

    adapter = build_registry(["qwen3"]).get("qwen3")
    t0 = time.time()
    timings = {}
    if tiny:
        teacher, selection, items, arch = tiny_inputs()
        arch = {**arch, **(design.get("architecture") or {})}
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
    if arch.get("projection_sharing") == "independent":
        return _run_independent(design, out, arch=arch, teacher=teacher, w0=w0.to(device), items=items,
                                probes=probes or [], device=device, tiny=tiny, timings=timings,
                                adapter=adapter)
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
    # the structural fold check: CPU, float32, the first item's first 256 tokens
    probe = torch.tensor([items[0]["ids"][:256]], device="cpu")
    student_cpu = student.to("cpu")
    folded = fold_to_qwen3(student_cpu, dtype=torch.float32)
    with torch.no_grad():
        a = student_cpu(probe, use_cache=False).logits
        b = folded(probe, use_cache=False).logits
        fold_fp32 = float((a - b).abs().max())
        logit_scale = float(a.abs().max())
    del folded
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
        "structural_fold_check": {"device": "cpu", "dtype": "float32", "probe_tokens": int(probe.shape[1]),
                                  "max_abs_logit_diff": fold_fp32, "max_abs_logit": logit_scale,
                                  "relative": fold_fp32 / logit_scale if logit_scale else None,
                                  "_": "whether the fold is the same function; device numerics are fold_numerics'"},
        "timings": timings, "device": device,
    }
    (out / "materialization.json").write_text(json.dumps(record, indent=1) + "\n")
    return record


def _run_independent(design, out, *, arch, teacher, w0, items, probes, device, tiny, timings, adapter):
    """Steps 2-5 for four independent projections per layer (see the module docstring)."""
    import torch

    from aadistill.evaluation.function_error import function_error
    from aadistill.initialization.transforms.projected_residual import (
        INDEPENDENT_BASES, SublayerStatsCollector, independent_bases, materialize_projected)
    from aadistill.models.cost_model import parameter_counts

    roles = tuple(dict.fromkeys(INDEPENDENT_BASES.values()))
    t1 = time.time()
    col = SublayerStatsCollector(adapter, w0, roles)
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
        b, dg = independent_bases({r: st[r][i] for r in roles}, int(arch["internal_size"]))
        bases.append(b)
        diags.append({"layer": i, **{f"{site}_{INDEPENDENT_BASES[site]}_energy_captured":
                                     dg[site][f"{INDEPENDENT_BASES[site]}_energy_captured"] for site in b},
                      **{f"{site}_orthonormality_error": dg[site]["orthonormality_error"] for site in b}})
    timings["bases_s"] = round(time.time() - t2, 1)
    student = materialize_projected(adapter, w0, bases, dtype=torch.float32,
                                    projection_activation=arch.get("projection_activation"))
    student = student.to(device)
    t3 = time.time()

    def batches(seqs):
        for ids in seqs:
            x = torch.tensor([ids], device=device)
            yield x, torch.ones(1, x.shape[1] - 1, dtype=torch.bool, device=device)

    calib = [it["ids"] for it in items]
    error = {"in_sample_calibration": {
                 "vs_full_width_parent": function_error(w0, student, batches(calib)),
                 "vs_teacher": function_error(teacher.to(device), student, batches(calib))}}
    if probes:
        error["probe"] = {"vs_full_width_parent": function_error(w0, student, batches(probes)),
                          "vs_teacher": function_error(teacher, student, batches(probes)),
                          "prompts": len(probes)}
    timings["initialization_error_s"] = round(time.time() - t3, 1)
    del teacher
    out.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(out / "model")
    shards = sorted((out / "model").glob("*.safetensors"))
    keys = ("vocab_size", "hidden_size", "internal_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "head_dim", "tie_word_embeddings",
            "projection_sharing", "projection_activation")
    geom = {k: getattr(student.config, k) for k in keys}
    counted = sum(p.numel() for p in student.parameters())
    record = {
        "schema": "aadistill.stage1.projected_residual.materialization/v2",
        "tiny_smoke": tiny,
        "teacher": None if tiny else design["teacher"],
        "selection_content_sha256": None if tiny else design["selection"]["content_sha256"],
        "bases": {site: role for site, role in INDEPENDENT_BASES.items()},
        "calibration_positions": positions,
        "geometry": geom, "num_parameters": counted,
        "num_parameters_cost_model": parameter_counts(geom)["total"],
        "weights": [{"file": p.name, "sha256": file_sha256(p)} for p in shards],
        "config_sha256": file_sha256(out / "model" / "config.json"),
        "per_layer": diags,
        "initialization_error": error,
        "_initialization_error": ("float32 on the materializing device, every real position; the student is an "
                                  "approximation of W0 (projections) and of the teacher (selection + projections), "
                                  "never claimed equal to either"),
        "timings": timings, "device": device,
    }
    (out / "materialization.json").write_text(json.dumps(record, indent=1) + "\n")
    return record


def fold_numerics(student, folded, probes: list[list[int]], *, device: str, factor: float) -> dict:
    """The factored student against its fold, BOTH on `device`, float32 then bfloat16.

    `probes` are token-id prompts, run one at a time (no padding). Per prompt and
    over all positions: the fold's float32 difference, its bfloat16 difference,
    the factored model's own bfloat16-vs-float32 difference (the reference error
    design.json :: evaluation.fold_acceptance scales), and, as diagnostics only,
    the mean bfloat16 difference and argmax agreement. `criterion_met` is the
    acceptance: max |folded_bf16 - factored_bf16| <= factor x max |factored_bf16 -
    factored_fp32|. Both models are cast to bfloat16 IN PLACE at the end; the
    caller passes copies it can lose. The device is whatever the models are on;
    the caller records which card that was.
    """
    import torch

    with torch.no_grad():
        return _fold_numerics(student, folded, probes, device=device, factor=factor)


def _fold_numerics(student, folded, probes, *, device, factor):
    import torch

    student, folded = student.to(device).eval(), folded.to(device).eval()
    ref = []
    for ids in probes:
        x = torch.tensor([ids], device=device)
        a = student(x, use_cache=False).logits.float()
        b = folded(x, use_cache=False).logits.float()
        ref.append((a.cpu(), float((a - b).abs().max())))
        del a, b
    student.to(torch.bfloat16)
    folded.to(torch.bfloat16)
    rows = []
    for ids, (a, fp32_diff) in zip(probes, ref):
        x = torch.tensor([ids], device=device)
        a16 = student(x, use_cache=False).logits.float().cpu()
        b16 = folded(x, use_cache=False).logits.float().cpu()
        d = (a16 - b16).abs()
        rows.append({"tokens": len(ids), "fold_fp32_max_abs": fp32_diff, "fold_bf16_max_abs": float(d.max()),
                     "fold_bf16_mean_abs": float(d.mean()), "bf16_vs_fp32_max_abs": float((a16 - a).abs().max()),
                     "argmax_agree_bf16": float((a16.argmax(-1) == b16.argmax(-1)).float().mean()),
                     "argmax_agree_folded_bf16_vs_factored_fp32": float((b16.argmax(-1) == a.argmax(-1))
                                                                         .float().mean())})
    n = sum(r["tokens"] for r in rows)
    out = {"prompts": rows, "positions": n,
           "max_abs_logit_diff_fp32": max(r["fold_fp32_max_abs"] for r in rows),
           "max_abs_logit_diff_bf16": max(r["fold_bf16_max_abs"] for r in rows),
           "bf16_vs_fp32_same_model": max(r["bf16_vs_fp32_max_abs"] for r in rows),
           "diagnostics": {"mean_abs_logit_diff_bf16": sum(r["fold_bf16_mean_abs"] * r["tokens"] for r in rows) / n,
                           "argmax_agreement_bf16": sum(r["argmax_agree_bf16"] * r["tokens"] for r in rows) / n},
           "factor": factor}
    out["criterion_met"] = out["max_abs_logit_diff_bf16"] <= factor * out["bf16_vs_fp32_same_model"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--design", default="configs/stages/stage-1/projected_residual/design.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tiny", action="store_true", help="host smoke test on a random small teacher")
    ap.add_argument("--tiny-independent", action="store_true",
                    help="the host smoke test with four independent, gated projections per layer")
    ap.add_argument("--probe", default=None,
                    help="a JSON list of token-id prompts: the out-of-sample initialization-error probe")
    a = ap.parse_args(argv)
    tiny = a.tiny or a.tiny_independent
    design = json.loads((REPO / a.design).read_text()) if not tiny else (
        {"architecture": {"projection_sharing": "independent",
                          "projection_activation": "learnable_silu_residual"}} if a.tiny_independent else {})
    probes = json.loads(Path(a.probe).read_text()) if a.probe else None
    rec = run(design, Path(a.out), device=a.device, tiny=tiny, probes=probes)
    print(json.dumps({k: rec[k] for k in ("num_parameters", "num_parameters_cost_model", "structural_fold_check",
                                          "initialization_error", "timings") if k in rec}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
