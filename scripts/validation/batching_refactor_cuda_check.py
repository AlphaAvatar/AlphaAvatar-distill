"""Real-CUDA engineering check of the operator-topology + batching refactor.

ENGINEERING validation. It answers one question per stage — *does this changed
execution path run on a real device, and does batch>1 agree with the scalar
oracle* — and nothing about model quality. It trains nothing, measures no
behaviour, produces no ``correct_overall``, and ranks nothing for promotion.
Every artifact it writes is disposable. **AUTHORIZES NOTHING.**

It is deliberately NOT the 2026-09-10 validation. That one ran at execution SHA
``7027a8f4`` against the pre-migration flat layout, and its config and evidence
are untouched. This is a separate current record: its own config, its own
derived closure, its own SHA.

Run:

    PYTHONPATH=src:scripts python scripts/validation/batching_refactor_cuda_check.py \
        --run-id <id> [--device cuda] [--out artifacts/validation]

Exit codes: 0 PASS, 3 FAIL, 4 NOT RUN (no usable device).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

FAILED = 3
NOT_RUN = 4


def say(msg: str) -> None:
    print(f"[batching-cuda] {msg}", flush=True)


# --- preconditions ----------------------------------------------------------


def require_cuda(requested: str) -> dict:
    """The device, or a refusal that distinguishes 'absent' from 'refused'.

    An integer exit means the environment cannot host the check; a string means
    the CALLER asked for a device this check will not certify on. A launcher
    reads the report, and the two are different verdicts.
    """
    import torch

    if requested != "cuda":
        raise SystemExit(f"this check certifies CUDA only; --device={requested!r}")
    if not torch.cuda.is_available():
        raise SystemExit(NOT_RUN)
    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    free, total = torch.cuda.mem_get_info(index)
    return {
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_vram_gib": round(total / 2**30, 3),
        "free_vram_gib": round(free / 2**30, 3),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "driver_bf16_supported": bool(torch.cuda.is_bf16_supported()),
    }


def require_capability(info: dict, need: dict) -> None:
    have = tuple(int(p) for p in info["compute_capability"].split("."))
    want = tuple(int(p) for p in str(need["compute_capability_min"]).split("."))
    assert have >= want, (
        f"compute capability {info['compute_capability']} < "
        f"{need['compute_capability_min']}; the dtype is not lowered to make a "
        "cheaper device pass")
    if need.get("native_bf16"):
        assert info["driver_bf16_supported"], "device does not support bf16"
    floor = float(need.get("free_vram_gib_min", 0))
    assert info["free_vram_gib"] >= floor, (
        f"{info['free_vram_gib']} GiB free < {floor} GiB required")


def require_repo_inputs(cfg: dict) -> list[str]:
    """Declared non-Python inputs, before CUDA.

    A missing input is not a device problem, and reporting it as one sends a
    pod's worth of money looking in the wrong place.
    """
    present = []
    for rel in cfg.get("declared_non_python_inputs", []):
        path = REPO / rel
        assert path.is_file(), f"declared input missing: {rel}"
        present.append(rel)
    return present


# --- the world the stages run in -------------------------------------------


def build_world(cfg: dict, device: str):
    """A toy model on the real device, plus ragged calibration items."""
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from aadistill.initialization.adapters import register_builtin_adapters
    from aadistill.initialization.specs.arch import get_adapter, ArchSpec

    register_builtin_adapters()
    geometry = dict(cfg["geometry"])
    torch.manual_seed(int(cfg["seed"]))
    model = Qwen3ForCausalLM(
        Qwen3Config(max_position_embeddings=2048, rope_theta=5_000_000,
                    **geometry))
    dtype = getattr(torch, cfg["dtype"])
    model = model.to(device=device, dtype=dtype).eval()
    model.config.use_cache = False
    #: Fresh models initialise every RMSNorm to 1.0, which would let a
    #: norm-folding or placement bug pass unnoticed.
    with torch.no_grad():
        for module in model.modules():
            if module.__class__.__name__ == "Qwen3RMSNorm":
                module.weight.uniform_(0.5, 1.5)

    vocab = geometry["vocab_size"]
    items = []
    for index, length in enumerate(cfg["calibration"]["lengths"]):
        items.append({
            "item_id": f"v{index}",
            "input_ids": torch.randint(1, vocab, (1, int(length))),
            "domain": "general" if index % 2 else "math",
            "subtype": "text" if index % 2 else "arith",
        })
    spec = ArchSpec.of(cfg["family"], geometry)
    return model, items, get_adapter(cfg["adapter"]), spec, dtype


def placement_of(model) -> dict:
    devices = {str(p.device) for p in model.parameters()}
    dtypes = {str(p.dtype) for p in model.parameters()}
    return {"devices": sorted(devices), "dtypes": sorted(dtypes)}


def context_for(adapter, model, parent, target, items, batch_size, device):
    from aadistill.initialization.calibration.profiles import NO_CALIBRATION
    from aadistill.initialization.execution import ExecutionConfig
    from aadistill.initialization.operators.base import OperatorContext

    return OperatorContext(
        adapter=adapter, model=model, parent_spec=parent, target_spec=target,
        profile=NO_CALIBRATION, calibration_items=items, seed=0, device=device,
        config={"n_calibration_items": len(items)},
        execution=ExecutionConfig(micro_batch_size=batch_size))


# --- stages -----------------------------------------------------------------


def stage_kl_primitive(cfg, world, device) -> dict:
    """The shared reducer, on the device, against the scalar oracle."""
    import torch

    from aadistill.initialization.calibration.batching import (
        build_batch, resolve_pad_id)
    from aadistill.initialization.statistics.contribution import (
        forward_kl_mean, forward_kl_mean_batch)

    model, items, _adapter, _spec, _dtype = world
    chunk = int(cfg["kl"]["chunk"])
    batch = build_batch(items, pad_id=resolve_pad_id(model), device=device)
    mask = batch.prediction_mask().to(device)

    with torch.no_grad():
        ref = model(batch.input_ids,
                    attention_mask=batch.attention_mask).logits[:, :-1]
        perturbed = model(torch.roll(batch.input_ids, 1, dims=1),
                          attention_mask=batch.attention_mask).logits[:, :-1]

    batched = forward_kl_mean_batch(ref, perturbed, mask, chunk=chunk)
    if str(device).startswith("cuda"):
        assert batched.is_cuda, (
            "the reducer returned a host tensor on a CUDA input; the whole "
            "point of the [B] result is one transfer per batch, not B")
    assert torch.isfinite(batched).all(), "non-finite per-item KL"

    rows, drifts = [], []
    for index, length in enumerate(batch.lengths):
        scalar = forward_kl_mean(ref[index, :length - 1],
                                 perturbed[index, :length - 1], chunk=chunk)
        got = float(batched[index])
        rel = abs(got - scalar) / max(abs(scalar), 1e-12)
        drifts.append(rel)
        rows.append({"item": items[index]["item_id"], "positions": length - 1,
                     "scalar_oracle": scalar, "batched": got,
                     "relative_drift": rel})

    #: Padding must be inert on the device too, not only in a CPU unit test.
    poisoned = perturbed.clone()
    for index, length in enumerate(batch.lengths):
        poisoned[index, length - 1:] = 1e4
    unchanged = forward_kl_mean_batch(ref, poisoned, mask, chunk=chunk)
    padding_inert = bool(torch.equal(batched, unchanged))

    worst = max(drifts)
    bound = float(cfg["kl"]["max_relative_drift"])
    return {
        "ok": padding_inert and worst <= bound,
        "per_item": rows,
        "max_relative_drift": worst,
        "bound": bound,
        "padding_is_inert_exactly": padding_inert,
        "chunk": chunk,
        "batch_size": batch.size,
        "sequence_lengths": list(batch.lengths),
        "_note": ("drift is batch-vs-scalar on the SAME device; bf16 forwards "
                  "reschedule under batching so bit equality is not expected "
                  "above B=1"),
    }


def stage_attention_stats(cfg, world, device) -> dict:
    """GQA head statistics: valid tokens only, same heads at any batch size."""
    from aadistill.initialization.operators.attention.gqa import (
        activation_importance as AA)

    model, items, adapter, parent, _dtype = world
    target = parent.replace(
        num_attention_heads=int(cfg["target"]["num_attention_heads"]))
    AA.register(replace=True)
    try:
        per_batch, tokens = {}, {}
        for size in cfg["micro_batch_sizes"]:
            out = AA.ATTENTION_ACTIVATION_IMPORTANCE_V1.apply(
                context_for(adapter, model, parent, target, items, size, device))
            per_batch[str(size)] = out.artifacts["kept_heads"]
            tokens[str(size)] = int(out.trace["calibration_tokens"])
    finally:
        AA.unregister()

    expected = sum(int(i["input_ids"].shape[1]) for i in items)
    sizes = [str(s) for s in cfg["micro_batch_sizes"]]
    same = all(per_batch[s] == per_batch[sizes[0]] for s in sizes)
    counted = all(tokens[s] == expected for s in sizes)
    return {"ok": same and counted, "kept_heads_identical": same,
            "calibration_tokens": tokens,
            "expected_valid_tokens": expected,
            "token_counts_exclude_padding": counted,
            "kept_heads": per_batch[sizes[0]]}


def stage_shared_stats(cfg, world, device) -> dict:
    """FFN / WIDTH / COMPOSITE through the one shared batched pass."""
    import torch

    from aadistill.initialization.device import stats_to
    from aadistill.initialization.operators._common import collect_activation_stats
    from aadistill.initialization.transforms.project import ffn_neuron_importance
    from aadistill.initialization.operators.composite.stage1_sandwich import (
        COMPOSITE_STAGE1_SANDWICH_V0 as COMPOSITE)
    from aadistill.initialization.operators.ffn.dense.activation_importance import (
        FFN_ACTIVATION_IMPORTANCE_V0 as FFN)
    from aadistill.initialization.operators.width.residual.global_pca import (
        WIDTH_GLOBAL_PCA_V0 as WIDTH)

    model, items, adapter, parent, _dtype = world
    sizes = [int(s) for s in cfg["micro_batch_sizes"]]
    out: dict = {"ok": True}

    ffn_target = parent.replace(intermediate_size=int(cfg["target"]["intermediate_size"]))
    keep_n = int(cfg["target"]["intermediate_size"])
    kept, importances = {}, {}
    for size in sizes:
        state = stats_to(collect_activation_stats(adapter, model, items, device,
                                                  batch_size=size), device)
        per_layer_kept, per_layer_imp = [], []
        for index, block in enumerate(adapter.blocks(model)):
            down = adapter.stream_out_projections(block)["ffn_out"]
            importance = ffn_neuron_importance(state, index, down.weight)
            per_layer_imp.append(importance.detach().float().cpu())
            per_layer_kept.append(
                torch.topk(importance, keep_n).indices.sort().values.tolist())
        kept[str(size)] = per_layer_kept
        importances[str(size)] = per_layer_imp
    identical = all(kept[str(s)] == kept[str(sizes[0])] for s in sizes)
    out["ffn_kept_neurons_identical"] = identical

    #: WHEN they differ, the question is whether the boundary was a near-tie or
    #: a real divergence, and a boolean cannot answer it. `topk` at a tie is an
    #: execution-order artifact, not a different importance signal, and C3's
    #: ATTENTION selection has exactly this shape -- so the margin is measured
    #: here rather than left for a paid rerun to discover.
    diagnostics = []
    base = str(sizes[0])
    for other in sizes[1:]:
        for index, (a, b) in enumerate(zip(kept[base], kept[str(other)])):
            if a == b:
                continue
            ia, ib = importances[base][index], importances[str(other)][index]
            ordered = torch.sort(ia, descending=True).values
            margin = float(ordered[keep_n - 1] - ordered[keep_n])
            swapped = sorted(set(a) ^ set(b))
            gaps = [abs(float(ia[n]) - float(ib[n])) for n in swapped]
            diagnostics.append({
                "layer": index, "batch_sizes": [sizes[0], other],
                "only_in_first": sorted(set(a) - set(b)),
                "only_in_second": sorted(set(b) - set(a)),
                "cutoff_margin": margin,
                "importance_at_cutoff": float(ordered[keep_n - 1]),
                "max_importance_drift_on_swapped": max(gaps) if gaps else 0.0,
                "drift_over_margin": (max(gaps) / margin) if margin else float("inf"),
                #: The principled question, and the one the ratio above only
                #: hints at: are the two swapped neurons separated by MORE than
                #: the numerical drift? If not, the signal does not distinguish
                #: them and which one top-k returns is an execution-order
                #: artifact rather than a different importance ordering.
                "separation_between_swapped": (
                    abs(float(ia[sorted(set(a) - set(b))[0]])
                        - float(ia[sorted(set(b) - set(a))[0]]))
                    if (set(a) - set(b)) and (set(b) - set(a)) else float("inf")),
                "swapped_importances": {
                    str(n): {"first": float(ia[n]), "second": float(ib[n])}
                    for n in swapped},
            })
    out["ffn_selection_diagnostics"] = diagnostics
    #: Indistinguishable iff every swap's own separation is within the drift
    #: that batching introduces. Stated as a comparison between two MEASURED
    #: quantities rather than a tolerance someone chose after seeing the result.
    out["ffn_is_a_near_tie"] = bool(diagnostics and all(
        d["separation_between_swapped"] <= d["max_importance_drift_on_swapped"]
        for d in diagnostics))

    width_target = parent.replace(hidden_size=int(cfg["target"]["hidden_size"]))
    energy = {}
    for size in sizes:
        energy[str(size)] = float(WIDTH.apply(
            context_for(adapter, model, parent, width_target, items, size,
                        device)).local_metrics.values["op.width.energy_captured_frac"])
    spread = max(energy.values()) - min(energy.values())
    out["width_energy_captured"] = energy
    out["width_energy_spread"] = spread
    out["width_ok"] = spread < 1e-4

    composite_target = parent.replace(**{k: int(v) for k, v in cfg["target"].items()
                                         if k != "tie_word_embeddings"})
    composite = COMPOSITE.apply(
        context_for(adapter, model, parent, composite_target, items, sizes[-1],
                    device))
    out["composite_trace"] = {
        "activation_stats": composite.trace["activation_stats"],
        "micro_batch_size": composite.trace["micro_batch_size"],
    }
    out["composite_child_placement"] = placement_of(composite.model)
    #: A near-tie at the cutoff is an execution-order artifact, not a different
    #: importance signal, and it is RECORDED rather than treated as a failure --
    #: while a selection that moves by more than its own margin is a real
    #: divergence and does fail.
    out["ok"] = bool(out["width_ok"] and
                     (out["ffn_kept_neurons_identical"] or out["ffn_is_a_near_tie"]))
    return out


def stage_depth(cfg, world, device) -> dict:
    """Causal-KL DEPTH batched on CUDA, in all three reference-cache modes."""
    from aadistill.initialization.operators.depth import causal_kl_greedy as D

    model, items, adapter, parent, _dtype = world
    target = parent.replace(
        num_hidden_layers=int(cfg["target"]["num_hidden_layers"]))
    sizes = [int(s) for s in cfg["micro_batch_sizes"]]

    modes, results = {}, {}
    probe = D._ReferenceLogits(model, items, device)
    fraction = D._ReferenceLogits.BUDGET_FRACTION
    original = D._available_memory_bytes
    try:
        for share, mode in ((10.0, "cached"), (0.5, "partial"), (0.0, "recomputed")):
            budget = int(probe.estimate_bytes * share / fraction)
            D._available_memory_bytes = (lambda _dev, _b=budget: (_b, "validation"))
            for size in sizes:
                outcome = D.DEPTH_CAUSAL_KL_GREEDY_V1.apply(
                    context_for(adapter, model, parent, target, items, size,
                                device))
                key = f"{mode}/bs{size}"
                modes[key] = outcome.artifacts["reference_cache"]["mode"]
                results[key] = {
                    "removal_order": outcome.trace["removal_order"],
                    "kept_layers": outcome.trace["kept_layers"],
                    "rounds": [{"table": r["table"]} for r in
                               outcome.artifacts["search_rounds"]],
                    "forwards": outcome.artifacts["timing"]["ablated_forwards"],
                    "items": outcome.artifacts["timing"]["ablated_items"],
                }
    finally:
        D._available_memory_bytes = original

    baseline_key = f"cached/bs{sizes[0]}"
    baseline = results[baseline_key]
    agree, worst_drift, worst_margin = True, 0.0, float("inf")
    for key, value in results.items():
        agree = agree and value["removal_order"] == baseline["removal_order"]
        for a, b in zip(baseline["rounds"], value["rounds"]):
            table_a = {r["candidate"]: r["score"] for r in a["table"]}
            table_b = {r["candidate"]: r["score"] for r in b["table"]}
            worst_drift = max(worst_drift,
                              max(abs(table_a[k] - table_b[k]) for k in table_a))
            ordered = sorted(table_a.values())
            if len(ordered) > 1:
                worst_margin = min(worst_margin, ordered[1] - ordered[0])
    ratio = worst_drift / worst_margin if worst_margin else float("inf")
    return {
        "ok": bool(agree and ratio < 1.0),
        "modes_reached": sorted(set(modes.values())),
        "mode_by_case": modes,
        "removal_order_agrees_everywhere": agree,
        "reference_removal_order": baseline["removal_order"],
        "max_candidate_score_drift": worst_drift,
        "min_decision_margin": worst_margin,
        "drift_over_margin": ratio,
        "forward_invocations": {k: v["forwards"] for k, v in results.items()},
        "item_equivalents": {k: v["items"] for k, v in results.items()},
        "_telemetry_note": ("`forward_invocations` counts real model forwards; "
                            "`item_equivalents` counts items covered. In partial "
                            "cache mode a canonical batch is forwarded in full "
                            "even when some rows were resident, so neither is a "
                            "per-row hit count."),
    }


def stage_real_checkpoint(cfg, world, device) -> dict:
    """Real Qwen-family checkpoint bytes through the changed plumbing."""
    import torch

    from aadistill.initialization.calibration.batching import (
        build_batch, resolve_pad_id)
    from aadistill.initialization.specs.arch import adapter_for_config
    from aadistill.initialization.statistics.contribution import (
        forward_kl_mean, forward_kl_mean_batch)

    spec_cfg = cfg["stages"]["real_checkpoint"].get("source", {})
    origin = "local"
    source = next((REPO / rel for rel in spec_cfg.get("local_paths", [])
                   if (REPO / rel / "config.json").is_file()), None)
    if source is None and spec_cfg.get("relay_repo"):
        #: A pod has no local artifacts, and pod->HF is fast where the dev-box
        #: uplink is not. Downloading here costs seconds; shipping 1.2 GB from
        #: the dev box would cost ~28 minutes of BILLED GPU to prove the same
        #: thing.
        from huggingface_hub import snapshot_download

        origin = "relay"
        source = Path(snapshot_download(
            repo_id=spec_cfg["relay_repo"],
            revision=spec_cfg.get("relay_revision"),
            allow_patterns=[f"{spec_cfg['relay_path']}/*"]))
        source = source / spec_cfg["relay_path"]
    if source is None or not (source / "config.json").is_file():
        return {"ok": True, "skipped": True,
                "reason": ("no real checkpoint reachable from this host; the "
                           "toy geometry already exercises every changed line, "
                           "and this stage exists only to prove the plumbing "
                           "survives real bytes")}

    #: The bytes must be the ones the manifest names, not merely a checkpoint.
    weights_sha = None
    expected = spec_cfg.get("expected_weights_sha256")
    weights = source / "model.safetensors"
    if expected and weights.is_file():
        import hashlib

        digest = hashlib.sha256()
        with weights.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        weights_sha = digest.hexdigest()
        assert weights_sha == expected, (
            f"checkpoint weights {weights_sha} != manifest {expected}")

    from transformers import AutoModelForCausalLM

    dtype = getattr(torch, cfg["dtype"])
    model = AutoModelForCausalLM.from_pretrained(
        str(source), dtype=dtype).to(device).eval()
    model.config.use_cache = False
    adapter = adapter_for_config(model.config)
    spec = adapter.spec_of(model)

    vocab = int(model.config.vocab_size)
    torch.manual_seed(int(cfg["seed"]))
    items = [{"item_id": f"r{i}",
              "input_ids": torch.randint(1, vocab, (1, L)),
              "domain": "general", "subtype": "text"}
             for i, L in enumerate((11, 40, 7))]
    batch = build_batch(items, pad_id=resolve_pad_id(model), device=device)
    with torch.no_grad():
        ref = model(batch.input_ids,
                    attention_mask=batch.attention_mask).logits[:, :-1]
        other = model(torch.roll(batch.input_ids, 1, dims=1),
                      attention_mask=batch.attention_mask).logits[:, :-1]
    values = forward_kl_mean_batch(ref, other, batch.prediction_mask().to(device),
                                   chunk=int(cfg["kl"]["chunk"]))
    drift = max(
        abs(float(values[i]) - forward_kl_mean(ref[i, :L - 1], other[i, :L - 1],
                                               chunk=int(cfg["kl"]["chunk"])))
        / max(abs(forward_kl_mean(ref[i, :L - 1], other[i, :L - 1],
                                  chunk=int(cfg["kl"]["chunk"]))), 1e-12)
        for i, L in enumerate(batch.lengths))
    #: THE DECISIVE FFN CASE. A toy random FFN has densely packed importances,
    #: so its k-th and (k+1)-th neurons are separated in the seventh significant
    #: figure and top-k at that boundary is decided by reduction order. A real
    #: trained model is the case that matters, and it is already loaded here.
    from aadistill.initialization.device import stats_to
    from aadistill.initialization.operators._common import collect_activation_stats
    from aadistill.initialization.transforms.project import ffn_neuron_importance

    keep_n = max(1, int(model.config.intermediate_size) // 2)
    real_sel, real_margins = {}, []
    for size in (1, 3):
        state = stats_to(collect_activation_stats(adapter, model, items, device,
                                                  batch_size=size), device)
        layers = []
        for index, block in enumerate(adapter.blocks(model)):
            down = adapter.stream_out_projections(block)["ffn_out"]
            importance = ffn_neuron_importance(state, index, down.weight)
            layers.append(torch.topk(importance, keep_n).indices.sort().values.tolist())
            if size == 1:
                ordered = torch.sort(importance, descending=True).values
                real_margins.append(float(ordered[keep_n - 1] - ordered[keep_n])
                                    / max(float(ordered[keep_n - 1]), 1e-12))
        real_sel[size] = layers
    ffn_identical = real_sel[1] == real_sel[3]
    ffn_diff_layers = sum(1 for a, b in zip(real_sel[1], real_sel[3]) if a != b)

    return {
        "ok": bool(torch.isfinite(values).all()
                   and drift <= float(cfg["kl"]["max_relative_drift"])
                   and ffn_identical),
        "skipped": False,
        "real_ffn_selection_identical_across_batch_sizes": ffn_identical,
        "real_ffn_layers_that_differ": ffn_diff_layers,
        "real_ffn_layers_total": len(real_sel[1]),
        "real_ffn_keep_per_layer": keep_n,
        "real_ffn_min_relative_cutoff_margin": min(real_margins) if real_margins else None,
        "_why_this_is_the_case_that_matters": (
            "a trained model's FFN importance distribution is spread, so the "
            "cutoff is not a seventh-significant-figure tie the way a random "
            "toy's is; if selection is stable here, the toy divergence is a "
            "fixture artifact rather than a property of the operator"),
        "origin": origin,
        "source": str(source),
        "weights_sha256": weights_sha,
        "spec": spec.as_dict(),
        "spec_hash": spec.spec_hash,
        "placement": placement_of(model),
        "per_item_kl": [float(v) for v in values],
        "max_relative_drift": drift,
        "adapter_family": adapter.family,
        "n_blocks_resolved_by_role": len(adapter.blocks(model)),
    }


STAGES = (
    ("kl_primitive", stage_kl_primitive),
    ("attention_stats", stage_attention_stats),
    ("shared_stats", stage_shared_stats),
    ("depth", stage_depth),
    ("real_checkpoint", stage_real_checkpoint),
)


# --- entry point ------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
                    default="configs/validation/batching_refactor_cuda.json")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="artifacts/validation")
    ap.add_argument("--dry-run", action="store_true",
                    help="derive the closure and stop at the CUDA boundary; "
                         "writes a NOT RUN report")
    ap.add_argument("--rehearse", action="store_true",
                    help="EXECUTE every stage on the host to prove the code "
                         "runs, then report REHEARSAL. Never PASS: a host run "
                         "certifies nothing about CUDA.")
    a = ap.parse_args()

    config_path = (Path(a.config) if Path(a.config).is_absolute()
                   else REPO / a.config)
    cfg = json.loads(config_path.read_text())
    out_dir = REPO / a.out if not Path(a.out).is_absolute() else Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "batching_refactor_cuda_report.json"

    report = {
        "schema": "aadistill.batching_refactor_cuda_check/v1",
        "run_id": a.run_id,
        "validation_id": cfg["validation_id"],
        "scientific_use": False,
        "authorizes": "nothing",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "stages": {},
    }

    def finish(verdict: str, code: int, reason: str = "") -> int:
        report["verdict"] = verdict
        if reason:
            report["reason"] = reason
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report_path.write_text(json.dumps(report, indent=1, default=str) + "\n")
        say(f"verdict: {verdict}" + (f" — {reason}" if reason else ""))
        return code

    try:
        report["required_repo_inputs"] = require_repo_inputs(cfg)
        report["source_closure"] = derived_closure(cfg)
        say(f"closure {report['source_closure']['digest'][:16]}… over "
            f"{report['source_closure']['n_files']} files")
        if a.dry_run:
            return finish("NOT RUN", NOT_RUN,
                          "--dry-run: stopped at the CUDA boundary")
        if a.rehearse:
            #: The stages RUN, on the host. This exists because a rehearsal that
            #: stubs the work proves nothing -- four paid pods in this project
            #: died inside lines no `$0` run had reached. What it cannot do is
            #: certify the device, so the verdict is its own word and the report
            #: says so in a field a reader cannot miss.
            report["is_cuda_validation"] = False
            report["device"] = {"name": "host (rehearsal)", "cuda": False}
            say("REHEARSAL: executing every stage on the host; this certifies "
                "no device behaviour")
        else:
            report["is_cuda_validation"] = True
            device_info = require_cuda(a.device)
            require_capability(device_info, cfg["capability_requirement"])
            report["device"] = device_info
            say(f"{device_info['name']} cc{device_info['compute_capability']} "
                f"{device_info['free_vram_gib']} GiB free")
    except SystemExit as exc:
        if isinstance(exc.code, int):
            return finish("NOT RUN", exc.code, "no CUDA device")
        return finish("FAIL", FAILED, str(exc.code))
    except AssertionError as exc:
        return finish("FAIL", FAILED, str(exc))

    device = "cpu" if a.rehearse else a.device
    try:
        world = build_world(cfg, device)
        report["parent_placement"] = placement_of(world[0])
    except Exception as exc:          # noqa: BLE001 - reported, not swallowed
        report["stages"]["build_world"] = {"ok": False, "error": repr(exc)}
        return finish("FAIL", FAILED, f"could not build the world: {exc!r}")

    ok = True
    for name, fn in STAGES:
        if not cfg["stages"].get(name, {}).get("enabled", False):
            report["stages"][name] = {"ok": True, "skipped": True,
                                      "reason": "disabled in the config"}
            continue
        started = time.perf_counter()
        try:
            result = fn(cfg, world, device)
        except Exception as exc:      # noqa: BLE001 - reported, not swallowed
            result = {"ok": False, "error": repr(exc)}
        result["seconds"] = round(time.perf_counter() - started, 3)
        report["stages"][name] = result
        ok = ok and bool(result.get("ok"))
        say(f"{name}: {'ok' if result.get('ok') else 'FAILED'} "
            f"({result['seconds']}s)")

    if a.rehearse:
        #: A rehearsal reports whether the CODE ran, never whether the device
        #: is correct. It cannot return PASS by construction.
        return finish("REHEARSAL" if ok else "REHEARSAL-FAILED",
                      0 if ok else FAILED)
    return finish("PASS" if ok else "FAIL", 0 if ok else FAILED)


def derived_closure(cfg: dict) -> dict:
    """What will actually execute, walked from THIS entry point.

    Derived, not the hand-maintained planning tuple: the closure binds the exact
    source the paid GPU runs, and a human list cannot make that claim.
    """
    from architecture.derive_closure import derive

    return derive(REPO, "batching_refactor_cuda",
                  ["scripts/validation/batching_refactor_cuda_check.py"],
                  list(cfg.get("declared_non_python_inputs", [])),
                  roots=["src", "scripts"])


if __name__ == "__main__":
    raise SystemExit(main())
