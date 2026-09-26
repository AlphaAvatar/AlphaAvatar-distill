"""Root-cause diagnostic: is a batched forward the same computation as a solo one?

The a4 validation reported that FFN top-k selection moves between micro-batch
sizes on a real model in bf16, and a hand-assembled finding attributed it to the
dtype. That attribution was not earned: the decisive numbers came from ad-hoc
scripts rather than the committed executable, and the environment was the pod
image's own torch with unpinned `pip install`. This file exists so that every
number a conclusion rests on is emitted by the executable that produced it.

The question, stated so it can fail:

    same checkpoint, same tokens, same positions, same dtype, same backend --
    does evaluating an item ALONE and evaluating it BESIDE others give the same
    scientific decision, and if not, which operation first stops agreeing?

Stages, in EXECUTION order, which is also the order the answers depend on each
other (and the order `STAGES` below declares -- a docstring that disagrees with
the run is how a report gets misread):

    -  environment          what actually ran, as data. Not a stage; recorded first
    1  repeatability        is each shape internally deterministic at all?
    2  case matrix A..F     mask presence / batch shape / neighbour CONTENT /
                            raggedness / padding, one change at a time
    3  first divergence     layer by layer, through adapter roles
    4  gemm isolation       does a bare GEMM already disagree at this dtype?
    5  reduction control    does allow_bf16_reduced_precision_reduction fix it?
    6  length sweep         padded width, batch size and true length, each
                            varied with the other two held fixed
    7  stats decomposition  forward drift vs reduction-order drift, separately
    8  ffn selection        what moved, against the cutoff margins it turned on
    9  causal KL            solo vs batched, the quantity a DEPTH search reads
   10  backend matrix       eager, and each SDPA kernel the build exposes
   11  fp32 control         the dtype hypothesis. A DIAGNOSTIC, never a proposal

The verdict is not written beside the numbers: `derive_conclusion` computes it
from the stage outputs, and `tests/validation/test_batch_invariance_conclusion`
tables that function. Nothing here changes an operator. It measures.

    PYTHONPATH=src:scripts python scripts/validation/batch_invariance_diagnostic.py \
        --run-id <id> [--device cuda] [--dtype bfloat16] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

NOT_RUN = 4
FAILED = 3


def say(msg: str) -> None:
    print(f"[batch-invariance] {msg}", flush=True)


# --- metrics ----------------------------------------------------------------
#
# One definition, used everywhere, chosen BEFORE any result was seen. A single
# pointwise relative error against a near-zero logit can read as enormous while
# the real disagreement is negligible, which is exactly how the 4.88e-02 headline
# came to stand alone. So every comparison reports a spread of statistics and the
# formulas live here rather than in prose.


def compare(a, b) -> dict:
    """Elementwise agreement between two tensors of identical shape."""
    import torch

    assert a.shape == b.shape, (tuple(a.shape), tuple(b.shape))
    x, y = a.detach().float().flatten(), b.detach().float().flatten()
    d = (x - y).abs()
    nx = x.norm()
    return {
        "n": int(d.numel()),
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "rms": float((d * d).mean().sqrt()),
        "p50_abs": float(d.quantile(0.50)) if d.numel() < 16_000_000 else None,
        "p95_abs": float(d.quantile(0.95)) if d.numel() < 16_000_000 else None,
        "p99_abs": float(d.quantile(0.99)) if d.numel() < 16_000_000 else None,
        # ||x_b - x_s||_2 / max(||x_s||_2, eps) -- scale-free and not hostage to
        # one near-zero element the way a pointwise ratio is.
        "rel_l2": float((x - y).norm() / torch.clamp(nx, min=1e-12)),
        "cosine": float(torch.nn.functional.cosine_similarity(
            x.unsqueeze(0), y.unsqueeze(0)).squeeze()),
        "bitwise_identical": bool(torch.equal(a, b)),
        "_rel_l2_formula": "||b-a||_2 / max(||a||_2, 1e-12)",
    }


def compare_logits(a, b) -> dict:
    """Agreement plus the decision-level quantities logits actually feed."""
    import torch
    import torch.nn.functional as F

    out = compare(a, b)
    pa, pb = F.log_softmax(a.float(), -1), F.log_softmax(b.float(), -1)
    out["forward_kl_per_position_mean"] = float(
        (pa.exp() * (pa - pb)).sum(-1).mean())
    out["argmax_agreement"] = float((a.argmax(-1) == b.argmax(-1)).float().mean())
    for k in (5, 20):
        ta = a.topk(k, dim=-1).indices.sort(dim=-1).values
        tb = b.topk(k, dim=-1).indices.sort(dim=-1).values
        out[f"top{k}_set_agreement"] = float((ta == tb).all(-1).float().mean())
    return out


# --- environment ------------------------------------------------------------


def executable_identity() -> dict:
    """WHICH executable produced this report.

    P4 asks for the evaluation code version and the reports did not carry it.
    It became load-bearing the moment one investigation collected reports from
    two commits: without this a reader compares numbers from different code and
    has no way to notice.
    """
    import hashlib
    import subprocess

    src = Path(__file__).resolve()
    out = {"file": src.name,
           "sha256": hashlib.sha256(src.read_bytes()).hexdigest()}
    try:
        r = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=20)
        out["git_head"] = r.stdout.strip() or None
        d = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain",
                            "--", str(src)], capture_output=True, text=True,
                           timeout=20)
        #: A dirty file means the sha256 above is the truth and the commit is
        #: not. Said plainly rather than left for a reader to infer.
        out["this_file_is_clean_at_git_head"] = not d.stdout.strip()
    except Exception:                                       # noqa: BLE001
        out["git_head"] = None
        out["this_file_is_clean_at_git_head"] = None
    return out


def environment_identity(device: str) -> dict:
    """What actually ran. Recorded as data, never inferred from an image name."""
    import torch

    env: dict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "torch_cudnn": (torch.backends.cudnn.version()
                        if torch.backends.cudnn.is_available() else None),
        "container_image": _read_env_file("AAD_CONTAINER_IMAGE"),
        "wheelhouse_source": _read_env_file("AAD_WHEELHOUSE_SOURCE"),
        "requirements_file": _read_env_file("AAD_REQUIREMENTS"),
    }
    for name in ("transformers", "tokenizers", "safetensors", "huggingface_hub",
                 "numpy", "triton"):
        try:
            env[name] = __import__(name).__version__
        except Exception:                                   # noqa: BLE001
            env[name] = None
    if device.startswith("cuda") and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        env.update({
            "gpu": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "total_vram_gib": round(props.total_memory / 2**30, 3),
            "driver_bf16_supported": bool(torch.cuda.is_bf16_supported()),
            "nvidia_driver": _nvidia_driver(),
        })
    # Numerical policy knobs, read from the running build rather than assumed.
    env["numerics"] = _numeric_policy()
    return env


def _read_env_file(key: str):
    import os
    return os.environ.get(key)


def _nvidia_driver():
    import subprocess
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                               "--format=csv,noheader"],
                              capture_output=True, text=True,
                              timeout=20).stdout.strip() or None
    except Exception:                                       # noqa: BLE001
        return None


def _numeric_policy() -> dict:
    """The knobs, and whether this build even has them.

    Read rather than assumed: `allow_bf16_reduced_precision_reduction` is not
    guaranteed to exist, and a diagnostic that silently assumes it would report
    a control it never applied.
    """
    import torch

    policy = {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    for attr in ("allow_bf16_reduced_precision_reduction",
                 "allow_bf16_reduced_precision_reduction_split_k",
                 "allow_fp16_reduced_precision_reduction",
                 "allow_fp16_reduced_precision_reduction_split_k",
                 "allow_tf32"):
        #: `getattr(..., default)` does not work here: this object's
        #: `__getattr__` raises AttributeError for an unknown name, which the
        #: default swallows -- but on torch 2.9.1 the `_split_k` names do not
        #: exist at all, and reporting them as a default would misdescribe the
        #: runtime. Caught explicitly so "absent" is a recorded fact.
        try:
            policy[f"cuda.matmul.{attr}"] = getattr(
                torch.backends.cuda.matmul, attr)
        except AttributeError:
            policy[f"cuda.matmul.{attr}"] = "<absent in this build>"
    policy["_split_k_is_readable"] = not isinstance(
        policy.get("cuda.matmul.allow_bf16_reduced_precision_reduction_split_k"),
        str)
    policy["cudnn.allow_tf32"] = getattr(torch.backends.cudnn, "allow_tf32",
                                         "<absent in this build>")
    return policy


# --- the bf16 reduction policy, read and written HONESTLY --------------------
#
# `allow_bf16_reduced_precision_reduction = False` does NOT disable split-K.
# torch's own parser is explicit about it:
#
#     if isinstance(value, bool):
#         return value, True          # <- allow_splitk stays True
#
# and the tuple form is the only way to reach the second flag:
#
#     allow_splitk = _ensure_bool(value[1], "allow_splitk")
#
# The previous round's `reduction_control` set the bool, described itself as
# turning off "bf16 split-k reduction", and reported that disabling it did not
# remove the divergence. The first half was wrong, and the second half was
# measured at the LOGIT level while the per-projection table underneath showed
# `attn_out` and `ffn_out` going bit-exact. Both are corrected here.
#
# Verified at $0 against the pytorch sources for the two runtimes in play:
#   v2.11.0  tuple SUPPORTED, `..._split_k` readable   <- the science runtime
#   v2.9.1   bool only, no split_k attribute           <- engineering runtime

BF16_REDUCED = "allow_bf16_reduced_precision_reduction"
BF16_SPLIT_K = "allow_bf16_reduced_precision_reduction_split_k"

#: The three states this investigation must keep apart. Named, so that a report
#: can never again call two different things "disabled".
CONTROL_A = "default"                                  # cuBLAS,   True/True
CONTROL_B = "reduced_precision_off_splitk_on"          # cuBLAS,   False/True
#: The backend switch ALONE. `allow_splitk=False` requires cuBLASLt, so the
#: intervention changes the BLAS library as well as the split-K flag -- and
#: without this control an improvement could not be attributed to either one.
CONTROL_D = "cublaslt_defaults"                        # cuBLASLt, True/True
CONTROL_C = "reduced_precision_off_splitk_off"         # cuBLASLt, False/False


def read_bf16_policy() -> dict:
    """What the runtime says the two flags actually are, right now."""
    import torch

    out = {}
    for name, attr in (("allow_reduced_precision", BF16_REDUCED),
                       ("allow_splitk", BF16_SPLIT_K)):
        try:
            out[name] = bool(getattr(torch.backends.cuda.matmul, attr))
            out[f"{name}_readable"] = True
        except AttributeError:
            out[name] = None
            out[f"{name}_readable"] = False
    return out


def _restore(before: dict, before_blas: str) -> None:
    """Put the policy and the BLAS backend back where the run found them.

    Both, not just the policy: a stage that selected cuBLASLt and restored only
    the reduction flags would leave every later stage on a different BLAS
    library than the one whose numbers it is being compared against.
    """
    set_bf16_policy(bool(before["allow_reduced_precision"]),
                    before["allow_splitk"] if before["allow_splitk_readable"]
                    else None,
                    before_blas if "<" not in before_blas else None)


def read_blas_backend():
    """Which BLAS library CUDA matmuls will use. cuBLAS unless told otherwise."""
    import torch

    try:
        return str(torch.backends.cuda.preferred_blas_library())
    except Exception as exc:                                # noqa: BLE001
        return f"<unreadable: {type(exc).__name__}: {exc}>"


def set_blas_backend(name: str | None) -> dict:
    """Select cuBLAS or cuBLASLt. Needed because `allow_splitk=False` requires Lt.

    This is NOT a free plumbing detail: switching the BLAS library is itself a
    numerical change, so an arm that changes it is changing two things and
    needs its own control. See `CONTROL_D`.
    """
    import torch

    out = {"requested": name, "error": None}
    if name is not None:
        try:
            torch.backends.cuda.preferred_blas_library(name)
        except Exception as exc:                            # noqa: BLE001
            out["error"] = f"{type(exc).__name__}: {exc}"
    out["readback"] = read_blas_backend()
    #: Did it STICK? `preferred_blas_library` can accept a name and leave the
    #: backend where it was -- on a CPU-only build it does exactly that. A
    #: request that silently no-ops would make control C look like a cuBLASLt
    #: arm while it ran on cuBLAS, which is the one mistake this whole round
    #: exists to stop repeating.
    out["honoured"] = (
        name is None
        or (isinstance(out["readback"], str)
            and name.lower() in out["readback"].lower()))
    return out


def _policy_runs_a_gemm() -> dict:
    """Does a real bf16 GEMM actually EXECUTE under the policy now in force?

    THE fix for the defect that cost attempt d4. `setattr` accepted
    `(False, False)` and returned; torch defers the validation, and the refusal
    -- "allow_splitk=False requires the cuBLASLt backend" -- arrived from
    inside `F.linear` on the first real matmul, killing four stages that had
    been told the policy was supported.

    So `supported` now means "a GEMM ran", not "the setter returned". A
    64x256 @ 256x64 bf16 matmul costs nothing and is the only honest test.
    """
    import torch

    if not torch.cuda.is_available():
        return {"probed": False,
                "reason": "no CUDA; this policy is a cuBLAS setting and is "
                          "inert here, so a probe would pass vacuously"}
    try:
        a = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(256, 64, device="cuda", dtype=torch.bfloat16)
        (a @ b).sum().item()          # .item() forces the launch to complete
        return {"probed": True, "ran": True, "error": None}
    except Exception as exc:                                # noqa: BLE001
        return {"probed": True, "ran": False,
                "error": f"{type(exc).__name__}: {exc}"}


def set_bf16_policy(reduced: bool, splitk: bool | None,
                    blas: str | None = None) -> dict:
    """Request a policy, read it back, and PROVE a GEMM runs under it.

    `splitk=None` means the boolean form, which is control B and leaves split-K
    on. A build that cannot express `splitk=False` -- or a backend that will
    not honour it -- is recorded UNSUPPORTED rather than silently given the
    boolean, because that substitution is the whole defect being corrected.
    """
    import torch

    matmul = torch.backends.cuda.matmul
    requested = {"allow_reduced_precision": reduced, "allow_splitk": splitk,
                 "blas_library": blas}
    result = {"requested": requested, "supported": None, "error": None,
              "blas": set_blas_backend(blas)}
    try:
        if splitk is None:
            setattr(matmul, BF16_REDUCED, bool(reduced))
            result["form"] = "bool"
        else:
            setattr(matmul, BF16_REDUCED, (bool(reduced), bool(splitk)))
            result["form"] = "tuple"
        result["setter_accepted"] = True
    except (TypeError, AttributeError, RuntimeError) as exc:
        result["setter_accepted"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["form"] = "UNSUPPORTED"

    result["readback"] = read_bf16_policy()
    #: The setter returning is NOT support. Only a GEMM running is.
    probe = _policy_runs_a_gemm() if result["setter_accepted"] else {
        "probed": False, "reason": "the setter itself refused"}
    result["gemm_probe"] = probe
    result["supported"] = bool(
        result["setter_accepted"] and probe.get("ran", not probe.get("probed")))
    if not result["supported"] and not result["error"]:
        result["error"] = probe.get("error")

    rb = result["readback"]
    result["honoured"] = (
        result["supported"]
        and rb["allow_reduced_precision"] == bool(reduced)
        and (splitk is None or rb["allow_splitk"] == bool(splitk)))
    return result


# --- the world --------------------------------------------------------------


def load_model(cfg: dict, device: str, dtype_name: str, attn: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM

    from aadistill.initialization.adapters import register_builtin_adapters
    from aadistill.initialization.specs.arch import adapter_for_config

    register_builtin_adapters()
    kwargs = {"dtype": getattr(torch, dtype_name)}
    if attn:
        kwargs["attn_implementation"] = attn
    model = AutoModelForCausalLM.from_pretrained(cfg["checkpoint"], **kwargs)
    model = model.to(device).eval()
    model.config.use_cache = False
    return model, adapter_for_config(model.config)


def resolve_checkpoint(cfg: dict) -> str:
    """The PARENT, because the reported divergence was about the parent.

    The a4 finding was FFN neuron selection over the parent's own activation
    statistics, so the diagnostic reads the same weights. It is a public hub
    model, which is why nothing here goes through the private relay: the pod
    downloads it directly and the dev-box uplink never enters the picture.
    """
    source = cfg["source"]
    for rel in source.get("local_paths", ()):
        if (REPO / rel / "config.json").is_file():
            return str(REPO / rel)
    from huggingface_hub import snapshot_download

    if source.get("relay_repo"):
        root = snapshot_download(source["relay_repo"],
                                 allow_patterns=[f"{source['relay_path']}/*"])
        return str(Path(root) / source["relay_path"])
    return snapshot_download(
        source["hub_id"],
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])


def focal_and_neighbours(cfg: dict, vocab: int):
    """The focal item, byte-identical in every case, plus its neighbours."""
    import torch

    g = torch.Generator().manual_seed(int(cfg["seed"]))
    focal = torch.randint(1, vocab, (1, int(cfg["focal_length"])), generator=g)
    others = [torch.randint(1, vocab, (1, int(n)), generator=g)
              for n in cfg["neighbour_lengths"]]
    return focal, others


def attention_implementation_of(model) -> dict:
    cfg = model.config
    return {
        "config._attn_implementation": getattr(cfg, "_attn_implementation", None),
        "config.attn_implementation": getattr(cfg, "attn_implementation", None),
        "first_block_attn_class": type(
            model.model.layers[0].self_attn).__name__,
    }


# --- 1. is each execution shape internally deterministic? -------------------


def stage_repeatability(cfg, ctx) -> dict:
    """Before any cross-shape claim: does one shape even repeat itself?

    A cross-shape difference means nothing if the SAME shape gives different
    answers twice. This separates shape-dependent deterministic numerics from
    run-to-run nondeterminism, and it runs first for that reason.
    """
    import torch

    model, focal, others, device = ctx["model"], ctx["focal"], ctx["others"], ctx["device"]
    out = {}
    with torch.no_grad():
        solo = [model(focal.to(device)).logits.clone() for _ in range(3)]
        batch_ids, mask, _ = _ragged(focal, others, ctx)
        batched = [model(batch_ids, attention_mask=mask).logits[:1].clone()
                   for _ in range(3)]
    for name, runs in (("solo", solo), ("batch", batched)):
        pairs = [compare(runs[0], runs[i]) for i in (1, 2)]
        identical = all(p["bitwise_identical"] for p in pairs)
        out[name] = {
            "repeats": len(runs),
            "bitwise_identical_across_repeats": identical,
            "max_abs_between_repeats": max(p["max_abs"] for p in pairs),
            "classification": ("bit-identical" if identical else
                               "stable-not-bit-identical"
                               if max(p["max_abs"] for p in pairs) < 1e-3
                               else "NONDETERMINISTIC"),
        }
    out["both_shapes_internally_deterministic"] = all(
        out[s]["bitwise_identical_across_repeats"] for s in ("solo", "batch"))
    return out


def _ragged(focal, others, ctx):
    """The focal item first, then ragged right-padded neighbours."""
    from aadistill.initialization.calibration.batching import build_batch, resolve_pad_id

    items = [{"item_id": "focal", "input_ids": focal}]
    items += [{"item_id": f"n{i}", "input_ids": t} for i, t in enumerate(others)]
    b = build_batch(items, pad_id=resolve_pad_id(ctx["model"]), device=ctx["device"])
    return b.input_ids, b.attention_mask, b


# --- 2. the case matrix -----------------------------------------------------


def stage_case_matrix(cfg, ctx) -> dict:
    """A..F, each changing ONE thing, so a difference can be attributed."""
    import torch

    model, focal, others = ctx["model"], ctx["focal"], ctx["others"]
    device, L = ctx["device"], focal.shape[1]
    ids = focal.to(device)

    def focal_rows(logits):
        #: Only the focal item's REAL positions, in every case. A padded
        #: position never enters a metric, and a pooled batch tensor is never
        #: compared against a solo one.
        return logits[0, :L]

    cases = {}
    with torch.no_grad():
        cases["A_solo_no_mask"] = focal_rows(model(ids).logits)
        cases["B_solo_ones_mask"] = focal_rows(
            model(ids, attention_mask=torch.ones_like(ids)).logits)

        # C: equal-length batch of identical copies -- pure batch dimension.
        dup = ids.repeat(int(cfg["equal_batch"]), 1)
        cases["C_equal_batch_duplicates"] = focal_rows(
            model(dup, attention_mask=torch.ones_like(dup)).logits)

        # D: equal length, DIFFERENT neighbour tokens. The focal row cannot
        # mathematically depend on them; if it moves, that is a cross-row effect.
        g = torch.Generator().manual_seed(int(cfg["seed"]) + 1)
        rows = [ids] + [torch.randint(1, int(model.config.vocab_size), (1, L),
                                      generator=g).to(device)
                        for _ in range(int(cfg["equal_batch"]) - 1)]
        diff = torch.cat(rows, 0)
        cases["D_equal_batch_diff_neighbours"] = focal_rows(
            model(diff, attention_mask=torch.ones_like(diff)).logits)

        # E: the intended production path -- ragged, right-padded, masked.
        bi, bm, _ = _ragged(focal, others, ctx)
        cases["E_ragged_padded_masked"] = focal_rows(
            model(bi, attention_mask=bm).logits)

        # F: the same ragged batch WITHOUT a mask. Diagnostic only: it isolates
        # padding shape from the mask path and is not a candidate implementation.
        cases["F_ragged_padded_no_mask"] = focal_rows(model(bi).logits)

    names = list(cases)
    widest = max([L] + [int(t.shape[1]) for t in others])
    out = {"cases": names, "focal_positions": int(L),
           "batch_padded_width": widest,
           #: When the focal item is the LONGEST in its group, cases C, E and F
           #: put no padding on the focal ROW, and a zero in `C_vs_E` then means
           #: "this group did not pad the focal item", not "the ragged path is
           #: exact". The frozen mixture's first item is 629 tokens and its next
           #: three are shorter, so this is the real production case, not a
           #: hypothetical -- read `C_vs_E` together with the length sweep.
           "focal_row_is_padded_in_this_batch": widest > int(L),
           "comparisons": {}, "_only_focal_real_positions_compared": True}
    for lhs, rhs in (("A_solo_no_mask", "B_solo_ones_mask"),
                     ("B_solo_ones_mask", "C_equal_batch_duplicates"),
                     ("C_equal_batch_duplicates", "D_equal_batch_diff_neighbours"),
                     ("C_equal_batch_duplicates", "E_ragged_padded_masked"),
                     ("E_ragged_padded_masked", "F_ragged_padded_no_mask"),
                     ("A_solo_no_mask", "E_ragged_padded_masked")):
        out["comparisons"][f"{lhs[0]}_vs_{rhs[0]}"] = compare_logits(
            cases[lhs], cases[rhs])
    out["_reading"] = {
        "A_vs_B": "mask-presence path",
        "B_vs_C": "pure batch-shape/kernel effect",
        "C_vs_D": "cross-row dependence (must be nil)",
        "C_vs_E": "ragged/padding path",
        "E_vs_F": "mask vs padding shape, for a causal decoder",
        "A_vs_F": "the end-to-end solo-vs-production difference",
    }
    ctx["case_tensors"] = cases
    return out


# --- 3. where does it FIRST diverge? ---------------------------------------


def stage_first_divergence(cfg, ctx) -> dict:
    """Layer by layer, through adapter roles, until the error leaves the noise.

    The two passes are compared INCREMENTALLY: the solo pass stores each tap, the
    batched pass compares against the stored tensor and frees both. Retaining
    both passes in full is what an obvious implementation does, and on a 36-layer
    parent the gate/up taps alone are several GiB of it -- the first rehearsal of
    this stage was OOM-killed at 15.8 GiB doing exactly that.
    """
    import torch

    model, adapter = ctx["model"], ctx["adapter"]
    focal, others, device = ctx["focal"], ctx["others"], ctx["device"]
    L = focal.shape[1]
    stored: dict[str, Any] = {}
    measured: dict[str, dict] = {}
    order: list[str] = []
    phase = ["record"]

    def tap(name):
        def hook(_m, _in, out):
            t = out[0] if isinstance(out, tuple) else out
            #: Kept in the model's own dtype and on its own device: widening
            #: here would double the bytes without changing a single comparison,
            #: since `compare` upcasts what it is given.
            row = t.detach()[0, :L].clone()
            if phase[0] == "record":
                stored[name] = row
                order.append(name)
            else:
                first = stored.pop(name, None)
                if first is not None:
                    measured[name] = compare(first, row)
                del first
        return hook

    handles = []
    handles.append(adapter.embedding(model).register_forward_hook(tap("embedding")))
    for index, block in enumerate(adapter.blocks(model)):
        for role, (linear, _norm) in adapter.stream_in_projections(block).items():
            handles.append(linear.register_forward_hook(
                tap(f"L{index:02d}.in.{role}")))
        for role, linear in adapter.stream_out_projections(block).items():
            handles.append(linear.register_forward_hook(
                tap(f"L{index:02d}.out.{role}")))
        handles.append(adapter.attn_norm(block).register_forward_hook(
            tap(f"L{index:02d}.attn_norm")))
        handles.append(adapter.ffn_norm(block).register_forward_hook(
            tap(f"L{index:02d}.ffn_norm")))
        handles.append(block.register_forward_hook(tap(f"L{index:02d}.block_out")))
    handles.append(adapter.final_norm(model).register_forward_hook(tap("final_norm")))

    try:
        with torch.no_grad():
            model(focal.to(device))                       # pass 0: solo, records
            phase[0] = "compare"
            bi, bm, _ = _ragged(focal, others, ctx)
            model(bi, attention_mask=bm)                  # pass 1: batched
    finally:
        for h in handles:
            h.remove()
        stored.clear()

    rows, first = [], None
    #: Execution order, not alphabetical: "first" has to mean first COMPUTED.
    for name in order:
        m = measured.get(name)
        if m is None:
            continue
        rows.append({"tap": name, "max_abs": m["max_abs"], "rel_l2": m["rel_l2"],
                     "cosine": m["cosine"],
                     "bitwise_identical": m["bitwise_identical"]})
        if first is None and not m["bitwise_identical"]:
            first = {"tap": name, **m}
    cases = ctx.get("case_tensors")
    logits = (compare_logits(cases["A_solo_no_mask"],
                             cases["E_ragged_padded_masked"])
              if cases else _solo_vs_batched_logits(ctx))
    #: The last reader of the case matrix's tensors. Six `[T, 151936]` blocks is
    #: 2.3 GiB at fp32 and every later stage wants that headroom for its own
    #: logits, so they are dropped here rather than held to the end of the run.
    ctx.pop("case_tensors", None)
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return {
        "n_taps": len(rows),
        "first_tap_that_is_not_bit_identical": first,
        "per_tap": rows,
        "final_logits": logits,
        "_note": ("taps are listed in EXECUTION order, so the first "
                  "non-identical tap is where solo and batched stop agreeing"),
    }


# --- 4. does a bare GEMM already disagree? ----------------------------------


def stage_gemm_isolation(cfg, ctx) -> dict:
    """Strip away attention, norms and softmax: just `x @ W^T`, two shapes.

    If the same rows, fed through the same weight, already differ when the
    tensor beside them is wider, then nothing above this is the cause and the
    attention backend is not the cause either. This is the cheapest possible
    place for the answer to be, so it is asked directly.
    """
    import torch

    model, adapter, device = ctx["model"], ctx["adapter"], ctx["device"]
    block = adapter.blocks(model)[int(cfg["focal_layer"])]
    named = {name: linear for name, (linear, _n) in
             adapter.stream_in_projections(block).items()}
    named.update(adapter.stream_out_projections(block))
    named["lm_head"] = model.lm_head

    rows = {}
    g = torch.Generator(device="cpu").manual_seed(int(cfg["seed"]) + 7)
    for name, linear in named.items():
        in_features = linear.weight.shape[1]
        T, B, W = int(cfg["gemm_rows"]), int(cfg["equal_batch"]), int(cfg["gemm_pad"])
        x = (torch.randn(1, T, in_features, generator=g) * 0.02).to(
            device=device, dtype=linear.weight.dtype)
        wide = torch.zeros(B, T + W, in_features,
                           device=device, dtype=linear.weight.dtype)
        wide[0, :T] = x[0]
        wide[1:] = (torch.randn(B - 1, T + W, in_features, generator=g) * 0.02
                    ).to(device=device, dtype=linear.weight.dtype)
        with torch.no_grad():
            solo = linear(x)[0, :T]
            batched = linear(wide)[0, :T]
        rows[name] = {
            "in_features": int(in_features),
            "out_features": int(linear.weight.shape[0]),
            "dtype": str(linear.weight.dtype),
            **compare(solo, batched),
        }
    any_moved = any(not r["bitwise_identical"] for r in rows.values())
    return {
        "shape_solo": [1, int(cfg["gemm_rows"])],
        "shape_batched": [int(cfg["equal_batch"]),
                          int(cfg["gemm_rows"]) + int(cfg["gemm_pad"])],
        "per_projection": rows,
        "a_bare_gemm_is_shape_dependent": any_moved,
        "_reading": ("true here means the divergence is a GEMM-level property "
                     "of this dtype/backend and is not introduced by attention, "
                     "masking, padding semantics or the operator code"),
    }


# --- 5. the THREE bf16 controls, and the split-K causal test ----------------


def stage_bf16_controls(cfg, ctx) -> dict:
    """Default vs reduced-precision-off vs split-K-off. Three states, named.

    The previous round ran two and called the second one "split-K off". It was
    not: the boolean form leaves `allow_splitk = True`. So the causal question
    -- does forbidding split-K remove the shape dependence? -- had never been
    asked. This asks it, and reads the runtime back after every request rather
    than trusting the assignment.
    """
    import torch

    before = read_bf16_policy()
    before_blas = read_blas_backend()
    controls = {
        CONTROL_A: (True, None, "cublas"),    # the historical default
        CONTROL_B: (False, None, "cublas"),   # boolean form: split-K STAYS ON
        CONTROL_D: (True, None, "cublaslt"),  # the BACKEND change, alone
        CONTROL_C: (False, False, "cublaslt"),  # the intervention
    }
    out: dict = {"policy_before": before, "controls": {}}
    try:
        for name, (reduced, splitk, blas) in controls.items():
            applied = set_bf16_policy(reduced, splitk, blas)
            row = {"policy": applied}
            if applied["supported"]:
                row["gemm"] = stage_gemm_isolation(cfg, ctx)
                row["logits_solo_vs_batched"] = _solo_vs_batched_logits(ctx)
            else:
                row["gemm"] = None
                row["logits_solo_vs_batched"] = None
                row["_why_absent"] = (
                    "this build cannot express the requested policy; recorded "
                    "UNSUPPORTED rather than substituting the boolean form")
            out["controls"][name] = row
    finally:
        _restore(before, before_blas)

    c = out["controls"].get(CONTROL_C) or {}
    out["blas_before"] = before_blas
    out["tuple_form_supported"] = bool((c.get("policy") or {}).get("supported"))
    out["split_k_state"] = {
        name: ((row.get("policy") or {}).get("readback") or {}).get("allow_splitk")
        for name, row in out["controls"].items()}
    out["blas_library"] = {
        name: ((row.get("policy") or {}).get("blas") or {}).get("readback")
        for name, row in out["controls"].items()}
    out["blas_request_honoured"] = {
        name: ((row.get("policy") or {}).get("blas") or {}).get("honoured")
        for name, row in out["controls"].items()}
    out["gemm_probe_ran"] = {
        name: ((row.get("policy") or {}).get("gemm_probe") or {}).get("ran")
        for name, row in out["controls"].items()}

    def moved(row):
        g = row.get("gemm") or {}
        return sorted(n for n, v in (g.get("per_projection") or {}).items()
                      if not v["bitwise_identical"])

    out["projections_still_shape_dependent"] = {
        name: moved(row) for name, row in out["controls"].items()
        if row.get("gemm")}
    still = out["projections_still_shape_dependent"]
    #: THE causal question, as a field.
    out["split_k_off_makes_every_gemm_exact"] = (
        CONTROL_C in still and still[CONTROL_C] == [])
    out["split_k_off_is_strictly_better_than_boolean"] = (
        CONTROL_B in still and CONTROL_C in still
        and set(still[CONTROL_C]) < set(still[CONTROL_B]))
    #: ATTRIBUTION. `allow_splitk=False` requires cuBLASLt, so control C
    #: changes the BLAS library AND the flag. Control D changes only the
    #: library. Without comparing the two, an improvement at C could be either.
    d = out["controls"].get(CONTROL_D) or {}
    if CONTROL_D in still and CONTROL_C in still:
        only_lt, both = set(still[CONTROL_D]), set(still[CONTROL_C])
        out["backend_switch_alone_fixes_it"] = not only_lt
        out["split_k_flag_adds_something_beyond_the_backend"] = bool(
            both < only_lt)
        out["improvement_attributable_to"] = (
            "cublaslt_backend_alone" if not only_lt
            else "the_split_k_flag" if both < only_lt
            else "neither" if both == only_lt else "undetermined")
    else:
        out["backend_switch_alone_fixes_it"] = None
        out["split_k_flag_adds_something_beyond_the_backend"] = None
        out["improvement_attributable_to"] = "undetermined"

    out["_reading"] = {
        CONTROL_A: "cuBLAS,   allow_reduced_precision=True,  allow_splitk=True",
        CONTROL_B: "cuBLAS,   allow_reduced_precision=False, allow_splitk=True  "
                   "<- what the previous round actually ran",
        CONTROL_D: "cuBLASLt, allow_reduced_precision=True,  allow_splitk=True  "
                   "<- the BACKEND change alone, so C is attributable",
        CONTROL_C: "cuBLASLt, allow_reduced_precision=False, allow_splitk=False "
                   "<- the intervention that had never been performed",
        "_the_boolean_does_not_disable_split_k":
            "torch's parser returns (value, True) for a bool; only the tuple "
            "form reaches the second flag",
        "_and_the_tuple_needs_cublaslt":
            "torch accepts (False, False) at the setter and then raises inside "
            "F.linear unless the BLAS library is cuBLASLt -- which is why every "
            "policy here is proved by running a real bf16 GEMM under it",
    }
    return out


def stage_solo_preservation(cfg, ctx) -> dict:
    """Does the historical SOLO path survive the new policy, unchanged?

    As important as making batching agree, and easy to forget: a policy that
    made batch and solo agree by moving BOTH to a third numerical result would
    not be invariance, it would be a new experiment wearing invariance's name.
    Same checkpoint, same item, same shape -- only the policy moves.
    """
    import torch

    model, focal, device = ctx["model"], ctx["focal"], ctx["device"]
    ids, L = focal.to(device), focal.shape[1]
    before = read_bf16_policy()
    before_blas = read_blas_backend()
    try:
        set_bf16_policy(True, None, "cublas")
        with torch.no_grad():
            solo_default = model(ids).logits[0, :L].clone()

        applied = set_bf16_policy(False, False, "cublaslt")
        if not applied["supported"]:
            return {"ran": False, "reason": "tuple policy unsupported",
                    "policy": applied}
        with torch.no_grad():
            solo_splitk_off = model(ids).logits[0, :L].clone()
            #: C -- equal-length batch of identical copies. The pure batch
            #: dimension, with no padding involved at all.
            dup = ids.repeat(int(cfg["equal_batch"]), 1)
            equal_splitk_off = model(
                dup, attention_mask=torch.ones_like(dup)).logits[0, :L].clone()
            #: E -- the production path: ragged, right-padded, masked.
            bi, bm, _ = _ragged(focal, ctx["others"], ctx)
            batch_splitk_off = model(bi, attention_mask=bm).logits[0, :L].clone()
    finally:
        _restore(before, before_blas)

    solo_vs_solo = compare_logits(solo_default, solo_splitk_off)
    batch_vs_default = compare_logits(solo_default, batch_splitk_off)
    equal_vs_default = compare_logits(solo_default, equal_splitk_off)
    equal_vs_solo_off = compare_logits(solo_splitk_off, equal_splitk_off)
    ragged_vs_solo_off = compare_logits(solo_splitk_off, batch_splitk_off)
    return {
        "ran": True,
        "focal_positions": int(L),
        "equal_batch_size": int(cfg["equal_batch"]),
        "solo_default_vs_solo_splitk_off": solo_vs_solo,
        "batch_splitk_off_vs_solo_default": batch_vs_default,
        #: A / C / E under the new policy, which is the minimum full-model
        #: comparison: A is the historical shape, C the pure batch dimension,
        #: E the ragged padded production path.
        "equal_batch_splitk_off_vs_solo_default": equal_vs_default,
        "equal_batch_vs_solo_both_splitk_off": equal_vs_solo_off,
        "ragged_batch_vs_solo_both_splitk_off": ragged_vs_solo_off,
        "all_three_shapes_agree_under_splitk_off": (
            equal_vs_solo_off["bitwise_identical"]
            and ragged_vs_solo_off["bitwise_identical"]),
        #: The clean execution-only repair, as two propositions.
        "historical_solo_output_is_preserved": solo_vs_solo["bitwise_identical"],
        "batch_under_splitk_off_equals_historical_solo":
            batch_vs_default["bitwise_identical"],
        "equal_batch_under_splitk_off_equals_historical_solo":
            equal_vs_default["bitwise_identical"],
        "_the_outcome_to_avoid": (
            "solo and batch agreeing with EACH OTHER at a third value. If "
            "`historical_solo_output_is_preserved` is false, the policy is a "
            "numerical change to the historical path and not a transparent fix."),
    }


def _solo_vs_batched_logits(ctx) -> dict:
    """Case A vs case E, recomputed under whatever policy is in force now."""
    import torch

    model, focal, others = ctx["model"], ctx["focal"], ctx["others"]
    L = focal.shape[1]
    with torch.no_grad():
        solo = model(focal.to(ctx["device"])).logits[0, :L]
        bi, bm, _ = _ragged(focal, others, ctx)
        batched = model(bi, attention_mask=bm).logits[0, :L]
    return compare_logits(solo, batched)


# --- 6. the attention backend matrix ----------------------------------------


def stage_backend_matrix(cfg, ctx) -> dict:
    """eager vs sdpa, and within sdpa each kernel that this build will run.

    A flash or memory-efficient kernel tiles over the sequence, and its tiling
    is chosen from the padded width -- an obvious candidate for a shape-dependent
    result. Recorded per backend rather than assumed from the default.
    """
    import torch

    results, ran = {}, []
    for attn in ("eager", "sdpa"):
        try:
            model, adapter = load_model(cfg, ctx["device"], ctx["dtype_name"],
                                        attn=attn)
        except Exception as exc:                                # noqa: BLE001
            results[attn] = {"loaded": False, "error": f"{type(exc).__name__}: {exc}"}
            continue
        sub = {"loaded": True, "reported": attention_implementation_of(model)}
        local = dict(ctx, model=model, adapter=adapter)
        sub["default_kernel"] = _solo_vs_batched_logits(local)
        if attn == "sdpa":
            sub["kernels"] = _sdpa_kernels(cfg, local)
        results[attn] = sub
        ran.append(attn)
        del model, local
        if ctx["device"].startswith("cuda"):
            torch.cuda.empty_cache()
    moved = {k: v["default_kernel"]["max_abs"] for k, v in results.items()
             if v.get("loaded")}
    for name, sub in results.items():
        for kernel, kv in (sub.get("kernels") or {}).items():
            if isinstance(kv, dict) and kv.get("max_abs") is not None:
                moved[f"{name}:{kernel}"] = kv["max_abs"]
    diverging = [k for k, m in moved.items() if m > 0]
    exact = [k for k, m in moved.items() if m == 0]
    return {
        "backends": results,
        "max_abs_by_backend": moved,
        "backends_measured": sorted(moved),
        "backends_diverging": sorted(diverging),
        "backends_exact": sorted(exact),
        #: Three distinct states, not two. "not every backend diverges" is true
        #: both when one kernel is exact and the rest are not, and when NONE
        #: diverge -- and those demand opposite conclusions.
        "every_backend_diverges": bool(moved) and not exact,
        "no_backend_diverges": bool(moved) and not diverging,
        "_reading": ("if every backend diverges by a similar amount the cause is "
                     "below attention; if some are exact and some are not, the "
                     "cause is the diverging kernel's shape-dependent tiling; if "
                     "none diverge there is nothing here to attribute"),
    }


def _sdpa_kernels(cfg, ctx) -> dict:
    """Each SDPA kernel this build exposes, one at a time."""
    import torch

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except Exception as exc:                                    # noqa: BLE001
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    out = {"available": True}
    for name in ("MATH", "FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION"):
        backend = getattr(SDPBackend, name, None)
        if backend is None:
            out[name] = {"present_in_build": False}
            continue
        try:
            with sdpa_kernel(backend):
                out[name] = {"present_in_build": True,
                             **_solo_vs_batched_logits(ctx)}
        except Exception as exc:                                # noqa: BLE001
            out[name] = {"present_in_build": True, "ran": False,
                         "error": f"{type(exc).__name__}: {exc}"}
    return out


# --- 7. does it depend on the padded width? ---------------------------------


def stage_length_sweep(cfg, ctx) -> dict:
    """Separate 'there are other rows' from 'the rows are longer'.

    Two knobs are confounded in the production path: batch size B and padded
    width T. Each is swept with the other held fixed, so a result can say which
    one the divergence tracks.
    """
    import torch

    model, focal, device = ctx["model"], ctx["focal"], ctx["device"]
    L = focal.shape[1]
    ids = focal.to(device)
    with torch.no_grad():
        ref = model(ids).logits[0, :L]

    by_batch, by_pad = [], []
    g = torch.Generator().manual_seed(int(cfg["seed"]) + 3)
    vocab = int(model.config.vocab_size)
    for b in cfg["batch_sweep"]:
        rows = [ids] + [torch.randint(1, vocab, (1, L), generator=g).to(device)
                        for _ in range(int(b) - 1)]
        x = torch.cat(rows, 0)
        with torch.no_grad():
            got = model(x, attention_mask=torch.ones_like(x)).logits[0, :L]
        by_batch.append({"batch_size": int(b), "padded_width": int(L),
                         **compare(ref, got)})
    for pad in cfg["pad_sweep"]:
        pad = int(pad)
        if pad == 0:
            x, m = ids, torch.ones_like(ids)
        else:
            tail = torch.full((1, pad), int(_pad_id(ctx)), device=device,
                              dtype=ids.dtype)
            x = torch.cat([ids, tail], 1)
            m = torch.cat([torch.ones_like(ids),
                           torch.zeros_like(tail)], 1)
        with torch.no_grad():
            got = model(x, attention_mask=m).logits[0, :L]
        by_pad.append({"batch_size": 1, "padded_width": int(L + pad),
                       "pad_positions": pad, **compare(ref, got)})
    # A third axis: the item's OWN length, with the padding held at a fixed
    # count. If the effect grows with T it is an accumulation-length property;
    # if it is flat it is a tiling-boundary one.
    by_length = []
    pad = int(cfg["length_sweep_pad"])
    tail = torch.full((1, pad), int(_pad_id(ctx)), device=device, dtype=ids.dtype)
    for frac in cfg["length_fractions"]:
        n = max(8, int(L * float(frac)))
        head = ids[:, :n]
        with torch.no_grad():
            solo_n = model(head).logits[0, :n]
            x = torch.cat([head, tail], 1)
            m = torch.cat([torch.ones_like(head), torch.zeros_like(tail)], 1)
            got = model(x, attention_mask=m).logits[0, :n]
        by_length.append({"true_length": n, "pad_positions": pad,
                          **compare(solo_n, got)})

    return {
        "vary_batch_size_at_fixed_length": by_batch,
        "vary_padding_at_batch_size_1": by_pad,
        "vary_true_length_at_fixed_padding": by_length,
        "padding_alone_is_inert": all(r["bitwise_identical"] for r in by_pad),
        "batch_size_alone_moves_it": any(not r["bitwise_identical"]
                                         for r in by_batch),
        "grows_with_true_length": (
            by_length[-1]["max_abs"] > 2 * by_length[0]["max_abs"]
            if len(by_length) > 1 and by_length[0]["max_abs"] > 0 else None),
        "_reading": {
            "vary_padding_at_batch_size_1": "padded width alone",
            "vary_batch_size_at_fixed_length": "batch dimension alone",
            "vary_true_length_at_fixed_padding":
                "accumulation length alone; growth with T points at reduction "
                "width, flatness at a tiling boundary",
        },
    }


def _pad_id(ctx) -> int:
    from aadistill.initialization.calibration.batching import resolve_pad_id
    return resolve_pad_id(ctx["model"])


# --- 8. forward drift vs reduction drift ------------------------------------


def stage_statistics_decomposition(cfg, ctx) -> dict:
    """Two candidate causes of a moved statistic, measured apart.

    A statistic can move because the ACTIVATIONS the forward produced moved, or
    because the same activations were SUMMED in a different order. Those demand
    different responses, so they are separated rather than reported as one
    number: the captured activations are compared elementwise, and then the
    captured activations are re-summed in both groupings.
    """
    import torch
    from aadistill.initialization.calibration.batching import build_batch, micro_batches
    from aadistill.initialization.statistics.collect import ActivationStatsCollector

    model, adapter, device = ctx["model"], ctx["adapter"], ctx["device"]
    items = ctx["items"]
    pad = _pad_id(ctx)
    layer = int(cfg["focal_layer"])
    down = adapter.stream_out_projections(adapter.blocks(model)[layer])["ffn_out"]

    captured: list = []

    def grab(_m, args):
        captured.append(args[0].detach())

    # (a) the live statistic, through the production collector, at EVERY micro
    #     batch size under test. The rejected finding quoted bs=1 vs bs=3 and
    #     DEFAULT_MICRO_BATCH_SIZE is 4, so measuring one of them would leave
    #     the comparison inexact in the one place it has to be exact.
    #
    #     `residual_sqsum` is [P, H, H] float64 -- 1.94 GiB at a 4B parent -- so
    #     the reference state is held and each batched state is compared and
    #     released, rather than accumulating one per sweep point.
    def collect(bs):
        collector = ActivationStatsCollector(
            model, [adapter.stream_out_projections(b)["ffn_out"]
                    for b in adapter.blocks(model)])
        try:
            for batch in micro_batches(items, bs, pad_id=pad, device=device):
                collector.process_batch(batch)
            return collector.state()
        finally:
            collector.close()

    ref_state = collect(1)
    sweep, kept_importance = {}, {1: ref_state}
    for bs in cfg["micro_batch_sweep"]:
        bs = int(bs)
        if bs == 1:
            continue
        bat_state = collect(bs)
        row = {name: compare(ref_state[name].float(), bat_state[name].float())
               for name in ("ffn_abs_sum", "ffn_sq_sum", "residual_sum")}
        row["residual_sqsum"] = compare(ref_state["residual_sqsum"].float(),
                                        bat_state["residual_sqsum"].float())
        row["token_counts_identical"] = bool(
            torch.equal(ref_state["token_counts"], bat_state["token_counts"]))
        row["residual_count_identical"] = bool(
            torch.equal(ref_state["residual_count"], bat_state["residual_count"]))
        row["every_accumulator_bit_identical"] = all(
            row[name]["bitwise_identical"] for name in
            ("ffn_abs_sum", "ffn_sq_sum", "residual_sum", "residual_sqsum"))
        sweep[bs] = row
        #: Only the FFN term survives the sweep -- [L, I] float64, ~2.8 MiB --
        #: because that is all `ffn_neuron_importance` reads. Keeping whole
        #: states would be ~2 GiB each.
        kept_importance[bs] = {
            "ffn_abs_sum": bat_state["ffn_abs_sum"],
            "residual_count": bat_state["residual_count"]}
        del bat_state
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    live = sweep.get(int(cfg["micro_batch_size"]),
                     sweep[min(sweep)] if sweep else {})

    # (b) the SAME items' down_proj inputs, captured in both shapes.
    handle = down.register_forward_pre_hook(grab)
    try:
        captured.clear()
        with torch.no_grad():
            for item in items:
                model(item["input_ids"].to(device))
        solo_rows = torch.cat([t.reshape(-1, t.shape[-1]) for t in captured], 0)

        captured.clear()
        with torch.no_grad():
            for batch in micro_batches(items, int(cfg["micro_batch_size"]),
                                       pad_id=pad, device=device):
                model(batch.input_ids.to(device),
                      attention_mask=batch.attention_mask.to(device))
                captured[-1] = batch.valid_tokens(
                    captured[-1].reshape(batch.size, -1, captured[-1].shape[-1]))
        batched_rows = torch.cat([t.reshape(-1, t.shape[-1]) for t in captured], 0)
    finally:
        handle.remove()

    out = {"layer": layer, "n_items": len(items),
           "item_lengths": [int(i["input_ids"].shape[1]) for i in items],
           "micro_batch_size": int(cfg["micro_batch_size"]),
           "micro_batch_sweep": [int(b) for b in cfg["micro_batch_sweep"]],
           "by_micro_batch_size": sweep,
           "every_batch_size_bit_identical": bool(sweep) and all(
               r["every_accumulator_bit_identical"] for r in sweep.values()),
           "live_collector_state": live}
    if solo_rows.shape != batched_rows.shape:
        out["forward_drift"] = {"comparable": False,
                                "solo_rows": list(solo_rows.shape),
                                "batched_rows": list(batched_rows.shape)}
        return out

    out["forward_drift"] = {"comparable": True, **compare(solo_rows, batched_rows)}
    # (c) reduction order, with the activations held FIXED. Same values, two
    #     groupings: one row block, or per-item blocks added together.
    fixed = solo_rows.to(torch.float64).abs()
    whole = fixed.sum(0)
    piecewise = torch.zeros_like(whole)
    at = 0
    for item in items:
        n = int(item["input_ids"].shape[1])
        piecewise += fixed[at:at + n].sum(0)
        at += n
    out["reduction_order_drift_at_fixed_activations"] = compare(whole, piecewise)
    out["_reading"] = {
        "forward_drift": "the activations themselves moved",
        "reduction_order_drift_at_fixed_activations":
            "float64 accumulation grouping alone, activations held identical",
        "live_collector_state": "what the operator actually consumes = both",
    }
    ctx["stats_states"] = kept_importance
    return out


# --- 9. what the FFN operator would decide ----------------------------------


def stage_ffn_selection(cfg, ctx) -> dict:
    """The decision, and how close it was to going the other way.

    A moved statistic only matters if it moves a DECISION, and a moved decision
    only means something relative to how far apart the two neurons at the cutoff
    were. Both are reported; neither is summarised into a verdict here.

    Swept over the SAME keep ratios the rejected finding quoted -- 0.32, 0.50,
    0.75, 0.90 -- and over every micro batch size the statistics stage
    collected, so "26 of 28 at keep=0.32" has a like-for-like counterpart here
    instead of a nearby number measured differently.
    """
    import torch
    from aadistill.initialization.transforms.project import ffn_neuron_importance

    states = ctx.get("stats_states")
    if not states or 1 not in states:
        return {"ran": False, "reason": "statistics decomposition did not run"}
    model, adapter = ctx["model"], ctx["adapter"]
    ref_state = states[1]
    others = sorted(b for b in states if b != 1)
    if not others:
        return {"ran": False, "reason": "only one micro batch size was collected"}

    #: Importance is `E[|a_j|] * ||down_proj column j||_2`, and the column norms
    #: are a property of the WEIGHTS -- identical in every arm by construction.
    #: Computed once, so a per-layer loop does not re-reduce a [H, I] matrix
    #: four times per keep ratio.
    importance = {}
    for index, block in enumerate(adapter.blocks(model)):
        w_down = adapter.stream_out_projections(block)["ffn_out"].weight.detach().cpu()
        importance[index] = {
            bs: ffn_neuron_importance(state, index, w_down)
            for bs, state in ((1, ref_state), *((b, states[b]) for b in others))}

    by_ratio: dict = {}
    for ratio in cfg["ffn_keep_ratios"]:
        ratio = float(ratio)
        per_bs: dict = {}
        for bs in others:
            rows, moved = [], 0
            for index in sorted(importance):
                imp_ref, imp_bat = importance[index][1], importance[index][bs]
                keep = max(1, int(round(imp_ref.numel() * ratio)))
                sel_ref = torch.topk(imp_ref, keep).indices.sort().values
                sel_bat = torch.topk(imp_bat, keep).indices.sort().values
                same = bool(torch.equal(sel_ref, sel_bat))
                moved += 0 if same else 1
                ordered = imp_ref.sort(descending=True).values
                # The gap the decision turned on: last kept vs first dropped.
                margin = (float(ordered[keep - 1] - ordered[keep])
                          if keep < ordered.numel() else float("inf"))
                scale = float(ordered[keep - 1])
                drift = compare(imp_ref, imp_bat)
                rows.append({
                    "layer": index, "keep": keep, "selection_identical": same,
                    #: Set difference, not an elementwise compare of two sorted
                    #: index vectors -- those can disagree in many positions
                    #: because ONE neuron was swapped, which reads as a much
                    #: larger change than occurred.
                    "n_neurons_swapped": len(
                        set(sel_ref.tolist()) - set(sel_bat.tolist())),
                    "cutoff_margin_abs": margin,
                    "cutoff_margin_relative": margin / scale if scale else None,
                    "importance_max_abs_drift": drift["max_abs"],
                    "importance_rel_l2_drift": drift["rel_l2"],
                    "drift_exceeds_margin": bool(drift["max_abs"] > margin),
                })
            per_bs[bs] = {
                "n_layers": len(rows),
                "n_layers_with_moved_selection": moved,
                "selection_is_batch_invariant": moved == 0,
                "total_neurons_swapped": sum(r["n_neurons_swapped"] for r in rows),
                "min_cutoff_margin_relative": min(
                    (r["cutoff_margin_relative"] for r in rows
                     if r["cutoff_margin_relative"] is not None), default=None),
                "n_layers_where_drift_exceeds_margin": sum(
                    1 for r in rows if r["drift_exceeds_margin"]),
                "per_layer": rows,
            }
        by_ratio[f"{ratio:.2f}"] = per_bs

    default_bs = int(cfg["micro_batch_size"])
    headline_bs = default_bs if default_bs in others else others[0]
    headline = by_ratio[f"{float(cfg['ffn_keep_ratio']):.2f}"][headline_bs]
    return {
        "ran": True,
        "keep_ratios": [float(r) for r in cfg["ffn_keep_ratios"]],
        "micro_batch_sizes_compared_against_1": others,
        "by_keep_ratio": by_ratio,
        #: The headline pair is DEFAULT_MICRO_BATCH_SIZE at the recipe's own
        #: keep ratio -- the configuration a formal run would actually execute.
        "headline_micro_batch_size": headline_bs,
        "headline_keep_ratio": float(cfg["ffn_keep_ratio"]),
        "n_layers": headline["n_layers"],
        "n_layers_with_moved_selection": headline["n_layers_with_moved_selection"],
        "selection_is_batch_invariant": headline["selection_is_batch_invariant"],
        "selection_is_batch_invariant_at_every_ratio_and_size": all(
            sub["selection_is_batch_invariant"]
            for per_bs in by_ratio.values() for sub in per_bs.values()),
        "_reading": ("cutoff_margin is the importance gap the decision turned "
                     "on; a drift far below every margin cannot move a "
                     "selection, and one above some margin will move it "
                     "sometimes, as a function of the checkpoint, not the code"),
    }


# --- 9b. the thing the project actually cares about -------------------------


def _ffn_selection(model, adapter, states, ratio):
    """kept-neuron index per layer, per collected batch size."""
    import torch
    from aadistill.initialization.transforms.project import ffn_neuron_importance

    out: dict = {}
    for index, block in enumerate(adapter.blocks(model)):
        w = adapter.stream_out_projections(block)["ffn_out"].weight.detach().cpu()
        for bs, state in states.items():
            imp = ffn_neuron_importance(state, index, w)
            keep = max(1, int(round(imp.numel() * ratio)))
            out.setdefault(bs, {})[index] = torch.topk(
                imp, keep).indices.sort().values
    return out


def _attention_selection(model, adapter, states, keep_q):
    """kept query-head index per layer, per collected batch size.

    `head_write_energy` refuses a host statistic meeting a device weight, so
    the o_proj weight is brought to the statistic rather than the reverse --
    the caller owns the co-location, and this caller chooses the host.
    """
    from aadistill.initialization.operators.attention.gqa._common import (
        select_q_heads_by_score)
    from aadistill.initialization.operators.attention.gqa._statistics import (
        head_write_energy)

    spec = model.config
    n_q, n_kv = spec.num_attention_heads, spec.num_key_value_heads
    head_dim = getattr(spec, "head_dim", None) or spec.hidden_size // n_q
    out: dict = {}
    for index, block in enumerate(adapter.blocks(model)):
        w = adapter.stream_out_projections(block)["attn_out"].weight.detach().cpu()
        for bs, state in states.items():
            scores = head_write_energy(state, index, w, n_q, head_dim)
            out.setdefault(bs, {})[index] = select_q_heads_by_score(
                scores, n_q, n_kv, keep_q)
    return out


def stage_operator_acceptance(cfg, ctx) -> dict:
    """bs=1 vs bs=4 discrete operator outputs, under each bf16 control.

    Raw floats agreeing bit-for-bit is not the requirement; the requirement is
    that the DECISION does not move. So this runs the two real selections --
    FFN kept neurons and ATTENTION kept query heads -- on the frozen mixture
    and asks whether the index sets are equal.
    """
    import torch
    from aadistill.initialization.calibration.batching import micro_batches
    from aadistill.initialization.statistics.collect import ActivationStatsCollector
    from aadistill.initialization.operators.attention.gqa._statistics import (
        AttentionHeadStatsCollector)

    model, adapter, device = ctx["model"], ctx["adapter"], ctx["device"]
    items, pad = ctx["items"], _pad_id(ctx)
    sizes = [1, int(cfg["micro_batch_size"])]
    spec = model.config
    n_q, n_kv = spec.num_attention_heads, spec.num_key_value_heads
    head_dim = getattr(spec, "head_dim", None) or spec.hidden_size // n_q
    keep_q = max(n_kv, (n_q // 2 // n_kv) * n_kv)          # half, group-aligned
    ffn_ratio = float(cfg["ffn_keep_ratio"])

    def collect(bs):
        """One pass per collector, each alone. NOT both at once.

        Both collectors DRIVE their own forward -- `process_batch` calls
        `self.model(...)` -- so running them together is two forwards, not one,
        and each collector's hooks also fire during the OTHER's forward with
        its own `_valid_mask` unset. That silently accumulates padded rows.
        The first rehearsal of this stage read 28 of 28 FFN layers moved on CPU
        float32, where the true answer is zero: I had manufactured the
        divergence I was measuring. Separate passes cost a second forward and
        are the only correct way to drive two self-driving collectors.
        """
        ffn = ActivationStatsCollector(
            model, [adapter.stream_out_projections(b)["ffn_out"]
                    for b in adapter.blocks(model)])
        try:
            for batch in micro_batches(items, bs, pad_id=pad, device=device):
                ffn.process_batch(batch)
            ffn_state = ffn.state()
        finally:
            ffn.close()

        attn = AttentionHeadStatsCollector(
            model, [adapter.stream_out_projections(b)["attn_out"]
                    for b in adapter.blocks(model)],
            num_heads=n_q, head_dim=head_dim)
        try:
            for batch in micro_batches(items, bs, pad_id=pad, device=device):
                attn.process_batch(batch)
            attn_state = attn.state()
        finally:
            attn.close()
        return ffn_state, attn_state

    before = read_bf16_policy()
    before_blas = read_blas_backend()
    results: dict = {}
    wanted = {CONTROL_A: (True, None, "cublas"),
              CONTROL_D: (True, None, "cublaslt"),
              CONTROL_C: (False, False, "cublaslt")}
    try:
        for name, (reduced, splitk, blas) in wanted.items():
            applied = set_bf16_policy(reduced, splitk, blas)
            if not applied["supported"]:
                results[name] = {"ran": False, "policy": applied}
                continue
            ffn_states, attn_states = {}, {}
            for bs in sizes:
                f, a = collect(bs)
                ffn_states[bs], attn_states[bs] = f, a
            fsel = _ffn_selection(model, adapter, ffn_states, ffn_ratio)
            asel = _attention_selection(model, adapter, attn_states, keep_q)
            ref = sizes[0]
            rows = []
            for index in sorted(fsel[ref]):
                same_f = bool(torch.equal(fsel[ref][index], fsel[sizes[1]][index]))
                same_a = asel[ref][index] == asel[sizes[1]][index]
                rows.append({
                    "layer": index,
                    "ffn_selection_identical": same_f,
                    "ffn_neurons_swapped": len(
                        set(fsel[ref][index].tolist())
                        - set(fsel[sizes[1]][index].tolist())),
                    "attention_selection_identical": same_a,
                    "attention_heads_swapped": len(
                        set(asel[ref][index]) - set(asel[sizes[1]][index])),
                })
            results[name] = {
                "ran": True,
                "policy": applied,
                "n_layers": len(rows),
                "ffn_layers_moved": sum(1 for r in rows
                                        if not r["ffn_selection_identical"]),
                "attention_layers_moved": sum(
                    1 for r in rows if not r["attention_selection_identical"]),
                "ffn_selection_invariant": all(
                    r["ffn_selection_identical"] for r in rows),
                "attention_selection_invariant": all(
                    r["attention_selection_identical"] for r in rows),
                "per_layer": rows,
            }
            del ffn_states, attn_states
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    finally:
        _restore(before, before_blas)

    c = results.get(CONTROL_C) or {}
    return {
        "n_items": len(items),
        "micro_batch_sizes": sizes,
        "ffn_keep_ratio": ffn_ratio,
        "attention_keep_q": keep_q,
        "attention_heads": [n_q, n_kv, head_dim],
        "by_control": results,
        #: THE acceptance criterion, as one field per operator.
        "ffn_invariant_under_splitk_off": c.get("ffn_selection_invariant"),
        "attention_invariant_under_splitk_off": c.get(
            "attention_selection_invariant"),
        "both_operators_invariant_under_splitk_off": bool(
            c.get("ffn_selection_invariant")
            and c.get("attention_selection_invariant")),
        "_reading": ("bit-identical floats are not the requirement; an "
                     "unmoved DECISION is. These are the two real selections "
                     "an initialization run makes."),
    }


def stage_performance(cfg, ctx) -> dict:
    """What the invariant path costs. Three wall times, same workload.

    Not a benchmark. The question is only whether a batched path that is
    numerically invariant still buys anything over running items one at a time.
    """
    import time

    import torch
    from aadistill.initialization.calibration.batching import micro_batches
    from aadistill.initialization.statistics.collect import ActivationStatsCollector

    model, adapter, device = ctx["model"], ctx["adapter"], ctx["device"]
    #: CAPPED, and the cap is reported. Six timed passes over the full frozen
    #: mixture would cost more GPU minutes than the measurement is worth, and
    #: the question -- is the invariant batched path still faster than one item
    #: at a time -- is a ratio that a representative slice answers.
    items = ctx["items"][: int(cfg["perf_items"])]
    pad = _pad_id(ctx)
    sync = torch.cuda.synchronize if device.startswith("cuda") else (lambda: None)

    def timed(bs):
        collector = ActivationStatsCollector(
            model, [adapter.stream_out_projections(b)["ffn_out"]
                    for b in adapter.blocks(model)])
        try:
            sync(); t0 = time.perf_counter()
            for batch in micro_batches(items, bs, pad_id=pad, device=device):
                collector.process_batch(batch)
            sync()
            return time.perf_counter() - t0
        finally:
            collector.close()

    before = read_bf16_policy()
    before_blas = read_blas_backend()
    out: dict = {"n_items": len(items),
                 "n_items_available": len(ctx["items"]),
                 "_items_are_capped": len(items) < len(ctx["items"]),
                 "total_tokens": sum(int(i["input_ids"].shape[1]) for i in items)}
    bs = int(cfg["micro_batch_size"])
    try:
        set_bf16_policy(True, None, "cublas")
        timed(bs)                                  # warm the kernels, discard
        out[f"default_bs{bs}_seconds"] = timed(bs)
        out["default_bs1_seconds"] = timed(1)
        applied = set_bf16_policy(False, False, "cublaslt")
        out["splitk_off_supported"] = bool(applied["supported"])
        if applied["supported"]:
            timed(bs)
            out[f"splitk_off_bs{bs}_seconds"] = timed(bs)
            out["splitk_off_bs1_seconds"] = timed(1)
    finally:
        _restore(before, before_blas)

    d_b = out.get(f"default_bs{bs}_seconds")
    s_b = out.get(f"splitk_off_bs{bs}_seconds")
    s_1 = out.get("splitk_off_bs1_seconds")
    out["splitk_off_slowdown_vs_default_batched"] = (
        s_b / d_b if d_b and s_b else None)
    out["invariant_batched_speedup_vs_solo"] = (s_1 / s_b if s_b and s_1 else None)
    out["_reading"] = ("the second ratio is the one that matters: if the "
                       "invariant batched path is still faster than one item "
                       "at a time, batching is worth keeping")
    return out


# --- 10. the quantity C3's DEPTH search would actually read -----------------


def stage_causal_kl(cfg, ctx) -> dict:
    """Forward KL of a bypassed block, solo vs batched, per item.

    This is the scored quantity, so it is measured end-to-end through the same
    two reducers the operator uses -- `forward_kl_mean` per item and
    `forward_kl_mean_batch` over the block -- rather than inferred from a logit
    norm.
    """
    import torch
    from aadistill.initialization.calibration.batching import build_batch
    from aadistill.initialization.statistics.contribution import (
        bypassed_blocks, forward_kl_mean, forward_kl_mean_batch)

    model, device, items = ctx["model"], ctx["device"], ctx["items"]
    skip = frozenset({int(cfg["bypass_layer"])})
    pad = _pad_id(ctx)

    solo = []
    with torch.no_grad():
        for item in items:
            ids = item["input_ids"].to(device)
            ref = model(ids).logits[:, :-1]
            with bypassed_blocks(model, skip):
                abl = model(ids).logits[:, :-1]
            solo.append(float(forward_kl_mean(ref[0], abl[0])))

        batch = build_batch(items, pad_id=pad, device=device)
        ids, mask = batch.input_ids.to(device), batch.attention_mask.to(device)
        ref_b = model(ids, attention_mask=mask).logits[:, :-1]
        with bypassed_blocks(model, skip):
            abl_b = model(ids, attention_mask=mask).logits[:, :-1]
        batched = [float(v) for v in forward_kl_mean_batch(
            ref_b, abl_b, batch.prediction_mask().to(device))]

    per_item = [{"item_id": it.get("item_id", f"i{n}"),
                 "length": int(it["input_ids"].shape[1]),
                 "kl_solo": s, "kl_batched": b,
                 "abs_diff": abs(s - b),
                 "rel_diff": abs(s - b) / s if s else None}
                for n, (it, s, b) in enumerate(zip(items, solo, batched))]
    order_solo = sorted(range(len(solo)), key=lambda i: -solo[i])
    order_batched = sorted(range(len(batched)), key=lambda i: -batched[i])
    mean_s, mean_b = sum(solo) / len(solo), sum(batched) / len(batched)
    return {
        "bypassed_layer": int(cfg["bypass_layer"]),
        "n_items": len(items),
        "per_item": per_item,
        "max_abs_diff": max(p["abs_diff"] for p in per_item),
        "max_rel_diff": max((p["rel_diff"] or 0.0) for p in per_item),
        "mean_kl_solo": mean_s, "mean_kl_batched": mean_b,
        "mean_abs_diff": abs(mean_s - mean_b),
        "item_ranking_identical": order_solo == order_batched,
        "_reading": ("the DEPTH search compares candidate subsets by their mean "
                     "KL, so what matters is whether the drift is small next to "
                     "the gap between competing candidates -- measured by the "
                     "search's own margins, not here"),
    }


# --- the fp32 control -------------------------------------------------------


def stage_fp32_control(cfg, ctx) -> dict:
    """The same case matrix in fp32. A DIAGNOSTIC, never a proposal.

    fp32 is not the deployment numeric and nothing here argues for running the
    search in it. Its only job is to say whether the divergence is a property of
    bf16's reduction width: if fp32 collapses it by orders of magnitude, the
    cause is named; if fp32 diverges too, the cause is not the dtype.
    """
    import torch

    if ctx["dtype_name"] == "float32":
        return {"ran": False, "reason": "the run is already float32"}
    #: The last stage, and the only one that needs a SECOND full copy of the
    #: parent at twice the width. 8 GiB of bf16 parent plus 16 GiB of fp32 parent
    #: plus six fp32 `[T, 151936]` case tensors is most of an L40S, so the bf16
    #: model is released first -- nothing after this reads it.
    released = ctx.pop("model", None)
    del released
    if ctx["device"].startswith("cuda"):
        torch.cuda.empty_cache()
    try:
        model, adapter = load_model(cfg, ctx["device"], "float32")
    except Exception as exc:                                    # noqa: BLE001
        return {"ran": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        local = dict(ctx, model=model, adapter=adapter, dtype_name="float32")
        out = {"ran": True, "dtype": "float32",
               "case_matrix": stage_case_matrix(cfg, local),
               "gemm": stage_gemm_isolation(cfg, local)}
    finally:
        del model
        if ctx["device"].startswith("cuda"):
            torch.cuda.empty_cache()
    bf16 = ctx["report"]["stages"].get("case_matrix", {}).get("comparisons", {})
    fp32 = out["case_matrix"]["comparisons"]
    out["ratio_bf16_over_fp32"] = {
        key: (bf16[key]["max_abs"] / fp32[key]["max_abs"]
              if key in bf16 and fp32.get(key, {}).get("max_abs") else None)
        for key in fp32}
    return out


# --- configuration ----------------------------------------------------------
#
# Every knob the stages read, in one literal, so the report can carry the exact
# inputs beside the numbers and a reader never has to guess what "the batch" was.

DEFAULTS: dict = {
    "source": {
        # The parent is a PUBLIC hub model, so a pod fetches it directly and
        # neither the private relay nor the 0.72 MB/s dev-box uplink is involved.
        "local_paths": [],
        "relay_repo": None,
        "relay_path": None,
        "hub_id": "Qwen/Qwen3-4B-Thinking-2507",
    },
    #: Qualified ids carry a `v`. `calib.domain_balanced@1` silently fell through
    #: to synthetic tokens in the first rehearsal, which is the difference
    #: between a claim about the frozen mixture and a claim about random ids.
    "calibration_profile": "calib.domain_balanced@v1",
    "n_items": 8,
    "micro_batch_size": 4,
    "equal_batch": 4,
    "focal_length": 512,
    "neighbour_lengths": [384, 640, 256],
    "focal_layer": 17,
    "bypass_layer": 17,
    "ffn_keep_ratio": 0.5,
    #: The four the rejected finding quoted, so its "26 of 28 at keep=0.32" has
    #: a like-for-like counterpart rather than a nearby number.
    "ffn_keep_ratios": [0.32, 0.50, 0.75, 0.90],
    #: 3 because the finding quoted bs=1 vs bs=3; 4 because that is
    #: DEFAULT_MICRO_BATCH_SIZE and therefore what a formal run would execute.
    "micro_batch_sweep": [1, 2, 3, 4],
    "gemm_rows": 512,
    "gemm_pad": 128,
    "batch_sweep": [1, 2, 3, 4, 8],
    "pad_sweep": [0, 1, 8, 64, 128],
    "length_fractions": [0.125, 0.25, 0.5, 1.0],
    "length_sweep_pad": 64,
    "perf_items": 16,
    "seed": 20260925,
}


def load_items(cfg: dict, vocab: int) -> tuple[list, dict]:
    """Real frozen calibration items when they resolve; synthetic otherwise.

    Which one ran is recorded, because a numerics claim made on random token ids
    is a claim about the arithmetic and a claim made on the frozen mixture is a
    claim about the decision C3 would take. They are not interchangeable.
    """
    import torch

    from aadistill.initialization.calibration.items import (
        CalibrationItemError, prepare_calibration_items)
    from aadistill.initialization.calibration.profiles import (
        CalibrationError, get_profile, load_profiles)

    provenance: dict = {"requested": cfg["calibration_profile"]}
    try:
        doc = json.loads((REPO / "configs/calibration/profiles.json").read_text())
        load_profiles(doc)
        profile = get_profile(cfg["calibration_profile"])
        raw = profile.resolve(REPO)[: int(cfg["n_items"])]
        items = prepare_calibration_items(raw, profile_id=profile.qualified_id)
        provenance.update({
            "kind": "frozen_calibration_mixture",
            #: PROPERTIES, not methods. Calling `profile_hash()` raised
            #: `TypeError: 'str' object is not callable`, a broad `except`
            #: turned that into a silent fall-through to random token ids, and
            #: the run then measured the wrong thing while reporting cleanly.
            "profile_hash": profile.profile_hash,
            "content_sha256": profile.content_sha256,
            "items_path": profile.items_path,
            "n_items": len(items),
            "lengths": [int(i["input_ids"].shape[1]) for i in items],
        })
        return items, provenance
    #: NARROW on purpose. "The mixture is not on this machine" is a legitimate
    #: reason to fall back and say so; a bug in this function is not, and must
    #: reach the caller rather than be recorded as an environment fact.
    except (CalibrationError, CalibrationItemError, KeyError,
            FileNotFoundError) as exc:
        provenance["frozen_mixture_error"] = f"{type(exc).__name__}: {exc}"

    g = torch.Generator().manual_seed(int(cfg["seed"]) + 11)
    lengths = [int(cfg["focal_length"])] + [
        int(n) for n in cfg["neighbour_lengths"]]
    while len(lengths) < int(cfg["n_items"]):
        lengths.append(lengths[len(lengths) % 4])
    items = [{"item_id": f"synth{i}",
              "input_ids": torch.randint(1, vocab, (1, n), generator=g)}
             for i, n in enumerate(lengths[: int(cfg["n_items"])])]
    provenance.update({"kind": "synthetic_random_ids", "n_items": len(items),
                       "lengths": lengths[: int(cfg["n_items"])],
                       "seed": int(cfg["seed"]) + 11})
    return items, provenance


# --- the run ----------------------------------------------------------------

STAGES = (
    ("repeatability", stage_repeatability),
    ("case_matrix", stage_case_matrix),
    ("first_divergence", stage_first_divergence),
    ("gemm_isolation", stage_gemm_isolation),
    ("bf16_controls", stage_bf16_controls),
    ("solo_preservation", stage_solo_preservation),
    ("length_sweep", stage_length_sweep),
    ("statistics_decomposition", stage_statistics_decomposition),
    ("ffn_selection", stage_ffn_selection),
    ("causal_kl", stage_causal_kl),
    ("operator_acceptance", stage_operator_acceptance),
    ("performance", stage_performance),
    ("backend_matrix", stage_backend_matrix),
    ("fp32_control", stage_fp32_control),
)


def derive_conclusion(report: dict) -> dict:
    """The verdict, COMPUTED from the stages, never written beside them.

    A prose conclusion can contradict the arithmetic printed above it and
    survive a reading. So the fields a conclusion would cite are derived here,
    from the stage outputs, and the narrative in the record has to match them.
    """
    s = report["stages"]

    def got(stage, *path, default=None):
        """A field from a stage, or `default` if that stage has nothing to say.

        A stage that did not run, errored, or declared `ran: False` legitimately
        has no fields, and its absence is UNKNOWN. A stage that DID run and is
        missing the key being asked for is a typo, and it raises: this function
        returned `None` for
        `selection_is_batch_invariant_at_every_ratio_and_batch_size` while the
        stage emitted `..._at_every_ratio_and_size`, and the conclusion carried
        a silent `None` into a field advertising a real answer.
        """
        node = s.get(stage)
        if not isinstance(node, dict) or "error" in node or node.get("ran") is False:
            return default
        for depth, key in enumerate(path):
            if not isinstance(node, dict):
                return default
            if key not in node:
                if depth == 0:
                    raise KeyError(
                        f"stage {stage!r} ran but has no field {key!r}; "
                        f"it has {sorted(k for k in node if not k.startswith('_'))}")
                #: Nested paths index DATA (a comparison name, a batch size),
                #: not a field contract, so an absent one is a legitimate
                #: "this run did not produce that case".
                return default
            node = node[key]
        return node

    deterministic = got("repeatability", "both_shapes_internally_deterministic")
    gemm_moves = got("gemm_isolation", "a_bare_gemm_is_shape_dependent")
    pad_inert = got("length_sweep", "padding_alone_is_inert")
    batch_moves = got("length_sweep", "batch_size_alone_moves_it")
    cross_row = got("case_matrix", "comparisons", "C_vs_D", "bitwise_identical")
    #: The name changed because the old one described something it did not do.
    #: `reduced_precision_off_splitk_on` is what the boolean actually gave.
    splitk_off_exact = got("bf16_controls", "split_k_off_makes_every_gemm_exact")
    splitk_off_better = got("bf16_controls",
                            "split_k_off_is_strictly_better_than_boolean")
    tuple_supported = got("bf16_controls", "tuple_form_supported")
    solo_preserved = got("solo_preservation",
                         "historical_solo_output_is_preserved")
    every_backend = got("backend_matrix", "every_backend_diverges")
    no_backend = got("backend_matrix", "no_backend_diverges")
    some_backend = bool(got("backend_matrix", "backends_diverging") or [])
    selection_invariant = got("ffn_selection", "selection_is_batch_invariant")

    #: Anything at all, anywhere. Checked BEFORE attributing a cause: the first
    #: rehearsal reported locus `attention_backend_kernel` on a run in which
    #: every measured comparison was bit-identical, because "not every backend
    #: diverges" had been read as "some backend is exact while others are not".
    observed = [
        key for key, value in {
            "case_matrix": any(
                not c.get("bitwise_identical", True)
                for c in (got("case_matrix", "comparisons", default={}) or {}).values()),
            "length_sweep": bool(batch_moves) or (pad_inert is False),
            "gemm_isolation": bool(gemm_moves),
            "backend_matrix": some_backend,
            "first_divergence": got(
                "first_divergence", "first_tap_that_is_not_bit_identical") is not None,
            #: The statistics too. A `--only statistics_decomposition,ffn_selection`
            #: run reported locus `no_divergence_observed` while its own
            #: `every_batch_size_bit_identical` was False -- the report
            #: contradicting itself, which is the exact failure this function
            #: exists to prevent.
            "statistics_decomposition": got(
                "statistics_decomposition",
                "every_batch_size_bit_identical") is False,
            "ffn_selection": got(
                "ffn_selection", "selection_is_batch_invariant") is False,
        }.items() if value
    ]

    #: WHAT moves it, on an axis orthogonal to WHERE. The length sweep varies
    #: padded width at batch size 1 and batch size at fixed width, so it can
    #: separate the two knobs that the production path confounds.
    pad_moves = pad_inert is False
    driver = ({(True, True): "padded_width_and_batch_size",
               (True, False): "padded_width",
               (False, True): "batch_size",
               (False, False): "neither"}[(bool(pad_moves), bool(batch_moves))]
              if pad_inert is not None and batch_moves is not None else None)

    if deterministic is False:
        locus = "run_to_run_nondeterminism"
    elif not observed:
        locus = "no_divergence_observed"
    elif gemm_moves:
        locus = "gemm_level_shape_dependent_accumulation"
    elif some_backend and not every_backend:
        locus = "attention_backend_kernel"
    elif every_backend:
        locus = "below_attention_above_gemm"
    elif batch_moves:
        locus = "above_gemm_below_logits"
    #: Last, because it is the weakest localization: the sweep saw it but no
    #: component test did. It still beats `undetermined`, which was what the
    #: ladder returned for a run whose padding sweep had named the knob exactly.
    elif pad_moves:
        locus = "padded_width_dependent"
    #: Weaker still: the only stages that ran were the ones downstream of the
    #: forward, so they can say THAT it moved and nothing about where.
    elif observed:
        locus = "observed_but_not_localized"
    else:
        locus = "undetermined"

    return {
        "each_shape_internally_deterministic": deterministic,
        "any_divergence_observed": bool(observed),
        "stages_reporting_a_divergence": observed,
        "divergence_locus": locus,
        "divergence_driver": driver,
        "a_bare_gemm_is_shape_dependent": gemm_moves,
        "padding_alone_is_inert": pad_inert,
        "batch_size_alone_moves_the_forward": batch_moves,
        "focal_row_independent_of_neighbour_tokens": cross_row,
        "bf16_tuple_policy_supported": tuple_supported,
        "split_k_off_makes_every_gemm_exact": splitk_off_exact,
        "split_k_off_beats_reduced_precision_off_alone": splitk_off_better,
        "split_k_state_by_control": got("bf16_controls", "split_k_state"),
        "blas_library_by_control": got("bf16_controls", "blas_library"),
        "improvement_attributable_to": got("bf16_controls",
                                           "improvement_attributable_to"),
        "backend_switch_alone_fixes_it": got("bf16_controls",
                                             "backend_switch_alone_fixes_it"),
        "projections_still_shape_dependent_by_control": got(
            "bf16_controls", "projections_still_shape_dependent"),
        "historical_solo_output_is_preserved": solo_preserved,
        "all_three_shapes_agree_under_splitk_off": got(
            "solo_preservation", "all_three_shapes_agree_under_splitk_off"),
        "batch_under_splitk_off_equals_historical_solo": got(
            "solo_preservation", "batch_under_splitk_off_equals_historical_solo"),
        "ffn_selection_invariant_under_splitk_off": got(
            "operator_acceptance", "ffn_invariant_under_splitk_off"),
        "attention_selection_invariant_under_splitk_off": got(
            "operator_acceptance", "attention_invariant_under_splitk_off"),
        "both_operators_invariant_under_splitk_off": got(
            "operator_acceptance", "both_operators_invariant_under_splitk_off"),
        "invariant_batched_speedup_vs_solo": got(
            "performance", "invariant_batched_speedup_vs_solo"),
        "splitk_off_slowdown_vs_default_batched": got(
            "performance", "splitk_off_slowdown_vs_default_batched"),
        "every_attention_backend_diverges": every_backend,
        "no_attention_backend_diverges": no_backend,
        "attention_backends_diverging": got("backend_matrix", "backends_diverging"),
        "attention_backends_exact": got("backend_matrix", "backends_exact"),
        "attention_implementation_of_the_main_run":
            (report.get("model", {}).get("attention") or {}).get(
                "config._attn_implementation"),
        "ffn_selection_is_batch_invariant": selection_invariant,
        "ffn_layers_with_moved_selection":
            got("ffn_selection", "n_layers_with_moved_selection"),
        #: The headline pair is DEFAULT_MICRO_BATCH_SIZE at the recipe's keep
        #: ratio. This one is over EVERY swept ratio and size, so a reader
        #: cannot mistake "the configuration we run" for "any configuration".
        "ffn_selection_invariant_at_every_ratio_and_batch_size":
            got("ffn_selection",
                "selection_is_batch_invariant_at_every_ratio_and_size"),
        "ffn_headline_micro_batch_size":
            got("ffn_selection", "headline_micro_batch_size"),
        "ffn_headline_keep_ratio": got("ffn_selection", "headline_keep_ratio"),
        "statistics_bit_identical_at_every_batch_size":
            got("statistics_decomposition", "every_batch_size_bit_identical"),
        "causal_kl_max_rel_diff": got("causal_kl", "max_rel_diff"),
        "causal_kl_item_ranking_identical":
            got("causal_kl", "item_ranking_identical"),
        "_derivation": (
            "locus (WHERE): nondeterminism first; then, only if SOME stage "
            "observed a divergence at all, a bare GEMM, then a backend that "
            "diverges while another is exact, then every backend diverging, "
            "then anything between a GEMM and the logits, then padded width as "
            "the weakest localization. driver (WHAT) is independent and comes "
            "from the length sweep, which varies padded width and batch size "
            "one at a time."),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--n-items", type=int, default=None)
    ap.add_argument("--micro-batch-size", type=int, default=None)
    ap.add_argument("--only", default=None,
                    help="comma-separated stage names; default is all")
    ap.add_argument("--attn", default=None,
                    help="attn_implementation for the MAIN model (eager, sdpa). "
                         "Default is whatever transformers selects. The backend "
                         "matrix sweeps every backend regardless; this pins the "
                         "one the statistics, FFN selection and causal-KL stages "
                         "run under, since those are the stages whose answer is "
                         "a scientific decision rather than a logit norm.")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="run on CPU. A CPU run answers a DIFFERENT question "
                         "and is marked as such; it cannot close this one.")
    args = ap.parse_args(argv)

    import torch

    cfg = dict(DEFAULTS)
    for key, value in (("n_items", args.n_items),
                       ("micro_batch_size", args.micro_batch_size)):
        if value is not None:
            cfg[key] = value

    out_dir = Path(args.out) if args.out else (
        REPO / "artifacts/validation/batch_invariance" / args.run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        if not args.allow_cpu:
            say("no CUDA. This diagnostic is about accelerator arithmetic; "
                "pass --allow-cpu to run it anyway, and read the result as a "
                "different question.")
            return NOT_RUN
        device = "cpu"

    report: dict = {
        "run_id": args.run_id,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "question": ("does evaluating a calibration item alone and beside "
                     "others give the same scientific decision, and if not, "
                     "which operation first stops agreeing"),
        "device": device,
        "dtype": args.dtype,
        "answers_the_accelerator_question": device.startswith("cuda"),
        "config": cfg,
        "executable": executable_identity(),
        "environment": environment_identity(device),
        "stages": {},
        "stage_status": {},
    }

    say(f"device={device} dtype={args.dtype} torch={torch.__version__}")
    cfg["checkpoint"] = args.checkpoint or resolve_checkpoint(cfg)
    report["checkpoint"] = cfg["checkpoint"]
    say(f"checkpoint {cfg['checkpoint']}")

    model, adapter = load_model(cfg, device, args.dtype, attn=args.attn)
    report["requested_attn_implementation"] = args.attn
    report["model"] = {
        "hidden_size": int(model.config.hidden_size),
        "num_hidden_layers": int(model.config.num_hidden_layers),
        "intermediate_size": int(model.config.intermediate_size),
        "vocab_size": int(model.config.vocab_size),
        "attention": attention_implementation_of(model),
        "param_dtype": str(next(model.parameters()).dtype),
    }
    items, provenance = load_items(cfg, int(model.config.vocab_size))
    report["calibration"] = provenance
    focal, others = focal_and_neighbours(cfg, int(model.config.vocab_size))
    if provenance["kind"] == "frozen_calibration_mixture":
        # Prefer real tokens for the focal item too: the numbers a conclusion
        # cites should be about the mixture C3 would actually calibrate on.
        focal = items[0]["input_ids"]
        others = [i["input_ids"] for i in items[1:4]]
        say(f"calibration: frozen mixture {cfg['calibration_profile']} "
            f"({provenance['n_items']} items, lengths {provenance['lengths']})")
    else:
        # Loud, because the difference between these two is the difference
        # between a claim about C3's decisions and a claim about random ids.
        say("calibration: FROZEN MIXTURE UNAVAILABLE -- falling back to "
            f"synthetic token ids. {provenance.get('frozen_mixture_error')}")
        say("            the FFN-selection and causal-KL stages below are "
            "therefore about arithmetic, not about a C3 decision.")
    report["focal"] = {"length": int(focal.shape[1]),
                       "neighbour_lengths": [int(t.shape[1]) for t in others]}

    ctx = {"model": model, "adapter": adapter, "device": device,
           "dtype_name": args.dtype, "focal": focal, "others": others,
           "items": items, "report": report}

    wanted = set(args.only.split(",")) if args.only else None
    for name, fn in STAGES:
        if wanted and name not in wanted:
            report["stage_status"][name] = "skipped"
            continue
        say(f"stage {name}")
        try:
            report["stages"][name] = fn(cfg, ctx)
            report["stage_status"][name] = "ok"
        except Exception as exc:                                # noqa: BLE001
            import traceback
            report["stages"][name] = {"error": f"{type(exc).__name__}: {exc}",
                                      "traceback": traceback.format_exc()}
            report["stage_status"][name] = "error"
            say(f"  stage {name} FAILED: {type(exc).__name__}: {exc}")

    report["conclusion"] = derive_conclusion(report)
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    path = out_dir / "report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n")
    say(f"wrote {path}")
    say(f"locus = {report['conclusion']['divergence_locus']}")
    errors = [n for n, s in report["stage_status"].items() if s == "error"]
    if errors:
        say(f"stages that failed: {', '.join(errors)}")
        return FAILED
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
