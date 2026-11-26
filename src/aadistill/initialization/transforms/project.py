"""Stage 1 projections derived from Stage 0 sufficient statistics.

The student residual stream is defined as ``y = P^T x`` for a single global
orthonormal projection ``P`` (d_teacher x d_student). Using one ``P`` for the
whole stream (rather than per-layer or per-group projections) keeps the
residual skip path exactly consistent: ``y_{l+1} = y_l + P^T (block delta)``
needs no change-of-basis anywhere. Per-group projections would insert a
rotation mismatch on the skip path at every group boundary that a standard
decoder graph has no place to absorb; that variant stays a logged baseline
candidate for later ablation.

``P`` is computed from the *uncentered* second moments ``E[x x^T]`` because
Stage 1 must reconstruct the actual signal (``P P^T x ~ x``), including its
large mean component — centering would discard the mean direction the network
relies on. Each residual point's moment is trace-normalized before averaging
so late layers (whose residual magnitudes are much larger) do not dominate.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from aadistill.initialization.statistics.collect import statistics_denominator


def uncentered_moment(state: dict[str, torch.Tensor], point: int) -> torch.Tensor:
    count = int(state["residual_count"][0])
    if count < 2:
        raise ValueError(f"Need at least 2 tokens, have {count}")
    return state["residual_sqsum"][point].to(torch.float64) / statistics_denominator(state)


def stream_weighting(spec, n_layers: int) -> tuple[list[int], list[float]]:
    """The residual collection points and their weights a CALLER declares, validated, for `stream_projection`.

    The core holds no weighting of its own: which stream points enter the projection and how much each counts is a
    recipe's choice, supplied by the configuration or the application that runs it, with where the numbers come
    from. ``spec`` is either

    * ``{"points": [p, ...], "weights": [w, ...], "source": ...}`` -- explicit points (each in ``0..n_layers``, the
      pre-norm stream states and ``n_layers`` the post-final-norm point; no repeats) and one weight each, or
    * ``{"endpoint_weights": {"first": a, "interior": b, "last": c}, "source": ...}`` -- every point
      ``0..n_layers``, the first weighted ``a``, the last ``c`` and each other ``b`` (one rule for any depth).

    Every weight is a finite number > 0. ``source`` (a non-empty string) is required. Returns ``(points, weights)``.
    """
    if not isinstance(spec, Mapping):
        raise ValueError("a stream weighting must be a mapping (the caller's declaration; the core has no default)")
    if not isinstance(spec.get("source"), str) or not spec["source"].strip():
        raise ValueError("a stream weighting must name its source (where the weights come from)")
    forms = [k for k in ("points", "endpoint_weights") if k in spec]
    if len(forms) != 1:
        raise ValueError("a stream weighting declares either points + weights or endpoint_weights")
    n = int(n_layers)
    if forms == ["endpoint_weights"]:
        ew = spec["endpoint_weights"]
        if not isinstance(ew, Mapping) or set(ew) != {"first", "interior", "last"}:
            raise ValueError("endpoint_weights must be {first, interior, last}")
        points = list(range(n + 1))
        weights = [ew["first"]] + [ew["interior"]] * (n - 1) + [ew["last"]]
    else:
        points, weights = spec["points"], spec.get("weights")
        if (not isinstance(points, list) or not points
                or not all(isinstance(x, int) and not isinstance(x, bool) for x in points)):
            raise ValueError("points must be a non-empty list of ints")
        if len(set(points)) != len(points) or not all(0 <= x <= n for x in points):
            raise ValueError(f"points must be distinct and in 0..{n}")
        if not isinstance(weights, list) or len(weights) != len(points):
            raise ValueError("weights must be a list with one weight per point")
    if not all(isinstance(w, (int, float)) and not isinstance(w, bool) and math.isfinite(w) and w > 0
               for w in weights):
        raise ValueError("every stream weight must be a finite number > 0")
    return list(points), [float(w) for w in weights]


def stream_projection(
    state: dict[str, torch.Tensor],
    d_student: int,
    points: list[int],
    weights: list[float] | None = None,
) -> tuple[torch.Tensor, dict]:
    """Global stream projection P (d_teacher x d_student), columns orthonormal.

    ``points`` selects which residual collection points contribute to the
    trace-normalized average; ``weights`` (default all-1) upweights points.
    In practice the mid-stream points are captured almost perfectly by any
    reasonable P (their outlier directions dominate each moment), while the
    embedding-output and post-final-norm points — the two interfaces wired to
    the tied embedding/head — are the worst-captured, so a recipe may upweight
    those ends; which points and weights is the caller's declaration
    (`stream_weighting`), never a core default.
    """
    if not points:
        raise ValueError("points must be non-empty")
    if weights is None:
        weights = [1.0] * len(points)
    if len(weights) != len(points):
        raise ValueError("weights must match points")
    d_teacher = state["residual_sqsum"].shape[-1]
    if not (0 < d_student <= d_teacher):
        raise ValueError(f"d_student {d_student} not in (0, {d_teacher}]")

    # On the statistics' device, in float64. `uncentered_moment` slices
    # `residual_sqsum`, so the summands land wherever the caller placed the
    # state — `composite.apply` does `stats_to(state, model_device(parent))`
    # before calling in, so on a pod that is cuda:0. A host accumulator here is
    # what killed a paid session, on the `+=` below, and no CPU
    # rehearsal can see it: on the dev box both operands are already host and
    # the arithmetic is correct. The dtype is unchanged; only the placement is.
    avg = torch.zeros(d_teacher, d_teacher, dtype=torch.float64,
                      device=state["residual_sqsum"].device)
    for p, w in zip(points, weights):
        m = uncentered_moment(state, p)
        avg += w * (m / m.trace())
    avg /= sum(weights)

    eigvals, eigvecs = torch.linalg.eigh(avg)  # ascending
    proj = eigvecs[:, -d_student:].flip(-1).contiguous()
    total = eigvals.sum()
    captured = eigvals[-d_student:].sum()
    diagnostics = {
        "points": points,
        "energy_captured_frac": (captured / total).item(),
        "top_eigenvalue": eigvals[-1].item(),
        "min_kept_eigenvalue": eigvals[-d_student].item(),
        # `proj` follows `avg`, so once that is placed correctly this identity
        # matrix is the NEXT host tensor to meet a device one. It was a latent
        # second defect on the same line of reasoning: fixing only the
        # accumulator would have moved the same RuntimeError eleven lines down
        # and cost another paid session to find.
        "orthonormality_error": (proj.T @ proj
                                 - torch.eye(d_student, dtype=torch.float64,
                                             device=proj.device))
        .abs()
        .max()
        .item(),
    }
    return proj, diagnostics


def ffn_neuron_importance(
    state: dict[str, torch.Tensor], layer: int, w_down: torch.Tensor
) -> torch.Tensor:
    """Per-neuron importance = E[|a_j|] * ||down_proj column j||_2.

    ``E[|a_j|]`` comes from the Stage 0 cache (down_proj input, i.e. the
    post-SwiGLU intermediate activation); the column norm weights it by how
    strongly the neuron can write to the residual stream.
    """
    n = statistics_denominator(state)
    mean_abs = state["ffn_abs_sum"][layer].to(torch.float64) / n
    col_norms = w_down.to(torch.float64).norm(dim=0)
    return mean_abs * col_norms


def final_norm_weights(
    state: dict[str, torch.Tensor],
    proj: torch.Tensor,
    teacher_norm_weight: torch.Tensor,
    post_norm_point: int,
) -> torch.Tensor:
    """Least-squares diagonal for the student final RMSNorm.

    The teacher's final norm weight cannot be folded into the lm head when
    embeddings are tied (folding would break the tie), and an elementwise
    weight does not transfer into the rotated student basis. Instead choose
    the diagonal ``w_s`` minimizing ``E || w_s * (P^T z) - P^T (w_f * z) ||^2``
    over normalized final-stream states ``z``:

        w_s[i] = (P_i^T C diag(w_f) P_i) / (P_i^T C P_i)

    where ``C = E[z z^T]`` is recovered from the cached post-final-norm moment
    ``C_post = diag(w_f) C diag(w_f)``. The per-token RMS factor cancels as a
    common weight inside the expectation.
    """
    w_f = teacher_norm_weight.to(torch.float64)
    if (w_f.abs() < 1e-8).any():
        raise ValueError("Teacher final norm has ~zero entries; cannot invert fold")
    c_post = uncentered_moment(state, post_norm_point)
    inv = 1.0 / w_f
    c = inv[:, None] * c_post * inv[None, :]
    p64 = proj.to(torch.float64)
    num = (p64 * (c @ (w_f[:, None] * p64))).sum(0)
    den = (p64 * (c @ p64)).sum(0)
    if (den <= 0).any():
        raise ValueError("Non-positive projected variance in final-norm solve")
    return num / den
