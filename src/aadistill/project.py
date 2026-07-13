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

import torch


def uncentered_moment(state: dict[str, torch.Tensor], point: int) -> torch.Tensor:
    n = int(state["residual_count"][0])
    if n < 2:
        raise ValueError(f"Need at least 2 tokens, have {n}")
    return state["residual_sqsum"][point].to(torch.float64) / n


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
    the tied embedding/head — are the worst-captured, so callers upweight
    those ends (see init_student).
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

    avg = torch.zeros(d_teacher, d_teacher, dtype=torch.float64)
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
        "orthonormality_error": (proj.T @ proj - torch.eye(d_student, dtype=torch.float64))
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
    n = int(state["residual_count"][0])
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
