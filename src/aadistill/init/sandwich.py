"""Stage 1 sandwich initialization: teacher weights -> student weights.

Every linear that reads the residual stream is initialized as
``scale * S W diag(w_norm) P`` and every linear that writes it as
``P^T W S^T`` where:

- ``P`` is the global stream projection (see project.py);
- ``diag(w_norm)`` folds the preceding RMSNorm's elementwise weight into the
  linear exactly (the student's corresponding norm weight is set to ones);
- ``S`` selects structure (Q-head selection for attention, top-k neuron
  selection for the FFN intermediate);
- ``scale = sqrt(d_teacher / d_student)`` compensates RMSNorm's 1/sqrt(d)
  factor: for energy-preserving P, the student's normalized stream state is
  ``sqrt(d_s/d_t) * P^T x_hat``, so in-projections are scaled up to match the
  teacher's post-norm magnitudes. Residual-writing projections need no scale.

RMSNorm is scale-invariant, so the remaining init error is directional
(energy outside span(P), dropped heads/neurons/layers), not a global scale.

Depth mapping (36 -> 28 style): the first ``2*(T-S)`` teacher layers are
merged pairwise (representative = the later layer of each pair, whose output
feeds the next span) and the remaining teacher layers map 1:1, so late
teacher layers are compressed least aggressively per AGENTS.md Stage 1.

KV heads and head_dim must match between teacher and student (Q heads may be
subsampled within each GQA group, preserving grouping); anything else fails
loudly rather than silently interpolating.
"""

from __future__ import annotations

import math

import torch

from .project import ffn_neuron_importance, final_norm_weights, stream_projection


def depth_span_map(teacher_layers: int, student_layers: int) -> list[dict]:
    """Map each student layer to a teacher span and a representative layer.

    Pairs are merged in the *middle band*: ~1/5 of the surviving 1:1 layers
    stay before the band and the rest after, so both the earliest and the
    late layers map 1:1. Layer-drop studies (Gromov et al. 2024,
    arXiv:2403.17887) and this project's single-axis ablation (2026-07-14
    Stage 1 experiment log) agree: merging the early band collapsed the model
    (holdout NLL 10.48) while the middle band kept it close to the teacher
    (3.88). The representative is the FIRST layer of each span — its expected
    input matches the incoming stream state (last-of-span ablated to 7.54).
    """
    n_merge = teacher_layers - student_layers
    if n_merge < 0:
        raise ValueError(f"Student deeper than teacher: {student_layers} > {teacher_layers}")
    if 2 * n_merge > teacher_layers:
        raise ValueError(
            f"Pairwise-merge map cannot compress {teacher_layers} -> {student_layers}"
        )
    head_keep = 0 if n_merge == 0 else max(1, round(0.2 * (teacher_layers - 2 * n_merge)))
    spans = []
    for s in range(head_keep):
        spans.append({"student": s, "teacher_span": [s, s + 1], "representative": s})
    for i in range(n_merge):
        a = head_keep + 2 * i
        spans.append({"student": head_keep + i, "teacher_span": [a, a + 2],
                      "representative": a})
    band_end = head_keep + 2 * n_merge
    for k in range(teacher_layers - band_end):
        spans.append({"student": head_keep + n_merge + k,
                      "teacher_span": [band_end + k, band_end + k + 1],
                      "representative": band_end + k})
    return spans


def select_q_heads(
    w_q: torch.Tensor, w_o: torch.Tensor,
    t_q_heads: int, t_kv_heads: int, s_q_heads: int, head_dim: int,
) -> list[int]:
    """Pick student Q heads per GQA group by a weight-magnitude proxy.

    Importance of head h = ||W_q rows of h||_F * ||W_o columns of h||_F —
    how strongly the head can form queries and write back to the stream.
    Activation-based head importance would need attention hooks that Stage 0
    does not cache yet; logged as a future baseline.
    """
    if t_q_heads % t_kv_heads or s_q_heads % t_kv_heads:
        raise ValueError("Q heads must be divisible by KV heads (GQA grouping)")
    per_g_t = t_q_heads // t_kv_heads
    per_g_s = s_q_heads // t_kv_heads
    if per_g_s > per_g_t:
        raise ValueError(f"Cannot keep {per_g_s} of {per_g_t} Q heads per group")
    scores = torch.tensor([
        w_q[h * head_dim:(h + 1) * head_dim, :].norm().item()
        * w_o[:, h * head_dim:(h + 1) * head_dim].norm().item()
        for h in range(t_q_heads)
    ])
    kept: list[int] = []
    for g in range(t_kv_heads):
        group = list(range(g * per_g_t, (g + 1) * per_g_t))
        top = sorted(group, key=lambda h: -scores[h].item())[:per_g_s]
        kept.extend(sorted(top))
    return kept


def _head_rows(heads: list[int], head_dim: int) -> torch.Tensor:
    return torch.tensor([h * head_dim + i for h in heads for i in range(head_dim)])


def _in_proj(w: torch.Tensor, norm_w: torch.Tensor, proj: torch.Tensor, scale: float) -> torch.Tensor:
    """scale * W diag(norm_w) P, computed in float64."""
    return scale * ((w.to(torch.float64) * norm_w.to(torch.float64)[None, :]) @ proj)


@torch.no_grad()
def init_student(
    teacher, student, state: dict[str, torch.Tensor],
    proj_override: torch.Tensor | None = None,
) -> dict:
    """Initialize a student Qwen3-style model in place from the teacher.

    Returns a diagnostics record for the run manifest. All math is float64;
    results are cast to the student's parameter dtype at assignment.

    ``proj_override`` replaces the activation-PCA stream projection with a
    caller-supplied orthonormal basis; used by tests (identity projection at
    equal width must reproduce the teacher exactly) and future ablations.
    """
    t_cfg, s_cfg = teacher.config, student.config
    if s_cfg.num_key_value_heads != t_cfg.num_key_value_heads:
        raise ValueError("KV head count must match teacher (KV selection not implemented)")
    if getattr(s_cfg, "head_dim", None) != getattr(t_cfg, "head_dim", None):
        raise ValueError("head_dim must match teacher (RoPE basis compatibility)")
    if s_cfg.vocab_size != t_cfg.vocab_size:
        raise ValueError("vocab_size must match teacher (same tokenizer)")
    if not s_cfg.tie_word_embeddings:
        raise ValueError("Recipe v0 assumes tied student embeddings")

    d_t, d_s = t_cfg.hidden_size, s_cfg.hidden_size
    head_dim = t_cfg.head_dim
    scale = math.sqrt(d_t / d_s)
    dtype = student.model.embed_tokens.weight.dtype

    # All pre-norm stream states plus the post-final-norm point, with the two
    # end points upweighted: the tied embedding reads point 0 and the tied lm
    # head reads point N (post-norm), and both are otherwise the worst-captured
    # points (0.74/0.75 vs >0.94 mid-stream on Qwen3-4B). Weights 9/8 chosen by
    # the 2026-07-14 width-only ablation (holdout top-1 0.036 -> 0.082).
    n_l = t_cfg.num_hidden_layers
    stream_points = list(range(n_l + 1))
    point_weights = [9.0] + [1.0] * (n_l - 1) + [8.0]
    if proj_override is not None:
        if proj_override.shape != (d_t, d_s):
            raise ValueError(f"proj_override shape {tuple(proj_override.shape)} != ({d_t}, {d_s})")
        proj = proj_override.to(torch.float64)
        proj_diag = {"override": True, "points": stream_points}
    else:
        proj, proj_diag = stream_projection(state, d_s, stream_points, point_weights)
        proj_diag["point_weights"] = point_weights

    spans = depth_span_map(t_cfg.num_hidden_layers, s_cfg.num_hidden_layers)
    t_layers, s_layers = teacher.model.layers, student.model.layers
    layer_records = []
    for span in spans:
        tl = t_layers[span["representative"]]
        sl = s_layers[span["student"]]

        w_att_norm = tl.input_layernorm.weight
        kept_q = select_q_heads(
            (tl.self_attn.q_proj.weight.to(torch.float64)
             * w_att_norm.to(torch.float64)[None, :]),
            tl.self_attn.o_proj.weight.to(torch.float64),
            t_cfg.num_attention_heads, t_cfg.num_key_value_heads,
            s_cfg.num_attention_heads, head_dim,
        )
        q_rows = _head_rows(kept_q, head_dim)
        sl.self_attn.q_proj.weight.copy_(
            _in_proj(tl.self_attn.q_proj.weight, w_att_norm, proj, scale)[q_rows].to(dtype))
        sl.self_attn.k_proj.weight.copy_(
            _in_proj(tl.self_attn.k_proj.weight, w_att_norm, proj, scale).to(dtype))
        sl.self_attn.v_proj.weight.copy_(
            _in_proj(tl.self_attn.v_proj.weight, w_att_norm, proj, scale).to(dtype))
        sl.self_attn.o_proj.weight.copy_(
            (proj.T @ tl.self_attn.o_proj.weight.to(torch.float64)[:, q_rows]).to(dtype))
        sl.self_attn.q_norm.weight.copy_(tl.self_attn.q_norm.weight.to(dtype))
        sl.self_attn.k_norm.weight.copy_(tl.self_attn.k_norm.weight.to(dtype))
        sl.input_layernorm.weight.fill_(1.0)

        importance = ffn_neuron_importance(
            state, span["representative"], tl.mlp.down_proj.weight)
        kept_n = torch.topk(importance, s_cfg.intermediate_size).indices.sort().values
        w_ffn_norm = tl.post_attention_layernorm.weight
        sl.mlp.gate_proj.weight.copy_(
            _in_proj(tl.mlp.gate_proj.weight, w_ffn_norm, proj, scale)[kept_n].to(dtype))
        sl.mlp.up_proj.weight.copy_(
            _in_proj(tl.mlp.up_proj.weight, w_ffn_norm, proj, scale)[kept_n].to(dtype))
        sl.mlp.down_proj.weight.copy_(
            (proj.T @ tl.mlp.down_proj.weight.to(torch.float64)[:, kept_n]).to(dtype))
        sl.post_attention_layernorm.weight.fill_(1.0)

        layer_records.append({
            "student_layer": span["student"],
            "teacher_span": span["teacher_span"],
            "representative": span["representative"],
            "kept_q_heads": kept_q,
            "ffn_kept_frac": len(kept_n) / t_cfg.intermediate_size,
        })

    student.model.embed_tokens.weight.copy_(
        (teacher.model.embed_tokens.weight.to(torch.float64) @ proj).to(dtype))
    student.tie_weights()

    w_final = final_norm_weights(
        state, proj, teacher.model.norm.weight, post_norm_point=t_cfg.num_hidden_layers)
    student.model.norm.weight.copy_((scale * w_final).to(dtype))

    return {
        "projection": proj_diag,
        "scale_compensation": scale,
        "depth_map": [
            {k: r[k] for k in ("student_layer", "teacher_span", "representative")}
            for r in layer_records
        ],
        "kept_q_heads": {r["student_layer"]: r["kept_q_heads"] for r in layer_records},
        "final_norm_weight_range": [
            w_final.min().item(), w_final.max().item()
        ],
    }
