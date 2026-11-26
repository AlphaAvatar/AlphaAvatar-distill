"""Hidden-state distillation at logical layers: the student's layer outputs against the teacher's, row by row.

**What is compared.** For each configured pair ``(student layer s, teacher layer t)``: the student's residual stream
after logical layer ``s`` and the teacher's after its layer ``t`` -- both the raw output of a whole decoder block
(attention, FFN, both norms and both residual adds), before any final norm. The final RMSNorm's output is not a
hidden-state target; it reaches the loss only through the output head (CE and KD). Both models run the same packed
input ids under the same position rule; the teacher runs without a graph and its states are targets only, never
inputs to the student.

**Rows.** The prediction positions of the KD term (``rows``: flat indices ``b * T + t`` into the ``[B, T]`` sequence,
``t`` the position whose state predicts token ``t + 1``) -- the same positions, with no extra offset, whose final
state reaches the output head for KD.

**The error** is the per-row relative squared error, ``||s(t) - h(t)||^2 / ||h(t)||^2``: the squared error of the
whole ``H``-dimensional state divided by the target's own energy at that row, so the term is dimensionless, the
hidden width cancels, and every (row, layer) counts once whatever the norm of the residual stream at that depth.
The trainer divides the sum by (number of pairs x the step's KD rows): a mean over layers and rows.

**Memory.** `relative_squared_error_sum` is its own autograd function: it keeps a reference to the student state (the
tensor the next block's checkpoint keeps anyway), the target rows and their inverse energies -- never a gathered
copy of the student rows -- and builds the ``[B, T, H]`` gradient only in the backward.

**No double registration.** The student's states are handed over by its own logical-layer loop
(`RecurrentDepthModel.forward(hidden_supervision=...)`), outside any checkpointed block call; the teacher's are taken
by forward hooks in a forward without a graph, each hooked layer required to run exactly once.

Nothing here names a model, a depth or a schedule: the pairs and the weight are the caller's configuration (P3).
"""
from __future__ import annotations

import contextlib
from collections.abc import Iterable, Mapping, Sequence

import torch

OBJECTIVE_ID = "hidden_state_relative_squared_error/v1"
ERRORS = ("relative_squared_error_v1",)
ROW_SETS = ("kd",)
REDUCTION_ID = ("sum over pairs and KD rows of ||s - h||^2 / ||h||^2, divided by (pairs x the step's KD rows); "
                "s the student's logical-layer output, h the teacher's layer output, both before the final norm")

__all__ = ["OBJECTIVE_ID", "ERRORS", "ROW_SETS", "REDUCTION_ID", "HiddenStateError", "validate_config",
           "objective_identity", "flat_rows", "relative_squared_error_sum", "HiddenSupervision",
           "capture_layer_outputs"]


class HiddenStateError(ValueError):
    pass


def validate_config(spec) -> dict:
    """``loss.hidden_state``: ``{weight > 0, pairs: [[student_layer, teacher_layer], ...], rows: "kd",
    error: "relative_squared_error_v1"}``; student layers unique, every index a non-negative int."""
    if not isinstance(spec, dict):
        raise HiddenStateError("loss.hidden_state must be an object")
    extra = sorted(set(spec) - {"weight", "pairs", "rows", "error"})
    if extra:
        raise HiddenStateError(f"loss.hidden_state has unknown keys {extra}")
    w = spec.get("weight")
    if not isinstance(w, (int, float)) or isinstance(w, bool) or not w > 0:
        raise HiddenStateError("loss.hidden_state.weight must be a number > 0")
    pairs = spec.get("pairs")
    if (not isinstance(pairs, list) or not pairs or not all(
            isinstance(p, list) and len(p) == 2 and all(isinstance(x, int) and not isinstance(x, bool) and x >= 0
                                                         for x in p) for p in pairs)):
        raise HiddenStateError("loss.hidden_state.pairs must be a non-empty list of [student_layer, teacher_layer]")
    students = [p[0] for p in pairs]
    if len(set(students)) != len(students):
        raise HiddenStateError("loss.hidden_state.pairs names a student layer twice")
    if spec.get("rows") not in ROW_SETS:
        raise HiddenStateError(f"loss.hidden_state.rows must be one of {ROW_SETS}")
    if spec.get("error") not in ERRORS:
        raise HiddenStateError(f"loss.hidden_state.error must be one of {ERRORS}")
    return spec


def objective_identity(spec: Mapping) -> dict:
    return {"objective": OBJECTIVE_ID, "error": spec["error"], "rows": spec["rows"], "weight": float(spec["weight"]),
            "pairs": [list(p) for p in spec["pairs"]], "reduction": REDUCTION_ID,
            "student_states": "logical-layer outputs from the model's own loop (no hook, no recomputed registration)",
            "teacher_states": "layer outputs by forward hook, no graph, each layer run exactly once"}


def flat_rows(mask: torch.Tensor) -> torch.Tensor:
    """``[B, T-1]`` prediction-position mask -> flat indices ``b * T + t`` into the ``[B, T]`` sequence."""
    b, tm1 = mask.shape
    idx = torch.nonzero(mask, as_tuple=False)
    return idx[:, 0] * (tm1 + 1) + idx[:, 1]


def _acc(dtype: torch.dtype) -> torch.dtype:
    """The accumulation dtype: float32, or float64 for a float64 state (never below float32)."""
    return torch.float64 if dtype == torch.float64 else torch.float32


class _RelativeSquaredErrorSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, rows, target, inv_energy):
        acc = _acc(hidden.dtype)
        h = hidden.reshape(-1, hidden.shape[-1])
        diff = h.index_select(0, rows).to(acc) - target.to(acc)
        out = ((diff * diff).sum(-1) * inv_energy.to(acc)).sum()
        ctx.save_for_backward(hidden, rows, target, inv_energy)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        hidden, rows, target, inv_energy = ctx.saved_tensors
        acc = _acc(hidden.dtype)
        h = hidden.reshape(-1, hidden.shape[-1])
        diff = h.index_select(0, rows).to(acc) - target.to(acc)
        g_rows = diff * (inv_energy.to(acc) * (2.0 * grad_out))[:, None]
        grad = torch.zeros_like(h)
        grad.index_add_(0, rows, g_rows.to(h.dtype))
        return grad.reshape(hidden.shape), None, None, None


def relative_squared_error_sum(hidden: torch.Tensor, rows: torch.Tensor, target: torch.Tensor,
                               inv_energy: torch.Tensor) -> torch.Tensor:
    """``sum_r ||hidden[rows[r]] - target[r]||^2 * inv_energy[r]`` (float32; float64 for a float64 state),
    differentiable in ``hidden``."""
    if rows.numel() == 0:
        return hidden.sum() * 0.0
    if hidden.requires_grad and torch.is_grad_enabled():
        return _RelativeSquaredErrorSum.apply(hidden, rows, target, inv_energy)
    with torch.no_grad():
        acc = _acc(hidden.dtype)
        h = hidden.reshape(-1, hidden.shape[-1]).index_select(0, rows).to(acc)
        return (((h - target.to(acc)) ** 2).sum(-1) * inv_energy.to(acc)).sum()


class HiddenSupervision:
    """The targets of one forward, handed to the student's logical-layer loop (``targets``, ``add``).

    ``rows``: flat row indices; ``targets``: student layer -> ``[N, H]`` teacher states at those rows (no graph).
    After the forward, ``loss_sum`` is the summed relative squared error over every pair and row (a graph-carrying
    float32 scalar), ``count`` the number of (pair, row) terms, ``by_layer`` each layer's detached sum."""

    def __init__(self, rows: torch.Tensor, targets: Mapping[int, torch.Tensor]):
        self.rows = rows
        self.targets = dict(targets)
        self.inv_energy = {}
        for p, t in self.targets.items():
            if t.shape[0] != rows.shape[0]:
                raise HiddenStateError(f"layer {p}: {t.shape[0]} target rows for {rows.shape[0]} rows")
            e = (t.to(_acc(t.dtype)) ** 2).sum(-1)
            if bool((e <= 0).any()):
                raise HiddenStateError(f"layer {p}: a target row with zero energy has no relative error")
            self.inv_energy[p] = 1.0 / e
        self.loss_sum: torch.Tensor | None = None
        self.by_layer: dict[int, torch.Tensor] = {}
        self.seen: list[int] = []

    @property
    def count(self) -> int:
        return int(self.rows.shape[0]) * len(self.targets)

    def add(self, layer: int, hidden: torch.Tensor) -> None:
        if layer in self.by_layer:
            raise HiddenStateError(f"logical layer {layer} handed over twice in one forward")
        s = relative_squared_error_sum(hidden, self.rows, self.targets[layer], self.inv_energy[layer])
        self.loss_sum = s if self.loss_sum is None else self.loss_sum + s
        self.by_layer[layer] = s.detach()
        self.seen.append(layer)

    def finish(self) -> None:
        """Every target layer must have been handed over exactly once."""
        missing = sorted(set(self.targets) - set(self.by_layer))
        if missing:
            raise HiddenStateError(f"the student's forward handed over no state for layers {missing}")


@contextlib.contextmanager
def capture_layer_outputs(model, layers: Iterable[int], rows: torch.Tensor):
    """Forward hooks on ``model.get_decoder().layers[l]`` for each ``l``: yields a dict filled, during one forward
    without a graph, with ``l -> [N, H]`` outputs at ``rows`` (the layer's own dtype). A layer run twice in that
    forward is refused (its output would be ambiguous)."""
    decoder = model.get_decoder()
    mods = decoder.layers
    got: dict[int, torch.Tensor] = {}
    handles = []

    def hook_for(l):
        def hook(_m, _args, out):
            if l in got:
                raise HiddenStateError(f"teacher layer {l} ran twice in one forward")
            h = out[0] if isinstance(out, tuple) else out
            got[l] = h.detach().reshape(-1, h.shape[-1]).index_select(0, rows)
        return hook

    for l in sorted(set(int(x) for x in layers)):
        if not 0 <= l < len(mods):
            raise HiddenStateError(f"teacher layer {l} is not one of its {len(mods)} layers")
        handles.append(mods[l].register_forward_hook(hook_for(l)))
    try:
        yield got
    finally:
        for h in handles:
            h.remove()


def pairs_of(spec: Mapping) -> list[tuple[int, int]]:
    return [(int(s), int(t)) for s, t in spec["pairs"]]


def summarize_by_layer(sums: Mapping[int, float], rows: int) -> dict[str, float]:
    """Per student layer, the mean relative squared error over ``rows`` (for logs)."""
    return {str(k): round(float(v) / rows, 6) for k, v in sorted(sums.items())} if rows else {}


def teacher_layers(pairs: Sequence[tuple[int, int]]) -> list[int]:
    return sorted({t for _, t in pairs})
