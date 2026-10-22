"""What a recovery run trains on and what it costs, counted from its own blocks.

Data-scaling and data-scheme comparisons need quantities that a nominal
"2.96M tokens" hides, and they do not move together when two corpora differ in
supervision density:

* **U** -- unique supervised positions: each training session's supervised
  target counted once, whatever context it carries;
* **E** -- supervised exposure: supervised positions actually processed, with
  every repetition (epochs, the wrap of the last step) counted;
* **C** -- training compute: every EXECUTED position, through the student's
  forward and backward and the teacher's forward, including prompt context,
  earlier sessions in the block, padding when it is forwarded, attention and
  the output heads;
* optimizer steps, supervised positions per step (the effective batch),
  epochs, and the reasoning / answer split of U and E;
* KD positions by kind -- prompt (real positions that are not supervised:
  system, user, tool and earlier-turn context), reasoning and answer -- under
  the run's KD scope.

The trainer's own schedule rule is mirrored, not reinvented: ``steps =
ceil(blocks * epochs / blocks_per_step)`` unless a run fixes its steps, and
each step consumes ``blocks_per_step`` blocks. ``training_accounting``'s
exposure uses the pack's mean supervision per block (a PLAN figure);
``consumed_exposure`` sums the blocks a run actually consumes, given the
trainer's own block stream for its seed (the EXACT figure). ``match_schedule``
is the rule that sets two or more packs to the same step count.

FLOPs per executed token come from ``aadistill.runtime.cost
.forward_flops_per_token`` (projections, causal attention, MLP and the output
head); a backward pass is counted as twice a forward. Nothing here names a
model, a corpus or a rung.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from ..runtime.cost import forward_flops_per_token

ACCOUNTING_SCHEMA = "aadistill.training.accounting/v2"
BACKWARD_FACTOR = 2.0


def _flops_per_block(spec: Mapping, executed: int) -> float:
    return executed * forward_flops_per_token(spec, executed) if executed else 0.0


def _kd_split(blocks: Sequence[Mapping], kd_scope: str | None) -> dict:
    """KD positions per pass, by kind. Scope ``all`` is every real prediction
    position (the trainer's ``prediction_mask``: real tokens after the first);
    ``assistant`` is the supervised positions only."""
    if kd_scope is None:
        return {}
    if kd_scope not in ("all", "assistant"):
        raise ValueError(f"unknown kd scope {kd_scope!r}")
    reason = sum(int(b.get("reasoning", 0)) for b in blocks)
    answer = sum(int(b.get("answer", 0)) for b in blocks)
    sup = sum(int(b["supervised"]) for b in blocks)
    prompt = sum(max(0, int(b["real_length"]) - 1 - int(b["supervised"])) for b in blocks) \
        if kd_scope == "all" else 0
    return {"prompt": prompt, "reasoning": reason, "answer": answer,
            "unsplit_supervised": sup - reason - answer}


def training_accounting(blocks: Sequence[Mapping], *, epochs: float, blocks_per_step: int,
                        block_len: int, truncate_padding: bool,
                        student_spec: Mapping, teacher_spec: Mapping | None,
                        steps: int | None = None, kd_scope: str | None = None) -> dict:
    """U, E, C and the schedule for one pack under one run configuration.

    ``blocks`` rows carry ``real_length`` (unpadded tokens), ``supervised`` and,
    optionally, ``reasoning`` and ``answer`` (a split of ``supervised``).
    ``teacher_spec`` is None for a run with no online teacher. ``steps`` fixes
    the step count (a matched schedule); otherwise the trainer's ceil rule over
    ``epochs`` sets it. E here is the PLAN figure (mean supervision per block).
    """
    n = len(blocks)
    if n == 0:
        raise ValueError("no blocks")
    if blocks_per_step < 1:
        raise ValueError("blocks_per_step must be >= 1")
    steps = int(steps) if steps is not None else math.ceil(n * epochs / blocks_per_step)
    passes = steps * blocks_per_step
    repeat = passes / n
    u = sum(int(b["supervised"]) for b in blocks)
    u_reason = sum(int(b.get("reasoning", 0)) for b in blocks)
    u_answer = sum(int(b.get("answer", 0)) for b in blocks)
    executed = [int(b["real_length"]) if truncate_padding else int(block_len) for b in blocks]
    real = sum(int(b["real_length"]) for b in blocks)
    stu = sum(_flops_per_block(student_spec, x) for x in executed) * (1 + BACKWARD_FACTOR)
    tea = sum(_flops_per_block(teacher_spec, x) for x in executed) if teacher_spec else 0.0
    kd = _kd_split(blocks, kd_scope)
    out = {
        "schema": ACCOUNTING_SCHEMA,
        "blocks": n, "epochs": round(repeat, 6), "blocks_per_step": blocks_per_step,
        "optimizer_steps": steps, "block_passes": passes,
        "U_supervised": u, "U_reasoning": u_reason, "U_answer": u_answer,
        "E_supervised": round(u * repeat), "E_reasoning": round(u_reason * repeat),
        "E_answer": round(u_answer * repeat),
        "supervised_per_step": round(u * repeat / steps, 1),
        "real_tokens_per_epoch": real,
        "padding_per_epoch": n * block_len - real,
        "executed_positions": round(sum(executed) * repeat),
        "executed_positions_per_step": round(sum(executed) * repeat / steps, 1),
        "padding_executed": 0 if truncate_padding else round((n * block_len - real) * repeat),
        "C_student_flops": stu * repeat,
        "C_teacher_flops": tea * repeat,
        "C_total_flops": (stu + tea) * repeat,
        "truncate_padding": truncate_padding,
    }
    if kd:
        out["kd_scope"] = kd_scope
        out["KD_positions_per_pass"] = kd
        out["KD_positions_exposure"] = {k: round(v * repeat) for k, v in kd.items()}
    return out


def consumed_exposure(blocks: Sequence[Mapping], consumed: Sequence[int], *,
                      kd_scope: str | None = None) -> dict:
    """EXACT exposure of one run: the sums over the blocks it consumes, in the
    order its trainer consumes them (``consumed`` -- block indices, repeats
    included, e.g. the trainer's seeded stream for ``steps * blocks_per_step``
    positions). U is the blocks touched at least once."""
    touched = set(consumed)
    if any(i < 0 or i >= len(blocks) for i in touched):
        raise ValueError("consumed index outside the pack")

    def total(rows, key):
        return sum(int(rows[i].get(key, 0)) for i in consumed)

    kd = _kd_split([blocks[i] for i in consumed], kd_scope)
    out = {"block_passes": len(consumed), "blocks_touched": len(touched),
           "U_supervised_touched": sum(int(blocks[i]["supervised"]) for i in touched),
           "E_supervised": total(blocks, "supervised"), "E_reasoning": total(blocks, "reasoning"),
           "E_answer": total(blocks, "answer"), "real_positions": total(blocks, "real_length")}
    if kd:
        out["kd_scope"] = kd_scope
        out["KD_positions_exposure"] = kd
    return out


def match_schedule(n_blocks: Mapping[str, int], *, epochs: float,
                   reference_blocks_per_step: int, search: float = 0.1) -> dict:
    """One step count T for every pack, and an integer blocks-per-step each.

    The rule, fixed before any pack it is applied to exists:

    1. the REFERENCE is the pack with the fewest blocks (ties: name order) and
       keeps ``reference_blocks_per_step``; T0 = ceil(n_ref * epochs / bps_ref);
    2. for every T in [floor(T0 (1 - search)), ceil(T0 (1 + search))], each
       other pack takes bps = max(1, round(n * epochs / T)), and its passes are
       T * bps / n;
    3. the chosen T minimizes the largest relative deviation of any pack's
       passes from ``epochs``; ties go to the T nearest T0, then the smaller T.

    Every pack then runs exactly T steps, so the learning-rate schedule is
    identical; the passes each pack realizes are returned, not hidden.
    """
    if not n_blocks or min(n_blocks.values()) < 1:
        raise ValueError("every pack needs at least one block")
    ref = min(n_blocks, key=lambda k: (n_blocks[k], k))
    bps_ref = int(reference_blocks_per_step)
    t0 = math.ceil(n_blocks[ref] * epochs / bps_ref)
    lo, hi = max(1, math.floor(t0 * (1 - search))), math.ceil(t0 * (1 + search))
    best = None
    for t in range(lo, hi + 1):
        bps = {k: bps_ref if k == ref else max(1, round(n * epochs / t)) for k, n in n_blocks.items()}
        dev = max(abs(t * bps[k] / n - epochs) / epochs for k, n in n_blocks.items())
        key = (round(dev, 12), abs(t - t0), t)
        if best is None or key < best[0]:
            best = (key, t, bps)
    _, t, bps = best
    return {"rule": "aadistill.training.accounting.match_schedule/v1",
            "reference": ref, "reference_blocks_per_step": bps_ref, "T0": t0,
            "searched_steps": [lo, hi], "steps": t, "blocks_per_step": bps,
            "passes": {k: round(t * bps[k] / n, 6) for k, n in n_blocks.items()},
            "max_pass_deviation": round(best[0][0], 6)}
