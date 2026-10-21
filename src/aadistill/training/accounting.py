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
  epochs, and the reasoning / answer split of U and E.

The trainer's own schedule rule is mirrored, not reinvented: ``steps =
ceil(blocks * epochs / blocks_per_step)`` and each step consumes
``blocks_per_step`` blocks. Exposure per pass uses the pack's mean supervision
per block, because which blocks the wrapped last step repeats depends on the
trainer's per-epoch shuffle; the exact figure is the trainer's own log.

FLOPs per executed token come from ``aadistill.runtime.cost
.forward_flops_per_token`` (projections, causal attention, MLP and the output
head); a backward pass is counted as twice a forward. Nothing here names a
model, a corpus or a rung.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from ..runtime.cost import forward_flops_per_token

ACCOUNTING_SCHEMA = "aadistill.training.accounting/v1"
BACKWARD_FACTOR = 2.0


def _flops_per_block(spec: Mapping, executed: int) -> float:
    return executed * forward_flops_per_token(spec, executed) if executed else 0.0


def training_accounting(blocks: Sequence[Mapping], *, epochs: float, blocks_per_step: int,
                        block_len: int, truncate_padding: bool,
                        student_spec: Mapping, teacher_spec: Mapping | None) -> dict:
    """U, E, C and the schedule for one pack under one run configuration.

    ``blocks`` rows carry ``real_length`` (unpadded tokens), ``supervised`` and,
    optionally, ``reasoning`` and ``answer`` (a split of ``supervised``).
    ``teacher_spec`` is None for a run with no online teacher.
    """
    n = len(blocks)
    if n == 0:
        raise ValueError("no blocks")
    if blocks_per_step < 1:
        raise ValueError("blocks_per_step must be >= 1")
    steps = math.ceil(n * epochs / blocks_per_step)
    passes = steps * blocks_per_step
    repeat = passes / n
    u = sum(int(b["supervised"]) for b in blocks)
    u_reason = sum(int(b.get("reasoning", 0)) for b in blocks)
    u_answer = sum(int(b.get("answer", 0)) for b in blocks)
    executed = [int(b["real_length"]) if truncate_padding else int(block_len) for b in blocks]
    real = sum(int(b["real_length"]) for b in blocks)
    stu = sum(_flops_per_block(student_spec, x) for x in executed) * (1 + BACKWARD_FACTOR)
    tea = sum(_flops_per_block(teacher_spec, x) for x in executed) if teacher_spec else 0.0
    return {
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


def matched_steps(blocks_reference: int, blocks_other: int, *, epochs: float,
                  reference_blocks_per_step: int) -> dict:
    """Blocks-per-step for a second pack so both take (about) the same steps.

    The reference keeps its own batch; the other pack's batch is the integer
    nearest to the reference's step count, and the realized mismatch is
    returned rather than hidden.
    """
    t_ref = math.ceil(blocks_reference * epochs / reference_blocks_per_step)
    bps = max(1, round(blocks_other * epochs / t_ref))
    t_other = math.ceil(blocks_other * epochs / bps)
    return {"reference_steps": t_ref, "other_blocks_per_step": bps,
            "other_steps": t_other, "step_ratio": round(t_other / t_ref, 4)}
