"""Arm R: student-prefix on-policy KD plus teacher recovery continuation.

One R example is `[prompt] + [student prefix] + [teacher recovery]`, with CE on
the recovery only and `kd_scope=all` putting KD over every real position —
including the student prefix. That is why the arm is named the way it is, and
why a result from it may not be attributed to the recovery continuation alone.

This module owns the two things that must not be improvised at generation time:

* **construction** — how the three pieces become one token stream and one loss
  mask, at token level, never by re-applying the chat template (which deletes
  earlier reasoning traces);
* **the ten registered quality gates** — each returning a machine-readable
  reason, because the rejection census is reported by task and source seed and a
  free-text failure cannot be counted.

Nothing here generates. The caller supplies token ids from the student and the
teacher, so the same code path serves the pilot and the full run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

GATES = (
    "exact_prefix_echo",          # teacher received, and kept, the supplied prefix
    "non_empty_continuation",
    "natural_termination",
    "no_context_limit",
    "roundtrip_stable",           # survives serialization and loader round-trip
    "mask_matches_supervision",   # CE-mask count == continuation length
    "not_held_out",               # evaluation and reserved splits excluded
    "not_duplicate_target",       # duplicate recovery targets measured/controlled
    "answer_valid",               # task verifier, where one exists
    "within_context_budget",
)


class GateFailure(ValueError):
    """A rejected R sample. `reason` is one of GATES and is always countable."""

    def __init__(self, reason: str, detail: str = ""):
        if reason not in GATES:
            raise ValueError(f"unregistered gate reason {reason!r}")
        self.reason = reason
        super().__init__(f"{reason}{': ' + detail if detail else ''}")


@dataclass
class RecoveryExample:
    """One built R example, before packing."""

    ids: list[int]
    mask: list[bool]
    n_prompt_tokens: int
    n_prefix_tokens: int          # student tokens kept
    n_continuation_tokens: int    # teacher tokens supervised
    source_session_id: str
    source_seed: str
    truncation_index: int
    truncation_fraction: float
    data_type: str
    meta: dict = field(default_factory=dict)

    @property
    def n_total_tokens(self) -> int:
        return len(self.ids)

    def to_record(self) -> dict:
        return {
            "id": f"{self.source_session_id}#r{self.truncation_index}",
            "source_session_id": self.source_session_id,
            "source_seed": self.source_seed,
            "truncation_index": self.truncation_index,
            "truncation_fraction": round(self.truncation_fraction, 4),
            "data_type": self.data_type,
            "arm": "R",
            "prefix_source": "student_generated",
            "n_prompt_tokens": self.n_prompt_tokens,
            "n_prefix_tokens": self.n_prompt_tokens + self.n_prefix_tokens,
            "n_student_prefix_tokens": self.n_prefix_tokens,
            "n_continuation_tokens": self.n_continuation_tokens,
            "n_total_tokens": self.n_total_tokens,
            **self.meta,
        }


def build_example(*, prompt_ids: list[int], student_prefix_ids: list[int],
                  teacher_continuation_ids: list[int], source_session_id: str,
                  source_seed: str, truncation_index: int,
                  truncation_fraction: float, data_type: str,
                  meta: dict | None = None) -> RecoveryExample:
    """Concatenate at token level and mask the recovery only.

    `n_prefix_tokens` in the emitted record is `prompt + student prefix`: that is
    the context the continuation is conditioned on, and it is the quantity the
    paired selector buckets and compares against C. The student-only count is
    kept beside it so the two are never confused.
    """
    if not student_prefix_ids:
        raise GateFailure("exact_prefix_echo", "empty student prefix")
    if not teacher_continuation_ids:
        raise GateFailure("non_empty_continuation", "no teacher tokens")
    ids = list(prompt_ids) + list(student_prefix_ids) + list(teacher_continuation_ids)
    mask = ([False] * (len(prompt_ids) + len(student_prefix_ids))
            + [True] * len(teacher_continuation_ids))
    return RecoveryExample(
        ids=ids, mask=mask,
        n_prompt_tokens=len(prompt_ids),
        n_prefix_tokens=len(student_prefix_ids),
        n_continuation_tokens=len(teacher_continuation_ids),
        source_session_id=source_session_id, source_seed=source_seed,
        truncation_index=truncation_index, truncation_fraction=truncation_fraction,
        data_type=data_type, meta=meta or {})


def check_gates(example: RecoveryExample, *, echoed_prefix_ids: list[int],
                student_prefix_ids: list[int], stop_ids: set[int] | frozenset[int],
                context_limit_hit: bool, block_len: int, n_system_tokens: int,
                held_out_ids: set[str], seen_targets: set[tuple],
                answer_ok: bool | None) -> None:
    """Run every registered gate; raise `GateFailure` on the first that fails.

    `echoed_prefix_ids` is what the engine reports it actually conditioned on.
    Comparing it to what was supplied is the only way to catch a serving path
    that silently re-tokenized, normalized or dropped the prefix — which would
    make the arm train on a state the student never visited.
    """
    if list(echoed_prefix_ids) != list(student_prefix_ids):
        raise GateFailure(
            "exact_prefix_echo",
            f"engine conditioned on {len(echoed_prefix_ids)} tokens, supplied "
            f"{len(student_prefix_ids)}")
    if example.n_continuation_tokens <= 0:
        raise GateFailure("non_empty_continuation")
    if example.ids[-1] not in stop_ids:
        raise GateFailure("natural_termination",
                          f"ends on {example.ids[-1]}, not a stop token")
    if context_limit_hit:
        raise GateFailure("no_context_limit")
    if example.source_session_id in held_out_ids:
        raise GateFailure("not_held_out", example.source_session_id)
    total = example.n_total_tokens + n_system_tokens
    if total > block_len:
        raise GateFailure("within_context_budget", f"{total} > {block_len}")
    if sum(example.mask) != example.n_continuation_tokens:
        raise GateFailure("mask_matches_supervision",
                          f"{sum(example.mask)} != {example.n_continuation_tokens}")
    key = (example.source_session_id, tuple(
        example.ids[-min(64, example.n_continuation_tokens):]))
    if key in seen_targets:
        raise GateFailure("not_duplicate_target", str(key[0]))
    if answer_ok is False:
        raise GateFailure("answer_valid")
    seen_targets.add(key)


def roundtrip_ok(example: RecoveryExample, reloaded_ids: list[int],
                 reloaded_mask: list[bool]) -> None:
    """Serialization and loader round-trip must be exact, or the gate fails."""
    if list(reloaded_ids) != list(example.ids):
        raise GateFailure("roundtrip_stable",
                          f"{len(reloaded_ids)} ids != {len(example.ids)}")
    if list(map(bool, reloaded_mask)) != list(map(bool, example.mask)):
        raise GateFailure("roundtrip_stable", "mask changed across round-trip")


def kd_decomposition(kd_per_token: list[float], mask: list[bool]) -> dict:
    """Split the KD signal into its prefix and continuation halves.

    `kd_scope=all` means KD covers context as well as target, so the arm's
    headline KD number blends two very different things: the teacher's opinion
    of the *student's own* prefix, and its opinion of its own recovery. Reporting
    only the blend hides how much signal the student-visited states actually
    carry, which is the thing R exists to add.

    `kd_per_token` is aligned to prediction positions, i.e. one entry per token
    from index 1 onward, which is how the trainer's KD mask is built.
    """
    pred_mask = mask[1:]
    if len(kd_per_token) != len(pred_mask):
        raise ValueError(
            f"kd_per_token has {len(kd_per_token)} entries, expected "
            f"{len(pred_mask)} prediction positions")
    cont = [v for v, m in zip(kd_per_token, pred_mask) if m]
    pre = [v for v, m in zip(kd_per_token, pred_mask) if not m]
    total = cont + pre
    return {
        "prefix_kd_tokens": len(pre),
        "continuation_kd_tokens": len(cont),
        "prefix_kd_mean": round(sum(pre) / len(pre), 6) if pre else None,
        "continuation_kd_mean": round(sum(cont) / len(cont), 6) if cont else None,
        "total_kd_mean": round(sum(total) / len(total), 6) if total else None,
        "prefix_share_of_kd_mass": (
            round(sum(pre) / sum(total), 4) if sum(total) else None),
        "prefix_share_of_kd_tokens": (
            round(len(pre) / len(total), 4) if total else None),
    }


def loss_attribution(*, ce_mean: float, kd_mean: float, ce_weight: float,
                     kd_weight: float, decomposition: dict) -> dict:
    """How much of the weighted loss the student-visited prefix states carry.

    The normalization is correct — proven separately on the real trainer — but
    correctness says nothing about magnitude. If prefix KD is a rounding error in
    the total loss, R's treatment is weaker than its description suggests and the
    write-up must say so.
    """
    w_ce = ce_weight * ce_mean
    w_kd = kd_weight * kd_mean
    total = w_ce + w_kd
    share = decomposition.get("prefix_share_of_kd_mass")
    prefix_kd_contribution = None if share is None else w_kd * share
    return {
        "weighted_ce_contribution": round(w_ce, 6),
        "weighted_kd_contribution": round(w_kd, 6),
        "total_weighted_loss": round(total, 6),
        "kd_share_of_total_loss": round(w_kd / total, 4) if total else None,
        "prefix_kd_contribution": (None if prefix_kd_contribution is None
                                   else round(prefix_kd_contribution, 6)),
        "prefix_kd_share_of_total_loss": (
            None if prefix_kd_contribution is None or not total
            else round(prefix_kd_contribution / total, 4)),
    }
