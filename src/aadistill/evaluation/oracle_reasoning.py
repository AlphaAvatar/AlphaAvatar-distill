"""Answer-only generation after gold reasoning is supplied.

`generation_mode = answer_only_after_gold_reasoning`.

The model is given the exact token prefix

    system/user prompt + template-preopened <think> + gold reasoning + </think>

and must autoregressively produce only `content + <|im_end|>`. That separates two
abilities the free rollout conflates: producing reliable reasoning, and producing
the answer once correct reasoning exists.

Finding the boundary
--------------------
The reasoning/content boundary is a property of the **chat template**, and this
module never goes looking for a `</think>` substring — a reasoning trace that
happens to discuss the literal string `</think>` would break that, and so would
any template change that alters the separator between the think block and the
answer.

Instead the session is rendered **three** times: once with its real content and
once with each of two probe strings whose first tokens differ. The two probe
renderings share exactly the structural prefix and diverge at the first content
token, so their longest common prefix *is* the boundary. It is then asserted to
be a prefix of the real rendering. This is template-aware, deterministic, and
uses the same `render_session` path training uses — no string concatenation and
no decode/re-encode anywhere.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from ..data.sessions import render_session

# Probes must tokenize to different FIRST tokens, and must not appear in any
# template control sequence.
_PROBE_A = "Qzx"
_PROBE_B = "Wvy"


class OracleBoundaryError(ValueError):
    """The structural boundary could not be established for this session."""


@dataclass
class OraclePrefix:
    """Everything the answer-only mode needs, plus what it asserts."""

    session_id: str
    data_type: str
    prefix_ids: list[int]       # prompt + <think> + gold reasoning + </think> + sep
    gold_answer_ids: list[int]  # the content span plus its terminator
    full_ids: list[int]         # the complete rendered session
    boundary: int               # index of the first content token in full_ids
    n_reasoning_tokens: int

    @property
    def total_tokens(self) -> int:
        return len(self.full_ids)


def _render_ids(tokenizer, session: dict, content: str | None) -> list[int]:
    """Render one session, optionally replacing the final assistant content."""
    if content is not None:
        session = copy.deepcopy(session)
        for m in reversed(session["messages"]):
            if m["role"] == "assistant":
                m["content"] = content
                break
    r = render_session(tokenizer, session)
    return list(r.body_ids)


def _common_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def build_oracle_prefix(tokenizer, session: dict) -> OraclePrefix:
    """Build the answer-only prefix, asserting every property that makes it valid."""
    full = _render_ids(tokenizer, session, None)
    pa = _render_ids(tokenizer, session, _PROBE_A)
    pb = _render_ids(tokenizer, session, _PROBE_B)

    boundary = _common_prefix(pa, pb)
    if boundary == 0:
        raise OracleBoundaryError("probe renderings diverge at token 0")
    if boundary >= len(full):
        raise OracleBoundaryError("structural prefix is not shorter than the render")
    # (1) the boundary must be a genuine prefix of the REAL rendering
    if full[:boundary] != pa[:boundary]:
        raise OracleBoundaryError("probe prefix is not a prefix of the real render")

    prefix = full[:boundary]
    answer = full[boundary:]
    # (2) prefix + remaining gold tokens must reconstruct the render exactly
    if prefix + answer != full:
        raise OracleBoundaryError("prefix + answer does not reconstruct the render")

    think_close = tokenizer.convert_tokens_to_ids("</think>")
    think_open = tokenizer.convert_tokens_to_ids("<think>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")

    # (3) the prefix must carry the assistant turn's structural close. Earlier
    # <|im_end|> tokens are legitimate -- they terminate the system and user
    # turns -- so what must be absent is a terminator AFTER the close, which
    # would mean the assistant turn had already ended.
    if think_close not in prefix:
        raise OracleBoundaryError("no structural </think> inside the prefix")
    close_at = len(prefix) - 1 - prefix[::-1].index(think_close)
    if im_end in prefix[close_at:]:
        raise OracleBoundaryError("prefix contains <|im_end|> after </think>")
    if think_close in answer:
        raise OracleBoundaryError("a </think> survives in the answer span")
    if think_open in answer:
        raise OracleBoundaryError("a <think> survives in the answer span")
    # (4) the answer span must end on the assistant terminator
    if im_end not in answer:
        raise OracleBoundaryError("answer span carries no <|im_end|>")

    open_at = (len(prefix) - 1 - prefix[::-1].index(think_open)
               if think_open in prefix else None)
    return OraclePrefix(
        session_id=session.get("id", ""),
        data_type=session.get("data_type", ""),
        prefix_ids=prefix,
        gold_answer_ids=answer,
        full_ids=full,
        boundary=boundary,
        n_reasoning_tokens=(close_at - open_at - 1) if open_at is not None
        else close_at,
    )


def fits(prefix: OraclePrefix, context: int, min_allowance: int) -> bool:
    """Does the complete prefix plus a real generation allowance fit the context?

    Reasoning is never truncated to make room: a session that cannot hold its
    whole prefix and a usable allowance is rejected and counted, because a
    truncated oracle prefix answers a different question.
    """
    return len(prefix.prefix_ids) + min_allowance <= context


def validate_answer_only(raw: str, *, im_end: str = "<|im_end|>",
                         think_open: str = "<think>",
                         think_close: str = "</think>") -> dict:
    """Validate a generation produced in answer-only mode.

    Everything generated here is answer continuation: the think block was already
    closed by the prefix. So the rules differ from the free-rollout validator —
    it must NOT expect another `<think>`, and re-opening one is leakage.
    """
    body = raw.split(im_end)[0]
    reopened = think_open in body
    leaked_close = think_close in body
    answer = body.strip()
    return {
        "terminated": im_end in raw,
        "protocol_valid": bool(im_end in raw and not reopened
                               and not leaked_close and answer),
        "reason": ("ok" if (im_end in raw and not reopened and not leaked_close
                            and answer)
                   else "reopened_think" if reopened
                   else "reasoning_leakage" if leaked_close
                   else "empty_answer" if not answer
                   else "not_terminated"),
        "reopened_think": reopened,
        "reasoning_leakage": leaked_close,
        "empty_answer": not answer,
        "answer": answer,
    }
