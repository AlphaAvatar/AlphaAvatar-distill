"""The Stage 2/3 primary behaviour metric: can the student roll out on its own?

Stage 2/3 exist to restore **stable autonomous rollout capability**, so that the
student can later produce usable trajectories for Stage 5/6 on-policy work. The
primary question is therefore not "is the answer right" but "did the model
produce a well-formed, self-terminating trajectory at all".

    usable_rollout = non_empty
                     AND natural_termination
                     AND no_severe_repetition
                     AND no_context_limit
                     AND protocol_valid

All five components are reported alongside the conjunction. The conjunction on
its own hides which failure dominates, and the components on their own hide that
a model can score well on each while rarely satisfying all five at once.

**This metric is deliberately blind to correctness.** A reply of "42" that opens
and closes `<think>` and stops on `<|im_end|>` is a perfectly usable rollout and
a useless answer. That is not a defect to be patched by folding correctness in —
it is why correctness is reported separately as the secondary axis, and why
`correct_given_usable` is reported as well. Do not combine them into one number.

Two record schemas are supported, both produced by existing harnesses:

* `three_mode` — `run_three_mode_diagnostic.py` free-mode generations, which
  already carry all five components;
* `behavior_v0` — the Experiment 1 / behaviour-wave generations, which carry
  termination, degeneration and context-limit but predate `protocol_valid`, so
  those two components are recomputed from the retained `raw` text.

Recomputing from retained text is a rescore, not a new measurement: the
generations are untouched and no model is run.
"""

from __future__ import annotations

from aadistill.evaluation.behavior import split_generation
from aadistill.evaluation.strict_answer import protocol_valid

COMPONENTS = (
    "non_empty",
    "natural_termination",
    "no_severe_repetition",
    "no_context_limit",
    "protocol_valid",
)


def _from_three_mode(rec: dict, *, think_preopened: bool) -> dict:
    return {
        "non_empty": not bool(rec["empty_answer"]),
        "natural_termination": bool(rec["natural_termination"]),
        "no_severe_repetition": not bool(rec["degenerate"]),
        "no_context_limit": not bool(rec["context_limit"]),
        "protocol_valid": bool(rec["protocol_valid"]),
    }


def _from_behavior_v0(rec: dict, *, think_preopened: bool) -> dict:
    raw = rec.get("raw") or ""
    valid, _ = protocol_valid(raw, think_preopened=think_preopened)
    answer = split_generation(raw, think_preopened=think_preopened)["answer"]
    return {
        "non_empty": bool(answer.strip()),
        "natural_termination": bool(rec["natural_termination"]),
        "no_severe_repetition": not bool(rec["degeneration_triggered"]),
        "no_context_limit": not bool(rec["context_limit_reached"]),
        "protocol_valid": valid,
    }


def detect_schema(rec: dict) -> str:
    """Name the record schema, or raise. Never guess silently."""
    if "empty_answer" in rec and "degenerate" in rec:
        return "three_mode"
    if "degeneration_triggered" in rec and "context_limit_reached" in rec:
        return "behavior_v0"
    raise ValueError(
        "unrecognised generation record; expected three_mode "
        f"(empty_answer/degenerate) or behavior_v0 "
        f"(degeneration_triggered/context_limit_reached), got keys {sorted(rec)}")


def components(rec: dict, *, think_preopened: bool = True,
               schema: str | None = None) -> dict:
    """The five component booleans for one generation record."""
    schema = schema or detect_schema(rec)
    if schema == "three_mode":
        out = _from_three_mode(rec, think_preopened=think_preopened)
    elif schema == "behavior_v0":
        out = _from_behavior_v0(rec, think_preopened=think_preopened)
    else:
        raise ValueError(f"unknown schema {schema!r}")
    assert set(out) == set(COMPONENTS), sorted(out)
    return out


def usable(rec: dict, *, think_preopened: bool = True,
           schema: str | None = None) -> bool:
    """True only when every component holds."""
    return all(components(rec, think_preopened=think_preopened,
                          schema=schema).values())


def summarize(records: list[dict], *, think_preopened: bool = True) -> dict:
    """Rates for the conjunction and every component, plus the failure census.

    `first_failure` attributes each unusable rollout to a single component in the
    fixed COMPONENTS order, so the census sums to the unusable count. It is a
    presentation aid: failures co-occur (a degeneration stop also fails natural
    termination), and the per-component rates above it are the honest view.
    """
    n = len(records)
    if not n:
        return {"n": 0}
    comp = [components(r, think_preopened=think_preopened) for r in records]
    out = {"n": n,
           "usable_rollout_rate": round(sum(all(c.values()) for c in comp) / n, 4)}
    for k in COMPONENTS:
        out[k] = round(sum(c[k] for c in comp) / n, 4)
    census = {}
    for c in comp:
        if all(c.values()):
            continue
        first = next(k for k in COMPONENTS if not c[k])
        census[first] = census.get(first, 0) + 1
    out["first_failure"] = dict(sorted(census.items(), key=lambda kv: -kv[1]))
    return out
