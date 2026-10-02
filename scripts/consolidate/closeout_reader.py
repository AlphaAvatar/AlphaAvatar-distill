#!/usr/bin/env python3
"""WHERE A CLOSEOUT KEEPS ITS VERDICT AND ITS COST, for every reader.

This existed twice — once in `render_log_navigation.py`, which fills the
snapshot's `latest_run` block, and once inline in `record_run_index.py`, which
classifies a run as unrecorded and says why. Both knew about one family of
closeouts and silently DEGRADED on the others, and the same defect was then
found three times:

* eight `phase_c2_behavioural` closeouts rendered as "no classification
  stated" with no dollars, while their own records plainly said
  `RETIRED_AT_ZERO_NO_RESOURCE_ACQUIRED` and `$0.0000`;
* `a3_attempt38`, the run that finished A3, rendered as the raw driver exit
  string `DRIVER_EXITED:0` with `$1.43` dropped, and the run index put all 38
  A3 runs in the "predates the run-manifest convention" bucket even though
  every one of them states a status;
* nine engineering-validation closeouts under `c2_full_search_cuda`,
  `c2_full_search_perf` and `c2_state_eval_cert` keep theirs under `verdict`
  and had never once been read, because none had ever been the newest run of
  the experiment a snapshot named.

Each time, the fix was to add a key name to a tuple — in one of the two
copies. So the tuples live here and both readers import them. A fourth family
now needs one edit, in one place, and the test that checks the readers against
EVERY closeout in the tree (`tests/docs/test_log_organisation.py`) covers both
callers at once.

Ordered alternatives rather than a schema registry: the point is to read what
is there, and a silent degradation is worse than either a loud refusal or a
second key name.
"""
from __future__ import annotations

#: PRECEDENCE MATTERS. `terminal` is a driver exit descriptor, not a
#: classification — reaching it while the record also states a status is the
#: defect, not the fallback. So it stays last.
VERDICT_KEYS: tuple[str, ...] = (
    "classification",   # phase_c2, baseline_completion, full_search, replay
    "status",           # phase_a3, phase_c3
    "verdict",          # the engineering validations
    "terminal",         # last resort: a driver exit code
)

#: `cost.actual_usd` goes LAST so every closeout that already rendered keeps
#: the figure it rendered: several records carry both `budget.this_attempt`
#: and `cost.actual_usd`, and the first was already authoritative for them.
COST_PATHS: tuple[tuple[str, ...], ...] = (
    ("budget", "this_attempt"),      # phase_c2, baseline_completion, full_search, replay
    ("cost", "this_attempt"),        # replay
    ("money", "all_in_usd"),         # behavioural
    ("money", "spent_usd"),          # behavioural, before all_in_usd existed
    ("cost", "actual_usd"),          # phase_a3, phase_c3
)


def closeout_verdict(doc: dict) -> str | None:
    """What this closeout says happened, or None if it says nothing."""
    for key in VERDICT_KEYS:
        value = doc.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def closeout_cost(doc: dict) -> float | None:
    """What this attempt cost, or None if the record states no figure.

    A stated `null` means a pod existed and no teardown cost could be read
    from the log; its own `_null_means` says it is NOT zero. Returning None
    keeps that distinction, because reading an unknown as zero understates
    spend — the one direction that matters.
    """
    for path in COST_PATHS:
        value: object = doc
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        #: `bool` is an `int`; a flag must never be rendered as a dollar figure.
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None
