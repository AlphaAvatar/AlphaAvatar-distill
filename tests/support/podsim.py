"""The pod simulator's free-space floor under the test suite, kept apart from a real sweep's.

A real sweep's floor is `record_pod_environment.min_free_gib()`, which the
recorder passes to the simulator because it knows the sweep's footprint; an
override there may raise it and never lower it. The suite's own simulator runs
move a synthetic tree of a few kilobytes, so the floor they need is the
suite's, not a launch's. ``AAD_TEST_PODSIM_MIN_FREE_GIB`` sets it for the suite
alone (unset: the simulator's own default).

The launch variables a developer may have exported (``PODSIM_MIN_FREE_GIB``,
``AAD_PODSIM_MIN_FREE_GIB``) never reach a test's simulator run, so neither
lowers a test's floor by accident nor changes what a default-value test reads.
"""
from __future__ import annotations

import os

TEST_VAR = "AAD_TEST_PODSIM_MIN_FREE_GIB"
LAUNCH_VARS = ("PODSIM_MIN_FREE_GIB", "AAD_PODSIM_MIN_FREE_GIB")


def simulator_env(**extra: str) -> dict[str, str]:
    """The environment a test runs the simulator under: the inherited one without the launch
    variables, the suite's own floor when set, then the test's explicit values."""
    env = {k: v for k, v in os.environ.items() if k not in LAUNCH_VARS}
    if os.environ.get(TEST_VAR):
        env["PODSIM_MIN_FREE_GIB"] = os.environ[TEST_VAR]
    return {**env, **extra}
