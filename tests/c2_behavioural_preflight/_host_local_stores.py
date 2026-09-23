"""What this scope can and cannot certify about a pod.

NOT a `conftest.py`, deliberately. `tests/` carries no `__init__.py`, so pytest
puts each test directory on `sys.path` and a bare `from conftest import X`
resolves to whichever one landed first. `tests/pod` and `tests/autoinit` both
import their own helpers that way, so a `conftest` here shadowed theirs and
broke 9 modules at collection the moment more than one scope ran together. A
pod would never have seen it -- it collects this directory alone -- which is
exactly the kind of divergence this file exists to remove. Hence a unique name,
leading underscore so nothing collects it as a test.

This directory is the ONE test directory a behavioural session's pod does not
ignore, so every case here runs inside a paid session's blocking gate. That
makes it the wrong place to depend on anything only the dev box holds — and 27
cases did.

They reach the out-of-tree durable store: the five reconstructed replay
products and the campaign's completed probes, under `$HOME/aad-artifacts`. A
session stages an asset's BYTES and never the store they were frozen in, so no
pod has it. On the dev box the store is there and the cases ran; on the pod
`candidate_manifest` raised `BehaviouralProposalError` and the gate refused.
That is the shape that cost C1 attempt 14 $0.40 and behavioural attempt9 $0.11.

The predicate below is keyed on THE CONDITION -- does this machine have the
store -- and never on a simulator marker, because a pod is behaviourally
identical to a machine that never had the store while carrying none of the
simulator's markers, so "am I in the simulation?" is inverted exactly where it
matters. `tests/pod/test_c1_session_contract.py` settled this already; this is
the same predicate for this scope.

It only tells the truth in the simulator because the stores are now located
through `$HOME` (`aadistill.runtime.cpu_test_env.host_local_store`). While they
were absolute constants, the simulator's fresh empty HOME could not reach them,
the sweep saw a store the pod would not have, and the two skip sets disagreed
-- which is the refusal attempt9 hit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments.phase_c2 import behavioural as _BH  # noqa: E402


def host_local_stores_are_absent() -> bool:
    """Does this machine lack the out-of-tree stores these cases read?

    Both are asked about, not just the one a given case happens to touch: they
    are absent together on a pod and present together on the dev box, and a
    predicate that covered one would skip inconsistently on a machine holding
    exactly the other.
    """
    replay = Path(_BH.DURABLE_STORE)
    probes = Path(_BH.DURABLE_STORE).parents[1] / "phase_c2_behavioural"
    return not (replay.is_dir() and probes.is_dir())


#: Applied to a case that reads the dev box's frozen products. NOT a licence to
#: skip anything inconvenient: a case that can build its own fixture must do
#: that instead, because a skip on the pod certifies nothing about the pod.
needs_host_local_stores = pytest.mark.skipif(
    host_local_stores_are_absent(),
    reason=("reads the out-of-tree durable store under $HOME/aad-artifacts, "
            "which a pod never receives -- a session stages an asset's bytes "
            "and not the store they were frozen in"),
)
