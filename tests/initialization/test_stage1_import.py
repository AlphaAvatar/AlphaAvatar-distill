"""The Stage-1 importer refuses to be a trusted deserializer.

The journal is evidence, not a serialization format. Everything else about this
module — that it rebuilds leaves and a control from bytes, and the nine refusals
that make a continuation safe to start — is in
`test_stage1_import_refusals.py`, proved against a search built under
`tmp_path`.

**This file used to be a Phase-A integration test.** It read attempt 12's real
records and its five preserved 1.11-GiB checkpoints from a host-local store,
under a `skipif`, which made the core suite's result depend on whether an old
experiment's bytes were still on this machine. That verification now lives in
`scripts/stages/stage-1/phase_a/tests/test_phase_a_attempt12_import.py`,
where the historical evidence is the subject rather than the fixture.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


def test_there_is_no_permissive_state_deserializer():
    """A generic `InitializationState.from_dict` would let any recorded line
    become a live candidate. The import reconstructs field by field from values
    it has re-derived instead.

    Asserted on the source, so it needs no evidence and no checkpoints: the
    property is about what the class and the module offer, not about what any
    particular search produced.
    """
    from aadistill.initialization.specs.state import InitializationState

    assert not hasattr(InitializationState, "from_dict"), (
        "a permissive deserializer appeared; the strict import exists so the "
        "journal never becomes a trusted input")
    src = (REPO / "src/aadistill/initialization/planning/stage1_import.py").read_text()
    assert "from_dict" not in src.split('"""')[2], (
        "the import gained a from_dict path")
