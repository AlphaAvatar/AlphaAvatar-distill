"""The C3 driver runs C3's experiment, not the one it was adapted from.

WHY THIS EXISTS. `autoinit_c3_driver.py` was adapted from
`autoinit_c1_driver.py` rather than written from nothing -- the right call
for 1300 lines of budget accounting, device handoff, artifact preservation
and trainer plumbing that is genuinely shared. But a port keeps the shape and
quietly keeps the *constants*, and three of C1's would each have produced a
complete, plausible, wrong result:

* **`self.seeds = derive_recovery_seeds()`** returns C1's three seeds. C3's
  nine probes would have run on `1635674081, 1656475568, 696460635` instead
  of the mechanically derived `217230555, 1151307191, 2045359208` -- the two
  sets are disjoint -- while every probe id, coverage check and count looked
  correct. The preregistration would have described an experiment that never
  ran.
* **`len(out) == 6`** and **`len(self.training) != 6`**: assertions that pass
  quietly while running two thirds of the design.
* **stage E naming `weight_proxy_v0`**, C1's incumbent, not C3's
  `activation_importance_v1`.

Every one is a constant that was *right for C1*. So these tests do not ask
"does the driver work" -- they ask "is it still C1's experiment anywhere".
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from experiments.phase_c3 import session as CS  # noqa: E402

DRIVER = REPO / "scripts/pod/autoinit_c3_driver.py"
SRC = DRIVER.read_text()


def executable_source(text: str) -> str:
    """The driver with comments and string literals removed.

    These tests are about what RUNS. Scanning the raw file made two of them
    fail on this file's own explanations -- a comment saying why
    `derive_recovery_seeds()` is absent read as a call to it, and a docstring
    quoting the old `len(out) == 6` read as that assertion surviving. A guard
    that cannot tell code from prose about code teaches you to delete the
    prose, which is the opposite of what it should encourage.
    """
    import io
    import tokenize

    out = []
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.COMMENT:
            continue
        #: Every string literal, docstring or not -- none of them can contain
        #: a call or an assertion that executes.
        out.append('""' if tok.type == tokenize.STRING else tok.string)
    return " ".join(out)


CODE = executable_source(SRC)
PREREG = json.loads(
    (REPO / "logs/stages/stage-1/phase_c3/plans/c3_preregistration.json").read_text())


def test_the_driver_parses_and_declares_no_c1_seed_source():
    """The defect that would have been invisible in the results."""
    ast.parse(SRC)
    #: It may be NAMED in a comment explaining why it is absent; it may not be
    #: imported or called.
    tree = ast.parse(SRC)
    imported = {
        alias.name for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) for alias in node.names}
    assert "derive_recovery_seeds" not in imported, (
        "the C3 driver imports C1's seed derivation; that is one line away "
        "from running the nine probes on the wrong replicates")
    assert not re.search(r"\bderive_recovery_seeds\s*\(", CODE), (
        "the C3 driver calls C1's seed derivation")


def test_the_driver_takes_its_seeds_from_the_frozen_c3_plan():
    assert "self.seeds = list(CS.recovery_seeds())" in SRC
    assert list(CS.recovery_seeds()) == PREREG["seeds"]["recovery"]
    assert CS.recovery_seeds() == (217230555, 1151307191, 2045359208)


def test_c1_and_c3_seeds_are_disjoint_so_the_mistake_is_detectable():
    """If they overlapped, the wrong-seed defect would be unfalsifiable."""
    from experiments.phase_c1.isolation import derive_recovery_seeds

    assert not set(CS.recovery_seeds()) & set(derive_recovery_seeds())


def test_no_probe_count_is_a_literal_six():
    """`== 6` is the assertion that passes while running the wrong design."""
    offenders = re.findall(r"[^\n]{0,60}(?:!=|==)\s*6\b[^\n]{0,20}", CODE)
    assert not offenders, (
        f"a hard-coded probe count survives in the C3 driver: {offenders}. "
        "The count is 3 arms x 3 seeds and must be derived from the plan.")


def test_the_driver_derives_its_probe_count_from_the_plan():
    assert "len(CS.arm_ids()) * len(self.seeds)" in SRC
    assert len(CS.arm_ids()) * len(CS.recovery_seeds()) == 9


def test_stage_e_names_c3_s_incumbent_not_c1_s():
    """weight_proxy_v0 was C1's incumbent; C3's is activation_importance_v1."""
    header = SRC.split('"""')[1]
    assert "activation_importance_v1" in header
    assert "weight_proxy" not in header, (
        "the driver's stage map still names C1's incumbent operator")
    assert CS.arm("A_incumbent")["attention"][0] == \
        "attention.activation_importance_v1"


def test_the_driver_writes_under_c3_s_own_roots():
    """A C3 probe under artifacts/stage3/c1 is collected as C1's evidence."""
    for wrong in ('REPO / "artifacts/audit/autoinit_c1"',
                  'REPO / "artifacts/stage3/c1"',
                  'REPO / "artifacts/eval/c1"',
                  'REPO / "artifacts/autoinit/c1_arms"'):
        assert wrong not in SRC, f"the C3 driver still writes to {wrong}"
    for right in ("artifacts/audit/autoinit_c3", "artifacts/stage3/c3",
                  "artifacts/eval/c3", "artifacts/autoinit/c3_arms"):
        assert right in SRC


def test_the_driver_loads_a_c3_authorization():
    assert "AUTHORIZATION_TYPE = C3Authorization" in SRC
    assert "C1Authorization" not in SRC, (
        "a C1 authorization measures a different harness and carries a "
        "ceiling derived for six probes on two arms")


def test_the_three_contrasts_are_computed_in_their_frozen_roles():
    """Roles fixed before results; nothing picks a contrast by looking."""
    for role in ("primary", "secondary", "practical"):
        assert f'"{role}"' in SRC, f"the driver computes no {role} contrast"
    #: The verdict is built from the primary alone.
    assert 'primary = reported["primary"]' in SRC
    assert "boot=primary[\"bootstrap\"]" in SRC
    assert "per_seed_delta=primary[\"per_seed_delta\"]" in SRC


def test_the_decision_record_states_who_owns_the_verdict():
    assert '_verdict_owner' in SRC
    assert "b1_vs_b3_is_a_post_c3_maintainer_decision" in SRC
    assert "c4_is_not_started" in SRC


def test_the_arms_must_materialize_to_distinct_digests():
    """B1 and B3 share an implementation; only the hashed config parts them.

    If the calibration protocol stopped being identity-bearing, both causal
    arms would build the same state and the session would run one treatment
    twice while reporting two.
    """
    assert "two arms materialized to the same artifact digest" in SRC


def test_each_initialization_is_built_once_and_fanned_out():
    """Re-running the causal scorers per seed would cost 2h+ for nothing."""
    assert "_each_initialization_built_once" in SRC
    #: The incumbent is NOT rebuilt in stage F -- stage DE already gated it.
    assert "causal_arms = [a for a in CS.arm_ids() if a != incumbent_id]" in SRC


def test_the_frozen_plan_is_built_by_iterating_the_plan_s_arms():
    """A literal arm tuple is how the executor drifted from the plan before.

    BEHAVIOURAL. This used to grep the driver for `for arm_id in CS.arm_ids()`,
    which asserted where a loop is written rather than that the arms come from
    the plan -- and it failed the moment the resolution moved to its owner in
    `session.py`, with the property completely unchanged. What matters is that
    an arm the plan does not declare cannot be resolved, and that is asked of
    the real resolver.
    """
    import copy

    from experiments.phase_c3 import session as CS

    real = CS.preregistration()
    assert CS.primary_operands() == ("A_incumbent", "B_causal_b1")

    #: An arm the resolver cannot tokenize must RAISE, not be skipped: a
    #: silently dropped arm is how a two-arm executor ran a three-arm plan.
    renamed = copy.deepcopy(real)
    renamed["arms"]["D_unrecognised"] = renamed["arms"]["C_causal_b3"]
    saved = CS._PREREG
    try:
        CS._PREREG = renamed
        with pytest.raises(CS.C3SessionError, match="contrast token"):
            CS.primary_operands()
    finally:
        CS._PREREG = saved

    #: And no C1 two-arm literal survives in the driver.
    assert 'C1Arm("c1.incumbent"' not in SRC
    assert "CS.INCUMBENT_ATTENTION" not in SRC and "CS.TREATMENT_ATTENTION" not in SRC


@pytest.mark.parametrize("stale", [
    "TREATMENT_SUFFIX_START_INDEX", "EXPECTED_PARENT_DIGEST",
    "EXPECTED_INCUMBENT_DIGEST", "TREATMENT_CONFIG",
])
def test_no_module_constant_from_the_two_arm_session_survives(stale):
    """Those names no longer exist on the C3 session; a reference would raise
    at runtime, on a paid pod, after the teacher had already been fetched."""
    assert f"CS.{stale}" not in SRC
    assert not hasattr(CS, stale), (
        f"the C3 session still exposes {stale}; it belonged to the two-arm "
        "design and its presence invites a stale reference")
