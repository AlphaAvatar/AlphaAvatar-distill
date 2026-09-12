"""The semantic-hardcode gate, pinned in both directions.

A rule that only ever fires is as useless as one that never does: the first
buries a real finding under noise until nobody reads the report, and the second
certifies a core that is full of one study's decisions. So every rule here gets
a case it MUST catch and a case it MUST NOT, written as synthetic modules rather
than read off the tree -- a property pinned by construction keeps holding, while
one read off `src/aadistill` only describes what it contains today.

The must-not cases are the real false positives the first pass produced, kept as
regressions: `COMPONENTS` and `SPLITS` are what an evaluation module is FOR,
`attempt = 0` is an ordinal, `DEFAULT_STATS_SPEC` is a safe empty value, and
`RelayInput.repo` defaulting to the deployment's one artifact store is the
opposite of the defect.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "semantic_hardcode_under_test", REPO / "scripts/architecture/semantic_hardcode.py")
SH = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SH)


def rules(source: str) -> list[str]:
    """Which rules fire on this module body."""
    w = SH.SemanticWalker("synthetic.py", source)
    import ast

    w.visit(ast.parse(source))
    return sorted(f["rule"] for f in w.findings)


# --- must catch --------------------------------------------------------------

class TestItCatchesCurrentExperimentPolicy:
    def test_a_concrete_seed(self):
        assert "concrete_seed" in rules("SEED_SA = 20260726\n")

    def test_a_seed_as_a_keyword_default(self):
        assert "concrete_seed" in rules(
            "def f(*, seed: int = 20260806): pass\n")

    def test_a_capability_tuple(self):
        assert "concrete_capability_tuple" in rules(
            'CAPABILITIES = ("gsm8k", "math_verified", "multihop", "rag")\n')

    def test_a_domain_tuple(self):
        assert "concrete_capability_tuple" in rules(
            'E8A_DOMAINS = ("general", "math", "rag_multihop", "code", "tool")\n')

    def test_a_phase_named_policy_constant(self):
        assert "phase_policy_constant" in rules(
            'PHASE_A_BATTERY = "recovery_search_v2"\n')

    def test_a_study_sized_rung_constant(self):
        assert "phase_policy_constant" in rules("NESTED_RUNG_INCREMENT = 735_603\n")

    def test_a_permission_property(self):
        assert "phase_policy_property" in rules(
            "class A:\n"
            "    @property\n"
            "    def allows_phase_a(self): return False\n")

    def test_a_phase_named_class(self):
        assert "phase_policy_class" in rules("class PhaseAPolicy: pass\n")

    def test_a_hardware_sku(self):
        assert "hardware_policy" in rules('GPU = "NVIDIA L40S"\n')

    def test_a_vram_requirement(self):
        assert "hardware_policy" in rules('NEED = "40 GiB free"\n')

    def test_a_price(self):
        assert "price_policy" in rules('RATE = "$1.09 per hour"\n')

    def test_a_current_experiment_identity(self):
        assert "current_experiment_identity" in rules(
            'battery_asset_id = "recovery_search_v2"\n')

    def test_a_dataclass_default_that_instantiates_policy(self):
        """The exact defect: a plan built with no arguments becomes this study."""
        src = ("from dataclasses import dataclass\n"
               "CATASTROPHIC_V1 = CatastrophicRule(metric='x', control_min=0.1)\n"
               "@dataclass\n"
               "class Plan:\n"
               "    catastrophic: CatastrophicRule = CATASTROPHIC_V1\n")
        assert "dataclass_default_instantiates_policy" in rules(src)


# --- must NOT catch ----------------------------------------------------------

class TestItLeavesGenericMechanismAlone:
    """Each of these fired on the first pass and is a false positive."""

    @pytest.mark.parametrize("src,why", [
        ('COMPONENTS = ("non_empty", "natural_termination", "no_context_limit")\n',
         "a metric's own components are what the module is for"),
        ('SPLITS = ("train", "val", "test")\n', "dataset splits are vocabulary"),
        ('GATES = ("shape", "dtype", "device")\n', "gate names are vocabulary"),
        ('BEHAVIOR_GROUPS = ("code", "math", "rag")\n',
         "the behaviour grouping IS the mechanism"),
    ])
    def test_mechanism_vocabulary_is_not_a_capability_tuple(self, src, why):
        assert "concrete_capability_tuple" not in rules(src), why

    @pytest.mark.parametrize("src", [
        "attempt = 0\n", "rung = 1\n", "stage_index = 3\n",
    ])
    def test_an_ordinal_is_not_phase_policy(self, src):
        assert "phase_policy_constant" not in rules(src)

    @pytest.mark.parametrize("src", [
        "z = 1.96\n", "chunk = 512\n", "temperature = 1.0\n",
        "version: int = 1\n", "max_retries = 3\n", "poll_seconds = 60\n",
        "epsilon = 1e-8\n",
    ])
    def test_generic_algorithm_parameters_are_untouched(self, src):
        assert rules(src) == [], f"{src.strip()} is an algorithm parameter"

    def test_a_derived_count_property_is_not_permission(self):
        assert "phase_policy_property" not in rules(
            "class P:\n"
            "    @property\n"
            "    def rung1_probes(self): return 6\n")

    @pytest.mark.parametrize("const,builder", [
        ("DEFAULT_STATS_SPEC", "StatsSpec()"),
        ("RUN_LAYOUT_VERSION", "int(3)"),
        ("MAIN_RELAY", "_main_relay()"),
    ])
    def test_a_sentinel_version_or_config_value_is_not_policy(self, const, builder):
        src = ("from dataclasses import dataclass\n"
               f"{const} = {builder}\n"
               "@dataclass\n"
               "class T:\n"
               f"    field: object = {const}\n")
        assert "dataclass_default_instantiates_policy" not in rules(src)

    def test_prose_naming_a_seed_does_not_fire(self):
        """Docstrings and comments are not findings."""
        src = ('"""SEED_SA = 20260726 is the study seed."""\n'
               "def f():\n"
               '    """Uses seed 20260726."""\n'
               "    return 1\n")
        assert rules(src) == []


# --- the real tree -----------------------------------------------------------

class TestTheCoreIsClean:
    def test_no_semantic_experiment_policy_remains(self):
        findings = SH.scan()
        assert findings == [], (
            "current-experiment policy in src/aadistill:\n"
            + "\n".join(f"  {f['rule']}: {f['path']}::{f['name']} -- {f['why']}"
                        for f in findings))

    def test_the_recorded_report_agrees_with_a_live_scan(self):
        path = REPO / "logs/maintenance/inventories/architecture_semantic_hardcode.json"
        if not path.is_file():
            pytest.skip("no recorded report yet")
        recorded = json.loads(path.read_text())
        assert recorded["total"] == len(SH.scan()), (
            "re-run scripts/architecture/semantic_hardcode.py --write")
