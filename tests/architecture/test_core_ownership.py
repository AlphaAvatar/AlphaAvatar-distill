"""The eight false negatives, each restored as a defect that must go red.

The literal and semantic gates both reported ZERO while every defect below was
live in `src/aadistill`. That is the finding: a detector's silence had been
read as compliance, and I had personally cleared three of these as "generic
prose" and "deployment facts" in the previous round.

So each rule here is written against a RESTORED version of the exact defect,
not against a synthetic lookalike. `restore_*` fixtures write the real prior
code into a scratch module and require the checker to flag it; the paired
`test_the_core_is_clean` case requires the live tree to be silent. A rule that
cannot be made to fire has not been shown to check anything.

Deliberately NOT flagged, and pinned as such below: ordinary mathematical
constants, schema versions, generic terminal types, and the reusable operator
taxonomy. Those were the false POSITIVES of the first semantic pass, and a
detector that trades one error for the other has not improved.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CORE = REPO / "src/aadistill"

_spec = importlib.util.spec_from_file_location(
    "core_ownership_under_test", REPO / "scripts/architecture/core_ownership.py")
OWN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(OWN)


#: The path matters for one rule: `numeric_stage_policy` is scoped to
#: infrastructure/runtime/governance, because `Stage 1` in `initialization/` is
#: this project's own workflow vocabulary rather than a driver's marker number.
INFRA = "src/aadistill/infrastructure/synthetic.py"
PLAIN = "src/aadistill/initialization/planning/synthetic.py"


def findings(source: str, rule: str | None = None, path: str = PLAIN) -> list[dict]:
    out = OWN.scan_source(path, source)
    return [f for f in out if rule is None or f["rule"] == rule]


def rules(source: str, path: str = PLAIN) -> set[str]:
    return {f["rule"] for f in findings(source, path=path)}


# --- 1. a phase name inside a serialized payload ----------------------------

class TestPhaseNameInAPayload:
    """The one I cleared as prose. It is a phase name in a wire artifact."""

    RESTORED = '''
def as_dict(self):
    payload = {
        "authorization_id": self.authorization_id,
        "enforcement": (
            "the launcher loads this artifact and refuses to create a pod "
            "whose priced hard threshold exceeds hard_cap_usd, refuses a "
            "stage not in authorized_stages, and has no code path to Phase A"),
    }
    return payload
'''

    def test_the_restored_defect_is_caught(self):
        assert "phase_in_payload" in rules(self.RESTORED)

    def test_the_same_sentence_in_a_docstring_is_not(self):
        """Explaining a rule is not enforcing one."""
        src = ('def as_dict(self):\n'
               '    """Refuses a stage not in authorized_stages, and has no code\n'
               '    path to Phase A."""\n'
               '    return {"a": 1}\n')
        assert "phase_in_payload" not in rules(src)

    def test_a_phase_name_in_a_raised_message_is_not_a_payload(self):
        """A refusal explains itself to a human; it is not stored."""
        src = ('def f():\n'
               '    raise ValueError("Phase A is not authorized here")\n')
        assert "phase_in_payload" not in rules(src)


# --- 2. a concrete plan-id default ------------------------------------------

class TestConcretePlanIdDefault:
    RESTORED = ('from dataclasses import dataclass\n'
                '@dataclass\n'
                'class PreflightPlan:\n'
                '    plan_id: str = "autoinit.micro_preflight"\n')

    def test_the_restored_defect_is_caught(self):
        assert "concrete_id_default" in rules(self.RESTORED)

    def test_a_required_id_is_not(self):
        src = ('from dataclasses import dataclass, field\n'
               '@dataclass\n'
               'class PreflightPlan:\n'
               '    plan_id: str = field(kw_only=True)\n')
        assert "concrete_id_default" not in rules(src)

    def test_a_schema_version_string_is_not(self):
        """`schema` identifies a FORMAT, not a study."""
        src = ('from dataclasses import dataclass\n'
               '@dataclass\n'
               'class R:\n'
               '    schema: str = "aadistill.thing/v1"\n')
        assert "concrete_id_default" not in rules(src)


# --- 3. experiment metric/control defaults in a generic dataclass ------------

class TestExperimentDefaultsInADataclass:
    RESTORED = ('from dataclasses import dataclass\n'
                '@dataclass\n'
                'class SuccessiveHalvingPlan:\n'
                '    feasibility_metric: str = "usable_rollout_rate"\n'
                '    primary_metric: str = "correct_overall"\n'
                '    secondary_metric: str = "correct_given_usable"\n'
                '    include_canonical_control: bool = True\n')

    def test_every_restored_default_is_caught(self):
        f = findings(self.RESTORED, "experiment_default")
        names = {x["name"] for x in f}
        assert {"feasibility_metric", "primary_metric", "secondary_metric",
                "include_canonical_control"} <= names, names

    def test_required_fields_are_not(self):
        src = ('from dataclasses import dataclass, field\n'
               '@dataclass\n'
               'class P:\n'
               '    feasibility_metric: str = field(kw_only=True)\n'
               '    include_canonical_control: bool = field(kw_only=True)\n')
        assert "experiment_default" not in rules(src)

    @pytest.mark.parametrize("src", [
        'from dataclasses import dataclass\n@dataclass\nclass P:\n    version: int = 1\n',
        'from dataclasses import dataclass\n@dataclass\nclass P:\n    alpha: float = 0.05\n',
        'from dataclasses import dataclass\n@dataclass\nclass P:\n    iterations: int = 10000\n',
    ])
    def test_ordinary_parameters_are_not(self, src):
        assert "experiment_default" not in rules(src)


# --- 4. concrete scoring semantics in a generic planning function -----------

class TestScoringSemanticsInPlanning:
    RESTORED = ('def score_recovery_row(*, usable, scorer_correct, scorable=True):\n'
                '    correct = bool(scorable and usable and scorer_correct)\n'
                '    return {"correct": correct}\n')

    def test_the_restored_defect_is_caught(self):
        assert "scoring_semantics" in rules(self.RESTORED)

    def test_delegating_to_a_supplied_rule_is_not(self):
        src = ('def score_recovery_row(*, usable, scorer_correct, scorable=True,\n'
               '                       rule):\n'
               '    correct = bool(rule.decide(scorable, usable, scorer_correct))\n'
               '    return {"correct": correct}\n')
        assert "scoring_semantics" not in rules(src)


# --- 5. a core read of configs/ or logs/ assembled through Path -------------

class TestCoreReadsRepositoryPaths:
    RESTORED = ('from pathlib import Path\n'
                'def provider_cli_fallbacks():\n'
                '    config = (Path(__file__).resolve().parents[3]\n'
                '              / "configs/infrastructure/provider_cli.json")\n'
                '    return tuple(config.read_text())\n')

    def test_the_restored_defect_is_caught(self):
        assert "core_reads_repository" in rules(self.RESTORED)

    def test_a_bare_literal_config_path_is_caught(self):
        src = 'from pathlib import Path\nLEDGER = Path("configs/autoinit/x.json")\n'
        assert "core_reads_repository" in rules(src)

    def test_a_caller_supplied_path_is_not(self):
        src = ('def verify_ledger(path, *, repo_root="."):\n'
               '    return (repo_root, path)\n')
        assert "core_reads_repository" not in rules(src)


# --- 6. an import-time call that loads deployment configuration -------------

class TestImportTimeDeploymentLoad:
    RESTORED = ('def _main_relay():\n'
                '    import json\n'
                '    from pathlib import Path\n'
                '    return json.loads(Path("configs/x.json").read_text())["r"]\n'
                'MAIN_RELAY = _main_relay()\n')

    def test_the_restored_defect_is_caught(self):
        assert "import_time_deployment_load" in rules(self.RESTORED)

    def test_a_function_that_is_not_called_at_import_is_not(self):
        src = ('def main_relay(config):\n'
               '    import json\n'
               '    return json.loads(config.read_text())["r"]\n')
        assert "import_time_deployment_load" not in rules(src)

    def test_a_pure_constant_is_not(self):
        assert "import_time_deployment_load" not in rules("LIMIT = int(3)\n")


# --- 7. a numeric stage key or stage-specific helper in infrastructure ------

class TestNumericStageInInfrastructure:
    RESTORED = ('class MarkerPolicy:\n'
                '    def stage2_passed(self, terminal, driver_stages):\n'
                '        return bool(driver_stages.get("2") or terminal == self.success)\n')

    def test_the_restored_defect_is_caught(self):
        assert "numeric_stage_policy" in rules(self.RESTORED, path=INFRA)

    def test_the_projects_own_stage_vocabulary_is_not(self):
        """`Stage 1` is this project's workflow (AGENTS.md defines stages 0-6)
        and an entire module is named for it. Flagging it would trade a false
        negative for a false positive, which is not an improvement."""
        src = ("def import_stage1_result(*, search_result, states_path):\n"
               "    return (search_result, states_path)\n")
        assert "numeric_stage_policy" not in rules(
            src, path="src/aadistill/initialization/planning/stage1_import.py")

    def test_a_named_eligibility_predicate_is_not(self):
        src = ('class MarkerPolicy:\n'
               '    def products_are_eligible(self, terminal, driver_stages):\n'
               '        if self.products_eligible is not None:\n'
               '            return bool(self.products_eligible(terminal, driver_stages))\n'
               '        return terminal == self.success\n')
        assert "numeric_stage_policy" not in rules(src, path=INFRA)

    def test_a_generic_stage_index_parameter_is_not(self):
        src = 'def require_stage(self, stage):\n    return stage in self.authorized_stages\n'
        assert "numeric_stage_policy" not in rules(src, path=INFRA)


# --- 8. an untyped result-shape branch encoding an artifact protocol -------

class TestUntypedResultShapeBranch:
    RESTORED = ('def fetch_result_ok(entry):\n'
                '    if isinstance(entry, dict):\n'
                '        return entry.get("rc") == 0\n'
                '    return True\n')

    def test_the_restored_defect_is_caught(self):
        """It fails OPEN: every non-dict counted as a completed transfer."""
        assert "untyped_result_branch" in rules(self.RESTORED)

    def test_the_fail_closed_version_is_not(self):
        src = ('def default_fetch_result_ok(entry):\n'
               '    if isinstance(entry, ProductFetchResult):\n'
               '        return entry.ok\n'
               '    if isinstance(entry, dict):\n'
               '        return entry.get("rc") == 0\n'
               '    return False\n')
        assert "untyped_result_branch" not in rules(src)

    def test_an_isinstance_branch_that_does_not_return_success_is_not(self):
        src = ('def norm(x):\n'
               '    if isinstance(x, dict):\n'
               '        return dict(x)\n'
               '    return list(x)\n')
        assert "untyped_result_branch" not in rules(src)


# --- the live tree ----------------------------------------------------------

class TestTheCoreIsClean:
    def test_no_ownership_violations_remain(self):
        found = OWN.scan(CORE)
        assert found == [], (
            "core-ownership violations in src/aadistill:\n"
            + "\n".join(f"  {f['rule']}: {f['path']}:{f['line']} "
                        f"{f['name']} -- {f['why']}" for f in found))

    def test_every_rule_is_exercised_by_a_restored_defect(self):
        """Guards the guard: a rule with no red case proves nothing.

        The eight classes the maintainer named must each have a test above that
        makes the checker fire. If a rule is added without one, this fails.
        """
        exercised = set()
        for cls in (TestPhaseNameInAPayload, TestConcretePlanIdDefault,
                    TestExperimentDefaultsInADataclass,
                    TestScoringSemanticsInPlanning, TestCoreReadsRepositoryPaths,
                    TestImportTimeDeploymentLoad, TestNumericStageInInfrastructure,
                    TestUntypedResultShapeBranch):
            exercised |= rules(cls.RESTORED, path=INFRA)
        assert set(OWN.RULES) <= exercised, (
            f"rules with no restored-defect case: {sorted(set(OWN.RULES) - exercised)}")

    def test_the_recorded_report_agrees_with_a_live_scan(self):
        import json
        path = REPO / "logs/architecture_core_ownership.json"
        if not path.is_file():
            pytest.skip("no recorded report yet")
        assert json.loads(path.read_text())["total"] == len(OWN.scan(CORE))
