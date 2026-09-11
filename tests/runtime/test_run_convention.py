"""This repository's run-directory convention, and that it is not C1's.

`aadistill.runtime.run_layout` is covered by `test_run_layout_all_stages.py`:
identifiers, containment, role uniqueness, manifest self-hash. This module
covers the layer above it — `experiments.run_layout`, the one place that says
the root is `logs/runs` and that a run's files live in five areas.

Two claims are worth protecting here, and they pull in opposite directions.

*The areas are shared.* A role path that sits in the run's own root, or names an
area without naming anything inside it, is refused — otherwise a run accumulates
loose files with no owner, which is the flat-`logs/` habit this convention
replaces.

*What lives in them is not.* The role vocabularies below are deliberately
disjoint: a Stage-0 activation collection, a Stage-3 recovery run and a Stage-4
rollout batch share no role name at all, and each experiment's `ArtifactSpec`
refuses the others'. If a future edit taught this module what a `replay_record`
is, `test_one_experiments_roles_are_refused_by_anothers_spec` is the one that
fails.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aadistill.runtime.run_layout import RunLayoutError
from experiments.run_layout import (
    AREAS, ArtifactSpec, MANIFEST_NAME, RUNS_ROOT, RunConventionError, area_of,
    check_roles, is_recorded, layout_for, manifest_path, open_run,
    present_roles, read_run, record_run,
)

# --- three experiments, three vocabularies, one mechanism --------------------
#
# Not one shape parameterized by a count. Different *stages*, with different
# artifacts, different required sets and different area usage — a Stage-0 run
# produces no evidence of a decision, and a rollout batch has no governance
# snapshot to keep because nothing one-use authorized it.

STAGE0 = ("stage0_collect", ArtifactSpec(
    spec_id="stage0_activation_collection_v1",
    required=("cache_manifest",),
    optional=("collector_log", "coverage_report")), {
        "cache_manifest": "artifacts/activation_cache_manifest.json",
        "collector_log": "runtime/collect.log",
        "coverage_report": "evidence/prompt_coverage.json"})

STAGE3 = ("stage3_recovery", ArtifactSpec(
    spec_id="stage3_recovery_v1",
    required=("train_log", "checkpoint_manifest"),
    optional=("grant", "eval_rows", "outcome")), {
        "train_log": "runtime/train_log.jsonl",
        "checkpoint_manifest": "artifacts/checkpoints.json",
        "grant": "governance/grant.json",
        "eval_rows": "evidence/per_sample.jsonl",
        "outcome": "closeout/outcome.json"})

STAGE4 = ("stage4_rollout", ArtifactSpec(
    spec_id="stage4_rollout_v1",
    required=("rollout_tokens",),
    optional=("engine_identity",)), {
        "rollout_tokens": "artifacts/rollouts.jsonl",
        "engine_identity": "evidence/engine.json"})

EXPERIMENTS = (STAGE0, STAGE3, STAGE4)


def _fill(layout, roles, present=None):
    """Create the files for `present` (default: all roles)."""
    for role in (roles if present is None else present):
        target = layout.path(roles[role])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}\n")


# --- the shared part: the five areas ----------------------------------------

def test_the_root_is_named_here_and_only_here():
    """The core states no repository location; this module does.

    Read from the core's CODE with docstrings stripped, not from its source
    text: its module docstring explains that the previous version named
    `logs/runs` and that this is why it could not be reused, and a check that
    cannot tell an explanation from the thing it explains flags the sentence
    describing the fix.
    """
    import ast
    import aadistill.runtime.run_layout as core

    assert RUNS_ROOT == "logs/runs"
    tree = ast.parse(open(core.__file__).read())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)) and ast.get_docstring(node):
            node.body = node.body[1:]
    literals = [n.value for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    assert not [s for s in literals if "logs/" in s], (
        "the core names a repository directory again")


@pytest.mark.parametrize("area", AREAS)
def test_every_area_accepts_a_path_inside_it(area):
    assert area_of(f"{area}/thing.json") == area
    #: A whole-area role. The CUDA validation declares `artifacts/`, and a
    #: convention that refused it would push that run back to the flat root.
    assert area_of(f"{area}/") == area


@pytest.mark.parametrize("bad", [
    "outcome.json",            # the run's own root
    "manifest.json",           # the index is not a role
    "logs/outcome.json",       # an area this repository does not have
    "governance",              # the area itself, with nothing inside
    "/governance/x.json",      # absolute
    "",
])
def test_a_path_outside_the_areas_is_refused(bad):
    with pytest.raises(RunConventionError):
        area_of(bad)


def test_check_roles_refuses_the_whole_mapping_on_one_bad_path():
    with pytest.raises(RunConventionError):
        check_roles({"good": "runtime/a.json", "bad": "loose.json"})


def test_traversal_is_still_refused_by_the_core(tmp_path):
    """The area check is additional, not a replacement."""
    layout = layout_for(tmp_path, "stage3_recovery", "r1")
    with pytest.raises(RunLayoutError):
        layout.path("runtime/../../../escape.json")


@pytest.mark.parametrize("crafted", [
    "governance/../manifest.json",   # lands on the run's own index
    "runtime/./session.json",
    "artifacts/../../other_run/x.json",
])
def test_a_path_that_starts_in_an_area_and_leaves_it_is_refused(crafted):
    """The area a path RESOLVES to, not the one it is spelled with.

    The core would allow the first of these: it resolves inside the run root,
    which is all the core promises. It is the run's `manifest.json` — a role
    that could overwrite the index describing it.
    """
    with pytest.raises(RunConventionError) as exc:
        area_of(crafted)
    assert "traversal" in str(exc.value)


def test_the_core_really_would_have_allowed_the_crafted_path(tmp_path):
    """The premise of the test above, executed rather than asserted.

    If the core ever starts refusing it, this test says so instead of leaving a
    guard whose stated reason has quietly stopped being true.
    """
    layout = layout_for(tmp_path, "stage3_recovery", "r1")
    assert layout.path("governance/../manifest.json") == (
        layout.root / "manifest.json").resolve()


# --- the per-experiment part: nothing here knows these roles ----------------

@pytest.mark.parametrize("experiment_id,spec,roles", EXPERIMENTS,
                         ids=[e[0] for e in EXPERIMENTS])
def test_each_experiment_opens_records_and_reads_back(tmp_path, experiment_id,
                                                      spec, roles):
    layout = open_run(tmp_path, experiment_id, "r1", roles=roles)
    assert layout.root == tmp_path / RUNS_ROOT / experiment_id / "r1"
    _fill(layout, roles)
    doc = record_run(layout, spec=spec, plan={"n": 1}, implementation={},
                     status={"passed": True}, roles=present_roles(layout, roles))
    assert doc["artifact_spec"] == spec.spec_id
    assert doc["roles"] == dict(sorted(roles.items()))
    assert read_run(tmp_path, experiment_id, "r1")["self_sha256"] == doc["self_sha256"]


def test_one_experiments_roles_are_refused_by_anothers_spec(tmp_path):
    """The guard against this module growing a favourite experiment.

    Stage-3's roles under Stage-0's spec must fail, and the message must name
    the undeclared roles rather than quietly recording them.
    """
    _, stage0_spec, _ = STAGE0
    _, _, stage3_roles = STAGE3
    layout = open_run(tmp_path, "stage3_recovery", "r1", roles=stage3_roles)
    _fill(layout, stage3_roles)
    with pytest.raises(RunLayoutError) as exc:
        record_run(layout, spec=stage0_spec, plan={}, implementation={},
                   status={}, roles=stage3_roles)
    assert "train_log" in str(exc.value) or "cache_manifest" in str(exc.value)


def test_a_missing_required_role_is_refused_not_recorded(tmp_path):
    _, spec, roles = STAGE3
    layout = open_run(tmp_path, "stage3_recovery", "r1", roles=roles)
    _fill(layout, roles, present=["train_log"])          # no checkpoint manifest
    with pytest.raises(RunLayoutError) as exc:
        record_run(layout, spec=spec, plan={}, implementation={}, status={},
                   roles=present_roles(layout, roles))
    assert "checkpoint_manifest" in str(exc.value)


def test_an_aborted_run_still_records_what_it_produced(tmp_path):
    """Stage 4 aborted before the engine identity was written. That is a run."""
    _, spec, roles = STAGE4
    layout = open_run(tmp_path, "stage4_rollout", "r1", roles=roles)
    _fill(layout, roles, present=["rollout_tokens"])
    doc = record_run(layout, spec=spec, plan={}, implementation={},
                     status={"passed": False}, roles=present_roles(layout, roles))
    assert list(doc["roles"]) == ["rollout_tokens"]


# --- present_roles reports what exists, not what was declared ---------------

def test_an_empty_declared_directory_is_not_reported_as_produced(tmp_path):
    """`open_run` creates every role's directory, so existence is not evidence."""
    roles = {"artifacts": "artifacts/", "log": "runtime/a.log"}
    layout = open_run(tmp_path, "stage4_rollout", "r1", roles=roles)
    assert layout.path("artifacts/").is_dir()
    assert present_roles(layout, roles) == {}
    (layout.path("artifacts/") / "x.jsonl").write_text("{}\n")
    assert present_roles(layout, roles) == {"artifacts": "artifacts/"}


# --- a run is opened once ---------------------------------------------------

def test_a_recorded_run_is_not_reopened(tmp_path):
    _, spec, roles = STAGE4
    layout = open_run(tmp_path, "stage4_rollout", "r1", roles=roles)
    _fill(layout, roles)
    record_run(layout, spec=spec, plan={}, implementation={}, status={},
               roles=present_roles(layout, roles))
    assert is_recorded(layout)
    with pytest.raises(RunConventionError) as exc:
        open_run(tmp_path, "stage4_rollout", "r1", roles=roles)
    assert MANIFEST_NAME in str(exc.value)


def test_an_occupied_run_with_no_manifest_is_not_reopened(tmp_path):
    """The launcher-died case. Overwriting it destroys the only evidence.

    Distinct from the recorded case on purpose: a manifest-less directory full
    of files is the shape a session leaves when the launcher dies between
    creating the run and closing it, and that is exactly when the files matter.
    """
    _, _, roles = STAGE4
    layout = open_run(tmp_path, "stage4_rollout", "r1", roles=roles)
    _fill(layout, roles, present=["rollout_tokens"])
    assert not is_recorded(layout)
    with pytest.raises(RunConventionError) as exc:
        open_run(tmp_path, "stage4_rollout", "r1", roles=roles)
    assert "died before recording" in str(exc.value)


def test_an_empty_run_directory_may_be_opened(tmp_path):
    """Only files block. A bare directory is what `create` itself leaves."""
    _, _, roles = STAGE4
    (tmp_path / RUNS_ROOT / "stage4_rollout" / "r1").mkdir(parents=True)
    assert open_run(tmp_path, "stage4_rollout", "r1", roles=roles)


# --- inputs prepared before the run opens -----------------------------------
#
# Not every file in a run is written by the run. A maintainer's grant is
# authored and committed while the tree is still clean, because the launch-bound
# readiness sweep and the authorization issued from it both require it to be
# there already. The occupancy rule could not tell that declared, expected input
# from the residue of a launcher that died, so the only two places such a file
# could go were a flat `logs/<experiment>_attemptN_grant.json` — nine of those
# exist — or nowhere.
#
# `prepared` is an exemption from THAT rule and nothing else, which is what the
# next four cases pin.

def _prepared_run(tmp_path, body='{"granted_by": "maintainer"}\n'):
    """A run directory holding only its prepared grant, not yet opened."""
    _, _, roles = STAGE3
    grant = tmp_path / RUNS_ROOT / "stage3_recovery" / "r1" / roles["grant"]
    grant.parent.mkdir(parents=True)
    grant.write_text(body)
    return roles, grant


def test_a_prepared_role_may_already_exist_when_the_run_opens(tmp_path):
    roles, grant = _prepared_run(tmp_path)
    layout = open_run(tmp_path, "stage3_recovery", "r1", roles=roles,
                      prepared=("grant",))
    assert layout.path(roles["grant"]).read_text() == grant.read_text(), (
        "opening the run must not rewrite or truncate a prepared input")


def test_the_same_file_undeclared_still_blocks_the_run(tmp_path):
    """The mutation. Drop the declaration and the refusal comes straight back.

    Without this the first case proves nothing: an `open_run` that had simply
    stopped checking occupancy would pass it just as well.
    """
    roles, _ = _prepared_run(tmp_path)
    with pytest.raises(RunConventionError) as exc:
        open_run(tmp_path, "stage3_recovery", "r1", roles=roles)
    assert "died before recording" in str(exc.value)


def test_a_prepared_role_exempts_itself_and_nothing_else(tmp_path):
    """A half-written evidence tree beside the grant is still the dead-launcher
    case, and naming the grant must not smuggle it through."""
    roles, _ = _prepared_run(tmp_path)
    stray = tmp_path / RUNS_ROOT / "stage3_recovery" / "r1" / roles["eval_rows"]
    stray.parent.mkdir(parents=True, exist_ok=True)
    stray.write_text("{}\n")
    with pytest.raises(RunConventionError) as exc:
        open_run(tmp_path, "stage3_recovery", "r1", roles=roles,
                 prepared=("grant",))
    assert roles["eval_rows"] in str(exc.value)
    assert roles["grant"] not in str(exc.value), (
        "the exempt input must not be counted among the files that refuse")


def test_a_prepared_name_that_is_not_a_declared_role_is_refused(tmp_path):
    """A typo would exempt a path the run records no owner for."""
    _, _, roles = STAGE3
    with pytest.raises(RunConventionError) as exc:
        open_run(tmp_path, "stage3_recovery", "r1", roles=roles,
                 prepared=("grnat",))
    assert "grnat" in str(exc.value)


def test_a_recorded_run_is_refused_even_with_a_prepared_input(tmp_path):
    """`prepared` relaxes occupancy. It never reopens a finished run."""
    _, spec, roles = STAGE3
    layout = open_run(tmp_path, "stage3_recovery", "r1", roles=roles)
    _fill(layout, roles)
    record_run(layout, spec=spec, plan={}, implementation={}, status={},
               roles=present_roles(layout, roles))
    with pytest.raises(RunConventionError) as exc:
        open_run(tmp_path, "stage3_recovery", "r1", roles=roles,
                 prepared=("grant",))
    assert MANIFEST_NAME in str(exc.value)


def test_a_prepared_input_is_recorded_as_one_of_the_runs_roles(tmp_path):
    """Which is the point of declaring it rather than exempting a path.

    An input the run consumes but never records has the same ownership problem
    as the flat file it replaces: it exists, and nothing says which run it
    belongs to.
    """
    _, spec, roles = STAGE3
    _prepared_run(tmp_path)
    layout = open_run(tmp_path, "stage3_recovery", "r1", roles=roles,
                      prepared=("grant",))
    _fill(layout, roles, present=["train_log", "checkpoint_manifest"])
    record_run(layout, spec=spec, plan={}, implementation={}, status={},
               roles=present_roles(layout, roles))
    assert read_run(tmp_path, "stage3_recovery", "r1")["roles"]["grant"] == (
        roles["grant"])


def test_a_whole_directory_prepared_role_exempts_what_is_inside_it(tmp_path):
    """`artifacts/` is a legal role, so it has to be a legal prepared one.

    The second half is the boundary: exempting the tree must not exempt a file
    that merely sits beside it.
    """
    roles = {"staged": "artifacts/", "log": "runtime/a.log"}
    for run_id in ("r1", "r2"):
        staged = tmp_path / RUNS_ROOT / "e" / run_id / "artifacts"
        staged.mkdir(parents=True)
        (staged / "input.jsonl").write_text("{}\n")
    assert open_run(tmp_path, "e", "r1", roles=roles, prepared=("staged",))

    beside = tmp_path / RUNS_ROOT / "e" / "r2" / roles["log"]
    beside.parent.mkdir(parents=True, exist_ok=True)
    beside.write_text("stale\n")
    with pytest.raises(RunConventionError) as exc:
        open_run(tmp_path, "e", "r2", roles=roles, prepared=("staged",))
    assert roles["log"] in str(exc.value)


# --- reading back is verification, not deserialization ----------------------

def test_a_tampered_manifest_is_refused_on_read(tmp_path):
    _, spec, roles = STAGE4
    layout = open_run(tmp_path, "stage4_rollout", "r1", roles=roles)
    _fill(layout, roles)
    record_run(layout, spec=spec, plan={}, implementation={},
               status={"passed": False}, roles=present_roles(layout, roles))
    doc = json.loads(manifest_path(layout).read_text())
    doc["status"]["passed"] = True
    manifest_path(layout).write_text(json.dumps(doc, indent=1))
    with pytest.raises(RunConventionError) as exc:
        read_run(tmp_path, "stage4_rollout", "r1")
    assert "self_sha256" in str(exc.value)


def test_a_role_pointing_at_a_file_the_run_never_wrote_is_refused(tmp_path):
    """`build_run_manifest` validates the declaration; the read validates disk."""
    _, spec, roles = STAGE4
    layout = open_run(tmp_path, "stage4_rollout", "r1", roles=roles)
    _fill(layout, roles)
    record_run(layout, spec=spec, plan={}, implementation={}, status={},
               roles=present_roles(layout, roles))
    layout.path(roles["rollout_tokens"]).unlink()
    with pytest.raises(RunConventionError) as exc:
        read_run(tmp_path, "stage4_rollout", "r1")
    assert "does not exist" in str(exc.value)


def test_read_refuses_a_run_that_never_recorded_itself(tmp_path):
    with pytest.raises(RunConventionError) as exc:
        read_run(tmp_path, "stage4_rollout", "never")
    assert MANIFEST_NAME in str(exc.value)


# --- runs are grouped by the stage their work belongs to ---------------------
#
# `logs/runs/stage-<stage_id>/<experiment_id>/<run_id>/`. The stage is DECLARED
# by the experiment's config. Phase (A/B/C), pipeline stage (0-6) and attempt
# are three dimensions; `phase_c1` is an experiment id, not a stage.

def test_the_stage_segment_is_declared_not_guessed():
    from experiments.run_layout import stage_segment

    assert stage_segment("3") == "stage-3"
    assert stage_segment("shared") == "stage-shared"
    for bad in ("c1", "phase_c1", "C", "attempt10", "", "-1", "1.5", "../x",
                "shared2", "01x"):
        with pytest.raises(RunConventionError):
            stage_segment(bad)


def test_a_stage_the_pipeline_grows_later_needs_no_change_here(tmp_path):
    """`range(7)` encoded today's AGENTS.md in the mechanism.

    A stage is DECLARED by an experiment's config. Capping which ones may exist
    means adding a stage requires editing public code, which is the opposite of
    declaring it -- and a wider fixed range has the same shape. The whole path
    must derive, and the run must reach the index, with nothing here edited.
    """
    import importlib.util

    roles = {"session_record": "runtime/session.json"}
    art = ArtifactSpec(spec_id="stage7_v1", required=("session_record",))
    layout = open_run(tmp_path, "some_future_experiment", "r1", roles=roles,
                      stage_id="7")
    assert layout.root == (tmp_path / RUNS_ROOT / "stage-7"
                           / "some_future_experiment" / "r1")
    _fill(layout, roles)
    record_run(layout, spec=art, plan={}, implementation={}, status={},
               roles=present_roles(layout, roles))

    spec = importlib.util.spec_from_file_location(
        "rri_stage7",
        Path(__file__).resolve().parents[2] / "scripts/architecture/record_run_index.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    found = {(r["experiment_id"], r.get("stage")) for r in mod.discover_v3(tmp_path)}
    assert ("some_future_experiment", "7") in found, (
        "a declared stage beyond today's range did not reach the index")


def test_a_stage_grouped_run_lands_under_its_stage(tmp_path):
    _, spec, roles = STAGE3
    layout = open_run(tmp_path, "stage3_recovery", "r1", roles=roles, stage_id="3")
    assert layout.root == tmp_path / RUNS_ROOT / "stage-3" / "stage3_recovery" / "r1"
    _fill(layout, roles)
    record_run(layout, spec=spec, plan={}, implementation={}, status={},
               roles=present_roles(layout, roles))
    assert (layout.root / MANIFEST_NAME).is_file()


def test_non_pipeline_work_gets_an_explicit_bucket(tmp_path):
    """Storage maintenance and engineering validation are not Stage 3."""
    roles = {"evidence": "evidence/probe.json"}
    layout = open_run(tmp_path, "cuda_stage_f", "s1", roles=roles,
                      stage_id="shared")
    assert "stage-shared" in str(layout.root)


def test_the_legacy_root_is_still_addressable(tmp_path):
    """Runs whose grant and closeout name a two-level path are read there.

    Moving a run whose identity was issued against its location breaks the
    lineage that makes it evidence, so old runs are found, not relocated.
    """
    _, _, roles = STAGE4
    layout = open_run(tmp_path, "stage4_rollout", "r1", roles=roles)
    assert layout.root == tmp_path / RUNS_ROOT / "stage4_rollout" / "r1"


# --- a README describes a directory; it is never evidence --------------------

def test_a_readme_is_the_one_other_file_a_run_root_may_hold():
    from experiments.run_layout import README_NAME, area_of

    assert area_of(README_NAME) == "root"
    #: By NAME, not by extension: "ignore Markdown" would let any .md file
    #: accumulate in a run root unowned.
    for other in ("NOTES.md", "readme.md", "README.txt", "docs.md"):
        with pytest.raises(RunConventionError):
            area_of(other)


def test_a_run_root_holding_only_a_readme_can_still_be_opened(tmp_path):
    from experiments.run_layout import README_NAME

    _, _, roles = STAGE4
    root = tmp_path / RUNS_ROOT / "stage-4" / "stage4_rollout" / "r1"
    root.mkdir(parents=True)
    (root / README_NAME).write_text("# what this directory is\n")
    layout = open_run(tmp_path, "stage4_rollout", "r1", roles=roles, stage_id="4")
    assert (layout.root / README_NAME).read_text().startswith("# what")


def test_a_readme_does_not_excuse_anything_else_in_the_root(tmp_path):
    """The dead-launcher rule survives README support."""
    from experiments.run_layout import README_NAME

    _, _, roles = STAGE4
    root = tmp_path / RUNS_ROOT / "stage-4" / "stage4_rollout" / "r1"
    (root / "artifacts").mkdir(parents=True)
    (root / README_NAME).write_text("# desc\n")
    (root / "artifacts" / "rollouts.jsonl").write_text("{}\n")
    with pytest.raises(RunConventionError) as exc:
        open_run(tmp_path, "stage4_rollout", "r1", roles=roles, stage_id="4")
    assert "died before recording" in str(exc.value)
    assert README_NAME not in str(exc.value)
