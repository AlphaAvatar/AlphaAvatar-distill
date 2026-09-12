"""One schema for every stage, and one index entry per run.

Two claims, and the second is the one the maintainer corrected.

**The schema is not compression-shaped.** The previous contract hard-coded
`logs/runs` as the root, fixed the roles to
`replay/treatment/training/evaluation/decision`, and required every manifest to
carry `adapter`, `source_spec_hash`, `target_spec_hash`, `compression` and
`operator_path`. A dataset build has no operator path; a throughput benchmark
has no source spec. `test_the_schema_carries_every_stage_without_a_source_change`
builds ten genuinely different runs — including two stages that do not exist yet
— through the same functions.

**The index counts runs, not artifact roots.** It reported 77. Attempt 9 is one
run with two surviving components (`logs/autoinit_c1_attempt9` and
`…_grant.json`); the old schema recorded them as two peers, so "how many C1
attempts?" answered 19 for a phase that has had 9. There are **41 logical runs
over 77 components**, and nothing moved to achieve that.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.runtime.run_layout import (  # noqa: E402
    MANIFEST_SCHEMA, RUN_LAYOUT_VERSION, ArtifactSpec, RunLayout,
    RunLayoutError, build_run_manifest, digest_of, verify_run_manifest,
)

INDEX = REPO / "logs/runs/index.json"


def _make(tmp_path, experiment, run, roles, spec=None, **over):
    layout = RunLayout(tmp_path, experiment, run).create(roles)
    for rel in roles.values():
        p = layout.path(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        if not rel.endswith("/"):
            p.write_text("{}\n")
        else:
            p.mkdir(parents=True, exist_ok=True)
    doc = build_run_manifest(
        layout,
        plan=over.get("plan", {"plan_id": f"{experiment}.plan"}),
        implementation=over.get("implementation", {"commit": "0" * 40}),
        status=over.get("status", {"provider": "none", "scientific": "not run"}),
        roles=roles, spec=spec)
    layout.path("manifest.json").write_text(json.dumps(doc, indent=1) + "\n")
    return layout, doc


# --- the schema is stage-agnostic -------------------------------------------

#: Ten runs from different parts of the project, two of them stages nobody has
#: named yet. Same builder, same verifier, no source change.
STAGES = [
    ("dataset_build", {"corpus": "corpus/shards.jsonl", "manifest_of_source": "source.json"}),
    ("teacher_rollout", {"rollouts": "rollouts.jsonl", "engine_probe": "engine.json"}),
    ("calibration_stats", {"statistics": "stats.safetensors", "coverage": "coverage.json"}),
    ("initialization_search", {"journal": "search/journal.jsonl", "frontier": "frontier.json"}),
    ("fixed_path_materialization", {"steps": "steps/", "identities": "identities.json"}),
    ("recovery_training", {"train_log": "train/log.jsonl", "checkpoints": "train/ckpt/"}),
    ("evaluation_scoring", {"generations": "eval/gen.jsonl", "scores": "eval/scores.json"}),
    ("quantized_benchmark", {"latency": "bench/latency.json", "memory": "bench/mem.json"}),
    ("infrastructure_validation", {"environment": "env.json", "verdict": "verdict.json"}),
    ("future_stage_seven", {"whatever_this_needs": "a/b/c.json"}),
]


@pytest.mark.parametrize("experiment,roles", STAGES, ids=[s[0] for s in STAGES])
def test_the_schema_carries_every_stage_without_a_source_change(
        experiment, roles, tmp_path):
    layout, doc = _make(tmp_path, experiment, "run1", roles)
    ok, why = verify_run_manifest(doc, tmp_path)
    assert ok, why
    assert doc["schema"] == MANIFEST_SCHEMA
    assert doc["layout_version"] == RUN_LAYOUT_VERSION
    assert set(doc["roles"]) == set(roles)


def test_the_core_requires_no_compression_vocabulary(tmp_path):
    """None of the previously mandatory fields is required, and none is invented."""
    _, doc = _make(tmp_path, "dataset_build", "run1", {"corpus": "c.jsonl"})
    for gone in ("adapter", "source_spec_hash", "target_spec_hash",
                 "compression", "operator_path", "components"):
        assert gone not in doc, f"{gone!r} is still a required manifest field"
    assert doc["plan"] == {"plan_id": "dataset_build.plan"}


def test_the_core_names_no_repository_directory():
    """The run root is the caller's. `logs/runs` must not appear in core."""
    import aadistill.runtime.run_layout as RL

    src = Path(RL.__file__).read_text()
    body = src.split('"""', 2)[-1]          # past the module docstring
    for literal in ('"logs/runs"', "'logs/runs'", '"logs"'):
        assert literal not in body, f"{literal} is hard-coded in the core layout"


def test_a_role_vocabulary_is_the_experiments_not_the_cores(tmp_path):
    """Two experiments, disjoint roles, both valid — because core reads none."""
    _, a = _make(tmp_path, "teacher_rollout", "r1", {"rollouts": "r.jsonl"})
    _, b = _make(tmp_path, "quantized_benchmark", "r1", {"latency": "l.json"})
    assert set(a["roles"]).isdisjoint(set(b["roles"]))
    assert verify_run_manifest(a, tmp_path)[0]
    assert verify_run_manifest(b, tmp_path)[0]


# --- what the core DOES enforce ---------------------------------------------

def test_an_artifact_spec_decides_which_roles_are_required(tmp_path):
    spec = ArtifactSpec("rollout.v1", required=("rollouts",), optional=("engine_probe",))
    with pytest.raises(RunLayoutError, match="required role"):
        _make(tmp_path, "teacher_rollout", "r1", {"engine_probe": "e.json"}, spec=spec)
    _, doc = _make(tmp_path, "teacher_rollout", "r2",
                   {"rollouts": "r.jsonl", "engine_probe": "e.json"}, spec=spec)
    assert verify_run_manifest(doc, tmp_path, spec=spec)[0]


def test_a_role_outside_the_spec_is_refused(tmp_path):
    spec = ArtifactSpec("rollout.v1", required=("rollouts",))
    with pytest.raises(RunLayoutError, match="not declared"):
        _make(tmp_path, "teacher_rollout", "r3",
              {"rollouts": "r.jsonl", "surprise": "s.json"}, spec=spec)


def test_a_spec_cannot_declare_a_role_twice(tmp_path):
    with pytest.raises(RunLayoutError, match="required and optional"):
        ArtifactSpec("bad.v1", required=("x",), optional=("x",))


def test_one_path_one_owner(tmp_path):
    with pytest.raises(RunLayoutError, match="one path, one owner"):
        _make(tmp_path, "dataset_build", "r1",
              {"corpus": "same.json", "shadow": "same.json"})


@pytest.mark.parametrize("bad", ["/etc/passwd", "../escape.json",
                                 "a/../../escape.json"])
def test_a_role_path_may_not_escape_the_run(bad, tmp_path):
    layout = RunLayout(tmp_path, "dataset_build", "r1")
    with pytest.raises(RunLayoutError):
        layout.path(bad)


@pytest.mark.parametrize("bad", ["..", "../x", "/abs", "a/b", "", "Upper", "a-b"])
def test_identifiers_are_single_safe_segments(bad, tmp_path):
    with pytest.raises(RunLayoutError, match="not valid here"):
        RunLayout(tmp_path, bad, "r1")
    with pytest.raises(RunLayoutError, match="not valid here"):
        RunLayout(tmp_path, "e1", bad)


def test_a_manifest_naming_an_absent_file_is_refused(tmp_path):
    layout, doc = _make(tmp_path, "dataset_build", "r1", {"corpus": "c.jsonl"})
    layout.path("c.jsonl").unlink()
    ok, why = verify_run_manifest(doc, tmp_path)
    assert not ok and "does not exist" in why


def test_an_edited_manifest_is_refused(tmp_path):
    _, doc = _make(tmp_path, "dataset_build", "r1", {"corpus": "c.jsonl"})
    doc["status"] = {"provider": "tampered"}
    ok, why = verify_run_manifest(doc, tmp_path)
    assert not ok and "self_sha256" in why


def test_reserved_manifest_fields_cannot_be_shadowed(tmp_path):
    layout = RunLayout(tmp_path, "dataset_build", "r1").create()
    with pytest.raises(RunLayoutError, match="reserved"):
        build_run_manifest(layout, plan={}, implementation={}, status={},
                           roles={}, extra={"roles": {"sneaky": "x"}})


# --- the index: one entry per logical run -----------------------------------

@pytest.fixture(scope="module")
def index() -> dict:
    return json.loads(INDEX.read_text())


def test_one_experiment_and_run_id_appears_exactly_once(index):
    keys = [(r["experiment_id"], r["run_id"]) for r in index["runs"]]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    assert not dupes, f"duplicated logical runs: {dupes}"


def test_the_index_counts_runs_not_artifact_roots(index):
    """The correction. 41 runs over 77 components, where v1 reported 77 runs."""
    runs = [r for r in index["runs"] if r["layout_version"] == 1]
    components = sum(r["n_components"] for r in runs)
    assert index["counts"]["runs_legacy_v1"] == len(runs)
    assert index["counts"]["legacy_components"] == components
    assert components > len(runs), (
        "every run has exactly one component, so this schema change bought "
        "nothing — check the discovery patterns")
    #: `by_experiment` counts BOTH layouts since 2026-09-12 -- it counted only
    #: `legacy` while `runs` held legacy + modern, so every experiment that had
    #: migrated under-reported itself. The original protection is kept by
    #: checking the legacy portion directly, which is what "not counting
    #: artifacts again" was ever about.
    legacy_c1 = [r for r in runs if r["experiment_id"] == "phase_c1"]
    assert len(legacy_c1) == 10, (
        "C1 has had attempts 1-9 plus 3r in the legacy layout; any other "
        "number is counting artifacts again")
    c1 = index["counts"]["by_experiment"]["phase_c1"]
    assert c1["recorded"] + c1["unrecorded"] == c1["total"]
    assert c1["total"] > len(legacy_c1), (
        "the per-experiment count ignores the current layout again")


def test_every_component_path_has_exactly_one_owner(index):
    owners: dict[str, list[str]] = {}
    for r in index["runs"]:
        for role, path in r["components"].items():
            owners.setdefault(path, []).append(
                f"{r['experiment_id']}/{r['run_id']}::{role}")
    clashes = {p: o for p, o in owners.items() if len(o) > 1}
    assert not clashes, f"paths claimed by more than one run/role: {clashes}"


def test_every_historical_component_is_byte_identical(index):
    drift = []
    for r in index["runs"]:
        if r["layout_version"] != 1:
            continue
        for role, path in r["components"].items():
            live = digest_of(REPO / path)["digest"]
            if live != r["component_digests"][role]:
                drift.append(f"{r['experiment_id']}/{r['run_id']}::{role} ({path})")
    assert not drift, "registered historical evidence has changed:\n  " + "\n  ".join(drift)


def test_the_aggregate_digest_detects_change(index, tmp_path):
    """Addition, removal and a single byte must all move the aggregate."""
    def aggregate(components, digests):
        return hashlib.sha256("".join(
            f"{r}:{components[r]}:{digests[r]}\n" for r in sorted(components)
        ).encode()).hexdigest()

    r = next(x for x in index["runs"] if x["layout_version"] == 1
             and x["n_components"] >= 2)
    comp, dig = dict(r["components"]), dict(r["component_digests"])
    assert aggregate(comp, dig) == r["aggregate_digest"]

    role = sorted(comp)[0]
    changed = dict(dig, **{role: "0" * 64})
    assert aggregate(comp, changed) != r["aggregate_digest"], "byte change undetected"

    removed_c = {k: v for k, v in comp.items() if k != role}
    removed_d = {k: v for k, v in dig.items() if k != role}
    assert aggregate(removed_c, removed_d) != r["aggregate_digest"], "removal undetected"

    added_c = dict(comp, extra="logs/new_thing.json")
    added_d = dict(dig, extra="1" * 64)
    assert aggregate(added_c, added_d) != r["aggregate_digest"], "addition undetected"


def test_no_run_dual_writes_a_legacy_path(index):
    legacy = {p for r in index["runs"] if r["layout_version"] == 1
              for p in r["components"].values()}
    for r in index["runs"]:
        if r["layout_version"] == 1:
            continue
        for p in r["components"].values():
            assert p not in legacy, f"{p} is written by both a legacy and a current run"
    for p in legacy:
        assert not p.startswith("logs/runs/"), (
            f"{p} is legacy but sits inside the hierarchical tree; one fact, one home")


def test_no_unregistered_attempt_entry_sits_directly_under_logs(index):
    import re

    registered = {p for r in index["runs"] for p in r["components"].values()}
    offenders = [f"logs/{e.name}" for e in sorted((REPO / "logs").iterdir())
                 if re.search("attempt", e.name, re.I)
                 and f"logs/{e.name}" not in registered]
    assert not offenders, (
        f"attempt-specific entries under logs/ that the index does not "
        f"register: {offenders}")


def test_latest_run_resolves_to_exactly_one_entry(index):
    state = json.loads((REPO / "logs/current_state.json").read_text())
    latest = state.get("latest_run")
    assert latest is not None, "current_state.json declares no latest_run"
    #: Across BOTH `runs` and `unrecorded`: the latest run may legitimately be
    #: PREPARED and not executed -- a grant is committed into the run before the
    #: launch-bound sweep, so that state is reachable on purpose. Searching only
    #: `runs` would force the snapshot to name an older run as the latest one.
    everything = [*index["runs"], *index["unrecorded"]]
    matches = [r for r in everything
               if r["experiment_id"] == latest["experiment_id"]
               and r["run_id"] == latest["run_id"]]
    assert len(matches) == 1, f"latest_run {latest} matched {len(matches)} entries"
    entry = matches[0]
    roots = set((entry.get("components") or {}).values())
    if entry.get("root"):
        roots.add(entry["root"])
    assert latest["root"] in roots, (
        f"latest_run root {latest['root']} is not a path of its own entry")
    #: And a prepared run must say so, rather than reading as an executed one.
    if entry in index["unrecorded"]:
        assert "PREPARED" in entry.get("why", "") , entry
        assert "PREPARED" in latest.get("state", ""), (
            "the snapshot names a prepared run as latest without saying it is "
            "prepared, which reads as an execution that happened")


def test_the_index_authorizes_nothing(index):
    assert index["authorizes"] == "nothing"
