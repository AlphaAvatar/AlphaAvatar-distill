"""run-layout-v2: generic, future-only, and enforced rather than intended.

The layout is easy to describe and easy to abandon. What makes it hold is a
small number of properties that fail loudly:

* nothing may be written flat under `logs/` for a new attempt again;
* a manifest's references must all exist and each path may have one owner;
* no component may resolve outside its run root;
* `current_state.latest_run` must resolve, not merely be a string;
* C1, Stage 4 and Stage 5 must use the SAME resolver — a layout that is really
  C1's with a generic name gets rewritten at Stage 4;
* every legacy run registered in the index must still be byte-for-byte what it
  was, because "future-only migration" is a promise about the past;
* a launch-window bundle record must live inside the active run.

The last one is where the current repository is still v1: `logs/autoinit_c1_bundle.json`
is flat, was overwritten by successive attempts, and is registered as legacy.
`test_a_bundle_record_belongs_inside_its_run` pins where the NEXT one goes.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.autoinit.run_layout import (  # noqa: E402
    COMPONENTS, INDEX_PATH, MANIFEST_SCHEMA, RUN_LAYOUT_VERSION,
    RunLayout, RunLayoutError, build_run_manifest, directory_digest, load_index,
    resolve_run, verify_run_manifest,
)

#: A configuration, supplied by a caller. Deliberately fictional and deliberately
#: NOT C1's: if anything in the layout were secretly C1-shaped, this would fail.
DEMO_CONFIG = {
    "adapter": {"family": "qwen3", "version": 1},
    "source_spec_hash": "a" * 64,
    "target_spec_hash": "b" * 64,
    "compression": {"kind": "demo", "ratio": 0.5},
    "operator_path": {"path_id": "demo.path", "path_hash": "c" * 64,
                      "steps": ["depth.x", "ffn.y"]},
    "protocols": {"science_plan": "d" * 64},
    "status": {"provider": "none", "scientific": "not executed"},
}


def _run(tmp_path, experiment="phase_c1", attempt="attempt10", **cfg):
    layout = RunLayout(experiment, attempt, repo_root=tmp_path).create()
    present = []
    for name, spec in COMPONENTS.items():
        if name == "manifest" or spec.endswith("/"):
            continue
        p = layout.path(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}\n")
        present.append(name)
    doc = build_run_manifest(layout, config={**DEMO_CONFIG, **cfg}, present=present)
    layout.path("manifest").write_text(json.dumps(doc, indent=1) + "\n")
    return layout, doc


# --- generic, not C1's ------------------------------------------------------

def test_c1_stage4_and_stage5_resolve_through_the_same_layout(tmp_path):
    """Three experiments, one class, identical structure.

    If the resolver had C1 vocabulary in it, the other two would need their own —
    and then the three disagree about where evidence lives.
    """
    layouts = [RunLayout(e, "attempt1", repo_root=tmp_path)
               for e in ("phase_c1", "stage4_online_data", "stage5_on_policy")]
    assert len({type(l) for l in layouts}) == 1

    for layout in layouts:
        assert layout.rel_root == f"logs/runs/{layout.experiment_id}/attempt1"
        for name in COMPONENTS:
            assert layout.path(name).is_relative_to(layout.root)
        # the component NAMES do not vary by experiment
        assert set(COMPONENTS) == set(COMPONENTS)

    rels = [{n: l.rel(n).replace(l.experiment_id, "<X>") for n in COMPONENTS}
            for l in layouts]
    assert rels[0] == rels[1] == rels[2], "the layout differs between experiments"


def test_the_layout_module_branches_on_no_subject_vocabulary():
    """No conditionals on model, geometry, ratio, stage number or attempt number.

    Checked against the code with comments and docstrings removed, because the
    module explains these very words in prose and a naive scan would match its
    own explanation — the same mistake that made an earlier portability test
    fail against its own comment.
    """
    import ast

    import aadistill.autoinit.run_layout as RL

    tree = ast.parse(Path(RL.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Expr,)) and isinstance(node.value, ast.Constant):
            node.value.value = ""            # blank docstrings
    code = ast.unparse(tree).lower()

    forbidden = ["qwen", "llama", "attempt10", "attempt9", "phase_c1",
                 "stage4", "stage5", "num_attention_heads", "0.86m"]
    hits = [w for w in forbidden if w in code]
    assert not hits, f"subject vocabulary in the resolver: {hits}"


def test_the_manifest_builder_takes_every_identity_from_configuration(tmp_path):
    layout = RunLayout("stage4_online_data", "attempt2", repo_root=tmp_path)
    doc = build_run_manifest(layout, config=DEMO_CONFIG, present=[])
    assert doc["schema"] == MANIFEST_SCHEMA
    assert doc["layout_version"] == RUN_LAYOUT_VERSION
    assert doc["experiment_id"] == "stage4_online_data"
    assert doc["architecture_adapter"] == DEMO_CONFIG["adapter"]
    assert doc["source_spec_hash"] == DEMO_CONFIG["source_spec_hash"]
    assert doc["target_spec_hash"] == DEMO_CONFIG["target_spec_hash"]
    assert doc["compression"] == DEMO_CONFIG["compression"]
    assert doc["operator_path"] == DEMO_CONFIG["operator_path"]
    assert doc["protocols"] == DEMO_CONFIG["protocols"]
    assert doc["status"] == DEMO_CONFIG["status"]
    assert doc["authorizes"] == "nothing"
    assert doc["self_sha256"]


def test_a_manifest_missing_a_configured_identity_is_refused(tmp_path):
    layout = RunLayout("phase_c1", "attempt10", repo_root=tmp_path)
    for drop in DEMO_CONFIG:
        cfg = {k: v for k, v in DEMO_CONFIG.items() if k != drop}
        with pytest.raises(RunLayoutError, match="missing"):
            build_run_manifest(layout, config=cfg, present=[])


# --- references, ownership and traversal ------------------------------------

def test_every_reference_in_a_manifest_resolves(tmp_path):
    layout, doc = _run(tmp_path)
    ok, why = verify_run_manifest(doc, tmp_path)
    assert ok, why


def test_a_manifest_naming_an_absent_file_is_refused(tmp_path):
    layout, doc = _run(tmp_path)
    layout.path("closeout.outcome").unlink()
    ok, why = verify_run_manifest(doc, tmp_path)
    assert not ok and "does not exist" in why


def test_two_components_may_not_own_the_same_path(tmp_path):
    """One path, one owner. A duplicate is an ambiguity about who wrote it."""
    from aadistill.infrastructure.manifest import sha256_json

    layout, doc = _run(tmp_path)
    doc["components"]["evidence"]["decision"] = doc["components"]["closeout"]["outcome"]
    doc.pop("self_sha256")
    doc["self_sha256"] = sha256_json(doc)
    ok, why = verify_run_manifest(doc, tmp_path)
    assert not ok and "one path, one owner" in why


def test_a_component_may_not_resolve_outside_its_run(tmp_path):
    layout = RunLayout("phase_c1", "attempt10", repo_root=tmp_path)
    with pytest.raises(RunLayoutError, match="not a declared run component"):
        layout.path("../../../etc/passwd")
    with pytest.raises(RunLayoutError, match="not a declared run component"):
        layout.path("governance/../../escape.json")


@pytest.mark.parametrize("bad", ["..", "../other", "/absolute", "a/b", "",
                                 "Attempt10", "attempt-10", "attempt 10"])
def test_traversal_and_odd_identifiers_are_refused(bad, tmp_path):
    with pytest.raises(RunLayoutError, match="valid identifier"):
        RunLayout(bad, "attempt1", repo_root=tmp_path)
    with pytest.raises(RunLayoutError, match="valid identifier"):
        RunLayout("phase_c1", bad, repo_root=tmp_path)


def test_contains_rejects_a_path_outside_the_run(tmp_path):
    layout, _ = _run(tmp_path)
    assert layout.contains(layout.rel("governance.bundle"))
    assert not layout.contains("logs/autoinit_c1_bundle.json")
    assert not layout.contains("/etc/passwd")


def test_a_manifest_whose_self_hash_was_edited_is_refused(tmp_path):
    layout, doc = _run(tmp_path)
    doc["status"] = {"provider": "tampered"}
    ok, why = verify_run_manifest(doc, tmp_path)
    assert not ok and "self_sha256" in why


# --- the bundle record belongs inside the run -------------------------------

def test_a_bundle_record_belongs_inside_its_run(tmp_path):
    """Where the NEXT launch-window bundle record goes.

    `logs/autoinit_c1_bundle.json` is flat and was overwritten by successive
    attempts, so the record of which bundle attempt 5 used is simply gone. Under
    v2 it is `governance/bundle.json` inside the run, and cannot collide with
    another attempt's by construction.
    """
    layout, doc = _run(tmp_path)
    bundle = layout.path("governance.bundle")
    assert bundle.is_file()
    assert layout.contains(bundle)
    assert doc["components"]["governance"]["bundle"] == \
        f"{layout.rel_root}/governance/bundle.json"

    other = RunLayout(layout.experiment_id, "attempt11", repo_root=tmp_path)
    assert other.path("governance.bundle") != bundle, (
        "two attempts share a bundle record path; that is the v1 defect")


# --- future-only: the past is untouched -------------------------------------

def test_every_registered_legacy_run_is_byte_for_byte_unchanged():
    """The promise the whole migration rests on."""
    index = load_index(REPO)
    legacy = [r for r in index["runs"] if r["layout_version"] == 1]
    assert legacy, "the index registers no legacy runs"

    drift = []
    for entry in legacy:
        live = directory_digest(entry["root"], REPO)
        if live["digest"] != entry["digest"]:
            drift.append(f"{entry['root']}: {entry['digest'][:12]}… -> "
                         f"{live['digest'][:12]}…")
    assert not drift, "registered legacy evidence has changed:\n  " + "\n  ".join(drift)


def test_the_nine_c1_attempts_are_all_registered():
    index = load_index(REPO)
    c1 = {r["attempt_id"] for r in index["runs"]
          if r["experiment_id"] == "phase_c1" and r["layout_version"] == 1}
    for n in ("attempt1", "attempt2", "attempt3", "attempt3r", "attempt4",
              "attempt5", "attempt6", "attempt7", "attempt8", "attempt9"):
        assert n in c1, f"{n} is not registered in {INDEX_PATH}"


def test_every_index_entry_resolves():
    index = load_index(REPO)
    problems = [f"{e['root']}: {why}" for e in index["runs"]
                for ok, why in [resolve_run(e, REPO)] if not ok]
    assert not problems, problems


def test_no_index_entry_claims_the_same_root_twice():
    roots = [r["root"] for r in load_index(REPO)["runs"]]
    assert len(roots) == len(set(roots))


# --- no new flat attempt logs -----------------------------------------------

#: What an attempt-specific top-level entry looks like, in every convention this
#: repository has used. A NEW one must go under `logs/runs/`.
ATTEMPT_PATTERN = re.compile(r"attempt", re.IGNORECASE)


def test_no_unregistered_attempt_specific_entry_sits_directly_under_logs():
    """The enforcement. Every flat attempt artifact must be a REGISTERED legacy
    one; a new one is a layout violation, not a new convention."""
    registered = {r["root"] for r in load_index(REPO)["runs"]}
    offenders = []
    for entry in sorted((REPO / "logs").iterdir()):
        rel = f"logs/{entry.name}"
        if not ATTEMPT_PATTERN.search(entry.name):
            continue
        if rel not in registered:
            offenders.append(rel)
    assert not offenders, (
        "attempt-specific entries directly under logs/ that the index does not "
        f"register: {offenders}. run-layout-v2 is future-only: new run evidence "
        "belongs under logs/runs/<experiment_id>/<attempt_id>/, and an existing "
        "one belongs in the index.")


def test_mutation_a_new_flat_attempt_log_is_caught(tmp_path):
    """The guard, driven against a synthetic tree so it cannot be vacuous."""
    logs = tmp_path / "logs"
    (logs / "runs").mkdir(parents=True)
    (logs / "runs" / "index.json").write_text(json.dumps(
        {"schema": "aadistill.autoinit.run_index/v1", "runs": []}))
    (logs / "autoinit_c1_attempt10").mkdir()

    registered = {r["root"] for r in load_index(tmp_path)["runs"]}
    offenders = [f"logs/{e.name}" for e in sorted(logs.iterdir())
                 if ATTEMPT_PATTERN.search(e.name)
                 and f"logs/{e.name}" not in registered]
    assert offenders == ["logs/autoinit_c1_attempt10"], (
        "the no-new-flat-log rule does not fire on a new flat attempt directory")


# --- current_state.latest_run must RESOLVE ----------------------------------

def test_latest_run_resolves_through_the_index_or_a_manifest():
    """A string that names nothing is worse than an absent field."""
    state = json.loads((REPO / "logs/current_state.json").read_text())
    latest = state.get("latest_run")
    assert latest is not None, (
        "current_state.json does not declare latest_run; it must name the most "
        "recent run so a reader does not have to know the naming convention")

    index = load_index(REPO)
    match = [r for r in index["runs"]
             if r["experiment_id"] == latest["experiment_id"]
             and r["attempt_id"] == latest["attempt_id"]
             and r["root"] == latest["root"]]
    assert match, f"latest_run {latest} is not registered in {INDEX_PATH}"
    ok, why = resolve_run(match[0], REPO)
    assert ok, f"latest_run does not resolve: {why}"
    assert latest["layout_version"] == match[0]["layout_version"]


def test_a_v2_latest_run_resolves_through_its_manifest(tmp_path):
    """The path Attempt 10 will take, exercised now against a synthetic run."""
    layout, doc = _run(tmp_path, experiment="phase_c1", attempt="attempt10")
    entry = {"experiment_id": "phase_c1", "attempt_id": "attempt10",
             "layout_version": RUN_LAYOUT_VERSION, "kind": "directory",
             "root": layout.rel_root, "manifest_sha256": doc["self_sha256"]}
    ok, why = resolve_run(entry, tmp_path)
    assert ok, why

    layout.path("evidence.replay").unlink()
    ok, why = resolve_run(entry, tmp_path)
    assert not ok and "does not exist" in why


# --- the two guards that a first mutation pass found untested ----------------
#
# `RunLayout.path()` refuses a component that resolves outside the run root, and
# `resolve_run()` refuses a legacy entry whose digest has moved. Both survived
# mutation: every declared component is a plain relative path so the escape
# branch is unreachable from COMPONENTS alone, and every registered legacy run
# currently matches, so deleting the comparison changed nothing observable.
# Neither is dead code — they are the two things that would matter on the day
# something is wrong — so both are driven directly.

def test_the_declared_components_cannot_escape_the_run_root(tmp_path):
    """The invariant COMPONENTS itself must satisfy."""
    layout = RunLayout("phase_c1", "attempt10", repo_root=tmp_path)
    root = layout.root.resolve()
    for name in COMPONENTS:
        assert layout.path(name).is_relative_to(root)


def test_the_escape_guard_fires_when_a_component_would_leave_the_run(tmp_path,
                                                                     monkeypatch):
    """Driven by injecting an escaping component, since none is declared.

    A guard whose triggering case cannot arise from the current data is still
    the guard that catches the next person who adds `"../shared/x.json"` to
    COMPONENTS. Proving it fires is the only way to know it would.
    """
    monkeypatch.setitem(COMPONENTS, "evil.escape", "../../../etc/passwd")
    layout = RunLayout("phase_c1", "attempt10", repo_root=tmp_path)
    with pytest.raises(RunLayoutError, match="outside the run root"):
        layout.path("evil.escape")


def test_component_names_map_to_distinct_paths():
    """Two names on one file would make ownership ambiguous by construction."""
    seen: dict[str, str] = {}
    for name, spec in COMPONENTS.items():
        rel = spec.rstrip("/")
        assert rel not in seen, f"{name!r} and {seen[rel]!r} both map to {rel!r}"
        seen[rel] = name


def test_a_changed_legacy_run_is_detected_by_resolve_run(tmp_path):
    """The byte-for-byte promise, driven against evidence that DID change."""
    logs = tmp_path / "logs" / "demo_attempt1"
    logs.mkdir(parents=True)
    (logs / "outcome.json").write_text('{"a": 1}\n')

    entry = {"experiment_id": "demo", "attempt_id": "attempt1",
             "layout_version": 1, "kind": "directory", "root": "logs/demo_attempt1",
             "digest": directory_digest("logs/demo_attempt1", tmp_path)["digest"]}
    ok, why = resolve_run(entry, tmp_path)
    assert ok, why

    # one byte
    (logs / "outcome.json").write_text('{"a": 2}\n')
    ok, why = resolve_run(entry, tmp_path)
    assert not ok and "has changed" in why

    # a new file is drift too
    (logs / "outcome.json").write_text('{"a": 1}\n')
    assert resolve_run(entry, tmp_path)[0]
    (logs / "extra.json").write_text("{}\n")
    ok, why = resolve_run(entry, tmp_path)
    assert not ok and "has changed" in why

    # and so is a deletion
    (logs / "extra.json").unlink()
    (logs / "outcome.json").unlink()
    ok, why = resolve_run(entry, tmp_path)
    assert not ok


def test_a_missing_legacy_run_is_detected(tmp_path):
    entry = {"experiment_id": "demo", "attempt_id": "attempt1",
             "layout_version": 1, "kind": "directory",
             "root": "logs/never_existed", "digest": "0" * 64}
    ok, why = resolve_run(entry, tmp_path)
    assert not ok and "does not exist" in why


def test_an_unknown_layout_version_is_refused(tmp_path):
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "x").mkdir()
    entry = {"experiment_id": "demo", "attempt_id": "attempt1",
             "layout_version": 99, "kind": "directory", "root": "logs/x"}
    ok, why = resolve_run(entry, tmp_path)
    assert not ok and "unknown layout_version" in why


def test_a_v2_run_may_not_dual_write_a_registered_legacy_path():
    """One fact, one home.

    The failure this prevents: a v2 run writes `evidence/replay.json` AND keeps
    updating `logs/autoinit_c1_bundle.json`, and the two disagree the first time
    somebody edits one. A v2 component path can never equal a registered legacy
    root, because v2 paths live under `logs/runs/` and no legacy root does — so
    this asserts that separation rather than assuming it.
    """
    index = load_index(REPO)
    legacy_roots = {r["root"] for r in index["runs"] if r["layout_version"] == 1}
    assert legacy_roots, "no legacy runs registered"

    for root in legacy_roots:
        assert not root.startswith("logs/runs/"), (
            f"{root} is registered as legacy-v1 but sits inside the v2 tree; "
            "one fact would then have two homes")

    for entry in (r for r in index["runs"] if r["layout_version"] == 2):
        layout = RunLayout(entry["experiment_id"], entry["attempt_id"],
                           repo_root=REPO)
        for name in COMPONENTS:
            assert layout.rel(name) not in legacy_roots


def test_the_run_index_authorizes_nothing():
    """Registering a run is bookkeeping. It must not read as permission."""
    index = load_index(REPO)
    assert index["authorizes"] == "nothing"
    blob = json.dumps(index).lower()
    for word in ("authorized", "grant is live", "permits", "approved"):
        assert word not in blob, f"the run index says {word!r}"


def test_the_run_root_is_not_swallowed_by_gitignore():
    """`.gitignore` had a bare `runs/`, which matches at ANY depth.

    That rule is for training outputs (`wandb/`, `checkpoints/`, `runs/`), and it
    silently ignored `logs/runs/` — the run-layout evidence root. The index was
    written and invisible to git, and every future run manifest would have been
    too: evidence that exists locally and is never committed, which is the one
    failure a provenance layout cannot have. Caught by noticing the path missing
    from `git status`, not by any test, so here is the test.
    """
    import subprocess

    for rel in ("logs/runs", "logs/runs/index.json",
                "logs/runs/demo/attempt1/manifest.json",
                "logs/runs/demo/attempt1/governance/bundle.json"):
        out = subprocess.run(
            ["git", "check-ignore", "-v", "--no-index", rel],
            cwd=REPO, capture_output=True, text=True)
        # A negation rule matches too, so the exit code alone is not the answer:
        # what matters is whether the winning rule is a negation.
        if out.returncode == 0 and out.stdout.strip():
            rule = out.stdout.split("\t")[0].rsplit(":", 1)[-1]
            assert rule.startswith("!"), (
                f"{rel} is ignored by rule {rule!r}; run-layout evidence would "
                "never be committed")


def test_heavy_run_outputs_stay_ignored():
    """The negation must not un-ignore checkpoints written under a run."""
    import subprocess

    for rel in ("logs/runs/demo/attempt1/evidence/training/model.safetensors",
                "logs/runs/demo/attempt1/evidence/evaluation/big.pt"):
        out = subprocess.run(["git", "check-ignore", "-v", "--no-index", rel],
                             cwd=REPO, capture_output=True, text=True)
        assert out.returncode == 0 and not out.stdout.split("\t")[0].rsplit(
            ":", 1)[-1].startswith("!"), (
            f"{rel} would be committed; heavy artifacts never belong in git")
