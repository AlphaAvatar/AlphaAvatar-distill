"""The cleanup's behaviours, exercised rather than asserted about.

Each test here runs the thing it checks: the budget deriver against the real
records, the relocator's planner against the real tree, the index builder over
both layouts, the link resolver over every document. A test that asserted "the
file exists" or "the string is in the README" would have passed on every defect
this cleanup fixed — the index counted the wrong set, the deriver divided the
wrong balance, and the README named the wrong directory, all while existing.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))


def load(rel: str) -> dict:
    return json.loads((REPO / rel).read_text())


def _missing_tracked_logs() -> list[str]:
    """Tracked `logs/` files that are not on disk right now.

    A pod receives a STAGED subset of the repository, and the pod-environment
    simulator models that by moving the rest aside. Anything here that reads the
    `logs/` tree as a whole -- the catalog renderer, the relocation record --
    describes a different tree there, and three of these tests failed inside the
    sweep for exactly that reason. On a paid pod that is a CPU-gate failure.

    The condition is OBSERVED, not flagged: git says what should be present and
    the filesystem says what is. A `skipif` keyed on a simulator variable would
    be inverted on the pod, which does not set it -- this repository has already
    lost a session to that.
    """
    out = subprocess.run(["git", "ls-files", "logs"], cwd=REPO,
                         capture_output=True, text=True, check=True).stdout.split()
    return [f for f in out if not (REPO / f).exists()]


#: Applied to the tests that read the tree as a whole. The rest run everywhere.
needs_whole_tree = pytest.mark.skipif(
    bool(_missing_tracked_logs()),
    reason=("the logs/ tree is partially staged, so a check that reads it as a "
            "whole would describe a different tree"))


# --- the four balances -----------------------------------------------------

class TestTheBudgetIsDerivedNotRestated:
    @pytest.fixture(scope="class")
    def derived(self):
        from consolidate.derive_budget import derive
        return derive(REPO)

    def test_the_fundable_count_divides_the_formal_allowance(self, derived):
        """Not the package balance. The engineering allowance cannot pay for a
        formal probe, and dividing the package total by the ceiling is what
        produced a count one too high in four documents at once."""
        formal = derived["formal"]["remaining_usd"]
        ceiling = derived["per_session_ceiling_usd"]
        assert derived["full_ceiling_sessions_fundable"] == int(formal // ceiling)

        package = derived["package"]["remaining_usd"]
        if int(package // ceiling) != int(formal // ceiling):
            #: The two really do disagree right now, which is why this matters.
            assert derived["full_ceiling_sessions_fundable"] < int(package // ceiling)

    def test_the_limits_do_not_add_up_to_each_other_by_accident(self, derived):
        assert (derived["formal"]["allowance_usd"]
                + derived["engineering"]["allowance_usd"]
                == pytest.approx(derived["package"]["allowance_usd"]))

    def test_the_package_book_agrees_with_the_sessions(self, derived):
        priced = [s for s in derived["sessions"] if s["cost_usd"] is not None]
        assert priced, "no session cost was read at all"
        assert sum(s["cost_usd"] for s in priced) == pytest.approx(
            derived["package"]["spent_usd"], abs=1e-4)

    def test_the_snapshot_carries_what_the_deriver_computes(self, derived):
        """The snapshot may hold the numbers; it may not hold DIFFERENT ones."""
        b = load("logs/current_state.json")["budget"]
        assert b["formal_remaining_usd"] == derived["formal"]["remaining_usd"]
        assert b["engineering_remaining_usd"] == derived["engineering"]["remaining_usd"]
        assert b["full_ceiling_sessions_fundable"] == \
            derived["full_ceiling_sessions_fundable"]
        assert b["_derived_by"] == "scripts/consolidate/derive_budget.py"

    def test_an_unreadable_session_is_not_read_as_free(self):
        """The deriver reported `$0.0000 spent` against a package that had
        booked `$0.3960`, because the index entry shape it expected was not the
        one on disk. Silence about spend is the dangerous direction."""
        from consolidate import derive_budget as m
        entry = {"experiment_id": "phase_c1", "run_id": "attempt10",
                 "components": {"root": "logs/runs/phase_c1/attempt10"}}
        assert m._outcome_paths(REPO, entry), (
            "a legacy entry's closeout is unreachable, so its cost would read "
            "as unknown and the package would look unspent")


# --- the index sees every run, in both layouts ------------------------------

class TestTheIndexAccountsForBothLayouts:
    @pytest.fixture(scope="class")
    def index(self):
        return load("logs/runs/index.json")

    def test_by_experiment_totals_reconcile_with_the_entries(self, index):
        """It counted `legacy` while `runs` held legacy + modern, so every
        experiment that had migrated under-reported itself."""
        counted = sum(v["total"] for v in index["counts"]["by_experiment"].values())
        assert counted == len(index["runs"]) + len(index["unrecorded"])

    def test_each_experiment_count_matches_its_own_entries(self, index):
        for exp, c in index["counts"]["by_experiment"].items():
            recorded = sum(1 for r in index["runs"] if r["experiment_id"] == exp)
            unrec = sum(1 for r in index["unrecorded"] if r["experiment_id"] == exp)
            assert (c["recorded"], c["unrecorded"]) == (recorded, unrec), exp

    def test_a_run_in_each_layout_is_found(self, index):
        roots = [r.get("root") or (r.get("components") or {}).get("root") or ""
                 for r in [*index["runs"], *index["unrecorded"]]]
        assert any("/runs/stage-" in r for r in roots), "no stage-grouped run found"
        assert any(r.startswith("logs/runs/") and "/stage-" not in r
                   for r in roots), "no two-level run found"
        assert any(not r.startswith("logs/runs/") for r in roots), \
            "no pre-run-directory evidence found"

    def test_the_kinds_are_declared_so_an_entry_need_not_be_inferred(self, index):
        for k in ("recorded_run", "prepared_not_executed",
                  "historical_evidence_no_manifest", "legacy_aggregate",
                  "description_only"):
            assert k in index["kinds"], k

    def test_a_prepared_run_is_not_reported_as_a_dead_launcher(self, index):
        prepared = [r for r in index["unrecorded"] if "PREPARED" in r.get("why", "")]
        for r in prepared:
            root = REPO / r["root"]
            areas = {p.relative_to(root).parts[0] for p in root.rglob("*")
                     if p.is_file() and len(p.relative_to(root).parts) > 1}
            assert areas == {"governance"}, (
                f"{r['root']} is called prepared but holds {areas}")


# --- entry points point at owners ------------------------------------------

class TestTheEntryPointsResolve:
    def test_every_owner_the_readme_names_exists(self):
        import re
        text = (REPO / "logs/README.md").read_text()
        for target in re.findall(r"\]\(([^)#\s]+)\)", text):
            if target.startswith(("http", "mailto:")):
                continue
            assert (REPO / "logs" / target).exists(), f"logs/README.md -> {target}"

    def test_state_md_names_an_owner_for_every_figure_it_states(self):
        """STATE.md may state a number only beside the thing that owns it."""
        text = (REPO / "logs/STATE.md").read_text()
        assert "derive_budget.py" in text, (
            "the budget table states figures without naming the deriver")
        assert "BUDGET_LEDGER.md" in text and "runs/index.json" in text

    def test_the_phase_c1_readme_locates_attempts_where_they_are(self):
        """It said attempts 1-12 were under `runs/phase_c1/`; three are."""
        index = load("logs/runs/index.json")
        roots = {r["run_id"]: (r.get("root")
                               or (r.get("components") or {}).get("root", ""))
                 for r in [*index["runs"], *index["unrecorded"]]
                 if r["experiment_id"] == "phase_c1"}
        text = (REPO / "logs/runs/stage-1/phase_c1/README.md").read_text()
        under_runs = [k for k, v in roots.items() if v.startswith("logs/runs/phase_c1/")]
        assert len(under_runs) == 3, under_runs
        assert "1–9" in text or "1-9" in text, (
            "the README does not say where attempts 1-9 are")
        #: Quoted spans excluded: the correction QUOTES the old wording to say
        #: what it fixed, and a check that cannot tell a citation from a claim
        #: forces documentation to describe its own defects vaguely.
        import re as _re
        unquoted = _re.sub(r'\*"[^"]*"\*', "", text)
        assert "attempts 1–12 remain" not in unquoted

    def test_the_readme_is_not_counted_as_a_run_product(self):
        """A directory holding only its own README has not executed."""
        from experiments.run_layout import present_roles, layout_for
        import tempfile
        with tempfile.TemporaryDirectory() as t:
            lay = layout_for(Path(t), "exp", "r1", "1")
            lay.create({"evidence": "evidence/e.json"})
            (lay.root / "evidence").mkdir(parents=True, exist_ok=True)
            (lay.root / "evidence" / "README.md").write_text("# describes\n")
            assert present_roles(lay, {"evidence": "evidence/"}) == {}


# --- protected history ------------------------------------------------------

class TestRegisteredEvidenceIsProtected:
    def test_the_relocator_refuses_a_registered_component(self):
        """24 were moved before this guard existed."""
        from consolidate.relocate_logs import plan
        p = plan(REPO)
        reasons = {e["path"]: e["reason"] for e in p["exceptions"]}
        index = load("logs/runs/index.json")
        registered = [rel for e in index["runs"]
                      for rel in (e.get("components") or {}).values()]
        loose = [r for r in registered
                 if (REPO / r).is_file() and Path(r).parent.as_posix() == "logs"]
        assert loose, "no registered component sits loose at logs/ root"
        moving = {m["from"] for m in p["moves"]}
        for r in loose:
            #: What matters is that it does not MOVE. Which guard stops it is
            #: an ordering detail -- several apply to the same file, and
            #: demanding one particular reason would fail on a file that is
            #: also read by an executable.
            assert r not in moving, (
                f"{r} is a registered component and the planner would move it")
            assert r in reasons, f"{r} is neither moved nor explained"

    def test_the_link_fixer_will_not_edit_inside_a_registered_directory(self):
        """It edited 15 READMEs there; the digest covers the directory."""
        from consolidate.fix_doc_links import protected_dirs, repair
        dirs = protected_dirs(REPO)
        assert dirs, "no registered directories found, so the guard is vacuous"
        r = repair(REPO, write=False)
        assert r["skipped_registered_evidence"], (
            "nothing was skipped, so the guard did not engage")
        for doc in r["fixed"]:
            assert not any(doc["doc"].startswith(d + "/") for d in dirs), doc

    def test_every_registered_component_still_hashes_to_its_record(self):
        from architecture.record_run_index import digest_of
        index = load("logs/runs/index.json")
        drift = []
        for e in index["runs"]:
            for role, rel in (e.get("components") or {}).items():
                want = (e.get("component_digests") or {}).get(role)
                if not want:
                    continue
                if digest_of(REPO / rel)["digest"] != want:
                    drift.append(f"{e['experiment_id']}/{e['run_id']}::{role} {rel}")
        assert not drift, "registered evidence changed:\n  " + "\n  ".join(drift)


# --- the relocation is followable both ways ---------------------------------

@needs_whole_tree
class TestTheRelocationIsTraceable:
    @pytest.fixture(scope="class")
    def record(self):
        return load("logs/maintenance/log_relocation.json")

    def test_every_move_landed_and_its_origin_is_gone(self, record):
        for m in record["moves"]:
            assert (REPO / m["to"]).exists(), m["to"]
            assert not (REPO / m["from"]).exists(), (
                f"{m['from']} still exists: this was a copy, not a move, and "
                "two editable records of one fact is what we are ending")

    def test_the_reverted_moves_are_back_and_recorded(self, record):
        rev = record["reverted"]["moved_and_restored"]
        assert rev, "the record claims no reverts"
        for r in rev:
            assert (REPO / r).is_file(), f"{r} was reverted but is not there"

    def test_every_exception_states_what_holds_it(self, record):
        allowed = {"entry_point", "named_by_executable", "index_registered",
                   "pinned_by_record", "unattributed",
                   "cited_by_frozen_evidence"}
        for e in record["exceptions"]:
            assert e["reason"] in allowed, e
            assert e.get("detail"), e
            assert "for safety" not in e["detail"].lower()

    def test_no_document_still_cites_a_moved_path(self, record):
        """Outside registered evidence. A citation INSIDE a digest-covered
        directory is left alone by design -- and the relocator now refuses to
        move anything such a document names, so the citation still resolves."""
        from consolidate.fix_doc_links import protected_dirs
        frozen = protected_dirs(REPO)
        stale = []
        for m in record["moves"]:
            for f in REPO.rglob("*.md"):
                if ".git" in f.parts:
                    continue
                rel = f.relative_to(REPO).as_posix()
                if any(rel.startswith(d + "/") for d in frozen):
                    continue
                try:
                    if m["from"] in f.read_text():
                        stale.append((f.relative_to(REPO).as_posix(), m["from"]))
                except (OSError, UnicodeDecodeError):
                    continue
        assert not stale, f"documents still cite moved paths: {stale[:5]}"


# --- the generated navigation is regenerable --------------------------------

class TestTheNavigationIsDerivedFromTheTree:
    """It went stale thirty times in one afternoon while being hand-edited."""

    @needs_whole_tree
    def test_rendering_again_changes_nothing(self):
        """Run the renderer against the live tree; its output must already be
        what is committed. A drift means a file moved and the navigation was
        not regenerated — which is the state this replaced."""
        from consolidate.render_log_navigation import (
            CATALOG, MARK_END, MARK_START, render_catalog,
            render_experiment_readme, runs_by_experiment)
        text = (REPO / CATALOG).read_text()
        i, j = text.index(MARK_START), text.index(MARK_END)
        assert text[i:j] == render_catalog(REPO) + "\n", (
            "CATALOG.md's classification is stale; re-run "
            "scripts/consolidate/render_log_navigation.py --write")

        runs = runs_by_experiment(REPO)
        for d in sorted((REPO / "logs/experiments").iterdir()):
            if not d.is_dir():
                continue
            assert (d / "README.md").read_text() == render_experiment_readme(
                d, REPO, runs), f"{d.name}/README.md is stale"

    @needs_whole_tree
    def test_the_catalog_classifies_every_entry(self):
        import re
        named = set(re.findall(r"`([^`]+)`", (REPO / CATALOG_REL).read_text()))
        missing = [p.name for p in sorted((REPO / "logs").iterdir())
                   if p.name not in named and f"{p.name}/" not in named]
        assert not missing, f"unclassified: {missing}"

    def test_a_generated_readme_states_no_cost_or_status(self):
        """A generated document restating an owned fact is the duplication the
        cleanup removes; it would also go stale silently."""
        import re
        for d in sorted((REPO / "logs/experiments").iterdir()):
            if not d.is_dir():
                continue
            body = (d / "README.md").read_text()
            assert not re.search(r"\$\d", body), f"{d.name}: states a cost"
            assert not re.search(r"\b[0-9a-f]{8,}\b", body), f"{d.name}: states a SHA"


CATALOG_REL = "logs/CATALOG.md"


# --- consecutive sweeps do not overwrite each other's evidence --------------

class TestSweepOutputsAreIsolated:
    """Two executions of ONE commit must not overwrite each other.

    The first fix keyed the path on the commit and this suite asserted that the
    same tree "is the same claim and may share" it. That contract was wrong. A
    sweep that fails and a sweep that then passes are two observations of one
    tree, and the simulator redirects with `>`, so the second truncated the
    first -- including raw output a committed record cites.
    """

    def _mod(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "rpe", REPO / "scripts/autoinit/record_pod_environment.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def test_two_trees_get_two_locations(self, tmp_path):
        m = self._mod()
        a = m.allocate_execution("a" * 40, root=str(tmp_path))
        b = m.allocate_execution("b" * 40, root=str(tmp_path))
        assert set(a[1:]).isdisjoint(set(b[1:])), (a, b)

    def test_two_executions_of_one_commit_get_two_locations(self, tmp_path):
        """The contract this replaces asserted the opposite."""
        m = self._mod()
        first = m.allocate_execution("c" * 40, root=str(tmp_path))
        second = m.allocate_execution("c" * 40, root=str(tmp_path))
        assert first[0] != second[0], "two executions share one identity"
        assert set(first[1:]).isdisjoint(set(second[1:])), (first, second)

    def test_a_retry_after_failure_does_not_truncate_the_failed_evidence(
            self, tmp_path):
        """The exact sequence that lost evidence: run, fail, fix, run again."""
        import hashlib

        m = self._mod()
        head = "d" * 40
        eid1, junit1, log1 = m.allocate_execution(head, root=str(tmp_path))
        Path(junit1).write_text("<testsuite failures='3'/>")
        Path(log1).write_text("first run, 3 failed\n")
        before = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                  for p in (junit1, log1)}

        eid2, junit2, log2 = m.allocate_execution(head, root=str(tmp_path))
        Path(junit2).write_text("<testsuite failures='0'/>")
        Path(log2).write_text("second run, 0 failed\n")

        for p, h in before.items():
            assert Path(p).is_file(), f"{p} disappeared"
            assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == h, (
                f"{p} was rewritten by the retry")
        assert Path(junit1).read_text() != Path(junit2).read_text()

    def test_an_interrupted_second_execution_leaves_the_first_intact(
            self, tmp_path):
        """An execution that claims a directory and writes nothing still must
        not consume or damage the previous one."""
        import hashlib

        m = self._mod()
        head = "e" * 40
        _, junit1, log1 = m.allocate_execution(head, root=str(tmp_path))
        Path(junit1).write_text("<testsuite/>")
        Path(log1).write_text("complete\n")
        h = hashlib.sha256(Path(junit1).read_bytes()).hexdigest()

        eid2, junit2, _ = m.allocate_execution(head, root=str(tmp_path))
        # interrupted: the directory exists, the output never arrives
        assert not Path(junit2).exists()
        assert Path(junit1).is_file()
        assert hashlib.sha256(Path(junit1).read_bytes()).hexdigest() == h

        # and a THIRD execution does not reuse the abandoned directory
        eid3, junit3, _ = m.allocate_execution(head, root=str(tmp_path))
        assert eid3 not in (eid2,)
        assert Path(junit3).parent != Path(junit2).parent

    def test_an_explicit_path_that_already_exists_is_refused(self, tmp_path):
        """Naming your own output is allowed; writing over evidence is not."""
        import subprocess as sp

        taken = tmp_path / "junit.xml"
        taken.write_text("<testsuite/>")
        r = sp.run([sys.executable,
                    str(REPO / "scripts/autoinit/record_pod_environment.py"),
                    "--junit", str(taken)],
                   capture_output=True, text=True,
                   env={"PYTHONPATH": f"{REPO}/src", "PATH": "/usr/bin:/bin",
                        "HOME": str(tmp_path)})
        assert r.returncode != 0, "an existing output path was accepted"
        assert "already exists" in (r.stdout + r.stderr)
        assert taken.read_text() == "<testsuite/>", "the refusal still wrote"

    def test_reading_an_existing_sweep_is_a_separate_path(self, tmp_path):
        """`--from-existing` must name what it reads, allocate nothing and run
        nothing. It shared a branch with execution and so still resolved a
        default output path it had just created."""
        import subprocess as sp

        r = sp.run([sys.executable,
                    str(REPO / "scripts/autoinit/record_pod_environment.py"),
                    "--from-existing"],
                   capture_output=True, text=True,
                   env={"PYTHONPATH": f"{REPO}/src", "PATH": "/usr/bin:/bin",
                        "HOME": str(tmp_path)})
        assert r.returncode != 0
        assert "cannot guess which one" in (r.stdout + r.stderr)

    def test_no_shared_default_remains(self):
        src = (REPO / "scripts/autoinit/record_pod_environment.py").read_text()
        assert "podsim_junit.xml" not in src, "the shared JUnit default is back"
        assert "podsim_pytest.log" not in src, "the shared log default is back"
        assert "def sweep_outputs" not in src, (
            "the commit-only path helper is back; it cannot distinguish two "
            "executions of one tree")

    def test_a_committed_record_still_names_its_own_raw_output(self):
        """The record's `evidence` block must point at paths that belong to the
        sweep it describes, so the two cannot drift apart again."""
        rec = load("logs/c1_pod_environment_verification.json")
        ev = rec.get("evidence") or {}
        assert ev.get("junit") and ev.get("pytest_log"), ev


# --- engineering spend is attributed, not assumed ---------------------------

class TestEngineeringSpendIsAttributedByPackage:
    """It read ONE hardcoded campaign and returned zero for anything CLOSED, so
    a second campaign was invisible and a campaign that spent inside this
    package stopped counting the moment it closed."""

    def _mod(self):
        from consolidate import derive_budget as m
        return m

    def _campaign(self, root: Path, name: str, *, cost, granted=None,
                  package_id=None, status="OPEN"):
        d = root / "logs/validations" / name / "v1"
        d.mkdir(parents=True, exist_ok=True)
        camp = {"campaign_id": name, "booked_usd": cost, "status": status,
                "authorization": f"logs/validations/{name}/v1/authorization.json"}
        if package_id:
            camp["package_id"] = package_id
        (d / "campaign.json").write_text(json.dumps(camp))
        (d / "authorization.json").write_text(json.dumps(
            {"granted_utc": granted} if granted else {}))
        return d

    def test_a_closed_campaign_inside_the_package_still_counts(self, tmp_path):
        """Closure says no MORE may be spent, not that nothing was."""
        m = self._mod()
        self._campaign(tmp_path, "inside", cost=1.25,
                       granted="2026-09-12T00:00:00Z", status="CLOSED")
        got = [m._attribute(c, "pkg", "2026-09-11")
               for c in m.engineering_campaigns(tmp_path)]
        assert [c["attribution"] for c in got] == ["this_package"]
        assert got[0]["cost_usd"] == 1.25

    def test_a_campaign_before_the_package_is_not_charged_twice(self, tmp_path):
        m = self._mod()
        self._campaign(tmp_path, "earlier", cost=0.04,
                       granted="2026-09-10T00:00:00Z")
        got = [m._attribute(c, "pkg", "2026-09-11")
               for c in m.engineering_campaigns(tmp_path)]
        assert got[0]["attribution"] == "before_this_package"

    def test_several_campaigns_are_all_found(self, tmp_path):
        """A second engineering run was invisible to the hardcoded path."""
        m = self._mod()
        self._campaign(tmp_path, "one", cost=0.5, granted="2026-09-12T00:00:00Z")
        self._campaign(tmp_path, "two", cost=0.25, granted="2026-09-13T00:00:00Z")
        got = [m._attribute(c, "pkg", "2026-09-11")
               for c in m.engineering_campaigns(tmp_path)]
        assert len(got) == 2
        assert sum(c["cost_usd"] for c in got
                   if c["attribution"] == "this_package") == 0.75

    def test_an_explicit_package_id_beats_the_date(self, tmp_path):
        m = self._mod()
        self._campaign(tmp_path, "stated", cost=2.0,
                       granted="2026-09-12T00:00:00Z", package_id="other")
        got = [m._attribute(c, "pkg", "2026-09-11")
               for c in m.engineering_campaigns(tmp_path)]
        assert got[0]["attribution"] == "other_package"

    def test_an_unknown_cost_is_flagged_not_zeroed(self, tmp_path):
        m = self._mod()
        d = tmp_path / "logs/validations/nocost/v1"
        d.mkdir(parents=True)
        (d / "campaign.json").write_text(json.dumps({"campaign_id": "nocost"}))
        got = [m._attribute(c, "pkg", "2026-09-11")
               for c in m.engineering_campaigns(tmp_path)]
        assert got[0]["attribution"] == "unknown"
        assert got[0]["cost_usd"] is None

    def test_an_unknown_attribution_is_flagged_not_dated_away(self, tmp_path):
        m = self._mod()
        self._campaign(tmp_path, "undated", cost=1.0, granted=None)
        got = [m._attribute(c, "pkg", "2026-09-11")
               for c in m.engineering_campaigns(tmp_path)]
        assert got[0]["attribution"] == "unknown"

    def test_one_resource_is_not_counted_twice(self, tmp_path):
        """Cost comes from the campaign's own book when it states one, not from
        re-summing subruns on top of it."""
        m = self._mod()
        d = tmp_path / "logs/validations/dup/v1"
        d.mkdir(parents=True)
        (d / "campaign.json").write_text(json.dumps(
            {"campaign_id": "dup", "booked_usd": 0.04,
             "subruns": [{"cost_usd": 0.0073}, {"cost_usd": 0.0145},
                         {"cost_usd": 0.0182}]}))
        got = m.engineering_campaigns(tmp_path)
        assert got[0]["cost_usd"] == 0.04, "the subruns were added on top"

    def test_the_live_deriver_reports_no_pending_reconciliation(self):
        d = self._mod().derive(REPO)
        assert d["pending_reconciliation"] == [], d["pending_reconciliation"]
        assert d["full_ceiling_sessions_fundable"] is not None

    def test_an_unreconciled_balance_yields_no_fundable_count(self, monkeypatch):
        """An arithmetic summary over an unknown is not a number a launch may
        be planned against."""
        m = self._mod()
        real = m.engineering_campaigns
        monkeypatch.setattr(m, "engineering_campaigns", lambda root: [
            *real(root),
            {"campaign": "logs/validations/x/v1/campaign.json", "cost_usd": None,
             "package_id": None, "granted_utc": None, "attribution": None}])
        d = m.derive(REPO)
        assert d["full_ceiling_sessions_fundable"] is None
        assert d["pending_reconciliation"] == [
            "logs/validations/x/v1/campaign.json"]


# --- the current view cannot go stale against its own owner -----------------

class TestTheCurrentViewReadsItsOwner:
    """STATE.md said the readiness record was a launch-bound FAILURE while the
    file it linked to was a diagnostic PASS. Three facts had been collapsed into
    one prose line, so the line was wrong about all three at once."""

    def _view(self):
        from consolidate.render_log_navigation import readiness_view
        return readiness_view(REPO)

    def test_state_md_agrees_with_the_record_it_links_to(self):
        from consolidate.render_log_navigation import (R_BEGIN, R_END,
                                                       render_readiness)
        text = (REPO / "logs/STATE.md").read_text()
        i, j = text.index(R_BEGIN), text.index(R_END) + len(R_END)
        assert text[i:j] == render_readiness(REPO), (
            "STATE.md's readiness block is stale; re-run "
            "scripts/consolidate/render_log_navigation.py --write")

    def test_the_three_facts_are_stated_separately(self):
        v = self._view()
        assert set(v) >= {"latest", "launch_bound_ready",
                          "launch_bound_failures"}
        #: Independent by CONSTRUCTION, not by today's values: this asserted
        #: `verdict == "PASS"`, which pinned a moment and went red the next time
        #: a sweep failed -- the very coupling the split exists to remove.
        assert v["latest"]["verdict"] in ("PASS", "FAIL")
        #: A diagnostic record can never make a launch-bound ready, whatever it
        #: found. That is the conjunction one combined line could not express.
        if v["latest"]["kind"] == "diagnostic":
            assert v["launch_bound_ready"] is False
        assert v["launch_bound_failures"], "the failed launch-bound is not kept"

    def test_the_snapshot_carries_the_same_derivation(self):
        lv = load("logs/current_state.json")["latest_verification"]
        v = self._view()
        assert lv["latest_sweep"]["verdict"] == v["latest"]["verdict"]
        assert lv["latest_sweep"]["kind"] == v["latest"]["kind"]
        assert lv["launch_bound_ready"] == v["launch_bound_ready"]

    def test_a_launch_bound_claim_requires_the_current_tree(self, tmp_path,
                                                            monkeypatch):
        """A PASS swept at another commit is not a prepared launch-bound."""
        from consolidate import render_log_navigation as m
        v = self._view()
        #: The conjunction, asserted directly rather than through whichever
        #: combination happens to be live: readiness requires kind AND verdict
        #: AND the tree, so failing any one of them denies it.
        ready = (v["latest"]["kind"] == "launch_bound"
                 and v["latest"]["verdict"] == "PASS"
                 and v["latest"]["describes_head"])
        assert v["launch_bound_ready"] == ready
        assert not (v["latest"]["kind"] == "diagnostic"
                    and v["launch_bound_ready"]), (
            "a diagnostic record is being read as launch-bound readiness")

    @needs_whole_tree
    def test_every_superseded_verdict_is_preserved(self):
        """The live record holds ONE sweep and the next replaces it, so a
        launch-bound FAILURE survived only in git history."""
        hist = load("logs/experiments/phase_c1/readiness_history.json")
        kinds = {(e["record_kind"], e["verdict"]) for e in hist["entries"]}
        assert ("launch_bound", "FAIL") in kinds, (
            "the failed launch-bound sweep is not preserved anywhere outside "
            "git history")
        assert len(hist["entries"]) > 10, len(hist["entries"])


# --- a run owns its own governance evidence ---------------------------------

class TestRunOwnedGovernanceEvidence:
    """Both the readiness record and the issued authorization were single
    repository-root files that each run overwrote in turn, with the closeout
    copying them into the run afterwards. The evidence a session launched under
    therefore lived where the next session would replace it."""

    def _L(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "L", REPO / "scripts/pod/autoinit_c1_launch.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def test_each_run_gets_its_own_readiness_path(self):
        from experiments.phase_c1.pod_environment import record_path_for
        a = record_path_for("attempt13", "1")
        b = record_path_for("attempt14", "1")
        assert a != b
        assert a.startswith("logs/runs/stage-1/phase_c1/attempt13/")
        assert a.endswith("governance/readiness.json")

    def test_each_run_gets_its_own_authorization_path(self):
        L = self._L()
        a, b = L.auth_path_for("attempt13"), L.auth_path_for("attempt14")
        assert a != b
        assert a.endswith("governance/authorization.json")

    def test_the_lineage_exemption_names_one_file_not_the_directory(self):
        """The narrowness IS the protection: a grant committed after the sweep
        must still invalidate it, which is what forces grant-then-sweep."""
        from experiments.phase_c1.pod_environment import (
            permitted_post_sweep_paths)
        permitted = permitted_post_sweep_paths("attempt13", "1")
        assert len(permitted) == 1, permitted
        assert permitted[0].endswith("governance/readiness.json")
        for p in permitted:
            assert not p.endswith("governance")
            assert not p.endswith("governance/")

    def test_a_grant_in_the_same_directory_is_not_exempt(self):
        """Same directory, different file: the exemption is per path."""
        from experiments.phase_c1.pod_environment import (
            permitted_post_sweep_paths)
        L = self._L()
        grant = (f"logs/runs/stage-1/phase_c1/attempt13/"
                 f"{L.C1_RUN_ROLES['grant']}")
        assert grant not in permitted_post_sweep_paths("attempt13", "1")

    def test_neither_artifact_is_copied_in_by_the_closeout(self):
        """Copying was the workaround for living at an overwritten path."""
        L = self._L()
        collected = [role for _src, role in L._RUN_GOVERNANCE]
        assert "authorization" not in collected
        assert "readiness_record" not in collected
        #: and both are still declared roles of the run
        assert "authorization" in L.C1_RUN_ROLES
        assert "readiness_record" in L.C1_RUN_ROLES

    def test_both_are_exempt_from_occupancy_by_name(self):
        L = self._L()
        assert set(L._RUN_PREPARED) == {"grant", "readiness_record"}

    def test_the_issuer_refuses_to_overwrite_an_authorization(self, tmp_path):
        """One-use means one artifact. Replacing one in place is how a consumed
        authorization becomes indistinguishable from a fresh one."""
        src = (REPO / "scripts/autoinit/issue_c1_authorization.py").read_text()
        assert "already exists. An authorization is one-use" in src
        assert "--run-id" in src

    def test_the_global_paths_are_pointers_not_records(self):
        from experiments.phase_c1 import pod_environment as pe
        assert pe.RECORD_POINTER == "logs/c1_pod_environment_verification.json"
        #: The alias stays: every pre-2026-09-12 record is at that path.
        assert pe.RECORD_PATH == pe.RECORD_POINTER
        L = self._L()
        assert L.auth_path_for(None) == L.AUTH_POINTER


# --- a relocation never edits what it does not own --------------------------

class TestARelocationRewritesOnlyWhatItOwns:
    """A bulk path rewrite edited `code_state.untracked_files` inside the frozen
    battery manifest under `artifacts/`. That list is covered by the manifest's
    own `manifest_sha256`, so the scorer refused it and 34 tests plus the pod
    CPU gate failed. Restored byte-identical from the canonical store.

    A path string inside an artifact may be part of what a hash covers."""

    def test_the_protected_trees_are_declared(self):
        from consolidate.relocate_logs import NEVER_REWRITE, rewritable
        assert "artifacts/" in NEVER_REWRITE
        assert not rewritable("artifacts/stage3/c1_confirmation_v1/manifest.json")
        assert not rewritable("logs/archive/anything.md")
        assert not rewritable("logs/runs/phase_c1/attempt10/manifest.json")
        assert rewritable("logs/README.md")

    def test_the_frozen_battery_manifest_verifies(self):
        """The check the scorer makes, made here at $0."""
        from aadistill.infrastructure.manifest import sha256_json
        p = REPO / "artifacts/stage3/c1_confirmation_v1/manifest.json"
        if not p.is_file():
            pytest.skip("the battery working copy is not present")
        m = json.loads(p.read_text())
        calc = sha256_json({k: v for k, v in m.items() if k != "manifest_sha256"})
        assert calc == m.get("manifest_sha256"), (
            "the battery manifest does not match its own manifest_sha256; "
            "something edited a frozen artifact")

    def test_it_matches_the_canonical_store(self):
        from aadistill.infrastructure.manifest import sha256_file
        a = REPO / "artifacts/stage3/c1_confirmation_v1/manifest.json"
        b = Path("/home/ecs-user/aad-artifacts/autoinit/c1_confirmation_v1/"
                 "manifest.json")
        if not (a.is_file() and b.is_file()):
            pytest.skip("one of the copies is not present on this machine")
        assert sha256_file(a) == sha256_file(b), (
            "the working copy has drifted from the canonical artifact")
