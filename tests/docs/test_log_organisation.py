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
    """Both raw-output paths were fixed, so every sweep replaced the previous
    one's — including sweeps whose record cites them as evidence."""

    def _mod(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "rpe", REPO / "scripts/autoinit/record_pod_environment.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def test_two_trees_get_two_locations(self):
        m = self._mod()
        a = m.sweep_outputs("a" * 40)
        b = m.sweep_outputs("b" * 40)
        assert set(a).isdisjoint(set(b)), (
            f"a sweep of a different tree writes the same files: {a} {b}")

    def test_the_same_tree_is_the_same_claim_and_may_share(self):
        m = self._mod()
        assert m.sweep_outputs("c" * 40) == m.sweep_outputs("c" * 40)

    def test_the_path_is_reproducible_from_the_record(self):
        """Keyed on the head commit, not a clock: a record that names an
        unreproducible path cannot be checked afterwards."""
        m = self._mod()
        j, _ = m.sweep_outputs("d" * 40)
        assert "d" * 12 in j and not any(ch.isdigit() for ch in Path(j).parent.name[6:])

    def test_no_shared_default_remains(self):
        src = (REPO / "scripts/autoinit/record_pod_environment.py").read_text()
        assert "podsim_junit.xml" not in src, "the shared JUnit default is back"
        assert "podsim_pytest.log" not in src, "the shared log default is back"

    def test_a_committed_record_still_names_its_own_raw_output(self):
        """The record's `evidence` block must point at paths that belong to the
        sweep it describes, so the two cannot drift apart again."""
        rec = load("logs/c1_pod_environment_verification.json")
        ev = rec.get("evidence") or {}
        assert ev.get("junit") and ev.get("pytest_log"), ev
