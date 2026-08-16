"""The pod simulator must leave the tree exactly as it found it.

On 2026-08-15 it did not. `restore` used `mv "$saved" "$dest"`, and when the test
run had RECREATED the destination — `artifacts/audit` is recreated by any driver
rehearsal — `mv` moved the saved directory *inside* the recreated one. It
happened twice, burying the real `artifacts/audit` at
`artifacts/audit/artifacts@audit/artifacts@audit/` and turning 11 tests into
skips. Skips are not failures, so the suite still read green and the corruption
was found only because a skip COUNT looked wrong.

The second defect was that two sweeps could overlap: the loser's `restore` walks
`$HIDE` and adopts the winner's saved paths.

This drives the real shipped script against a temporary tree, not a copy of its
logic.
"""

import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/pod/simulate_pod_env.sh"


def run_sim(root: Path, hide: Path, hidden_paths: str, cmd: str, **extra):
    env = {**os.environ,
           "PODSIM_ROOT": str(root),
           "HIDE_DIR": str(hide),
           "PODSIM_LOCK": str(hide) + ".lock",
           "HIDDEN_PATHS": hidden_paths,
           "PODSIM_CMD": cmd,
           **extra}
    return subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True,
                          env=env)


def snapshot(root: Path) -> dict[str, str]:
    """Every file under root, by relative path -> contents."""
    return {str(p.relative_to(root)): p.read_text()
            for p in sorted(root.rglob("*")) if p.is_file()}


def build_tree(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "artifacts/audit/three_mode").mkdir(parents=True)
    (root / "artifacts/audit/three_mode/anchor.json").write_text('{"real": 1}')
    (root / "artifacts/audit/e4_comparison.json").write_text('{"e4": 1}')
    (root / "artifacts/keep").mkdir(parents=True)
    (root / "artifacts/keep/untouched.txt").write_text("keep me")
    return root


def test_restore_reproduces_the_exact_pre_state_when_the_run_recreates_it(tmp_path):
    """The exact 2026-08-15 corruption, reproduced and refused.

    The command recreates `artifacts/audit` while it is hidden — as any driver
    rehearsal does. Restore must put the ORIGINAL back, not nest it underneath
    the recreation.
    """
    root = build_tree(tmp_path)
    before = snapshot(root)
    hide = tmp_path / "hidden"

    # The run recreates the very path that is hidden.
    recreate = ("mkdir -p artifacts/audit/autoinit_preflight && "
                "echo made-by-the-run > artifacts/audit/autoinit_preflight/x.json")
    r = run_sim(root, hide, "artifacts/audit", recreate)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "restored the hidden artifacts" in r.stdout

    after = snapshot(root)
    # THE ASSERTION THAT MATTERS: byte-identical to the pre-simulation state.
    assert after == before, (
        "the tree was not restored exactly; "
        f"added={sorted(set(after) - set(before))} "
        f"lost={sorted(set(before) - set(after))}")
    # And specifically, no nesting of the encoded name.
    assert not (root / "artifacts/audit/artifacts@audit").exists()
    assert (root / "artifacts/audit/three_mode/anchor.json").read_text() == '{"real": 1}'


def test_the_recreated_content_is_quarantined_not_deleted(tmp_path):
    """Deleting on a restore path is how a bug becomes data loss."""
    root = build_tree(tmp_path)
    hide = tmp_path / "hidden"
    recreate = ("mkdir -p artifacts/audit && "
                "echo made-by-the-run > artifacts/audit/evidence.json")
    r = run_sim(root, hide, "artifacts/audit", recreate)
    assert r.returncode == 0, r.stdout + r.stderr
    quarantined = list((Path(str(hide) + ".recreated")).rglob("evidence.json"))
    assert quarantined, "the recreated content was destroyed rather than quarantined"
    assert quarantined[0].read_text().strip() == "made-by-the-run"


def test_a_clean_run_that_recreates_nothing_still_restores_exactly(tmp_path):
    root = build_tree(tmp_path)
    before = snapshot(root)
    hide = tmp_path / "hidden"
    r = run_sim(root, hide, "artifacts/audit", "true")
    assert r.returncode == 0, r.stdout + r.stderr
    assert snapshot(root) == before
    assert not Path(str(hide) + ".recreated").exists()


def test_a_concurrent_sweep_is_refused_rather_than_racing(tmp_path):
    """The loser must not run AND must not restore the winner's saved paths.

    `$HIDE` is deliberately populated with a winner's saved path: an empty
    `$HIDE` leaves the loser nothing to steal, so the second assertion would
    hold vacuously. The property itself is provided by arming the trap only
    after the lock is acquired, which
    `test_the_trap_is_armed_only_after_the_lock_is_held` pins directly.
    """
    root = build_tree(tmp_path)
    hide = tmp_path / "hidden"
    lock = Path(str(hide) + ".lock")
    lock.mkdir(parents=True)                     # a sweep is "already running"
    hide.mkdir(parents=True)
    winners_saved = hide / "artifacts@keep"      # mid-flight, owned by the winner
    winners_saved.mkdir()
    (winners_saved / "untouched.txt").write_text("keep me")
    try:
        r = run_sim(root, hide, "artifacts/audit", "true")
        assert r.returncode == 3, r.stdout + r.stderr
        assert "another pod simulation holds" in r.stderr
        # It refused BEFORE hiding anything of its own.
        assert (root / "artifacts/audit/three_mode/anchor.json").is_file()
        # And it did NOT restore the winner's saved path out from under it.
        assert winners_saved.is_dir(), (
            "the loser restored the winner's saved paths; that is the race")
        assert (winners_saved / "untouched.txt").is_file()
    finally:
        lock.rmdir()


def test_a_leftover_hide_dir_is_refused_rather_than_adopted(tmp_path):
    """A previous sweep that died before restoring must be handled by a human,
    not silently adopted under this run's assumptions."""
    root = build_tree(tmp_path)
    hide = tmp_path / "hidden"
    hide.mkdir(parents=True)
    (hide / "artifacts@somethingelse").write_text("orphaned by a dead sweep")
    r = run_sim(root, hide, "artifacts/audit", "true")
    assert r.returncode == 4, r.stdout + r.stderr
    assert "did not restore" in r.stderr
    # The orphan is untouched and the lock is released.
    assert (hide / "artifacts@somethingelse").is_file()
    assert not Path(str(hide) + ".lock").exists()


def test_the_lock_is_released_after_a_normal_run(tmp_path):
    root = build_tree(tmp_path)
    hide = tmp_path / "hidden"
    r = run_sim(root, hide, "artifacts/audit", "true")
    assert r.returncode == 0
    assert not Path(str(hide) + ".lock").exists(), (
        "a stale lock would block every future sweep")


def test_the_trap_is_armed_only_after_the_lock_is_held():
    """The ordering IS the anti-race mechanism, so it is asserted directly.

    A loser exits at the lock check; if `trap restore` were installed before
    that check, the loser's exit would restore the winner's saved paths. There
    is no runtime guard doing this job -- an in-function holder check was
    unreachable and was removed rather than left as decoration -- so the
    property lives in the ordering and is pinned here.
    """
    src = SCRIPT.read_text()
    lock_at = src.index('if ! mkdir "$LOCK"')
    trap_at = src.index("trap restore EXIT INT TERM")
    leftover_at = src.index('REFUSING: $HIDE is not empty')
    assert lock_at < trap_at, (
        "the trap is armed before the lock is acquired; a losing sweep would "
        "restore the winner's saved paths on its way out")
    assert leftover_at < trap_at, (
        "the leftover-$HIDE check must also precede the trap, or refusing it "
        "would still trigger a restore")
