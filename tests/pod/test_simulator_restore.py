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
import sys
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
           # The interpreter is an INPUT. `PODSIM_ROOT` here is a synthetic tree
           # with no venv, which is also a pod's condition — and the ambient
           # `command -v python3` the script used to fall back to is what cost C1
           # attempt 6 all 18 cases in this module at the pod CPU gate.
           "PODSIM_PYTHON": sys.executable,
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


# --- the second dimension: HOME and Hugging Face -----------------------------
#
# Hiding gitignored ARTIFACTS was only half of what a pod does not have. C1
# attempt 3R died at the pod's CPU test gate on 2026-09-04 with `14 failed,
# 2650 passed`. Seven were renderer-parity cases reading a `$HOME` Hugging Face
# dataset cache no fresh pod possesses, and two were repository state; the other
# five remain UNEXPLAINED. The simulator existed, was green, and modelled a
# machine that does not exist.
#
# These drive the real script and assert the isolation it now applies, because a
# simulator that quietly kept the dev box's cache is worse than none — it is the
# same green light with a stronger claim attached.

def run_env_sim(root: Path, hide: Path, cmd: str, **extra):
    """A sweep whose PODSIM_CMD reports on the environment it was given."""
    return run_sim(root, hide, "artifacts/audit", cmd,
                   PODSIM_ENV_ROOT=str(hide) + ".env", **extra)


def test_the_simulated_home_is_a_fresh_empty_directory(tmp_path):
    root = build_tree(tmp_path)
    r = run_env_sim(root, tmp_path / "hidden",
                    'echo "HOME=$HOME"; echo "ENTRIES=$(ls -A "$HOME" | wc -l)"')
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ENTRIES=0" in r.stdout, r.stdout
    assert f"HOME={tmp_path}" in r.stdout


def test_the_dev_box_hf_cache_is_not_visible(tmp_path):
    """The exact condition the seven renderer-parity cases must meet to skip."""
    root = build_tree(tmp_path)
    r = run_env_sim(root, tmp_path / "hidden",
                    'test -e "$HOME/.cache/huggingface" && s=present || s=absent;'
                    ' echo "CACHE=$s"; echo "HUB=$HF_HUB_CACHE";'
                    ' echo "DATASETS=$(ls -A "$HF_HUB_CACHE" | wc -l)"')
    assert r.returncode == 0, r.stdout + r.stderr
    # Read the command's own output, not the `running:` echo of its source text.
    assert "CACHE=absent" in r.stdout and "CACHE=present" not in r.stdout
    assert "DATASETS=0" in r.stdout


def test_hf_hub_cache_sits_under_the_isolated_hf_home(tmp_path):
    root = build_tree(tmp_path)
    r = run_env_sim(root, tmp_path / "hidden",
                    'echo "HF_HOME=$HF_HOME"; echo "HUB=$HF_HUB_CACHE"')
    assert r.returncode == 0, r.stdout + r.stderr
    hf_home = [l for l in r.stdout.splitlines() if l.startswith("HF_HOME=")][0][8:]
    hub = [l for l in r.stdout.splitlines() if l.startswith("HUB=")][0][4:]
    assert hub == f"{hf_home}/hub"


def test_variables_that_would_defeat_the_isolation_are_unset(tmp_path):
    """Each of these names the real cache on its own, whatever HF_HOME says."""
    root = build_tree(tmp_path)
    leaky = {"HUGGINGFACE_HUB_CACHE": "/real/hub", "HF_DATASETS_CACHE": "/real/ds",
             "TRANSFORMERS_CACHE": "/real/tf", "XDG_CACHE_HOME": "/real/xdg"}
    r = run_env_sim(root, tmp_path / "hidden",
                    'for v in HUGGINGFACE_HUB_CACHE HF_DATASETS_CACHE '
                    'TRANSFORMERS_CACHE XDG_CACHE_HOME; do '
                    'echo "$v=${!v-UNSET}"; done', **leaky)
    assert r.returncode == 0, r.stdout + r.stderr
    for v in leaky:
        assert f"{v}=UNSET" in r.stdout, r.stdout


def test_a_pod_equivalent_token_is_exported_and_never_printed(tmp_path):
    """Pod setup exports HF_TOKEN before its test gate, so the simulation must
    too — and the value must not reach a log."""
    root = build_tree(tmp_path)
    secret = "hf_thisMustNeverBePrinted12345"
    r = run_env_sim(root, tmp_path / "hidden",
                    'test -n "$HF_TOKEN" && echo TOKEN_PRESENT || echo TOKEN_MISSING',
                    PODSIM_HF_TOKEN=secret)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "TOKEN_PRESENT" in r.stdout
    assert secret not in r.stdout and secret not in r.stderr


def test_the_isolation_is_torn_down_after_a_normal_run(tmp_path):
    root = build_tree(tmp_path)
    hide = tmp_path / "hidden"
    r = run_env_sim(root, hide, "true")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "restored the environment" in r.stdout
    assert not Path(str(hide) + ".env").exists()


def test_the_isolation_is_torn_down_even_when_the_suite_fails(tmp_path):
    """`finally`, not `if it worked`. A failing sweep is the normal case here.

    The command is a subprocess that returns non-zero, which is what a failing
    pytest run is. (A PODSIM_CMD that calls `exit` in the simulator's own shell
    still restores — the EXIT trap fires — but it does so while the command's
    output redirection is live, so the messages land in the log instead of on
    stdout. `test_restoration_survives_a_command_that_exits_the_shell` covers
    that shape by its observable effects rather than its chatter.)
    """
    root = build_tree(tmp_path)
    hide = tmp_path / "hidden"
    before = snapshot(root)
    r = run_env_sim(root, hide, 'bash -c "exit 7"')
    assert r.returncode == 7, (r.returncode, r.stdout + r.stderr)
    assert "restored the environment" in r.stdout
    assert "restored the hidden artifacts" in r.stdout
    assert not Path(str(hide) + ".env").exists()
    assert snapshot(root) == before


def test_restoration_survives_a_command_that_exits_the_shell(tmp_path):
    """`eval "exit 7"` terminates the simulator itself. The EXIT trap must still
    put the tree and the environment back, however little it gets to say."""
    root = build_tree(tmp_path)
    hide = tmp_path / "hidden"
    before = snapshot(root)
    r = run_env_sim(root, hide, "exit 7")
    assert r.returncode == 7, (r.returncode, r.stdout + r.stderr)
    assert snapshot(root) == before, "the hidden artifacts were not restored"
    assert not Path(str(hide) + ".env").exists()
    assert not Path(str(hide) + ".lock").exists()


def test_the_suite_exit_code_reaches_the_caller(tmp_path):
    """It used to end in `| tail -12`, so every sweep exited 0 no matter what.
    A readiness record built on that would assert a pass that never happened."""
    root = build_tree(tmp_path)
    assert run_env_sim(root, tmp_path / "hidden", "exit 3").returncode == 3
    assert run_env_sim(root, tmp_path / "hidden", "true").returncode == 0


def test_failing_nodeids_are_reported_before_the_tail(tmp_path):
    """Attempt 3R's fourteen failures arrived as three names."""
    root = build_tree(tmp_path)
    cmd = ('echo "FAILED tests/a.py::test_one"; echo "FAILED tests/b.py::test_two";'
           ' echo "ERROR tests/c.py::test_three"; echo "1 failed"; exit 1')
    r = run_env_sim(root, tmp_path / "hidden", cmd)
    assert r.returncode == 1
    for nodeid in ("tests/a.py::test_one", "tests/b.py::test_two",
                   "tests/c.py::test_three"):
        assert nodeid in r.stdout, r.stdout


def test_a_junit_report_is_requested_when_asked(tmp_path):
    root = build_tree(tmp_path)
    junit = tmp_path / "reports/j.xml"
    r = run_env_sim(root, tmp_path / "hidden", 'echo "CMD=$0"',
                    PODSIM_JUNIT=str(junit))
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"--junitxml={junit}" in r.stdout
    assert junit.parent.is_dir()


def test_the_default_command_matches_the_pod_gates_selection():
    """The simulator must not run a different suite from the pod. The interpreter
    differs by design — the pod has /opt/train — but the selection must not."""
    src = SCRIPT.read_text()
    assert "-m pytest tests/ -q" in src
    assert "uv run pytest" not in src, (
        "`uv run` reaches into $HOME/.cache/uv, which the isolation empties")


def test_the_environment_is_saved_before_it_is_changed():
    """`save_env` must precede the first export, or the trap restores nothing."""
    src = SCRIPT.read_text()
    # The first change is now the CPU-test contract eval rather than a literal
    # `export HOME=...`: the variables are declared once in
    # `aadistill.autoinit.cpu_test_env` and shared with the pod's gate, so the
    # diagnostic and the paid pod cannot drift into two different environments.
    assert src.index("save_env\n") < src.index('eval "$PODSIM_ENV_SH"')
    assert src.index("restore_env") < src.index('for p in "$HIDE"/*')


def test_a_nested_simulation_does_not_inherit_this_ones_control_variables(tmp_path):
    """The 2026-09-04 sweep failed five of its own tests on this.

    Every `PODSIM_*` variable is exported, and the tests above drive the real
    script as a subprocess. So a nested run inherited the OUTER sweep's
    `PODSIM_JUNIT`, appended a second `--junitxml=` to its own one-line command,
    and overwrote the outer report — turning correct tests red and corrupting the
    evidence the sweep existed to produce. They are inputs to one invocation.
    """
    root = build_tree(tmp_path)
    names = ("PODSIM_JUNIT PODSIM_LOG PODSIM_CMD PODSIM_ROOT PODSIM_ENV_ROOT "
             "PODSIM_HF_TOKEN HIDE_DIR PODSIM_LOCK HIDDEN_PATHS")
    # `bash -c '...' _` so the appended `--junitxml=` lands as an ignored
    # positional argument: the real PODSIM_CMD is a pytest invocation that takes
    # the flag, and this probe must tolerate it the same way.
    probe = f"""bash -c 'for v in {names}; do echo "$v=${{!v-UNSET}}"; done' _"""
    r = run_env_sim(root, tmp_path / "hidden", probe,
                    PODSIM_JUNIT=str(tmp_path / "outer.xml"))
    assert r.returncode == 0, r.stdout + r.stderr
    for v in names.split():
        assert f"{v}=UNSET" in r.stdout, (
            f"{v} leaked into the suite; a nested simulation would inherit it\n"
            + r.stdout)


def test_the_junit_flag_reaches_the_suite_but_not_its_children(tmp_path):
    """The flag must still be appended for THIS run — the fix must not disable
    the reporting it exists for."""
    root = build_tree(tmp_path)
    junit = tmp_path / "reports/j.xml"
    r = run_env_sim(root, tmp_path / "hidden", 'echo ran',
                    PODSIM_JUNIT=str(junit))
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"--junitxml={junit}" in r.stdout          # appended to the command
    assert "ran" in r.stdout


def test_the_default_log_path_is_per_invocation_not_a_shared_file(tmp_path):
    """A fixed default log is a shared mutable file across nested runs.

    Unsetting PODSIM_LOG so nested simulations cannot inherit it made every one
    of them fall back to the SAME default and truncate the outer sweep's log
    while the outer shell still held an open fd at its own offset. Sweep A's log
    came back with its `FAILED` lines punched out: `grep` matched nothing in a
    file whose own tail read `3 failed`. Deriving the default from `$HIDE` makes
    it per-invocation by construction.
    """
    src = SCRIPT.read_text()
    assert 'PODSIM_LOG=${PODSIM_LOG:-"${HIDE}.pytest.log"}' in src, (
        "the default log path is not derived from the per-invocation $HIDE")

    root = build_tree(tmp_path)
    hide = tmp_path / "hidden"
    r = run_env_sim(root, hide, "echo hello-from-the-suite")
    assert r.returncode == 0, r.stdout + r.stderr
    log = Path(str(hide) + ".pytest.log")
    assert log.is_file(), f"no log at the per-invocation default {log}"
    assert "hello-from-the-suite" in log.read_text()
    assert not Path("/home/ecs-user/aad-scratch/podsim_pytest.log").samefile(log) \
        if Path("/home/ecs-user/aad-scratch/podsim_pytest.log").exists() else True


# --- the interpreter is an input, never a guess -------------------------------
#
# C1 attempt 6 lost all 18 cases in this module at the pod CPU gate, for $0.3665.
# The simulator resolved its interpreter as "repo .venv, else `command -v
# python3`". A pod checkout has no repo .venv — it uses /opt/train — so the pod
# ALWAYS took a fallback the dev box never exercised.

#: The exact nodeids attempt 6 reported. Named, so the repair is checked against
#: what actually failed rather than against this module in general.
ATTEMPT_6_FAILED = (
    "test_a_clean_run_that_recreates_nothing_still_restores_exactly",
    "test_a_junit_report_is_requested_when_asked",
    "test_a_nested_simulation_does_not_inherit_this_ones_control_variables",
    "test_a_pod_equivalent_token_is_exported_and_never_printed",
    "test_failing_nodeids_are_reported_before_the_tail",
    "test_hf_hub_cache_sits_under_the_isolated_hf_home",
    "test_restoration_survives_a_command_that_exits_the_shell",
    "test_restore_reproduces_the_exact_pre_state_when_the_run_recreates_it",
    "test_the_default_log_path_is_per_invocation_not_a_shared_file",
    "test_the_dev_box_hf_cache_is_not_visible",
    "test_the_isolation_is_torn_down_after_a_normal_run",
    "test_the_isolation_is_torn_down_even_when_the_suite_fails",
    "test_the_junit_flag_reaches_the_suite_but_not_its_children",
    "test_the_lock_is_released_after_a_normal_run",
    "test_the_recreated_content_is_quarantined_not_deleted",
    "test_the_simulated_home_is_a_fresh_empty_directory",
    "test_the_suite_exit_code_reaches_the_caller",
    "test_variables_that_would_defeat_the_isolation_are_unset",
)


def test_the_ambient_python_fallback_is_gone_from_the_code():
    """Comments may NAME the defect; executable lines may not contain it."""
    code = "\n".join(ln for ln in SCRIPT.read_text().splitlines()
                     if not ln.lstrip().startswith("#"))
    for ambient in ("command -v python3", "/usr/bin/python3", "which python3"):
        assert ambient not in code, (
            f"{ambient!r} is back in executable code. An ambient interpreter is "
            "whatever the machine happens to have; on a pod that is not the one "
            "running the suite.")
    assert 'PODSIM_PYTHON' in code and "REFUSING: no interpreter" in SCRIPT.read_text()


def test_every_nodeid_attempt_6_lost_is_in_this_module():
    """The repair is checked against the recorded failures, not a proxy."""
    src = SCRIPT.parent.parent.parent / "tests/pod/test_simulator_restore.py"
    text = src.read_text()
    missing = [n for n in ATTEMPT_6_FAILED if f"def {n}(" not in text]
    assert not missing, f"attempt 6 named nodeids this module no longer has: {missing}"
    assert len(ATTEMPT_6_FAILED) == 18


def test_an_explicit_interpreter_works_where_there_is_no_repo_venv(tmp_path):
    """A synthetic PODSIM_ROOT with no `.venv` — which is a pod's condition."""
    root = build_tree(tmp_path)
    assert not (root / ".venv").exists()
    r = run_env_sim(root, tmp_path / "hidden", 'echo "HOME=$HOME"')
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"HOME={tmp_path}" in r.stdout


def test_no_interpreter_and_no_repo_venv_refuses_loudly(tmp_path):
    """Fail closed. A silent fallback is how the pod ran different code."""
    fake = tmp_path / "checkout/scripts/pod"
    fake.mkdir(parents=True)
    (fake / SCRIPT.name).write_text(SCRIPT.read_text())
    (fake / "cpu_test_env_args.py").write_text(
        (SCRIPT.parent / "cpu_test_env_args.py").read_text())
    assert not (tmp_path / "checkout/.venv").exists()

    root = build_tree(tmp_path)
    env = {k: v for k, v in os.environ.items() if k != "PODSIM_PYTHON"}
    env.update({"PODSIM_ROOT": str(root), "HIDE_DIR": str(tmp_path / "h"),
                "PODSIM_LOCK": str(tmp_path / "h.lock"),
                "HIDDEN_PATHS": "artifacts/audit", "PODSIM_CMD": "true"})
    r = subprocess.run(["bash", str(fake / SCRIPT.name)], capture_output=True,
                       text=True, env=env)
    assert r.returncode == 5, (r.returncode, r.stdout, r.stderr)
    assert "REFUSING: no interpreter" in r.stderr


def test_a_non_executable_interpreter_refuses_rather_than_falling_back(tmp_path):
    root = build_tree(tmp_path)
    r = run_env_sim(root, tmp_path / "hidden", 'echo hi',
                    PODSIM_PYTHON=str(tmp_path / "not-a-python"))
    assert r.returncode == 5, (r.returncode, r.stdout, r.stderr)
    assert "is not executable" in r.stderr


def test_a_nested_simulation_does_not_inherit_the_interpreter_control(tmp_path):
    """`PODSIM_PYTHON` is invocation-local, like every other PODSIM_* control.

    An outer sweep that leaked it would hand a nested run an interpreter chosen
    for a different machine — the same shape as the PODSIM_JUNIT leak that failed
    five correct tests in the 2026-09-04 sweep.
    """
    root = build_tree(tmp_path)
    r = run_env_sim(root, tmp_path / "hidden",
                    'echo "INNER=${PODSIM_PYTHON:-<unset>}"')
    assert r.returncode == 0, r.stdout + r.stderr
    assert "INNER=<unset>" in r.stdout, r.stdout
    src = SCRIPT.read_text()
    assert "PODSIM_LOCK HIDDEN_PATHS PODSIM_PYTHON" in src


# --- free space is checked BEFORE anything is moved --------------------------
#
# On 2026-09-11 the filesystem filled while a sweep was running. The EXIT trap
# never completed, 869 gitignored artifacts (5.9 GiB) were left in the hide
# directory, and `git status` read clean the whole time -- gitignored files are
# precisely what it does not see. Two tracked files were truncated to zero bytes
# in the same window.
#
# A sweep that runs out of disk is not a failed sweep. It is a displaced
# repository that reports nothing.

def _run_simulator(tmp_path, *, min_free_gib, cmd="true"):
    import subprocess

    env = {**os.environ,
           "HIDE_DIR": str(tmp_path / "hide"),
           "PODSIM_LOCK": str(tmp_path / "lock"),
           "PODSIM_MIN_FREE_GIB": str(min_free_gib),
           "PODSIM_CMD": cmd}
    return subprocess.run(["bash", str(REPO / "scripts/pod/simulate_pod_env.sh")],
                          capture_output=True, text=True, cwd=REPO, env=env,
                          timeout=300)


def test_the_sweep_refuses_when_free_space_is_below_the_requirement(tmp_path):
    """And refuses BEFORE hiding, which is the whole point.

    Refusing after the first `mv` would leave the caller to restore by hand,
    which is exactly the state this rule was learned from.
    """
    out = _run_simulator(tmp_path, min_free_gib=10**9)
    assert out.returncode == 5, out.stdout + out.stderr
    assert "REFUSING" in out.stderr
    assert "free" in out.stderr.lower()
    assert not (tmp_path / "hide").exists(), (
        "the hide directory was created before the space check")
    assert not (tmp_path / "lock").exists(), (
        "the lock survived a refusal, so the next sweep would refuse too")


def test_the_sweep_proceeds_when_the_requirement_is_met(tmp_path):
    """Both directions: a check that always refuses protects nothing."""
    out = _run_simulator(tmp_path, min_free_gib=1)
    assert "free space ok" in out.stdout, out.stdout + out.stderr
    assert "REFUSING" not in out.stderr


def test_the_threshold_is_an_input_not_a_constant_in_the_core():
    """The number lives with the caller that knows the footprint.

    `src/aadistill` must not learn this machine's capacity, and the simulator
    must not hard-code a figure derived from one workload.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "rpe", REPO / "scripts/autoinit/record_pod_environment.py")
    rpe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rpe)
    assert rpe.min_free_gib() >= 20, "the derived requirement lost its headroom"

    sim = (REPO / "scripts/pod/simulate_pod_env.sh").read_text()
    assert "PODSIM_MIN_FREE_GIB" in sim, "the simulator no longer takes the input"
    core = (REPO / "src/aadistill").rglob("*.py")
    offenders = [p.name for p in core if "MIN_FREE_GIB" in p.read_text()]
    assert not offenders, f"a disk threshold reached the core: {offenders}"
