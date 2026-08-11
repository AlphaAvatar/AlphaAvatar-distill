"""The two setup gates that killed E8b-S2's first three host draws.

Both reported `HOST_COLD`, and neither host was cold:

* **draw 1, 26.4 min.** The uv tripwire killed on a *single* 20 s window with under
  20 MB of disk growth. uv writes nothing while it resolves or builds a wheel, so a
  working host is indistinguishable from a hung one if one sample decides. `UV_MAX_S`
  did not protect it — that variable never extended the deadline, it only stops the
  growth check from applying at all, which is why `--uv-max-s 3600` changed nothing.
* **draws 2 and 3, ~27 min each.** The suite exceeded its 900 s box on 128-vCPU A100
  hosts, against 88 s on a 16-core dev box. The cause was not the suite being slow: a
  container reports the HOST's cpu count while the cgroup grants a fraction of it, so
  the subprocess `test_depth_search_driver` spawns sized its thread pools from
  `nproc` 128 and burned 900+ s at 1338% CPU on work that takes 7.4 s, while the
  parent pytest waited in `sigsuspend` at 2% CPU. Env caps did not bound it because
  the child re-derives its own pool.

These assert against the real `e8b_setup.sh`, not a copy, because the file that runs
on the pod is the thing that was wrong. `test_cold_host_tripwire.sh` covers a lifted
structural copy and cannot catch a regression in the deployed script.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SETUP = REPO / "scripts/pod/e8b_setup.sh"
LAUNCH = REPO / "scripts/pod/e8b_launch.py"


@pytest.fixture(scope="module")
def setup_text() -> str:
    return SETUP.read_text()


# --- fault 1: the uv tripwire ---------------------------------------------

def test_the_tripwire_needs_consecutive_stalls_not_one(setup_text):
    assert "STALL_LIMIT=${UV_STALL_LIMIT:-3}" in setup_text
    assert 'stalls=$(( stalls + 1 ))' in setup_text
    assert '[ "$stalls" -lt "$STALL_LIMIT" ]' in setup_text, (
        "a kill must require STALL_LIMIT consecutive quiet windows")


def test_progress_resets_the_stall_counter(setup_text):
    # Otherwise three scattered quiet windows across a 40-minute install would
    # accumulate to a kill on a host that was making progress throughout.
    grace = re.search(r"graced=\$\(\( graced \+ 1 \)\)[^\n]*", setup_text)
    assert grace and "stalls=0" in grace.group(0), (
        "the progress branch must reset stalls to 0")


def test_the_stall_count_is_recorded_in_the_marker(setup_text):
    # The next reader has to be able to tell a hang from a slow mirror.
    assert "stalls${stalls}" in setup_text


def test_uv_max_s_is_documented_as_a_ceiling_not_an_extension(setup_text):
    # The misunderstanding that cost draw 1 was believing --uv-max-s moved the
    # deadline. If someone deletes this note the trap is reset. Normalize first:
    # the comment is line-wrapped, so the phrase spans a "\n# " continuation.
    flat = re.sub(r"\s+", " ", setup_text.replace("\n#", " "))
    assert "never extended the deadline" in flat


# --- fault 2: the test gate ------------------------------------------------

def test_the_suite_is_pinned_to_a_kernel_enforced_cpu_set(setup_text):
    assert 'taskset -c "$CPUS"' in setup_text, (
        "env thread caps are advisory and a child re-derives its own pool; the cpu "
        "set must be enforced by the kernel, where children inherit it")
    assert "CPUS=${TESTS_CPUS:-0-$(( NCPU - 1 ))}" in setup_text


def test_the_budget_comes_from_the_cgroup_not_from_nproc(setup_text):
    # `nproc` is the host's count inside a container and is the number that caused
    # the blowup. The quota is the real limit.
    assert "cpu_budget()" in setup_text
    assert "/sys/fs/cgroup/cpu.max" in setup_text, "cgroup v2 quota not read"
    assert "cpu.cfs_quota_us" in setup_text, "cgroup v1 quota not read"
    assert "NCPU=$(cpu_budget)" in setup_text


def test_the_fallback_does_not_use_bare_nproc(setup_text):
    """`nproc` honours OMP_NUM_THREADS, so it reports our cap, not the machine.

    This is documented coreutils behaviour and it is genuinely surprising: with
    OMP_NUM_THREADS=8 exported, `nproc` printed 8 on a 13-cpu affinity set. Using it
    as the no-quota fallback would feed our own thread cap back in as the cpu budget.
    """
    fn = _cpu_budget_fn(setup_text)
    assert "sched_getaffinity" in fn, (
        "the no-quota fallback must read the affinity mask")
    bare = re.findall(r"\$\(nproc\)", fn)
    assert not bare, f"bare $(nproc) in cpu_budget: {bare} — honours OMP_NUM_THREADS"


def test_the_env_thread_caps_track_the_derived_budget(setup_text):
    # taskset bounds total parallelism; the env caps still stop each library from
    # spawning a pool sized to `nproc` inside that set — but they must follow the
    # budget, not a constant that could exceed it on a 4-cpu container.
    assert "NTHREADS=$(( NCPU < 8 ? NCPU : 8 ))" in setup_text
    for var in ("OMP_NUM_THREADS=$NTHREADS", "MKL_NUM_THREADS=$NTHREADS",
                "OPENBLAS_NUM_THREADS=$NTHREADS"):
        assert var in setup_text


def test_the_visible_and_granted_counts_are_both_logged(setup_text):
    # The discrepancy between them IS the bug; a log that shows only one hides it.
    line = next(l for l in setup_text.splitlines() if "CPU test suite (" in l)
    assert "$(nproc) vCPUs visible" in line and "cgroup budget ${NCPU}" in line


def test_the_box_is_large_enough_for_the_current_suite(setup_text):
    boxes = {int(m) for m in re.findall(r"TESTS_MAX_S:-(\d+)", setup_text)}
    assert boxes, "no TESTS_MAX_S fallback found"
    assert min(boxes) >= 2700, f"box {min(boxes)}s is the value draw 2 exceeded"
    assert len(boxes) == 1, f"inconsistent fallbacks {boxes} — one will be wrong"


def test_the_launcher_default_agrees_with_the_script(setup_text):
    m = re.search(r'"--tests-max-s",\s*type=int,\s*default=(\d+)', LAUNCH.read_text())
    assert m, "launcher no longer exposes --tests-max-s"
    launcher = int(m.group(1))
    script = min(int(x) for x in re.findall(r"TESTS_MAX_S:-(\d+)", setup_text))
    # The launcher passes TESTS_MAX_S explicitly, so its default wins over the
    # script's fallback. A stale launcher default silently reinstates the old box.
    assert launcher >= script, (
        f"launcher default {launcher}s would override the script's {script}s")


def test_the_elapsed_suite_time_is_recorded_on_success(setup_text):
    # So the box is calibrated from measurements instead of guessed a third time.
    assert 'mark "TESTS_OK:${tt}s"' in setup_text


def test_a_timeout_still_marks_host_cold_and_exits_90(setup_text):
    # The gate must keep failing closed: a suite that cannot finish is a bad host
    # for this workload even after the box is raised.
    block = setup_text[setup_text.index("CPU test suite"):]
    assert '"$RC" -eq 124' in block
    assert "HOST_COLD:tests:" in block
    assert "exit 90" in block


# --- behavioural: the new stall logic, driven against synthetic hosts ------

TRIPWIRE = r"""
tripwire() {
  local TRIP_S=$1 GRACE_S=$2 STALL_LIMIT=$3 DIR=$4; shift 4
  say() { :; }
  "$@" & local UV_PID=$!
  local t0 el graced a b stalls
  t0=$(date -u +%s); graced=0; stalls=0
  while kill -0 "$UV_PID" 2>/dev/null; do
    sleep 1
    el=$(( $(date -u +%s) - t0 ))
    [ "$el" -lt "$TRIP_S" ] && continue
    a=$(du -sb "$DIR" 2>/dev/null | cut -f1 || echo 0); sleep 1
    b=$(du -sb "$DIR" 2>/dev/null | cut -f1 || echo 0)
    if [ "$((b - a))" -gt 100000 ]; then
      graced=$(( graced + 1 )); stalls=0
      TRIP_S=$(( TRIP_S + GRACE_S )); continue
    fi
    stalls=$(( stalls + 1 ))
    if [ "$stalls" -lt "$STALL_LIMIT" ]; then continue; fi
    kill -9 "$UV_PID" 2>/dev/null || true
    echo "COLD:stalls${stalls}"
    return 90
  done
  wait "$UV_PID" || return 1
  echo "OK"
  return 0
}
"""


def run_tripwire(script: str, tmp_path: Path) -> tuple[int, str]:
    body = f'set -uo pipefail\nDIR="{tmp_path}"\n{TRIPWIRE}\n{script}\n'
    p = subprocess.run(["bash", "-c", body], capture_output=True, text=True,
                       timeout=120)
    return p.returncode, p.stdout.strip()


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_a_quiet_host_that_finishes_is_not_killed(tmp_path):
    # Two quiet windows then completion: the old logic killed on the first.
    rc, out = run_tripwire('tripwire 1 1 3 "$DIR" sleep 4', tmp_path)
    assert (rc, out) == (0, "OK"), out


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_a_genuinely_hung_host_still_dies(tmp_path):
    rc, out = run_tripwire('tripwire 1 1 3 "$DIR" sleep 300', tmp_path)
    assert rc == 90
    assert "COLD:stalls3" in out, out


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_a_progressing_host_survives_past_the_trip_point(tmp_path):
    grower = tmp_path / "grow.sh"
    grower.write_text(
        f'for i in 1 2 3 4 5 6; do dd if=/dev/zero of="{tmp_path}/f$i" bs=1M '
        'count=2 status=none; sleep 1; done\n')
    rc, out = run_tripwire(f'tripwire 1 1 3 "$DIR" bash {grower}', tmp_path)
    assert (rc, out) == (0, "OK"), out


# --- behavioural: cpu_budget against synthetic cgroups ---------------------

def _cpu_budget_fn(setup_text: str) -> str:
    start = setup_text.index("cpu_budget() {")
    return setup_text[start:setup_text.index("NCPU=$(cpu_budget)")]


def run_budget(setup_text: str, tmp_path: Path, quota: str | None) -> str:
    body = _cpu_budget_fn(setup_text)
    if quota is None:
        body = (body.replace("/sys/fs/cgroup/cpu.max", str(tmp_path / "absent"))
                    .replace("/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
                             str(tmp_path / "absent1")))
    else:
        (tmp_path / "cpu.max").write_text(quota + "\n")
        body = body.replace("/sys/fs/cgroup/cpu.max", str(tmp_path / "cpu.max"))
    p = subprocess.run(["bash", "-c", body + "\ncpu_budget\n"],
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    return p.stdout.strip()


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.parametrize("quota,want", [
    ("400000 100000", "4"),      # a 4-cpu container
    ("800000 100000", "8"),      # an 8-cpu container
    ("3200000 100000", "16"),    # 32 granted, capped: the suite needs no more
])
def test_the_quota_decides_the_budget(setup_text, tmp_path, quota, want):
    assert run_budget(setup_text, tmp_path, quota) == want


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_an_unlimited_quota_falls_back_to_the_affinity_mask(setup_text, tmp_path):
    import os
    want = str(min(16, len(os.sched_getaffinity(0))))
    assert run_budget(setup_text, tmp_path, "max 100000") == want


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_a_bare_host_with_no_cgroup_files_still_works(setup_text, tmp_path):
    import os
    want = str(min(16, len(os.sched_getaffinity(0))))
    assert run_budget(setup_text, tmp_path, None) == want, (
        "no quota means the affinity mask decides; it must not return empty or 0")


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_the_fallback_ignores_an_ambient_omp_num_threads(setup_text, tmp_path, monkeypatch):
    """The regression that broke a paid pod: this test itself runs with the cap set.

    The setup script exports OMP_NUM_THREADS for the suite, so any test asserting
    on the budget runs in an environment where `nproc` lies. Pin that the budget is
    unmoved by it.
    """
    import os
    want = str(min(16, len(os.sched_getaffinity(0))))
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    assert run_budget(setup_text, tmp_path, None) == want
