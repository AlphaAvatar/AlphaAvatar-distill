"""Pod scripts must only reference relay and artifact paths that actually exist.

This exists because of a real failure. `e4_setup.sh` was generated from
`e3_setup.sh` with `sed 's/e3_/e4_/g'`, which also rewrote the *middle* of
`stag`**`e3_`**`recovery_corpus_v2` into `stage4_recovery_corpus_v2`. The pod
downloaded zero files, failed on the first `iterdir`, and deleted itself — $0.05
and a clean fail, but only because the setup script happened to touch the
directory immediately.

The lesson is not "sed carefully". It is that a pod script's external paths are
a contract with the relay, and a contract belongs in a test. A typo in any of
these is otherwise invisible until a GPU is already running.
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
POD = REPO / "scripts/pod"

# Prefixes known to exist in the private relay repo, established by the sessions
# that successfully downloaded from them. Add a row only when a real upload
# created it.
RELAY_PREFIXES = {
    "stage1/qwen3_0p6b_init_v0/checkpoint",
    "stage3_recovery_corpus_v2/ladder_uniform",
    "stage3_recovery_corpus_v2/sessions.jsonl",
    "e1_scaling_20260801",
    "e5_start",
    "transfer",
    "e7_streams_20260809",
    # Created by scripts/data/stage_e8_inputs.py, roundtrip-verified there.
    "e8_inputs_20260810",
    # Created by the documented dev-box upload of the treatment initialization,
    # which `e8b_launch.py` refuses to create a pod without.
    "e8_init_20260810",
}
# Local trees inside the checked-out repo on the pod.
LOCAL_PREFIXES = {
    # The whole Stage 1 tree, not one checkpoint: E8 adds a calibration set, a
    # depth search and a second initialization under it, and enumerating each new
    # directory would make this a changelog rather than a guard against `stage4/`.
    "artifacts/stage1",
    "artifacts/eval",
    "data/eval_behavior_v0",
    "artifacts/stage1/qwen3_0p6b_init_v0",
    "artifacts/stage3",
    "artifacts/audit",
    "configs/stage3",
    "data/warmup",
    "scripts",
    "tests",
    "src",
}
# `e[0-9]*` rather than `e[0-9]`: E8 splits into `e8a_*` and `e8b_*`, and the
# narrower glob silently excluded them — a contract test that does not cover the
# newest scripts is the failure mode it exists to prevent.
SCRIPTS = (sorted(POD.glob("e[0-9]*_setup.sh")) + sorted(POD.glob("e[0-9]*_driver.py"))
           + sorted(POD.glob("e[0-9]*_pilot.py")))

# `stageN` names that are real. Anything else is almost certainly sed damage:
# the project has stages 0-6 but only these directory families exist.
VALID_STAGE_TOKENS = {"stage0", "stage1", "stage2", "stage3", "stage2_v1",
                      "stage3_recovery_corpus_v2", "stage3_pilot"}


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_no_invented_stage_directories(script):
    """`stage4_…`/`stage5_…` do not exist; finding one means a rename went wide."""
    text = script.read_text()
    # Only PATH-like uses: a stage token followed by a slash. Without the
    # lookahead this fires on identifiers such as `stage1_init_sha256`, which is
    # a JSON key and not a directory.
    for token in set(re.findall(r"stage\d+[a-z0-9_]*(?=/)", text)):
        assert token in VALID_STAGE_TOKENS, (
            f"{script.name} references {token!r}, which is not a real stage "
            "directory — most likely collateral from a global rename")


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_relay_paths_are_known(script):
    """Every hf_hub_download / snapshot_download path resolves to a known prefix."""
    text = script.read_text()
    quoted = re.findall(r"""['"]([A-Za-z0-9_][A-Za-z0-9_/*.:-]{6,})['"]""", text)
    for value in quoted:
        # Only look at things that look like relay object paths.
        if not re.match(r"^(stage[0-9]|e1_scaling|e5_start|transfer)", value):
            continue
        if "/" not in value:
            continue                      # an identifier, not a relay object path
        stem = value.rstrip("*").rstrip("/")
        assert any(stem == p or stem.startswith(p + "/") or p.startswith(stem)
                   for p in RELAY_PREFIXES), \
            f"{script.name}: relay path {value!r} is not a known prefix"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_local_repo_paths_are_known(script):
    text = script.read_text()
    for value in set(re.findall(r"(?:/workspace/aad/)?((?:artifacts|configs|data|scripts|src|tests)/[A-Za-z0-9_/.*-]+)", text)):
        assert any(value == p or value.startswith(p + "/") or p.startswith(value)
                   for p in LOCAL_PREFIXES), \
            f"{script.name}: local path {value!r} is outside the known tree"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_pinned_hashes_are_full_length_sha256(script):
    """A truncated hash silently weakens a verification gate to a prefix check."""
    text = script.read_text()
    for name, value in re.findall(r"(want|sha|INIT_SHA|HOLDOUT_SHA)\s*=\s*'([0-9a-f]{8,})'", text):
        assert len(value) == 64, f"{script.name}: {name} is {len(value)} chars, not 64"


@pytest.mark.parametrize("prefix", sorted(
    {f.name.split("_", 1)[0] for f in POD.glob("e[0-9]*_*")}))
def test_setup_driver_and_launcher_agree_on_the_status_file(prefix):
    """All three components must name the SAME marker file.

    This is the third instance of one blind spot. `sed s/e3_/e4_/g` does not
    match `e3.status` (a dot, not an underscore), so the E4 launcher polled a
    file the E4 driver never wrote. The run would have finished at 11:00 and the
    launcher would have idled to its 400-minute timeout — roughly $2.65 of
    billing for nothing, against a $4.00 authorization. Caught live and patched
    with a symlink; this test is the durable fix.
    """
    paths = {}
    for name, pattern in (("setup", f"{prefix}_setup.sh"),
                          ("driver", f"{prefix}_driver.py"),
                          ("launcher", f"{prefix}_launch.sh"),
                          ("launcher_py", f"{prefix}_launch.py")):
        f = POD / pattern
        if not f.is_file():
            continue
        text = f.read_text()
        found = set(re.findall(r"/?workspace/(e\d+[a-z]?\.status)", text))
        found |= set(re.findall(r"STATUS=\$WS/(e\d+[a-z]?\.status)", text))
        found |= set(re.findall(r'STATUS = f?"\{WS\}/(e\d+[a-z]?\.status)"', text))
        if found:
            paths[name] = found
    if len(paths) < 2:
        pytest.skip(f"{prefix}: fewer than two components reference a status file")
    everything = set().union(*paths.values())
    assert len(everything) == 1, (
        f"{prefix}: components disagree on the status file — {paths}. "
        "The launcher polls it for ALL_DONE; a mismatch means teardown never "
        "fires on completion and the pod idle-bills to its timeout.")


def test_the_stage1_fork_point_hash_is_identical_everywhere():
    """Every script that *verifies* the init must verify the SAME init.

    A prefix followed by an ellipsis (`sha256 86fbba78…`) is prose meant for a
    human and is allowed; a bare prefix in code is a weakened gate and is not.
    """
    want = "86fbba78e8a2a32481ca77e5ac362ed1f17a39dbc30bcbc952cabd5df2633e54"
    verifying = 0
    for script in POD.glob("*"):
        if not script.is_file():
            continue
        text = script.read_text(errors="ignore")
        for match in re.finditer(r"\b86fbba78[0-9a-f]*", text):
            tail = text[match.end():match.end() + 3]
            if tail.startswith("…") or tail.startswith("..."):
                continue                      # documented display prefix
            assert match.group() == want, (
                f"{script.name}: truncated/altered init hash {match.group()}")
            verifying += 1
    assert verifying >= 1, "no pod script verifies the Stage 1 fork point"


@pytest.mark.parametrize("script", sorted(POD.glob("e[0-9]_launch.sh")),
                         ids=lambda p: p.name)
def test_launcher_scratch_paths_match_its_own_experiment(script):
    """`e3.state` in the E4 launcher survived a rename because of the dot.

    Same blind spot as `stage3_recovery_corpus_v2` and `e3.status`: a global
    `s/e3_/e4_/` matches neither. The state file is what the teardown watchdog
    reads, so a stale one makes the watchdog report an empty state forever.
    """
    prefix = script.name[:2]
    text = script.read_text()
    for token in set(re.findall(r"\$SCR/(e\d+)[._]", text)):
        assert token == prefix, (
            f"{script.name} references scratch path {token!r} but belongs to "
            f"{prefix!r} — collateral from a global rename")
    for token in set(re.findall(r"configs/stage3/(e\d+)\b", text)):
        assert token == prefix, (
            f"{script.name} bundles {token!r} configs but belongs to {prefix!r} "
            "— the returned side bundle would carry the wrong experiment")


# --- Experiment 5 gate-2 amendment -------------------------------------------
# Gate 2 must not price R's training at C's measured rate. These assertions pin
# the amendment's three moving parts so a later refactor cannot quietly restore
# the cheaper-but-wrong projection.

E5_DRIVER = REPO / "scripts/pod/e5_driver.py"
E5_BENCH = REPO / "scripts/training/benchmark_e5_throughput.py"


def _e5_driver() -> str:
    return E5_DRIVER.read_text()


def test_final_benchmark_runs_between_pairing_and_gate_2():
    src = _e5_driver()
    order = src[src.index("STAGES = {"):src.index("BLOCKING = (")]
    for earlier, later in (("pair", "final_benchmark"),
                           ("final_benchmark", "budget_gate_2"),
                           ("budget_gate_2", "train")):
        assert order.index(f'"{earlier}"') < order.index(f'"{later}"'), \
            f"{earlier} must be ordered before {later}"
    blocking = src[src.index("BLOCKING = ("):src.index("def main()")]
    assert '"final_benchmark"' in blocking, "the gate-2 benchmark must be blocking"


def test_gate_2_projects_from_the_slower_final_pack_measurement():
    src = _e5_driver()
    gate = src[src.index("def stage_budget_gate_2"):src.index("# Order matters")]
    assert "e5_throughput_final.json" in gate, \
        "gate 2 must read the final-pack benchmark, not the gate-1 one"
    assert "sec_per_step_for_projection" in gate, \
        "gate 2 must project from a measured absolute sec/step"
    assert "measured_wall_clock_speedup" not in gate, \
        "gate 2 must not reuse C's speedup as R's rate"
    assert "_spent(args)" in gate, \
        "gate 2 must charge elapsed pod time, which covers its own benchmark"


def test_final_benchmark_measures_both_arms_on_the_registered_path():
    src = _e5_driver()
    stage = src[src.index("def stage_final_benchmark"):
                src.index("def stage_budget_gate_2")]
    assert "--absolute-only" in stage, "the full-width reference is not repeated"
    assert "e5_pack_{a}_{seed}" in stage, "must measure the FINAL packs"
    assert '("c", "r")' in stage, "both arms must be measured"


def test_absolute_only_mode_reports_the_slowest_arm():
    src = E5_BENCH.read_text()
    assert '"--absolute-only"' in src
    block = src[src.index("if args.absolute_only:"):src.index("ids, ce, content, sel_real, real = load(args.pack)")]
    assert 'max(results, key=lambda k: results[k]["sec_per_step"])' in block, \
        "the projection rate must be the SLOWER arm"
    assert '"benchmark_cost_usd"' in block, "the benchmark must price itself"


def test_a_stopped_gate_is_not_recorded_as_a_completed_run():
    src = _e5_driver()
    main = src[src.index("def main()"):]
    assert main.index('mark("ABORTED_AT_GATE")') < main.index('mark("ALL_DONE")')
    aborted = main[main.index('mark("ABORTED_AT_GATE")"'[:-1]):]
    assert aborted.split("\n")[1].strip() == "return", \
        "aborting must return, or the last status line becomes ALL_DONE"


def test_launcher_hands_the_driver_a_real_starting_balance():
    """A zero starting balance would hide startup + the ~53-min setup from both
    gates, which is roughly $1 of the authorization."""
    src = (REPO / "scripts/pod/e5_launch.sh").read_text()
    invoke = src[src.index("scripts/pod/e5_driver.py --stage all"):]
    assert "--spent-usd" in invoke.split("disown")[0]
    assert "--authorized-usd" in invoke.split("disown")[0]
    assert "pod_start_epoch" in src[:src.index("scripts/pod/e5_driver.py --stage all")]
    assert "GATE_CEILING=$(echo \"$BACKSTOP_MINUTES/60*$MAX_PRICE\"" in src, \
        "the gate ceiling must be bound to the RunPod deadline it has to fit inside"


# --- 2026-08-07: uncaught exception left a finished pod billing ---------------

def test_driver_marks_every_stage_failure_not_just_three_types():
    """A narrow except tuple let a ValueError escape main(), so no marker was
    written and the launcher had no terminal state to poll for."""
    src = _e5_driver()
    main = src[src.index("def main()"):]
    assert "except Exception as exc:" in main, \
        "an uncaught exception on a paid pod is a billing event"
    assert "except (subprocess.CalledProcessError" not in main


def test_launcher_stops_when_the_driver_process_dies():
    src = (REPO / "scripts/pod/e5_launch.sh").read_text()
    poll = src[src.index("DEADLINE_TS=$(( $(date -u +%s) + POLL_LIMIT_MIN"):]
    assert "[e]5_driver.py" in poll, \
        "liveness must be checked without pgrep matching its own command line"
    assert "DRIVER PROCESS GONE" in poll


def test_pairing_merges_both_arms_system_blocks():
    src = _e5_driver()
    assert "def _system_ids(" in src
    fn = src[src.index("def _system_ids("):src.index("def stage_pair(")]
    assert 'for arm in ("c", "r")' in fn, "R's system blocks must be included"
    assert "clashes" in fn, "disagreement between arms must be reported"
    pair = src[src.index("def stage_pair("):src.index("def stage_train(")]
    assert 'e5_arm_c_{seed}"\n                             / "system_ids.json"' not in pair


# --- E5 attempt 2: $7.55, arm C reused, records verified from disk ------------

AUTHORIZED_USD = 4.84     # what survived attempts 1-7


def test_the_pod_deadline_cannot_exceed_the_remaining_authorization():
    """The RunPod-side deadline is the one layer that fires even if launcher,
    driver and poller are all dead, so it must itself sit under the
    authorization -- including any pod abandoned earlier in the session."""
    src = (REPO / "scripts/pod/e5_launch.sh").read_text()
    backstop = int(re.search(r"BACKSTOP_MINUTES=\$\{BACKSTOP_MINUTES:-(\d+)\}",
                             src).group(1))
    rate = float(re.search(r"MAX_PRICE=\$\{MAX_PRICE:-([\d.]+)\}", src).group(1))
    assert backstop / 60 * rate <= AUTHORIZED_USD, \
        f"{backstop} min at ${rate}/h exceeds the ${AUTHORIZED_USD} authorization"
    poll = int(re.search(r"POLL_LIMIT_MIN=\$\{POLL_LIMIT_MIN:-(\d+)\}", src).group(1))
    assert poll < backstop, "polling must end before the pod is killed under it"


def test_arm_c_is_staged_and_verified_never_rebuilt():
    setup = (REPO / "scripts/pod/e5_setup.sh").read_text()
    assert "e5_start/e5_arm_c.tar.gz" in setup, "arm C must be staged from the relay"
    assert "ARM C BUNDLE MISMATCH" in setup and "ARM C MISMATCH" in setup, \
        "both the bundle and the per-file hashes must be asserted"
    assert "fails the current contract" in setup, \
        "a reused corpus must satisfy the CURRENT packing contract, not just a hash"
    driver = _e5_driver()
    gen = driver[driver.index("def stage_generate("):driver.index("def stage_verify_records(")]
    assert "build_e5_arm_c.py" not in gen, "the driver must not rebuild arm C"
    assert "staged in setup, not built here" in gen


def test_persisted_records_are_verified_from_disk_before_pairing():
    src = _e5_driver()
    order = src[src.index("STAGES = {"):src.index("BLOCKING = (")]
    for earlier, later in (("generate", "verify_records"), ("verify_records", "pair")):
        assert order.index(f'"{earlier}"') < order.index(f'"{later}"')
    assert '"verify_records"' in src[src.index("BLOCKING = ("):src.index("def main()")]
    stage = src[src.index("def stage_verify_records("):src.index("def _system_ids(")]
    assert 'examples.jsonl").open()' in stage, "must re-read the file, not reuse memory"
    assert "example_to_rendered(rec)" in stage
    assert 'for arm in ("c", "r")' in stage, "both producers must be verified"


def test_a_replaced_pod_does_not_reset_the_session_billing_origin():
    """2026-08-07: a pod that never exposed TCP 22 was deleted and replaced. The
    replacement reset `pod_start_epoch`, hiding the abandoned pod's $0.25 from
    every gate and handing the replacement a fresh full deadline."""
    src = (REPO / "scripts/pod/e5_launch.sh").read_text()
    assert '[ -f "$SCR/pod_start_epoch" ] || date -u +%s > "$SCR/pod_start_epoch"' in src, \
        "the billing origin must be written once per session, not once per pod"
    assert 'date -u +%s > "$SCR/pod_start_epoch"; echo' not in src


def test_gate_1_prices_r_generation_from_the_measurement():
    """152 min was a pre-measurement estimate; attempt 1 measured 74.1 min. The
    phantom was worth $1.29 and failed a run short by $0.55."""
    src = _e5_driver()
    gate = src[src.index("def stage_budget_gate_1"):src.index("def stage_final_benchmark")]
    assert '"r_generation": 152' not in gate, "the stale estimate must be gone"
    # Two prices: 0 when the corpora are staged, else a margin over the measured
    # 74.1 min. Both must be present -- the gate chooses between them at runtime.
    measured = int(re.search(r'"r_generation": 0 if staged else (\d+)', gate).group(1))
    assert 74 <= measured <= 110, \
        f"r_generation {measured} is not a margin over the measured 74.1 min"
    assert "MEASURED 74.1 min" in gate, "the basis must travel with the number"


# --- cold-host tripwire and redraw (2026-08-07) -------------------------------

def test_the_tripwire_behavioural_suite_passes():
    """Runs the shell tripwire against synthetic warm/cold/linking hosts.

    Shell, not a python reimplementation: the thing that runs on a pod is shell,
    and a reimplementation would prove nothing about it."""
    import subprocess
    r = subprocess.run(["bash", str(REPO / "tests/pod/test_cold_host_tripwire.sh")],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "7 passed, 0 failed" in r.stdout, r.stdout


def test_setup_and_the_tripwire_test_stay_in_sync():
    """The test lifted the tripwire's structure; if setup's version drifts, the
    passing test stops meaning anything."""
    setup = (REPO / "scripts/pod/e5_setup.sh").read_text()
    for token in ('TRIP_S=${UV_TRIP_S:-360}', 'GRACE_S=${UV_GRACE_S:-180}',
                  'uv sync --group dev &', 'kill -0 "$UV_PID"',
                  'graced=0', 'exit 90',
                  # without the comparison the loop never trips at all
                  '[ "$el" -lt "$TRIP_S" ] && continue',
                  'el=$(( $(date -u +%s) - t0 ))'):
        assert token in setup, f"tripwire lost {token!r}"
    assert 'if [ "$graced" -eq 0 ] && [ -x /opt/train/bin/python ]' in setup, \
        "the grace clause must check the venv before abandoning a linking host"
    assert 'wait "$UV_PID" || { say "uv sync failed"; exit 1; }' in setup, \
        "a genuine sync failure must exit 1, never 90 -- 90 means redraw"


def test_launcher_redraws_on_a_cold_host_and_charges_it():
    src = (REPO / "scripts/pod/e5_launch.sh").read_text()
    assert 'for draw in $(seq 1 "$MAX_HOST_DRAWS"); do' in src
    block = src[src.index("SETUP_RC=$?"):src.index("say \"starting FORMAL E5")]
    assert '[ "$SETUP_RC" -eq 90 ]' in block, "90 must mean redraw"
    assert 'runpodctl remove pod "$POD_ID"' in block, "the cold pod must be deleted"
    assert 'redraws.log' in block, "every redraw must be recorded"
    assert 'billed_to_date' in block, "with its cost"
    assert '[ "$SETUP_RC" -ne 0 ]' in block, "other failures stay fatal"
    # The meter must not restart on a redraw.
    assert '[ -f "$SCR/pod_start_epoch" ] || date -u +%s > "$SCR/pod_start_epoch"' in src


def test_gate_1_stays_conservative_about_the_binding_arm():
    """Under nested selection C binds, not R: C is measured at 872/880 blocks on
    the real corpus while R estimates at ~591/596. The gate-1 figure must sit
    ABOVE the measured binding arm, so it can only be wrong in the safe
    direction until gate 2 measures the real packs."""
    import re as _re
    src = _e5_driver()
    n = int(_re.search(r'"--assumed-blocks", type=int, default=(\d+)', src).group(1))
    # 904 is now MEASURED on the staged corpora, not assumed, and reproduced
    # offline. It must never drop below that measurement.
    assert n >= 904, f"assumed-blocks {n} is below the measured 904"


def test_the_claim_boundary_travels_with_the_result():
    # Read the EVALUATED constant, not the source text: the literal is split
    # across source lines, so substring checks on the raw file silently pass.
    import ast
    tree = ast.parse(_e5_driver())
    cb = next(ast.literal_eval(n.value) for n in tree.body
              if isinstance(n, ast.Assign)
              and getattr(n.targets[0], "id", None) == "CLAIM_BOUNDARY")
    for phrase in ("training composition is no longer identical",
                   "does NOT isolate the pure causal effect",
                   "per fixed CE-supervision budget",
                   "do not remove the training-composition difference"):
        assert phrase in cb, f"claim boundary lost: {phrase!r}"
    assert '"claim_boundary": CLAIM_BOUNDARY' in _e5_driver(), \
        "the boundary must be written into the feasibility report"


def test_generated_corpora_are_retained_before_anything_can_fail():
    """Twice an R corpus costing ~$1.20 of GPU time was generated, accepted, and
    lost at teardown because the side bundle ships manifests, not examples."""
    src = _e5_driver()
    assert "_retain_corpora()" in src
    fn = src[src.index("def _retain_corpora("):src.index("def _system_ids(")]
    assert "e5_arm_r_corpora.tar.gz" in fn and "upload_file" in fn
    assert "except Exception as exc:" in fn, "retention must never fail the run"
    # It must run immediately after verification, not at teardown.
    stage = src[src.index("def stage_verify_records("):src.index("def _retain_corpora(")]
    assert stage.rstrip().endswith("_retain_corpora()")
    launcher = (REPO / "scripts/pod/e5_launch.sh").read_text()
    tar = launcher[launcher.index("tar czf"):launcher.index("cp /workspace/e5_run.log")]
    assert "e5_arm_r_*/" in tar, "the side bundle must carry the corpora too"
    assert "e5_final_*.jsonl" in tar, "and the paired selection"


def test_the_common_block_count_is_even_so_three_passes_divide():
    """`verify_pack` requires steps * blocks_per_step == 3 * n, so an odd n can
    never satisfy three passes at two blocks per step. Attempt 4 landed on 759
    and failed that check for no other reason."""
    src = _e5_driver()
    pair = src[src.index("common = max(minima.values())"):src.index("def stage_train(")]
    assert "odd_bump = common % 2" in pair and "common += odd_bump" in pair
    assert "rounded_up_for_even_passes" in pair, "the bump must be reported"
    # And the arithmetic must still be exact afterwards.
    assert 'report["optimizer_steps"] = common * 3 // 2' in pair


def test_the_launcher_waits_out_a_capacity_drought():
    """Waiting for capacity costs nothing -- no pod exists yet -- so giving up
    after two tries five minutes apart trades a free wait for a lost launch.
    Both of attempt 5's creates failed inside that window."""
    src = (REPO / "scripts/pod/e5_launch.sh").read_text()
    tries = int(re.search(r"MAX_POD_ATTEMPTS=\$\{MAX_POD_ATTEMPTS:-(\d+)\}", src).group(1))
    delay = int(re.search(r"CREATE_RETRY_DELAY_S=\$\{CREATE_RETRY_DELAY_S:-(\d+)\}", src).group(1))
    assert tries >= 8, f"{tries} create attempts is not patient enough"
    assert tries * delay >= 3600, \
        f"{tries} x {delay}s = {tries*delay}s waits out under an hour of drought"
    # Backing off must happen before a pod exists, so it must not be billed.
    assert "no capacity; retrying in" in src


def test_the_runtime_deadline_is_computed_per_create_not_at_launch():
    """Free capacity waiting must not consume the runtime backstop.

    Attempt 5 waited 7 minutes for capacity: the failed create at 16:47:36
    computed a 00:07:36Z deadline and created nothing, and the successful create
    at 16:54:37 computed a fresh 00:14:37Z. That only holds because `deadline` is
    computed INSIDE create_pod, once per attempt. Hoisting it to script scope
    would silently charge every drought minute against the pod's runtime.
    """
    src = (REPO / "scripts/pod/e5_launch.sh").read_text()
    fn = src[src.index("create_pod() {"):src.index("\n}", src.index("create_pod() {"))]
    assert 'deadline=$(date -u -d "+${BACKSTOP_MINUTES} minutes"' in fn, \
        "the deadline must be computed inside create_pod, per attempt"
    assert '--terminate-after "$deadline"' in fn
    # And it must not also be computed at script scope, where it would freeze.
    before = src[:src.index("create_pod() {")]
    assert "BACKSTOP_MINUTES} minutes" not in before, \
        "a script-scope deadline would be fixed before capacity is acquired"


def test_the_spending_meter_starts_at_the_successful_create():
    """`pod_start_epoch` is what every gate charges from, so it must be written
    only once a pod actually exists -- never during a capacity drought."""
    src = (REPO / "scripts/pod/e5_launch.sh").read_text()
    loop = src[src.index("POD_ID=$(create_pod"):src.index("# runpodctl 2.7.1")]
    epoch_at = loop.index('date -u +%s > "$SCR/pod_start_epoch"')
    fail_at = loop.index('if [ -z "$POD_ID" ]; then')
    assert fail_at < epoch_at, \
        "the failed-create branch must return before the meter starts"
    assert '[ -f "$SCR/pod_start_epoch" ] ||' in loop, "written once per session"
    wd_note = (REPO / "scripts/pod/e5_launch.sh").read_text()
    assert "no capacity; retrying in" in wd_note


def test_arm_r_is_staged_and_contract_checked_like_arm_c():
    """Attempt 5's R corpora cost ~$1.24 and survived; regenerating them costs
    90 min and pushes the run past the authorization."""
    setup = (REPO / "scripts/pod/e5_setup.sh").read_text()
    assert "e5_start/e5_arm_r.tar.gz" in setup
    assert "ARM R BUNDLE MISMATCH" in setup, "the staged bundle must be hashed"
    assert "scripts/data/verify_staged_r.py" in setup, \
        "a reused corpus needs independent verification, not just a bundle hash"
    v = (REPO / "scripts/data/verify_staged_r.py").read_text()
    for required in ("record count", "teacher revision", "decoding preset",
                     "P2-0.86M checkpoint identity", "chat_template.jinja",
                     "example_to_rendered", "system block matches its key",
                     "seed derivation calls sha256"):
        assert required in v, f"staged-R verification lost: {required}"
    assert "sys.exit(f\"STAGED R VERIFICATION FAILED" in v, "failures must be fatal"
    # The driver must then SKIP generation rather than overwrite it.
    src = _e5_driver()
    gen = src[src.index("def stage_generate("):src.index("def stage_verify_records(")]
    assert 'if (d / "examples.jsonl").exists():' in gen
    assert "skipping" in gen


def test_c_draws_from_its_full_pool_not_the_intersection():
    """The intersection was needed only while both arms shared a composition.
    Keeping it cost C 23.8% of its pool on sb and made attempt 5 infeasible."""
    src = _e5_driver()
    pair = src[src.index("def stage_pair("):src.index("def stage_final_benchmark(")]
    assert 'pools[f"C_{seed}"] = as_bundles(c_rows)' in pair, \
        "C must pool over its full corpus"
    assert 'pools[f"R_{seed}"] = as_bundles(rk)' in pair, "R stays intersected"
    assert 'kept[seed] = (c_rows, rk, census)' in pair


def test_corpus_retention_does_not_depend_on_an_inherited_env_var():
    """The driver is detached with setsid and does not inherit setup's HF_TOKEN;
    attempt 5 raised KeyError and fell back to the side bundle."""
    src = _e5_driver()
    fn = src[src.index("def _retain_corpora("):src.index("def _system_ids(")]
    assert 'os.environ.get("HF_TOKEN")' in fn and "/workspace/hf/token" in fn
    assert 'os.environ["HF_TOKEN"]' not in fn


def test_gate_1_does_not_budget_generation_that_will_be_skipped():
    """With the corpora staged, reserving 90 min for generation is a $1.48
    phantom. It failed gate 1 by $0.94 on 2026-08-07 while the real plan fitted."""
    src = _e5_driver()
    g = src[src.index("def stage_budget_gate_1"):src.index("def stage_final_benchmark")]
    assert 'staged = all(' in g and 'e5_arm_r_{s_}/examples.jsonl' in g
    assert '"r_generation": 0 if staged else 90' in g
    assert '"r_corpora_staged"' in g, "the gate must record which plan it priced"
    # The stale 1.30x-R rationale must not outlive the design it described.
    assert "conservative R = 1.30x" not in g


def test_the_checkpoint_tag_is_derived_not_hard_coded():
    """`step_000738` was a constant from the superseded 492-block design. It
    silently matched nothing and lost all four trained checkpoints on
    2026-08-07, after they had cost 117 minutes of GPU time."""
    src = (REPO / "scripts/pod/e5_launch.sh").read_text()
    # Executable lines only. The comment that explains this fix names the stale
    # tag, and a whole-file search flags the documentation for the bug it
    # documents -- the same trap the `hash()` docstring sprang earlier.
    code = [ln for ln in src.splitlines() if not ln.lstrip().startswith("#")]
    assert not [ln for ln in code if "step_000738" in ln], "the stale tag must not return"
    assert "STEP_TAG=$(" in src and "optimizer_steps" in src, \
        "the tag must come from the feasibility report's measured step count"
    fetch = src[src.index("hashing checkpoints on the pod"):]
    assert "$STEP_TAG" in fetch, "the fetch must use the derived tag"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_no_pod_script_checks_rope_on_a_meta_model(script):
    """A meta model has no buffer values, so the RoPE check cannot read them.

    E8 pod A died exactly here. The setup script built the teacher under
    `torch.device('meta')` to avoid materializing 16 GB in fp32, and
    `assert_rope_matches_config` went looking for `inv_freq` values that a meta
    tensor does not have. Setup exited 1 after TEACHER_READY and the session
    self-terminated at $0.08.

    The pod-environment simulation runs the *test suite*, not the setup script, so
    nothing local exercised that snippet. This is the cheap standing check:
    `assert_rope_from_config(config)` builds only the rotary embedding — 128 buffer
    elements, no model — and is what these scripts must use.
    """
    text = script.read_text()
    if "assert_rope_matches_config" not in text:
        return
    assert 'torch.device("meta")' not in text and "torch.device('meta')" not in text, (
        f"{script.name}: combines a meta-device model with "
        "assert_rope_matches_config, which cannot read a meta tensor. Use "
        "assert_rope_from_config(config) instead."
    )


def test_the_simulator_and_the_pod_gate_ignore_the_same_files():
    """The simulator is only evidence if it runs the pod's command.

    `simulate_pod_env.sh` exists to answer "would this suite pass on a pod?".
    An ignore list that differs from the pod's turns a green simulation into a
    statement about a command nobody runs. `--ignore` is the one place the two
    are allowed to be long, so it is the one place they are pinned.
    """
    import re

    def ignores(path):
        # Stops at shell quoting so the trailing `"}` of a ${VAR:-"..."}
        # default is not captured as part of the path.
        return sorted(re.findall(r"--ignore=([^\s\"']+)",
                                 (REPO / path).read_text().replace("\\\n", " ")))

    pod = ignores("scripts/pod/autoinit_preflight_setup.sh")
    sim = ignores("scripts/pod/simulate_pod_env.sh")
    assert pod, "the pod gate ignores nothing; the extractor is not matching"
    assert pod == sim, (
        f"the pod gate ignores {pod} but the simulator ignores {sim}; the "
        "simulation would not describe the pod's run")
    assert "tests/pod/test_phase_a_stages1_5_execute.py" in pod, (
        "the ~20-minute pre-flight rehearsal is back in the pod's 2700 s gate, "
        "whose timeout exits 90 and kills a paid session")
