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
    "transfer",
}
# Local trees inside the checked-out repo on the pod.
LOCAL_PREFIXES = {
    "artifacts/stage1/qwen3_0p6b_init_v0",
    "artifacts/stage3",
    "artifacts/audit",
    "configs/stage3",
    "data/warmup",
    "scripts",
    "tests",
    "src",
}
SCRIPTS = sorted(POD.glob("e[0-9]_setup.sh")) + sorted(POD.glob("e[0-9]_driver.py"))

# `stageN` names that are real. Anything else is almost certainly sed damage:
# the project has stages 0-6 but only these directory families exist.
VALID_STAGE_TOKENS = {"stage0", "stage1", "stage2", "stage3", "stage2_v1",
                      "stage3_recovery_corpus_v2", "stage3_pilot"}


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_no_invented_stage_directories(script):
    """`stage4_…`/`stage5_…` do not exist; finding one means a rename went wide."""
    text = script.read_text()
    for token in set(re.findall(r"stage\d+[a-z0-9_]*", text)):
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
        if not re.match(r"^(stage[0-9]|e1_scaling|transfer)", value):
            continue
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


@pytest.mark.parametrize("prefix", sorted({f.name[:2] for f in POD.glob("e[0-9]_*")}))
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
                          ("launcher", f"{prefix}_launch.sh")):
        f = POD / pattern
        if not f.is_file():
            continue
        found = set(re.findall(r"/?workspace/(e\d+\.status)", f.read_text()))
        found |= set(re.findall(r"STATUS=\$WS/(e\d+\.status)", f.read_text()))
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
