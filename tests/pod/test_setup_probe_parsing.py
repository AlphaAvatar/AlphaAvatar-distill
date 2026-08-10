"""The launcher's setup probe, and the $0.19 misread that motivated it.

A cold host is supposed to cost one draw and a redraw. On E8 pod A it cost the
whole session: the tripwire fired correctly at 360 s, the setup exited 90, and the
launcher classified it as `setup_failed` and aborted.

The cause was positional parsing of shell output. The probe was

    grep -c 'MARKER:SETUP_DONE' status || echo 0
    grep -c 'HOST_COLD'         status || echo 0
    tail -1 setup.log

and `grep -c` prints "0" **and exits 1** when there are no matches — so the
`|| echo 0` also runs and that command emits two lines. With SETUP_DONE absent,
line 1 was the stray "0" from the first command rather than the HOST_COLD count.
Everything shifted by one and the cold host became invisible.

These tests pin the replacement: labelled `KEY=value` output, read by key, with the
setup script's own exit code 90 as an independent cold-host signal.
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/pod"))

import e8a_launch  # noqa: E402
import e8b_launch  # noqa: E402

LAUNCHERS = (e8a_launch, e8b_launch)


@pytest.mark.parametrize("mod", LAUNCHERS, ids=lambda m: m.__name__)
def test_a_cold_host_is_read_as_cold(mod):
    probe = mod.parse_setup_probe(
        "SETUP_DONE=0\nHOST_COLD=1\nSETUP_RC=90\nTAIL=SETUP_RC=90\n")
    assert probe["host_cold"] == "1"
    assert probe["setup_rc"] == "90"
    assert probe["setup_done"] == "0"


@pytest.mark.parametrize("mod", LAUNCHERS, ids=lambda m: m.__name__)
def test_a_successful_setup_is_read_as_done(mod):
    probe = mod.parse_setup_probe(
        "SETUP_DONE=1\nHOST_COLD=0\nSETUP_RC=0\nTAIL=setup complete\n")
    assert probe["setup_done"] == "1"
    assert probe["host_cold"] == "0"
    assert probe["setup_rc"] == "0"


@pytest.mark.parametrize("mod", LAUNCHERS, ids=lambda m: m.__name__)
def test_the_exact_output_that_fooled_the_old_parser(mod):
    """The old code read this as host_cold="0". By key, it cannot."""
    old_style = "0\n0\n1\nSETUP_RC=90\n"          # what the shell actually emitted
    probe = mod.parse_setup_probe(old_style)
    # Unlabelled lines contribute nothing rather than shifting a field.
    assert probe["host_cold"] == "0"
    assert probe["setup_rc"] == "90"              # still recovered, from the label
    # And the launcher treats rc 90 as cold regardless of the marker count, so this
    # input can no longer produce an abort.


@pytest.mark.parametrize("mod", LAUNCHERS, ids=lambda m: m.__name__)
def test_missing_and_garbage_lines_default_instead_of_shifting(mod):
    assert mod.parse_setup_probe("") == {
        "setup_done": "0", "host_cold": "0", "setup_rc": "", "tail": ""}
    probe = mod.parse_setup_probe("garbage\nHOST_COLD=2\nUNKNOWN=x\n")
    assert probe["host_cold"] == "2"
    assert probe["setup_done"] == "0"


@pytest.mark.parametrize("mod", LAUNCHERS, ids=lambda m: m.__name__)
def test_the_probe_command_never_uses_the_duplicating_idiom(mod):
    """`grep -c … || echo 0` is the idiom that emitted two lines."""
    cmd = mod.PROBE_COMMAND
    assert "|| echo" not in cmd
    assert "SETUP_DONE=" in cmd and "HOST_COLD=" in cmd and "SETUP_RC=" in cmd
    # Every value is produced inside a command substitution and labelled, so a
    # command that prints nothing yields an empty value rather than no line.
    assert cmd.count("echo ") >= 4


@pytest.mark.parametrize("mod", LAUNCHERS, ids=lambda m: m.__name__)
def test_the_launcher_treats_exit_code_90_as_cold(mod):
    """The setup script's own signal, independent of the marker file."""
    source = Path(mod.__file__).read_text()
    assert 'probe["setup_rc"] == "90"' in source
    assert 'return "cold"' in source
