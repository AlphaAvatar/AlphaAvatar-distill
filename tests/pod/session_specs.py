"""Load every session specification the way the launcher's own `main` does.

Shared by the rebound launcher tests and by the structural checks, so that a
test asserting on a session asserts on **the real parser and the real spec** —
not on a regex over the launcher's source, which is how a transcription and the
thing it transcribes come to disagree.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: (name, extra argv). The continuation requires `--transport`; everything else
#: runs on defaults, and `session_args` supplies `--run-id` to any launcher whose
#: parser declares one. Keep this list complete: a session missing from it is a
#: session no structural check covers.
SESSION_LAUNCHERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("autoinit_preflight_launch", ()),
    ("autoinit_phase_a_launch", ()),
    ("autoinit_continuation_launch", ("--transport", "relay")),
    ("autoinit_device_canary_launch", ()),
    ("autoinit_measurement_launch", ()),
    ("autoinit_recovery_continuation_launch", ()),
    ("autoinit_c1_launch", ()),
)


def load_session_launcher(name: str):
    path = REPO / f"scripts/pod/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def session_args(mod, extra: tuple[str, ...] = (), **overrides):
    """The namespace the launcher's REAL parser produces.

    Device-canary attempt 1 died at $0.0603 on an attribute a hand-written
    namespace would have had and the real parser did not. Tests build their
    namespace here for that reason.

    A launcher that requires `--run-id` gets one, unless the caller supplied its
    own. Asked of the REAL parser rather than of a list kept here, so a second
    session adopting the run layout needs no edit — and so this helper cannot
    quietly satisfy a required argument that the parser dropped.
    """
    argv = ["--scr", "/tmp/session-spec-test",
            "--session-commit", "0" * 40,
            "--bundle", "aad_test.bundle", *extra]
    parser = mod.build_parser()
    accepts_run_id = any("--run-id" in (a.option_strings or ())
                         for a in parser._actions)
    if accepts_run_id and "--run-id" not in argv:
        argv += ["--run-id", "spec_check"]
    args = parser.parse_args(argv)
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def all_specs():
    """`(name, module, args, spec)` for every session."""
    out = []
    for name, extra in SESSION_LAUNCHERS:
        mod = load_session_launcher(name)
        args = session_args(mod, extra)
        out.append((name, mod, args, mod.spec(args)))
    return out
