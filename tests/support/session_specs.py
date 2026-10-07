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
    #: A3. Added WITH the launcher, which is what the comment above asks for:
    #: it needs no `extra` any more, because `session_args` now fills every
    #: required option from the parser itself -- `--max-price` included.
    ("autoinit_a3_launch", ()),
    #: D1. Absent until 2026-10-05, so none of the four structural modules that
    #: read this list covered it: not the declared-vs-read `required_env` check,
    #: not the local-asset install-root check, not the setup-env forwarding
    #: check. D1 declares `TEACHER_REVISION` and three local assets, and both of
    #: those declarations are exactly what those modules exist to verify against
    #: the shell that consumes them.
    ("autoinit_d1_launch", ()),
    #: D1's REPLAY, and the same omission as the entry above, found the same
    #: way: the four structural modules that read this list -- the
    #: declared-vs-read `required_env` check, the local-asset install-root
    #: check, the setup-env forwarding check, and the sweep-registry check --
    #: covered every session except this one. Its FIRST paid subrun died in
    #: setup on an asset it had not declared, which is precisely the class the
    #: local-asset check exists to catch at $0.
    ("autoinit_d1_replay_launch", ()),
)


def load_session_launcher(name: str):
    path = REPO / f"scripts/pod/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _placeholder(action) -> str:
    """A value the action's OWN type and choices accept.

    Typed from the parser, not guessed: a required `type=float` flag answered
    with `"spec_check"` is exit 2 inside the conversion instead of exit 2 at
    the missing-argument check, which looks like a different bug.
    """
    if action.choices:
        return str(next(iter(action.choices)))
    conv = getattr(action, "type", None)
    if conv is float:
        return "1.0"
    if conv is int:
        return "1"
    return "spec_check"


def session_args(mod, extra: tuple[str, ...] = (), **overrides):
    """The namespace the launcher's REAL parser produces.

    Device-canary attempt 1 died at $0.0603 on an attribute a hand-written
    namespace would have had and the real parser did not. Tests build their
    namespace here for that reason.

    A launcher that requires `--run-id` gets one, unless the caller supplied its
    own. Asked of the REAL parser rather than of a list kept here, so a second
    session adopting the run layout needs no edit — and so this helper cannot
    quietly satisfy a required argument that the parser dropped.

    **Every OTHER required option is filled the same way, from the parser's own
    metadata.** It used to be `--run-id` and nothing else, with anything new
    arriving through the caller's `extra` — so a launcher that declared a
    required flag this helper did not know about did not merely lose one check:
    `argparse` exits 2 during `parse_args`, which takes the whole launcher out
    of `derive_session`, out of `all_specs()` and out of every structural check
    that iterates sessions. A3's `--max-price` is `type=float` and required, so
    the readiness sweep it is the second step of could not build a session spec
    at all. Filling from `action.required` keeps that impossible for the next
    one, and the value is typed from the action rather than passed as a string,
    because `float("spec_check")` is the same exit 2 one line later.
    """
    argv = ["--scr", "/tmp/session-spec-test",
            "--session-commit", "0" * 40,
            "--bundle", "aad_test.bundle", *extra]
    parser = mod.build_parser()
    accepts_run_id = any("--run-id" in (a.option_strings or ())
                         for a in parser._actions)
    if accepts_run_id and "--run-id" not in argv:
        argv += ["--run-id", "spec_check"]
    for action in parser._actions:
        if not (action.required and action.option_strings):
            continue
        flag = action.option_strings[0]
        if flag in argv or any(a in argv for a in action.option_strings):
            continue
        if action.nargs == 0:                 # a required store_true, rare
            argv.append(flag)
            continue
        argv += [flag, _placeholder(action)]
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
