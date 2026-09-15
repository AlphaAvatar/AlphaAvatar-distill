"""Which optional setup steps a session declared. One rule, two callers.

`SetupManifest.setup_markers` has been declared by every session since the
shared setup script existed, and until 2026-09-16 it was read by **nothing**.
The script ran its sections unconditionally and emitted the markers as it went,
so the declaration described what would happen rather than deciding it — and a
session that omitted a marker got the step anyway. A paid session was lost to
it; `docs/core-provenance.md` records which.

So the declaration is now the contract, and this module is the rule both sides
read:

* the runner exports it as `SESSION_SETUP_MARKERS`, built from the manifest;
* the shared shell asks this module, per step, whether to run that step.

**One implementation, not two.** A `case` statement in the shell and a predicate
in Python would be two rules the moment either is edited, and the one that
decides what a paid pod does is the one nobody can unit-test. The shell calls
the CLI below; a test calls `requires()` directly.

Two properties make it safe to be asked by a shell, and both were learned the
hard way at `$0`:

**The exit codes cannot collide with an interpreter failure.** `0` is declared,
`3` is not declared, everything else is unusable. `1` and `2` are deliberately
absent: a Python that cannot run this file at all exits `1` with a traceback,
and `runpy` and `argparse` exit `2`. With `1` meaning "not declared", a syntax
error under an old interpreter read as *skip this step* — silently skipping the
authorization check, the test gate and the frozen-asset gate, which is the
original defect inverted and worse. A crash now lands in the unusable branch
and the setup script aborts.

**It runs on an old interpreter.** The shell asks this before any project venv
exists, with whatever `python3` the pod image ships, so the file avoids every
syntax feature that would make it unparseable rather than merely unusual: no
`from __future__ import annotations`, no `X | Y` annotations, no `collections.abc`
subscripting. The dev box's `python3` is 3.6 and it parses this; if it did not,
the failure would be indistinguishable from an undeclared step.

Nothing here knows an experiment, a phase, a model or a step's meaning. The
marker names are the session's vocabulary: a future stage declaring
`CALIBRATION_READY` needs no change here.
"""

import os
import sys

#: The environment value the runner exports and the shell reads.
DECLARATION_ENV = "SESSION_SETUP_MARKERS"

#: Exit codes. `1` and `2` are RESERVED FOR FAILURE and never returned: see the
#: module docstring. The shell must treat anything outside {DECLARED,
#: NOT_DECLARED} as fatal.
DECLARED = 0
NOT_DECLARED = 3
UNUSABLE = 4


def parse(declaration):
    """The declared marker names, in declaration order, duplicates removed.

    Accepts the environment form (whitespace- or comma-separated) and an
    iterable, so the runner and a test can hand it the same thing in either
    shape without a second parser.
    """
    if declaration is None:
        return ()
    if isinstance(declaration, str):
        names = declaration.replace(",", " ").split()
    else:
        names = [str(n).strip() for n in declaration]
    out = []
    for name in names:
        if name and name not in out:
            out.append(name)
    return tuple(out)


def requires(declaration, step):
    """Did this session declare `step`?

    Case- and whitespace-insensitive on the step NAME only: a marker is a
    constant in the session's own declaration, and a caller typing
    `vllm_ready` for `VLLM_READY` is asking the same question.
    """
    wanted = str(step).strip().upper()
    return any(name.upper() == wanted for name in parse(declaration))


def main(argv=None, environ=None):
    """`python3 -m aadistill.runtime.setup_steps <STEP>` -> 0 / 3 / 4."""
    argv = list(sys.argv[1:] if argv is None else argv)
    env = os.environ if environ is None else environ
    if len(argv) != 1 or not argv[0].strip():
        sys.stderr.write(
            "usage: python3 -m aadistill.runtime.setup_steps <STEP>\n")
        return UNUSABLE
    declared = parse(env.get(DECLARATION_ENV))
    if not declared:
        sys.stderr.write(
            DECLARATION_ENV + " is absent or empty, so this session declared "
            "no setup contract. Refusing to decide which optional steps to "
            "run: an unusable declaration is not permission to skip them.\n")
        return UNUSABLE
    return DECLARED if requires(declared, argv[0]) else NOT_DECLARED


if __name__ == "__main__":
    sys.exit(main())
