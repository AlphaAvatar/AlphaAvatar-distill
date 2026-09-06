"""The one CPU-test environment, declared once and used by both machines.

C1 attempt 5's `--strict` skip-set comparison was correct machinery pointed at
two DIFFERENT environments. The launch-bound diagnostic runs on a CPU dev box
with an isolated empty HF cache; the pod runs on an L40S with the pinned teacher
already downloaded. Compared exactly, a healthy pod would have been refused for
being a healthy pod:

* a GPU predicate decides one way here and the OTHER way there;
* a teacher-cache predicate decides one way here and the OTHER way there.

The repair is not to weaken the comparison. It is to make the two commands run
under the same deliberately hardware- and cache-neutral environment, declared
here so neither side can drift from a prose list.

**Command scope only.** These variables wrap the pytest invocation and nothing
else. Setup still installs the real environment, downloads the pinned teacher and
proves CUDA — all outside this scope, and re-asserted after pytest so the
isolation is shown not to have leaked into the science runtime.

**The token is not isolated.** A real pod exports a real `HF_TOKEN` and the
simulation injects a synthetic one; both are non-empty, and the CACHE is what is
neutralized. Whether a token is real or synthetic must never be a skip predicate
— `AAD_SYNTHETIC_HF_TOKEN` is never set on a pod, and a guard keyed on it is the
defect that cost attempt 5 a grant.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

CONTRACT = "aadistill.autoinit.c1_cpu_test_env/v1"

#: Declares the SUITE, never the machine. Both the simulator and the paid pod set
#: it; nothing else does. A predicate keyed on it says "my output is consumed by
#: the readiness record this very invocation produces" — the one question that is
#: genuinely the same on both machines and genuinely unanswerable inside the
#: sweep. It is emphatically NOT a simulator flag: keying an artifact premise on
#: one of those is what cost attempt 5 a grant.
SCOPE_MARKER = "AAD_C1_CPU_TEST_SCOPE"

#: Set for the pytest command. `HOME` is a fresh empty directory, so anything
#: that locates host-local state through the home directory finds nothing —
#: which is the pod's condition.
def overlay(scope_root: str | Path) -> dict[str, str]:
    """The variables the pytest command runs under, derived from one root.

    The argument is the SCOPE root, not `HOME` itself: `HOME` is a subdirectory
    of it and the Hugging Face root is a SIBLING, so `HOME` stays genuinely
    empty. Nesting the HF root inside `HOME` would put one entry in it, and
    "the simulated HOME is empty" is an invariant worth keeping literally true.
    """
    root = str(scope_root).rstrip("/")
    return {
        # WHICH SUITE THIS IS, not which machine runs it. Set identically by the
        # simulator and by the pod, so it can never become a simulator marker.
        #
        # `record_pod_environment.py` moves the previous readiness record aside
        # while producing its replacement, so a test that CONSUMES that record
        # cannot meaningfully run inside the sweep that produces it: it skips in
        # every sweep and runs on every pod, forever. That is a measurement
        # circularity, not a machine difference, and it is the only thing this
        # marker is allowed to express.
        SCOPE_MARKER: "1",
        # No accelerator, on either machine. The pod has an L40S; a predicate
        # that asks `torch.cuda.is_available()` would otherwise decide the
        # OPPOSITE way there from the diagnostic.
        "CUDA_VISIBLE_DEVICES": "",
        # A fresh, EMPTY home. Host-local artifact stores are located under it,
        # so an empty one makes them invisible on both machines.
        "HOME": f"{root}/home",
        # One isolated Hugging Face root, empty on both machines, so a
        # cache-dependent case makes the same decision on both.
        "HF_HOME": f"{root}/hf",
        "HF_HUB_CACHE": f"{root}/hf/hub",   # a sibling of HOME, not inside it
    }


#: Cleared for the pytest command: each can silently redirect a cache lookup
#: back at the host's real one, which is exactly the divergence being closed.
UNSET: tuple[str, ...] = (
    "HUGGINGFACE_HUB_CACHE",
    "HF_DATASETS_CACHE",
    "TRANSFORMERS_CACHE",
    "XDG_CACHE_HOME",
)

#: Deliberately NOT touched: `HF_TOKEN` (presence is the same on both machines,
#: and the cache — not the credential — is what differs) and the session's own
#: `SESSION_*` variables, which the manifest sets identically for both.
PRESERVED: tuple[str, ...] = ("HF_TOKEN", "SESSION_KIND", "SESSION_TEST_IGNORES")


def in_cpu_test_scope() -> bool:
    """Is this pytest run the C1 CPU candidate suite the readiness record consumes?

    True in the launch-bound/diagnostic sweep AND on the paid pod, false in an
    ordinary `pytest tests/` run. The two machines therefore agree, which is the
    whole point: a marker that only one of them sets is the attempt-5 defect.
    """
    import os
    return bool(os.environ.get(SCOPE_MARKER))


def env_args(scope_root: str | Path) -> list[str]:
    """Arguments for `env(1)`, so the scope is ONE command and not the shell.

    Exporting these over the whole setup would neutralize the teacher download
    and the CUDA proof that must happen in the real environment.
    """
    args: list[str] = []
    for key in UNSET:
        args += ["-u", key]
    args += [f"{k}={v}" for k, v in sorted(overlay(scope_root).items())]
    return args


def digest() -> str:
    """Deterministic over the contract, so a record can bind the environment it
    was produced under and a later change invalidates it."""
    body = {"contract": CONTRACT, "set": sorted(overlay("<SCOPE>")),
            "unset": list(UNSET), "preserved": list(PRESERVED),
            "template": overlay("<SCOPE>")}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def describe(scope_root: str | Path = "<SCOPE>") -> dict:
    return {"contract": CONTRACT, "digest": digest(), "scope_root": str(scope_root),
            "set": overlay(scope_root), "unset": list(UNSET),
            "preserved_not_isolated": list(PRESERVED)}


def host_local_store(name: str = "aad-artifacts") -> Path:
    """The out-of-tree artifact store, LOCATED THROUGH `$HOME`.

    It was `Path("/home/ecs-user/aad-artifacts")` in a dozen places. On the real
    dev box this resolves identically, so nothing about normal operation changes
    — but a hardcoded absolute path is immune to the fresh empty `HOME` above,
    which is what let host-local cases run in the diagnostic and skip on the pod.
    Deriving it is what makes the declared isolation actually isolate.
    """
    return Path.home() / name


def main() -> int:                                        # pragma: no cover
    """`env_args` for a shell: `env $(… --home DIR) <command>`."""
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--home", required=True,
                    help="the SCOPE root; HOME becomes <root>/home")
    ap.add_argument("--format", choices=("envargs", "sh", "json"),
                    default="envargs")
    a = ap.parse_args()
    if a.format == "json":
        print(json.dumps(describe(a.home), indent=1))
    elif a.format == "sh":
        # For the simulator, which isolates its whole subshell and restores it
        # from an EXIT trap. The pod uses `envargs` instead, because there the
        # scope must be ONE command.
        print("unset " + " ".join(UNSET))
        for k, v in sorted(overlay(a.home).items()):
            print(f'export {k}="{v}"')
    else:
        print(" ".join(env_args(a.home)))
    return 0


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())
