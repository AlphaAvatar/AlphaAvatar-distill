#!/usr/bin/env python3
"""Everything deterministic that a launch-bound sweep assumes, checked in seconds.

    PYTHONPATH=src python scripts/consolidate/converge_before_sweep.py
    PYTHONPATH=src python scripts/consolidate/converge_before_sweep.py --write

Three of the first four launch-bound sweeps failed on derived records that had
not been regenerated: a skip predicate changed without re-running the audit, a
sweep wrote a record without the snapshot following it, a stage index built
against a tree the pod does not have. Each cost thirteen minutes to discover
something a second of generation would have shown.

So this runs every generator, runs each a SECOND time, and requires the second
pass to change nothing. A generator that is not at a fixed point is the whole
failure mode: it means the committed document and the tree disagree, and the
sweep is the most expensive possible way to find that out.

It is NOT a test suite. Nothing here executes pytest, and it must stay that way
— the point is that it costs seconds. `--write` regenerates; without it,
nothing is modified and drift is reported.

The launch preconditions it also checks are the ones the chain silently
depends on:

* the working tree is clean, because the issuer refuses a dirty one and the
  session commit must describe what the pod checks out;
* the grant resolves to this run's canonical path and validates;
* that path and the run's readiness record are TRACKABLE — `runs/` is
  gitignored, and until the stage-first evidence root was exempted the one path
  the lineage rule permits was a path git could not see;
* a launch-bound sweep will not rewrite the global pointer, which is tracked
  and would otherwise put an unpermitted change into the swept tree.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

#: (label, argv). Order matters: the stage index and the run index feed the
#: navigation, and the inventory reads the tree they leave behind.
GENERATORS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("skip-predicate audit",
     ("scripts/autoinit/audit_skip_predicates.py", "--write")),
    ("stage index",
     ("scripts/consolidate/stage_attribution.py", "--write")),
    ("run index",
     ("scripts/architecture/record_run_index.py", "--write")),
    ("navigation + snapshot",
     ("scripts/consolidate/render_log_navigation.py", "--write")),
    ("log inventory",
     ("scripts/consolidate/build_log_inventory.py",
      "--out", "logs/maintenance/inventories/log_inventory.json")),
    ("document links",
     ("scripts/consolidate/fix_doc_links.py", "--write")),
)


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args],
                          capture_output=True, text=True, check=True).stdout


def run(argv: tuple[str, ...]) -> int:
    return subprocess.run([sys.executable, *argv], cwd=REPO,
                          capture_output=True, text=True).returncode


#: Fields a generator stamps with the moment it ran. A document whose ONLY
#: change is one of these has not drifted -- it has been regenerated -- and
#: counting it as drift makes the fixed-point check unsatisfiable and leaves the
#: tree dirty for the very sweep that requires it clean.
STAMP_FIELDS = ('"created_utc"', '"generated_utc"', '"recorded_utc"')


def dirty() -> list[str]:
    return [l[3:] for l in git("status", "--porcelain").splitlines() if l.strip()]


def stamp_only(rel: str) -> bool:
    """Is this file's whole diff a regenerated timestamp?"""
    diff = git("diff", "-U0", "--", rel).splitlines()
    body = [l for l in diff
            if (l.startswith(("+", "-"))
                and not l.startswith(("+++", "---")))]
    return bool(body) and all(any(f in l for f in STAMP_FIELDS) for l in body)


def revert_stamp_only(paths) -> list[str]:
    """Put back the files whose only change is when they were generated."""
    reverted = [rel for rel in paths if stamp_only(rel)]
    if reverted:
        subprocess.run(["git", "-C", str(REPO), "checkout", "--", *reverted],
                       check=True)
    return reverted


def generators(write: bool) -> list[str]:
    """Run each generator twice; report anything the second pass still moves."""
    problems = []
    before = set(dirty())
    for label, argv in GENERATORS:
        if run(argv):
            problems.append(f"{label}: generator exited non-zero")
    first = set(dirty()) - before
    for label, argv in GENERATORS:
        run(argv)
    #: EVERY dirty file, not only the newly dirty ones. Scoped to the new set,
    #: a document already carrying nothing but a fresh timestamp when this
    #: started stayed dirty and blocked the clean-tree check it exists to serve.
    second = set(dirty())
    stamped = set(revert_stamp_only(sorted(second)))
    moved = second - stamped - first - before
    if moved:
        problems.append("not at a fixed point after a second pass: "
                        + ", ".join(sorted(moved)))
    real = first - stamped
    if real and not write:
        problems.append("regeneration changed: " + ", ".join(sorted(real))
                        + " (re-run with --write intended, then commit)")
    if stamped:
        print("  regenerated, timestamp only, reverted: "
              + ", ".join(sorted(stamped)))
    return problems


def launch_preconditions(run_id: str, stage_id: str) -> list[str]:
    from experiments.phase_c1 import pod_environment as pe
    from experiments.phase_c1.authorization_payload import (
        C1AuthorizationRefused, build_c1_authorization_payload)
    from experiments.run_layout import rel_run_dir

    problems = []
    grant_rel = f"{rel_run_dir('phase_c1', run_id, stage_id)}/governance/grant.json"
    grant_p = REPO / grant_rel
    if not grant_p.is_file():
        problems.append(f"no grant at {grant_rel}")
    else:
        try:
            build_c1_authorization_payload(
                grant=json.loads(grant_p.read_text()),
                session_commit=git("rev-parse", "HEAD").strip(),
                granted_utc="1970-01-01T00:00:00+00:00", repo_root=REPO,
                grant_path="<dry-run>")
        except C1AuthorizationRefused as exc:
            problems.append(f"the issuer would refuse this grant: {exc}")

    record_rel = pe.record_path_for(run_id, stage_id)
    auth_rel = f"{rel_run_dir('phase_c1', run_id, stage_id)}/governance/authorization.json"
    for rel in (grant_rel, record_rel, auth_rel):
        if subprocess.run(["git", "-C", str(REPO), "check-ignore", "-q", rel]
                          ).returncode == 0:
            problems.append(f"{rel} is gitignored, so it can never be the "
                            "permitted post-sweep path the lineage rule names")

    permitted = pe.permitted_post_sweep_paths(run_id, stage_id)
    if list(permitted) != [record_rel]:
        problems.append(f"the lineage exemption is {permitted}, not exactly "
                        f"[{record_rel}]")
    if any(ch in p for p in permitted for ch in "*?["):
        problems.append(f"the lineage exemption contains a glob: {permitted}")

    src = (REPO / "scripts/autoinit/record_pod_environment.py").read_text()
    if 'args.kind != "launch_bound"' not in src:
        problems.append("the recorder does not guard the global pointer "
                        "against a launch_bound sweep")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true",
                    help="regenerate; without it nothing is modified")
    ap.add_argument("--run-id", default="attempt13")
    ap.add_argument("--stage-id", default="1")
    a = ap.parse_args()

    problems = generators(a.write)
    problems += launch_preconditions(a.run_id, a.stage_id)
    left = dirty()
    if left:
        problems.append(f"{len(left)} uncommitted change(s); the sweep must "
                        "describe a committed tree")

    for p in problems:
        print("  PROBLEM:", p)
    if problems:
        print(f"\nNOT READY — {len(problems)} problem(s)")
        return 1
    print(f"ready: generators at a fixed point, tree clean, grant valid at "
          f"{a.run_id}, lineage exemption exact, pointer guarded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
