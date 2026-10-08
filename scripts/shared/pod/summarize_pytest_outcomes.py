#!/usr/bin/env python
"""The complete outcome of the pod CPU test gate, named rather than tailed.

C1 attempt 3R lost fourteen failures to a four-line tail. The `grep '^FAILED'`
added afterwards fixed that half — attempt 5's two failing nodeids arrived
exactly — but only that half: attempt 5's diagnostics named every FAILED nodeid
and no SKIPPED one, and the counts prove a skip divergence that is still
unexplained, because the pod's skip list died with the pod.

This reads the JUnit report the gate already writes and produces:

* every FAILED, ERROR and SKIPPED nodeid, with skip reasons;
* the totals;
* a deterministic digest over the complete skip set;
* the exact set difference against the launch-bound record's skip set.

Two differences matter and are printed separately. Attempt 5 produced
BOTH, and they must not be conflated:

* `expected_but_ran` — the sweep skipped it and the pod EXECUTED it. This is
  attempt 5's two FAILURES: the sweep recorded both staging-contract cases as
  expected skips, and the pod ran them and failed.
* `unexpected_skip` — the pod skipped something the sweep ran. This is attempt
  5's unexplained RESIDUAL: sweep 2770 passed / 100 skipped against pod 2769 /
  99, so after the two skip→fail transitions one test the sweep PASSED must have
  SKIPPED on the pod. It has never been named, because the pod's skip list was
  not captured.

    python scripts/pod/summarize_pytest_outcomes.py --junit X --out Y \
        [--expected logs/stages/stage-1/phase_c1/analyses/c1_pod_environment_verification.json] [--strict]

The record compared against is the one that AUTHORIZED this session: the
`readiness.json` beside `$SESSION_AUTH_PATH`, which the launcher already exports
to setup. `--expected` is the pre-2026-09-12 entry point and is now a pointer to
whichever run swept last; it is used only when this session owns no record.

`--strict` exits non-zero on any difference, and on an unavailable comparison
when the session owns a record — attempt 14 had one, compared against nothing,
and paid for the divergence at the pod gate. Entirely local: it reads two files
and writes one.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "src"))

from aadistill.runtime import pod_environment as pe

#: stdout is a `tail -40` window on a failing setup, so the full lists go to the
#: JSON and only what a human must see immediately is printed.
PRINT_LIMIT = 12

#: The pointer that used to be the record. Before 2026-09-12 the repository-root
#: file WAS the launch-bound readiness record; now it names the run that owns
#: the live one. Reading it as a record is how C1 attempt 14 compared against
#: nothing at all.
POINTER_SCHEMA = "aadistill.autoinit.c1_readiness_pointer/v1"


def resolve_expected(expected: Path | None, session_auth: str | None,
                     repo: Path) -> tuple[Path | None, bool, str | None]:
    """Which record THIS pod must match, and whether matching is mandatory.

    The record that authorized this session is the one beside the authorization
    the launcher named, because the issuer, the sweep and the launcher all
    address a run by the same governance directory. The repository-root path is
    a POINTER and names whichever run swept last: on attempt 14 it named attempt
    13, carried no skip set at all, and the comparison silently reported
    UNAVAILABLE on the one pod that needed it.

    Returns `(path, mandatory, why_not)`. `mandatory` is true exactly when a
    run-owned record was found: having one and not comparing against it is the
    blind spot, while a session that predates the layout keeps the old
    behaviour rather than being failed for its filing.
    """
    if session_auth:
        auth = Path(session_auth)
        auth = auth if auth.is_absolute() else repo / auth
        owned = auth.parent / "readiness.json"
        if owned.is_file():
            return owned, True, None
    if expected and expected.is_file():
        try:
            doc = json.loads(expected.read_text())
        except json.JSONDecodeError:
            doc = {}
        if doc.get("schema") == POINTER_SCHEMA:
            return expected, False, (
                f"{expected} is a pointer to run {doc.get('run_id')!r}, not a "
                "record. This session's own record was not found, so there is "
                "nothing it is entitled to be compared against.")
    return expected, False, None


def summarize(junit: Path, expected_record: Path | None, repo: Path,
              *, why_not: str | None = None) -> dict:
    parsed = pe.read_junit(junit, repo)
    outcomes = parsed["outcomes"]
    reasons = parsed.get("skip_reasons", {})
    details = parsed.get("failure_details", {})

    def ids(status: str) -> list[str]:
        return sorted(n for n, s in outcomes.items() if s == status)

    skipped = ids("skipped")
    out = {
        "schema": "aadistill.autoinit.c1_cpu_test_outcomes/v1",
        "counts": parsed["counts"],
        "total": parsed["total"],
        "failed_nodeids": ids("failed"),
        "error_nodeids": ids("error"),
        "all_skipped_nodeids": skipped,
        "skip_reasons": reasons,
        # Every failure and error with its message and traceback. Attempt 6
        # preserved 18 exact nodeids and no reason for any of them.
        "failure_details": details,
        "skip_set_digest": pe.skip_set_digest(skipped),
        "comparison": None,
    }

    if why_not:
        out["comparison"] = {"available": False, "why": why_not}
    elif expected_record and expected_record.is_file():
        rec = json.loads(expected_record.read_text())
        exp = (rec.get("findings") or {}).get("all_skipped_nodeids")
        if exp is None:
            out["comparison"] = {
                "available": False,
                "record": str(expected_record),
                "why": ("the committed readiness record predates the complete "
                        "skip set and carries only the named groups, so no exact "
                        "comparison is possible"),
            }
        else:
            cmp = pe.compare_skip_sets(exp, skipped)
            cmp["available"] = True
            cmp["record"] = str(expected_record)
            out["comparison"] = cmp
    else:
        out["comparison"] = {"available": False,
                             "why": f"no readiness record at {expected_record}"}
    return out


def report(out: dict) -> None:
    c = out["counts"]
    print(f"--- CPU test outcome: {c['passed']} passed, {c['skipped']} skipped, "
          f"{c['failed']} failed, {c['error']} error, {out['total']} selected ---")
    for label, key in (("FAILED", "failed_nodeids"), ("ERROR", "error_nodeids")):
        for nodeid in out[key]:
            print(f"{label} {nodeid}")
    # One line of WHY per failure, so the launcher's `tail -40` window carries a
    # mechanism and not only a list of names. The full bodies go to the JSON.
    for nodeid, d in list(out.get("failure_details", {}).items())[:PRINT_LIMIT]:
        first = (d.get("message") or d.get("body") or "").strip().splitlines()
        if first:
            print(f"  WHY {nodeid.rsplit('::', 1)[-1]}: {first[0][:160]}")
    print(f"skip set digest: {out['skip_set_digest'][:16]}… "
          f"({len(out['all_skipped_nodeids'])} skipped, all named in the JSON)")

    cmp = out.get("comparison") or {}
    if not cmp.get("available"):
        print(f"skip-set comparison UNAVAILABLE: {cmp.get('why')}")
        return
    if cmp["identical"]:
        print("skip-set comparison: IDENTICAL to the launch-bound sweep")
        return
    print(f"skip-set comparison: DIFFERS from the launch-bound sweep "
          f"({cmp['expected_skips']} expected vs {cmp['actual_skips']} actual)")
    for label, key in (("expected-skip-but-RAN", "expected_but_ran"),
                       ("unexpected POD-ONLY skip", "unexpected_skip")):
        items = cmp[key]
        if not items:
            continue
        print(f"  {len(items)} {label}:")
        for nodeid in items[:PRINT_LIMIT]:
            print(f"    {nodeid}")
        if len(items) > PRINT_LIMIT:
            print(f"    … {len(items) - PRINT_LIMIT} more, all in the JSON")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--junit", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expected", default="")
    ap.add_argument("--session-authorization",
                    default=os.environ.get("SESSION_AUTH_PATH", ""),
                    help="the authorization this session runs under. Its "
                         "sibling readiness.json is the record this pod must "
                         "match. Defaults to $SESSION_AUTH_PATH, which the "
                         "launcher already exports to setup.")
    ap.add_argument("--repo", default=str(HERE.parents[2]))
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()

    junit = Path(a.junit)
    if not junit.is_file():
        print(f"--- CPU test outcome: no JUnit report at {junit} ---")
        Path(a.out).write_text(json.dumps(
            {"schema": "aadistill.autoinit.c1_cpu_test_outcomes/v1",
             "error": f"no JUnit report at {junit}"}, indent=1) + "\n")
        return 0                      # never mask the suite's own exit code

    record, mandatory, why_not = resolve_expected(
        Path(a.expected) if a.expected else None,
        a.session_authorization or None, Path(a.repo))
    out = summarize(junit, record, Path(a.repo), why_not=why_not)
    out["expected_record"] = str(record) if record else None
    out["comparison_is_mandatory"] = mandatory
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    report(out)

    cmp = out.get("comparison") or {}
    if a.strict and cmp.get("available") and not cmp.get("identical"):
        print("REFUSING: the pod's skip set is not the launch-bound sweep's. "
              "The sweep therefore did not certify this machine, which is the "
              "condition that cost attempt 5 a grant.")
        return 1
    #: A session that OWNS a launch-bound record and cannot be compared against
    #: it is the attempt-14 condition: the divergence that failed the gate would
    #: have been named here for nothing, and was not. An unavailable comparison
    #: is a failed comparison whenever the record exists.
    if a.strict and mandatory and not cmp.get("available"):
        print(f"REFUSING: this session's launch-bound record is {record} and "
              "the pod's skip set could not be compared against it: "
              f"{cmp.get('why')}. A record that cannot be checked certifies "
              "nothing.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
