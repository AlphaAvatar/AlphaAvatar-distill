#!/usr/bin/env python3
"""One closeout per A3 attempt, DERIVED from that attempt's own evidence.

    PYTHONPATH=src:scripts python scripts/autoinit/write_a3_attempt_closeouts.py --write

`derive_budget` discovers a session's cost through the run index and reads it
out of the run's own closeout (`cost.actual_usd` or `budget.this_attempt`). A3's
pre-science attempts are *unrecorded* runs -- the launcher aborted before
writing a manifest -- so every one of them reported `cost_usd: None` and the
`$2.7400` they really spent was invisible to the derived formal balance while
the ledger prose carried it. One fact, two owners, and the machine-readable one
was empty.

So each attempt gets the record the deriver reads, built from the two
authoritative local sources and nothing else:

* the launcher's own `runtime/session.json`, which it writes on EVERY path
  including an abort;
* the launcher log's teardown line, which is where the provider-confirmed
  minutes and dollars are stated.

Nothing is invented. An attempt whose log states no teardown cost is written
with `actual_usd: null` and a reason, because reading an unknown as zero
understates spend -- which is the one direction that matters.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

RUNS = REPO / "logs/stages/stage-1/phase_a3/runs"
SESSIONS = Path("/home/ecs-user/aad-artifacts/phase_a3/_sessions")

SCHEMA = "aadistill.autoinit.a3_attempt_outcome/v1"

#: `pod deleted — 4.6 min, $0.08; provider confirms gone: True`
_TEARDOWN = re.compile(
    r"pod deleted\D+([0-9.]+) min, \$([0-9.]+).*?gone: (\w+)")
#: `created e0a2yb7ps0z545 at $1.09/h`
_CREATED = re.compile(r"created (\w+) at \$([0-9.]+)/h")
_SETUP_FAIL = re.compile(
    r"autoinit_preflight_setup\.sh: line \d+: ([A-Z_]+): (.*)")


def normalized_signature(log: str) -> str | None:
    """The attempt's deterministic failure signature, or None.

    The SAME string the same-failure rule compares, so a closeout cannot
    disagree with the gate about what an attempt failed on.
    """
    from aadistill.infrastructure.failure_signature import signature_of

    return signature_of(log)


#: WHAT AN ATTEMPT'S OWN EVIDENCE SAYS IT REACHED. The first version of this
#: writer asserted `PRE_SCIENCE_ABORT_NO_MEASUREMENT` and
#: `measurement_began: False` for every attempt, which was true of the 37 that
#: aborted and false of `a3_attempt38`, the one that finished: it produced three
#: scored probe records and the closeout said no probe was trained. A derived
#: record that cannot describe the successful path is not derived, it is a
#: default. So the classification is read from the files the attempt wrote.
def reached(run_dir: Path) -> dict:
    ev = run_dir / "evidence"
    scored = sorted(p.name for p in ev.glob("*_c1_confirmation.json"))
    admitted = sorted(p.name for p in ev.glob("*_generation_admission.json"))
    #: `evidence/probes` is written by the preservation stage, so it is the
    #: trained-and-durable fact `a3_attempt35` owns and the resume consumed.
    preserved = (ev / "probes").is_dir()
    return {"scored": scored, "admitted": admitted, "preserved": preserved}


def classify(pods: list[str], ev: dict) -> tuple[str, str]:
    if not pods:
        return ("PRE_PROVIDER_ABORT_NOTHING_CREATED",
                "one consumed one-use A3 chain that created no provider "
                "resource. No probe was trained and no measurement was taken.")
    if ev["scored"]:
        return (f"MEASUREMENT_COMPLETE_{len(ev['scored'])}_PROBES_SCORED",
                "a complete A3 measurement. The scientific figures belong to "
                "the comparison artifact, which is computed off pod from this "
                "attempt's evidence; this record carries cost and cause.")
    if ev["preserved"] or ev["admitted"]:
        return ("TRAINED_PRESERVED_NOT_VALIDLY_SCORED",
                "probes were trained and preserved and no valid endpoint was "
                "produced -- AGENTS.md P8.4 state 2. The checkpoints are a "
                "resume input, not a result, and nothing here authorizes "
                "pooling or reusing them.")
    return ("PRE_SCIENCE_ABORT_NO_MEASUREMENT",
            "one consumed one-use A3 chain. No probe was trained, no "
            "generation was produced and no measurement was taken, so this "
            "record carries cost and cause and no scientific field.")


def attempt_record(run_dir: Path) -> dict:
    name = run_dir.name
    log_path = SESSIONS / name / "launcher.log"
    log = log_path.read_text(errors="replace") if log_path.is_file() else ""

    pods = [m.group(1) for m in _CREATED.finditer(log)]
    rates = [float(m.group(2)) for m in _CREATED.finditer(log)]
    teardowns = [(float(m.group(1)), float(m.group(2)), m.group(3) == "True")
                 for m in _TEARDOWN.finditer(log)]
    minutes = round(sum(t[0] for t in teardowns), 2) if teardowns else 0.0
    usd = round(sum(t[1] for t in teardowns), 4) if teardowns else None
    if not pods:
        usd = 0.0                      # no resource existed; $0 is a FACT here

    session = run_dir / "runtime/session.json"
    sess = json.loads(session.read_text()) if session.is_file() else {}
    setup_fail = _SETUP_FAIL.search(log)

    ev = reached(run_dir)
    status, what = classify(pods, ev)

    return {
        "schema": SCHEMA,
        "run_id": name,
        "experiment_id": "phase_a3",
        #: STATED, so `formal_sessions` attributes this run to the package
        #: that funds it rather than to the declared-id fallback.
        "package_id": "phase_c1.execution_package.2026-09-11",
        "status": status,
        "_what_this_is": what,
        "provider_resource_created": bool(pods),
        "pod_ids": pods,
        "all_pods_confirmed_gone": bool(teardowns) and all(t[2] for t in teardowns),
        "terminal": sess.get("terminal"),
        "launcher_exit": sess.get("launcher_exit"),
        #: DERIVED, not asserted. Training, an admitted generation or a scored
        #: record all mean formal work began, and the retry rules turn on that.
        "measurement_began": bool(ev["scored"] or ev["admitted"]
                                  or ev["preserved"]),
        "probe_records_scored": ev["scored"],
        "generations_admitted": len(ev["admitted"]),
        "probe_checkpoints_preserved": ev["preserved"],
        "session_commit": sess.get("session_commit"),
        "deterministic_failure_signature": normalized_signature(log),
        "setup_refusal": (f"{setup_fail.group(1)}: {setup_fail.group(2)}"
                          if setup_fail else None),
        "cost": {
            "actual_usd": usd,
            "elapsed_minutes": minutes,
            "price_per_hour": rates[0] if rates else None,
            "_source": ("summed from this attempt's own launcher-log teardown "
                        "lines, each of which states provider confirmation"),
            "_null_means": ("a pod existed and no teardown cost could be read "
                            "from the log; it is NOT zero"),
        },
        "authorizes": "nothing",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)

    total, n_pods, written = 0.0, 0, 0
    unknown = []
    for run_dir in sorted(RUNS.iterdir()):
        if not run_dir.is_dir():
            continue
        rec = attempt_record(run_dir)
        usd = rec["cost"]["actual_usd"]
        if usd is None:
            unknown.append(rec["run_id"])
        else:
            total += usd
        n_pods += len(rec["pod_ids"])
        out = run_dir / "closeout/outcome.json"
        if args.write:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
            written += 1
        print(f"  {rec['run_id']:14s} pods={len(rec['pod_ids'])} "
              f"${'none' if usd is None else f'{usd:.4f}':>8s}  "
              f"{(rec['deterministic_failure_signature'] or '-')[:52]}")

    print(f"\n{written} closeout(s) written · {n_pods} pod(s) created · "
          f"${total:.4f} total")
    if unknown:
        print(f"UNKNOWN cost for {unknown}; reading an unknown as zero would "
              "understate spend")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
