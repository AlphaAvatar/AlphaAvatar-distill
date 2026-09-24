#!/usr/bin/env python
"""Recompute the C2 behavioural verdict FROM THE COMMITTED ARCHIVE. `$0`.

The verdict attempt13 produced was correct and is not in question. What was
missing is the ability to establish that independently: probe 11's 950
per-sample rows and its `result.json` were scored on the pod and never
collected, so the archive held eleven complete probes and one that looked
trained-but-unscored, and the terminal decision could not be recomputed from
committed evidence. AGENTS.md P4 makes that a defect in the result, not a
footnote to it.

This reads the six confirmation probes' rows out of the durable destination and
runs the frozen path — `decision_inputs` -> `paired_differences` ->
`stratified_cluster_bootstrap` -> `decide` — through `behavioural_decision.confirm`,
which is the same function the driver's stage D calls. It re-implements none of
it. In particular it does NOT align the rows itself: `decision_inputs` refuses a
duplicate prompt id, a prompt set that differs between probes, a scorable count
that is not 850 and a total that is not 950, and every one of those would
otherwise still produce a number.

It also does not decide on a partial field. `confirm` requires six probes and
the three frozen seeds; with anything less this exits non-zero and says what is
missing, which is the behaviour that makes the repair verifiable rather than
assumed.

Arms come from each probe's own `probe_record.json`, not from parsing its id:
the record is what the driver persisted and it carries `arm`, `seed` and `rung`
directly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments.phase_c2 import behavioural_decision as BD  # noqa: E402
from experiments.phase_c2 import behavioural_schedule as SCH  # noqa: E402

DEFAULT_STORE = ("/home/ecs-user/aad-artifacts/phase_c2_behavioural"
                 "/c2-behavioural-12probe-v1")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def confirmation_probes(store: Path) -> dict[str, dict[str, Any]]:
    """Every confirmation probe the archive holds, keyed by probe id.

    Walks the attempt directories because a campaign's probes are produced by
    whichever resource happened to run: probe 11 is under `attempt12` and the
    incumbent's third seed under `attempt13`, and both belong to one experiment.
    """
    found: dict[str, dict[str, Any]] = {}
    for attempt_dir in sorted(p for p in store.iterdir() if p.is_dir()):
        for probe_dir in sorted(p for p in attempt_dir.iterdir() if p.is_dir()):
            record = probe_dir / "probe_record.json"
            if not record.is_file():
                continue
            try:
                rec = json.loads(record.read_text())
            except json.JSONDecodeError:
                continue
            if rec.get("rung") != "confirmation":
                continue
            pid = str(rec.get("probe_id") or probe_dir.name)
            rows = probe_dir / "per_sample.jsonl"
            result = probe_dir / "result.json"
            #: Last writer wins only if it is MORE complete. A probe can appear
            #: under two attempts -- one that trained it and one that scored it
            #: -- and the one carrying rows is the one the verdict can read.
            prior = found.get(pid)
            if prior and prior["has_rows"] and not rows.is_file():
                continue
            found[pid] = {
                "probe_id": pid, "arm": rec.get("arm"),
                "seed": int(rec["seed"]) if rec.get("seed") is not None else None,
                "dir": probe_dir, "attempt": attempt_dir.name,
                "has_rows": rows.is_file(), "has_result": result.is_file(),
                "rows_path": rows, "result_path": result,
            }
    return found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", default=DEFAULT_STORE)
    ap.add_argument("--out", required=True, help="where to write the record")
    ap.add_argument("--compare-runtime", default=None,
                    help="attempt13's driver_evidence.json, to check the "
                         "reconstruction against what the pod observed")
    a = ap.parse_args(argv)

    store = Path(a.store)
    probes = confirmation_probes(store)
    rule = BD.decision_rule(REPO)

    print(f"confirmation probes in the archive: {len(probes)}")
    incomplete = []
    for pid, p in sorted(probes.items()):
        mark = "ok " if (p["has_rows"] and p["has_result"]) else "GAP"
        print(f"  [{mark}] {pid[:56]:58s} arm={str(p['arm'])[:34]:36s} "
              f"seed={p['seed']} from={p['attempt']}")
        if not (p["has_rows"] and p["has_result"]):
            incomplete.append(pid)

    if incomplete:
        print(f"\nREFUSING: {len(incomplete)} confirmation probe(s) hold no "
              f"rows or no result in the archive: {incomplete}. The frozen "
              "rule is a paired difference over six probes and three seeds; a "
              "verdict from a partial field is not the preregistered "
              "experiment. This is the state the P4 repair exists to end.")
        return 2

    rows: dict[tuple[str, int], list[dict]] = {}
    fingerprints: dict[str, Any] = {}
    for pid, p in sorted(probes.items()):
        arm = (BD.INCUMBENT_ARM if p["arm"] == SCH.ANCHOR
               else BD.TREATMENT_ARM)
        rows[(arm, p["seed"])] = [json.loads(l) for l in
                                  p["rows_path"].open() if l.strip()]
        fingerprints[pid] = {
            "arm": p["arm"], "decision_arm": arm, "seed": p["seed"],
            "source_attempt": p["attempt"],
            "per_sample_sha256": sha256_of(p["rows_path"]),
            "result_sha256": sha256_of(p["result_path"]),
            "n_rows": len(rows[(arm, p["seed"])]),
        }

    decision = BD.confirm(rows, rule=rule)
    decision["_recomputed_from"] = "the committed archive, not a pod"
    decision["_why"] = (
        "P4 reproducibility repair. attempt13's runtime produced this verdict "
        "and its rows for probe 11 were never collected; that probe's "
        "evaluation was reconstructed from its preserved, identity-verified "
        "checkpoint under the frozen protocol, and the decision then recomputed "
        "here from the complete six-probe field.")
    decision["probe_fingerprints"] = fingerprints

    d = decision["decision"]
    print(f"\nterminal_state   {decision['terminal_state']}")
    for k in ("delta", "lcb_one_sided", "ucb_one_sided", "sesoi"):
        if k in d:
            print(f"{k:16s} {d[k]}")
    print(f"per_seed_delta   {decision['per_seed_delta_correct']}")
    print(f"usable_pooled    {decision['usable_pooled_delta']}")
    print(f"criteria         {json.dumps(d.get('criteria'))}")

    if a.compare_runtime:
        obs = (json.loads(Path(a.compare_runtime).read_text())
               .get("stages", {}).get("D", {}))
        same_state = obs.get("terminal_state") == decision["terminal_state"]
        drift = {k: (obs.get(k), d.get(k)) for k in ("delta", "lcb")
                 if obs.get(k) is not None}
        cmp = {"runtime_terminal_state": obs.get("terminal_state"),
               "recomputed_terminal_state": decision["terminal_state"],
               "terminal_state_agrees": same_state,
               "runtime_delta": obs.get("delta"),
               "recomputed_delta": d.get("delta"),
               "runtime_lcb": obs.get("lcb"),
               "recomputed_lcb": d.get("lcb_one_sided")}
        decision["runtime_comparison"] = cmp
        print(f"\nruntime comparison: {json.dumps(cmp, indent=1)}")
        if not same_state:
            print("\nMATERIAL DISAGREEMENT: the reconstruction does not "
                  "reproduce the runtime's terminal state. This is a "
                  "scientific boundary, not something to average or conceal. "
                  "The record is written; stop here.")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(decision, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
