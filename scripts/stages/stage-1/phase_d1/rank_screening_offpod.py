#!/usr/bin/env python3
"""Rank a screening field from its SECURED evidence, off-pod, at `$0`.

WHY THIS EXISTS. `rank_screening` had exactly one caller: the pod driver, at
the end of the run. So a session that scored all ten probes and then died in
ranking, artifact collection or teardown would have had every number it
needed sitting in the secured store and no way to use them -- the only route
back to a selection was a fresh pod and ten retrainings, about thirteen hours
and $16 to recompute a pure function of rows already on disk.

It is a pure function. `rank_screening` reads four fields per probe -- arm,
seed, correct_overall, usable_rollout_rate -- and every one of them is in the
`result.json` the launcher's poll hook secures the moment a probe is scored.
Nothing about ranking needs a GPU, the weights, or the pod.

WHAT THIS DOES NOT DO. It does not relax the completeness requirement.
`rank_screening` refuses a partial field, and so does this: a field missing
any (arm, seed) is a field where ranking would select on who happened to
finish. If probes are missing, the answer is not a partial ranking.

It also does not rescue a session that failed BEFORE scoring all ten. Those
probes' measurements do not exist, and `protocol_uniformity` forbids training
the remainder in another session and pooling the two -- so that case is a
full retrain, by the frozen protocol rather than by any limitation here.

    python scripts/stages/stage-1/phase_d1/rank_screening_offpod.py \\
        --store <scr>/store/probes --out ranking.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _extra in ("src", "scripts"):
    if str(REPO / _extra) not in sys.path:
        sys.path.insert(0, str(REPO / _extra))

from stages.phase_d1 import behavioural as B  # noqa: E402


class OffpodRankingError(RuntimeError):
    """The secured store does not support a ranking."""


def load_scored(store: Path) -> tuple[list[dict], dict[str, str]]:
    """Every secured probe's ranking row, with the digest of each file read.

    Reads the destination the poll hook writes -- `<store>/<probe_id>/
    result.json` -- so this consumes the evidence that actually came home.
    Every file's sha256 is returned beside the rows: an artifact that cannot
    name the bytes it consumed is not reproducible (P4).
    """
    rows: list[dict] = []
    digests: dict[str, str] = {}
    schedule = {p.probe_id: p for p in B.probes("screening", REPO)}
    for probe_dir in sorted(p for p in store.glob("*") if p.is_dir()):
        result = probe_dir / "result.json"
        if not result.is_file():
            continue
        data = result.read_bytes()
        digests[probe_dir.name] = hashlib.sha256(data).hexdigest()
        doc = json.loads(data)
        missing = [k for k in ("correct_overall", "usable_rollout_rate")
                   if doc.get(k) is None]
        if missing:
            raise OffpodRankingError(
                f"{result} has no {missing}; rank_screening reads these per "
                "probe and a probe without them is not scored")
        #: THE ARM COMES FROM THE FROZEN SCHEDULE, NOT THE SCORER.
        #: `score_d1_screening` deliberately records an opaque `label` and
        #: leaves `arm` null -- screening has five arms and C1's
        #: incumbent/treatment pair does not describe them, so nothing
        #: downstream may read a screening field through that vocabulary. The
        #: pod driver gets the arm from `probe.arm_id` for the same reason.
        #: Taking it from the schedule rather than parsing the probe id also
        #: means an id this session never scheduled is refused instead of
        #: being silently split into an arm and a seed.
        probe = schedule.get(probe_dir.name)
        if probe is None:
            raise OffpodRankingError(
                f"{probe_dir.name} is not in the frozen screening schedule; "
                "the store holds a probe this rung does not define")
        rows.append({"arm": probe.arm_id, "seed": probe.seed,
                     "correct_overall": float(doc["correct_overall"]),
                     "usable_rollout_rate": float(doc["usable_rollout_rate"]),
                     "probe_id": probe_dir.name})
    if not rows:
        raise OffpodRankingError(
            f"no secured result.json under {store}; there is nothing to rank")
    return rows, digests


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True, type=Path,
                    help="the secured probe store: <scr>/store/probes")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    rows, digests = load_scored(args.store)
    #: `rank_screening` owns completeness, the pooling, the tie-break and the
    #: guardrail veto. This file computes none of them -- a second ranking
    #: formula is a second thing that can disagree about the selection.
    ranking = B.rank_screening(rows, REPO)
    selection = B.advance_one(ranking)

    out = {
        "schema": "aadistill.phase_d1.offpod_screening_ranking/v1",
        "_contract": ("the frozen ranking and mechanical selection, computed "
                      "off-pod from secured evidence. The rule, the pooling, "
                      "the tie-break and the guardrail veto are "
                      "behavioural.rank_screening's; nothing is recomputed "
                      "here."),
        "store": str(args.store),
        "n_probes_read": len(rows),
        "result_sha256_by_probe": digests,
        "ranking": ranking,
        "selection": selection,
        "_this_authorizes_no_promotion": (
            "screening advances ONE candidate to confirmation. Promotion or "
            "adoption of any checkpoint is a separate scientific decision and "
            "a separate maintainer decision."),
    }
    text = json.dumps(out, indent=1) + "\n"
    if args.out:
        args.out.write_text(text)
        print(f"wrote {args.out}")
    print(f"\n{len(rows)} probes, pooled over {len(ranking['seeds'])} seeds:")
    for row in ranking["ranked"]:
        print(f"  {row.get('arm'):4} delta {row.get('delta_vs_b'):+.4f} "
              f"guardrail={'PASS' if row.get('guardrail_ok', True) else 'VETO'}")
    if selection.get("advanced"):
        print(f"\nADVANCING: {selection['arm']} delta "
              f"{selection['delta_vs_b']:+.4f}")
    else:
        print(f"\nNO_CANDIDATE ADVANCES: {selection.get('why')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
