#!/usr/bin/env python3
"""Reconstruct D1's unretained quality-order finalists on one GPU, digest-pinned.

    PYTHONPATH=src:scripts python scripts/pod/autoinit_d1_replay_driver.py \
        --out artifacts/audit/autoinit_d1_replay --run-id <id> --device cuda

Two leaves, four pinned steps each. The session decides nothing: the candidate
set is read from the maintainer's retention decision and every step is pinned to
the artifact digest the completed search recorded for it, so the only two
outcomes are "byte-identical to the search" and "STOP".

A digest mismatch is TERMINAL for that leaf and is not retried: the path is
deterministic, so the same inputs diverge the same way. It is a finding for
review, not an engineering failure to repair.

The product here is WEIGHTS. Each leaf is written where the launcher's
`fetch_products` can find it, and a leaf that reconstructs is reported as such
the moment it finishes -- never gated on the whole session succeeding, because
`if terminal == "ALL_DONE"` has already destroyed verified checkpoints in this
programme.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
for _extra in ("src", "scripts", "scripts/autoinit"):
    if str(REPO_ROOT / _extra) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / _extra))

SUCCESS_MARKER = "ALL_DONE"
FAILURE_MARKER = "RUN_FAILED"
MISMATCH_MARKER = "DIGEST_MISMATCH"
CHECK_ONLY_MARKER = "CHECK_ONLY_OK"


def mark(status_path: str, name: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(status_path, "a") as fh:
        fh.write(f"{stamp} MARKER:{name}\n")
        fh.flush()


def terminal_marker(*, check_only: bool, status: str,
                    mismatch: bool, evidence_written: bool) -> str:
    """The marker the launcher polls for. One predicate, ONE caller.

    The first version marked inside the `--check-only` branch AND again in the
    `finally`, so a clean check-only run appended `CHECK_ONLY_OK` and then
    `RUN_FAILED` -- and a launcher polling the status file would have read the
    second. There is now exactly one `mark` call, in the `finally`, and this
    function is the whole decision.

    `evidence_written` gates success on BOTH paths: a run that reconstructed
    everything and could not write its record has not produced the thing the
    session owes.
    """
    if mismatch:
        return MISMATCH_MARKER
    if not evidence_written:
        return FAILURE_MARKER
    if check_only:
        return CHECK_ONLY_MARKER if status == "CHECK_ONLY_OK" else FAILURE_MARKER
    return SUCCESS_MARKER if status == "COMPLETE" else FAILURE_MARKER


def build_parser() -> argparse.ArgumentParser:
    """At module scope so the dispatch probe can build it. Every required
    argument is a STRING: the probe fills them with one."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True,
                    help="evidence directory, repository-relative on the pod")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--status", default=None,
                    help="where the terminal marker is appended")
    ap.add_argument("--plan", default=None,
                    help="the resolved replay plan, written by the launcher "
                         "from the source run's journal. The journal itself "
                         "does not travel: 67 MB over a 0.72 MB/s uplink is "
                         "~93 min of billing to ship what resolves to ~8 KB")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--products", default=None,
                    help="where reconstructed leaves are written")
    ap.add_argument("--check-only", action="store_true",
                    help="resolve the plan and every symbol, materialize "
                         "nothing, create no artifact")
    return ap


def main(argv: list[str] | None = None) -> int:
    from experiments.phase_d1 import d1_session as D1S

    args = build_parser().parse_args(argv)
    status = args.status or f"{D1S.POD_WORKSPACE}/autoinit_d1_replay.status"
    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    products = Path(args.products) if args.products else out / "products"
    products.mkdir(parents=True, exist_ok=True)

    record: dict[str, Any] = {
        "schema": "aadistill.phase_d1.replay_session/v1",
        "run_id": args.run_id,
        "experiment_id": D1S.EXPERIMENT_ID,
        "stage_id": D1S.STAGE_ID,
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": args.device,
        "status": "RUNNING",
        "leaves": [],
    }
    evidence = out / "d1_replay.json"

    def save() -> None:
        evidence.write_text(json.dumps(record, indent=2, default=str) + "\n")

    mark(status, "DRIVER_START")
    mismatch = False
    try:
        from aadistill.initialization.specs.arch import get_adapter
        from aadistill.initialization.planning.fixed_path import (
            FixedPathDigestMismatch, materialize_fixed_path,
        )
        from experiments.phase_d1 import replay_specs as R

        if args.plan:
            plan_doc = json.loads(Path(args.plan).read_text())
            leaves = R.leaves_from_plan(plan_doc)
            record["plan"] = plan_doc
            record["plan_source"] = str(args.plan)
        else:
            #: Only reachable where the journal is present, which is the dev
            #: box. A pod always receives `--plan`.
            leaves = R.replay_plan(REPO_ROOT)
            record["plan"] = R.describe(leaves)
            record["plan_source"] = R.DEFAULT_JOURNAL
        save()

        from aadistill.initialization.specs.materialization import (
            NumericalEnvironment,  # noqa: F401  (resolved here, used on the pod)
        )

        specs = [(leaf, R.fixed_path_spec(leaf, device=args.device,
                                          repo_root=REPO_ROOT))
                 for leaf in leaves]
        record["spec_hashes"] = {leaf.state_id: spec.spec_hash
                                 for leaf, spec in specs}
        save()

        if args.check_only:
            record["status"] = "CHECK_ONLY_OK"
            record["_check_only"] = (
                "the plan resolved, every step is pinned, both FixedPathSpecs "
                "built and every symbol the materialization path names "
                "imported. No teacher was loaded and no artifact was written.")
            save()
            return 0

        #: All four process-global registries, via D1's own single owner. This
        #: registered adapters alone and step 0 raised on an empty calibration
        #: registry after setup had been paid for.
        D1S._register_frozen_operators()
        adapter = get_adapter(R.FAMILY)

        def load_root(spec):
            """The teacher THIS path is pinned to, reloaded per path.

            Reloaded rather than shared because operators MUTATE the module
            they are given: a second path starting from the first path's root
            would begin from an already-compressed model and diverge at step
            one. After the first load the weights are in the local cache, so
            this is a disk read; the alternative is wrong.

            No config overrides. D1's search produced ONE root state for every
            level-0 expansion -- all eight operator x profile combinations
            recorded a single config hash per profile -- so unlike the C2
            replay there is no historical root mutation to reconstruct. If that
            reading is wrong, step 0 of the cheapest path says so in under a
            minute; see the ordering below.
            """
            import torch
            from transformers import AutoModelForCausalLM

            return AutoModelForCausalLM.from_pretrained(
                spec.root_repo_id, dtype=torch.bfloat16,
                revision=spec.root_revision).to(args.device).eval()

        #: CHEAPEST FIRST STEP FIRST. If the root state is not what the search
        #: expanded from, every digest diverges at step 0 -- and the cost of
        #: learning that depends entirely on which operator sits there. The
        #: search measured `depth.causal_kl_greedy_v1` at up to 41 min and the
        #: other three at under 50 s, so a path whose first step is FFN reports
        #: a wrong root for about a minute of GPU time instead of forty.
        _first_cost = {"depth.causal_kl_greedy_v1": 2}
        specs.sort(key=lambda ls: _first_cost.get(ls[1].steps[0].impl_id, 1))
        record["leaf_order"] = [leaf.state_id for leaf, _ in specs]
        record["_leaf_order_why"] = (
            "cheapest first operator first, so a wrong root state is reported "
            "after about a minute rather than after the DEPTH operator's 41")
        save()

        for leaf, spec in specs:
            leaf_dir = products / leaf.state_id
            leaf_dir.mkdir(parents=True, exist_ok=True)
            entry: dict[str, Any] = {
                "state_id": leaf.state_id,
                "quality_position": leaf.quality_position,
                "path_label": leaf.path_label,
                "spec_hash": spec.spec_hash,
                "steps_completed": 0,
                "reconstructed": False,
                "adopted": False,
            }
            record["leaves"].append(entry)
            save()
            print(f"[{leaf.state_id}] q{leaf.quality_position} "
                  f"{leaf.path_label}", flush=True)

            def on_step(result, _entry=entry) -> None:
                _entry["steps_completed"] = int(result.index) + 1
                _entry.setdefault("step_seconds", []).append(
                    round(float(result.seconds), 2))
                print(f"  step {result.index} {result.impl_id} -> "
                      f"{result.identity.artifact_digest[:12]} "
                      f"(pinned {str(result.digest_expected)[:12]}, "
                      f"{result.seconds:.0f}s)", flush=True)
                save()

            t0 = time.time()
            try:
                results = materialize_fixed_path(
                    spec, adapter=adapter,
                    root_loader=lambda _s=spec: load_root(_s),
                    workdir=leaf_dir, repo_root=REPO_ROOT, on_step=on_step)
            except FixedPathDigestMismatch as exc:
                #: TERMINAL for this leaf. A deterministic replay that diverges
                #: is a finding: not retried, not substituted.
                mismatch = True
                entry["digest_mismatch"] = {
                    "step_index": getattr(exc, "step_index", None),
                    "label": getattr(exc, "label", None),
                    "expected": getattr(exc, "expected", None),
                    "actual": getattr(exc, "actual", None),
                    "verdict": (
                        "replay mismatch. The path is deterministic, so the "
                        "same inputs would diverge the same way; this is not "
                        "retried and no near-equivalent checkpoint is "
                        "substituted. Adoption of this candidate stops until "
                        "the materialization discrepancy is diagnosed."),
                }
                entry["seconds"] = round(time.time() - t0, 1)
                save()
                print(f"  DIGEST MISMATCH at step "
                      f"{entry['digest_mismatch']['step_index']}", flush=True)
                continue

            entry["seconds"] = round(time.time() - t0, 1)
            entry["reconstructed"] = True
            artifact = results[-1].identity if results else None
            ok, detail = R.adoption_matches(leaf, artifact)
            entry["identity"] = detail
            entry["adopted"] = bool(ok)
            entry["checkpoint_path"] = (
                results[-1].checkpoint_path if results else str(leaf_dir))
            save()
            print(f"  {'ADOPTED' if ok else 'IDENTITY MISMATCH'} "
                  f"({entry['seconds']:.0f}s)", flush=True)
            if not ok:
                mismatch = True

        adopted = [e for e in record["leaves"] if e.get("adopted")]
        record["n_adopted"] = len(adopted)
        record["status"] = ("COMPLETE" if len(adopted) == len(specs)
                            else "INCOMPLETE")
    except BaseException as exc:                                  # noqa: BLE001
        import traceback

        record["status"] = "FAILED"
        record["failure"] = {"type": type(exc).__name__, "message": str(exc),
                             "traceback": traceback.format_exc()}
        raise
    finally:
        record["ended_utc"] = datetime.now(timezone.utc).isoformat(
            timespec="seconds")
        written = False
        try:
            save()
            written = True
        except Exception as exc:                                  # noqa: BLE001
            print(f"could not write {evidence}: {exc}", flush=True)
        print(f"\nwrote {evidence}  status={record['status']}", flush=True)
        mark(status, terminal_marker(
            check_only=bool(args.check_only), status=record["status"],
            mismatch=mismatch, evidence_written=written))
    return 0 if record["status"] == "COMPLETE" else 11


if __name__ == "__main__":
    raise SystemExit(main())
