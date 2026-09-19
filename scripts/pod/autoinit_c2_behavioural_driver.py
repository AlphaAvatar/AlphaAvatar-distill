#!/usr/bin/env python3
"""The Phase-C2 behavioural selection: twelve probes, two rungs, one decision.

    PYTHONPATH=src:scripts python \
        scripts/pod/autoinit_c2_behavioural_driver.py --stage all ...

Stages, in the order the frozen protocol fixes them:

    P  prepare  — verify the five durable candidates and MATERIALIZE incumbent B
                  from the frozen C1 treatment path, gated on B's exact identity
    S  screen   — six fresh recovery probes, one seed, five candidates plus B
    R  rank     — score `c2_screening_v1`, rank by paired delta, advance ONE
    C  confirm  — six fresh probes, the advanced candidate and B, three seeds
    D  decide   — score `c1_confirmation_v1`, apply C1's frozen rule, STOP

It does not implement recovery training or battery scoring. Those are C1's,
already executed on real hardware across three attempts, and `train_one` /
`score_one` are seams C1 built precisely so a `$0` regression can replace them.
This driver subclasses that machinery and changes what it has to: which probes
exist, when the second rung may begin, and who advances.

**Screening never decides.** It ranks and advances exactly one candidate, and
`assert_screening_emits_no_verdict` refuses a screening result that carries a
verdict, an incumbent or a promotion. Only the confirmation rung, on three
disjoint seeds, may name a C2 incumbent — and `NO_GO` and `INCONCLUSIVE` are
results, not failures to be retried.

**Confirmation cannot start early.** All six screening probes must be trained
AND scored and the winner mechanically determined first; otherwise the candidate
being confirmed was chosen from whoever happened to finish.

**Finished probes are persisted as they complete.** C1 attempt 17 trained six
probes over ten hours and lost every one; `preserve_probe` is inherited for that
reason and is called from both rungs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
for _extra in ("src", "scripts", "scripts/autoinit", "scripts/pod"):
    if str(REPO_ROOT / _extra) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / _extra))

from autoinit_c1_driver import (  # noqa: E402
    AUDIT, C1Driver, C1DriverError, mark, say,
)

from experiments.phase_c2 import behavioural as BH  # noqa: E402
from experiments.phase_c2 import behavioural_schedule as SCH  # noqa: E402

RUN_ID = "autoinit.v1.phase_c2.behavioural"
SUCCESS_MARKER = "C2_BEHAVIOURAL_ALL_DONE"
FAILURE_MARKER = "C2_BEHAVIOURAL_FAILED"
B_READY_MARKER = "C2_BEHAVIOURAL_B_READY"
PROBE_MARKER = "C2_BEHAVIOURAL_PROBE_DONE"
ADVANCED_MARKER = "C2_BEHAVIOURAL_ADVANCED"


class BehaviouralDriverError(RuntimeError):
    """This session cannot proceed on the evidence it has."""


class C2BehaviouralDriver(C1Driver):
    """C1's probe machinery, two rungs, and a selection between them.

    Subclassed rather than reimplemented: a harness that rebuilt the training
    loop could not notice the loop being broken, and this programme has already
    certified a defective line that way once.
    """

    def __init__(self, a) -> None:
        super().__init__(a)
        self.candidates: list[dict[str, Any]] = []
        self.anchor: dict[str, Any] = {}
        self.screening: list[SCH.Probe] = []
        self.confirmation: list[SCH.Probe] = []
        self.ranked: list[dict[str, Any]] = []
        self.advanced: dict[str, Any] | None = None
        #: Keyed by probe_id, not by (arm, seed): the same arm appears in both
        #: rungs and on repeated seeds, and a key that collided would let one
        #: rung's probe satisfy the other's completeness check.
        self.probe_training: dict[str, dict] = {}
        self.probe_scores: dict[str, dict] = {}

    # -- P: the inputs, including the one that does not exist yet ------------
    def stage_p(self) -> None:
        """Verify the five candidates; materialize and gate the sixth arm."""
        mark("STAGE_START:P")
        self.candidates = BH.candidate_manifest(REPO_ROOT)
        say(f"  {len(self.candidates)} durable candidates verified against the "
            "frozen selection")

        binding = BH.b_binding(REPO_ROOT, device=self.a.device)
        say(f"  B construction {binding['construction']['spec_hash'][:12]}… == "
            f"frozen {binding['construction']['expected_spec_hash'][:12]}…")

        if binding["availability"]["available"]:
            say("  B is already durable; staging rather than rebuilding")
            b_path = binding["availability"]["durable_path"]
        else:
            b_path = self.materialize_b(binding)

        self.anchor = {
            "state_id": SCH.ANCHOR,
            "artifact_digest": binding["required_identity"]["artifact_digest"],
            "durable_path": b_path,
        }
        mark(B_READY_MARKER)
        self.complete("P", candidates=len(self.candidates),
                      b_spec_hash=binding["construction"]["spec_hash"],
                      b_artifact_digest=self.anchor["artifact_digest"],
                      b_materialized=not binding["availability"]["available"])

    def materialize_b(self, binding: dict[str, Any]) -> str:
        """Build B from the frozen C1 treatment path, then gate on its identity.

        The construction comes from `baseline.frozen_baseline_spec`, which builds
        it with C1's own constructor; this method only runs it and checks the
        result. If the result is not B, no screening probe may start: an anchor
        that is not the frozen incumbent makes every delta meaningless.
        """
        from aadistill.initialization.planning.fixed_path import (
            materialize_fixed_path,
        )
        from aadistill.initialization.specs.arch import get_adapter

        from experiments.phase_c2 import baseline as BL

        say("  B is not durable — materializing it from the frozen treatment path")
        spec = BL.frozen_baseline_spec(device=self.a.device)
        BL.assert_frozen_construction(spec)

        workdir = Path(self.a.b_workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        results = materialize_fixed_path(
            spec, adapter=get_adapter("qwen3"),
            root_loader=lambda: self.load_teacher(), workdir=workdir,
            repo_root=REPO_ROOT)
        final = results[-1]

        required = binding["required_identity"]
        observed = final.identity
        mismatched = [
            f"{field}: required {required[field]!r}, built "
            f"{getattr(observed, field)!r}"
            for field in ("artifact_digest", "weights_digest",
                          "single_shard_sha256", "arch_signature",
                          "num_parameters")
            if getattr(observed, field) != required[field]
        ]
        if mismatched:
            raise BehaviouralDriverError(
                "the rebuilt B is not the frozen incumbent — "
                + "; ".join(mismatched)
                + ". NO SCREENING PROBE MAY START: an anchor that is not B "
                  "makes every delta meaningless. This is a provenance "
                  "finding, not a retryable failure.")
        say(f"  B rebuilt and identity-gated in {(time.time() - t0) / 60:.1f} min")

        #: Persisted immediately. B costs half an hour of GPU to build and a
        #: later infrastructure failure must not force it again.
        dest = Path(self.a.b_durable)
        preserved = self.preserve_probe("incumbent_b", Path(final.checkpoint_path),
                                        {"probe_id": "incumbent_b",
                                         "artifact_digest": required["artifact_digest"]})
        (AUDIT / "incumbent_b.identity.json").write_text(json.dumps({
            "schema": "aadistill.autoinit.c2_incumbent_b/v1",
            "construction": binding["construction"],
            "required_identity": required,
            "built_identity": final.as_dict(),
            "preserved": preserved,
            "durable_path": str(dest),
            "_not_a_probe": binding["_not_a_thirteenth_probe"],
        }, indent=1) + "\n")
        return str(final.checkpoint_path)

    # -- the two rungs -------------------------------------------------------
    def descriptors(self) -> list[dict]:
        """C1's contract, answered for whichever rung is in flight.

        The parent's loop reads this; overriding it is what makes the inherited
        stage-G body train THIS session's probes without the loop being copied.
        """
        probes = self.confirmation if self.advanced else self.screening
        return [{**p.as_dict(), "student_path": p.initialization_path}
                for p in probes]

    def run_rung(self, rung: str, probes: list[SCH.Probe], battery: str) -> None:
        """Train and score one rung, persisting each probe as it completes."""
        for probe in probes:
            name = probe.probe_id
            if name in self.probe_scores:
                say(f"  {name}: restored from the journal")
                continue
            if not self.afford(self.a.probe_train_minutes
                               + self.a.probe_battery_minutes, name):
                raise BehaviouralDriverError(
                    f"budget refuses {name}; no probe is skipped to make "
                    "progress and a partial rung decides nothing")
            d = {**probe.as_dict(), "student_path": probe.initialization_path}
            config = self.probe_config(d)
            t0 = time.time()
            out_dir = self.train_one(name, config)
            record = {"probe_id": name, "rung": rung, "arm": probe.arm,
                      "seed": probe.seed, "out_dir": str(out_dir),
                      "train_minutes": round((time.time() - t0) / 60, 2)}
            record["preserved"] = self.preserve_probe(
                name, Path(out_dir), record)
            self.probe_training[name] = record
            score = self.score_probe(name, out_dir, battery)
            self.probe_scores[name] = score
            (AUDIT / "probes" / f"{name}.json").write_text(
                json.dumps({**record, "score": score}, indent=1) + "\n")
            self.save()
            mark(f"{PROBE_MARKER}:{name}")
            say(f"  {name}: trained and scored, correct_overall="
                f"{score['correct_overall']:.4f}")

    def score_probe(self, name: str, out_dir: Path, battery: str) -> dict:
        """Score one probe on one battery. A SEAM, like C1's `train_one`.

        Everything around it — the rung loop, the completeness gate, the
        ranking — is this driver's business and is exercised without hardware.
        """
        raise NotImplementedError(
            "score_probe is bound at session construction to C1's battery "
            "scorer; a driver that reached here has not been wired")

    def stage_s(self) -> None:
        mark("STAGE_START:S")
        proto = BH.protocol(REPO_ROOT)["behavioural_selection"]
        self.screening = SCH.screening_probes(
            self.candidates, self.anchor, proto["seeds"]["screening"])
        self.run_rung("screening", self.screening,
                      proto["batteries"]["screening"]["asset_id"])
        ok, why = SCH.screening_is_complete(
            self.screening, self.probe_training, self.probe_scores)
        if not ok:
            raise BehaviouralDriverError(why)
        self.complete("S", probes=len(self.screening), **{"gate": why})

    def stage_r(self) -> None:
        """Rank and advance exactly one. NO VERDICT LEAVES THIS STAGE."""
        mark("STAGE_START:R")
        ok, why = SCH.screening_is_complete(
            self.screening, self.probe_training, self.probe_scores)
        if not ok:
            raise BehaviouralDriverError(
                f"{why} Ranking a partial field selects on who finished first.")

        scores = {p.arm: self.probe_scores[p.probe_id]["correct_overall"]
                  for p in self.screening}
        self.ranked = SCH.rank_screening(scores, self.candidates)
        result = {"ranked": self.ranked}
        SCH.assert_screening_emits_no_verdict(result)
        self.advanced = SCH.advance_one(self.ranked)
        mark(f"{ADVANCED_MARKER}:{self.advanced['state_id']}")
        say(f"  advanced {self.advanced['state_id'][:12]}…, delta vs B "
            f"{self.advanced['delta_vs_b']:+.4f}"
            + (" (tie broken on the frozen order)"
               if self.advanced["tie_broken"] else ""))
        self.complete("R", ranked=self.ranked, advanced=self.advanced,
                      emits_verdict=False)

    def stage_c(self) -> None:
        mark("STAGE_START:C")
        if self.advanced is None:
            raise BehaviouralDriverError("no candidate has advanced")
        proto = BH.protocol(REPO_ROOT)["behavioural_selection"]
        chosen = next(c for c in self.candidates
                      if c["state_id"] == self.advanced["state_id"])
        self.confirmation = SCH.confirmation_probes(
            chosen, self.anchor, proto["seeds"]["confirmation"])
        self.run_rung("confirmation", self.confirmation,
                      proto["batteries"]["confirmation"]["asset_id"])
        self.complete("C", probes=len(self.confirmation),
                      arm=self.advanced["state_id"])

    def stage_d(self) -> None:
        """Apply C1's frozen decision rule. The ONLY stage that may name one."""
        mark("STAGE_START:D")
        deltas = []
        for seed in {p.seed for p in self.confirmation}:
            pair = {p.arm: self.probe_scores[p.probe_id]["correct_overall"]
                    for p in self.confirmation if p.seed == seed}
            deltas.append({"seed": seed,
                           "delta": round(pair[self.advanced["state_id"]]
                                          - pair[SCH.ANCHOR], 10)})
        verdict = self.apply_frozen_decision_rule(deltas)
        self.complete("D", deltas=deltas, **verdict)

    def apply_frozen_decision_rule(self, deltas: list[dict]) -> dict:
        """C1's rule, imported rather than restated. A SEAM for the same reason."""
        raise NotImplementedError(
            "apply_frozen_decision_rule is bound at session construction to "
            "C1's frozen rule; a driver that reached here has not been wired")

    # -- run -----------------------------------------------------------------
    def run(self) -> int:
        stages = (("P", self.stage_p), ("S", self.stage_s), ("R", self.stage_r),
                  ("C", self.stage_c), ("D", self.stage_d))
        for letter, fn in stages:
            say(f"stage {letter}")
            try:
                fn()
            except Exception as exc:                            # noqa: BLE001
                self.fail(letter, exc)
                mark(FAILURE_MARKER)
                return 1
        mark(SUCCESS_MARKER)
        return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--b-workdir", required=True,
                    help="where incumbent B is materialized before it is gated")
    ap.add_argument("--b-durable", default=BH.B_DURABLE_PATH)
    ap.add_argument("--probe-train-minutes", type=float, required=True)
    ap.add_argument("--probe-battery-minutes", type=float, required=True)
    ap.add_argument("--rate", type=float, required=True)
    ap.add_argument("--soft-stop-usd", type=float, required=True)
    ap.add_argument("--authorized-usd", type=float, required=True)
    ap.add_argument("--image-digest", default="")
    return ap
