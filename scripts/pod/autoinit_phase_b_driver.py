#!/usr/bin/env python3
"""Phase B: the joint P=2 calibration-distribution search, and its cross-phase rungs.

Phase B differs from Phase A in four places and inherits everything else. The
inheritance is the point: the affordability arithmetic, the CPU budget, the suite
binding, the leaf-shortfall rule, the durability boundary, the probe runner, the
battery, the selection rows, the journal and the teardown are the machinery Phase
A proved on hardware, and a Phase-B copy of them would be a second implementation
of the parts that spend money.

What differs:

1. **The governing artifacts.** A `PhaseBAuthorization` and the Phase-B session
   plan, both class attributes, so this driver cannot be run under Phase A's
   consumed grant or advance through Phase A's stages.
2. **Stage 0 additionally** binds the Phase-B executable-source digest, binds
   **both** calibration profiles by spec *and* content hash, and — once
   comparability has passed — imports the verified historical Phase-A probe
   records into this run's journal.
3. **Stage 1 searches two profiles jointly**, via the `run_search` seam.
4. **`restore_probe` accepts an imported record under a comparable protocol**,
   not only an identical one. See below; this is the difference between the
   priced run and one that silently re-buys three probes.

**Reuse is implemented as journal seeding, not as a special case.** An imported
probe becomes a file in `AUDIT/probes/`, and from that point the inherited rung
machinery treats it exactly as it treats a probe this session already ran:
`restore_probe` finds it, `run_rung` skips the budget check, and
`pooled_over_rungs` pools it. Nothing downstream needs to know which probes were
bought today. The probe ids match by construction because `probe_configs` derives
them from the **science** plan id, which Phase B reuses unchanged.

**Stage-0 comparability failure terminates.** Stage 0 is blocking, so a failure
stops the run before stage 1 — inherited behaviour, and the intended one. Phase B
does not respond to it by re-running eight candidates: the frozen feasibility
floor and equivalence interval were materialized under Phase A's runtime, so
under a non-comparable runtime they describe nothing this session could produce,
and a larger run would not restore them.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "autoinit"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aadistill.autoinit.authorization import AuthorizationError  # noqa: E402
from aadistill.autoinit.calibration import (  # noqa: E402
    DOMAIN_BALANCED_V1, REASONING_HEAVY_V2,
)
from aadistill.autoinit.phase_b import (  # noqa: E402
    CANONICAL_CONTROL, PHASE_A_EXCLUDED_LEAVES, PHASE_A_IMPORTED_FINALISTS,
    PHASE_B_PLAN_V1, PhaseBAuthorization, phase_b_source_digest,
)
#: Deliberately the SAME audit root the Phase-A driver and the recovery
#: continuation write to. It is a per-pod scratch directory that the session
#: archives wholesale; giving Phase B its own would fork the artifact policy for
#: no scientific reason, and every inherited method writes there.
import autoinit_phase_a_driver as _phase_a  # noqa: E402
from autoinit_phase_a_driver import (  # noqa: E402
    AUDIT, PhaseADriver, mark, say,
)

WS = Path("/workspace")
STATUS = WS / "autoinit_phase_b.status"

#: Phase B searches into its OWN workdir, not Phase A's — the two sessions retain
#: different journals and a shared path would let one overwrite the other's.
#:
#: It is a module constant rather than a literal inside `run_search` because the
#: artifact specs that COLLECT this journal live in different files, and attempt 3
#: proved they can disagree silently: the specs named `phase_a_search` while the
#: driver wrote `phase_b_search`, the collector matched nothing, `min_matches: 0`
#: reported `missing: 0`, and the search journal was deleted with the pod at the
#: one moment it mattered — a deadline failure with no per-state timings.
#: `tests/pod/test_phase_b_artifact_paths.py` now holds writer and both
#: collectors to this constant.
SEARCH_WORKDIR = REPO / "artifacts/autoinit/phase_b_search"

#: `mark()` is a module function that appends to the module global `STATUS`, and
#: every inherited stage calls it. Without this line a Phase-B run would write
#: its markers to `autoinit_phase_a.status` while `autoinit_phase_b_launch.py`
#: polls `autoinit_phase_b.status` — the launcher would see an empty file for the
#: whole session, conclude nothing was happening, and kill a working run.
#:
#: Reassigned rather than parameterized because the alternative is threading a
#: path through a dozen inherited call sites in a driver Phase A has already
#: proved on hardware. A test asserts the redirection took effect, and mutating
#: it away fails that test.
_phase_a.STATUS = STATUS

#: The verified reuse record, produced at `$0` by
#: `scripts/autoinit/verify_historical_probe_reuse.py` and re-checked here.
REUSE_RECORD = REPO / "logs/autoinit_historical_probe_reuse.json"
#: Where the historical probe records themselves live, in the committed tree.
HISTORICAL_PROBES = REPO / "logs/autoinit_recovery_continuation_attempt7/probes"
#: The Phase-A durability record: the canonical ids and the digests their bytes
#: must reproduce.
LEAF_RETENTION = REPO / "logs/autoinit_recovery_continuation_attempt7/leaf_retention.json"
#: Where the launcher stages the two retained finalists on the pod. Read-only
#: inputs: they are measured on the state-evaluation suite and never trained.
STAGED_FINALISTS = REPO / "artifacts/autoinit/phase_a_selected"


class PhaseBDriver(PhaseADriver):
    AUTHORIZATION_TYPE = PhaseBAuthorization
    AUTHORIZATION_PATH = "logs/autoinit_phase_b_authorization.json"
    PLAN = PHASE_B_PLAN_V1

    def __init__(self, a):
        #: NOT `super().__init__(a)`. The parent binds `PHASE_A_PLAN_V1` and
        #: stamps the Phase-A evidence schema, and a Phase-B run must be
        #: governed by neither. Written out rather than patched afterwards so
        #: the Phase-B contract is visible at the point it is established; a
        #: test asserts this constructor leaves no attribute the inherited
        #: methods rely on unset.
        import time

        self.a = a
        self.t0 = time.time()
        self.results: dict[int, dict] = {}
        self.evaluation_protocol = None
        self.plan = None                 # the frozen SuccessiveHalvingPlan
        self.search_result = None
        self.leaves: list = []
        self.control_state = None
        self.rung1 = None
        self.rung2 = None
        self.imported_probe_ids: set[str] = set()
        self.imported_finalists: list = []
        self.comparability = None
        self.ev: dict = {
            "schema": "aadistill.autoinit.phase_b_evidence/v1",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "phase": "B",
            "hypothesis": ("does the preferred composition change when the "
                           "calibration distribution changes"),
            "profiles": [DOMAIN_BALANCED_V1.qualified_id,
                         REASONING_HEAVY_V2.qualified_id],
            "retrains_permanent_controls": False,
            "redefines_thresholds": False,
            "followon_started": False,
            "followon_reachable_from_this_driver": False,
            "excluded_phase_a_leaves": dict(sorted(PHASE_A_EXCLUDED_LEAVES.items())),
            "stages": {}}
        AUDIT.mkdir(parents=True, exist_ok=True)
        (AUDIT / "probes").mkdir(parents=True, exist_ok=True)
        self.auth = self.AUTHORIZATION_TYPE.load(REPO / self.AUTHORIZATION_PATH)
        self.auth.require_plan(self.PLAN.plan_hash)
        self.ev["authorization"] = self.auth.as_dict()

    def enter(self, stage: int) -> None:
        """The Phase-B plan decides stage ordering, not Phase A's."""
        self.auth.require_stage(stage)
        self.PLAN.advance_to(stage, self.results)
        mark(f"STAGE_START:{stage}")

    # -- stage 0 additions -------------------------------------------------
    def stage0(self) -> bool:
        """Phase A's stage 0, then what Phase B additionally binds.

        Order matters. The inherited stage runs the comparability gate; the
        imports below are only legitimate once it has passed, because a probe
        measured under a non-comparable protocol is not evidence about this
        session's candidates.
        """
        if not super().stage0():
            #: Inherited failure, including comparability. Restated here so the
            #: evidence says what Phase B does about it rather than leaving a
            #: reader to infer it from a Phase-A message.
            self.ev["stage0_failure_policy"] = (
                "TERMINATE. Phase B does not answer a comparability failure by "
                "re-running all eight candidates: the frozen feasibility floor "
                "and equivalence interval were materialized under Phase A's "
                "runtime and would not describe anything this session could "
                "produce. The Stage-3 controls are not rematerialized and the "
                "thresholds are not redefined.")
            self.save()
            return False

        try:
            source = self.auth.require_source(REPO)
            for profile in (DOMAIN_BALANCED_V1, REASONING_HEAVY_V2):
                self.auth.require_calibration(profile)
        except AuthorizationError as exc:
            return self.record(0, False, str(exc)[-1500:])

        try:
            imported = self.import_historical_probes()
        except Exception as exc:                                  # noqa: BLE001
            return self.record(0, False,
                               f"historical probe import: {type(exc).__name__}: "
                               f"{exc}"[-1500:],
                               **self.preserve_traceback(0, exc))

        binding = {
            "executable_source_digest": source["digest"],
            "calibration": {
                p.qualified_id: {"profile_hash": p.profile_hash,
                                 "content_sha256": p.content_sha256,
                                 "items_file_sha256": p.items_file_sha256}
                for p in (DOMAIN_BALANCED_V1, REASONING_HEAVY_V2)},
            "imported_probes": imported,
        }
        (AUDIT / "phase_b_stage0_binding.json").write_text(
            json.dumps(binding, indent=2, default=str) + "\n")
        self.ev["phase_b_binding"] = binding
        say(f"phase B: source {source['digest'][:12]}…, both mixtures bound, "
            f"{len(imported['probe_ids'])} historical probes imported")
        # The inherited stage already recorded a pass; enrich it rather than
        # recording stage 0 twice, which would overwrite its attestation.
        self.ev["stages"]["0"].update(phase_b_binding=binding)
        self.save()
        return True

    def import_historical_probes(self) -> dict:
        """Seed the journal with the probes whose reconstruction was verified.

        Refuses to import anything the `$0` record does not list as *admitted and
        reusable*, and re-checks the record against the probe bytes on the pod —
        a verification performed on the dev box says nothing about what was
        staged here.
        """
        record = json.loads(REUSE_RECORD.read_text())
        if not record.get("reuse_verified"):
            raise RuntimeError(
                "the historical probe reuse record is not verified; Phase B may "
                "not cite evidence whose reconstruction was never proved")

        admitted = set(record.get("admitted_reusable_probes") or ())
        if not admitted:
            raise RuntimeError("the reuse record admits no probe")

        wanted = {*PHASE_A_IMPORTED_FINALISTS, CANONICAL_CONTROL}
        copied, skipped = [], []
        for path in sorted(HISTORICAL_PROBES.glob("*.json")):
            probe = json.loads(path.read_text())
            probe_id = probe.get("probe_id", "")
            candidate = probe_id.split(".")[-2] if "." in probe_id else ""
            seed_name = probe_id.rsplit(".", 1)[-1]
            key = f"{candidate}/{seed_name}"
            if candidate not in wanted:
                #: The three excluded Phase-A leaves. Their sa probes are
                #: verified and cost nothing, and they are STILL not imported:
                #: zero marginal cost is not an admission criterion.
                skipped.append({"probe_id": probe_id, "reason":
                                "candidate is not in the Phase-B candidate set"})
                continue
            if key not in admitted:
                skipped.append({"probe_id": probe_id,
                                "reason": "not admitted-and-reusable in the record"})
                continue
            shutil.copyfile(path, AUDIT / "probes" / f"{probe_id}.json")
            copied.append(probe_id)
            self.imported_probe_ids.add(probe_id)

        if not copied:
            raise RuntimeError("no historical probe was imported; the priced run "
                               "assumes three, so this is a defect not a saving")
        return {"probe_ids": sorted(copied), "skipped": skipped,
                "probes_dir_digest": record.get("probes_dir_digest"),
                "source": str(HISTORICAL_PROBES.relative_to(REPO))}

    # -- the cross-phase candidate universe --------------------------------
    def candidate_universe(self) -> list:
        """Eight: the Phase-B Top-5, the two retained Phase-A finalists, the control.

        The inherited universe is `leaves + control` — six here — and journal
        seeding alone does not fix that: a probe record is only consulted if a
        DESCRIPTOR is generated for its candidate, and descriptors come from this
        list. Without the finalists the run would seed six citations, use three,
        and quietly compare a different set from the one the preregistration
        froze.

        `self.leaves` still means exactly the Phase-B Top-5. The imported states
        are kept separate so nothing blurs what the P=2 search produced.
        """
        return [*self.leaves, *self.imported_finalists, self.control_state]

    def require_citable(self, rung_seed_names: tuple[str, ...]) -> None:
        """Every imported finalist must already have the evidence it will be asked
        for. Fail closed rather than fall through into training bytes that were
        never staged for training."""
        record = json.loads(REUSE_RECORD.read_text())
        admitted = set(record.get("admitted_reusable_probes") or ())
        missing = [f"{c}/{s}" for c in PHASE_A_IMPORTED_FINALISTS
                   for s in rung_seed_names if f"{c}/{s}" not in admitted]
        if missing:
            raise RuntimeError(
                f"imported Phase-A finalists lack verified evidence for {missing}. "
                "Phase B cites their behaviour; it does not retrain them, and "
                "their checkpoints are staged read-only for measurement. "
                "Refusing rather than falling through into a probe.")

    # -- stage 1: the joint P=2 search -------------------------------------
    def run_search(self, run_phase_a_search):
        """Both mixtures, searched jointly, in one beam.

        The loader dispatches on the profile it is asked about. A loader that
        ignored its argument — the historical shape everywhere in this project —
        would label half the states with one mixture and feed them another, and
        the engine additionally asks it about the `calib.none` sentinel.
        """
        from phase_a_search import as_operator_items

        items = {p.qualified_id: as_operator_items(p.resolve(REPO))
                 for p in (DOMAIN_BALANCED_V1, REASONING_HEAVY_V2)}
        # Refuse before the search rather than after it: if the citations these
        # candidates depend on are not verified, no amount of searching helps.
        self.require_citable(("sa", "sb"))
        say(f"phase B search: P=2 over {sorted(items)}; "
            f"{len(PHASE_A_IMPORTED_FINALISTS)} retained finalists injected")
        found = run_phase_a_search(
            workdir=SEARCH_WORKDIR,
            state_eval=REPO / "artifacts/stage1/state_eval_v1",
            top_n=self.plan.searched_leaves, device="cuda", repo_root=REPO,
            profiles=(DOMAIN_BALANCED_V1, REASONING_HEAVY_V2),
            calibration_items=items,
            retained_candidates=self.retained_candidate_specs(),
            search_minutes=self.a.search_deadline_minutes)
        # Published here rather than in the inherited stage 1, which knows only
        # about leaves and the control. Kept OFF `self.leaves`: those are the
        # Phase-B Top-5 and nothing else.
        self.imported_finalists = list(found.imported)
        if len(self.imported_finalists) != len(PHASE_A_IMPORTED_FINALISTS):
            raise RuntimeError(
                f"expected {len(PHASE_A_IMPORTED_FINALISTS)} retained finalists, "
                f"measured {len(self.imported_finalists)}; the cross-phase "
                "comparison would be missing a preregistered candidate")
        say(f"retained finalists measured: "
            f"{[s.state_id[:12] for s in self.imported_finalists]}")
        return found

    def retained_candidate_specs(self) -> list[dict]:
        """The two Phase-A finalists, by canonical id and recorded digest.

        The id is the ORIGINAL one, because `probe_configs` derives probe ids
        from `state_id[:12]` and a renamed candidate would stop matching the
        historical records that are its entire evidence.
        """
        retention = json.loads(LEAF_RETENTION.read_text())
        by_id = {e["canonical_id"]: e for e in retention["entries"]}
        specs = []
        for candidate in PHASE_A_IMPORTED_FINALISTS:
            entry = next((e for cid, e in by_id.items()
                          if cid.startswith(candidate)), None)
            if entry is None:
                raise RuntimeError(
                    f"{candidate} is not in the Phase-A leaf retention record; "
                    "its identity cannot be established and it must not be "
                    "compared on an assumed digest")
            specs.append({
                "candidate_id": entry["canonical_id"],
                "checkpoint_dir": str(STAGED_FINALISTS / entry["canonical_id"]),
                "expected_artifact_digest": entry["artifact_digest"],
                "provenance": "retained_phase_a_finalist",
                "description": ("a retained Phase-A finalist, cited rather than "
                                "re-searched; its behaviour is imported evidence"),
            })
        return specs

    # -- citing an imported probe ------------------------------------------
    def restore_probe(self, descriptor: dict) -> dict | None:
        """As inherited, except that an IMPORTED record binds by comparability.

        The inherited check requires the journalled record's
        `evaluation_protocol_hash` to equal this run's exactly. For a probe this
        session ran, that is right. For one imported from Phase A it is the wrong
        predicate and would silently re-buy it: Phase A's own record notes its
        raw protocol hash differs from the Stage-3 one by driver patch alone,
        which `generation_runtime_comparability@v2` declares non-material — and
        that same relation is what stage 0 has already established for this run.

        Everything else stays strict. A different student or a different seed is
        a different probe whatever the protocol says.
        """
        probe_id = descriptor["probe_id"]
        if probe_id not in self.imported_probe_ids:
            return super().restore_probe(descriptor)

        path = AUDIT / "probes" / f"{probe_id}.json"
        if not path.is_file():
            return None
        record = json.loads(path.read_text())
        for field in ("student_artifact_digest", "seed"):
            if record.get(field) != descriptor[field]:
                say(f"  {probe_id}: imported record does not bind to this run "
                    f"({field} moved) — refusing to cite it")
                return None
        if not record.get("complete"):
            return None
        say(f"  {probe_id}: cited from verified Phase-A evidence "
            f"(protocol {str(record.get('evaluation_protocol_hash'))[:12]}…, "
            "comparable under v2)")
        record["resumed"] = True
        record["imported_from_phase_a"] = True
        return record


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    #: There is no stage 6. Phase B is a terminus, like Phase A.
    ap.add_argument("--stage", default="all", choices=("all",))
    ap.add_argument("--image-digest", required=True)
    ap.add_argument("--rate", type=float, required=True)
    ap.add_argument("--spent-usd", type=float, default=0.0)
    ap.add_argument("--soft-stop-usd", type=float, required=True)
    ap.add_argument("--authorized-usd", type=float, required=True)
    ap.add_argument("--search-minutes", type=float, required=True)
    ap.add_argument("--search-deadline-minutes", type=float, required=True)
    ap.add_argument("--probe-train-minutes", type=float, default=62.0)
    ap.add_argument("--probe-battery-minutes", type=float, default=10.0)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    driver = PhaseBDriver(args)
    driver.auth.require_within_cap(args.authorized_usd, what="session backstop")
    return driver.run()


if __name__ == "__main__":
    raise SystemExit(main())
