#!/usr/bin/env python3
"""Finish Phase B's behavioural selection. The P=2 search is NOT reachable here.

Attempt 5 completed Stage 0 and the joint P=2 search, emitted an authoritative
Top-5 and a durable Stage-1 selection, and ran three rung-1 `sa` probes. Rung 1
was then completed at `$0` under the identity-collapse amendment. What remains is
**one missing `sb`** and **at most two conditional `sc`**.

So this driver imports the completed state instead of recomputing it:

    stage 0  attestation, comparability, and every cited identity re-checked
    stage 1  import the completed behavioural state — NO search, NO new sa
    stage 2  rung 2 on sb: buy only what does not exist
    stage 3  the frozen pooled sa+sb decision, unchanged
    stage 4  conditional sc, only where no verified sc exists
    stage 5  the frozen final selection and report

**Why this is a separate driver and not a `--skip-search` flag.** A flag leaves a
16.5 h purchase one mistake away, and this project has paid four times for code
that only one path reaches. Here the search is unreachable by construction: the
inherited `stage1` is overridden to refuse, this driver's stage map contains no
search stage, and `phase_a_search` is imported by the parent only *inside* the
stage-1 body that never runs. A test asserts the import closure never reaches it.

Everything scientific is inherited unchanged. `stage3`, `stage4` and `stage5` are
the Phase-A implementations, so the pooled `sa+sb` equivalence rule, the
conditional tie-break and the final selection are the frozen ones — including
`winner=None / unresolved_equivalence` as a valid terminal result.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
#: `phase_a_frozen` lives here. It is a SEPARATE module from
#: `phase_a_search` — importing it reaches no search — and it is where the
#: frozen target geometry and teacher identity live.
sys.path.insert(0, str(REPO / "scripts/autoinit"))

from aadistill.autoinit.authorization import AuthorizationError  # noqa: E402
from aadistill.autoinit.identity_collapse import (  # noqa: E402
    IdentityCollapseError, collapse, universe_identity,
)
from aadistill.autoinit.phase_b_continuation import (  # noqa: E402
    CONTINUATION_PLAN_V1, ContinuationAuthorization, continuation_source_digest,
)
from aadistill.autoinit.recovery import RecoveryAdmissionError  # noqa: E402
from aadistill.autoinit.state import make_control_state, make_retained_state  # noqa: E402

import autoinit_phase_a_driver as _phase_a  # noqa: E402
from autoinit_phase_a_driver import (  # noqa: E402
    AUDIT, PhaseADriver, WS, mark, say,
)

STATUS = WS / "autoinit_continuation_b.status"


def bind_status_file() -> None:
    """Point the inherited `mark()` at the file THIS launcher polls.

    Phase B does the same rebinding, but at import time. Two modules writing the
    same shared global means whichever is imported last wins — harmless on a pod,
    where exactly one driver is ever imported, and wrong everywhere else. Doing it
    at import here made the Phase-B driver's own marker test fail as soon as this
    module was imported first in the same process, which is a fair warning about
    hidden global state rather than a test problem.

    So it happens on entry instead, where "this process is the continuation" is
    actually true.
    """
    _phase_a.STATUS = STATUS

#: The completed evidence this session stands on. Every one is re-checked at
#: stage 0 against the authorization's bound hashes.
STAGE1_SELECTION = REPO / "logs/autoinit_phase_b_attempt5/stage1_selection.json"
AMENDMENT = REPO / "logs/autoinit_phase_b_identity_collapse_amendment.json"
HISTORICAL_REUSE = REPO / "logs/autoinit_historical_probe_reuse.json"
ATTEMPT5_REUSE = REPO / "logs/autoinit_attempt5_probe_reuse.json"
HISTORICAL_PROBES = REPO / "logs/autoinit_recovery_continuation_attempt7/probes"
ATTEMPT5_PROBES = REPO / "logs/autoinit_phase_b_attempt5/probes"

CANONICAL_CONTROL = "control-qwen"
#: The canonical control's own id. Its collapsed state id is the truncated
#: `control-qwen`; the state this driver builds is `control-<CONTROL_ID>`.
CONTROL_ID = "qwen3_0p6b_init_v0"

#: The two numbers that keep evidence and workload apart, asserted rather than
#: assumed. SIX distinct behavioural candidates carry the completed `sa`
#: evidence and the identity-collapse result. THREE of them — the two frozen
#: rung-1 survivors plus the auto-advancing control — are all that may enter
#: `sb`, the pooled decision, `sc` and the final selection.
#:
#: Rung 1 is COMPLETE. This session does not recompute it, and the three searched
#: non-survivors must not reach a probe.
EXPECTED_EVIDENCE_UNIVERSE = 6
EXPECTED_ACTIVE_FINALISTS = 3


class ContinuationDriver(PhaseADriver):
    """Phase A's probe machinery, without Phase A's search."""

    #: Attempt 2 died at stage 0 for `$0.3146` because these were not set.
    #:
    #: `PhaseADriver` declares them as a seam precisely so a subclass can name
    #: its own grant, and its comment there names THIS subclass as the reason.
    #: Left unset, the driver loaded `logs/autoinit_phase_a_authorization.json`
    #: — a real, committed Phase-A grant — and `stage_bind` then called
    #: `require_evidence`, which only `ContinuationAuthorization` has. Every
    #: other `PhaseADriver` subclass sets both; this one did not.
    AUTHORIZATION_TYPE = ContinuationAuthorization
    AUTHORIZATION_PATH = "logs/autoinit_continuation_b_authorization.json"
    PLAN = CONTINUATION_PLAN_V1

    def __init__(self, args) -> None:
        #: NOT `super().__init__(args)`. The parent binds three things this
        #: session must not inherit: the Phase-A authorization type and path,
        #: `PHASE_A_PLAN_V1`, and the Phase-A evidence schema and scope. Setting
        #: the two class attributes above and still calling the parent would only
        #: move attempt 2's failure one line later, to the parent's closing
        #: `require_plan(PHASE_A_PLAN_V1.plan_hash)` — a `ContinuationAuthorization`
        #: refuses it, since it binds `a2ef4cd68a4b` and that check asks for
        #: `9377a2dc61f2`.
        #:
        #: Written out rather than patched afterwards, the same shape
        #: `PhaseBDriver` already uses, so the continuation's contract is visible
        #: at the point it is established. A test asserts this constructor leaves
        #: no attribute the inherited stages rely on unset, and that it produces
        #: a real `ContinuationAuthorization` BEFORE any test supplies one.
        import time
        from datetime import datetime, timezone

        self.a = args
        self.t0 = time.time()
        self.results: dict[int, dict] = {}
        self.evaluation_protocol = None
        self.plan = None                 # the frozen SuccessiveHalvingPlan
        #: Inherited stage 3/4/5 read these. The search never runs here, so
        #: `search_result` and `leaves` stay empty rather than being populated by
        #: a stage-1 this driver refuses.
        self.search_result = None
        self.leaves: list = []
        self.control_state = None
        self.rung1 = None
        self.rung2 = None
        self.plan_spec = CONTINUATION_PLAN_V1
        #: SIX distinct candidates: evidence and provenance.
        self.evidence_universe: list = []
        #: THREE active finalists: the only states this session may probe.
        self.finalists: list = []
        self.imported_probe_ids: set[str] = set()
        self.evidence_observed: dict[str, str] = {}
        #: The evidence envelope describes THIS session. The artifact pathname is
        #: unchanged — the collection contract names it and moving it would widen
        #: this repair — but what the file claims must be the continuation's, not
        #: a Phase-A schema and scope inherited along with the writer.
        self.ev: dict = {
            "schema": "aadistill.autoinit.continuation_b_evidence/v1",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "phase": "B-continuation",
            "question": ("which of the three active finalists survives the "
                         "frozen pooled sa+sb rule, and the conditional sc"),
            "runs_search": False,
            "stage1_imported_not_recomputed": True,
            "retrains_permanent_controls": False,
            "redefines_thresholds": False,
            "followon_started": False,
            "followon_reachable_from_this_driver": False,
            "stages": {}}
        AUDIT.mkdir(parents=True, exist_ok=True)
        (AUDIT / "probes").mkdir(parents=True, exist_ok=True)
        self.auth = self.AUTHORIZATION_TYPE.load(REPO / self.AUTHORIZATION_PATH)
        self.auth.require_plan(self.PLAN.plan_hash)
        self.ev["authorization"] = self.auth.as_dict()

    def enter(self, stage: int) -> None:
        """The CONTINUATION plan decides stage ordering, not Phase A's.

        The inherited `enter` advances `PHASE_A_PLAN_V1`, whose stage 1 is the
        search. Ordering this session against it would check the wrong
        preconditions at every stage.
        """
        self.auth.require_stage(stage)
        CONTINUATION_PLAN_V1.advance_to(stage, self.results)
        mark(f"STAGE_START:{stage}")

    # -- stage 1 is NOT a search here -------------------------------------
    def stage1(self) -> bool:
        raise RecoveryAdmissionError(
            "the behavioural continuation has no search stage. Phase-B Stage 1 is "
            "complete, retained and authoritative; buying it again is a different "
            "session with a different ceiling. This method exists only to make "
            "that refusal explicit if anything ever calls it.")

    def run_search(self, *a, **k):
        raise RecoveryAdmissionError(
            "the behavioural continuation may not run a search")

    # -- evidence ----------------------------------------------------------
    @staticmethod
    def observed_evidence() -> dict[str, str]:
        """Re-derive every bound identity from the artifacts on this machine.

        Static because it reads only module constants, and because the launcher's
        `$0` evidence gate calls it before any driver exists. That gate used to
        pass a `SessionContext` as `self` with a `type: ignore`, which worked only
        for as long as nobody added an attribute access here.
        """
        selection = json.loads(STAGE1_SELECTION.read_text())
        amendment = json.loads(AMENDMENT.read_text())
        historical = json.loads(HISTORICAL_REUSE.read_text())
        attempt5 = json.loads(ATTEMPT5_REUSE.read_text())
        rung1 = amendment["rung1_selection"]
        from aadistill.infrastructure.manifest import sha256_json

        return {
            "stage1_selection_sha256": selection["selection_sha256"],
            "identity_collapse_amendment_sha256": amendment["amendment_sha256"],
            "collapsed_universe_identity":
                amendment["collapsed_universe"]["universe_identity"],
            "historical_reuse_probes_dir_digest": historical["probes_dir_digest"],
            "attempt5_reuse_probes_dir_digest": attempt5["probes_dir_digest"],
            "rung1_selection_digest": sha256_json({
                "selected_searched": rung1["selected_searched"],
                "auto_advanced_control": rung1["auto_advanced_control"],
                "advancing": rung1["advancing"]}),
        }

    def build_evidence_universe(self) -> list:
        """The SIX distinct behavioural candidates, rebuilt from the amendment.

        Evidence, not workload. This layer proves that identity collapse still
        produces the universe the grant was issued against, and that the `sa`
        record covering it is complete. It reads no checkpoint bytes, because
        five-sixths of it is never probed and requiring its bytes would put ~3.6
        GiB of checkpoints on a pod to look at their filenames.
        """
        amendment = json.loads(AMENDMENT.read_text())
        recorded = amendment["collapsed_universe"]
        entries = [{"state_id": c["state_id"], "artifact_digest": c["artifact_digest"],
                    "role": role, "checkpoint_path": c.get("checkpoint_path")}
                   for c in recorded["candidates"] for role in c["roles"]]
        collapsed = collapse(entries)
        if universe_identity(collapsed) != recorded["universe_identity"]:
            raise IdentityCollapseError(
                "the collapsed universe rebuilt to a different identity than the "
                "amendment records; refusing to continue against a universe that "
                "is not the one authorized")
        if len(collapsed) != EXPECTED_EVIDENCE_UNIVERSE:
            raise IdentityCollapseError(
                f"the evidence universe is {len(collapsed)} candidates, not "
                f"{EXPECTED_EVIDENCE_UNIVERSE}; the amendment this session cites "
                "describes a different experiment")
        return collapsed

    def build_finalist_states(self) -> list:
        """The THREE active finalists, rebuilt from retained bytes.

        Rung 1 is complete and frozen. Only its survivors and the auto-advancing
        control may enter `sb`, the pooled decision, `sc` and the final selection,
        so only their bytes are staged and re-identified here. Rebuilding the
        other three would not merely waste transfer: it would put searched
        non-survivors one filter away from a probe this session must not buy.
        """
        amendment = json.loads(AMENDMENT.read_text())
        advancing = set(amendment["rung1_selection"]["advancing"])
        # The frozen geometry and teacher identity come from `phase_a_frozen`,
        # which is deliberately a separate module from `phase_a_search`: importing
        # it reaches no search. The teacher hash comes from the state-eval
        # manifest, the same place the search read it.
        from transformers import AutoConfig

        import phase_a_frozen
        from aadistill.autoinit.arch import ArchSpec, get_adapter
        from aadistill.autoinit.artifact import identify_checkpoint

        adapter = get_adapter("qwen3")
        target = ArchSpec.of("qwen3", phase_a_frozen.TARGET_GEOMETRY)
        manifest = json.loads(
            (REPO / "artifacts/stage1/state_eval_v1/manifest.json").read_text())
        teacher_sha = manifest.get("teacher_sha256") or "0" * 64

        states = []
        for c in self.evidence_universe:
            control = c.primary_role == "control"
            # The control's collapsed id is truncated (`control-qwen`); its state
            # id is the full `control-<control_id>`. Compare on what `advancing`
            # actually holds, under both spellings, rather than assuming one.
            full_control_id = f"control-{CONTROL_ID}"
            if not (c.state_id in advancing
                    or (control and full_control_id in advancing)):
                continue
            directory = Path(c.checkpoint_path)
            if not directory.is_dir():
                raise RecoveryAdmissionError(
                    f"{c.state_id} is an ADVANCING finalist but is not staged at "
                    f"{directory}; the continuation probes it and cannot rebuild "
                    "it from nothing")
            spec = adapter.spec_from_config(AutoConfig.from_pretrained(str(directory)))
            n_params = adapter.param_count(spec)
            artifact = identify_checkpoint(directory, adapter=adapter, spec=spec,
                                           num_parameters=n_params)
            if artifact.artifact_digest != c.artifact_digest:
                raise IdentityCollapseError(
                    f"{c.state_id} re-derives to {artifact.artifact_digest[:12]} "
                    f"but the amendment records {c.artifact_digest[:12]}")
            if control:
                states.append(make_control_state(
                    control_id=CONTROL_ID, artifact=artifact, spec=spec,
                    target_spec=target, num_parameters=n_params,
                    root_teacher_id=phase_a_frozen.TEACHER_ID,
                    root_teacher_sha256=teacher_sha,
                    description="the retained canonical initialization",
                    expected_single_file_sha256=None))
            else:
                states.append(make_retained_state(
                    state_id=c.state_id, artifact=artifact, spec=spec,
                    target_spec=target, num_parameters=n_params,
                    root_teacher_id=phase_a_frozen.TEACHER_ID,
                    root_teacher_sha256=teacher_sha,
                    description=f"Phase-B searched leaf; roles {','.join(c.roles)}",
                    provenance="retained_imported",
                    expected_artifact_digest=c.artifact_digest))
        return states

    def candidate_universe(self) -> list:
        """What the probe stages may touch: the THREE active finalists.

        The inherited `stage3`/`stage4` filter this by `rung1["advancing"]` as
        well, so the boundary is enforced twice. That redundancy is deliberate —
        this method is the one a future edit is most likely to widen.
        """
        return list(self.finalists)

    def import_completed_probes(self) -> dict:
        """Seed the journal with every strictly reconstructed observation.

        Both records, and only what each ADMITS: the historical Phase-A citations
        and the three `sa` probes Attempt 5 paid for. A probe that did not pass
        strict reconstruction is not seeded, so the rung buys it instead of
        citing something unproved.
        """
        historical = json.loads(HISTORICAL_REUSE.read_text())
        attempt5 = json.loads(ATTEMPT5_REUSE.read_text())
        if not historical.get("reuse_verified"):
            raise RecoveryAdmissionError(
                "the historical reuse record is unverified; the continuation may "
                "not cite evidence whose reconstruction was never proved")
        if not attempt5.get("reuse_verified"):
            raise RecoveryAdmissionError(
                "the Attempt-5 reuse record is unverified; those three sa probes "
                "would have to be re-bought, which is a larger session")

        admitted = set(historical.get("admitted_reusable_probes") or ())
        admitted |= set(attempt5.get("reusable_probes") or ())
        known = {c.state_id[:12] for c in
                 collapse([{"state_id": c["state_id"],
                            "artifact_digest": c["artifact_digest"], "role": r}
                           for c in json.loads(AMENDMENT.read_text())
                           ["collapsed_universe"]["candidates"] for r in c["roles"]])}
        known.add(CANONICAL_CONTROL)

        dest = AUDIT / "probes"
        dest.mkdir(parents=True, exist_ok=True)
        copied, skipped = [], []
        for source in (HISTORICAL_PROBES, ATTEMPT5_PROBES):
            for path in sorted(source.glob("*.json")):
                probe = json.loads(path.read_text())
                probe_id = probe.get("probe_id", "")
                parts = probe_id.split(".")
                candidate = parts[-2] if len(parts) > 1 else ""
                seed_name = parts[-1]
                key = f"{candidate}/{seed_name}"
                if key not in admitted or candidate not in known:
                    skipped.append({"probe_id": probe_id,
                                    "reason": ("not admitted by a verified reuse "
                                               "record, or outside the collapsed "
                                               "universe")})
                    continue
                target = dest / f"{probe_id}.json"
                if target.is_file():
                    # One observation per (initialization, seed): the collapsed
                    # candidates are cited once, not once per role.
                    skipped.append({"probe_id": probe_id,
                                    "reason": "already seeded; one observation per seed"})
                    continue
                target.write_text(path.read_text())
                self.imported_probe_ids.add(probe_id)
                copied.append(probe_id)
        return {"imported": sorted(copied), "skipped": skipped,
                "n_imported": len(copied)}

    # -- citing an imported probe -------------------------------------------
    def restore_probe(self, descriptor: dict) -> dict | None:
        """As inherited, except that an IMPORTED record binds by comparability.

        The inherited check requires the journalled record's
        `evaluation_protocol_hash` to equal this run's exactly. For a probe this
        session ran, that is right. For an imported one it is the wrong predicate
        and would silently re-buy it — and here that is the whole economics: the
        eight citable probes carry TWO raw protocol hashes (`7327e880…` from the
        Phase-A continuation, `250f72ef…` from Attempt 5) that differ by host
        driver patch alone. `generation_runtime_comparability@v2` declares the
        driver patch non-material and the raw hash itself non-material; all of
        them share the comparable identity `70a26e0b…`, which stage 0 has already
        re-established for this run.

        Requiring exact equality would re-buy all eight probes, at roughly nine
        times the priced ceiling, while reporting success.

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
        say(f"  {probe_id}: cited from verified evidence "
            f"(protocol {str(record.get('evaluation_protocol_hash'))[:12]}…, "
            "comparable under v2)")
        record["resumed"] = True
        record["imported_evidence"] = True
        return record

    # -- stages -------------------------------------------------------------
    def stage_bind(self) -> bool:
        """Stage 0: the inherited attestation, THEN the cited evidence.

        Order matters, and it is the same order Phase B uses. The inherited
        stage runs the frozen-asset gate, binds the preregistration, materializes
        both thresholds, probes the engine and establishes
        `generation_runtime_comparability@v2` — and it is what sets `self.plan`
        and `self.evaluation_protocol`, which stages 3, 4 and 5 read. Citing
        completed observations is only legitimate once comparability has passed,
        because a probe measured under a non-comparable protocol is not evidence
        about this session's candidates.
        """
        if not super().stage0():
            self.ev["stage0_failure_policy"] = (
                "TERMINATE. The continuation does not answer a comparability "
                "failure by re-running the candidates: every observation it "
                "cites — the eight reused probes AND the frozen rung-1 result — "
                "was materialized under a comparable runtime, and re-buying them "
                "is a larger session than this grant prices.")
            self.save()
            return False

        observed = self.observed_evidence()
        self.evidence_observed = observed
        try:
            self.auth.require_evidence(observed)
            self.auth.require_plan(CONTINUATION_PLAN_V1.plan_hash)
            source = self.auth.require_source(REPO)
        except AuthorizationError as exc:
            return self.record(0, False, str(exc)[-1500:])

        binding = {"executable_source_digest": source["digest"],
                   "cited_evidence": observed}
        (AUDIT / "continuation_b_stage0_binding.json").write_text(
            json.dumps(binding, indent=2, default=str) + "\n")
        self.ev["continuation_binding"] = binding
        say(f"continuation evidence bound: universe "
            f"{observed['collapsed_universe_identity'][:12]}…, stage-1 selection "
            f"{observed['stage1_selection_sha256'][:12]}…")
        # The inherited stage already recorded a pass; enrich it rather than
        # recording stage 0 twice, which would overwrite its attestation.
        self.ev["stages"]["0"].update(continuation_binding=binding)
        self.save()
        return True

    def stage_import(self) -> bool:
        """Stage 1: import the completed behavioural state. No search.

        Where six becomes three. The evidence universe is rebuilt and checked
        whole; the workload is then narrowed to the frozen rung-1 result, which
        this session imports rather than recomputes.
        """
        self.enter(1)
        try:
            self.evidence_universe = self.build_evidence_universe()
        except IdentityCollapseError as exc:
            return self.record(1, False, str(exc)[-1500:])
        seeded = self.import_completed_probes()

        amendment = json.loads(AMENDMENT.read_text())
        self.rung1 = amendment["rung1_selection"]
        expected = set(self.rung1["advancing"])

        try:
            self.finalists = self.build_finalist_states()
        except (IdentityCollapseError, RecoveryAdmissionError) as exc:
            return self.record(1, False, str(exc)[-1500:])

        present = {s.state_id for s in self.finalists}
        missing = sorted(expected - present)
        if missing:
            return self.record(1, False,
                               f"the frozen rung-1 survivors {missing} were not "
                               "rebuilt from retained bytes")
        if len(self.finalists) != EXPECTED_ACTIVE_FINALISTS:
            return self.record(1, False,
                               f"{len(self.finalists)} finalists entered the probe "
                               f"stages, not {EXPECTED_ACTIVE_FINALISTS}; rung 1 is "
                               "complete and frozen and is not recomputed here")
        excluded = sorted({c.state_id for c in self.evidence_universe}
                          - {CANONICAL_CONTROL} - present)
        say(f"imported {seeded['n_imported']} completed observations; "
            f"{len(self.evidence_universe)} distinct candidates as evidence; "
            f"{len(self.finalists)} advancing {sorted(present)}; "
            f"{len(excluded)} searched non-survivors excluded {excluded}")
        return self.record(1, True,
                           evidence_universe=len(self.evidence_universe),
                           active_finalists=sorted(present),
                           excluded_non_survivors=excluded,
                           seeded=seeded, rung1=self.rung1)

    # -- run ----------------------------------------------------------------
    def run(self) -> int:
        mark("DRIVER_START")
        # No stage 1 search anywhere in this map. `stage3`/`stage4`/`stage5` are
        # the inherited, frozen implementations.
        # Keys are the CONTINUATION plan's stage numbers, which are the
        # inherited ones. There is no stage 2: Phase A's stage 2 is rung 1 on
        # seed sa, and this session imports that result instead of buying it.
        stages = {0: self.stage_bind, 1: self.stage_import, 3: self.stage3,
                  4: self.stage4, 5: self.stage5}
        blocking = {s.stage for s in CONTINUATION_PLAN_V1.stages if s.blocking}
        failed: list[int] = []
        for stage in sorted(stages):
            try:
                ok = stages[stage]()
            except (RecoveryAdmissionError, AuthorizationError) as exc:
                self.record(stage, False, f"refused: {exc}"[-1500:])
                ok = False
            except Exception as exc:                              # noqa: BLE001
                self.record(stage, False, f"{type(exc).__name__}: {exc}"[-1500:],
                            **self.preserve_traceback(stage, exc))
                ok = False
            if ok:
                continue
            failed.append(stage)
            if stage in blocking:
                mark("PHASE_A_FAILED")
                say("stopping before any later stage; completed observations are "
                    "journalled and the permanent controls are untouched")
                return 1
        mark("ALL_DONE" if not failed else "PHASE_A_INCOMPLETE")
        return 0 if not failed else 1


def main() -> int:                                             # pragma: no cover
    import argparse

    bind_status_file()

    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--image-digest", default="")
    ap.add_argument("--rate", type=float, default=0.99)
    ap.add_argument("--spent-usd", type=float, default=0.0)
    ap.add_argument("--soft-stop-usd", type=float, default=0.0)
    ap.add_argument("--authorized-usd", type=float, default=0.0)
    ap.add_argument("--probe-train-minutes", type=float, default=61.55)
    ap.add_argument("--probe-battery-minutes", type=float, default=9.82)
    args = ap.parse_args()
    return ContinuationDriver(args).run()


if __name__ == "__main__":                                     # pragma: no cover
    raise SystemExit(main())
