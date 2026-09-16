#!/usr/bin/env python3
"""Phase C2 baseline completion: establish B, measure it once, compare against frozen C.

    python scripts/pod/autoinit_phase_c2_baseline_driver.py \
        --protocol logs/stages/stage-1/phase_c2/plans/phase_c2_baseline_completion_protocol.json \
        --frozen-inputs .../evidence/c2_frozen_comparison_inputs.json \
        --selection-record .../evidence/stage1_selection.json \\
        --rebuild-minutes N --soft-stop-usd X --rate R --spent-usd S

Attempt 4's beam search completed and committed a ranking of five candidates,
then its conditional baseline rebuild hit a reserve that could not fund the
work. So the candidate side of the B->C comparison is measured and frozen, and
the baseline side does not exist. This driver supplies only the missing half.

**The beam search is unreachable from here.** Not by instruction -- by
construction. Nothing in this module imports `run_phase_a_search`, `BeamSearch`,
`SCHEDULE_V1` or any search entry point, and there is no code path that could
generate a candidate. `tests/pod/test_phase_c2_baseline_completion.py` asserts
that over the module's import graph, so an edit that reintroduces the search
fails a test rather than quietly widening what a grant authorizes.

**Nothing about C is recomputed.** The five candidate measurements are read from
the frozen record, verified against their own hash, and passed to the comparison
builder untouched. There is no code path that measures a candidate.

**Two stages, and the first one spends nothing.** Every identity the comparison
depends on -- the suite, the policy and its epsilon, the teacher, the B spec,
and the evaluator implementation itself -- is checked before the teacher is
loaded. A mismatch there means the two halves of the comparison were produced by
different machinery, and that is a stop, not something to compensate for.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _extra in ("src", "scripts", "scripts/autoinit"):
    if str(REPO / _extra) not in sys.path:
        sys.path.insert(0, str(REPO / _extra))

from aadistill.infrastructure.manifest import sha256_file, sha256_json  # noqa: E402
from aadistill.initialization.planning import stage1_selection  # noqa: E402
from aadistill.initialization.planning.ranking import PARETO_V1  # noqa: E402
from aadistill.initialization.specs.state import make_retained_state  # noqa: E402
from experiments.phase_c2 import baseline as B  # noqa: E402
from experiments.phase_c2 import comparison as C  # noqa: E402
from experiments.phase_c2.frozen_inputs import (  # noqa: E402
    load_frozen_candidates, load_record)
#: At MODULE scope, because the calibration mixtures are data that an
#: application bootstrap registers and stage A resolves profiles. C2 attempt 3
#: died one second into its first stage for want of exactly this import being
#: here rather than inside the stage that needed it.
from experiments.calibration import register_builtin_profiles  # noqa: E402

register_builtin_profiles()

WS = Path("/workspace")
STATUS = WS / "autoinit_phase_c2_baseline.status"
AUDIT = REPO / "artifacts/audit/autoinit_phase_c2_baseline"
WORK = REPO / "artifacts/autoinit/phase_c2_baseline"

SCHEMA = "aadistill.autoinit.c2_baseline_completion_evidence/v1"


class CompletionError(RuntimeError):
    """The completion cannot proceed on the evidence it was given."""


def mark(name: str) -> None:
    line = f"{datetime.now(timezone.utc):%FT%TZ} MARKER:{name}"
    print(line, flush=True)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    with STATUS.open("a") as handle:
        handle.write(line + "\n")


def say(message: str) -> None:
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {message}", flush=True)


class BaselineCompletionDriver:
    """Two stages: check every shared identity, then rebuild, measure, compare."""

    def __init__(self, args) -> None:
        self.a = args
        self.t0 = time.time()
        self.completed: list[str] = []
        AUDIT.mkdir(parents=True, exist_ok=True)
        self.protocol = json.loads(Path(args.protocol).read_text())
        self.ev: dict = {
            "schema": SCHEMA,
            "_contract": (
                "the Phase-C2 baseline-completion session record. It supplies the "
                "baseline half of a comparison whose candidate half was measured "
                "by Attempt 4 and is read frozen. No beam search is reachable "
                "from this session and no candidate is remeasured."),
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": self.protocol.get("protocol_sha256"),
            "frozen_inputs": str(args.frozen_inputs),
            "rate_usd_per_hour": args.rate,
            "soft_stop_usd": args.soft_stop_usd,
            "trains_anything": False,
            "runs_a_beam_search": False,
            "remeasures_any_candidate": False,
            "stages": {},
        }
        self.save()

    # -- bookkeeping -------------------------------------------------------
    def usd(self) -> float:
        return self.a.spent_usd + (time.time() - self.t0) / 3600.0 * self.a.rate

    def save(self) -> None:
        self.ev["elapsed_min"] = round((time.time() - self.t0) / 60, 2)
        self.ev["spend_usd"] = round(self.usd(), 4)
        self.ev["stages_completed"] = list(self.completed)
        (AUDIT / "c2_baseline_completion_evidence.json").write_text(
            json.dumps(self.ev, indent=2, default=str) + "\n")

    def record(self, stage: str, passed: bool, detail, **extra) -> None:
        self.ev["stages"][stage] = {
            "stage_id": stage, "passed": passed,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "spend_usd": round(self.usd(), 4), "detail": detail, **extra}
        if passed:
            self.completed.append(stage)
            mark(f"STAGE_PASSED:{stage}")
        else:
            mark(f"STAGE_FAILED:{stage}")
        self.save()

    def afford(self, minutes: float, what: str) -> None:
        projected = self.usd() + minutes / 60.0 * self.a.rate
        if projected > self.a.soft_stop_usd:
            raise CompletionError(
                f"{what} needs {minutes:.1f} min (${projected:.2f} projected) and "
                f"the soft stop is ${self.a.soft_stop_usd:.2f}")

    # -- stage A -----------------------------------------------------------
    def bind_identities(self) -> bool:
        """Everything the two halves must share, before anything is loaded."""
        from experiments.phase_c2.search_space import register_c2_operators
        from phase_a_frozen import TEACHER_ID, TEACHER_REVISION

        #: `attention.activation_importance_v1` is not a shipped default and the
        #: frozen B path's last step names it. Registered FIRST, as everywhere.
        register_c2_operators()

        frozen_record = load_record(self.a.frozen_inputs)
        bound = self.protocol["cross_session_comparability_contract"]["bound"]

        #: The evaluator that measured C, by source hash. This is the check that
        #: makes "comparable across sessions" a verified statement rather than an
        #: assumption: if the implementation moved, the two halves were produced
        #: by different machinery and no amount of identical configuration makes
        #: their numbers one measurement series.
        drifted = {path: sha256_file(REPO / path)
                   for path, expected in bound["evaluator_implementation_sha256"].items()
                   if sha256_file(REPO / path) != expected}
        if drifted:
            raise CompletionError(
                "the evaluator implementation has moved since the candidate side "
                f"was frozen: {json.dumps(drifted, indent=1)}. STOP: the frozen C "
                "measurements and a new B measurement would not be the same "
                "measurement series, and that is not something to compensate for.")

        checks = {
            "suite_hash": (frozen_record["suite"]["hash"], bound["state_eval_suite_hash"]),
            "policy_hash": (frozen_record["policy"]["hash"], bound["pareto_policy_hash"]),
            "live_policy_hash": (PARETO_V1.policy_hash, bound["pareto_policy_hash"]),
            "teacher_repo_id": (TEACHER_ID, bound["teacher_repo_id"]),
            "teacher_revision": (TEACHER_REVISION, bound["teacher_revision"]),
        }
        disagreements = {k: {"live": a, "bound": b} for k, (a, b) in checks.items() if a != b}
        if disagreements:
            raise CompletionError(
                "a frozen identity does not match the protocol: "
                + json.dumps(disagreements, indent=1))

        live_epsilon = {k: float(v) for k, v in PARETO_V1.epsilon.items()}
        if live_epsilon != {k: float(v) for k, v in
                            self.protocol["frozen_identities_that_must_not_move"]
                            ["pareto_epsilon"].items()}:
            raise CompletionError(
                f"the live epsilon {live_epsilon} is not the protocol's. Epsilon may "
                "not change, and it may certainly not change before B is measured.")

        spec = B.frozen_baseline_spec(device=self.a.device)
        construction = B.assert_frozen_construction(spec)

        candidates = load_frozen_candidates(
            self.a.frozen_inputs,
            expect_suite_hash=bound["state_eval_suite_hash"],
            expect_policy_hash=bound["pareto_policy_hash"])
        say(f"  {len(candidates)} frozen candidate measurement(s) loaded, "
            f"record {frozen_record['self_sha256'][:12]}…")
        say(f"  frozen baseline construction verified: {spec.spec_hash[:12]}… "
            f"== {B.B_SPEC_HASH[:12]}…")

        #: The BEAM RANKING, at a path the caller supplies and this stage
        #: VERIFIES: its own commitment hash must be the one the frozen
        #: extraction recorded. So a caller can say where the ranking is and
        #: cannot say which ranking it is.
        selection = stage1_selection.load(self.a.selection_record)
        committed = frozen_record["sources"]["selection_commitment_sha256"]
        if selection["selection_sha256"] != committed:
            raise CompletionError(
                f"{self.a.selection_record} commits "
                f"{selection['selection_sha256']} and the frozen candidate "
                f"record was extracted from a selection committing {committed}. "
                "These are different rankings.")

        #: The Search-1 identities the comparison must describe, taken from the
        #: frozen record and NOT from the command line. A caller must not be
        #: able to make a valid B measurement produce a record claiming the
        #: candidates came from another search configuration.
        self.frozen = {
            "search_run_id": frozen_record["search"]["run_id"],
            "search_config_hash": frozen_record["search"]["config_hash"],
            "selection_commitment_sha256": committed,
            "suite_hash": frozen_record["suite"]["hash"],
            "policy_hash": frozen_record["policy"]["hash"],
        }
        if selection["search"]["config_hash"] != self.frozen["search_config_hash"]:
            raise CompletionError(
                "the ranking and the frozen extraction disagree about the search "
                f"config: {selection['search']['config_hash']} vs "
                f"{self.frozen['search_config_hash']}")
        say(f"  bound Search-1 identity: {self.frozen['search_run_id']} "
            f"config {self.frozen['search_config_hash'][:12]}…")

        self.candidates = candidates
        self.ev["frozen_candidate_state_ids"] = [c.state_id for c in candidates]
        self.ev["bound_search_identity"] = dict(self.frozen)
        self.record("bind_identities", True, {
            "bound_search_identity": dict(self.frozen),
            "_identities_are_bound_not_supplied": (
                "the Search-1 run id and config hash come from the frozen "
                "candidate record; there is no command-line source for either"),
            "beam_ranking_cited": str(self.a.selection_record),
            "frozen_inputs_self_sha256": frozen_record["self_sha256"],
            "frozen_inputs_extraction_rule": frozen_record["extraction_rule"],
            "n_frozen_candidates": len(candidates),
            "source_journal_sha256": frozen_record["sources"]["journal_sha256"],
            "selection_commitment_sha256":
                frozen_record["sources"]["selection_commitment_sha256"],
            "suite": frozen_record["suite"],
            "policy": frozen_record["policy"],
            "epsilon": live_epsilon,
            "teacher": {"repo_id": TEACHER_ID, "revision": TEACHER_REVISION},
            "baseline_construction": construction,
            "evaluator_implementation_verified": sorted(
                bound["evaluator_implementation_sha256"]),
            "_no_measurement_here": "stage A loads no model and measures nothing",
        })
        return True

    # -- the two expensive loads, isolated so a $0 test can stub them --------
    def load_suite_bundle(self):
        """The frozen suite, from the root its own declaration names.

        NOT the repository root, which is what this driver asked for until
        2026-09-16: `load_state_eval.load` reads `manifest.json` and
        `items.jsonl` directly beneath the root it is given, and the repository
        root has neither. The failure would have landed on the pod, after setup,
        inside the only stage that spends money.
        """
        from load_state_eval import load as load_suite
        from experiments.phase_c2.frozen_assets import state_eval_root

        root = state_eval_root(REPO)
        say(f"  state_eval root resolved from its declaration: {root}")
        return (root, *load_suite(root))

    def load_original_teacher(self):
        """The pinned original teacher, ONCE, at the bound revision.

        By repo id and `revision=`, exactly as `run_phase_a_search` loaded the
        teacher the frozen C measurements were scored against -- so the revision
        is enforced by the loader rather than asserted about a directory.
        """
        import torch
        from transformers import AutoModelForCausalLM
        from phase_a_frozen import TEACHER_ID, TEACHER_REVISION

        say(f"  loading the original teacher {TEACHER_ID}@{TEACHER_REVISION[:12]}… "
            f"once, on {self.a.device}")
        return AutoModelForCausalLM.from_pretrained(
            TEACHER_ID, dtype=torch.bfloat16, revision=TEACHER_REVISION,
        ).to(self.a.device).eval()

    # -- stage B -----------------------------------------------------------
    def rebuild_measure_compare(self) -> bool:
        """The one unit of scientific work this session exists to do."""
        from transformers import AutoConfig

        from aadistill.initialization.adapters.qwen3 import QWEN3_ADAPTER
        from aadistill.initialization.specs.arch import ArchSpec
        from aadistill.initialization.specs.artifact import identify_checkpoint
        from phase_a_frozen import TARGET_GEOMETRY, TEACHER_ID
        from aadistill.initialization.planning.metrics import StateEvaluator
        from aadistill.initialization.specs.metrics import ReferenceStrategy
        from experiments.phase_c2.frozen_inputs import (
            numerical_sensitivity_disclosure)

        suite_root, suite, items, suite_manifest = self.load_suite_bundle()
        if suite.suite_hash != self.frozen["suite_hash"]:
            raise CompletionError(
                f"the staged suite at {suite_root} hashes to {suite.suite_hash} "
                f"and the frozen candidates were measured on "
                f"{self.frozen['suite_hash']}. Values from two suites are not "
                "comparable.")
        target_spec = ArchSpec.of("qwen3", TARGET_GEOMETRY)

        #: RECOMPUTE, explicitly, because it is the strategy the frozen C
        #: measurements recorded and the property the comparability contract
        #: rests on: no candidate is normalized against any other.
        evaluator = StateEvaluator(
            suite, items, device=self.a.device,
            reference_strategy=ReferenceStrategy.RECOMPUTE)

        #: ONE teacher object, primed as the reference AND handed to the
        #: rebuild. Attempt 4's path did exactly this -- `run_phase_a_search`
        #: primes the evaluator and passes `lambda: teacher` to the conditional
        #: baseline hook -- and the semantics matter twice over: `evaluate`
        #: REFUSES against an unprimed evaluator, and a second independently
        #: loaded teacher would put a second 4B model on the device and let B be
        #: built from a different object than the one B is scored against.
        teacher = self.load_original_teacher()
        evaluator.prime_reference(teacher)
        say("  evaluator primed with the original teacher (RECOMPUTE)")

        fallback = B.BaselineFallback(
            adapter=QWEN3_ADAPTER, workdir=WORK,
            rebuild_minutes=self.a.rebuild_minutes, afford=self.afford,
            repo_root=REPO, device=self.a.device, say=say)

        #: The rebuild's own entry point, not the search's conditional hook. The
        #: hook exists to be called BY a beam; this session has none. The SAME
        #: teacher object the evaluator was primed with.
        entry = fallback.rebuild(lambda: teacher)
        outcome = fallback.outcome

        directory = Path(entry["checkpoint_dir"])
        spec = QWEN3_ADAPTER.spec_from_config(AutoConfig.from_pretrained(str(directory)))
        artifact = identify_checkpoint(directory, adapter=QWEN3_ADAPTER, spec=spec,
                                       num_parameters=QWEN3_ADAPTER.param_count(spec))
        state = make_retained_state(
            state_id=entry["candidate_id"], artifact=artifact, spec=spec,
            target_spec=target_spec,
            num_parameters=QWEN3_ADAPTER.param_count(spec),
            root_teacher_id=TEACHER_ID,
            root_teacher_sha256=suite_manifest.get("teacher_sha256", "") or "0" * 64,
            description=entry["description"], provenance=entry["provenance"],
            expected_artifact_digest=entry["expected_artifact_digest"])

        #: ONCE. The identical canonical-reload/state_eval path every searched
        #: candidate took, on the same suite and the same evaluator.
        self.afford(4.0, "the single state_eval of B")
        state.attach_evaluation(evaluator.evaluate(
            QWEN3_ADAPTER.load(str(directory), device=self.a.device),
            artifact.artifact_digest))
        say(f"  B measured once on {suite.qualified_id}: "
            f"{state.evaluation.values['state.teacher_kl.equal_domain_mean']:.6f} "
            "equal-domain mean KL")

        #: THE DURABILITY BOUNDARY, and it is here rather than after the
        #: comparison for one reason: everything below this line is arithmetic
        #: over numbers that already exist, and a failure in it would otherwise
        #: discard the one GPU measurement this session was funded to take.
        #:
        #: Phase-B attempt 4 completed eight hours of science, ranked it, and
        #: raised in a summary dict -- ending with no authoritative record of
        #: what it had measured. The beam learned to commit its ranking the
        #: instant it existed; this is the same rule for a single measurement.
        #:
        #: It writes into the session evidence document, which the artifact
        #: specs already collect on BOTH the success and the failure path, so no
        #: new artifact framework is involved. After this returns, a
        #: post-processing repair is a $0 host-side job over the durable
        #: evaluation plus the frozen five, and B is never remeasured.
        self.persist_baseline_measurement(
            state=state, artifact=artifact, outcome=outcome, suite=suite,
            suite_root=suite_root)

        #: Pure post-processing from here. One measured B, five frozen C.
        #:
        #: The disclosure is built BEFORE the record and handed to the builder,
        #: so it is inside `record_sha256` rather than appended after it. A
        #: field added afterwards would leave the document differing from the
        #: object its own self-hash describes.
        disclosure = self.protocol["cross_session_comparability_contract"][
            "preregistered_disclosure_RULE_not_a_rule_change"]
        interpretation = numerical_sensitivity_disclosure(
            state.evaluation.values, self.candidates,
            objectives=tuple(o.key for o in PARETO_V1.objectives),
            threshold=float(disclosure["threshold"]))
        interpretation["numerical_sensitivity"]["_registered_before_b_existed"] = (
            disclosure["rule"])

        record = C.build(
            baseline=state, baseline_outcome=outcome,
            candidates=list(self.candidates), suite=suite, policy=PARETO_V1,
            #: The SEARCH that produced C, from the frozen record. This
            #: session's own run identity lives in its governance and session
            #: records; conflating the two would let a comparison describe the
            #: completion session as though it had searched.
            run_id=self.frozen["search_run_id"],
            config_hash=self.frozen["search_config_hash"],
            #: The beam ranking, verified in stage A against the commitment the
            #: frozen extraction recorded.
            selection_record=str(self.a.selection_record),
            #: The derived extraction, cited separately from the ranking.
            frozen_candidate_inputs=str(self.a.frozen_inputs),
            interpretation=interpretation)

        sensitivity = record["numerical_sensitivity"]
        if sensitivity["status"] == "FLAGGED":
            say(f"  DISCLOSURE: {sensitivity['n_flagged_pairs']} margin(s) at or "
                f"below {sensitivity['threshold']}; the verdict is unaltered")

        #: The builder nests its computed verdict under `comparison`. Reading
        #: `record["verdict"]` raised a KeyError -- after B had been rebuilt and
        #: measured, which is the most expensive possible moment to learn it.
        verdict = record["comparison"]["verdict"]
        path = C.commit(record, AUDIT)
        say(f"  comparison written: {path.name}, verdict {verdict}")
        self.record("rebuild_measure_compare", True, {
            "baseline_identity": outcome["identity_matches"],
            "baseline_state_id": state.state_id,
            "baseline_artifact_digest": artifact.artifact_digest,
            "state_eval_measurements_performed": 1,
            "candidates_compared": len(self.candidates),
            "candidates_remeasured": 0,
            "teacher_instances_loaded": 1,
            "verdict": verdict,
            "numerical_sensitivity": sensitivity["status"],
            "n_flagged_pairs": sensitivity["n_flagged_pairs"],
            "cross_session_variance": sensitivity["cross_session_variance"],
            "suite_root": str(suite_root),
            "cites_beam_ranking": str(self.a.selection_record),
            "cites_frozen_candidate_inputs": str(self.a.frozen_inputs),
            "record_sha256": record["record_sha256"],
            "comparison_record": str(path),
        })
        return True


    # -- the durability boundary -------------------------------------------
    def persist_baseline_measurement(self, *, state, artifact, outcome, suite,
                                     suite_root) -> None:
        """Write the ONE formal B measurement into the evidence, immediately.

        Everything needed to reconstruct the comparison later WITHOUT the
        checkpoint bytes and without measuring anything again: B's identity,
        its complete `StateEvaluation`, the suite and reference strategy it was
        taken under, the teacher it was scored against, and the frozen Search-1
        identities the comparison will be computed against.
        """
        from phase_a_frozen import TEACHER_ID, TEACHER_REVISION

        evaluation = state.evaluation
        self.ev["baseline_measurement"] = {
            "status": "MEASURED",
            "state_eval_measurements_performed": 1,
            "measured_utc": (getattr(evaluation, "measured_utc", None)
                             or datetime.now(timezone.utc).isoformat()),
            "persisted_utc": datetime.now(timezone.utc).isoformat(),
            "state_id": state.state_id,
            "path_label": getattr(state, "path_label", None),
            "construction": outcome.get("construction"),
            "expected_artifact_digest": B.B_ARTIFACT_DIGEST,
            "actual_artifact_digest": artifact.artifact_digest,
            "artifact_identity": {
                "weights_digest": getattr(artifact, "weights_digest", None),
                "config_sha256": getattr(artifact, "config_sha256", None),
                "arch_signature": getattr(artifact, "arch_signature", None),
                "tokenizer_sha256": getattr(artifact, "tokenizer_sha256", None),
                "index_sha256": getattr(artifact, "index_sha256", None),
                "single_shard_sha256": getattr(artifact, "single_shard_sha256",
                                               None),
                "num_parameters": getattr(artifact, "num_parameters", None),
            },
            "identity_matches": outcome.get("identity_matches"),
            #: COMPLETE, so a later session rehydrates the measurement rather
            #: than recomputing it.
            "evaluation": evaluation.as_dict(),
            "suite": {"id": suite.qualified_id, "hash": suite.suite_hash,
                      "root": str(suite_root)},
            "reference_strategy": (evaluation.detail or {}).get(
                "reference_strategy"),
            "teacher": {"repo_id": TEACHER_ID, "revision": TEACHER_REVISION},
            "bound_search1_identity": dict(self.frozen),
            "frozen_candidate_inputs": str(self.a.frozen_inputs),
            "no_candidate_was_remeasured": (
                "the five C measurements were READ from the frozen record and "
                "passed to the comparison untouched. No code path in this "
                "session measures a candidate, and the count above is the "
                "session's total."),
            "_this_is_the_durability_boundary": (
                "written the instant the measurement existed and before any "
                "post-processing. If the comparison below fails, THIS survives "
                "and the comparison can be rebuilt from it plus the frozen five "
                "at $0. B is not remeasured."),
        }
        self.save()
        mark("BASELINE_MEASURED")
        say("  B measurement PERSISTED — durable before any post-processing")

    # -- the loop ----------------------------------------------------------
    def run(self) -> int:
        mark("BASELINE_COMPLETION_START")
        stages = (("bind_identities", self.bind_identities),
                  ("rebuild_measure_compare", self.rebuild_measure_compare))
        for name, function in stages:
            try:
                ok = function()
            except Exception as exc:                            # noqa: BLE001
                self.record(name, False, f"{type(exc).__name__}: {exc}"[-2000:],
                            traceback=traceback.format_exc()[-6000:])
                ok = False
            if not ok:
                self.ev["outcome"] = "FAILED"
                self.ev["failed_stage"] = name
                self.save()
                mark("BASELINE_COMPLETION_FAILED")
                return 1
        self.ev["outcome"] = "ALL_DONE"
        self.ev["successful"] = True
        self.save()
        mark("BASELINE_COMPLETION_ALL_DONE")
        return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    #: Every argument here is a LOCATION or a BUDGET. None of them is a
    #: scientific identity: the Search-1 run id, its config hash, the suite
    #: hash, the policy hash and the selection commitment are all bound in
    #: stage A from the frozen candidate record, so a caller cannot make a
    #: valid B measurement produce a record attributing C to another search.
    #:
    #: `--config-hash` used to be here and was exactly that hole. There is no
    #: `--teacher-path` either: the teacher is loaded by pinned repo id and
    #: revision, so the revision is enforced by the loader.
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--frozen-inputs", required=True)
    ap.add_argument("--selection-record", required=True,
                    help=("the committed beam ranking. Its own commitment hash "
                          "is checked in stage A against the one the frozen "
                          "extraction recorded."))
    ap.add_argument("--rebuild-minutes", type=float, required=True)
    ap.add_argument("--rate", type=float, required=True)
    ap.add_argument("--spent-usd", type=float, default=0.0)
    ap.add_argument("--soft-stop-usd", type=float, required=True)
    ap.add_argument("--device", default="cuda")
    return ap


def main(argv=None) -> int:
    return BaselineCompletionDriver(build_parser().parse_args(argv)).run()


if __name__ == "__main__":
    raise SystemExit(main())
