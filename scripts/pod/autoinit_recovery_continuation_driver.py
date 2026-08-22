#!/usr/bin/env python3
"""Phase-A recovery continuation: import Stage 1, then run Stages 2-5.

Attempts 11 and 12 produced byte-identical Stage-1 searches, and attempt 12
preserved the five selected checkpoints off-pod with re-verified digests. A third
search would cost 203 minutes and add no scientific information. This driver
consumes that result instead.

**It cannot run a search, and that is structural rather than conventional.** This
module never imports `phase_a_search`; the frozen identities come from
`phase_a_frozen`, which contains no search code. `PhaseADriver.stage1` — the one
that calls `run_phase_a_search` — is overridden here and never delegated to. A
test asserts all three properties, and mutating any of them fails it.

What replaces the search:

1. `import_stage1_result` rebuilds the five leaves from the **staged bytes** and
   the committed attempt-12 evidence, refusing a different config hash, a
   reordering, a substitution or a digest the bytes contradict;
2. the canonical control comes back **unmeasured** — its Stage-1 evaluation is
   not in the evidence — so it is measured here, once, on the same frozen suite,
   through the same evaluator contract, and the measurement is persisted;
3. the complete `[five leaves + control]` set goes through the same
   `admit_leaves` gate a live search's output goes through;
4. the teacher and evaluator are released and the recovery-trainer headroom
   contract is checked before rung 1 starts.

Step 4 costs almost nothing here, because this session never held the search's
device residency — which is the whole point of not running it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "autoinit"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aadistill.autoinit.device_handoff import (  # noqa: E402
    DeviceHandoffError, complete_release, cuda_memory, require_headroom,
    require_released,
)
from aadistill.autoinit.recovery import admit_leaves  # noqa: E402
from aadistill.autoinit.recovery_continuation import (  # noqa: E402
    RecoveryContinuationAuthorization,
)
from aadistill.autoinit.stage1_import import (  # noqa: E402
    Stage1ImportError, import_stage1_result,
)
#: The frozen identities WITHOUT the search module. Importing `phase_a_search`
#: here would put `run_phase_a_search` one attribute lookup away.
from phase_a_frozen import (  # noqa: E402
    CANONICAL_INIT, CANONICAL_INIT_SHA256, TARGET_GEOMETRY, TEACHER_ID,
    TEACHER_REVISION,
)
from autoinit_phase_a_driver import (  # noqa: E402
    AUDIT, RECOVERY_TRAINER_BYTES, STATE_EVAL, PhaseADriver, mark, say,
)

#: Where the launcher stages attempt 12's five preserved leaves.
STAGED_LEAVES = REPO / "artifacts/autoinit/phase_a_selected"
#: The committed evidence the import is bound to.
EVIDENCE = REPO / "logs/autoinit_phase_a_attempt12"


class RecoveryContinuationDriver(PhaseADriver):
    """Stages 0 and 2-5 of Phase A, with Stage 1 imported rather than searched."""

    #: ITS OWN artifact and type. Inheriting the parent's would have loaded
    #: attempt 12's consumed Phase-A authorization — which is committed at that
    #: path — and enforced a $23.0484 ceiling on a session priced at $16.7456.
    AUTHORIZATION_TYPE = RecoveryContinuationAuthorization
    AUTHORIZATION_PATH = "logs/autoinit_recovery_continuation_authorization.json"

    def stage1(self) -> bool:
        """Import the verified Stage-1 result. No search is reachable from here."""
        self.enter(1)
        from aadistill.autoinit.arch import get_adapter

        adapter = get_adapter("qwen3")
        result = json.loads((EVIDENCE / "search_result.json").read_text())
        durability = json.loads(
            (EVIDENCE / "selected_leaf_durability.json").read_text())

        try:
            imported = import_stage1_result(
                search_result=result,
                states_path=EVIDENCE / "search_states_reduced.jsonl",
                checkpoint_store=STAGED_LEAVES, durability=durability,
                adapter=adapter,
                expected_config_hash=result["config_hash"],
                target_geometry=TARGET_GEOMETRY,
                control_dir=REPO / CANONICAL_INIT,
                control_sha256=CANONICAL_INIT_SHA256)
        except Stage1ImportError as exc:
            return self.record(
                1, False, f"the persisted Stage-1 result did not verify: {exc}")

        say(f"imported {len(imported.leaves)} stage-1 leaves, "
            f"config {result['config_hash'][:12]}…, re-identified from bytes")
        (AUDIT / "stage1_import.json").write_text(
            json.dumps(imported.verification, indent=2, default=str) + "\n")
        mark(f"STAGE1_IMPORTED:{len(imported.leaves)}")

        # --- the control, measured here because it cannot be imported -------
        # Its stage-1 evaluation is absent from the evidence, and inventing one
        # would make the baseline incomparable to the leaves it is the baseline
        # for. One evaluation, on the same suite, through the same contract.
        try:
            control_eval, teacher, evaluator = self.measure_control(
                imported.control, adapter)
        except Exception as exc:                       # noqa: BLE001
            return self.record(
                1, False, f"the canonical control could not be measured: "
                          f"{type(exc).__name__}: {exc}")
        imported.control.attach_evaluation(control_eval)
        (AUDIT / "control_measurement.json").write_text(
            json.dumps(control_eval.as_dict(), indent=2, default=str) + "\n")
        say(f"control measured on {control_eval.suite_id} "
            f"({control_eval.positions} positions)")

        # --- the same gate a live search's output passes --------------------
        try:
            admit_leaves([*imported.leaves, imported.control], self.plan)
        except Exception as exc:                       # noqa: BLE001
            return self.record(1, False, f"admission refused: {exc}")

        self.leaves = imported.leaves
        self.control_state = imported.control
        self.search_result = imported.summary

        # --- hand the card to recovery --------------------------------------
        # The BEFORE snapshot is taken here, by the frame that owns the names,
        # and the `del` happens BEFORE `complete_release` measures the result.
        # `evaluator` must go too: `prime_reference` stores the teacher on
        # `StateEvaluator._teacher`, so dropping `teacher` alone leaves the
        # model alive behind the evaluator.
        before = cuda_memory()
        del teacher, evaluator
        handoff = complete_release(before)
        (AUDIT / "device_handoff.json").write_text(
            json.dumps(handoff, indent=2, default=str) + "\n")
        self.ev.setdefault("runtime", {})["device_handoff"] = handoff
        say(f"device handoff: {handoff.get('verdict', 'n/a')}")
        try:
            require_released(handoff, what="the stage-2 recovery trainer")
            require_headroom(handoff["after"], need_bytes=RECOVERY_TRAINER_BYTES,
                             what="the stage-2 recovery trainer")
        except DeviceHandoffError as exc:
            return self.record(1, False, str(exc))

        return self.record(
            1, True, imported=True,
            stage1_source="phase-a attempt 12, verified from bytes",
            config_hash=result["config_hash"],
            selected=[s.state_id for s in self.leaves],
            control_measured=True, device_handoff=handoff.get("verdict"))

    def measure_control(self, control, adapter):
        """Measure the canonical control on the frozen state-evaluation suite.

        Returns the evaluation and the live teacher/evaluator so the caller can
        release them explicitly at the handoff rather than leaving them to a
        closure nobody remembers holds them.
        """
        import torch
        from transformers import AutoModelForCausalLM

        from aadistill.autoinit.metrics import StateEvaluator
        from load_state_eval import load as load_suite

        suite, items, _manifest = load_suite(STATE_EVAL)
        teacher = AutoModelForCausalLM.from_pretrained(
            TEACHER_ID, revision=TEACHER_REVISION, dtype=torch.bfloat16,
        ).to("cuda").eval()
        evaluator = StateEvaluator(suite, items, device="cuda")
        evaluator.prime_reference(teacher)
        model = adapter.load(str(REPO / CANONICAL_INIT), device="cuda")
        evaluation = evaluator.evaluate(model, control.artifact_digest)
        del model
        return evaluation, teacher, evaluator


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    #: No "search" stage exists to name. The continuation's stage 1 IS the
    #: import, and there is no flag that selects a search.
    ap.add_argument("--stage", default="all", choices=("all",))
    ap.add_argument("--image-digest", required=True)
    ap.add_argument("--rate", type=float, required=True)
    ap.add_argument("--spent-usd", type=float, default=0.0)
    ap.add_argument("--soft-stop-usd", type=float, required=True)
    ap.add_argument("--authorized-usd", type=float, required=True)
    ap.add_argument("--probe-train-minutes", type=float, required=True)
    ap.add_argument("--probe-battery-minutes", type=float, required=True)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    #: The full driver expects these two; the continuation owes neither, because
    #: it runs no search. Supplied as zero so the shared budget arithmetic is
    #: explicit about that rather than inheriting a search allowance.
    args.search_minutes = 0.0
    args.search_deadline_minutes = 0.0
    driver = RecoveryContinuationDriver(args)
    driver.auth.require_within_cap(args.authorized_usd, what="session backstop")
    return driver.run()


if __name__ == "__main__":
    raise SystemExit(main())
