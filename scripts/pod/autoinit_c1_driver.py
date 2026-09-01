#!/usr/bin/env python3
"""Phase C1: replay one frozen path under two digest gates, then run six probes.

The stage order, the gates and the arm construction are
`aadistill.autoinit.c1_session`; the decision rule is `c1_isolation`. This driver
executes them and records evidence. It restates no scientific constant.

**It cannot search, rank, eliminate or tie-break.** `PhaseADriver.stage1` — the
one method that calls `run_phase_a_search` — is overridden here and never
delegated to, and this module never imports `phase_a_search`, so the search entry
point is not one attribute lookup away. `PhaseADriver.run_rung` is overridden
too: successive halving is what it does, and C1 runs every arm on every seed.

**The two replay gates are the session's first product.** If the frozen path does
not reproduce `eea90c91…` at stage D, or `c313d1b4…` at stage E, the driver marks
`C1_REPLAY_MISMATCH`, writes the full evidence — every intermediate identity, the
DEPTH/FFN/ATTENTION selections, the WIDTH diagnostics, the runtime triple — and
stops. **No recovery training starts.** That outcome costs about $1.30 and is a
real scientific finding, not an outage; it is emphatically not a condition to
retry through.

What the six probes are: two arms — the incumbent's current
`attention.weight_proxy_v0` and the replacement `attention.activation_importance_v1`
— each trained at 0.86M on all three preregistered fresh seeds, then each
evaluated exactly once on `c1_confirmation_v1`. Both arms are built from the
*same verified parent*, so they differ by exactly one operator by construction.
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

from aadistill.autoinit import c1_session as CS  # noqa: E402
from aadistill.autoinit.c1_authorization import C1Authorization  # noqa: E402
from aadistill.autoinit.c1_isolation import (  # noqa: E402
    C1Arm, C1IsolationPlan, decide, derive_recovery_seeds, paired_differences,
    stratified_cluster_bootstrap,
)
from aadistill.autoinit.fixed_path import (  # noqa: E402
    FixedPathDigestMismatch, materialize_fixed_path, write_replay_record,
)
from aadistill.autoinit.operators import attention_activation  # noqa: E402
#: The frozen identities WITHOUT the search module, for the same reason the
#: recovery continuation imports it: `phase_a_search` would put
#: `run_phase_a_search` one attribute lookup away.
from phase_a_frozen import TEACHER_ID, TEACHER_REVISION  # noqa: E402
from autoinit_phase_a_driver import AUDIT, PhaseADriver, mark, say  # noqa: E402

BATTERY = REPO / "artifacts/stage3/c1_confirmation_v1"
BATTERY_IDENTITY = REPO / "logs/phase_c1_battery.json"
TEACHER_BINDING = REPO / "logs/phase_c1_teacher_binding.json"
WORK = REPO / "artifacts/stage3/c1"


class C1ReplayMismatch(RuntimeError):
    """A frozen digest did not reproduce. The session stops here, by design."""


class C1Driver(PhaseADriver):
    """Stages A-J of the C1 isolation. No search, no ranking, no elimination."""

    #: ITS OWN artifact and type. Inheriting the parent's would load a consumed
    #: Phase-A grant committed at that path and enforce a ceiling derived for a
    #: beam search this session cannot reach.
    AUTHORIZATION_TYPE = C1Authorization
    AUTHORIZATION_PATH = "logs/autoinit_c1_authorization.json"

    # --- the three things this session structurally cannot do --------------

    def stage1(self) -> bool:
        raise NotImplementedError(
            "C1 runs no search. The fixed path is replayed by "
            "materialize_fixed_path under two digest gates; there is no stage "
            "that could produce or consume a beam.")

    def run_rung(self, stage, descriptors, label):
        raise NotImplementedError(
            "C1 has no rungs. Successive halving eliminates arms on partial "
            "evidence; C1 runs every arm on every seed and eliminates nothing.")

    def selection_row(self, records):
        raise NotImplementedError(
            "C1 does not rank. The paired decision rule in c1_isolation is the "
            "only thing that reads probe results.")

    # --- the session --------------------------------------------------------

    def teacher_verify(self) -> dict:
        """Stage B. Every fetched file must hash to the bound value."""
        binding = json.loads(TEACHER_BINDING.read_text())
        if binding["revision"] != TEACHER_REVISION:
            raise RuntimeError(
                f"teacher binding pins {binding['revision']} but the session "
                f"declares {TEACHER_REVISION}")
        from huggingface_hub import snapshot_download
        import hashlib

        local = snapshot_download(TEACHER_ID, revision=TEACHER_REVISION)
        bad = []
        for name, want in binding["expected_shard_sha256"].items():
            p = Path(local) / name
            if not p.is_file():
                bad.append(f"{name}: absent after fetch")
                continue
            got = hashlib.sha256(p.read_bytes()).hexdigest()
            if got != want:
                bad.append(f"{name}: {got} != {want}")
        if bad:
            raise RuntimeError("teacher verification FAILED: " + "; ".join(bad))
        say(f"teacher {TEACHER_ID}@{TEACHER_REVISION[:12]} verified, "
            f"{len(binding['expected_shard_sha256'])} shards")
        return {"repo_id": TEACHER_ID, "revision": TEACHER_REVISION,
                "local_path": local, "shards_verified": True}

    def register_operator(self) -> dict:
        """Stage C. Explicit, because importing the module does not register."""
        impl = attention_activation.register(replace=True)
        say(f"registered {impl.impl_id} ({impl.signature_hash[:12]})")
        return {"impl_id": impl.impl_id, "signature_hash": impl.signature_hash}

    def replay_and_build_arms(self, teacher_path: str) -> dict:
        """Stages D, E and F. The gates live inside `materialize_fixed_path`.

        The incumbent arm carries both pins, so replaying it exercises the whole
        end-to-end check; the treatment arm is then built from the *same* parent
        rather than from a second replay, which is what makes the two arms
        differ by exactly one operator rather than by one operator plus whatever
        a second replay did differently.
        """
        from aadistill.autoinit.adapters.qwen3 import QWEN3_ADAPTER
        from transformers import AutoModelForCausalLM

        arms = CS.build_arm_specs(workdir_device="cuda")
        if not CS.arm_prefix_is_shared(arms):
            raise RuntimeError("the two arms do not share their prefix")

        def root():
            return AutoModelForCausalLM.from_pretrained(
                teacher_path, dtype="bfloat16").eval()

        runtime = self.runtime_identity()
        try:
            steps = materialize_fixed_path(
                arms["incumbent"], adapter=QWEN3_ADAPTER, root_loader=root,
                workdir=WORK / "incumbent", repo_root=str(REPO))
        except FixedPathDigestMismatch as exc:
            record = {"schema": "aadistill.autoinit.c1_replay_mismatch/v1",
                      "stage": "D" if exc.step_index < 3 else "E",
                      "step_index": exc.step_index, "label": exc.label,
                      "expected": exc.expected, "actual": exc.actual,
                      "runtime": runtime, "evidence": exc.evidence,
                      "training_started": False,
                      "meaning": (
                          "the frozen path did not reproduce its recorded digest "
                          "under this runtime. NO recovery training was started. "
                          "This is the session's product: refer it to review with "
                          "the evidence attached. Do not retry, and do not "
                          "substitute a rebuilt parent for the historical one "
                          "without a reviewed amendment.")}
            (AUDIT / "c1_replay_record.json").write_text(
                json.dumps(record, indent=1) + "\n")
            mark("C1_REPLAY_MISMATCH")
            raise C1ReplayMismatch(str(exc)) from exc

        write_replay_record(arms["incumbent"], steps,
                            AUDIT / "c1_replay_record.json",
                            runtime=runtime,
                            root_binding=json.loads(TEACHER_BINDING.read_text()))
        parent = steps[2]                       # the verified pre-ATTENTION parent
        say(f"parent {parent.identity.artifact_digest[:12]} and incumbent "
            f"{steps[3].identity.artifact_digest[:12]} both match their pins")

        # Stage F: the treatment arm, from the SAME verified parent.
        treatment = materialize_fixed_path(
            arms["treatment"], adapter=QWEN3_ADAPTER,
            root_loader=lambda: QWEN3_ADAPTER.load(parent.checkpoint_path,
                                                   device="cuda"),
            workdir=WORK / "treatment", repo_root=str(REPO))
        identities = {
            "parent": parent.as_dict(),
            "incumbent": steps[3].as_dict(),
            "treatment": treatment[-1].as_dict(),
            "shared_parent": True,
            "runtime": runtime,
        }
        (AUDIT / "c1_arm_identities.json").write_text(
            json.dumps(identities, indent=1) + "\n")
        return identities

    def runtime_identity(self) -> dict:
        import torch
        import transformers

        return {"image_digest": getattr(self.a, "image_digest", None),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "cuda_runtime": getattr(torch.version, "cuda", None),
                "gpu": (torch.cuda.get_device_name(0)
                        if torch.cuda.is_available() else None)}

    def probe_descriptors(self) -> list[dict]:
        """Stage G's schedule: every arm on every seed. Six, always."""
        seeds = derive_recovery_seeds()
        out = []
        for arm in ("incumbent", "treatment"):
            for seed in seeds:
                out.append({"probe_id": f"autoinit.v1.phase_c1.{arm}.{seed}",
                            "arm": arm, "seed": seed, "rung": 1})
        assert len(out) == 6, "C1 is exactly six probes"
        return out

    def apply_decision(self, per_probe: dict) -> dict:
        """Stage I. Only after all six results exist."""
        if len(per_probe) != 6:
            raise RuntimeError(
                f"{len(per_probe)} probe results, not 6; the decision rule may "
                "not run on a partial design")
        battery = json.loads(BATTERY_IDENTITY.read_text())
        plan = C1IsolationPlan(
            plan_id="autoinit.v1.phase_c1",
            arms=(C1Arm("c1.incumbent", "incumbent", *CS.INCUMBENT_ATTENTION),
                  C1Arm("c1.treatment", "treatment", *CS.TREATMENT_ATTENTION)),
            seeds=tuple(derive_recovery_seeds()),
            battery_asset_id=battery["asset_id"],
            battery_content_sha256=battery["content_sha256"])

        inc = {s: per_probe[("incumbent", s)]["correct"] for s in plan.seeds}
        trt = {s: per_probe[("treatment", s)]["correct"] for s in plan.seeds}
        d = paired_differences(inc, trt)
        strata = per_probe[("incumbent", plan.seeds[0])]["strata"]
        boot = stratified_cluster_bootstrap(d, strata)
        per_seed = [
            sum(bool(trt[s][j]) - bool(inc[s][j]) for j in d) / len(d)
            for s in plan.seeds]
        result = decide(
            plan, boot=boot, per_seed_delta=per_seed,
            usable_pooled_delta=per_probe["usable_pooled_delta"],
            usable_per_seed_delta=per_probe["usable_per_seed_delta"],
            catastrophic_violations=per_probe.get("catastrophic_violations", ()))
        result["plan_hash"] = plan.plan_hash
        (AUDIT / "c1_decision.json").write_text(json.dumps(result, indent=1) + "\n")
        return result


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Phase C1 fixed-path ATTENTION isolation")
    ap.add_argument("--stage", default="all")
    ap.add_argument("--image-digest", default="")
    ap.add_argument("--rate", type=float, default=0.99)
    ap.add_argument("--spent-usd", type=float, default=0.0)
    ap.add_argument("--soft-stop-usd", type=float, required=True)
    ap.add_argument("--authorized-usd", type=float, required=True)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    say(f"C1: {CS.C1_SESSION_CONTRACT.n_probes} probes, ceiling "
        f"${args.authorized_usd:.4f}, soft stop ${args.soft_stop_usd:.4f}")
    for s in CS.C1_STAGES:
        say(f"  {s.letter}: {s.stage_id}"
            + ("   [blocks training on failure]" if s.blocks_training else ""))
    return C1Driver(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
