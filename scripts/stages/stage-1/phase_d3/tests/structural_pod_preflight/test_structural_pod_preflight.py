"""The D3 STRUCTURAL pod's blocking CPU gate. Pod-faithful, seconds.

`SetupManifest.test_paths` for the structural session points HERE. What
belongs here is what a mis-staged or mis-checked-out structural pod would get
wrong, checkable on its CPU before the teacher is loaded:

* the bindings agree: the registration verifies its own hash, and the control
  plan and control record it binds are the bytes on this checkout;
* the registered treatment re-derives HERE -- the staged calibration mixtures
  give q2's three recorded untreated config hashes and the registered DEPTH
  one -- and differs from the control only as declared;
* the core's declared KL reference behaves on this interpreter: on a toy model
  at the root, the root reference reproduces the parent path exactly;
* the governance and dispatch surface loads and refuses what it must.

DELIBERATELY ABSENT: the teacher (the driver verifies its staged shards), and
anything touching the network or the dev host.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[6]
for _extra in ("src", "scripts", "scripts/stages/stage-1", "tests"):
    if str(REPO / _extra) not in sys.path:
        sys.path.insert(0, str(REPO / _extra))

from stages.phase_d3 import d3_design as D  # noqa: E402


class TestTheBindingsAgree:

    def test_the_registration_verifies_its_own_hash(self):
        assert len(D.registration_hash(REPO)) == 64

    def test_the_control_plan_and_record_are_the_bound_bytes(self):
        reg = D.registration(REPO)["control"]
        for rel, want in ((reg["replay_plan"], reg["replay_plan_sha256"]),
                          (reg["control_record"],
                           reg["control_record_sha256"])):
            assert hashlib.sha256((REPO / rel).read_bytes()).hexdigest() == want

    def test_the_control_is_the_frozen_q2(self):
        leaf = D.control_leaf(REPO)
        assert leaf.state_id == D.CONTROL_STATE_ID
        assert leaf.expected_artifact_digest == D.CONTROL_ARTIFACT_DIGEST


@pytest.fixture(scope="module")
def spec():
    return D.registered_treatment_spec(REPO)


class TestTheTreatmentReDerivesHere:

    def test_the_spec_is_the_registered_one(self, spec):
        assert spec.spec_hash == \
            D.registration(REPO)["treatment"]["treatment_path_hash"]

    def test_every_operator_config_derives_to_its_pin(self, spec):
        rows = D.operator_config_hashes(spec, REPO)
        assert all(r["config_hash"] == r["pinned"] for r in rows), rows

    def test_it_differs_from_the_control_only_as_declared(self, spec):
        contract = D.treatment_contract(
            D.control_spec(D.control_leaf(REPO)), spec)
        assert {d["step"] for d in contract["differences"]} <= {1, 2, 3}
        assert all(d["declared"] for d in contract["differences"])


class TestTheDeclaredReferenceOnThisInterpreter:

    def test_at_the_root_the_root_reference_is_the_parent_path(self):
        torch = pytest.importorskip("torch")
        from support.toy import TEACHER_GEOMETRY, build_tiny_model, make_items
        from aadistill.initialization.adapters import register_builtin_adapters
        from aadistill.initialization.calibration.items import (
            prepare_calibration_items,
        )
        from aadistill.initialization.execution import ExecutionConfig
        from aadistill.initialization.operators.base import OperatorContext
        from aadistill.initialization.operators.depth.causal_kl_greedy import (
            DepthCausalKLGreedyV1,
        )
        from aadistill.initialization.operators.register import (
            register_builtin_operators,
        )
        from aadistill.initialization.scoring.kl_reference import (
            KL_REFERENCE_KEY, ROOT_REFERENCE_V1,
        )
        from aadistill.initialization.scoring.positions import (
            SUPERVISED_TARGET_V1, policy_config,
        )
        from aadistill.initialization.scoring.reference_store import (
            build_root_reference_store, root_reference_identity,
            store_execution,
        )
        from aadistill.initialization.scoring.support import reference_topk_tail
        from aadistill.initialization.specs.arch import ArchSpec, get_adapter

        register_builtin_adapters()
        register_builtin_operators()
        geometry = dict(TEACHER_GEOMETRY)
        model = build_tiny_model(geometry)
        model.config.use_cache = False
        items = prepare_calibration_items(make_items(), profile_id="toy@v1")
        support = reference_topk_tail(8)
        ex = ExecutionConfig(micro_batch_size=3,
                             calibration_batch_packing="length_sorted_v1")
        store = build_root_reference_store(
            model, items, support=support,
            identity=root_reference_identity(
                root_identity={"teacher_id": "toy"}, profile_hash="p" * 64,
                support=support, execution=store_execution(
                    micro_batch_size=3, packing="length_sorted_v1",
                    device="cpu", model=model)),
            micro_batch_size=3, packing="length_sorted_v1", device="cpu")

        def run(declared):
            config = {"n_calibration_items": len(items),
                      **policy_config(SUPERVISED_TARGET_V1),
                      "distribution_support": support.as_dict(),
                      **({KL_REFERENCE_KEY: ROOT_REFERENCE_V1}
                         if declared else {})}
            ctx = OperatorContext(
                adapter=get_adapter("qwen3"), model=model,
                parent_spec=ArchSpec.of("qwen3", geometry),
                target_spec=ArchSpec.of("qwen3", {**geometry,
                                                  "num_hidden_layers": 4}),
                profile=None, calibration_items=items, seed=0, device="cpu",
                config=config, distribution_support=support, execution=ex,
                position_policy=SUPERVISED_TARGET_V1,
                root_reference=store if declared else None)
            return DepthCausalKLGreedyV1().execute(ctx)

        with torch.no_grad():
            parent, root = run(False), run(True)
        assert root.trace["removal_order"] == parent.trace["removal_order"]
        assert root.local_metrics.reference == "root_reference"


class TestTheGovernanceAndDispatchSurfaceHolds:

    def test_the_authorization_type_refuses_an_edited_artifact(self, tmp_path):
        from stages.phase_d3.d3_authorization import (
            D3AuthorizationRefused, D3StructuralAuthorization,
        )

        path = tmp_path / "edited.json"
        path.write_text(json.dumps({"authorization_sha256": "x" * 64,
                                    "schema": "wrong"}))
        with pytest.raises(D3AuthorizationRefused):
            D3StructuralAuthorization.load(path)

    def test_the_shell_dispatch_branch_exists_for_this_kind(self):
        src = (REPO / "scripts/shared/pod/autoinit_preflight_setup.sh"
               ).read_text()
        branch = src.split('"$SESSION_KIND" = "d3_structural"', 1)[1]
        branch = branch.split("\nelif", 1)[0]
        assert "D3StructuralAuthorization.load(" in branch
        assert "assert a.authorizes_d3_structural is True" in branch
        for flag in ("authorizes_recovery_training", "authorizes_d3_probes",
                     "authorizes_beam_search", "authorizes_reopen_d1_d2",
                     "authorizes_cancelled_d3", "authorizes_promotion",
                     "authorizes_automatic_followon_start"):
            assert f"assert a.{flag} is False" in branch, flag

    def test_the_pod_entry_points_are_present_and_parse(self):
        import ast

        for rel in ("scripts/stages/stage-1/phase_d3/"
                    "autoinit_d3_structural_driver.py",
                    "scripts/stages/stage-1/phase_d3/d3_design.py",
                    "scripts/stages/stage-1/phase_d3/d3_authorization.py"):
            ast.parse((REPO / rel).read_text())
