"""Execute `run_phase_a_search` for real, at toy scale. Zero cost, CPU only.

Not a simulation and not a stub. This calls the **same function the pod calls**,
with a real (32-wide, 6-layer) Qwen3 teacher, the real operators, real
checkpoints written to disk, the real `from_pretrained` reload, real hashing,
real measurement against the real policy, and the real `make_control_state`
injection. Only the model is scaled down.

It exists because the wrapper's own lines had never run. `dry_run_search.py`
exercises `BeamSearch`, but not this function — and this project has lost four
paid pods inside lines that no test had ever executed, including one that died
with `KeyError: 'metrics'` after both models had loaded.
"""

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/autoinit"))

from aadistill.autoinit.arch import ArchSpec, get_adapter  # noqa: E402
from aadistill.autoinit.calibration import (  # noqa: E402
    CalibrationProfile, CalibrationSource,
)
from aadistill.autoinit.metrics import StateEvalSuite, SuiteItem  # noqa: E402
from aadistill.autoinit.recovery import (  # noqa: E402
    RecoveryAdmissionError, admit_leaves, probe_configs,
)

TEACHER_GEOMETRY = dict(hidden_size=32, num_hidden_layers=6, intermediate_size=48,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                        vocab_size=128, tie_word_embeddings=True)
TARGET_GEOMETRY = dict(hidden_size=16, num_hidden_layers=4, intermediate_size=24,
                       num_attention_heads=2, num_key_value_heads=2, head_dim=8,
                       vocab_size=128, tie_word_embeddings=True)
DOMAINS = {"general": ("text",), "math": ("arith",)}


def _teacher(seed=4242):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(
        Qwen3Config(max_position_embeddings=256, rope_theta=5_000_000,
                    **TEACHER_GEOMETRY)).float().eval()
    with torch.no_grad():
        for module in model.modules():
            if module.__class__.__name__ == "Qwen3RMSNorm":
                module.weight.uniform_(0.5, 1.5)
    return model


def _items(seed, n, seq_len=24):
    torch.manual_seed(seed)
    out = []
    for domain, (subtype,) in DOMAINS.items():
        for k in range(n):
            ids = torch.randint(0, TEACHER_GEOMETRY["vocab_size"], (1, seq_len))
            targets = ids[0, 1:]
            out.append({"item_id": f"{subtype}-{k}", "input_ids": ids,
                        "domain": domain, "subtype": subtype,
                        "tags": {"eos_like": targets == 0,
                                 "answer_like": targets % 17 == 0}})
    return out


def _suite_bundle(seed=99):
    suite = StateEvalSuite(
        suite_id="rehearsal.state_eval", version=1, domains=tuple(DOMAINS),
        subtypes=DOMAINS, critical_tags=("eos_like", "answer_like"),
        description="toy stand-in; structure mirrors state_eval_v1")
    items = [SuiteItem(item_id=i["item_id"], input_ids=i["input_ids"],
                       domain=i["domain"], subtype=i["subtype"], tags=i["tags"])
             for i in _items(seed, 2)]
    return suite, items, {"teacher_sha256": "0" * 64}


def _profile():
    return CalibrationProfile(
        profile_id="rehearsal.domain_balanced", version=1,
        description="toy stand-in for calib.domain_balanced@v1",
        sources=tuple(CalibrationSource("rehearsal", "local", d, 2) for d in DOMAINS),
        domain_weights={d: 1.0 / len(DOMAINS) for d in DOMAINS},
        token_budget=1, sample_rule="fixed order", seed=1000)


@pytest.fixture(scope="module")
def searched(tmp_path_factory):
    """One real search, reused by the assertions below (it takes ~a minute)."""
    from phase_a_search import run_phase_a_search

    tmp = tmp_path_factory.mktemp("phase_a_search")
    adapter = get_adapter("qwen3")
    teacher = _teacher()
    target_spec = ArchSpec.of("qwen3", TARGET_GEOMETRY)

    # The "retained canonical control": built once OUTSIDE the search, which is
    # the same relationship the real one has to the real search.
    control_dir = tmp / "canonical_control"
    control_model = adapter.build_model(
        adapter.build_config(teacher.config, target_spec), torch.float32, 4242)
    adapter.save(control_model, str(control_dir))

    return run_phase_a_search(
        workdir=tmp / "search", state_eval=tmp / "unused", top_n=3,
        device="cpu", repo_root=tmp,
        teacher_id="rehearsal-tiny-teacher",
        canonical_init="canonical_control",
        canonical_sha256=None,
        teacher_loader=lambda: teacher,
        target_geometry=TARGET_GEOMETRY,
        suite_bundle=_suite_bundle(),
        calibration_items=_items(7, 2),
        profile=_profile())


def test_the_search_ran_and_produced_measured_leaves(searched):
    assert searched.summary["summary"]["n_states"] > 0
    assert searched.summary["summary"]["n_complete_leaves"] > 0
    assert searched.leaves, "no leaves selected"
    for leaf in searched.leaves:
        # The mandatory cycle: a leaf must carry its own hash-bound measurement
        # or it cannot enter a recovery probe.
        leaf.require_recovery_admissible()
        assert leaf.artifact_digest
        assert leaf.checkpoint_path and Path(leaf.checkpoint_path).is_dir()


def test_the_control_is_injected_measured_and_marked_retained(searched):
    control = searched.control
    assert control.provenance == "retained_canonical"
    assert not control.steps, "the control was produced by no operator of this search"
    # Measured on the SAME suite as the leaves, or its step-0 metrics would not
    # be comparable with theirs.
    control.require_recovery_admissible()
    assert control.evaluation.suite_hash == searched.summary["suite_hash"]


def test_the_summary_is_serializable_and_names_what_a_driver_reads(searched):
    import json

    payload = json.loads(json.dumps(searched.summary, default=str))
    for key in ("suite_hash", "summary", "levels", "top_n", "control",
                "config_hash", "resumed_state_ids"):
        assert key in payload, key
    assert payload["top_n"]["selected"], "the driver reads top_n.selected"
    for entry in payload["top_n"]["selected"]:
        for field in ("state_id", "artifact_digest", "checkpoint_path"):
            assert entry.get(field), field


def test_the_searched_leaves_and_control_pass_the_recovery_gate(searched):
    """`admit_leaves` + `probe_configs` are what the driver's rungs call.

    Running them here means the boundary between the search and the rungs is
    executed at $0 rather than discovered at hour four of a paid session.
    """
    from aadistill.autoinit.recovery import (
        CAPABILITY_SCHEMA_V1, CATASTROPHIC_V1, E1_KD_HEAVY_0860K,
        EquivalenceRule, FeasibilityRule, SuccessiveHalvingPlan,
    )

    leaves = list(searched.leaves)
    plan = SuccessiveHalvingPlan(
        plan_id="rehearsal.phase_a", recipe=E1_KD_HEAVY_0860K,
        searched_leaves=len(leaves), survivors=max(1, len(leaves) - 1),
        feasibility_min=0.0,
        equivalence=EquivalenceRule(n_pooled=340),
        feasibility=FeasibilityRule(n_pooled=380),
        catastrophic=CATASTROPHIC_V1, capability_schema=CAPABILITY_SCHEMA_V1,
        survivor_rule="rehearsal", winner_rule="rehearsal",
        battery_asset_id="recovery_search_v2")

    admitted = admit_leaves([*leaves, searched.control], plan)
    assert len(admitted) == len(leaves) + 1

    probes = probe_configs(admitted, plan, rung=1)
    assert len(probes) == len(leaves) + 1
    assert sum(1 for p in probes if p["is_control"]) == 1, (
        "exactly one probe must be the control")
    for probe in probes:
        assert probe["seed"] == 20260726, "rung 1 is seed sa"
        assert probe["student_checkpoint"], "a probe with no student trains nothing"


def test_a_control_that_is_not_the_retained_checkpoint_is_refused(tmp_path):
    """The frozen-hash injection, exercised. A control that is not the retained
    checkpoint is not a control."""
    from aadistill.autoinit.artifact import identify_checkpoint
    from aadistill.autoinit.state import StateError, make_control_state

    adapter = get_adapter("qwen3")
    teacher = _teacher()
    target_spec = ArchSpec.of("qwen3", TARGET_GEOMETRY)
    d = tmp_path / "ckpt"
    adapter.save(adapter.build_model(
        adapter.build_config(teacher.config, target_spec), torch.float32, 1), str(d))
    artifact = identify_checkpoint(d, adapter=adapter, spec=target_spec,
                                   num_parameters=adapter.param_count(target_spec))
    with pytest.raises(StateError, match="not the retained checkpoint"):
        make_control_state(
            control_id="wrong", artifact=artifact, spec=target_spec,
            target_spec=target_spec,
            num_parameters=adapter.param_count(target_spec),
            root_teacher_id="t", root_teacher_sha256="0" * 64,
            description="a checkpoint that is not the frozen one",
            expected_single_file_sha256="0" * 64)


def test_the_search_resumes_without_repeating_a_measured_state(tmp_path):
    """Resume is the difference between losing a pod and losing an hour."""
    from phase_a_search import run_phase_a_search

    adapter = get_adapter("qwen3")
    teacher = _teacher()
    target_spec = ArchSpec.of("qwen3", TARGET_GEOMETRY)
    control_dir = tmp_path / "canonical_control"
    adapter.save(adapter.build_model(
        adapter.build_config(teacher.config, target_spec), torch.float32, 4242),
        str(control_dir))

    kwargs = dict(
        state_eval=tmp_path / "unused", top_n=2, device="cpu", repo_root=tmp_path,
        teacher_id="rehearsal-tiny-teacher", canonical_init="canonical_control",
        canonical_sha256=None, teacher_loader=lambda: teacher,
        target_geometry=TARGET_GEOMETRY, suite_bundle=_suite_bundle(),
        calibration_items=_items(7, 2), profile=_profile())

    first = run_phase_a_search(workdir=tmp_path / "search", **kwargs)
    assert not first.summary["resumed_state_ids"], "a fresh run resumed something"

    second = run_phase_a_search(workdir=tmp_path / "search", **kwargs)
    assert second.summary["resumed_state_ids"], (
        "the second pass re-measured every state; a lost pod would cost the "
        "whole search again")
    assert (second.summary["summary"]["n_states"]
            == first.summary["summary"]["n_states"])
