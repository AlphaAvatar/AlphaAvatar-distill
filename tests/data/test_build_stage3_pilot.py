"""Invariants that protect the Stage 3 teacher-target 2x2 from silent confounds.

The experiment's whole validity rests on the two arms differing in exactly one
thing. These tests check the properties that, if they broke, would produce a
plausible-looking result measuring something else entirely.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "build_stage3_pilot", REPO_ROOT / "scripts" / "data" / "build_stage3_pilot.py"
)
bsp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bsp)

from aadistill.data.dataset import load_split  # noqa: E402

PROMPT = "Answer from the context. Context: Paris is in France. Question: Where?"


def target_row(sid, group, source, accepted=True, prompt=PROMPT):
    """One row in the shape generate_teacher_answers.py writes."""
    assistant = {"role": "assistant", "content": "France."}
    if accepted:
        assistant["reasoning_content"] = "The context says Paris is in France."
    return {
        "id": sid, "group": group, "source": source, "format": "chat",
        "target_source": "teacher_verified" if accepted else "v1_public",
        "candidate_index": 0, "think_tokens": 11,
        "messages": [{"role": "user", "content": prompt}, assistant],
    }


def public_row(sid, group, source, prompt=PROMPT, answer="France"):
    return {
        "id": sid, "group": group, "source": source, "format": "chat",
        "messages": [{"role": "user", "content": prompt},
                     {"role": "assistant", "content": answer}],
    }


def corpus(n=12):
    """n accepted + 4 fallback prompts, with matching public targets."""
    targets, public = [], {}
    for i in range(n):
        sid = f"squad_v2-{i:06d}"
        targets.append(target_row(sid, "rag_evidence", "squad_v2"))
        public[sid] = public_row(sid, "rag_evidence", "squad_v2")
    for i in range(4):
        sid = f"gsm8k-{i:06d}"
        targets.append(target_row(sid, "code_math", "gsm8k", accepted=False))
        public[sid] = public_row(sid, "code_math", "gsm8k")
    return targets, public


def test_only_accepted_prompts_enter_either_arm():
    """A public-fallback prompt in the treatment arm would contaminate it with
    control data and turn the comparison into a mixture ablation."""
    targets, public = corpus()
    arms, ids, _ = bsp.build_arms(targets, public)
    assert len(ids) == 12
    assert all(i.startswith("squad_v2-") for i in ids)
    for arm in bsp.ARMS:
        assert len(arms[arm]) == 12
        assert not any(s["id"].startswith("gsm8k-") for s in arms[arm])


def test_arms_share_prompt_set_and_differ_only_in_assistant_turn():
    targets, public = corpus()
    arms, _, _ = bsp.build_arms(targets, public)
    ctrl = {s["id"]: s for s in arms["control"]}
    treat = {s["id"]: s for s in arms["treatment"]}
    assert ctrl.keys() == treat.keys()
    for sid in ctrl:
        cu = [m for m in ctrl[sid]["messages"] if m["role"] != "assistant"]
        tu = [m for m in treat[sid]["messages"] if m["role"] != "assistant"]
        # Byte-identical prompt prefix: the teacher must have answered in the
        # same position the student is trained on.
        assert cu == tu
        assert ctrl[sid]["group"] == treat[sid]["group"]
        c_asst = [m for m in ctrl[sid]["messages"] if m["role"] == "assistant"]
        t_asst = [m for m in treat[sid]["messages"] if m["role"] == "assistant"]
        assert "reasoning_content" not in c_asst[0]
        assert t_asst[0]["reasoning_content"]


def test_missing_public_target_fails_loudly():
    """Without a public target the control arm cannot cover the same prompt
    set, so the comparison is impossible — that must not degrade quietly."""
    targets, public = corpus()
    public.pop("squad_v2-000003")
    with pytest.raises(SystemExit, match="no public target"):
        bsp.build_arms(targets, public)


def test_public_target_carrying_a_trace_is_rejected():
    targets, public = corpus()
    public["squad_v2-000002"]["messages"][1]["reasoning_content"] = "hmm"
    with pytest.raises(SystemExit, match="already carries a reasoning trace"):
        bsp.build_arms(targets, public)


def test_duplicate_prompt_is_rejected():
    targets, public = corpus()
    targets.append(target_row("squad_v2-000000", "rag_evidence", "squad_v2"))
    with pytest.raises(SystemExit, match="duplicate accepted prompt"):
        bsp.build_arms(targets, public)


def test_no_accepted_targets_fails_loudly():
    targets, public = corpus(n=0)
    with pytest.raises(SystemExit, match="no accepted teacher targets"):
        bsp.build_arms(targets, public)


def test_out_of_scope_group_fails_loudly():
    """`refusal_uncertainty` is evaluation-only (scope frozen 2026-07-30). The
    2026-07-29 pilot corpus predates that and contains refusal rows, so a build
    that silently trained on them would widen the declared capability scope."""
    targets, public = corpus()
    sid = "squad_v2-900000"
    targets.append(target_row(sid, "refusal_uncertainty", "squad_v2"))
    public[sid] = public_row(sid, "refusal_uncertainty", "squad_v2")
    with pytest.raises(SystemExit, match="out-of-scope"):
        bsp.build_arms(targets, public)


def test_out_of_scope_group_can_be_dropped_and_is_recorded():
    targets, public = corpus()
    sid = "squad_v2-900000"
    targets.append(target_row(sid, "refusal_uncertainty", "squad_v2"))
    public[sid] = public_row(sid, "refusal_uncertainty", "squad_v2")
    arms, ids, scope = bsp.build_arms(
        targets, public, bsp.IN_SCOPE_GROUPS, drop_out_of_scope=True
    )
    assert sid not in ids
    assert scope["dropped_out_of_scope"] == {"refusal_uncertainty": 1}
    for arm in bsp.ARMS:
        assert all(s["group"] in bsp.IN_SCOPE_GROUPS for s in arms[arm])


def test_dropping_everything_fails_rather_than_writing_an_empty_arm():
    targets, public = [], {}
    sid = "squad_v2-900000"
    targets.append(target_row(sid, "refusal_uncertainty", "squad_v2"))
    public[sid] = public_row(sid, "refusal_uncertainty", "squad_v2")
    with pytest.raises(SystemExit, match="every accepted prompt was out of scope"):
        bsp.build_arms(targets, public, bsp.IN_SCOPE_GROUPS, drop_out_of_scope=True)


def test_val_assignment_is_arm_independent_and_seed_free():
    """Both arms must hold the same prompts in val, or their val CE — the R4
    abort guard rail — is not comparable across arms."""
    ids = [f"squad_v2-{i:06d}" for i in range(400)]
    first = [bsp.val_bucket(i, 0.1) for i in ids]
    second = [bsp.val_bucket(i, 0.1) for i in ids]
    assert first == second                      # deterministic
    assert 0.03 < sum(first) / len(first) < 0.2  # roughly the requested rate


def test_written_arms_load_and_hold_identical_split_membership(tmp_path):
    targets, public = corpus(n=40)
    arms, _, _ = bsp.build_arms(targets, public)
    for arm in bsp.ARMS:
        bsp.write_arm(tmp_path / arm, arms[arm], val_frac=0.2)

    for split in ("train", "val"):
        per_arm = {}
        for arm in bsp.ARMS:
            groups = load_split(tmp_path / arm, split)   # validates every sample
            per_arm[arm] = {s["id"] for rows in groups.values() for s in rows}
        assert per_arm["control"] == per_arm["treatment"]
        assert per_arm["control"]


def test_empty_val_split_fails_loudly(tmp_path):
    targets, public = corpus(n=6)
    arms, _, _ = bsp.build_arms(targets, public)
    with pytest.raises(SystemExit, match="empty val split"):
        bsp.write_arm(tmp_path / "control", arms["control"], val_frac=0.0)


def test_output_files_are_reproducible_byte_for_byte(tmp_path):
    """Reruns must not reorder records: the manifest hashes files, and a churn
    would make an unchanged corpus look like a new one (P4)."""
    targets, public = corpus(n=30)
    arms, _, _ = bsp.build_arms(targets, public)
    hashes = []
    for run in ("a", "b"):
        root = tmp_path / run
        bsp.write_arm(root, arms["treatment"], val_frac=0.2)
        hashes.append(sorted(
            (p.relative_to(root).as_posix(), p.read_bytes())
            for p in root.rglob("*.jsonl")
        ))
    assert hashes[0] == hashes[1]


def test_group_files_are_named_by_group(tmp_path):
    """load_split requires filename == group; a mismatch raises there, but the
    builder should not produce it in the first place."""
    targets, public = corpus(n=20)
    for i in range(6):
        sid = f"gsm8k-1{i:05d}"
        targets.append(target_row(sid, "code_math", "gsm8k"))
        public[sid] = public_row(sid, "code_math", "gsm8k")
    arms, _, _ = bsp.build_arms(targets, public)
    counts = bsp.write_arm(tmp_path / "treatment", arms["treatment"], val_frac=0.2)
    names = {p.stem for p in (tmp_path / "treatment").rglob("*.jsonl")}
    assert names <= {"rag_evidence", "code_math"}
    total = counts["train"]["total"] + counts["val"]["total"]
    assert total == 26
    for row in (tmp_path / "treatment" / "train" / "code_math.jsonl").read_text(
    ).splitlines():
        assert json.loads(row)["group"] == "code_math"
