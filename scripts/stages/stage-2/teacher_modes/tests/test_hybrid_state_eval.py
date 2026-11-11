"""The hybrid state-evaluation binding refuses a suite of another root, another type table or other items."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
for _p in ("src", "scripts", "scripts/stages/stage-2"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.teacher_modes import hybrid_state_eval as HSE  # noqa: E402

TEACHER = {"model": "org/hybrid", "revision": "r" * 40, "weight_shards": [{"lfs_sha256": "a" * 64}, {"lfs_sha256": "b" * 64}],
           "chat_template_sha256": "c" * 64, "vocabulary_sha256": "v" * 64}


def _manifest(**over):
    m = {"root_teacher": {"model": "org/hybrid", "revision": "r" * 40, "weight_shards_sha256": ["b" * 64, "a" * 64],
                          "chat_template_sha256": "c" * 64, "vocabulary_sha256": "v" * 64},
         "modes": {"H-T": {}, "H-N": {}}, "critical_groups": dict(HSE.CRITICAL_GROUPS), "types_sha256": "t" * 64,
         "items_sha256": "i" * 64, "content_sha256": "x" * 64}
    m.update(over)
    return m


def test_the_binding_accepts_its_own_suite_and_refuses_another_root_table_or_items():
    assert HSE.binding_problems(_manifest(), TEACHER, "t" * 64, "i" * 64) == []
    other = _manifest(root_teacher={**_manifest()["root_teacher"], "model": "org/thinking-2507"})
    assert any("root_teacher.model" in p for p in HSE.binding_problems(other, TEACHER, "t" * 64, "i" * 64))
    assert any("type table" in p for p in HSE.binding_problems(_manifest(), TEACHER, "u" * 64, "i" * 64))
    assert any("items.jsonl" in p for p in HSE.binding_problems(_manifest(), TEACHER, "t" * 64, "j" * 64))
    one_mode = _manifest(modes={"H-T": {}})
    assert any("both modes" in p for p in HSE.binding_problems(one_mode, TEACHER, "t" * 64, "i" * 64))


def test_the_binding_refuses_a_suite_without_one_critical_group_per_mode():
    """Without groups the evaluator averages all five (mode, class) pairs: thinking 3/5, non-thinking 2/5."""
    for groups in (None, {"H-T": ["think_close", "final_answer", "eos", "final_answer", "eos"]},
                   {"H-T": ["final_answer", "eos"], "H-N": ["think_close", "final_answer", "eos"]}):
        assert any("critical groups" in p for p in HSE.binding_problems(_manifest(critical_groups=groups), TEACHER,
                                                                         "t" * 64, "i" * 64))
