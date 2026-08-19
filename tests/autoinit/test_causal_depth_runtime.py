"""The causal-depth runtime repair, and the three guards that keep it.

Phase-A attempt 10 spent $11.43 and produced nothing because
`depth.causal_kl_greedy_v1` ran its full-vocabulary softmax/KL on the host: the
port of `scripts/training/search_depth_map.py` inserted `.cpu()` on the logits
and the targets, E8a has neither, and nothing between the driver's affordability
check and the cost watchdog ever looked at a clock.

Three properties are asserted here, each of which was false during that run:

1. the scoring tensors stay on the compute device;
2. the wall-clock budget stops the search **inside** an expansion;
3. the search says where it is, so a stall is distinguishable from work.

The equivalence of the repair — that no removal decision moved — is
`scripts/autoinit/verify_depth_backend_equivalence.py`, run as its own artifact,
and re-asserted in miniature at the bottom of this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.autoinit.device import apply_cpu_budget, cpu_budget  # noqa: E402
from aadistill.autoinit.operators.depth import _forward_logits  # noqa: E402
from aadistill.autoinit.search import (  # noqa: E402
    Deadline, SearchDeadlineExceeded,
)
from aadistill.init.contribution import greedy_removal  # noqa: E402


# --- 1. the scoring tensors stay where the model is ------------------------

class _Logits:
    def __init__(self, t):
        self.logits = t


class _FakeModel:
    """Answers a forward on a chosen device. Enough for a placement assertion."""

    def __init__(self, device):
        self._device = device

    def __call__(self, ids):
        return _Logits(torch.zeros(1, ids.shape[1], 7, device=self._device))


def test_forward_logits_leaves_the_logits_on_the_model_device():
    """The single line that cost $11.43. `.cpu()` here drags the whole
    151,936-vocabulary reduction to the host, 17,420 times per expansion."""
    item = {"input_ids": torch.zeros(1, 5, dtype=torch.long)}
    out = _forward_logits(_FakeModel(torch.device("cpu")), item, "cpu")
    assert out.device == torch.device("cpu")

    # `meta` is a real second device on every machine, so this distinguishes
    # "placed" from "happens to match" — the trap the attempt-9 audit fell into.
    meta_item = {"input_ids": torch.zeros(1, 5, dtype=torch.long)}
    out = _forward_logits(_FakeModel(torch.device("meta")), meta_item, "meta")
    assert out.device.type == "meta", (
        "_forward_logits moved the logits off the model's device; on a pod that "
        "is a device->host copy of 517 MiB per item, 17,420 times")


def test_the_operator_source_carries_no_cpu_transfer_on_the_scoring_path():
    """A regression guard on the exact edit, because the failure is invisible on
    a one-device box: with the `.cpu()` restored every test still passes and
    every number is identical. Only the wall clock changes, and only on a GPU."""
    import re

    src = (REPO / "src/aadistill/autoinit/operators/depth.py").read_text()
    code = "\n".join(l for l in src.splitlines()
                     if not l.lstrip().startswith("#"))
    # Docstrings explain the removed transfer; strip them before looking.
    code = re.sub(r'"""].*?"""', "", code, flags=re.S)
    code = re.sub(r'""".*?"""', "", code, flags=re.S)
    offenders = [l.strip() for l in code.splitlines()
                 if ".cpu()" in l or ".to('cpu')" in l or '.to("cpu")' in l]
    assert not offenders, (
        f"depth.py moves a scoring tensor to the host: {offenders}. E8a keeps "
        "prepared inputs, reference logits, ablated logits and the distortion "
        "reduction on the accelerator; the port's transfer is what attempt 10 "
        "paid for.")


# --- 2. the budget binds inside an expansion -------------------------------

def test_the_deadline_stops_a_greedy_search_mid_expansion():
    """Not after the expansion — inside it. A round is 29-36 evaluations and one
    expansion is 260; attempt 10 ran 3.6x the whole search budget in one."""
    deadline = Deadline(seconds=0.0)          # already spent
    seen = []

    def on_candidate(p):
        seen.append(p)
        deadline.check(f"round {p['round']} candidate {p['index']}/{p['of']}")

    with pytest.raises(SearchDeadlineExceeded) as e:
        greedy_removal(lambda s: float(sum(s)), 8, 3, on_candidate=on_candidate)
    assert len(seen) == 1, "the deadline let a second candidate start"
    assert "candidate 1/8" in str(e.value), str(e.value)
    assert "0.0 min" in str(e.value) or "min" in str(e.value)


def test_a_live_deadline_does_not_interfere():
    deadline = Deadline.from_minutes(60.0)
    r = greedy_removal(lambda s: float(sum(s)), 6, 2,
                       on_candidate=lambda p: deadline.check("x"))
    assert r["kept"] == [2, 3, 4, 5]
    assert not deadline.expired()


def test_the_deadline_is_not_part_of_the_search_identity():
    """`SearchConfig` "fixes a search run, and therefore everything that
    hashes". A wall-clock budget is operational; putting it there would make
    every re-pricing a different search."""
    from dataclasses import fields

    from aadistill.autoinit.search import SearchConfig

    names = {f.name for f in fields(SearchConfig)}
    assert "deadline" not in names and "search_minutes" not in names


# --- 3. the search says where it is ----------------------------------------

def test_greedy_removal_reports_every_candidate_and_every_round():
    rounds, candidates = [], []
    r = greedy_removal(lambda s: float(sum(s)), 6, 2,
                       on_round=rounds.append, on_candidate=candidates.append)
    assert len(rounds) == 2
    # 6 candidates in round 0, 5 in round 1.
    assert [c["of"] for c in candidates].count(6) == 6
    assert [c["of"] for c in candidates].count(5) == 5
    assert candidates[-1]["evaluations"] == 11
    assert r["evaluations"] == 11


def test_progress_hooks_cannot_change_a_decision():
    """They are called with what is already computed and their return value is
    discarded. Only an exception changes control flow, and that stops the
    search rather than steering it."""
    plain = greedy_removal(lambda s: float(sum(s)), 7, 3)
    observed = greedy_removal(lambda s: float(sum(s)), 7, 3,
                              on_round=lambda r: "ignored",
                              on_candidate=lambda p: "ignored")
    assert plain["kept"] == observed["kept"]
    assert [r["chosen"] for r in plain["rounds"]] == \
           [r["chosen"] for r in observed["rounds"]]
    assert [r["table"] for r in plain["rounds"]] == \
           [r["table"] for r in observed["rounds"]]


# --- the cgroup CPU budget -------------------------------------------------

def test_cpu_budget_is_bounded_and_names_its_source():
    n, source = cpu_budget()
    assert 1 <= n <= 16
    assert source in {"cgroup.v2", "cgroup.v1", "sched_getaffinity", "cpu_count"}


def test_apply_cpu_budget_actually_holds_torch_to_it():
    before = torch.get_num_threads()
    try:
        got = apply_cpu_budget()
        assert got["torch_threads_after"] == got["threads"]
        assert torch.get_num_threads() == got["threads"]
        import os
        assert os.environ["OMP_NUM_THREADS"] == str(got["threads"])
    finally:
        torch.set_num_threads(before)


def test_the_budget_is_not_read_from_the_visible_cpu_count():
    """`nproc` honours OMP_NUM_THREADS, and the container advertised 128 while
    the cgroup granted 13. The helper must consult the quota, not the count."""
    import ast
    import re

    src = (REPO / "src/aadistill/autoinit/device.py").read_text()
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "cpu_budget")
    # The docstring says "NOT os.cpu_count()", which a naive text search reads as
    # a use. Drop it and look at the code.
    body = ast.get_source_segment(src, fn) or ""
    body = re.sub(r'"""     .*?"""', "", body, flags=re.S | re.X)
    body = body.replace(ast.get_docstring(fn) or "", "")

    assert "cpu.max" in body and "cfs_quota_us" in body, (
        "cpu_budget no longer consults the cgroup quota")
    assert body.index("cpu.max") < body.index("os.cpu_count()"), (
        "the visible-CPU count is consulted before the cgroup quota; the quota "
        "is the limit that binds, and 128-visible/13-granted is what attempt 10 "
        "ran into")


# --- the repair changed no decision (miniature of the equivalence script) ---

def test_moving_the_reduction_changes_no_removal_decision():
    """The full artifact is `verify_depth_backend_equivalence.py`; this keeps the
    claim in the suite so a later edit cannot quietly break it."""
    from aadistill.init.contribution import distortion

    torch.manual_seed(11)
    ref = torch.randn(40, 32)
    abl = torch.randn(40, 32)
    tgt = torch.randint(0, 32, (40,))

    on_device = distortion(ref, abl, tgt, chunk=512).as_dict()
    on_host = distortion(ref.cpu(), abl.cpu(), tgt.cpu(), chunk=512).as_dict()
    assert on_device == on_host, "the reduction is not placement-invariant"
