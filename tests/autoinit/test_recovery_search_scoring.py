"""The Stage-3 scorer, run against known-bad policies on the REAL frozen battery.

Every scorer this project has trusted without this check had a defect. Three were
caught this way; a fourth (`score_battery.py` cannot read `recovery_search_v1` at
all — different manifest schema, different set list) was caught while wiring the
preflight. So the recovery-search scorer is validated before a pod is booked,
against the actual 190 frozen prompts rather than a fixture, using policies whose
correct verdicts are known in advance:

* **contentless-but-perfect** — `<think>ok</think>42`, terminates naturally.
  Every behaviour component passes and almost nothing is correct. This is the
  policy that proves `usable_rollout` is blind to correctness, which is precisely
  why correctness must never be folded into it.
* **oracle-then-loop** — the gold answer, then a context-limit hit. The scorer
  finds the right answer; `correct` must still be 0, because
  `score_recovery_row` defines correctness as *correct in a usable rollout*.
* **empty** — nothing. Every component fails.
* **degenerate** — a repetition loop caught by the degeneration stop.
* **oracle** — the gold answer, well formed. The upper bound: correctness should
  be high, and it must never exceed usable.

The last one also guards the direction of the whole instrument: if a policy that
emits the gold answer scores near zero, the scorer is broken, not the policy.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BATTERY = REPO / "artifacts/stage3/recovery_search_v1"
sys.path.insert(0, str(REPO / "src"))

pytestmark = pytest.mark.skipif(
    not (BATTERY / "manifest.json").is_file(),
    reason="recovery_search_v1 is a local artifact, not tracked in git")


def gold_answer(set_name: str, sample: dict) -> str:
    """What a perfect policy would say for this item, per set."""
    if set_name == "gsm8k":
        # A standalone marker line, which is what the strict extractor requires:
        # "the answer is 7" in prose is deliberately not a stated conclusion.
        return f"Final Answer: {sample['gsm8k_answer']}"
    if set_name == "math_verified":
        return f"So \\boxed{{{sample['boxed']}}}."
    if set_name == "knowledge":
        return f"Answer: {sample['aliases'][0]}"
    if set_name == "multihop":
        titles = ", ".join(sample.get("supporting_titles") or [])
        return f"According to {titles}, the answer is {sample['answer']}."
    if set_name == "rag":
        return f"The context states: \"{sample['gold']}\". Answer: {sample['gold']}."
    if set_name == "tool":
        calls = sample["reference_calls"]
        if isinstance(calls, str):
            calls = json.loads(calls)
        emitted = "\n".join(
            "<tool_call>\n"
            + json.dumps({"name": c.get("name") or c.get("function", {}).get("name"),
                          "arguments": c.get("arguments")
                          or c.get("function", {}).get("arguments") or {}})
            + "\n</tool_call>" for c in calls)
        return emitted
    return "def solve():\n    return 1\n"


POLICIES = {
    "contentless_perfect": dict(answer=lambda s, x: "42", terminates=True,
                                degenerate=False, context_limit=False),
    "oracle": dict(answer=gold_answer, terminates=True, degenerate=False,
                   context_limit=False),
    "oracle_then_loop": dict(answer=gold_answer, terminates=False,
                             degenerate=False, context_limit=True),
    "empty": dict(answer=lambda s, x: "", terminates=False, degenerate=False,
                  context_limit=True),
    "degenerate": dict(answer=lambda s, x: "the the the " * 200, terminates=False,
                       degenerate=True, context_limit=False),
}


def write_generations(out_dir: Path, policy: str) -> None:
    spec = POLICIES[policy]
    out_dir.mkdir(parents=True, exist_ok=True)
    for set_path in sorted(BATTERY.glob("*.jsonl")):
        name = set_path.stem
        with (out_dir / f"{name}.generations.jsonl").open("w") as f:
            for line in set_path.read_text().splitlines():
                if not line.strip():
                    continue
                sample = json.loads(line)
                answer = spec["answer"](name, sample)
                # The template pre-opens <think>, as it does in the real harness,
                # so the generation continues from inside it. `<|im_end|>` is
                # present exactly when the policy terminates naturally, because
                # `uncapped_eval` decodes with `skip_special_tokens=False` and
                # `protocol_valid` keys termination off that token.
                raw = f"reasoning.</think>\n{answer}"
                if spec["terminates"]:
                    raw += "<|im_end|>"
                f.write(json.dumps({
                    "id": sample["id"], "label": policy,
                    "group": sample.get("group"), "source": sample.get("source"),
                    "prompt_tokens": 100, "think_preopened": True,
                    "generated_tokens": 50,
                    "natural_termination": spec["terminates"],
                    "degeneration_triggered": spec["degenerate"],
                    "context_limit_reached": spec["context_limit"],
                    "stop_reason": ("eos" if spec["terminates"] else
                                    "degeneration" if spec["degenerate"]
                                    else "context_limit"),
                    "raw": raw,
                }) + "\n")


def score(tmp_path: Path, policy: str) -> dict:
    gen = tmp_path / policy
    write_generations(gen, policy)
    out = tmp_path / f"{policy}.json"
    rc = subprocess.run(
        [sys.executable, str(REPO / "scripts/autoinit/score_recovery_search.py"),
         "--generations", str(gen), "--label", policy, "--seed", "20260726",
         "--out", str(out)],
        capture_output=True, text=True, cwd=REPO,
        env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin",
             "HOME": str(tmp_path)})
    assert rc.returncode == 0, rc.stdout + rc.stderr
    return json.loads(out.read_text())


@pytest.fixture(scope="module")
def scored(tmp_path_factory) -> dict:
    tmp = tmp_path_factory.mktemp("recovery_search_scoring")
    return {policy: score(tmp, policy) for policy in POLICIES}


def test_a_contentless_policy_is_behaviourally_perfect_and_almost_never_correct(scored):
    """The whole reason correctness is a separate axis."""
    result = scored["contentless_perfect"]
    assert result["usable_rollout_rate"] == 1.0
    for component in ("non_empty", "natural_termination", "no_severe_repetition",
                      "no_context_limit", "protocol_valid"):
        assert result[component] == 1.0, component
    assert result["correct_overall"] < 0.10, (
        "a two-word contentless answer is scoring as correct; the correctness "
        "axis is measuring format, not content")


def test_correct_implies_usable_by_construction(scored):
    """The gold answer inside an unusable rollout counts for nothing."""
    loop = scored["oracle_then_loop"]
    assert loop["usable_rollout_rate"] == 0.0
    assert loop["correct"] == 0
    assert loop["correct_overall"] == 0.0
    assert loop["correct_given_usable"] is None, (
        "a checkpoint with no usable rollout has no conditional accuracy; "
        "reporting 0.0 would make it look measured")
    # The scorer did find the answer — that is what makes this a real check
    # rather than a tautology about an empty output.
    assert loop["scoring_contract"]["correct_but_unusable"] > 0
    for result in scored.values():
        assert result["correct"] <= result["usable"], result["label"]


def test_the_oracle_scores_high_so_the_instrument_points_the_right_way(scored):
    result = scored["oracle"]
    assert result["usable_rollout_rate"] == 1.0
    assert result["correct_overall"] > 0.60, (
        "a policy that emits the gold answer scores below 0.60 — the scorer is "
        "broken, not the policy")
    for name in ("gsm8k", "math_verified", "knowledge", "multihop", "tool"):
        assert result["per_capability"][name]["correct_overall"] > 0.5, name


def test_an_empty_and_a_degenerate_policy_both_fail_every_component(scored):
    empty, degen = scored["empty"], scored["degenerate"]
    assert empty["usable_rollout_rate"] == 0.0 and empty["non_empty"] == 0.0
    assert degen["usable_rollout_rate"] == 0.0
    assert degen["no_severe_repetition"] == 0.0
    # The census attributes each failure to one component and sums to the total.
    for result in (empty, degen):
        assert sum(result["first_failure"].values()) == result["n"]


def test_code_is_behaviour_only_and_never_enters_correctness(scored):
    result = scored["oracle"]
    assert result["n"] == 190 and result["n_scorable"] == 170
    assert result["per_set"]["code"]["n"] == 20
    assert result["per_set"]["code"]["correct"] == 0
    assert result["per_set"]["code"]["correct_overall"] is None
    assert result["per_set"]["code"]["usable_rollout_rate"] == 1.0, (
        "code prompts still contribute behaviour")
    assert "code" not in result["per_capability"]


def test_the_capability_schema_is_enforced_not_defaulted(scored):
    from aadistill.autoinit.recovery import CAPABILITY_SCHEMA_V1, CapabilitySchemaError

    result = scored["oracle"]
    assert result["capability_schema_enforced"] is True
    assert set(result["per_capability"]) == set(CAPABILITY_SCHEMA_V1.expected)
    # Drop one capability and the schema must raise rather than pass it.
    broken = {**result, "per_capability": {
        k: v for k, v in result["per_capability"].items() if k != "tool"}}
    with pytest.raises(CapabilitySchemaError, match="tool"):
        CAPABILITY_SCHEMA_V1.validate(broken, label="broken")


def test_a_short_set_is_refused_rather_than_inflating_a_rate(tmp_path):
    gen = tmp_path / "short"
    write_generations(gen, "oracle")
    path = gen / "gsm8k.generations.jsonl"
    kept = path.read_text().splitlines()[:5]
    path.write_text("\n".join(kept) + "\n")
    rc = subprocess.run(
        [sys.executable, str(REPO / "scripts/autoinit/score_recovery_search.py"),
         "--generations", str(gen), "--label", "short", "--seed", "1",
         "--out", str(tmp_path / "short.json")],
        capture_output=True, text=True, cwd=REPO,
        env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin",
             "HOME": str(tmp_path)})
    assert rc.returncode != 0
    assert "have no generation" in rc.stdout + rc.stderr


def test_a_correct_tool_call_is_a_usable_rollout(scored):
    """The defect that would have zeroed one of six capabilities on every arm.

    `protocol_valid`'s `unexpected_tool_call` rule is right for a model that
    invents a tool call unprompted, and wrong for a prompt that declared tools:
    it made a correct, well-terminated tool call structurally invalid, so
    `usable` was false, so `correct => usable` forced correctness to 0. Every
    candidate and every control would have scored exactly 0 on `tool`, and the
    Stage-3 characterization would have frozen thresholds derived from it.
    """
    tool = scored["oracle"]["per_capability"]["tool"]
    assert tool["n"] == 20
    assert tool["protocol_valid"] == 1.0, (
        "a well-formed tool call is being scored as a protocol violation")
    assert tool["usable_rollout_rate"] == 1.0
    assert tool["correct_overall"] > 0.5


def test_an_unprompted_tool_call_is_still_a_protocol_violation(scored):
    """The relaxation is scoped to prompts that declared tools, not global."""
    from aadistill.evaluation.strict_answer import protocol_valid

    raw = 'think</think>\n<tool_call>\n{"name": "x", "arguments": {}}\n</tool_call><|im_end|>'
    assert protocol_valid(raw) == (False, "unexpected_tool_call")
    assert protocol_valid(raw, tools_offered=False) == (False, "unexpected_tool_call")
    assert protocol_valid(raw, tools_offered=True)[0] is True
    # A non-tool set in the real battery must not have been relaxed.
    assert scored["oracle"]["per_capability"]["gsm8k"]["protocol_valid"] == 1.0
    assert scored["contentless_perfect"]["per_set"]["tool"]["correct"] == 0


def test_counts_not_rates_reach_the_pooled_aggregation(scored):
    """`pooled_counts@v1` refuses a float; the scorer must emit integers."""
    from aadistill.autoinit.recovery import POOLED_COUNTS_V1

    sa, sb = scored["oracle"], scored["contentless_perfect"]
    for result in (sa, sb):
        for key in ("n", "usable", "correct"):
            assert isinstance(result[key], int) and not isinstance(result[key], bool)
    pooled = POOLED_COUNTS_V1.pool([
        {"seed": 20260726, "n": sa["n"], "usable": sa["usable"],
         "correct": sa["correct"]},
        {"seed": 20260801, "n": sb["n"], "usable": sb["usable"],
         "correct": sb["correct"]}])
    assert pooled["n"] == sa["n"] + sb["n"]
    assert pooled["aggregation"] == "pooled_counts@v1"
