"""Zero-cost evidence that the recovery-search scorer is sound, before any pod.

    PYTHONPATH=src python scripts/autoinit/validate_recovery_scoring.py

Runs every policy in `tests/autoinit/test_recovery_search_scoring.py` over all
190 frozen prompts and emits one reviewable artifact. The tests are the gate; this
is the record a maintainer reads without running pytest, and the thing the
preregistration binds its scoring-contract digest to.

What it demonstrates, in the maintainer's terms:

    perfect oracle              -> no capability structurally capped below 1.0
    malformed tool policy       -> tool unusable, incorrect
    invalid-tool-name policy    -> tool unusable, incorrect
    valid-but-wrong-arguments   -> tool USABLE, incorrect
    unprompted tool-call policy -> protocol invalid on every non-tool set
    every policy                -> correct <= usable preserved
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.autoinit.recovery import recovery_scoring_contract  # noqa: E402
from aadistill.infrastructure.manifest import sha256_json  # noqa: E402

TESTS = REPO_ROOT / "tests/autoinit/test_recovery_search_scoring.py"
GENERIC = ("gsm8k", "math_verified", "multihop", "rag", "knowledge")

#: What each policy must produce. Written here as data so the artifact states its
#: own acceptance criteria rather than leaving a reader to infer them.
EXPECTATIONS = {
    "oracle": "every capability usable 1.0; correctness high; nothing capped",
    "contentless_perfect": ("behaviourally perfect and useless; tool unusable "
                            "because no call was emitted"),
    "oracle_then_loop": "scorer finds the answer; correct still 0",
    "empty": "every component fails",
    "degenerate": "repetition stop; no_severe_repetition 0",
    "tool_malformed_json": "tool unusable (parse), incorrect",
    "tool_undeclared_name": "tool unusable (name), incorrect",
    "tool_wrong_arguments": "tool USABLE, incorrect",
    "unprompted_tool_call": "protocol invalid on every set that offers no tools",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/autoinit_recovery_scoring_validation.json")
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("rs_tests", TESTS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    report = {
        "schema": "aadistill.autoinit.recovery_scoring_validation/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "battery": "artifacts/stage3/recovery_search_v1",
        "n_prompts": 190,
        "scoring_contract": recovery_scoring_contract(REPO_ROOT),
        "expectations": EXPECTATIONS,
        "policies": {},
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for policy in mod.POLICIES:
            gen = tmp / policy
            mod.write_generations(gen, policy)
            out = tmp / f"{policy}.json"
            rc = subprocess.run(
                [sys.executable,
                 str(REPO_ROOT / "scripts/autoinit/score_recovery_search.py"),
                 "--generations", str(gen), "--label", policy,
                 "--seed", "20260726", "--out", str(out)],
                capture_output=True, text=True, cwd=REPO_ROOT,
                env={"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": "/usr/bin:/bin",
                     "HOME": str(tmp)})
            if rc.returncode != 0:
                raise SystemExit(f"{policy}: scorer failed\n{rc.stdout}{rc.stderr}")
            result = json.loads(out.read_text())
            tool = result["per_capability"]["tool"]
            report["policies"][policy] = {
                "expectation": EXPECTATIONS.get(policy),
                "usable_rollout_rate": result["usable_rollout_rate"],
                "correct_overall": result["correct_overall"],
                "correct_given_usable": result["correct_given_usable"],
                "correct_le_usable": result["correct"] <= result["usable"],
                "per_capability_usable": {
                    k: v["usable_rollout_rate"]
                    for k, v in result["per_capability"].items()},
                "per_capability_correct": {
                    k: v["correct_overall"]
                    for k, v in result["per_capability"].items()},
                "tool": {k: tool.get(k) for k in
                         ("protocol_valid", "tool_call_emitted", "tool_call_parsed",
                          "tool_name_valid", "tool_structurally_executable",
                          "usable_rollout_rate", "correct_overall")},
                "protocol_valid": result["protocol_valid"],
                "first_failure": result["first_failure"],
            }

    p = report["policies"]
    checks = {
        "oracle_no_capability_capped": all(
            v == 1.0 for v in p["oracle"]["per_capability_usable"].values()),
        "oracle_correctness_high": p["oracle"]["correct_overall"] > 0.60,
        "malformed_tool_unusable": (
            p["tool_malformed_json"]["tool"]["usable_rollout_rate"] == 0.0
            and p["tool_malformed_json"]["tool"]["correct_overall"] == 0.0),
        "undeclared_tool_name_unusable": (
            p["tool_undeclared_name"]["tool"]["usable_rollout_rate"] == 0.0
            and p["tool_undeclared_name"]["tool"]["correct_overall"] == 0.0),
        "wrong_arguments_usable_but_incorrect": (
            p["tool_wrong_arguments"]["tool"]["usable_rollout_rate"] == 1.0
            and p["tool_wrong_arguments"]["tool"]["correct_overall"] == 0.0),
        "unprompted_tool_call_protocol_invalid": all(
            p["unprompted_tool_call"]["per_capability_usable"][k] == 0.0
            for k in GENERIC),
        "unprompted_tool_call_allowed_where_tools_offered": (
            p["unprompted_tool_call"]["tool"]["protocol_valid"] == 1.0),
        "correct_implies_usable_everywhere": all(
            v["correct_le_usable"] for v in p.values()),
        "contentless_is_behaviourally_perfect_and_useless": (
            all(p["contentless_perfect"]["per_capability_usable"][k] == 1.0
                for k in GENERIC)
            and p["contentless_perfect"]["correct_overall"] < 0.10),
    }
    report["checks"] = checks
    report["all_checks_pass"] = all(checks.values())
    report["report_sha256"] = sha256_json(report)
    (REPO_ROOT / args.out).write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps({
        "all_checks_pass": report["all_checks_pass"],
        "failed": [k for k, v in checks.items() if not v],
        "scoring_contract": report["scoring_contract"]["contract"],
        "scoring_contract_digest": report["scoring_contract"]["digest"],
        "tool_usable_by_policy": {
            k: v["tool"]["usable_rollout_rate"] for k, v in p.items()},
        "report_sha256": report["report_sha256"],
    }, indent=2))
    if not report["all_checks_pass"]:
        raise SystemExit("scorer validation FAILED — do not launch")


if __name__ == "__main__":
    main()
