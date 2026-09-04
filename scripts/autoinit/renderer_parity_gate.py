"""The $0 dev-box gate that keeps renderer parity a guarantee, not a hope.

    PYTHONPATH=src .venv/bin/python scripts/autoinit/renderer_parity_gate.py

`tests/data/test_c1_battery.py` asserts that the shared renderers in
`scripts/data/battery_render.py` reproduce the frozen `recovery_search_v2`
prompts byte for byte. That assertion needs the seven pinned Hugging Face source
snapshots — roughly four gigabytes that the dev box holds from earlier work and
the C1 pod is deliberately never given, because they are a **readiness** input
and never a C1 runtime or scientific one.

Until 2026-09-04 those seven cases simply *failed* wherever the snapshots were
absent, which is every pod. That is what aborted C1 attempt 3R at the setup test
gate for `$0.3482`, having reached `VLLM_READY → TEACHER_READY → ROPE_OK`.

The fix is not to weaken the guarantee. In the pytest suite the seven cases now
skip when their source is absent; here, on the one host that has the sources,
they must all be present and all pass. A skip is a failure of this gate. That
keeps the property enforced exactly once, on the machine that can enforce it,
before a provider is created rather than after one is billing.

The comparison itself is `battery_render.check_group_parity`, shared verbatim
with the pytest cases: two independent implementations of "byte for byte" could
disagree about what parity means, which is the one thing this cannot afford.

Exit 0 when all seven groups PASS; non-zero on any skip, mismatch or absence.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts/data"))

from battery_render import (FROZEN_SOURCES, check_group_parity,  # noqa: E402
                            hub_cache)

SCHEMA = "aadistill.autoinit.c1_renderer_parity/v1"
RECORD = REPO_ROOT / "logs/c1_renderer_parity.json"

#: Every group must be accounted for. A gate that silently checked six of seven
#: would be indistinguishable from one that checked all seven and passed.
EXPECTED_GROUPS = tuple(sorted(FROZEN_SOURCES))


def _head_commit() -> str:
    out = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                         capture_output=True, text=True, timeout=60)
    return out.stdout.strip() or "unknown"


def _tree_is_clean() -> bool:
    out = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
                         capture_output=True, text=True, timeout=60)
    return out.returncode == 0 and not out.stdout.strip()


def _harness_digest() -> str:
    from aadistill.autoinit.c1_authorization import c1_harness_digest

    return c1_harness_digest(REPO_ROOT)["digest"]


def run_parity() -> dict[str, Any]:
    """Execute all seven groups and return the evidence record.

    Never raises on a parity failure: the verdict belongs to `gate_verdict`, so
    the launcher gate and the command line reach the same conclusion from the
    same record rather than from two code paths.
    """
    groups = [check_group_parity(g) for g in EXPECTED_GROUPS]
    counts = {status: sum(1 for g in groups if g["status"] == status)
              for status in ("PASS", "FAIL", "SOURCE_ABSENT")}
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "_what_this_is": (
            "the $0 pre-provider proof that the shared renderers still reproduce "
            "the frozen recovery-search prompts byte for byte, executed on the "
            "dev box because it is the only host holding the seven pinned source "
            "snapshots. The C1 pod is not given them and does not need them."),
        "executable_head": _head_commit(),
        "tree_clean": _tree_is_clean(),
        "c1_harness_digest": _harness_digest(),
        "resolved_hub_cache": str(hub_cache()),
        "n_groups_expected": len(EXPECTED_GROUPS),
        "counts": counts,
        "groups": groups,
        "shared_implementation": "scripts/data/battery_render.py:check_group_parity",
        "also_executed_by": (
            "tests/data/test_c1_battery.py::"
            "test_the_shared_renderers_reproduce_the_frozen_battery_byte_for_byte"),
    }
    return record


def gate_verdict(record: dict[str, Any]) -> tuple[bool, str]:
    """The single place that decides what this gate accepts."""
    groups = record.get("groups") or []
    seen = tuple(sorted(g["group"] for g in groups))
    if seen != EXPECTED_GROUPS:
        return False, (f"expected the {len(EXPECTED_GROUPS)} frozen groups, got "
                       f"{len(seen)}: {list(seen)}")
    absent = [g for g in groups if g["status"] == "SOURCE_ABSENT"]
    if absent:
        return False, ("renderer parity was not proved: "
                       + "; ".join(f"{g['repo_id']}@{g['revision'][:12]}… absent at "
                                   f"{g['resolved_snapshot']}" for g in absent))
    failed = [g for g in groups if g["status"] != "PASS"]
    if failed:
        return False, ("renderer parity BROKEN: "
                       + "; ".join(f"{g['group']} mismatches={g['mismatches'][:3]} "
                                   f"missing={g['missing'][:3]}" for g in failed))
    checked = sum(g["n_checked"] for g in groups)
    return True, (f"{len(groups)}/{len(EXPECTED_GROUPS)} groups PASS, 0 skipped, "
                  f"{checked} frozen prompts re-rendered byte for byte")


def write_record(record: dict[str, Any], path: Path = RECORD) -> str:
    """Write the record with a self-hash over its own content."""
    body = dict(record)
    body.pop("self_sha256", None)
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    body["self_sha256"] = digest
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    return digest


def main() -> int:
    record = run_parity()
    ok, reason = gate_verdict(record)
    record["verdict"] = "PASS" if ok else "FAIL"
    record["reason"] = reason
    digest = write_record(record)
    for g in record["groups"]:
        print(f"  {g['status']:<13} {g['group']:<14} "
              f"{g['n_checked']}/{g['n_frozen']} re-rendered  {g['repo_id']}")
    print(f"\n{'PASS' if ok else 'FAIL'}: {reason}")
    print(f"record: {RECORD.relative_to(REPO_ROOT)} ({digest[:12]}…)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
