"""Run the REAL pre-provider gates at $0 and stop before create().

Not a rehearsal with stubs: it constructs the launcher's own SessionSpec, the
runner's own SessionContext, and calls the runner's own make_plan() and
run_prechecks(). The only thing it does not do is call create().

It exists because the runner prices BEFORE it gates -- so attempt 3, which
aborted on price at 0.0 elapsed, never ran a single gate. The ledger says
otherwise and is being corrected.
"""
import json, sys
from pathlib import Path
REPO = Path("/home/ecs-user/AlphaAvatar-distill")
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts/pod"))
import autoinit_c1_launch as L
from aadistill.infrastructure.session_runner import SessionRunner

args = L.build_parser().parse_args([
    "--scr", "unused-for-gates",
    "--session-commit", sys.argv[1],
    "--bundle", sys.argv[2],
    "--max-price", sys.argv[3],
    "--out", "logs/autoinit_c1_gate_rehearsal.json",
])
runner = SessionRunner(L.spec(args), args, REPO)
priced = runner.make_plan()
print(f"\nmake_plan -> {priced}   quoted ${runner.price}/h")
gated = runner.run_prechecks() if priced else False
res = runner.ev.get("prechecks", [])
print(f"\n=== {sum(1 for r in res if r['ok'])}/{len(res)} GATES PASS ===")
for r in res:
    print(f"  [{'PASS' if r['ok'] else 'FAIL'}] {r['check']}: {r['message']}")
print("\nrelay/local:", json.dumps(runner.ev.get("precheck"), indent=1)[:900])
print(f"\nprechecks -> {gated}. create() NOT called; no provider resource.")
sys.exit(0 if gated else 1)
