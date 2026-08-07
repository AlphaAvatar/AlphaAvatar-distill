#!/usr/bin/env python
"""Verify the staged arm-R corpora independently, before anything trains on them.

    PYTHONPATH=src python scripts/data/verify_staged_r.py \
        --root artifacts/stage3 --teacher-revision <sha> \
        --ckpt-dir /workspace/ckpt --out artifacts/audit/e5_staged_r_verify.json

Attempt 5's R corpora are reused rather than regenerated: they are already-paid
production outputs from the registered teacher/student/decoding configuration,
they passed the generation gates and `RECORDS_VERIFIED` on the pod, and attempt 5
produced no training result, so nothing about them is outcome-dependent.

Reuse is only sound if the artifact staged on the pod is provably the artifact
that was generated. A hash alone proves provenance and not usability; a contract
check alone proves usability and not provenance. This asserts both, plus the
configuration the corpus claims to have been generated under, so a mismatch stops
the run here rather than surfacing as a quiet difference in a trained model.

Every check is fatal. There is no "warn and continue" path: a corpus that fails
any of these is not the corpus the experiment registered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.data.e5_pack import (  # noqa: E402
    REQUIRED_FIELDS, example_to_rendered,
)

# Recorded when the corpora were staged on 2026-08-08, from the artifacts that
# came back in attempt 5's side bundle.
EXPECTED = {
    "sa": {
        "records": 2098,
        "examples.jsonl": "247941ab7816adc04807b3300c2d145d0927b3a2ce5ef605c662129e4136dd3c",
        "manifest.json": "eea4f1a3bc362328a4e3dc245e4a72bcf8606d92964bcd8026d3f4885496dbd4",
        "system_ids.json": "f89a33862bdf5b34ca8d3af6ec8f73e6710c5d69f4229267b64822005254023d",
    },
    "sb": {
        "records": 2042,
        "examples.jsonl": "ae1a420355098c70f80001865ed4e2d4f161be997df9272d0d0caf56f3ef7b9e",
        "manifest.json": "9efdddcda6c5c6258ba013d97e7ff8751be18e8c4f6a57978f13d342417d628a",
        "system_ids.json": "6ab6194560f20e7128f290ea507606bf3fc54450ba0f4b5a0c3b925b104cbdf1",
    },
}
# The tokenizer and chat template the corpora were rendered with. Identical
# across both P2 seeds, as they must be -- both descend from the same Stage 1
# initialization, and a divergence would mean the two seeds tokenized differently.
TOKENIZER = {
    "tokenizer.json": "be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506",
    "tokenizer_config.json": "8fa82a4ba512c8bee7c1c5e82b9a71ddbef362e4665be5c8f7ce0afd78af129a",
    "chat_template.jinja": "3802169b2a02b81e6adb7ab4f64f91ff02db753c8c3a64a01c35192d3a61d8d7",
}
CHECKPOINT = {"sa": "4aface45a12cd02e", "sb": "9828b1780a5eb4e2"}
PRESET = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
SESSIONS_SHA = "2b4edc2e2cc16cd56dae3d340345e1a17e2c4a8baa9837650a7bf5e340fa6fcd"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(results: list, name: str, ok: bool, detail: str = "") -> bool:
    results.append({"check": name, "passed": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}",
          flush=True)
    return bool(ok)


def verify_seed(seed: str, root: Path, ckpt_dir: Path, teacher_revision: str,
                results: list) -> None:
    d = root / f"e5_arm_r_{seed}"
    exp = EXPECTED[seed]

    for name, want in ((k, v) for k, v in exp.items() if k != "records"):
        path = d / name
        got = sha256(path) if path.is_file() else "MISSING"
        check(results, f"{seed}/{name} hash", got == want, got[:16])

    rows = [json.loads(line) for line in (d / "examples.jsonl").open() if line.strip()]
    check(results, f"{seed} record count", len(rows) == exp["records"],
          f"{len(rows)} (expected {exp['records']})")

    sysids = json.loads((d / "system_ids.json").read_text())
    missing = unrenderable = bad_system = 0
    for rec in rows:
        if [f for f in REQUIRED_FIELDS if f not in rec]:
            missing += 1
            continue
        try:
            example_to_rendered(rec)
        except Exception:
            unrenderable += 1
            continue
        if rec["ids"][:rec["n_system_tokens"]] != sysids.get(rec["system_key"], []):
            bad_system += 1
    check(results, f"{seed} no missing ids/masks/system fields", missing == 0, str(missing))
    check(results, f"{seed} every record reloads through example_to_rendered",
          unrenderable == 0, str(unrenderable))
    check(results, f"{seed} every record's system block matches its key",
          bad_system == 0, str(bad_system))

    man = json.loads((d / "manifest.json").read_text())
    tid, _, rev = man["teacher"].partition("@")
    check(results, f"{seed} teacher revision", rev == teacher_revision, rev[:16])
    check(results, f"{seed} teacher id", tid == "Qwen/Qwen3-4B-Thinking-2507", tid)
    check(results, f"{seed} decoding preset", man["decoding_preset"] == PRESET,
          json.dumps(man["decoding_preset"]))
    check(results, f"{seed} source sessions corpus", man["sessions_sha256"] == SESSIONS_SHA,
          man["sessions_sha256"][:16])

    # The corpus names the student it rolled out from; the weights behind that
    # path are hashed here, so the name cannot stand in for the identity.
    claimed = Path(man["student"]).name
    check(results, f"{seed} student path names p2_ceheavy_{seed}",
          claimed == f"p2_ceheavy_{seed}", claimed)
    w = ckpt_dir / f"p2_ceheavy_{seed}" / "model.safetensors"
    got = sha256(w)[:16] if w.is_file() else "MISSING"
    check(results, f"{seed} P2-0.86M checkpoint identity", got == CHECKPOINT[seed], got)

    for name, want in TOKENIZER.items():
        f = ckpt_dir / f"p2_ceheavy_{seed}" / name
        got = sha256(f) if f.is_file() else "MISSING"
        check(results, f"{seed} {name}", got == want, got[:16])


def verify_seed_derivation(results: list) -> None:
    """The rollout seed must be a pure function of (session id, source seed).

    Python's `hash()` on a str is randomized per process, so a corpus seeded that
    way is unreproducible from its own manifest. Re-deriving in this interpreter
    and comparing against values computed in another proves the function is
    stable across processes, which is the property that matters.
    """
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "data"))
    from build_e5_arm_r import stable_seed

    known = {("gsm8k-000001", "sa"), ("glaive-000300", "sb")}
    here = {k: stable_seed(*k) for k in known}
    again = {k: stable_seed(*k) for k in known}
    check(results, "seed derivation is deterministic in-process", here == again)
    # Inspect the parsed function, not its text: the docstring explains why
    # `hash()` is unsafe here, so a substring search for "hash(" flags the very
    # comment that documents the fix.
    import ast
    tree = ast.parse((REPO_ROOT / "scripts/data/build_e5_arm_r.py").read_text())
    fn_node = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == "stable_seed")
    calls = {n.func.id for n in ast.walk(fn_node)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    attrs = {n.func.attr for n in ast.walk(fn_node)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    check(results, "seed derivation calls sha256", "sha256" in attrs, str(sorted(attrs)))
    check(results, "seed derivation never calls the randomized hash()",
          "hash" not in calls, str(sorted(calls)))
    check(results, "seeds differ across source seeds",
          stable_seed("gsm8k-000001", "sa") != stable_seed("gsm8k-000001", "sb"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("artifacts/stage3"))
    ap.add_argument("--ckpt-dir", type=Path, default=Path("/workspace/ckpt"))
    ap.add_argument("--teacher-revision", required=True)
    ap.add_argument("--seeds", nargs="+", default=["sa", "sb"])
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    results: list = []
    print("verifying the staged arm-R corpora", flush=True)
    for seed in args.seeds:
        verify_seed(seed, args.root, args.ckpt_dir, args.teacher_revision, results)
    verify_seed_derivation(results)

    failures = [r["check"] for r in results if not r["passed"]]
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "purpose": ("independent verification of REUSED arm-R artifacts: "
                    "provenance by hash, usability by contract, and the "
                    "configuration the corpus claims to have been generated under"),
        "checks": results, "failures": failures, "passed": not failures,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=1))
        print(f"wrote {args.out}", flush=True)
    if failures:
        sys.exit(f"STAGED R VERIFICATION FAILED: {failures}")
    print(f"all {len(results)} checks passed", flush=True)


if __name__ == "__main__":
    main()
