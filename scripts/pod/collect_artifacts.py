#!/usr/bin/env python3
"""Manifest-driven artifact collection. Runs pod-side, then dev-box-side.

Replaces this construct, which is what lost E6b's training event streams:

    $SSH "root@$HOST" 'cd /workspace/aad && tar czf /workspace/e6b.tar.gz \\
      artifacts/audit/three_mode artifacts/audit/e6_checkpoint_manifest.json \\
      $(ls -d artifacts/stage3/e6b_*/train_log.jsonl 2>/dev/null)'

Two failure modes in one line. The literal list was inherited from a session
that did not train, so `train_log.jsonl` was simply absent from it. And where
the `$(ls -d …)` form *is* used (E3, E4, E5), the glob is expanded by the remote
login shell, and a pattern that matches nothing yields an empty substitution and
a silently smaller tarball that still exits 0, still hashes, and still verifies.

Here the spec is declared, expanded by Python that reports what it could not
find, and the archive is built from the resulting manifest — so a required
artifact that is missing fails **on the pod, while it can still be found**,
rather than after teardown.

Subcommands, in session order:

    manifest        pod-side: expand the spec, hash every file, report gaps
    archive         pod-side: tar exactly the manifest's files
    verify-archive  pod-side: confirm the archive covers the manifest
    verify-local    dev-side: re-hash the retrieved copies against the manifest
    gate            dev-side: may the pod be deleted?

Exit codes: 0 pass; 5 a required artifact is missing or a check failed.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aadistill.infrastructure.artifact_gate import (  # noqa: E402
    ArtifactManifest, ArtifactSpec, GATE_ORDER, build_manifest, create_archive,
    evaluate_teardown, verify_archive, verify_extracted,
)


def load_specs(path: str) -> tuple[ArtifactSpec, ...]:
    raw = json.loads(Path(path).read_text())
    return tuple(ArtifactSpec(**item) for item in raw)


def cmd_manifest(args) -> int:
    manifest = build_manifest(
        args.root, load_specs(args.spec),
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    manifest.write(args.out)
    print(f"{len(manifest.entries)} file(s) across "
          f"{len({e.artifact_class for e in manifest.entries})} class(es) "
          f"-> {args.out}")
    for gap in manifest.missing:
        print(f"  MISSING {gap['artifact_class']}: {gap['pattern']} "
              f"({gap['reason']})")
    if not manifest.ok:
        print("required artifacts are absent; find them before teardown — this "
              "is the point in the session where that is still possible")
        return 5
    return 0


def cmd_archive(args) -> int:
    manifest = ArtifactManifest.load(args.manifest)
    if not manifest.ok and not args.allow_incomplete:
        print("refusing to archive an incomplete manifest; pass "
              "--allow-incomplete only when the loss is accepted and recorded")
        return 5
    out = create_archive(manifest, args.out)
    # `create_archive` rewrites the entries to the bytes it actually archived.
    # Persist that, or the dev box verifies the transfer against a description
    # of bytes that no longer exist anywhere: the live canary on 2026-08-09 saw
    # `train_log.jsonl` grow by one event between manifest and tar, and the
    # teardown gate then blocked forever on `archive_contents_verified`.
    manifest.write(args.manifest)
    print(f"{out} ({out.stat().st_size} bytes, {len(manifest.entries)} files)")
    for row in manifest.appended_during_archive:
        print(f"  appended during archive: {row['path']} "
              f"{row['manifest_bytes']} -> {row['archived_bytes']} bytes")
    return 0


def cmd_verify_archive(args) -> int:
    manifest = ArtifactManifest.load(args.manifest)
    problems = verify_archive(args.archive, manifest)
    for p in problems:
        print(f"  {p}")
    print("archive covers the manifest" if not problems
          else f"{len(problems)} problem(s)")
    return 5 if problems else 0


def cmd_verify_local(args) -> int:
    manifest = ArtifactManifest.load(args.manifest)
    problems = verify_extracted(args.root, manifest)
    for p in problems:
        print(f"  {p}")
    print(f"{len(manifest.entries)} file(s) verified against the pod manifest"
          if not problems else f"{len(problems)} problem(s)")
    return 5 if problems else 0


def cmd_gate(args) -> int:
    state = json.loads(Path(args.state).read_text())
    decision = evaluate_teardown(
        state, emergency_budget=args.emergency,
        emergency_reason=args.emergency_reason)
    print(json.dumps(decision.as_dict(), indent=2))
    return 0 if decision.allowed else 5


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("manifest")
    m.add_argument("--root", required=True)
    m.add_argument("--spec", required=True, help="JSON list of ArtifactSpec")
    m.add_argument("--out", required=True)
    m.set_defaults(fn=cmd_manifest)

    a = sub.add_parser("archive")
    a.add_argument("--manifest", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--allow-incomplete", action="store_true")
    a.set_defaults(fn=cmd_archive)

    va = sub.add_parser("verify-archive")
    va.add_argument("--manifest", required=True)
    va.add_argument("--archive", required=True)
    va.set_defaults(fn=cmd_verify_archive)

    vl = sub.add_parser("verify-local")
    vl.add_argument("--manifest", required=True)
    vl.add_argument("--root", required=True)
    vl.set_defaults(fn=cmd_verify_local)

    g = sub.add_parser("gate", description="checks, in order: "
                                           + ", ".join(GATE_ORDER))
    g.add_argument("--state", required=True, help="JSON {check: bool}")
    g.add_argument("--emergency", action="store_true")
    g.add_argument("--emergency-reason", default="")
    g.set_defaults(fn=cmd_gate)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
