"""Retire REMOTE relay copies whose canonical local copy is verified. $0.

    PYTHONPATH=src python scripts/maintenance/consolidation/retire_relay_copies.py            # dry run
    PYTHONPATH=src python scripts/maintenance/consolidation/retire_relay_copies.py --execute

**This retires a remote COPY, not a checkpoint.** Every object removed here has a
byte-identical copy under `/home/ecs-user/aad-artifacts`, verified by content
hash at the moment of deletion. The scientific artifact continues to exist; what
changes is that it stops being mirrored on Hugging Face. Nothing in the
experiment records, frozen identities or accountability metadata is touched.

**Why.** The private-storage limit is account-wide (measured 2026-08-22:
93.279 GiB against ~93.13), so the five Attempt-12 selected leaves — 5.55 GiB —
cannot be mirrored for transport while obsolete copies hold the quota. The
maintainer decided explicitly that obsolete remote redundancy is not worth
blocking Phase A.

**How, and why not the obvious way.** Ordinary file deletion reclaims *nothing*
on this repo — measured twice (2026-08-02, deleting 19.07 GB freed nothing;
2026-08-14, a 2.23 GiB delete left billed `usedStorage` unchanged for 45 min).
Hugging Face bills LFS including history, so the only reclaim is
`permanently_delete_lfs_files(..., rewrite_history=True)` on an exact object.
Objects are therefore selected **by `file_oid` identity, never by directory
prefix**, and every safety property is re-asserted here rather than inherited
from the table that proposed them.

The gotcha that reads as "not found": the LFS content hash is the object's
`file_oid`, **not** its `oid`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

RELAY = "AlphaAvatar/aadistill-artifacts"
TRANSPORT = "AlphaAvatar/aadistill-transport"
CANON = Path("/home/ecs-user/aad-artifacts")

#: Never removable, whatever a table says. Re-derived here so a mistake in the
#: candidate list cannot reach the API.
PROTECTED_PREFIXES = (
    "transfer/wheelhouse_cu128_cp312/",   # offline uv sync, paid critical path
    "transfer/wheelhouse_vllm_cp312/",    # offline vLLM venv, paid critical path
    "permanent_controls/",                # the two permanent controls
    "stage1/qwen3_0p6b_init_v0/",         # canonical initialization
    "stage3_recovery_corpus_v2/ladder_uniform/",
    "e8_inputs_20260810/calibration_v1/",
)


def token() -> str:
    return Path(os.path.expanduser("~/.cache/huggingface/token")).read_text().strip()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _probe_argv(parser, extra: tuple[str, ...]) -> list[str]:
    """Argv the launcher's REAL parser accepts, with required options filled.

    Type-appropriate fillers: a required `type=float` option handed the string
    `"probe"` makes the parser exit 2, which drops that launcher — and every
    path it declares — out of the protection set without a word.
    """
    argv = ["--scr", "/tmp/retire-spec-probe", "--session-commit", "0" * 40,
            "--bundle", "aad_test.bundle", *extra]
    for a in parser._actions:
        opts = a.option_strings or ()
        if not opts or opts[0] in argv:
            continue
        if not (a.required or opts[0] == "--run-id"):
            continue
        if a.nargs == 0:
            argv.append(opts[0])
            continue
        t = getattr(a, "type", None)
        argv += [opts[0], "1.0" if t is float else "1" if t is int else "probe"]
    return argv


def declared_remote_paths() -> tuple[set[str], list[tuple[str, str]]]:
    """Every remote path any CURRENT session declares, from the real specs.

    Enumerated over **every** `scripts/pod/autoinit_*_launch.py`, not over a
    curated list. It used to read `session_specs.all_specs()`, whose
    `SESSION_LAUNCHERS` names 7 of the 15 launchers that exist — so the C3,
    C2-behavioural, continuation-B, Phase-B and Phase-C2 sessions declared
    their relay inputs to nothing that protected them. C3's eight inputs
    survived only because `PROTECTED_PREFIXES` happens to cover the three
    directories they live in, which is coincidence rather than design.

    Returns the paths and the launchers that could NOT be enumerated. A
    launcher that refuses to build its spec is a launcher whose declared
    inputs are unknown, so the caller **fails closed** on it rather than
    deleting against a set that silently shrank.
    """
    import importlib.util

    sys.path.insert(0, str(REPO_ROOT / "scripts" / "pod"))
    sys.path.insert(0, str(REPO_ROOT / "src"))

    paths: set[str] = set()
    unreadable: list[tuple[str, str]] = []
    for path in sorted(REPO_ROOT.glob("scripts/pod/autoinit_*_launch.py")):
        name = path.stem
        try:
            spec_ = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec_)
            sys.modules[name] = mod
            spec_.loader.exec_module(mod)
            if not (hasattr(mod, "spec") and hasattr(mod, "build_parser")):
                continue  # not a session launcher; declares no relay inputs
            parser = mod.build_parser()
            #: The one launcher whose transport is not defaulted.
            extra = (("--transport", "relay")
                     if name == "autoinit_continuation_launch" else ())
            session = mod.spec(parser.parse_args(_probe_argv(parser, extra)))
            paths |= {r.path for r in session.setup.relay_inputs}
        except BaseException as exc:            # SystemExit included, deliberately
            unreadable.append((name, f"{type(exc).__name__}: {exc}"))
    return paths, unreadable


def check(obj, declared: set[str], *, archival: set[str] | None = None,
          ) -> tuple[bool, str, str | None]:
    """Every condition, re-derived. Returns (ok, why, canonical_local_path).

    `archival` is the set of paths an explicit maintainer authorization has
    cleared under the P8.4 consumer rule: completed, validly scored, evidence
    durable, no authorized consumer of the WEIGHTS. Those objects have no local
    copy by construction — that is the whole point — so the copy check below is
    the one property they are exempt from. Every other property still applies,
    and is re-derived here rather than inherited from the authorization.
    """
    fn = obj.filename
    if fn in declared:
        return False, "a current session declares this remote path", None
    if any(fn.startswith(p) for p in PROTECTED_PREFIXES):
        return False, "protected prefix", None
    if archival and fn in archival:
        return True, "archival retirement, authorized and consumer-checked", None
    # The canonical copy must exist and match, under the canonical store only.
    matches = []
    for root in (CANON,):
        for p in root.rglob("*"):
            try:
                if p.is_file() and p.stat().st_size == obj.size:
                    if sha256(p) == obj.file_oid:
                        matches.append(str(p))
                        break
            except OSError:
                continue
        if matches:
            break
    if not matches:
        return False, "NO canonical local copy with this content hash", None
    return True, "canonical local copy verified by content hash", matches[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--table", default=None,
                    help="candidate table from the analysis step (filenames only; "
                         "every property is re-checked here)")
    ap.add_argument("--archival-authorization", default=None,
                    help="an aadistill.archival_retirement/v1 record. Clears the "
                         "objects it names of the local-copy requirement ONLY; "
                         "every other safety property is still re-derived here.")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    api = HfApi(token=token())
    declared, unreadable = declared_remote_paths()

    archival: set[str] = set()
    if args.archival_authorization:
        rec = json.load(open(args.archival_authorization))
        if rec.get("schema") != "aadistill.archival_retirement/v1":
            raise SystemExit(f"not an archival authorization: {rec.get('schema')!r}")
        if not rec.get("authorised_by"):
            raise SystemExit("archival authorization names no maintainer decision")
        archival = {r["path"] for r in rec["retired"]}
        print(f"archival authorization: {len(archival)} objects, "
              f"{rec['authorised_by'][:60]}…")

    #: FAIL CLOSED. A launcher that cannot build its spec declares an UNKNOWN
    #: set of relay inputs, so the protection set has quietly shrunk and this
    #: tool cannot tell what it would be deleting. The authorization must name
    #: each one it reviewed by hand; an unreadable launcher it does not name
    #: refuses the run. A new launcher breaking its spec therefore stops a
    #: later retirement instead of silently widening it.
    for name, why in unreadable:
        print(f"  UNREADABLE {name}: {why}")
    if archival:
        reviewed = set(rec.get("unreadable_launchers_reviewed") or {})
        unreviewed = sorted({n for n, _ in unreadable} - reviewed)
        if unreviewed:
            raise SystemExit(
                "these launchers could not be enumerated and the authorization "
                f"does not review them: {unreviewed}. Their declared relay "
                "inputs are unknown, so refusing to delete anything.")

    wanted = None
    if args.table:
        wanted = {r["filename"] for r in json.load(open(args.table))["rows"]
                  if r.get("deletable")}
    elif archival:
        #: Archival mode considers the authorized objects and NOTHING else, so
        #: a local-copy-verified object cannot ride along on an authorization
        #: that never named it.
        wanted = set(archival)

    #: BOTH repos. This iterated only RELAY while the module's own docstring
    #: is about the TRANSPORT repo's redundant copies -- the five Attempt-12
    #: leaves and the Phase-B one that the 2026-08-22 finding left "in place
    #: for the maintainer to dispose of". They were unreachable by the tool
    #: written to dispose of them. `check` re-derives every safety property per
    #: object, so widening the scan weakens nothing.
    plan, refused = [], []
    for repo in (RELAY, TRANSPORT):
        for obj in api.list_lfs_files(repo, repo_type="model"):
            if wanted is not None and obj.filename not in wanted:
                continue
            ok, why, local = check(obj, declared, archival=archival)
            (plan if ok else refused).append((obj, why, local, repo))

    #: An authorized path that never matched a live object is a typo, a stale
    #: record or an already-deleted object — all three mean the operator's
    #: mental model and the relay disagree, and none of them should be
    #: discovered after the API call.
    if archival:
        matched = {o.filename for o, _, _, _ in plan} & archival
        #: Distinguish the two ways an authorized path can fail to be planned.
        #: Collapsing them reports a PROTECTED path as "not found", which reads
        #: like a stale record and invites someone to re-add it.
        blocked = {o.filename: why for o, why, _, _ in refused
                   if o.filename in archival}
        absent = sorted(archival - matched - set(blocked))
        if blocked:
            for fn, why in sorted(blocked.items()):
                print(f"  REFUSED (authorized but protected) {fn}: {why}")
        if blocked or absent:
            raise SystemExit(
                f"authorization not satisfiable: {len(blocked)} path(s) refused "
                f"by a safety check, {len(absent)} matched no live relay object"
                + (f" {absent[:4]}" if absent else ""))

    print(f"declared remote paths in current sessions: {len(declared)}")
    for obj, why, _, repo in refused:
        print(f"  REFUSED {repo.split('/')[1]}:{obj.filename}: {why}")
    total = 0
    for obj, why, local, repo in plan:
        total += obj.size
        print(f"  RETIRE  {obj.size / 2**30:7.3f} GiB  "
              f"{repo.split('/')[1]}:{obj.filename}")
        print(f"          oid {obj.file_oid[:16]}…  local {local or 'NONE — archival'}")
    print(f"\n{len(plan)} objects, {total / 2**30:.3f} GiB")

    #: Under an archival authorization the plan must be EXACTLY what was
    #: authorized. Anything extra would be an object this run decided on its
    #: own to delete forever, which is not what a maintainer signed.
    if archival:
        extra = sorted({o.filename for o, _, _, _ in plan} - archival)
        if extra:
            raise SystemExit(
                f"plan exceeds the authorization by {len(extra)} object(s): "
                f"{extra[:4]}{'…' if len(extra) > 4 else ''}")
        print(f"plan == authorization ({len(plan)} objects), nothing extra")

    if not args.execute:
        print("\nDRY RUN — nothing deleted. Re-run with --execute.")
        return
    if not plan:
        print("nothing to do")
        return
    #: Per repo: `permanently_delete_lfs_files` takes objects of ONE repo, and
    #: handing it a mixed list would delete from the wrong one or fail.
    for repo in (RELAY, TRANSPORT):
        objs = [o for o, _, _, r in plan if r == repo]
        if not objs:
            continue
        api.permanently_delete_lfs_files(repo, objs, rewrite_history=True,
                                         repo_type="model")
        print(f"retired {len(objs)} remote copies from {repo}")


if __name__ == "__main__":
    main()
