"""Proof that the pod's CPU test gate can pass, bound to the tree that proved it.

A paid session reached every readiness marker and then died at the setup test
gate: fourteen failures out of ~2650, no scientific stage. Seven were
renderer-parity cases reading `$HOME` for a Hugging Face cache no pod has,
and two were repository state. **The other five remain UNEXPLAINED**: they were
attributed to leaf transport by a `$0` reproduction that ran with no `HF_TOKEN`,
which is a state no pod is in, and that attribution does not reproduce.

The seven passed on the dev box and could not pass on a pod, and several
launches went by without anyone finding out, because no session had reached
`TESTS_OK` before.

The suite that answers "would this pass on a pod?" takes about thirteen minutes,
which is far too slow to run inside a pre-provider gate while a pod waits. So it
is run once, deliberately, and what it produced is recorded here — and the cheap
gate checks that the recording still describes the code that would actually run.

That binding is to the **executable**, never to `HEAD`. The normal order of work
commits the executable first, runs the sweep against it, then writes a
preregistration and updates documentation in later commits; a gate keyed on the
commit hash would go stale the moment the paperwork landed, and the obvious way
to make it green again would be to re-run a sweep that had not changed in any way
that mattered. Two digests decide instead:

* the C1 harness digest — everything the paid session executes;
* the pod **test environment** digest — everything that decides what the pod's
  test gate does and is outside that harness: the simulator that models it, the
  whole test tree, the publisher whose tests the gate runs, and the recorder
  that writes this file. The pod's own setup script is NOT here: it is executed
  on the pod, so the grant measures it, in the harness.

Neither digest covers `logs/` or `docs/`, and the pod suite reads repository
state — so a third check, `swept_base_commit` lineage, closes that gap.

Change any of them and the record no longer describes the gate; the sweep is owed
again. Change a log, a decision record or a README, and it does not.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = "aadistill.autoinit.c1_pod_environment_verification/v1"

#: The record path and the named non-harness files are CALLER-SUPPLIED. They
#: used to be written here, which put one experiment's log name and one
#: experiment's tool list inside a reusable runtime module. They now live in
#: `scripts/experiments/phase_c1/pod_environment.py`.


#: The readiness contract a caller declares. The nine node-id groups that used
#: to sit here were C1's -- concrete test node ids, one session's expectation
#: about which of its own tests skip on a pod and which must pass. A reusable
#: runtime cannot own that: a second session watches different tests, and its
#: refusal messages would still have said "C1".
#:
#: So the groups are the CALLER's, and this module holds only what they mean.
#: `scripts/experiments/phase_c1/pod_environment.py` declares C1's.
@dataclass(frozen=True)
class ReadinessGroups:
    """Which node ids a session expects to skip, and which must pass.

    Every field is required. A default would make one session's contract the
    silent answer for every other, which is the defect this extraction closes --
    and an empty default would be worse, because a group of zero node ids passes
    its check vacuously.
    """

    #: Must SKIP: their source is a readiness input the session does not stage.
    expected_skips: Mapping[str, Sequence[str]]
    #: Must PASS: named groups whose absence or skip is a finding.
    must_pass: Mapping[str, Sequence[str]]
    #: A single node id that must pass because the session DOES stage its source.
    staged_role_nodeid: str
    #: Legitimate skips inside a watched module that are not environment-driven.
    known_non_environment_skips: Sequence[str]
    #: Extra context appended to a group's refusal, by group name. The
    #: session owns the explanation -- "recovery_search_v2 is a local asset"
    #: is a fact about C1's staging, not about sweeps.
    refusal_notes: Mapping[str, str] = field(default_factory=dict)
    #: Import path prefixes whose skips are watched at all.
    watched: Sequence[str] = ()

    def all_expected_skips(self) -> set[str]:
        return {n for g in self.expected_skips.values() for n in g}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pod_test_environment_digest(repo_root: str | Path = ".", *,
                                named_files: tuple[str, ...]) -> dict[str, Any]:
    """Digest over everything that decides the pod test gate's outcome.

    The whole test tree is in here on purpose. A new test is exactly as capable
    of failing on a pod as a new line of production code — that is the entire
    lesson of attempt 3R — so a readiness record must not survive one.
    """
    root = Path(repo_root)
    rels = sorted(
        {str(p.relative_to(root)) for p in (root / "tests").rglob("*.py")}
        | set(named_files))
    entries = []
    for rel in rels:
        p = root / rel
        if not p.is_file():
            raise FileNotFoundError(
                f"declared pod test-environment source {rel!r} is missing; "
                "refusing to digest a smaller environment than the one the pod "
                "gate would run")
        entries.append({"path": rel, "sha256": _sha256_file(p)})
    digest = hashlib.sha256(
        "".join(f"{e['path']}:{e['sha256']}\n" for e in entries).encode()).hexdigest()
    return {"digest": digest, "n_files": len(entries),
            "named_files": list(named_files),
            "rule": ("sha256 over sorted 'path:sha256' lines of tests/**/*.py "
                     "plus the named non-harness files")}


# --- reading what the sweep actually did ------------------------------------

def _nodeid(case: ET.Element, repo_root: Path) -> str:
    """Reconstruct a pytest nodeid from a JUnit `testcase` element.

    pytest writes **no `file` attribute** — only `classname` and `name`. Assuming
    otherwise produced nodeids like `::test_x` for every case, so the record's
    lookups all read `ABSENT` and it could not tell a passing sweep from a
    failing one. Found by running the sweep, which is the only reason it was
    found at all.

    `classname` is the dotted module path, with any test class appended:
    `tests.pod.test_x` or `tests.pod.test_x.TestGroup`. The split point is
    resolved against the filesystem — the longest dotted prefix that is a real
    `.py` file is the module — rather than guessed from naming convention.
    """
    cls = case.get("classname") or ""
    name = case.get("name") or ""
    parts = cls.split(".") if cls else []
    for cut in range(len(parts), 0, -1):
        rel = "/".join(parts[:cut]) + ".py"
        if (repo_root / rel).is_file():
            inner = "::".join(parts[cut:])
            return f"{rel}::{inner}::{name}" if inner else f"{rel}::{name}"
    # No module resolved: keep the raw classname rather than silently emitting a
    # nodeid with an empty path, which is what hid the defect the first time.
    return f"{cls}::{name}" if cls else name


def read_junit(path: str | Path, repo_root: str | Path = ".") -> dict[str, Any]:
    """Per-nodeid outcomes from a JUnit XML report.

    JUnit is a *reporting* flag: it changes nothing about which tests are
    selected or how they run, so the sweep still executes the pod's own command.
    It is the only way to name every skip and every pass exactly, and naming them
    is the point — attempt 3R's four-line tail is why fourteen failures arrived
    as three.
    """
    root = Path(repo_root)
    tree = ET.parse(str(path))
    outcomes: dict[str, str] = {}
    reasons: dict[str, str] = {}
    details: dict[str, dict[str, str]] = {}
    n_cases = 0
    for case in tree.iter("testcase"):
        n_cases += 1
        nid = _nodeid(case, root)
        status = "passed"
        for child in case:
            tag = child.tag.lower()
            if tag in ("failure", "error", "skipped"):
                status = {"failure": "failed", "error": "error",
                          "skipped": "skipped"}[tag]
                if tag == "skipped":
                    # Why it skipped, which is the half attempt 5 could not see.
                    reasons[nid] = (child.get("message")
                                    or (child.text or "").strip())[:300]
                else:
                    # WHY it failed. A diagnostic that names every failing
                    # nodeid and not one message leaves the mechanism attributed
                    # rather than proven — the same shape as a four-line tail and
                    # attempt 5's missing skip list, one layer further in.
                    details[nid] = {
                        "kind": status,
                        "type": child.get("type") or "",
                        "message": (child.get("message") or "")[:2000],
                        "body": (child.text or "").strip()[:8000],
                    }
                break
        outcomes[nid] = status
    counts = {s: sum(1 for v in outcomes.values() if v == s)
              for s in ("passed", "skipped", "failed", "error")}
    # A collision means two testcases mapped to one nodeid and one outcome was
    # silently dropped — which is exactly how a failure would disappear.
    if n_cases != len(outcomes):
        raise ValueError(
            f"{n_cases} testcases collapsed to {len(outcomes)} nodeids; the "
            "reconstruction is lossy and an outcome would be lost")
    return {"outcomes": outcomes, "counts": counts, "total": len(outcomes),
            "skip_reasons": dict(sorted(reasons.items())),
            "failure_details": dict(sorted(details.items()))}


def skip_set_digest(nodeids: Sequence[str]) -> str:
    """Deterministic over the COMPLETE skip set, order-independent.

    Two sweeps with the same digest skipped exactly the same tests. A diagnostic
    that names every FAILED nodeid and no SKIPPED one cannot make that
    comparison: a divergence that only moved a skip is invisible, and one
    such divergence is still unexplained.
    """
    body = "\n".join(sorted(set(nodeids)))
    return hashlib.sha256(body.encode()).hexdigest()


def compare_skip_sets(expected: Sequence[str], actual: Sequence[str]
                      ) -> dict[str, Any]:
    """Which tests changed their mind between two machines.

    `expected_but_ran` — the sweep skipped it, the pod ran it. That is attempt
    5's two FAILURES. `unexpected_skip` — the pod skipped what the sweep ran.
    That is attempt 5's unexplained RESIDUAL (sweep 2770/100 vs pod 2769/99: one
    test the sweep PASSED skipped on the pod, and was never named).
    """
    exp, act = set(expected), set(actual)
    ran = sorted(exp - act)
    extra = sorted(act - exp)
    return {
        "expected_skips": len(exp), "actual_skips": len(act),
        "expected_but_ran": ran,
        "unexpected_skip": extra,
        "identical": not ran and not extra,
        "expected_digest": skip_set_digest(exp),
        "actual_digest": skip_set_digest(act),
    }


def evaluate_sweep(outcomes: dict[str, str],
                   skip_reasons: dict[str, str] | None = None,
                   *, groups: ReadinessGroups) -> dict[str, Any]:
    """Turn per-nodeid outcomes into the pass/fail findings the record asserts.

    `groups` is REQUIRED and comes from the session. This function used to name
    one session's node-id tuples directly, so a reusable runtime carried that
    experiment's expectations and refused in its vocabulary -- naming which of
    this project's phases failed to skip under which session kind is not
    something a generic module can say.

    The output keys are derived from the caller's own group names, so the record
    schema is the session's too. C1's names reproduce the existing keys exactly.
    """
    counts = {s: sum(1 for v in outcomes.values() if v == s)
              for s in ("passed", "skipped", "failed", "error")}
    failed = sorted(n for n, s in outcomes.items() if s in ("failed", "error"))

    out: dict[str, Any] = {"counts": counts, "failed_nodeids": failed}
    problems: list[str] = []
    if failed:
        problems.append(f"{len(failed)} failed/errored: {failed[:10]}")

    def note(name: str) -> str:
        extra = groups.refusal_notes.get(name, "")
        return f" {extra}" if extra else ""

    #: A group of zero node ids would pass its own check vacuously, which is the
    #: `battery_v2` defect in a different costume. An empty declared group is a
    #: refusal, not a pass.
    for name, ids in groups.must_pass.items():
        seen = {n: outcomes.get(n, "ABSENT") for n in ids}
        ok = bool(ids) and all(v == "passed" for v in seen.values())
        out[name] = seen
        out[f"{name}_all_passed"] = ok
        if not ok:
            problems.append(
                f"{name}: did not pass {len(ids)}/{len(ids)}: {seen}." + note(name))

    for name, ids in groups.expected_skips.items():
        seen = {n: outcomes.get(n, "ABSENT") for n in ids}
        ok = bool(ids) and all(v == "skipped" for v in seen.values())
        out[f"{name}_expected_skips"] = seen
        out[f"{name}_skipped_as_expected"] = ok
        if not ok:
            problems.append(
                f"{name}: expected {len(ids)} skip(s), got {seen}." + note(name))

    staged_role = outcomes.get(groups.staged_role_nodeid, "ABSENT")
    if staged_role != "passed":
        problems.append(
            f"the staged-role case did not pass: {staged_role!r} for "
            f"{groups.staged_role_nodeid!r}." + note("staged_role"))

    # Any OTHER skip in a watched module is an unexpected environment skip: a
    # test that quietly stopped running under an empty HOME is
    # indistinguishable from one that never existed.
    watched = tuple(groups.watched)
    expected = groups.all_expected_skips()
    unexpected = sorted(
        n for n, s in outcomes.items()
        if s == "skipped" and watched and n.startswith(watched)
        and n not in expected
        and n not in set(groups.known_non_environment_skips))
    if unexpected:
        problems.append(f"unexpected environment skips: {unexpected}")

    out.update({
        "battery_staged_role_nodeid": groups.staged_role_nodeid,
        "battery_staged_role_outcome": staged_role,
        "all_skipped_nodeids": sorted(n for n, s in outcomes.items()
                                      if s == "skipped"),
        "n_skipped": counts["skipped"],
        "skip_set_digest": skip_set_digest(
            [n for n, s in outcomes.items() if s == "skipped"]),
        "skip_reasons": dict(sorted((skip_reasons or {}).items())),
        "skip_set_is_forensic_not_scientific":
            "the complete skip list exists so a pod/sweep divergence can be "
            "NAMED. The total is not a target and carries no scientific "
            "content; only the declared groups above express expectations.",
        "expected_environment_skips": sorted(expected),
        "known_non_environment_skips": list(groups.known_non_environment_skips),
        "unexpected_environment_skips": unexpected,
        "problems": problems,
        "verdict": "PASS" if not problems else "FAIL",
    })
    return out


# --- the record and the gate that reads it ----------------------------------

def self_hash(record: dict[str, Any]) -> str:
    body = {k: v for k, v in record.items() if k != "self_sha256"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def head_commit(repo_root: str | Path = ".") -> str:
    out = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                         capture_output=True, text=True, timeout=60)
    return out.stdout.strip() or "unknown"


def tree_is_clean(repo_root: str | Path = ".") -> bool:
    out = subprocess.run(["git", "-C", str(repo_root), "status", "--porcelain"],
                         capture_output=True, text=True, timeout=60)
    return out.returncode == 0 and not out.stdout.strip()


def permitted_post_sweep_paths(record_path: str) -> tuple[str, ...]:
    """The only tracked path that may differ between the swept and session commits.

    The record is written after the sweep by construction, so its own commit can
    never be part of the swept tree. Which path that is belongs to whoever owns
    the record.
    """
    return (record_path,)


def lineage_from_swept_base(repo_root: Path, base: str | None, commit: str,
                            allowed_paths: tuple[str, ...]) -> dict[str, Any]:
    """`lineage_from_authorized_base`, asked about more than one permitted path.

    It **delegates**: the git plumbing — does the commit exist, does `commit`
    descend from `base`, what differs between them — runs once, in the frozen
    function, and only the permitted-set predicate is re-evaluated here on the
    `changed_paths` it already returned. There is no second copy of the risky
    part, which is what the session gates learned the hard way when two
    hand-written copies of a lineage rule drifted.

    It lives HERE rather than in `session_prechecks` on purpose.
    `session_prechecks.py` is a member of two closed phases' frozen executable
    sets, and generalizing it in place moved both of those digests — for a
    feature neither closed phase will ever use. This module is in the current
    harness alone, so the cost lands where the benefit does.
    """
    from aadistill.infrastructure.session_prechecks import lineage_from_authorized_base

    allowed = tuple(allowed_paths)
    out = lineage_from_authorized_base(repo_root, base, commit,
                                       allowed[0] if allowed else "")
    if out["changed_paths"] is None:
        return out                      # refused before the diff; nothing to widen
    unexpected = [p for p in out["changed_paths"] if p not in allowed]
    out["allowed_paths"] = list(allowed)
    out["unexpected_paths"] = unexpected
    out["ok"] = not unexpected
    if unexpected:
        out["reason"] = (
            f"{len(unexpected)} path(s) other than {list(allowed)} changed "
            f"between the swept base and the session commit: {unexpected[:8]}")
    else:
        out["reason"] = (f"only {sorted(set(out['changed_paths']))} differs from "
                         "the swept base" if out["changed_paths"]
                         else "identical to the swept base")
    return out


#: What a readiness record may claim to be.
#:
#: `diagnostic` proves the machinery and that the suite passes on a tree. It is
#: real readiness evidence and stays valid as such. `launch_bound` is the sweep a
#: maintainer-approved paid launch rests on — the same procedure, run deliberately
#: as the thing the money will rest on rather than as a check that the plumbing
#: works.
#:
#: The distinction was defined on 2026-09-04 and enforced by nothing: the paid
#: gate accepted any PASS record and merely copied the kind into evidence, so a
#: diagnostic sweep could have satisfied gate 12 and the promised launch-bound
#: sweep need never have happened.
RECORD_KINDS: tuple[str, ...] = ("diagnostic", "launch_bound")

#: What `pod_environment_gate` requires before a provider resource is created.
LAUNCH_BOUND: str = "launch_bound"


def verify_record(record: dict[str, Any], repo_root: str | Path = ".", *,
                  session_commit: str | None = None,
                  authorization_path: str | None = None,
                  required_kind: str | None = None,
                  staging_contract_digest: str | None = None,
                  harness_digest: "Callable[[Path], str] | None" = None,
                  named_files: tuple[str, ...] = (),
                  record_path: str | None = None,
                  ) -> tuple[bool, str]:
    """The cheap pre-provider check: does this record still describe live code?

    Never re-runs the sweep. It answers one question — is the thing that was
    proved still the thing that would run — and refuses when it cannot tell.

    Two digests are not enough on their own. They cover the harness and the pod
    test environment, and deliberately ignore `logs/` and `docs/` — but the pod's
    pytest suite READS repository state: `current_state.json`, `STATE.md`,
    `CATALOG.md`. The 2026-09-04 sweep was recorded at `0457bab` and four
    documentation commits landed afterwards without invalidating anything, so the
    record described a suite that had not been run against the tree it was
    certifying.

    Enumerating every pytest data dependency is a losing game — the next one
    added would silently not be in the list. So this asks git instead, reusing
    the session-lineage rule verbatim: the session commit must descend from
    `swept_base_commit`, and the ONLY tracked paths permitted to differ are the
    readiness record itself and, once a session is issued, the canonical
    authorization artifact the existing lineage contract already allows. Any
    other change — `logs/**`, docs, README, preregistration, state, tests,
    source — means the sweep is owed again.
    """
    # `harness_digest` is INJECTED. Which harness a readiness record describes
    # is an experiment-instance fact, and the core reaching into
    # `experiments.phase_c1` to find out inverted the dependency -- the reusable
    # runtime would have named one experiment, and adding a second would have
    # meant editing this file.
    if harness_digest is None:
        return False, (
            "no harness_digest provider was supplied; which harness this record "
            "describes is the caller's fact, and this function will not guess "
            "it. Pass the experiment's digest function.")

    if record.get("schema") != SCHEMA:
        return False, f"unexpected schema {record.get('schema')!r}"
    stored = record.get("self_sha256")
    if not stored or stored != self_hash(record):
        return False, "the record's self-hash does not match its contents"
    if record.get("verdict") != "PASS":
        return False, (f"the recorded sweep verdict is {record.get('verdict')!r}: "
                       f"{record.get('problems')}")

    # AFTER the self-hash check, deliberately: editing `record_kind` in place to
    # promote a diagnostic record is tampering, and must be reported as tampering
    # rather than as a kind mismatch.
    kind = record.get("record_kind")
    if kind not in RECORD_KINDS:
        return False, (f"the record declares record_kind {kind!r}, which is not "
                       f"one of {list(RECORD_KINDS)}; a record that cannot say "
                       "what it is cannot be relied on for anything")
    # The staged view the sweep ran under must be the one this session stages.
    # A sweep that used the pod simulator's GENERIC default HIDDEN_PATHS
    # modelled a machine 55 tests more generous than the real pod and certified
    # a tree that then failed six ways. A launch-bound record must carry a
    # contract derived from the session's own SetupManifest, and it must still
    # describe the live one.
    recorded_staging = record.get("staging_contract_digest")
    if required_kind == LAUNCH_BOUND and not recorded_staging:
        return False, (
            "the record carries no staging_contract_digest, so the sweep it "
            "describes may have used the generic simulator default rather than "
            "this session's staging manifest. That is what aborted attempt 4.")
    if (staging_contract_digest is not None and recorded_staging
            and staging_contract_digest != recorded_staging):
        return False, (
            f"the record was swept under staging contract {recorded_staging[:12]}…, "
            f"the live session stages {staging_contract_digest[:12]}… — relay "
            "inputs, local assets, test ignores or the session kind moved, so the "
            "sweep is owed again")

    if required_kind is not None and kind != required_kind:
        return False, (
            f"this record is {kind!r} and {required_kind!r} is required. A "
            "diagnostic sweep proves the machinery works on a tree; it is not the "
            "sweep a paid launch rests on. Re-run "
            "`record_pod_environment.py --kind launch_bound` on the final CLEAN "
            "PRE-AUTHORIZATION tree -- after the grant and all metadata are "
            "committed and BEFORE the authorization is issued -- then commit only "
            "the readiness record, and only then issue and commit the "
            "authorization. Running it on the authorized tree instead adds a "
            "second path to the lineage diff and session_commit_gate refuses; "
            "that ordering was once reported backwards.")

    try:
        live_harness = harness_digest(repo_root)
        live_env = pod_test_environment_digest(
            repo_root, named_files=named_files)["digest"]
    except Exception as exc:                                   # noqa: BLE001
        return False, f"cannot digest the live tree: {exc}"

    if record.get("c1_harness_digest") != live_harness:
        return False, (f"the record was made against C1 harness "
                       f"{str(record.get('c1_harness_digest'))[:12]}…, the live tree "
                       f"is {live_harness[:12]}… — the pod sweep is owed again")
    if record.get("pod_test_environment_digest") != live_env:
        return False, (f"the record was made against pod test environment "
                       f"{str(record.get('pod_test_environment_digest'))[:12]}…, the "
                       f"live tree is {live_env[:12]}… — the pod sweep is owed again")
    if not record.get("tree_clean"):
        return False, "the sweep was recorded against a dirty working tree"

    # --- post-sweep lineage -------------------------------------------------
    root = Path(repo_root)
    base = record.get("swept_base_commit")
    if not base:
        return False, ("the record names no swept_base_commit, so nothing "
                       "constrains what changed after the sweep")
    target = session_commit or head_commit(root)
    if record_path is None:
        return False, ("no record_path was supplied; which artifact the sweep\n"
                       "writes is the caller's fact and this will not guess it")
    allowed = list(permitted_post_sweep_paths(record_path))
    if authorization_path:
        allowed.append(authorization_path)
    lineage = lineage_from_swept_base(root, base, target, tuple(allowed))
    if not lineage["ok"]:
        return False, (f"post-sweep drift: {lineage['reason']}. The pod suite "
                       "reads repository state, so anything beyond the permitted "
                       "paths means the recorded sweep no longer describes the "
                       "tree that would run")

    c = record.get("counts") or {}
    return True, (f"pod sweep {c.get('passed')} passed / {c.get('skipped')} skipped "
                  f"/ {c.get('failed', 0) + c.get('error', 0)} failed, 7 renderer "
                  f"skips, leaf transport 5/5, binds harness {live_harness[:12]}… "
                  f"and environment {live_env[:12]}…, staging contract "
                  f"{str(recorded_staging)[:12]}…, swept at {base[:8]} with "
                  f"{lineage['reason']}")


def load_record(repo_root: str | Path = ".", *, record_path: str) -> dict[str, Any]:
    return json.loads((Path(repo_root) / record_path).read_text())
