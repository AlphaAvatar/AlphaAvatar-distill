"""P1's launch wiring at $0: what binds a session, what refuses it, what it brings home.

No pod, no provider call, no grant: artifacts are issued in tmp_path from a
test grant document, and the provider reconciliation is injected.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from stages.reasoning_data import assets as A
from stages.reasoning_data import governance as GOV

from aadistill.governance.authorization import AuthorizationError, harness_source_digest
from support.session_specs import load_session_launcher, session_args

L = load_session_launcher("h1_generation_launch")


def _grant(tmp_path, **over) -> Path:
    g = {"granted_by": "test", "decision": "test only", "phase": "P1", "launch_identity_sha256": GOV.launch_hash(),
         "harness_digest": harness_source_digest(A.REPO, files=GOV.HARNESS)["digest"],
         "expected_usd": 1.0, "hard_cap_usd": 2.0, **over}
    g = {k: v for k, v in g.items() if v is not None}
    p = tmp_path / "grant.json"
    p.write_text(json.dumps(g))
    return p


def test_the_launch_identity_binds_the_protocol_and_every_input(tmp_path):
    ident = GOV.launch_identity()
    assert ident["generation_protocol"]["path"] == str(GOV.PROTOCOL)
    assert ident["inputs"] and ident["pools"]
    repo = tmp_path / "repo"
    for f in [ident["generation_plan"], ident["generation_protocol"], *ident["inputs"], *ident["pools"]]:
        rel = f.get("path") or f["file"]
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_bytes((A.REPO / rel).read_bytes())
    assert GOV.launch_hash(repo) == GOV.launch_hash()
    proto = repo / GOV.PROTOCOL                       # a protocol edit moves the identity: it is bound
    proto.write_text(proto.read_text().replace('"v2"', '"v1"', 1))
    assert GOV.launch_hash(repo) != GOV.launch_hash()
    inp = repo / ident["inputs"][0]["file"]            # an input that differs from the plan refuses
    inp.write_text(inp.read_text() + "\n")
    with pytest.raises(AuthorizationError, match="not the plan's"):
        GOV.launch_identity(repo)


def test_the_issuer_needs_a_grant_naming_this_launch_identity(tmp_path):
    out = tmp_path / "authorization.json"
    with pytest.raises(AuthorizationError, match="no grant document at"):
        GOV.issue("r1", grant_path=tmp_path / "missing.json", out=out)
    with pytest.raises(AuthorizationError, match="does not derive"):
        GOV.issue("r1", grant_path=_grant(tmp_path, launch_identity_sha256="0" * 64), out=out)
    with pytest.raises(AuthorizationError, match="not one of"):
        GOV.issue("r1", grant_path=_grant(tmp_path, phase="P2"), out=out)
    with pytest.raises(AuthorizationError, match="grants run 'other'"):
        GOV.issue("r1", grant_path=_grant(tmp_path, run_id="other"), out=out)
    with pytest.raises(AuthorizationError, match="does not derive"):    # P1's identity under P1t's phase
        GOV.issue("r1", grant_path=_grant(tmp_path, phase="P1t"), out=out)
    path = GOV.issue("r1", grant_path=_grant(tmp_path), out=out)
    auth = GOV.load(path)                              # the artifact binds identity and harness
    auth.require_plan(GOV.launch_hash())
    assert auth.require_harness(A.REPO)["digest"] == harness_source_digest(A.REPO, files=GOV.HARNESS)["digest"]
    raw = json.loads(path.read_text())
    assert raw["authorizes_teacher_generation"] and not raw["authorizes_training"] and not raw["authorizes_paid_storage"]
    with pytest.raises(AuthorizationError, match="exists"):
        GOV.issue("r1", grant_path=_grant(tmp_path), out=out)


def test_the_issuer_needs_the_approved_harness_and_never_adopts_a_new_one(tmp_path, monkeypatch):
    out = tmp_path / "authorization.json"
    with pytest.raises(AuthorizationError, match="lacks 'harness_digest'"):
        GOV.issue("h1", grant_path=_grant(tmp_path, harness_digest=None), out=out)
    with pytest.raises(AuthorizationError, match="never adopted at issue"):
        GOV.issue("h1", grant_path=_grant(tmp_path, harness_digest="0" * 64), out=out)
    grant = _grant(tmp_path)                        # approved against today's code ...
    real = GOV.harness_source_digest

    def edited(root, files):                         # ... and the code changes before issue
        d = real(root, files=files)
        return {**d, "digest": hashlib.sha256(d["digest"].encode()).hexdigest()}
    monkeypatch.setattr(GOV, "harness_source_digest", edited)
    with pytest.raises(AuthorizationError, match="executable changed since the grant"):
        GOV.issue("h1", grant_path=grant, out=out)
    assert not out.exists()
    monkeypatch.setattr(GOV, "harness_source_digest", real)
    path = GOV.issue("h1", grant_path=grant, out=out)  # an exact match issues, bound to that digest
    auth = GOV.load(path)
    assert auth.harness_source_digest == json.loads(grant.read_text())["harness_digest"]
    assert auth.stage_conditions["granted_harness_digest"] == auth.harness_source_digest


def test_an_artifact_claiming_more_or_edited_after_issue_is_refused(tmp_path):
    path = GOV.issue("r2", grant_path=_grant(tmp_path), out=tmp_path / "a.json")
    raw = json.loads(path.read_text())
    edited = tmp_path / "edited.json"
    edited.write_text(json.dumps({**raw, "hard_cap_usd": 99.0}))
    with pytest.raises(AuthorizationError, match="edited"):
        GOV.load(edited)
    with pytest.raises(AuthorizationError, match="no authorization at"):
        GOV.load(tmp_path / "none.json")


def test_the_harness_covers_the_pod_paths_whole_import_closure():
    code = ("import sys, importlib\n"
            "for m in ('stages.reasoning_data.generation_driver', 'stages.reasoning_data.generation_adapter',"
            " 'stages.reasoning_data.checkers', 'stages.reasoning_data.supporting_verify',"
            " 'stages.calibration_volume.run_battery', 'stages.calibration_volume.score_battery',"
            " 'shared.evaluation.uncapped_eval', 'aadistill.evaluation.degeneration'):\n"
            "    importlib.import_module(m)\n"
            "print('\\n'.join(sorted(getattr(m, '__file__', '') or '' for m in list(sys.modules.values()))))\n")
    got = subprocess.run([sys.executable, "-c", code], cwd=A.REPO, capture_output=True, text=True,
                         env={"PYTHONPATH": "src:scripts", "PATH": "/usr/bin:/bin"}, check=True).stdout.split()
    root = str(A.REPO) + "/"
    closure = {p[len(root):] for p in got if p.startswith(root) and "/.venv/" not in p
               and not p.endswith("__init__.py")}
    missing = sorted(p for p in closure - set(GOV.HARNESS) if not p.startswith(GOV.HARNESS_EXEMPT_PREFIXES))
    assert not missing, f"executed on the pod but not in the authorized harness: {missing}"


def test_the_session_spec_validates_and_binds_the_launch_identity():
    args = session_args(L)
    spec = L.spec(args)
    spec.validate()
    assert spec.plan_hash == GOV.launch_hash()
    assert "TESTS_OK" not in spec.setup.setup_markers and not spec.setup.test_paths   # no suite on the paid pod
    assert spec.authorization_path == str(GOV.authorization_path(args.run_id))
    cmd = spec.driver_command(type("Ctx", (), {"args": args})(), None)
    assert "-m stages.reasoning_data.generation_driver" in cmd and "--resume-from" not in cmd
    b = spec.budget
    assert b.contingency_fraction > 0 and b.other_phases[0].minutes > 0


def test_the_prechecks_refuse_a_plan_or_commit_that_is_not_the_bound_one(monkeypatch):
    for phase in GOV.PHASE_PLANS:
        ok, why = L.plan_is_the_chosen_admitted_one(type("Ctx", (), {"args": type("A", (), {"phase": phase})()})())
        assert ok, (phase, why)
    readme = subprocess.run(["git", "show", "HEAD:README.md"], cwd=A.REPO, capture_output=True).stdout
    ctx = type("Ctx", (), {"args": type("A", (), {"session_commit": "HEAD", "phase": "P1"})()})()
    ident = {"generation_plan": {"path": "README.md", "sha256": hashlib.sha256(readme).hexdigest()},
             "generation_protocol": {"path": "README.md", "sha256": hashlib.sha256(readme).hexdigest()},
             "inputs": [], "pools": []}
    monkeypatch.setattr(GOV, "launch_identity", lambda root=None, plan_path=None: ident)
    assert L.bound_files_are_in_the_session_commit(ctx)[0]
    ident["pools"] = [{"path": "README.md", "sha256": "0" * 64}]
    ok, why = L.bound_files_are_in_the_session_commit(ctx)
    assert not ok and "README.md" in why


def test_a_launch_without_an_authorization_is_refused_before_any_provider_call(tmp_path, capsys):
    for phase in GOV.PHASE_PLANS:
        rc = L.main(["--phase", phase, "--run-id", "no_such_run", "--scr", str(tmp_path), "--session-commit",
                     "0" * 40, "--bundle", "x", "--max-price", "1.0", "--dry-run"])
        assert rc == 11 and "REFUSED at $0" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="no network volume"):
        L.main(["--phase", "P1t", "--run-id", "r", "--scr", str(tmp_path), "--session-commit", "0" * 40,
                "--bundle", "x", "--max-price", "1.0", "--network-volume-id", "v"])


def test_closeout_rehashes_every_shard_and_reconciles_the_provider(tmp_path, monkeypatch):
    scr, store = tmp_path / "scr", tmp_path / "scr/store"
    store.mkdir(parents=True)
    good = store / "pass0_g0.part0000.jsonl"
    good.write_text('{"task_id": "a"}\n{"task_id": "b"}\n')
    good.with_suffix(".done").write_text(hashlib.sha256(good.read_bytes()).hexdigest() + "\n")
    (store / "pass0_g0.complete").write_text(json.dumps({"label": "pass0_g0", "tasks": 2, "shards": [
        {"name": good.name, "sha256": hashlib.sha256(good.read_bytes()).hexdigest()}]}))
    (store / L.EVIDENCE).write_text(json.dumps({"terminal": "ALL_DONE"}))
    monkeypatch.setattr(GOV, "run_dir", lambda run_id: Path("runs") / run_id)
    (tmp_path / "runs/r/runtime").mkdir(parents=True)
    (tmp_path / "runs/r/runtime/session.json").write_text(json.dumps(
        {"passed": True, "cost": {"actual_usd": 3.3331, "elapsed_minutes": 111.73, "price_per_hour": 1.79}}))
    out = L.closeout("r", scr, repo_root=tmp_path, reconcile=lambda: {"ok": True, "why": "0 pods"})
    assert out["complete"] and out["shards"] == {"secured": 1, "unverified": [], "records": 2}
    assert out["status"] == "COMPLETE" and out["money"]["all_in_usd"] == 3.3714      # GPU + 150 GB disk
    sys.path.insert(0, str(A.REPO / "scripts/maintenance/consolidation"))
    from closeout_reader import closeout_cost, closeout_verdict                     # what the navigation reads
    assert closeout_verdict(out) == "COMPLETE" and closeout_cost(out) == 3.3714
    assert out["provider_reconciliation_after"]["ok"]
    good.write_text('{"task_id": "a"}\n')                 # a shard that no longer matches its marker
    out = L.closeout("r", scr, repo_root=tmp_path, reconcile=lambda: {"ok": True, "why": "0 pods"})
    assert not out["complete"] and out["shards"]["unverified"] == [good.name]


def _setup_branch() -> str:
    text = (A.REPO / "scripts/shared/pod/autoinit_preflight_setup.sh").read_text()
    start = text.index('elif [ "$SESSION_KIND" = "stage2_h1_generation" ]; then\n') + len(
        'elif [ "$SESSION_KIND" = "stage2_h1_generation" ]; then\n')
    return text[start:text.index("\nelif [", start)]


@pytest.mark.parametrize("phase", sorted(GOV.PHASE_PLANS))
@pytest.mark.parametrize("plan_hash,rc", [("current", 0), ("0" * 64, 98), ("other phase", 98)])
def test_the_pods_setup_branch_runs_and_fails_closed(tmp_path, phase, plan_hash, rc):
    """The branch the pod executes, executed: it loads the artifact by its own
    type, recomputes the launch identity of the plan the artifact's phase names
    and the harness from the checkout, and a mismatch -- including the other
    phase's identity -- marks AUTHORIZATION_MISMATCH and exits 98."""
    lid = GOV.launch_hash(A.REPO, GOV.PHASE_PLANS[phase])
    other = next(GOV.launch_hash(A.REPO, p) for k, p in GOV.PHASE_PLANS.items() if k != phase)
    path = GOV.issue("setup_probe", grant_path=_grant(tmp_path, phase=phase, launch_identity_sha256=lid),
                     out=tmp_path / "auth.json")
    script = ('say() { echo "SAY $*"; }; mark() { echo "MARK $*"; }\n' + _setup_branch() + "\nexit 0\n")
    env = {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "REPO": str(A.REPO),
           "SESSION_AUTH_PATH": str(path),
           "SESSION_PLAN_HASH": {"current": lid, "other phase": other}.get(plan_hash, plan_hash)}
    script = script.replace("/opt/train/bin/python", sys.executable)
    res = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, cwd=A.REPO)
    assert res.returncode == rc, res.stdout + res.stderr
    if rc == 0:
        assert "setup_probe" in res.stdout and f"launch {lid[:12]}" in res.stdout
    else:
        assert "MARK AUTHORIZATION_MISMATCH" in res.stdout


# -- securing shards: a failed listing is UNKNOWN, never an empty success -----

from aadistill.infrastructure.remote import CommandResult  # noqa: E402


class Pod:
    """A pod's shard directory behind a scripted ssh."""

    def __init__(self, files: dict[str, bytes], result=None):
        self.files, self.result = files, result

    def run(self, command, *, timeout):
        if self.result is not None:
            return self.result
        if "for f in pass*.part*.done" in command:
            lines = [f"{n[:-6]}.done {hashlib.sha256(b).hexdigest()}" for n, b in sorted(self.files.items())]
            return CommandResult(0, "\n".join(lines + [L.LISTED]) + "\n", "")
        return CommandResult(0, "", "")


def _ctx(tmp_path, pod):
    return type("Ctx", (), {"target": pod, "host": "h", "evidence": {}, "say": staticmethod(lambda m: None),
                            "args": type("A", (), {"scr": str(tmp_path)})()})()


@pytest.fixture
def fetch(monkeypatch):
    """scp from the scripted pod; `corrupt` names files that arrive damaged."""
    corrupt = set()

    def scp(ctx, remote, dest, *, limit_s=1200):
        name = Path(remote).name
        if name in corrupt and name.endswith(".raise"):
            raise OSError("scp died")
        if f"{name}.raise" in corrupt:
            raise OSError("scp died")
        if ctx.target is None or name not in getattr(ctx.target, "files", {}):
            return 1
        data = ctx.target.files[name]
        Path(dest).write_bytes(data[:-1] if name in corrupt else data)
        return 0
    monkeypatch.setattr(L, "_scp", scp)
    return corrupt


@pytest.mark.parametrize("result,why", [
    (CommandResult(124, "", "", timed_out=True), "timed out"),
    (CommandResult(255, "", "ssh: connect to host"), "exited 255"),
    (CommandResult(0, "", ""), "no sentinel"),                      # an empty stdout is not an empty pod
    (CommandResult(0, "pass0.part0000.done not-a-digest\n" + "__H1_SHARD_LISTING_COMPLETE__\n", ""),
     "unreadable"),
])
def test_a_listing_that_failed_is_unknown_never_zero_shards_secured(tmp_path, fetch, result, why):
    sleeps = []
    ctx = _ctx(tmp_path, Pod({}, result))
    ok, msg = L.shards_secured(ctx, sleep=sleeps.append)
    assert not ok and msg.startswith("UNKNOWN") and why in msg
    assert len(sleeps) == L.SECURE_ATTEMPTS - 1                     # bounded, then recorded
    assert ctx.evidence["shards_unverified"]["attempts"] == L.SECURE_ATTEMPTS


def test_no_target_is_unknown(tmp_path, fetch):
    ok, msg = L.shards_secured(_ctx(tmp_path, None), sleep=lambda s: None)
    assert not ok and "no target" in msg


def test_a_pod_that_really_holds_no_shard_is_secured_as_such(tmp_path, fetch):
    ok, msg = L.shards_secured(_ctx(tmp_path, Pod({})), sleep=lambda s: None)
    assert ok and "holds no sealed shard" in msg


def test_listed_shards_are_secured_by_their_local_bytes_not_by_a_flag(tmp_path, fetch):
    files = {"pass0_g0.part0000.jsonl": b'{"task_id": "a"}\n', "pass0_g0.part0001.jsonl": b'{"task_id": "b"}\n'}
    ctx = _ctx(tmp_path, Pod(files))
    ok, msg = L.shards_secured(ctx, sleep=lambda s: None)
    assert ok and "2 sealed shard(s)" in msg
    assert all((tmp_path / "store" / n).read_bytes() == b for n, b in files.items())
    (tmp_path / "store" / "pass0_g0.part0001.jsonl").write_bytes(b"damaged")   # the flag still says ok ...
    assert ctx.evidence["shards_secured"]["pass0_g0.part0001.jsonl"]["ok"]
    ok, _ = L.shards_secured(ctx, sleep=lambda s: None)                  # ... the bytes are re-fetched
    assert ok and (tmp_path / "store" / "pass0_g0.part0001.jsonl").read_bytes() == files["pass0_g0.part0001.jsonl"]


def test_a_shard_that_arrives_damaged_is_not_secured(tmp_path, fetch):
    files = {"pass0_g0.part0000.jsonl": b'{"task_id": "a"}\n'}
    fetch.add("pass0_g0.part0000.jsonl")
    ok, msg = L.shards_secured(_ctx(tmp_path, Pod(files)), sleep=lambda s: None)
    assert not ok and msg.startswith("UNVERIFIED") and "pass0_g0.part0000.jsonl" in msg


def test_the_poll_hook_records_an_unknown_listing_and_secures_nothing(tmp_path, fetch):
    ctx = _ctx(tmp_path, Pod({}, CommandResult(255, "", "")))
    L.secure_shards(ctx)
    assert ctx.evidence["listing_unknown"] and not ctx.evidence.get("shards_secured")


def test_a_stale_secured_flag_never_stands_in_for_the_bytes(tmp_path, fetch):
    files = {"pass0_g0.part0000.jsonl": b'{"task_id": "a"}\n', "pass0_g0.part0001.jsonl": b'{"task_id": "b"}\n'}
    ctx = _ctx(tmp_path, Pod(files))
    assert L.shards_secured(ctx, sleep=lambda s: None)[0]
    (tmp_path / "store" / "pass0_g0.part0001.jsonl").unlink()          # the local copy is lost ...
    fetch.add("pass0_g0.part0001.jsonl.raise")                          # ... and the re-fetch dies
    assert ctx.evidence["shards_secured"]["pass0_g0.part0001.jsonl"]["ok"]   # a stale flag says ok
    ok, msg = L.shards_secured(ctx, sleep=lambda s: None)
    assert not ok and "pass0_g0.part0001.jsonl" in msg


def test_a_blocking_download_cannot_outlive_the_one_deadline(tmp_path, monkeypatch):
    """Every copy and listing gets at most the time left; a copy that hangs to
    its own limit uses that time up, and what is left undone is recorded --
    the verified shard stays verified and the check returns within the deadline."""
    clock = [0.0]
    monkeypatch.setattr(L, "_now", lambda: clock[0])
    files = {"pass0_g0.part0000.jsonl": b'{"task_id": "a"}\n'}
    given = []

    class SlowPod(Pod):
        def run(self, command, *, timeout):
            given.append(("run", timeout, clock[0]))
            if "ls *.complete" in command:
                return CommandResult(0, "pass0_g0.complete pass1_g0.complete\n", "")
            return super().run(command, timeout=timeout)

    def scp(ctx, remote, dest, *, limit_s=1200):
        name = Path(remote).name
        given.append((name, limit_s, clock[0]))
        if name.endswith(".complete"):            # this download hangs until its own limit
            clock[0] += limit_s
            return 124
        Path(dest).write_bytes(files[name])
        return 0
    monkeypatch.setattr(L, "_scp", scp)
    ctx = _ctx(tmp_path, SlowPod(files))
    ok, msg = L.shards_secured(ctx, sleep=lambda s: None)
    assert ok and "1 sealed shard(s)" in msg                       # the verified shard is kept
    assert all(limit <= L.SECURE_BUDGET_SECONDS - at + 1e-9 for _n, limit, at in given)
    assert clock[0] <= L.SECURE_BUDGET_SECONDS                     # nothing ran past the deadline
    names = [n for n, _l, _a in given]
    assert names.count("pass0_g0.complete") == 1 and "pass1_g0.complete" not in names
    assert {"pass1_g0.complete", L.EVIDENCE} <= set(ctx.evidence["secure_deadline_skipped"])


def test_a_shard_download_that_hangs_leaves_the_rest_unverified_within_the_deadline(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(L, "_now", lambda: clock[0])
    files = {"pass0_g0.part0000.jsonl": b"a\n", "pass0_g0.part0001.jsonl": b"b\n"}

    def scp(ctx, remote, dest, *, limit_s=1200):
        clock[0] += limit_s                         # every download hangs to its limit
        return 124
    monkeypatch.setattr(L, "_scp", scp)
    ctx = _ctx(tmp_path, Pod(files))
    ok, msg = L.shards_secured(ctx, sleep=lambda s: None)
    assert not ok and msg.startswith("UNVERIFIED") and clock[0] <= L.SECURE_BUDGET_SECONDS
    assert ctx.evidence["shards_unverified"]["why"] == msg
    assert "pass0_g0.part0001.jsonl" in ctx.evidence["secure_deadline_skipped"]
