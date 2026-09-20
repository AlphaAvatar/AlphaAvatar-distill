"""Same-campaign continuation across a REPLACEMENT resource, on two filesystems.

The property under test is the one a shared-filesystem test cannot reach. An
earlier continuation test handed the second driver the first driver's own
`--audit-dir`, `--eval-dir` and `--b-workdir`, so it proved that a second
process can read a first process's files. A real replacement pod has none of
them: the previous pod's `audit/probes/*.json` is gone and every `model_dir` it
recorded points at nothing. So every test here builds **two disjoint pod
roots** and moves work between them only through the durable destination on the
launcher host, which is the only copy a replaced resource leaves behind.

**What is replaced and what is not.** Exactly one seam is substituted: the
transport. `scp` becomes a local copy between the two pod roots and the
destination, and `ssh` runs `mkdir`/`test` against those same real
directories. Everything that decides anything runs for real — the manifest
build, the destination re-identification, the on-pod re-identification, the
path derivation, the score-evidence hashing, the campaign predicate, the
descriptor comparison, the ranking commitment and the remaining-work
arithmetic. A fake that stood in for any of those would certify itself; one
already did, in this programme, and an end-to-end harness signed off a
defective line because its fake matched the consumer rather than the producer.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for _p in ("src", "scripts", "scripts/autoinit", "scripts/pod"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from experiments.phase_c2 import behavioural as BH  # noqa: E402
from experiments.phase_c2 import behavioural_continuation as BC  # noqa: E402
from experiments.phase_c2 import behavioural_governance as BG  # noqa: E402
from experiments.phase_c2 import behavioural_schedule as SCH  # noqa: E402

import autoinit_c2_behavioural_launch as L  # noqa: E402
import autoinit_c2_behavioural_driver as D  # noqa: E402

GPU_HARD, DISK_HARD, RUNTIME = 32.7097, 0.5002, 1800.53
ALL_IN = round(GPU_HARD + DISK_HARD, 4)


# ---------------------------------------------------------------------------
# the one substituted seam: transport between two real, disjoint filesystems
# ---------------------------------------------------------------------------

class _Pod:
    """A pod's filesystem, as a local directory tree.

    An absolute pod path like `/workspace/aad/...` maps under `root`, so the
    launcher's real path constants are exercised rather than replaced.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def local(self, remote: str) -> Path:
        return self.root / remote.lstrip("/")


class _Ctx:
    """A `SessionContext` as the restore and durability steps actually use it."""

    def __init__(self, pod: _Pod, store: Path, run_id: str, *,
                 scr: Path | None = None) -> None:
        self.pod = pod
        self.evidence: dict = {}
        self.host = "fake-host"
        self.scp = ("scp", "-P", "2222", "-o", "StrictHostKeyChecking=no")
        self.said: list[str] = []
        self.args = type("A", (), {
            "run_id": run_id, "ckpt_store": str(store),
            "ckpt_fetch_limit_min": 20, "restore_limit_min": 150,
            "scr": str(scr or pod.root / "scr")})()
        self.auth = type("Auth", (), {
            "campaign_id": BG.CAMPAIGN_ID, "gpu_hard_usd": GPU_HARD,
            "disk_hard_usd": DISK_HARD, "all_in_hard_usd": ALL_IN,
            "rate_usd_per_hour": 1.09, "hard_runtime_minutes": RUNTIME})()
        self.target = _Target(pod)

    def say(self, msg: str) -> None:
        self.said.append(msg)


class _Target:
    """`ssh` against the pod's real directory tree."""

    def __init__(self, pod: _Pod) -> None:
        self.pod = pod
        self.port = "2222"

    def run(self, command: str, *, timeout: float):
        #: Rewrite absolute pod paths to this pod's root and run for real, so
        #: `mkdir -p` and `test -s … && echo PRESENT=1` mean what they mean on
        #: a pod. A stubbed "PRESENT=1" would assert nothing about arrival.
        rewritten = command.replace("/workspace", str(self.pod.local("/workspace")))
        out = subprocess.run(["bash", "-c", rewritten], capture_output=True,
                             text=True, timeout=timeout)
        return type("R", (), {"returncode": out.returncode,
                              "stdout": out.stdout, "stderr": out.stderr})()


@pytest.fixture
def transport(monkeypatch):
    """`scp` as a local copy. The ONLY substitution in this module.

    Registered pods are resolved by the `root@host:` target, so a push to a
    replacement pod cannot land in the producing pod's tree by accident — which
    is the confusion the whole module exists to rule out.
    """
    pods: dict[str, _Pod] = {}
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        argv = [str(a) for a in argv]
        if "scp" not in argv:
            #: Not a transfer. `timeout 20m scp …` puts scp at index 2, so
            #: looking only at argv[0] would let every real fetch through to
            #: the real `scp` and fail with rc=1 — which is exactly what the
            #: first version of this fake did.
            return _real_run(argv, **kwargs)
        calls.append(argv)
        #: Positional operands only: drop flags and the VALUES that follow
        #: `-P` and `-o`.
        cleaned, skip = [], False
        for a in argv[argv.index("scp") + 1:]:
            if skip:
                skip = False
                continue
            if a in ("-P", "-o"):
                skip = True
                continue
            if a.startswith("-"):
                continue
            cleaned.append(a)
        assert len(cleaned) >= 2, f"scp with no source/target: {argv}"
        sources, target = cleaned[:-1], cleaned[-1]

        def resolve(spec: str) -> Path:
            if spec.startswith("root@"):
                host, _, remote = spec[len("root@"):].partition(":")
                pod = pods.get(host)
                if pod is None:
                    raise AssertionError(f"scp to unregistered pod {host!r}")
                return pod.local(remote)
            return Path(spec)

        dest = resolve(target)
        rc = 0
        for src in sources:
            s = resolve(src)
            if not s.exists():
                rc = 1
                continue
            d = dest / s.name if (dest.is_dir() or target.endswith("/")) else dest
            d.parent.mkdir(parents=True, exist_ok=True)
            if s.is_dir():
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
        return type("R", (), {"returncode": rc, "stdout": b"", "stderr": b""})()

    _real_run = subprocess.run
    monkeypatch.setattr(L.subprocess, "run", fake_run)
    return type("T", (), {"pods": pods, "calls": calls})()


# ---------------------------------------------------------------------------
# attempt 1: produce probes on pod 1 and secure them to the destination
# ---------------------------------------------------------------------------

def _write_checkpoint(directory: Path) -> Path:
    import torch
    from safetensors.torch import save_file

    directory.mkdir(parents=True, exist_ok=True)
    save_file({"w": torch.zeros(4, 4, dtype=torch.float32)},
              str(directory / "model.safetensors"))
    (directory / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "hidden_size": 4}) + "\n")
    return directory


def _identity(directory: Path) -> dict:
    from aadistill.initialization.specs.arch import get_adapter
    from aadistill.runtime.leaf_durability import identify_for_transfer

    ident = identify_for_transfer(directory, adapter=get_adapter("qwen3"),
                                  arch_signature="sig", num_parameters=16)
    return json.loads(json.dumps(ident.as_dict() if hasattr(ident, "as_dict")
                                 else ident.__dict__, default=str))


def _produce(pod: _Pod, probe_id: str, *, rung: str, arm: str, seed: int,
             scored: bool = True) -> dict:
    """What a real driver leaves on its pod for one finished probe.

    Every recorded path is POD-ABSOLUTE, as a driver running on a pod records
    them — `/workspace/aad/…`, not a host path. The fake transport resolves
    those under the pod's root, which is the whole point: a fixture that
    recorded host paths would make `scp root@host:<hostpath>` resolve by
    accident and prove nothing about the real argv.
    """
    audit = pod.local(L.AUDIT_DIR)
    (audit / "probes").mkdir(parents=True, exist_ok=True)
    model_remote = f"{L.WORKDIR}/_train/{probe_id}/model"
    _write_checkpoint(pod.local(model_remote))
    identity = _identity(pod.local(model_remote))

    record = {"probe_id": probe_id, "campaign": BG.CAMPAIGN_ID,
              "rung": rung, "arm": arm, "seed": seed,
              "model_dir": model_remote,
              "initialization_artifact_digest": f"init-{arm}",
              "config_sha256": "c" * 64, "complete": True,
              "durable": {"identity": identity}}
    if scored:
        rows = [{"id": f"p{i:04d}", "set": "gsm8k", "scorable": True,
                 "correct": i % 2 == 0, "usable": True} for i in range(20)]
        per_sample = audit / f"{probe_id}_per_sample.jsonl"
        per_sample.write_text("".join(json.dumps(r) + "\n" for r in rows))
        result = audit / f"{probe_id}_result.json"
        result.write_text(json.dumps({"correct_overall": 0.5,
                                      "usable_rollout_rate": 1.0}) + "\n")
        from aadistill.infrastructure.manifest import sha256_file

        record["score"] = {
            "probe_id": probe_id, "rung": rung, "arm": arm, "seed": seed,
            "correct_overall": 0.5 if arm == SCH.ANCHOR else 0.9,
            "usable_rollout_rate": 1.0,
            "result_path": f"{L.AUDIT_DIR}/{probe_id}_result.json",
            "result_sha256": sha256_file(result),
            "per_sample_path": f"{L.AUDIT_DIR}/{probe_id}_per_sample.jsonl",
            "per_sample_sha256": sha256_file(per_sample)}
    (audit / "probes" / f"{probe_id}.json").write_text(
        json.dumps(record, indent=2) + "\n")
    return {"unit_id": probe_id, "kind": "probe", "path": model_remote,
            "identity": identity, "campaign": BG.CAMPAIGN_ID}


def _secure(ctx: _Ctx, units: list[dict]) -> list:
    """The launcher's REAL durability path: bytes, then science evidence."""
    scr = Path(ctx.args.scr) / "relay"
    scr.mkdir(parents=True, exist_ok=True)
    (scr / "c2_behavioural_evidence.json").write_text(
        json.dumps({"durable_units": units}) + "\n")
    fetched = L._fetch_and_verify(ctx, units)
    L.secure_probe_evidence(ctx, units)
    L.secure_campaign_ranking(ctx)
    return fetched


def _commit_ranking(pod: _Pod, advanced: str) -> None:
    audit = pod.local(L.AUDIT_DIR)
    audit.mkdir(parents=True, exist_ok=True)
    (audit / BC.RANKING_NAME).write_text(json.dumps({
        "schema": "aadistill.autoinit.c2_screening_ranking/v1",
        "campaign": BG.CAMPAIGN_ID, "run_attempt": "attempt1",
        "ranked": [{"state_id": advanced, "delta_vs_b": 0.4,
                    "frozen_search_rank": 0, "screening_position": 0}],
        "advanced": {"state_id": advanced, "delta_vs_b": 0.4,
                     "advanced": True, "tie_broken": False,
                     "frozen_search_rank": 0}}, indent=2) + "\n")


def _candidate_ids() -> list[str]:
    """The five candidate state ids, from the FROZEN RECORD in the repository.

    Not `candidate_manifest`, which joins the selection to the dev box's
    durable products under `/home/ecs-user/aad-artifacts/...`. This directory
    is the pod selection and runs inside the paid session's blocking gate, on a
    container that has no such path.
    """
    return [leaf.state_id for leaf in BG.candidate_leaves(REPO)]


def _screening_ids() -> list[tuple[str, str, int]]:
    """`(probe_id, arm, seed)` for the real screening rung."""
    proto = BH.protocol(REPO)["behavioural_selection"]
    seed = int(proto["seeds"]["screening"][0])
    return [(f"screening.{a}.s{seed}", a, seed)
            for a in [*_candidate_ids(), SCH.ANCHOR]]


# ---------------------------------------------------------------------------
# A. a paid attempt that finished no probe is still a prior resource
# ---------------------------------------------------------------------------

def _session_record(repo_root: Path, run_id: str, *, created=True,
                    confirmed_gone=True, actual_usd=0.0) -> Path:
    path = repo_root / L.session_record_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "provider_resource_created": created,
        "provider_confirms_gone": confirmed_gone,
        "cost": {"actual_usd": actual_usd}}))
    return path


def _run_manifest(repo_root: Path, run_id: str,
                  campaign: str | None = BG.CAMPAIGN_ID) -> Path:
    path = (repo_root / L.rel_run_dir(L.EXPERIMENT_ID, run_id, L.STAGE_ID)
            / "manifest.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    plan = {} if campaign is None else {"campaign_id": campaign}
    path.write_text(json.dumps({"plan": plan}))
    return path


def _mirror_repo(root: Path) -> Path:
    """The real repository, with `…/phase_c2_behavioural/runs/` writable.

    Two roots are being separated, and conflating them is why this exists. The
    gate reads repo CONTENT — the frozen protocol, the selection, the journal,
    the telemetry, the calibration profiles — which must be the real thing. It
    also reads and this module must WRITE run records, and writing those into
    the actual repository would leave stray evidence of runs that never
    happened, indistinguishable to a later reader from the real thing.

    So everything is symlinked except the one branch that has to be writable,
    which is recreated as real directories with its siblings symlinked. No file
    is copied and nothing in the repository is touched.
    """
    writable = Path(L.rel_run_dir(L.EXPERIMENT_ID, "_", L.STAGE_ID)).parent.parts
    root.mkdir(parents=True, exist_ok=True)
    cursor, source = root, REPO
    for depth, part in enumerate(writable):
        for child in source.iterdir():
            if child.name != part and child.name != ".git":
                (cursor / child.name).symlink_to(child)
        cursor, source = cursor / part, source / part
        cursor.mkdir()
        if depth == len(writable) - 1:
            #: The runs directory itself: real, empty, and ours.
            for child in source.iterdir() if source.is_dir() else ():
                if child.name != "runs":
                    (cursor / child.name).symlink_to(child)
    return root


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = _mirror_repo(tmp_path / "repo")
    monkeypatch.setattr(L, "REPO_ROOT", root)
    return root


def test_a_paid_attempt_with_no_durable_probe_is_still_a_predecessor(
        tmp_path, repo):
    """THE C2 defect: a pod that billed and died before its first probe.

    Discovered from durable probes alone this was invisible, so the next
    attempt called itself the campaign's first resource, summed no spend and
    checked no release — against R9, whose content is that the ceiling is
    cumulative across every resource and that a predecessor must be confirmed
    non-billing.
    """
    store = tmp_path / "store"
    _run_manifest(repo, "attempt1")
    _session_record(repo, "attempt1", actual_usd=3.41)

    prior = L.campaign_attempts(BG.CAMPAIGN_ID, exclude="attempt2",
                                store=store, repo_root=repo)
    assert prior == ["attempt1"], (
        "an attempt that created a provider resource and left no probe is "
        "still a resource of this campaign")

    ctx = _Ctx(_Pod(tmp_path / "pod2"), store, "attempt2")
    ok, why = L.campaign_continuation_gate(ctx)
    assert "first resource" not in why
    assert ctx.evidence["campaign"]["settled_campaign_spend_usd"] == 3.41
    #: A full session's remaining work plus $3.41 does not fit one ceiling.
    assert not ok
    assert "cumulative across every resource" in why


def test_an_unconfirmed_paid_attempt_with_no_probe_refuses(tmp_path, repo):
    """Two resources on the meter, which is what R9 exists to prevent."""
    store = tmp_path / "store"
    _run_manifest(repo, "attempt1")
    _session_record(repo, "attempt1", confirmed_gone=False, actual_usd=0.11)
    ctx = _Ctx(_Pod(tmp_path / "pod2"), store, "attempt2")
    ok, why = L.campaign_continuation_gate(ctx)
    assert not ok
    assert "never confirmed released" in why


def test_a_run_of_another_campaign_is_not_a_predecessor(tmp_path, repo):
    store = tmp_path / "store"
    _run_manifest(repo, "otherrun", campaign="c2-behavioural-some-other")
    _session_record(repo, "otherrun", actual_usd=9.99)
    assert L.campaign_attempts(BG.CAMPAIGN_ID, exclude="attempt1",
                               store=store, repo_root=repo) == []


def test_a_run_whose_campaign_cannot_be_read_is_included(tmp_path, repo):
    """UNKNOWN is a stop condition, not an omission."""
    store = tmp_path / "store"
    _run_manifest(repo, "attempt1", campaign=None)
    assert L.campaign_attempts(BG.CAMPAIGN_ID, exclude="attempt2",
                               store=store, repo_root=repo) == ["attempt1"]
    ctx = _Ctx(_Pod(tmp_path / "pod2"), store, "attempt2")
    ok, why = L.campaign_continuation_gate(ctx)
    assert not ok and "UNKNOWN" in why


def test_a_first_resource_is_still_recognised_as_one(tmp_path, repo):
    """The permitted case, so the refusals above are known to be selective."""
    ctx = _Ctx(_Pod(tmp_path / "pod1"), tmp_path / "store", "attempt1")
    ok, why = L.campaign_continuation_gate(ctx)
    assert ok, why
    assert "first resource" in why
    assert ctx.evidence["campaign"]["remaining_work"][
        "n_probes_remaining"] == 12


# ---------------------------------------------------------------------------
# B. a true fresh-filesystem replacement
# ---------------------------------------------------------------------------

def test_a_replacement_pod_restores_probes_from_the_durable_destination(
        tmp_path, repo, transport):
    """THE C1 property: two disjoint filesystems, joined only by the destination.

    Attempt 1 finishes three screening probes on pod 1 and they are secured —
    bytes and science evidence — to the launcher host. Pod 1 then ceases to
    exist. Attempt 2 is a fresh tree that has never seen any of it, and the
    launcher's real `materialize_inputs` step puts the verified probes there.
    """
    store = tmp_path / "store"
    pod1, pod2 = _Pod(tmp_path / "pod1"), _Pod(tmp_path / "pod2")
    transport.pods["fake-host"] = pod1

    ids = _screening_ids()[:3]
    units = [_produce(pod1, pid, rung="screening", arm=arm, seed=seed)
             for pid, arm, seed in ids]
    ctx1 = _Ctx(pod1, store, "attempt1")
    fetched = _secure(ctx1, units)
    assert all(f["matched"] for f in fetched), fetched

    #: The destination holds bytes AND evidence for each.
    state = BC.campaign_state(L.campaign_store(BG.CAMPAIGN_ID, store))
    assert sorted(state["probes"]) == sorted(p for p, _, _ in ids)
    assert all(state["probes"][p]["scored"] for p, _, _ in ids)

    #: POD 1 IS GONE. Nothing below may read it.
    shutil.rmtree(pod1.root)
    transport.pods["fake-host"] = pod2

    ctx2 = _Ctx(pod2, store, "attempt2")
    assert L.restore_campaign_probes(ctx2) is True, ctx2.said

    manifest_path = pod2.local(L.RESTORE_MANIFEST)
    assert manifest_path.is_file(), "the manifest must always travel"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["campaign_id"] == BG.CAMPAIGN_ID
    assert len(manifest["probes"]) == 3
    for entry in manifest["probes"]:
        #: No entry DIRECTS the pod at the producing pod's location. The
        #: announced identity travels verbatim and its `path` is provenance
        #: that verification never reads; what the driver acts on is
        #: `pod_path`, and that is under this pod's restore root.
        assert "model_dir" not in entry
        assert entry["pod_path"].startswith(L.RESTORE_DIR)
        assert "_train" not in entry["pod_path"]
        assert "_train" not in entry["durable_path"]
        landed = pod2.local(entry["pod_path"])
        assert (landed / "model.safetensors").is_file()
        assert (landed / "config.json").is_file()
        assert (landed / BC.PER_SAMPLE_NAME).is_file()
    #: And nothing outside the identities mentions the producing pod at all.
    without_identities = json.dumps(
        [{k: v for k, v in e.items() if k != "identity"}
         for e in manifest["probes"]])
    assert "_train" not in without_identities


def test_the_replacement_driver_re_identifies_at_the_new_local_path(
        tmp_path, repo, transport):
    """The driver's own admission, on bytes that are on ITS filesystem."""
    store = tmp_path / "store"
    pod1, pod2 = _Pod(tmp_path / "pod1"), _Pod(tmp_path / "pod2")
    transport.pods["fake-host"] = pod1
    ids = _screening_ids()[:2]
    units = [_produce(pod1, pid, rung="screening", arm=arm, seed=seed)
             for pid, arm, seed in ids]
    _secure(_Ctx(pod1, store, "attempt1"), units)
    shutil.rmtree(pod1.root)
    transport.pods["fake-host"] = pod2
    assert L.restore_campaign_probes(_Ctx(pod2, store, "attempt2")) is True

    driver = _driver(pod2, "attempt2")
    summary = driver.restore_campaign()
    assert summary["n"] == 2
    journal = driver.load_campaign_journal()
    assert sorted(r["probe_id"] for r in journal["restored"]) == sorted(
        p for p, _, _ in ids)
    assert journal["rejected"] == []
    #: The journal's model_dir is on THIS pod, and the score's evidence too.
    for pid, _, _ in ids:
        entry = driver.training[pid]
        assert entry["model_dir"].startswith(str(pod2.root))
        assert Path(driver.scores[pid]["per_sample_path"]).is_file()
        assert "_train" not in entry["model_dir"]


def test_a_restored_probe_whose_bytes_changed_in_transit_is_refused(
        tmp_path, repo, transport):
    """The manifest is a document; the bytes are the evidence."""
    import torch
    from safetensors.torch import save_file

    store = tmp_path / "store"
    pod1, pod2 = _Pod(tmp_path / "pod1"), _Pod(tmp_path / "pod2")
    transport.pods["fake-host"] = pod1
    pid, arm, seed = _screening_ids()[0]
    units = [_produce(pod1, pid, rung="screening", arm=arm, seed=seed)
             for pid, arm, seed in _screening_ids()[:1]]
    _secure(_Ctx(pod1, store, "attempt1"), units)
    shutil.rmtree(pod1.root)
    transport.pods["fake-host"] = pod2
    assert L.restore_campaign_probes(_Ctx(pod2, store, "attempt2")) is True

    #: Corrupt the arrival on the replacement pod.
    landed = pod2.local(f"{L.RESTORE_DIR}/{pid}")
    save_file({"w": torch.ones(4, 4, dtype=torch.float32)},
              str(landed / "model.safetensors"))

    driver = _driver(pod2, "attempt2")
    with pytest.raises(D.C2DriverError, match="do not reproduce"):
        driver.restore_campaign()


def test_the_launcher_refuses_to_ship_a_probe_that_no_longer_verifies(
        tmp_path, repo, transport):
    """Re-identified at the DESTINATION too, before a byte is sent."""
    import torch
    from safetensors.torch import save_file

    store = tmp_path / "store"
    pod1, pod2 = _Pod(tmp_path / "pod1"), _Pod(tmp_path / "pod2")
    transport.pods["fake-host"] = pod1
    units = [_produce(pod1, pid, rung="screening", arm=arm, seed=seed)
             for pid, arm, seed in _screening_ids()[:1]]
    _secure(_Ctx(pod1, store, "attempt1"), units)
    shutil.rmtree(pod1.root)
    transport.pods["fake-host"] = pod2

    held = next(iter(BC.campaign_state(
        L.campaign_store(BG.CAMPAIGN_ID, store))["probes"].values()))
    save_file({"w": torch.ones(4, 4, dtype=torch.float32)},
              str(Path(held["durable_path"]) / "model.safetensors"))

    ctx = _Ctx(_Pod(tmp_path / "pod2b"), store, "attempt2")
    transport.pods["fake-host"] = ctx.pod
    assert L.restore_campaign_probes(ctx) is False
    assert any("no longer re-identifies" in m for m in ctx.said)


def test_a_probe_without_a_destination_ack_is_preserved_but_not_reusable(
        tmp_path, repo, transport):
    """Preservation is not permission, and an ack is the admission token."""
    store = tmp_path / "store"
    pod1 = _Pod(tmp_path / "pod1")
    transport.pods["fake-host"] = pod1
    units = [_produce(pod1, pid, rung="screening", arm=arm, seed=seed)
             for pid, arm, seed in _screening_ids()[:1]]
    _secure(_Ctx(pod1, store, "attempt1"), units)

    root = L.campaign_store(BG.CAMPAIGN_ID, store)
    ack = next(root.rglob(BC.ACK_NAME))
    ack.unlink()
    state = BC.campaign_state(root)
    assert state["probes"] == {}
    assert any("no durable_ack" in r["why"] for r in state["rejected"])


# ---------------------------------------------------------------------------
# C-E. continuation at each campaign state
# ---------------------------------------------------------------------------

def _pod_view(pod: _Pod) -> Path:
    """The launcher's manifest, with pod-absolute paths resolved to this root.

    Part of the ONE substituted seam and nothing more: a pod's filesystem is
    rooted at `/`, and this module's pods are rooted at `pod.root`. The
    launcher wrote `/workspace/…` because that is what it writes on a real pod,
    and the driver resolves it literally because that is correct there. So the
    harness applies exactly the mapping it already applies to `scp` and `ssh`.

    Nothing about the manifest's MEANING changes — the entries, identities,
    scores and the committed ranking travel untouched, and the tests still
    assert that what lands is under this pod's restore root and not the
    producing pod's.
    """
    raw = pod.local(L.RESTORE_MANIFEST).read_text()
    manifest = json.loads(raw)
    manifest["pod_root"] = str(pod.local(manifest["pod_root"]))
    for entry in manifest["probes"]:
        entry["pod_path"] = str(pod.local(entry["pod_path"]))
    out = pod.root / "continuation_view.json"
    out.write_text(json.dumps(manifest, indent=1) + "\n")
    return out


def _driver(pod: _Pod, run_attempt: str):
    """The real driver, on this pod's filesystem, with no hardware seam used."""
    args = D.build_parser().parse_args([
        "--campaign", BG.CAMPAIGN_ID, "--run-attempt", run_attempt,
        "--continuation-manifest", str(_pod_view(pod)),
        "--audit-dir", str(pod.local(L.AUDIT_DIR)),
        "--eval-dir", str(pod.local(L.EVAL_DIR)),
        "--b-workdir", str(pod.local(L.ARM_DIR)),
        "--status-path", str(pod.root / "status.txt"),
        "--b-build-minutes", "31", "--probe-train-minutes", "75",
        "--probe-battery-minutes", "45", "--rate", "1.09",
        "--soft-stop-usd", "30", "--authorized-usd", "32"])
    driver = D.C2BehaviouralDriver.__new__(D.C2BehaviouralDriver)
    driver.a = args
    driver.audit = Path(args.audit_dir)
    driver.audit.mkdir(parents=True, exist_ok=True)
    driver.training, driver.scores = {}, {}
    driver.ev = {}
    return driver


def _continue_to(tmp_path, store: Path, transport, *, complete, ranking=None,
                 unscored=()):
    """Produce `complete` probes on pod 1, secure them, and restore onto pod 2."""
    pod1, pod2 = _Pod(tmp_path / "pod1"), _Pod(tmp_path / "pod2")
    transport.pods["fake-host"] = pod1
    units = []
    for pid, rung, arm, seed in complete:
        units.append(_produce(pod1, pid, rung=rung, arm=arm, seed=seed,
                              scored=pid not in unscored))
    if ranking:
        _commit_ranking(pod1, ranking)
    _secure(_Ctx(pod1, store, "attempt1"), units)
    shutil.rmtree(pod1.root)
    transport.pods["fake-host"] = pod2
    ctx = _Ctx(pod2, store, "attempt2")
    assert L.restore_campaign_probes(ctx) is True, ctx.said
    return pod2, ctx


def test_partial_screening_continuation_owes_only_the_rest(tmp_path, repo,
                                                           transport):
    """C. Three of six screening probes done: three remain, and only three."""
    store = tmp_path / "store"
    done = [(pid, "screening", arm, seed)
            for pid, arm, seed in _screening_ids()[:3]]
    pod2, ctx = _continue_to(tmp_path, store, transport, complete=done)

    work = BC.remaining_work(
        REPO, state=BC.campaign_state(L.campaign_store(BG.CAMPAIGN_ID, store)))
    assert len(work["probes_complete"]) == 3
    assert work["n_probes_remaining"] == 9
    #: Only the arms the REMAINING probes need, plus the dearest admissible
    #: candidate for the unranked confirmation rung.
    assert len(work["arms_needed"]) == 4
    assert work["decomposition"]["hard_minutes"] < 1800.53

    driver = _driver(pod2, "attempt2")
    driver.restore_campaign()
    driver.load_campaign_journal()
    assert len(driver.scores) == 3
    #: And the schedule still refuses to rank a partial field.
    probes = SCH.screening_probes(
        [{"state_id": sid, "artifact_digest": "d", "durable_path": "p"}
         for sid in _candidate_ids()],
        {"state_id": SCH.ANCHOR, "artifact_digest": "d", "durable_path": "p"},
        BH.protocol(REPO)["behavioural_selection"]["seeds"]["screening"])
    ok, why = SCH.screening_is_complete(probes, driver.training, driver.scores)
    assert not ok and "may not begin" in why


def test_committed_screening_continuation_keeps_the_same_candidate(
        tmp_path, repo, transport):
    """D. Screening complete and committed: confirmation only, same candidate."""
    store = tmp_path / "store"
    advanced = _candidate_ids()[0]
    done = [(pid, "screening", arm, seed)
            for pid, arm, seed in _screening_ids()]
    pod2, _ = _continue_to(tmp_path, store, transport, complete=done,
                           ranking=advanced)

    state = BC.campaign_state(L.campaign_store(BG.CAMPAIGN_ID, store))
    assert state["committed_candidate"] == advanced
    work = BC.remaining_work(REPO, state=state)
    assert work["n_probes_remaining"] == 6
    assert work["screening_committed"] is True
    #: Exactly the committed candidate and the anchor need rebuilding.
    assert set(work["arms_needed"]) == {advanced, SCH.ANCHOR}

    driver = _driver(pod2, "attempt2")
    driver.restore_campaign()
    committed = driver.committed_ranking(
        driver.audit / "c2_screening_ranking.json")
    assert committed["advanced"]["state_id"] == advanced
    assert committed["campaign"] == BG.CAMPAIGN_ID


def test_partial_confirmation_continuation_owes_only_the_rest(tmp_path, repo,
                                                              transport):
    """E. Screening complete, ranking committed, confirmation 4/6."""
    store = tmp_path / "store"
    advanced = _candidate_ids()[0]
    seeds = BH.protocol(REPO)["behavioural_selection"]["seeds"]["confirmation"]
    done = [(pid, "screening", arm, seed)
            for pid, arm, seed in _screening_ids()]
    for seed in seeds[:2]:
        for arm in (advanced, SCH.ANCHOR):
            done.append((f"confirmation.{arm}.s{seed}", "confirmation", arm,
                         int(seed)))
    pod2, _ = _continue_to(tmp_path, store, transport, complete=done,
                           ranking=advanced)

    work = BC.remaining_work(
        REPO, state=BC.campaign_state(L.campaign_store(BG.CAMPAIGN_ID, store)))
    assert work["n_probes_remaining"] == 2
    assert len(work["probes_complete"]) == 10
    assert set(work["arms_needed"]) == {advanced, SCH.ANCHOR}
    assert work["decomposition"]["hard_minutes"] < 1000.0


def test_a_trained_but_unscored_probe_owes_its_scoring_not_its_training(
        tmp_path, repo, transport):
    """The preregistered policy: it resumes AT SCORING, and scoring reads weights.

    This is why the bytes have to reach the replacement pod at all. A manifest
    of identities would leave an unscored probe unscoreable, and retraining it
    is forbidden.
    """
    store = tmp_path / "store"
    ids = _screening_ids()[:3]
    done = [(pid, "screening", arm, seed) for pid, arm, seed in ids]
    pod2, _ = _continue_to(tmp_path, store, transport, complete=done,
                           unscored={ids[2][0]})

    state = BC.campaign_state(L.campaign_store(BG.CAMPAIGN_ID, store))
    work = BC.remaining_work(REPO, state=state)
    assert ids[2][0] in work["probes_trained_not_scored"]
    assert ids[2][0] in work["probes_remaining"]
    assert len(work["probes_complete"]) == 2
    #: Its bytes are restored anyway, because scoring needs them.
    assert ids[2][0] in work["restore"]["probes"]
    landed = pod2.local(f"{L.RESTORE_DIR}/{ids[2][0]}")
    assert (landed / "model.safetensors").is_file()

    driver = _driver(pod2, "attempt2")
    driver.restore_campaign()
    driver.load_campaign_journal()
    assert ids[2][0] in driver.training
    assert ids[2][0] not in driver.scores, (
        "a probe with no score must resume at scoring, not count as complete")


def test_a_complete_campaign_owes_nothing_and_may_not_continue(tmp_path, repo,
                                                               transport):
    """A verdict is terminal. NO_GO and INCONCLUSIVE are complete results."""
    store = tmp_path / "store"
    advanced = _candidate_ids()[0]
    seeds = BH.protocol(REPO)["behavioural_selection"]["seeds"]["confirmation"]
    done = [(pid, "screening", arm, seed)
            for pid, arm, seed in _screening_ids()]
    for seed in seeds:
        for arm in (advanced, SCH.ANCHOR):
            done.append((f"confirmation.{arm}.s{seed}", "confirmation", arm,
                         int(seed)))
    _continue_to(tmp_path, store, transport, complete=done, ranking=advanced)

    work = BC.remaining_work(
        REPO, state=BC.campaign_state(L.campaign_store(BG.CAMPAIGN_ID, store)))
    assert work["n_probes_remaining"] == 0
    assert len(work["probes_complete"]) == 12

    _run_manifest(repo, "attempt1")
    _session_record(repo, "attempt1", actual_usd=20.0)
    ctx = _Ctx(_Pod(tmp_path / "pod3"), store, "attempt3")
    ok, why = L.campaign_continuation_gate(ctx)
    assert not ok
    assert "TERMINAL" in why


# ---------------------------------------------------------------------------
# F. the remaining-work budget
# ---------------------------------------------------------------------------

def test_a_fresh_campaign_prices_the_full_authorized_session(tmp_path):
    """The default path and the continuation path are ONE derivation."""
    state = BC.campaign_state(tmp_path / "empty")
    work = BC.remaining_work(REPO, state=state)
    assert work["n_probes_remaining"] == 12
    assert len(work["arms_needed"]) == 6
    assert work["restore"]["minutes"] == 0.0
    d = work["decomposition"]
    assert (d["expected_minutes"], d["hard_minutes"]) == (1294.87, 1800.53), (
        "a fresh campaign must still price exactly the authorized session")


def test_remaining_work_shrinks_monotonically_as_probes_complete(tmp_path,
                                                                 repo,
                                                                 transport):
    """More verified probes can only mean less owed."""
    store = tmp_path / "store"
    pod1 = _Pod(tmp_path / "pod1")
    transport.pods["fake-host"] = pod1
    seen, last = [], None
    for pid, arm, seed in _screening_ids():
        seen.append(_produce(pod1, pid, rung="screening", arm=arm, seed=seed))
        _secure(_Ctx(pod1, store, "attempt1"), seen)
        work = BC.remaining_work(
            REPO,
            state=BC.campaign_state(L.campaign_store(BG.CAMPAIGN_ID, store)))
        hard = work["decomposition"]["hard_minutes"]
        if last is not None:
            assert hard < last, (
                f"{len(seen)} probes verified and the bound did not fall: "
                f"{hard} vs {last}")
        last = hard


def test_a_continuation_that_fits_the_remaining_ceiling_is_permitted(
        tmp_path, repo, transport):
    """F. The case the old whole-session reservation refused by construction.

    Reserving a fresh full session per attempt meant `settled + approved >
    approved` for any prior spend above `$0`, so no continuation could ever
    pass. Priced on remaining work, one that fits does.
    """
    store = tmp_path / "store"
    advanced = _candidate_ids()[0]
    seeds = BH.protocol(REPO)["behavioural_selection"]["seeds"]["confirmation"]
    done = [(pid, "screening", arm, seed)
            for pid, arm, seed in _screening_ids()]
    for seed in seeds[:2]:
        for arm in (advanced, SCH.ANCHOR):
            done.append((f"confirmation.{arm}.s{seed}", "confirmation", arm,
                         int(seed)))
    pod2, _ = _continue_to(tmp_path, store, transport, complete=done,
                           ranking=advanced)

    _run_manifest(repo, "attempt1")
    _session_record(repo, "attempt1", actual_usd=20.0)
    ctx = _Ctx(pod2, store, "attempt2")
    ok, why = L.campaign_continuation_gate(ctx)
    assert ok, why
    campaign = ctx.evidence["campaign"]
    assert campaign["settled_campaign_spend_usd"] == 20.0
    assert (campaign["settled_campaign_spend_usd"]
            + campaign["this_session_planned_all_in_usd"]) <= (
        campaign["campaign_approved_all_in_usd"] + BG.DOLLAR_QUANTUM_USD)
    assert campaign["remaining_work"]["n_probes_remaining"] == 2


def test_a_continuation_that_does_not_fit_refuses_and_names_the_decision(
        tmp_path, repo, transport):
    """Fail closed. The experiment is not shortened and the ceiling not raised."""
    store = tmp_path / "store"
    done = [(pid, "screening", arm, seed)
            for pid, arm, seed in _screening_ids()[:1]]
    pod2, _ = _continue_to(tmp_path, store, transport, complete=done)
    _run_manifest(repo, "attempt1")
    _session_record(repo, "attempt1", actual_usd=25.0)
    ctx = _Ctx(pod2, store, "attempt2")
    ok, why = L.campaign_continuation_gate(ctx)
    assert not ok
    assert "maintainer decision" in why
    assert "NOT shortened" in why


def test_a_cheaper_card_does_not_buy_more_remaining_work(tmp_path):
    """The window shortens with price; the WORK does not grow to fill it."""
    state = BC.campaign_state(tmp_path / "empty")
    work = BC.remaining_work(REPO, state=state)
    minutes = work["decomposition"]["hard_minutes"]
    for rate in (0.50, 1.09, 2.00):
        again = BC.remaining_work(REPO, state=state)
        assert again["decomposition"]["hard_minutes"] == minutes, (
            "the remaining work is a property of the campaign, not of the "
            f"price (rate {rate})")
    #: And the deadline still cannot outlive either bound.
    assert BG.window_minutes(2.00, gpu_hard_usd=GPU_HARD,
                             hard_runtime_minutes=RUNTIME) < RUNTIME
    assert BG.window_minutes(0.10, gpu_hard_usd=GPU_HARD,
                             hard_runtime_minutes=RUNTIME) == RUNTIME


def test_the_restore_bound_is_the_slowest_recorded_uplink(tmp_path):
    """A bound takes the slowest observation. An average is not a bound."""
    assert BC.RESTORE_MB_PER_SECOND == 0.23
    assert "slowest" in BC.RESTORE_RATE_BASIS
    one_probe = int(1.11 * 2**30)
    minutes = BC.restore_minutes(one_probe)
    assert 70 < minutes < 95, (
        f"one 1.11 GiB probe bounds at {minutes} min; the figure a reader "
        "needs is that this is real billed pod time")
    #: Monotone, and zero bytes is zero minutes.
    assert BC.restore_minutes(0) == 0.0
    assert BC.restore_minutes(2 * one_probe) > minutes


# ---------------------------------------------------------------------------
# the destination gate, and the `campaign_attempts` cases with no run record
# ---------------------------------------------------------------------------

def test_the_destination_gate_checks_the_volume_the_fetcher_writes_to(
        tmp_path, repo):
    """One flag decides where probes are secured, and every consumer reads it.

    The gate checked the `DURABLE_STORE` constant while `probe_destination`
    honoured `--ckpt-store`, so an operator passing a different store would
    have had capacity verified on a volume nothing writes to — a durability
    check that is green about the wrong disk. No test called this gate at all,
    which is how the two came apart.
    """
    ctx = _Ctx(_Pod(tmp_path / "pod"), tmp_path / "store", "attempt1")
    ok, why = L.destination_gate(ctx)
    assert ok, why
    root = L.campaign_store(BG.CAMPAIGN_ID, ctx.args.ckpt_store)
    assert str(root) in why
    assert L.probe_destination(ctx, "screening.B.s1").is_relative_to(root)


def test_the_destination_gate_refuses_an_unusable_store(tmp_path, repo):
    """A run that trains work it cannot preserve is a run that will lose it."""
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("a file, so no store can be created beneath it")
    ok, why = L.destination_gate(
        _Ctx(_Pod(tmp_path / "pod"), blocked / "store", "attempt1"))
    assert not ok
    assert "unusable" in why


def test_this_run_is_never_its_own_predecessor(tmp_path, repo, transport):
    """A re-entered launcher must not refuse on its own durable probes."""
    store = tmp_path / "store"
    pod = _Pod(tmp_path / "pod2")
    transport.pods["fake-host"] = pod
    units = [_produce(pod, pid, rung="screening", arm=arm, seed=seed)
             for pid, arm, seed in _screening_ids()[:1]]
    _secure(_Ctx(pod, store, "attempt2"), units)
    assert L.campaign_attempts(BG.CAMPAIGN_ID, exclude="attempt2",
                               store=store, repo_root=repo) == []


def test_an_empty_attempt_directory_is_not_a_predecessor(tmp_path, repo):
    """An empty directory is not durable work, and refusing on one would
    strand every campaign whose store was merely created."""
    (L.campaign_store(BG.CAMPAIGN_ID, tmp_path / "store")
     / "attempt1").mkdir(parents=True)
    assert L.campaign_attempts(BG.CAMPAIGN_ID, exclude="attempt2",
                               store=tmp_path / "store",
                               repo_root=repo) == []
