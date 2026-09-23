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

import inspect
import json
import re
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
from experiments.phase_c2 import scoring as C2S  # noqa: E402

import autoinit_c2_behavioural_launch as L  # noqa: E402
import autoinit_c2_behavioural_driver as D  # noqa: E402

from conftest import needs_host_local_stores  # noqa: E402

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
                 scr: Path | None = None, all_in: float = ALL_IN,
                 campaign: float | None = None) -> None:
        self.pod = pod
        self.evidence: dict = {}
        self.host = "fake-host"
        self.scp = ("scp", "-P", "2222", "-o", "StrictHostKeyChecking=no")
        self.said: list[str] = []
        self.args = type("A", (), {
            "run_id": run_id, "ckpt_store": str(store),
            "ckpt_fetch_limit_min": 20,
            #: The attached network volume the campaign's probes are
            #: pre-staged on. Spelled exactly as the launcher's own defaults,
            #: because `volume_probe_root` builds the pod path from these and
            #: a test that invented its own mount would exercise a path no
            #: session uses.
            "network_volume_id": L.CAMPAIGN_VOLUME_ID,
            "volume_mount_path": L.VOLUME_MOUNT,
            "volume_gb": L.CAMPAIGN_VOLUME_GB,
            "data_center_ids": L.VOLUME_DATACENTER,
            "scr": str(scr or pod.root / "scr")})()
        #: The two ceilings default to the same figure so that the tests
        #: written when they WERE the same figure keep testing the gate's
        #: arithmetic. Which of the two the gate actually reads is asserted
        #: separately, by driving them apart.
        self.auth = type("Auth", (), {
            "campaign_id": BG.CAMPAIGN_ID, "gpu_hard_usd": GPU_HARD,
            "disk_hard_usd": DISK_HARD, "all_in_hard_usd": all_in,
            "campaign_all_in_hard_usd": all_in if campaign is None else campaign,
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
        #: `mkdir -p`, `test -s … && echo PRESENT=1` and `cat <index>` mean
        #: what they mean on a pod. A stubbed "PRESENT=1" would assert nothing
        #: about arrival.
        #:
        #: BOTH pod-absolute roots, not just the workspace: the campaign's
        #: probes now live on an attached volume, so a rewrite that knew only
        #: `/workspace` would send every presence check to a path outside the
        #: fixture and report absence for probes that are there.
        #: AT PATH BOUNDARIES ONLY. A plain `str.replace` of the mount prefix
        #: rewrites it wherever it appears, including inside a FILENAME:
        #: `.../evidence/<pid>/durable_ack.json` contains `/durable`, so the
        #: naive version turned it into `.../<pid><pod-root>/durable_ack.json`
        #: and the presence check reported a file that had arrived perfectly as
        #: absent. A fixture that fabricates a failure is worse than one that
        #: misses a real one, because the failure it invents is investigated.
        rewritten = command
        for prefix in ("/workspace", L.VOLUME_MOUNT):
            rewritten = re.sub(
                re.escape(prefix) + r"(?=/|\s|$)",
                str(self.pod.local(prefix)).replace("\\", "\\\\"),
                rewritten)
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
                #: A driver run in-process is given HOST paths for its audit
                #: and eval directories, so what it records is already inside
                #: this pod's root and must not be prefixed twice. A
                #: pod-absolute path (`/workspace/…`) still maps.
                if Path(remote).is_relative_to(pod.root):
                    return Path(remote)
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

def _prestage(pod: _Pod, store: Path, *, campaign_id: str = BG.CAMPAIGN_ID,
              only: set[str] | None = None) -> Path:
    """Put the destination's probes on this pod's attached volume.

    What `scripts/autoinit/stage_c2_probes_to_volume.py` does to a real
    volume, against this fixture's filesystem: the probe directories are
    COPIED — real bytes, at the flat per-probe layout the launcher derives —
    and a `staged_index.json` records which attempt each copy came from.

    This replaces what used to be a `scp` per probe during the session. The
    launcher's job is no longer to move them but to establish that they are
    here and are the right ones, so a test that faked the index instead of
    copying would be asserting against its own fixture rather than against
    bytes.
    """
    root = pod.local(f"{L.VOLUME_MOUNT}/{campaign_id}/probes")
    root.mkdir(parents=True, exist_ok=True)
    state = BC.campaign_state(store / campaign_id)
    index = {"schema": "aadistill.autoinit.c2_probe_volume_index/v2",
             "campaign_id": campaign_id, "probe_root": str(root),
             "staged_utc": "2026-09-23T00:00:00Z", "probes": {}}
    for pid, held in sorted(state["probes"].items()):
        if only is not None and pid not in only:
            continue
        shutil.copytree(held["durable_path"], root / pid, dirs_exist_ok=True)
        index["probes"][pid] = {
            "source_attempt": held["attempt"],
            "artifact_digest": (held["identity"] or {}).get("artifact_digest"),
            "bytes": held["bytes"], "scored": held["scored"]}
    (root / "staged_index.json").write_text(json.dumps(index, indent=1) + "\n")
    return root


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
              "initialization_artifact_digest": _init_digest(arm),
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


def _init_digest(arm: str) -> str:
    """The REAL initialization digest the schedule derives for this arm.

    Not a placeholder. `assert_reuse_matches` compares a restored probe's
    descriptor against the descriptor the rung builds from the frozen record,
    so a fixture using `init-<arm>` describes a probe no rung would accept —
    and any test that drove the real schedule would fail for the wrong reason.
    """
    if arm == SCH.ANCHOR:
        return BH.b_binding(REPO, device="cuda")[
            "required_identity"]["artifact_digest"]
    return next(leaf.artifact_digest for leaf in BG.candidate_leaves(REPO)
                if leaf.state_id == arm)


def _screening_ids() -> list[tuple[str, str, int]]:
    """`(probe_id, arm, seed)` for the real screening rung."""
    proto = BH.protocol(REPO)["behavioural_selection"]
    seed = int(proto["seeds"]["screening"][0])
    return [(f"screening.{a}.s{seed}", a, seed)
            for a in [*_candidate_ids(), SCH.ANCHOR]]


# ---------------------------------------------------------------------------
# A. a paid attempt that finished no probe is still a prior resource
# ---------------------------------------------------------------------------

def _refuse_to_write_into_the_repository(path: Path) -> Path:
    """Never write a run artifact into the actual repository.

    The mirror is supposed to make this impossible, and for one commit it did
    not: it symlinked the real `runs/` children into the tmp tree, so
    `_run_manifest` and `_session_record` wrote THROUGH those symlinks and left
    a fabricated session record in the repository claiming a provider resource
    had been created and $1.09 spent. `campaign_continuation_gate` would have
    read that as a real predecessor, and `open_run` would have refused the real
    launch as an already-recorded run.

    So the helpers check the destination rather than trusting the fixture. A
    guard at the point of writing costs nothing and does not depend on another
    function being right.
    """
    resolved = path.resolve()
    if resolved.is_relative_to(REPO):
        raise AssertionError(
            f"refusing to write {resolved}: it is inside the repository. A run "
            "artifact written here is indistinguishable, to a later reader, "
            "from evidence of a run that actually happened.")
    return path


def _session_record(repo_root: Path, run_id: str, *, created=True,
                    confirmed_gone=True, actual_usd=0.0,
                    elapsed_minutes: float | None = None,
                    omit_elapsed: bool = False) -> Path:
    """A predecessor's session record, in the shape `SessionRunner` writes.

    `elapsed_minutes` is not decoration: the container disk is billed for the
    pod's whole lifetime and the runner records only GPU dollars, so the
    predecessor's disk has to be derived from its minutes. A record without
    them is UNKNOWN, which `omit_elapsed` exists to exercise.

    Defaults to the minutes that GPU spend implies at the authorized rate, so
    a test that only cares about dollars stays self-consistent.
    """
    path = _refuse_to_write_into_the_repository(
        repo_root / L.session_record_path(run_id))
    path.parent.mkdir(parents=True, exist_ok=True)
    cost: dict = {"actual_usd": actual_usd}
    if not omit_elapsed:
        cost["elapsed_minutes"] = (
            actual_usd / 1.09 * 60.0 if elapsed_minutes is None
            else elapsed_minutes)
    path.write_text(json.dumps({
        "provider_resource_created": created,
        "provider_confirms_gone": confirmed_gone,
        "cost": cost}))

    #: AND ITS CLOSEOUT, because a predecessor that billed publishes one. The
    #: fixture wrote only the session record, which made every simulated
    #: predecessor look like an attempt that spent money and never published
    #: what it spent — a real and serious state, but not the ordinary one, and
    #: modelling only it hid the reconciliation between the two figures from
    #: every test here.
    #:
    #: `money.all_in_usd` is the shape the closeout publishes, and it is
    #: all-in: GPU plus the separately billed container disk, derived from the
    #: same basis the gate uses so the two agree by construction rather than by
    #: a number typed here.
    if not omit_elapsed:
        minutes = cost["elapsed_minutes"]
        all_in = (0.0 if not created
                  else round(actual_usd + _expected_disk_usd(minutes), 4))
        closeout = (repo_root / L.rel_run_dir(L.EXPERIMENT_ID, run_id,
                                              L.STAGE_ID)
                    / "closeout/outcome.json")
        closeout.parent.mkdir(parents=True, exist_ok=True)
        closeout.write_text(json.dumps({
            "provider_resource_created": created,
            "money": {"all_in_usd": all_in}}, indent=1) + "\n")
    return path


def _expected_disk_usd(minutes: float) -> float:
    """The disk the gate must charge a predecessor, from the SAME basis.

    Ceiled to the 4-decimal quantum, because a derived spend accumulating
    against a ceiling rounds UP.
    """
    import math

    return math.ceil(DISK_HARD / RUNTIME * minutes * 10_000) / 10_000


def _run_manifest(repo_root: Path, run_id: str,
                  campaign: str | None = BG.CAMPAIGN_ID) -> Path:
    path = _refuse_to_write_into_the_repository(
        repo_root / L.rel_run_dir(L.EXPERIMENT_ID, run_id, L.STAGE_ID)
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

    So everything is symlinked except the branches that have to be writable,
    which are recreated as real directories with their siblings symlinked. No
    file is copied and nothing in the repository is touched.

    **A symlinked branch is a live wire.** There are two writable branches now,
    and the second was added after the first version of the staging-record
    helper wrote through `logs/` — a symlink — into the ACTUAL repository, and
    overwrote the record of a staging run that was in flight at the time. That
    is precisely the failure the paragraph above says this function prevents,
    and it prevents it only for the branches it is told about.
    """
    #: The writable LEAVES. Each is materialized real and EMPTY; everything
    #: above them is real with its other children symlinked; everything else is
    #: a symlink to the real repository.
    #:
    #: The staging leaf is the directory that HOLDS the records, not `logs` or
    #: `logs/shared` — those have siblings this module reads, and making either
    #: of them the leaf would either empty them or fail this function's own
    #: emptiness check.
    writable = {
        #: run records this module writes
        Path(L.rel_run_dir(L.EXPERIMENT_ID, "_", L.STAGE_ID)).parent.parts,
        #: staging records `staged_probe_index` reads. From the ROOT the
        #: launcher declares, never by slicing the glob: the two branches
        #: overlap under one experiment directory, and a slice that guessed
        #: the depth would make a shared ancestor the empty leaf.
        Path(L.STAGING_RECORD_ROOT).parts,
    }
    real: set[tuple[str, ...]] = set()
    for branch in writable:
        for depth in range(1, len(branch) + 1):
            real.add(branch[:depth])

    root.mkdir(parents=True, exist_ok=True)
    for rel in sorted(real, key=len):
        (root / Path(*rel)).mkdir(parents=True, exist_ok=True)
    for rel in sorted({()} | real, key=len):
        #: A LEAF SYMLINKS NOTHING. At a leaf the source directory's children
        #: are exactly the records this module must not inherit — the previous
        #: version's filter excluded a directory NAME, which at leaf depth
        #: matched none of them, and every test that asserts "this campaign has
        #: no predecessor" silently inherited the real repository's runs.
        if rel in writable:
            continue
        source = REPO / Path(*rel)
        if not source.is_dir():
            continue
        for child in source.iterdir():
            if child.name == ".git" or rel + (child.name,) in real:
                continue
            link = root / Path(*rel) / child.name
            if not link.exists() and not link.is_symlink():
                link.symlink_to(child)

    #: Each writable LEAF is left REAL, EMPTY and ours. The runs leaf used to
    #: symlink its source's children under a filter that excluded a name called
    #: "runs" — but at that depth the source IS the runs directory, so its
    #: children are ATTEMPTS and the filter excluded none of them. That was
    #: invisible while the repository had no behavioural runs and became a
    #: three-test failure the moment attempt1's grant created one: every test
    #: that asserts "this campaign has no predecessor" inherited a real
    #: attempt1 from the actual repository.
    #:
    #: The launch-bound sweep caught it, which is what a launch-bound sweep is
    #: for — the same three tests run in the paid pod's blocking TESTS_OK gate,
    #: on a tree that always has a run directory in it by the time a pod
    #: exists.
    for branch in writable:
        leaf = root / Path(*branch)
        assert not any(leaf.iterdir()), f"{leaf} must be empty"
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
    #: ALL-IN: the record's GPU actual PLUS the disk that pod's minutes imply.
    camp = ctx.evidence["campaign"]
    minutes = 3.41 / 1.09 * 60.0
    assert camp["settled_campaign_gpu_usd"] == pytest.approx(3.41, abs=1e-4)
    assert camp["settled_campaign_disk_usd"] == pytest.approx(
        _expected_disk_usd(minutes), abs=1e-4)
    assert camp["settled_campaign_spend_usd"] == pytest.approx(
        3.41 + _expected_disk_usd(minutes), abs=1e-4)
    assert camp["settled_campaign_spend_usd"] > 3.41, (
        "the predecessor's container disk is missing from the cumulative "
        "campaign spend again")
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

    _prestage(pod2, store)
    ctx2 = _Ctx(pod2, store, "attempt2")
    assert L.restore_campaign_probes(ctx2) is True, ctx2.said

    manifest_path = pod2.local(L.RESTORE_MANIFEST)
    assert manifest_path.is_file(), "the manifest must always travel"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["campaign_id"] == BG.CAMPAIGN_ID

    #: ARTIFACTS FOLLOW CONSUMERS. All three probes are completed and validly
    #: scored, so nothing remaining reads their weights and none of them is in
    #: the `probes` section at all. What travels is their evidence.
    assert manifest["probes"] == [], (
        "a completed and validly scored probe's checkpoint was named for "
        "restore; no remaining operation reads it")
    assert len(manifest["evidence"]) == 3
    for entry in manifest["evidence"]:
        #: No entry DIRECTS the pod at the producing pod's location. The
        #: announced identity travels verbatim and its `path` is provenance
        #: that verification never reads; what the driver acts on is
        #: `pod_path`.
        assert "model_dir" not in entry
        assert entry["pod_path"].startswith(L.EVIDENCE_DIR)
        assert "_train" not in entry["pod_path"]
        assert "_train" not in entry["durable_path"]
        landed = pod2.local(entry["pod_path"])
        for name in BC.EVIDENCE_NAMES:
            assert (landed / name).is_file(), name
        #: And the checkpoint is NOT there. That is the point.
        assert not (landed / "model.safetensors").exists(), (
            "22 GiB of weights followed evidence nothing reads")
    #: And nothing outside the identities mentions the producing pod at all.
    without_identities = json.dumps(
        [{k: v for k, v in e.items() if k != "identity"}
         for e in manifest["evidence"]])
    assert "_train" not in without_identities


def test_the_replacement_driver_re_identifies_at_the_new_local_path(
        tmp_path, repo, transport):
    """The driver's own admission, on bytes that are on ITS filesystem.

    Two probes, and they are admitted by DIFFERENT routes because they are in
    different lifecycle states. The unscored one's weights are what the scorer
    will read, so they travel and are re-identified from the bytes that arrived
    here. The scored one contributes rows the verdict reads; its checkpoint has
    no remaining consumer and stays archival.
    """
    store = tmp_path / "store"
    pod1, pod2 = _Pod(tmp_path / "pod1"), _Pod(tmp_path / "pod2")
    transport.pods["fake-host"] = pod1
    scored_id, unscored_id = [p for p, _, _ in _screening_ids()[:2]]
    units = [_produce(pod1, pid, rung="screening", arm=arm, seed=seed,
                      scored=pid != unscored_id)
             for pid, arm, seed in _screening_ids()[:2]]
    _secure(_Ctx(pod1, store, "attempt1"), units)
    shutil.rmtree(pod1.root)
    transport.pods["fake-host"] = pod2
    _prestage(pod2, store)
    assert L.restore_campaign_probes(_Ctx(pod2, store, "attempt2")) is True

    driver = _driver(pod2, "attempt2")
    summary = driver.restore_campaign()
    assert summary["n"] == 1, "only the unscored probe's weights travel"
    assert summary["n_evidence"] == 1
    journal = driver.load_campaign_journal()
    assert sorted(r["probe_id"] for r in journal["restored"]) == sorted(
        [scored_id, unscored_id])
    assert journal["rejected"] == []

    #: The unscored one: weights on THIS pod, re-identified from them.
    entry = driver.training[unscored_id]
    assert entry["model_dir"].startswith(str(pod2.root))
    assert "_train" not in entry["model_dir"]
    assert not entry.get("evidence_only")

    #: The scored one: no model_dir at all, because a path that cannot be
    #: checked is not evidence, and its rows are here and hash-checked.
    scored = driver.training[scored_id]
    assert scored["evidence_only"] is True
    assert "model_dir" not in scored
    assert Path(driver.scores[scored_id]["per_sample_path"]).is_file()


def test_a_restored_probe_whose_bytes_changed_in_transit_is_refused(
        tmp_path, repo, transport):
    """The manifest is a document; the bytes are the evidence."""
    import torch
    from safetensors.torch import save_file

    store = tmp_path / "store"
    pod1, pod2 = _Pod(tmp_path / "pod1"), _Pod(tmp_path / "pod2")
    transport.pods["fake-host"] = pod1
    pid, arm, seed = _screening_ids()[0]
    #: UNSCORED, because only a probe whose weights the scorer will read
    #: travels. A completed and validly scored probe's checkpoint is archival.
    units = [_produce(pod1, pid, rung="screening", arm=arm, seed=seed,
                      scored=False)
             for pid, arm, seed in _screening_ids()[:1]]
    _secure(_Ctx(pod1, store, "attempt1"), units)
    shutil.rmtree(pod1.root)
    transport.pods["fake-host"] = pod2
    _prestage(pod2, store)
    assert L.restore_campaign_probes(_Ctx(pod2, store, "attempt2")) is True

    #: Corrupt the arrival on the replacement pod.
    landed = pod2.local(
        f"{L.VOLUME_MOUNT}/{BG.CAMPAIGN_ID}/probes/{pid}")
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
    #: UNSCORED, because only a probe whose weights the scorer will read
    #: travels. A completed and validly scored probe's checkpoint is archival.
    units = [_produce(pod1, pid, rung="screening", arm=arm, seed=seed,
                      scored=False)
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
    _prestage(ctx.pod, store)
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
    if manifest.get("evidence_root"):
        manifest["evidence_root"] = str(pod.local(manifest["evidence_root"]))
    #: BOTH SECTIONS. `probes` are the ones whose weights a remaining
    #: operation reads; `evidence` are the completed and validly scored ones,
    #: whose rows the verdict reads and whose checkpoints stay archival. A
    #: mapping applied to one and not the other sent the driver at a
    #: pod-absolute evidence path that does not exist on this host, and it
    #: refused a probe whose rows had arrived perfectly.
    for section in ("probes", "evidence"):
        for entry in manifest.get(section) or []:
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
    _prestage(pod2, store)
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
    assert work["n_train_and_score"] == 9 and work["n_score_only"] == 0
    #: SCREENING HAS NOT COMMITTED, so any of the five can still be advanced
    #: and every one of them is materialized. A cost proxy must never decide
    #: which candidate's bytes exist — which is what naming the dearest
    #: admissible candidate as the confirmation arm did.
    assert set(work["arms_needed"]) == set(_candidate_ids()) | {SCH.ANCHOR}
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
    assert ids[2][0] in work["transfer"]["weights"]["probes"]
    landed = pod2.local(
        f"{L.VOLUME_MOUNT}/{BG.CAMPAIGN_ID}/probes/{ids[2][0]}")
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
    assert work["transfer"]["minutes"] == 0.0
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
    minutes = 20.0 / 1.09 * 60.0
    assert campaign["settled_campaign_gpu_usd"] == pytest.approx(20.0, abs=1e-4)
    assert campaign["settled_campaign_spend_usd"] == pytest.approx(
        20.0 + _expected_disk_usd(minutes), abs=1e-4)
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


def test_the_restore_is_a_named_reserve_not_a_throughput_claim():
    """A rate was the right shape for a push; it is wrong for everything since.

    This quantity has had four shapes. `RESTORE_MB_PER_SECOND` times bytes was
    right while the dev box PUSHED to an already-billing pod. A flat transport
    reserve was right once the bytes were to be pulled from an object store,
    because the rate then belonged to somebody else's network. A flat
    verification reserve was right once the bytes sat on an attached volume and
    were only read. And now the class is chosen by WHAT ACTUALLY MOVES, because
    a campaign whose whole remaining working set is 8.1 MiB of per-sample rows
    was being charged the reserve for reading and hashing 22 GiB.

    Through all four the invariant is what this pins: the figure is a RESERVE,
    flat within its class, never a rate. The exact numbers may move when the
    architecture does; a rate creeping back in may not.
    """
    for word in ("NOT a measured", "reserve", "diagnostic"):
        assert word in BC.PROBE_AVAILABILITY_RESERVE_BASIS, word
    assert not hasattr(BC, "RESTORE_MB_PER_SECOND"), (
        "the rate is back; a time-varying quantity must not be a hard bound")
    assert not hasattr(BC, "TRANSPORT_RESERVE_MINUTES")
    assert not hasattr(BC, "restore_minutes"), (
        "the old entry point still resolves, so a caller can get the weights "
        "reserve for a working set that is evidence only")

    weights = BC.PROBE_AVAILABILITY_RESERVE_MINUTES
    evidence = BC.EVIDENCE_RESERVE_MINUTES
    assert 0 < evidence < weights, (
        "an evidence-only working set is three orders of magnitude smaller "
        "than a weights one and must not be reserved the same minutes")

    #: FLAT WITHIN A CLASS: one probe's weights and ten cost the same reserved
    #: minutes, because the reserve bounds the BILL and not the schedule.
    one, ten = int(2.22 * 2**30), int(22.2 * 2**30)
    assert BC.availability_minutes(one, 0) == weights
    assert BC.availability_minutes(ten, 0) == weights
    #: Evidence beside weights does not add a second reserve.
    assert BC.availability_minutes(ten, 8 << 20) == weights

    #: EVIDENCE ONLY gets the evidence reserve, whatever the skipped weights
    #: would have been.
    assert BC.availability_minutes(0, 8 << 20) == evidence
    assert BC.availability_minutes(0, 1) == evidence

    #: And nothing at all is charged nothing: a fresh campaign holds no probe,
    #: and charging it would move the FROZEN full-session ceiling.
    assert BC.availability_minutes(0, 0) == 0.0


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


# ---------------------------------------------------------------------------
# Stage P consumes `arms_needed`, and `run_rung` has three states
# ---------------------------------------------------------------------------

class _StageP(D.C2BehaviouralDriver):
    """The real driver with only the hardware seams replaced.

    `materialize_arm` and `materialize_b` RECORD what they were asked to build
    and write a real checkpoint; the budget admission inside them is production
    code and still runs. Everything that decides WHICH arms are asked for —
    `arm_is_needed`, the frozen-order assertion, the metadata assembly — is the
    production path.
    """

    def __init__(self, args) -> None:
        super().__init__(args)
        self.built: list[str] = []

    def verify_teacher(self) -> None:
        self.teacher_path = "/fake/teacher"

    def release_device(self) -> dict:
        return {"verdict": "no device"}

    def materialize_arm(self, label, spec, *, required, bounded_minutes,
                        config_overrides=None) -> str:
        if not self.afford(bounded_minutes, f"materializing {label}"):
            raise D.C2DriverError(f"budget refuses {label}")
        self.built.append(label)
        d = _write_checkpoint(Path(self.a.b_workdir) / label / "model")
        return str(d)

    def materialize_b(self, binding) -> str:
        if not self.afford(self.a.b_build_minutes, "materializing B"):
            raise D.C2DriverError("budget refuses B")
        self.built.append(SCH.ANCHOR)
        return str(_write_checkpoint(Path(self.a.b_workdir) / "B" / "model"))


def _stage_p_driver(pod: _Pod, run_attempt: str, *, manifest: Path | None):
    args = D.build_parser().parse_args([
        "--campaign", BG.CAMPAIGN_ID, "--run-attempt", run_attempt,
        *(["--continuation-manifest", str(manifest)] if manifest else []),
        "--audit-dir", str(pod.local(L.AUDIT_DIR)),
        "--eval-dir", str(pod.local(L.EVAL_DIR)),
        "--b-workdir", str(pod.local(L.ARM_DIR)),
        "--status-path", str(pod.root / "status.txt"), "--device", "cuda",
        "--b-build-minutes", "31", "--probe-train-minutes", "75",
        "--probe-battery-minutes", "45", "--rate", "1.09",
        "--soft-stop-usd", "300", "--authorized-usd", "320"])
    return _StageP(args)


def test_a_fresh_campaign_stage_p_materializes_all_six(tmp_path, repo):
    """No manifest, nothing verified: every arm is owed and every arm is built."""
    pod = _Pod(tmp_path / "pod1")
    driver = _stage_p_driver(pod, "attempt1", manifest=None)
    driver.restore_campaign()
    assert driver.arms_needed is None
    driver.stage_p()

    assert len(driver.built) == 6, driver.built
    assert SCH.ANCHOR in driver.built
    assert sorted(set(driver.built) - {SCH.ANCHOR}) == sorted(_candidate_ids())
    assert all(c["materialized"] for c in driver.candidates)
    assert driver.ev["stages"]["P"]["arms_needed"] is None

    #: And the budget for the same state agrees it owes six.
    work = BC.remaining_work(REPO, state=BC.campaign_state(tmp_path / "none"))
    assert len(work["arms_needed"]) == 6


def test_a_partial_confirmation_continuation_stage_p_builds_only_two(
        tmp_path, repo, transport):
    """THE budget/work parity that was broken.

    `remaining_work` charged a continuation for `{advanced, B}` while stage P
    rebuilt all six unconditionally, so the pre-provider budget could sit below
    the GPU work the driver would actually do.
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

    work = BC.remaining_work(
        REPO, state=BC.campaign_state(L.campaign_store(BG.CAMPAIGN_ID, store)))
    assert set(work["arms_needed"]) == {advanced, SCH.ANCHOR}

    driver = _stage_p_driver(pod2, "attempt2", manifest=_pod_view(pod2))
    driver.restore_campaign()
    assert driver.arms_needed == {advanced, SCH.ANCHOR}
    driver.stage_p()

    assert sorted(driver.built) == sorted([advanced, SCH.ANCHOR]), driver.built
    assert driver.ev["stages"]["P"]["arms_materialized"] == sorted(
        [advanced, SCH.ANCHOR])
    #: Metadata for all five candidates survives, so the schedule, the ranking
    #: and the frozen tie-break are unchanged by building only two.
    assert [c["state_id"] for c in driver.candidates] == _candidate_ids()
    assert [c["rank_in_frozen_selection"] for c in driver.candidates] == list(
        range(5))
    unbuilt = [c for c in driver.candidates if not c["materialized"]]
    assert len(unbuilt) == 4
    assert all(c["durable_path"] == D.C2BehaviouralDriver.NOT_MATERIALIZED
               for c in unbuilt)
    assert all(c["artifact_digest"] and c["num_parameters"] for c in unbuilt)


def test_training_from_an_unbuilt_arm_is_refused(tmp_path, repo):
    """A budget and a work plan that disagree must stop, not improvise."""
    pod = _Pod(tmp_path / "pod")
    driver = _stage_p_driver(pod, "attempt2", manifest=None)
    probe = SCH.Probe("screening", "some-arm", 1, "a" * 64,
                      D.C2BehaviouralDriver.NOT_MATERIALIZED)
    with pytest.raises(D.C2DriverError, match="did not materialize"):
        driver.require_materialized_arm(probe)
    #: A real path passes.
    driver.require_materialized_arm(
        SCH.Probe("screening", "some-arm", 1, "a" * 64, "/arms/some-arm"))


class _NeverTrains(_StageP):
    """`train_one` is a defect here: a restored probe must never be retrained."""

    def train_one(self, name, config):
        raise AssertionError(
            f"{name} was retrained; R3 forbids retraining a completed probe "
            "for any outcome")

    def attest(self, battery):
        self.evaluation_protocol = object()
        return {"battery": battery.name, "evaluation_protocol_hash": "x"}

    def score_probe(self, probe, model_dir, *, battery, run_completion) -> dict:
        assert Path(model_dir).is_dir(), model_dir
        assert run_completion is None, (
            "the producing pod's run_completion cannot exist here")
        self.scored_from = getattr(self, "scored_from", [])
        self.scored_from.append(str(model_dir))
        return {"probe_id": probe.probe_id, "rung": probe.rung,
                "arm": probe.arm, "seed": probe.seed,
                "correct_overall": 0.5, "usable_rollout_rate": 1.0,
                "result_path": "r", "result_sha256": "s",
                "per_sample_path": "p", "per_sample_sha256": "t"}


def test_a_restored_unscored_probe_is_scored_not_retrained(tmp_path, repo,
                                                           transport):
    """State 2 of the rung, which `run_rung` did not implement.

    Its only test was `name in self.scores`, so a trained, destination-verified
    probe whose scoring had failed fell through to `train_one` and was
    retrained — against R3 and against this module's own contract.
    """
    store = tmp_path / "store"
    ids = _screening_ids()[:1]
    done = [(pid, "screening", arm, seed) for pid, arm, seed in ids]
    pod2, _ = _continue_to(tmp_path, store, transport, complete=done,
                           unscored={ids[0][0]})

    driver = _NeverTrains(_stage_p_driver(pod2, "attempt2",
                                          manifest=_pod_view(pod2)).a)
    driver.restore_campaign()
    driver.load_campaign_journal()
    name, arm, seed = ids[0]
    assert name in driver.training and name not in driver.scores

    probe = SCH.Probe("screening", arm, seed, _init_digest(arm), "/unused")
    #: `train_one` raises if called. The rung must not call it.
    driver.run_rung("screening", [probe], REPO / C2S.BATTERY_PATH)
    assert name in driver.scores
    assert driver.scored_from == [driver.training[name]["model_dir"]]
    assert driver.training[name]["scored_on_restored_checkpoint"] is True
    #: And the record on disk now carries both the descriptor and the score.
    on_disk = json.loads(
        (driver.audit / "probes" / f"{name}.json").read_text())
    assert on_disk["arm"] == arm and on_disk["seed"] == seed
    assert on_disk["score"]["correct_overall"] == 0.5


def test_scoring_a_restored_probe_charges_the_battery_not_the_trainer(
        tmp_path, repo, transport):
    """Charging the trainer again would refuse affordable continuations."""
    store = tmp_path / "store"
    ids = _screening_ids()[:1]
    pod2, _ = _continue_to(
        tmp_path, store, transport,
        complete=[(pid, "screening", arm, seed) for pid, arm, seed in ids],
        unscored={ids[0][0]})
    args = _stage_p_driver(pod2, "attempt2", manifest=_pod_view(pod2)).a
    #: Enough for the battery, NOT enough for training plus the battery.
    args.soft_stop_usd = args.rate * (args.probe_battery_minutes + 5) / 60
    args.authorized_usd = args.soft_stop_usd
    driver = _NeverTrains(args)
    driver.restore_campaign()
    driver.load_campaign_journal()
    name, arm, seed = ids[0]
    driver.run_rung("screening",
                    [SCH.Probe("screening", arm, seed, _init_digest(arm), "/x")],
                    REPO / C2S.BATTERY_PATH)
    assert name in driver.scores


# ---------------------------------------------------------------------------
# the REAL scoring-failure path, in production order
# ---------------------------------------------------------------------------

class _ScoringFails(_StageP):
    """Trains for real, announces for real, and then scoring raises.

    This is the sequence that actually happens on a pod: training succeeds, the
    checkpoint becomes durable off-pod, and scoring dies. Before this round the
    per-probe record was written only AFTER a successful score, so that
    sequence left the destination holding verified bytes and an ack carrying
    only the checkpoint's identity — nothing to say which probe it was.
    """

    def train_one(self, name, config) -> Path:
        out = Path(self.a.eval_dir) / "_train" / name
        _write_checkpoint(out / "checkpoints" / "step_0001023" / "model")
        (out / "checkpoints" / "latest.txt").write_text("step_0001023\n")
        (out / "run_completion.json").write_text(
            json.dumps({"final_step": 1023}) + "\n")
        return out

    def probe_config(self, probe) -> Path:
        cfg = Path(self.a.eval_dir) / f"{probe.probe_id}.json"
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text(json.dumps({"probe": probe.probe_id}) + "\n")
        return cfg

    def attest(self, battery):
        self.evaluation_protocol = object()
        return {"battery": battery.name, "evaluation_protocol_hash": "x"}

    def score_probe(self, probe, model_dir, *, battery, run_completion):
        raise D.C2DriverError(
            f"{probe.probe_id}: the evaluator died after training")


@needs_host_local_stores
def test_a_real_scoring_failure_leaves_a_continuable_probe(tmp_path, repo,
                                                           transport):
    """End to end, in production order, with nothing pre-written.

    The fixture does NOT write a completed probe record up front — doing that
    is what hid this defect, because it is stronger than anything the
    production failure path produces.
    """
    store = tmp_path / "store"
    pod1, pod2 = _Pod(tmp_path / "pod1"), _Pod(tmp_path / "pod2")
    transport.pods["fake-host"] = pod1

    name, arm, seed = _screening_ids()[0]
    driver = _ScoringFails(_stage_p_driver(pod1, "attempt1", manifest=None).a)
    driver.arm_arch = {arm: ("sig", 16)}
    probe = SCH.Probe("screening", arm, seed, f"init-{arm}", "/arms/x")

    with pytest.raises(D.C2DriverError, match="evaluator died"):
        driver.run_rung("screening", [probe], REPO / C2S.BATTERY_PATH)

    #: TRAINED, ANNOUNCED, AND ITS DESCRIPTOR IS ON DISK — written before
    #: scoring was attempted.
    record_path = driver.audit / "probes" / f"{name}.json"
    assert record_path.is_file(), (
        "the training descriptor was not persisted before scoring, so a "
        "scoring failure leaves bytes nothing can identify as this probe")
    on_disk = json.loads(record_path.read_text())
    assert "score" not in on_disk
    for field in ("rung", "arm", "seed", "initialization_artifact_digest",
                  "config_sha256"):
        assert on_disk[field], field
    unit = driver.durable[-1]
    assert unit["kind"] == "probe" and unit["identity"]

    #: The launcher secures the bytes and whatever evidence exists.
    ctx1 = _Ctx(pod1, store, "attempt1")
    fetched = _secure(ctx1, [unit])
    assert all(f["matched"] for f in fetched), fetched

    state = BC.campaign_state(L.campaign_store(BG.CAMPAIGN_ID, store))
    assert name in state["probes"], state["rejected"]
    assert state["probes"][name]["scored"] is False
    assert state["probes"][name]["record"]["arm"] == arm

    #: THE PRODUCER IS GONE.
    shutil.rmtree(pod1.root)
    transport.pods["fake-host"] = pod2
    _prestage(pod2, store)
    ctx2 = _Ctx(pod2, store, "attempt2")
    assert L.restore_campaign_probes(ctx2) is True, ctx2.said

    work = BC.remaining_work(REPO, state=state)
    assert name in work["probes_trained_not_scored"]
    assert arm in work["arms_needed"]

    #: The replacement scores it and never trains it.
    second = _NeverTrains(_stage_p_driver(pod2, "attempt2",
                                          manifest=_pod_view(pod2)).a)
    second.restore_campaign()
    journal = second.load_campaign_journal()
    assert [r["probe_id"] for r in journal["restored"]] == [name]
    assert journal["restored"][0]["scored"] is False
    second.run_rung("screening", [probe], REPO / C2S.BATTERY_PATH)
    assert name in second.scores
    assert second.training[name]["scored_on_restored_checkpoint"] is True


def test_bytes_without_a_descriptor_are_preserved_but_not_consumable(
        tmp_path, repo, transport):
    """The state the old code would have left, refused explicitly.

    Re-identified bytes prove the file is what was announced; they say nothing
    about WHICH probe it is. Without the training record no rung may consume
    them, and saying so is better than a manifest of `null` descriptors that
    the driver refuses later.
    """
    store = tmp_path / "store"
    pod1 = _Pod(tmp_path / "pod1")
    transport.pods["fake-host"] = pod1
    units = [_produce(pod1, pid, rung="screening", arm=arm, seed=seed)
             for pid, arm, seed in _screening_ids()[:1]]
    _secure(_Ctx(pod1, store, "attempt1"), units)

    root = L.campaign_store(BG.CAMPAIGN_ID, store)
    next(root.rglob(BC.RECORD_NAME)).unlink()
    state = BC.campaign_state(root)
    assert state["probes"] == {}
    assert any("no training descriptor" in r["why"]
               for r in state["rejected"]), state["rejected"]
    #: And the campaign therefore owes that probe again, honestly.
    work = BC.remaining_work(REPO, state=state)
    assert work["n_probes_remaining"] == 12


# ---------------------------------------------------------------------------
# cumulative all-in, and UNKNOWN
# ---------------------------------------------------------------------------

def test_prior_disk_actual_enters_cumulative_campaign_spend(tmp_path, repo):
    """A ten-hour predecessor's disk is real money and was being dropped."""
    _run_manifest(repo, "attempt1")
    _session_record(repo, "attempt1", actual_usd=10.90,
                    elapsed_minutes=600.0)
    ctx = _Ctx(_Pod(tmp_path / "pod2"), tmp_path / "store", "attempt2")
    actual = L.prior_attempt_actual(ctx, "attempt1")

    assert actual["gpu_usd"] == pytest.approx(10.90, abs=1e-4)
    assert actual["elapsed_minutes"] == 600.0
    assert actual["disk_usd"] == pytest.approx(_expected_disk_usd(600.0),
                                              abs=1e-4)
    assert actual["all_in_usd"] == pytest.approx(
        10.90 + _expected_disk_usd(600.0), abs=1e-4)
    assert actual["all_in_usd"] > actual["gpu_usd"], (
        "the predecessor's container disk is not in its all-in actual")

    L.campaign_continuation_gate(ctx)
    camp = ctx.evidence["campaign"]
    assert camp["settled_campaign_gpu_usd"] == pytest.approx(10.90, abs=1e-4)
    assert camp["settled_campaign_disk_usd"] == pytest.approx(
        _expected_disk_usd(600.0), abs=1e-4)
    assert camp["settled_campaign_spend_usd"] == pytest.approx(
        actual["all_in_usd"], abs=1e-4)


def test_a_predecessor_whose_cost_cannot_be_established_refuses(tmp_path, repo):
    """UNKNOWN, never $0. Defaulting a paid predecessor to zero is the defect."""
    _run_manifest(repo, "attempt1")
    _session_record(repo, "attempt1", actual_usd=7.02, omit_elapsed=True)
    ctx = _Ctx(_Pod(tmp_path / "pod2"), tmp_path / "store", "attempt2")

    actual = L.prior_attempt_actual(ctx, "attempt1")
    assert "unknown" in actual
    assert "gpu_usd" not in actual

    ok, why = L.campaign_continuation_gate(ctx)
    assert not ok
    assert "UNKNOWN" in why
    assert "not zero" in why


def test_a_zero_dollar_pre_provider_refusal_owes_nothing(tmp_path, repo):
    """No resource existed, so neither GPU nor disk was billed."""
    _run_manifest(repo, "attempt1")
    _session_record(repo, "attempt1", created=False, confirmed_gone=False,
                    actual_usd=0.0, omit_elapsed=True)
    ctx = _Ctx(_Pod(tmp_path / "pod2"), tmp_path / "store", "attempt2",
               all_in=ALL_IN + 1.0)
    actual = L.prior_attempt_actual(ctx, "attempt1")
    assert actual["all_in_usd"] == 0.0
    assert "unknown" not in actual
    ok, why = L.campaign_continuation_gate(ctx)
    assert ok, why


def test_the_disk_rate_comes_from_the_authorizations_own_figures(tmp_path,
                                                                 repo):
    """One basis for the ceiling and for the predecessor, not two."""
    _run_manifest(repo, "attempt1")
    _session_record(repo, "attempt1", actual_usd=1.09, elapsed_minutes=60.0)
    ctx = _Ctx(_Pod(tmp_path / "pod2"), tmp_path / "store", "attempt2")
    actual = L.prior_attempt_actual(ctx, "attempt1")
    #: An hour of the provisioned volume, derived from disk_hard_usd over the
    #: authorized runtime — the same quotient the ceiling was built from.
    assert actual["disk_usd_per_minute"] == pytest.approx(
        DISK_HARD / RUNTIME, abs=1e-8)
    assert actual["disk_usd"] == _expected_disk_usd(60.0)
    #: A ceiling, so never under the raw arithmetic and never by more than the
    #: quantum.
    raw = DISK_HARD / RUNTIME * 60.0
    assert raw <= actual["disk_usd"] <= raw + BG.DOLLAR_QUANTUM_USD


# ---------------------------------------------------------------------------
# R10: a cost proxy must never decide which candidate's bytes exist
# ---------------------------------------------------------------------------

class _FullFlow(_StageP):
    """Drives the real P -> S -> R -> C, with only hardware seams replaced.

    `score_probe` returns a controlled `correct_overall` per arm so the test
    decides which candidate the MECHANICAL ranking advances. Everything that
    chooses arms, refuses unbuilt ones, ranks, breaks ties and schedules
    confirmation is production code.
    """

    #: arm -> correct_overall. Anything unlisted scores low.
    wants: dict = {}

    def train_one(self, name, config) -> Path:
        out = Path(self.a.eval_dir) / "_train" / name
        _write_checkpoint(out / "checkpoints" / "step_0001023" / "model")
        (out / "checkpoints" / "latest.txt").write_text("step_0001023\n")
        (out / "run_completion.json").write_text(
            json.dumps({"final_step": 1023}) + "\n")
        return out

    def probe_config(self, probe) -> Path:
        cfg = Path(self.a.eval_dir) / f"{probe.probe_id}.cfg.json"
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text(json.dumps({"probe": probe.probe_id}) + "\n")
        return cfg

    def attest(self, battery):
        self.evaluation_protocol = object()
        return {"battery": battery.name, "evaluation_protocol_hash": "x"}

    def score_probe(self, probe, model_dir, *, battery, run_completion) -> dict:
        rows = [{"id": f"p{i:04d}", "set": "gsm8k", "scorable": True,
                 "correct": True, "usable": True} for i in range(10)]
        per_sample = self.audit / f"{probe.probe_id}_per_sample.jsonl"
        per_sample.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return {"probe_id": probe.probe_id, "rung": probe.rung,
                "arm": probe.arm, "seed": probe.seed,
                "correct_overall": self.wants.get(probe.arm, 0.10),
                "usable_rollout_rate": 1.0,
                "result_path": "r", "result_sha256": "s",
                "per_sample_path": str(per_sample), "per_sample_sha256": "t"}


@needs_host_local_stores
def test_the_screening_winner_may_be_any_candidate_not_the_cost_proxy(
        tmp_path, repo, transport):
    """R10 edge case 1, driven through P -> S -> R -> C for real.

    The dearest candidate is a COST bound for an unranked confirmation rung. It
    used to be handed to Stage P as an arm identity, so if the mechanical
    ranking advanced any other leaf — which screening scores decide, and they
    have nothing to do with build cost — stage C met `NOT_MATERIALIZED` and the
    campaign failed for a perfectly legitimate winner. The old partial-screening
    test never ran past the rank, so it could not see this.
    """
    store = tmp_path / "store"
    per_arm = BC.arm_minutes(REPO)
    candidates = _candidate_ids()
    dearest = max(candidates, key=lambda a: per_arm[a])

    #: A partial screening continuation: three probes restored, three owed.
    done = [(pid, "screening", arm, seed)
            for pid, arm, seed in _screening_ids()[:3]]
    restored_arms = {arm for _, arm, _ in _screening_ids()[:3]}
    #: The CHEAPEST candidate whose screening probe this session still runs —
    #: so the controlled score decides the ranking, and the winner is the last
    #: arm any cost proxy would have named.
    winner = min((c for c in candidates if c not in restored_arms),
                 key=lambda a: per_arm[a])
    assert winner != dearest
    pod2, _ = _continue_to(tmp_path, store, transport, complete=done)

    work = BC.remaining_work(
        REPO, state=BC.campaign_state(L.campaign_store(BG.CAMPAIGN_ID, store)))
    assert work["screening_committed"] is False
    assert winner in work["arms_needed"], (
        "the winner's arm is not in the set the budget funded, so a cost "
        "proxy is still deciding which candidate's bytes exist")
    assert dearest in work["arms_needed"]

    driver = _FullFlow(_stage_p_driver(pod2, "attempt2",
                                       manifest=_pod_view(pod2)).a)
    driver.wants = {winner: 0.99}
    driver.restore_campaign()
    driver.load_campaign_journal()
    driver.stage_p()

    #: Every candidate that can still win was built, the proxy among them.
    assert winner in driver.built and dearest in driver.built
    assert all(c["materialized"] for c in driver.candidates)

    driver.stage_s()
    driver.stage_r()
    assert driver.advanced["state_id"] == winner, (
        "this test only means something if the ranking advances a leaf that "
        "is not the cost proxy")
    assert driver.advanced["state_id"] != dearest

    #: THE POINT: confirmation runs for the winner, from real bytes.
    driver.stage_c()
    assert len(driver.confirmation) == 6
    assert {p.arm for p in driver.confirmation} == {winner, SCH.ANCHOR}
    for probe in driver.confirmation:
        assert probe.initialization_path != D.C2BehaviouralDriver.NOT_MATERIALIZED
        assert probe.probe_id in driver.scores


def test_only_a_trained_unscored_probe_remains(tmp_path, repo, transport):
    """R10 edge case 2: no arm is owed and nothing is trained.

    Its checkpoint exists and was verified; rebuilding the initialization it
    was trained from would be building bytes nothing reads, and pricing it as a
    full probe charges a trainer that will not run.
    """
    store = tmp_path / "store"
    advanced = _candidate_ids()[0]
    seeds = BH.protocol(REPO)["behavioural_selection"]["seeds"]["confirmation"]
    done = [(pid, "screening", arm, seed)
            for pid, arm, seed in _screening_ids()]
    last = None
    for seed in seeds:
        for arm in (advanced, SCH.ANCHOR):
            pid = f"confirmation.{arm}.s{seed}"
            done.append((pid, "confirmation", arm, int(seed)))
            last = pid
    pod2, _ = _continue_to(tmp_path, store, transport, complete=done,
                           ranking=advanced, unscored={last})

    work = BC.remaining_work(
        REPO, state=BC.campaign_state(L.campaign_store(BG.CAMPAIGN_ID, store)))
    assert work["probes_trained_not_scored"] == [last]
    assert work["probes_untrained"] == []
    assert work["n_train_and_score"] == 0
    assert work["n_score_only"] == 1
    assert work["arms_needed"] == [], (
        "a trained-but-unscored probe already holds its checkpoint; its "
        "initialization must not be rebuilt")
    assert work["materialization_minutes"] == 0.0

    d = work["decomposition"]
    assert d["train_and_score_probes"] == 0 and d["score_only_probes"] == 1
    assert not any(n.endswith("_probes_train_and_score")
                   for n, _ in d["expected_phases"])
    assert any(n == "1_probes_score_only" for n, _ in d["expected_phases"])
    #: The minutes come from the RECORD's own per-probe eval mean — not from a
    #: second call to the function under test, which would move with it and
    #: could never fail.
    cost = json.loads((REPO / BH.PRICING).read_text())[
        "behavioural_selection"]["probe_cost"]
    eval_mean = float(cost["eval_minutes"]["mean"])
    train_mean = float(cost["train_minutes"]["mean"])
    assert dict(d["expected_phases"])["1_probes_score_only"] == pytest.approx(
        eval_mean, abs=0.01), (
        "a score-only probe is being charged the trainer")
    assert d["per_probe_minutes"]["train_mean"] == train_mean
    #: And materially cheaper than a full probe, by about the training mean.
    as_full = BH.session_decomposition(
        REPO, materialization_minutes=0.0, train_and_score_probes=1,
        score_only_probes=0, restore_minutes=d["restore_minutes"])
    assert d["hard_minutes"] < as_full["hard_minutes"]
    assert (as_full["expected_minutes"] - d["expected_minutes"]
            ) == pytest.approx(train_mean, abs=0.02)

    #: And the driver really does only score it.
    driver = _NeverTrains(_stage_p_driver(pod2, "attempt2",
                                          manifest=_pod_view(pod2)).a)
    driver.restore_campaign()
    assert driver.arms_needed == set(), (
        "an empty arms_needed must mean 'none owed', not 'no manifest'")
    driver.stage_p()
    assert driver.built == []
    driver.load_campaign_journal()
    arm, seed = advanced if last.startswith("confirmation." + advanced) else SCH.ANCHOR, int(seeds[-1])
    probe = SCH.Probe("confirmation", last.split(".")[1], seed,
                      _init_digest(last.split(".")[1]), "/unused")
    driver.run_rung("confirmation", [probe], REPO / C2S.BATTERY_PATH)
    assert last in driver.scores


def test_a_mixed_campaign_state_buckets_every_probe_correctly(tmp_path, repo,
                                                              transport):
    """R10, all three buckets at once, and Stage P prepares only the untrained.

    Complete, trained-but-unscored and untrained probes must land in different
    cost buckets, and only the untrained ones may cause an initialization to be
    built.
    """
    store = tmp_path / "store"
    ids = _screening_ids()
    #: Four restored: three scored, one trained-only. Two never started.
    done = [(pid, "screening", arm, seed) for pid, arm, seed in ids[:4]]
    unscored_id = ids[3][0]
    pod2, _ = _continue_to(tmp_path, store, transport, complete=done,
                           unscored={unscored_id})

    work = BC.remaining_work(
        REPO, state=BC.campaign_state(L.campaign_store(BG.CAMPAIGN_ID, store)))
    assert len(work["probes_complete"]) == 3
    assert work["probes_trained_not_scored"] == [unscored_id]
    #: Two screening probes never started, plus all six confirmation probes.
    assert len(work["probes_untrained"]) == 8
    assert work["n_train_and_score"] == 8
    assert work["n_score_only"] == 1
    assert work["n_probes_remaining"] == 9

    #: The unscored probe's arm is NOT owed on its account. Screening is
    #: uncommitted, so every candidate is owed for the confirmation rung
    #: anyway — what must hold is that the anchor and candidates come from the
    #: untrained probes, never from the restored one.
    from_untrained = {ids_arm for pid, ids_arm, _ in ids
                      if pid in work["probes_untrained"]}
    assert from_untrained <= set(work["arms_needed"])

    driver = _NeverTrains(_stage_p_driver(pod2, "attempt2",
                                          manifest=_pod_view(pod2)).a)
    driver.restore_campaign()
    assert driver.arms_needed == set(work["arms_needed"])
    driver.stage_p()
    assert sorted(driver.built) == sorted(work["arms_needed"])

    #: Pricing: eight trainings and one scoring, not nine trainings — asserted
    #: in absolute minutes from the record's own per-probe figures.
    d = work["decomposition"]
    cost = json.loads((REPO / BH.PRICING).read_text())[
        "behavioural_selection"]["probe_cost"]
    train_mean = float(cost["train_minutes"]["mean"])
    eval_mean = float(cost["eval_minutes"]["mean"])
    phases = dict(d["expected_phases"])
    assert phases["8_probes_train_and_score"] == pytest.approx(
        8 * (train_mean + eval_mean), abs=0.01)
    assert phases["1_probes_score_only"] == pytest.approx(eval_mean, abs=0.01)
    nine_full = BH.session_decomposition(
        REPO, materialization_minutes=work["materialization_minutes"],
        train_and_score_probes=9, score_only_probes=0,
        restore_minutes=d["restore_minutes"])
    assert d["hard_minutes"] < nine_full["hard_minutes"]
    assert (nine_full["expected_minutes"] - d["expected_minutes"]
            ) == pytest.approx(train_mean, abs=0.02)


# ---------------------------------------------------------------------------
# the readiness gate: a gate that cannot pass, and a gate that passes a FAIL
# ---------------------------------------------------------------------------

def _readiness(repo_root: Path, run_id: str, **over) -> Path:
    """A readiness record in the shape the recorder actually writes."""
    from experiments.phase_c2 import behavioural_pod_environment as BPE

    path = repo_root / BPE.record_path_for(run_id, L.STAGE_ID)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"record_kind": "launch_bound", "verdict": "PASS",
              "counts": {"passed": 152, "failed": 0, "skipped": 0, "error": 0},
              "problems": []}
    record.update(over)
    path.write_text(json.dumps(record))
    return path


@pytest.mark.parametrize("over,why", [
    ({}, None),
    ({"record_kind": "diagnostic"}, "not 'launch_bound'"),
    ({"verdict": "FAIL", "problems": ["3 failed"]}, "does not rest on a sweep"),
    ({"verdict": None}, "does not rest on a sweep"),
])
def test_the_readiness_gate_reads_record_kind_and_the_verdict(
        tmp_path, monkeypatch, over, why):
    """The gate read `kind`, which NO record carries, so it could not pass.

    Every readiness record this repository writes names the field
    `record_kind`; `kind` is absent, so the comparison was
    `None != "launch_bound"` on a perfectly good record. The replay launcher's
    docstring names that shape as the defect the full-search launcher was
    repaired for. It would have refused this launch at gate seven of eight.

    And it never asked the verdict, which is the more dangerous half: the first
    behavioural sweep FAILED, and a gate that only asks "is this launch-bound"
    lets a failing tree through.
    """
    from experiments.phase_c2 import behavioural_pod_environment as BPE

    root = _mirror_repo(tmp_path / "repo")
    monkeypatch.setattr(L, "REPO_ROOT", root)
    _readiness(root, "attempt1", **over)
    #: `verify_record` binds the record to a tree and is not what is under test
    #: here; it has its own coverage in the pod-environment contract.
    monkeypatch.setattr(BPE, "verify_record",
                        lambda *a, **k: {"ok": True})

    ctx = _Ctx(_Pod(tmp_path / "pod"), tmp_path / "store", "attempt1")
    ok, message = L.readiness_gate(ctx)
    if why is None:
        assert ok, message
        assert "launch_bound" in message and "PASS" in message
    else:
        assert not ok
        assert why in message


def test_a_missing_readiness_record_refuses(tmp_path, monkeypatch):
    root = _mirror_repo(tmp_path / "repo")
    monkeypatch.setattr(L, "REPO_ROOT", root)
    ok, message = L.readiness_gate(
        _Ctx(_Pod(tmp_path / "pod"), tmp_path / "store", "attempt1"))
    assert not ok
    assert "no behavioural readiness record" in message


def _closeout(repo_root: Path, run_id: str, **over) -> Path:
    """A retired chain's closeout, in the shape the launcher writes."""
    path = _refuse_to_write_into_the_repository(
        repo_root / L.rel_run_dir(L.EXPERIMENT_ID, run_id, L.STAGE_ID)
        / "closeout" / "outcome.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"run_id": run_id,
              "terminal": "CHAIN_RETIRED_BEFORE_PROVIDER_CONTACT",
              "provider_resource_created": False, "spend_usd": 0.0}
    record.update(over)
    path.write_text(json.dumps(record))
    return path


def test_a_retired_chain_is_zero_not_unknown(tmp_path, repo):
    """A chain retired before the launcher ran created nothing, and says so.

    No session record has TWO causes that must not be conflated: a launcher
    that died mid-flight, which may have created a resource, and a chain
    retired before the launcher ran at all. Only an affirmative statement tells
    them apart. A retirement is not rare — any repair inside the executable
    closure forces one — so without this the campaign's first retirement blocks
    every later attempt forever.
    """
    _run_manifest(repo, "attempt1")
    _closeout(repo, "attempt1")
    ctx = _Ctx(_Pod(tmp_path / "pod"), tmp_path / "store", "attempt2")

    actual = L.prior_attempt_actual(ctx, "attempt1")
    assert "unknown" not in actual
    assert actual["all_in_usd"] == 0.0 and actual["basis"] == "closeout"

    ok, why = L.campaign_continuation_gate(ctx)
    assert ok, why
    assert ctx.evidence["campaign"]["settled_campaign_spend_usd"] == 0.0


@pytest.mark.parametrize("over,label", [
    ({"provider_resource_created": True}, "a closeout that created a resource"),
    ({}, "no closeout at all"),
])
def test_a_run_without_an_affirmative_closeout_stays_unknown(tmp_path, repo,
                                                             over, label):
    """Absence is never zero, and a closeout that DID create refuses too."""
    _run_manifest(repo, "attempt1")
    if over:
        _closeout(repo, "attempt1", **over)
    ctx = _Ctx(_Pod(tmp_path / "pod"), tmp_path / "store", "attempt2")

    assert "unknown" in L.prior_attempt_actual(ctx, "attempt1"), label
    ok, why = L.campaign_continuation_gate(ctx)
    assert not ok and "UNKNOWN" in why, label


def test_an_unreadable_closeout_stays_unknown(tmp_path, repo):
    _run_manifest(repo, "attempt1")
    path = (repo / L.rel_run_dir(L.EXPERIMENT_ID, "attempt1", L.STAGE_ID)
            / "closeout" / "outcome.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not json")
    ctx = _Ctx(_Pod(tmp_path / "pod"), tmp_path / "store", "attempt2")
    actual = L.prior_attempt_actual(ctx, "attempt1")
    assert "unreadable closeout" in actual["unknown"]


# ---------------------------------------------------------------------------
# stage P must release what it writes, or the storage bound is fiction
# ---------------------------------------------------------------------------

class _Step:
    def __init__(self, impl_id: str, path: Path) -> None:
        self.impl_id, self.checkpoint_path = impl_id, str(path)


def _path_on_disk(workdir: Path, n: int) -> list:
    """An n-step arm build, every step written, as fixed_path leaves it."""
    steps = []
    for i in range(n):
        d = workdir / f"step_{i}"
        _write_checkpoint(d)
        steps.append(_Step(f"impl.{i}", d))
    return steps


def test_stage_p_releases_intermediates_and_keeps_the_final(tmp_path):
    """attempt3 died here, at $2.50, with five arms already rebuilt exactly.

    `materialize_fixed_path` writes every step of a four-step path and returns
    them all, and nothing deleted them — so all six arms' full paths stayed
    resident for the whole of stage P. The storage derivation charged that
    transient ONCE, in as many words: "resident while that arm builds and
    released when it is verified". Nothing released them, so the bound
    described a program that did not exist and B failed at `Writing model
    shards` with ENOSPC after 32.8 minutes of completed compute.
    """
    driver = D.C2BehaviouralDriver.__new__(D.C2BehaviouralDriver)
    driver.ev = {}
    workdir = tmp_path / "arms" / "leafA"
    steps = _path_on_disk(workdir, 4)

    out = driver.release_intermediates("leafA", steps, workdir)

    assert out["failed"] == []
    assert len(out["removed_steps"]) == 3
    assert out["freed_gib"] >= 0
    for step in steps[:-1]:
        assert not Path(step.checkpoint_path).exists(), step.impl_id
    #: The final survives, with its bytes.
    final = Path(steps[-1].checkpoint_path)
    assert (final / "model.safetensors").is_file()
    assert driver.ev["intermediates_released"][0]["arm"] == "leafA"


def test_release_never_touches_the_final_or_anything_outside_the_workdir(
        tmp_path):
    """Two guards, because this deletes."""
    driver = D.C2BehaviouralDriver.__new__(D.C2BehaviouralDriver)
    driver.ev = {}
    workdir = tmp_path / "arms" / "leafB"
    steps = _path_on_disk(workdir, 2)

    #: A step that points OUTSIDE the arm's workdir — the shape a path-handling
    #: slip would produce — must be left alone rather than deleted.
    outside = tmp_path / "somewhere_else"
    _write_checkpoint(outside)
    steps.insert(0, _Step("impl.outside", outside))
    #: And a step whose path IS the final must never be removed.
    steps.insert(1, _Step("impl.dup", Path(steps[-1].checkpoint_path)))

    driver.release_intermediates("leafB", steps, workdir)

    assert (outside / "model.safetensors").is_file(), (
        "release deleted a path outside the arm's workdir")
    assert (Path(steps[-1].checkpoint_path) / "model.safetensors").is_file(), (
        "release deleted the final checkpoint")


def test_release_never_raises(tmp_path, monkeypatch):
    """A cleanup failure must not destroy a verified, announced arm."""
    import shutil

    driver = D.C2BehaviouralDriver.__new__(D.C2BehaviouralDriver)
    driver.ev = {}
    workdir = tmp_path / "arms" / "leafC"
    steps = _path_on_disk(workdir, 3)
    monkeypatch.setattr(
        shutil, "rmtree",
        lambda *a, **k: (_ for _ in ()).throw(OSError("device busy")))

    out = driver.release_intermediates("leafC", steps, workdir)
    assert out["removed_steps"] == []
    assert len(out["failed"]) == 2
    #: Everything still there — nothing half-deleted, nothing raised.
    for step in steps:
        assert Path(step.checkpoint_path).exists()


def test_a_single_step_path_releases_nothing(tmp_path):
    """The final is the only step; there is nothing to free."""
    driver = D.C2BehaviouralDriver.__new__(D.C2BehaviouralDriver)
    driver.ev = {}
    workdir = tmp_path / "arms" / "leafD"
    steps = _path_on_disk(workdir, 1)
    out = driver.release_intermediates("leafD", steps, workdir)
    assert out["removed_steps"] == [] and out["freed_gib"] == 0.0
    assert (Path(steps[0].checkpoint_path) / "model.safetensors").is_file()


def test_materialize_arm_calls_the_release(tmp_path, monkeypatch):
    """The wiring, not just the mechanism.

    Found by mutation: deleting the `release_intermediates(...)` call from
    `materialize_arm` left every test above green, because they call the
    release directly. Tests prove a mechanism works; only this proves stage P
    uses it — and stage P not using it is precisely what cost attempt3.

    Drives the REAL `materialize_arm` with only the builder replaced.
    """
    from aadistill.initialization.planning import fixed_path as FP

    workdir = tmp_path / "arms" / "leafE"
    steps = _path_on_disk(workdir, 4)

    class _Identity:
        arch_signature, num_parameters = "sig", 16
    for s in steps:
        s.identity = _Identity()

    monkeypatch.setattr(FP, "materialize_fixed_path",
                        lambda *a, **k: steps)

    binding = json.loads(
        (REPO / "logs/stages/stage-1/phase_c1/plans/teacher_binding.json"
         ).read_text())
    spec = type("S", (), {"root_repo_id": binding["repo_id"],
                          "root_revision": binding["revision"]})()

    class _Arm(D.C2BehaviouralDriver):
        def reuse_arm(self, label, required):
            return None

        def announce_durable(self, *a, **k):
            return {"identity": None}

        def afford(self, minutes, what):
            return True

    driver = _Arm.__new__(_Arm)
    driver.a = type("A", (), {"b_workdir": str(tmp_path / "arms"),
                              "device": "cpu", "campaign": BG.CAMPAIGN_ID})()
    driver.ev, driver.durable = {}, []
    driver.teacher_path = "/fake/teacher"

    out = driver.materialize_arm(
        "leafE", spec,
        required={"arch_signature": "sig", "num_parameters": 16},
        bounded_minutes=1.0)

    assert out == steps[-1].checkpoint_path
    assert driver.ev.get("intermediates_released"), (
        "materialize_arm did not release its intermediates; stage P would "
        "retain every step of all six arms, which is what filled the disk")
    for step in steps[:-1]:
        assert not Path(step.checkpoint_path).exists(), step.impl_id
    assert (Path(steps[-1].checkpoint_path) / "model.safetensors").is_file()


def test_a_failed_release_stops_stage_p_before_the_next_arm(tmp_path,
                                                            monkeypatch):
    """The 120 GB bound holds only if intermediates are actually released.

    `release_intermediates` stays non-raising — a cleanup error must not
    destroy a verified, announced arm — so the fail-closed decision belongs to
    the CALLER. If a release failed, the lifecycle assumption the storage bound
    rests on has been falsified, and building the next arm under a bound that
    no longer describes the program is how attempt3 reached ENOSPC five arms
    later with no verdict.

    Proves all three: the verified arm survives, the failure is recorded, and
    no next arm begins.
    """
    import shutil

    from aadistill.initialization.planning import fixed_path as FP

    built: list[str] = []

    class _Identity:
        arch_signature, num_parameters = "sig", 16

    def _steps_for(label):
        steps = _path_on_disk(tmp_path / "arms" / label, 3)
        for s in steps:
            s.identity = _Identity()
        return steps

    made = {}

    def _fake(*a, **k):
        label = Path(k["workdir"]).name
        built.append(label)
        made[label] = _steps_for(label)
        return made[label]

    monkeypatch.setattr(FP, "materialize_fixed_path", _fake)
    monkeypatch.setattr(
        shutil, "rmtree",
        lambda *a, **k: (_ for _ in ()).throw(OSError("device busy")))

    binding = json.loads(
        (REPO / "logs/stages/stage-1/phase_c1/plans/teacher_binding.json"
         ).read_text())
    spec = type("S", (), {"root_repo_id": binding["repo_id"],
                          "root_revision": binding["revision"]})()

    class _Arm(D.C2BehaviouralDriver):
        def reuse_arm(self, label, required):
            return None

        def announce_durable(self, *a, **k):
            self.announced.append(a[0])
            return {"identity": None}

        def afford(self, minutes, what):
            return True

    driver = _Arm.__new__(_Arm)
    driver.a = type("A", (), {"b_workdir": str(tmp_path / "arms"),
                              "device": "cpu", "campaign": BG.CAMPAIGN_ID})()
    driver.ev, driver.durable, driver.announced = {}, [], []
    driver.teacher_path = "/fake/teacher"
    required = {"arch_signature": "sig", "num_parameters": 16}

    with pytest.raises(D.C2DriverError, match="NO FURTHER ARM MAY BE BUILT"):
        driver.materialize_arm("armOne", spec, required=required,
                               bounded_minutes=1.0)

    #: 1. the verified arm survives, bytes intact
    final = Path(made["armOne"][-1].checkpoint_path)
    assert (final / "model.safetensors").is_file()
    #: 2. it was announced BEFORE the release was attempted, so it is durable
    assert driver.announced == ["armOne"]
    #: 3. the failure is recorded rather than swallowed
    rec = driver.ev["intermediates_released"][0]
    assert rec["arm"] == "armOne" and len(rec["failed"]) == 2
    #: 4. NO NEXT ARM BEGAN — the raise propagates out of materialize_arm, so
    #:    stage P cannot proceed to the next one.
    assert built == ["armOne"], built


def test_a_clean_release_lets_the_next_arm_begin(tmp_path, monkeypatch):
    """The permitted case, so the refusal above is known to be selective."""
    from aadistill.initialization.planning import fixed_path as FP

    class _Identity:
        arch_signature, num_parameters = "sig", 16

    built: list[str] = []

    def _fake(*a, **k):
        label = Path(k["workdir"]).name
        built.append(label)
        steps = _path_on_disk(tmp_path / "arms" / label, 3)
        for s in steps:
            s.identity = _Identity()
        return steps

    monkeypatch.setattr(FP, "materialize_fixed_path", _fake)
    binding = json.loads(
        (REPO / "logs/stages/stage-1/phase_c1/plans/teacher_binding.json"
         ).read_text())
    spec = type("S", (), {"root_repo_id": binding["repo_id"],
                          "root_revision": binding["revision"]})()

    class _Arm(D.C2BehaviouralDriver):
        def reuse_arm(self, label, required):
            return None

        def announce_durable(self, *a, **k):
            return {"identity": None}

        def afford(self, minutes, what):
            return True

    driver = _Arm.__new__(_Arm)
    driver.a = type("A", (), {"b_workdir": str(tmp_path / "arms"),
                              "device": "cpu", "campaign": BG.CAMPAIGN_ID})()
    driver.ev, driver.durable = {}, []
    driver.teacher_path = "/fake/teacher"
    required = {"arch_signature": "sig", "num_parameters": 16}

    for label in ("armOne", "armTwo"):
        driver.materialize_arm(label, spec, required=required,
                               bounded_minutes=1.0)
    assert built == ["armOne", "armTwo"]
    assert all(not r["failed"] for r in driver.ev["intermediates_released"])


# ---------------------------------------------------------------------------
# Two ceilings: the campaign's funds another attempt and buys nothing else
# ---------------------------------------------------------------------------

def test_the_continuation_gate_reads_the_campaign_ceiling_not_the_session_one(
        tmp_path, repo):
    """Drive them apart, and check which number the gate actually compares to.

    They were the same figure until attempt3 spent $2.5425 without reaching a
    verdict. While they agreed, a gate reading either one passed every test —
    so the only way to know which it reads is to make them disagree.
    """
    _run_manifest(repo, "attempt1")
    _session_record(repo, "attempt1", actual_usd=2.5042, elapsed_minutes=137.8)

    remaining = None
    for campaign, expect in ((ALL_IN, False), (ALL_IN + 3.0, True)):
        ctx = _Ctx(_Pod(tmp_path / f"pod-{campaign}"), tmp_path / "store",
                   "attempt2", all_in=ALL_IN, campaign=campaign)
        ok, why = L.campaign_continuation_gate(ctx)
        assert ok is expect, (campaign, why)
        ev = ctx.evidence["campaign"]
        #: The session ceiling is recorded either way, so a reader can see both
        #: numbers and tell which one bound the decision.
        assert ev["session_all_in_hard_usd"] == pytest.approx(ALL_IN)
        assert ev["campaign_approved_all_in_usd"] == pytest.approx(campaign)
        #: The remaining planned work does NOT move with the campaign ceiling.
        if remaining is None:
            remaining = ev["this_session_planned_all_in_usd"]
        assert ev["this_session_planned_all_in_usd"] == pytest.approx(remaining)

    #: And the refusal is the honest one: a full session's remaining work no
    #: longer fits beside a spent predecessor under the OLD ceiling.
    assert remaining > 0


def test_raising_the_campaign_ceiling_buys_no_runtime_disk_probes_or_seeds():
    """The maintainer's constraint, asserted against the production derivation.

    A larger cumulative ceiling funds another attempt. It must not extend the
    session window, raise the GPU dollars, enlarge the disk, add probes, add
    seeds, or alter scientific scope — all of which come from the frozen
    record through the SESSION ceiling.
    """
    small = BG.authorization_terms(REPO, rate_usd_per_hour=1.09,
                                   campaign_all_in_hard_usd=33.2099)
    large = BG.authorization_terms(REPO, rate_usd_per_hour=1.09,
                                   campaign_all_in_hard_usd=99.0)

    #: Every session amount is identical, field by field.
    for field in BG.AUTHORIZATION_AMOUNT_FIELDS:
        assert small[field] == large[field], field
    assert small["provisioned_disk_gb"] == large["provisioned_disk_gb"]
    assert small["expected_all_in_usd"] == large["expected_all_in_usd"]
    #: Only the campaign field moved.
    assert small[BG.CAMPAIGN_AMOUNT_FIELD] != large[BG.CAMPAIGN_AMOUNT_FIELD]

    #: The window the launcher runs on is unmoved.
    assert BG.window_minutes(
        1.09, gpu_hard_usd=small["gpu_hard_usd"],
        hard_runtime_minutes=small["hard_runtime_minutes"]) == \
        BG.window_minutes(
            1.09, gpu_hard_usd=large["gpu_hard_usd"],
            hard_runtime_minutes=large["hard_runtime_minutes"])

    #: And the science: the schedule is a function of the frozen record, and
    #: `window_minutes` takes no campaign figure at all, so there is no path
    #: from the campaign ceiling into either.
    assert "campaign" not in inspect.signature(BG.window_minutes).parameters
    sched = BH.session_decomposition(REPO, materialization_minutes=100.0)
    assert sched["train_and_score_probes"] == 12


def test_a_campaign_ceiling_below_the_session_ceiling_is_refused():
    """Not a funding decision — an incoherent one: the session cannot run."""
    with pytest.raises(BG.BehaviouralGovernanceError, match="campaign"):
        BG.authorization_terms(REPO, rate_usd_per_hour=1.09,
                               campaign_all_in_hard_usd=10.0)


def test_an_authorization_missing_the_campaign_ceiling_will_not_load(tmp_path):
    """Fail closed: a document with one ceiling cannot say which it means."""
    doc = json.loads(
        (REPO / "logs/stages/stage-1/phase_c2_behavioural/runs/attempt3"
         / "governance/authorization.json").read_text())
    doc.pop(BG.CAMPAIGN_AMOUNT_FIELD, None)
    p = tmp_path / "authorization.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(Exception, match="campaign"):
        BG.BehaviouralAuthorization.load(p)


# ---------------------------------------------------------------------------
# A dry run must not consume the chain it exists to de-risk
# ---------------------------------------------------------------------------

def test_a_dry_run_writes_to_its_own_run_directory(tmp_path):
    """attempt4 was consumed at `$0` by the flag meant to protect it.

    Every launcher invocation records its run — deliberately, so a launcher
    that died mid-flight cannot be silently re-invoked against one
    authorization — and `open_run` then refuses a recorded or occupied
    directory. `--dry-run` inherited both, so the invocation that advertises
    "run every $0 gate and stop before provider creation" consumed attempt4's
    one-use chain, and the real launch seconds later was refused by the
    occupancy rule.

    The GATE inputs must stay on the real run id: a dry run resolving its
    grant, readiness record, authorization, bundle and prior campaign attempts
    from a different id would check a different chain and prove nothing about
    the one about to launch. Only the outputs move.
    """
    src = (REPO / "scripts/pod/autoinit_c2_behavioural_launch.py").read_text()
    body = src.split("def main(", 1)[1]

    #: The output locations are keyed on the dry-run id...
    for call in ("claim_output_root(args.scr, EXPERIMENT_ID, layout_run_id",
                 "layout = open_run(REPO_ROOT, EXPERIMENT_ID, layout_run_id",
                 "args.out = session_record_path(layout_run_id)"):
        assert call in body, call
    #: ...and it differs from the real one exactly when --dry-run is set.
    assert 'layout_run_id = (f"{args.run_id}{DRY_RUN_SUFFIX}" if args.dry_run'\
           in body
    #: ...while `args.run_id` is NOT reassigned, so every gate still binds the
    #: real chain. A reassignment is the one edit that would make this pass
    #: while checking the wrong authorization.
    assert "args.run_id =" not in body, (
        "args.run_id is reassigned in main(); the gates would resolve a "
        "different chain's governance artifacts")


def test_the_dry_run_id_is_a_VALID_run_id(tmp_path):
    """THE defect string comparison could not see.

    Both checks above read the implementation's own text, so they confirmed
    that the id is derived and distinct -- and said nothing about whether
    `run_layout` will accept it. `attempt6-dryrun` contains a hyphen, which
    `^[a-z0-9][a-z0-9_]*$` refuses rather than resolves, so the dry run raised
    inside `open_run` on the very launch it existed to de-risk. It cost
    nothing, because it crashed before any output was claimed; a rehearsal
    nobody can run is the whole loss.

    So this builds the layout FOR REAL, which is the only thing that would
    have caught it.
    """
    import autoinit_c2_behavioural_launch as LL
    from experiments.run_layout import layout_for

    from aadistill.runtime.run_layout import RunLayoutError

    rid = f"attempt6{LL.DRY_RUN_SUFFIX}"
    #: Constructing the layout is what VALIDATES the id. It raised on the
    #: hyphen, and no string comparison could have known.
    layout = layout_for(tmp_path, LL.EXPERIMENT_ID, rid, LL.STAGE_ID)
    assert layout.run_id == rid

    #: And the validator really does refuse: without this the line above
    #: passes for any id at all, which is how the hyphen survived.
    with pytest.raises(RunLayoutError, match="not valid here"):
        layout_for(tmp_path, LL.EXPERIMENT_ID, "attempt6-dryrun", LL.STAGE_ID)

    #: `newest_attempt`-style numeric parsing must not see it as an attempt: a
    #: dry-run directory beside the real ones must not be mistaken for the
    #: newest, or the launch guards would call the real chain stale.
    assert not rid[len("attempt"):].isdigit()
    #: And it is a distinct path from the run it rehearses, which is the whole
    #: point: the real directory stays unoccupied.
    assert LL.session_record_path(rid) != LL.session_record_path("attempt6")
    assert rid in LL.session_record_path(rid)


# ---------------------------------------------------------------------------
# the volume gate: the pre-staged probes must exist BEFORE a pod is drawn
# ---------------------------------------------------------------------------

def _staging_record(repo: Path, run: str, *, probes, volume=None,
                    campaign=BG.CAMPAIGN_ID, terminal="ALL_STAGED",
                    finished="2026-09-23T00:00:00+00:00") -> Path:
    """One staging run's record, where `staged_probe_index` looks for it."""
    path = repo / L.STAGING_RECORD_ROOT / run / "staging_record.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema": "aadistill.autoinit.c2_probe_volume_staging/v1",
        "campaign_id": campaign,
        "volume_id": volume or L.CAMPAIGN_VOLUME_ID,
        "terminal": terminal,
        "finished_utc": finished,
        "probes": [{"probe_id": p, "verified": True, "bytes": 2 * 2**30,
                    "source_attempt": "attempt1"} for p in probes],
    }, indent=1) + "\n")
    return path


def test_a_fresh_campaign_owes_no_pre_staging(tmp_path, repo, transport):
    """A campaign with no verified probe has nothing to stage and nothing owed.

    Demanding a staging record here would refuse the FIRST attempt of every
    future campaign, which is the opposite of what this gate is for.
    """
    ctx = _Ctx(_Pod(tmp_path / "pod"), tmp_path / "empty", "attempt1")
    ok, why = L.volume_gate(ctx)
    assert ok, why
    assert "no probe needs its weights" in why


def test_the_gate_refuses_a_continuation_whose_probes_were_never_staged(
        tmp_path, repo, transport):
    """The one new way to fail, refused at `$0` instead of on a billing pod.

    On the pod this surfaces after setup, after the image pull and after the
    teacher is materialized — and it is unrecoverable, because a completed
    probe may not be retrained, so the session can only abort.
    """
    store = tmp_path / "store"
    pod1 = _Pod(tmp_path / "pod1")
    transport.pods["fake-host"] = pod1
    ids = _screening_ids()[:2]
    #: UNSCORED: the gate asks for the probes whose WEIGHTS a remaining
    #: operation reads, and a completed and validly scored probe has none.
    units = [_produce(pod1, pid, rung="screening", arm=arm, seed=seed,
                      scored=False)
             for pid, arm, seed in ids]
    _secure(_Ctx(pod1, store, "attempt1"), units)

    ctx = _Ctx(_Pod(tmp_path / "pod2"), store, "attempt2")
    ok, why = L.volume_gate(ctx)
    assert not ok
    assert "never verified onto volume" in why
    assert "stage_c2_probes_to_volume" in why

    #: Stage only one of the two: still refused, and it names the gap.
    _staging_record(repo, "stage1", probes=[ids[0][0]])
    ok, why = L.volume_gate(ctx)
    assert not ok
    assert ids[1][0] in why

    #: Both staged: the gate passes and says what it found.
    _staging_record(repo, "stage2", probes=[p for p, _, _ in ids])
    ok, why = L.volume_gate(ctx)
    assert ok, why
    assert L.CAMPAIGN_VOLUME_ID in why and L.VOLUME_DATACENTER in why


def test_only_a_completed_staging_run_for_this_campaign_and_volume_counts(
        tmp_path, repo, transport):
    """Three ways a record is not evidence about THIS volume."""
    store = tmp_path / "store"
    pod1 = _Pod(tmp_path / "pod1")
    transport.pods["fake-host"] = pod1
    pid, arm, seed = _screening_ids()[0]
    _secure(_Ctx(pod1, store, "attempt1"),
            [_produce(pod1, pid, rung="screening", arm=arm, seed=seed,
                      scored=False)])
    ctx = _Ctx(_Pod(tmp_path / "pod2"), store, "attempt2")

    #: A run that could not confirm its own resource was released is not
    #: evidence about what survived it.
    _staging_record(repo, "unreconciled", probes=[pid],
                    terminal="STAGED_BUT_UNRECONCILED")
    assert not L.volume_gate(ctx)[0]

    #: Another campaign's staging. One experiment's probes are never pooled.
    _staging_record(repo, "other_campaign", probes=[pid],
                    campaign="some-other-campaign")
    assert not L.volume_gate(ctx)[0]

    #: The right probes, staged onto a DIFFERENT volume than this session will
    #: attach. The bytes are somewhere; they are not where the pod will look.
    _staging_record(repo, "other_volume", probes=[pid], volume="some-other-vol")
    assert not L.volume_gate(ctx)[0]

    _staging_record(repo, "good", probes=[pid])
    assert L.volume_gate(ctx)[0]


def test_the_gate_refuses_a_mount_that_would_cover_the_checkout(
        tmp_path, repo, transport):
    """/workspace is the checkout root, and the provider's own default."""
    store = tmp_path / "store"
    pod1 = _Pod(tmp_path / "pod1")
    transport.pods["fake-host"] = pod1
    pid, arm, seed = _screening_ids()[0]
    _secure(_Ctx(pod1, store, "attempt1"),
            [_produce(pod1, pid, rung="screening", arm=arm, seed=seed,
                      scored=False)])
    _staging_record(repo, "stage1", probes=[pid])

    ctx = _Ctx(_Pod(tmp_path / "pod2"), store, "attempt2")
    ctx.args.volume_mount_path = "/workspace"
    ok, why = L.volume_gate(ctx)
    assert not ok
    assert "checkout root" in why

    #: And all three of volume, mount and datacenter are required together.
    for field in ("network_volume_id", "volume_mount_path", "data_center_ids"):
        ctx = _Ctx(_Pod(tmp_path / f"pod_{field}"), store, "attempt2")
        setattr(ctx.args, field, "")
        ok, why = L.volume_gate(ctx)
        assert not ok, field
        assert "All three are required" in why


def test_the_volume_is_attached_only_when_a_probe_needs_its_weights(
        tmp_path, repo, transport):
    """The DRAW CONSTRAINT follows a consumer, not the campaign.

    Attaching the volume pins acquisition to its one datacenter, and attempt8
    died there: eight create calls over forty minutes, "no longer any
    instances available with the requested specifications" from EU-NL-1, for a
    volume no remaining operation was going to read. `$0`, but the chain was
    consumed.

    Both directions are asserted, because a derivation that always answered
    one way would pass half of this on its own.
    """
    store = tmp_path / "store"
    pod1 = _Pod(tmp_path / "pod1")
    transport.pods["fake-host"] = pod1
    pid, arm, seed = _screening_ids()[0]

    #: An UNSCORED probe: scoring reads the weights, so the constraint buys
    #: something and the session accepts it.
    _secure(_Ctx(pod1, store, "attempt1"),
            [_produce(pod1, pid, rung="screening", arm=arm, seed=seed,
                      scored=False)])
    args = _Ctx(_Pod(tmp_path / "pod2"), store, "attempt2").args
    attach = L.volume_attachment(args)
    assert attach["attach"] is True
    assert attach["probes"] == [pid]
    #: And the triple survives, or the gate would refuse the very restore the
    #: session is attaching for.
    assert (args.network_volume_id, args.volume_mount_path,
            args.data_center_ids) == (L.CAMPAIGN_VOLUME_ID, L.VOLUME_MOUNT,
                                      L.VOLUME_DATACENTER)

    #: The SAME probe, scored. Nothing reads its checkpoint now: the verdict
    #: consumes rows and scores, and a newly trained probe is preserved by
    #: `_fetch_and_verify` to the launcher host, not to the volume.
    pod3 = _Pod(tmp_path / "pod3")
    transport.pods["fake-host"] = pod3
    store2 = tmp_path / "store2"
    _secure(_Ctx(pod3, store2, "attempt1"),
            [_produce(pod3, pid, rung="screening", arm=arm, seed=seed,
                      scored=True)])
    args = _Ctx(pod3, store2, "attempt2").args
    attach = L.volume_attachment(args)
    assert attach["attach"] is False
    assert attach["probes"] == []
    assert "no remaining operation reads a pre-staged checkpoint" in attach["why"]
    #: THE POINT: the draw is no longer pinned to one datacenter.
    assert (args.network_volume_id, args.volume_mount_path,
            args.data_center_ids) == ("", "", "")


def test_an_unattached_session_pins_no_datacenter_and_claims_no_pod_root(
        tmp_path, repo, transport):
    """What `attach: False` must actually produce, at the two surfaces.

    The command line is the one that mattered: a `--data-center-ids` the
    session had no use for is what starved the draw. And `volume_probe_root`
    must not answer with a path rooted at `/`, which would record the bytes as
    living somewhere they have never been.
    """
    from aadistill.infrastructure.session_runner import SessionRunner

    ctx = _Ctx(_Pod(tmp_path / "pod"), tmp_path / "store", "attempt2")
    assert L.volume_probe_root(ctx).startswith(L.VOLUME_MOUNT)

    #: An empty store owes no restore, so resolving against it is what a
    #: session with nothing to fetch really does — not a hand-cleared fixture.
    assert L.volume_attachment(ctx.args)["attach"] is False
    assert L.volume_probe_root(ctx) == ""

    flags = SessionRunner.attached_volume(
        type("R", (), {"a": ctx.args, "ws": "/workspace"})())
    assert "--data-center-ids" not in flags
    assert "--network-volume-id" not in flags
    assert flags == ("--volume-in-gb", "0")


def test_the_launcher_refuses_a_volume_index_for_another_campaign(
        tmp_path, repo, transport):
    """One experiment's probes may never be pooled into another's."""
    store = tmp_path / "store"
    pod1, pod2 = _Pod(tmp_path / "pod1"), _Pod(tmp_path / "pod2")
    transport.pods["fake-host"] = pod1
    pid, arm, seed = _screening_ids()[0]
    _secure(_Ctx(pod1, store, "attempt1"),
            [_produce(pod1, pid, rung="screening", arm=arm, seed=seed,
                      scored=False)])
    shutil.rmtree(pod1.root)
    transport.pods["fake-host"] = pod2
    root = _prestage(pod2, store)

    index = json.loads((root / "staged_index.json").read_text())
    index["campaign_id"] = "a-different-campaign"
    (root / "staged_index.json").write_text(json.dumps(index, indent=1) + "\n")

    ctx = _Ctx(pod2, store, "attempt2")
    assert L.restore_campaign_probes(ctx) is False
    assert any("never be pooled" in m for m in ctx.said)


def test_the_launcher_refuses_a_copy_staged_from_a_different_attempt(
        tmp_path, repo, transport):
    """Re-identified bytes prove WHAT a file is, never WHICH measurement.

    Probe ids are unique within a campaign and `campaign_state` collapses them
    across attempts, so a stale copy from an earlier attempt can sit under
    exactly the right name. Without this it would fail re-identification on the
    pod — a correct refusal arriving at the most expensive possible moment.
    """
    store = tmp_path / "store"
    pod1, pod2 = _Pod(tmp_path / "pod1"), _Pod(tmp_path / "pod2")
    transport.pods["fake-host"] = pod1
    pid, arm, seed = _screening_ids()[0]
    _secure(_Ctx(pod1, store, "attempt1"),
            [_produce(pod1, pid, rung="screening", arm=arm, seed=seed,
                      scored=False)])
    shutil.rmtree(pod1.root)
    transport.pods["fake-host"] = pod2
    root = _prestage(pod2, store)

    index = json.loads((root / "staged_index.json").read_text())
    index["probes"][pid]["source_attempt"] = "attempt0"
    (root / "staged_index.json").write_text(json.dumps(index, indent=1) + "\n")

    ctx = _Ctx(pod2, store, "attempt2")
    assert L.restore_campaign_probes(ctx) is False
    assert any("not the measurement the manifest identifies" in m
               for m in ctx.said)


def test_a_probe_named_for_restore_but_absent_from_the_volume_aborts(
        tmp_path, repo, transport):
    """A completed probe may not be retrained, so absence can only stop."""
    store = tmp_path / "store"
    pod1, pod2 = _Pod(tmp_path / "pod1"), _Pod(tmp_path / "pod2")
    transport.pods["fake-host"] = pod1
    ids = _screening_ids()[:2]
    _secure(_Ctx(pod1, store, "attempt1"),
            [_produce(pod1, pid, rung="screening", arm=arm, seed=seed,
                      scored=False)
             for pid, arm, seed in ids])
    shutil.rmtree(pod1.root)
    transport.pods["fake-host"] = pod2
    #: Stage only the first. The second is named by the manifest and missing.
    _prestage(pod2, store, only={ids[0][0]})

    ctx = _Ctx(pod2, store, "attempt2")
    assert L.restore_campaign_probes(ctx) is False
    assert any("never staged onto the volume" in m for m in ctx.said)


def test_the_restore_moves_no_weights_for_a_completed_probe(
        tmp_path, repo, transport):
    """What the volume and the consumer rule buy, counted in scp calls.

    `scp` is the substituted seam in this module, so the transport fixture
    records every call. Three completed and validly scored probes cost: one
    manifest, plus one evidence copy each. What they do NOT cost is a
    checkpoint — not over the wire, because it is pre-staged, and not at all,
    because nothing remaining reads it.
    """
    store = tmp_path / "store"
    pod1, pod2 = _Pod(tmp_path / "pod1"), _Pod(tmp_path / "pod2")
    transport.pods["fake-host"] = pod1
    ids = _screening_ids()[:3]
    _secure(_Ctx(pod1, store, "attempt1"),
            [_produce(pod1, pid, rung="screening", arm=arm, seed=seed)
             for pid, arm, seed in ids])
    shutil.rmtree(pod1.root)
    transport.pods["fake-host"] = pod2
    _prestage(pod2, store)

    before = len(transport.calls)
    ctx = _Ctx(pod2, store, "attempt2")
    assert L.restore_campaign_probes(ctx) is True, ctx.said
    sent = transport.calls[before:]
    assert len(sent) == 1 + 3, (
        f"a continuation sent {len(sent)} scp calls; expected the manifest "
        "plus one evidence copy per completed probe")
    assert any(a.endswith("campaign_continuation.json") for a in sent[0])

    #: Not one call carries a checkpoint.
    for call in sent:
        assert not any(a.endswith("model.safetensors") for a in call), call

    #: And the preflight says so, in the numbers.
    ev = ctx.evidence["campaign_restore"]
    assert ev["n_weights_probes"] == 0
    assert ev["n_evidence_probes"] == 3
    assert ev["skipped_checkpoints"] == 3
    assert ev["avoided_gib"] >= 0.0
    assert "GiB avoided" in ev["transfer_plan"]["summary"]

    #: The rows the verdict reads are on the pod; the weights are not.
    manifest = json.loads(pod2.local(L.RESTORE_MANIFEST).read_text())
    for entry in manifest["evidence"]:
        landed = pod2.local(entry["pod_path"])
        assert (landed / BC.PER_SAMPLE_NAME).is_file()
        assert not (landed / "model.safetensors").exists()


def test_an_indexed_probe_whose_bytes_are_not_on_the_volume_aborts(
        tmp_path, repo, transport):
    """The index is a document; the bytes are the evidence.

    A probe can be named by the staged index and still be unreadable — a
    staging run that was interrupted between writing a directory and writing
    the index, a volume that lost it. The index check cannot see that, so
    without a real presence test the launcher would hand the driver a path with
    nothing at it and the failure would surface inside stage S.
    """
    store = tmp_path / "store"
    pod1, pod2 = _Pod(tmp_path / "pod1"), _Pod(tmp_path / "pod2")
    transport.pods["fake-host"] = pod1
    ids = _screening_ids()[:2]
    _secure(_Ctx(pod1, store, "attempt1"),
            [_produce(pod1, pid, rung="screening", arm=arm, seed=seed,
                      scored=False)
             for pid, arm, seed in ids])
    shutil.rmtree(pod1.root)
    transport.pods["fake-host"] = pod2
    root = _prestage(pod2, store)

    #: Indexed, and gone. The index is left untouched.
    shutil.rmtree(root / ids[1][0])
    index = json.loads((root / "staged_index.json").read_text())
    assert ids[1][0] in index["probes"], "the index must still claim it"

    ctx = _Ctx(pod2, store, "attempt2")
    assert L.restore_campaign_probes(ctx) is False
    assert any("not readable on the volume" in m for m in ctx.said)


def test_a_scored_probe_whose_rows_did_not_survive_is_caught_before_the_verdict(
        tmp_path, repo, transport):
    """The decision rule reads per-sample ROWS, not the summary.

    A completed probe contributes evidence and nothing else, so its rows are
    the whole of what it brings. If they are missing on the launcher host the
    continuation has nothing to decide with, and discovering that in stage D
    wastes the entire run — which is why the check is on the host, before a
    byte is sent, rather than a presence test after.
    """
    store = tmp_path / "store"
    pod1, pod2 = _Pod(tmp_path / "pod1"), _Pod(tmp_path / "pod2")
    transport.pods["fake-host"] = pod1
    pid, arm, seed = _screening_ids()[0]
    _secure(_Ctx(pod1, store, "attempt1"),
            [_produce(pod1, pid, rung="screening", arm=arm, seed=seed)])
    shutil.rmtree(pod1.root)
    transport.pods["fake-host"] = pod2
    _prestage(pod2, store)

    held = BC.campaign_state(L.campaign_store(BG.CAMPAIGN_ID, store))
    #: `campaign_state` reads a probe whose rows are gone as UNSCORED, which is
    #: correct and a different route -- so the case this covers is rows that
    #: exist for the state read and are removed before the transfer.
    assert held["probes"][pid]["scored"] is True
    ctx = _Ctx(pod2, store, "attempt2")
    work = L.campaign_remaining_work(ctx)
    assert pid in work["transfer"]["evidence"]["probes"]
    (Path(held["probes"][pid]["durable_path"]) / BC.PER_SAMPLE_NAME).unlink()

    assert L.restore_campaign_probes(ctx) is False
    assert any("evidence is incomplete on this host" in m for m in ctx.said)


# ---------------------------------------------------------------------------
# a session may not be authorized to spend past its campaign's ceiling
# ---------------------------------------------------------------------------

RATE = 1.09


def test_a_fresh_campaign_gets_the_whole_frozen_session():
    """Nothing settled, nothing shortened. The derivation is untouched."""
    terms = BG.authorization_terms(REPO, rate_usd_per_hour=RATE,
                                   campaign_all_in_hard_usd=42.0)
    assert terms["hard_runtime_minutes"] == RUNTIME
    assert terms["all_in_hard_usd"] == ALL_IN
    assert round(terms["gpu_hard_usd"] + terms["disk_hard_usd"], 4) == ALL_IN


def test_a_session_is_never_authorized_past_what_the_campaign_has_left():
    """THE GAP: the runaway bound exceeded the bound it was supposed to obey.

    The continuation gate charges the campaign for the work a session PLANS.
    Nothing charged it for what that session could cost if the work went wrong,
    because `all_in_hard_usd` came from the frozen full-session decomposition
    and was the same figure for the first attempt and the fifth. With `$2.5425`
    settled that was safe by one cent; with `$22.2466` settled a full session
    would have been authorized to reach `$55.4565` against a `$42.0000`
    campaign ceiling, and every gate it passed would have said yes.
    """
    settled = 22.2466
    campaign = 42.0
    terms = BG.authorization_terms(REPO, rate_usd_per_hour=RATE,
                                   campaign_all_in_hard_usd=campaign,
                                   settled_campaign_all_in_usd=settled)
    assert terms["all_in_hard_usd"] < ALL_IN
    assert settled + terms["all_in_hard_usd"] <= campaign, (
        "a session may not be authorized to spend past its campaign's "
        "cumulative ceiling")
    #: The runtime is shortened WITH the money. A window the money does not
    #: fund is the defect the session amounts are kept apart to prevent.
    assert terms["hard_runtime_minutes"] < RUNTIME
    assert round(terms["gpu_hard_usd"] + terms["disk_hard_usd"], 4) == round(
        terms["all_in_hard_usd"], 4)
    #: Derived at the SAME rate the session will be billed at.
    assert round(terms["hard_runtime_minutes"] / 60 * RATE, 4) == round(
        terms["gpu_hard_usd"], 4)


def test_the_shortened_ceiling_rounds_in_the_safe_direction():
    """A LIMIT rounds down. Rounding a limit up spends money nobody granted."""
    settled, campaign = 22.2466, 42.0
    terms = BG.authorization_terms(REPO, rate_usd_per_hour=RATE,
                                   campaign_all_in_hard_usd=campaign,
                                   settled_campaign_all_in_usd=settled)
    available = campaign - settled
    assert terms["all_in_hard_usd"] <= available
    #: And not absurdly below it: flooring costs at most a rounding quantum,
    #: never a meaningful share of the window.
    assert terms["all_in_hard_usd"] > available - 0.01


def test_a_campaign_with_nothing_left_refuses_rather_than_shortening_to_zero():
    """Absent money is a refusal, not a very short session."""
    with pytest.raises(BG.BehaviouralGovernanceError, match="nothing"):
        BG.authorization_terms(REPO, rate_usd_per_hour=RATE,
                               campaign_all_in_hard_usd=42.0,
                               settled_campaign_all_in_usd=42.0)
    with pytest.raises(BG.BehaviouralGovernanceError, match="nothing"):
        BG.authorization_terms(REPO, rate_usd_per_hour=RATE,
                               campaign_all_in_hard_usd=42.0,
                               settled_campaign_all_in_usd=99.0)


def test_the_shortened_window_still_funds_the_work_a_continuation_owes(
        tmp_path, repo, transport):
    """Shortening the runaway bound must not shorten the experiment.

    This is the check that makes the cap safe to APPLY rather than merely safe:
    once the money is capped, the work the campaign still owes has to fit
    inside the window that money buys. It is asserted against a campaign built
    here — ten completed probes, the state this campaign is actually in — so it
    depends on no host's artifact store and skips on no machine.
    """
    store = tmp_path / "store"
    pod1 = _Pod(tmp_path / "pod1")
    transport.pods["fake-host"] = pod1
    advanced = _candidate_ids()[0]
    proto = BH.protocol(REPO)["behavioural_selection"]
    conf_seeds = [int(x) for x in proto["seeds"]["confirmation"]]

    units = [_produce(pod1, pid, rung="screening", arm=arm, seed=seed)
             for pid, arm, seed in _screening_ids()]
    #: Two of the three confirmation seeds, both arms: exactly where this
    #: campaign stands after attempt5.
    for seed in conf_seeds[:2]:
        for arm in (advanced, SCH.ANCHOR):
            units.append(_produce(pod1, f"confirmation.{arm}.s{seed}",
                                  rung="confirmation", arm=arm, seed=seed))
    _commit_ranking(pod1, advanced)
    _secure(_Ctx(pod1, store, "attempt1"), units)

    state = BC.campaign_state(L.campaign_store(BG.CAMPAIGN_ID, store))
    work = BC.remaining_work(REPO, state=state)
    assert work["n_probes_remaining"] == 2, sorted(work["probes_remaining"])

    terms = BG.authorization_terms(REPO, rate_usd_per_hour=RATE,
                                   campaign_all_in_hard_usd=42.0,
                                   settled_campaign_all_in_usd=22.2466)
    assert terms["hard_runtime_minutes"] < RUNTIME, "the cap did not bind"
    assert work["decomposition"]["hard_minutes"] < terms["hard_runtime_minutes"], (
        f"the campaign owes {work['decomposition']['hard_minutes']} minutes "
        f"and the money it has left buys {terms['hard_runtime_minutes']}")


def test_two_readings_of_settled_spend_must_agree_in_the_dangerous_direction(
        tmp_path, repo, transport):
    """The issuer caps against one figure; this gate charges another.

    Both are needed — the gate reconciles each predecessor RESOURCE under R9,
    while the issuer must cap a session using a figure independent of the
    authorization it is deriving. They must not drift: a session capped against
    a published total SMALLER than what the gate charges has room its campaign
    does not have, and every gate it passed would say yes.
    """
    store = tmp_path / "store"
    _run_manifest(repo, "attempt1")
    _session_record(repo, "attempt1", actual_usd=8.0, elapsed_minutes=440.0)
    ctx = _Ctx(_Pod(tmp_path / "pod2"), store, "attempt2",
               campaign=ALL_IN + 12.0)

    #: Consistent to begin with.
    ok, why = L.campaign_continuation_gate(ctx)
    assert ok, why

    #: Now the closeout under-reports what the resource actually billed.
    closeout = (repo / L.rel_run_dir(L.EXPERIMENT_ID, "attempt1", L.STAGE_ID)
                / "closeout/outcome.json")
    doc = json.loads(closeout.read_text())
    doc["money"]["all_in_usd"] = 1.0
    closeout.write_text(json.dumps(doc, indent=1) + "\n")

    ctx = _Ctx(_Pod(tmp_path / "pod3"), store, "attempt2",
               campaign=ALL_IN + 12.0)
    ok, why = L.campaign_continuation_gate(ctx)
    assert not ok
    assert "reconciles to" in why and "the closeouts those attempts" in why


def test_an_over_reported_closeout_is_safe_and_does_not_block(
        tmp_path, repo, transport):
    """The other direction is a stricter cap, not a hazard.

    Refusing it would block a launch over an over-conservative number, which is
    a different failure from the one the reconciliation exists to catch.
    """
    store = tmp_path / "store"
    _run_manifest(repo, "attempt1")
    _session_record(repo, "attempt1", actual_usd=8.0, elapsed_minutes=440.0)
    closeout = (repo / L.rel_run_dir(L.EXPERIMENT_ID, "attempt1", L.STAGE_ID)
                / "closeout/outcome.json")
    doc = json.loads(closeout.read_text())
    doc["money"]["all_in_usd"] = float(doc["money"]["all_in_usd"]) + 2.0
    closeout.write_text(json.dumps(doc, indent=1) + "\n")

    ctx = _Ctx(_Pod(tmp_path / "pod2"), store, "attempt2",
               campaign=ALL_IN + 12.0)
    ok, why = L.campaign_continuation_gate(ctx)
    assert ok, why


# ---------------------------------------------------------------------------
# the durable destination is charged for what the session still owes
# ---------------------------------------------------------------------------

@needs_host_local_stores
def test_the_destination_is_charged_for_the_probes_this_session_produces(
        tmp_path, repo, transport, monkeypatch):
    """A probe already in the store OCCUPIES it; it is not also a need.

    This gate charged the whole twelve-probe requirement — 26.654 GiB — every
    time, which is right for a fresh campaign and double-counts for a
    continuation: the ten probes already there are subtracted from free space
    AND added to the need. On the launcher host, with 16.4 GiB free and ten
    probes held, that refused a session whose real appetite is two probes and
    4.4 GiB.
    """
    import shutil as _shutil

    store = tmp_path / "store"
    pod1, pod2 = _Pod(tmp_path / "pod1"), _Pod(tmp_path / "pod2")
    transport.pods["fake-host"] = pod1
    advanced = _candidate_ids()[0]
    proto = BH.protocol(REPO)["behavioural_selection"]
    conf = [int(s) for s in proto["seeds"]["confirmation"]]
    units = [_produce(pod1, pid, rung="screening", arm=arm, seed=seed)
             for pid, arm, seed in _screening_ids()]
    for seed in conf[:2]:
        for arm in (advanced, SCH.ANCHOR):
            units.append(_produce(pod1, f"confirmation.{arm}.s{seed}",
                                  rung="confirmation", arm=arm, seed=seed))
    _commit_ranking(pod1, advanced)
    _secure(_Ctx(pod1, store, "attempt1"), units)

    ctx = _Ctx(pod2, store, "attempt2")
    work = L.campaign_remaining_work(ctx)
    assert work["n_probes_remaining"] == 2

    #: Free space that holds the two probes owed and NOT the whole protocol.
    two_probes = 5 * 2**30
    monkeypatch.setattr(_shutil, "disk_usage",
                        lambda p: type("U", (), {"free": two_probes,
                                                 "total": 0, "used": 0})())
    ok, why = L.destination_gate(ctx)
    assert ok, why
    assert "2 probe(s)" in why and "already there" in why
    ev = ctx.evidence["durable_destination"]
    assert ev["probes_owed"] == 2
    assert ev["need_bytes"] < ev["whole_protocol_need_bytes"], (
        "the continuation was charged the whole protocol's requirement")

    #: And it still refuses when the remainder genuinely does not fit.
    ctx2 = _Ctx(_Pod(tmp_path / "pod3"), store, "attempt2")
    monkeypatch.setattr(_shutil, "disk_usage",
                        lambda p: type("U", (), {"free": 2**30, "total": 0,
                                                 "used": 0})())
    ok, why = L.destination_gate(ctx2)
    assert not ok
    assert "still owes" in why


@needs_host_local_stores
def test_a_fresh_campaign_is_still_charged_for_every_probe(tmp_path, repo,
                                                           monkeypatch):
    """The remainder of a fresh campaign is all of it, so nothing is weakened.

    That is the property that makes charging the remainder safe rather than
    lenient: one derivation serves both, and a first attempt is still refused
    if it cannot preserve the twelve probes it will produce.
    """
    import shutil as _shutil

    ctx = _Ctx(_Pod(tmp_path / "pod"), tmp_path / "empty", "attempt1")
    work = L.campaign_remaining_work(ctx)
    assert work["n_probes_remaining"] == 12
    monkeypatch.setattr(_shutil, "disk_usage",
                        lambda p: type("U", (), {"free": 5 * 2**30,
                                                 "total": 0, "used": 0})())
    ok, why = L.destination_gate(ctx)
    assert not ok
    assert "12 probe(s) this session still owes" in why


def test_the_launcher_and_the_governance_module_name_one_runs_root():
    """Two spellings of one path is how a gate comes to enumerate nothing.

    `settled_campaign_all_in` reads this campaign's closeouts and the launcher's
    `campaign_attempts` enumerates its run records; both derive the root from
    `rel_run_dir`, but from their own copies of the experiment and stage ids. If
    those ever diverge, one of them reports a campaign with no paid
    predecessors — which is the exact bug the enumeration was added to fix.
    """
    assert BG.EXPERIMENT_ID == L.EXPERIMENT_ID
    assert BG.STAGE_ID == L.STAGE_ID
    assert BG.campaign_runs_rel() == L.runs_root_rel()


def test_a_campaign_of_only_scored_probes_needs_no_staging_at_all(
        tmp_path, repo, transport):
    """The gate must not ask for what has no consumer.

    This is C2's actual state: ten completed and validly scored probes and two
    untrained ones. If the gate asks for every held probe it demands a
    22.21 GiB pre-stage to satisfy nobody — and then refuses a launch whose
    real working set is 8.1 MiB of rows. A mutation that widened the gate back
    to every held probe passed every other test in this module.
    """
    store = tmp_path / "store"
    pod1 = _Pod(tmp_path / "pod1")
    transport.pods["fake-host"] = pod1
    ids = _screening_ids()[:3]
    _secure(_Ctx(pod1, store, "attempt1"),
            [_produce(pod1, pid, rung="screening", arm=arm, seed=seed)
             for pid, arm, seed in ids])

    ctx = _Ctx(_Pod(tmp_path / "pod2"), store, "attempt2")
    work = L.campaign_remaining_work(ctx)
    assert work["transfer"]["weights"]["n"] == 0
    assert work["transfer"]["evidence"]["n"] == 3
    assert work["transfer"]["skipped_weights"]["n"] == 3

    #: NO staging record exists, and the gate passes anyway, because nothing
    #: needs staging.
    ok, why = L.volume_gate(ctx)
    assert ok, why
    assert "no probe needs its weights" in why
    assert "have no remaining consumer" in why


def test_an_unscored_probe_may_not_be_admitted_as_evidence(tmp_path, repo,
                                                           transport):
    """A probe that has not been scored resumes AT SCORING, and scoring reads
    the model. Admitting one on its descriptor alone would let the scorer run
    against weights that are not there — and the whole reason a completed
    probe may travel light is that its measurement is already finished."""
    store = tmp_path / "store"
    pod1, pod2 = _Pod(tmp_path / "pod1"), _Pod(tmp_path / "pod2")
    transport.pods["fake-host"] = pod1
    pid, arm, seed = _screening_ids()[0]
    _secure(_Ctx(pod1, store, "attempt1"),
            [_produce(pod1, pid, rung="screening", arm=arm, seed=seed)])
    shutil.rmtree(pod1.root)
    transport.pods["fake-host"] = pod2
    _prestage(pod2, store)
    ctx = _Ctx(pod2, store, "attempt2")
    assert L.restore_campaign_probes(ctx) is True, ctx.said

    #: Strip the score from the evidence entry the pod will read.
    view = pod2.local(L.RESTORE_MANIFEST)
    manifest = json.loads(view.read_text())
    assert manifest["evidence"] and manifest["evidence"][0]["score"]
    manifest["evidence"][0]["score"] = None
    view.write_text(json.dumps(manifest, indent=1) + "\n")

    driver = _driver(pod2, "attempt2")
    with pytest.raises(D.C2DriverError, match="no score"):
        driver.restore_campaign()


def test_the_journal_refuses_an_evidence_entry_without_a_score(tmp_path,
                                                               monkeypatch):
    """The same rule at the other admission point.

    `reidentify` short-circuits for an evidence-only entry because there are
    deliberately no weights to check. That shortcut must not become a way in
    for an entry that has neither weights nor a completed measurement.
    """
    drv = D.C2BehaviouralDriver.__new__(D.C2BehaviouralDriver)
    ok, why = drv.reidentify({"evidence_only": True,
                              "score": {"result_sha256": "x"}})
    assert ok and "evidence-only" in why
    ok, why = drv.reidentify({"evidence_only": True})
    assert not ok and "no score" in why


def test_a_dry_run_directory_is_not_a_campaign_RESOURCE(tmp_path, repo):
    """THE third defect in the dry-run mechanism, and the deepest of them.

    `--dry-run` writes to its own run id so it cannot consume the chain it
    exists to de-risk. That puts its directory under `runs/`, beside the real
    attempts — and `campaign_attempts` enumerates exactly that. attempt7's dry
    run reached the ninth gate, created `attempt7_dryrun`, and then the gate
    read it as a prior RESOURCE of the campaign whose billing state could not
    be established, and refused the launch it had just rehearsed.

    A dry run contacts no provider. That is what the flag means, and both
    enumerations of this campaign's run directories have to agree about it,
    which is why the suffix is owned by `behavioural_governance` rather than by
    the launcher that happens to derive it.
    """
    store = tmp_path / "store"
    _run_manifest(repo, "attempt1")
    _session_record(repo, "attempt1", actual_usd=3.41)
    #: A dry-run directory that looks exactly like a run that billed and left
    #: no session record — the shape that refuses.
    _run_manifest(repo, f"attempt2{BG.DRY_RUN_SUFFIX}")

    prior = L.campaign_attempts(BG.CAMPAIGN_ID, exclude="attempt2",
                                store=store, repo_root=repo)
    assert prior == ["attempt1"], (
        "a dry-run directory was counted as a prior provider resource")

    #: And it contributes nothing to SETTLED SPEND either, for the same
    #: reason. A dry-run directory with no closeout is skipped anyway, so the
    #: exclusion only does work when one HAS a closeout — which is what a dry
    #: run's own `finally` would write. Give it one.
    dry = f"attempt3{BG.DRY_RUN_SUFFIX}"
    out = (repo / L.rel_run_dir(L.EXPERIMENT_ID, dry, L.STAGE_ID)
           / "closeout/outcome.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"provider_resource_created": False,
                               "money": {"all_in_usd": 99.99}}) + "\n")
    settled = BG.settled_campaign_all_in(repo)
    assert not any(BG.is_dry_run_id(k) for k in settled["per_attempt"]), (
        settled["per_attempt"])
    assert settled["total_usd"] < 99.99, (
        "a dry run's closeout entered the campaign's settled spend")


def test_the_dry_run_suffix_has_one_owner():
    """Two enumerations agreeing by coincidence is two chances to diverge."""
    assert L.DRY_RUN_SUFFIX is BG.DRY_RUN_SUFFIX
    assert BG.is_dry_run_id("attempt7_dryrun")
    assert not BG.is_dry_run_id("attempt7")
    #: And it is a valid run id, which a hyphen was not.
    from experiments.run_layout import layout_for
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        layout_for(tmp, L.EXPERIMENT_ID, f"attempt7{BG.DRY_RUN_SUFFIX}",
                   L.STAGE_ID)
